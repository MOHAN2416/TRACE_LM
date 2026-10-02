"""
Token-by-token text generation with real-time hidden-state monitoring for TRACE-LM.
Extracts hidden states from layers 3-6 during generation without modifying the model.
"""

from typing import Tuple, List, Dict, Any, Optional, Callable
import torch
import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizer

from trace_lm.config import (
    MONITORED_LAYERS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from trace_lm.features import (
    extract_token_uncertainty,
    compute_layer_hidden_metrics,
    enrich_trajectory,
)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies temperature scaling and top-p (nucleus) filtering to logits,
    then samples the next token.

    Args:
        logits: Logits tensor for the last token position (shape: [1, vocab_size]).
        temperature: Temperature parameter (> 0.0).
        top_p: Top-p nucleus sampling threshold (0.0 < top_p <= 1.0).

    Returns:
        (next_token_tensor, softmax_probabilities_tensor)
    """
    temperature = max(1e-5, temperature)
    scaled_logits = logits / temperature

    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mask tokens with cumulative probability above threshold
    mask = cumulative_probs > top_p
    # Shift mask right to keep the first token exceeding threshold
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False

    sorted_logits[mask] = float("-inf")

    filtered_logits = torch.full_like(scaled_logits, float("-inf"))
    filtered_logits.scatter_(1, sorted_indices, sorted_logits)

    probs = torch.softmax(filtered_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token, probs


def generate_with_trajectory(
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[str, pd.DataFrame, torch.Tensor]:
    """
    Generates text token-by-token from a user prompt, genuinely extracting hidden states
    from layers 3, 4, 5, and 6 at each step to build the temporal trajectory.

    Args:
        prompt: Raw user prompt string.
        model: Frozen GPT-2-medium model.
        tokenizer: GPT-2 tokenizer.
        device: Torch device (CPU or GPU).
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p sampling probability threshold.
        step_callback: Optional callback invoked after each token generation step.

    Returns:
        (generated_text, enriched_trajectory_df, generated_token_ids)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = inputs["input_ids"].clone()

    records: List[Dict[str, Any]] = []
    previous_hidden: Dict[int, Optional[torch.Tensor]] = {
        layer: None for layer in MONITORED_LAYERS
    }

    for step in range(1, max_new_tokens + 1):
        with torch.no_grad():
            outputs = model(
                input_ids=generated_ids,
                attention_mask=torch.ones_like(generated_ids),
                output_hidden_states=True,
            )

        logits = outputs.logits[:, -1, :]
        next_token, probs = sample_next_token(logits, temperature=temperature, top_p=top_p)

        token_id = int(next_token.item())
        token_str = tokenizer.decode([token_id])

        # Compute uncertainty features
        prob, entropy, norm_entropy = extract_token_uncertainty(probs, token_id)

        # Extract hidden states for monitored layers
        current_hidden: Dict[int, torch.Tensor] = {}
        for layer in MONITORED_LAYERS:
            # outputs.hidden_states is a tuple of length (n_layer + 1)
            # shape of each: [batch_size, seq_len, hidden_dim]
            h = outputs.hidden_states[layer][0, -1, :].detach().clone()
            current_hidden[layer] = h

        # Compute layer metrics (norm, delta, cosine similarity)
        layer_metrics = compute_layer_hidden_metrics(current_hidden, previous_hidden)

        # Update previous hidden states
        previous_hidden = current_hidden

        row: Dict[str, Any] = {
            "step": step,
            "token": token_str,
            "token_id": token_id,
            "token_probability": prob,
            "entropy": entropy,
            "normalized_entropy": norm_entropy,
        }
        row.update(layer_metrics)
        records.append(row)

        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        if step_callback is not None:
            step_callback(row)

        if token_id == tokenizer.eos_token_id:
            break

    # Build and enrich trajectory DataFrame
    raw_trajectory = pd.DataFrame(records)
    trajectory_df = enrich_trajectory(raw_trajectory)

    # Decode newly generated tokens only (excluding prompt)
    newly_generated_tokens = generated_ids[0, prompt_length:]
    generated_text = tokenizer.decode(
        newly_generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return generated_text, trajectory_df, generated_ids
