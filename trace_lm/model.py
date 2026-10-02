"""
Model loading and verification for TRACE-LM.
Ensures GPT-2-medium is loaded in frozen evaluation mode and layers 3-6 are accessible.
"""

import logging
from typing import Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel

from trace_lm.config import MODEL_NAME, MONITORED_LAYERS

logger = logging.getLogger(__name__)


def get_device(requested_device: Optional[str] = None) -> torch.device:
    """
    Detect GPU availability automatically, safely falling back to CPU.
    """
    if requested_device:
        device = torch.device(requested_device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def verify_frozen_model(model: PreTrainedModel) -> bool:
    """
    Programmatically verifies that all model parameters are frozen (requires_grad is False).
    """
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    is_frozen = trainable_count == 0 and all(not p.requires_grad for p in model.parameters())
    if not is_frozen:
        raise RuntimeError(
            f"Model safety check failed: model is trainable ({trainable_count} trainable parameters). "
            f"TRACE-LM requires frozen GPT-2-medium."
        )
    return True


def verify_monitored_layers(model: PreTrainedModel) -> bool:
    """
    Verifies that the monitored layers (3, 4, 5, 6) exist in the model architecture.
    """
    num_layers = getattr(model.config, "n_layer", getattr(model.config, "num_hidden_layers", 0))
    for layer in MONITORED_LAYERS:
        if layer >= num_layers:
            raise ValueError(
                f"Monitored layer {layer} exceeds model layer depth {num_layers}."
            )
    return True


def load_gpt2_medium(
    device: Optional[torch.device] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, torch.device]:
    """
    Loads GPT-2-medium and tokenizer, freezes all parameters, sets eval mode,
    and returns (model, tokenizer, device).
    """
    if device is None:
        device = get_device()

    logger.info("Loading tokenizer for %s...", MODEL_NAME)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tokenizer for '{MODEL_NAME}'. Check internet connection "
            f"or local Hugging Face cache. Details: {e}"
        ) from e

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model for %s onto %s...", MODEL_NAME, device)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            output_hidden_states=True
        ).to(device)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model for '{MODEL_NAME}'. Check internet connection "
            f"or local Hugging Face cache. Details: {e}"
        ) from e

    # Set eval mode
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Programmatic verification
    verify_frozen_model(model)
    verify_monitored_layers(model)

    logger.info(
        "Successfully loaded %s on %s (Frozen: True, Monitored Layers: %s)",
        MODEL_NAME,
        device,
        MONITORED_LAYERS,
    )

    return model, tokenizer, device
