"""
Unit tests for GPT-2-medium loading, frozen parameter status, and layer access in TRACE-LM.
"""

import pytest
import torch
from trace_lm.config import MODEL_NAME, MONITORED_LAYERS
from trace_lm.model import load_gpt2_medium, verify_frozen_model, verify_monitored_layers


@pytest.fixture(scope="module")
def gpt2_instance():
    """
    Loads GPT-2-medium once for test suite.
    """
    model, tokenizer, device = load_gpt2_medium()
    return model, tokenizer, device


def test_model_loads_and_device(gpt2_instance):
    """Verifies that model and tokenizer are successfully instantiated."""
    model, tokenizer, device = gpt2_instance
    assert model is not None
    assert tokenizer is not None
    assert isinstance(device, torch.device)


def test_model_parameters_are_frozen(gpt2_instance):
    """
    Ensures ALL model parameters are frozen (requires_grad is False).
    Mandated by TRACE-LM specification.
    """
    model, _, _ = gpt2_instance
    all_frozen = all(not p.requires_grad for p in model.parameters())
    assert all_frozen is True
    assert verify_frozen_model(model) is True

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_count == 0


def test_model_monitored_layers_accessible(gpt2_instance):
    """Verifies that layers 3, 4, 5, and 6 are accessible and output hidden states."""
    model, tokenizer, device = gpt2_instance
    assert verify_monitored_layers(model) is True

    test_input = tokenizer("Test prompt", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**test_input, output_hidden_states=True)

    assert outputs.hidden_states is not None
    # For GPT-2-medium (24 layers), hidden_states has 25 elements (embedding + 24 layers)
    assert len(outputs.hidden_states) >= max(MONITORED_LAYERS) + 1

    for layer in MONITORED_LAYERS:
        h = outputs.hidden_states[layer]
        # Shape: [batch_size, seq_len, hidden_dim]
        # GPT-2-medium hidden dimension is 1024
        assert h.dim() == 3
        assert h.shape[0] == 1
        assert h.shape[1] == test_input["input_ids"].shape[1]
        assert h.shape[2] == 1024
