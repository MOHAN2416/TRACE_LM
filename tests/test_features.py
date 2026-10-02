"""
Unit tests for TRACE-LM feature engineering, trajectory metrics, and statistical aggregations.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from trace_lm.config import (
    MONITORED_LAYERS,
    BASE_FEATURE_COLUMNS,
    ALL_MODEL_FEATURE_COLUMNS,
    TRAJECTORY_FEATURE_COLUMNS,
    UNCERTAINTY_FEATURE_COLUMNS,
)
from trace_lm.features import (
    extract_token_uncertainty,
    compute_layer_hidden_metrics,
    enrich_trajectory,
    aggregate_token_features,
    compute_checkpoint_features,
)


def test_token_uncertainty_bounds():
    """Verifies uncertainty calculations (probability, entropy, normalized entropy)."""
    vocab_size = 50257
    probs = torch.zeros(1, vocab_size)
    probs[0, 42] = 1.0  # Deterministic token

    prob, ent, norm_ent = extract_token_uncertainty(probs, 42)
    assert prob == 1.0
    assert pytest.approx(ent, abs=1e-5) == 0.0
    assert pytest.approx(norm_ent, abs=1e-5) == 0.0

    # Uniform distribution
    uniform_probs = torch.full((1, vocab_size), 1.0 / vocab_size)
    prob_u, ent_u, norm_ent_u = extract_token_uncertainty(uniform_probs, 10)
    assert pytest.approx(prob_u, rel=1e-3) == 1.0 / vocab_size
    assert pytest.approx(norm_ent_u, abs=1e-4) == 1.0


def test_compute_layer_hidden_metrics_first_and_subsequent_steps():
    """Verifies metrics on first step (NaNs) and subsequent step (deltas and cosine)."""
    dim = 1024
    h1 = {layer: torch.ones(dim) for layer in MONITORED_LAYERS}
    prev_none = {layer: None for layer in MONITORED_LAYERS}

    metrics_step1 = compute_layer_hidden_metrics(h1, prev_none)
    for layer in MONITORED_LAYERS:
        assert f"L{layer}_norm" in metrics_step1
        assert np.isnan(metrics_step1[f"L{layer}_delta"])
        assert np.isnan(metrics_step1[f"L{layer}_cosine"])

    # Step 2: Identical vector -> delta = 0, cosine = 1.0
    metrics_step2 = compute_layer_hidden_metrics(h1, h1)
    for layer in MONITORED_LAYERS:
        assert pytest.approx(metrics_step2[f"L{layer}_delta"], abs=1e-5) == 0.0
        assert pytest.approx(metrics_step2[f"L{layer}_cosine"], abs=1e-5) == 1.0

    # Cross-layer drift metrics verification (identical ones across layers -> drift=0.0, cosine=1.0)
    assert pytest.approx(metrics_step1["L3_to_L4_drift"], abs=1e-5) == 0.0
    assert pytest.approx(metrics_step1["L4_to_L5_drift"], abs=1e-5) == 0.0
    assert pytest.approx(metrics_step1["L5_to_L6_drift"], abs=1e-5) == 0.0
    assert pytest.approx(metrics_step1["L3_to_L4_cosine"], abs=1e-5) == 1.0
    assert pytest.approx(metrics_step1["L4_to_L5_cosine"], abs=1e-5) == 1.0
    assert pytest.approx(metrics_step1["L5_to_L6_cosine"], abs=1e-5) == 1.0


def test_enrich_trajectory_and_relative_deltas():
    """Verifies relative delta formulas and cross-layer statistics."""
    records = []
    for step in range(1, 5):
        row = {
            "step": step,
            "token": f"tok_{step}",
            "token_probability": 0.8,
            "entropy": 0.5,
            "normalized_entropy": 0.1,
        }
        for layer in MONITORED_LAYERS:
            row[f"L{layer}_norm"] = 10.0 + step
            row[f"L{layer}_delta"] = np.nan if step == 1 else 2.0
            row[f"L{layer}_cosine"] = np.nan if step == 1 else 0.95
        records.append(row)

    raw_df = pd.DataFrame(records)
    enriched = enrich_trajectory(raw_df)

    assert "mean_relative_delta" in enriched.columns
    assert "mean_layer_norm" in enriched.columns

    # Step 2 relative delta should be 2.0 / (11.0 + 1e-8)
    expected_rel_delta = 2.0 / 11.0
    assert pytest.approx(enriched["relative_delta_L3"].iloc[1], rel=1e-3) == expected_rel_delta
    assert pytest.approx(enriched["mean_relative_delta"].iloc[1], rel=1e-3) == expected_rel_delta


def test_aggregate_features_shape_and_columns():
    """Verifies aggregated features produce 52 features with mean, std, max, min."""
    records = []
    for step in range(1, 10):
        row = {
            "step": step,
            "token": f"word_{step}",
            "token_probability": 0.7,
            "entropy": 1.2,
            "normalized_entropy": 0.15,
        }
        for layer in MONITORED_LAYERS:
            row[f"L{layer}_norm"] = 15.0
            row[f"L{layer}_delta"] = np.nan if step == 1 else 1.5
            row[f"L{layer}_cosine"] = np.nan if step == 1 else 0.9
        records.append(row)

    df = enrich_trajectory(pd.DataFrame(records))
    agg_df = aggregate_token_features(df)

    assert len(agg_df) == 1
    # Check all 52 columns exist
    for col in ALL_MODEL_FEATURE_COLUMNS:
        assert col in agg_df.columns, f"Missing feature column: {col}"

    assert agg_df["num_tokens"].iloc[0] == 8  # 9 steps minus step 1 NaN row

    # All values must be finite
    assert np.isfinite(agg_df[ALL_MODEL_FEATURE_COLUMNS].to_numpy()).all()


def test_edge_case_single_token():
    """Single token generation must not crash and must produce finite features."""
    record = {
        "step": 1,
        "token": "Paris",
        "token_probability": 0.95,
        "entropy": 0.2,
        "normalized_entropy": 0.02,
    }
    for layer in MONITORED_LAYERS:
        record[f"L{layer}_norm"] = 25.0
        record[f"L{layer}_delta"] = np.nan
        record[f"L{layer}_cosine"] = np.nan

    df = enrich_trajectory(pd.DataFrame([record]))
    agg_df = aggregate_token_features(df)

    assert len(agg_df) == 1
    vals = agg_df[ALL_MODEL_FEATURE_COLUMNS].to_numpy()
    assert np.isfinite(vals).all()
    assert not np.isnan(vals).any()


def test_edge_case_zero_vectors_and_nan_imputation():
    """Verifies that zero vectors, NaNs, and Infs are safely imputed."""
    record = {
        "step": 1,
        "token": "zero",
        "token_probability": 0.0,
        "entropy": 0.0,
        "normalized_entropy": 0.0,
    }
    for layer in MONITORED_LAYERS:
        record[f"L{layer}_norm"] = 0.0
        record[f"L{layer}_delta"] = 0.0
        record[f"L{layer}_cosine"] = 0.0

    df = enrich_trajectory(pd.DataFrame([record]))
    agg_df = aggregate_token_features(df)

    vals = agg_df[ALL_MODEL_FEATURE_COLUMNS].to_numpy()
    assert np.isfinite(vals).all()


def test_verify_monitored_layers_in_features():
    """
    Verifies that Layers 3, 4, 5, 6 are explicitly represented in the active detector feature matrix.
    Ensures that removing any monitored layer triggers a validation error.
    """
    from trace_lm.features import verify_monitored_layers_in_features

    # Default features must pass
    assert verify_monitored_layers_in_features() is True

    # If any layer (e.g. Layer 4) is omitted, it must fail
    incomplete_features = [col for col in TRAJECTORY_FEATURE_COLUMNS if "L4" not in col]
    with pytest.raises(ValueError, match="missing from active detector features"):
        verify_monitored_layers_in_features(incomplete_features)

