"""
Feature extraction, trajectory computation, and statistical aggregation for TRACE-LM.
Preserves the exact feature engineering formulas from the research notebook.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch

from trace_lm.config import (
    MONITORED_LAYERS,
    BASE_FEATURE_COLUMNS,
    STAT_AGGREGATIONS,
    ALL_MODEL_FEATURE_COLUMNS,
    TRAJECTORY_FEATURE_COLUMNS,
    UNCERTAINTY_FEATURE_COLUMNS,
    ACTIVE_DETECTOR_FEATURES,
    CHECKPOINT_FRACTIONS,
)


def extract_token_uncertainty(
    probs: torch.Tensor,
    token_id: int,
) -> Tuple[float, float, float]:
    """
    Computes token probability, entropy, and normalized entropy for a given token.

    Args:
        probs: Softmax probability distribution tensor of shape (1, vocab_size).
        token_id: Selected token id integer.

    Returns:
        (token_probability, entropy, normalized_entropy)
    """
    token_probability = float(probs[0, token_id].item())
    entropy = float(-torch.sum(probs * torch.log(probs + 1e-12)).item())

    vocab_size = max(probs.shape[-1], 2)
    normalized_entropy = float(entropy / np.log(vocab_size))

    # Defensive bounds check
    token_probability = max(0.0, min(1.0, token_probability))
    entropy = max(0.0, entropy)
    normalized_entropy = max(0.0, min(1.0, normalized_entropy))

    return token_probability, entropy, normalized_entropy


def compute_layer_hidden_metrics(
    current_hidden: Dict[int, torch.Tensor],
    previous_hidden: Dict[int, Optional[torch.Tensor]],
) -> Dict[str, float]:
    """
    Computes norms, deltas, and cosine similarities for monitored layers.

    Args:
        current_hidden: Dict mapping layer index to 1D hidden state tensor.
        previous_hidden: Dict mapping layer index to previous 1D hidden state tensor (or None).

    Returns:
        Dict containing L{layer}_norm, L{layer}_delta, L{layer}_cosine for each monitored layer.
    """
    metrics: Dict[str, float] = {}

    for layer in MONITORED_LAYERS:
        if layer in current_hidden:
            h = current_hidden[layer]
            norm = float(torch.norm(h).item())
            metrics[f"L{layer}_norm"] = norm

            prev = previous_hidden.get(layer)
            if prev is None:
                metrics[f"L{layer}_delta"] = np.nan
                metrics[f"L{layer}_cosine"] = np.nan
            else:
                delta = float(torch.norm(h - prev).item())
                cosine = float(
                    torch.nn.functional.cosine_similarity(
                        h.unsqueeze(0),
                        prev.unsqueeze(0)
                    ).item()
                )
                metrics[f"L{layer}_delta"] = delta
                metrics[f"L{layer}_cosine"] = cosine

    # Cross-layer drift and cosine alignment across adjacent monitored layers
    layer_pairs = [(3, 4), (4, 5), (5, 6)]
    for l_src, l_tgt in layer_pairs:
        if l_src in current_hidden and l_tgt in current_hidden:
            h_src = current_hidden[l_src]
            h_tgt = current_hidden[l_tgt]
            drift_norm = float(torch.norm(h_tgt - h_src).item())
            drift_cosine = float(
                torch.nn.functional.cosine_similarity(
                    h_tgt.unsqueeze(0),
                    h_src.unsqueeze(0)
                ).item()
            )
            metrics[f"L{l_src}_to_L{l_tgt}_drift"] = drift_norm
            metrics[f"L{l_src}_to_L{l_tgt}_cosine"] = drift_cosine
        else:
            metrics[f"L{l_src}_to_L{l_tgt}_drift"] = np.nan
            metrics[f"L{l_src}_to_L{l_tgt}_cosine"] = np.nan

    return metrics


def enrich_trajectory(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived trajectory metrics:
    - relative_delta_L{layer} = L{layer}_delta / (L{layer}_norm_{t-1} + 1e-8)
    - mean_relative_delta
    - mean_layer_norm

    Args:
        trajectory_df: Raw token-by-token trajectory DataFrame.

    Returns:
        Enriched DataFrame with cross-layer trajectory metrics.
    """
    df = trajectory_df.copy()

    for layer in MONITORED_LAYERS:
        norm_col = f"L{layer}_norm"
        delta_col = f"L{layer}_delta"
        rel_col = f"relative_delta_L{layer}"

        if norm_col in df.columns and delta_col in df.columns:
            prev_norm = df[norm_col].shift(1)
            df[rel_col] = df[delta_col] / (prev_norm + 1e-8)
        else:
            df[rel_col] = np.nan

    rel_cols = [f"relative_delta_L{layer}" for layer in MONITORED_LAYERS if f"relative_delta_L{layer}" in df.columns]
    norm_cols = [f"L{layer}_norm" for layer in MONITORED_LAYERS if f"L{layer}_norm" in df.columns]

    df["mean_relative_delta"] = df[rel_cols].mean(axis=1)
    df["mean_layer_norm"] = df[norm_cols].mean(axis=1)

    return df


def aggregate_token_features(
    trajectory_df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Computes mean, std, max, and min for all feature columns over the given trajectory.
    Handles first-token edge cases, short trajectories, and NaN/Inf safely.

    Args:
        trajectory_df: Trajectory DataFrame (should be enriched).
        feature_columns: Base feature columns to aggregate (defaults to BASE_FEATURE_COLUMNS).

    Returns:
        Single-row DataFrame containing aggregated features.
    """
    if feature_columns is None:
        feature_columns = BASE_FEATURE_COLUMNS

    df = trajectory_df.copy()

    # If df does not have derived columns, enrich it
    if "mean_relative_delta" not in df.columns:
        df = enrich_trajectory(df)

    # In the research notebook, usable rows drop NaNs (step 1 has NaN deltas)
    cols_to_check = [col for col in feature_columns if col in df.columns]
    usable = df[cols_to_check].dropna().reset_index(drop=True)

    # Edge case: If 1 token or all rows had NaNs, impute NaNs with neutral defaults
    if len(usable) == 0:
        usable = df[cols_to_check].copy()
        for col in cols_to_check:
            if "cosine" in col:
                usable[col] = usable[col].fillna(1.0)
            elif "drift" in col:
                usable[col] = usable[col].fillna(0.0)
            elif "delta" in col:
                usable[col] = usable[col].fillna(0.0)
            elif "norm" in col:
                usable[col] = usable[col].fillna(1.0)
            else:
                usable[col] = usable[col].fillna(0.0)

    aggregated: Dict[str, float] = {}

    for col in feature_columns:
        if col in usable.columns and len(usable[col]) > 0:
            series = usable[col]
            mean_val = float(series.mean())
            std_val = float(series.std(ddof=1)) if len(series) > 1 else 0.0
            max_val = float(series.max())
            min_val = float(series.min())
        else:
            mean_val = 0.0
            std_val = 0.0
            max_val = 0.0
            min_val = 0.0

        # Replace NaN / Inf with 0.0
        mean_val = 0.0 if np.isnan(mean_val) or np.isinf(mean_val) else mean_val
        std_val = 0.0 if np.isnan(std_val) or np.isinf(std_val) else std_val
        max_val = 0.0 if np.isnan(max_val) or np.isinf(max_val) else max_val
        min_val = 0.0 if np.isnan(min_val) or np.isinf(min_val) else min_val

        aggregated[f"{col}_mean"] = mean_val
        aggregated[f"{col}_std"] = std_val
        aggregated[f"{col}_max"] = max_val
        aggregated[f"{col}_min"] = min_val

    aggregated["num_tokens"] = len(usable)

    result_df = pd.DataFrame([aggregated])
    return result_df


def verify_monitored_layers_in_features(
    feature_columns: Optional[List[str]] = None,
) -> bool:
    """
    Verifies programmatically that all monitored layers (3, 4, 5, 6) are explicitly
    represented and used in the active feature matrix.

    Args:
        feature_columns: Feature names to check (defaults to ACTIVE_DETECTOR_FEATURES).

    Returns:
        True if all monitored layers are present, raises ValueError otherwise.
    """
    cols = feature_columns or ACTIVE_DETECTOR_FEATURES
    missing_layers = []
    for layer in MONITORED_LAYERS:
        layer_repr = f"L{layer}"
        has_layer = any(layer_repr in col for col in cols)
        if not has_layer:
            missing_layers.append(layer)

    if missing_layers:
        raise ValueError(
            f"Monitored layers {missing_layers} are missing from active detector features!"
        )
    return True


def compute_checkpoint_features(
    trajectory_df: pd.DataFrame,
    fractions: Optional[List[float]] = None,
    feature_columns: Optional[List[str]] = None,
    planned_max_tokens: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Computes prefix feature representations for specified generation checkpoints (e.g. 25%, 50%, 75%, 100%).
    Handles short generations and early EOS termination by flagging unreached checkpoints.

    Rounding Convention:
        cutoff = max(1, int(np.ceil(target_tokens * frac)))
        Consistent with the research implementation in Untitled34.ipynb.

    Args:
        trajectory_df: Complete token-by-token trajectory DataFrame.
        fractions: List of fraction cutoffs (defaults to [0.25, 0.50, 0.75, 1.00]).
        feature_columns: Base feature columns to aggregate.
        planned_max_tokens: Optional planned sequence length (e.g., max_new_tokens).
            If provided, determines whether checkpoints were reached when generation terminates early.

    Returns:
        List of dicts, each with fraction, cutoff token count, reached boolean, and feature DataFrame.
    """
    if fractions is None:
        fractions = CHECKPOINT_FRACTIONS
    if feature_columns is None:
        feature_columns = BASE_FEATURE_COLUMNS

    df = enrich_trajectory(trajectory_df)
    total_tokens = len(df)
    target_tokens = planned_max_tokens if planned_max_tokens is not None else total_tokens

    checkpoint_records = []

    for frac in fractions:
        # Standardized ceiling rounding convention
        cutoff = max(1, int(np.ceil(target_tokens * frac)))

        if total_tokens < cutoff:
            # Checkpoint was not reached due to early EOS or short generation
            checkpoint_records.append({
                "fraction": frac,
                "reached": False,
                "cutoff": cutoff,
                "tokens_seen": total_tokens,
                "tokens_remaining": 0,
                "feature_df": None,
            })
        else:
            prefix_slice = df.iloc[:cutoff].copy()
            feature_df = aggregate_token_features(prefix_slice, feature_columns=feature_columns)

            checkpoint_records.append({
                "fraction": frac,
                "reached": True,
                "cutoff": cutoff,
                "tokens_seen": cutoff,
                "tokens_remaining": max(0, target_tokens - cutoff),
                "feature_df": feature_df,
            })

    return checkpoint_records

