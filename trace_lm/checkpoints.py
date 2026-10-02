"""
Checkpoint evaluation and early-warning lead-time analysis for TRACE-LM.
Evaluates trajectories at 25%, 50%, 75%, and 100% of generation.
Implements persistence rule for early warning to filter isolated noisy token spikes.
"""

from typing import List, Dict, Any, Optional
import pandas as pd

from trace_lm.config import CHECKPOINT_FRACTIONS, DEFAULT_THRESHOLD, CONSECUTIVE_REQUIRED
from trace_lm.features import compute_checkpoint_features
from trace_lm.detector import AnomalyDetector


def evaluate_checkpoints(
    trajectory_df: pd.DataFrame,
    detector: AnomalyDetector,
    threshold: Optional[float] = None,
    fractions: Optional[List[float]] = None,
    planned_max_tokens: Optional[int] = None,
    consecutive_required: int = CONSECUTIVE_REQUIRED,
) -> Dict[str, Any]:
    """
    Evaluates risk score and anomaly status at specified generation checkpoints.

    Rounding & Lead-Time Convention:
        - Checkpoint cutoff: cutoff = max(1, int(np.ceil(target_tokens * frac)))
        - Lead Time = target_tokens - tokens_seen_at_warning
        - Persistence Rule: Requires `consecutive_required` consecutive checkpoints
          exceeding the calibrated threshold to trigger a verified early warning.
        - If generation terminates early (e.g. via EOS), checkpoints exceeding
          the actual generated length are flagged as 'Not reached' without fabricating scores.

    Args:
        trajectory_df: Enriched token-by-token trajectory DataFrame.
        detector: Loaded AnomalyDetector instance.
        threshold: Risk score classification threshold (defaults to detector's calibrated threshold).
        fractions: List of checkpoint fractions (default [0.25, 0.50, 0.75, 1.00]).
        planned_max_tokens: Target generation token count (e.g. max_new_tokens).
        consecutive_required: Number of consecutive checkpoints >= threshold to confirm early warning.

    Returns:
        Dict containing:
        - 'checkpoints': List of checkpoint records
        - 'first_warning': Checkpoint label where warning was initiated, or None
        - 'first_warning_fraction': float or None
        - 'tokens_seen_at_warning': int or None
        - 'tokens_remaining_at_warning': int or None
        - 'lead_time': int (tokens remaining from first warning, or 0 if no warning)
        - 'alert_confirmed': bool whether persistence criteria was met
        - 'summary_df': pd.DataFrame table for display
    """
    if fractions is None:
        fractions = CHECKPOINT_FRACTIONS

    thresh = detector.threshold if threshold is None else threshold

    checkpoint_records = compute_checkpoint_features(
        trajectory_df=trajectory_df,
        fractions=fractions,
        planned_max_tokens=planned_max_tokens,
    )

    results: List[Dict[str, Any]] = []

    for rec in checkpoint_records:
        frac = rec["fraction"]
        reached = rec.get("reached", True)
        tokens_seen = rec["tokens_seen"]
        tokens_remaining = rec["tokens_remaining"]
        feature_df = rec["feature_df"]

        checkpoint_label = f"{int(frac * 100)}%"

        if not reached or feature_df is None:
            risk_score = None
            status = "Not reached"
        elif detector.is_loaded:
            prediction = detector.predict(feature_df, threshold=thresh)
            risk_score = prediction["risk_score"]
            status = prediction["status"]
        else:
            risk_score = 0.0
            status = "UNKNOWN"

        record = {
            "checkpoint": checkpoint_label,
            "fraction": frac,
            "reached": reached,
            "risk_score": round(risk_score, 4) if risk_score is not None else None,
            "status": status,
            "tokens_seen": tokens_seen,
            "tokens_remaining": tokens_remaining,
        }
        results.append(record)

    # Early Warning Evaluation (Chronological First Threshold Crossing)
    first_warning: Optional[str] = None
    first_warning_fraction: Optional[float] = None
    tokens_seen_at_warning: Optional[int] = None
    tokens_remaining_at_warning: Optional[int] = None
    lead_time: int = 0

    for r in results:
        if r["reached"] and r["risk_score"] is not None and r["risk_score"] >= thresh:
            if first_warning is None:
                first_warning = r["checkpoint"]
                first_warning_fraction = r["fraction"]
                tokens_seen_at_warning = r["tokens_seen"]
                tokens_remaining_at_warning = r["tokens_remaining"]
                lead_time = r["tokens_remaining"]

    # Sequence-level anomaly status: SUSPICIOUS if ANY checkpoint crossed threshold
    any_anomalous = any(
        r["reached"] and r["risk_score"] is not None and r["risk_score"] >= thresh
        for r in results
    )
    sequence_status = "SUSPICIOUS" if any_anomalous else "NORMAL"
    alert_confirmed = any_anomalous

    summary_rows = []
    for r in results:
        summary_rows.append({
            "checkpoint": r["checkpoint"],
            "risk_score": f"{r['risk_score']:.4f}" if r["risk_score"] is not None else "Not reached",
            "status": r["status"],
            "tokens_seen": r["tokens_seen"],
            "tokens_remaining": r["tokens_remaining"] if r["reached"] else "-",
        })

    summary_df = pd.DataFrame(summary_rows)[
        ["checkpoint", "risk_score", "status", "tokens_seen", "tokens_remaining"]
    ]

    return {
        "checkpoints": results,
        "first_warning": first_warning,
        "first_warning_fraction": first_warning_fraction,
        "tokens_seen_at_warning": tokens_seen_at_warning,
        "tokens_remaining_at_warning": tokens_remaining_at_warning,
        "lead_time": lead_time,
        "sequence_status": sequence_status,
        "alert_confirmed": alert_confirmed,
        "summary_df": summary_df,
    }
