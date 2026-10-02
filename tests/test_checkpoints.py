"""
Unit tests for TRACE-LM checkpoint analysis and early-warning lead-time calculations.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from trace_lm.config import MONITORED_LAYERS, CHECKPOINT_FRACTIONS
from trace_lm.features import enrich_trajectory
from trace_lm.checkpoints import evaluate_checkpoints


@pytest.fixture
def sample_trajectory_32_tokens():
    """Builds a realistic 32-token trajectory DataFrame."""
    records = []
    for step in range(1, 33):
        row = {
            "step": step,
            "token": f"token_{step}",
            "token_probability": 0.85,
            "entropy": 0.4,
            "normalized_entropy": 0.04,
        }
        for layer in MONITORED_LAYERS:
            row[f"L{layer}_norm"] = 20.0 + step * 0.1
            row[f"L{layer}_delta"] = np.nan if step == 1 else 1.2
            row[f"L{layer}_cosine"] = np.nan if step == 1 else 0.96
        records.append(row)

    return enrich_trajectory(pd.DataFrame(records))


def test_checkpoint_token_counts(sample_trajectory_32_tokens):
    """
    Verifies tokens_seen and tokens_remaining at 25%, 50%, 75%, 100% for 32 tokens:
    25% -> 8 tokens seen, 24 tokens remaining
    50% -> 16 tokens seen, 16 tokens remaining
    75% -> 24 tokens seen, 8 tokens remaining
    100% -> 32 tokens seen, 0 tokens remaining
    """
    mock_detector = MagicMock()
    mock_detector.is_loaded = True
    mock_detector.predict.return_value = {"risk_score": 0.2, "status": "NORMAL"}

    res = evaluate_checkpoints(sample_trajectory_32_tokens, mock_detector, threshold=0.5)
    checkpoints = res["checkpoints"]

    assert len(checkpoints) == 4
    # 25%
    assert checkpoints[0]["checkpoint"] == "25%"
    assert checkpoints[0]["tokens_seen"] == 8
    assert checkpoints[0]["tokens_remaining"] == 24

    # 50%
    assert checkpoints[1]["checkpoint"] == "50%"
    assert checkpoints[1]["tokens_seen"] == 16
    assert checkpoints[1]["tokens_remaining"] == 16

    # 75%
    assert checkpoints[2]["checkpoint"] == "75%"
    assert checkpoints[2]["tokens_seen"] == 24
    assert checkpoints[2]["tokens_remaining"] == 8

    # 100%
    assert checkpoints[3]["checkpoint"] == "100%"
    assert checkpoints[3]["tokens_seen"] == 32
    assert checkpoints[3]["tokens_remaining"] == 0


def test_first_warning_and_lead_time_at_50_percent(sample_trajectory_32_tokens):
    """
    Tests scenario:
    25% -> NORMAL
    50% -> SUSPICIOUS
    75% -> SUSPICIOUS
    100% -> SUSPICIOUS
    Expected: first_warning = '50%', lead_time = 16 tokens.
    """
    mock_detector = MagicMock()
    mock_detector.is_loaded = True

    # Return NORMAL for first call (25%), SUSPICIOUS for subsequent
    mock_detector.predict.side_effect = [
        {"risk_score": 0.3, "status": "NORMAL"},
        {"risk_score": 0.75, "status": "SUSPICIOUS"},
        {"risk_score": 0.8, "status": "SUSPICIOUS"},
        {"risk_score": 0.85, "status": "SUSPICIOUS"},
    ]

    res = evaluate_checkpoints(sample_trajectory_32_tokens, mock_detector, threshold=0.5)

    assert res["first_warning"] == "50%"
    assert res["first_warning_fraction"] == 0.50
    assert res["tokens_seen_at_warning"] == 16
    assert res["tokens_remaining_at_warning"] == 16
    assert res["lead_time"] == 16


def test_first_warning_at_25_percent(sample_trajectory_32_tokens):
    """
    Tests early warning triggered immediately at 25%:
    lead_time = 24 tokens.
    """
    mock_detector = MagicMock()
    mock_detector.is_loaded = True
    mock_detector.predict.return_value = {"risk_score": 0.9, "status": "SUSPICIOUS"}

    res = evaluate_checkpoints(sample_trajectory_32_tokens, mock_detector, threshold=0.5)

    assert res["first_warning"] == "25%"
    assert res["tokens_seen_at_warning"] == 8
    assert res["tokens_remaining_at_warning"] == 24
    assert res["lead_time"] == 24


def test_no_warning_all_normal(sample_trajectory_32_tokens):
    """
    Tests scenario where no checkpoint triggers warning:
    Expected: first_warning = None, lead_time = 0.
    """
    mock_detector = MagicMock()
    mock_detector.is_loaded = True
    mock_detector.predict.return_value = {"risk_score": 0.15, "status": "NORMAL"}

    res = evaluate_checkpoints(sample_trajectory_32_tokens, mock_detector, threshold=0.5)

    assert res["first_warning"] is None
    assert res["tokens_seen_at_warning"] is None
    assert res["tokens_remaining_at_warning"] is None
    assert res["lead_time"] == 0


def test_early_eos_checkpoints_not_reached(sample_trajectory_32_tokens):
    """
    Tests scenario where generation stops early (e.g. only 6 tokens generated
    when planned_max_tokens was 32).
    Checkpoints requiring more than 6 tokens must be flagged as 'Not reached'.
    """
    short_trajectory = sample_trajectory_32_tokens.iloc[:6].copy()
    mock_detector = MagicMock()
    mock_detector.is_loaded = True
    mock_detector.predict.return_value = {"risk_score": 0.2, "status": "NORMAL"}

    res = evaluate_checkpoints(
        short_trajectory,
        mock_detector,
        threshold=0.5,
        planned_max_tokens=32,
    )

    checkpoints = res["checkpoints"]
    assert len(checkpoints) == 4

    # 25% requires 8 tokens -> 6 < 8 -> Not reached
    assert checkpoints[0]["reached"] is False
    assert checkpoints[0]["status"] == "Not reached"
    assert checkpoints[0]["risk_score"] is None

    # Summary table should reflect 'Not reached'
    summary_df = res["summary_df"]
    assert summary_df.loc[0, "status"] == "Not reached"
    assert summary_df.loc[0, "risk_score"] == "Not reached"


def test_regression_early_anomaly_followed_by_normal_trajectory(sample_trajectory_32_tokens):
    """
    Automated Regression Test: Early anomaly followed by normal-looking trajectory.
    Matches the observed interactive demonstration run:
    Prompt: 'What is the capital of France?'
    Observed response: 'There are no capital gains tax rates in France. The main income tax rate is 40%...'
    Checkpoints:
    25%  = 0.9931 (SUSPICIOUS)
    50%  = 0.4459 (NORMAL)
    75%  = 0.4423 (NORMAL)
    100% = 0.1817 (NORMAL)
    Threshold = 0.6900

    Expected sequence-level reporting:
    Final Risk (100% checkpoint) = 0.1817
    Sequence Status = SUSPICIOUS
    First Warning = 25%
    Lead Time = 24 tokens
    """
    mock_detector = MagicMock()
    mock_detector.is_loaded = True
    mock_detector.threshold = 0.6900
    mock_detector.predict.side_effect = [
        {"risk_score": 0.9931, "status": "SUSPICIOUS"},
        {"risk_score": 0.4459, "status": "NORMAL"},
        {"risk_score": 0.4423, "status": "NORMAL"},
        {"risk_score": 0.1817, "status": "NORMAL"},
    ]

    res = evaluate_checkpoints(
        sample_trajectory_32_tokens,
        mock_detector,
        threshold=0.6900,
        planned_max_tokens=32,
    )

    assert res["first_warning"] == "25%"
    assert res["first_warning_fraction"] == 0.25
    assert res["tokens_seen_at_warning"] == 8
    assert res["tokens_remaining_at_warning"] == 24
    assert res["lead_time"] == 24
    assert res["sequence_status"] == "SUSPICIOUS"
    assert res["checkpoints"][-1]["risk_score"] == 0.1817


def test_regression_persistent_anomaly(sample_trajectory_32_tokens):
    """
    Automated Regression Test: Persistent anomaly across checkpoints.
    Checkpoints:
    25%  = 0.9999 (SUSPICIOUS)
    50%  = 0.9962 (SUSPICIOUS)
    75%  = 0.8585 (SUSPICIOUS)
    100% = 0.9402 (SUSPICIOUS)
    Threshold = 0.6900

    Expected:
    Final Risk (100% checkpoint) = 0.9402
    Sequence Status = SUSPICIOUS
    First Warning = 25%
    Lead Time = 24 tokens
    """
    mock_detector = MagicMock()
    mock_detector.is_loaded = True
    mock_detector.threshold = 0.6900
    mock_detector.predict.side_effect = [
        {"risk_score": 0.9999, "status": "SUSPICIOUS"},
        {"risk_score": 0.9962, "status": "SUSPICIOUS"},
        {"risk_score": 0.8585, "status": "SUSPICIOUS"},
        {"risk_score": 0.9402, "status": "SUSPICIOUS"},
    ]

    res = evaluate_checkpoints(
        sample_trajectory_32_tokens,
        mock_detector,
        threshold=0.6900,
        planned_max_tokens=32,
    )

    assert res["first_warning"] == "25%"
    assert res["lead_time"] == 24
    assert res["sequence_status"] == "SUSPICIOUS"
    assert res["checkpoints"][-1]["risk_score"] == 0.9402


def test_regression_all_nominal_checkpoints(sample_trajectory_32_tokens):
    """
    Automated Regression Test: All checkpoints below threshold (No anomaly).
    All checkpoint scores below calibrated threshold (0.6900).

    Expected:
    Sequence Status = NORMAL
    First Warning = None
    Lead Time = 0
    """
    mock_detector = MagicMock()
    mock_detector.is_loaded = True
    mock_detector.threshold = 0.6900
    mock_detector.predict.return_value = {"risk_score": 0.05, "status": "NORMAL"}

    res = evaluate_checkpoints(
        sample_trajectory_32_tokens,
        mock_detector,
        threshold=0.6900,
        planned_max_tokens=32,
    )

    assert res["first_warning"] is None
    assert res["lead_time"] == 0
    assert res["sequence_status"] == "NORMAL"


