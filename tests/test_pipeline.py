"""
Unit and integration tests for TRACE-LM end-to-end pipeline.
Verifies complete execution flow and dataset independence during live inference.
"""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import torch

from trace_lm.pipeline import TraceLMPipeline
from trace_lm.detector import AnomalyDetector
from trace_lm.config import DISCLAIMER


@pytest.fixture
def mock_pipeline():
    """Builds a mock pipeline to test end-to-end orchestration without heavy generation."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_detector = MagicMock(spec=AnomalyDetector)
    mock_detector.is_loaded = True
    mock_detector.threshold = 0.5
    mock_detector.predict.return_value = {
        "risk_score": 0.35,
        "status": "NORMAL",
        "threshold": 0.5,
        "disclaimer": DISCLAIMER,
    }

    # Pipeline with mocks
    with patch("trace_lm.pipeline.load_gpt2_medium"):
        pipeline = TraceLMPipeline(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device=torch.device("cpu"),
            detector=mock_detector,
        )
    return pipeline


def test_pipeline_output_fields(mock_pipeline):
    """Verifies that pipeline returns all fields expected by the Streamlit frontend."""
    # Mock generation output
    fake_records = [
        {
            "step": i,
            "token": f"tok_{i}",
            "token_id": i,
            "token_probability": 0.8,
            "entropy": 0.5,
            "normalized_entropy": 0.05,
            "L3_norm": 20.0,
            "L4_norm": 20.0,
            "L5_norm": 20.0,
            "L6_norm": 20.0,
            "L3_delta": 1.0,
            "L4_delta": 1.0,
            "L5_delta": 1.0,
            "L6_delta": 1.0,
            "L3_cosine": 0.95,
            "L4_cosine": 0.95,
            "L5_cosine": 0.95,
            "L6_cosine": 0.95,
            "L3_to_L4_drift": 1.0,
            "L4_to_L5_drift": 1.0,
            "L5_to_L6_drift": 1.0,
            "L3_to_L4_cosine": 0.95,
            "L4_to_L5_cosine": 0.95,
            "L5_to_L6_cosine": 0.95,
        }
        for i in range(1, 17)
    ]
    fake_df = pd.DataFrame(fake_records)

    with patch("trace_lm.pipeline.generate_with_trajectory", return_value=("Paris.", fake_df, None)):
        result = mock_pipeline.run("What is the capital of France?", max_new_tokens=16)

    assert result["prompt"] == "What is the capital of France?"
    assert result["generated_text"] == "Paris."
    assert result["tokens_generated"] == 16
    assert "generation_time_seconds" in result
    assert "device" in result
    assert "overall_risk_score" in result
    assert "overall_status" in result
    assert "first_warning" in result
    assert "lead_time" in result
    assert "checkpoints" in result
    assert "checkpoints_summary" in result
    assert "trajectory_df" in result
    assert result["disclaimer"] == DISCLAIMER


def test_live_prototype_does_not_require_dataset():
    """
    Requirement 3 & 19: The LIVE prototype must NOT require downloading or loading
    HaluEval, SQuAD, TruthfulQA, or any dataset when detector artifacts exist.
    """
    mock_detector = MagicMock(spec=AnomalyDetector)
    mock_detector.is_loaded = True
    mock_detector.predict.return_value = {"risk_score": 0.2, "status": "NORMAL"}

    with patch("trace_lm.pipeline.load_gpt2_medium"):
        pipeline = TraceLMPipeline(
            model=MagicMock(),
            tokenizer=MagicMock(),
            device=torch.device("cpu"),
            detector=mock_detector,
        )

    # Verify no dataset loading modules/functions are invoked in pipeline.py
    import inspect
    pipeline_code = inspect.getsource(pipeline.run)
    assert "halueval" not in pipeline_code.lower()
    assert "requests.get" not in pipeline_code
    assert "qa_data.json" not in pipeline_code


def test_pipeline_early_warning_preserved(mock_pipeline):
    """
    Verifies that pipeline separates final_risk from peak_risk,
    and preserves sequence_status = 'SUSPICIOUS' when an early checkpoint is anomalous.
    """
    mock_pipeline.detector.predict.side_effect = [
        {"risk_score": 0.9931, "status": "SUSPICIOUS"},  # 25%
        {"risk_score": 0.4459, "status": "NORMAL"},      # 50%
        {"risk_score": 0.4423, "status": "NORMAL"},      # 75%
        {"risk_score": 0.1817, "status": "NORMAL"},      # 100%
    ]
    mock_pipeline.detector.threshold = 0.6900

    fake_records = [
        {
            "step": i,
            "token": f"tok_{i}",
            "token_id": i,
            "token_probability": 0.8,
            "entropy": 0.5,
            "normalized_entropy": 0.05,
            "L3_norm": 20.0, "L4_norm": 20.0, "L5_norm": 20.0, "L6_norm": 20.0,
            "L3_delta": 1.0, "L4_delta": 1.0, "L5_delta": 1.0, "L6_delta": 1.0,
            "L3_cosine": 0.95, "L4_cosine": 0.95, "L5_cosine": 0.95, "L6_cosine": 0.95,
            "L3_to_L4_drift": 1.0, "L4_to_L5_drift": 1.0, "L5_to_L6_drift": 1.0,
            "L3_to_L4_cosine": 0.95, "L4_to_L5_cosine": 0.95, "L5_to_L6_cosine": 0.95,
        }
        for i in range(1, 33)
    ]
    fake_df = pd.DataFrame(fake_records)

    with patch("trace_lm.pipeline.generate_with_trajectory", return_value=("Sample answer", fake_df, None)):
        result = mock_pipeline.run("Test prompt", max_new_tokens=32, threshold=0.6900)

    # Final risk must be the 100% checkpoint risk (0.1817), NOT peak risk (0.9931)
    assert result["final_risk"] == 0.1817
    assert result["overall_risk_score"] == 0.1817
    assert result["peak_risk"] == 0.9931

    # Sequence status must be SUSPICIOUS because 25% crossed the threshold
    assert result["sequence_status"] == "SUSPICIOUS"
    assert result["overall_status"] == "SUSPICIOUS"

    # First warning must be 25% with 24 tokens lead time
    assert result["first_warning"] == "25%"
    assert result["lead_time"] == 24
    assert result["tokens_seen_at_warning"] == 8
    assert result["tokens_remaining_at_warning"] == 24
