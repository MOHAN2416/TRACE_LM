"""
Unit tests for TRACE-LM anomaly detector (StandardScaler + LogisticRegression).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

from trace_lm.config import (
    TRAJECTORY_FEATURE_COLUMNS,
    ACTIVE_DETECTOR_FEATURES,
    DEFAULT_THRESHOLD,
)
from trace_lm.detector import AnomalyDetector


@pytest.fixture
def mock_detector_artifacts(tmp_path):
    """
    Creates temporary mock scaler, detector, and config artifacts for unit testing.
    """
    import json
    scaler_path = tmp_path / "scaler.joblib"
    detector_path = tmp_path / "detector.joblib"
    config_path = tmp_path / "config.json"

    np.random.seed(42)
    n_features = len(ACTIVE_DETECTOR_FEATURES)
    X = np.random.randn(20, n_features)
    y = np.random.randint(0, 2, size=20)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression()
    clf.fit(X_scaled, y)

    joblib.dump(scaler, scaler_path)
    joblib.dump({"model": clf, "feature_names": ACTIVE_DETECTOR_FEATURES}, detector_path)

    config_data = {
        "model_name": "gpt2-medium",
        "monitored_layers": [3, 4, 5, 6],
        "feature_cols": ACTIVE_DETECTOR_FEATURES,
        "threshold": 0.45,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f)

    return detector_path, scaler_path, config_path


def test_detector_initialization_and_loading(mock_detector_artifacts):
    """Verifies that AnomalyDetector correctly loads saved artifacts."""
    detector_path, scaler_path, config_path = mock_detector_artifacts
    detector = AnomalyDetector(detector_path=detector_path, scaler_path=scaler_path, config_path=config_path)

    assert detector.is_loaded is True
    assert detector.model is not None
    assert detector.scaler is not None
    assert len(detector.expected_features) == len(ACTIVE_DETECTOR_FEATURES)
    assert detector.threshold == 0.45


def test_detector_predict_output_structure(mock_detector_artifacts):
    """Verifies predict returns risk_score, status, threshold, and disclaimer."""
    detector_path, scaler_path, config_path = mock_detector_artifacts
    detector = AnomalyDetector(detector_path=detector_path, scaler_path=scaler_path, config_path=config_path)

    feature_data = {col: [1.0] for col in ACTIVE_DETECTOR_FEATURES}
    feature_df = pd.DataFrame(feature_data)

    res = detector.predict(feature_df, threshold=0.5)

    assert "risk_score" in res
    assert "status" in res
    assert "threshold" in res
    assert "disclaimer" in res
    assert 0.0 <= res["risk_score"] <= 1.0
    assert res["status"] in ["NORMAL", "SUSPICIOUS"]


def test_detector_threshold_logic(mock_detector_artifacts):
    """Verifies classification flips correctly at threshold boundary."""
    detector_path, scaler_path, config_path = mock_detector_artifacts
    detector = AnomalyDetector(detector_path=detector_path, scaler_path=scaler_path, config_path=config_path)

    feature_data = {col: [0.0] for col in ACTIVE_DETECTOR_FEATURES}
    feature_df = pd.DataFrame(feature_data)

    # If threshold is 0.0, any score >= 0 is SUSPICIOUS
    res_low = detector.predict(feature_df, threshold=0.0)
    assert res_low["status"] == "SUSPICIOUS"

    # If threshold is 1.0001, score < threshold is NORMAL
    res_high = detector.predict(feature_df, threshold=1.0001)
    assert res_high["status"] == "NORMAL"


def test_detector_handles_nan_inf_features(mock_detector_artifacts):
    """Verifies that passing NaNs and Infs does not crash detector and produces valid result."""
    detector_path, scaler_path, config_path = mock_detector_artifacts
    detector = AnomalyDetector(detector_path=detector_path, scaler_path=scaler_path, config_path=config_path)

    feature_data = {col: [np.nan if i % 2 == 0 else np.inf] for i, col in enumerate(ACTIVE_DETECTOR_FEATURES)}
    feature_df = pd.DataFrame(feature_data)

    res = detector.predict(feature_df)
    assert 0.0 <= res["risk_score"] <= 1.0
    assert res["status"] in ["NORMAL", "SUSPICIOUS"]


def test_validate_artifacts_success(mock_detector_artifacts):
    """Verifies that validate_artifacts passes when all files and metadata match."""
    detector_path, scaler_path, config_path = mock_detector_artifacts
    detector = AnomalyDetector(detector_path=detector_path, scaler_path=scaler_path, config_path=config_path)
    assert detector.validate_artifacts() is True


def test_validate_artifacts_fails_on_model_mismatch(mock_detector_artifacts, tmp_path):
    """Verifies that validate_artifacts fails loudly on model mismatch."""
    import json
    detector_path, scaler_path, config_path = mock_detector_artifacts
    bad_config_path = tmp_path / "bad_config.json"
    bad_data = {
        "model_name": "other-model",
        "monitored_layers": [3, 4, 5, 6],
        "feature_cols": ACTIVE_DETECTOR_FEATURES,
        "threshold": 0.45,
    }
    with open(bad_config_path, "w") as f:
        json.dump(bad_data, f)

    detector = AnomalyDetector(detector_path=detector_path, scaler_path=scaler_path, config_path=bad_config_path)
    with pytest.raises(RuntimeError, match="Model name mismatch"):
        detector.validate_artifacts()


def test_missing_artifacts_message(tmp_path):
    """Verifies clear error when artifacts are absent."""
    missing_detector = tmp_path / "nonexistent_detector.joblib"
    missing_scaler = tmp_path / "nonexistent_scaler.joblib"

    detector = AnomalyDetector(detector_path=missing_detector, scaler_path=missing_scaler)
    assert detector.is_loaded is False

    with pytest.raises(RuntimeError, match="Run 'python training/train_detector.py'"):
        detector.predict(pd.DataFrame())
