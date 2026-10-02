"""
Anomaly detector module for TRACE-LM.
Wraps StandardScaler and LogisticRegression for temporal trajectory anomaly detection.
Supports artifact validation and configuration loading from config.json.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from trace_lm.config import (
    DETECTOR_PATH,
    SENTINEL_DETECTOR_PATH,
    SCALER_PATH,
    CONFIG_JSON_PATH,
    DEFAULT_THRESHOLD,
    ACTIVE_DETECTOR_FEATURES,
    MODEL_NAME,
    MONITORED_LAYERS,
    DISCLAIMER,
)

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Lightweight Logistic Regression detector for LLM trajectory anomalies.
    """

    def __init__(
        self,
        detector_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        threshold: Optional[float] = None,
        expected_features: Optional[List[str]] = None,
        validate_on_init: bool = False,
    ):
        # Prefer sentinel_logreg.joblib if it exists, else fallback to detector.joblib
        if detector_path is not None:
            self.detector_path = Path(detector_path)
        elif SENTINEL_DETECTOR_PATH.exists():
            self.detector_path = SENTINEL_DETECTOR_PATH
        else:
            self.detector_path = DETECTOR_PATH

        self.scaler_path = Path(scaler_path) if scaler_path else SCALER_PATH
        self.config_path = Path(config_path) if config_path else CONFIG_JSON_PATH
        self.expected_features = expected_features or ACTIVE_DETECTOR_FEATURES

        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.config_data: Dict[str, Any] = {}
        self.threshold: float = threshold if threshold is not None else DEFAULT_THRESHOLD
        self._is_loaded: bool = False

        self.try_load()

        if validate_on_init and self._is_loaded:
            self.validate_artifacts()

    def try_load(self) -> bool:
        """
        Attempts to load the saved scaler, detector models, and configuration from disk.
        Returns True if successful, False otherwise.
        """
        # 1. Load config.json if available
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config_data = json.load(f)
                if "threshold" in self.config_data:
                    self.threshold = float(self.config_data["threshold"])
                if "feature_cols" in self.config_data:
                    self.expected_features = self.config_data["feature_cols"]
                logger.info("Loaded detector configuration from %s (Threshold: %.4f)", self.config_path, self.threshold)
            except Exception as e:
                logger.warning("Could not read config.json: %s", e)

        # 2. Check model and scaler files
        # Check primary detector path or fallback
        active_detector_path = self.detector_path
        if not active_detector_path.exists() and DETECTOR_PATH.exists():
            active_detector_path = DETECTOR_PATH

        if active_detector_path.exists() and self.scaler_path.exists():
            try:
                self.scaler = joblib.load(self.scaler_path)
                detector_bundle = joblib.load(active_detector_path)

                if isinstance(detector_bundle, dict):
                    self.model = detector_bundle.get("model")
                    if "feature_names" in detector_bundle and not self.config_data:
                        self.expected_features = detector_bundle["feature_names"]
                else:
                    self.model = detector_bundle

                self._is_loaded = True
                self.detector_path = active_detector_path
                logger.info(
                    "Loaded detector from %s and scaler from %s (Features: %d)",
                    self.detector_path,
                    self.scaler_path,
                    len(self.expected_features),
                )
                return True
            except Exception as e:
                logger.warning("Failed to load detector artifacts: %s", e)
                self._is_loaded = False
                return False
        else:
            self._is_loaded = False
            return False

    def validate_artifacts(self) -> bool:
        """
        Validates artifact consistency as required by Section 16:
        - all artifacts exist
        - feature count matches
        - feature names match
        - monitored layers match
        - threshold exists
        - model name matches
        Fails loudly with a descriptive RuntimeError if validation fails.
        """
        if not self.scaler_path.exists():
            raise RuntimeError(f"Missing required scaler artifact at: {self.scaler_path}")
        if not self.detector_path.exists():
            raise RuntimeError(f"Missing required detector artifact at: {self.detector_path}")
        if not self.config_path.exists():
            raise RuntimeError(f"Missing required detector config file at: {self.config_path}")

        if not self.config_data:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config_data = json.load(f)

        # 1. Model name validation
        cfg_model = self.config_data.get("model_name")
        if cfg_model != MODEL_NAME:
            raise RuntimeError(
                f"Model name mismatch in {self.config_path}: expected '{MODEL_NAME}', got '{cfg_model}'"
            )

        # 2. Monitored layers validation
        cfg_layers = self.config_data.get("monitored_layers", [])
        if list(cfg_layers) != list(MONITORED_LAYERS):
            raise RuntimeError(
                f"Monitored layers mismatch in {self.config_path}: expected {MONITORED_LAYERS}, got {cfg_layers}"
            )

        # 3. Threshold existence
        if "threshold" not in self.config_data or self.config_data["threshold"] is None:
            raise RuntimeError(f"Calibrated threshold missing in {self.config_path}")

        # 4. Feature count and names validation
        cfg_features = self.config_data.get("feature_cols", [])
        if len(cfg_features) != len(self.expected_features):
            raise RuntimeError(
                f"Feature count mismatch: config has {len(cfg_features)}, detector expected {len(self.expected_features)}"
            )

        if list(cfg_features) != list(self.expected_features):
            mismatches = [f"cfg={c} vs exp={e}" for c, e in zip(cfg_features, self.expected_features) if c != e]
            raise RuntimeError(f"Feature name/order mismatch: {mismatches[:5]}")

        # 5. Scaler dimension check
        if hasattr(self.scaler, "n_features_in_") and self.scaler.n_features_in_ != len(self.expected_features):
            raise RuntimeError(
                f"Scaler feature dimension mismatch: scaler expects {self.scaler.n_features_in_}, "
                f"config specifies {len(self.expected_features)}"
            )

        logger.info("TRACE-LM artifact validation PASSED successfully.")
        return True

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self.model is not None and self.scaler is not None

    @property
    def feature_cols(self) -> List[str]:
        return list(self.expected_features)

    def validate_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures all expected feature columns are present and free of NaNs/Infs.
        Aligns columns exactly to the order expected by the detector.
        """
        df = feature_df.copy()

        # Fill missing expected features with 0.0
        for col in self.expected_features:
            if col not in df.columns:
                df[col] = 0.0

        # Select only expected features in the exact ordering
        selected = df[self.expected_features].copy()

        # Sanitize NaNs and infinite values
        values = selected.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            selected = selected.fillna(0.0)
            values = selected.to_numpy(dtype=np.float64)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            selected = pd.DataFrame(values, columns=self.expected_features)

        return selected

    def predict(
        self,
        feature_df: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates risk score and classification for the provided feature representation.

        Args:
            feature_df: Single-row or multi-row DataFrame of features.
            threshold: Optional override for anomaly decision threshold.

        Returns:
            Dict containing risk_score, status ('NORMAL' or 'SUSPICIOUS'), threshold, disclaimer.
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Detector artifacts are missing or not loaded. "
                "Run 'python training/train_detector.py' to generate detector, scaler, and config artifacts."
            )

        thresh = self.threshold if threshold is None else threshold
        validated_df = self.validate_features(feature_df)

        scaled_features = self.scaler.transform(validated_df.to_numpy())

        # Risk score is probability of class 1 (anomaly / suspicious)
        risk_score = float(self.model.predict_proba(scaled_features)[0, 1])
        risk_score = max(0.0, min(1.0, risk_score))

        status = "SUSPICIOUS" if risk_score >= thresh else "NORMAL"

        return {
            "risk_score": float(risk_score),
            "status": status,
            "threshold": float(thresh),
            "disclaimer": DISCLAIMER,
        }
