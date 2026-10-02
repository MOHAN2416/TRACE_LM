"""
Configuration for TRACE-LM (Trajectory Anomaly Checkpoint & Evaluation in LLMs).
Centralizes feature definitions, model settings, and artifact paths.
"""

from pathlib import Path
from typing import List

# Paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = PROJECT_ROOT / "models"
DETECTOR_PATH: Path = MODELS_DIR / "detector.joblib"
SENTINEL_DETECTOR_PATH: Path = MODELS_DIR / "sentinel_logreg.joblib"
SCALER_PATH: Path = MODELS_DIR / "scaler.joblib"
CONFIG_JSON_PATH: Path = MODELS_DIR / "config.json"
DATA_DIR: Path = PROJECT_ROOT / "data"

# Model Configuration
MODEL_NAME: str = "gpt2-medium"
MONITORED_LAYERS: List[int] = [3, 4, 5, 6]

# Inference Hyperparameters
DEFAULT_MAX_NEW_TOKENS: int = 32
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_TOP_P: float = 0.9
DEFAULT_THRESHOLD: float = 0.40  # Default fallback; overridden by calibrated threshold in config.json
RANDOM_SEED: int = 42
CONSECUTIVE_REQUIRED: int = 2
SMOOTHING_ALPHA: float = 0.3

# Checkpoint Fractions
CHECKPOINT_FRACTIONS: List[float] = [0.25, 0.50, 0.75, 1.00]

# Base Token-level Features (23 features)
# Captures Uncertainty, Internal Layer Norms, Velocity, and Cross-Layer Drift
BASE_FEATURE_COLUMNS: List[str] = [
    # 1. Uncertainty Features
    "token_probability",
    "entropy",
    "normalized_entropy",
    # 2. Aggregated Internal Dynamics
    "mean_relative_delta",
    "mean_layer_norm",
    # 3. Monitored Layer Norms (L3-L6)
    "L3_norm",
    "L4_norm",
    "L5_norm",
    "L6_norm",
    # 4. Temporal State Change / Velocity (Deltas)
    "L3_delta",
    "L4_delta",
    "L5_delta",
    "L6_delta",
    # 5. Temporal Alignment (Cosine Similarities with t-1)
    "L3_cosine",
    "L4_cosine",
    "L5_cosine",
    "L6_cosine",
    # 6. Cross-Layer Drift (Norm of Representation Difference between Adjacent Monitored Layers)
    "L3_to_L4_drift",
    "L4_to_L5_drift",
    "L5_to_L6_drift",
    # 7. Cross-Layer Directional Alignment (Cosine Similarity between Adjacent Monitored Layers)
    "L3_to_L4_cosine",
    "L4_to_L5_cosine",
    "L5_to_L6_cosine",
]

STAT_AGGREGATIONS: List[str] = ["mean", "std", "max", "min"]

# Aggregated Response Features (92 features: 23 base features * 4 statistics)
ALL_MODEL_FEATURE_COLUMNS: List[str] = [
    f"{col}_{stat}" for col in BASE_FEATURE_COLUMNS for stat in STAT_AGGREGATIONS
]

# Trajectory Features Subset (80 features: internal norms, deltas, cosines, drifts, relative deltas)
TRAJECTORY_FEATURE_COLUMNS: List[str] = [
    col for col in ALL_MODEL_FEATURE_COLUMNS
    if any(key in col for key in ["delta", "cosine", "norm", "drift", "relative_delta"])
]

# Uncertainty Features Subset (12 features)
UNCERTAINTY_FEATURE_COLUMNS: List[str] = [
    col for col in ALL_MODEL_FEATURE_COLUMNS
    if any(key in col for key in ["token_probability", "entropy"])
]

# Active detector features used by trained Logistic Regression detector
# Centralized definition used by both training and live inference
ACTIVE_DETECTOR_FEATURES: List[str] = ALL_MODEL_FEATURE_COLUMNS

# Disclaimer String
DISCLAIMER: str = (
    "Research prototype. Risk score is a learned anomaly detection score based on "
    "hidden-state trajectory and uncertainty features, not a calibrated probability of factual hallucination."
)
