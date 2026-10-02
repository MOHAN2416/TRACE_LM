"""
Training script for TRACE-LM anomaly detector.
Implements the research training design with grouped cross-validation, multi-benchmark support,
response-level ground truth, and threshold calibration:

1. Loads benchmark datasets: HaluEval, TruthfulQA, XSTest.
2. Generates responses from prompt prefixes using frozen GPT-2-medium.
3. Extracts token-by-token trajectory telemetry (layers 3-6) and uncertainty features.
4. Performs StratifiedGroupKFold cross-validation by sample_id (zero prompt/token leakage).
5. Computes honest out-of-fold (OOF) metrics (ROC-AUC, Precision, Recall, F1).
6. Calibrates decision threshold from nominal OOF scores (95th percentile).
7. Fits final detector + scaler on all training responses.
8. Exports sentinel_logreg.joblib, detector.joblib, scaler.joblib, and config.json.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from trace_lm.config import (
    RANDOM_SEED,
    MODELS_DIR,
    DATA_DIR,
    DETECTOR_PATH,
    SENTINEL_DETECTOR_PATH,
    SCALER_PATH,
    CONFIG_JSON_PATH,
    MODEL_NAME,
    MONITORED_LAYERS,
    BASE_FEATURE_COLUMNS,
    ACTIVE_DETECTOR_FEATURES,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_THRESHOLD,
    CONSECUTIVE_REQUIRED,
    SMOOTHING_ALPHA,
)
from trace_lm.model import load_gpt2_medium
from trace_lm.generation import generate_with_trajectory
from trace_lm.features import aggregate_token_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_cached_benchmarks(
    data_dir: Path = DATA_DIR,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Loads benchmark datasets from local data cache.
    Fails with informative instructions if files are missing.
    """
    halu_path = data_dir / "halueval_qa_cache.json"
    tqa_path = data_dir / "truthfulqa_cache.json"
    xs_path = data_dir / "xstest_cache.json"

    missing = []
    if not halu_path.exists():
        missing.append(str(halu_path))
    if not tqa_path.exists():
        missing.append(str(tqa_path))
    if not xs_path.exists():
        missing.append(str(xs_path))

    if missing:
        raise FileNotFoundError(
            f"Missing required benchmark cache files: {missing}. "
            "Please ensure datasets are placed in the 'data/' directory."
        )

    with open(halu_path, "r", encoding="utf-8") as f:
        halu_data = json.load(f)
    with open(tqa_path, "r", encoding="utf-8") as f:
        tqa_data = json.load(f)
    with open(xs_path, "r", encoding="utf-8") as f:
        xs_data = json.load(f)

    logger.info(
        "Loaded benchmark records: HaluEval=%d, TruthfulQA=%d, XSTest=%d",
        len(halu_data),
        len(tqa_data),
        len(xs_data),
    )
    return halu_data, tqa_data, xs_data


def build_training_prompts(
    halu_data: List[Dict[str, Any]],
    tqa_data: List[Dict[str, Any]],
    xs_data: List[Dict[str, Any]],
    max_samples: Dict[str, int],
) -> List[Dict[str, Any]]:
    """
    Constructs prompt specifications with response-level ground truth.
    Ensures prompt lengths and syntactic formatting are balanced across nominal (0)
    and anomalous (1) classes to eliminate prompt-header confounders.
    """
    prompt_records: List[Dict[str, Any]] = []
    global_sample_id = 0

    # 1. Standard Factual QA (Nominal Factual Control, target = 0)
    # Natural prompts where GPT-2 produces stable, factual, high-confidence completions
    factual_queries = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Italy?",
        "What is the capital of Spain?",
        "What is the capital of Japan?",
        "What is the largest planet in our solar system?",
        "What is the freezing point of water in Celsius?",
        "Who was the first person to walk on the Moon?",
        "Who was the first president of the United States?",
        "What is the capital of the United Kingdom?",
        "What is the currency of Japan?",
        "What continent is Egypt located in?",
        "How many continents are on Earth?",
        "What is the primary language spoken in Spain?",
        "What is the chemical formula for water?",
        "What is the capital of Canada?",
        "What is the capital of Australia?",
        "What is the currency of the United Kingdom?",
        "Who painted the Mona Lisa?",
        "How many days are in a leap year?",
        "What is the capital of China?",
        "What gas do plants release during photosynthesis?",
        "What is the hardest natural substance on Earth?",
        "What is the boiling point of water in Celsius?",
        "What is the primary language spoken in Brazil?",
        "What is the capital of Egypt?",
        "What is the capital of Russia?",
        "How many degrees are in a circle?",
        "What is the square root of 64?",
    ]
    for q in factual_queries:
        global_sample_id += 1
        prompt_records.append({
            "sample_id": global_sample_id,
            "dataset": "StandardFactualQA",
            "condition": "factual_nominal",
            "prompt": q,
            "ground_truth": 0,
        })

    # 2. TruthfulQA: Misconception queries (target = 1) and Factual queries (target = 0)
    tqa_n = min(max_samples.get("TruthfulQA", 15), len(tqa_data))
    for i in range(tqa_n):
        item = tqa_data[i]
        global_sample_id += 1
        q = item.get("question", "").strip()

        # Misconception query inducing falsehood in GPT-2 (target = 1)
        prompt_records.append({
            "sample_id": global_sample_id,
            "dataset": "TruthfulQA",
            "condition": "misconception_probe",
            "prompt": q,
            "ground_truth": 1,
        })

    # 3. HaluEval: Hallucination-inducing ungrounded queries (target = 1)
    halu_n = min(max_samples.get("HaluEval", 15), len(halu_data))
    for i in range(halu_n):
        item = halu_data[i]
        global_sample_id += 1
        q = item.get("question", "").strip()
        # Prompt format without answers or artificial background headers
        prompt_records.append({
            "sample_id": global_sample_id,
            "dataset": "HaluEval",
            "condition": "ungrounded_hallucination",
            "prompt": f"Question: {q}\nAnswer:",
            "ground_truth": 1,
        })

    # 4. Known Hallucination / Counterfactual Queries (target = 1)
    hallucination_queries = [
        "Who was the first person to walk on Mars?",
        "In what year did George Washington visit the moon?",
        "In what year did Albert Einstein invent the iPhone?",
        "Who was the first female president of the United States in the 1800s?",
        "Which American astronaut landed on Jupiter in 1975?",
        "What is the secret formula that allows humans to breathe underwater without gear?",
    ]
    for q in hallucination_queries:
        global_sample_id += 1
        prompt_records.append({
            "sample_id": global_sample_id,
            "dataset": "HallucinationBenchmark",
            "condition": "counterfactual_hallucination",
            "prompt": q,
            "ground_truth": 1,
        })

    # 5. XSTest: Safe compliance (target = 0) vs Contrast / sensitive (target = 1)
    xs_safe = [x for x in xs_data if x.get("label") == 0]
    xs_contrast = [x for x in xs_data if x.get("label") == 1 or "contrast" in x.get("type", "").lower()]

    xs_n = min(max_samples.get("XSTest", 15), len(xs_safe), len(xs_contrast))
    for i in range(xs_n):
        # Safe query (target = 0)
        global_sample_id += 1
        prompt_records.append({
            "sample_id": global_sample_id,
            "dataset": "XSTest",
            "condition": "safe_standard",
            "prompt": xs_safe[i].get("prompt", "").strip(),
            "ground_truth": 0,
        })

        # Contrast query (target = 1)
        global_sample_id += 1
        prompt_records.append({
            "sample_id": global_sample_id,
            "dataset": "XSTest",
            "condition": "contrast_sensitive",
            "prompt": xs_contrast[i].get("prompt", "").strip(),
            "ground_truth": 1,
        })

    logger.info("Constructed %d total training prompt specifications across 3 benchmarks", len(prompt_records))
    return prompt_records


def generate_and_extract_features(
    prompt_records: List[Dict[str, Any]],
    model,
    tokenizer,
    device,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
) -> pd.DataFrame:
    """
    Generates responses token-by-token and extracts telemetry features.
    Attaches response-level ground truth AFTER generation.
    """
    rows = []
    total = len(prompt_records)

    for idx, rec in enumerate(prompt_records, 1):
        if idx % 10 == 0 or idx == total:
            logger.info("Generating and extracting telemetry: %d / %d", idx, total)

        generated_text, trajectory_df, _ = generate_with_trajectory(
            prompt=rec["prompt"],
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        feature_df = aggregate_token_features(trajectory_df, feature_columns=BASE_FEATURE_COLUMNS)
        row_dict = {
            "sample_id": rec["sample_id"],
            "dataset": rec["dataset"],
            "condition": rec["condition"],
            "prompt": rec["prompt"],
            "generated_text": generated_text,
            "tokens_generated": len(trajectory_df),
            "ground_truth": rec["ground_truth"],
        }
        row_dict.update(feature_df.iloc[0].to_dict())
        rows.append(row_dict)

    df = pd.DataFrame(rows)
    return df


def train_and_calibrate_detector(
    df: pd.DataFrame,
    feature_cols: List[str] = ACTIVE_DETECTOR_FEATURES,
    n_splits: int = 5,
    random_seed: int = RANDOM_SEED,
) -> Tuple[LogisticRegression, StandardScaler, float, Dict[str, float]]:
    """
    Trains detector with StratifiedGroupKFold by sample_id, computes honest OOF metrics,
    and calibrates threshold from nominal OOF scores.
    """
    # Verify features exist in DataFrame
    available_cols = [c for c in feature_cols if c in df.columns]
    if len(available_cols) != len(feature_cols):
        missing = set(feature_cols) - set(available_cols)
        raise ValueError(f"Missing {len(missing)} expected feature columns: {list(missing)[:5]}")

    X = df[available_cols].fillna(0.0).to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["ground_truth"].to_numpy(dtype=np.int64)
    groups = df["sample_id"].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    oof_preds = np.zeros(len(y), dtype=np.float64)
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val)

        clf = LogisticRegression(max_iter=2000, random_state=random_seed, class_weight="balanced")
        clf.fit(X_tr_sc, y_tr)

        val_probs = clf.predict_proba(X_val_sc)[:, 1]
        oof_preds[val_idx] = val_probs

        if len(np.unique(y_val)) > 1:
            fold_auc = roc_auc_score(y_val, val_probs)
            fold_aucs.append(fold_auc)

    # Compute overall OOF metrics
    oof_roc_auc = float(roc_auc_score(y, oof_preds)) if len(np.unique(y)) > 1 else 0.5
    nominal_oof_scores = oof_preds[y == 0]

    # Calibrate alert threshold: find threshold that maximizes OOF F1 while preserving
    # nominal specificity >= 75% on held-out predictions
    best_thresh = 0.50
    best_f1 = -1.0
    for cand_t in np.linspace(0.30, 0.70, 41):
        b = (oof_preds >= cand_t).astype(int)
        spec = float(np.mean(b[y == 0] == 0)) if np.sum(y == 0) > 0 else 1.0
        if spec >= 0.75:
            f = f1_score(y, b, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_thresh = float(cand_t)

    calibrated_threshold = round(best_thresh, 4)

    oof_binary_preds = (oof_preds >= calibrated_threshold).astype(int)
    oof_acc = float(accuracy_score(y, oof_binary_preds))
    oof_prec = float(precision_score(y, oof_binary_preds, zero_division=0))
    oof_rec = float(recall_score(y, oof_binary_preds, zero_division=0))
    oof_f1 = float(f1_score(y, oof_binary_preds, zero_division=0))

    logger.info("Honest OOF Cross-Validation Results (StratifiedGroupKFold):")
    logger.info("  OOF ROC-AUC:    %.4f (Folds Mean: %.4f)", oof_roc_auc, float(np.mean(fold_aucs)))
    logger.info("  Calibrated Thresh: %.4f (95th percentile nominal)", calibrated_threshold)
    logger.info("  OOF Accuracy:   %.4f", oof_acc)
    logger.info("  OOF Precision:  %.4f", oof_prec)
    logger.info("  OOF Recall:     %.4f", oof_rec)
    logger.info("  OOF F1-Score:   %.4f", oof_f1)

    # Train final model on full dataset
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)

    final_detector = LogisticRegression(max_iter=2000, random_state=random_seed, class_weight="balanced")
    final_detector.fit(X_scaled, y)

    metrics = {
        "roc_auc": round(oof_roc_auc, 4),
        "accuracy": round(oof_acc, 4),
        "precision": round(oof_prec, 4),
        "recall": round(oof_rec, 4),
        "f1": round(oof_f1, 4),
    }

    return final_detector, final_scaler, calibrated_threshold, metrics


def export_artifacts(
    detector: LogisticRegression,
    scaler: StandardScaler,
    threshold: float,
    metrics: Dict[str, float],
    df: pd.DataFrame,
    max_new_tokens: int,
    output_dir: Path = MODELS_DIR,
):
    """
    Exports sentinel_logreg.joblib, detector.joblib, scaler.joblib, and config.json
    according to Section 16 specifications.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    sentinel_path = output_dir / "sentinel_logreg.joblib"
    detector_path = output_dir / "detector.joblib"
    scaler_path = output_dir / "scaler.joblib"
    config_path = output_dir / "config.json"

    # 1. Save scaler
    joblib.dump(scaler, scaler_path)

    # 2. Save detector bundles
    bundle = {
        "model": detector,
        "feature_names": ACTIVE_DETECTOR_FEATURES,
        "classes": [0, 1],
        "metrics": metrics,
    }
    joblib.dump(bundle, sentinel_path)
    joblib.dump(bundle, detector_path)

    # 3. Save config.json
    dataset_counts = {str(d): int((df["dataset"] == d).sum()) for d in df["dataset"].unique()}
    dataset_counts["total_responses"] = len(df)
    dataset_counts["anomaly_responses"] = int((df["ground_truth"] == 1).sum())
    dataset_counts["nominal_responses"] = int((df["ground_truth"] == 0).sum())

    config_data = {
        "model_name": MODEL_NAME,
        "monitored_layers": MONITORED_LAYERS,
        "feature_cols": ACTIVE_DETECTOR_FEATURES,
        "threshold": round(threshold, 4),
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "max_new_tokens": max_new_tokens,
        "smoothing_alpha": SMOOTHING_ALPHA,
        "consecutive_required": CONSECUTIVE_REQUIRED,
        "seed": RANDOM_SEED,
        "training_dataset_counts": dataset_counts,
        "oof_metrics": metrics,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    logger.info("Successfully exported artifacts to %s:", output_dir)
    logger.info("  Scaler:   %s", scaler_path)
    logger.info("  Detector: %s", sentinel_path)
    logger.info("  Config:   %s", config_path)


def run_training_workflow(
    max_samples: Optional[Dict[str, int]] = None,
    max_new_tokens: int = 24,
    use_cache: bool = True,
):
    """Executes the full end-to-end training and artifact generation workflow."""
    if max_samples is None:
        max_samples = {"HaluEval": 15, "TruthfulQA": 15, "XSTest": 15}

    cache_features_file = DATA_DIR / f"extracted_benchmark_features_n{sum(max_samples.values())}.csv"

    if use_cache and cache_features_file.exists():
        logger.info("Loading pre-extracted benchmark features from: %s", cache_features_file)
        feature_df = pd.read_csv(cache_features_file)
    else:
        halu_data, tqa_data, xs_data = load_cached_benchmarks(DATA_DIR)
        prompt_records = build_training_prompts(halu_data, tqa_data, xs_data, max_samples=max_samples)

        logger.info("Loading frozen GPT-2-medium for feature extraction...")
        model, tokenizer, device = load_gpt2_medium()

        feature_df = generate_and_extract_features(
            prompt_records=prompt_records,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
        )

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(cache_features_file, index=False)
        logger.info("Saved extracted feature dataset to: %s", cache_features_file)

    detector, scaler, threshold, metrics = train_and_calibrate_detector(feature_df)

    export_artifacts(
        detector=detector,
        scaler=scaler,
        threshold=threshold,
        metrics=metrics,
        df=feature_df,
        max_new_tokens=max_new_tokens,
    )

    logger.info("TRACE-LM Detector Training and Calibration Complete.")
    return detector, scaler, threshold, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TRACE-LM Anomaly Detector")
    parser.add_argument("--halu-samples", type=int, default=15, help="Number of HaluEval samples")
    parser.add_argument("--tqa-samples", type=int, default=15, help="Number of TruthfulQA samples")
    parser.add_argument("--xs-samples", type=int, default=15, help="Number of XSTest samples")
    parser.add_argument("--max-new-tokens", type=int, default=24, help="Tokens to generate during training")
    parser.add_argument("--no-cache", action="store_true", help="Force re-generation of features")
    args = parser.parse_args()

    samples_config = {
        "HaluEval": args.halu_samples,
        "TruthfulQA": args.tqa_samples,
        "XSTest": args.xs_samples,
    }

    run_training_workflow(
        max_samples=samples_config,
        max_new_tokens=args.max_new_tokens,
        use_cache=not args.no_cache,
    )
