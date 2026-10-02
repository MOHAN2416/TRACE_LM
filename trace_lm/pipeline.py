"""
End-to-end execution pipeline for TRACE-LM.
Connects prompt input, token-by-token generation, hidden-state monitoring,
checkpoint analysis, and early-warning detection.
Implements response-level aggregation strategy (max risk over trajectory checkpoints).
"""

import time
import logging
from typing import Dict, Any, Optional, Callable
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from trace_lm.config import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_THRESHOLD,
    CONSECUTIVE_REQUIRED,
    DISCLAIMER,
)
from trace_lm.model import load_gpt2_medium, get_device
from trace_lm.generation import generate_with_trajectory
from trace_lm.features import aggregate_token_features
from trace_lm.detector import AnomalyDetector
from trace_lm.checkpoints import evaluate_checkpoints

logger = logging.getLogger(__name__)


class TraceLMPipeline:
    """
    Unified pipeline for TRACE-LM research prototype inference.
    """

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        device: Optional[torch.device] = None,
        detector: Optional[AnomalyDetector] = None,
        consecutive_required: int = CONSECUTIVE_REQUIRED,
    ):
        self.device = device or get_device()

        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.model, self.tokenizer, self.device = load_gpt2_medium(device=self.device)

        self.detector = detector or AnomalyDetector()
        self.consecutive_required = consecutive_required

    def run(
        self,
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        threshold: Optional[float] = None,
        step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inference on a user prompt:
        Prompt -> Frozen GPT-2 -> Token generation & state extraction ->
        Checkpoint evaluation -> Anomaly & early-warning detection.

        Args:
            prompt: Text prompt string.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p nucleus sampling threshold.
            threshold: Risk score decision boundary (defaults to detector calibrated threshold).
            step_callback: Optional callback for streaming updates.

        Returns:
            Dictionary containing comprehensive results.
        """
        start_time = time.time()

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        detector_thresh = getattr(self.detector, "threshold", DEFAULT_THRESHOLD)
        effective_threshold = detector_thresh if threshold is None else threshold

        # Step 1: Token-by-token generation with trajectory tracking
        generated_text, trajectory_df, _ = generate_with_trajectory(
            prompt=prompt.strip(),
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            step_callback=step_callback,
        )

        generation_time = time.time() - start_time
        num_tokens = len(trajectory_df)

        # Step 2: Compute response-level features (100% completion)
        response_feature_df = aggregate_token_features(trajectory_df)

        # Step 3: Checkpoint evaluation (25%, 50%, 75%, 100%)
        checkpoint_results = evaluate_checkpoints(
            trajectory_df=trajectory_df,
            detector=self.detector,
            threshold=effective_threshold,
            planned_max_tokens=max_new_tokens,
            consecutive_required=self.consecutive_required,
        )

        # Step 4: Sequence-level decision strategy
        reached_scores = [
            c["risk_score"] for c in checkpoint_results["checkpoints"]
            if c.get("reached", True) and c["risk_score"] is not None
        ]

        if not self.detector.is_loaded:
            overall_risk_score = 0.0
            final_checkpoint_risk = 0.0
            max_checkpoint_risk = 0.0
            sequence_status = "DETECTOR_NOT_LOADED"
            overall_status = "DETECTOR_NOT_LOADED"
        elif reached_scores:
            max_checkpoint_risk = float(max(reached_scores))
            final_checkpoint_risk = float(reached_scores[-1])
            # Sequence-level decision strategy:
            # If ANY checkpoint crosses the calibrated threshold, sequence_status is SUSPICIOUS.
            sequence_status = checkpoint_results.get(
                "sequence_status",
                "SUSPICIOUS" if any(s >= effective_threshold for s in reached_scores) else "NORMAL"
            )
            overall_status = sequence_status
            # Final risk score is strictly the risk score at the 100% (final reached) checkpoint.
            # Do NOT make the final risk score equal to the maximum risk score.
            overall_risk_score = final_checkpoint_risk
        else:
            # Short sequence where no checkpoint was reached
            prediction = self.detector.predict(response_feature_df, threshold=effective_threshold)
            final_checkpoint_risk = float(prediction["risk_score"])
            max_checkpoint_risk = float(prediction["risk_score"])
            overall_risk_score = final_checkpoint_risk
            sequence_status = prediction["status"]
            overall_status = sequence_status

        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "tokens_generated": num_tokens,
            "generation_time_seconds": round(generation_time, 3),
            "device": str(self.device),
            "final_risk": round(final_checkpoint_risk, 4),
            "final_checkpoint_risk": round(final_checkpoint_risk, 4),
            "peak_risk": round(max_checkpoint_risk, 4),
            "max_checkpoint_risk": round(max_checkpoint_risk, 4),
            "overall_risk_score": round(overall_risk_score, 4),
            "sequence_status": sequence_status,
            "overall_status": overall_status,
            "threshold": effective_threshold,
            "first_warning": checkpoint_results["first_warning"],
            "first_warning_fraction": checkpoint_results["first_warning_fraction"],
            "tokens_seen_at_warning": checkpoint_results["tokens_seen_at_warning"],
            "tokens_remaining_at_warning": checkpoint_results["tokens_remaining_at_warning"],
            "lead_time": checkpoint_results["lead_time"],
            "alert_confirmed": checkpoint_results.get("alert_confirmed", sequence_status == "SUSPICIOUS"),
            "checkpoints": checkpoint_results["checkpoints"],
            "checkpoints_summary": checkpoint_results["summary_df"],
            "trajectory_df": trajectory_df,
            "response_feature_df": response_feature_df,
            "disclaimer": DISCLAIMER,
        }
