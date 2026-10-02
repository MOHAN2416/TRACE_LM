"""
TRACE-LM: Trajectory Anomaly Checkpoint & Evaluation in LLMs
Streamlit Interactive Research Prototype Application.
"""

import os
import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Configure matplotlib temporary directory for read-only environments
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

# Ensure trace_lm package is on python path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from trace_lm.config import (
    MODEL_NAME,
    MONITORED_LAYERS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_THRESHOLD,
    DETECTOR_PATH,
    SENTINEL_DETECTOR_PATH,
    SCALER_PATH,
    CONFIG_JSON_PATH,
    ACTIVE_DETECTOR_FEATURES,
    RANDOM_SEED,
    DISCLAIMER,
)
from trace_lm.model import load_gpt2_medium, get_device
from trace_lm.detector import AnomalyDetector
from trace_lm.pipeline import TraceLMPipeline

# Page setup
st.set_page_config(
    page_title="TRACE-LM: Trajectory Anomaly Checkpoint & Evaluation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.1rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2563EB;
        margin-bottom: 0.2rem;
    }
    .caption-text {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.4rem;
    }
    .live-badge {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1D4ED8;
        font-size: 0.85rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 4px;
        border: 1px solid #BFDBFE;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .benchmark-badge {
        display: inline-block;
        background-color: #F8FAFC;
        color: #475569;
        font-size: 0.85rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 4px;
        border: 1px solid #CBD5E1;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-title {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .metric-val-normal {
        font-size: 1.85rem;
        font-weight: 800;
        color: #059669;
    }
    .metric-val-suspicious {
        font-size: 1.85rem;
        font-weight: 800;
        color: #DC2626;
    }
    .metric-sub {
        font-size: 0.82rem;
        color: #64748B;
        margin-top: 4px;
    }
    .response-box {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 18px;
        font-family: monospace;
        font-size: 0.95rem;
        color: #0F172A;
        line-height: 1.5;
        white-space: pre-wrap;
        margin-bottom: 10px;
    }
    .disclaimer-box {
        background-color: #FFFBEB;
        border-left: 4px solid #D97706;
        padding: 14px 18px;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #92400E;
        margin-top: 2rem;
        line-height: 1.4;
    }
    .benchmark-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 18px;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing frozen GPT-2-medium backbone into memory...")
def get_cached_backbone():
    """Loads GPT-2-medium and tokenizer once into memory and verifies it is frozen."""
    device = get_device()
    model, tokenizer, device = load_gpt2_medium(device=device)
    return model, tokenizer, device


def get_live_pipeline() -> TraceLMPipeline:
    """Constructs the active TraceLMPipeline with freshly loaded detector artifacts."""
    model, tokenizer, device = get_cached_backbone()
    detector = AnomalyDetector()
    return TraceLMPipeline(
        model=model,
        tokenizer=tokenizer,
        device=device,
        detector=detector,
    )


def compute_reproducibility_fingerprint(pipeline: TraceLMPipeline) -> dict:
    """Computes exact SHA256 hashes, modification timestamps, and configuration for reproducibility audit."""
    import hashlib
    fp = {
        "model_id": MODEL_NAME,
        "monitored_layers": list(MONITORED_LAYERS),
        "random_seed": RANDOM_SEED,
        "feature_count": len(pipeline.detector.feature_cols) if pipeline.detector.is_loaded else 0,
        "calibrated_threshold": float(pipeline.detector.threshold) if pipeline.detector.is_loaded else DEFAULT_THRESHOLD,
        "artifacts": {},
    }
    for name, path in [
        ("detector", SENTINEL_DETECTOR_PATH),
        ("scaler", SCALER_PATH),
        ("config", CONFIG_JSON_PATH),
    ]:
        p = Path(path)
        if p.exists():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))
            fp["artifacts"][name] = {
                "path": str(p),
                "sha256": h,
                "sha256_short": h[:16],
                "mtime": mtime,
                "size_bytes": p.stat().st_size,
            }
        else:
            fp["artifacts"][name] = None
    return fp


# Sidebar Configuration and Model Information
with st.sidebar:
    st.markdown("### 🔬 System Information")
    try:
        pipeline_obj = get_live_pipeline()
        device_str = str(pipeline_obj.device).upper()
        detector_loaded = pipeline_obj.detector.is_loaded
        model_frozen = all(not p.requires_grad for p in pipeline_obj.model.parameters())
    except Exception as e:
        pipeline_obj = None
        device_str = "CPU"
        detector_loaded = False
        model_frozen = True
        st.error(f"Initialization error: {e}")

    st.markdown(
        f"""
        **Project:** `TRACE-LM`  
        **Model:** `{MODEL_NAME}`  
        **Model Status:** `{'Frozen (eval mode, requires_grad=False)' if model_frozen else 'Trainable (ERROR)'}`  
        **Monitored Layers:** `Layers 3, 4, 5, 6`  
        **Detector:** `Logistic Regression (StandardScaler)`  
        **Execution:** `{device_str}`  
        **Detector Artifacts:** `{'Loaded' if detector_loaded else 'Missing / Unavailable'}`  
        """
    )

    if not detector_loaded:
        st.warning(
            "⚠️ Detector artifacts (`sentinel_logreg.joblib`, `scaler.joblib`, `config.json`) not found in `models/`.\n"
            "Run `python training/train_detector.py` to train artifacts."
        )

    # Section 10 & 12: Reproducibility Fingerprint & Diagnostics
    if pipeline_obj is not None:
        fp = compute_reproducibility_fingerprint(pipeline_obj)
        with st.expander("🔐 Reproducibility Fingerprint", expanded=False):
            st.markdown(
                f"""
                **Model ID:** `{fp['model_id']}`  
                **Monitored Layers:** `{fp['monitored_layers']}`  
                **Feature Count:** `{fp['feature_count']} features`  
                **Alert Threshold:** `{fp['calibrated_threshold']:.4f}`  
                **Random Seed:** `{fp['random_seed']}`  
                """
            )
            for k, art in fp["artifacts"].items():
                if art:
                    st.markdown(
                        f"""
                        **`{Path(art['path']).name}`**  
                        - SHA-256: `{art['sha256_short']}...`  
                        - Modified: `{art['mtime']}`  
                        - Size: `{art['size_bytes']} bytes`
                        """
                    )
                else:
                    st.markdown(f"**`{k}`**: *Missing*")

    with st.expander("🔬 Detector Diagnostics", expanded=True):
        if detector_loaded and pipeline_obj and pipeline_obj.detector.config_data:
            cfg = pipeline_obj.detector.config_data
            oof = cfg.get("oof_metrics", {})
            counts = cfg.get("training_dataset_counts", {})
            st.markdown(
                f"""
                **Detector:** `Logistic Regression`  
                **Training:** `Grouped OOF (StratifiedGroupKFold)`  
                **Training Responses:** `{counts.get('total_responses', 'N/A')}`  
                **Anomaly Responses:** `{counts.get('anomaly_responses', 'N/A')}`  
                **Nominal Responses:** `{counts.get('nominal_responses', 'N/A')}`  
                **OOF ROC-AUC:** `{oof.get('roc_auc', 'N/A')}`  
                **OOF Precision:** `{oof.get('precision', 'N/A')}`  
                **OOF Recall:** `{oof.get('recall', 'N/A')}`  
                **OOF F1:** `{oof.get('f1', 'N/A')}`  
                **Calibrated Threshold:** `{pipeline_obj.detector.threshold:.4f}`  
                **Feature Count:** `{len(cfg.get('feature_cols', []))}`  
                **Model:** `{cfg.get('model_name', MODEL_NAME)}`  
                **Monitored Layers:** `{', '.join(str(l) for l in cfg.get('monitored_layers', MONITORED_LAYERS))}`  
                """
            )
        else:
            st.markdown(
                """
                **Detector:** `Logistic Regression`  
                **Training:** `Grouped OOF / Cross-Validation`  
                *(Waiting for artifact loading / training)*
                """
            )

    if st.button("🔄 Reload Detector Artifacts", help="Clears Streamlit cached resources and reloads detector artifacts from models/"):
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Inference Parameters")
    max_tokens = st.slider(
        "Maximum New Tokens",
        min_value=8,
        max_value=128,
        value=DEFAULT_MAX_NEW_TOKENS,
        step=4,
        help="Default is 32 tokens for responsive CPU execution.",
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=1.5,
        value=DEFAULT_TEMPERATURE,
        step=0.05,
    )
    top_p = st.slider(
        "Top-p (Nucleus)",
        min_value=0.1,
        max_value=1.0,
        value=DEFAULT_TOP_P,
        step=0.05,
    )

    calibrated_val = float(pipeline_obj.detector.threshold) if (pipeline_obj and detector_loaded) else DEFAULT_THRESHOLD
    st.caption(f"🎯 **Calibrated Alert Threshold:** `{calibrated_val:.4f}`")
    threshold = st.slider(
        "Detection Threshold",
        min_value=0.10,
        max_value=0.90,
        value=round(calibrated_val, 2),
        step=0.01,
        help="Learned anomaly detection score threshold based on hidden-state trajectory and uncertainty features.",
    )

    st.markdown("---")
    st.caption(
        "ℹ️ **Monitoring Notice:** Token-level features and layer trajectories are extracted "
        "step-by-step during autoregressive generation in the backend. Full trajectory analytics "
        "and checkpoint decisions render immediately upon generation completion."
    )


# Main Interface Header
st.markdown('<div class="main-title">TRACE-LM</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Trajectory Anomaly Checkpoint & Evaluation in LLMs</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="caption-text">Temporal Hidden-State Monitoring for Early Detection of Anomalous LLM Generation</div>',
    unsafe_allow_html=True,
)

# Preset prompt selection or custom prompt
sample_prompts = [
    "What is the capital of France?",
    "Who was the first person to walk on Mars?",
    "Who was the first person to walk on the Moon?",
    "What happens to you if you eat watermelon seeds?",
    "How can I kill a Python process?",
    "What is the largest planet in our solar system?",
    "Explain the theory of relativity briefly.",
    "Custom Prompt...",
]

selected_sample = st.selectbox(
    "Select a demonstration prompt or choose Custom Prompt:",
    sample_prompts,
    index=0,
)

default_prompt_text = "What is the capital of France?" if selected_sample == "Custom Prompt..." else selected_sample

prompt_input = st.text_area(
    "Enter your prompt:",
    value=default_prompt_text,
    height=90,
    help="Enter standard user prompt. Reference answers are never injected during live inference.",
)

col_btn, col_info = st.columns([1, 4])
with col_btn:
    run_clicked = st.button("RUN TRACE-LM", type="primary", width="stretch")

with col_info:
    if pipeline_obj is not None and not pipeline_obj.detector.is_loaded:
        st.warning("Detector artifacts are missing. Generation will execute, but detector risk scoring requires artifacts.")

# Execute Pipeline
if run_clicked:
    if not prompt_input.strip():
        st.warning("Please enter a valid text prompt before clicking RUN TRACE-LM.")
    elif pipeline_obj is None:
        st.error("System pipeline is not initialized. Check model loading logs.")
    else:
        try:
            with st.spinner("Generating tokens & monitoring hidden-state trajectory token-by-token..."):
                result = pipeline_obj.run(
                    prompt=prompt_input,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    threshold=threshold,
                )

            st.markdown("---")
            st.markdown('<div class="live-badge">Live TRACE-LM Detection</div>', unsafe_allow_html=True)

            # Generated Response Display
            st.markdown("#### Generated Response")
            st.markdown(f'<div class="response-box">{result["generated_text"]}</div>', unsafe_allow_html=True)

            st.caption(
                f"⏱️ Generation Time: **{result['generation_time_seconds']}s** | "
                f"🖥️ Execution Device: **{result['device']}** | "
                f"🔤 Tokens Generated: **{result['tokens_generated']}**"
            )

            # Core Live Metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                final_score = result["final_risk"]
                peak_score = result.get("peak_risk", final_score)
                val_class = "metric-val-suspicious" if final_score >= threshold else "metric-val-normal"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">FINAL RISK</div>
                        <div class="{val_class}">{final_score:.4f}</div>
                        <div class="metric-sub">100% Checkpoint | Peak: {peak_score:.4f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with m2:
                seq_status = result["sequence_status"]
                val_class = "metric-val-suspicious" if seq_status == "SUSPICIOUS" else "metric-val-normal"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">SEQUENCE STATUS</div>
                        <div class="{val_class}">{seq_status}</div>
                        <div class="metric-sub">Threshold: {threshold:.2f} (Any Checkpoint)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with m3:
                fw = result["first_warning"]
                fw_display = fw if fw is not None else "None"
                fw_class = "metric-val-suspicious" if fw is not None else "metric-val-normal"
                seen_txt = f"Tokens seen: {result['tokens_seen_at_warning']}" if result['tokens_seen_at_warning'] is not None else "No warning triggered"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">FIRST WARNING</div>
                        <div class="{fw_class}">{fw_display}</div>
                        <div class="metric-sub">{seen_txt}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with m4:
                lead_time = result["lead_time"]
                lt_class = "metric-val-suspicious" if lead_time > 0 else "metric-val-normal"
                rem_txt = f"{lead_time} tokens saved" if lead_time > 0 else "No early warning lead time"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">DETECTION LEAD TIME</div>
                        <div class="{lt_class}">{lead_time} <span style="font-size: 1rem;">tokens</span></div>
                        <div class="metric-sub">{rem_txt}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Checkpoint Analysis Table
            st.markdown("### ⏱️ Checkpoint Analysis")
            checkpoint_df = result["checkpoints_summary"].copy()
            checkpoint_df.columns = ["Checkpoint", "Risk Score", "Status", "Tokens Seen", "Tokens Remaining"]
            st.dataframe(checkpoint_df, hide_index=True)

            # Trajectory Visualizations
            st.markdown("### 📈 Trajectory Visualizations")
            chart_col1, chart_col2 = st.columns(2)

            checkpoints_data = result["checkpoints"]
            reached_chks = [c for c in checkpoints_data if c.get("reached", True) and c["risk_score"] is not None]

            with chart_col1:
                st.markdown("#### Risk Score Trajectory Across Checkpoints")
                if reached_chks:
                    fig_risk, ax_risk = plt.subplots(figsize=(6, 4))
                    chk_fracs = [int(c["fraction"] * 100) for c in reached_chks]
                    chk_scores = [c["risk_score"] for c in reached_chks]

                    ax_risk.plot(chk_fracs, chk_scores, marker="o", color="#2563EB", linewidth=2.2, label="Risk Score")
                    ax_risk.axhline(threshold, color="#DC2626", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold:.2f})")

                    # Highlight first warning if present
                    if result["first_warning_fraction"] is not None:
                        warn_pct = int(result["first_warning_fraction"] * 100)
                        warn_score = next((c["risk_score"] for c in reached_chks if c["fraction"] == result["first_warning_fraction"]), None)
                        if warn_score is not None:
                            ax_risk.scatter([warn_pct], [warn_score], color="#DC2626", s=140, zorder=5, label=f"First Warning ({warn_pct}%)")

                    ax_risk.set_xlabel("Generation Progress (%)", fontsize=10)
                    ax_risk.set_ylabel("Risk Score", fontsize=10)
                    ax_risk.set_title("Live Checkpoint Risk Evolution", fontsize=11, fontweight="bold")
                    ax_risk.set_xticks(chk_fracs)
                    ax_risk.set_xticklabels([f"{p}%" for p in chk_fracs])
                    ax_risk.set_ylim(-0.05, 1.05)
                    ax_risk.grid(True, linestyle=":", alpha=0.5)
                    ax_risk.legend(fontsize=9, loc="best")
                    fig_risk.tight_layout()
                    st.pyplot(fig_risk)
                    plt.close(fig_risk)
                else:
                    st.info("No checkpoints reached to plot.")

            with chart_col2:
                st.markdown("#### Hidden-State Trajectory (Layers 3–6)")
                traj_df = result["trajectory_df"]
                fig_traj, ax_traj = plt.subplots(figsize=(6, 4))

                colors = {"L3": "#2563EB", "L4": "#7C3AED", "L5": "#059669", "L6": "#D97706"}
                for layer in MONITORED_LAYERS:
                    col_name = f"L{layer}_norm"
                    if col_name in traj_df.columns:
                        ax_traj.plot(
                            traj_df["step"],
                            traj_df[col_name],
                            label=f"Layer {layer}",
                            color=colors.get(f"L{layer}", "#64748B"),
                            linewidth=1.8,
                        )

                ax_traj.set_xlabel("Generated Token Step", fontsize=10)
                ax_traj.set_ylabel("L2 Norm of Hidden State", fontsize=10)
                ax_traj.set_title("Layer Hidden-State Norm Dynamics", fontsize=11, fontweight="bold")
                ax_traj.grid(True, linestyle=":", alpha=0.5)
                ax_traj.legend(fontsize=9, loc="best")
                fig_traj.tight_layout()
                st.pyplot(fig_traj)
                plt.close(fig_traj)

            # Token-Level Inspection Expander
            with st.expander("🔍 Token-by-Token Trajectory Log"):
                display_cols = [
                    "step",
                    "token",
                    "token_probability",
                    "entropy",
                    "normalized_entropy",
                    "mean_relative_delta",
                    "mean_layer_norm",
                ]
                avail_cols = [c for c in display_cols if c in traj_df.columns]
                token_display_df = traj_df[avail_cols].copy()
                for flt_col in ["token_probability", "entropy", "normalized_entropy", "mean_relative_delta", "mean_layer_norm"]:
                    if flt_col in token_display_df.columns:
                        token_display_df[flt_col] = token_display_df[flt_col].apply(
                            lambda v: f"{v:.4f}" if pd.notnull(v) else "-"
                        )
                st.dataframe(token_display_df, hide_index=True)

        except Exception as e:
            st.error(f"Inference error encountered: {e}")

# Controlled Research Benchmarks Section (Clearly Separated)
st.markdown("---")
st.markdown('<div class="benchmark-badge">Reference Baseline</div>', unsafe_allow_html=True)
st.markdown("### 📊 Controlled Research Benchmarks")
st.markdown(
    "*The values below represent **reported controlled-condition research results** "
    "from offline experiments on HaluEval QA (using paired correct vs. hallucinated reference prompts) "
    "as documented in the project research presentation. These are baseline reference results and "
    "are not generated by your current live prompt.*"
)

bench_df = pd.DataFrame({
    "Experiment / Checkpoint": [
        "Trajectory-only (All tokens)",
        "Uncertainty-only (All tokens)",
        "All features (Combined)",
        "Early Warning: 25% Checkpoint",
        "Early Warning: 50% Checkpoint",
        "Early Warning: 75% Checkpoint",
        "Early Warning: 100% Checkpoint",
    ],
    "Mean ROC-AUC": ["0.7700", "0.6030", "0.7500", "0.9095", "0.7625", "0.7090", "0.7030"],
    "Std": ["0.0417", "0.0402", "0.0499", "0.0618", "0.1069", "0.0933", "0.0938"],
    "95% Confidence Interval": [
        "[0.7335, 0.8065]",
        "[0.5678, 0.6382]",
        "[0.7063, 0.7937]",
        "[0.8553, 0.9637]",
        "[0.6688, 0.8562]",
        "[0.6272, 0.7908]",
        "[0.6208, 0.7852]",
    ],
    "Reported Mean Tokens Remaining": ["-", "-", "-", "23.0 tokens", "15.0 tokens", "7.0 tokens", "0.0 tokens"],
})
st.dataframe(bench_df, hide_index=True)

st.markdown(
    """
    **Scientific Clarifications:**
    - **ROC-AUC is a ranking discrimination metric, not classification accuracy.** It measures the detector's capability to separate controlled normal versus anomalous representations across decision thresholds.
    - **Origin of 23.0 / 15.0 / 7.0 Lead Times:** In `Untitled34.ipynb`, hidden-state deltas between successive tokens require step $t \\ge 2$. Dropping the initial un-differenced state ($t=1$) leaves 31 usable trajectory steps for a 32-token sequence. Under ceiling rounding: $\\lceil 31 \\times 0.25 \\rceil = 8$ tokens seen, yielding exactly $31 - 8 = 23.0$ remaining tokens in the controlled offline benchmark.
    """
)

# Research Disclaimer Footer
st.markdown(
    f'<div class="disclaimer-box"><strong>Research Disclaimer:</strong> {DISCLAIMER}</div>',
    unsafe_allow_html=True,
)
