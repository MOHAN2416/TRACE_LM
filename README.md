# TRACE-LM: Trajectory Anomaly Checkpoint & Evaluation in LLMs

**Temporal Hidden-State Monitoring for Early Detection of Anomalous LLM Generation**

---

## 1. Project Title
**TRACE-LM**

## 2. Full Form
**T**rajectory **A**nomaly **C**heckpoint & **E**valuation in **L**arge **L**anguage **M**odels

---

## 3. Problem Statement
Large Language Models (LLMs) can generate incorrect, anomalous, or hallucinated content during open-ended text generation. Conventional hallucination and anomaly detection methods evaluate text only *post-hoc*, after the entire response has been generated. This end-of-sequence paradigm wastes computational resources and eliminates opportunities for early intervention, generation termination, or corrective rerouting.

## 4. Objective
TRACE-LM develops a lightweight, token-level monitoring mechanism that tracks the internal temporal evolution of hidden-state representations inside a frozen LLM (GPT-2-medium) during generation. By combining temporal trajectory features with token-level uncertainty metrics, TRACE-LM evaluates generation anomaly risk at designated progress checkpoints (25%, 50%, 75%, 100%) and provides an early-warning signal with measurable detection lead time.

---

## 5. System Architecture

```text
               User Prompt Input
                      ↓
          Frozen GPT-2-medium Backbone
                      ↓
          Token-by-Token Autoregressive Loop
                      ↓
       ┌──────────────────────────────┐
       │   Hidden-State Extraction    │
       │     (Layers 3, 4, 5, 6)      │
       └──────────────┬───────────────┘
                      │
       ┌──────────────┴───────────────┐
       ↓                              ↓
Temporal Trajectory Features   Uncertainty Features
- L2 Hidden Norms              - Token Probability
- Hidden-State Deltas          - Shannon Entropy
- Cosine Similarities          - Normalized Entropy
- Relative State Changes
       └──────────────┬───────────────┘
                      │
       ┌──────────────┴───────────────┐
       │     Feature Aggregation      │
       │   (Mean, Std, Max, Min)      │
       └──────────────┬───────────────┘
                      ↓
        Lightweight Anomaly Detector
       (StandardScaler + LogisticReg)
                      ↓
        Risk Score & Status Decision
             (NORMAL / SUSPICIOUS)
                      ↓
     Checkpoints (25% → 50% → 75% → 100%)
                      ↓
       First Warning & Detection Lead Time
```

---

## 6. Hidden-State Monitoring
- **Backbone Model:** `GPT-2-medium` (355M parameters, 24 transformer layers, hidden dimension $d = 1024$).
- **Model State:** **Completely Frozen**. Verified programmatically:
  $$\forall p \in \text{model.parameters}(),\; p.\text{requires\_grad} = \text{False}$$
- **Monitored Layers:** Layers **3, 4, 5, and 6**.
  - Internal representations at position $t$ are extracted via `output_hidden_states=True`:
    $$\mathbf{h}_t^{(l)} \in \mathbb{R}^{1024} \quad \text{for } l \in \{3, 4, 5, 6\}$$

---

## 7. Temporal Features
For each token step $t \ge 1$ and layer $l \in \{3, 4, 5, 6\}$:
1. **Hidden-State Norm:**
   $$\|\mathbf{h}_t^{(l)}\|_2$$
2. **Hidden-State Delta:**
   $$\Delta_t^{(l)} = \|\mathbf{h}_t^{(l)} - \mathbf{h}_{t-1}^{(l)}\|_2 \quad (t > 1)$$
3. **Cosine Similarity:**
   $$\cos\theta_t^{(l)} = \frac{\mathbf{h}_t^{(l)} \cdot \mathbf{h}_{t-1}^{(l)}}{\|\mathbf{h}_t^{(l)}\|_2 \|\mathbf{h}_{t-1}^{(l)}\|_2} \quad (t > 1)$$
4. **Relative State Change:**
   $$\delta_{\text{rel}, t}^{(l)} = \frac{\Delta_t^{(l)}}{\|\mathbf{h}_{t-1}^{(l)}\|_2 + 10^{-8}}$$
5. **Cross-Layer Dynamics:**
   $$\overline{\delta}_{\text{rel}, t} = \frac{1}{4} \sum_{l=3}^6 \delta_{\text{rel}, t}^{(l)}, \quad \overline{\|\mathbf{h}\|}_t = \frac{1}{4} \sum_{l=3}^6 \|\mathbf{h}_t^{(l)}\|_2$$

Edge cases (first token step $t=1$, missing previous states, zero vectors, NaNs) are imputed with neutral values ($0.0$ for deltas, $1.0$ for cosine similarity) to ensure numerical stability.

---

## 8. Uncertainty Features
For each generated token $y_t$:
1. **Token Probability:**
   $$P(y_t \mid y_{<t}, x) = \text{softmax}(\mathbf{z}_t)_{y_t}$$
2. **Shannon Entropy:**
   $$H(P_t) = -\sum_{v \in V} P(v) \log(P(v) + 10^{-12})$$
3. **Normalized Entropy:**
   $$\widetilde{H}(P_t) = \frac{H(P_t)}{\log(|V|)}$$

---

## 9. Detector
- **Architecture:** `StandardScaler` followed by `LogisticRegression(max_iter=2000, random_state=42)`.
- **Feature Aggregation:** Over the evaluated token trajectory, each base feature column produces 4 summary statistics:
  $$\text{Mean}, \quad \text{Standard Deviation (ddof=1)}, \quad \text{Maximum}, \quad \text{Minimum}$$
  Yielding 40 trajectory feature dimensions (and 12 uncertainty dimensions).
- **Risk Score:** The detector outputs risk probability $P(\text{anomaly}) \in [0, 1]$.
- **Decision Rule:**
  $$\text{Status} = \begin{cases} \text{SUSPICIOUS} & \text{if } \text{Risk Score} \ge \tau \\ \text{NORMAL} & \text{otherwise} \end{cases}$$
  where default threshold $\tau = 0.5$ (configurable in `config.py` and Streamlit UI).
- **Disclaimer:** The risk score is an anomaly detection score under controlled trajectory conditions and is **not** a calibrated probability of factual hallucination.

---

## 10. Checkpoints
Inference is monitored at 4 discrete generation fractions:
$$\text{Checkpoints} \in \{25\%, 50\%, 75\%, 100\%\}$$
For a generation sequence of $N$ total tokens:
$$\text{Cutoff}_k = \max(1, \lceil N \times f_k \rceil), \quad f_k \in \{0.25, 0.50, 0.75, 1.00\}$$
At each checkpoint $k$:
- Tokens seen: $N_{\text{seen}} = \text{Cutoff}_k$
- Tokens remaining: $N_{\text{remaining}} = N - N_{\text{seen}}$
- Detector evaluates the prefix sub-trajectory $\mathbf{h}_{1..\text{Cutoff}_k}$.

---

## 11. Early-Warning Mechanism
The early warning detector identifies the earliest checkpoint $k$ where:
$$\text{Risk Score}_k \ge \tau$$
- If any checkpoint crosses the threshold:
  $$\text{First Warning} = \text{Checkpoint}_k$$
- If no checkpoint crosses the threshold:
  $$\text{First Warning} = \text{None}$$

---

## 12. Lead-Time Calculation
- **Live Inference Definition:**
  Detection lead time measures the number of remaining tokens from the first anomalous warning checkpoint until completion:
  $$\text{Lead Time} = N_{\text{remaining at First Warning}} = N_{\text{total}} - N_{\text{seen at First Warning}}$$
  For example, for a 32-token generation with warning at 25% ($N_{\text{seen}} = 8$), the live lead time is $32 - 8 = 24$ tokens remaining.
  If no warning is triggered, Lead Time is $0$ tokens.

- **Origin of 23.0 / 15.0 / 7.0 / 0.0 in Controlled Research Benchmarks:**
  In the reference research experiments (`Untitled34.ipynb`), hidden-state delta and cosine transitions require at least two consecutive tokens ($t \ge 2$). Dropping the initial un-differenced state ($t=1$) leaves $N_{\text{usable}} = 31$ steps for 32-token sequences. Under ceiling rounding:
  - 25%: $\lceil 31 \times 0.25 \rceil = 8$ tokens seen $\implies 31 - 8 = \mathbf{23.0}$ mean tokens remaining.
  - 50%: $\lceil 31 \times 0.50 \rceil = 16$ tokens seen $\implies 31 - 16 = \mathbf{15.0}$ mean tokens remaining.
  - 75%: $\lceil 31 \times 0.75 \rceil = 24$ tokens seen $\implies 31 - 24 = \mathbf{7.0}$ mean tokens remaining.
  - 100%: $\lceil 31 \times 1.00 \rceil = 31$ tokens seen $\implies 31 - 31 = \mathbf{0.0}$ mean tokens remaining.

---

## 13. Hardware Requirements
TRACE-LM is engineered for local CPU execution on a standard laptop:
- **RAM:** 16 GB RAM.
- **CPU:** Standard 4-core / 8-thread x86_64 CPU (no CUDA required).
- **Storage:** ~3 GB free disk space (GPT-2-medium weights ~1.4 GB).
- **GPU:** Optional. CUDA is detected automatically, falling back safely to CPU.

---

## 14. Installation

```bash
# 1. Clone repository and navigate to folder
cd /home/mohan/Desktop/DL

# 2. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 15. Model Setup
GPT-2-medium is loaded automatically from the Hugging Face Hub on first run and cached locally in `~/.cache/huggingface`. Subsequent executions run completely offline from local cache.

---

## 16. Detector Training (Workflow B)
*Note: The live application does NOT require this step; it runs completely offline using exported artifacts.*

To retrain the detector using multi-benchmark data (HaluEval, TruthfulQA, XSTest) with grouped cross-validation and threshold calibration:
```bash
python training/train_detector.py --halu-samples 15 --tqa-samples 15 --xs-samples 15 --max-new-tokens 24
```
This exports:
- `models/scaler.joblib`: Fitted StandardScaler for 92 trajectory & uncertainty features
- `models/sentinel_logreg.joblib`: Trained balanced Logistic Regression detector
- `models/detector.joblib`: Backward-compatibility bundle
- `models/config.json`: Calibrated threshold, feature definitions, and OOF cross-validation metrics

---

## 17. Running the Live Application (Workflow A)
The live prototype launches with zero dataset downloads:
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 18. Testing & Evaluation Taxonomy

TRACE-LM maintains a strict separation across three evaluation modalities:

### A. Controlled Benchmark Tests
Standardized benchmark prompts evaluated under fixed inference conditions ($T = 0.7$, $p = 0.9$, $\text{max\_new\_tokens} = 32$, threshold $\tau = 0.6900$):
```bash
python tests/test_detector_correction.py
```
This executes the 5 controlled benchmark tests (`BENCHMARK-1` through `BENCHMARK-5`).

### B. Automated Regression Tests
Deterministic unit and integration regression suite covering checkpoint slicing, lead-time calculations, model freezing, and edge cases:
```bash
pytest -v
```
Currently 29/29 passing tests across 6 test modules.

---

## 19. Operational Metrics & Interactive Case Studies

### Metric Definitions
- **`FINAL RISK`:** Anomaly risk score evaluated at the completion of generation (100% checkpoint).
- **`PEAK RISK`:** Maximum risk score observed across any evaluated checkpoint in the sequence ($\max_k \text{Risk}_k$).
- **`SEQUENCE STATUS`:** Trajectory-level decision (`SUSPICIOUS` if **any** checkpoint crossed threshold, else `NORMAL`).
- **`FIRST WARNING`:** Earliest checkpoint crossing the threshold, or `None` if no checkpoint crossed.
- **`DETECTION LEAD TIME`:** Generated tokens remaining from the first warning checkpoint.
- **Early-Warning Persistence:** A `NORMAL` final checkpoint risk does **NOT** erase an earlier warning.

### Interactive Demonstration Case Studies

#### Case 1: Early Anomaly Followed by Normal Later Trajectory
- **Prompt:** `"What is the capital of France?"`
- **Generated Text:** `"There are no capital gains tax rates in France. The main income tax rate is 40%..."`
- **Checkpoints:** 25%: `0.9931` (SUSPICIOUS), 50%: `0.4459` (NORMAL), 75%: `0.4423` (NORMAL), 100%: `0.1817` (NORMAL)
- **Sequence-Level Reporting:**
  - `FINAL RISK:` `0.1817`
  - `PEAK RISK:` `0.9931`
  - `SEQUENCE STATUS:` `SUSPICIOUS`
  - `FIRST WARNING:` `25%`
  - `LEAD TIME:` `24 tokens`

#### Case 2: On-Distribution Nominal Generation
- **Prompt:** `"What is the capital of France?"`
- **Generated Text:** `"Paris is the capital of France. It is a city of great beauty and grandeur..."`
- **Checkpoints:** 25%: `0.2143` (NORMAL), 50%: `0.0863` (NORMAL), 75%: `0.1239` (NORMAL), 100%: `0.1741` (NORMAL)
- **Sequence-Level Reporting:**
  - `FINAL RISK:` `0.1741`
  - `PEAK RISK:` `0.2143`
  - `SEQUENCE STATUS:` `NORMAL`
  - `FIRST WARNING:` `None`
  - `LEAD TIME:` `0 tokens`

---

## 20. Research Results (Reported Controlled Conditions)
In controlled experiments reported in the project research presentation (using HaluEval QA paired with correct vs. hallucinated reference prompts):

| Experiment / Checkpoint | Mean ROC-AUC | Std | 95% Confidence Interval | Mean Tokens Remaining |
|---|---|---|---|---|
| Trajectory-only (All tokens) | 0.7700 | 0.0417 | [0.7335, 0.8065] | - |
| Uncertainty-only (All tokens) | 0.6030 | 0.0402 | [0.5678, 0.6382] | - |
| All Features (Combined) | 0.7500 | 0.0499 | [0.7063, 0.7937] | - |
| **Early Warning: 25% Checkpoint** | **0.9095** | 0.0618 | [0.8553, 0.9637] | **23.0 tokens** |
| Early Warning: 50% Checkpoint | 0.7625 | 0.1069 | [0.6688, 0.8562] | 15.0 tokens |
| Early Warning: 75% Checkpoint | 0.7090 | 0.0933 | [0.6272, 0.7908] | 7.0 tokens |
| Early Warning: 100% Checkpoint | 0.7030 | 0.0938 | [0.6208, 0.7852] | 0.0 tokens |

*Notice: Reported controlled-condition ROC-AUC reflects separation between controlled prompt conditions and is not classification accuracy.*

---

## 21. Limitations
1. **Controlled-Condition Proxy Labels:** Training utilizes contrastive controlled reference completions from HaluEval rather than human-annotated open-ended generation spans.
2. **Uncalibrated Detection Metric:** The risk score is an anomaly detection ranking score, not a calibrated posterior probability of factual correctness.
3. **Single Backbone Model:** Evaluated exclusively on frozen `GPT-2-medium`. Transferability across larger model families (e.g. LLaMA, Mistral) requires further research.
4. **Sampling Sensitivity:** Autoregressive sampling temperature and top-p filtering influence trajectory geometry.

---

## 22. Future Work
1. **Multi-Model Validation:** Benchmark trajectory signatures across LLaMA-3, Mistral-7B, and Gemma.
2. **Adaptive Checkpoints:** Implement dynamic entropy-guided checkpointing rather than uniform fixed percentages.
3. **Real-Time Token Interruption:** Automatically abort or backtrack generation when early warning triggers.
4. **Token-Span Attribution:** Localize specific tokens causing abrupt trajectory spikes.

---

## License
MIT License. Developed for research and educational purposes.
