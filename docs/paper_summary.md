# Paper → code map

**Paper:** Explainability-Based Wireless Intrusion Detection System: An Agreement-Based Confidence Framework
Using SHAP and LIME. Pramith Sagar V, Yashwanth D N, Sachin U S, Sudarshan G R. T. John Institute of
Technology, Bengaluru.

| Paper | What it says | Code | Notes |
|---|---|---|---|
| III-A.1 | 802.11 frames → features, Min-Max, SMOTE, SHAP-guided selection to 16–17 features | `frames.py`, `features.py`, `prep.py` | The paper mentions CICFlowMeter's 154 flow features. CICFlowMeter reads IP flows, not 802.11 MAC frames, so we compute 33 frame-level window features (the kind the paper's Section II-C describes) and keep 17. |
| III-A.2 | RF primary + XGBoost secondary, soft vote; Isolation Forest for zero-day | `models.py` | Isolation Forest is trained on Normal windows only. Its 5% flag is shown as a hint; escalation needs the 99.5th percentile (ours). |
| III-A.3 | TreeSHAP + tabular LIME | `explain.py` | Ensemble SHAP = weighted sum of member SHAP (valid because soft voting is linear). |
| III-A.4 | τ = 0.85; ≥ τ auto-resolved with SHAP archived; < τ escalated with SHAP + LIME | `triage.py` | Escalation rate = escalated / alerts. |
| III-A.5 | Dashboard, TP/FP verdicts, retrain every 50 labels, MLflow | `app/`, `feedback.py` | New model goes live only if validation macro-F1 does not drop by more than 0.5 points (ours). |
| III-B | ABEC: J = Jaccard of top-k SHAP / LIME; tiers High / Single-source / Disputed | `explain.abec` | "Negligible" = normalised score below 0.10 (ours). A top-k feature is Disputed if the other method finds it negligible or gives it the opposite sign. |
| IV-A | RF 500 trees depth 20; XGB 300, lr 0.05; IF contamination 0.05; stratified 10-fold CV | `config.yaml`, `03_evaluate.py --cv` | CV folds are group-aware (no temporal leakage). |
| IV-B | Tables 1, 2 | `03_evaluate.py` | Only datasets that are actually run can be reported. |
| IV-C | Table 3 fidelity / stability / latency; Table 4 J by outcome | `03_evaluate.py` | Stability is reported as instability = 1 − mean pairwise Jaccard of top-5 sets over 10 runs. |
| IV-D | Table 5 active learning in batches of 50 | `04_active_learning.py` | The analyst is simulated by ground-truth labels (an oracle); say so in the paper. |
| IV-E | Latency / throughput | `03_evaluate.py` | Measured on your machine. |
| V-D | Run ABEC on demand for escalated alerts | `app/service.py` | Explanations are computed when an alert is escalated or opened. |
