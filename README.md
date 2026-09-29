# XWIDS: Explainability-Based Wireless Intrusion Detection System

Working code for the paper *"Explainability-Based Wireless Intrusion Detection System: An Agreement-Based
Confidence Framework Using SHAP and LIME"* (Pramith Sagar V, Yashwanth D N, Sachin U S, Sudarshan G R,
T. John Institute of Technology).

It includes an analyst dashboard, a phone-friendly **Net Doctor** Wi-Fi console, and one-command deployment
to Hugging Face.

---

## The idea in one picture (explained simply)

Imagine a security guard (the AI) watching Wi-Fi traffic. When it shouts "attack!", a human boss asks
*"why do you think so?"*. We ask **two different explainers** (SHAP and LIME) to answer.
If they agree, we trust the reason. If they disagree, that is a warning sign, and a human looks closely.

```
Wi-Fi frames ──► cut into 1-second windows per access point ──► 33 measurements ──► keep the best 17
                                                                                        │
             ┌──────────────────────────────────────────────────────────────────────────┘
             ▼
   Random Forest (500 trees) + XGBoost (300 trees)  ──► "Deauthentication, 93% sure"
   Isolation Forest (trained on normal traffic)     ──► "this looks unusual"  (zero-day hint)
             │
             ▼
   Triage (τ = 0.85):   sure enough? ──yes──► AUTO: closed, SHAP reason saved
                             │
                             no
                             ▼
   ESCALATE to analyst with SHAP + LIME + ABEC
        ABEC:  J = |top-5 SHAP ∩ top-5 LIME| / |union|
               High confidence (both agree) · Single-source · Disputed (they disagree)
             │
             ▼
   Analyst clicks "True positive" / "False positive" ──► every 50 answers the model retrains itself
```

## What is in the box

| Part | Where | Paper section |
|---|---|---|
| AWID3 CSV loader (tolerant of tshark column names) | `xwids/frames.py` | III-A stage 1 |
| 802.11 traffic simulator (for testing without AWID3) | `xwids/simulate.py` | – |
| 33 window features → SHAP-guided selection of 17 | `xwids/features.py`, `xwids/prep.py` | III-A stage 1 |
| Min-Max scaling, SMOTE (train only), leak-free split | `xwids/prep.py`, `xwids/models.py` | III-A stage 1 |
| RF 500/depth 20 + XGBoost 300/lr 0.05 soft vote + Isolation Forest 0.05 | `xwids/models.py` | III-A stage 2, IV-A |
| TreeSHAP + LIME | `xwids/explain.py` | III-A stage 3 |
| **ABEC** (Jaccard J, three tiers) | `xwids/explain.py → abec()` | III-B, Eq. 1 |
| Triage τ = 0.85 | `xwids/triage.py` | III-A stage 4 |
| Analyst dashboard, verdicts, retrain every 50 labels, MLflow | `app/`, `xwids/feedback.py` | III-A stage 5 |
| Tables 1–4 (detection, per class, explanation quality, J by outcome) | `scripts/03_evaluate.py` | IV-B, IV-C |
| Table 5 (active-learning loop) | `scripts/04_active_learning.py` | IV-D |
| Net Doctor console (Dash, Diag, Reco, Zone, Devices, Timeline, Forecast) | `app/doctor.py` | extra |
| Docker, Hugging Face Space / Hub, Render | `Dockerfile`, `scripts/07_hf_push.py` | V-D |

## Run it on your laptop (Windows)

You need **Python 3.11** (from python.org; tick "Add to PATH") and this folder unzipped.

```powershell
cd path\to\xwids
.\setup.ps1          # one time: makes a private Python environment and installs everything (5-10 min)
.\run_demo.ps1       # opens the dashboard at http://127.0.0.1:5000
```

If PowerShell refuses to run scripts, first run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Log in with **analyst / xwids-demo**, press **Replay**, then open the **Analyst queue**.
On your phone (same Wi-Fi, or through the public link) open **/doctor/** for Net Doctor.

The first `run_demo.ps1` builds the synthetic dataset and trains the model (about 3–5 minutes). After that it starts instantly.

## Use the real dataset (AWID3)

1. Register (free, for research) at https://icsdweb.aegean.gr/awid/awid3 and download the **CSV** version.
2. Unzip the CSV folders into `data\raw\awid3\` (any sub-folders are fine).
3. Check that one file reads correctly:
   `python scripts\06_inspect_csv.py data\raw\awid3\<some folder>\<some file>.csv`
4. Run the pipeline:

```powershell
python scripts\01_prepare.py --dataset awid3       # features + SMOTE + SHAP selection
python scripts\02_train.py   --dataset awid3       # RF + XGBoost + Isolation Forest
python scripts\03_evaluate.py --dataset awid3 --n-explain 200 --cv   # Tables 1-4 -> reports\awid3\RESULTS.md
python scripts\04_active_learning.py --dataset awid3 --jaccard       # Table 5 -> reports\awid3\ACTIVE_LEARNING.md
.\run_demo.ps1 -Dataset awid3
```

AWID3 is very large, so `max_rows_per_file` in `config.yaml` limits how much is read (to fit in 16 GB RAM).

## Put it online (Hugging Face)

The zip already contains a ready `deploy/` folder (synthetic model + data, with its exact library versions
pinned), so you can push a working Space right away. Run `05_export_deploy.py` again later to ship an AWID3 model.

```powershell
pip install huggingface_hub ; huggingface-cli login
python scripts\05_export_deploy.py --dataset synthetic            # optional now; needed after training on awid3
$env:XWIDS_USERS="analyst:<strong-password>"
python scripts\07_hf_push.py --dataset synthetic --what space --repo <your-hf-name>/xwids-demo
```

`docs/DEPLOY.md` covers the other options: Render, Docker, and a public link straight from your laptop.

## Important: honest numbers

* The **synthetic** results in `reports/synthetic/` only prove the code runs. **Never** put them in the paper.
* The paper's tables (98.1% accuracy, FPR 4.2% → 1.9%, escalation 100% → 16%, SHAP 3.7 ms…) must be
  **replaced by what `03_evaluate.py` and `04_active_learning.py` measure on AWID3**. If a number cannot be
  reproduced, it must not be reported.
* A real finding already: on the synthetic data, SHAP and LIME agreement is **not** lower for misclassified
  windows (the paper's Table 4 hypothesis). Test it on AWID3 and report whichever way it comes out.
* Without the `shap` / `lime` / `xgboost` packages the code still runs, using built-in versions of the same
  algorithms. `requirements.txt` installs the real ones, and every report says which backend was used.
