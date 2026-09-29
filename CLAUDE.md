# CLAUDE.md: XWIDS (Explainability-Based Wireless IDS, ABEC framework)

You are working in the XWIDS repository with a student who has **no coding background**. Explain what you do
in plain words, like to a beginner. The student pastes steps from `CLAUDE_CODE_STEPS.md` one at a time. For
each step: do it, run its **check**, fix failures at the cause (never loosen a check or fake a number), then
add a line to `docs/CHANGELOG.md`. Keep going inside a step without asking unless it says **ASK**.

## What this project is
Implementation of the paper in `docs/paper_summary.md`: RF + XGBoost + Isolation Forest on 802.11 window
features (AWID3), SHAP + LIME fused by ABEC (Jaccard J, High-confidence / Single-source / Disputed tiers),
τ = 0.85 triage, analyst dashboard, retraining every 50 labels, and the Net Doctor mobile console.

## Map
- `xwids/`: library (frames → features → prep → models → explain/ABEC → triage → store → feedback)
- `app/`: Flask dashboard (`/` analyst console, `/doctor/` Net Doctor), `serve.py` starts it
- `scripts/01..07`: prepare, train, evaluate (Tables 1-4), active learning (Table 5), export, inspect CSV, HF push
- `config.yaml`: every paper hyper-parameter; values marked "(ours)" are our own choices, not from the paper
- `tests/`: `python -m pytest -q`; `XWIDS_FAST=1` shrinks forests for tests only

## Hard rules
1. **Honesty.** Numbers for the paper come only from `reports/awid3/*.md` produced by the scripts. Never copy
   the paper's claimed numbers into code, docs or slides as if measured. Synthetic results are never paper results.
2. If AWID3 columns do not parse, add aliases in `xwids/frames.py: ALIASES` and re-run
   `scripts/06_inspect_csv.py`. Do not drop the check.
3. Keep the split group-aware (`prep.split`): windows of the same BSS within 2 minutes stay on one side.
4. SMOTE only on the training split. Never on val/test.
5. Never use `XWIDS_FAST=1` for reported results.
6. Don't change the ABEC definition (explain.abec) without updating the docstring, the tests and the README.
7. Windows laptop, CPU only: keep memory under about 12 GB (use `max_rows_per_file`).
