---
title: XWIDS Explainable Wireless IDS
emoji: 📡
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# XWIDS: Explainability-Based Wireless Intrusion Detection System

Analyst dashboard for the ABEC framework (SHAP + LIME agreement) on 802.11 traffic windows ({{DATASET}}).
Log in, press **Replay** to stream held-out traffic through the Random Forest + XGBoost ensemble, then review
escalated alerts. Every 50 verdicts retrain the model.

Set Space secrets `XWIDS_USERS` (`name:password,name2:password2`), `XWIDS_SECRET_KEY` and optionally `XWIDS_API_KEY`.
Data written at runtime (verdicts, retrained models) is lost when the Space restarts: export the audit log first.
