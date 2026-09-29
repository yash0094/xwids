# XWIDS results: synthetic (model v1)

> **SYNTHETIC DATA.** These numbers prove the pipeline runs. Do not put them in the paper; rerun on AWID3.

Backends: shap: permutation-sampling Shapley (built-in); lime: tabular LIME (built-in); xgboost: sklearn HistGradientBoosting (install `xgboost`); smote: built-in SMOTE; mlflow: off (JSON log only)

## Table 1: detection performance (held-out test split)

| Dataset | Acc. (%) | Prec. (%) | Rec. (%) | F1 (%) | FPR (%) | Escal. (%) |
|---|---|---|---|---|---|---|
| synthetic (macro) | 98.8 | 98.3 | 92.2 | 94.6 | 0.2 | 14.9 |
| synthetic (weighted) | 98.8 | 98.7 | 98.8 | 98.7 | 0.2 | |

## Table 2: per attack class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| (Re)Association Flood | 0.991 | 1.000 | 0.995 | 108 |
| Deauthentication | 0.975 | 0.639 | 0.772 | 61 |
| Evil Twin | 0.988 | 1.000 | 0.994 | 84 |
| KRACK | 0.973 | 0.973 | 0.973 | 112 |
| Normal | 0.988 | 0.998 | 0.993 | 2035 |

## Table 3: explanation quality (top-5, 60 alert windows, 10 runs)

| Metric | SHAP-only | LIME-only | ABEC (fused) |
|---|---|---|---|
| Fidelity (conf. drop) | 0.363 | 0.287 | 0.375 |
| Instability (1 - mean pairwise J) | 0.000 | 0.188 | 0.163 |
| Avg. latency (ms) | 123.0 | 127.5 | 250.6 |
| Mean Jaccard J | – | – | 0.336 |

## Table 4: mean ABEC agreement J by outcome

| Predicted class | Correct | Misclassified | n (correct / wrong) |
|---|---|---|---|
| (Re)Association Flood | 0.15 | 1.00 | 10 / 1 |
| Deauthentication | 0.27 | 0.43 | 2 / 1 |
| Evil Twin | 0.37 | 0.67 | 8 / 1 |
| KRACK | 0.53 | 0.45 | 9 / 3 |
| Normal | 0.37 | 0.27 | 3 / 22 |
| ALL | 0.34 | 0.33 | 32 / 28 |

## Latency and throughput

- classification: 91.6 ms for one window, 0.063 ms/window batched
- SHAP: 123.0 ms · LIME: 127.5 ms per alert
- triage throughput: 10957 windows/s
