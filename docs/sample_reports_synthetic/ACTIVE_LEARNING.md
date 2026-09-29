# Table 5: active-learning feedback loop (synthetic)

> **SYNTHETIC DATA**: shows the loop works; not a paper result.

Seed: 30% of real training windows; 50 analyst labels per iteration; oracle = ground truth.

| Iteration | Labels added | F1 (%) | FPR (%) | Escal. (%) | Fidelity (SHAP top-5) | Retrain (s) |
|---|---|---|---|---|---|---|
| Baseline (manual review) | 0 | 92.7 | 0.39 | 100 (manual) / 22 (model) | 0.967 | – |
| Iteration 1 | 50 | 97.3 | 0.39 | 15 | 0.958 | 5 |
| Iteration 2 | 100 | 98.1 | 0.29 | 14 | 0.966 | 5 |
| Iteration 3 | 150 | 97.5 | 0.54 | 16 | 0.940 | 5 |
