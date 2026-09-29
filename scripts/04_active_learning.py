"""Paper Table 5: simulated human-in-the-loop active learning.

A model starts from a small labelled seed (default 30% of the real training windows). The rest of the training
windows arrive as an unlabelled stream. Each iteration: triage the stream, take the 50 escalated alerts with the
highest priority (lowest confidence; with --jaccard, lowest ABEC agreement first), label them with the ground
truth (standing in for the analyst), retrain, and measure F1 / FPR / escalation / fidelity on the test split.

    python scripts/04_active_learning.py --dataset synthetic
    python scripts/04_active_learning.py --dataset awid3 --iters 3 --batch 50 --jaccard
"""
import argparse
import time

import _boot  # noqa: F401
import numpy as np
import pandas as pd

from xwids.common import NORMAL, load_config, log, use_dataset, write_json
from xwids.explain import Explainer, top_k
from xwids.features import FEATURES
from xwids.feedback import build_training_set
from xwids.metrics import detection
from xwids.models import Detector
from xwids.prep import load_meta, load_split, smote
from xwids.registry import background
from xwids.triage import ESCALATE, escalation_rate, triage


def measure(det, cfg, test, n_fid, seed):
    tri = triage(det, test, cfg["triage"]["tau"])
    m = detection(test["label"], det.predict(test))
    m["escalation_rate"] = escalation_rate(tri)
    ex = Explainer(det, background(det, test, 500, seed), cfg)
    alerts = [i for i, r in enumerate(tri) if r["pred"] != NORMAL]
    rng = np.random.default_rng(seed)
    fids = []
    for i in rng.choice(alerts, min(n_fid, len(alerts)), replace=False) if alerts else []:
        x = test.iloc[[i]]
        c = tri[i]["pred_index"]
        fids.append(ex.fidelity(x, c, top_k(ex.shap_values(det.scale(x), c), cfg["explain"]["top_k"])))
    m["fidelity_shap_top5"] = float(np.mean(fids)) if fids else None
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--batch", type=int, default=None, help="labels per iteration (default: config retrain_every)")
    ap.add_argument("--seed-frac", type=float, default=0.3)
    ap.add_argument("--jaccard", action="store_true", help="rank candidates by ABEC agreement too (slower)")
    ap.add_argument("--n-fid", type=int, default=40)
    a = ap.parse_args()
    cfg = use_dataset(load_config(), a.dataset)
    meta = load_meta(cfg)
    feats = meta["selected"]
    batch = a.batch or cfg["feedback"]["retrain_every"]
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)
    real = load_split(cfg, "train")
    real = real[real["smote"] == 0].drop(columns="smote").reset_index(drop=True)
    test = load_split(cfg, "test")
    # stratified seed pool
    seed_idx = np.concatenate([rng.choice(np.where(real["label"] == c)[0],
                                          max(3, int(round(a.seed_frac * (real["label"] == c).sum()))), replace=False)
                               for c in real["label"].unique()])
    pool = real.drop(index=seed_idx).reset_index(drop=True)
    seed_df = real.iloc[seed_idx].reset_index(drop=True)
    base = smote(seed_df, FEATURES, seed) if cfg.get("smote") else seed_df.assign(smote=0)
    det = Detector(feats, cfg).fit(base)
    rows = []
    m0 = measure(det, cfg, test, a.n_fid, seed)
    rows.append({"iteration": "Baseline (manual review)", "labels": 0, **m0, "escalation_rate_manual": 1.0,
                 "retrain_s": None})
    log.info("baseline: F1 %.4f FPR %.4f escal %.3f", m0["f1_macro"], m0["fpr"], m0["escalation_rate"])
    labelled, used = [], np.zeros(len(pool), bool)
    for it in range(1, a.iters + 1):
        avail = np.where(~used)[0]
        tri = triage(det, pool.iloc[avail], cfg["triage"]["tau"])
        cand = [(avail[j], r) for j, r in enumerate(tri) if r["route"] == ESCALATE]
        if len(cand) < batch:  # not enough escalations left: fall back to the least confident alerts
            extra = sorted([(avail[j], r) for j, r in enumerate(tri) if r["route"] != ESCALATE],
                           key=lambda t: t[1]["confidence"])
            cand += extra[: batch - len(cand)]
        cand.sort(key=lambda t: -t[1]["priority"])
        if a.jaccard:
            ex = Explainer(det, background(det, base, 500, seed), cfg)
            top = cand[: batch * 3]
            J = {i: ex.explain(pool.iloc[[i]], cls=r["pred_index"])["abec"]["jaccard"] for i, r in top}
            cand = sorted(top, key=lambda t: (J[t[0]], t[1]["confidence"]))
        chosen = [i for i, _ in cand[:batch]]
        used[chosen] = True
        labelled += [{"features": pool.iloc[i][feats].to_dict(), "label": pool.iloc[i]["label"]} for i in chosen]
        t0 = time.time()
        df, w = build_training_set(base, labelled, feats, cfg["feedback"]["label_weight"])
        det = Detector(feats, cfg).fit(df, sample_weight=w)
        secs = time.time() - t0
        m = measure(det, cfg, test, a.n_fid, seed)
        rows.append({"iteration": f"Iteration {it}", "labels": len(labelled), **m, "retrain_s": secs,
                     "batch_true_classes": pd.Series([pool.iloc[i]["label"] for i in chosen]).value_counts().to_dict()})
        log.info("iteration %d: +%d labels, F1 %.4f FPR %.4f escal %.3f (%.0fs retrain)", it, batch, m["f1_macro"],
                 m["fpr"], m["escalation_rate"], secs)
    out = {"dataset": a.dataset, "synthetic": meta.get("synthetic", False), "seed_frac": a.seed_frac, "batch": batch,
           "jaccard_ranking": a.jaccard, "rows": rows}
    write_json(f"{cfg['paths']['reports']}/active_learning.json", out)
    L = [f"# Table 5: active-learning feedback loop ({a.dataset})", ""]
    if out["synthetic"]:
        L += ["> **SYNTHETIC DATA**: shows the loop works; not a paper result.", ""]
    L += [f"Seed: {a.seed_frac:.0%} of real training windows; {batch} analyst labels per iteration; oracle = ground truth.",
          "", "| Iteration | Labels added | F1 (%) | FPR (%) | Escal. (%) | Fidelity (SHAP top-5) | Retrain (s) |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        esc = "100 (manual) / " + f"{r['escalation_rate'] * 100:.0f} (model)" if r["labels"] == 0 else \
            f"{r['escalation_rate'] * 100:.0f}"
        fid = "–" if r["fidelity_shap_top5"] is None else f"{r['fidelity_shap_top5']:.3f}"
        rt = "–" if r["retrain_s"] is None else f"{r['retrain_s']:.0f}"
        L.append(f"| {r['iteration']} | {r['labels']} | {r['f1_macro'] * 100:.1f} | {r['fpr'] * 100:.2f} | {esc} | {fid} | {rt} |")
    md = "\n".join(L) + "\n"
    open(f"{cfg['paths']['reports']}/ACTIVE_LEARNING.md", "w", encoding="utf-8").write(md)
    print(md)


if __name__ == "__main__":
    main()
