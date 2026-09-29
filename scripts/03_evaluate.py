"""Regenerates the paper's Tables 1-4 (and the latency numbers of Section IV-E) from the live model.

    python scripts/03_evaluate.py --dataset synthetic
    python scripts/03_evaluate.py --dataset awid3 --n-explain 200 --cv     # --cv adds stratified 10-fold CV

Writes reports/<dataset>/evaluation.json and reports/<dataset>/RESULTS.md.
Table 1 : detection metrics on the held-out test split (+ 10-fold CV with --cv)
Table 2 : per-attack-class precision / recall / F1
Table 3 : explanation quality: fidelity (confidence drop when top-5 features are ablated), stability
          (1 - mean pairwise Jaccard of top-5 sets over 10 repeated runs; 0 = perfectly stable), latency, mean J
Table 4 : mean ABEC Jaccard J for correctly vs incorrectly classified windows, per attack class
"""
import argparse
import itertools
import time

import _boot  # noqa: F401
import numpy as np
import pandas as pd

from xwids import registry
from xwids.common import NORMAL, backends, load_config, log, use_dataset, write_json
from xwids.explain import top_k
from xwids.metrics import confusion, detection, per_class
from xwids.prep import load_meta, load_split
from xwids.triage import escalation_rate, triage


def jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def instability(sets):
    pairs = list(itertools.combinations(sets, 2))
    return float(1 - np.mean([jacc(a, b) for a, b in pairs])) if pairs else 0.0


def cross_validate(cfg, meta, folds):
    from sklearn.model_selection import StratifiedGroupKFold
    from xwids.models import Detector
    from xwids.prep import smote
    from xwids.features import FEATURES
    real = pd.concat([load_split(cfg, "train"), load_split(cfg, "val")])
    real = real[real["smote"] == 0].reset_index(drop=True)
    groups = real["bssid"].astype(str) + "|" + (real["win_start"] // 120).astype(int).astype(str)
    res = []
    for k, (a, b) in enumerate(StratifiedGroupKFold(folds, shuffle=True, random_state=cfg["seed"])
                               .split(real, real["label"], groups)):
        tr = smote(real.iloc[a].drop(columns="smote"), FEATURES, cfg["seed"]) if cfg.get("smote") else real.iloc[a]
        det = Detector(meta["selected"], cfg).fit(tr)
        res.append(detection(real.iloc[b]["label"], det.predict(real.iloc[b])))
        log.info("CV fold %d/%d: F1 %.4f", k + 1, folds, res[-1]["f1_macro"])
    keys = res[0].keys()
    return {m: {"mean": float(np.mean([r[m] for r in res])), "std": float(np.std([r[m] for r in res]))}
            for m in keys if m != "n"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--n-explain", type=int, default=100, help="alert windows used for Tables 3-4")
    ap.add_argument("--runs", type=int, default=10, help="repeated runs for the stability metric")
    ap.add_argument("--cv", action="store_true", help="also run stratified k-fold CV (slow at full size)")
    a = ap.parse_args()
    cfg = use_dataset(load_config(), a.dataset)
    meta = load_meta(cfg)
    det, ex, reg = registry.load(cfg)
    test = load_split(cfg, "test")
    k = cfg["explain"]["top_k"]
    tau = cfg["triage"]["tau"]
    rng = np.random.default_rng(cfg["seed"])

    # ---------------- Tables 1 + 2 and latency
    t0 = time.perf_counter()
    pred = det.predict(test)
    t_cls = (time.perf_counter() - t0) / len(test) * 1000
    t1 = time.perf_counter()
    tri = triage(det, test, tau)
    thr = len(test) / (time.perf_counter() - t1)
    t_single = []
    for i in rng.choice(len(test), min(50, len(test)), replace=False):
        s = time.perf_counter()
        det.predict_proba(test.iloc[[i]])
        t_single.append((time.perf_counter() - s) * 1000)
    table1 = {**detection(test["label"], pred), "escalation_rate": escalation_rate(tri)}
    table2 = per_class(test["label"], pred)

    # ---------------- Tables 3 + 4 on alert windows (attack predicted or escalated)
    alert_idx = [i for i, r in enumerate(tri) if r["is_alert"]]
    wrong = [i for i in alert_idx if pred[i] != test["label"].iloc[i]]
    right = [i for i in alert_idx if pred[i] == test["label"].iloc[i]]
    n_wrong = min(len(wrong), a.n_explain // 2)
    pick = list(rng.choice(wrong, n_wrong, replace=False)) if n_wrong else []
    pick += list(rng.choice(right, min(len(right), a.n_explain - n_wrong), replace=False)) if right else []
    rows = []
    for n, i in enumerate(pick):
        x = test.iloc[[i]]
        cls = tri[i]["pred_index"]
        runs = [ex.explain(x, cls=cls, lime=True, seed=s) for s in range(a.runs)]
        e0 = runs[0]
        s_top, l_top, f_top = top_k(e0["shap"], k), top_k(e0["lime"], k), e0["abec"]["fused_top"]
        rows.append({
            "true": test["label"].iloc[i], "pred": pred[i], "correct": pred[i] == test["label"].iloc[i],
            "J": e0["abec"]["jaccard"], "J_mean_runs": float(np.mean([r["abec"]["jaccard"] for r in runs])),
            "fid_shap": ex.fidelity(x, cls, s_top), "fid_lime": ex.fidelity(x, cls, l_top),
            "fid_abec": ex.fidelity(x, cls, f_top),
            "stab_shap": instability([top_k(r["shap"], k) for r in runs]),
            "stab_lime": instability([top_k(r["lime"], k) for r in runs]),
            "stab_abec": instability([r["abec"]["fused_top"] for r in runs]),
            "ms_shap": float(np.mean([r["shap_ms"] for r in runs])), "ms_lime": float(np.mean([r["lime_ms"] for r in runs])),
            "n_disputed": e0["abec"]["n_disputed"]})
        if (n + 1) % 20 == 0:
            log.info("explained %d/%d alert windows", n + 1, len(pick))
    R = pd.DataFrame(rows)
    table3 = {} if R.empty else {
        "SHAP-only": {"fidelity": R.fid_shap.mean(), "instability": R.stab_shap.mean(), "latency_ms": R.ms_shap.mean()},
        "LIME-only": {"fidelity": R.fid_lime.mean(), "instability": R.stab_lime.mean(), "latency_ms": R.ms_lime.mean()},
        "ABEC (fused)": {"fidelity": R.fid_abec.mean(), "instability": R.stab_abec.mean(),
                         "latency_ms": (R.ms_shap + R.ms_lime).mean(), "mean_J": R.J.mean()},
        "n_windows": int(len(R)), "runs": a.runs}
    table4 = {}
    if not R.empty:
        for c, g in R.groupby("pred"):
            table4[c] = {"correct_J": float(g[g.correct].J.mean()) if g.correct.any() else None,
                         "misclassified_J": float(g[~g.correct].J.mean()) if (~g.correct).any() else None,
                         "n_correct": int(g.correct.sum()), "n_wrong": int((~g.correct).sum())}
        table4["ALL"] = {"correct_J": float(R[R.correct].J.mean()) if R.correct.any() else None,
                         "misclassified_J": float(R[~R.correct].J.mean()) if (~R.correct).any() else None,
                         "n_correct": int(R.correct.sum()), "n_wrong": int((~R.correct).sum())}
    latency = {"classification_ms_per_window_batched": t_cls, "classification_ms_single_window": float(np.median(t_single)),
               "shap_ms": float(R.ms_shap.mean()) if len(R) else None,
               "lime_ms": float(R.ms_lime.mean()) if len(R) else None,
               "triage_windows_per_s": thr}
    out = {"dataset": a.dataset, "model_version": reg["live"], "synthetic": meta.get("synthetic", False),
           "backends": {**backends(), **ex.backend}, "features": det.features,
           "table1_detection": table1, "table2_per_class": table2, "confusion": confusion(test["label"], pred),
           "table3_explanation": table3, "table4_jaccard": table4, "latency": latency}
    if a.cv:
        out["table1_cv"] = cross_validate(cfg, meta, cfg["models"]["cv_folds"])
    write_json(f"{cfg['paths']['reports']}/evaluation.json", out)
    R.to_csv(f"{cfg['paths']['reports']}/explanations_sample.csv", index=False)
    md = results_md(out)
    open(f"{cfg['paths']['reports']}/RESULTS.md", "w", encoding="utf-8").write(md)
    print(md)


def pct(v):
    return "–" if v is None else f"{v * 100:.1f}"


def results_md(o):
    L = [f"# XWIDS results: {o['dataset']} (model v{o['model_version']})", ""]
    if o["synthetic"]:
        L += ["> **SYNTHETIC DATA.** These numbers prove the pipeline runs. Do not put them in the paper; rerun on AWID3.", ""]
    L += ["Backends: " + "; ".join(f"{k}: {v}" for k, v in o["backends"].items()), ""]
    t = o["table1_detection"]
    L += ["## Table 1: detection performance (held-out test split)", "",
          "| Dataset | Acc. (%) | Prec. (%) | Rec. (%) | F1 (%) | FPR (%) | Escal. (%) |", "|---|---|---|---|---|---|---|",
          f"| {o['dataset']} (macro) | {pct(t['accuracy'])} | {pct(t['precision_macro'])} | {pct(t['recall_macro'])} | "
          f"{pct(t['f1_macro'])} | {pct(t['fpr'])} | {pct(t['escalation_rate'])} |",
          f"| {o['dataset']} (weighted) | {pct(t['accuracy'])} | {pct(t['precision_weighted'])} | "
          f"{pct(t['recall_weighted'])} | {pct(t['f1_weighted'])} | {pct(t['fpr'])} | |", ""]
    if "table1_cv" in o:
        c = o["table1_cv"]
        L += [f"10-fold CV: accuracy {pct(c['accuracy']['mean'])} ± {pct(c['accuracy']['std'])}, macro-F1 "
              f"{pct(c['f1_macro']['mean'])} ± {pct(c['f1_macro']['std'])}, FPR {pct(c['fpr']['mean'])}", ""]
    L += ["## Table 2: per attack class", "", "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for c, v in o["table2_per_class"].items():
        L.append(f"| {c} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |")
    t3 = o["table3_explanation"]
    if t3:
        L += ["", f"## Table 3: explanation quality (top-5, {t3['n_windows']} alert windows, {t3['runs']} runs)", "",
              "| Metric | SHAP-only | LIME-only | ABEC (fused) |", "|---|---|---|---|"]
        names = ["SHAP-only", "LIME-only", "ABEC (fused)"]
        L.append("| Fidelity (conf. drop) | " + " | ".join(f"{t3[n]['fidelity']:.3f}" for n in names) + " |")
        L.append("| Instability (1 - mean pairwise J) | " + " | ".join(f"{t3[n]['instability']:.3f}" for n in names) + " |")
        L.append("| Avg. latency (ms) | " + " | ".join(f"{t3[n]['latency_ms']:.1f}" for n in names) + " |")
        L.append(f"| Mean Jaccard J | – | – | {t3['ABEC (fused)']['mean_J']:.3f} |")
    t4 = o["table4_jaccard"]
    if t4:
        L += ["", "## Table 4: mean ABEC agreement J by outcome", "", "| Predicted class | Correct | Misclassified | n (correct / wrong) |",
              "|---|---|---|---|"]
        for c, v in t4.items():
            f = lambda x: "–" if x is None else f"{x:.2f}"
            L.append(f"| {c} | {f(v['correct_J'])} | {f(v['misclassified_J'])} | {v['n_correct']} / {v['n_wrong']} |")
    la = o["latency"]
    L += ["", "## Latency and throughput", "",
          f"- classification: {la['classification_ms_single_window']:.1f} ms for one window, "
          f"{la['classification_ms_per_window_batched']:.3f} ms/window batched",
          f"- SHAP: {la['shap_ms'] or 0:.1f} ms · LIME: {la['lime_ms'] or 0:.1f} ms per alert",
          f"- triage throughput: {la['triage_windows_per_s']:.0f} windows/s", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
