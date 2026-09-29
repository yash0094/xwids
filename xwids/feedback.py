"""Stage 5: human-in-the-loop feedback. Every `retrain_every` (paper: 50) analyst labels trigger retraining.

New model = original training data + ALL analyst-labelled windows so far (weighted `label_weight`x).
It is evaluated on the validation split and promoted to live only if macro-F1 does not drop by more than
0.5 points (a guard so one bad batch of labels cannot silently degrade the detector).
Runs are logged to MLflow when installed, and always to reports/<dataset>/feedback_log.jsonl.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import registry
from .common import has, log
from .metrics import detection
from .models import Detector
from .triage import escalation_rate, triage


def build_training_set(base: pd.DataFrame, labelled: list[dict], features: list[str], weight: float):
    if labelled:
        lab = pd.DataFrame([{**r["features"], "label": r["label"]} for r in labelled])
        for f in features:
            if f not in lab.columns:
                lab[f] = 0.0
        lab["smote"] = 0
        df = pd.concat([base[features + ["label", "smote"]], lab[features + ["label", "smote"]]], ignore_index=True)
        w = np.r_[np.ones(len(base)), np.full(len(lab), weight)]
    else:
        df, w = base[features + ["label", "smote"]].reset_index(drop=True), np.ones(len(base))
    return df, w


def evaluate(det, val: pd.DataFrame, tau: float) -> dict:
    m = detection(val["label"], det.predict(val))
    m["escalation_rate"] = escalation_rate(triage(det, val, tau))
    return m


def retrain(cfg: dict, det_live, store, base_train: pd.DataFrame, val: pd.DataFrame, actor: str = "system") -> dict:
    t0 = time.time()
    labelled = store.labelled()
    new_ids = [r["id"] for r in store.labelled(unused_only=True)]
    df, w = build_training_set(base_train, labelled, det_live.features, cfg["feedback"]["label_weight"])
    cand = Detector(det_live.features, cfg).fit(df, sample_weight=w)
    tau = cfg["triage"]["tau"]
    before, after = evaluate(det_live, val, tau), evaluate(cand, val, tau)
    promote = after["f1_macro"] >= before["f1_macro"] - 0.005
    metrics = {"val_before": before, "val_after": after, "labels_total": len(labelled), "labels_new": len(new_ids)}
    v = registry.save(cfg, cand, registry.background(cand, base_train, seed=cfg["seed"]),
                      f"retrain on {len(labelled)} analyst labels", metrics, make_live=promote)
    store.mark_used(new_ids, v)
    secs = time.time() - t0
    store.x("INSERT INTO retrains(ts,version,labels_total,labels_new,seconds,metrics,promoted,note) "
            "VALUES(?,?,?,?,?,?,?,?)", (time.strftime("%Y-%m-%d %H:%M:%S"), v, len(labelled), len(new_ids), secs,
                                        json.dumps(metrics), int(promote),
                                        "promoted" if promote else "kept previous model (F1 dropped)"))
    store.audit(actor, "retrain", {"version": v, "promoted": promote, "labels": len(labelled),
                                   "f1_before": round(before["f1_macro"], 4), "f1_after": round(after["f1_macro"], 4)})
    _log_run(cfg, v, metrics, secs, promote)
    log.info("retrain -> v%d in %.1fs, F1 %.4f -> %.4f, %s", v, secs, before["f1_macro"], after["f1_macro"],
             "PROMOTED" if promote else "not promoted")
    return {"version": v, "promoted": promote, "seconds": secs, **metrics}


def _log_run(cfg, version, metrics, secs, promoted):
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "version": version, "seconds": round(secs, 1),
           "promoted": promoted, **{f"after_{k}": v for k, v in metrics["val_after"].items()},
           "labels_total": metrics["labels_total"]}
    p = Path(cfg["paths"]["reports"]) / "feedback_log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    if cfg["feedback"].get("mlflow") and has("mlflow"):
        try:
            import mlflow
            mlflow.set_experiment(f"xwids-{cfg['dataset']}")
            with mlflow.start_run(run_name=f"v{version}"):
                mlflow.log_params({"version": version, "labels_total": metrics["labels_total"],
                                   "promoted": promoted})
                mlflow.log_metrics({k: float(v) for k, v in metrics["val_after"].items()
                                    if isinstance(v, (int, float))})
        except Exception as e:  # MLflow problems must never break the SOC workflow
            log.warning("MLflow logging failed: %s", e)


class RetrainWorker:
    """Runs retraining in a background thread so the dashboard stays responsive."""

    def __init__(self):
        self.thread = None
        self.last = None
        self.error = None

    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, fn):
        if self.busy:
            return False

        def run():
            try:
                self.error = None
                self.last = fn()
            except Exception as e:  # surfaced on the dashboard
                log.exception("retraining failed")
                self.error = str(e)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return True
