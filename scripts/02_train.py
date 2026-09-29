"""Stage 2: train RF + XGBoost + Isolation Forest on the selected features and register the model.

    python scripts/02_train.py --dataset synthetic
"""
import argparse
import json

import _boot  # noqa: F401
from xwids import registry
from xwids.common import load_config, use_dataset
from xwids.metrics import detection
from xwids.models import Detector
from xwids.prep import load_meta, load_split
from xwids.triage import escalation_rate, triage

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic")
    a = ap.parse_args()
    cfg = use_dataset(load_config(), a.dataset)
    meta = load_meta(cfg)
    train, val, test = (load_split(cfg, s) for s in ("train", "val", "test"))
    det = Detector(meta["selected"], cfg).fit(train)
    out = {}
    for name, df in (("val", val), ("test", test)):
        m = detection(df["label"], det.predict(df))
        m["escalation_rate"] = escalation_rate(triage(det, df, cfg["triage"]["tau"]))
        out[name] = m
    v = registry.save(cfg, det, registry.background(det, train, seed=cfg["seed"]), "initial training", out)
    print(json.dumps({"version": v, **{k: {m: round(x, 4) for m, x in d.items()} for k, d in out.items()}}, indent=2))
    if meta.get("synthetic"):
        print("NOTE: synthetic data - these numbers test the code, they are not paper results.")
