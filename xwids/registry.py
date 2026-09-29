"""Versioned model store: models/<dataset>/v<N>/bundle.joblib + registry.json {live, history}."""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .common import log, read_json, write_json
from .explain import Explainer


def background(det, train: pd.DataFrame, n: int = 2000, seed: int = 42) -> np.ndarray:
    """Real (non-SMOTE) training rows, stratified, scaled: used by SHAP/LIME/ablation as the reference data."""
    real = train[train.get("smote", 0) == 0] if "smote" in train.columns else train
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in real.groupby("label"):
        m = max(5, int(round(n * len(g) / len(real))))
        parts.append(g.iloc[rng.choice(len(g), min(m, len(g)), replace=False)])
    return det.scale(pd.concat(parts))


def save(cfg: dict, det, bg: np.ndarray, note: str, metrics: dict | None = None, make_live: bool = True) -> int:
    root = Path(cfg["paths"]["models"])
    reg_p = root / "registry.json"
    reg = read_json(reg_p) if reg_p.exists() else {"live": None, "history": []}
    v = max([h["version"] for h in reg["history"]] + [0]) + 1
    det.version = v
    d = root / f"v{v}"
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump({"detector": det, "background": bg}, d / "bundle.joblib", compress=3)
    write_json(d / "model_card.json", {**det.info(), "note": note, "metrics": metrics or {}})
    reg["history"].append({"version": v, "note": note, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "metrics": metrics or {}})
    if make_live:
        reg["live"] = v
    write_json(reg_p, reg)
    log.info("saved model v%d (%s)%s", v, note, " [live]" if make_live else "")
    return v


def load(cfg: dict, version: int | None = None):
    root = Path(cfg["paths"]["models"])
    reg_p = root / "registry.json"
    if not reg_p.exists():
        raise SystemExit(f"No models in {root}. Run: python scripts/02_train.py --dataset {cfg['dataset']}")
    reg = read_json(reg_p)
    v = version or reg["live"]
    try:
        b = joblib.load(root / f"v{v}" / "bundle.joblib")
    except Exception as e:  # usually: model saved with different numpy / scikit-learn versions
        raise SystemExit(f"Could not load model v{v} ({type(e).__name__}: {e}).\nIt was probably trained with other "
                         f"library versions. Retrain here: python scripts/02_train.py --dataset {cfg['dataset']}")
    det = b["detector"]
    det.cfg = cfg
    return det, Explainer(det, b["background"], cfg), reg
