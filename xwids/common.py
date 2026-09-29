"""Shared helpers: config loading, paths, optional-package detection."""
from __future__ import annotations

import copy
import importlib.util
import json
import logging
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("xwids")
if not log.handlers:
    logging.basicConfig(level=os.environ.get("XWIDS_LOG", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

NORMAL = "Normal"


def has(pkg: str) -> bool:
    """True if an optional package (shap, lime, xgboost, mlflow, ...) is installed."""
    return importlib.util.find_spec(pkg) is not None


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path or os.environ.get("XWIDS_CONFIG", ROOT / "config.yaml"))
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(path.resolve().parent)
    return cfg


def use_dataset(cfg: dict, name: str) -> dict:
    """Return a copy of cfg whose paths point at one dataset's folders."""
    if name not in cfg["datasets"]:
        raise SystemExit(f"Unknown dataset '{name}'. Choose one of: {', '.join(cfg['datasets'])}")
    c = copy.deepcopy(cfg)
    root = Path(c["_root"])
    c["dataset"] = name
    c["ds"] = c["datasets"][name]
    p = c["paths"]
    p["raw"] = str(root / p["raw"])
    p["processed"] = str(root / p["processed"] / name)
    p["models"] = str(root / p["models"] / name)
    p["reports"] = str(root / p["reports"] / name)
    p["db"] = str(root / p["db"])
    return c


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def backends() -> dict:
    """Which real libraries are active vs. built-in fallbacks (shown on the dashboard and in reports)."""
    return {
        "shap": "shap" if has("shap") else "built-in permutation SHAP (install `shap` for exact TreeSHAP)",
        "lime": "lime" if has("lime") else "built-in tabular LIME (install `lime`)",
        "xgboost": "xgboost" if has("xgboost") else "sklearn HistGradientBoosting (install `xgboost`)",
        "smote": "imbalanced-learn" if has("imblearn") else "built-in SMOTE",
        "mlflow": "mlflow" if has("mlflow") else "off (JSON log only)",
    }
