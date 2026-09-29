"""Export a self-contained deployment bundle into deploy/ (used by the Dockerfile / Hugging Face Space / Render).

Copies the LIVE model and capped data splits (enough to replay traffic and to retrain on analyst feedback),
so the deployed app needs no raw dataset and no internet.

    python scripts/05_export_deploy.py --dataset synthetic
    python scripts/05_export_deploy.py --dataset awid3
"""
import argparse
import json
import shutil
from pathlib import Path

import _boot  # noqa: F401
import numpy as np

from xwids.common import ROOT, load_config, read_json, use_dataset, write_json
from xwids.prep import load_split


def stratified_cap(df, cap, seed):
    if len(df) <= cap:
        return df
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in df.groupby("label"):
        n = max(10, int(round(cap * len(g) / len(df))))
        parts.append(g.iloc[rng.choice(len(g), min(n, len(g)), replace=False)])
    return __import__("pandas").concat(parts).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--out", default=str(ROOT / "deploy"))
    a = ap.parse_args()
    base = load_config()
    cfg = use_dataset(base, a.dataset)
    out = Path(a.out)
    reg = read_json(Path(cfg["paths"]["models"]) / "registry.json")
    live = reg["live"]
    m_out, d_out = out / "models" / a.dataset, out / "data" / "processed" / a.dataset
    shutil.rmtree(m_out, ignore_errors=True)
    shutil.rmtree(d_out, ignore_errors=True)
    m_out.mkdir(parents=True)
    d_out.mkdir(parents=True)
    shutil.copytree(Path(cfg["paths"]["models"]) / f"v{live}", m_out / f"v{live}")
    write_json(m_out / "registry.json", {"live": live, "history": [h for h in reg["history"] if h["version"] == live]})
    rows = {}
    for name, cap in base["deploy"]["caps"].items():
        df = stratified_cap(load_split(cfg, name), cap, base["seed"])
        df.to_pickle(d_out / f"{name}.pkl")
        rows[name] = len(df)
    for f in ("meta.json", "DATA_CARD.md"):
        src = Path(cfg["paths"]["processed"]) / f
        if src.exists():
            shutil.copy(src, d_out / f)
    shutil.copy(ROOT / "config.yaml", out / "config.yaml")
    # pin the exact library versions the model was trained with, so the container can unpickle it
    from importlib.metadata import PackageNotFoundError, version
    pins = []
    for pkg in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "xgboost", "shap", "lime"):
        try:
            pins.append(f"{pkg}=={version(pkg)}")
        except PackageNotFoundError:
            pass
    (out / "versions.txt").write_text("\n".join(pins) + "\n", encoding="utf-8")
    mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    bundle = (m_out / f"v{live}" / "bundle.joblib").stat().st_size / 1e6
    print(json.dumps({"dataset": a.dataset, "live_version": live, "rows": rows, "model_MB": round(bundle, 1),
                      "deploy_MB": round(mb, 1), "path": str(out)}, indent=2))
    if bundle > 400:
        print("WARNING: model file is large for free hosting. Lower models.rf.n_estimators / max_depth, retrain, export again.")
