"""Publish to Hugging Face: processed dataset, model, and/or the live dashboard as a Docker Space.

    pip install huggingface_hub  &&  huggingface-cli login      (or set HF_TOKEN)
    python scripts/07_hf_push.py --dataset awid3 --what dataset --repo <you>/xwids-awid3-features
    python scripts/07_hf_push.py --dataset awid3 --what model   --repo <you>/xwids-awid3-model
    python scripts/07_hf_push.py --dataset awid3 --what space   --repo <you>/xwids-demo      (run 05_export_deploy first)

Dataset pushes include manifest.json with SHA-256 of every file; `datasets.hf` in config.yaml downloads and
verifies them, so a teammate can train from your processed features without AWID3 on their disk.
Space pushes set XWIDS_USERS / XWIDS_SECRET_KEY / XWIDS_API_KEY as Space secrets from your environment.
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

import _boot  # noqa: F401
import pandas as pd

from xwids.common import ROOT, has, load_config, read_json, use_dataset, write_json
from xwids.prep import sha256

SPACE_FILES = ["Dockerfile", "requirements-deploy.txt", "serve.py", "xwids", "app", "deploy"]


def api():
    if not has("huggingface_hub"):
        raise SystemExit("pip install huggingface_hub")
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get("HF_TOKEN"))


def push_dataset(cfg, repo, private):
    h = api()
    h.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    src = Path(cfg["paths"]["processed"])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for s in ("train", "val", "test"):
            pd.read_pickle(src / f"{s}.pkl").to_csv(tmp / f"{s}.csv.gz", index=False)
        shutil.copy(src / "meta.json", tmp / "meta.json")
        card = (src / "DATA_CARD.md").read_text(encoding="utf-8")
        (tmp / "README.md").write_text("---\nlicense: other\ntags: [intrusion-detection, wifi, 802.11, xai]\n---\n" + card,
                                       encoding="utf-8")
        write_json(tmp / "manifest.json", {"sha256": {f.name: sha256(f) for f in tmp.iterdir()
                                                      if f.name not in ("README.md",)}})
        h.upload_folder(folder_path=str(tmp), repo_id=repo, repo_type="dataset", commit_message="XWIDS processed features")
    print(f"pushed https://huggingface.co/datasets/{repo}\nSet in config.yaml:  datasets.hf.repo_id: {repo}")


def push_model(cfg, repo, private):
    h = api()
    h.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    reg = read_json(Path(cfg["paths"]["models"]) / "registry.json")
    d = Path(cfg["paths"]["models"]) / f"v{reg['live']}"
    card = read_json(d / "model_card.json")
    readme = (f"---\ntags: [intrusion-detection, wifi, shap, lime, random-forest, xgboost]\n---\n# XWIDS model v{reg['live']}\n\n"
              f"Random Forest + XGBoost soft-voting ensemble with Isolation Forest, trained on `{cfg['dataset']}` "
              f"windows. Load with `joblib.load('bundle.joblib')` inside the XWIDS repo.\n\n```json\n"
              f"{pd.Series(card).to_json(indent=2)}\n```\n")
    (d / "README.md").write_text(readme, encoding="utf-8")
    h.upload_folder(folder_path=str(d), repo_id=repo, repo_type="model", commit_message=f"XWIDS model v{reg['live']}")
    print(f"pushed https://huggingface.co/{repo}")


def push_space(cfg, repo, private):
    if not (ROOT / "deploy" / "config.yaml").exists():
        raise SystemExit("Run first: python scripts/05_export_deploy.py --dataset " + cfg["dataset"])
    h = api()
    h.create_repo(repo, repo_type="space", space_sdk="docker", private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for f in SPACE_FILES:
            p = ROOT / f
            (shutil.copytree(p, tmp / f, ignore=shutil.ignore_patterns("__pycache__", "hf_space")) if p.is_dir()
             else shutil.copy(p, tmp / f))
        readme = (ROOT / "deploy_templates" / "hf_space_README.md").read_text(encoding="utf-8")
        (tmp / "README.md").write_text(readme.replace("{{DATASET}}", cfg["dataset"]), encoding="utf-8")
        h.upload_folder(folder_path=str(tmp), repo_id=repo, repo_type="space", commit_message="XWIDS dashboard")
    for key in ("XWIDS_USERS", "XWIDS_SECRET_KEY", "XWIDS_API_KEY"):
        if os.environ.get(key):
            h.add_space_secret(repo, key, os.environ[key])
            print(f"set Space secret {key}")
    h.add_space_variable(repo, "XWIDS_DATASET", cfg["dataset"])
    if not os.environ.get("XWIDS_USERS"):
        print("WARNING: XWIDS_USERS not set -> the Space uses DEMO logins. Add it under Settings -> Secrets.")
    print(f"pushed https://huggingface.co/spaces/{repo}  (build takes a few minutes; check /healthz)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--what", choices=["dataset", "model", "space"], required=True)
    ap.add_argument("--repo", required=True, help="<hf-username>/<name>")
    ap.add_argument("--public", action="store_true", help="default is private")
    a = ap.parse_args()
    cfg = use_dataset(load_config(), a.dataset)
    {"dataset": push_dataset, "model": push_model, "space": push_space}[a.what](cfg, a.repo, not a.public)
