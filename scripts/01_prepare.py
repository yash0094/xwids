"""Stage 1: frames -> window features -> split -> SMOTE -> SHAP-guided selection.

    python scripts/01_prepare.py --dataset synthetic     # offline, no download needed
    python scripts/01_prepare.py --dataset awid3         # after putting AWID3 CSVs in data/raw/awid3/
"""
import argparse
import json

import _boot  # noqa: F401
from xwids.common import load_config, use_dataset
from xwids.prep import prepare

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic")
    a = ap.parse_args()
    meta = prepare(use_dataset(load_config(), a.dataset))
    print(json.dumps({k: meta[k] for k in ("dataset", "frames", "windows", "rows", "selected")}, indent=2))
