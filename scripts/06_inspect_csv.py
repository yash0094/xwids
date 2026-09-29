"""Check that an AWID3 (or any tshark-exported) CSV can be read before running the full pipeline.

    python scripts/06_inspect_csv.py data/raw/awid3/Deauth/Deauth_0.csv
Prints the columns found, which canonical field each maps to, and the first parsed rows + label counts.
"""
import sys

import _boot  # noqa: F401
import pandas as pd

from xwids import frames as FR

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    raw = pd.read_csv(sys.argv[1], nrows=20000, low_memory=False)
    print(f"{len(raw.columns)} columns in file")
    for canon, names in FR.ALIASES.items():
        hit = next((n for n in names if n.lower() in {c.lower().strip() for c in raw.columns}), None)
        print(f"  {canon:10s} <- {hit or '(missing: default used)'}")
    fr = FR.from_tshark_columns(raw, sys.argv[1])
    print(fr.head(8).to_string())
    print("labels:", fr["label"].value_counts().to_dict())
    print("subtypes:", fr["subtype"].value_counts().head(12).to_dict())
