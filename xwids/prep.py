"""Stage 1: data capture + preprocessing.

frames (AWID3 CSV / simulator) -> window features -> leakage-safe split -> Min-Max scaling -> SMOTE (train only)
-> SHAP-guided feature selection (33 candidates -> 17) -> data/processed/<dataset>/{train,val,test}.pkl
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from . import features as FT
from . import frames as FR
from . import simulate
from .common import NORMAL, backends, has, log, read_json, write_json

SPLITS = ("train", "val", "test")


# ------------------------------------------------------------------ loading
def load_frames(cfg: dict) -> pd.DataFrame:
    ds = cfg["ds"]
    if ds["source"] == "synthetic":
        return simulate.generate(ds.get("n_bss", 10), ds.get("seconds", 1200), cfg["seed"])
    if ds["source"] == "csv":
        pattern = ds["glob"]
        if not Path(pattern).is_absolute():
            pattern = str(Path(cfg["_root"]) / pattern)
        return FR.load_csv_frames(pattern, ds.get("max_rows_per_file"))
    raise ValueError(f"source {ds['source']} has no frame table (processed features only)")


# ------------------------------------------------------------------ split
def split(df: pd.DataFrame, cfg: dict) -> dict:
    """Group-aware stratified split: windows from the same BSS within the same 2-minute block stay together,
    so neighbouring windows of one attack episode cannot sit in both train and test (temporal leakage)."""
    groups = df["bssid"].astype(str) + "|" + (df["win_start"] // 120).astype(int).astype(str)
    y = df["label"].astype(str)
    n_test = max(2, round(1 / cfg["split"]["test"]))
    sgk = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=cfg["seed"])
    tr_idx, te_idx = next(sgk.split(df, y, groups))
    rest = df.iloc[tr_idx]
    val_share = cfg["split"]["val"] / (1 - cfg["split"]["test"])
    sgk2 = StratifiedGroupKFold(n_splits=max(2, round(1 / val_share)), shuffle=True, random_state=cfg["seed"] + 1)
    a, b = next(sgk2.split(rest, y.iloc[tr_idx], groups.iloc[tr_idx]))
    out = {"train": rest.iloc[a], "val": rest.iloc[b], "test": df.iloc[te_idx]}
    tr_g = set(groups.iloc[tr_idx].iloc[a])
    assert not (tr_g & set(groups.iloc[te_idx])), "group leakage between train and test"
    return {k: v.reset_index(drop=True) for k, v in out.items()}


# ------------------------------------------------------------------ SMOTE
def smote(df: pd.DataFrame, cols: list[str], seed: int) -> pd.DataFrame:
    """SMOTE in Min-Max space on the training split; new rows are mapped back to raw units and marked smote=1."""
    df = df.assign(smote=0)
    y = df["label"].to_numpy()
    counts = pd.Series(y).value_counts()
    target = int(counts.max())
    scaler = MinMaxScaler().fit(df[cols].to_numpy(float))
    Xs = scaler.transform(df[cols].to_numpy(float))
    if has("imblearn"):
        from imblearn.over_sampling import SMOTE
        k = int(max(1, min(5, counts.min() - 1)))
        Xr, yr = SMOTE(random_state=seed, k_neighbors=k).fit_resample(Xs, y)
        new = pd.DataFrame(scaler.inverse_transform(Xr[len(df):]), columns=cols)
        new["label"] = yr[len(df):]
    else:
        rng = np.random.default_rng(seed)
        parts = []
        for cls, c in counts.items():
            need = target - int(c)
            if need <= 0 or c < 2:
                continue
            Xc = Xs[y == cls]
            nn = NearestNeighbors(n_neighbors=min(6, len(Xc))).fit(Xc)
            _, idx = nn.kneighbors(Xc)
            i = rng.integers(0, len(Xc), need)
            j = idx[i, rng.integers(1, idx.shape[1], need)]
            gap = rng.random((need, 1))
            syn = Xc[i] + gap * (Xc[j] - Xc[i])
            p = pd.DataFrame(scaler.inverse_transform(syn), columns=cols)
            p["label"] = cls
            parts.append(p)
        new = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols + ["label"])
    new["smote"] = 1
    for m in ["bssid", "win_start", "n_frames", "attack_frames"]:
        new[m] = "smote" if m == "bssid" else -1
    log.info("SMOTE added %d synthetic training rows: %s", len(new), new["label"].value_counts().to_dict())
    return pd.concat([df, new[df.columns]], ignore_index=True)


# ------------------------------------------------------------------ SHAP-guided feature selection
def shap_importance(X: np.ndarray, y: np.ndarray, names: list[str], cfg: dict) -> pd.Series:
    rng = np.random.default_rng(cfg["seed"])
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=cfg["seed"])
    rf.fit(X, y)
    n = min(len(X), cfg["features"]["selection_sample"] if has("shap") else 300)
    S = X[rng.choice(len(X), n, replace=False)]
    if has("shap"):
        import shap
        sv = shap.TreeExplainer(rf).shap_values(S, check_additivity=False)
        arr = np.stack(sv, -1) if isinstance(sv, list) else np.asarray(sv)
        imp = np.abs(arr).mean(axis=(0, 2)) if arr.ndim == 3 else np.abs(arr).mean(0)
    else:
        from .explain import Explainer

        class _Det:  # minimal adapter so the built-in Shapley sampler can explain this RF
            w_rf, w_gb = 1.0, 0.0
            rf = gb = None
        det = _Det()
        det.rf = rf
        det.proba_scaled = rf.predict_proba
        ex = Explainer.__new__(Explainer)
        ex.det, ex.bg = det, X[rng.choice(len(X), min(len(X), 100), replace=False)]
        ex.cfg = {"fallback_permutations": 6}
        pred = rf.predict_proba(S).argmax(1)
        imp = np.zeros(X.shape[1])
        for x, c in zip(S, pred):
            imp += np.abs(ex._sampled_shapley(lambda Z, c=c: rf.predict_proba(Z)[:, c], x))
        imp /= len(S)
    return pd.Series(imp, index=names).sort_values(ascending=False)


# ------------------------------------------------------------------ main entry
def prepare(cfg: dict) -> dict:
    t0 = time.time()
    out = Path(cfg["paths"]["processed"])
    out.mkdir(parents=True, exist_ok=True)
    fr = load_frames(cfg)
    fcfg = cfg["features"]
    df = FT.extract(fr, fcfg["window_seconds"], fcfg["min_frames"], fcfg["attack_fraction"])
    keep = cfg["ds"].get("keep_classes")
    if keep:
        before = df["label"].value_counts()
        df = df[df["label"].isin(keep)].reset_index(drop=True)
        dropped = before.drop([k for k in keep if k in before.index], errors="ignore")
        if len(dropped):
            log.info("dropped windows of classes outside keep_classes: %s", dropped.to_dict())
    counts = df["label"].value_counts()
    rare = counts[counts < 10]
    if len(rare):
        log.warning("dropping classes with < 10 windows (cannot be split): %s", rare.to_dict())
        df = df[~df["label"].isin(rare.index)].reset_index(drop=True)
    log.info("feature rows: %d; classes %s", len(df), df["label"].value_counts().to_dict())
    parts = split(df, cfg)
    if cfg.get("smote", True):
        parts["train"] = smote(parts["train"], FT.FEATURES, cfg["seed"])
    else:
        parts["train"] = parts["train"].assign(smote=0)
    for k in ("val", "test"):
        parts[k] = parts[k].assign(smote=0)

    sc = MinMaxScaler().fit(parts["train"][FT.FEATURES].to_numpy(float))
    imp = shap_importance(sc.transform(parts["train"][FT.FEATURES].to_numpy(float)),
                          parts["train"]["label"].to_numpy(), FT.FEATURES, cfg)
    selected = list(imp.index[: fcfg["n_selected"]])
    log.info("SHAP-guided selection kept %d/%d features: %s", len(selected), len(FT.FEATURES), selected)

    for k, v in parts.items():
        v.to_pickle(out / f"{k}.pkl")
    meta = {
        "dataset": cfg["dataset"], "source": cfg["ds"]["source"], "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frames": int(len(fr)), "windows": int(len(df)), "window_seconds": fcfg["window_seconds"],
        "candidates": FT.FEATURES, "selected": selected,
        "shap_importance": {k: round(float(v), 6) for k, v in imp.items()},
        "rows": {k: int(len(v)) for k, v in parts.items()},
        "class_counts": {k: v["label"].value_counts().to_dict() for k, v in parts.items()},
        "smote_rows": int(parts["train"]["smote"].sum()), "backends": backends(),
        "synthetic": cfg["ds"]["source"] == "synthetic",
    }
    write_json(out / "meta.json", meta)
    (out / "DATA_CARD.md").write_text(data_card(meta), encoding="utf-8")
    log.info("prepared %s in %.1fs -> %s", cfg["dataset"], time.time() - t0, out)
    return meta


def data_card(meta: dict) -> str:
    lines = [f"# Data card: {meta['dataset']}", ""]
    if meta.get("synthetic"):
        lines += ["**SYNTHETIC DATA.** Produced by `xwids/simulate.py`. Numbers measured on it test the code only;"
                  " they must not be reported as results for the paper.", ""]
    lines += [f"- source: {meta['source']}", f"- frames: {meta['frames']:,}", f"- windows (rows): {meta['windows']:,}",
              f"- window: {meta['window_seconds']} s per access point (BSSID)",
              f"- split rows: {meta['rows']}", f"- SMOTE rows added to train: {meta['smote_rows']:,}", "",
              "## Class counts", ""]
    for k, v in meta["class_counts"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## SHAP-guided feature selection", "", "| rank | feature | mean abs SHAP | kept | meaning |",
              "|---|---|---|---|---|"]
    for i, (f, v) in enumerate(meta["shap_importance"].items(), 1):
        lines.append(f"| {i} | {f} | {v:.4f} | {'yes' if f in meta['selected'] else ''} | {FT.CANDIDATES[f]} |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ reading processed data
def load_split(cfg: dict, name: str) -> pd.DataFrame:
    p = Path(cfg["paths"]["processed"]) / f"{name}.pkl"
    if not p.exists():
        if cfg["ds"]["source"] == "hf":
            fetch_hf(cfg)
        else:
            raise SystemExit(f"{p} not found. Run: python scripts/01_prepare.py --dataset {cfg['dataset']}")
    return pd.read_pickle(p)


def load_meta(cfg: dict) -> dict:
    p = Path(cfg["paths"]["processed"]) / "meta.json"
    if not p.exists() and cfg["ds"]["source"] == "hf":
        fetch_hf(cfg)
    return read_json(p)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_hf(cfg: dict) -> None:
    """Download processed features from YOUR Hugging Face dataset repo (made by scripts/07_hf_push.py) and
    verify every file against the SHA-256 manifest pushed with it."""
    ds = cfg["ds"]
    if not ds.get("repo_id"):
        raise SystemExit("Set datasets.hf.repo_id in config.yaml (e.g. your-name/xwids-awid3-features).")
    if not has("huggingface_hub"):
        raise SystemExit("pip install huggingface_hub")
    from huggingface_hub import hf_hub_download
    out = Path(cfg["paths"]["processed"])
    out.mkdir(parents=True, exist_ok=True)
    rev = ds.get("revision")
    man = read_json(hf_hub_download(ds["repo_id"], "manifest.json", repo_type="dataset", revision=rev))
    for fname, digest in man["sha256"].items():
        local = hf_hub_download(ds["repo_id"], fname, repo_type="dataset", revision=rev)
        got = sha256(local)
        if got != digest:
            raise SystemExit(f"SHA-256 mismatch for {fname}: expected {digest}, got {got}. Refusing to use it.")
        if fname.endswith(".csv.gz"):
            pd.read_csv(local).to_pickle(out / fname.replace(".csv.gz", ".pkl"))
        else:
            Path(out / fname).write_bytes(Path(local).read_bytes())
    log.info("downloaded and verified %d files from hf://datasets/%s", len(man["sha256"]), ds["repo_id"])
