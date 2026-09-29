"""Stage 2: multi-model detection engine.

Random Forest (primary, 500 trees, depth 20) + XGBoost (secondary, 300 trees, lr 0.05) combined by soft voting,
plus an Isolation Forest (contamination 0.05) trained on Normal traffic only, for zero-day anomalies.
Min-Max scaling (paper, stage 1) lives inside the Detector so callers always pass raw feature values.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from .common import NORMAL, has, log


def fast_mode() -> bool:
    """XWIDS_FAST=1 shrinks the forests so tests and CI run in seconds. Never use it for reported results."""
    return os.environ.get("XWIDS_FAST") == "1"


def _make_gb(cfg: dict, seed: int, n_classes: int):
    p = dict(cfg["models"]["xgb"])
    if fast_mode():
        p["n_estimators"] = 40
    if has("xgboost"):
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                             max_depth=p.get("max_depth", 6), subsample=0.9, colsample_bytree=0.9,
                             tree_method="hist", n_jobs=-1, random_state=seed,
                             objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                             eval_metric="mlogloss" if n_classes > 2 else "logloss")
    from sklearn.ensemble import HistGradientBoostingClassifier
    log.warning("xgboost not installed: using sklearn HistGradientBoostingClassifier as the secondary model")
    return HistGradientBoostingClassifier(max_iter=p["n_estimators"], learning_rate=p["learning_rate"],
                                          max_depth=p.get("max_depth", 6), random_state=seed)


class Detector:
    def __init__(self, features: list[str], cfg: dict):
        self.features = list(features)
        self.cfg = cfg
        self.seed = cfg.get("seed", 42)
        self.classes_: list[str] = []
        self.scaler = MinMaxScaler(clip=True)
        self.rf = None
        self.gb = None
        self.iforest = None
        w = cfg["models"]["vote_weights"]
        self.w_rf, self.w_gb = float(w["rf"]), float(w["xgb"])
        self.version = 1
        self.trained_at = None
        self.backend_gb = "xgboost" if has("xgboost") else "sklearn-hgb"

    # ---------- helpers ----------
    def _X(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X[self.features].to_numpy(dtype=float)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def scale(self, X) -> np.ndarray:
        return self.scaler.transform(self._X(X))

    def unscale(self, Xs) -> np.ndarray:
        return self.scaler.inverse_transform(np.atleast_2d(Xs))

    # ---------- training ----------
    def fit(self, df: pd.DataFrame, sample_weight=None, fit_scaler: bool = True):
        t0 = time.time()
        X = self._X(df)
        y = df["label"].astype(str).to_numpy()
        if fit_scaler:
            self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        self.classes_ = sorted(set(y))
        yi = np.searchsorted(self.classes_, y)
        rfp = dict(self.cfg["models"]["rf"])
        if fast_mode():
            rfp["n_estimators"] = 60
        self.rf = RandomForestClassifier(n_estimators=rfp["n_estimators"], max_depth=rfp["max_depth"],
                                         n_jobs=-1, random_state=self.seed, class_weight=None)
        self.rf.fit(Xs, yi, sample_weight=sample_weight)
        self.gb = _make_gb(self.cfg, self.seed, len(self.classes_))
        self.gb.fit(Xs, yi, sample_weight=sample_weight)
        normal = (y == NORMAL)
        if "smote" in df.columns:
            normal &= df["smote"].to_numpy() == 0
        ip = self.cfg["models"]["iforest"]
        self.iforest = IsolationForest(n_estimators=ip.get("n_estimators", 200), contamination=ip["contamination"],
                                       random_state=self.seed, n_jobs=-1)
        ref = Xs[normal] if normal.sum() > 50 else Xs
        self.iforest.fit(ref)
        # (ours) a Normal-predicted window is escalated as a possible zero-day only if it is more unusual than
        # `zero_day_quantile` of real Normal training traffic; the 5% contamination flag alone is shown as a hint
        q = self.cfg["triage"].get("zero_day_quantile", 0.995)
        self.zero_day_threshold = float(np.quantile(-self.iforest.score_samples(ref), q))
        self.trained_at = time.strftime("%Y-%m-%d %H:%M:%S")
        log.info("trained detector on %d rows, %d features, classes=%s in %.1fs",
                 len(X), len(self.features), self.classes_, time.time() - t0)
        return self

    # ---------- inference (all take RAW feature values) ----------
    def proba_scaled(self, Xs) -> np.ndarray:
        Xs = np.atleast_2d(Xs)
        return self.w_rf * self.rf.predict_proba(Xs) + self.w_gb * self.gb.predict_proba(Xs)

    def predict_proba(self, X) -> np.ndarray:
        return self.proba_scaled(self.scale(X))

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.classes_)[self.predict_proba(X).argmax(1)]

    def anomaly(self, X):
        """(is_anomaly bool array, anomaly score: higher = more unusual)."""
        Xs = self.scale(X)
        return self.iforest.predict(Xs) == -1, -self.iforest.score_samples(Xs)

    def member_proba(self, X) -> dict:
        Xs = self.scale(X)
        return {"rf": self.rf.predict_proba(Xs), "xgb": self.gb.predict_proba(Xs)}

    def info(self) -> dict:
        return {"version": self.version, "trained_at": self.trained_at, "features": self.features,
                "classes": self.classes_, "rf_trees": self.rf.n_estimators if self.rf else None,
                "secondary": self.backend_gb, "vote_weights": {"rf": self.w_rf, "xgb": self.w_gb}}
