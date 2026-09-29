"""Stage 3 (SHAP + LIME) and the ABEC framework (paper Section III-B).

All explanations are for the probability of ONE class (normally the predicted class) and are computed in the
scaled feature space the models see. Positive = pushes the model towards that class.

SHAP:  exact TreeSHAP via the `shap` package (RF: path-dependent; XGBoost: interventional, probability output).
       The ensemble is a weighted sum of probabilities, so ensemble SHAP = w_rf*SHAP_rf + w_xgb*SHAP_xgb.
       Without `shap` installed, a permutation-sampling Shapley estimate is used (same definition, sampled).
LIME:  `lime` LimeTabularExplainer (continuous features, sampling around the instance).
       Without `lime`, a built-in implementation of the same algorithm is used.
"""
from __future__ import annotations

import time
import zlib

import numpy as np

from .common import has, log


class Explainer:
    def __init__(self, detector, background_scaled: np.ndarray, cfg: dict):
        self.det = detector
        self.cfg = cfg["explain"]
        rng = np.random.default_rng(cfg.get("seed", 42))
        bg = np.asarray(background_scaled, dtype=float)
        n = min(len(bg), self.cfg["shap_background"])
        self.bg = bg[rng.choice(len(bg), n, replace=False)] if len(bg) > n else bg
        big = min(len(bg), 5000)
        self.lime_train = bg[rng.choice(len(bg), big, replace=False)] if len(bg) > big else bg
        self.bg_mean = self.bg.mean(0)
        self.std = self.lime_train.std(0)
        self.std[self.std == 0] = 1.0
        self.mean = self.lime_train.mean(0)
        self.use_shap = has("shap")
        self.use_lime = has("lime")
        self._shap_rf = self._shap_gb = None
        self._lime = None
        self._shap_exact_gb = False
        if self.use_shap:
            import shap
            self._shap_rf = shap.TreeExplainer(self.det.rf)
            try:  # probability-space SHAP for XGBoost needs the interventional mode
                self._shap_gb = shap.TreeExplainer(self.det.gb, data=self.bg, feature_perturbation="interventional",
                                                   model_output="probability")
                self._shap_exact_gb = True
            except Exception as e:  # e.g. sklearn HistGradientBoosting fallback model
                log.warning("TreeSHAP unavailable for the secondary model (%s); sampling its Shapley values", e)
        if self.use_lime:
            from lime.lime_tabular import LimeTabularExplainer
            self._lime_cls = LimeTabularExplainer
        self.backend = {"shap": "TreeSHAP (shap)" if self.use_shap else "permutation-sampling Shapley (built-in)",
                        "lime": "lime" if self.use_lime else "tabular LIME (built-in)"}

    # ------------------------------------------------------------------ SHAP
    @staticmethod
    def _pick_class(vals, cls: int, d: int) -> np.ndarray:
        """Handle every output shape shap has used across versions."""
        if isinstance(vals, list):
            return np.asarray(vals[cls]).reshape(-1)[:d]
        v = np.asarray(vals)
        if v.ndim == 3:  # (n, d, K)
            return v[0, :, cls]
        if v.ndim == 2 and v.shape[0] == 1:  # binary model returning one output
            return v[0] if cls == 1 else -v[0]
        return v.reshape(-1)[:d]

    def shap_values(self, xs: np.ndarray, cls: int) -> np.ndarray:
        xs = np.atleast_2d(xs)
        d = xs.shape[1]
        if self.use_shap:
            phi_rf = self._pick_class(self._shap_rf.shap_values(xs, check_additivity=False), cls, d)
            if self._shap_exact_gb:
                phi_gb = self._pick_class(self._shap_gb.shap_values(xs), cls, d)
            else:
                phi_gb = self._sampled_shapley(lambda Z: self.det.gb.predict_proba(Z)[:, cls], xs[0])
            return self.det.w_rf * phi_rf + self.det.w_gb * phi_gb
        return self._sampled_shapley(lambda Z: self.det.proba_scaled(Z)[:, cls], xs[0])

    def _sampled_shapley(self, f, x: np.ndarray) -> np.ndarray:
        """Interventional Shapley values by antithetic permutation sampling (Strumbelj & Kononenko 2014).
        Each (permutation, background row) pair walks from the background row to x one feature at a time;
        the contributions telescope, so sum(phi) = f(x) - mean f(background) exactly (local accuracy)."""
        d = len(x)
        m = int(self.cfg.get("fallback_permutations", 24))
        rng = np.random.default_rng(zlib.crc32(np.round(x, 6).tobytes()))  # deterministic per input
        perms = [rng.permutation(d) for _ in range(m)]
        perms += [p[::-1] for p in perms]  # antithetic pairs reduce variance
        bidx = rng.integers(0, len(self.bg), len(perms))
        rows = np.empty((len(perms) * (d + 1), d))
        k = 0
        for p, b in zip(perms, bidx):
            z = self.bg[b].copy()
            rows[k] = z
            k += 1
            for j in p:
                z[j] = x[j]
                rows[k] = z
                k += 1
        out = f(rows).reshape(len(perms), d + 1)
        deltas = np.diff(out, axis=1)
        phi = np.zeros(d)
        for i, p in enumerate(perms):
            phi[p] += deltas[i]
        return phi / len(perms)

    # ------------------------------------------------------------------ LIME
    def lime_weights(self, xs: np.ndarray, cls: int, seed: int = 0) -> np.ndarray:
        x = np.atleast_2d(xs)[0]
        d = len(x)
        n = int(self.cfg["lime_samples"])
        if self.use_lime:
            exp = self._lime_cls(self.lime_train, mode="classification", discretize_continuous=False,
                                 sample_around_instance=True, random_state=seed)
            e = exp.explain_instance(x, self.det.proba_scaled, labels=(cls,), num_features=d, num_samples=n)
            w = np.zeros(d)
            for j, v in e.as_map()[cls]:
                w[j] = v
            return w
        # built-in: the LIME tabular algorithm with continuous features
        rng = np.random.default_rng(seed)
        Z = rng.normal(0, 1, (n, d)) * self.std + x
        Z[0] = x
        Zstd = (Z - self.mean) / self.std
        xstd = (x - self.mean) / self.std
        dist = np.sqrt(((Zstd - xstd) ** 2).sum(1))
        width = 0.75 * np.sqrt(d)
        wts = np.sqrt(np.exp(-(dist ** 2) / width ** 2))
        y = self.det.proba_scaled(Z)[:, cls]
        from sklearn.linear_model import Ridge
        r = Ridge(alpha=1.0, fit_intercept=True, random_state=seed)
        r.fit(Zstd, y, sample_weight=wts)
        return r.coef_

    # ------------------------------------------------------------------ full explanation
    def explain(self, x_raw, cls: int | None = None, lime: bool = True, seed: int = 0) -> dict:
        xs = self.det.scale(x_raw)
        p = self.det.proba_scaled(xs)[0]
        cls = int(p.argmax()) if cls is None else int(cls)
        t0 = time.perf_counter()
        sv = self.shap_values(xs, cls)
        t_shap = (time.perf_counter() - t0) * 1000
        out = {"class": self.det.classes_[cls], "class_index": cls, "confidence": float(p[cls]),
               "features": self.det.features, "values": self.det._X(x_raw)[0].tolist(),
               "shap": sv.tolist(), "shap_base": float(p[cls] - sv.sum()), "shap_ms": round(t_shap, 2),
               "backend": self.backend}
        if lime:
            t1 = time.perf_counter()
            lw = self.lime_weights(xs, cls, seed=seed)
            out["lime_ms"] = round((time.perf_counter() - t1) * 1000, 2)
            out["lime"] = lw.tolist()
            out["abec"] = abec(sv, lw, self.det.features, self.cfg["top_k"], self.cfg["negligible"])
        return out

    def fidelity(self, x_raw, cls: int, top_idx) -> float:
        """Confidence drop when the given top features are replaced by the background mean (paper Table 3)."""
        xs = self.det.scale(x_raw)[0]
        before = self.det.proba_scaled(xs)[0, cls]
        z = xs.copy()
        z[list(top_idx)] = self.bg_mean[list(top_idx)]
        return float(before - self.det.proba_scaled(z)[0, cls])


def _norm(v: np.ndarray) -> np.ndarray:
    m = np.abs(v).max()
    return v / m if m > 0 else v * 0


def top_k(v, k: int) -> list[int]:
    return [int(i) for i in np.argsort(-np.abs(np.asarray(v)), kind="stable")[:k]]


def abec(shap_v, lime_v, features: list[str], k: int = 5, negligible: float = 0.10) -> dict:
    """Agreement-Based Explanation Confidence.

    J = |Sk ∩ Lk| / |Sk ∪ Lk|   (Sk, Lk = top-k features by |SHAP| and |LIME|)
    Tiers for every feature in Sk ∪ Lk:
      High Confidence : in both top-k sets with the same sign; score = mean of the normalised SHAP and LIME scores
      Disputed        : top-k in one method but negligible (|normalised| < `negligible`) or of opposite sign
                        in the other (this includes features in both sets whose signs disagree)
      Single-Source   : top-k in one method only, same sign and non-negligible in the other;
                        reported with that method's score and labelled with its source
    Scores are normalised per method by dividing by the largest |score| (min-max on magnitudes, sign kept).
    """
    s, l = np.asarray(shap_v, float), np.asarray(lime_v, float)
    ns, nl = _norm(s), _norm(l)
    S, L = set(top_k(s, k)), set(top_k(l, k))
    union, inter = S | L, S & L
    J = len(inter) / len(union) if union else 1.0
    rows = []
    for j in sorted(union, key=lambda i: -max(abs(ns[i]), abs(nl[i]))):
        opposite = np.sign(ns[j]) != np.sign(nl[j]) and ns[j] != 0 and nl[j] != 0
        if j in inter and not opposite:
            tier, src, score = "High Confidence", "SHAP+LIME", (ns[j] + nl[j]) / 2
        else:
            src = "SHAP+LIME" if j in inter else ("SHAP" if j in S else "LIME")
            other = nl[j] if j in S else ns[j]
            if opposite or abs(other) < negligible:
                tier = "Disputed"
            else:
                tier = "Single-Source"
            score = ns[j] if j in S else nl[j]
        rows.append({"feature": features[j], "index": int(j), "tier": tier, "source": src,
                     "score": round(float(score), 4), "shap_norm": round(float(ns[j]), 4),
                     "lime_norm": round(float(nl[j]), 4)})
    order = {"High Confidence": 0, "Single-Source": 1, "Disputed": 2}
    ranked = sorted(rows, key=lambda r: (order[r["tier"]], -abs(r["score"])))
    return {"jaccard": round(J, 4), "k": k, "tiers": rows,
            "n_high": sum(r["tier"] == "High Confidence" for r in rows),
            "n_single": sum(r["tier"] == "Single-Source" for r in rows),
            "n_disputed": sum(r["tier"] == "Disputed" for r in rows),
            "fused_top": [r["index"] for r in ranked][:k]}
