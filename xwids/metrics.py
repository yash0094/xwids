"""Detection metrics used in the paper's tables."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .common import NORMAL


def fpr(y_true, y_pred) -> float:
    """False Positive Rate: share of truly Normal windows that were flagged as any attack."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    normal = y_true == NORMAL
    return float((y_pred[normal] != NORMAL).mean()) if normal.any() else 0.0


def detection(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    atk_t, atk_p = y_true != NORMAL, y_pred != NORMAL
    dr = float((atk_p[atk_t]).mean()) if atk_t.any() else 0.0
    return {"n": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(p_m), "recall_macro": float(r_m), "f1_macro": float(f_m),
            "precision_weighted": float(p_w), "recall_weighted": float(r_w), "f1_weighted": float(f_w),
            "fpr": fpr(y_true, y_pred), "detection_rate": dr}


def per_class(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str)
    labels = sorted(set(y_true) | set(y_pred))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {c: {"precision": float(a), "recall": float(b), "f1": float(d), "support": int(e)}
            for c, a, b, d, e in zip(labels, p, r, f, s)}


def confusion(y_true, y_pred) -> dict:
    labels = sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    return {"labels": labels, "matrix": confusion_matrix(np.asarray(y_true).astype(str),
                                                         np.asarray(y_pred).astype(str), labels=labels).tolist()}
