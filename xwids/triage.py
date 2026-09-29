"""Stage 4: alert triage engine (paper: tau = 0.85).

AUTO      confidence >= tau and no anomaly: resolved autonomously, SHAP explanation archived
ESCALATE  confidence <  tau, or the classifier says Normal but the Isolation Forest score is beyond the
          zero-day threshold (possible unknown attack): sent to the analyst with SHAP + LIME + ABEC
The Isolation Forest's own contamination-0.05 flag is kept as the `anomaly` hint shown on every alert.

An "alert" is any window that is predicted as an attack or escalated. Escalation rate = escalated / alerts,
so a fully manual SOC (every alert reviewed by a person) is 100%, as in the paper's Table 5 baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import NORMAL

AUTO, ESCALATE = "AUTO", "ESCALATE"


def triage(det, X: pd.DataFrame, tau: float = 0.85) -> list[dict]:
    P = det.predict_proba(X)
    anom, ascore = det.anomaly(X)
    classes = np.asarray(det.classes_)
    out = []
    for i in range(len(P)):
        k = int(P[i].argmax())
        pred, conf = str(classes[k]), float(P[i, k])
        reasons = []
        if conf < tau:
            reasons.append(f"confidence {conf:.2f} < tau {tau:.2f}")
        zero_day = bool(pred == NORMAL and ascore[i] > getattr(det, "zero_day_threshold", np.inf))
        if zero_day:
            reasons.append("strong Isolation Forest anomaly while classifier says Normal (possible zero-day)")
        route = ESCALATE if reasons else AUTO
        if route == AUTO:
            reasons.append("auto-resolved attack, SHAP archived" if pred != NORMAL else "auto-cleared as Normal")
        is_alert = pred != NORMAL or route == ESCALATE
        priority = (1 - conf) + (0.5 if zero_day else 0.1 if anom[i] else 0.0) + (0.25 if pred != NORMAL else 0.0)
        out.append({"pred": pred, "pred_index": k, "confidence": conf, "probs": P[i].round(5).tolist(),
                    "anomaly": bool(anom[i]), "zero_day": zero_day, "anomaly_score": float(ascore[i]), "route": route,
                    "reason": "; ".join(reasons), "is_alert": bool(is_alert), "priority": round(float(priority), 4)})
    return out


def escalation_rate(results: list[dict]) -> float:
    alerts = [r for r in results if r["is_alert"]]
    return float(np.mean([r["route"] == ESCALATE for r in alerts])) if alerts else 0.0
