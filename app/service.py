"""The running system behind the dashboard: model, explainer, event store, replay stream and feedback loop."""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from xwids import features as FT
from xwids import registry
from xwids.common import NORMAL, backends, load_config, log, read_json, use_dataset
from xwids.feedback import RetrainWorker, retrain
from xwids.prep import load_meta, load_split
from xwids.store import Store
from xwids.triage import AUTO, ESCALATE, triage


class Service:
    def __init__(self, dataset: str | None = None):
        dataset = dataset or os.environ.get("XWIDS_DATASET", "synthetic")
        self.cfg = use_dataset(load_config(), dataset)
        if os.environ.get("XWIDS_RESET") == "1" and Path(self.cfg["paths"]["db"]).exists():
            Path(self.cfg["paths"]["db"]).unlink()
            log.info("XWIDS_RESET=1: started with an empty event database")
        self.store = Store(self.cfg["paths"]["db"])
        self.meta = load_meta(self.cfg)
        self.lock = threading.RLock()
        self._load_model()
        self.base_train = load_split(self.cfg, "train")
        self.val = load_split(self.cfg, "val")
        test = load_split(self.cfg, "test")
        # replay stream = held-out test windows in time order (as if arriving live)
        self.stream = test.sort_values(["win_start", "bssid"]).reset_index(drop=True)
        self.cursor = int((self.store.one("SELECT COUNT(*) c FROM events WHERE source='replay'") or {"c": 0})["c"])
        self.worker = RetrainWorker()
        self.tau = self.cfg["triage"]["tau"]
        self.every = self.cfg["feedback"]["retrain_every"]
        self.store.audit("system", "startup", {"dataset": dataset, "model": self.det.version})

    def _load_model(self):
        self.det, self.ex, reg = registry.load(self.cfg)
        self.live_version = reg["live"]

    def maybe_reload(self):
        reg = read_json(Path(self.cfg["paths"]["models"]) / "registry.json")
        if reg["live"] != self.live_version:
            with self.lock:
                self._load_model()
            log.info("loaded new live model v%s", self.live_version)

    # ------------------------------------------------------------------ ingest
    def _ingest(self, df: pd.DataFrame, source: str, true_labels=None) -> list[int]:
        with self.lock:
            res = triage(self.det, df, self.tau)
            rows = []
            for i, r in enumerate(res):
                x = df.iloc[[i]]
                expl = None
                if r["route"] == ESCALATE:
                    expl = self.ex.explain(x, cls=r["pred_index"], lime=True, seed=0)  # SHAP + LIME + ABEC
                elif r["pred"] != NORMAL:
                    expl = self.ex.explain(x, cls=r["pred_index"], lime=False)  # archived SHAP (paper stage 4)
                rows.append({**r, "source": source, "features": {f: float(x.iloc[0][f]) for f in self.det.features},
                             "bssid": str(x.iloc[0].get("bssid", "")), "win_start": float(x.iloc[0].get("win_start", 0)),
                             "true_label": None if true_labels is None else str(true_labels[i]),
                             "model_version": self.det.version, "explanation": expl,
                             "telemetry": {f: float(x.iloc[0][f]) for f in FT.FEATURES if f in x.columns},
                             "jaccard": expl["abec"]["jaccard"] if expl and "abec" in expl else None})
            ids = self.store.add_events(rows)
        return ids

    def replay(self, n: int = 25) -> dict:
        n = max(1, min(int(n), 500))
        idx = [(self.cursor + i) % len(self.stream) for i in range(n)]
        self.cursor += n
        df = self.stream.iloc[idx].reset_index(drop=True)
        ids = self._ingest(df, "replay", df["label"].tolist())
        esc = self.store.one(f"SELECT COUNT(*) c FROM events WHERE id IN ({','.join(map(str, ids))}) "
                             f"AND route='ESCALATE'")["c"]
        return {"ingested": len(ids), "escalated": esc}

    def score_rows(self, rows: list[dict], source: str = "api") -> list[dict]:
        df = pd.DataFrame(rows)
        missing = [f for f in self.det.features if f not in df.columns]
        if missing:
            raise ValueError(f"missing features: {missing}")
        ids = self._ingest(df, source)
        return [self.public(self.store.event(i)) for i in ids]

    def score_frames(self, frames: list[dict], source: str = "api-frames") -> list[dict]:
        from xwids import frames as FR
        raw = pd.DataFrame(frames)
        fr = FR.from_tshark_columns(raw, "api") if "wlan.fc.type_subtype" in raw.columns else raw
        for c in FR.COLUMNS:
            if c not in fr.columns:
                fr[c] = NORMAL if c == "label" else 0
        fc = self.cfg["features"]
        feats = FT.extract(fr, fc["window_seconds"], 1, fc["attack_fraction"])
        return self.score_rows(feats.to_dict("records"), source) if len(feats) else []

    @staticmethod
    def public(e: dict) -> dict:
        keep = ["id", "pred", "confidence", "route", "reason", "anomaly", "jaccard", "model_version", "bssid"]
        return {k: e[k] for k in keep}

    # ------------------------------------------------------------------ analyst
    def explanation(self, eid: int) -> dict | None:
        e = self.store.event(eid)
        if not e:
            return None
        if not e["explanation"] or "abec" not in e["explanation"]:  # computed on demand (paper V-D)
            with self.lock:
                x = pd.DataFrame([e["features"]])
                cls = self.det.classes_.index(e["pred"]) if e["pred"] in self.det.classes_ else None
                e["explanation"] = self.ex.explain(x, cls=cls, lime=True, seed=0)
            self.store.set_explanation(eid, e["explanation"])
            e = self.store.event(eid)
        return e

    def verdict(self, eid: int, label: str, analyst: str, note: str = "") -> dict:
        e = self.store.event(eid)
        if not e:
            raise KeyError(eid)
        if label not in self.det.classes_:
            raise ValueError(f"unknown class {label}")
        kind = ("TP" if label == e["pred"] else "RECLASS") if label != NORMAL else ("FP" if e["pred"] != NORMAL else "TN")
        if e["pred"] == NORMAL and label != NORMAL:
            kind = "FN"
        self.store.verdict(eid, label, kind, analyst, note)
        self.store.audit(analyst, "verdict", {"event": eid, "pred": e["pred"], "verdict": label, "kind": kind,
                                              "confidence": round(e["confidence"], 4), "jaccard": e["jaccard"]})
        started = self.maybe_retrain(analyst)
        return {"kind": kind, "retrain_started": started}

    def maybe_retrain(self, actor: str = "system", force: bool = False) -> bool:
        pending = self.store.stats()["pending"]
        if (pending >= self.every or (force and pending > 0)) and not self.worker.busy:
            self.store.audit(actor, "retrain_triggered", {"pending_labels": pending, "forced": force})
            return self.worker.start(lambda: self._retrain_and_reload(actor))
        return False

    def _retrain_and_reload(self, actor):
        out = retrain(self.cfg, self.det, self.store, self.base_train, self.val, actor)
        self.maybe_reload()
        return out

    # ------------------------------------------------------------------ views
    def overview(self) -> dict:
        s = self.store.stats()
        s["pending_to_retrain"] = max(0, self.every - s["pending"])
        truth = self.store.one("SELECT COUNT(*) n, SUM(verdict=true_label) ok FROM events "
                               "WHERE verdict IS NOT NULL AND true_label IS NOT NULL") or {}
        s["analyst_checked"], s["analyst_agree"] = truth.get("n") or 0, truth.get("ok") or 0
        return {"stats": s, "model": self.det.info(), "backends": {**backends(), **self.ex.backend},
                "dataset": self.cfg["dataset"], "synthetic": self.meta.get("synthetic", False),
                "tau": self.tau, "retrain_every": self.every, "busy": self.worker.busy,
                "last_error": self.worker.error, "stream": {"cursor": self.cursor, "size": len(self.stream)}}

    def retrain_history(self) -> list[dict]:
        import json
        rows = self.store.q("SELECT * FROM retrains ORDER BY id")
        for r in rows:
            r["metrics"] = json.loads(r["metrics"])
        return rows
