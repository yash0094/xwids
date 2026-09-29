"""SQLite store for events, analyst verdicts, retraining runs and a hash-chained (tamper-evident) audit log."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT, source TEXT, bssid TEXT, win_start REAL,
  features TEXT,                -- JSON {feature: raw value}
  true_label TEXT,              -- only for replayed test data; hidden from the analyst (blind review)
  pred TEXT, confidence REAL, probs TEXT, anomaly INTEGER, anomaly_score REAL,
  route TEXT, reason TEXT, is_alert INTEGER, priority REAL,
  model_version INTEGER,
  explanation TEXT,             -- JSON: SHAP (always) + LIME + ABEC (escalated, computed on demand)
  jaccard REAL,
  verdict TEXT, verdict_kind TEXT, analyst TEXT, verdict_ts TEXT, note TEXT,
  used_in_version INTEGER,
  telemetry TEXT                -- JSON: all 33 window measurements (used by the Net Doctor console)
);
CREATE INDEX IF NOT EXISTS ix_route ON events(route, verdict);
CREATE TABLE IF NOT EXISTS retrains (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, version INTEGER, labels_total INTEGER, labels_new INTEGER,
  seconds REAL, metrics TEXT, promoted INTEGER, note TEXT
);
CREATE TABLE IF NOT EXISTS ap_names (bssid TEXT PRIMARY KEY, name TEXT);
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, actor TEXT, action TEXT, detail TEXT, prev_hash TEXT, hash TEXT
);
"""


class Store:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(events)")}
        if "telemetry" not in cols:  # database created by an older version
            self.db.execute("ALTER TABLE events ADD COLUMN telemetry TEXT")
        self.db.commit()

    # ---------------------------------------------------------------- generic
    def q(self, sql: str, args=()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    def one(self, sql: str, args=()):
        r = self.q(sql, args)
        return r[0] if r else None

    def x(self, sql: str, args=()) -> int:
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur.lastrowid

    # ---------------------------------------------------------------- events
    def add_events(self, rows: list[dict]) -> list[int]:
        ids = []
        with self.lock:
            for r in rows:
                cur = self.db.execute(
                    "INSERT INTO events(ts,source,bssid,win_start,features,true_label,pred,confidence,probs,anomaly,"
                    "anomaly_score,route,reason,is_alert,priority,model_version,explanation,jaccard,telemetry) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (time.strftime("%Y-%m-%d %H:%M:%S"), r.get("source", "replay"), r.get("bssid"), r.get("win_start"),
                     json.dumps(r["features"]), r.get("true_label"), r["pred"], r["confidence"], json.dumps(r["probs"]),
                     int(r["anomaly"]), r["anomaly_score"], r["route"], r["reason"], int(r["is_alert"]),
                     r["priority"], r["model_version"], json.dumps(r.get("explanation")) if r.get("explanation")
                     else None, r.get("jaccard"), json.dumps(r.get("telemetry") or {})))
                ids.append(cur.lastrowid)
            self.db.commit()
        return ids

    def set_explanation(self, eid: int, expl: dict):
        j = expl.get("abec", {}).get("jaccard")
        self.x("UPDATE events SET explanation=?, jaccard=COALESCE(?, jaccard) WHERE id=?", (json.dumps(expl), j, eid))

    def event(self, eid: int):
        e = self.one("SELECT * FROM events WHERE id=?", (eid,))
        if e:
            e["features"] = json.loads(e["features"])
            e["probs"] = json.loads(e["probs"])
            e["explanation"] = json.loads(e["explanation"]) if e["explanation"] else None
        return e

    def verdict(self, eid: int, label: str, kind: str, analyst: str, note: str = ""):
        self.x("UPDATE events SET verdict=?, verdict_kind=?, analyst=?, verdict_ts=?, note=? WHERE id=?",
               (label, kind, analyst, time.strftime("%Y-%m-%d %H:%M:%S"), note, eid))

    def labelled(self, unused_only: bool = False) -> list[dict]:
        sql = "SELECT id, features, verdict FROM events WHERE verdict IS NOT NULL"
        if unused_only:
            sql += " AND used_in_version IS NULL"
        return [{"id": r["id"], "features": json.loads(r["features"]), "label": r["verdict"]} for r in self.q(sql)]

    def mark_used(self, ids: list[int], version: int):
        with self.lock:
            self.db.executemany("UPDATE events SET used_in_version=? WHERE id=?", [(version, i) for i in ids])
            self.db.commit()

    def stats(self) -> dict:
        s = self.one("SELECT COUNT(*) n, SUM(is_alert) alerts, SUM(route='ESCALATE') esc, SUM(route='AUTO') auto, "
                     "SUM(route='ESCALATE' AND verdict IS NULL) queue, SUM(verdict IS NOT NULL) labelled, "
                     "SUM(verdict IS NOT NULL AND used_in_version IS NULL) pending, "
                     "SUM(verdict_kind='FP') fp, SUM(verdict_kind='TP') tp, "
                     "SUM(route='AUTO' AND pred!='Normal') auto_attacks FROM events") or {}
        s = {k: (v or 0) for k, v in s.items()}
        s["escalation_rate"] = (s["esc"] / s["alerts"]) if s["alerts"] else 0.0
        return s

    # ---------------------------------------------------------------- audit (hash chain)
    def audit(self, actor: str, action: str, detail: dict | str = ""):
        detail = detail if isinstance(detail, str) else json.dumps(detail, sort_keys=True)
        with self.lock:
            last = self.db.execute("SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
            prev = last[0] if last else "0" * 64
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            h = hashlib.sha256(f"{prev}|{ts}|{actor}|{action}|{detail}".encode()).hexdigest()
            self.db.execute("INSERT INTO audit(ts,actor,action,detail,prev_hash,hash) VALUES(?,?,?,?,?,?)",
                            (ts, actor, action, detail, prev, h))
            self.db.commit()

    def verify_audit(self) -> tuple[bool, int]:
        prev = "0" * 64
        rows = self.q("SELECT * FROM audit ORDER BY id")
        for r in rows:
            h = hashlib.sha256(f"{prev}|{r['ts']}|{r['actor']}|{r['action']}|{r['detail']}".encode()).hexdigest()
            if r["prev_hash"] != prev or r["hash"] != h:
                return False, r["id"]
            prev = r["hash"]
        return True, len(rows)
