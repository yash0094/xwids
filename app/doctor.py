"""Net Doctor console: a phone-friendly Wi-Fi diagnostic view on top of the IDS.

Seven tabs, like the Net Doctor prototype, but every number is computed from real data:
  DASH      health score 0-100 from recent IDS windows (threats, signal, retries, encryption, backlog, anomalies)
  DIAG      8-step diagnostic: browser->server ping / download / upload measured live, fresh traffic capture,
            then IDS scans for deauth floods, evil twin, KRACK + association floods, and radio health
  RECO      rule-based recommendations triggered by what was actually measured, with the evidence shown
  ZONE      signal map: one heat spot per access point (BSSID), coloured by mean RSSI (>= -65 good, -65..-75
            moderate, < -75 weak). Positions are illustrative; you can name each AP (e.g. "Kitchen")
  DEVI      access points with client counts / traffic / threats, plus the phone's own connection
  TIME      timeline built from the audit log and detections
  FORE      linear-trend forecast of attack activity over recent traffic blocks
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np
from flask import Blueprint, Response, abort, jsonify, redirect, render_template, request, session, url_for

from xwids.common import NORMAL

GOOD, WEAK = -65.0, -75.0
ATTACK_INFO = {
    "Deauthentication": "spoofed deauthentication frames knocking clients off the network",
    "(Re)Association Flood": "a flood of fake (re)association requests exhausting the access point",
    "Evil Twin": "a rogue access point cloning your network name / BSSID",
    "KRACK": "replayed WPA handshake messages (key reinstallation attack)",
}


def _rssi_band(v):
    if v is None:
        return "unknown"
    return "good" if v >= GOOD else ("moderate" if v >= WEAK else "weak")


def recent_events(store, n=600):
    rows = store.q("SELECT id, ts, bssid, win_start, pred, confidence, route, anomaly, verdict, telemetry "
                   "FROM events ORDER BY id DESC LIMIT ?", (n,))
    for r in rows:
        r["t"] = json.loads(r.pop("telemetry") or "{}")
    return rows[::-1]


def ap_table(store, events):
    names = {r["bssid"]: r["name"] for r in store.q("SELECT * FROM ap_names")}
    by = defaultdict(list)
    for e in events:
        if e["bssid"]:
            by[e["bssid"]].append(e)
    aps = []
    for b, ev in by.items():
        t = [e["t"] for e in ev if e["t"]]
        mean = lambda k: float(np.mean([x[k] for x in t if k in x])) if any(k in x for x in t) else None
        attacks = Counter(e["pred"] for e in ev if e["pred"] != NORMAL)
        rssi = mean("rssi_mean")
        aps.append({"bssid": b, "name": names.get(b) or "", "windows": len(ev), "rssi": rssi, "band": _rssi_band(rssi),
                    "clients": int(max([x.get("unique_ta", 0) for x in t] or [0])),
                    "frames_per_s": mean("frames_per_s"), "retry": mean("retry_rate"),
                    "protected": mean("protected_ratio"), "channels": int(max([x.get("n_channels", 0) for x in t] or [0])),
                    "attacks": dict(attacks), "attack_share": sum(attacks.values()) / len(ev),
                    "last_id": ev[-1]["id"]})
    aps.sort(key=lambda a: (-a["attack_share"], a["bssid"]))
    return aps


def health(store, events, aps, client=None):
    """Score = 100 minus weighted penalties (each penalty is 0..1). Weights sum to 100."""
    n = len(events) or 1
    atk = sum(e["pred"] != NORMAL for e in events) / n
    anom = sum(bool(e["anomaly"]) for e in events) / n
    weak = (sum(a["band"] == "weak" for a in aps) / len(aps)) if aps else 0
    retry = float(np.mean([a["retry"] for a in aps if a["retry"] is not None])) if aps else 0
    prot = float(np.mean([a["protected"] for a in aps if a["protected"] is not None])) if aps else 1
    queue = (store.one("SELECT COUNT(*) c FROM events WHERE route='ESCALATE' AND verdict IS NULL") or {"c": 0})["c"]
    parts = [("Threats detected", 40, min(1.0, atk / 0.25), f"{atk:.0%} of recent windows classified as attacks"),
             ("Weak signal", 15, weak, f"{weak:.0%} of access points below {WEAK:.0f} dBm"),
             ("Retransmissions", 15, min(1.0, retry / 0.30), f"mean retry rate {retry:.1%}"),
             ("Unencrypted data", 10, min(1.0, max(0.0, 1 - prot) / 0.2), f"{prot:.0%} of data frames encrypted"),
             ("Unreviewed alerts", 10, min(1.0, queue / 20), f"{queue} alerts waiting for an analyst"),
             ("Anomalies", 10, min(1.0, anom / 0.15), f"{anom:.0%} of windows unusual to the Isolation Forest")]
    if client and client.get("latency_ms") is not None:
        lat = client["latency_ms"]
        parts.append(("Your connection", 0, 0.0, f"{lat:.0f} ms to the server, "
                      f"{client.get('down_mbps', 0):.1f} Mbps down (shown, not scored)"))
    score = round(100 - sum(w * p for _, w, p, _ in parts))
    label = "GOOD" if score >= 80 else ("MODERATE" if score >= 60 else "POOR")
    return {"score": max(0, score), "label": label, "parts": parts, "windows": len(events)}


def checks(events, aps):
    """IDS findings for DIAG steps 5-8."""
    n = len(events) or 1
    cnt = Counter(e["pred"] for e in events)

    def find(cls):
        hit = [e for e in events if e["pred"] == cls]
        where = Counter(e["bssid"] for e in hit).most_common(3)
        return {"count": len(hit), "share": len(hit) / n, "where": [w for w, _ in where],
                "status": "fail" if len(hit) else "pass"}

    weak = [a for a in aps if a["band"] == "weak"]
    hiretry = [a for a in aps if (a["retry"] or 0) > 0.15]
    unenc = [a for a in aps if (a["protected"] if a["protected"] is not None else 1) < 0.95]
    radio_bad = len(weak) + len(hiretry) + len(unenc)
    return {"windows": len(events), "classes": dict(cnt),
            "deauth": find("Deauthentication"), "evil_twin": find("Evil Twin"), "krack": find("KRACK"),
            "flood": find("(Re)Association Flood"),
            "radio": {"status": "warn" if radio_bad else "pass", "weak": [a["bssid"] for a in weak],
                      "high_retry": [a["bssid"] for a in hiretry], "unencrypted": [a["bssid"] for a in unenc]}}


def recommendations(events, aps, client=None):
    c = checks(events, aps)
    recs = []

    def add(impact, title, why, how, evidence, key):
        recs.append({"impact": impact, "title": title, "why": why, "how": how, "evidence": evidence, "key": key})

    if c["deauth"]["count"]:
        add("HIGH", "Turn on Protected Management Frames (802.11w)",
            "Deauthentication attacks work because management frames are not authenticated. With PMF required, "
            "spoofed deauth frames are ignored.",
            ["Router admin page -> Wireless -> Security", "Set 'Protected Management Frames' to Required (or Capable)",
             "Prefer WPA3-Personal / WPA2+WPA3 mixed mode"],
            f"{c['deauth']['count']} deauth windows on {', '.join(c['deauth']['where'])}", "pmf")
    if c["evil_twin"]["count"]:
        add("HIGH", "Hunt down the Evil Twin access point",
            "A second radio is beaconing with your BSSID, often on another channel, luring clients onto open Wi-Fi.",
            ["Walk the area with a Wi-Fi analyser app and look for your SSID on an unexpected channel",
             "Use WPA3 / 802.1X so clients reject an AP that cannot prove the password",
             "Tell users not to accept 'open' versions of the network"],
            f"{c['evil_twin']['count']} evil-twin windows on {', '.join(c['evil_twin']['where'])}", "twin")
    if c["krack"]["count"]:
        add("HIGH", "Patch clients and router against KRACK",
            "Replayed handshake messages (repeated sequence numbers with EAPOL) point to key reinstallation.",
            ["Update router firmware", "Update every client OS (phones, laptops, IoT)", "Move to WPA3 where possible"],
            f"{c['krack']['count']} KRACK windows on {', '.join(c['krack']['where'])}", "krack")
    if c["flood"]["count"]:
        add("MEDIUM", "Rate-limit association requests",
            "Thousands of fake association requests fill the AP's client table so real users cannot join.",
            ["Enable client limits / association rate limiting on the AP",
             "Enable 'airtime fairness' and block MAC addresses that never finish the handshake"],
            f"{c['flood']['count']} flood windows on {', '.join(c['flood']['where'])}", "flood")
    if c["radio"]["unencrypted"]:
        add("HIGH", "Encrypt all data traffic",
            "Unprotected data frames were seen: anyone nearby can read them.",
            ["Set security to WPA2-AES or WPA3; never 'Open' or WEP", "Check for a guest network left open"],
            f"low encryption share on {', '.join(c['radio']['unencrypted'][:3])}", "encrypt")
    if c["radio"]["high_retry"]:
        add("MEDIUM", "Change channel or switch to 5 GHz",
            "Many frames are being re-sent, which usually means interference or a crowded 2.4 GHz channel.",
            ["Pick channel 1, 6 or 11 on 2.4 GHz (least crowded)", "Move capable devices to 5 GHz"],
            f"retry rate above 15% on {', '.join(c['radio']['high_retry'][:3])}", "channel")
    if c["radio"]["weak"]:
        add("MEDIUM", "Fix weak-signal zones",
            "Access points heard below -75 dBm give slow, unstable connections.",
            ["Move the router higher and away from walls / metal", "Add a mesh node or access point in that area"],
            f"weak signal at {', '.join(c['radio']['weak'][:3])}", "signal")
    if client and client.get("latency_ms") and client["latency_ms"] > 80:
        add("MEDIUM", "High latency from this device",
            "Round trips to the server are slow; video calls and games will stutter.",
            ["Move closer to the router or use 5 GHz", "Enable QoS for real-time traffic", "Pause big downloads"],
            f"{client['latency_ms']:.0f} ms average, jitter {client.get('jitter_ms', 0):.0f} ms", "latency")
    if not recs:
        add("LOW", "No action needed", "No attacks or radio problems in the recent traffic.",
            ["Keep firmware updated", "Run the diagnostic again after changes"], f"{c['windows']} windows checked", "none")
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    return sorted(recs, key=lambda r: order[r["impact"]])


def zone_layout(aps, w=600, h=460):
    """Deterministic, illustrative positions (hash of the BSSID) so the map is stable between visits."""
    out = []
    for i, a in enumerate(aps):
        hv = int(hashlib.sha1(a["bssid"].encode()).hexdigest(), 16)
        x = 60 + (hv % 1000) / 1000 * (w - 120)
        y = 50 + ((hv >> 12) % 1000) / 1000 * (h - 100)
        out.append({**a, "x": round(x, 1), "y": round(y, 1)})
    return out


def forecast(events, block=25, threshold=0.20):
    """Share of attack windows per block of `block` events; least-squares trend; when will it cross threshold?"""
    blocks = [events[i:i + block] for i in range(0, len(events) - len(events) % block, block)][-10:]
    if len(blocks) < 3:
        return {"ready": False, "need": block * 3, "have": len(events)}
    share = np.array([np.mean([e["pred"] != NORMAL for e in b]) for b in blocks])
    secs = [max(1.0, (max(e["win_start"] or 0 for e in b) - min(e["win_start"] or 0 for e in b))) for b in blocks]
    x = np.arange(len(share), dtype=float)
    slope, icpt = np.polyfit(x, share, 1)
    pred = slope * x + icpt
    ss_res, ss_tot = float(((share - pred) ** 2).sum()), float(((share - share.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    now = slope * x[-1] + icpt
    per_block_min = float(np.median(secs)) / 60
    if slope > 0.005 and now < threshold:
        blocks_left = (threshold - now) / slope
        minutes = blocks_left * per_block_min
        status = "rising"
        head = f"Attack activity likely to pass {threshold:.0%} in about {max(1, round(minutes))} min"
    elif now >= threshold:
        status, minutes = "high", 0
        head = f"Attack activity already above {threshold:.0%} of traffic"
    else:
        status, minutes = "stable", None
        head = "No rise in attack activity predicted"
    last = [e for b in blocks[-3:] for e in b if e["pred"] != NORMAL]
    top = Counter(e["bssid"] for e in last).most_common(1)
    cls = Counter(e["pred"] for e in last).most_common(1)
    return {"ready": True, "status": status, "headline": head, "minutes": minutes, "slope": float(slope),
            "confidence": max(0.0, min(1.0, r2)), "current": float(max(0.0, now)), "series": share.round(3).tolist(),
            "threshold": threshold, "block": block, "top_bssid": top[0][0] if top else None,
            "top_class": cls[0][0] if cls else None}


def timeline(store, limit=60):
    items = []
    for a in store.q("SELECT * FROM audit WHERE action IN ('verdict','retrain','fix_done','replay','ap_named',"
                     "'diagnostic') ORDER BY id DESC LIMIT ?", (limit,)):
        d = json.loads(a["detail"]) if a["detail"].startswith("{") else {}
        if a["action"] == "verdict":
            st, title = "SUCCESS", f"Analyst verdict: {d.get('verdict')} ({d.get('kind')})"
            body = f"{a['actor']} reviewed alert #{d.get('event')} predicted as {d.get('pred')}."
        elif a["action"] == "retrain":
            st = "SUCCESS" if d.get("promoted") else "WARNING"
            title = f"Model retrained -> v{d.get('version')}"
            body = f"Learned from {d.get('labels')} analyst labels; F1 {d.get('f1_before')} -> {d.get('f1_after')}."
        elif a["action"] == "fix_done":
            st, title, body = "SUCCESS", f"Fix applied: {d.get('title')}", f"Marked done by {a['actor']}."
        elif a["action"] == "replay":
            st, title = "INFO", f"Traffic captured: {d.get('ingested')} windows"
            body = f"{d.get('escalated')} escalated to the analyst queue."
        elif a["action"] == "diagnostic":
            st, title = "INFO", f"Diagnostic run: health {d.get('score')}/100"
            body = f"{d.get('latency_ms', '?')} ms latency, {d.get('down_mbps', '?')} Mbps down."
        else:
            st, title, body = "INFO", f"Access point named {d.get('name')}", d.get("bssid", "")
        items.append({"ts": a["ts"], "status": st, "title": title, "body": body})
    for e in store.q("SELECT id, ts, bssid, pred, route, confidence FROM events WHERE pred!=? ORDER BY id DESC LIMIT ?",
                     (NORMAL, limit)):
        st = "RESOLVED" if e["route"] == "AUTO" else "ALERT"
        items.append({"ts": e["ts"], "status": st, "title": f"{e['pred']} detected",
                      "body": (f"Auto-resolved at {e['confidence']:.0%} confidence" if e["route"] == "AUTO" else
                               f"Escalated to analyst ({e['confidence']:.0%} confidence)") + f" on {e['bssid']}.",
                      "link": f"/alert/{e['id']}"})
    items.sort(key=lambda i: i["ts"], reverse=True)
    return items[:limit]


def ago(ts: str) -> str:
    try:
        s = time.time() - time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return ts
    for unit, n in (("d", 86400), ("h", 3600), ("m", 60)):
        if s >= n:
            return f"{int(s // n)}{unit} ago"
    return "just now"


# ---------------------------------------------------------------------------------------------- routes
def register(app, svc, login_required):
    bp = Blueprint("doctor", __name__, url_prefix="/doctor")
    client_metrics = {}  # last browser measurements per user (memory only)

    def ctx(tab):
        ev = recent_events(svc.store)
        aps = ap_table(svc.store, ev)
        cm = client_metrics.get(session.get("user"))
        return ev, aps, cm, {"tab": tab, "ago": ago, "net": network_name(aps), "synthetic": svc.meta.get("synthetic")}

    def network_name(aps):
        ch = sorted({a["channels"] for a in aps})
        return {"name": f"XWIDS / {svc.cfg['dataset'].upper()}", "aps": len(aps),
                "sub": f"{len(aps)} access points / model v{svc.det.version} / tau {svc.tau}"}

    @bp.route("/")
    @login_required
    def dash():
        ev, aps, cm, c = ctx("dash")
        return render_template("doctor/dash.html", h=health(svc.store, ev, aps, cm), **c)

    @bp.route("/diag")
    @login_required
    def diag():
        ev, aps, cm, c = ctx("diag")
        return render_template("doctor/diag.html", auto=request.args.get("run") == "1", **c)

    @bp.route("/reco")
    @login_required
    def reco():
        ev, aps, cm, c = ctx("reco")
        done = {json.loads(a["detail"]).get("key") for a in svc.store.q(
            "SELECT detail FROM audit WHERE action='fix_done'")}
        return render_template("doctor/reco.html", recs=recommendations(ev, aps, cm), done=done, **c)

    @bp.route("/zone")
    @login_required
    def zone():
        ev, aps, cm, c = ctx("zone")
        return render_template("doctor/zone.html", zones=zone_layout(aps), **c)

    @bp.route("/devices")
    @login_required
    def devices():
        ev, aps, cm, c = ctx("devi")
        return render_template("doctor/devices.html", aps=aps, **c)

    @bp.route("/timeline")
    @login_required
    def timeline_page():
        ev, aps, cm, c = ctx("time")
        return render_template("doctor/timeline.html", items=timeline(svc.store), **c)

    @bp.route("/forecast")
    @login_required
    def forecast_page():
        ev, aps, cm, c = ctx("fore")
        return render_template("doctor/forecast.html", f=forecast(ev), **c)

    @bp.route("/report")
    @login_required
    def report():
        ev, aps, cm, c = ctx("dash")
        return render_template("doctor/report.html", h=health(svc.store, ev, aps, cm), aps=aps,
                               recs=recommendations(ev, aps, cm), chk=checks(ev, aps), f=forecast(ev), cm=cm,
                               now=time.strftime("%Y-%m-%d %H:%M"), **c)

    # ---------------- JSON / measurement endpoints (used by the DIAG page)
    @bp.route("/api/ping")
    @login_required
    def ping():
        return jsonify({"t": time.time()})

    @bp.route("/api/blob")
    @login_required
    def blob():
        kb = max(64, min(int(request.args.get("kb", 1024)), 8192))
        return Response(os.urandom(kb * 1024), mimetype="application/octet-stream",
                        headers={"Cache-Control": "no-store"})

    @bp.route("/api/upload", methods=["POST"])
    @login_required
    def upload():
        n = len(request.get_data(cache=False)[: 9 * 1024 * 1024])
        return jsonify({"bytes": n})

    @bp.route("/api/capture", methods=["POST"])
    @login_required
    def capture():
        r = svc.replay(int(request.args.get("n", 25)))
        svc.store.audit(session["user"], "replay", r)
        return jsonify(r)

    @bp.route("/api/checks")
    @login_required
    def api_checks():
        ev = recent_events(svc.store)
        aps = ap_table(svc.store, ev)
        return jsonify(checks(ev, aps))

    @bp.route("/api/client", methods=["POST"])
    @login_required
    def api_client():
        d = request.get_json(force=True, silent=True) or {}
        keep = {k: float(d[k]) for k in ("latency_ms", "jitter_ms", "loss", "down_mbps", "up_mbps") if k in d
                and d[k] is not None}
        client_metrics[session["user"]] = keep
        ev = recent_events(svc.store)
        h = health(svc.store, ev, ap_table(svc.store, ev), keep)
        svc.store.audit(session["user"], "diagnostic", {**{k: round(v, 1) for k, v in keep.items()}, "score": h["score"]})
        return jsonify({"score": h["score"], "label": h["label"]})

    @bp.route("/fix/<key>", methods=["POST"])
    @login_required
    def fix_done(key):
        svc.store.audit(session["user"], "fix_done", {"key": key[:40], "title": request.form.get("title", "")[:120]})
        return redirect(url_for("doctor.reco"))

    @bp.route("/name", methods=["POST"])
    @login_required
    def name_ap():
        b, n = request.form.get("bssid", "")[:40], request.form.get("name", "").strip()[:40]
        if not b:
            abort(400)
        if n:
            svc.store.x("INSERT OR REPLACE INTO ap_names(bssid,name) VALUES(?,?)", (b, n))
        else:
            svc.store.x("DELETE FROM ap_names WHERE bssid=?", (b,))
        svc.store.audit(session["user"], "ap_named", {"bssid": b, "name": n})
        return redirect(request.form.get("back") or url_for("doctor.zone"))

    app.register_blueprint(bp)
