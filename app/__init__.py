"""Flask analyst dashboard (paper stage 5): escalated-alert queue, SHAP waterfall + LIME + ABEC tiers, verdicts,
retraining every 50 labels, audit log. Logins come from XWIDS_USERS="name:password,name2:password2"."""
from __future__ import annotations

import csv
import hmac
import io
import os
import secrets
import time
from functools import wraps

from flask import (Flask, Response, abort, flash, g, jsonify, redirect, render_template, request, session,
                   url_for)

from xwids.common import NORMAL

from . import charts
from .service import Service

DEFAULT_USERS = "analyst:xwids-demo,reviewer:xwids-review"


def _users() -> dict:
    out = {}
    for pair in os.environ.get("XWIDS_USERS", DEFAULT_USERS).split(","):
        if ":" in pair:
            u, p = pair.split(":", 1)
            out[u.strip()] = p.strip()
    return out


def create_app(dataset: str | None = None) -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("XWIDS_SECRET_KEY") or secrets.token_hex(32)
    app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")
    svc = Service(dataset)
    app.svc = svc
    users = _users()
    api_key = os.environ.get("XWIDS_API_KEY", "")
    if os.environ.get("XWIDS_USERS") is None:
        app.logger.warning("XWIDS_USERS not set: using the DEMO logins %s. Set real passwords before sharing a URL.",
                           DEFAULT_USERS)

    # ------------------------------------------------------------------ auth + CSRF
    def login_required(fn):
        @wraps(fn)
        def wrap(*a, **k):
            if "user" not in session:
                return redirect(url_for("login", next=request.path))
            g.user = session["user"]
            return fn(*a, **k)
        return wrap

    @app.before_request
    def csrf():
        session.setdefault("csrf", secrets.token_hex(16))
        if request.method == "POST" and not request.path.startswith("/api/"):
            sent = request.headers.get("X-CSRF-Token") or request.form.get("csrf", "")
            if not hmac.compare_digest(sent, session["csrf"]):
                abort(400, "CSRF token missing or wrong - reload the page")

    @app.context_processor
    def inject():
        return {"csrf": session.get("csrf", ""), "user": session.get("user"), "NORMAL": NORMAL}

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            u, p = request.form.get("username", ""), request.form.get("password", "")
            if u in users and hmac.compare_digest(users[u].encode(), p.encode()):
                session["user"] = u
                svc.store.audit(u, "login")
                nxt = request.args.get("next", "/")
                return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else "/")
            time.sleep(0.5)
            flash("Wrong username or password")
        return render_template("login.html", demo=os.environ.get("XWIDS_USERS") is None)

    @app.route("/logout", methods=["POST"])
    def logout():
        session.pop("user", None)
        return redirect(url_for("login"))

    from . import doctor
    doctor.register(app, svc, login_required)

    # ------------------------------------------------------------------ pages
    @app.route("/")
    @login_required
    def index():
        svc.maybe_reload()
        ov = svc.overview()
        recent = svc.store.q("SELECT id,ts,bssid,pred,confidence,route,jaccard,anomaly,verdict FROM events "
                             "WHERE is_alert=1 ORDER BY id DESC LIMIT 12")
        return render_template("index.html", ov=ov, recent=recent)

    @app.route("/replay", methods=["POST"])
    @login_required
    def replay():
        r = svc.replay(int(request.form.get("n", 25)))
        svc.store.audit(g.user, "replay", r)
        flash(f"Replayed {r['ingested']} traffic windows: {r['escalated']} escalated to the analyst queue")
        return redirect(request.form.get("back") or url_for("index"))

    @app.route("/queue")
    @login_required
    def queue():
        order = request.args.get("order", "priority")
        sql = ("SELECT id,ts,bssid,pred,confidence,anomaly,jaccard,reason,priority FROM events "
               "WHERE route='ESCALATE' AND verdict IS NULL ORDER BY ")
        low = svc.cfg["triage"]["low_agreement"]
        sql += {"jaccard": "COALESCE(jaccard,1) ASC, confidence ASC",
                "confidence": "confidence ASC",
                "newest": "id DESC"}.get(order, f"(priority + (COALESCE(jaccard,1) < {low}) * 0.5) DESC, id ASC")
        rows = svc.store.q(sql + " LIMIT 300")
        return render_template("queue.html", rows=rows, order=order, low=low, ov=svc.overview())

    @app.route("/alert/<int:eid>")
    @login_required
    def alert(eid):
        e = svc.explanation(eid)
        if not e:
            abort(404)
        ex = e["explanation"]
        feats = ex["features"]
        shap_svg = charts.waterfall(feats, ex["shap"], ex["values"], ex["shap_base"])
        lime_svg = charts.bars(feats, ex["lime"]) if ex.get("lime") else ""
        nxt = svc.store.one("SELECT id FROM events WHERE route='ESCALATE' AND verdict IS NULL AND id!=? "
                            "ORDER BY priority DESC, id LIMIT 1", (eid,))
        from xwids.features import CANDIDATES
        return render_template("alert.html", e=e, ex=ex, shap_svg=shap_svg, lime_svg=lime_svg,
                               classes=svc.det.classes_, desc=CANDIDATES, nxt=nxt,
                               probs=sorted(zip(svc.det.classes_, e["probs"]), key=lambda t: -t[1]))

    @app.route("/alert/<int:eid>/verdict", methods=["POST"])
    @login_required
    def verdict(eid):
        label = request.form.get("label", "")
        if label == "__other__":
            label = request.form.get("label_other", "")
        try:
            r = svc.verdict(eid, label, g.user, request.form.get("note", "")[:500])
        except (KeyError, ValueError) as err:
            abort(400, str(err))
        msg = f"Saved verdict {label} ({r['kind']})."
        if r["retrain_started"]:
            msg += f" {svc.every} new labels reached: retraining started in the background."
        flash(msg)
        nxt = svc.store.one("SELECT id FROM events WHERE route='ESCALATE' AND verdict IS NULL "
                            "ORDER BY priority DESC, id LIMIT 1")
        return redirect(url_for("alert", eid=nxt["id"]) if nxt and request.form.get("go_next") else url_for("queue"))

    @app.route("/auto")
    @login_required
    def auto():
        rows = svc.store.q("SELECT id,ts,bssid,pred,confidence,explanation FROM events WHERE route='AUTO' "
                           "AND pred!=? ORDER BY id DESC LIMIT 200", (NORMAL,))
        import json
        for r in rows:
            ex = json.loads(r.pop("explanation") or "null")
            if ex:
                order = sorted(range(len(ex["shap"])), key=lambda i: -abs(ex["shap"][i]))[:3]
                r["top"] = ", ".join(f"{ex['features'][i]} ({ex['shap'][i]:+.2f})" for i in order)
        return render_template("auto.html", rows=rows)

    @app.route("/learning")
    @login_required
    def learning():
        svc.maybe_reload()
        hist = svc.retrain_history()
        from xwids.common import read_json
        from pathlib import Path
        reg = read_json(Path(svc.cfg["paths"]["models"]) / "registry.json")
        pts = []
        base = next((h for h in reg["history"] if h["version"] == 1), None)
        if base and base["metrics"].get("val"):
            pts.append(("v1", base["metrics"]["val"]))
        for h in hist:
            pts.append((f"v{h['version']}", h["metrics"]["val_after"]))
        charts_ = {k: charts.trend([(n, m[k]) for n, m in pts], label=k)
                   for k in ("f1_macro", "fpr", "escalation_rate")}
        return render_template("learning.html", hist=hist, charts=charts_, ov=svc.overview(), reg=reg)

    @app.route("/retrain", methods=["POST"])
    @login_required
    def retrain_now():
        ok = svc.maybe_retrain(g.user, force=True)
        flash("Retraining started in the background." if ok else "Nothing new to learn from (or already running).")
        return redirect(url_for("learning"))

    @app.route("/audit")
    @login_required
    def audit():
        ok, n = svc.store.verify_audit()
        rows = svc.store.q("SELECT * FROM audit ORDER BY id DESC LIMIT 300")
        return render_template("audit.html", rows=rows, ok=ok, n=n)

    @app.route("/audit.csv")
    @login_required
    def audit_csv():
        buf = io.StringIO()
        rows = svc.store.q("SELECT * FROM audit ORDER BY id")
        w = csv.DictWriter(buf, fieldnames=["id", "ts", "actor", "action", "detail", "prev_hash", "hash"])
        w.writeheader()
        w.writerows(rows)
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=xwids_audit.csv"})

    @app.route("/verdicts.csv")
    @login_required
    def verdicts_csv():
        rows = svc.store.q("SELECT id,ts,bssid,pred,confidence,jaccard,route,verdict,verdict_kind,analyst,verdict_ts,"
                           "note,true_label,model_version FROM events WHERE verdict IS NOT NULL ORDER BY id")
        buf = io.StringIO()
        if rows:
            w = csv.DictWriter(buf, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=xwids_verdicts.csv"})

    # ------------------------------------------------------------------ API
    def api_auth():
        if not api_key:
            abort(403, "API disabled: set XWIDS_API_KEY")
        if not hmac.compare_digest(request.headers.get("X-API-Key", ""), api_key):
            abort(401)

    @app.route("/api/score", methods=["POST"])
    def api_score():
        """POST {"rows": [{feature: value, ...}]} or {"frames": [tshark/AWID3-style frame dicts]}."""
        api_auth()
        body = request.get_json(force=True, silent=True) or {}
        try:
            if "frames" in body:
                out = svc.score_frames(body["frames"])
            else:
                out = svc.score_rows(body.get("rows", []))
        except ValueError as err:
            return jsonify({"error": str(err)}), 400
        return jsonify({"results": out, "model_version": svc.det.version})

    @app.route("/api/features")
    def api_features():
        return jsonify({"features": svc.det.features, "classes": svc.det.classes_})

    @app.route("/healthz")
    def healthz():
        return jsonify({"ok": True, "model_version": svc.det.version, "dataset": svc.cfg["dataset"]})

    return app
