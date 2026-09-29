"""Fast checks (XWIDS_FAST=1 shrinks the forests). Run: pytest -q"""
import os

os.environ["XWIDS_FAST"] = "1"

import numpy as np
import pandas as pd
import pytest

from xwids import features as FT
from xwids import frames as FR
from xwids import simulate
from xwids.common import NORMAL, load_config, use_dataset
from xwids.explain import Explainer, abec
from xwids.models import Detector
from xwids.prep import smote, split
from xwids.store import Store
from xwids.triage import ESCALATE, escalation_rate, triage


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    c = use_dataset(load_config(), "synthetic")
    c["paths"]["db"] = str(tmp_path_factory.mktemp("db") / "t.sqlite")
    return c


@pytest.fixture(scope="module")
def windows():
    fr = simulate.generate(n_bss=4, seconds=400, seed=3)
    return FT.extract(fr)


def test_simulator_schema_and_labels():
    fr = simulate.generate(n_bss=2, seconds=120, seed=1)
    assert list(fr.columns) == FR.COLUMNS
    assert fr["time"].is_monotonic_increasing
    assert fr["seq"].between(-1, 4095).all()


def test_features_have_no_nans_and_attack_signatures(windows):
    assert not windows[FT.FEATURES].isna().any().any()
    med = windows.groupby("label")[["deauth_rate", "eapol_rate", "assoc_req_rate"]].median()
    if "Deauthentication" in med.index:
        assert med.loc["Deauthentication", "deauth_rate"] > med.loc[NORMAL, "deauth_rate"]
    if "(Re)Association Flood" in med.index:
        assert med.loc["(Re)Association Flood", "assoc_req_rate"] > med.loc[NORMAL, "assoc_req_rate"]


def test_awid3_style_csv_is_parsed():
    raw = pd.DataFrame({"frame.time_epoch": [1.0, 1.1, 1.2], "wlan.fc.type_subtype": ["0x000c", "0x0008", "0x0028"],
                        "wlan.bssid": ["AA:BB:CC:00:00:01"] * 3, "wlan.ta": ["aa:bb:cc:00:00:01"] * 3,
                        "wlan.ra": ["ff:ff:ff:ff:ff:ff"] * 3, "radiotap.dbm_antsignal": ["-40,-42", "-41", "-45"],
                        "wlan.seq": [1, 2, 3], "wlan.fc.retry": ["False", "True", "0"], "Label": ["Deauth", "Normal", "Normal"]})
    fr = FR.from_tshark_columns(raw, "t")
    assert fr["subtype"].tolist() == [12, 8, 40]
    assert fr["rssi"].tolist() == [-40, -41, -45]
    assert fr["retry"].tolist() == [0, 1, 0]
    assert fr["label"].tolist() == ["Deauthentication", NORMAL, NORMAL]
    assert fr["bssid"].iloc[0] == "aa:bb:cc:00:00:01"


def test_split_has_no_group_leakage(windows, cfg):
    parts = split(windows, cfg)
    g = lambda d: set(d["bssid"] + "|" + (d["win_start"] // 120).astype(int).astype(str))
    assert not (g(parts["train"]) & g(parts["test"]))


def test_smote_balances_training_only(windows):
    out = smote(windows, FT.FEATURES, 0)
    counts = out["label"].value_counts()
    assert counts.max() == counts.min()
    assert (out["smote"] == 1).sum() == len(out) - len(windows)


def test_abec_tiers_and_jaccard():
    feats = [f"f{i}" for i in range(8)]
    s = np.array([5, 4, 3, 2, 1, 0, 0, 0], float)
    l = np.array([5, 4, -3, 0, 0, 2, 1, 0], float)  # f2 opposite sign, f3/f4 negligible in LIME
    r = abec(s, l, feats, k=5, negligible=0.1)
    tiers = {t["feature"]: t["tier"] for t in r["tiers"]}
    assert tiers["f0"] == tiers["f1"] == "High Confidence"
    assert tiers["f2"] == "Disputed"          # both top-5, opposite signs
    assert tiers["f3"] == "Disputed"          # SHAP top-5, negligible in LIME
    assert tiers["f5"] == "Disputed"          # LIME top-5 only, SHAP is 0 (negligible)
    assert r["jaccard"] == pytest.approx(3 / 7, abs=1e-3)


def test_detector_shap_local_accuracy_and_triage(windows, cfg):
    parts = split(windows, cfg)
    tr = smote(parts["train"], FT.FEATURES, 0)
    det = Detector(FT.FEATURES[:17], cfg).fit(tr)
    ex = Explainer(det, det.scale(parts["train"]), cfg)
    x = parts["test"].iloc[[0]]
    e = ex.explain(x)
    assert e["shap_base"] + sum(e["shap"]) == pytest.approx(e["confidence"], abs=1e-6)
    assert 0 <= e["abec"]["jaccard"] <= 1
    res = triage(det, parts["test"], 0.85)
    assert all(r["route"] == ESCALATE for r in res if r["confidence"] < 0.85)
    assert 0 <= escalation_rate(res) <= 1


def test_audit_chain_detects_tampering(cfg):
    st = Store(cfg["paths"]["db"])
    st.audit("a", "x", {"k": 1})
    st.audit("b", "y", {"k": 2})
    assert st.verify_audit()[0]
    st.x("UPDATE audit SET detail='{\"k\": 9}' WHERE action='x'")
    assert not st.verify_audit()[0]
