"""Frame table -> one feature row per access point (BSS) per time window.

The paper uses "a compact set of 16-17 MAC-layer and Radiotap-header features" such as frame subtype counts,
retry rate, sequence-number gaps and RSSI variance. We extract the 33 candidates below, then
prep.select_features() keeps the 17 with the highest mean |SHAP| (the paper's "SHAP-guided feature selection").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import frames as F
from .common import NORMAL

CANDIDATES = {
    "frames_per_s": "frames seen in the window, per second",
    "mgmt_ratio": "share of management frames",
    "ctrl_ratio": "share of control frames",
    "data_ratio": "share of data frames",
    "deauth_rate": "deauthentication frames / s",
    "disassoc_rate": "disassociation frames / s",
    "assoc_req_rate": "(re)association requests / s",
    "auth_rate": "authentication frames / s",
    "beacon_rate": "beacons / s",
    "probe_req_rate": "probe requests / s",
    "probe_resp_rate": "probe responses / s",
    "eapol_rate": "EAPOL (WPA handshake) frames / s",
    "retry_rate": "share of frames with the retry flag",
    "protected_ratio": "share of data frames that are encrypted",
    "seq_gap_mean": "mean sequence-number jump per transmitter",
    "seq_gap_max": "largest sequence-number jump",
    "seq_anomaly_rate": "repeated or backwards sequence numbers / frame",
    "rssi_mean": "mean signal strength (dBm)",
    "rssi_std": "signal strength spread (dBm)",
    "rssi_range": "max - min signal strength",
    "ap_rssi_std": "signal spread of frames that claim to come from the AP",
    "len_mean": "mean frame length",
    "len_std": "frame length spread",
    "dur_mean": "mean NAV duration field",
    "iat_mean": "mean gap between frames (s)",
    "iat_std": "spread of gaps between frames",
    "unique_ta": "distinct transmitter MACs",
    "unique_ra": "distinct receiver MACs",
    "new_ta_ratio": "distinct transmitters per management frame",
    "broadcast_ratio": "share of frames sent to broadcast",
    "reason_rate": "frames carrying a deauth/disassoc reason code / s",
    "n_channels": "distinct radio channels used by this BSSID",
    "beacon_iat_std": "jitter between beacons (s)",
}
FEATURES = list(CANDIDATES)
META = ["bssid", "win_start", "n_frames", "attack_frames", "label"]


def _fill_bssid(fr: pd.DataFrame) -> pd.DataFrame:
    b = fr["bssid"].astype("string").fillna("")
    known = set(b[b != ""].unique()) - {"ff:ff:ff:ff:ff:ff"}
    miss = b == ""
    if miss.any():
        ra_ok = miss & fr["ra"].isin(known)
        b = b.mask(ra_ok, fr["ra"])
        miss = b == ""
        ta_ok = miss & fr["ta"].isin(known)
        b = b.mask(ta_ok, fr["ta"])
    fr = fr.assign(bssid=b)
    return fr[(fr["bssid"] != "") & (fr["bssid"] != "ff:ff:ff:ff:ff:ff")]


def extract(frames: pd.DataFrame, window_seconds: float = 1.0, min_frames: int = 5,
            attack_fraction: float = 0.10) -> pd.DataFrame:
    fr = _fill_bssid(frames.copy())
    if fr.empty:
        return pd.DataFrame(columns=FEATURES + META)
    fr["win"] = np.floor(fr["time"] / window_seconds).astype(np.int64)
    fr = fr.sort_values(["bssid", "win", "time"], kind="stable").reset_index(drop=True)
    g_id = fr.groupby(["bssid", "win"], sort=False).ngroup().to_numpy()
    fr["g"] = g_id
    st = fr["subtype"].to_numpy()
    fr["is_mgmt"] = st < 16
    fr["is_ctrl"] = (st >= 16) & (st < 32)
    fr["is_data"] = st >= 32
    fr["is_deauth"] = st == F.DEAUTH
    fr["is_disassoc"] = st == F.DISASSOC
    fr["is_assoc"] = np.isin(st, [F.ASSOC_REQ, F.REASSOC_REQ])
    fr["is_auth"] = st == F.AUTH
    fr["is_beacon"] = st == F.BEACON
    fr["is_preq"] = st == F.PROBE_REQ
    fr["is_presp"] = st == F.PROBE_RESP
    fr["is_bcast"] = fr["ra"] == "ff:ff:ff:ff:ff:ff"
    fr["has_reason"] = fr["reason"] > 0
    fr["prot_data"] = fr["is_data"] & (fr["protected"] > 0)
    fr["from_ap"] = fr["ta"] == fr["bssid"]
    fr["rssi_ap"] = fr["rssi"].where(fr["from_ap"])
    fr["is_attack"] = fr["label"] != NORMAL

    # time gaps inside each window
    t = fr["time"].to_numpy()
    same = np.r_[False, g_id[1:] == g_id[:-1]]
    iat = np.where(same, np.r_[0.0, np.diff(t)], np.nan)
    fr["iat"] = iat

    # sequence-number gaps per transmitter inside each window
    s = fr[(fr["seq"] >= 0) & (fr["ta"] != "")][["g", "ta", "time", "seq"]].sort_values(["g", "ta", "time"],
                                                                                         kind="stable")
    sg, sta, sseq = s["g"].to_numpy(), s["ta"].to_numpy(), s["seq"].to_numpy()
    cont = np.r_[False, (sg[1:] == sg[:-1]) & (sta[1:] == sta[:-1])]
    gap = np.where(cont, (np.r_[0, np.diff(sseq)]) % 4096, np.nan)
    s = s.assign(gap=gap, anomaly=cont & ((gap == 0) | (gap > 2048)))
    seq_stats = s.groupby("g").agg(seq_gap_mean=("gap", "mean"), seq_gap_max=("gap", "max"),
                                   seq_anom=("anomaly", "sum"))

    # beacon jitter
    bf = fr[fr["is_beacon"]][["g", "time"]]
    bg = bf["g"].to_numpy()
    biat = np.where(np.r_[False, bg[1:] == bg[:-1]], np.r_[0.0, np.diff(bf["time"].to_numpy())], np.nan)
    beacon_stats = bf.assign(biat=biat).groupby("g").agg(beacon_iat_std=("biat", "std"))

    grp = fr.groupby("g", sort=True)
    agg = grp.agg(
        bssid=("bssid", "first"), win=("win", "first"), n=("time", "size"),
        mgmt=("is_mgmt", "sum"), ctrl=("is_ctrl", "sum"), data=("is_data", "sum"),
        deauth=("is_deauth", "sum"), disassoc=("is_disassoc", "sum"), assoc=("is_assoc", "sum"),
        auth=("is_auth", "sum"), beacon=("is_beacon", "sum"), preq=("is_preq", "sum"), presp=("is_presp", "sum"),
        eapol=("eapol", "sum"), retry=("retry", "sum"), prot_data=("prot_data", "sum"),
        rssi_mean=("rssi", "mean"), rssi_std=("rssi", "std"), rssi_min=("rssi", "min"), rssi_max=("rssi", "max"),
        ap_rssi_std=("rssi_ap", "std"), len_mean=("length", "mean"), len_std=("length", "std"),
        dur_mean=("duration", "mean"), iat_mean=("iat", "mean"), iat_std=("iat", "std"),
        unique_ra=("ra", "nunique"), bcast=("is_bcast", "sum"), reason=("has_reason", "sum"),
        n_channels=("channel", "nunique"), attack_frames=("is_attack", "sum"),
    )
    ta_nonempty = fr[fr["ta"] != ""].groupby("g")["ta"].nunique()
    agg["unique_ta"] = ta_nonempty.reindex(agg.index).fillna(0)
    agg = agg.join(seq_stats, how="left").join(beacon_stats, how="left")

    w = window_seconds
    n = agg["n"].astype(float)
    out = pd.DataFrame({
        "frames_per_s": n / w,
        "mgmt_ratio": agg["mgmt"] / n, "ctrl_ratio": agg["ctrl"] / n, "data_ratio": agg["data"] / n,
        "deauth_rate": agg["deauth"] / w, "disassoc_rate": agg["disassoc"] / w, "assoc_req_rate": agg["assoc"] / w,
        "auth_rate": agg["auth"] / w, "beacon_rate": agg["beacon"] / w, "probe_req_rate": agg["preq"] / w,
        "probe_resp_rate": agg["presp"] / w, "eapol_rate": agg["eapol"] / w,
        "retry_rate": agg["retry"] / n,
        "protected_ratio": np.where(agg["data"] > 0, agg["prot_data"] / agg["data"].clip(lower=1), 1.0),
        "seq_gap_mean": agg["seq_gap_mean"].fillna(1.0), "seq_gap_max": agg["seq_gap_max"].fillna(1.0),
        "seq_anomaly_rate": agg["seq_anom"].fillna(0) / n,
        "rssi_mean": agg["rssi_mean"], "rssi_std": agg["rssi_std"].fillna(0),
        "rssi_range": (agg["rssi_max"] - agg["rssi_min"]).fillna(0),
        "ap_rssi_std": agg["ap_rssi_std"].fillna(0),
        "len_mean": agg["len_mean"], "len_std": agg["len_std"].fillna(0), "dur_mean": agg["dur_mean"],
        "iat_mean": agg["iat_mean"].fillna(w), "iat_std": agg["iat_std"].fillna(0),
        "unique_ta": agg["unique_ta"], "unique_ra": agg["unique_ra"],
        "new_ta_ratio": agg["unique_ta"] / agg["mgmt"].clip(lower=1),
        "broadcast_ratio": agg["bcast"] / n, "reason_rate": agg["reason"] / w,
        "n_channels": agg["n_channels"], "beacon_iat_std": agg["beacon_iat_std"].fillna(0),
    }, index=agg.index)
    out["rssi_mean"] = out["rssi_mean"].fillna(-100)
    out = out.astype(float)

    # window label: the dominant attack if it makes up >= attack_fraction of the frames, else Normal
    atk = fr[fr["is_attack"]].groupby(["g", "label"]).size().rename("c").reset_index()
    if len(atk):
        top = atk.sort_values("c", ascending=False).drop_duplicates("g").set_index("g")
        lab = pd.Series(NORMAL, index=agg.index, dtype=object)
        frac = top["c"] / agg.loc[top.index, "n"]
        hit = frac[frac >= attack_fraction].index
        lab.loc[hit] = top.loc[hit, "label"]
    else:
        lab = pd.Series(NORMAL, index=agg.index, dtype=object)
    out["bssid"] = agg["bssid"].astype(str)
    out["win_start"] = agg["win"] * w
    out["n_frames"] = agg["n"]
    out["attack_frames"] = agg["attack_frames"]
    out["label"] = lab.astype(str)
    out = out[out["n_frames"] >= min_frames].reset_index(drop=True)
    return out[FEATURES + META]
