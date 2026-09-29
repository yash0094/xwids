"""802.11 frame table: one canonical schema, plus a tolerant loader for AWID3 CSV exports.

Every data source (AWID3 CSVs, the synthetic simulator, a live capture exported with tshark) is converted
to this frame table first. The feature extractor (features.py) only ever sees these columns:

    time       float  seconds
    bssid      str    access point MAC the frame belongs to (may be empty for control frames)
    ta, ra     str    transmitter / receiver MAC
    subtype    int    wlan.fc.type_subtype: 0-15 management, 16-31 control, 32-47 data
    retry      int    0/1 retry flag
    protected  int    0/1 protected (encrypted) flag
    seq        int    sequence number 0-4095 (-1 if absent, e.g. ACK)
    rssi       float  signal strength, dBm
    length     int    frame length, bytes
    duration   int    NAV duration field, microseconds
    channel    int    radio channel
    reason     int    reason code (deauth / disassoc), 0 if none
    eapol      int    0/1 frame carries EAPOL (WPA handshake)
    label      str    ground-truth class (training data only)
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from .common import NORMAL, log

COLUMNS = ["time", "bssid", "ta", "ra", "subtype", "retry", "protected", "seq", "rssi", "length",
           "duration", "channel", "reason", "eapol", "label"]

# 802.11 type_subtype codes used by the features
ASSOC_REQ, ASSOC_RESP, REASSOC_REQ, REASSOC_RESP = 0, 1, 2, 3
PROBE_REQ, PROBE_RESP, BEACON, DISASSOC, AUTH, DEAUTH = 4, 5, 8, 10, 11, 12
RTS, CTS, ACK = 27, 28, 29
DATA, QOS_DATA = 32, 40

# tshark field names seen in AWID3 and other Wi-Fi datasets -> canonical column
ALIASES = {
    "time": ["frame.time_epoch", "frame.time_relative", "frame.time_delta_displayed", "frame.time"],
    "bssid": ["wlan.bssid"],
    "ta": ["wlan.ta", "wlan.sa", "wlan.addr2"],
    "ra": ["wlan.ra", "wlan.da", "wlan.addr1"],
    "subtype": ["wlan.fc.type_subtype"],
    "retry": ["wlan.fc.retry"],
    "protected": ["wlan.fc.protected"],
    "seq": ["wlan.seq"],
    "rssi": ["radiotap.dbm_antsignal", "wlan_radio.signal_dbm", "radiotap.dbm_antsignal.0"],
    "length": ["frame.len", "frame.cap_len"],
    "duration": ["wlan.duration", "wlan_radio.duration"],
    "channel": ["wlan_radio.channel", "radiotap.channel.freq", "radiotap.channel"],
    "reason": ["wlan.fixed.reason_code"],
    "eapol": ["eapol.type", "eapol.keydes.type", "eapol.version", "wlan_rsna_eapol.keydes.msgnr"],
    "label": ["Label", "label", "Class", "class", "attack_cat"],
}

# AWID3 label spellings -> the class names used in the paper
LABEL_MAP = {
    "normal": NORMAL,
    "deauth": "Deauthentication", "deauthentication": "Deauthentication",
    "disas": "Disassociation", "disassociation": "Disassociation",
    "(re)assoc": "(Re)Association Flood", "reassoc": "(Re)Association Flood", "re_assoc": "(Re)Association Flood",
    "(re)association flood": "(Re)Association Flood",
    "evil_twin": "Evil Twin", "evil twin": "Evil Twin", "eviltwin": "Evil Twin",
    "krack": "KRACK", "kr00k": "Kr00k", "rogue_ap": "Rogue AP",
}


def _pick(df: pd.DataFrame, names: list[str]):
    lower = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return df[lower[n.lower()]]
    return None


def _num(s: pd.Series, default=0) -> pd.Series:
    """Parse numbers that may be ints, floats, hex strings ('0x000c'), booleans or multi-values ('-45,-47')."""
    if s is None:
        return None
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    t = s.astype(str).str.split(",").str[0].str.strip().str.lower()
    t = t.replace({"true": "1", "false": "0", "nan": "", "none": "", "?": "", "-": ""})
    hexmask = t.str.startswith("0x")
    out = pd.to_numeric(t.where(~hexmask), errors="coerce")
    if hexmask.any():
        out[hexmask] = t[hexmask].apply(lambda v: int(v, 16) if v else np.nan)
    return out.astype(float)


def _time(s: pd.Series) -> pd.Series:
    n = _num(s)
    if n is not None and n.notna().mean() > 0.9:
        return n
    return pd.to_datetime(s, errors="coerce").astype("int64") / 1e9


def normalise_label(v) -> str:
    k = str(v).strip()
    return LABEL_MAP.get(k.lower(), k)


def from_tshark_columns(raw: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """Map any AWID3 / tshark-style CSV onto the canonical frame table."""
    out = pd.DataFrame(index=raw.index)
    missing = []
    for col, names in ALIASES.items():
        s = _pick(raw, names)
        if s is None:
            missing.append(col)
        out[col] = s
    # subtype can also be split into wlan.fc.type + wlan.fc.subtype
    if out["subtype"].isna().all():
        t, st = _pick(raw, ["wlan.fc.type"]), _pick(raw, ["wlan.fc.subtype"])
        if t is not None and st is not None:
            out["subtype"] = _num(t) * 16 + _num(st)
            missing.remove("subtype")
    need = {"time", "subtype"}
    if need & set(missing):
        raise ValueError(f"{source}: cannot find required columns {sorted(need & set(missing))}. "
                         f"Run `python scripts/inspect_csv.py <file>` and add the right names to ALIASES in xwids/frames.py.")
    if missing:
        log.info("%s: columns not found (filled with defaults): %s", source or "csv", ", ".join(missing))

    out["time"] = _time(out["time"])
    for c in ["subtype", "retry", "protected", "seq", "length", "duration", "channel", "reason"]:
        v = _num(out[c]) if out[c].notna().any() else pd.Series(np.nan, index=out.index)
        out[c] = v
    out["seq"] = out["seq"].fillna(-1)
    out[["retry", "protected", "length", "duration", "reason"]] = \
        out[["retry", "protected", "length", "duration", "reason"]].fillna(0)
    ch = out["channel"]
    out["channel"] = np.where(ch > 1000, ((ch - 2407) / 5).round(), ch)  # frequency MHz -> channel number
    out["channel"] = out["channel"].fillna(0)
    out["rssi"] = _num(out["rssi"]) if out["rssi"].notna().any() else np.nan
    out["eapol"] = out["eapol"].notna().astype(int) if out["eapol"].notna().any() else 0
    for c in ["bssid", "ta", "ra"]:
        out[c] = out[c].astype("string").str.lower().fillna("")
    out["label"] = out["label"].map(normalise_label) if out["label"].notna().any() else NORMAL
    out = out.dropna(subset=["time", "subtype"])
    int_cols = ["subtype", "retry", "protected", "seq", "length", "duration", "channel", "reason", "eapol"]
    out[int_cols] = out[int_cols].astype(np.int64)
    return out[COLUMNS]


def load_csv_frames(pattern: str, max_rows_per_file: int | None = None) -> pd.DataFrame:
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise SystemExit(
            f"No CSV files match '{pattern}'.\n"
            "AWID3 is free for research but needs registration: https://icsdweb.aegean.gr/awid/awid3\n"
            "Download the CSV version, unzip it into data/raw/awid3/, then run this again.\n"
            "No dataset yet? Use --dataset synthetic to test everything end to end.")
    parts = []
    t_offset = 0.0
    for f in files:
        raw = pd.read_csv(f, nrows=max_rows_per_file, low_memory=False)
        fr = from_tshark_columns(raw, Path(f).name)
        # each AWID3 file is a separate capture: shift time so windows never mix two files
        fr["time"] = fr["time"] - fr["time"].min() + t_offset
        t_offset = fr["time"].max() + 10.0
        parts.append(fr)
        log.info("loaded %s: %d frames, labels %s", Path(f).name, len(fr), fr["label"].value_counts().to_dict())
    return pd.concat(parts, ignore_index=True)
