"""Synthetic IEEE 802.11 traffic generator (frame level) for testing XWIDS without AWID3.

It produces the SAME frame table that the AWID3 loader produces, so the whole pipeline (window features ->
selection -> models -> SHAP/LIME/ABEC -> dashboard) is exercised exactly as it will be on real data.

Simulated: access points with beacons, stations with bursty encrypted data + ACKs, probes, and ordinary
roaming (a legitimate deauth + re-auth + 4-way EAPOL handshake now and then, so single deauths or EAPOL
frames are NOT unique to attacks). Attacks, following the paper's Table 2 classes:

  Deauthentication        spoofed deauth frames (TA = the AP's MAC) at 10-300 frames/s, attacker's own
                          sequence counter and RSSI -> sequence gaps + RSSI spread on "AP" frames;
                          victims drop off and re-authenticate
  (Re)Association Flood   assoc / reassoc / auth requests from random spoofed MACs at 30-400 frames/s
  Evil Twin               a rogue AP cloning the BSSID, usually on another channel, beaconing with a
                          different RSSI, sending a few deauths, and luring clients onto open (unprotected) data
  KRACK                   replayed EAPOL message 3 with the SAME sequence number (nonce reuse), retries,
                          and replayed data frames, relayed through a channel-based man in the middle

This is a simulator. Results on it only prove the code works; they are NOT results for the paper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import frames as F
from .common import NORMAL, log

ATTACKS = ["Deauthentication", "(Re)Association Flood", "Evil Twin", "KRACK"]
BROADCAST = 0


class _Sim:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.cols = {c: [] for c in F.COLUMNS + ["seqkey", "seqmode"]}
        self.seq = {}
        self.keys, self.key_start = {}, []
        self.next_mac = 5_000_000

    def new_mac(self, n=1):
        ids = np.arange(self.next_mac, self.next_mac + n)
        self.next_mac += n
        return ids

    def seqs(self, key, n, dup=False):
        """Sequence numbers are assigned at the end in time order per counter key (like a real radio).
        dup=True: a replay that re-uses one sequence number for every frame."""
        if key not in self.keys:
            self.keys[key] = len(self.keys)
            self.key_start.append(int(self.rng.integers(0, 4096)))
        return ("dup" if dup else "auto", self.keys[key])

    def emit(self, n, t0, *, ta, ra, bssid, subtype, rssi, rssi_sd=2.0, length=(60, 1500), retry_p=0.0,
             protected=0, seq=None, duration=44, channel=1, reason=0, eapol=0, label=NORMAL):
        n = int(n)
        if n <= 0:
            return
        r = self.rng
        c = self.cols
        c["time"].append(t0 + np.sort(r.random(n)))
        c["ta"].append(np.broadcast_to(ta, n).copy() if np.ndim(ta) == 0 else np.asarray(ta))
        c["ra"].append(np.broadcast_to(ra, n).copy() if np.ndim(ra) == 0 else np.asarray(ra))
        c["bssid"].append(np.full(n, bssid))
        c["subtype"].append(np.broadcast_to(subtype, n).copy() if np.ndim(subtype) == 0 else np.asarray(subtype))
        c["retry"].append((r.random(n) < retry_p).astype(int))
        c["protected"].append(np.full(n, protected))
        if isinstance(seq, tuple):
            c["seq"].append(np.full(n, -1))
            c["seqkey"].append(np.full(n, seq[1]))
            c["seqmode"].append(np.full(n, 1 if seq[0] == "auto" else 2))
        else:
            c["seq"].append(np.full(n, -1) if seq is None else np.asarray(seq))
            c["seqkey"].append(np.full(n, -1))
            c["seqmode"].append(np.zeros(n, int))
        c["rssi"].append(np.round(rssi + r.normal(0, rssi_sd, n)))
        lo, hi = length if isinstance(length, tuple) else (length, length)
        c["length"].append(r.integers(lo, hi + 1, n))
        c["duration"].append(np.full(n, duration) + (r.integers(0, 3, n) * 2 if duration else 0))
        c["channel"].append(np.broadcast_to(channel, n).copy() if np.ndim(channel) == 0 else np.asarray(channel))
        c["reason"].append(np.full(n, reason))
        c["eapol"].append(np.full(n, eapol))
        c["label"].append(np.full(n, label, dtype=object))


def _schedule(rng, seconds):
    """Alternate quiet periods and attack episodes for one BSS: list of (start, end, attack, intensity)."""
    eps, t = [], int(rng.integers(20, 120))
    while t < seconds - 20:
        dur = int(rng.integers(15, 60))
        eps.append((t, min(t + dur, seconds), ATTACKS[int(rng.integers(0, len(ATTACKS)))], float(rng.random())))
        t += dur + int(rng.integers(50, 220))
    return eps


def generate(n_bss: int = 10, seconds: int = 1200, seed: int = 42) -> pd.DataFrame:
    S = _Sim(seed)
    r = S.rng
    for b in range(n_bss):
        ap = 1000 + b
        ch = int(r.choice([1, 6, 11, 36, 40, 44]))
        rogue_ch = int(r.choice([c for c in [1, 6, 11, 36, 40, 44] if c != ch]))
        ap_rssi = float(r.uniform(-65, -35))
        n_sta = int(r.integers(3, 12))
        stas = 10_000 + b * 100 + np.arange(n_sta)
        sta_rssi = r.uniform(-85, -45, n_sta)
        eps = _schedule(r, seconds)
        activity = r.uniform(0.5, 8, n_sta)
        for sec in range(seconds):
            t0 = float(sec)
            if sec % 60 == 0:  # bursty users: new activity level every minute
                activity = r.uniform(0.5, 8, n_sta) * (r.random(n_sta) < 0.8)
            ep = next((e for e in eps if e[0] <= sec < e[1]), None)
            attack, inten = (ep[2], ep[3]) if ep else (None, 0.0)
            # beacons (~10/s at 102.4 ms)
            nb = int(r.integers(9, 12))
            S.emit(nb, t0, ta=ap, ra=BROADCAST, bssid=ap, subtype=F.BEACON, rssi=ap_rssi,
                   length=(180, 320), seq=S.seqs(ap, nb), duration=0, channel=ch)
            online = np.ones(n_sta, bool)
            if attack == "Deauthentication":
                online = r.random(n_sta) < 0.3  # most victims are knocked off
            for i, sta in enumerate(stas):
                lam = activity[i] * (1.0 if online[i] else 0.1)
                retry_p = float(np.clip(0.02 + (-60 - sta_rssi[i]) * 0.006, 0.01, 0.35))
                up, down = r.poisson(lam), r.poisson(lam * 1.6)
                S.emit(up, t0, ta=sta, ra=ap, bssid=ap, subtype=F.QOS_DATA, rssi=sta_rssi[i], retry_p=retry_p,
                       protected=1, seq=S.seqs(sta, up), channel=ch)
                S.emit(down, t0, ta=ap, ra=sta, bssid=ap, subtype=F.QOS_DATA, rssi=ap_rssi, retry_p=retry_p,
                       protected=1, seq=S.seqs(ap, down), channel=ch)
                S.emit(up, t0, ta=-1, ra=sta, bssid=ap, subtype=F.ACK, rssi=ap_rssi, length=14, channel=ch)
                S.emit(down, t0, ta=-1, ra=ap, bssid=ap, subtype=F.ACK, rssi=sta_rssi[i], length=14, channel=ch)
                # re-authentication by victims of a deauth attack (legitimate frames)
                if not online[i] and r.random() < 0.35:
                    _handshake(S, t0, ap, sta, ch, ap_rssi, sta_rssi[i], with_deauth=False)
            # probes
            npr = r.poisson(0.6)
            if npr:
                S.emit(npr, t0, ta=S.new_mac(npr), ra=BROADCAST, bssid=ap, subtype=F.PROBE_REQ,
                       rssi=float(r.uniform(-90, -50)), length=(60, 140), seq=r.integers(0, 4096, npr), channel=ch)
                S.emit(npr, t0, ta=ap, ra=BROADCAST, bssid=ap, subtype=F.PROBE_RESP, rssi=ap_rssi,
                       length=(180, 320), seq=S.seqs(ap, npr), channel=ch)
            # hard NORMAL look-alikes (so the classifier has genuinely uncertain cases for the analyst):
            # band steering / load balancing: the real AP deauths several clients (its own seq counter + RSSI)
            if r.random() < 0.012:
                steer = S.seq.setdefault(("steer", ap, sec // 8), int(r.integers(3, 25)))
                S.emit(steer, t0, ta=ap, ra=r.choice(stas, steer), bssid=ap, subtype=F.DEAUTH, rssi=ap_rssi,
                       length=26, seq=S.seqs(ap, steer), channel=ch, reason=5)
            # crowd arrival (a lecture starts): many new clients authenticate + associate at once
            if r.random() < 0.01:
                for _ in range(int(r.integers(5, 40))):
                    _handshake(S, t0, ap, int(S.new_mac()[0]), ch, ap_rssi, float(r.uniform(-85, -45)),
                               with_deauth=False)
            # ordinary roaming: legit deauth + re-auth + 4-way handshake
            if r.random() < 0.02:
                i = int(r.integers(0, n_sta))
                _handshake(S, t0, ap, stas[i], ch, ap_rssi, sta_rssi[i], with_deauth=True)
            if attack:
                _attack(S, attack, inten, t0, ap, stas, ch, rogue_ch, ap_rssi, sta_rssi)
    cols = {k: np.concatenate(v) for k, v in S.cols.items()}
    df = pd.DataFrame(cols)
    mac = lambda i: "ff:ff:ff:ff:ff:ff" if i == 0 else ("" if i < 0 else "02:%02x:%02x:%02x:%02x:%02x" % tuple(
        (int(i) >> s) & 0xFF for s in (32, 24, 16, 8, 0)))
    for c in ["ta", "ra", "bssid"]:
        codes, uniq = pd.factorize(df[c])
        df[c] = pd.Series(np.array([mac(u) for u in uniq], dtype=object)[codes]).astype("string")
    df = df.sort_values("time", kind="stable").reset_index(drop=True)
    starts = np.array(S.key_start + [0])
    auto = df["seqmode"] == 1
    k = df.loc[auto, "seqkey"]
    df.loc[auto, "seq"] = (starts[k.to_numpy()] + k.groupby(k).cumcount().to_numpy()) % 4096
    dup = df["seqmode"] == 2
    df.loc[dup, "seq"] = starts[df.loc[dup, "seqkey"].to_numpy()]
    for c in ["subtype", "retry", "protected", "seq", "length", "duration", "channel", "reason", "eapol"]:
        df[c] = df[c].astype(np.int64)
    log.info("simulated %d frames; frame labels: %s", len(df), df["label"].value_counts().to_dict())
    return df[F.COLUMNS]


def _handshake(S, t0, ap, sta, ch, ap_rssi, sta_rssi, with_deauth):
    if with_deauth:
        S.emit(1, t0, ta=ap, ra=sta, bssid=ap, subtype=F.DEAUTH, rssi=ap_rssi, length=26, seq=S.seqs(ap, 1),
               channel=ch, reason=int(S.rng.choice([3, 8])))
    S.emit(1, t0, ta=sta, ra=ap, bssid=ap, subtype=F.AUTH, rssi=sta_rssi, length=30, seq=S.seqs(sta, 1), channel=ch)
    S.emit(1, t0, ta=ap, ra=sta, bssid=ap, subtype=F.AUTH, rssi=ap_rssi, length=30, seq=S.seqs(ap, 1), channel=ch)
    S.emit(1, t0, ta=sta, ra=ap, bssid=ap, subtype=F.ASSOC_REQ, rssi=sta_rssi, length=(80, 160), seq=S.seqs(sta, 1),
           channel=ch)
    S.emit(1, t0, ta=ap, ra=sta, bssid=ap, subtype=F.ASSOC_RESP, rssi=ap_rssi, length=(80, 160), seq=S.seqs(ap, 1),
           channel=ch)
    S.emit(2, t0, ta=ap, ra=sta, bssid=ap, subtype=F.QOS_DATA, rssi=ap_rssi, length=(120, 160), seq=S.seqs(ap, 2),
           channel=ch, eapol=1)
    S.emit(2, t0, ta=sta, ra=ap, bssid=ap, subtype=F.QOS_DATA, rssi=sta_rssi, length=(120, 160),
           seq=S.seqs(sta, 2), channel=ch, eapol=1)


def _attack(S, attack, inten, t0, ap, stas, ch, rogue_ch, ap_rssi, sta_rssi):
    r = S.rng
    # attacker signal strength; sometimes close to the AP's own (attacker standing near the AP = harder)
    atk_rssi = S.seq.setdefault(("rssi", ap, attack),
                                float(ap_rssi + r.normal(0, 3)) if r.random() < 0.35 else float(r.uniform(-85, -30)))
    if attack == "Deauthentication":
        n = r.poisson(3 + inten ** 3 * 300)
        ra = np.where(r.random(n) < 0.5, BROADCAST, r.choice(stas, n))
        S.emit(n, t0, ta=ap, ra=ra, bssid=ap, subtype=F.DEAUTH, rssi=atk_rssi, rssi_sd=3, length=26,
               seq=S.seqs(("deauth", ap), n), channel=ch, reason=7, label=attack)
    elif attack == "(Re)Association Flood":
        n = r.poisson(8 + inten ** 3 * 390)
        sub = r.choice([F.ASSOC_REQ, F.REASSOC_REQ, F.AUTH], n, p=[0.45, 0.35, 0.20])
        S.emit(n, t0, ta=S.new_mac(n), ra=ap, bssid=ap, subtype=sub, rssi=atk_rssi, rssi_sd=4, length=(60, 200),
               seq=r.integers(0, 4096, n), channel=ch, label=attack)
        k = r.binomial(n, 0.4)  # AP answers part of them (legitimate frames)
        S.emit(k, t0, ta=ap, ra=S.new_mac(k), bssid=ap, subtype=F.ASSOC_RESP, rssi=ap_rssi, length=(80, 160),
               seq=S.seqs(ap, k), channel=ch)
    elif attack == "Evil Twin":
        rch = rogue_ch if inten > 0.5 else ch
        n = r.integers(9, 11)
        S.emit(n, t0, ta=ap, ra=BROADCAST, bssid=ap, subtype=F.BEACON, rssi=atk_rssi, length=(180, 320),
               seq=S.seqs(("twin", ap), n), duration=0, channel=rch, label=attack)
        nd = r.poisson(1 + 3 * inten)
        S.emit(nd, t0, ta=ap, ra=r.choice(stas, max(nd, 1))[:nd], bssid=ap, subtype=F.DEAUTH, rssi=atk_rssi,
               length=26, seq=S.seqs(("twin", ap), nd), channel=ch, reason=7, label=attack)
        lured = stas[r.random(len(stas)) < 0.3 + 0.4 * inten]
        for sta in lured:  # victims talking to the open rogue AP
            m = r.poisson(4)
            S.emit(m, t0, ta=sta, ra=ap, bssid=ap, subtype=F.QOS_DATA, rssi=atk_rssi, length=(60, 1500),
                   protected=0, seq=S.seqs(("twin", sta), m), channel=rch, label=attack)
            S.emit(m, t0, ta=ap, ra=sta, bssid=ap, subtype=F.QOS_DATA, rssi=atk_rssi, length=(60, 1500),
                   protected=0, seq=S.seqs(("twin", ap), m), channel=rch, label=attack)
    elif attack == "KRACK":
        victim = stas[int(r.integers(0, len(stas)))]
        n = r.poisson(1 + 12 * inten ** 2)
        S.emit(n, t0, ta=ap, ra=victim, bssid=ap, subtype=F.QOS_DATA, rssi=atk_rssi, length=(120, 160),
               retry_p=0.8, seq=S.seqs(("krack-msg3", ap, victim), n, dup=True), channel=rogue_ch, eapol=1,
               label=attack)
        m = r.poisson(2 + 20 * inten ** 2)  # replayed data frames -> nonce / sequence reuse
        S.emit(m, t0, ta=victim, ra=ap, bssid=ap, subtype=F.QOS_DATA, rssi=atk_rssi, length=(60, 1500),
               retry_p=0.5, protected=1, seq=S.seqs(("krack-data", victim), m, dup=True), channel=rogue_ch,
               label=attack)
