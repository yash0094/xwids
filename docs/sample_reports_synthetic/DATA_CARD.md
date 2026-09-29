# Data card: synthetic

**SYNTHETIC DATA.** Produced by `xwids/simulate.py`. Numbers measured on it test the code only; they must not be reported as results for the paper.

- source: synthetic
- frames: 1,760,935
- windows (rows): 12,000
- window: 1.0 s per access point (BSSID)
- split rows: {'train': 35370, 'val': 1200, 'test': 2400}
- SMOTE rows added to train: 26,970

## Class counts

- train: {'(Re)Association Flood': 7074, 'Normal': 7074, 'KRACK': 7074, 'Deauthentication': 7074, 'Evil Twin': 7074}
- val: {'Normal': 1011, '(Re)Association Flood': 58, 'KRACK': 55, 'Deauthentication': 50, 'Evil Twin': 26}
- test: {'Normal': 2035, 'KRACK': 112, '(Re)Association Flood': 108, 'Evil Twin': 84, 'Deauthentication': 61}

## SHAP-guided feature selection

| rank | feature | mean abs SHAP | kept | meaning |
|---|---|---|---|---|
| 1 | seq_anomaly_rate | 0.0810 | yes | repeated or backwards sequence numbers / frame |
| 2 | data_ratio | 0.0805 | yes | share of data frames |
| 3 | beacon_rate | 0.0788 | yes | beacons / s |
| 4 | seq_gap_mean | 0.0517 | yes | mean sequence-number jump per transmitter |
| 5 | deauth_rate | 0.0445 | yes | deauthentication frames / s |
| 6 | seq_gap_max | 0.0432 | yes | largest sequence-number jump |
| 7 | mgmt_ratio | 0.0425 | yes | share of management frames |
| 8 | assoc_req_rate | 0.0424 | yes | (re)association requests / s |
| 9 | reason_rate | 0.0372 | yes | frames carrying a deauth/disassoc reason code / s |
| 10 | unique_ta | 0.0361 | yes | distinct transmitter MACs |
| 11 | new_ta_ratio | 0.0343 | yes | distinct transmitters per management frame |
| 12 | ctrl_ratio | 0.0322 | yes | share of control frames |
| 13 | retry_rate | 0.0290 | yes | share of frames with the retry flag |
| 14 | len_mean | 0.0248 | yes | mean frame length |
| 15 | protected_ratio | 0.0245 | yes | share of data frames that are encrypted |
| 16 | n_channels | 0.0218 | yes | distinct radio channels used by this BSSID |
| 17 | ap_rssi_std | 0.0178 | yes | signal spread of frames that claim to come from the AP |
| 18 | eapol_rate | 0.0160 |  | EAPOL (WPA handshake) frames / s |
| 19 | broadcast_ratio | 0.0156 |  | share of frames sent to broadcast |
| 20 | auth_rate | 0.0140 |  | authentication frames / s |
| 21 | unique_ra | 0.0073 |  | distinct receiver MACs |
| 22 | rssi_mean | 0.0060 |  | mean signal strength (dBm) |
| 23 | len_std | 0.0051 |  | frame length spread |
| 24 | beacon_iat_std | 0.0041 |  | jitter between beacons (s) |
| 25 | iat_mean | 0.0033 |  | mean gap between frames (s) |
| 26 | frames_per_s | 0.0033 |  | frames seen in the window, per second |
| 27 | rssi_std | 0.0028 |  | signal strength spread (dBm) |
| 28 | dur_mean | 0.0026 |  | mean NAV duration field |
| 29 | iat_std | 0.0022 |  | spread of gaps between frames |
| 30 | rssi_range | 0.0017 |  | max - min signal strength |
| 31 | probe_req_rate | 0.0006 |  | probe requests / s |
| 32 | probe_resp_rate | 0.0005 |  | probe responses / s |
| 33 | disassoc_rate | 0.0000 |  | disassociation frames / s |
