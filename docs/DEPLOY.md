# Deploying XWIDS

The dashboard is one Flask app (`serve.py`). It needs the **exported bundle** (live model + a capped sample of
the data for replay and retraining), and nothing else: no dataset download and no internet.

## 0. Export the bundle (before every hosted option)

```
python scripts/05_export_deploy.py --dataset synthetic      # or awid3
```
This writes `deploy/`. If it warns that the model is large (> 400 MB), lower `models.rf.n_estimators` and
`max_depth` in `config.yaml`, retrain, and export again.

## A. Laptop + public link (quickest for a live demo)

```
winget install --id Cloudflare.cloudflared      # once
.\run_demo.ps1 -Public
```
A second window shows an `https://…trycloudflare.com` link. It works on the judges' phones and opens straight
into Net Doctor at `/doctor/`. Set real passwords first: `$env:XWIDS_USERS="analyst:<pw>,judge:<pw>"`.

## B. Hugging Face Space (free, Docker)

```
pip install huggingface_hub
huggingface-cli login                                   # paste a WRITE token from huggingface.co/settings/tokens
$env:XWIDS_USERS="analyst:<pw>,judge:<pw>"
$env:XWIDS_SECRET_KEY="<long random text>"
python scripts/07_hf_push.py --dataset synthetic --what space --repo <you>/xwids-demo
```
The script creates the Space (SDK: Docker, port 7860), uploads the code and `deploy/`, and stores your
passwords as **Space secrets**. The build takes 3–8 minutes. Check `https://<you>-xwids-demo.hf.space/healthz`.
Spaces are private by default; add `--public` to make one public.

Also on the Hub (optional):
```
python scripts/07_hf_push.py --dataset awid3 --what dataset --repo <you>/xwids-awid3-features   # processed features + SHA-256 manifest
python scripts/07_hf_push.py --dataset awid3 --what model   --repo <you>/xwids-awid3-model
```
A teammate can then set `datasets.hf.repo_id: <you>/xwids-awid3-features` in `config.yaml` and train with
`--dataset hf`. Each file is checked against its SHA-256 before use.

Free Spaces sleep when idle and lose runtime data (verdicts, retrained models) on restart. Download the audit
and verdict CSVs from the Audit page after a session.

## C. Render

Push the repo, including `deploy/`, to GitHub (use Git LFS for `*.joblib` / `*.pkl`; `.gitattributes` is
included). In Render: **New → Blueprint**, pick the repo, and set `XWIDS_USERS` when asked. The free 512 MB
plan is too small for a 500-tree forest. Use the Starter plan, or retrain with `n_estimators: 150` for a demo.

## D. Docker anywhere

```
docker compose up --build -d          # edit the passwords in docker-compose.yml first
```
Verdicts and the audit log survive restarts in the `xwids-runtime` volume.

## Sending live data to the API

Set `XWIDS_API_KEY`, then POST either feature rows or raw tshark/AWID3-style frames:
```
curl -X POST https://<host>/api/score -H "X-API-Key: <key>" -H "Content-Type: application/json" \
     -d '{"frames": [{"frame.time_epoch": 1.0, "wlan.fc.type_subtype": 12, "wlan.bssid": "aa:bb:cc:dd:ee:ff", "wlan.ta": "aa:bb:cc:dd:ee:ff", "wlan.ra": "ff:ff:ff:ff:ff:ff"}]}'
```
To capture frames live on Linux with a monitor-mode Wi-Fi card, run
`tshark -i wlan0mon -T fields -E header=y -E separator=, -e frame.time_epoch -e wlan.fc.type_subtype -e wlan.bssid -e wlan.ta -e wlan.ra -e wlan.seq -e wlan.fc.retry -e wlan.fc.protected -e radiotap.dbm_antsignal -e frame.len -e wlan.duration -e wlan_radio.channel -e wlan.fixed.reason_code -e eapol.type`,
then post the rows once per second.

## Security checklist before sharing a link

- [ ] `XWIDS_USERS` set, with one login per person (verdicts are recorded under each login)
- [ ] `XWIDS_SECRET_KEY` set (otherwise logins reset on every restart)
- [ ] `XWIDS_API_KEY` set only if a sensor posts data, and kept out of Git
- [ ] Say in the demo that Net Doctor's "Mark fix done" only records the fix; it does not change router settings
