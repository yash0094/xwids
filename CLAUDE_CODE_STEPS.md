# Step-by-step prompts for Claude Code (VS Code)

Open the `xwids` folder in VS Code, open Claude Code, and paste **one step at a time**. Wait for it to finish.

---

### Step 1: Set up and self-test
> Read CLAUDE.md. Create a Python 3.11 virtual environment in `.venv`, install `requirements.txt`, and run
> `python -m pytest -q`. **Check:** all tests pass. Then tell me in simple words which backends are active
> (`python -c "from xwids.common import backends; print(backends())"`). All five should be the real libraries.

### Step 2: Demo on synthetic data
> Run `python scripts/01_prepare.py --dataset synthetic`, `python scripts/02_train.py --dataset synthetic` and
> `python scripts/03_evaluate.py --dataset synthetic`. **Check:** `reports/synthetic/RESULTS.md` exists and the
> backends line says `TreeSHAP (shap)` and `lime`. Then start `.\run_demo.ps1` and tell me the login.

### Step 3: Load AWID3. ASK me first if `data/raw/awid3` is empty
> Run `python scripts/06_inspect_csv.py` on one CSV from each attack folder in `data/raw/awid3`.
> **Check:** time, subtype, bssid, ta, ra, seq, rssi and label are all found. If any are missing, add the
> right column names to `ALIASES` in `xwids/frames.py` and re-check. Show me the label counts.

### Step 4: Prepare AWID3
> Run `python scripts/01_prepare.py --dataset awid3`. If memory runs out, halve `max_rows_per_file` in
> config.yaml and retry. **Check:** `data/processed/awid3/DATA_CARD.md` lists all 5 classes, each with at
> least 100 windows. If a class is missing, find out why (label spelling? window threshold?) and fix it.

### Step 5: Train and evaluate on AWID3
> Run `python scripts/02_train.py --dataset awid3`, then
> `python scripts/03_evaluate.py --dataset awid3 --n-explain 200 --cv`. **Check:** RESULTS.md has Tables 1-4
> and the backends are the real libraries. Explain each table to me in simple words, and say honestly where
> they differ from the paper's claimed numbers.

### Step 6: Feedback loop (Table 5)
> Run `python scripts/04_active_learning.py --dataset awid3 --jaccard`. **Check:** ACTIVE_LEARNING.md has a
> baseline and 3 iterations. Explain whether F1 went up and FPR and escalation went down, and by how much.

### Step 7: Update the paper text
> Using ONLY `reports/awid3/RESULTS.md` and `ACTIVE_LEARNING.md`, write `docs/paper_tables_measured.md` with
> Tables 1-5 in the paper's format, plus a list of every sentence in the paper whose number must change.
> Do not invent numbers. If the paper reports a dataset we did not run (CICIDS2017, NSL-KDD, N-BaIoT), mark
> that row "not run".

### Step 8: Deploy
> Run `python scripts/05_export_deploy.py --dataset awid3`, then help me log in to Hugging Face and run
> `python scripts/07_hf_push.py --dataset awid3 --what space --repo <my-name>/xwids-demo` with strong passwords
> in `XWIDS_USERS`. **Check:** `/healthz` on the Space answers `ok`. Open `/doctor/` on my phone.

### Step 9: Demo rehearsal
> Write `docs/DEMO_SCRIPT.md`: a 5-minute live demo for judges. Log in → Replay → open the lowest-J alert →
> explain the ABEC tiers → give 5 verdicts → show the Feedback loop page → open Net Doctor on the phone →
> Run diagnostic → Recommendations. Include what to say if the Wi-Fi fails (use `run_demo.ps1` locally).
