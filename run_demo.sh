#!/usr/bin/env bash
# XWIDS demo launcher (macOS / Linux):  ./run_demo.sh [dataset] [port]
set -e
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && . .venv/bin/activate
DATASET="${1:-synthetic}"; PORT="${2:-5000}"
if [ ! -f "models/$DATASET/registry.json" ]; then
  echo "No trained model for '$DATASET' yet - preparing + training first..."
  python scripts/01_prepare.py --dataset "$DATASET"
  python scripts/02_train.py --dataset "$DATASET"
fi
export XWIDS_USERS="${XWIDS_USERS:-analyst:xwids-demo,reviewer:xwids-review}"
export XWIDS_SECRET_KEY="${XWIDS_SECRET_KEY:-$(python -c 'import secrets;print(secrets.token_hex(32))')}"
export XWIDS_DATASET="$DATASET" PORT="$PORT" HOST=127.0.0.1 XWIDS_RESET=1
echo "Logins: $XWIDS_USERS"; echo "Open http://127.0.0.1:$PORT  (phone view: /doctor/)"
python serve.py
