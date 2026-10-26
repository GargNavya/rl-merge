#!/usr/bin/env bash
# Build fixed splits, then score the untouched base model on all tasks.
# Base model comes from configs/base.yaml (override: BASE=... bash scripts/eval_base.sh).
# Extra args pass through, e.g.:  bash scripts/eval_base.sh --set eval.n_test=50
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=${BASE:-$(python -c "from src.utils.config import load_config; print(load_config()['model']['name'])")}

python -m src.data.load
python -m src.eval.evaluate --model "$BASE" --label base "$@"