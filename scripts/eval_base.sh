#!/usr/bin/env bash
# Build fixed splits, then score the untouched base model on all tasks.
# Extra args pass through, e.g.:  bash scripts/eval_base.sh --set eval.n_test=50
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=${BASE:-Qwen/Qwen2.5-0.5B-Instruct}

python -m src.data.load
python -m src.eval.evaluate --model "$BASE" --label base "$@"
