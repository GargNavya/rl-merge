#!/usr/bin/env bash
# Build fixed splits, then score the untouched base model on all tasks.
# Base model, run and seed come from configs/base.yaml (override: BASE=... RUN=... SEED=...).
# Scores go to results/<run>/seed<k>/ so RRR for that run + seed can find them.
# Extra args pass through, e.g.:  bash scripts/eval_base.sh --set eval.n_test=50
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=${BASE:-$(python -c "from src.utils.config import load_config; print(load_config()['model']['name'])")}
RUN=${RUN:-$(python -m src.utils.config run)}
SEED=${SEED:-$(python -m src.utils.config seed)}

python -m src.data.load
python -m src.eval.evaluate --model "$BASE" --label base --run "$RUN" --train-seed "$SEED" "$@"