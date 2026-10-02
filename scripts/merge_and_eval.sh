#!/usr/bin/env bash
# Merge all specialists with each method, score the merged models, then compute RRR.
#   METHODS="ties" LAM=0.5 bash scripts/merge_and_eval.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=${BASE:-Qwen/Qwen2.5-0.5B-Instruct}
SEED=${SEED:-42}
# TASKS=${TASKS:-"gsm8k countdown logic sudoku code"}
TASKS=${TASKS:-"gsm8k logic code"}
METHODS=${METHODS:-"avg task_arithmetic ties"}
LAM=${LAM:-1.0}
DENSITY=${DENSITY:-0.2}

SPECS=()
for t in $TASKS; do SPECS+=("checkpoints/specialists/$t/seed$SEED/merged"); done

LABELS=()
for m in $METHODS; do
  label="merge-$m"
  python -m src.merge.merge --base "$BASE" --method "$m" --lam "$LAM" --density "$DENSITY" \
      --specialists "${SPECS[@]}" --out "checkpoints/merged/$m"
  python -m src.eval.evaluate --model "checkpoints/merged/$m" --label "$label"
  LABELS+=("$label")
done

python -m src.eval.rrr --merged "${LABELS[@]}" --tasks $TASKS
