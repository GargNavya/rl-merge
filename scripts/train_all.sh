#!/usr/bin/env bash
# Train one GRPO specialist per task, then score each specialist on ALL tasks
# (the off-diagonal scores show how one skill's training affects the others).
#   TASKS="gsm8k countdown" SEED=1 bash scripts/train_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SEED=${SEED:-42}
TASKS=${TASKS:-"gsm8k countdown logic sudoku code"}

for t in $TASKS; do
  python -m src.train.grpo --task "$t" --seed "$SEED"
  python -m src.eval.evaluate --model "checkpoints/specialists/$t/seed$SEED/merged" --label "spec-$t"
done
