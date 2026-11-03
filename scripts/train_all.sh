#!/usr/bin/env bash
# Train one GRPO specialist per task, then score each specialist on ALL tasks
# (the off-diagonal scores show how one skill's training affects the others).
# Run and seed come from configs/base.yaml unless set here.
#   TASKS="gsm8k logic" SEED=43 RUN=v2 bash scripts/train_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-$(python -m src.utils.config run)}
SEED=${SEED:-$(python -m src.utils.config seed)}
TASKS=${TASKS:-$(python -c "from src.utils.config import TASKS; print(' '.join(TASKS))")}

for t in $TASKS; do
  python -m src.train.grpo --task "$t" --seed "$SEED" --set run="$RUN"
  python -m src.eval.evaluate --model "$(python -m src.utils.config spec "$t" --seed "$SEED" --run "$RUN")/merged" \
      --label "spec-$t" --run "$RUN" --train-seed "$SEED"
done