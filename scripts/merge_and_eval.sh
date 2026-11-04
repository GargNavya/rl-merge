#!/usr/bin/env bash
# Merge all specialists of one run + seed with each method, score the merged models, then compute RRR.
# Merges go to checkpoints/merged/<run>/seed<k>/<merge name>, labels are merge-<merge name>.
#   METHODS="ties" LAM=0.5 DENSITY=0.3 bash scripts/merge_and_eval.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=${BASE:-$(python -c "from src.utils.config import load_config; print(load_config()['model']['name'])")}
RUN=${RUN:-$(python -m src.utils.config run)}
SEED=${SEED:-$(python -m src.utils.config seed)}
TASKS=${TASKS:-$(python -c "from src.utils.config import MERGE_TASKS; print(' '.join(MERGE_TASKS))")}
EVAL_TASKS=${EVAL_TASKS:-$(python -c "from src.utils.config import TASKS; print(' '.join(TASKS))")}
METHODS=${METHODS:-"avg task_arithmetic ties"}
LAM=${LAM:-1.0}
DENSITY=${DENSITY:-0.2}

LABELS=()
for m in $METHODS; do
  name=$(python -m src.utils.config mergename "$m" --lam "$LAM" --density "$DENSITY")
  python -m src.merge.merge --base "$BASE" --method "$m" --lam "$LAM" --density "$DENSITY" \
      --run "$RUN" --seed "$SEED" --tasks $TASKS
  python -m src.eval.evaluate --model "$(python -m src.utils.config merged "$name" --seed "$SEED" --run "$RUN")" \
      --label "merge-$name" --run "$RUN" --train-seed "$SEED" --tasks $EVAL_TASKS
  LABELS+=("merge-$name")
done

python -m src.eval.rrr --merged "${LABELS[@]}" --tasks $TASKS --run "$RUN" --seed "$SEED"