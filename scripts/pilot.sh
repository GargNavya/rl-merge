#!/usr/bin/env bash
# End-to-end dry run of the whole pipeline at small scale.
# Everything goes to checkpoints/pilot/ and results/pilot/, so real results stay clean.
#   STEPS=30 N_TEST=50 bash scripts/pilot.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=${BASE:-$(python -c "from src.utils.config import load_config; print(load_config()['model']['name'])")}
TASKS=${TASKS:-"gsm8k logic"}
STEPS=${STEPS:-30}
N_TEST=${N_TEST:-50}
CK=checkpoints/pilot
RES=results/pilot
EVAL=(--results "$RES" --tasks $TASKS --set eval.n_test=$N_TEST)

rm -rf "$CK" "$RES"
echo "### 1/5 base eval"
python -m src.eval.evaluate --model "$BASE" --label base "${EVAL[@]}"

SPECS=()
for t in $TASKS; do
  echo "### 2/5 train $t ($STEPS steps)"
  python -m src.train.grpo --task "$t" --out "$CK/$t" --set grpo.max_steps=$STEPS grpo.save_steps=100000
  python -m src.eval.verify weights --specialist "$CK/$t" --base "$BASE"
  python -m src.eval.evaluate --model "$CK/$t/merged" --label "spec-$t" "${EVAL[@]}"
  SPECS+=("$CK/$t/merged")
done

echo "### 3/5 verify merge"
python -m src.eval.verify merge --specialists "${SPECS[@]}" --base "$BASE"

echo "### 4/5 merge + eval"
LABELS=()
for m in avg task_arithmetic ties; do
  python -m src.merge.merge --base "$BASE" --method "$m" --specialists "${SPECS[@]}" --out "$CK/merged-$m"
  python -m src.eval.evaluate --model "$CK/merged-$m" --label "merge-$m" "${EVAL[@]}"
  LABELS+=("merge-$m")
done

echo "### 5/5 RRR"
python -m src.eval.rrr --scores "$RES/scores.csv" --merged "${LABELS[@]}" --tasks $TASKS
echo "### pilot done. Numbers are tiny-scale: check that everything RAN, not what it says."