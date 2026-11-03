#!/usr/bin/env bash
# One-time move of the existing 1.5B seed-42 outputs into the run layout, as run "v1".
#   DRY=1 bash scripts/migrate_to_v1.sh     # print what would happen, change nothing
#   bash scripts/migrate_to_v1.sh           # do it
# Moves stay on the same disk, so they are instant and need no extra space.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=v1
SEED=seed42
TASKS="gsm8k logic code"
DRY=${DRY:-0}
run() { echo "+ $*"; [ "$DRY" = 1 ] || "$@"; }

SPEC=$(readlink -f checkpoints/specialists)
MERG=$(readlink -f checkpoints/merged)
echo "specialists folder: $SPEC"
echo "merges folder:      $MERG"

# stop before touching anything if the layout is not what we expect
for t in $TASKS; do [ -d "$SPEC/$t/$SEED" ] || { echo "STOP: $SPEC/$t/$SEED not found"; exit 1; }; done
for m in avg task_arithmetic ties; do [ -d "$MERG/$m" ] || { echo "STOP: $MERG/$m not found"; exit 1; }; done
[ -f results/scores.csv ] || { echo "STOP: results/scores.csv not found"; exit 1; }
for d in "$SPEC/$RUN" "$MERG/$RUN" "results/$RUN"; do
  [ -e "$d" ] && { echo "STOP: $d already exists"; exit 1; }
done

echo "### 1 specialists: <task>/$SEED -> $RUN/<task>/$SEED"
run mkdir -p "$SPEC/$RUN"
for t in $TASKS; do run mv "$SPEC/$t" "$SPEC/$RUN/$t"; done

echo "### 2 merges -> $RUN/$SEED/<merge name>"
run mkdir -p "$MERG/$RUN/$SEED"
run mv "$MERG/avg"             "$MERG/$RUN/$SEED/avg"
run mv "$MERG/task_arithmetic" "$MERG/$RUN/$SEED/task_arithmetic_lam1"
run mv "$MERG/ties"            "$MERG/$RUN/$SEED/ties_d0.2_lam1"

echo "### 3 results -> results/$RUN/$SEED (results/pilot stays)"
run mkdir -p "results/$RUN/$SEED"
for f in results/*; do
  case "$(basename "$f")" in pilot|"$RUN") continue ;; esac
  run mv "$f" "results/$RUN/$SEED/"
done

echo "### 4 merge labels: generations folders, scores.csv, rrr.csv"
G=results/$RUN/$SEED/generations
run mv "$G/merge-task_arithmetic" "$G/merge-task_arithmetic_lam1"
run mv "$G/merge-ties"            "$G/merge-ties_d0.2_lam1"
if [ "$DRY" = 1 ]; then
  echo "+ rename labels and model paths in results/$RUN/$SEED/{scores,rrr}.csv"
else
python - "$RUN" "$SEED" <<'EOF'
import csv, sys
run, seed = sys.argv[1], sys.argv[2]
d = f"results/{run}/{seed}"
LABELS = {"merge-task_arithmetic": "merge-task_arithmetic_lam1", "merge-ties": "merge-ties_d0.2_lam1"}
MODELS = {"checkpoints/merged/avg": f"checkpoints/merged/{run}/{seed}/avg",
          "checkpoints/merged/task_arithmetic": f"checkpoints/merged/{run}/{seed}/task_arithmetic_lam1",
          "checkpoints/merged/ties": f"checkpoints/merged/{run}/{seed}/ties_d0.2_lam1"}
for t in ["gsm8k", "logic", "code"]:
    MODELS[f"checkpoints/specialists/{t}/{seed}/merged"] = f"checkpoints/specialists/{run}/{t}/{seed}/merged"
for name in ["scores.csv", "rrr.csv"]:
    path = f"{d}/{name}"
    try:
        rows = list(csv.DictReader(open(path, newline="")))
    except FileNotFoundError:
        print(f"  (no {path})"); continue
    fields = list(rows[0].keys())
    n = 0
    for r in rows:
        if r["label"] in LABELS:
            r["label"] = LABELS[r["label"]]; n += 1
        if "model" in r and r["model"] in MODELS:
            r["model"] = MODELS[r["model"]]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"  {path}: {n} labels renamed")
EOF
fi

echo "### done. Check with:"
echo "  python -m src.eval.verify rescore"
echo "  python -m src.eval.rrr --merged merge-avg merge-task_arithmetic_lam1 merge-ties_d0.2_lam1"
