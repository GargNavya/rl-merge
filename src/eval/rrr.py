"""Reasoning Retention Ratio (RRR) + cross-task gain matrix, read from results/<run>/seed<k>/scores.csv.

Label convention: base = "base", specialists = "spec-<task>", merged = anything else you pass.

RRR_j = (merged_j - base_j) / (spec_j - base_j)
  1 = merged kept the specialist's full gain, 0 = gain gone, < 0 = worse than base.

Example:
  python -m src.eval.rrr --merged merge-avg merge-task_arithmetic_lam1 merge-ties_d0.2_lam1
"""
import argparse
import csv
from pathlib import Path

from src.utils.config import TASKS, results_dir

MIN_GAIN = 0.02  # below this the specialist barely improved and RRR is noise


def latest_scores(path):
    table = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            table[(r["label"], r["task"])] = float(r["accuracy"])  # later rows overwrite earlier
    return table


def compute_rrr(s: dict, label: str, tasks: list[str]):
    """s maps (label, task) -> accuracy. Returns (per-task RRR or None, macro, worst)."""
    vals = []
    for t in tasks:
        gain = s[(f"spec-{t}", t)] - s[("base", t)]
        if (label, t) not in s or gain < MIN_GAIN:
            vals.append(None)
            continue
        vals.append((s[(label, t)] - s[("base", t)]) / gain)
    ok = [v for v in vals if v is not None]
    macro = sum(ok) / len(ok) if ok else float("nan")
    worst = min(ok) if ok else float("nan")
    return vals, macro, worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", default=None, help="default: results/<run>/seed<k>/scores.csv")
    p.add_argument("--run", default=None, help="default: run in configs/base.yaml")
    p.add_argument("--seed", type=int, default=None, help="training seed; default: seed in base.yaml")
    p.add_argument("--merged", nargs="+", required=True)
    p.add_argument("--tasks", nargs="+", default=TASKS)
    a = p.parse_args()
    scores = a.scores or str(results_dir(a.seed, a.run) / "scores.csv")

    s = latest_scores(scores)
    tasks = [t for t in a.tasks if ("base", t) in s and (f"spec-{t}", t) in s]
    missing = sorted(set(a.tasks) - set(tasks))
    if missing:
        print(f"[rrr] skipping tasks without base+specialist scores: {missing}")

    print("\nGain over base (rows = model, cols = task). Diagonal = specialist on its own task.")
    rows = [f"spec-{t}" for t in tasks] + a.merged
    print(f"{'':<30}" + "".join(f"{t:>11}" for t in tasks))
    for lab in rows:
        cells = [f"{s[(lab, t)] - s[('base', t)]:+11.3f}" if (lab, t) in s else f"{'-':>11}"
                 for t in tasks]
        print(f"{lab:<30}" + "".join(cells))

    out_rows = []
    print("\nRRR per task")
    print(f"{'':<30}" + "".join(f"{t:>11}" for t in tasks) + f"{'macro':>9}{'worst':>9}")
    for lab in a.merged:
        vals, macro, worst = compute_rrr(s, lab, tasks)
        print(f"{lab:<30}" + "".join(f"{v:11.2f}" if v is not None else f"{'n/a':>11}" for v in vals)
              + f"{macro:9.2f}{worst:9.2f}")
        out_rows.append({"label": lab, **{t: v for t, v in zip(tasks, vals)},
                         "macro_rrr": macro, "worst_rrr": worst})

    low = [t for t in tasks if s[(f"spec-{t}", t)] - s[("base", t)] < MIN_GAIN]
    if low:
        print(f"\n[rrr] n/a = specialist gained < {MIN_GAIN} on: {low}. Train longer or adjust difficulty.")

    path = Path(scores).parent / "rrr.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", *tasks, "macro_rrr", "worst_rrr"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n[rrr] saved {path}")


if __name__ == "__main__":
    main()