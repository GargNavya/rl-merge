"""Re-grade saved replies of one run with the current checkers, into a NEW results folder.
No model, no GPU: only the checkers run (code tests run on CPU).

The source folder is never changed. Output gets the same labels, so RRR works on it directly.

Example:
  python -m src.eval.regrade --src-run v1                       # -> results/v1_regrade/seed42/
  python -m src.eval.rrr --run v1_regrade --merged merge-avg merge-task_arithmetic_lam1 merge-ties_d0.2_lam1
"""
import argparse
import csv
import json
import shutil
import sys

from src.data.load import DATA_DIR
from src.rewards import get_checker
from src.utils.config import results_dir, train_seed
from src.utils.tracking import append_score, git_commit, now


def latest_rows(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[(r["label"], r["task"])] = r   # later rows overwrite earlier, like rrr.py
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-run", required=True)
    p.add_argument("--seed", type=int, default=None, help="training seed; default: seed in base.yaml")
    p.add_argument("--out-run", default=None, help="default: <src-run>_regrade")
    p.add_argument("--tasks", nargs="+", default=None, help="only these tasks (default: all saved)")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    seed = train_seed(a.seed)
    src = results_dir(seed, a.src_run)
    out = results_dir(seed, a.out_run or f"{a.src_run}_regrade")
    if not (src / "scores.csv").exists():
        sys.exit(f"[regrade] no {src}/scores.csv")
    if out.exists():
        if not a.overwrite:
            sys.exit(f"[regrade] {out} exists. Pass --overwrite to replace it.")
        shutil.rmtree(out)

    old = latest_rows(src / "scores.csv")
    files = sorted((src / "generations").glob("*/*.jsonl"))
    if a.tasks:
        files = [f for f in files if f.stem in a.tasks]
    commit = git_commit()
    print(f"[regrade] {src} -> {out}\n")
    print(f"{'label':<30}{'task':<8}{'old':>8}{'new':>8}{'diff':>8}{'flips +/-':>12}")

    for f in files:
        label, task = f.parent.name, f.stem
        rows = {r["id"]: r for r in map(json.loads, open(DATA_DIR / task / "test.jsonl"))}
        gens = [json.loads(l) for l in open(f)]
        check = get_checker(task)
        new = []
        for g in gens:
            r = rows[g["id"]]
            new.append({**g, "correct": check(g["completion"], r["answer"], r["meta"])})
        up = sum(n["correct"] > g["correct"] for n, g in zip(new, gens))
        down = sum(n["correct"] < g["correct"] for n, g in zip(new, gens))
        acc_new = sum(n["correct"] for n in new) / len(new)

        gen_out = out / "generations" / label / f"{task}.jsonl"
        gen_out.parent.mkdir(parents=True, exist_ok=True)
        with open(gen_out, "w") as h:
            for n in new:
                h.write(json.dumps(n) + "\n")

        o = old.get((label, task), {})
        append_score({**o, "time": now(), "label": label, "task": task, "n": len(new),
                      "accuracy": round(acc_new, 4), "git_commit": commit}, out / "scores.csv")
        acc_old = float(o["accuracy"]) if o else float("nan")
        print(f"{label:<30}{task:<8}{acc_old:8.3f}{acc_new:8.3f}{acc_new - acc_old:+8.3f}{f'+{up}/-{down}':>12}")

    (out / "SOURCE.txt").write_text(f"re-graded from {src} at {now()}, commit {commit}\n")
    print(f"\n[regrade] saved {out}/scores.csv. format_rate and mean_len are copied: replies did not change.")


if __name__ == "__main__":
    main()