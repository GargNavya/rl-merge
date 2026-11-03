"""Print random evaluated examples to proofread the checker by eye.
Reads results/<run>/seed<k>/generations/ (run and seed from configs/base.yaml unless given).

Examples:
  python -m src.eval.show_samples --label base --task gsm8k
  python -m src.eval.show_samples --label base --task logic --n 10 --only wrong
  python -m src.eval.show_samples --label spec-code --task code --only right --seed 3
"""
import argparse
import importlib
import json
import random

from src.utils.config import ROOT, results_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--only", choices=["right", "wrong"], default=None)
    p.add_argument("--seed", type=int, default=0, help="which random examples to show")
    p.add_argument("--run", default=None, help="default: run in configs/base.yaml")
    p.add_argument("--train-seed", type=int, default=None, help="default: seed in base.yaml")
    p.add_argument("--results", default=None, help="explicit results folder (overrides run/train-seed)")
    p.add_argument("--full", action="store_true", help="don't shorten long replies")
    a = p.parse_args()
    res = ROOT / a.results if a.results else results_dir(a.train_seed, a.run)

    gens = [json.loads(l) for l in open(res / "generations" / a.label / f"{a.task}.jsonl")]
    rows = {r["id"]: r for r in map(json.loads, open(ROOT / "data" / a.task / "test.jsonl"))}
    extract = importlib.import_module(f"src.rewards.{a.task}").extract

    if a.only:
        gens = [g for g in gens if (g["correct"] == 1.0) == (a.only == "right")]
    picks = random.Random(a.seed).sample(gens, min(a.n, len(gens)))
    total = sum(g["correct"] for g in gens)
    print(f"{a.label} / {a.task}: showing {len(picks)} of {len(gens)}"
          + (f" ({a.only})" if a.only else f" (acc {total / max(len(gens), 1):.3f})"))

    for g in picks:
        row = rows[g["id"]]
        reply = g["completion"]
        if not a.full and len(reply) > 1500:
            reply = reply[:500] + "\n   [... cut, use --full ...]\n" + reply[-900:]
        print("\n" + "=" * 90)
        print(f"ID: {g['id']}    VERDICT: {'RIGHT' if g['correct'] == 1.0 else 'WRONG'}")
        print("-" * 34 + " QUESTION " + "-" * 46)
        print(row["question"].strip())
        print("-" * 34 + " MODEL REPLY " + "-" * 43)
        print(reply.strip())
        print("-" * 90)
        print(f"EXTRACTED: {extract(g['completion'], row['meta'])!r}")
        gold = row["answer"] or "(unit tests) " + " | ".join(json.loads(row["meta"]).get("tests", []))
        print(f"GOLD:      {gold!r}")


if __name__ == "__main__":
    main()