"""Score one model on one or more tasks. Same code and same questions for every model.

Example:
  python -m src.eval.evaluate --model Qwen/Qwen2.5-0.5B-Instruct --label base
  python -m src.eval.evaluate --model checkpoints/specialists/gsm8k/seed42/merged --label spec-gsm8k
"""
import argparse
import json
from pathlib import Path

import torch

from src.data.load import load_split
from src.data.prompts import build_messages
from src.rewards import format_ok, get_checker
from src.utils.config import ROOT, TASKS, load_config
from src.utils.model import load_model, load_tokenizer
from src.utils.seed import set_seed
from src.utils.tracking import append_score, git_commit, now


@torch.no_grad()
def generate(model, tok, questions, batch_size, max_new_tokens):
    outs = []
    for i in range(0, len(questions), batch_size):
        texts = [tok.apply_chat_template(build_messages(q), tokenize=False, add_generation_prompt=True)
                 for q in questions[i:i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  {min(i + batch_size, len(questions))}/{len(questions)}", end="\r", flush=True)
    print(" " * 20, end="\r")
    return outs


def evaluate(model_path, label, tasks, seed=0, overrides=(), results="results"):
    res = ROOT / results
    set_seed(seed)
    tok = load_tokenizer(model_path)
    model = load_model(model_path, load_config(overrides=overrides)["model"]["dtype"]).eval()
    accs = {}
    for task in tasks:
        cfg = load_config(task=task, overrides=overrides)
        rows = load_split(task, "test", cfg["eval"]["n_test"])
        comps = generate(model, tok, [r["question"] for r in rows],
                         cfg["eval"]["batch_size"], cfg["eval"]["max_new_tokens"])
        checker = get_checker(task)
        scores = [checker(c, r["answer"], r["meta"]) for c, r in zip(comps, rows)]
        acc = sum(scores) / len(scores)
        fmt = sum(format_ok(c) for c in comps) / len(comps)
        mean_len = sum(len(tok(c)["input_ids"]) for c in comps) / len(comps)

        gen_path = res / "generations" / label / f"{task}.jsonl"
        gen_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gen_path, "w") as f:
            for r, c, s in zip(rows, comps, scores):
                f.write(json.dumps({"id": r["id"], "completion": c, "correct": s,
                                    "answer": r["answer"]}) + "\n")

        append_score({"time": now(), "label": label, "model": str(model_path), "task": task,
                      "n": len(rows), "accuracy": round(acc, 4), "format_rate": round(fmt, 4),
                      "mean_len": round(mean_len, 1), "seed": seed, "git_commit": git_commit()},
                     res / "scores.csv")
        accs[task] = acc
        print(f"[eval] {label:<24} {task:<10} acc={acc:.3f} format={fmt:.2f} len={mean_len:.0f}")
    return accs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True, help="base | spec-<task> | merge-<method> ...")
    p.add_argument("--tasks", nargs="+", default=TASKS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", nargs="*", default=[], help="overrides like eval.n_test=50")
    p.add_argument("--results", default="results", help="folder for scores.csv + generations")
    a = p.parse_args()
    evaluate(a.model, a.label, a.tasks, a.seed, a.set, a.results)


if __name__ == "__main__":
    main()