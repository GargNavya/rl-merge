"""Pipeline checks. Each prints PASS / FAIL / INFO lines and exits non-zero on any FAIL.

  python -m src.eval.verify data                                   # splits + checkers on gold/junk
  python -m src.eval.verify eval    --task gsm8k                   # base model from configs/base.yaml
  python -m src.eval.verify weights --specialist checkpoints/specialists/gsm8k/seed42
  python -m src.eval.verify merge   --specialists checkpoints/specialists/*/seed42/merged
  python -m src.eval.verify rescore --results results/warmup_seed42   # CPU: re-grade saved replies

See VERIFY.md for the order and what to do when something fails.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

from src.utils.config import TASKS, load_config

FAILS = []
DEFAULT_BASE = load_config()["model"]["name"]


def report(ok, msg, info=False):
    tag = "INFO" if info else ("PASS" if ok else "FAIL")
    if not ok and not info:
        FAILS.append(msg)
    print(f"  [{tag}] {msg}")


# ------------------------------------------------------------------ data
def verify_data(tasks):
    from src.data.load import load_split
    from src.rewards import get_checker

    for task in tasks:
        print(f"\n== data: {task}")
        cfg = load_config(task=task)["task"]
        train, test = load_split(task, "train"), load_split(task, "test")
        check = get_checker(task)

        report(len(train) == cfg["n_train"], f"train size {len(train)} (config {cfg['n_train']})")
        report(len(test) == cfg["n_test"], f"test size {len(test)} (config {cfg['n_test']})")
        ids = [r["id"] for r in train + test]
        report(len(ids) == len(set(ids)), "all ids unique")
        tq = {r["question"] for r in train}
        overlap = sum(r["question"] in tq for r in test)
        report(overlap == 0, f"train/test question overlap = {overlap}")
        report(len({r['question'] for r in test}) == len(test), "no duplicate test questions")

        if task == "code":
            if "solution" not in json.loads(test[0]["meta"]):
                report(False, "no reference solutions in meta: delete data/code and rerun")
                continue
            passed = [check(f"<answer>```python\n{json.loads(r['meta'])['solution']}\n```</answer>",
                            r["answer"], r["meta"]) for r in test]
            rate = sum(passed) / len(passed)
            # a few MBPP reference solutions are known to be broken or slow
            report(rate >= 0.95, f"reference solutions pass tests: {rate:.1%}")
        else:
            gold = [check(f"<answer>{r['answer']}</answer>", r["answer"], r["meta"]) for r in test]
            rate = sum(gold) / len(gold)
            report(rate == 1.0, f"gold answer in tags scores correct: {rate:.1%}")
            bare = sum(check(r["answer"], r["answer"], r["meta"]) for r in test) / len(test)
            report(True, f"gold answer WITHOUT tags scores correct: {bare:.1%} (fallback reader)", info=True)

        junk = ["", "I don't know.", "<answer></answer>", "<answer>I don't know</answer>"]
        false_pass = sum(check(j, r["answer"], r["meta"]) for r in test[:50] for j in junk)
        report(false_pass == 0, f"junk answers scored correct: {int(false_pass)} (want 0)")


# ------------------------------------------------------------------ rescore
def verify_rescore(results_dir):
    """Re-grade every saved reply with this machine's checkers; compare to stored verdicts + scores.csv."""
    from src.data.load import DATA_DIR
    from src.eval.rrr import latest_scores
    from src.rewards import get_checker
    from src.utils.config import ROOT

    res = ROOT / results_dir
    print(f"\n== rescore: {res}")
    scores = latest_scores(res / "scores.csv")
    files = sorted((res / "generations").glob("*/*.jsonl"))
    report(len(files) > 0, f"{len(files)} saved generation files found")
    for f in files:
        label, task = f.parent.name, f.stem
        rows = {r["id"]: r for r in map(json.loads, open(DATA_DIR / task / "test.jsonl"))}
        gens = [json.loads(l) for l in open(f)]
        missing = [g["id"] for g in gens if g["id"] not in rows]
        if missing:
            report(False, f"{label}/{task}: {len(missing)} ids not in data/{task}/test.jsonl (different splits?)")
            continue
        check = get_checker(task)
        new = [check(g["completion"], rows[g["id"]]["answer"], rows[g["id"]]["meta"]) for g in gens]
        flips = [g["id"] for g, n in zip(gens, new) if n != g["correct"]]
        acc = sum(new) / len(new)
        stored = scores.get((label, task))
        same_acc = stored is not None and abs(stored - acc) < 1e-3
        report(not flips and same_acc,
               f"{label:<22} {task:<6} acc {acc:.3f} vs scores.csv {stored}  verdict changes: {len(flips)}"
               + (f" e.g. {flips[:3]}" if flips else ""))


# ------------------------------------------------------------------ eval
def verify_eval(model_path, task, n):
    from src.data.load import load_split
    from src.eval.evaluate import eval_dtype, generate
    from src.utils.model import load_model, load_tokenizer

    print(f"\n== eval: {model_path} on {task}")
    cfg = load_config(task=task)
    tok = load_tokenizer(model_path)
    model = load_model(model_path, eval_dtype(cfg)).eval()
    qs = [r["question"] for r in load_split(task, "test", n)]
    bs, mx = cfg["eval"]["batch_size"], cfg["eval"]["max_new_tokens"]

    a = generate(model, tok, qs, bs, mx)
    b = generate(model, tok, qs, bs, mx)
    same = sum(x == y for x, y in zip(a, b))
    report(same == len(qs), f"same settings twice -> identical replies: {same}/{len(qs)}")

    k = min(8, len(qs))
    c = generate(model, tok, qs[:k], 1, mx)
    agree = sum(x == y for x, y in zip(a[:k], c))
    report(True, f"batch {bs} vs batch 1 identical replies: {agree}/{k} "
                 "(small differences are normal: padding changes GPU arithmetic slightly)", info=True)


# ------------------------------------------------------------------ weights
@torch.no_grad()
def verify_weights(spec_dir, base_path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.merge.merge import LazyWeights

    spec_dir = Path(spec_dir)
    print(f"\n== weights: {spec_dir}")
    info = json.loads((spec_dir / "run_info.json").read_text())
    targets = set(info["config"]["lora"]["target_modules"])

    base, spec = LazyWeights(base_path), LazyWeights(spec_dir / "merged")
    report(set(base.keys()) == set(spec.keys()), "same tensor names as base")
    changed, rel = [], []
    for k in base.keys():
        b, s = base.get(k).float(), spec.get(k).float()   # base is bf16, saved specialist float32
        if b.shape != s.shape:
            report(False, f"{k}: shape differs from base")
            continue
        if not torch.equal(b, s):
            changed.append(k)
            rel.append(((s - b).norm() / b.norm()).item())
    outside = [k for k in changed if k.split(".")[-2] not in targets]
    report(len(changed) > 0, f"{len(changed)} tensors changed by training")
    report(not outside, f"only LoRA target modules changed (outside: {outside[:3]})")
    if rel:
        report(True, f"relative change |dW|/|W|: mean {sum(rel)/len(rel):.2e}, max {max(rel):.2e}", info=True)

    logs = [h for h in info.get("log_history", []) if "reward_std" in h]
    if logs:
        zero = sum(float(h.get("frac_reward_zero_std", 0)) for h in logs) / len(logs)
        report(zero < 0.9, f"GRPO groups with no reward variation: {zero:.0%} (want well below 90%)")
        r0, r1 = float(logs[0]["reward"]), float(logs[-1]["reward"])
        report(True, f"training reward first log {r0:.3f} -> last log {r1:.3f}", info=True)

    # saved full model must behave like base + adapter
    tok = AutoTokenizer.from_pretrained(spec_dir / "merged")
    text = tok.apply_chat_template([{"role": "user", "content": "What is 17 + 25? Explain."}],
                                   tokenize=False, add_generation_prompt=True) + "17 + 25 equals 42 because"
    ids = tok(text, return_tensors="pt").input_ids
    peft = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.float32), str(spec_dir / "adapter"))
    full = AutoModelForCausalLM.from_pretrained(spec_dir / "merged", dtype=torch.float32)
    la, lb = peft(ids).logits[0], full(ids).logits[0]
    top1 = (la.argmax(-1) == lb.argmax(-1)).float().mean().item()
    report(top1 >= 0.95, f"saved full model vs base+adapter: same top token at {top1:.0%} of positions")
    report(True, f"max logit difference {(la - lb).abs().max().item():.3f} (float32 save: expect ~0)", info=True)


# ------------------------------------------------------------------ merge
def verify_merge(spec_paths, base_path):
    from src.merge.merge import LazyWeights, merge_models

    print(f"\n== merge: {len(spec_paths)} specialist(s)")
    base, a = LazyWeights(base_path), LazyWeights(spec_paths[0])

    def equal_to(merged, ref):   # compare values, not dtypes (base bf16, merges float32)
        return all(torch.equal(merged[k].float(), ref.get(k).float()) for k in merged)

    m, _ = merge_models(base_path, [spec_paths[0]], "avg")
    report(equal_to(m, a), "avg of one specialist == that specialist")
    m, _ = merge_models(base_path, [spec_paths[0], spec_paths[0]], "avg")
    report(equal_to(m, a), "avg of a specialist with itself == that specialist")
    m, _ = merge_models(base_path, [spec_paths[0]], "ties", lam=1.0, density=1.0)
    report(equal_to(m, a), "TIES of one specialist (keep all values) == that specialist")
    m, changed = merge_models(base_path, spec_paths, "task_arithmetic", lam=0.0)
    report(equal_to(m, base), "task arithmetic with scale 0 == base")

    if len(spec_paths) >= 2:
        specs = [LazyWeights(p) for p in spec_paths]
        m, changed = merge_models(base_path, spec_paths, "avg")
        n_changed = [sum(not torch.equal(s.get(k).float(), base.get(k).float()) for k in base.keys())
                     for s in specs]
        report(changed >= max(n_changed), f"avg changes {changed} tensors (each specialist: {n_changed})")
        differs = [not equal_to(m, s) for s in specs]
        report(all(differs), "merged model differs from every individual specialist")
        k = next(k for k in base.keys() if not torch.equal(specs[0].get(k).float(), base.get(k).float()))
        expect = base.get(k).float() + torch.stack(
            [s.get(k).float() - base.get(k).float() for s in specs]).mean(0)
        err = (m[k].float() - expect).abs().max().item()
        report(err < 1e-6, f"avg on '{k}' matches hand computation (max err {err:.1e})")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--tasks", nargs="+", default=TASKS)
    e = sub.add_parser("eval")
    e.add_argument("--model", default=DEFAULT_BASE)
    e.add_argument("--task", default="gsm8k")
    e.add_argument("--n", type=int, default=16)
    w = sub.add_parser("weights")
    w.add_argument("--specialist", required=True, help="checkpoints/specialists/<task>/seed<k>")
    w.add_argument("--base", default=DEFAULT_BASE)
    m = sub.add_parser("merge")
    m.add_argument("--specialists", nargs="+", required=True, help=".../merged folders")
    m.add_argument("--base", default=DEFAULT_BASE)
    r = sub.add_parser("rescore")
    r.add_argument("--results", default="results", help="folder holding scores.csv + generations/")
    a = p.parse_args()

    if a.cmd == "data":
        verify_data(a.tasks)
    elif a.cmd == "rescore":
        verify_rescore(a.results)
    elif a.cmd == "eval":
        verify_eval(a.model, a.task, a.n)
    elif a.cmd == "weights":
        verify_weights(a.specialist, a.base)
    elif a.cmd == "merge":
        verify_merge(a.specialists, a.base)

    print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED:\n  - " + "\n  - ".join(FAILS)))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()