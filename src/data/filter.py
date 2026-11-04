"""Difficulty filter: keep training questions the base model sometimes solves.

For each question in a pool, sample n answers from the base model exactly as GRPO would
(training prompt, temperature, max tokens), grade them, and keep questions with
keep_min..keep_max correct out of n (default 1..7 of 8). Questions solved 0/8 or 8/8 give
GRPO no signal: every answer in the group gets the same reward.

Writes, next to the untouched train.jsonl / test.jsonl:
  data/<task>/<out>.jsonl              kept questions (same row format as train.jsonl)
  data/<task>/<out>_passcounts.jsonl   every pool question with its k/n (resumable progress log)
  data/<task>/<out>_stats.json         histogram, sizes, settings
Training reads <out>.jsonl when the task config has `train_split: <out>`.

Pool: gsm8k = all of GSM8K train. Reasoning Gym tasks = --pool generated items (same train seed).
Pool questions that appear in test.jsonl are dropped. Code is not filtered (only 474 problems).

Examples:
  CUDA_VISIBLE_DEVICES=0 python -m src.data.filter --task gsm8k
  CUDA_VISIBLE_DEVICES=1 python -m src.data.filter --task logic_hard --pool 8000
  CUDA_VISIBLE_DEVICES=0 python -m src.data.filter --task gsm8k --backend vllm     # if vLLM is installed
  python -m src.data.filter --task gsm8k --pool 64 --out filter_try               # quick try on 64 questions
"""
import argparse
import collections
import copy
import json
import random
import sys
import time

import torch

from src.data.load import BUILDERS, DATA_DIR, SPLIT_SEED, build_splits, load_split
from src.data.prompts import build_messages
from src.rewards import get_checker
from src.utils.config import load_config, task_config
from src.utils.model import load_model, load_tokenizer
from src.utils.tracking import git_commit, now, write_json


def make_pool(task, pool_size):
    cfg = copy.deepcopy(task_config(task))
    if cfg["source"] == "hf_mbpp":
        sys.exit("[filter] code is not filtered: MBPP has only 474 training problems.")
    if cfg["source"] == "reasoning_gym":
        cfg["n_train"] = pool_size or 3 * cfg["n_train"]
    rows = BUILDERS[cfg["source"]](cfg)["train"]
    random.Random(SPLIT_SEED).shuffle(rows)
    test_q = {r["question"] for r in load_split(task, "test")}
    rows = [r for r in rows if r["question"] not in test_q]
    return rows[:pool_size] if pool_size else rows


def texts_for(tok, rows, task, style):
    return [tok.apply_chat_template(build_messages(r["question"], task, style), tokenize=False,
                                    add_generation_prompt=True) for r in rows]


class HFSampler:
    def __init__(self, name, dtype, n, temperature, max_new_tokens):
        self.tok = load_tokenizer(name)
        self.model = load_model(name, dtype).eval()
        self.kw = dict(do_sample=True, num_return_sequences=n, temperature=temperature, top_p=1.0, top_k=0,
                       max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
        self.n = n

    @torch.no_grad()
    def __call__(self, texts):
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        out = self.model.generate(**enc, **self.kw)
        dec = self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return [dec[i * self.n:(i + 1) * self.n] for i in range(len(texts))]


class VLLMSampler:
    def __init__(self, name, dtype, n, temperature, max_new_tokens):
        from vllm import LLM, SamplingParams
        self.tok = load_tokenizer(name)
        self.llm = LLM(model=name, dtype="bfloat16" if dtype == "auto" else dtype, gpu_memory_utilization=0.85)
        self.sp = SamplingParams(n=n, temperature=temperature, top_p=1.0, max_tokens=max_new_tokens)

    def __call__(self, texts):
        return [[o.text for o in r.outputs] for r in self.llm.generate(texts, self.sp, use_tqdm=False)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--pool", type=int, default=None, help="pool size (default: all gsm8k train; 3x n_train for RG)")
    p.add_argument("--n", type=int, default=8, help="answers per question (match grpo.num_generations)")
    p.add_argument("--keep-min", type=int, default=1)
    p.add_argument("--keep-max", type=int, default=None, help="default: n - 1")
    p.add_argument("--batch", type=int, default=64, help="questions per generate call (x n sequences)")
    p.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    p.add_argument("--out", default="train_filtered", help="file name stem under data/<task>/")
    p.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. prompt_style=v2")
    p.add_argument("--overwrite", action="store_true", help="start over instead of resuming")
    a = p.parse_args()

    cfg = load_config(task=a.task, overrides=a.set)
    keep_max = a.n - 1 if a.keep_max is None else a.keep_max
    style, name = cfg.get("prompt_style", "v1"), cfg["model"]["name"]
    temperature, max_new = cfg["grpo"]["temperature"], cfg["grpo"]["max_completion_length"]
    d = DATA_DIR / a.task
    out_rows, counts_path, stats_path = d / f"{a.out}.jsonl", d / f"{a.out}_passcounts.jsonl", d / f"{a.out}_stats.json"
    if a.out in ("train", "test"):
        sys.exit("[filter] --out must not be train or test")
    if out_rows.exists() and not a.overwrite:
        sys.exit(f"[filter] {out_rows} exists. Pass --overwrite to redo it.")
    if a.overwrite:
        counts_path.unlink(missing_ok=True)

    build_splits(a.task)
    pool = make_pool(a.task, a.pool)
    done = {}
    if counts_path.exists():
        done = {r["id"]: r["k"] for r in map(json.loads, open(counts_path))}
        print(f"[filter] resuming: {len(done)} of {len(pool)} already sampled")
    todo = [r for r in pool if r["id"] not in done]
    print(f"[filter] {a.task}: pool {len(pool)}, to sample {len(todo)}, n={a.n}, T={temperature}, "
          f"max {max_new} tokens, prompt_style {style}, backend {a.backend}")

    sampler = (HFSampler if a.backend == "hf" else VLLMSampler)(name, cfg["model"]["dtype"], a.n, temperature, max_new)
    check = get_checker(a.task)
    t0 = time.time()
    step = a.batch if a.backend == "hf" else 4096
    with open(counts_path, "a") as log:
        for i in range(0, len(todo), step):
            chunk = todo[i:i + step]
            answers = sampler(texts_for(sampler.tok, chunk, a.task, style))
            for r, ans in zip(chunk, answers):
                k = int(sum(check(x, r["answer"], r["meta"]) for x in ans))
                done[r["id"]] = k
                log.write(json.dumps({"id": r["id"], "k": k}) + "\n")
            log.flush()
            n_done = min(i + step, len(todo))
            rate = (time.time() - t0) / n_done
            print(f"  {n_done}/{len(todo)}  {rate * (len(todo) - n_done) / 60:.0f} min left", flush=True)

    kept = [r for r in pool if a.keep_min <= done[r["id"]] <= keep_max]
    with open(out_rows, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    hist = collections.Counter(done[r["id"]] for r in pool)
    write_json({"task": a.task, "pool": len(pool), "kept": len(kept), "keep": [a.keep_min, keep_max], "n": a.n,
                "histogram_k": {k: hist.get(k, 0) for k in range(a.n + 1)},
                "base_mean_pass_rate": sum(done[r["id"]] for r in pool) / (a.n * len(pool)),
                "model": name, "prompt_style": style, "temperature": temperature, "max_new_tokens": max_new,
                "backend": a.backend, "git_commit": git_commit(), "time": now()}, stats_path)
    print(f"\n[filter] k/{a.n} histogram: " + "  ".join(f"{k}:{hist.get(k, 0)}" for k in range(a.n + 1)))
    print(f"[filter] kept {len(kept)} of {len(pool)} -> {out_rows}")
    print(f"[filter] to train on it, set in configs/tasks/{a.task}.yaml under task:  train_split: {a.out}")


if __name__ == "__main__":
    main()