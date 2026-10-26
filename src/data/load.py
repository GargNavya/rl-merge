"""Build fixed train/test splits per task and cache them in data/<task>/{train,test}.jsonl.

Splits are built once and then always read from the cache, so every model is
scored on exactly the same questions. Delete data/<task>/ to rebuild.
"""
import json
import random
from pathlib import Path

from datasets import Dataset

from src.data.prompts import CODE_SUFFIX, build_messages
from src.utils.config import ROOT, task_config

DATA_DIR = ROOT / "data"
SPLIT_SEED = 1234  # fixed forever; independent of training seed


def _row(task, split, i, question, answer="", meta=None):
    return {"id": f"{task}-{split}-{i}", "task": task, "question": question,
            "answer": str(answer), "meta": json.dumps(meta or {}, default=str)}


def _gsm8k(cfg):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")
    out = {}
    for split in ["train", "test"]:
        out[split] = [_row("gsm8k", split, i, ex["question"],
                           ex["answer"].split("####")[-1].strip().replace(",", ""))
                      for i, ex in enumerate(ds[split])]
    return out


def _mbpp(cfg):
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "full")

    def conv(ex, split, i):
        q = f"{ex['text']}\nYour code should pass this test:\n{ex['test_list'][0]}{CODE_SUFFIX}"
        meta = {"tests": ex["test_list"], "setup": ex.get("test_setup_code", "") or "",
                "solution": ex["code"]}  # reference code, only used by verify.py
        return _row("code", split, i, q, "", meta)

    train = [conv(ex, "train", i) for i, ex in
             enumerate(list(ds["train"]) + list(ds["validation"]) + list(ds["prompt"]))]
    test, seen = [], set()
    for i, ex in enumerate(ds["test"]):
        r = conv(ex, "test", i)
        if r["question"] not in seen:          # MBPP test repeats a few problems
            seen.add(r["question"])
            test.append(r)
    return {"train": train, "test": test}


def _reasoning_gym(cfg):
    import reasoning_gym as rg
    name, rcfg = cfg["rg_name"], cfg.get("rg_config", {}) or {}
    task = cfg["name"]
    train_ds = rg.create_dataset(name, size=cfg["n_train"], seed=SPLIT_SEED, **rcfg)
    # RG seeds item i with (seed + i), so test seed must be far away or items repeat.
    test_ds = rg.create_dataset(name, size=cfg["n_test"] * 2, seed=SPLIT_SEED + 10_000_000, **rcfg)
    cut, suffix = cfg.get("prompt_cut"), cfg.get("prompt_suffix", "")

    def question(e):
        # Replace RG's own answer instructions (e.g. "no other text") which stop the model reasoning
        q = e["question"]
        if cut and cut in q:
            q = q[:q.index(cut)].rstrip() + "\n\n" + suffix.strip()
        return q

    train = [_row(task, "train", i, question(e), e["answer"], e) for i, e in enumerate(train_ds)]
    seen = {r["question"] for r in train}
    test = [_row(task, "test", i, question(e), e["answer"], e)
            for i, e in enumerate(test_ds) if question(e) not in seen]
    return {"train": train, "test": test}


BUILDERS = {"hf_gsm8k": _gsm8k, "hf_mbpp": _mbpp, "reasoning_gym": _reasoning_gym}


def _write(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _read(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_splits(task: str, force: bool = False) -> None:
    cfg = task_config(task)
    d = DATA_DIR / task
    if not force and (d / "train.jsonl").exists() and (d / "test.jsonl").exists():
        return
    splits = BUILDERS[cfg["source"]](cfg)
    rng = random.Random(SPLIT_SEED)
    for split, n in [("train", cfg["n_train"]), ("test", cfg["n_test"])]:
        rows = splits[split]
        rng.shuffle(rows)
        _write(rows[:n], d / f"{split}.jsonl")
    print(f"[data] built {task}: train={min(len(splits['train']), cfg['n_train'])} "
          f"test={min(len(splits['test']), cfg['n_test'])}")


def load_split(task: str, split: str, n: int | None = None) -> list[dict]:
    build_splits(task)
    rows = _read(DATA_DIR / task / f"{split}.jsonl")
    return rows[:n] if n else rows


def to_grpo_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_list([{**r, "prompt": build_messages(r["question"])} for r in rows])


if __name__ == "__main__":
    import argparse
    from src.utils.config import TASKS
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=TASKS)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    for t in a.tasks:
        build_splits(t, force=a.force)