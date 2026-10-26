"""Train one GRPO specialist. Every specialist uses this same script; only the task config differs.

Example:
  python -m src.train.grpo --task gsm8k
  python -m src.train.grpo --task gsm8k --seed 1 --set grpo.max_steps=20
"""
import argparse
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from src.data.load import load_split, to_grpo_dataset
from src.rewards import make_reward_funcs
from src.utils.config import ROOT, load_config
from src.utils.model import load_model, load_tokenizer, pick_dtype
from src.utils.seed import set_seed
from src.utils.tracking import git_commit, now, write_json


def save_full_weights(base_name, adapter_dir, out_dir, tok):
    """base + LoRA -> full model, saved in float32. bf16 would round away most of the LoRA update
    (notebook 05: 60% of entries lost, direction cosine 0.84). Untouched weights stay bit-identical
    to the bf16 base, because bf16 -> float32 is exact (important: merging reads specialist - base)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--set", nargs="*", default=[], help="overrides like grpo.max_steps=20")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    cfg = load_config(task=a.task, overrides=a.set)
    task = cfg["task"]["name"]
    seed = a.seed if a.seed is not None else cfg["seed"]
    set_seed(seed)
    out = Path(a.out) if a.out else ROOT / "checkpoints" / "specialists" / task / f"seed{seed}"

    tok = load_tokenizer(cfg["model"]["name"])
    model = load_model(cfg["model"]["name"], cfg["model"]["dtype"])
    dtype = pick_dtype(cfg["model"]["dtype"])

    train_ds = to_grpo_dataset(load_split(task, "train"))
    reward_funcs = make_reward_funcs(task)

    args = GRPOConfig(
        output_dir=str(out / "trainer"),
        seed=seed,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        reward_weights=[1.0, cfg["reward"]["format_weight"]],
        **cfg["grpo"],
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=train_ds,
        processing_class=tok,
        peft_config=LoraConfig(task_type="CAUSAL_LM", **cfg["lora"]),
    )

    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60

    trainer.model.save_pretrained(out / "adapter")
    save_full_weights(cfg["model"]["name"], out / "adapter", out / "merged", tok)

    write_json({
        "task": task, "seed": seed, "config": cfg, "git_commit": git_commit(),
        "finished": now(), "train_minutes": round(minutes, 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
        "log_history": trainer.state.log_history,
    }, out / "run_info.json")
    print(f"[train] {task} done in {minutes:.1f} min -> {out}")


if __name__ == "__main__":
    main()