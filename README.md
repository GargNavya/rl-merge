# rl-merge

Warm-up for *Does RL-Induced Reasoning Survive Merging?* (Gate 0: pipeline check).

Five GRPO specialists from one base (Qwen2.5-0.5B-Instruct, LoRA), merged three ways,
scored with the Reasoning Retention Ratio (RRR).

| Task | Data | Checker |
|---|---|---|
| gsm8k | GSM8K | final number match |
| countdown | Reasoning Gym `countdown` (easy settings) | expression evaluates to target |
| logic | Reasoning Gym `knights_knaves` | exact match |
| sudoku | Reasoning Gym `mini_sudoku` | valid solved grid |
| code | MBPP (full) | unit tests in a subprocess |

## Setup

```bash
# 1. PyTorch with CUDA for your GPU: https://pytorch.org/get-started/locally/
pip install -U setuptools wheel
pip install -r requirements.txt
git init && git add -A && git commit -m "init"   # commit hash is logged with every result
pytest -q                                         # checkers must pass before any training
```

## Run

```bash
bash scripts/eval_base.sh        # 1. base scores (want ~10-70% per task)
bash scripts/train_all.sh        # 2. five specialists + cross-task scores
bash scripts/merge_and_eval.sh   # 3. merge (avg, task_arithmetic, ties) + RRR
```

Single pieces:

```bash
python -m src.train.grpo --task gsm8k --set grpo.max_steps=20      # quick trial run
python -m src.eval.evaluate --model <path> --label <name> --tasks gsm8k logic
python -m src.merge.merge --method ties --specialists <dirs...> --out checkpoints/merged/x
python -m src.eval.rrr --merged merge-ties
```

Any config value can be overridden with `--set key.sub=value`.

## Outputs

- `results/scores.csv` - every eval ever run, one row per (model, task). Source of all numbers.
- `results/rrr.csv` - RRR per merged model.
- `results/generations/<label>/<task>.jsonl` - raw model answers. Read them.
- `checkpoints/specialists/<task>/seed<k>/` - `adapter/`, `merged/` (full weights), `run_info.json`.

## Rules

- Change a task only through its config. Never fork the training script.
- Splits in `data/` are built once. Don't edit them. If you must, delete and re-score everything.
- Label convention: `base`, `spec-<task>`, `merge-<method>`. `rrr.py` depends on it.

## If it runs out of GPU memory

In `configs/base.yaml`, in this order: `per_device_train_batch_size: 1` (and double
`gradient_accumulation_steps`), `max_completion_length: 256`, `num_generations: 4`,
`eval.batch_size: 8`.
