import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS = ["gsm8k", "logic", "code"]  # countdown, sudoku: base model near 0%, skipped for now
DEFAULT_RUN = "v1"


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_override(cfg: dict, item: str) -> None:
    """item like 'grpo.max_steps=5'."""
    key, _, raw = item.partition("=")
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(raw)


def load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(base="configs/base.yaml", task=None, overrides=()) -> dict:
    cfg = load_yaml(ROOT / base if not Path(base).is_absolute() else base)
    if task:
        task_path = Path(task)
        if not task_path.suffix:
            task_path = ROOT / "configs" / "tasks" / f"{task}.yaml"
        cfg = deep_merge(cfg, load_yaml(task_path))
    for o in overrides or ():
        apply_override(cfg, o)
    return cfg


def task_config(name: str) -> dict:
    return load_yaml(ROOT / "configs" / "tasks" / f"{name}.yaml")["task"]


# default task lists: `tasks:` (train + eval) and `merge_tasks:` (merge + RRR) in configs/base.yaml
TASKS = list(load_config().get("tasks", DEFAULT_TASKS))
MERGE_TASKS = list(load_config().get("merge_tasks", TASKS))


# ---------------------------------------------------------------- run layout
# One "run" = one training setup (reward, format, data, hyperparameters). Everything a run
# produces sits under its name, so a new setup never overwrites an old one:
#   checkpoints/specialists/<run>/<task>/seed<k>/
#   checkpoints/merged/<run>/seed<k>/<merge name>/
#   results/<run>/seed<k>/{scores.csv, rrr.csv, generations/}

def run_name(run=None) -> str:
    return str(run or load_config().get("run", DEFAULT_RUN))


def train_seed(seed=None) -> int:
    return int(seed if seed is not None else load_config()["seed"])


def spec_dir(task, seed=None, run=None) -> Path:
    return ROOT / "checkpoints" / "specialists" / run_name(run) / task / f"seed{train_seed(seed)}"


def merged_dir(name, seed=None, run=None) -> Path:
    return ROOT / "checkpoints" / "merged" / run_name(run) / f"seed{train_seed(seed)}" / name


def results_dir(seed=None, run=None) -> Path:
    return ROOT / "results" / run_name(run) / f"seed{train_seed(seed)}"


def merge_name(method, lam=1.0, density=0.2) -> str:
    """Folder name and label suffix of a merge: avg | task_arithmetic_lam0.5 | ties_d0.2_lam1."""
    lam, density = float(lam), float(density)
    if method == "avg":
        return "avg"
    if method == "task_arithmetic":
        return f"task_arithmetic_lam{lam:g}"
    if method == "ties":
        return f"ties_d{density:g}_lam{lam:g}"
    raise ValueError(method)


if __name__ == "__main__":
    # For shell scripts. Prints paths relative to the repo root.
    #   python -m src.utils.config run
    #   python -m src.utils.config seed
    #   python -m src.utils.config spec gsm8k [--seed 42] [--run v1]
    #   python -m src.utils.config merged ties_d0.2_lam1 [--seed 42] [--run v1]
    #   python -m src.utils.config results [--seed 42] [--run v1]
    #   python -m src.utils.config mergename ties --lam 1 --density 0.2
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["run", "seed", "spec", "merged", "results", "mergename"])
    p.add_argument("name", nargs="?")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run", default=None)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--density", type=float, default=0.2)
    a = p.parse_args()
    rel = lambda x: x.relative_to(ROOT)
    print({"run": lambda: run_name(a.run),
           "seed": lambda: train_seed(a.seed),
           "spec": lambda: rel(spec_dir(a.name, a.seed, a.run)),
           "merged": lambda: rel(merged_dir(a.name, a.seed, a.run)),
           "results": lambda: rel(results_dir(a.seed, a.run)),
           "mergename": lambda: merge_name(a.name, a.lam, a.density)}[a.what]())