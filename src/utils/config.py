import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TASKS = ["gsm8k", "logic", "code"]


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
