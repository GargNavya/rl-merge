"""Merge specialists that share one base model. Works tensor by tensor, so RAM stays low.

Task vector  tv_i = specialist_i - base
  avg              base + mean(tv_i)                   (= plain weight averaging)
  task_arithmetic  base + lam * sum(tv_i)
  ties             base + lam * TIES(tv_i)             (trim small values, elect sign, disjoint mean)

Example:
  python -m src.merge.merge --method ties --lam 1.0 --density 0.2 \
      --specialists checkpoints/specialists/*/seed42/merged --out checkpoints/merged/ties
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from src.utils.tracking import git_commit, now, write_json


class LazyWeights:
    """Reads single tensors from a model folder (single or sharded safetensors)."""

    def __init__(self, path):
        path = Path(path)
        if not path.exists():
            path = Path(snapshot_download(str(path), allow_patterns=["*.safetensors", "*.json"]))
        self.path = path
        index = path / "model.safetensors.index.json"
        if index.exists():
            wm = json.loads(index.read_text())["weight_map"]
            self.key_file = {k: path / f for k, f in wm.items()}
        else:
            f = path / "model.safetensors"
            with safe_open(f, "pt") as h:
                self.key_file = {k: f for k in h.keys()}
        self._handles = {}

    def keys(self):
        return list(self.key_file)

    def get(self, key) -> torch.Tensor:
        f = self.key_file[key]
        if f not in self._handles:
            self._handles[f] = safe_open(f, "pt")
        return self._handles[f].get_tensor(key)


def ties(tvs: torch.Tensor, density: float) -> torch.Tensor:
    """tvs: [n_models, ...]. Returns merged task vector."""
    flat = tvs.reshape(tvs.shape[0], -1)
    k = max(1, int(density * flat.shape[1]))
    thresh = flat.abs().kthvalue(flat.shape[1] - k + 1, dim=1, keepdim=True).values
    trimmed = torch.where(flat.abs() >= thresh, flat, torch.zeros_like(flat))
    sign = torch.sign(trimmed.sum(0))
    agree = (torch.sign(trimmed) == sign) & (trimmed != 0)
    num = (trimmed * agree).sum(0)
    cnt = agree.sum(0).clamp(min=1)
    return (num / cnt).reshape(tvs.shape[1:])


def merge_tensor(base, specs, method, lam, density):
    tvs = torch.stack([s.float() - base.float() for s in specs])
    if not tvs.any():
        return base
    if method == "avg":
        delta = tvs.mean(0)
    elif method == "task_arithmetic":
        delta = lam * tvs.sum(0)
    elif method == "ties":
        delta = lam * ties(tvs, density)
    else:
        raise ValueError(method)
    return (base.float() + delta).to(base.dtype)


def merge_models(base_path, spec_paths, method, lam=1.0, density=0.2):
    """Returns (merged state dict, number of tensors that differ from base)."""
    base = LazyWeights(base_path)
    specs = [LazyWeights(s) for s in spec_paths]
    merged, changed = {}, 0
    for key in base.keys():
        b = base.get(key)
        m = merge_tensor(b, [s.get(key) for s in specs], method, lam, density)
        changed += int(m is not b)
        merged[key] = m.contiguous()
    return merged, changed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--specialists", nargs="+", required=True)
    p.add_argument("--method", choices=["avg", "task_arithmetic", "ties"], required=True)
    p.add_argument("--lam", type=float, default=1.0, help="scale on merged task vector")
    p.add_argument("--density", type=float, default=0.2, help="TIES: fraction of values kept")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    merged, changed = merge_models(a.base, a.specialists, a.method, a.lam, a.density)
    save_file(merged, out / "model.safetensors", metadata={"format": "pt"})

    src = Path(a.specialists[0])
    for f in src.iterdir():
        if f.suffix in {".json", ".txt", ".model", ".jinja"} and "safetensors" not in f.name:
            shutil.copy(f, out / f.name)

    write_json({"method": a.method, "lam": a.lam, "density": a.density, "base": a.base,
                "specialists": a.specialists, "tensors_changed": changed,
                "git_commit": git_commit(), "time": now()}, out / "merge_info.json")
    print(f"[merge] {a.method}: {len(a.specialists)} models, {changed} tensors changed -> {out}")


if __name__ == "__main__":
    main()
