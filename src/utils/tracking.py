import csv
import datetime as dt
import json
import subprocess
from pathlib import Path

from .config import ROOT

SCORES_CSV = ROOT / "results" / "scores.csv"
SCORE_FIELDS = ["time", "label", "model", "task", "n", "accuracy", "format_rate",
                "mean_len", "seed", "git_commit"]


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                      stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=ROOT, stderr=subprocess.DEVNULL)
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def append_score(row: dict, path: Path = SCORES_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCORE_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in SCORE_FIELDS})


def write_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
