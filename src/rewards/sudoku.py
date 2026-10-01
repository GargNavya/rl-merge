import re

from .common import extract_final, rg_score

_ROW = re.compile(r"^[\s|`*\-]*([1-4])[\s,|]+([1-4])[\s,|]+([1-4])[\s,|]+([1-4])[\s|`*]*$")


def extract(text: str, meta: str = "") -> str | None:
    """Last block of 4 consecutive lines that each look like a 4x4 sudoku row."""
    final = extract_final(text)
    src = final if final is not None else (text or "")
    rows = [_ROW.match(line) for line in src.splitlines()]
    last = None
    for i in range(len(rows) - 3):
        if all(rows[i:i + 4]):
            last = i
    if last is None:
        return None
    return "\n".join(" ".join(m.groups()) for m in rows[last:last + 4])


def check(completion: str, answer: str, meta: str) -> float:
    return rg_score(extract(completion), meta)