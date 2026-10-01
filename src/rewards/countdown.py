import re

from .common import extract_final, rg_score

# a run of numbers joined by + - * / with optional parentheses
_EXPR = re.compile(r"[\d(][\d\s+\-*/().]*[\d)]")


def _last_expression(text: str) -> str | None:
    cands = [m.strip() for m in _EXPR.findall(text.replace("×", "*").replace("÷", "/"))]
    cands = [c for c in cands if re.search(r"[+\-*/]", c) and len(re.findall(r"\d+", c)) >= 2]
    return cands[-1] if cands else None


def extract(text: str, meta: str = "") -> str | None:
    final = extract_final(text)
    src = final if final is not None else (text or "")
    if "=" in src:  # "16 = 19*46/31 - 20" or "19*46/31 - 20 = 16": keep the side with operators
        sides = [s for s in src.split("=") if _last_expression(s)]
        src = sides[-1] if sides else src
    return _last_expression(src)


def check(completion: str, answer: str, meta: str) -> float:
    return rg_score(extract(completion), meta)