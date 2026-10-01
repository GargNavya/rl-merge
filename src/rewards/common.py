"""Answer extraction shared by all checkers.

Order: <answer> tag -> \\boxed{} -> task-specific reader on the full text.
Always the LAST occurrence, so listing several candidates never helps the model.
Format compliance is measured separately (format_ok) and never affects accuracy.
"""
import json
import re

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)


def extract_tagged(text: str) -> str | None:
    found = _ANSWER.findall(text or "")
    return found[-1].strip() if found else None


def extract_boxed(text: str) -> str | None:
    """Last \\boxed{...}, handling nested braces."""
    start = (text or "").rfind("\\boxed{")
    if start == -1:
        return None
    i, depth = start + len("\\boxed{"), 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j].strip()
    return None


def extract_final(text: str) -> str | None:
    t = extract_tagged(text)
    return t if t is not None else extract_boxed(text)


def format_ok(text: str) -> bool:
    return (text or "").count("<answer>") == 1 and extract_tagged(text) is not None


# ---------------- Reasoning Gym ----------------
_RG_SCORERS = {}


def rg_score(pred: str | None, meta: str) -> float:
    """Binary: 1 only if Reasoning Gym's own scorer gives full credit."""
    import reasoning_gym as rg
    if not pred:
        return 0.0
    entry = json.loads(meta)
    name = entry["metadata"]["source_dataset"]
    if name not in _RG_SCORERS:
        _RG_SCORERS[name] = rg.create_dataset(name, size=1, seed=0)
    try:
        score = _RG_SCORERS[name].score_answer(answer=pred, entry=entry)
    except Exception:
        return 0.0
    return 1.0 if score >= 1.0 else 0.0