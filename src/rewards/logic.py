import json
import re

from .common import extract_final, rg_score


def extract(text: str, meta: str) -> str | None:
    """Rebuild 'A is a X, and B is a Y' from the LAST claim made about each person."""
    final = extract_final(text)
    src = final if final is not None else (text or "")
    md = json.loads(meta)["metadata"]
    terms = md["knight_knave_terms"]
    roles = "|".join(re.escape(terms[k]) for k in ("knight", "knave"))
    parts = []
    for name in md["names"]:
        pat = rf"\b{re.escape(name)}\b\W*(?:is|must be|would be)\s+(?:an?\s+)?({roles})\b(?!\s*/)"
        found = re.findall(pat, src, flags=re.I)
        if not found:
            return None
        parts.append(f"{name} is a {found[-1].lower()}")
    return ", and ".join(parts)


def check(completion: str, answer: str, meta: str) -> float:
    return rg_score(extract(completion, meta), meta)