import re

from .common import extract_final

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_number(s: str) -> float | None:
    nums = _NUM.findall(s.replace(",", "").replace("$", ""))
    return float(nums[-1]) if nums else None


def extract(text: str, meta: str = "") -> str | None:
    final = extract_final(text)
    if final is not None:
        return final
    if "####" in (text or ""):
        return text.rsplit("####", 1)[-1]
    return text  # fallback: last number anywhere in the reply


def check(completion: str, answer: str, meta: str = "") -> float:
    pred = extract(completion)
    p = parse_number(pred) if pred else None
    return 1.0 if p is not None and abs(p - float(answer)) < 1e-6 else 0.0