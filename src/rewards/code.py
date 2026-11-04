"""Runs model code against MBPP unit tests in a separate Python process.

This is isolation, not a real security sandbox. Fine for MBPP on your own machine;
use a container (e.g. Docker) before running untrusted code at scale.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

from .common import extract_final

_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)
TIMEOUT_S = 5.0


def extract(text: str, meta: str = "") -> str | None:
    """The code answer is a ```python block, wherever it sits; <answer> tags are ignored for code.
    1. last code block in the reply that defines something (def/class); else the last code block
    2. no code block at all: whatever is inside <answer>/boxed (raw code without a fence)
    Until 4 Nov, tags came first: a number inside <answer> hid correct code written outside the tags."""
    blocks = _BLOCK.findall(text or "")
    if blocks:
        defining = [b for b in blocks if re.search(r"^\s*(def|class)\s", b, re.M)]
        return (defining or blocks)[-1]
    return extract_final(text)


def run_tests(code: str, tests: list[str], setup: str = "", timeout: float = TIMEOUT_S) -> bool:
    program = "\n".join([setup, code, *tests])
    with tempfile.TemporaryDirectory() as d:
        try:
            r = subprocess.run([sys.executable, "-I", "-c", program], cwd=d,
                               capture_output=True, timeout=timeout,
                               env={"PATH": os.environ.get("PATH", "")})
        except subprocess.TimeoutExpired:
            return False
    return r.returncode == 0


def check(completion: str, answer: str = "", meta: str = "{}") -> float:
    code = extract(completion)
    if not code:
        return 0.0
    m = json.loads(meta)
    return 1.0 if run_tests(code, m["tests"], m.get("setup", "")) else 0.0