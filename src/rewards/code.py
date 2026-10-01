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
    """Last code block inside <answer>/boxed if present, else last code block in the reply."""
    final = extract_final(text)
    if final is not None:
        blocks = _BLOCK.findall(final)
        return blocks[-1] if blocks else final
    blocks = _BLOCK.findall(text or "")
    return blocks[-1] if blocks else None


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