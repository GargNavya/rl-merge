from . import code, countdown, gsm8k, logic, logic_hard, sudoku
from .common import format_ok

CHECKERS = {
    "gsm8k": gsm8k.check,
    "countdown": countdown.check,
    "logic": logic.check,
    "logic_hard": logic_hard.check,
    "sudoku": sudoku.check,
    "code": code.check,
}


def get_checker(task: str):
    return CHECKERS[task]


def get_format_check(task: str, style: str | None = None):
    """Is the answer in this task's answer slot? prompt_style v1: <answer> tags for every task.
    v2: code = a ```python block; every other task = <answer> tags."""
    from src.data.prompts import prompt_style
    if task == "code" and prompt_style(style) == "v2":
        return code.has_block
    return format_ok


def _text(c) -> str:
    # TRL passes conversational completions as [{"role": "assistant", "content": ...}]
    return c[0]["content"] if isinstance(c, list) else c


def make_reward_funcs(task: str, style: str | None = None):
    checker = get_checker(task)
    fmt = get_format_check(task, style)

    def correctness(prompts, completions, answer, meta, **kwargs):
        return [checker(_text(c), a, m) for c, a, m in zip(completions, answer, meta)]

    def formatting(prompts, completions, **kwargs):
        return [1.0 if fmt(_text(c)) else 0.0 for c in completions]

    correctness.__name__ = f"{task}_correct"
    formatting.__name__ = "format"
    return [correctness, formatting]