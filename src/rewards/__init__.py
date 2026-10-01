from . import code, countdown, gsm8k, logic, sudoku
from .common import format_ok

CHECKERS = {
    "gsm8k": gsm8k.check,
    "countdown": countdown.check,
    "logic": logic.check,
    "sudoku": sudoku.check,
    "code": code.check,
}


def get_checker(task: str):
    return CHECKERS[task]


def _text(c) -> str:
    # TRL passes conversational completions as [{"role": "assistant", "content": ...}]
    return c[0]["content"] if isinstance(c, list) else c


def make_reward_funcs(task: str):
    checker = get_checker(task)

    def correctness(prompts, completions, answer, meta, **kwargs):
        return [checker(_text(c), a, m) for c, a, m in zip(completions, answer, meta)]

    def formatting(prompts, completions, **kwargs):
        return [1.0 if format_ok(_text(c)) else 0.0 for c in completions]

    correctness.__name__ = f"{task}_correct"
    formatting.__name__ = "format"
    return [correctness, formatting]
