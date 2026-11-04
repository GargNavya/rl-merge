"""Prompts. `prompt_style` in configs/base.yaml picks the version; v1 is kept exactly as it was.

v1: one system prompt for every task asks for <answer> tags (code too, through CODE_SUFFIX).
v2: one answer slot per task. gsm8k / logic / logic_hard: <answer> tags. code: a ```python block, no tags.
    The checker reads the same slot (src/rewards/code.py ignores tags for code).
"""
from src.utils.config import load_config

SYSTEM_PROMPT = (
    "You are a careful problem solver. Reason step by step. "
    "Then give only your final answer inside <answer> </answer> tags, "
    "for example: <answer>42</answer>"
)

# Stored inside every cached code question (data/code/*.jsonl). Kept unchanged so cached data stays valid.
CODE_SUFFIX = (
    "\n\nWrite the complete Python function. Put the final code inside <answer> </answer> "
    "as a ```python code block."
)

SYSTEM_PROMPT_V2 = "You are a careful problem solver. Reason step by step, then give your final answer."
SUFFIX_V2 = {
    "gsm8k": "\n\nGive only the final number inside <answer> </answer> tags, for example: <answer>42</answer>",
    "code": "\n\nWrite the complete Python function in a single ```python code block.",
}
DEFAULT_SUFFIX_V2 = "\n\nGive only your final answer inside <answer> </answer> tags."


def prompt_style(style=None) -> str:
    return str(style or load_config().get("prompt_style", "v1"))


def build_messages(question: str, task: str | None = None, style: str | None = None) -> list[dict]:
    style = prompt_style(style)
    if style == "v1":
        system, q = SYSTEM_PROMPT, question
    elif style == "v2":
        if task is None:
            raise ValueError("prompt_style v2 needs the task name")
        system = SYSTEM_PROMPT_V2
        q = question.replace(CODE_SUFFIX, "").rstrip() + SUFFIX_V2.get(task, DEFAULT_SUFFIX_V2)
    else:
        raise ValueError(f"unknown prompt_style {style}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": q},
    ]