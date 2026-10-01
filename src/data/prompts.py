SYSTEM_PROMPT = (
    "You are a careful problem solver. Reason step by step. "
    "Then give only your final answer inside <answer> </answer> tags, "
    "for example: <answer>42</answer>"
)

CODE_SUFFIX = (
    "\n\nWrite the complete Python function. Put the final code inside <answer> </answer> "
    "as a ```python code block."
)


def build_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]