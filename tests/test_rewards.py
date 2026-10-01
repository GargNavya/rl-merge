"""Hand-made right and wrong answers for every checker. Run: python -m pytest -q

A checker bug silently ruins every result after it, so these must pass before any training.
Several cases below are real base-model outputs.
"""
import json

import pytest

from src.rewards import format_ok, get_checker
from src.rewards.common import extract_boxed, extract_final, extract_tagged


def tag(ans, reasoning="some reasoning"):
    return f"{reasoning}\n<answer>{ans}</answer>"


# ---------- common ----------
def test_tag_takes_last():
    assert extract_tagged("<answer>1</answer> then <answer> 2 </answer>") == "2"


def test_boxed_nested():
    assert extract_boxed(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"


def test_tag_beats_boxed():
    assert extract_final(r"\boxed{1} <answer>2</answer>") == "2"


def test_format():
    assert format_ok(tag("5"))
    assert not format_ok("the answer is 5")
    assert not format_ok(tag("5") + "<answer>6</answer>")


# ---------- gsm8k ----------
@pytest.mark.parametrize("ans,expected", [
    ("18", 1.0), ("$18", 1.0), ("18.0", 1.0), ("The answer is 18", 1.0), ("17", 0.0),
])
def test_gsm8k_tagged(ans, expected):
    assert get_checker("gsm8k")(tag(ans), "18") == expected


def test_gsm8k_commas_and_negative():
    assert get_checker("gsm8k")(tag("1,234"), "1234") == 1.0
    assert get_checker("gsm8k")(tag("-5"), "-5") == 1.0
    assert get_checker("gsm8k")(tag("5"), "-5") == 0.0


def test_gsm8k_boxed():
    assert get_checker("gsm8k")(r"... so \boxed{18}.", "18") == 1.0


def test_gsm8k_untagged_uses_last_number():
    real = "Percentage increase = 0.1 x 100%\n= 10%\n\nSo, the factory is making $10 more per day now."
    assert get_checker("gsm8k")(real, "10") == 1.0
    assert get_checker("gsm8k")("12 x 2 = $24\n\nSo, Janet makes **$24** every day.", "18") == 0.0


def test_gsm8k_tag_wins_over_text():
    assert get_checker("gsm8k")("I first thought 18.\n<answer>17</answer>", "18") == 0.0


# ---------- countdown ----------
def cd_meta(numbers, target):
    return json.dumps({"question": "q", "answer": "",
                       "metadata": {"source_dataset": "countdown", "source_index": 0,
                                    "numbers": numbers, "target": target, "expression": "",
                                    "difficulty": {}}})


def test_countdown_any_valid_expression():
    meta, check = cd_meta([2, 3, 4], 14), get_checker("countdown")
    assert check(tag("3*4+2"), "", meta) == 1.0
    assert check(tag("2 + 3 * 4"), "", meta) == 1.0
    assert check(tag("2*3+4"), "", meta) == 0.0       # wrong value
    assert check(tag("14"), "", meta) == 0.0          # no numbers used
    assert check(tag("3*4+2+0"), "", meta) == 0.0     # number not given


def test_countdown_untagged_with_equals():
    meta, check = cd_meta([2, 3, 4], 14), get_checker("countdown")
    assert check("Let me try.\n14 = 3 * 4 + 2", "", meta) == 1.0
    assert check("So 3 × 4 + 2 = 14", "", meta) == 1.0
    assert check("16 = 19 * 46 / 31 - 20", "", meta) == 0.0   # real wrong base output


# ---------- logic (knights & knaves) ----------
def kk(seed=7):
    import reasoning_gym as rg
    e = rg.create_dataset("knights_knaves", size=1, seed=seed)[0]
    return e, json.dumps(e, default=str)


def test_logic_gold_and_wrong():
    e, meta = kk()
    check = get_checker("logic")
    assert check(tag(e["answer"]), e["answer"], meta) == 1.0
    assert check(tag("definitely wrong"), e["answer"], meta) == 0.0


def test_logic_untagged_list_style():
    e, meta = kk()
    md = e["metadata"]
    t = md["knight_knave_terms"]
    sol = dict(zip(md["names"], [t["knight"] if s else t["knave"] for s in md["solution"]]))
    lines = "\n".join(f"- {n} is a {r}," for n, r in sol.items())
    assert get_checker("logic")(f"Thus, the final answer is:\n{lines}", e["answer"], meta) == 1.0


def test_logic_template_echo_is_not_an_answer():
    e, meta = kk()
    n1, n2 = e["metadata"]["names"]
    t = e["metadata"]["knight_knave_terms"]
    echo = f"Final Answer: {n1} is a {t['knight']}/{t['knave']}, and {n2} is a {t['knight']}/{t['knave']}."
    assert get_checker("logic")(echo, e["answer"], meta) == 0.0


def test_logic_uses_last_claim():
    e, meta = kk()
    md = e["metadata"]
    t = md["knight_knave_terms"]
    right = [t["knight"] if s else t["knave"] for s in md["solution"]]
    wrong = [t["knave"] if s else t["knight"] for s in md["solution"]]
    n1, n2 = md["names"]
    text = (f"Maybe {n1} is a {wrong[0]} and {n2} is a {wrong[1]}. No wait. "
            f"{n1} is a {right[0]} and {n2} is a {right[1]}.")
    assert get_checker("logic")(text, e["answer"], meta) == 1.0


# ---------- sudoku ----------
def sudoku():
    import reasoning_gym as rg
    e = rg.create_dataset("mini_sudoku", size=1, seed=7)[0]
    return e, json.dumps(e, default=str)


def test_sudoku_tagged_and_wrong():
    e, meta = sudoku()
    check = get_checker("sudoku")
    assert check(tag(e["answer"]), e["answer"], meta) == 1.0
    assert check(tag("1 2 3 4\n1 2 3 4\n1 2 3 4\n1 2 3 4"), e["answer"], meta) == 0.0


def test_sudoku_untagged_codeblock_and_separators():
    e, meta = sudoku()
    grid = "\n".join("| " + " | ".join(r.split()) + " |" for r in e["answer"].splitlines())
    text = f"Here is the solution:\n```\n{grid}\n```\nAll rows are unique."
    assert get_checker("sudoku")(text, e["answer"], meta) == 1.0


def test_sudoku_incomplete_grid_fails():
    e, meta = sudoku()
    real = "```\n4 2 3\n_ 3 1\n3 1 2\n1 2 4\n```"   # real base output
    assert get_checker("sudoku")(real, e["answer"], meta) == 0.0


# ---------- code ----------
CODE_META = json.dumps({"tests": ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"], "setup": ""})
GOOD = "```python\ndef add(a, b):\n    return a + b\n```"


def test_code_tagged_pass():
    assert get_checker("code")(tag(GOOD), "", CODE_META) == 1.0


def test_code_tagged_without_fence():
    assert get_checker("code")(tag("def add(a, b):\n    return a + b"), "", CODE_META) == 1.0


def test_code_untagged_block():
    assert get_checker("code")(f"Here you go:\n{GOOD}\nThis adds numbers.", "", CODE_META) == 1.0


def test_code_untagged_plain_text_fails():
    assert get_checker("code")("def add(a, b): return a + b", "", CODE_META) == 0.0


def test_code_wrong():
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    assert get_checker("code")(tag(bad), "", CODE_META) == 0.0


def test_code_syntax_error():
    assert get_checker("code")(tag("```python\ndef add(a, b) return\n```"), "", CODE_META) == 0.0


def test_code_infinite_loop_times_out():
    c = tag("```python\ndef add(a, b):\n    while True: pass\n```")
    assert get_checker("code")(c, "", CODE_META) == 0.0


def test_code_setup_used():
    meta = json.dumps({"tests": ["assert f() == 3"], "setup": "X = 3"})
    assert get_checker("code")(tag("def f():\n    return X"), "", meta) == 1.0