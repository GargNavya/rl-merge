# Verifying the pipeline

Do this once before the full runs, and again after any change to data, checkers, training or merging.
It is the project doc's **Gate 0**: checkers correct, merge code correct, RRR computed correctly.

Total time: ~1.5 hours, mostly the pilot (step 6).

| # | What is checked | Command | Time |
|---|---|---|---|
| 1 | Unit tests (checkers, merge math, RRR math) | `python -m pytest -q` | 1 min |
| 2 | Data splits + checkers on gold and junk answers | `python -m src.eval.verify data` | 1-2 min |
| 3 | Checkers on real model replies, by eye | `python -m src.eval.show_samples ...` | 20 min of reading |
| 4 | Evaluation gives the same answer twice | `python -m src.eval.verify eval` | 2 min |
| 5 | Training actually learns (short run) | watch the log of step 6 | - |
| 6 | Whole pipeline at small scale | `bash scripts/pilot.sh` | ~45 min |

Every `verify` command prints `[PASS]`, `[FAIL]` or `[INFO]` per check and ends with
`ALL CHECKS PASSED` or a list of failures. `[INFO]` lines are numbers to look at, never failures.

---

## 1. Unit tests

```bash
python -m pytest -q
```

**Pass:** `37 passed`.
These use hand-made inputs: right answers, wrong answers, answers without tags, real base-model
replies, code that loops forever. Also the merge formulas and the RRR formula on small known numbers.

**If it fails:** don't continue. Paste the failure.

## 2. Data and checkers on known answers

The code split now stores MBPP's reference solution, so rebuild it once:

```bash
rm -rf data/code
python -m src.eval.verify data
```

For every task it checks:

| Check | Why |
|---|---|
| train/test sizes match config | splits built as intended |
| ids unique, no repeated test questions | no double counting |
| train/test overlap = 0 | test must be unseen, or gains are memorisation |
| gold answer inside `<answer>` scores correct (100%) | checker accepts right answers |
| code: reference solutions pass the tests (>= 95%) | test runner works; a few MBPP references are known broken |
| junk answers (`""`, `I don't know`, empty tags) score 0 | checker rejects wrong answers |
| INFO: gold WITHOUT tags | how often the fallback reader finds an untagged answer |

**If it fails:** overlap or size problems usually mean a stale `data/<task>/` folder. Delete it and rerun.
Gold failing means a checker bug. Paste the output.

## 3. Checkers on real replies, by eye

Steps 1-2 use clean answers. Real replies are messy. Read them:

```bash
python -m src.eval.show_samples --label base --task gsm8k --only wrong --n 10
python -m src.eval.show_samples --label base --task gsm8k --only right --n 10
```

Repeat for `logic` and `code`. Each example shows the question, the reply, `EXTRACTED` (what the
checker read) and `GOLD`.

| You see | Meaning | Severity |
|---|---|---|
| marked WRONG but the reply is right | checker too strict: misses answers | medium: lowers scores |
| marked RIGHT but the reply is wrong | checker too lenient: false pass | **high**: RL will learn to exploit it |
| `EXTRACTED: None` but an answer is visible | extraction missed a format | medium |

Write down any mismatch with its ID and send it. One false pass in 20 is worth fixing before training.

## 4. Evaluation is deterministic

```bash
python -m src.eval.verify eval --task gsm8k
```

**Pass:** same settings twice give identical replies (16/16).
Otherwise score differences between models could be noise from the eval itself.
INFO line: batch size 16 vs 1. A few differences here are normal on GPU.

## 5 + 6. Pilot: the whole pipeline, small

```bash
nohup bash scripts/pilot.sh > results/pilot.log 2>&1 &
tail -f results/pilot.log
```

It trains 2 specialists (gsm8k, logic) for 30 steps each, merges them 3 ways, evaluates everything
on 50 questions per task, and computes RRR. Output goes to `checkpoints/pilot/` and `results/pilot/`,
so your real `results/scores.csv` is untouched. Settings: `STEPS=60 N_TEST=100 bash scripts/pilot.sh`.

It runs two checks on the way:

**`verify weights`** (after each specialist):

| Check | Why |
|---|---|
| tensors changed > 0 | training did something |
| only LoRA target modules changed | nothing else was touched by accident |
| groups with no reward variation well below 90% | GRPO had a learning signal |
| saved full model gives same top token as base + adapter (>= 95%) | the file we merge = the model we trained |

**`verify merge`** (before merging for real):

| Check | Why |
|---|---|
| average of one specialist == that specialist | merge code changes nothing it shouldn't |
| average of a specialist with itself == itself | same |
| TIES with keep-everything on one specialist == itself | TIES trim/sign steps are correct |
| task arithmetic with scale 0 == base | scaling is correct |
| merged model differs from each specialist | the merge actually combined them |
| one tensor checked against a hand computation | the formula is right on real weights |

**Pass:** script reaches `### pilot done`, every check block says `ALL CHECKS PASSED`, and the
two RRR tables print. With only 30 steps, gains may be below 0.02 and RRR shows `n/a`. That is
fine: the pilot tests the plumbing, not the science.

Also look at the training log:
- `rewards/<task>_correct/mean` should drift up over the 30 steps (noisy)
- `completions/clipped_ratio` mostly under 0.2
- no `nan` in `loss`

Afterwards, read a few pilot replies:
```bash
python -m src.eval.show_samples --results results/pilot --label spec-gsm8k --task gsm8k --only right
```

## When everything passes

Commit, then do the real runs:

```bash
git add -A && git commit -m "pipeline verified"
bash scripts/eval_base.sh
nohup bash scripts/train_all.sh > results/train_all.log 2>&1 &
bash scripts/merge_and_eval.sh      # after training finishes
```

Rerun `verify weights` on each real specialist and `verify merge` on the real set. They are cheap.
