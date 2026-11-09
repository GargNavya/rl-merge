# Remote runner: jobs run on a second machine from a queue on GitHub

After setup, jobs are added and checked through GitHub, so the machine's owner doesn't have to be disturbed
for routine work. He stays in control of his machine at all times (see "Notes for the machine's owner").

| branch | written by | holds |
|---|---|---|
| `navya-1.5B` | Navya (airavat) | code; the runner updates itself from here |
| `data-v2` | `prep_branches.sh`, once | exact `data/` splits + `SHA256SUMS` |
| `queue-remote` | Navya, anywhere (GitHub web editor is fine) | `jobs.txt` |
| `results-remote` | the runner only | `STATUS.md`, `jobs/`, `logs/` (tails), `results/`, `checkpoints_info/` |

Cron runs `remote/tick.sh` every 10 minutes. Each tick: read `jobs.txt` → mark finished / failed / lost jobs →
apply `cancel <id>` lines → start waiting jobs on free GPUs (each in its own worktree at the job's commit) →
push `STATUS.md` and results. Commits carry Navya's git name and email; pushes use a deploy key (belongs to the repo,
not to any person's GitHub account). Checkpoints stay on the remote disk; `run_info.json` / `merge_info.json` are copied.

On the remote machine (`<BIG>` = the folder chosen at setup):
`repo/` (runner code), `venv/`, `hf_cache/`, `shared/{data,checkpoints,results}`, `wt/<commit>/` (job code),
`state/`, `logs/<job>.log` (full logs), `out/` (results branch), `remote.env` (settings), `tick.log`, `PAUSE` (optional).

---

## A. Before the session (airavat, ~15 min)

airavat has no GitHub remote, so the branches are made there and your laptop carries them to GitHub.
```bash
# on airavat
source /22tdata/$USER/oml_nav_rlmerge/env.sh; cd ~/novel/oml/RL-merge_analysis/rl-merge_airavat
# copy the remote/ folder into the repo first, then:
bash remote/prep_branches.sh                         # local branches data-v2, queue-remote, results-remote + requirements-lock.txt
git add remote && git commit -m "remote runner kit"
git tag v2-code <commit v2 seed 42 trained with>

# on the laptop (inside rl-merge)
git fetch server && git push origin 'refs/remotes/server/*:refs/heads/*'
git fetch server tag v2-code && git push origin v2-code
```
The commit is in `checkpoints/specialists/v2/gsm8k/seed42/run_info.json` (`git_commit`). If it ends in `-dirty`,
tag the commit after it that contains those changes.

Bring: your laptop logged in to GitHub (to add the deploy key and edit `jobs.txt`).

## B. The session (60 min, together at his machine)

In the steps below, "you" means Navya.

| min | do | done when |
|---|---|---|
| 0-5 | Paste the machine check (below); pick `BIG` = a folder on a disk with >= 200 GB free | output sent to Claude |
| 5-10 | Make the deploy key; add it on GitHub (Settings → Deploy keys → Add, tick **Allow write access**) | key listed on GitHub |
| 10-15 | Clone the repo into `$BIG/repo` | `git log -1` shows your last commit |
| 15-40 | `bash $BIG/repo/remote/setup.sh` (answers: GPUs, jobs per GPU, your name, your GitHub email) | ends with `DONE` |
| 40-50 | Add the `smoke` job to `jobs.txt` on GitHub; run one tick by hand | `STATUS.md` on GitHub shows `smoke` running, then done |
| 50-55 | Paste `jobs_v2_seeds.txt` into `jobs.txt`; one more tick | 4+ jobs running |
| 55-60 | Go through the notes for the machine's owner (below) together | he knows how to pause or stop it |

**Machine check** (minute 0):
```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv; ldd --version | head -1
head -2 /etc/os-release; python3 --version; nproc; free -g; df -h ~ /tmp /data* /scratch* 2>/dev/null
git --version; command -v crontab flock conda tmux; ssh -T git@github.com 2>&1 | head -1
```

**Deploy key and clone** (minutes 5-15):
```bash
BIG=/path/with/space/rlmerge; mkdir -p $BIG/.ssh
ssh-keygen -t ed25519 -N "" -C rlm-runner -f $BIG/.ssh/deploy_key; cat $BIG/.ssh/deploy_key.pub
# add the key on GitHub, then:
GIT_SSH_COMMAND="ssh -i $BIG/.ssh/deploy_key -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new" \
  git clone -b navya-1.5B git@github.com:<you>/<repo>.git $BIG/repo
```
If port 22 is blocked (clone hangs), use `ssh://git@ssh.github.com:443/<you>/<repo>.git` instead.

**Smoke job** (minute 40), add to `jobs.txt` on GitHub, then run `$BIG/repo/remote/tick.sh; tail -3 $BIG/tick.log`:
```
smoke | 1 | navya-1.5B | - | python -m src.train.grpo --task code --out checkpoints/pilot/smoke --set grpo.max_steps=3 grpo.save_steps=100000 && python -m src.eval.evaluate --model checkpoints/pilot/smoke/merged --label smoke --tasks code --set eval.n_test=16 --results results/pilot_smoke
```

**Notes for the machine's owner**:
- **What it uses:** only the GPUs chosen at setup, the folder `<BIG>`, and one cron line that runs every 10 minutes.
  It never uses a GPU on which someone else has more than 2 GB in use, so your own work takes priority.
- **Need the GPUs for a while?** `touch <BIG>/PAUSE`. No new jobs start; running ones finish (up to a few hours).
  `rm <BIG>/PAUSE` lets it continue. A quick message to Navya helps her plan around it.
- **Need a GPU right now?** It's fine to stop a running job (`kill <pid>`, shown by `nvidia-smi`). The runner marks it
  as failed and Navya re-queues it later; nothing breaks.
- **Restarting the machine** is fine too. Jobs running at that moment are marked as lost and re-queued later;
  a heads-up is appreciated.
- **Stop everything for good:** `crontab -l | grep -v rlm-runner | crontab -`, then stop any running `python -m src...`
  jobs. Deleting `<BIG>` removes every file it made.
- Please leave the files under `<BIG>` as they are while it runs, so results stay consistent.

## C. Afterwards (from anywhere)

- **Watch:** `STATUS.md` on branch `results-remote` (GitHub web), or `git fetch origin && git show origin/results-remote:STATUS.md`.
- **Add work:** append lines to `jobs.txt` on `queue-remote`. Never reuse an id; a finished id is never rerun.
- **Stop a job:** add `cancel <id>`. **Retry:** add the line again with a new id (training refuses a non-empty
  output folder, so first point it elsewhere with `--out`, or add `--overwrite`).
- **Change code:** push to `navya-1.5B`, then queue jobs with that branch or a new tag. Running jobs are not affected.
- **Get results on airavat:** `git fetch origin && git archive origin/results-remote results | tar -x -C remote_results/`
  (generations are `.jsonl.gz`; `gunzip` them for `show_samples --results`).
- **Problems:** `STATUS.md` → Problems and Finished (status `failed` / `lost`); the job's log tail is in `logs/<id>.txt`.

The deploy key can push to any branch of the repo; only `results-remote` is ever pushed by the runner.