#!/usr/bin/env python3
"""One tick of the remote job runner. Cron calls remote/tick.sh every 10 minutes; nothing stays running.

Each tick:
  1. reads the queue: jobs.txt on branch queue-remote (edited by Navya, e.g. in the GitHub web editor)
  2. updates job states: finished / failed / lost (process gone without an exit code, e.g. reboot)
  3. cancels jobs listed as `cancel <id>`
  4. starts pending jobs on free GPUs, each in its own git worktree at the job's commit
  5. publishes STATUS.md, job states, log tails, results and run_info files to branch results-remote

Queue line:   id | gpus | ref | after | command
  gpus   1 = one GPU, 0 = CPU only
  ref    branch, tag or commit of the code (resolved at start; the exact commit is recorded)
  after  "-" or comma-separated job ids that must have finished OK first
  command runs with bash in the worktree root; $RLM_GPU holds the GPU index
Other lines:  "# comment", "cancel <id>"
Pause: create the file $RLM_BIG/PAUSE (no new jobs start; running ones continue).

Only the Python standard library is used.
"""
import datetime as dt
import gzip
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
from pathlib import Path

BIG = Path(os.environ["RLM_BIG"])
REPO, WT, SHARED, STATE, LOGS, OUT = (BIG / x for x in ("repo", "wt", "shared", "state", "logs", "out"))
GPUS = [g.strip() for g in os.environ.get("RLM_GPUS", "0").split(",") if g.strip()]
JOBS_PER_GPU = int(os.environ.get("RLM_JOBS_PER_GPU", "1"))
BUSY_MB = int(os.environ.get("RLM_BUSY_MB", "2000"))       # a GPU we don't use, with more memory in use, belongs to someone else
MAX_CPU_JOBS = int(os.environ.get("RLM_MAX_CPU_JOBS", "2"))
MIN_FREE_GB = int(os.environ.get("RLM_MIN_FREE_GB", "50"))
VENV = Path(os.environ.get("RLM_VENV", BIG / "venv"))
QUEUE_REF = "origin/queue-remote"
SHARED_DIRS = ("data", "checkpoints", "results")
NOW = dt.datetime.now().replace(microsecond=0)


def run(cmd, cwd=REPO, check=True):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))}: {r.stderr.strip()[-500:]}")
    return r.stdout


def git(*args, cwd=REPO, check=True):
    return run(["git", *args], cwd=cwd, check=check)


# ------------------------------------------------------------------ queue
def parse_queue(text):
    jobs, cancels, errors = {}, set(), []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("cancel "):
            cancels.add(line.split(None, 1)[1].strip())
            continue
        parts = [p.strip() for p in line.split("|", 4)]
        if len(parts) != 5:
            errors.append(f"line {n}: needs 5 fields separated by |")
            continue
        jid, gpus, ref, after, cmd = parts
        if not jid.replace("-", "").replace("_", "").replace(".", "").isalnum():
            errors.append(f"line {n}: bad id {jid!r} (letters, digits, - _ . only)")
            continue
        if jid in jobs:
            errors.append(f"line {n}: duplicate id {jid}")
            continue
        if gpus not in ("0", "1"):
            errors.append(f"line {n}: gpus must be 0 or 1")
            continue
        jobs[jid] = {"id": jid, "gpus": int(gpus), "ref": ref, "cmd": cmd, "line": n,
                     "after": [] if after in ("-", "") else [a.strip() for a in after.split(",")]}
    return jobs, cancels, errors


# ------------------------------------------------------------------ state
def load_states():
    STATE.mkdir(parents=True, exist_ok=True)
    return {p.stem: json.loads(p.read_text()) for p in STATE.glob("*.json")}


def save_state(s):
    (STATE / f"{s['id']}.json").write_text(json.dumps(s, indent=1))


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def update_running(states):
    for s in states.values():
        if s["status"] != "running":
            continue
        exitf = STATE / f"{s['id']}.exit"
        if exitf.exists():
            code = int(exitf.read_text().strip() or 1)
            s.update(status="done" if code == 0 else "failed", exit=code, end=str(NOW))
            save_state(s)
        elif not alive(s["pid"]):
            s.update(status="lost", end=str(NOW), note="process gone without exit code (reboot or kill?)")
            save_state(s)


def cancel(states, jobs, cancels):
    for jid in cancels:
        s = states.get(jid)
        if s and s["status"] == "running":
            try:
                os.killpg(s["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
            s.update(status="cancelled", end=str(NOW))
            save_state(s)
        elif s is None and jid in jobs:
            states[jid] = {"id": jid, "status": "cancelled", "end": str(NOW), "cmd": jobs[jid]["cmd"]}
            save_state(states[jid])


# ------------------------------------------------------------------ resources
def gpu_info():
    try:
        out = run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                   "--format=csv,noheader,nounits"], cwd=BIG)
    except Exception as e:
        return {}, str(e)
    info = {}
    for line in out.strip().splitlines():
        i, used, total, util = [x.strip() for x in line.split(",")]
        info[i] = {"used": int(used), "total": int(total), "util": util}
    return info, None


def free_gpus(states, info):
    ours = {g: 0 for g in GPUS}
    for s in states.values():
        if s["status"] == "running" and s.get("gpu") in ours:
            ours[s["gpu"]] += 1
    free = []
    for g in GPUS:
        if g not in info or ours[g] >= JOBS_PER_GPU:
            continue
        if ours[g] == 0 and info[g]["used"] > BUSY_MB:
            continue                                   # someone else is using it
        free += [g] * (JOBS_PER_GPU - ours[g])
    return free, ours


# ------------------------------------------------------------------ launching
def worktree(sha):
    path = WT / sha[:12]
    if not path.exists():
        git("worktree", "add", "--detach", str(path), sha)
        for name in SHARED_DIRS:
            tracked = [f for f in git("ls-files", name, cwd=path).splitlines() if f]
            if tracked:   # hide tracked files under the shared dirs, so git_commit() does not say "-dirty"
                git("update-index", "--skip-worktree", *tracked, cwd=path)
            target = path / name
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.exists():
                shutil.rmtree(target)
            (SHARED / name).mkdir(parents=True, exist_ok=True)
            target.symlink_to(SHARED / name)
    return path


def resolve(ref):
    for cand in (f"origin/{ref}", ref):
        r = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{cand}^{{commit}}"],
                           cwd=REPO, capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout.strip()
    raise RuntimeError(f"unknown ref {ref!r}")


def launch(job, gpu):
    sha = resolve(job["ref"])
    path = worktree(sha)
    LOGS.mkdir(parents=True, exist_ok=True)
    exitf = STATE / f"{job['id']}.exit"
    exitf.unlink(missing_ok=True)
    env = dict(os.environ)
    env.update(PATH=f"{VENV / 'bin'}:{env.get('PATH', '/usr/bin:/bin')}", VIRTUAL_ENV=str(VENV),
               HF_HOME=str(BIG / "hf_cache"), PIP_CACHE_DIR=str(BIG / "pip_cache"),
               PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", RLM_JOB=job["id"])
    if gpu is None:
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env.update(CUDA_VISIBLE_DEVICES=gpu, RLM_GPU=gpu)
    header = f"### job {job['id']}  start {NOW}  host {socket.gethostname()}  gpu {gpu}  commit {sha}\n### {job['cmd']}\n"
    with open(LOGS / f"{job['id']}.log", "w") as log:
        log.write(header)
        log.flush()
        p = subprocess.Popen(["bash", "-c", f"(\n{job['cmd']}\n)\necho $? > '{exitf}'"], cwd=path, env=env,
                             stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
    s = {"id": job["id"], "status": "running", "gpu": gpu, "sha": sha, "ref": job["ref"], "pid": p.pid,
         "start": str(NOW), "cmd": job["cmd"]}
    save_state(s)
    return s


def schedule(jobs, states, info, problems):
    waiting = {}
    if (BIG / "PAUSE").exists():
        return {j: "paused (PAUSE file)" for j in jobs if j not in states}
    free_gb = shutil.disk_usage(BIG).free / 1e9
    if free_gb < MIN_FREE_GB:
        return {j: f"low disk ({free_gb:.0f} GB free)" for j in jobs if j not in states}
    gpus, _ = free_gpus(states, info)
    cpu_running = sum(1 for s in states.values() if s["status"] == "running" and s.get("gpu") is None)
    for jid, job in sorted(jobs.items(), key=lambda kv: kv[1]["line"]):     # queue order
        if jid in states:
            continue
        bad = [a for a in job["after"] if a not in jobs and a not in states]
        if bad:
            waiting[jid] = f"after: unknown job {bad}"
            continue
        not_ok = [a for a in job["after"] if states.get(a, {}).get("status") != "done"]
        if not_ok:
            failed = [a for a in not_ok if states.get(a, {}).get("status") in ("failed", "lost", "cancelled")]
            waiting[jid] = f"blocked: {failed} did not finish OK" if failed else f"waiting for {not_ok}"
            continue
        if job["gpus"] == 1:
            if not gpus:
                waiting[jid] = "no free GPU"
                continue
            gpu = gpus.pop(0)
        else:
            if cpu_running >= MAX_CPU_JOBS:
                waiting[jid] = "CPU job limit"
                continue
            gpu, cpu_running = None, cpu_running + 1
        try:
            states[jid] = launch(job, gpu)
        except Exception as e:
            problems.append(f"{jid}: could not start: {e}")
            waiting[jid] = "start error (see Problems)"
            if gpu is not None:
                gpus.insert(0, gpu)
    return waiting


# ------------------------------------------------------------------ publishing
def tail(path, n=150):
    if not path.exists():
        return ""
    data = path.read_bytes()[-400_000:].decode("utf-8", "replace").replace("\r", "\n")
    lines = [l for l in data.split("\n") if l.strip()]
    return "\n".join(lines[-n:]) + "\n"


def copy_if_changed(src, dst, compress=False):
    data = src.read_bytes()
    if compress:
        data, dst = gzip.compress(data, mtime=0), dst.with_name(dst.name + ".gz")
    if dst.exists() and dst.read_bytes() == data:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)


def publish_files(states):
    for d in ("jobs", "logs"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    for s in states.values():
        (OUT / "jobs" / f"{s['id']}.json").write_text(json.dumps(s, indent=1))
        log = LOGS / f"{s['id']}.log"
        if log.exists():
            (OUT / "logs" / f"{s['id']}.txt").write_text(tail(log))
    res = SHARED / "results"
    if res.exists():
        for f in res.rglob("*"):
            if not f.is_file() or f.stat().st_size > 50_000_000:
                continue
            rel = f.relative_to(res)
            if f.suffix in (".csv", ".json", ".txt", ".md"):
                copy_if_changed(f, OUT / "results" / rel)
            elif f.suffix == ".jsonl":
                copy_if_changed(f, OUT / "results" / rel, compress=True)
    ck = SHARED / "checkpoints"
    if ck.exists():
        for name in ("run_info.json", "merge_info.json"):
            for f in ck.rglob(name):
                if "trainer" not in f.parts:
                    copy_if_changed(f, OUT / "checkpoints_info" / f.relative_to(ck))


def status_md(jobs, states, waiting, info, gpu_err, ours, problems, queue_errors):
    L = [f"# Runner status: {NOW}  (host {socket.gethostname()})", ""]
    du = shutil.disk_usage(BIG)
    L.append(f"Disk free on {BIG}: {du.free / 1e9:.0f} GB of {du.total / 1e9:.0f} GB. "
             f"{'**PAUSED** (PAUSE file present). ' if (BIG / 'PAUSE').exists() else ''}"
             f"GPUs allowed: {','.join(GPUS)}, jobs per GPU: {JOBS_PER_GPU}.")
    L += ["", "## GPUs", "", "| gpu | used / total MB | util % | our jobs |", "|---|---|---|---|"]
    if gpu_err:
        L.append(f"| nvidia-smi failed: {gpu_err} | | | |")
    for g, v in sorted(info.items()):
        mine = [s["id"] for s in states.values() if s["status"] == "running" and s.get("gpu") == g]
        L.append(f"| {g} | {v['used']} / {v['total']} | {v['util']} | {', '.join(mine) or '-'} |")
    L += ["", "## Running", "", "| job | gpu | since | last log line |", "|---|---|---|---|"]
    for s in sorted(states.values(), key=lambda s: s.get("start", "")):
        if s["status"] == "running":
            last = tail(LOGS / f"{s['id']}.log", 1).strip().replace("|", "/")[-160:]
            L.append(f"| {s['id']} | {s.get('gpu') if s.get('gpu') is not None else 'cpu'} | {s['start']} | `{last}` |")
    L += ["", "## Waiting", "", "| job | reason |", "|---|---|"]
    for jid, why in waiting.items():
        L.append(f"| {jid} | {why} |")
    L += ["", "## Finished (newest first)", "", "| job | status | exit | start | end | commit |", "|---|---|---|---|---|---|"]
    done = [s for s in states.values() if s["status"] != "running"]
    for s in sorted(done, key=lambda s: s.get("end", ""), reverse=True)[:40]:
        L.append(f"| {s['id']} | {s['status']} | {s.get('exit', '')} | {s.get('start', '')} | {s.get('end', '')} | "
                 f"{s.get('sha', '')[:8]} |")
    gone = [jid for jid in states if jid not in jobs]
    if gone:
        L += ["", f"Jobs with a state but no longer in jobs.txt: {', '.join(sorted(gone))}"]
    if problems or queue_errors:
        L += ["", "## Problems", ""] + [f"- {p}" for p in queue_errors + problems]
    return "\n".join(L) + "\n"


def push(message):
    git("add", "-A", cwd=OUT)
    if not git("status", "--porcelain", cwd=OUT).strip():
        return "nothing to publish"
    git("commit", "-q", "-m", message, cwd=OUT)
    r = subprocess.run(["git", "push", "-q", "origin", "HEAD:results-remote"], cwd=OUT, capture_output=True, text=True)
    if r.returncode != 0:
        git("pull", "-q", "--rebase", "origin", "results-remote", cwd=OUT, check=False)
        r = subprocess.run(["git", "push", "-q", "origin", "HEAD:results-remote"], cwd=OUT, capture_output=True, text=True)
    return "pushed" if r.returncode == 0 else f"push failed: {r.stderr.strip()[-300:]}"


def main():
    problems = []
    try:
        git("fetch", "-q", "--prune", "origin")
    except Exception as e:
        problems.append(f"git fetch failed (network?): {e}")
    try:
        queue_text = git("show", f"{QUEUE_REF}:jobs.txt")
    except Exception as e:
        queue_text = ""
        problems.append(f"cannot read jobs.txt on queue-remote: {e}")
    jobs, cancels, queue_errors = parse_queue(queue_text)
    states = load_states()
    update_running(states)
    cancel(states, jobs, cancels)
    info, gpu_err = gpu_info()
    waiting = schedule(jobs, states, info, problems)
    _, ours = free_gpus(states, info)
    publish_files(states)
    (OUT / "STATUS.md").write_text(status_md(jobs, states, waiting, info, gpu_err, ours, problems, queue_errors))
    n_run = sum(s["status"] == "running" for s in states.values())
    try:
        result = push(f"tick {NOW:%Y-%m-%d %H:%M}: {n_run} running, {len(waiting)} waiting")
    except Exception as e:
        result = f"publish error: {e}"
    print(f"[tick {NOW}] running {n_run}, waiting {len(waiting)}, {result}"
          + (f", problems: {problems}" if problems else ""))


if __name__ == "__main__":
    sys.exit(main())
