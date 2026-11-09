#!/usr/bin/env bash
# One-time setup of the remote runner. Run on the remote machine after cloning the repo into <BIG>/repo:
#   bash <BIG>/repo/remote/setup.sh
# Safe to re-run: finished steps are skipped. Every step prints "== n/10 ..."; stop at the first error.
set -euo pipefail
BIG=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
REPO=$BIG/repo
CODE_BRANCH=${CODE_BRANCH:-navya-1.5B}
step() { echo; echo "== $*"; }
ask() { local v; read -r -p "$1 [$2]: " v; echo "${v:-$2}"; }

step "1/10 machine check"
command -v nvidia-smi > /dev/null || { echo "STOP: no nvidia-smi"; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
DRIVER_CUDA=$(nvidia-smi | grep -o "CUDA Version: [0-9.]*" | grep -o "[0-9.]*$")
echo "driver supports CUDA $DRIVER_CUDA; $(ldd --version | head -1)"
FREE_GB=$(df -BG --output=avail "$BIG" | tail -1 | tr -dc 0-9)
echo "free disk in $BIG: ${FREE_GB} GB"; [ "$FREE_GB" -ge 150 ] || echo "WARNING: under 150 GB free"
command -v crontab > /dev/null || { echo "STOP: no crontab command"; exit 1; }
command -v flock > /dev/null || { echo "STOP: no flock command"; exit 1; }

step "2/10 settings"
ALL=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
if [ -f "$BIG/remote.env" ]; then
  echo "using existing $BIG/remote.env"; source "$BIG/remote.env"
else
  GPUS=$(ask "GPUs the runner may use" "$ALL")
  JPG=$(ask "jobs per GPU (2 only for >= 40 GB GPUs)" "1")
  NAME=$(ask "git author name" "Navya Garg")
  EMAIL=$(ask "git author email (same as on GitHub)" "")
  cat > "$BIG/remote.env" <<EOF
export RLM_BIG="$BIG"
export RLM_GPUS="$GPUS"
export RLM_JOBS_PER_GPU="$JPG"
export RLM_VENV="$BIG/venv"
export RLM_CODE_BRANCH="$CODE_BRANCH"
export RLM_GIT_NAME="$NAME"
export RLM_GIT_EMAIL="$EMAIL"
export HF_HOME="$BIG/hf_cache"
export PIP_CACHE_DIR="$BIG/pip_cache"
EOF
  source "$BIG/remote.env"
fi
cat "$BIG/remote.env"

step "3/10 git: identity, key, branches"
KEY=$BIG/.ssh/deploy_key
git -C "$REPO" config user.name "$RLM_GIT_NAME"
git -C "$REPO" config user.email "$RLM_GIT_EMAIL"
git -C "$REPO" config core.sshCommand "ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
git -C "$REPO" config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"
git -C "$REPO" fetch -q origin
for b in "$CODE_BRANCH" data-v2 queue-remote results-remote; do
  git -C "$REPO" rev-parse -q --verify "origin/$b" > /dev/null || { echo "STOP: branch $b missing on GitHub (run prep_branches.sh on airavat)"; exit 1; }
done
git -C "$REPO" checkout -q --detach "origin/$CODE_BRANCH"
echo "ok: $(git -C "$REPO" log -1 --format='%h %s')"

step "4/10 python venv"
if [ ! -x "$RLM_VENV/bin/python" ]; then
  PY=""
  for p in python3.13 python3.12 python3.11 python3.10 python3; do command -v $p > /dev/null && { PY=$p; break; }; done
  [ -n "$PY" ] || { echo "STOP: no python3"; exit 1; }
  echo "using $($PY --version)"
  if ! $PY -m venv "$RLM_VENV" 2> /dev/null; then
    command -v conda > /dev/null || { echo "STOP: '$PY -m venv' failed and no conda"; exit 1; }
    conda create -y -q -p "$BIG/pyenv" python=3.13 > /dev/null
    "$BIG/pyenv/bin/python" -m venv "$RLM_VENV"
  fi
fi
source "$RLM_VENV/bin/activate"
python -m pip install -q -U pip setuptools wheel

step "5/10 torch for this driver"
LOCK=$REPO/remote/requirements-lock.txt
TORCH_VER=${TORCH_VER:-$(grep -i "^torch==" "$LOCK" | cut -d= -f3 | cut -d+ -f1)}
[ -n "$TORCH_VER" ] || { echo "STOP: no 'torch==' line in requirements-lock.txt; rerun as TORCH_VER=<version> bash setup.sh"; exit 1; }
if python -c "import torch, sys; sys.exit(0 if torch.__version__.split('+')[0] == '$TORCH_VER' and torch.cuda.is_available() else 1)" 2> /dev/null; then
  echo "torch $TORCH_VER already installed"
else
  DMAJ=${DRIVER_CUDA%%.*}; DMIN=${DRIVER_CUDA#*.}; DNUM=$((DMAJ * 10 + ${DMIN%%.*}))
  OK=""
  for cu in 130 128 126 124 121 118; do
    [ "$cu" -le "$DNUM" ] || continue
    echo "trying torch==$TORCH_VER from cu$cu"
    if pip install -q "torch==$TORCH_VER" --index-url "https://download.pytorch.org/whl/cu$cu"; then OK=cu$cu; break; fi
  done
  [ -n "$OK" ] || { echo "STOP: no torch $TORCH_VER build for driver CUDA $DRIVER_CUDA"; exit 1; }
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count(), 'ok' if torch.cuda.is_available() else 'NO CUDA')"
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "STOP: torch cannot see the GPU"; exit 1; }

step "6/10 other packages (same versions as airavat)"
grep -v -i -E "^(torch|torchvision|torchaudio|triton|nvidia-)" "$LOCK" > "$BIG/lock-no-torch.txt"
if ! pip install -q -r "$BIG/lock-no-torch.txt"; then
  echo "WARNING: exact versions failed; installing requirements.txt instead (note it in the log)"
  pip install -q -r "$REPO/requirements.txt"
fi
python -c "import transformers, trl, peft, reasoning_gym; print('transformers', transformers.__version__, 'trl', trl.__version__, 'peft', peft.__version__)"

step "7/10 base model into $HF_HOME"
MODEL=$(python -c "import yaml; print(yaml.safe_load(open('$REPO/configs/base.yaml'))['model']['name'])")
python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL'))"

step "8/10 data snapshot + results branch"
mkdir -p "$BIG/shared/data" "$BIG/shared/checkpoints" "$BIG/shared/results" "$BIG/state" "$BIG/logs" "$BIG/wt"
git -C "$REPO" archive origin/data-v2 | tar -x -C "$BIG/shared"
(cd "$BIG/shared" && sha256sum -c --quiet SHA256SUMS) && echo "data checksums OK: $(wc -l < "$BIG/shared/SHA256SUMS") files"
if [ ! -d "$BIG/out" ]; then
  git -C "$REPO" worktree add -q -B results-remote "$BIG/out" origin/results-remote
fi
git -C "$BIG/out" config user.name "$RLM_GIT_NAME"; git -C "$BIG/out" config user.email "$RLM_GIT_EMAIL"

step "9/10 tests in a job worktree"
WTP=$(cd "$REPO/remote" && RLM_BIG=$BIG python -c "import tick; print(tick.worktree(tick.resolve('$CODE_BRANCH')))")
echo "worktree: $WTP"
(cd "$WTP" && python -m pytest -q 2>&1 | tail -3)
(cd "$WTP" && python -m src.eval.verify data --tasks gsm8k logic logic_hard code 2>&1 | tail -4)

step "10/10 cron + first tick"
chmod +x "$REPO/remote/tick.sh"
LINE="*/10 * * * * $REPO/remote/tick.sh  # rlm-runner"
(crontab -l 2> /dev/null | grep -v "rlm-runner"; echo "$LINE") | crontab -
crontab -l | grep rlm-runner
"$REPO/remote/tick.sh"; tail -3 "$BIG/tick.log"
echo
echo "DONE. Check STATUS.md on branch results-remote on GitHub. Pause new jobs: touch $BIG/PAUSE"
