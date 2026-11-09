#!/usr/bin/env bash
# Cron entry point (every 10 min). Updates the runner code from the code branch, then runs one tick.
# Settings come from $RLM_BIG/remote.env, written by setup.sh.
set -uo pipefail
BIG=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)     # <BIG>/repo/remote/tick.sh -> <BIG>
source "$BIG/remote.env"
export PATH="/usr/local/bin:/usr/bin:/bin:$(dirname "$(command -v nvidia-smi 2>/dev/null || echo /usr/bin/x)")"
LOG="$BIG/tick.log"
[ -f "$LOG" ] && [ "$(stat -c %s "$LOG")" -gt 5000000 ] && mv "$LOG" "$LOG.old"
exec 9>"$BIG/tick.lock"
flock -n 9 || { echo "[$(date)] previous tick still running" >> "$LOG"; exit 0; }
{
  cd "$BIG/repo" || exit 1
  if git fetch -q origin "$RLM_CODE_BRANCH" 2>/dev/null; then
    git checkout -q --detach "origin/$RLM_CODE_BRANCH" 2>/dev/null || echo "[$(date)] could not update runner code"
  fi
  "$RLM_VENV/bin/python" "$BIG/repo/remote/tick.py"
} >> "$LOG" 2>&1
