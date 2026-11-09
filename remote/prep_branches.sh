#!/usr/bin/env bash
# Run ONCE on airavat, from the repo root, before setting up the remote machine.
# Creates three branches without touching your working copy or current branch:
#   data-v2         exact data/ splits used by v2 (+ SHA256SUMS), so the remote machine scores the same questions
#   queue-remote    jobs.txt: the remote machine's to-do list (you edit it, the runner reads it)
#   results-remote  the runner publishes STATUS.md, job states, logs and results here
# Also writes remote/requirements-lock.txt (exact package versions of this venv); commit and push it yourself.
#   bash remote/prep_branches.sh            # refuses to overwrite existing branches
#   FORCE=1 bash remote/prep_branches.sh    # replace data-v2 (e.g. after rebuilding data)
set -euo pipefail
cd "$(dirname "$0")/.."
# airavat has no GitHub remote: the branches are created locally, and the usual laptop sync
#   git fetch server && git push origin 'refs/remotes/server/*:refs/heads/*'
# carries them to GitHub. If a remote named $REMOTE exists, they are pushed to it directly instead.
GD=$(git rev-parse --absolute-git-dir)
REMOTE=${REMOTE:-origin}
if git remote get-url "$REMOTE" > /dev/null 2>&1; then MODE=push; git fetch -q "$REMOTE"; else MODE=local; fi
echo "mode: $MODE"

exists() {
  if [ "$MODE" = push ]; then git ls-remote --exit-code --heads "$REMOTE" "$1" > /dev/null 2>&1
  else git show-ref --verify --quiet "refs/heads/$1"; fi
}

push_tree() {   # push_tree <branch> <dir> <message>: commit the contents of <dir> as the whole branch
  local branch=$1 dir=$2 msg=$3 idx tree commit
  if exists "$branch" && [ "${FORCE:-0}" != 1 ]; then
    echo "  $branch exists on origin, kept (FORCE=1 to replace)"; return
  fi
  idx=$(mktemp); rm -f "$idx"
  GIT_DIR=$GD GIT_INDEX_FILE=$idx git -C "$dir" --work-tree=. add -A -f .
  tree=$(GIT_DIR=$GD GIT_INDEX_FILE=$idx git write-tree)
  commit=$(git commit-tree "$tree" -m "$msg")
  rm -f "$idx"
  if [ "$MODE" = push ]; then
    git push -q -f "$REMOTE" "$commit:refs/heads/$branch"; echo "  pushed $branch ($commit)"
  else
    git update-ref "refs/heads/$branch" "$commit"; echo "  created local branch $branch ($commit)"
  fi
}

echo "== data-v2"
for t in gsm8k logic logic_hard code; do
  [ -f "data/$t/test.jsonl" ] || { echo "STOP: data/$t/test.jsonl missing"; exit 1; }
done
[ -f data/gsm8k/train_filtered.jsonl ] || { echo "STOP: run the filters first"; exit 1; }
tmp=$(mktemp -d)
mkdir -p "$tmp/data"
for t in gsm8k logic logic_hard code; do
  mkdir -p "$tmp/data/$t"
  cp data/$t/test.jsonl data/$t/train.jsonl "$tmp/data/$t/"
  [ -f data/$t/train_filtered.jsonl ] && cp data/$t/train_filtered* "$tmp/data/$t/"
done
rm -f "$tmp"/data/*/filter_try*
(cd "$tmp" && find data -type f | sort | xargs sha256sum > SHA256SUMS)
echo "  $(wc -l < "$tmp/SHA256SUMS") files, $(du -sh "$tmp" | cut -f1)"
push_tree data-v2 "$tmp" "data snapshot for v2 ($(git rev-parse --short HEAD))"
rm -rf "$tmp"

echo "== queue-remote"
tmp=$(mktemp -d)
cat > "$tmp/jobs.txt" <<'EOF'
# Remote job queue. One job per line:   id | gpus | ref | after | command
#   gpus: 1 = one GPU, 0 = CPU only.  ref: branch, tag or commit.  after: "-" or job ids that must finish OK first.
#   Commands run with bash in the repo root. Add "cancel <id>" to stop a job. Never reuse an id.
# Example:
# smoke | 1 | navya-1.5B | - | python -m src.train.grpo --task code --out checkpoints/pilot/smoke --set grpo.max_steps=3 grpo.save_steps=100000
EOF
push_tree queue-remote "$tmp" "empty job queue"
rm -rf "$tmp"

echo "== results-remote"
tmp=$(mktemp -d)
printf "# Remote results\n\nWritten by remote/tick.py every 10 minutes. Start with STATUS.md.\n" > "$tmp/README.md"
push_tree results-remote "$tmp" "results branch"
rm -rf "$tmp"

echo "== requirements lock"
pip freeze --exclude-editable | grep -v -E "^(-e |#)" > remote/requirements-lock.txt
echo "  remote/requirements-lock.txt: $(wc -l < remote/requirements-lock.txt) packages, torch $(grep -i '^torch==' remote/requirements-lock.txt)"
echo "Now: git add remote && git commit -m 'remote runner kit'"
[ "$MODE" = local ] && echo "Then on the laptop: git fetch server && git push origin 'refs/remotes/server/*:refs/heads/*'"
true