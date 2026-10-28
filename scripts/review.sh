#!/usr/bin/env bash
# Usage: scripts/review.sh "<scope description>"
set -euo pipefail
scope="${1:?usage: review.sh \"<scope>\"}"
repo=/tank/code/mimir-advisor
wt="$repo/.worktrees/review"
out="$repo/.worktrees/reviews/$(date +%F-%H%M)"
mkdir -p "$out"

git -C "$wt" merge --ff-only main
cd "$wt"

prompt() { cat <<P
Review: $scope
Follow AGENTS.md, Reviewer role: do not edit any source, test or config file.
Do not open any REVIEW-*.md file anywhere.
Write findings to REVIEW-$1.md in this directory. One finding per entry: file, line, issue, severity (high/medium/low), suggested fix.
Check: correctness, error handling, test gaps, hidden network calls, data paths that could leak outside the index dir, config not going through Settings/.env.template.
Do not run anything that reaches the network or reads outside this directory.
End with the three findings you would fix first.
P
}

for tool in agy codex; do
  case "$tool" in
    agy)   agy -p "$(prompt agy)" --sandbox ;;
    codex) codex exec --sandbox workspace-write "$(prompt codex)" ;;
  esac
  [ -z "$(git status --porcelain -- . ':!REVIEW-*.md')" ] || { echo "$tool edited files, stopping" >&2; exit 1; }
  mv "REVIEW-$tool.md" "$out/"
done
echo "Reviews in $out"
