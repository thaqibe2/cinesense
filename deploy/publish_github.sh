#!/usr/bin/env bash
# Publish CineSense to GitHub. Run from the cinesense/ folder.
# Prereq: git installed and you are logged in to git (https) or have the GitHub CLI `gh`.
set -e
USER="${1:?Usage: ./deploy/publish_github.sh YOUR_GITHUB_USERNAME [repo_name]}"
REPO="${2:-cinesense}"
cd "$(dirname "$0")/.."
git init
git add .
git commit -m "CineSense: ML Numeric + NLP movie rating fusion project"
git branch -M main
# Option A: GitHub CLI (creates the repo and pushes in one step)
if command -v gh >/dev/null 2>&1; then
  gh repo create "$USER/$REPO" --public --source=. --remote=origin --push
  gh api -X PUT "repos/$USER/$REPO/collaborators/jasminh" >/dev/null && echo "invited jasminh"
  gh api -X PUT "repos/$USER/$REPO/collaborators/bkuehnis" >/dev/null && echo "invited bkuehnis"
else
  echo ">> No gh CLI. First create an EMPTY repo named '$REPO' at https://github.com/new"
  echo ">> then run:"
  echo "   git remote add origin https://github.com/$USER/$REPO.git"
  echo "   git push -u origin main"
  echo ">> Then add collaborators jasminh and bkuehnis under Settings > Collaborators."
fi
