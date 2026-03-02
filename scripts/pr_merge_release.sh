#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/stefan/ml4t/libraries/ml4t-backtest"
BRANCH="feature/parity-hardening-coverage"
BASE="main"
TAG="v0.1.0a8"
PR_TITLE="Parity hardening + coverage hardening"
PR_BODY="Includes parity config wiring, contract tests, and result.py error-path coverage improvements."

cd "$REPO_DIR"

echo "==> Preflight"
current_branch="$(git branch --show-current)"
echo "Current branch: ${current_branch}"
git status --short

echo "==> Auth/network checks"
gh auth status
gh api user >/dev/null
git ls-remote --heads origin >/dev/null

echo "==> Ensure feature branch and push"
git checkout "$BRANCH"
git push -u origin "$BRANCH"

echo "==> Create PR (or reuse existing one)"
if gh pr view "$BRANCH" >/dev/null 2>&1; then
  echo "PR already exists for branch ${BRANCH}"
else
  gh pr create \
    --base "$BASE" \
    --head "$BRANCH" \
    --title "$PR_TITLE" \
    --body "$PR_BODY"
fi

echo "==> Watch CI checks"
gh pr checks "$BRANCH" --watch

echo "==> Merge PR"
gh pr merge "$BRANCH" --merge --delete-branch

echo "==> Sync main"
git checkout "$BASE"
git pull origin "$BASE"

echo "==> Create and push release tag ${TAG}"
if git rev-parse "$TAG" >/dev/null 2>&1; then
  echo "Tag ${TAG} already exists locally"
else
  git tag "$TAG"
fi
git push origin "$TAG"

echo "==> Done"
echo "Release workflow should now run from tag ${TAG}."
