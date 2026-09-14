#!/usr/bin/env bash
set -euo pipefail
BASE="${HOME}/.reviewsys"; DATA="${BASE}/observability-data"
mkdir -p "$DATA"
"${BASE}/venv/bin/reviewsys" --config "${BASE}/config.toml" export-status --output "$DATA"
cd "$DATA"
if [ ! -d .git ]; then git init -q; git checkout -q -b observability-data; git remote add origin https://github.com/thepastaclaw/review-system.git; fi
mkdir -p .github/workflows
cp "${BASE}/src/.github/workflows/pages.yml" .github/workflows/pages.yml
git add status.json
git add .github/workflows/pages.yml
git diff --cached --quiet && exit 0
git -c user.name=reviewsys -c user.email=reviewsys@thepastaclaw.ai commit -q -m "chore: publish review status"
git push -q origin observability-data
