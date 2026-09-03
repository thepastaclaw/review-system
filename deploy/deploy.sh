#!/usr/bin/env bash
# Deploy reviewsys to the Mac Studio as user `claw`.
#
# Refuses to deploy anything that is not a tagged commit with a green CI run,
# so the OpenClaw agent (which may edit this repo) cannot push untested code
# into production by editing files on the box.
#
# Usage:  deploy/deploy.sh <tag>            (run from anywhere with gh + ssh access)
#         deploy/deploy.sh <tag> --force    (skip CI check; emergencies only)
set -euo pipefail

TAG="${1:?usage: deploy.sh <tag> [--force] [--shadow]}"
shift
FORCE=""; SHADOW=""
for a in "$@"; do case "$a" in --force) FORCE=1;; --shadow) SHADOW=1;; esac; done
REPO="thepastaclaw/review-system"
HOST="${REVIEWSYS_HOST:-claw@100.81.48.28}"
LABEL="ai.thepastaclaw.reviewsys"

SHA=$(gh api "repos/$REPO/git/ref/tags/$TAG" --jq .object.sha)
if [ -z "$FORCE" ]; then
  STATE=$(gh api "repos/$REPO/commits/$SHA/check-runs" --jq '[.check_runs[] | select(.name=="check")][0].conclusion // "none"')
  if [ "$STATE" != "success" ]; then
    echo "refusing to deploy $TAG ($SHA): CI conclusion is '$STATE' (need success). Use --force to override." >&2
    exit 1
  fi
fi

echo "deploying $TAG ($SHA) to $HOST"
ssh -o BatchMode=yes "$HOST" bash -s -- "$TAG" "$SHA" "$LABEL" "$SHADOW" <<'REMOTE'
set -euo pipefail
TAG="$1"; SHA="$2"; LABEL="$3"; SHADOW="$4"
export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.npm-global/bin
BASE=$HOME/.reviewsys
SRC=$BASE/src
mkdir -p "$BASE"
if [ ! -d "$SRC/.git" ]; then
  [ -d "$SRC" ] && mv "$SRC" "$SRC.pre-git-$(date -u +%Y%m%dT%H%M%SZ)"
  git clone -q https://github.com/thepastaclaw/review-system.git "$SRC"
fi
git -C "$SRC" fetch -q --tags origin
git -C "$SRC" checkout -q --detach "$SHA"
if [ ! -x "$BASE/venv/bin/python" ]; then uv venv -q --python 3.14 "$BASE/venv"; fi
uv pip install -q --python "$BASE/venv/bin/python" "$SRC"
[ -f "$BASE/config.toml" ] || "$BASE/venv/bin/reviewsys" --config "$BASE/config.toml" init-config
"$BASE/venv/bin/reviewsys" --config "$BASE/config.toml" doctor --no-models
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
if [ -n "$SHADOW" ]; then
  sed 's#<string>daemon</string>#<string>daemon</string><string>--no-spawn</string><string>--no-wake</string>#' "$SRC/deploy/ai.thepastaclaw.reviewsys.plist" > "$PLIST"
  echo "mode: SHADOW (daemon --no-spawn --no-wake)"
else
  cp "$SRC/deploy/ai.thepastaclaw.reviewsys.plist" "$PLIST"
  echo "mode: LIVE"
fi
plutil -lint "$PLIST" >/dev/null
echo "$TAG $SHA $(date -u +%FT%TZ)" >> "$BASE/deployed.log"
# restart if running (launchctl kickstart needs a GUI session; kill + KeepAlive respawn works over ssh)
PID=$(pgrep -f "venv/bin/reviewsys daemon" | head -1 || true)
if [ -n "$PID" ]; then kill "$PID"; sleep 3; fi
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load -w "$PLIST" 2>/dev/null || launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null || true
sleep 3
NEW=$(pgrep -f "venv/bin/reviewsys daemon" | head -1 || true)
echo "daemon pid: ${NEW:-not running (load the plist from a GUI session or run: launchctl load -w ~/Library/LaunchAgents/$LABEL.plist)}"
"$BASE/venv/bin/reviewsys" --config "$BASE/config.toml" status
REMOTE
