#!/usr/bin/env bash
# The live v12 shadow runner on the box (`reviewsys shadow run`), apart from the production
# daemon: its own LaunchAgent label, its own root (~/.reviewsys-shadow), its own cron line.
# Stopping or removing it never touches the daemon.
#
# Usage: deploy/shadow.sh install   enable, install the LaunchAgent and the cron line, (re)start
#        deploy/shadow.sh start     start it if enabled and not running (what cron calls)
#        deploy/shadow.sh stop      disable and stop it (running replays are terminated and
#                                   their heads queued again); cron leaves it stopped
#        deploy/shadow.sh restart   stop and start, staying enabled (new code after a deploy)
#        deploy/shadow.sh status    enabled?, running?, guard present?, `reviewsys shadow status`
#
# launchd never supervises `claw` (no GUI login): as for the daemon, a cron line restarts it.
# It runs under `nice -n 10`; the plist adds ProcessType Background for the day launchd runs
# it. Settings: REVIEWSYS_SHADOW_SKILLS (default ~/Projects/skills-v12) and
# REVIEWSYS_SHADOW_GUARD (default the quota guard below; "none" for no guard).
set -euo pipefail
BASE="$HOME/.reviewsys"
ROOT="${REVIEWSYS_SHADOW_ROOT:-$HOME/.reviewsys-shadow}"
SKILLS="${REVIEWSYS_SHADOW_SKILLS:-$HOME/Projects/skills-v12}"
GUARD="${REVIEWSYS_SHADOW_GUARD:-/opt/homebrew/bin/python3 $HOME/.reviewsys-exp/bin/quota_guard.py}"
LABEL="ai.thepastaclaw.reviewsys-shadow"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.npm-global/bin:$HOME/.local/bin

pid() { local p; p=$(cat "$ROOT/shadow.pid" 2>/dev/null || true); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && echo "$p"; return 0; }

start() {
  [ -f "$ROOT/enabled" ] || { echo "shadow: not enabled (deploy/shadow.sh install)"; return 0; }
  [ -n "$(pid)" ] && return 0
  [ -f "$SKILLS/config.json" ] || { echo "shadow: $SKILLS has no config.json" >&2; return 1; }
  mkdir -p "$ROOT/logs"
  local guard=()
  [ "$GUARD" != "none" ] && guard=(--guard-cmd "$GUARD")
  "$BASE/venv/bin/python" - "$BASE" "$ROOT" "$SKILLS" ${guard[@]+"${guard[@]}"} <<'PY'
import os, subprocess, sys
base, root, skills, *guard = sys.argv[1:]
env = dict(os.environ, GODEBUG="netdns=go", REVIEWSYS_CONFIG=f"{base}/config.toml",
           PYTHONUNBUFFERED="1")
err = open(f"{root}/logs/shadow.stderr.log", "ab")
argv = ["/usr/bin/nice", "-n", "10", f"{base}/venv/bin/reviewsys", "--config",
        f"{base}/config.toml", "shadow", "run", "--root", root, "--skills", skills, *guard]
p = subprocess.Popen(argv, cwd=root, stdin=subprocess.DEVNULL, stdout=err, stderr=err, env=env,
                     start_new_session=True)
print(f"started reviewsys shadow runner pid {p.pid}")
PY
}

stop() {
  launchctl unload "$PLIST" 2>/dev/null || true
  local p; p=$(pid)
  [ -z "$p" ] && { echo "shadow: not running"; return 0; }
  kill "$p"
  # the runner terminates its replays' process groups (30 s grace) before it exits
  for _ in $(seq 1 120); do kill -0 "$p" 2>/dev/null || { echo "shadow: stopped"; return 0; }; sleep 1; done
  kill -9 "$p" 2>/dev/null || true
  echo "shadow: killed; the next start cleans up its replays" >&2
}

cron_line() {
  # lockf: an overlapping start exits at once; never `taskpolicy -b` (throttled I/O piles up)
  echo "* * * * * /usr/bin/lockf -s -t 0 $ROOT/cron.lock $SRC/deploy/shadow.sh start >/dev/null 2>&1"
}

case "${1:-}" in
  install)
    [ -f "$SKILLS/config.json" ] || { echo "shadow: $SKILLS has no config.json" >&2; exit 1; }
    mkdir -p "$ROOT/logs" "$HOME/Library/LaunchAgents"
    touch "$ROOT/enabled"
    sed -e "s#/opt/homebrew/bin/python3 /Users/claw/.reviewsys-exp/bin/quota_guard.py#$GUARD#" \
      -e "s#/Users/claw/Projects/skills-v12#$SKILLS#" -e "s#/Users/claw#$HOME#g" \
      "$SRC/deploy/$LABEL.plist" > "$PLIST"
    plutil -lint "$PLIST" >/dev/null
    { crontab -l 2>/dev/null | grep -v 'deploy/shadow.sh start' || true; cron_line; } | crontab -
    [ "$GUARD" = "none" ] || [ -f "${GUARD##* }" ] || echo "shadow: WARNING: quota guard ${GUARD##* } is missing: no replay launches until it is there" >&2
    stop; start
    launchctl load -w "$PLIST" 2>/dev/null || true  # a twin exits on the runner's lock
    ;;
  start) start ;;
  stop) rm -f "$ROOT/enabled"; stop ;;
  restart) stop; start ;;
  status)
    echo "enabled: $([ -f "$ROOT/enabled" ] && echo yes || echo no)"
    echo "runner pid: $(pid || true)"
    [ "$GUARD" = "none" ] || echo "guard: $GUARD ($([ -f "${GUARD##* }" ] && echo present || echo MISSING))"
    "$BASE/venv/bin/reviewsys" --config "$BASE/config.toml" shadow status --root "$ROOT" || true
    ;;
  *) sed -n '2,17p' "$0"; exit 2 ;;
esac
