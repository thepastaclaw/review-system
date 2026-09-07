#!/usr/bin/env bash
# Cron watchdog (every minute): restart the reviewsys daemon if it is not running.
# Mode (live/shadow) is read from ~/.reviewsys/mode so a redeploy can change it.
set -u
BASE="$HOME/.reviewsys"
LOG="$BASE/watchdog.log"
if pgrep -f "venv/bin/reviewsys daemon" >/dev/null; then exit 0; fi
for f in "$LOG" "$BASE/daemon.stderr.log"; do
  [ -f "$f" ] && [ "$(wc -c < "$f")" -gt 1048576 ] && { tail -n 500 "$f" > "$f.tmp" && mv "$f.tmp" "$f"; }
done
MODE=""; [ "$(cat "$BASE/mode" 2>/dev/null)" = "shadow" ] && MODE="--shadow"
echo "$(date -u +%FT%TZ) daemon not running; starting $MODE" >> "$LOG"
"$BASE/src/deploy/start-daemon.sh" $MODE >> "$LOG" 2>&1
