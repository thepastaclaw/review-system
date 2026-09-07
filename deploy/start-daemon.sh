#!/usr/bin/env bash
# Start the reviewsys daemon detached (launchd never supervises `claw`: no GUI login).
# Safe to call when one is already running: the daemon's flock makes the twin exit.
# Usage: start-daemon.sh [--shadow]
set -euo pipefail
BASE="$HOME/.reviewsys"
ARGS="daemon"; [ "${1:-}" = "--shadow" ] && ARGS="daemon --no-spawn --no-wake"
if pgrep -f "venv/bin/reviewsys daemon" >/dev/null; then exit 0; fi
"$BASE/venv/bin/python" - "$BASE" "$ARGS" <<'PY'
import os, subprocess, sys
base, args = sys.argv[1], sys.argv[2].split()
env = dict(os.environ, GODEBUG="netdns=go", REVIEWSYS_CONFIG=f"{base}/config.toml",
           REVIEWSYS_LOG_FILE=f"{base}/daemon.log", PYTHONUNBUFFERED="1",
           PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:"
                f"{os.environ['HOME']}/.npm-global/bin:{os.environ['HOME']}/.local/bin")
crash = open(f"{base}/daemon.stderr.log", "ab")
p = subprocess.Popen([f"{base}/venv/bin/reviewsys", *args], cwd=base, stdin=subprocess.DEVNULL,
                     stdout=crash, stderr=crash, env=env, start_new_session=True)
print(f"started reviewsys daemon pid {p.pid} ({' '.join(args)})")
PY
