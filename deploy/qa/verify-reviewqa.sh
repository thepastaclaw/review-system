#!/bin/bash
# Verify, from the claw account, that the reviewqa virtual GUI session is usable by a lane.
# Usage (on the Mac Studio as claw, or from anywhere with the claw key):
#   deploy/qa/verify-reviewqa.sh [host]     # default host: localhost
# Exit 0 only if ssh works AND a real screenshot comes back through the ssh hop.
set -euo pipefail
host="${1:-localhost}"
target="reviewqa@${host}"
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10)

echo "== ssh as reviewqa"
ssh "${ssh_opts[@]}" "$target" 'id -un; who | grep -c "^reviewqa.*console" || true'

echo "== gui session bound to this login"
ssh "${ssh_opts[@]}" "$target" 'launchctl print "gui/$(id -u)" >/dev/null 2>&1 && echo "gui domain: present" || { echo "gui domain: MISSING (no virtual session)"; exit 3; }'

echo "== screenshot through the hop"
tmp="$(mktemp -d)"
ssh "${ssh_opts[@]}" "$target" 'f="$(mktemp -t qa).png"; screencapture -x "$f" >/dev/null 2>&1 && cat "$f"; rm -f "$f"' > "$tmp/shot.png"
size=$(stat -f %z "$tmp/shot.png")
if [ "$size" -gt 10000 ]; then
  echo "OK: ${size} bytes -> $tmp/shot.png"
else
  echo "FAIL: got ${size} bytes; the session exists but capture is blocked (TCC) or the display is asleep"
  exit 4
fi

echo "== keepalive agent in dcg session"
ssh "${ssh_opts[@]}" "$target" 'pgrep -u dcg -fl "Screen Sharing" | head -2 || echo "no Screen Sharing.app client running as dcg (keepalive not installed)"' || true
