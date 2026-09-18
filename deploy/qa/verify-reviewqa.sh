#!/bin/bash
# Verify, from the claw account, that the reviewqa virtual GUI session is usable by a lane.
# Usage (on the Mac Studio as claw, or from anywhere with the claw key):
#   deploy/qa/verify-reviewqa.sh [host]     # default host: localhost
# Exit 0 only if ssh works AND a real screenshot comes back through the ssh hop.
set -euo pipefail
host="${1:-localhost}"
target="reviewqa@${host}"
ssh_opts=(-n -o BatchMode=yes -o ConnectTimeout=10)  # -n: never eat stdin (script may be piped via bash -s)

echo "== ssh as reviewqa"
ssh "${ssh_opts[@]}" "$target" 'id -un; who | grep -c "^reviewqa.*console" || true'

echo "== gui session bound to this login"
ssh "${ssh_opts[@]}" "$target" 'launchctl print "gui/$(id -u)" >/dev/null 2>&1 && echo "gui domain: present" || { echo "gui domain: MISSING (no virtual session)"; exit 3; }'

tmp="$(mktemp -d)"; tmp_win="$tmp/win.txt"
echo "== windows are composited (Calculator opened over the hop)"
# `screencapture` without the Screen Recording grant silently returns wallpaper + menu bar,
# so a non-empty PNG proves nothing. Prove rendering with the window list instead.
ssh "${ssh_opts[@]}" "$target" 'test -x ~/bin/winlist || swiftc -O -o ~/bin/winlist /Users/Shared/winlist.swift; open -a Calculator; sleep 4; ~/bin/winlist; pkill -x Calculator' > "$tmp_win" 2>&1 || true
grep -q '^Calculator' "$tmp_win" && echo "OK: Calculator window present in reviewqa session" || { echo "FAIL: no Calculator window:"; cat "$tmp_win"; exit 3; }

echo "== screenshot through the hop shows real windows"
ssh "${ssh_opts[@]}" "$target" 'open -a Calculator; sleep 4; f="$(mktemp -t qa).png"; screencapture -x "$f" >/dev/null 2>&1 && cat "$f"; rm -f "$f"; pkill -x Calculator' > "$tmp/shot.png"
ssh "${ssh_opts[@]}" "$target" 'f="$(mktemp -t qa).png"; screencapture -x "$f" >/dev/null 2>&1 && cat "$f"; rm -f "$f"' > "$tmp/blank.png"
size=$(stat -f %z "$tmp/shot.png")
if [ "$size" -lt 10000 ]; then
  echo "FAIL: got ${size} bytes; the session exists but capture is blocked or the display is asleep"
  exit 4
fi
if cmp -s "$tmp/shot.png" "$tmp/blank.png"; then
  echo "FAIL: capture with and without Calculator open are byte-identical (wallpaper only)."
  echo "      Screen Recording is not granted to the ssh path. In the reviewqa GUI session:"
  echo "      System Settings > Privacy & Security > Screen & System Audio Recording > + >"
  echo "      Cmd-Shift-G > /usr/libexec/sshd-keygen-wrapper, then re-run (new ssh session)."
  exit 5
fi
echo "OK: ${size} bytes, differs from the empty desktop -> $tmp/shot.png"

echo "== keepalive agent in dcg session"
ssh "${ssh_opts[@]}" "$target" 'pgrep -u dcg -fl "Screen Sharing" | head -2 || echo "no Screen Sharing.app client running as dcg (keepalive not installed)"' || true
