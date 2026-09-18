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
# Without the Screen Recording grant, screencapture over ssh silently returns wallpaper +
# menu bar. Crop both captures to the Calculator window's own rectangle and require a change.
ssh "${ssh_opts[@]}" "$target" 'open -a Calculator; sleep 4; ~/bin/winlist > "$TMPDIR/win.txt"; f="$(mktemp -t qa).png"; screencapture -x "$f" >/dev/null 2>&1 && cat "$f"; rm -f "$f"; pkill -x Calculator; sleep 2; g="$(mktemp -t qa).png"; screencapture -x "$g" >/dev/null 2>&1; cat "$TMPDIR/win.txt" >&2; rm -f "$TMPDIR/win.txt"' > "$tmp/shot.png" 2> "$tmp/win2.txt"
ssh "${ssh_opts[@]}" "$target" 'f="$(mktemp -t qa).png"; screencapture -x "$f" >/dev/null 2>&1 && cat "$f"; rm -f "$f"' > "$tmp/blank.png"
size=$(stat -f %z "$tmp/shot.png")
[ "$size" -gt 10000 ] || { echo "FAIL: got ${size} bytes; capture blocked or display asleep"; exit 4; }
disp_w=$(sed -n 's/^display=\([0-9]*\)x.*/\1/p' "$tmp/win2.txt")
cap_w=$(sips -g pixelWidth "$tmp/shot.png" | awk '/pixelWidth/{print $2}')
read -r wx wy ww wh < <(awk -F'\t' '$1=="Calculator"{split($3,a,/[ ,x]/); print a[1],a[2],a[3],a[4]; exit}' "$tmp/win2.txt")
[ -n "${ww:-}" ] || { echo "FAIL: Calculator window bounds not found"; cat "$tmp/win2.txt"; exit 3; }
scale=$(( cap_w / disp_w ))
for f in shot blank; do
  sips -c $((wh*scale)) $((ww*scale)) --cropOffset $((wy*scale)) $((wx*scale)) "$tmp/$f.png" --out "$tmp/$f-crop.png" >/dev/null
done
if cmp -s "$tmp/shot-crop.png" "$tmp/blank-crop.png"; then
  echo "FAIL: the Calculator window region is pixel-identical with and without the app (${ww}x${wh} at ${wx},${wy})."
  echo "      Screen Recording is not granted to the ssh path. In the reviewqa GUI session:"
  echo "      System Settings > Privacy & Security > Screen & System Audio Recording > + >"
  echo "      Cmd-Shift-G > /usr/libexec/sshd-keygen-wrapper > enable, then re-run."
  exit 5
fi
echo "OK: ${size} bytes, window region rendered -> $tmp/shot.png"

echo "== keepalive agent in dcg session"
ssh "${ssh_opts[@]}" "$target" 'pgrep -u dcg -fl "Screen Sharing" | head -2 || echo "no Screen Sharing.app client running as dcg (keepalive not installed)"' || true
