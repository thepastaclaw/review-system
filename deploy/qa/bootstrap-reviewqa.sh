#!/bin/bash
# One-time bootstrap of the `reviewqa` GUI account on the Mac Studio.
# Run INSIDE the reviewqa virtual Screen Sharing session (Terminal.app there):
#   bash /Users/Shared/bootstrap-reviewqa.sh
# Idempotent. Needs no sudo. It:
#   1. installs the claw + laptop ssh keys so reviewsys lanes can `ssh reviewqa@localhost`
#   2. keeps the session's display awake (no screen saver / no idle lock)
#   3. proves the session can capture its display (the whole point of the account)
set -euo pipefail
[ "$(id -un)" = reviewqa ] || { echo "run as reviewqa, not $(id -un)"; exit 1; }

echo "== ssh keys"
umask 077
mkdir -p ~/.ssh
touch ~/.ssh/authorized_keys
add_key() { grep -qF "$1" ~/.ssh/authorized_keys || echo "$1" >> ~/.ssh/authorized_keys; }
add_key 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPvEgKeDtQaLPOlVkWebo6LqMsOtk+JvJOOdEf4pRPxd thepastaclaw@github'
add_key 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGnTD5kFDNil2qWY1fL5tSAGEtZyGlVuWBOEDzfabkwG pasta@dashboost.org'
chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys
wc -l ~/.ssh/authorized_keys

echo "== display stays awake"
defaults -currentHost write com.apple.screensaver idleTime -int 0
defaults write com.apple.screensaver askForPassword -int 0
defaults write com.apple.screensaver askForPasswordDelay -int 0
# no sleep/screen-off inside this session; pmset needs sudo, caffeinate does not
pkill -u reviewqa -f "caffeinate -dimsu" 2>/dev/null || true
nohup caffeinate -dimsu >/dev/null 2>&1 &
echo "caffeinate pid $!"

echo "== gui domain"
launchctl print "gui/$(id -u)" | sed -n '1,3p'

echo "== screen capture"
out="${TMPDIR:-/tmp}/reviewqa-probe.png"
rm -f "$out"
if screencapture -x "$out" && [ -s "$out" ]; then
  echo "OK: $(stat -f %z "$out") bytes, $(sips -g pixelWidth -g pixelHeight "$out" | awk '/pixel/{printf "%s ",$2}')"
else
  echo "FAIL: screencapture produced nothing. Grant Terminal 'Screen & System Audio Recording' in System Settings > Privacy & Security, then re-run."
  exit 2
fi

echo "== toolchain visibility"
xcode-select -p; xcrun simctl list runtimes | tail -n +2 | head -3
command -v cargo >/dev/null && cargo --version || echo "cargo: not on PATH for reviewqa (expected; installed per-user later)"
echo "== done"
