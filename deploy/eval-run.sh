#!/usr/bin/env bash
# reviewsys policy eval on the box (v12 DESIGN.md §5): replay the eval set under two skills
# arms, judge what the labels cannot settle, score. Resumable: re-run the same command.
#
#   deploy/eval-run.sh --v12 SKILLS_COMMIT [options]     the real eval (production box)
#   deploy/eval-run.sh --dry [--eval-root DIR]           smoke test: fixture repo, fake lanes
#
# Options:
#   --v12 SHA            skills commit of the v12 arm (the feat/review-quality-v12 branch)
#   --v11 SHA            skills commit of the v11 arm (default 169fcb8)
#   --skills-repo DIR    skills repository the arms are checked out from (default: the
#                        production config's [paths] skills); never written to
#   --set FILE           eval set (default: $EVAL_ROOT/eval/set.json)
#   --labels FILE        labels (default: $EVAL_ROOT/eval/labels.jsonl)
#   --eval-root DIR      default ~/.reviewsys-eval
#   --concurrency N      replays (and judged heads) at once (default 3)
#   --limit N            only N heads, round-robin over the sets (smoke test; own out dir)
#   --pin-phase1-model M one Phase-1 model for both arms (default muse-spark-1.3-contributor;
#                        "none" keeps each arm's ladder)
#   --judge-model M      default gpt-6.1-sol;  --judge-effort E  default xhigh
#   --min-free-gb N      launch nothing while less is free (default 25)
#   --disk-wait-minutes N  then wait this long before stopping (default 30)
#   --retry-failed       also re-run failed and invalid replays
#   --refresh-snapshot   take a new snapshot of the production database
#   --only STAGES        comma list of: preflight,snapshot,arms,replay,score,judge
#   --allow-bot-token    run without REVIEWSYS_REPLAY_GH_TOKEN (see below)
#   --dry                every step on a fixture with fake lanes and a fake gh; no model, no
#                        GitHub, nothing of the production setup is touched
#
# What it touches: it reads the production database only through a read-only `.backup`
# snapshot ($EVAL_ROOT/source.db; the live one is opened read-only for the degraded-mode
# check and the gpt pool size), reads the production run artifacts (evidence.json), fetches
# into the production bare mirrors (as the daemon does) and takes lane slots in the
# production lane-slots directory at the lowest rank (behind every production lane, at most
# half of each pool). Everything it writes is under $EVAL_ROOT: its own venv with this
# checkout's reviewsys, the skills arms, scratch databases, per-replay pinned clones and
# worktrees (removed as each replay ends), reports, logs.
#
# GitHub: every replay read goes through a pinned view that refuses writes, but with the bot's
# token a bug there could still write as the bot on real PRs. So it requires a read-only token
# in REVIEWSYS_REPLAY_GH_TOKEN (a fine-grained PAT with read-only Contents, Pull requests,
# Issues, Checks on the reviewed repositories), and refuses to start without one unless
# --allow-bot-token is passed.
#
# Run it detached; it takes about a day and a half for the full set:
#   nohup deploy/eval-run.sh --v12 <sha> >/dev/null 2>&1 &
#   tail -f ~/.reviewsys-eval/logs/eval-run-*.log
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
ARGS=("$@")
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.npm-global/bin:$HOME/.local/bin:$PATH"
export GODEBUG=netdns=go PYTHONUNBUFFERED=1

EVAL_ROOT="${REVIEWSYS_EVAL_ROOT:-$HOME/.reviewsys-eval}"
PROD="${REVIEWSYS_PROD:-$HOME/.reviewsys}"
PROD_CONFIG="$PROD/config.toml"
SKILLS_REPO="" V11="169fcb8" V12="" SET="" LABELS=""
CONCURRENCY=3 LIMIT="" PIN_P1="muse-spark-1.3-contributor"
JUDGE_MODEL="gpt-6.1-sol" JUDGE_EFFORT="xhigh"
MIN_FREE_GB=25 DISK_WAIT=30 RETRY="" REFRESH="" ONLY="" ALLOW_BOT="" DRY=""
EXIT_ABORTED=75 EXIT_DISK=74

die() { echo "eval-run: $*" >&2; exit 1; }
need() { [ $# -ge 2 ] && [ -n "$2" ] || die "$1 needs a value"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --v12) need "$@"; V12="$2"; shift 2;;
    --v11) need "$@"; V11="$2"; shift 2;;
    --skills-repo) need "$@"; SKILLS_REPO="$2"; shift 2;;
    --set) need "$@"; SET="$2"; shift 2;;
    --labels) need "$@"; LABELS="$2"; shift 2;;
    --eval-root) need "$@"; EVAL_ROOT="$2"; shift 2;;
    --concurrency) need "$@"; CONCURRENCY="$2"; shift 2;;
    --limit) need "$@"; LIMIT="$2"; shift 2;;
    --pin-phase1-model) need "$@"; PIN_P1="$2"; shift 2;;
    --judge-model) need "$@"; JUDGE_MODEL="$2"; shift 2;;
    --judge-effort) need "$@"; JUDGE_EFFORT="$2"; shift 2;;
    --min-free-gb) need "$@"; MIN_FREE_GB="$2"; shift 2;;
    --disk-wait-minutes) need "$@"; DISK_WAIT="$2"; shift 2;;
    --retry-failed) RETRY=1; shift;;
    --refresh-snapshot) REFRESH=1; shift;;
    --only) need "$@"; ONLY=",$2,"; shift 2;;
    --allow-bot-token) ALLOW_BOT=1; shift;;
    --dry) DRY=1; shift;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0;;
    *) die "unknown argument $1 (see --help)";;
  esac
done
stage() { [ -z "$ONLY" ] || case "$ONLY" in *",$1,"*) return 0;; *) return 1;; esac; }
log() { echo "$(date -u +%FT%TZ) eval-run: $*"; }

mkdir -p "$EVAL_ROOT/logs"
EVAL_ROOT="$(cd "$EVAL_ROOT" && pwd)"

# everything goes to the log too: the script runs again as a child piped through tee
if [ -z "${EVAL_RUN_LOG:-}" ]; then
  EVAL_RUN_LOG="$EVAL_ROOT/logs/eval-run-$(date -u +%Y%m%dT%H%M%SZ)${DRY:+-dry}.log"
  export EVAL_RUN_LOG
  set +e
  "$0" ${ARGS[@]+"${ARGS[@]}"} 2>&1 | tee -a "$EVAL_RUN_LOG"
  exit "${PIPESTATUS[0]}"
fi

# one eval-run per root at a time
LOCK="$EVAL_ROOT/run.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  pid="$(cat "$LOCK/pid" 2>/dev/null || true)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then die "already running (pid $pid, $LOCK)"; fi
  rm -rf "$LOCK"; mkdir "$LOCK"
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT

log "log $EVAL_RUN_LOG; source $SRC ($(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo '?'))"

# this checkout's reviewsys, in the eval root's own venv (the daemon's venv is never touched)
if [ -n "${REVIEWSYS_EVAL_PYTHON:-}" ]; then
  PY="$REVIEWSYS_EVAL_PYTHON"
else
  PY="$EVAL_ROOT/venv/bin/python"
  want="$(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
  if [ ! -x "$PY" ] || [ "$(cat "$EVAL_ROOT/venv/.src-commit" 2>/dev/null)" != "$want" ]; then
    log "installing reviewsys $want into $EVAL_ROOT/venv"
    [ -x "$PY" ] || uv venv -q --python 3.14 "$EVAL_ROOT/venv"
    uv pip install -q --python "$PY" --reinstall-package reviewsys "$SRC"
    echo "$want" > "$EVAL_ROOT/venv/.src-commit"
  fi
fi
RS() { "$PY" -m reviewsys.cli "$@"; }
jget() { "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$@"; }

FAKE=()
if [ -n "$DRY" ]; then
  FIX="$EVAL_ROOT/fixture"
  if [ ! -f "$EVAL_ROOT/fixture.json" ]; then
    rm -rf "$FIX"
    RS replay-fixture "$FIX" > "$EVAL_ROOT/fixture.json"
  fi
  PROD_CONFIG="$(jget "$EVAL_ROOT/fixture.json" config)"
  SKILLS_REPO="$(jget "$EVAL_ROOT/fixture.json" skills_repo)"
  V11="$(jget "$EVAL_ROOT/fixture.json" v11)"; V12="$(jget "$EVAL_ROOT/fixture.json" v12)"
  SET="$(jget "$EVAL_ROOT/fixture.json" set)"; LABELS="$(jget "$EVAL_ROOT/fixture.json" labels)"
  export REVIEWSYS_REPLAY_GH_TOKEN="${REVIEWSYS_REPLAY_GH_TOKEN:-dry-run-not-a-token}"
  FAKE=(--fake-lanes)
  log "DRY RUN on the fixture in $FIX: fake lanes, fake gh, no model, no GitHub"
fi

[ -n "$V12" ] || die "--v12 SKILLS_COMMIT is required (the v12 arm)"
[ -f "$PROD_CONFIG" ] || die "no config at $PROD_CONFIG"
paths() {  # the configured [paths] value of $1 (db, work, skills), ~ expanded
  "$PY" -c 'import sys, tomllib, os; print(os.path.expanduser(tomllib.load(open(sys.argv[1], "rb"))["paths"][sys.argv[2]]))' "$PROD_CONFIG" "$1"
}
PROD_DB="$(paths db)"; PROD_WORK="$(paths work)"
SKILLS_REPO="${SKILLS_REPO:-$(paths skills)}"
SET="${SET:-$EVAL_ROOT/eval/set.json}"; LABELS="${LABELS:-$EVAL_ROOT/eval/labels.jsonl}"
[ -f "$SET" ] || die "no eval set at $SET (copy reviewsys-v12-design/eval/ to $EVAL_ROOT/eval/)"
[ -f "$LABELS" ] || die "no labels at $LABELS"
SNAP="$EVAL_ROOT/source.db"
OUT="$EVAL_ROOT/out${LIMIT:+-limit$LIMIT}"
LIM=(); if [ -n "$LIMIT" ]; then LIM=(--limit "$LIMIT"); fi
PIN=(); if [ "$PIN_P1" != "none" ]; then PIN=(--pin-phase1-model "$PIN_P1"); fi
RETRY_ARG=(); if [ -n "$RETRY" ]; then RETRY_ARG=(--retry-failed); fi

if [ -z "${REVIEWSYS_REPLAY_GH_TOKEN:-}" ]; then
  if [ -z "$ALLOW_BOT" ]; then
    die "REVIEWSYS_REPLAY_GH_TOKEN is not set. Give the eval a read-only GitHub token (a
fine-grained PAT with read-only Contents, Pull requests, Issues and Checks on the reviewed
repositories). Without one every replay reads GitHub with the bot's own gh credentials, which
can write: writes are refused by reviewsys code only. To accept that, pass --allow-bot-token."
  fi
  log "WARNING: --allow-bot-token: replays read GitHub with the bot's own credentials (write"
  log "WARNING: scope). Only reviewsys code (the pinned view, read-only runner) stands between"
  log "WARNING: a bug and a comment, review or label on a real PR as the bot."
fi

if stage preflight; then
  log "preflight: degraded mode (production database, read-only)"
  rc=0; RS --config "$PROD_CONFIG" replay-preflight || rc=$?
  [ $rc -eq $EXIT_ABORTED ] && die "degraded mode is active: an eval on stand-in models measures nothing; re-run when it is off"
  [ $rc -eq 0 ] || die "preflight failed (exit $rc)"
fi

if stage snapshot; then
  if [ ! -f "$SNAP" ] || [ -n "$REFRESH" ]; then
    log "snapshot: $PROD_DB -> $SNAP (sqlite backup of a read-only connection)"
    "$PY" - "$PROD_DB" "$SNAP.tmp" <<'PY'
import sqlite3, sys
src = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True, timeout=60)
dst = sqlite3.connect(sys.argv[2])
src.backup(dst)
dst.close(); src.close()
PY
    mv "$SNAP.tmp" "$SNAP"
  else
    log "snapshot: keeping $SNAP ($(date -r "$SNAP" -u +%FT%TZ)); --refresh-snapshot for a new one"
  fi
  rc=0; RS --config "$PROD_CONFIG" replay-preflight --set "$SET" --source-db "$SNAP" --source-runs "$PROD_WORK/runs" || rc=$?
  [ $rc -eq 0 ] || die "the set does not fit the snapshot (exit $rc)"
fi

arm_dir() { echo "$EVAL_ROOT/skills/$1"; }
if stage arms; then
  for arm in v11 v12; do
    if [ "$arm" = v11 ]; then sha="$V11"; else sha="$V12"; fi
    d="$(arm_dir $arm)"
    [ -d "$d/.git" ] || git clone -q --no-checkout "$SKILLS_REPO" "$d"
    if ! git -C "$d" cat-file -e "$sha^{commit}" 2>/dev/null; then
      git -C "$d" fetch -q origin 2>/dev/null || true
    fi
    if ! git -C "$d" cat-file -e "$sha^{commit}" 2>/dev/null; then
      url="$(git -C "$SKILLS_REPO" remote get-url origin 2>/dev/null || true)"
      [ -n "$url" ] && git -C "$d" fetch -q "$url" "$sha" || true
    fi
    git -C "$d" cat-file -e "$sha^{commit}" 2>/dev/null || die "skills commit $sha ($arm) not found in $SKILLS_REPO or its origin"
    full="$(git -C "$d" rev-parse "$sha^{commit}")"
    git -C "$d" checkout -q --detach "$full"
    [ -f "$d/config.json" ] || die "$d has no config.json at $full"
    log "arm $arm: $d at $full ($("$PY" -c 'import json,sys; p=json.load(open(sys.argv[1]))["review_model_policy"]; print(p["name"], "v", p.get("version"), "pipeline" if p.get("pipeline") else "no pipeline")' "$d/config.json"))"
  done
fi
POLICIES=(--policy "v11=$(arm_dir v11)" --policy "v12=$(arm_dir v12)")

if stage replay; then
  log "replay-set: $SET -> $OUT, concurrency $CONCURRENCY${LIMIT:+, limit $LIMIT}"
  rc=0
  RS --config "$PROD_CONFIG" replay-set "$SET" "${POLICIES[@]}" --out "$OUT" \
    --concurrency "$CONCURRENCY" --force-phase2 ${PIN[@]+"${PIN[@]}"} --source-db "$SNAP" \
    --source-runs "$PROD_WORK/runs" --gh-token-env REVIEWSYS_REPLAY_GH_TOKEN \
    --min-free-gb "$MIN_FREE_GB" --disk-wait-minutes "$DISK_WAIT" \
    ${LIM[@]+"${LIM[@]}"} ${RETRY_ARG[@]+"${RETRY_ARG[@]}"} ${FAKE[@]+"${FAKE[@]}"} || rc=$?
  case $rc in
    0) ;;
    1) log "replay-set: some replays failed (their logs are under $OUT/<arm>/); scoring the rest";;
    $EXIT_ABORTED) die "replay-set stopped: degraded mode or dry models; re-run this command later to resume";;
    $EXIT_DISK) die "replay-set stopped: less than $MIN_FREE_GB GB free; free space and re-run to resume";;
    *) die "replay-set exited $rc";;
  esac
fi

if stage score; then
  log "replay-score (before the judge)"
  RS replay-score "$SET" "$LABELS" --out "$OUT" --baseline v11 ${LIM[@]+"${LIM[@]}"}
fi

if stage judge; then
  log "replay-judge: $JUDGE_MODEL at $JUDGE_EFFORT"
  rc=0
  RS --config "$PROD_CONFIG" replay-judge "$OUT" --set "$SET" --labels "$LABELS" --baseline v11 \
    --model "$JUDGE_MODEL" --effort "$JUDGE_EFFORT" --concurrency "$CONCURRENCY" \
    --min-free-gb "$MIN_FREE_GB" ${LIM[@]+"${LIM[@]}"} ${FAKE[@]+"${FAKE[@]}"} || rc=$?
  case $rc in
    0) ;;
    1) log "replay-judge: some items got no verdict; re-run to ask them again";;
    $EXIT_DISK) die "replay-judge stopped: disk below $MIN_FREE_GB GB; re-run to resume";;
    *) die "replay-judge exited $rc";;
  esac
fi

log "done: $OUT/score.json, $OUT/judged.jsonl, $OUT/spot_check.jsonl"
if [ -n "$DRY" ]; then
  "$PY" - "$OUT" "$EVAL_ROOT/fixture" <<'PY'
import json, sys
from pathlib import Path
out, fix = Path(sys.argv[1]), Path(sys.argv[2])
score = json.loads((out / "score.json").read_text())
jf = out / "judged.jsonl"
judged = [json.loads(x) for x in jf.read_text().splitlines() if x.strip()] if jf.exists() else []
problems = []
if score["heads"]["paired"] != score["heads"]["in_set"]:
    problems.append(f"paired {score['heads']['paired']} of {score['heads']['in_set']} heads")
if not judged or score["to_judge"]:
    problems.append(f"judged {len(judged)}, still to judge {score['to_judge']}")
log = fix / "gh-calls.log"
calls = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
writes = [c for c in calls if "--method" in c and c[c.index("--method") + 1] != "GET"]
if writes:
    problems.append(f"gh writes reached the (fake) gh: {writes[:3]}")
left = [p for p in out.glob("*/run-*/work/*") if p.name in ("worktrees", "mirrors")]
left += [p for p in (out / "judge" / "work").glob("*/*") if p.parent.name in ("worktrees", "mirrors")]
if left:
    problems.append(f"worktrees or pinned clones left behind: {left[:3]}")
print(json.dumps({"paired": score["heads"]["paired"], "judged": len(judged), "gh_calls": len(calls),
                  "precision": {a: v.get("precision") for a, v in score["arms"].items()}}))
if problems:
    sys.exit("DRY RUN FAILED: " + "; ".join(problems))
print("DRY RUN OK")
PY
fi
