#!/usr/bin/env bash
# reviewsys policy eval on the box (v12 DESIGN.md §5): replay the eval set under two skills
# arms, judge what the labels cannot settle, score. Resumable: re-run the same command.
#
#   deploy/eval-run.sh --v12 SKILLS_COMMIT [options]     the real eval (production box)
#   deploy/eval-run.sh --dry [--eval-root DIR]           smoke test: fixture repo, fake lanes
#   deploy/eval-run.sh --stop [--eval-root DIR]          stop a running eval and clean up
#
# Options:
#   --v12 SHA            skills commit of the v12 arm (the feat/review-quality-v12 branch)
#   --v11 SHA            skills commit of the v11 arm (default 169fcb8)
#   --skills-repo DIR    skills repository the arms are checked out from (default: the
#                        production config's [paths] skills); never written to
#   --set FILE           eval set (default: $EVAL_ROOT/eval/set.json)
#   --labels FILE        labels (default: $EVAL_ROOT/eval/labels.jsonl)
#   --eval-root DIR      default ~/.reviewsys-eval (~/.reviewsys-eval-dry with --dry)
#   --concurrency N      replays (and judged heads) at once (default 3)
#   --limit N            only N heads, round-robin over the sets (smoke test; own out dir)
#   --pin-phase1-model M one Phase-1 model for both arms (default muse-spark-1.3-contributor;
#                        "none" keeps each arm's ladder)
#   --judge-model M      default gpt-6.1-sol;  --judge-effort E  default xhigh
#   --min-free-gb N      keep this much disk free (default 35), counting --disk-per-job-gb
#                        (default 6) for every replay or judged head running
#   --disk-wait-minutes N  with nothing running and the disk short, wait, then stop (30)
#   --min-usable-accounts N  stop launching below this many usable OpenAI accounts (2)
#   --busy-wait-minutes N  with production's queue long and nothing running, wait, then stop (60)
#   --replay-pool-slots P=N  replay lanes hold at most N slots of pool P (repeatable; default
#                        gpt=2 for the eval; "none" for the config's replay_pool_share only)
#   --retry-failed       also re-run failed and invalid replays
#   --refresh-snapshot   take a new snapshot of the production database
#   --only STAGES        comma list of: preflight,snapshot,arms,replay,score,judge
#   --allow-bot-token    run without REVIEWSYS_REPLAY_GH_TOKEN (see below)
#   --stop-grace S       --stop: seconds between TERM and KILL (default 60)
#   --dry                every step on a fixture with fake lanes and a fake gh; no model, no
#                        GitHub, nothing of the production setup is touched
#
# What it touches: it reads the production database only through a read-only `.backup`
# snapshot ($EVAL_ROOT/source.db; the live one is opened read-only before every launch for
# degraded mode, the OpenAI account reading, the live queue and the gpt pool size), reads the
# production run artifacts (evidence.json) and the production bare mirrors (never writes
# them: each replay fetches what a mirror lacks into its own clone; a repository with no
# production mirror is mirrored under $EVAL_ROOT/mirrors), and takes lane slots from the
# production lane-slots pools, waiting in its own `<pool>.replay/` directories (production
# never sees a replay waiting), only while no production lane waits, at most half of each
# pool's non-side-lane slots. Everything it writes is under $EVAL_ROOT: its own venv with this
# checkout's reviewsys, the skills arms, scratch databases, per-replay clones and worktrees
# (removed as each replay ends), the judge's work dir (judge-work, outside the scored out/),
# reports, logs.
#
# Launches stop (exit 75, resumable) when production goes degraded or has fewer than
# --min-usable-accounts usable OpenAI accounts; they wait while production's live queue is
# at half its Phase-1 skip threshold, and while the disk is short.
#
# GitHub: every replay read goes through a pinned view that refuses writes, but with a token
# that can write, a bug there could still write as that account on real PRs. So it requires a
# read-only token in REVIEWSYS_REPLAY_GH_TOKEN (a fine-grained PAT with read-only Contents,
# Pull requests, Issues and Checks on the reviewed repositories, made by an account that has
# no write access to them: GitHub reports the account's role, not the token's), refuses the
# bot's own token (`gh auth token`), a classic token with a write scope, and any token whose
# account can push to a set repository, and refuses to start without one unless
# --allow-bot-token is passed. Lanes never see the token: it stays with the replay's broker.
#
# Run it detached; the full set takes two to three days beside production:
#   nohup deploy/eval-run.sh --v12 <sha> >/dev/null 2>&1 &
#   tail -f ~/.reviewsys-eval/logs/eval-run-*.log
#   deploy/eval-run.sh --stop        # TERM to the eval's process group, KILL after the grace
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
ARGS=("$@")
export PATH="$PATH:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.npm-global/bin:$HOME/.local/bin"
export GODEBUG=netdns=go PYTHONUNBUFFERED=1

REAL_ROOT="${REVIEWSYS_EVAL_ROOT:-$HOME/.reviewsys-eval}"
EVAL_ROOT="$REAL_ROOT"
PROD="${REVIEWSYS_PROD:-$HOME/.reviewsys}"
PROD_CONFIG="$PROD/config.toml"
SKILLS_REPO="" V11="169fcb8" V12="" SET="" LABELS=""
CONCURRENCY=3 LIMIT="" PIN_P1="muse-spark-1.3-contributor"
JUDGE_MODEL="gpt-6.1-sol" JUDGE_EFFORT="xhigh"
MIN_FREE_GB=35 JOB_GB=6 DISK_WAIT=30 MIN_ACCOUNTS=2 BUSY_WAIT=60 STOP_GRACE=60
POOL_SLOTS=() POOL_SLOTS_SET=""
RETRY="" REFRESH="" ONLY="" ALLOW_BOT="" DRY="" STOP="" EVAL_ROOT_SET=""
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
    --eval-root) need "$@"; EVAL_ROOT="$2"; EVAL_ROOT_SET=1; shift 2;;
    --concurrency) need "$@"; CONCURRENCY="$2"; shift 2;;
    --limit) need "$@"; LIMIT="$2"; shift 2;;
    --pin-phase1-model) need "$@"; PIN_P1="$2"; shift 2;;
    --judge-model) need "$@"; JUDGE_MODEL="$2"; shift 2;;
    --judge-effort) need "$@"; JUDGE_EFFORT="$2"; shift 2;;
    --min-free-gb) need "$@"; MIN_FREE_GB="$2"; shift 2;;
    --disk-per-job-gb) need "$@"; JOB_GB="$2"; shift 2;;
    --disk-wait-minutes) need "$@"; DISK_WAIT="$2"; shift 2;;
    --min-usable-accounts) need "$@"; MIN_ACCOUNTS="$2"; shift 2;;
    --busy-wait-minutes) need "$@"; BUSY_WAIT="$2"; shift 2;;
    --stop-grace) need "$@"; STOP_GRACE="$2"; shift 2;;
    --replay-pool-slots) need "$@"; POOL_SLOTS_SET=1; [ "$2" = none ] || POOL_SLOTS+=("$2"); shift 2;;
    --retry-failed) RETRY=1; shift;;
    --refresh-snapshot) REFRESH=1; shift;;
    --only) need "$@"; ONLY=",$2,"; shift 2;;
    --allow-bot-token) ALLOW_BOT=1; shift;;
    --dry) DRY=1; shift;;
    --stop) STOP=1; shift;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0;;
    *) die "unknown argument $1 (see --help)";;
  esac
done
stage() { [ -z "$ONLY" ] || case "$ONLY" in *",$1,"*) return 0;; *) return 1;; esac; }
log() { echo "$(date -u +%FT%TZ) eval-run: $*"; }
canon() { local d="${1%/}"; if [ -d "$d" ]; then (cd "$d" && pwd -P); else echo "$d"; fi; }

if [ -n "$DRY" ]; then
  # a dry run never shares a root (out/, snapshot, arms) with a real eval
  [ -n "$EVAL_ROOT_SET" ] || EVAL_ROOT="$REAL_ROOT-dry"
  [ "$(canon "$EVAL_ROOT")" != "$(canon "$REAL_ROOT")" ] \
    || die "--dry refuses the real eval root $REAL_ROOT; give another --eval-root"
fi

# ---- --stop: the eval's process group, then what its replays left behind ----
if [ -n "$STOP" ]; then
  [ -d "$EVAL_ROOT" ] || die "no eval root at $EVAL_ROOT"
  EVAL_ROOT="$(canon "$EVAL_ROOT")"
  pg="$(cat "$EVAL_ROOT/eval.pgid" 2>/dev/null || true)"
  if [ -n "$pg" ] && kill -0 -- "-$pg" 2>/dev/null; then
    [ "$pg" != "$(perl -e 'print getpgrp()')" ] || die "refusing to signal my own process group"
    echo "eval-run: TERM to process group $pg"
    kill -TERM -- "-$pg" 2>/dev/null || true
    i=0
    while kill -0 -- "-$pg" 2>/dev/null && [ "$i" -lt "$STOP_GRACE" ]; do sleep 1; i=$((i + 1)); done
    if kill -0 -- "-$pg" 2>/dev/null; then
      echo "eval-run: still running after ${STOP_GRACE}s: KILL to process group $pg"
      kill -KILL -- "-$pg" 2>/dev/null || true
      sleep 1
    fi
  else
    echo "eval-run: no running eval under $EVAL_ROOT"
  fi
  n=0
  for d in "$EVAL_ROOT"/out*/*/run-*/work/worktrees "$EVAL_ROOT"/out*/*/run-*/work/mirrors \
           "$EVAL_ROOT"/judge-work/work/worktrees "$EVAL_ROOT"/judge-work/work/mirrors; do
    if [ -d "$d" ]; then rm -rf "$d"; n=$((n + 1)); fi
  done
  rm -f "$EVAL_ROOT/eval.pgid"; rm -rf "$EVAL_ROOT/run.lock"
  echo "eval-run: stopped; removed $n leftover worktree/clone dirs; re-run the eval command to resume"
  exit 0
fi

mkdir -p "$EVAL_ROOT/logs"
EVAL_ROOT="$(canon "$EVAL_ROOT")"

# everything goes to the log too, and the eval runs in a process group of its own (--stop
# signals that group): the script runs again as a child in a new group piped through tee
if [ -z "${EVAL_RUN_LOG:-}" ]; then
  EVAL_RUN_LOG="$EVAL_ROOT/logs/eval-run-$(date -u +%Y%m%dT%H%M%SZ)${DRY:+-dry}.log"
  export EVAL_RUN_LOG
  set +e
  (
    set -o pipefail
    perl -e 'setpgrp(0, 0); exec { $ARGV[0] } @ARGV or die "exec: $!\n"' \
      bash "$0" ${ARGS[@]+"${ARGS[@]}"} 2>&1 | tee -a "$EVAL_RUN_LOG"
  ) &
  wrapper=$!
  # an interrupt here (Ctrl-C) reaches the eval's own group too
  trap 'pg="$(cat "$EVAL_ROOT/eval.pgid" 2>/dev/null)"; [ -n "$pg" ] && kill -TERM -- "-$pg" 2>/dev/null' INT TERM
  rc=0
  wait "$wrapper"; rc=$?
  while kill -0 "$wrapper" 2>/dev/null; do wait "$wrapper"; rc=$?; done
  exit "$rc"
fi

# one eval-run per root at a time
LOCK="$EVAL_ROOT/run.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  pid="$(cat "$LOCK/pid" 2>/dev/null || true)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then die "already running (pid $pid, $LOCK)"; fi
  rm -rf "$LOCK"; mkdir "$LOCK"
fi
echo $$ > "$LOCK/pid"
perl -e 'print getpgrp(), "\n"' > "$EVAL_ROOT/eval.pgid"  # this shell's group (perl inherits it)
trap 'rm -rf "$LOCK" "$EVAL_ROOT/eval.pgid"' EXIT

log "log $EVAL_RUN_LOG; source $SRC ($(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo '?')); process group $(cat "$EVAL_ROOT/eval.pgid")"

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
  MIN_ACCOUNTS=0  # the fixture's database has no OpenAI account reading
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
# the eval's replays hold at most 2 gpt slots unless told otherwise: the gpt pool shrinks
# whenever the box parks an OpenAI account, and production must keep the rest
if [ -z "$POOL_SLOTS_SET" ]; then POOL_SLOTS=(gpt=2); fi
SLOTS=()
for ps in ${POOL_SLOTS[@]+"${POOL_SLOTS[@]}"}; do SLOTS+=(--replay-pool-slots "$ps"); done
GATE=(--min-free-gb "$MIN_FREE_GB" --disk-per-job-gb "$JOB_GB" --disk-wait-minutes "$DISK_WAIT"
      --min-usable-accounts "$MIN_ACCOUNTS" --busy-wait-minutes "$BUSY_WAIT"
      --eval-mirrors "$EVAL_ROOT/mirrors" ${SLOTS[@]+"${SLOTS[@]}"})

# ---- the GitHub token: read-only, not the bot's ----
TOKEN="${REVIEWSYS_REPLAY_GH_TOKEN:-}"
if [ -z "$TOKEN" ]; then
  if [ -z "$ALLOW_BOT" ]; then
    die "REVIEWSYS_REPLAY_GH_TOKEN is not set. Give the eval a read-only GitHub token (a
fine-grained PAT with read-only Contents, Pull requests, Issues and Checks on the reviewed
repositories, from an account without write access to them). Without one every replay reads
GitHub with the bot's own gh credentials, which can write: writes are refused by reviewsys
code only. To accept that, pass --allow-bot-token."
  fi
  log "WARNING: --allow-bot-token: replays read GitHub with the bot's own credentials (write"
  log "WARNING: scope). Only reviewsys code (the pinned view, read-only runner) stands between"
  log "WARNING: a bug and a comment, review or label on a real PR as the bot."
elif [ -z "$DRY" ]; then
  own="$(env -u GH_TOKEN -u GITHUB_TOKEN gh auth token 2>/dev/null || true)"
  if [ -n "$own" ] && [ "$own" = "$TOKEN" ]; then
    die "REVIEWSYS_REPLAY_GH_TOKEN is this machine's own gh token (gh auth token), which can write; give a read-only one"
  fi
  "$PY" - "$SET" <<'PY' > "$EVAL_ROOT/logs/token-check.json" || die "the token check failed: $(cat "$EVAL_ROOT/logs/token-check.json" 2>/dev/null)"
import json, os, subprocess, sys
env = {**os.environ, "GH_TOKEN": os.environ["REVIEWSYS_REPLAY_GH_TOKEN"]}
env.pop("GITHUB_TOKEN", None)
def gh(*args):
    r = subprocess.run(["gh", *args], capture_output=True, text=True, env=env, timeout=60)
    if r.returncode:
        print(json.dumps({"error": f"gh {' '.join(args)}: {r.stderr.strip()[:200]}"})); sys.exit(1)
    return r.stdout
raw = gh("api", "-i", "user")
head, _, body = raw.replace("\r\n", "\n").partition("\n\n")
scopes = next((l.split(":", 1)[1].strip() for l in head.splitlines()
               if l.lower().startswith("x-oauth-scopes:")), "")
login = json.loads(body).get("login")
problems = []
writing = [s for s in (x.strip() for x in scopes.split(",")) if s and s.split(":")[0] in
           ("repo", "public_repo", "write", "admin") or s.startswith(("write:", "admin:"))]
if writing:
    problems.append(f"classic token scopes that can write: {writing}")
top = json.load(open(sys.argv[1]))
repos = sorted({h["repo"] for h in (top.get("heads", top) if isinstance(top, dict) else top)
                if isinstance(h, dict) and h.get("repo")})
for repo in repos:
    perms = json.loads(gh("api", f"repos/{repo}")).get("permissions") or {}
    can = sorted(k for k in ("admin", "maintain", "push") if perms.get(k))
    if can:
        problems.append(f"{login} has {'/'.join(can)} on {repo}")
print(json.dumps({"login": login, "scopes": scopes, "repos": len(repos), "problems": problems}))
sys.exit(2 if problems else 0)
PY
  log "token: $(cat "$EVAL_ROOT/logs/token-check.json")"
fi

preflight() {  # production healthy now (and with --set, the set fits the snapshot)
  local rc=0
  RS --config "$PROD_CONFIG" replay-preflight --min-usable-accounts "$MIN_ACCOUNTS" "$@" || rc=$?
  [ $rc -eq $EXIT_ABORTED ] && die "production is not in a state to share with an eval (see above: degraded mode, OpenAI accounts or a long queue); re-run later to resume"
  [ $rc -eq 0 ] || die "preflight failed (exit $rc)"
}

if stage preflight; then
  log "preflight: production database (read-only)"
  preflight
fi

if stage snapshot; then
  if [ ! -f "$SNAP" ] || [ -n "$REFRESH" ]; then
    "$PY" - "$EVAL_ROOT" "$PROD_DB" "$MIN_FREE_GB" <<'PY' || die "not enough disk for the snapshot"
import os, shutil, sys
root, db, floor = sys.argv[1], sys.argv[2], float(sys.argv[3])
size = sum(os.path.getsize(p) for p in (db, db + "-wal") if os.path.exists(p))
free = shutil.disk_usage(root).free
if free - size < floor * 1024**3:
    sys.exit(f"{free / 1024**3:.1f} GB free, snapshot {size / 1024**3:.2f} GB, floor {floor:g} GB")
PY
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
  preflight --set "$SET" --source-db "$SNAP" --source-runs "$PROD_WORK/runs"
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
      if [ -n "$url" ]; then git -C "$d" fetch -q "$url" "$sha" || true; fi
    fi
    git -C "$d" cat-file -e "$sha^{commit}" 2>/dev/null || die "skills commit $sha ($arm) not found in $SKILLS_REPO or its origin"
    full="$(git -C "$d" rev-parse "$sha^{commit}")"
    git -C "$d" checkout -q --detach "$full"
    [ -f "$d/config.json" ] || die "$d has no config.json at $full"
    log "arm $arm: $d at $full ($("$PY" -c 'import json,sys; p=json.load(open(sys.argv[1]))["review_model_policy"]; print(p["name"], "v", p.get("version"), "pipeline" if p.get("pipeline") else "no pipeline")' "$d/config.json"))"
  done
fi
POLICIES=(--policy "v11=$(arm_dir v11)" --policy "v12=$(arm_dir v12)")

FAILED=""
if stage replay; then
  log "replay-set: $SET -> $OUT, concurrency $CONCURRENCY${LIMIT:+, limit $LIMIT}"
  rc=0
  RS --config "$PROD_CONFIG" replay-set "$SET" "${POLICIES[@]}" --out "$OUT" \
    --concurrency "$CONCURRENCY" --force-phase2 ${PIN[@]+"${PIN[@]}"} --source-db "$SNAP" \
    --source-runs "$PROD_WORK/runs" --gh-token-env REVIEWSYS_REPLAY_GH_TOKEN "${GATE[@]}" \
    ${LIM[@]+"${LIM[@]}"} ${RETRY_ARG[@]+"${RETRY_ARG[@]}"} ${FAKE[@]+"${FAKE[@]}"} || rc=$?
  case $rc in
    0) ;;
    1) FAILED="$FAILED replay-set"; log "replay-set: some replays failed (logs under $OUT/<arm>/); scoring the rest";;
    "$EXIT_ABORTED") die "replay-set stopped: production degraded, short of OpenAI accounts or busy, or a replay's models ran dry; re-run this command later to resume";;
    "$EXIT_DISK") die "replay-set stopped: the disk stayed below $MIN_FREE_GB GB; free space and re-run to resume";;
    *) die "replay-set exited $rc";;
  esac
fi

if stage score; then
  log "replay-score (before the judge)"
  RS replay-score "$SET" "$LABELS" --out "$OUT" --baseline v11 ${LIM[@]+"${LIM[@]}"}
fi

if stage judge; then
  log "preflight before the judge"
  preflight
  log "replay-judge: $JUDGE_MODEL at $JUDGE_EFFORT"
  rc=0
  RS --config "$PROD_CONFIG" replay-judge "$OUT" --set "$SET" --labels "$LABELS" --baseline v11 \
    --model "$JUDGE_MODEL" --effort "$JUDGE_EFFORT" --concurrency "$CONCURRENCY" \
    --work-dir "$EVAL_ROOT/judge-work" "${GATE[@]}" ${LIM[@]+"${LIM[@]}"} \
    ${FAKE[@]+"${FAKE[@]}"} || rc=$?
  case $rc in
    0) ;;
    1) FAILED="$FAILED replay-judge"; log "replay-judge: some items got no verdict; re-run to ask them again";;
    "$EXIT_ABORTED") die "replay-judge stopped: production degraded, short of OpenAI accounts or busy; re-run to resume";;
    "$EXIT_DISK") die "replay-judge stopped: disk below $MIN_FREE_GB GB; re-run to resume";;
    *) die "replay-judge exited $rc";;
  esac
fi

# what the out dir holds now, whatever this invocation ran
"$PY" - "$OUT" "$SET" ${LIMIT:+"$LIMIT"} <<'PY' > "$EVAL_ROOT/logs/summary.json"
import json, sys
from collections import Counter
from pathlib import Path
from reviewsys.replayset import limit_items, load_set, plan, Policy, load_report, report_state
out = Path(sys.argv[1])
items = limit_items(load_set(Path(sys.argv[2])), int(sys.argv[3]) if len(sys.argv) > 3 else None)
manifest = json.loads((out / "replay-set.json").read_text()) if (out / "replay-set.json").exists() else {}
arms = [Policy(p["name"], Path(p["skills"])) for p in manifest.get("policies") or []]
states = Counter(report_state(load_report(j, out)) for j in plan(items, arms)) if arms else Counter()
pending = sum(1 for x in (out / "to_judge.jsonl").read_text().splitlines() if x.strip()) \
    if (out / "to_judge.jsonl").exists() else 0
judged = sum(1 for x in (out / "judged.jsonl").read_text().splitlines() if x.strip()) \
    if (out / "judged.jsonl").exists() else 0
print(json.dumps({"replays": dict(states), "judged": judged, "to_judge": pending}))
PY
log "summary: $(cat "$EVAL_ROOT/logs/summary.json")"
log "outputs: $OUT/score.json, $OUT/judged.jsonl, $OUT/spot_check.jsonl"

if [ -n "$DRY" ]; then
  "$PY" - "$OUT" "$EVAL_ROOT" <<'PY'
import json, sys
from pathlib import Path
out, root = Path(sys.argv[1]), Path(sys.argv[2])
fix = root / "fixture"
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
left += [p for p in (root / "judge-work" / "work").glob("*/*") if p.parent.name in ("worktrees", "mirrors")]
if left:
    problems.append(f"worktrees or pinned clones left behind: {left[:3]}")
if (out / "judge-work").exists() or any(out.glob("*.judge-work")):
    problems.append("the judge's work dir is inside the scored out dir")
print(json.dumps({"paired": score["heads"]["paired"], "judged": len(judged), "gh_calls": len(calls),
                  "precision": {a: v.get("precision") for a, v in score["arms"].items()}}))
if problems:
    sys.exit("DRY RUN FAILED: " + "; ".join(problems))
print("DRY RUN OK")
PY
fi

if [ -n "$FAILED" ]; then
  log "finished with failures in:$FAILED (see the summary above); re-run to retry what is missing"
  exit 1
fi
