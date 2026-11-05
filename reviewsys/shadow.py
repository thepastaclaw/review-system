"""`reviewsys shadow`: replay live production reviews under policy v12 as they finish, beside
production, and compare the two head by head (DESIGN.md section 4, rollout step 4: a live
shadow, sampled, at the lowest rank, never publishing).

Every shadowed head is one `reviewsys replay` (replay.py) of a finished production run, in its
own process group: a scratch database seeded as of the run's start, the worker's dry run, the
pinned read-only gh broker and git pinned to the cutoff. Nothing reaches GitHub but reads
through the broker, and the production database is only ever opened read-only. Replay lanes
wait in `<pool>.replay/` and take a model slot only when no production lane waits
(lanepool.py), so a shadow never slows production.

State lives under the shadow root (default ~/.reviewsys-shadow): `shadow.db` (its own SQLite
database, schema below), `replays/run-<id>/` (each replay's report.json and artifacts),
`logs/` (one log per replay), `mirrors/` (repositories the box has no mirror of), and the
runner's lock and pidfile.

A head's life (`shadows.status`):
- discovered: a production run qualifies when it is `done`, a live review (no audit), not
  degraded, not a conversation or a carried head, and published a final review (a `reviews`
  row, phase `final`, not imported). Its head must still be the PR's newest. Runs that
  finished before the watermark (the first start, or `--since`) or more than
  `max_age_hours` ago are never looked at; `sample_rate` keeps a deterministic share (hashed
  on the run id). Every qualifying run gets exactly one row (`source_run` is unique): never
  shadowed twice. Rows the sample leaves out are `skipped` (`not sampled`).
- `queued` -> `waiting` (a launch gate holds it: reason `daily cap`, `accounts`, `backlog`,
  `disk`, `production unreadable`, `quota guard`) -> `running` -> `done` | `failed` |
  `skipped`. Newest first; a head no longer the PR's newest is `skipped` (`superseded`), one
  older than `max_age_hours` at launch time `skipped` (`aged out`), any while degraded mode is
  on `skipped` (`degraded`).
- Gates before every launch, in order: at most `max_heads_per_day` launches per UTC day; the
  production checks of replayset (degraded, usable OpenAI accounts, backlog) and the disk
  floor; then `guard_cmd`, a shell command whose exit 0 allows the launch (its one-line JSON
  stdout is logged). `stop_on_guard` terminates running replays after two failed guard
  checks in a row (they go back to the queue).
- done: the replay's report is compared with what production published for the same run
  (`compare`), and later `score` reads what happened to the PR since (outcomes).

Public vs private: shadow.db holds finding titles (the `findings` table) and failure text
(`shadows.detail`) and stays on the box. `public_export` (the status dashboard) gives only
numbers, verdicts, repo/number/sha/PR title, links and reasons from `REASONS`.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
import signal
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import IO, Any, Protocol

from .db import fmt_ts, kv_get, kv_set, now_dt, parse_ts, tx
from .exporter import SLOT_WAIT_RECORDED_SINCE
from .lanepool import pool_of
from .proc import alive
from .replay import EXIT_ABORTED, TOKEN_ENV, _has_table, read_json
from .replayscore import assign, matches
from .replayset import DEFAULT_JOB_GB, DEFAULT_MIN_FREE_GB, Gate, production_guard

log = logging.getLogger("reviewsys.shadow")

DEFAULT_ROOT = Path("~/.reviewsys-shadow")
ROOT_ENV = "REVIEWSYS_SHADOW_ROOT"
DB_NAME = "shadow.db"
POLICY_NAME = "v12"
SEVERITIES = ("blocking", "suggestion", "nitpick")
VERDICTS = ("APPROVE", "REQUEST_CHANGES", "COMMENT")
_STATE_EVENT = {"APPROVED": "APPROVE", "CHANGES_REQUESTED": "REQUEST_CHANGES",
                "COMMENTED": "COMMENT"}  # fmt: skip
LIFTING = frozenset({"WITHDRAWN", "INTENTIONALLY_DEFERRED"})
LEDGER_LIFTED = frozenset({"withdrawn", "deferred"})
GUARD_FAILS_TO_STOP = 2
GUARD_TIMEOUT_S = 120

# every reason a row or the runner can carry; the public export shows nothing else
REASONS = frozenset(
    {
        "not sampled",
        "superseded",
        "aged out",
        "degraded",
        "daily cap",
        "accounts",
        "backlog",
        "disk",
        "production unreadable",
        "quota guard",
        "interrupted",
        "timed out",
        "no report",
        "run failed",
        "invalid: live github",
        "invalid: quota",
        "launch error",
        "compare error",
    }
)

SCHEMA = (
    """
    CREATE TABLE shadows (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_run INTEGER NOT NULL UNIQUE,
        repo TEXT NOT NULL, number INTEGER NOT NULL, sha TEXT NOT NULL,
        title TEXT NOT NULL DEFAULT '', tier TEXT,
        source_started_at TEXT NOT NULL, source_finished_at TEXT NOT NULL,
        discovered_at TEXT NOT NULL,
        status TEXT NOT NULL, reason TEXT, detail TEXT,
        attempts INTEGER NOT NULL DEFAULT 0, pid INTEGER,
        started_at TEXT, finished_at TEXT, exit_code INTEGER,
        report_path TEXT, log_path TEXT, policy TEXT, policy_fingerprint TEXT,
        compared_at TEXT, prod_verdict TEXT, shadow_verdict TEXT,
        prod_blocking INTEGER, prod_suggestion INTEGER, prod_nitpick INTEGER,
        shadow_blocking INTEGER, shadow_suggestion INTEGER, shadow_nitpick INTEGER,
        matched INTEGER, prod_only INTEGER, shadow_only INTEGER,
        prod_wall_s REAL, prod_active_s REAL, shadow_wall_s REAL, shadow_active_s REAL,
        prod_gpt_lanes REAL, prod_gpt_tokens INTEGER, prod_tokens INTEGER,
        shadow_gpt_lanes REAL, shadow_gpt_tokens INTEGER, shadow_tokens INTEGER,
        shadow_tier TEXT, v12 TEXT, scoring_degraded INTEGER NOT NULL DEFAULT 0,
        scored_at TEXT, pr_state TEXT, merged_at TEXT,
        caught_earlier INTEGER, prod_false_alarms INTEGER, prod_confirmed INTEGER)
    """,
    "CREATE INDEX shadows_status ON shadows (status, source_finished_at)",
    """
    CREATE TABLE findings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        shadow_id INTEGER NOT NULL REFERENCES shadows(id),
        side TEXT NOT NULL, match TEXT NOT NULL, pair INTEGER,
        hash TEXT NOT NULL DEFAULT '', severity TEXT, file TEXT, line_start INTEGER,
        line_end INTEGER, category TEXT, title TEXT NOT NULL DEFAULT '')
    """,
    "CREATE INDEX findings_shadow ON findings (shadow_id, side)",
    """
    CREATE TABLE outcomes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        shadow_id INTEGER NOT NULL REFERENCES shadows(id),
        kind TEXT NOT NULL, hash TEXT NOT NULL DEFAULT '', severity TEXT,
        in_shadow INTEGER, later_run INTEGER, later_sha TEXT, status TEXT, source TEXT,
        observed_at TEXT NOT NULL,
        UNIQUE (shadow_id, kind, hash))
    """,
    """
    CREATE TABLE events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, kind TEXT NOT NULL,
        shadow_id INTEGER, detail TEXT)
    """,
    "CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
)
SCHEMA_VERSION = 1


# ---- the shadow database ----


def connect(path: Path) -> sqlite3.Connection:
    """The shadow database, created on first use. Never the production one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # one runner thread uses it at a time; `check_same_thread` only stops a stop-signal test
    # (or a caller) from handing the runner to another thread
    conn = sqlite3.connect(str(path), timeout=30, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    with tx(conn):
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            for stmt in SCHEMA:
                conn.execute(stmt)
            conn.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
    return conn


def connect_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _event(
    conn: sqlite3.Connection, kind: str, detail: str = "", shadow_id: int | None = None
) -> None:
    conn.execute(
        "INSERT INTO events (ts, kind, shadow_id, detail) VALUES (?,?,?,?)",
        (fmt_ts(now_dt()), kind, shadow_id, detail[:4000]),
    )


def sampled(run_id: int, rate: float) -> bool:
    """Whether a run is in the sample: deterministic in the run id, so a restart (or a second
    look at the same run) decides the same."""
    if rate >= 1:
        return True
    if rate <= 0:
        return False
    digest = hashlib.sha256(f"reviewsys-shadow:{run_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < rate


def canonical(event: str | None) -> str | None:
    return _STATE_EVENT.get(str(event), str(event)) if event else None


# ---- options ----


@dataclass(frozen=True, slots=True)
class Options:
    root: Path
    skills: Path
    prod_db: Path
    config: str | None = None
    sample_rate: float = 1.0
    max_age_hours: float = 24.0  # <= 0: no limit
    concurrency: int = 2
    max_heads_per_day: int = 40  # <= 0: no cap
    max_attempts: int = 2
    replay_timeout_hours: float = 8.0
    min_usable_accounts: int = 2
    backlog_limit: int = 10  # the box's backlog_skip_phase1_above
    min_free_gb: float = DEFAULT_MIN_FREE_GB
    disk_per_job_gb: float = DEFAULT_JOB_GB
    guard_cmd: str = ""
    stop_on_guard: bool = False
    gh_token_env: str = TOKEN_ENV
    run_timeout_minutes: int = 180
    fake_lanes: bool = False
    keep_artifacts_days: float = 14.0  # <= 0: keep forever
    tick_seconds: float = 60.0
    score_every_seconds: float = 3600.0
    score_days: float = 30.0

    @property
    def db_path(self) -> Path:
        return self.root / DB_NAME

    @property
    def replays(self) -> Path:
        return self.root / "replays"

    def report(self, run_id: int) -> Path:
        return self.replays / f"run-{run_id}" / "report.json"

    def log(self, run_id: int) -> Path:
        return self.root / "logs" / f"replay-run-{run_id}.log"


class Handle(Protocol):
    pid: int

    def poll(self) -> int | None: ...


Launcher = Callable[[Options, sqlite3.Row], Handle]
GateFn = Callable[[int], tuple[str, str, str]]  # running -> (go|wait|skip, reason, detail)
GuardFn = Callable[[], tuple[bool, str]]  # -> (allowed, one-line stdout)


def replay_argv(opts: Options, run_id: int) -> list[str]:
    argv = [sys.executable, "-m", "reviewsys.cli"]
    if opts.config:
        argv += ["--config", opts.config]
    argv += [
        "replay",
        str(run_id),
        "--out",
        str(opts.replays),
        "--skills",
        str(opts.skills),
        "--policy-name",
        POLICY_NAME,
        "--gh-token-env",
        opts.gh_token_env,
        "--eval-mirrors",
        str(opts.root / "mirrors"),
    ]
    if opts.fake_lanes:
        argv.append("--fake-lanes")
    return argv


def subprocess_launcher(opts: Options, row: sqlite3.Row) -> Handle:
    """`reviewsys replay` in its own session (its own process group: lanes included), output
    to the head's log."""
    argv = replay_argv(opts, int(row["source_run"]))
    path = opts.log(int(row["source_run"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as fh:
        fh.write(f"# {fmt_ts(now_dt())} {' '.join(argv)}\n".encode())
        fh.flush()
        return subprocess.Popen(
            argv,
            cwd=str(opts.root),
            stdin=subprocess.DEVNULL,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def command_guard(cmd: str) -> GuardFn:
    """`cmd` through the shell: exit 0 allows a launch. Its first stdout line (one-line JSON
    by convention) is what the runner logs."""

    def check() -> tuple[bool, str]:
        try:
            r = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=GUARD_TIMEOUT_S,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return False, json.dumps({"guard": "timed out", "after_s": GUARD_TIMEOUT_S})
        except OSError as exc:
            return False, json.dumps({"guard": "could not run", "error": str(exc)[:200]})
        line = (r.stdout.strip().splitlines() or [""])[0][:2000]
        if r.returncode != 0 and not line:
            line = json.dumps({"exit": r.returncode, "stderr": r.stderr.strip()[:300]})
        return r.returncode == 0, line

    return check


def production_gate(opts: Options) -> GateFn:
    """replayset's checks, sorted by what the shadow does: degraded mode skips the head (a
    replay on stand-ins would measure nothing), everything else holds launches."""
    disk = Gate(opts.replays, min_free_gb=opts.min_free_gb, job_gb=opts.disk_per_job_gb)

    def check(running: int) -> tuple[str, str, str]:
        # production_guard checks degraded mode first; its messages are matched here
        why, _ = production_guard(
            opts.prod_db, min_accounts=opts.min_usable_accounts, backlog_limit=opts.backlog_limit
        )
        if why:
            if "degraded" in why:
                return "skip", "degraded", why
            if "account" in why:
                return "wait", "accounts", why
            if "queued" in why:
                return "wait", "backlog", why
            return "wait", "production unreadable", why
        why, kind = disk.check(running)
        if kind:
            return "wait", "disk", why
        return "go", "", ""

    return check


# ---- discovery ----


def _later_head(prod: sqlite3.Connection, repo: str, number: int, run_id: int, sha: str) -> bool:
    """Whether the PR moved on from run `run_id`'s commit `sha`: the PR's recorded head is
    another commit, or a newer live head exists."""
    pr = prod.execute(
        "SELECT head_sha FROM prs WHERE repo=? AND number=?", (repo, number)
    ).fetchone()
    if pr is not None and pr["head_sha"] and pr["head_sha"] != sha:
        return True
    newer = prod.execute(
        "SELECT 1 FROM heads WHERE repo=? AND number=? AND queue='live' AND sha<>? "
        "AND id > (SELECT head_id FROM runs WHERE id=?) LIMIT 1",
        (repo, number, sha, run_id),
    ).fetchone()
    return newer is not None


def candidates(prod: sqlite3.Connection, since: str) -> list[sqlite3.Row]:
    """Production runs that qualify for a shadow (module docstring), finished at or after
    `since`, newest first. A run is an audit by the path it took (`runs.path`): an audit takes
    an uncleanly merged PR's live head over (`heads.queue` becomes `audit`) after its live
    runs finished. A carried head is left out by its run's `carry` step, not by
    `heads.carried_from`: a review someone asks for of a carried commit is a real review."""
    return prod.execute(
        "SELECT r.id, r.started_at, r.finished_at, r.tier, h.repo, h.number, "
        "h.sha, COALESCE(p.title, '') title FROM runs r JOIN heads h ON h.id=r.head_id "
        "LEFT JOIN prs p ON p.repo=h.repo AND p.number=h.number "
        "WHERE r.status='done' AND r.finished_at IS NOT NULL AND r.finished_at >= ? "
        "AND COALESCE(r.path, CASE WHEN h.queue='audit' THEN 'audit' END, '') <> 'audit' "
        "AND COALESCE(r.degraded, 0)=0 "
        "AND EXISTS (SELECT 1 FROM reviews v WHERE v.run_id=r.id AND v.phase='final' "
        "AND COALESCE(v.imported, 0)=0) "
        "AND NOT EXISTS (SELECT 1 FROM steps s WHERE s.run_id=r.id "
        "AND s.name IN ('converse', 'carry')) "
        "ORDER BY r.finished_at DESC, r.id DESC",
        (since,),
    ).fetchall()


# ---- comparison ----


def _prod_findings(prod: sqlite3.Connection, run_id: int) -> list[dict[str, Any]]:
    """What production's final review stood on: the run's `posted` rows of phase `final` (a
    publication's `kept`, as the replay's `published`), one per hash."""
    seen: dict[str, dict[str, Any]] = {}
    for r in prod.execute(
        "SELECT hash, severity, file, line_start, line_end, category, title FROM findings "
        "WHERE run_id=? AND stage='posted' AND phase='final' ORDER BY id",
        (run_id,),
    ):
        seen.setdefault(str(r["hash"]), dict(r))
    return list(seen.values())


def _prod_verdict(prod: sqlite3.Connection, run_id: int) -> str | None:
    row = prod.execute(
        "SELECT event FROM reviews WHERE run_id=? AND phase='final' AND COALESCE(imported,0)=0 "
        "ORDER BY id DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    return canonical(row["event"]) if row else None


def shadow_verdict(report: dict[str, Any]) -> str | None:
    pubs = report.get("publications") or {}
    pub = pubs.get("final") or pubs.get("preliminary") or {}
    return canonical(pub.get("event"))


def _sev(findings: list[dict[str, Any]]) -> dict[str, int]:
    c = Counter(str(f.get("severity") or "") for f in findings)
    return {s: c[s] for s in SEVERITIES}


def compare_findings(
    prod: list[dict[str, Any]], shadow: list[dict[str, Any]]
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    """(matched (prod, shadow) pairs, production-only, shadow-only), one to one, by the eval's
    matcher (same file within 10 lines and title overlap, or the same hash)."""
    got, shadow_only = assign(shadow, prod)
    pairs = [(prod[i], got[i]) for i in sorted(got)]
    prod_only = [f for i, f in enumerate(prod) if i not in got]
    return pairs, prod_only, shadow_only


def _prod_cost(
    prod: sqlite3.Connection, run_id: int, run_timeout_minutes: int
) -> dict[str, float | int | None]:
    gpt_lanes = gpt_tokens = tokens = 0
    for x in prod.execute(
        "SELECT model, COALESCE(tokens_in,0) ti, COALESCE(tokens_out,0) tout FROM lanes "
        "WHERE run_id=?",
        (run_id,),
    ):
        tokens += int(x["ti"]) + int(x["tout"])
        if pool_of(str(x["model"])) == "gpt":
            gpt_lanes += 1
            gpt_tokens += int(x["ti"]) + int(x["tout"])
    r = prod.execute(
        "SELECT started_at, finished_at, deadline_at FROM runs WHERE id=?", (run_id,)
    ).fetchone()
    wall = active = None
    if r is not None and r["started_at"] and r["finished_at"]:
        wall = (parse_ts(r["finished_at"]) - parse_ts(r["started_at"])).total_seconds()
        active = wall
        if r["deadline_at"] and str(r["started_at"]) >= SLOT_WAIT_RECORDED_SINCE:
            # lanepool pushes the deadline out by every stretch the run only waited for slots
            credited = (parse_ts(r["deadline_at"]) - parse_ts(r["started_at"])).total_seconds()
            active = max(0.0, wall - max(0.0, credited - run_timeout_minutes * 60))
    return {"gpt_lanes": gpt_lanes, "gpt_tokens": gpt_tokens, "tokens": tokens,
            "wall_s": wall, "active_s": active}  # fmt: skip


def compare(
    conn: sqlite3.Connection,
    prod: sqlite3.Connection,
    shadow_id: int,
    source_run: int,
    report: dict[str, Any],
    *,
    run_timeout_minutes: int = 180,
) -> dict[str, Any]:
    """Store the head's comparison (columns of `shadows`, rows of `findings`); returns it."""
    pf = _prod_findings(prod, source_run)
    sf = [f for f in report.get("published") or [] if isinstance(f, dict)]
    pairs, prod_only, shadow_only = compare_findings(pf, sf)
    cost = _prod_cost(prod, source_run, run_timeout_minutes)
    gpt = report.get("gpt_excl_reconcile") or report.get("gpt") or {}
    ps, ss = _sev(pf), _sev(sf)
    row = {
        "compared_at": fmt_ts(now_dt()),
        "prod_verdict": _prod_verdict(prod, source_run),
        "shadow_verdict": shadow_verdict(report),
        **{f"prod_{k}": v for k, v in ps.items()},
        **{f"shadow_{k}": v for k, v in ss.items()},
        "matched": len(pairs),
        "prod_only": len(prod_only),
        "shadow_only": len(shadow_only),
        "prod_wall_s": cost["wall_s"],
        "prod_active_s": cost["active_s"],
        "shadow_wall_s": report.get("wall_clock_s"),
        "shadow_active_s": report.get("active_s"),
        "prod_gpt_lanes": cost["gpt_lanes"],
        "prod_gpt_tokens": cost["gpt_tokens"],
        "prod_tokens": cost["tokens"],
        "shadow_gpt_lanes": gpt.get("lanes"),
        "shadow_gpt_tokens": int(gpt.get("tokens_in") or 0) + int(gpt.get("tokens_out") or 0),
        "shadow_tokens": int(report.get("tokens_in") or 0) + int(report.get("tokens_out") or 0),
        "shadow_tier": report.get("tier"),
        "policy": report.get("policy"),
        "policy_fingerprint": report.get("policy_fingerprint"),
        "v12": json.dumps(report.get("v12") or {}),
        # a thread-lane shard failed: some issues were never judged (replayscore leaves such
        # heads out; so does the summary)
        "scoring_degraded": int(bool(report.get("degraded_for_scoring"))),
    }
    keys = ("hash", "severity", "file", "line_start", "line_end", "category", "title")

    def finding(side: str, match: str, pair: int | None, f: dict[str, Any]) -> tuple[Any, ...]:
        return (shadow_id, side, match, pair, *(f.get(k) for k in keys))

    rows = []
    for n, (p, s) in enumerate(pairs):
        rows += [finding("prod", "matched", n, p), finding("shadow", "matched", n, s)]
    rows += [finding("prod", "only", None, f) for f in prod_only]
    rows += [finding("shadow", "only", None, f) for f in shadow_only]
    with tx(conn):
        conn.execute(
            f"UPDATE shadows SET {', '.join(f'{k}=?' for k in row)} WHERE id=?",
            (*row.values(), shadow_id),
        )
        conn.execute("DELETE FROM findings WHERE shadow_id=?", (shadow_id,))
        conn.executemany(
            "INSERT INTO findings (shadow_id, side, match, pair, hash, severity, file, "
            "line_start, line_end, category, title) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
    return row


# ---- outcomes ----


def _settlement(prod: sqlite3.Connection, repo: str, number: int, h: str) -> str | None:
    """The finding's standing lift (worker._latest_settlement on main): its newest row among
    conceded/verified/posted on the PR is a `conceded` one whose status is a judgement lift."""
    row = prod.execute(
        "SELECT f.stage, f.body FROM findings f JOIN runs r ON r.id=f.run_id "
        "JOIN heads h ON h.id=r.head_id WHERE h.repo=? AND h.number=? AND f.hash=? "
        "AND f.stage IN ('conceded','verified','verified-fresh','posted') "
        "ORDER BY r.id DESC, f.id DESC LIMIT 1",
        (repo, number, h),
    ).fetchone()
    if row is None or row["stage"] != "conceded":
        return None
    # main writes the status as the body's first line; older rows (this branch) leave the body
    # empty, and a conversation records a `conceded` row only for a judgement lift
    status = str(row["body"] or "").split("\n", 1)[0].strip()
    if not status:
        return "CONCEDED"
    return status if status in LIFTING else None


def _answers(prod: sqlite3.Connection, repo: str, number: int, since: str) -> dict[str, str]:
    """finding hash -> the last status a delivered thread answer gave it after `since`."""
    out: dict[str, str] = {}
    for e in prod.execute(
        "SELECT detail FROM events WHERE kind='thread.answered' AND repo=? AND number=? "
        "AND ts >= ? ORDER BY id",
        (repo, number, since),
    ):
        try:
            d = json.loads(e["detail"] or "{}")
        except ValueError:
            continue
        if not isinstance(d, dict) or not d.get("finding_hash"):
            continue
        if d.get("action") not in ("replied", "already_answered", "status_cleared", "no_thread"):
            continue
        out[str(d["finding_hash"])] = str(d.get("status") or "")
    return out


def _ledger_status(prod: sqlite3.Connection, repo: str, number: int) -> dict[str, str]:
    if not _has_table(prod, "issues"):
        return {}
    return {
        str(r["hash"]): str(r["status"])
        for r in prod.execute(
            "SELECT hash, status FROM issues WHERE repo=? AND number=?", (repo, number)
        )
    }


def score_head(
    conn: sqlite3.Connection, prod: sqlite3.Connection, s: sqlite3.Row
) -> dict[str, int]:
    """What happened on the PR since the shadowed run (module docstring: outcomes); replaces
    the head's outcome rows, keeping the first time each was seen."""
    repo, number, src = str(s["repo"]), int(s["number"]), int(s["source_run"])
    since = str(s["source_started_at"])
    found: dict[tuple[str, str], dict[str, Any]] = {}
    mine = conn.execute("SELECT * FROM findings WHERE shadow_id=?", (s["id"],)).fetchall()
    later = [
        dict(r)
        for r in prod.execute(
            "SELECT f.hash, f.file, f.line_start, f.line_end, f.category, f.title, f.severity, "
            "r.id run_id, h.sha FROM findings f JOIN runs r ON r.id=f.run_id "
            "JOIN heads h ON h.id=r.head_id WHERE h.repo=? AND h.number=? AND r.id<>? "
            "AND r.started_at > ? AND f.stage='posted' ORDER BY r.id",
            (repo, number, src, since),
        )
    ]
    for f in (dict(x) for x in mine if x["side"] == "shadow" and x["match"] == "only"):
        hit = next((g for g in later if matches(f, g)), None)
        if hit is not None:
            found[("caught_earlier", str(f["hash"]))] = {
                "severity": f["severity"], "in_shadow": 1, "later_run": hit["run_id"],
                "later_sha": hit["sha"], "status": None, "source": "posted",
            }  # fmt: skip
    answers = _answers(prod, repo, number, since)
    led = _ledger_status(prod, repo, number)
    for f in (dict(x) for x in mine if x["side"] == "prod"):
        h = str(f["hash"])
        in_shadow = int(f["match"] == "matched")
        settled = _settlement(prod, repo, number, h)
        status, source = (settled, "conceded") if settled else ("", "")
        if not status and answers.get(h) in LIFTING:
            status, source = answers[h], "thread"
        if not status and led.get(h) in LEDGER_LIFTED:
            status, source = led[h], "ledger"
        if status:
            found[("prod_false_alarm", h)] = {
                "severity": f["severity"], "in_shadow": in_shadow, "later_run": None,
                "later_sha": None, "status": status, "source": source,
            }  # fmt: skip
            continue
        if f["severity"] == "blocking" and (answers.get(h) == "FIXED" or led.get(h) == "fixed"):
            found[("prod_confirmed", h)] = {
                "severity": f["severity"], "in_shadow": in_shadow, "later_run": None,
                "later_sha": None, "status": "FIXED",
                "source": "thread" if answers.get(h) == "FIXED" else "ledger",
            }  # fmt: skip
    pr = prod.execute("SELECT state FROM prs WHERE repo=? AND number=?", (repo, number)).fetchone()
    merged_at = None
    if _has_table(prod, "audits"):
        a = prod.execute(
            "SELECT merged_at FROM audits WHERE repo=? AND number=?", (repo, number)
        ).fetchone()
        merged_at = a["merged_at"] if a is not None and a["merged_at"] else None
    # `prs.state` is only open/closed; a merge is known only from an audit row (a PR merged
    # over findings), never for a clean merge: `pr_closed` covers both, its status says
    # `merged` when known
    state = "merged" if merged_at else (str(pr["state"]) if pr is not None else None)
    if state in ("merged", "closed"):
        found[("pr_closed", "")] = {"severity": None, "in_shadow": None, "later_run": None,
                                    "later_sha": None, "status": state, "source": "prs"}  # fmt: skip
    at = fmt_ts(now_dt())
    counts = Counter(kind for kind, _ in found)
    with tx(conn):
        keep = conn.execute(
            "SELECT kind, hash FROM outcomes WHERE shadow_id=?", (s["id"],)
        ).fetchall()
        for k in keep:
            if (k["kind"], k["hash"]) not in found:
                conn.execute(
                    "DELETE FROM outcomes WHERE shadow_id=? AND kind=? AND hash=?",
                    (s["id"], k["kind"], k["hash"]),
                )
        for (kind, h), d in found.items():
            conn.execute(
                "INSERT INTO outcomes (shadow_id, kind, hash, severity, in_shadow, later_run, "
                "later_sha, status, source, observed_at) VALUES (?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(shadow_id, kind, hash) DO UPDATE SET severity=excluded.severity, "
                "in_shadow=excluded.in_shadow, later_run=excluded.later_run, "
                "later_sha=excluded.later_sha, status=excluded.status, source=excluded.source",
                (s["id"], kind, h, d["severity"], d["in_shadow"], d["later_run"],
                 d["later_sha"], d["status"], d["source"], at),
            )  # fmt: skip
        conn.execute(
            "UPDATE shadows SET scored_at=?, pr_state=?, merged_at=?, caught_earlier=?, "
            "prod_false_alarms=?, prod_confirmed=? WHERE id=?",
            (at, state, merged_at, counts["caught_earlier"], counts["prod_false_alarm"],
             counts["prod_confirmed"], s["id"]),
        )  # fmt: skip
    return {
        "caught_earlier": counts["caught_earlier"],
        "prod_false_alarms": counts["prod_false_alarm"],
        "prod_confirmed": counts["prod_confirmed"],
    }


def score(conn: sqlite3.Connection, prod_db: Path, *, days: float = 30.0) -> dict[str, int]:
    """Score every compared head discovered in the last `days` (all with days <= 0)."""
    since = fmt_ts(now_dt() - timedelta(days=days)) if days > 0 else ""
    prod = connect_ro(prod_db)
    totals: Counter[str] = Counter()
    try:
        for s in conn.execute(
            "SELECT * FROM shadows WHERE status='done' AND compared_at IS NOT NULL "
            "AND discovered_at >= ? ORDER BY id",
            (since,),
        ).fetchall():
            totals.update(score_head(conn, prod, s))
            totals["heads"] += 1
    finally:
        prod.close()
    return dict(totals)


# ---- the runner ----


@dataclass
class Runner:
    """Discovers, launches, reaps and compares (module docstring). One per root: `lock()`."""

    opts: Options
    conn: sqlite3.Connection
    launcher: Launcher = subprocess_launcher
    gate: GateFn | None = None
    guard: GuardFn | None = None
    clock: Callable[[], datetime] = now_dt
    kill_grace: float = 30.0
    running: dict[int, Handle] = field(default_factory=dict)  # shadow id -> process
    guard_fails: int = 0
    _guard_last: tuple[bool, str] | None = None
    _pidfile: bool = False
    _lock: IO[str] | None = None

    def __post_init__(self) -> None:
        if self.gate is None:
            self.gate = production_gate(self.opts)
        if self.guard is None and self.opts.guard_cmd:
            self.guard = command_guard(self.opts.guard_cmd)

    # -- bookkeeping --

    def now(self) -> str:
        return fmt_ts(self.clock())

    def set_watermark(self, since: str | None) -> str:
        """The earliest finish a run may have to be shadowed: `since` when given, else the
        first start's time (no backfill)."""
        with tx(self.conn):
            if since:
                kv_set(self.conn, "watermark", since)
            elif kv_get(self.conn, "watermark") is None:
                kv_set(self.conn, "watermark", self.now())
        return str(kv_get(self.conn, "watermark"))

    def _state(self, state: str, reason: str = "", detail: str = "") -> None:
        with tx(self.conn):
            kv_set(self.conn, "runner.state", json.dumps(
                {"state": state, "reason": reason, "detail": detail[:2000], "at": self.now()}
            ))  # fmt: skip

    def _set(self, shadow_id: int, **cols: Any) -> None:
        with tx(self.conn):
            self.conn.execute(
                f"UPDATE shadows SET {', '.join(f'{k}=?' for k in cols)} WHERE id=?",
                (*cols.values(), shadow_id),
            )

    def _oldest_allowed(self) -> str:
        wm = str(kv_get(self.conn, "watermark") or self.now())
        if self.opts.max_age_hours > 0:
            age = fmt_ts(self.clock() - timedelta(hours=self.opts.max_age_hours))
            return max(wm, age)
        return wm

    # -- discovery --

    def discover(self) -> dict[str, int]:
        """Book every newly qualifying production run: queued, or skipped (not sampled,
        superseded)."""
        counts: Counter[str] = Counter()
        prod = connect_ro(self.opts.prod_db)
        try:
            known = {int(r[0]) for r in self.conn.execute("SELECT source_run FROM shadows")}
            for c in candidates(prod, self._oldest_allowed()):
                if int(c["id"]) in known:
                    continue
                status, reason = "queued", None
                if not sampled(int(c["id"]), self.opts.sample_rate):
                    status, reason = "skipped", "not sampled"
                elif _later_head(prod, c["repo"], c["number"], int(c["id"]), c["sha"]):
                    status, reason = "skipped", "superseded"
                with tx(self.conn):
                    self.conn.execute(
                        "INSERT OR IGNORE INTO shadows (source_run, repo, number, sha, title, "
                        "tier, source_started_at, source_finished_at, discovered_at, status, "
                        "reason) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (c["id"], c["repo"], c["number"], c["sha"], c["title"], c["tier"],
                         c["started_at"], c["finished_at"], self.now(), status, reason),
                    )  # fmt: skip
                counts[reason or status] += 1
        finally:
            prod.close()
        return dict(counts)

    # -- launching --

    def launched_today(self) -> int:
        """Replays launched this UTC day, every attempt (a stopped one spent quota too)."""
        day = self.now()[:10]
        return int(
            self.conn.execute(
                "SELECT COUNT(*) FROM events WHERE kind='launch' AND ts >= ?", (day,)
            ).fetchone()[0]
        )

    def _check_guard(self) -> bool:
        if self.guard is None:
            return True
        ok, line = self.guard()
        self.guard_fails = 0 if ok else self.guard_fails + 1
        if (ok, line) != self._guard_last:
            self._guard_last = (ok, line)
            with tx(self.conn):
                _event(self.conn, "guard.ok" if ok else "guard.refused", line)
            log.info("shadow: quota guard %s: %s", "allows" if ok else "refuses", line)
        return ok

    def _next(self) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self.conn.execute(
            "SELECT * FROM shadows WHERE status IN ('queued','waiting') "
            "ORDER BY source_finished_at DESC, source_run DESC LIMIT 1"
        ).fetchone()
        return row

    def launch(self) -> dict[str, int]:
        """Start queued heads, newest first, while slots are free and every gate allows."""
        counts: Counter[str] = Counter()
        guard_checked = False
        with tx(self.conn):
            aged = self.conn.execute(
                "UPDATE shadows SET status='skipped', reason='aged out', finished_at=? "
                "WHERE status IN ('queued','waiting') AND source_finished_at < ?",
                (self.now(), self._oldest_allowed()),
            ).rowcount
        if aged:
            counts["aged out"] += aged
        prod = connect_ro(self.opts.prod_db)
        try:
            while len(self.running) < max(1, self.opts.concurrency):
                row = self._next()
                if row is None:
                    self._state("idle" if not self.running else "running")
                    break
                sid = int(row["id"])
                if _later_head(prod, row["repo"], row["number"], int(row["source_run"]),
                               row["sha"]):  # fmt: skip
                    self._set(sid, status="skipped", reason="superseded", finished_at=self.now())
                    counts["superseded"] += 1
                    continue
                cap = self.opts.max_heads_per_day
                if cap > 0 and self.launched_today() >= cap:
                    self._hold(sid, "daily cap", f"{cap} heads launched today (UTC)")
                    counts["daily cap"] += 1
                    break
                assert self.gate is not None
                action, reason, why = self.gate(len(self.running))
                if action == "skip":
                    self._set(sid, status="skipped", reason=reason, detail=why,
                              finished_at=self.now())  # fmt: skip
                    counts[reason] += 1
                    continue
                if action == "wait":
                    self._hold(sid, reason, why)
                    counts[reason] += 1
                    break
                guard_checked = True
                if not self._check_guard():
                    self._hold(sid, "quota guard", "")
                    counts["quota guard"] += 1
                    break
                if self._start(row):
                    counts["launched"] += 1
        finally:
            prod.close()
        if not guard_checked and self.running and self.opts.stop_on_guard and self.guard:
            # with every slot taken nothing launches, but a stop needs the guard's word
            self._check_guard()
        if self.opts.stop_on_guard and self.guard_fails >= GUARD_FAILS_TO_STOP and self.running:
            n = self.stop_all(reason="quota guard")
            counts["stopped by guard"] += n
        return dict(counts)

    def _hold(self, sid: int, reason: str, why: str) -> None:
        self._set(sid, status="waiting", reason=reason)
        self._state("waiting", reason, why)

    def _start(self, row: sqlite3.Row) -> bool:
        sid, rid = int(row["id"]), int(row["source_run"])
        report = self.opts.report(rid)
        report.unlink(missing_ok=True)  # a stale report would read as this attempt's
        try:
            handle = self.launcher(self.opts, row)
        except Exception as exc:  # a launch that cannot start fails its head only
            self._set(sid, status="failed", reason="launch error", detail=str(exc)[:2000],
                      finished_at=self.now())  # fmt: skip
            return False
        try:
            with tx(self.conn):
                self.conn.execute(
                    "UPDATE shadows SET status='running', reason=NULL, attempts=attempts+1, "
                    "pid=?, started_at=?, finished_at=NULL, report_path=?, log_path=? WHERE id=?",
                    (handle.pid, self.now(), str(report), str(self.opts.log(rid)), sid),
                )
                _event(self.conn, "launch", str(rid), sid)
        except BaseException:
            # unbooked, the head would be launched again next tick beside this replay
            kill_groups([(handle.pid, handle.poll)], grace=self.kill_grace)
            raise
        self.running[sid] = handle
        self._state("running")
        log.info("shadow: launched run %s (%s#%s) pid %s", rid, row["repo"], row["number"],
                 handle.pid)  # fmt: skip
        return True

    # -- reaping --

    def reap(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        limit = self.opts.replay_timeout_hours * 3600
        for sid, handle in list(self.running.items()):
            rc = handle.poll()
            row = self.conn.execute("SELECT * FROM shadows WHERE id=?", (sid,)).fetchone()
            if rc is None:
                started = parse_ts(str(row["started_at"]))
                if limit > 0 and (self.clock() - started).total_seconds() > limit:
                    kill_groups([(handle.pid, handle.poll)], grace=self.kill_grace)
                    self._cleanup(int(row["source_run"]))
                    self._set(sid, status="failed", reason="timed out", finished_at=self.now())
                    self.running.pop(sid)
                    counts["timed out"] += 1
                continue
            counts[self._finish_safely(row, rc)] += 1
            self.running.pop(sid)
        return dict(counts)

    def _finish_safely(self, row: sqlite3.Row, rc: int) -> str:
        try:
            return self.finish(row, rc)
        except Exception as exc:
            log.exception("shadow: booking run %s failed", row["source_run"])
            self._set(int(row["id"]), status="failed", reason="compare error",
                      detail=repr(exc)[:2000], exit_code=rc, finished_at=self.now())  # fmt: skip
            return "compare error"

    def finish(self, row: sqlite3.Row, rc: int) -> str:
        """Book a replay that exited with `rc`: compare it, or say why not."""
        sid, rid = int(row["id"]), int(row["source_run"])
        self._cleanup(rid)
        rep = read_json(self.opts.report(rid))
        at = self.now()
        if not isinstance(rep, dict):
            if rc == EXIT_ABORTED:  # refused to start: degraded mode
                self._set(sid, status="skipped", reason="degraded", exit_code=rc, finished_at=at)
                return "degraded"
            self._set(sid, status="failed", reason="no report", exit_code=rc, finished_at=at,
                      detail=_tail(self.opts.log(rid)))  # fmt: skip
            return "no report"
        if rep.get("invalid"):
            reason = "invalid: quota" if str(rep["invalid"]).startswith("quota") else (
                "invalid: live github")  # fmt: skip
            self._set(sid, status="failed", reason=reason, exit_code=rc, finished_at=at,
                      detail=str(rep.get("invalid")))  # fmt: skip
            return reason
        if rep.get("status") != "done":
            self._set(sid, status="failed", reason="run failed", exit_code=rc, finished_at=at,
                      detail=f"{rep.get('fail_kind') or ''}: {rep.get('reason') or ''}"[:2000],
                      shadow_wall_s=rep.get("wall_clock_s"))  # fmt: skip
            return "run failed"
        prod = connect_ro(self.opts.prod_db)
        try:
            compare(self.conn, prod, sid, rid, rep,
                    run_timeout_minutes=self.opts.run_timeout_minutes)  # fmt: skip
        finally:
            prod.close()
        self._set(sid, status="done", reason=None, exit_code=rc, finished_at=at)
        return "done"

    def _cleanup(self, run_id: int) -> None:
        """What a replay removes itself when it ends normally (its pinned clone and
        worktrees), for one that was killed."""
        work = self.opts.replays / f"run-{run_id}" / "work"
        for d in ("worktrees", "mirrors"):
            shutil.rmtree(work / d, ignore_errors=True)

    def stop_all(self, *, reason: str = "interrupted") -> int:
        """Terminate every running replay's process group and put its head back in the queue
        (the attempt does not count)."""
        n = 0
        for sid, handle in list(self.running.items()):
            rc = handle.poll()
            if rc is not None:  # it ended since the last reap: book it, never run it again
                row = self.conn.execute("SELECT * FROM shadows WHERE id=?", (sid,)).fetchone()
                self._finish_safely(row, rc)
                self.running.pop(sid)
        kill_groups([(h.pid, h.poll) for h in self.running.values()], grace=self.kill_grace)
        for sid in list(self.running):
            self.running.pop(sid)
            row = self.conn.execute("SELECT * FROM shadows WHERE id=?", (sid,)).fetchone()
            self._cleanup(int(row["source_run"]))
            self._set(sid, status="queued", reason=reason, pid=None, started_at=None,
                      attempts=max(0, int(row["attempts"]) - 1))  # fmt: skip
            n += 1
        return n

    def recover(self) -> int:
        """After a runner that died without stopping its replays: kill what is left of them
        (only processes that are still that replay) and queue their heads again, or fail them
        after `max_attempts`."""
        n = 0
        for row in self.conn.execute("SELECT * FROM shadows WHERE status='running'").fetchall():
            sid, rid, pid = int(row["id"]), int(row["source_run"]), int(row["pid"] or 0)
            if sid in self.running:
                continue
            n += 1
            mine = _is_replay(pid, rid) if pid and alive(pid) else False
            if mine is None:
                # alive, and nothing can say whether it is still that replay: never signal it,
                # never run the head again beside it
                self._set(sid, status="failed", reason="interrupted", pid=None,
                          finished_at=self.now(),
                          detail=f"pid {pid} alive but unverifiable at recovery")  # fmt: skip
                continue
            if mine:
                kill_groups([(pid, None)], grace=self.kill_grace)
            elif self.opts.report(rid).exists():  # it finished while no runner watched
                self._finish_safely(row, 0)
                continue
            self._cleanup(rid)
            if int(row["attempts"]) >= self.opts.max_attempts:
                self._set(sid, status="failed", reason="interrupted", pid=None,
                          finished_at=self.now())  # fmt: skip
            else:
                self._set(sid, status="queued", reason="interrupted", pid=None)
        return n

    def prune(self) -> int:
        """Lane artifacts and scratch databases of heads finished more than
        `keep_artifacts_days` ago; report.json stays."""
        if self.opts.keep_artifacts_days <= 0:
            return 0
        before = fmt_ts(self.clock() - timedelta(days=self.opts.keep_artifacts_days))
        n = 0
        for row in self.conn.execute(
            "SELECT source_run FROM shadows WHERE status IN ('done','failed') AND finished_at < ?",
            (before,),
        ):
            d = self.opts.replays / f"run-{row['source_run']}"
            for sub in ("work", "bin", "gh-config"):
                if (d / sub).exists():
                    shutil.rmtree(d / sub, ignore_errors=True)
                    n += 1
            self.opts.log(int(row["source_run"])).unlink(missing_ok=True)
        return n

    def tick(self) -> dict[str, Any]:
        return {"reaped": self.reap(), "discovered": self.discover(), "launched": self.launch()}

    def wait_all(self, poll: float = 5.0) -> dict[str, int]:
        """Reap until nothing runs (`shadow tick`: launched replays are seen to the end)."""
        counts: Counter[str] = Counter()
        while self.running:
            counts.update(self.reap())
            if self.running:
                time.sleep(poll)
        return dict(counts)

    # -- the long-running loop --

    def lock(self, *, pidfile: bool = True) -> None:
        """One runner per root: an exclusive flock (released by the kernel when the process
        dies, so a killed runner never blocks the next) and, for `shadow run`, a pidfile."""
        import fcntl

        self.opts.root.mkdir(parents=True, exist_ok=True)
        fd = open(self.opts.root / "shadow.lock", "a+")  # noqa: SIM115 - held for life
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fd.close()
            raise SystemExit(
                f"another shadow runner holds {self.opts.root / 'shadow.lock'}"
            ) from None
        self._lock = fd
        self._pidfile = pidfile
        if pidfile:
            (self.opts.root / "shadow.pid").write_text(f"{os.getpid()}\n")

    def unlock(self) -> None:
        if self._pidfile:
            with contextlib.suppress(OSError):
                (self.opts.root / "shadow.pid").unlink()
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def run_forever(self, stop: threading.Event) -> None:
        self.recover()
        last_score = last_prune = 0.0
        try:
            while not stop.is_set():
                try:
                    self.tick()
                except Exception:  # a bad tick is logged; the loop goes on
                    log.exception("shadow: tick failed")
                mono = time.monotonic()
                if mono - last_score >= self.opts.score_every_seconds:
                    last_score = mono
                    try:
                        log.info(
                            "shadow: scored %s",
                            score(self.conn, self.opts.prod_db, days=self.opts.score_days),
                        )
                    except Exception:
                        log.exception("shadow: scoring failed")
                if mono - last_prune >= 86400:
                    last_prune = mono
                    self.prune()
                stop.wait(self.opts.tick_seconds)
        finally:
            n = self.stop_all()
            self._state("stopped", "", f"{n} running replays stopped and queued again")
            log.info("shadow: stopped (%s replays terminated)", n)


def kill_groups(
    targets: list[tuple[int, Callable[[], int | None] | None]], *, grace: float
) -> None:
    """SIGTERM every process group led by one of the pids at once, SIGKILL whatever is left of
    them after `grace` seconds. `poll` (our own child's Popen.poll) reaps the leader the way
    its owner will read it; a pid that does not lead its own group (a replay always does) is
    never signalled. A group is signalled again only while it still has members: once empty,
    its id could be handed to another session (a production worker leads its own group too)."""
    groups = []
    for pid, poll in targets:
        if pid <= 0:
            continue
        try:
            if os.getpgid(pid) != pid:
                continue
        except ProcessLookupError:
            continue
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal.SIGTERM)
        groups.append((pid, poll))

    def members(pid: int, poll: Callable[[], int | None] | None) -> bool:
        if poll is not None:
            poll()  # an exited leader we started stays a zombie (and a member) until reaped
        try:
            os.killpg(pid, 0)
        except (ProcessLookupError, PermissionError):
            # macOS answers EPERM for a group whose only member is a zombie: nothing to signal
            return False
        return True

    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and any(members(p, q) for p, q in groups):
        time.sleep(0.2)
    for pid, poll in groups:
        if members(pid, poll):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pid, signal.SIGKILL)


def _is_replay(pid: int, run_id: int) -> bool | None:
    """Whether `pid` is still the replay of `run_id` (pids are reused); None when `ps` cannot
    say."""
    if not alive(pid):
        return False
    try:
        r = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True,
                           text=True, timeout=10, check=False)  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return False if not alive(pid) else None
    cmd = r.stdout.strip()
    return "reviewsys" in cmd and " replay " in cmd and f" {run_id} " in f"{cmd} "


def _tail(path: Path, n: int = 20) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])[-2000:]
    except OSError:
        return ""


# ---- reports ----


def _ratio(a: float, b: float) -> float | None:
    return round(a / b, 3) if b else None


def _median(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else None


def summary(
    conn: sqlite3.Connection, *, since: str = "", at: datetime | None = None
) -> dict[str, Any]:
    """The numbers `shadow report` prints and the dashboard shows (no finding text)."""
    at = at or now_dt()
    rows = conn.execute(
        "SELECT * FROM shadows WHERE discovered_at >= ? ORDER BY id", (since,)
    ).fetchall()
    status = Counter(str(r["status"]) for r in rows)
    skipped = Counter(_public_reason(r["reason"]) for r in rows if r["status"] == "skipped")
    failed = Counter(_public_reason(r["reason"]) for r in rows if r["status"] == "failed")
    waiting = Counter(_public_reason(r["reason"]) for r in rows if r["status"] == "waiting")
    compared = [r for r in rows if r["status"] == "done" and r["compared_at"]]
    done = [r for r in compared if not r["scoring_degraded"]]

    def within(hours: float) -> int:
        edge = fmt_ts(at - timedelta(hours=hours))
        return sum(1 for r in compared if str(r["finished_at"] or "") >= edge)

    labels = [*VERDICTS, "none"]
    confusion = {p: dict.fromkeys(labels, 0) for p in labels}

    def verdict(v: str | None) -> str:
        return v if v in VERDICTS else "none"

    for r in done:
        confusion[verdict(r["prod_verdict"])][verdict(r["shadow_verdict"])] += 1
    agree = sum(1 for r in done if verdict(r["prod_verdict"]) == verdict(r["shadow_verdict"]))
    m = sum(r["matched"] or 0 for r in done)
    po = sum(r["prod_only"] or 0 for r in done)
    so = sum(r["shadow_only"] or 0 for r in done)

    def mean(col: str) -> float | None:
        return round(statistics.fmean([r[col] or 0 for r in done]), 2) if done else None

    def total(col: str) -> float:
        return float(sum(r[col] or 0 for r in done))

    outcomes: Counter[str] = Counter()
    for o in conn.execute(
        "SELECT o.kind, o.in_shadow, o.severity, o.status FROM outcomes o JOIN shadows s ON s.id=o.shadow_id "
        "WHERE s.discovered_at >= ?",
        (since,),
    ):
        outcomes[o["kind"]] += 1
        if o["kind"] in ("prod_false_alarm", "prod_confirmed") and o["in_shadow"]:
            outcomes[f"{o['kind']}_in_shadow"] += 1
        if o["kind"] == "prod_false_alarm" and o["severity"] == "blocking":
            outcomes["prod_false_alarm_blockers"] += 1
        if o["kind"] == "pr_closed" and o["status"] == "merged":
            outcomes["pr_merged_known"] += 1
    return {
        "heads": {
            "discovered": len(rows),
            "shadowed_24h": within(24),
            "shadowed_7d": within(24 * 7),
            "shadowed_total": len(compared),
            "scoring_degraded": len(compared) - len(done),
            "by_status": dict(status),
            "skipped_by_reason": dict(skipped),
            "failed_by_reason": dict(failed),
            "waiting_by_reason": dict(waiting),
        },
        "verdict_agreement": round(agree / len(done), 4) if done else None,
        "confusion": confusion,  # production verdict -> shadow verdict -> heads
        "findings": {
            "matched": m,
            "prod_only": po,
            "shadow_only": so,
            "agreement": round(m / (m + po + so), 4) if m + po + so else None,
        },
        "per_head": {
            "prod_blockers": mean("prod_blocking"),
            "shadow_blockers": mean("shadow_blocking"),
            "prod_suggestions": mean("prod_suggestion"),
            "shadow_suggestions": mean("shadow_suggestion"),
        },
        "outcomes": {
            "shadow_caught_earlier": outcomes["caught_earlier"],
            "shadow_caught_earlier_heads": sum(1 for r in done if r["caught_earlier"]),
            "prod_false_alarms": outcomes["prod_false_alarm"],
            "prod_false_alarm_blockers": outcomes["prod_false_alarm_blockers"],
            "prod_false_alarms_shadow_also_raised": outcomes["prod_false_alarm_in_shadow"],
            "prod_confirmed_blockers": outcomes["prod_confirmed"],
            "prod_confirmed_shadow_also_raised": outcomes["prod_confirmed_in_shadow"],
            # closed covers merged; a merge is known only when the audit queue recorded it
            "prs_closed": outcomes["pr_closed"],
            "prs_merged_known": outcomes["pr_merged_known"],
        },
        "cost": {
            "gpt_tokens_ratio": _ratio(total("shadow_gpt_tokens"), total("prod_gpt_tokens")),
            "gpt_lanes_ratio": _ratio(total("shadow_gpt_lanes"), total("prod_gpt_lanes")),
            "tokens_ratio": _ratio(total("shadow_tokens"), total("prod_tokens")),
        },
        "latency": {
            "prod_wall_median_s": _median([r["prod_wall_s"] for r in done]),
            "shadow_wall_median_s": _median([r["shadow_wall_s"] for r in done]),
            "prod_active_median_s": _median([r["prod_active_s"] for r in done]),
            "shadow_active_median_s": _median([r["shadow_active_s"] for r in done]),
            "wall_ratio": _ratio(total("shadow_wall_s"), total("prod_wall_s")),
            "active_ratio": _ratio(total("shadow_active_s"), total("prod_active_s")),
        },
    }


def _public_reason(reason: str | None) -> str | None:
    if reason is None:
        return None
    return reason if reason in REASONS else "other"


def runner_state(conn: sqlite3.Connection, root: Path) -> dict[str, Any]:
    try:
        state = json.loads(kv_get(conn, "runner.state") or "{}")
    except ValueError:
        state = {}
    pid = None
    with contextlib.suppress(OSError, ValueError):
        pid = int((root / "shadow.pid").read_text().strip())
    return {
        "running": bool(pid and alive(pid)),
        "pid": pid,
        "state": state.get("state"),
        "reason": state.get("reason"),
        "detail": state.get("detail"),
        "at": state.get("at"),
        "watermark": kv_get(conn, "watermark"),
    }


RECENT = 30


def public_export(db_path: Path, *, at: datetime | None = None) -> dict[str, Any]:
    """The dashboard's `shadow` block: numbers, verdicts, repo/number/sha/PR title and links
    only. Never a finding title or body, never failure text: reasons come from `REASONS`."""
    conn = connect_ro(db_path)
    try:
        s = summary(conn, at=at)
        state = runner_state(conn, db_path.parent)
        recent = []
        for r in conn.execute(
            "SELECT * FROM shadows WHERE status <> 'skipped' OR reason <> 'not sampled' "
            "ORDER BY source_finished_at DESC, source_run DESC LIMIT ?",
            (RECENT,),
        ):
            pr_url = f"https://github.com/{r['repo']}/pull/{r['number']}"
            recent.append(
                {
                    "source_run": r["source_run"],
                    "repo": r["repo"],
                    "number": r["number"],
                    "sha": r["sha"],
                    "title": r["title"],
                    "pr_url": pr_url,
                    "commit_url": f"{pr_url}/commits/{r['sha']}",
                    "status": r["status"],
                    "reason": _public_reason(r["reason"]),
                    "finished_at": r["source_finished_at"],
                    "prod_verdict": r["prod_verdict"],
                    "shadow_verdict": r["shadow_verdict"],
                    "prod": {k: r[f"prod_{k}"] for k in SEVERITIES},
                    "shadow": {k: r[f"shadow_{k}"] for k in SEVERITIES},
                    "matched": r["matched"],
                    "prod_only": r["prod_only"],
                    "shadow_only": r["shadow_only"],
                    "prod_wall_s": r["prod_wall_s"],
                    "shadow_wall_s": r["shadow_wall_s"],
                    "caught_earlier": r["caught_earlier"],
                    "prod_false_alarms": r["prod_false_alarms"],
                    "disagree": bool(r["compared_at"])
                    and (r["prod_verdict"] or "") != (r["shadow_verdict"] or ""),
                }
            )
        return {
            "policy": POLICY_NAME,
            "runner": {
                "running": state["running"],
                "state": state["state"],
                "reason": _public_reason(state["reason"]) if state["reason"] else None,
                "at": state["at"],
            },
            "summary": s,
            "recent": recent,
        }
    finally:
        conn.close()


def export_root() -> Path:
    return Path(os.environ.get(ROOT_ENV) or DEFAULT_ROOT).expanduser()


def render(s: dict[str, Any], state: dict[str, Any] | None = None) -> str:
    """`shadow report` as text."""
    h, f, o, c, lat, ph = (s["heads"], s["findings"], s["outcomes"], s["cost"], s["latency"],
                           s["per_head"])  # fmt: skip

    def pct(x: float | None) -> str:
        return "—" if x is None else f"{x * 100:.1f}%"

    lines = []
    if state is not None:
        lines.append(
            f"runner: {'running' if state['running'] else 'not running'} "
            f"(state {state['state'] or '—'}{', ' + state['reason'] if state['reason'] else ''}) "
            f"watermark {state['watermark'] or '—'}"
        )
    lines += [
        f"heads: {h['shadowed_total']} compared ({h['shadowed_24h']} in 24 h, "
        f"{h['shadowed_7d']} in 7 d; {h['scoring_degraded']} left out of the numbers below: "
        f"a thread-lane shard failed) of {h['discovered']} discovered; by status "
        + ", ".join(f"{k} {v}" for k, v in sorted(h["by_status"].items())),
    ]
    for key, label in (("skipped_by_reason", "skipped"), ("failed_by_reason", "failed"),
                       ("waiting_by_reason", "waiting")):  # fmt: skip
        if h[key]:
            lines.append(f"  {label}: " + ", ".join(f"{k} {v}" for k, v in sorted(h[key].items())))
    lines.append(f"verdict agreement: {pct(s['verdict_agreement'])}")
    labels = [*VERDICTS, "none"]
    lines.append("confusion (rows production, columns shadow):")
    lines.append("  " + " " * 16 + "".join(f"{x[:15]:>16}" for x in labels))
    for p in labels:
        lines.append(
            "  " + f"{p[:15]:<16}" + "".join(f"{s['confusion'][p][x]:>16}" for x in labels)
        )
    lines += [
        f"findings: {f['matched']} matched, {f['prod_only']} production only, "
        f"{f['shadow_only']} shadow only (agreement {pct(f['agreement'])})",
        f"per head: blockers prod {ph['prod_blockers']} / shadow {ph['shadow_blockers']}; "
        f"suggestions prod {ph['prod_suggestions']} / shadow {ph['shadow_suggestions']}",
        f"outcomes: shadow caught earlier {o['shadow_caught_earlier']} "
        f"(on {o['shadow_caught_earlier_heads']} heads); production false alarms "
        f"{o['prod_false_alarms']} ({o['prod_false_alarm_blockers']} blockers, shadow also "
        f"raised {o['prod_false_alarms_shadow_also_raised']}); production blockers confirmed "
        f"fixed {o['prod_confirmed_blockers']} (shadow also raised "
        f"{o['prod_confirmed_shadow_also_raised']}); PRs closed {o['prs_closed']} (merged, "
        f"where known: {o['prs_merged_known']})",
        f"cost (shadow / production): gpt tokens {c['gpt_tokens_ratio']}, gpt lanes "
        f"{c['gpt_lanes_ratio']}, all tokens {c['tokens_ratio']}",
        f"latency: wall median prod {lat['prod_wall_median_s']} s / shadow "
        f"{lat['shadow_wall_median_s']} s (ratio {lat['wall_ratio']}); active median prod "
        f"{lat['prod_active_median_s']} s / shadow {lat['shadow_active_median_s']} s (ratio "
        f"{lat['active_ratio']})",
    ]
    return "\n".join(lines) + "\n"


def status_rows(conn: sqlite3.Connection, limit: int = 20) -> list[dict[str, Any]]:
    return [
        {k: r[k] for k in ("id", "source_run", "repo", "number", "sha", "status", "reason",
                           "attempts", "started_at", "finished_at", "prod_verdict",
                           "shadow_verdict", "matched", "prod_only", "shadow_only")}
        for r in conn.execute(
            "SELECT * FROM shadows ORDER BY id DESC LIMIT ?", (limit,)
        )
    ]  # fmt: skip


def doctor_line(root: Path) -> tuple[bool, str] | None:
    """For `reviewsys doctor`: None without a shadow root; else (healthy, what it says).
    Informational: a stopped shadow is not a production problem."""
    db = root / DB_NAME
    if not db.exists():
        return None
    try:
        conn = connect_ro(db)
    except sqlite3.Error as exc:
        return False, f"{db}: {exc}"
    try:
        st = runner_state(conn, root)
        n = conn.execute("SELECT COUNT(*) FROM shadows WHERE status='done'").fetchone()[0]
    except sqlite3.Error as exc:
        return False, f"{db}: {exc}"
    finally:
        conn.close()
    run = f"running pid {st['pid']}" if st["running"] else "not running"
    return True, f"{run}, state {st['state'] or '—'} {st['reason'] or ''}, {n} heads compared"
