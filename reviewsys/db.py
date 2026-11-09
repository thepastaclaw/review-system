"""SQLite store: WAL mode, numbered migrations, short write transactions."""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 10

MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE prs (
        repo TEXT NOT NULL, number INTEGER NOT NULL, head_sha TEXT NOT NULL,
        title TEXT NOT NULL DEFAULT '', author TEXT NOT NULL DEFAULT '',
        is_draft INTEGER NOT NULL DEFAULT 0, state TEXT NOT NULL DEFAULT 'open',
        updated_at TEXT NOT NULL, PRIMARY KEY (repo, number));
    CREATE TABLE heads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        repo TEXT NOT NULL, number INTEGER NOT NULL, sha TEXT NOT NULL,
        trigger TEXT NOT NULL, priority INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'queued',
        queued_at TEXT NOT NULL, eligible_at TEXT NOT NULL,
        superseded_at TEXT, finished_at TEXT, reason TEXT,
        attempts INTEGER NOT NULL DEFAULT 0,
        UNIQUE (repo, number, sha));
    CREATE INDEX heads_status ON heads (status, eligible_at);
    CREATE INDEX heads_pr ON heads (repo, number);
    CREATE TABLE runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        head_id INTEGER NOT NULL REFERENCES heads(id),
        attempt INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'spawned',
        reason TEXT, fail_kind TEXT, pid INTEGER, token TEXT NOT NULL,
        worktree TEXT, run_dir TEXT,
        started_at TEXT NOT NULL, heartbeat_at TEXT, deadline_at TEXT NOT NULL,
        finished_at TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0,
        phase TEXT, blocker_count INTEGER, review_id INTEGER, review_url TEXT);
    CREATE INDEX runs_status ON runs (status);
    CREATE INDEX runs_head ON runs (head_id);
    CREATE TABLE steps (
        run_id INTEGER NOT NULL REFERENCES runs(id), name TEXT NOT NULL,
        status TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
        detail TEXT, PRIMARY KEY (run_id, name));
    CREATE TABLE lanes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL REFERENCES runs(id), phase TEXT NOT NULL,
        role TEXT NOT NULL, agent TEXT NOT NULL, model TEXT NOT NULL,
        attempt INTEGER NOT NULL, attempt_id TEXT NOT NULL, status TEXT NOT NULL,
        exit_code INTEGER, tokens_in INTEGER, tokens_out INTEGER,
        started_at TEXT NOT NULL, finished_at TEXT, artifact_dir TEXT,
        prompt_sha TEXT, reason TEXT);
    CREATE INDEX lanes_run ON lanes (run_id);
    CREATE TABLE findings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL REFERENCES runs(id), phase TEXT NOT NULL,
        stage TEXT NOT NULL, hash TEXT NOT NULL, file TEXT, line_start INTEGER,
        line_end INTEGER, severity TEXT NOT NULL, confidence REAL,
        category TEXT, title TEXT NOT NULL, body TEXT NOT NULL);
    CREATE INDEX findings_run ON findings (run_id, stage);
    CREATE TABLE posted_findings (
        repo TEXT NOT NULL, number INTEGER NOT NULL, hash TEXT NOT NULL,
        sha TEXT NOT NULL, review_id INTEGER, comment_id INTEGER,
        posted_at TEXT NOT NULL, PRIMARY KEY (repo, number, hash));
    CREATE TABLE reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER REFERENCES runs(id), repo TEXT NOT NULL, number INTEGER NOT NULL,
        sha TEXT NOT NULL, phase TEXT NOT NULL, github_review_id INTEGER,
        event TEXT NOT NULL, posted_at TEXT NOT NULL, imported INTEGER NOT NULL DEFAULT 0);
    CREATE INDEX reviews_pr ON reviews (repo, number, sha);
    CREATE TABLE inbox (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id TEXT NOT NULL UNIQUE, kind TEXT NOT NULL,
        repo TEXT NOT NULL, number INTEGER, actor TEXT, body TEXT,
        occurred_at TEXT NOT NULL, seen_at TEXT NOT NULL, handled_at TEXT, action TEXT);
    CREATE INDEX inbox_unhandled ON inbox (handled_at) WHERE handled_at IS NULL;
    CREATE TABLE events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, kind TEXT NOT NULL,
        repo TEXT, number INTEGER, run_id INTEGER, detail TEXT);
    CREATE INDEX events_ts ON events (ts);
    CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """,
    2: """
    ALTER TABLE lanes ADD COLUMN effort TEXT;
    ALTER TABLE runs ADD COLUMN tier TEXT;
    """,
    3: """
    ALTER TABLE runs ADD COLUMN degraded INTEGER NOT NULL DEFAULT 0;
    """,
    # the audit queue: post-merge reviews of PRs merged without a clean review (see audit.py)
    4: """
    ALTER TABLE heads ADD COLUMN queue TEXT NOT NULL DEFAULT 'live';
    CREATE TABLE audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        head_id INTEGER UNIQUE REFERENCES heads(id),  -- NULL: recorded, never run (sync merge)
        repo TEXT NOT NULL, number INTEGER NOT NULL, sha TEXT NOT NULL,
        source TEXT NOT NULL, mode TEXT NOT NULL, rank INTEGER NOT NULL DEFAULT 100,
        title TEXT NOT NULL DEFAULT '', author TEXT NOT NULL DEFAULT '',
        merged_by TEXT NOT NULL DEFAULT '', merged_at TEXT, merge_commit TEXT,
        base_ref TEXT NOT NULL DEFAULT '', coverage TEXT NOT NULL DEFAULT '',
        queued_at TEXT NOT NULL, finished_at TEXT, run_id INTEGER,
        verdict TEXT, blockers INTEGER, still_present INTEGER, findings INTEGER,
        tip_ref TEXT, tip_sha TEXT, comment_url TEXT, issue_url TEXT,
        degraded INTEGER NOT NULL DEFAULT 0, escalated INTEGER NOT NULL DEFAULT 0,
        UNIQUE (repo, number));
    CREATE INDEX audits_rank ON audits (rank, merged_at);
    CREATE TABLE audit_findings (
        audit_id INTEGER NOT NULL REFERENCES audits(id), run_id INTEGER NOT NULL,
        hash TEXT NOT NULL, severity TEXT NOT NULL, file TEXT, line_start INTEGER,
        line_end INTEGER, category TEXT, title TEXT NOT NULL, body TEXT NOT NULL,
        status TEXT, evidence TEXT, fixed_by TEXT,
        PRIMARY KEY (audit_id, hash));
    """,
    # the path a run took (progress.path_of: priority, normal or audit), fixed as it starts: a
    # reply re-queues a finished head with `priority` set and a new `queued_at`, and an audit
    # takes the merged head over, so the head no longer says what its earlier runs were. Runs
    # from before the column are filled from their head, unless the head was queued again
    # after they started (left NULL: not known)
    5: """
    ALTER TABLE runs ADD COLUMN path TEXT;
    UPDATE runs SET path=(SELECT CASE WHEN h.queue='audit' THEN 'audit'
        WHEN h.priority<>0 THEN 'priority' ELSE 'normal' END
        FROM heads h WHERE h.id=runs.head_id AND h.queued_at<=runs.started_at);
    """,
    # events are kept forever by default (`[retention] event_days = 0`), and progress, compare
    # and the exporter look them up by run and kind; without this they scan the whole table
    6: """
    CREATE INDEX IF NOT EXISTS events_run ON events (run_id, kind);
    """,
    # the PR carries a `[scheduling] skip_labels` label: no automatic review (see ingest.py)
    7: """
    ALTER TABLE prs ADD COLUMN skip_label INTEGER NOT NULL DEFAULT 0;
    """,
    # the PR's own diff as a live run found it, and the earlier review it carries (worker._carry)
    8: """
    ALTER TABLE heads ADD COLUMN diff_print TEXT;
    ALTER TABLE heads ADD COLUMN carried_from TEXT;
    """,
    # the issue ledger (ledger.py): one row per issue the bot raised on a PR, keyed by its
    # `finding_hash` (byte-compatible with the markers on GitHub), and the event history it is
    # folded from. `issue_events.key` names the fact an event records (`posted:<run>:<hash>`,
    # ...): the backfill and the live write-through derive the same key for the same fact, so
    # either can run first and neither duplicates the other. Only additive statements: the old
    # code keeps working against this schema (it never names these tables or the new column).
    # `findings.candidate_id` is the harness-assigned candidate id of v12 finder lanes. The v12
    # review-strength query (a run's own events) is served by migration 6's `events_run`.
    9: """
    CREATE TABLE IF NOT EXISTS issues (
        repo TEXT NOT NULL, number INTEGER NOT NULL, hash TEXT NOT NULL,
        file TEXT NOT NULL DEFAULT '', line_start INTEGER, line_end INTEGER, anchor_sha TEXT,
        severity TEXT NOT NULL, category TEXT NOT NULL DEFAULT 'general',
        title TEXT NOT NULL, body TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL, status_sha TEXT, status_source TEXT NOT NULL,
        opened_run INTEGER, opened_sha TEXT, last_seen_run INTEGER, last_seen_sha TEXT,
        closed_sha TEXT, closed_reason TEXT, thread_id TEXT, comment_id INTEGER,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        PRIMARY KEY (repo, number, hash));
    CREATE INDEX IF NOT EXISTS issues_status ON issues (status);
    CREATE TABLE IF NOT EXISTS issue_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
        repo TEXT NOT NULL, number INTEGER NOT NULL, hash TEXT NOT NULL, key TEXT NOT NULL,
        kind TEXT NOT NULL, run_id INTEGER, sha TEXT, source TEXT NOT NULL,
        from_status TEXT, to_status TEXT, detail TEXT NOT NULL DEFAULT '{}',
        UNIQUE (repo, number, key));
    CREATE INDEX IF NOT EXISTS issue_events_issue ON issue_events (repo, number, hash);
    CREATE INDEX IF NOT EXISTS issue_events_run ON issue_events (run_id);
    ALTER TABLE findings ADD COLUMN candidate_id TEXT;
    """,
    # the check run (checks.py) last written for a commit: its GitHub id, and what it said
    10: """
    CREATE TABLE IF NOT EXISTS check_runs (
        repo TEXT NOT NULL, number INTEGER NOT NULL, sha TEXT NOT NULL,
        check_id INTEGER NOT NULL, status TEXT NOT NULL, conclusion TEXT,
        digest TEXT NOT NULL, updated_at TEXT NOT NULL,
        PRIMARY KEY (repo, number, sha));
    """,
}


def fmt_ts(value: datetime) -> str:
    """Render a datetime the way every timestamp column stores it: ISO-8601 with a `Z`."""
    return value.isoformat().replace("+00:00", "Z")


def now_dt() -> datetime:
    """Current UTC time at the whole-second resolution used for stored timestamps."""
    return datetime.now(UTC).replace(microsecond=0)


def now() -> str:
    return fmt_ts(now_dt())


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def connect(path: Path | str) -> sqlite3.Connection:
    p = Path(path)
    if str(p) != ":memory:":
        p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    # Switching a fresh file to WAL needs an exclusive lock and SQLite does not honour
    # busy_timeout for that particular pragma, so retry it briefly under contention.
    for attempt in range(50):
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            break
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc) or attempt == 49:
                raise
            time.sleep(0.05 * (attempt + 1))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    _retry_locked(lambda: migrate(conn))
    return conn


def connect_existing(path: Path | str) -> sqlite3.Connection:
    """Another connection to a database `connect` already opened and migrated in this
    process: same settings, no migration pass. One per thread (sqlite3 connections must not
    be shared between threads that use them at the same time)."""
    conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _retry_locked(fn: Callable[[], None], attempts: int = 50) -> None:
    for attempt in range(attempts):
        try:
            fn()
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc) or attempt == attempts - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    for version in sorted(MIGRATIONS):
        with tx(conn):
            # re-read under the write lock: another process may have migrated first
            row = conn.execute("SELECT version FROM schema_version").fetchone()
            current = int(row[0]) if row else 0
            if version > current:
                # executescript() would implicitly COMMIT; run statements one by one instead
                for stmt in MIGRATIONS[version].split(";"):
                    if stmt.strip() and not _column_exists(conn, stmt):
                        conn.execute(stmt)
                if row is None:
                    conn.execute("INSERT INTO schema_version VALUES (?)", (version,))
                else:
                    conn.execute("UPDATE schema_version SET version=?", (version,))


_ADD_COLUMN_RE = re.compile(r"\s*ALTER\s+TABLE\s+(\w+)\s+ADD\s+COLUMN\s+(\w+)", re.IGNORECASE)


def _column_exists(conn: sqlite3.Connection, stmt: str) -> bool:
    """`stmt` adds a column the table already has: a database the unmerged policy v12 branch
    migrated to its own schema v6 has `findings.candidate_id`, which migration 9 adds."""
    m = _ADD_COLUMN_RE.match(stmt)
    if not m:
        return False
    return any(r[1] == m.group(2) for r in conn.execute(f"PRAGMA table_info({m.group(1)})"))


@contextmanager
def tx(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Short write transaction. BEGIN IMMEDIATE takes the write lock up front."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def event(
    conn: sqlite3.Connection,
    kind: str,
    *,
    repo: str | None = None,
    number: int | None = None,
    run_id: int | None = None,
    detail: str = "",
) -> None:
    conn.execute(
        "INSERT INTO events (ts, kind, repo, number, run_id, detail) VALUES (?,?,?,?,?,?)",
        (now(), kind, repo, number, run_id, detail[:4000]),
    )


def kv_get(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    return str(row[0]) if row else default


def kv_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO kv (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
