"""SQLite store: WAL mode, numbered migrations, short write transactions."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1

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
    conn = sqlite3.connect(str(p), timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA synchronous=NORMAL")
    migrate(conn)
    return conn


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
                    if stmt.strip():
                        conn.execute(stmt)
                if row is None:
                    conn.execute("INSERT INTO schema_version VALUES (?)", (version,))
                else:
                    conn.execute("UPDATE schema_version SET version=?", (version,))


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
