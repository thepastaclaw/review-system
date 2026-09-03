"""Scheduler: claim eligible heads, spawn workers, retry/supersede policy.

Slot accounting is computed from the DB on every tick. Nothing is cached, so
no crash can leave a phantom slot occupied.
"""

from __future__ import annotations

import logging
import secrets
import sqlite3
import sys
from datetime import timedelta
from pathlib import Path

from . import proc
from .config import Config
from .db import event, fmt_ts, now, parse_ts, tx
from .models import FailKind, HeadStatus, RunStatus

log = logging.getLogger(__name__)


def active_runs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT r.*, h.repo, h.number, h.sha, h.priority FROM runs r JOIN heads h ON h.id=r.head_id WHERE r.status IN ('spawned','running')"
    ).fetchall()


def eligible_heads(conn: sqlite3.Connection, *, ts: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM heads WHERE status='queued' AND eligible_at<=? ORDER BY priority DESC, queued_at ASC",
        (ts,),
    ).fetchall()


def worker_argv(run_id: int) -> list[str]:
    return [sys.executable, "-m", "reviewsys.cli", "worker", "--run-id", str(run_id)]


def schedule(conn: sqlite3.Connection, cfg: Config, *, spawn: bool = True) -> list[int]:
    """One scheduling pass. Returns the run ids started."""
    ts = now()
    started: list[int] = []
    active = active_runs(conn)
    active_prs = {(r["repo"], r["number"]) for r in active}
    normal_used = sum(1 for r in active if not r["priority"])
    priority_used = sum(1 for r in active if r["priority"])
    for head in eligible_heads(conn, ts=ts):
        if (head["repo"], head["number"]) in active_prs:
            continue  # single-flight per PR
        total_active = normal_used + priority_used
        if head["priority"]:
            if total_active >= cfg.max_concurrent + cfg.priority_overflow:
                continue
        elif total_active >= cfg.max_concurrent:
            continue
        run_id = _start_run(conn, cfg, head, ts=ts, spawn=spawn)
        if run_id is None:
            continue
        started.append(run_id)
        active_prs.add((head["repo"], head["number"]))
        if head["priority"]:
            priority_used += 1
        else:
            normal_used += 1
    return started


def _start_run(
    conn: sqlite3.Connection, cfg: Config, head: sqlite3.Row, *, ts: str, spawn: bool
) -> int | None:
    attempt = int(head["attempts"]) + 1
    token = secrets.token_hex(8)
    deadline = fmt_ts(parse_ts(ts) + timedelta(minutes=cfg.run_timeout_minutes))
    with tx(conn):
        cur = conn.execute(
            "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at) VALUES (?,?,?,?,?,?)",
            (head["id"], attempt, RunStatus.SPAWNED.value, token, ts, deadline),
        )
        run_id = int(cur.lastrowid or 0)
        conn.execute(
            "UPDATE heads SET status='running', attempts=? WHERE id=?", (attempt, head["id"])
        )
        event(
            conn,
            "run.spawned",
            repo=head["repo"],
            number=head["number"],
            run_id=run_id,
            detail=f"attempt {attempt} sha={head['sha'][:8]}",
        )
    if not spawn:
        return run_id
    log_path = cfg.logs_dir / f"run-{run_id}.log"
    try:
        child = proc.spawn(
            worker_argv(run_id),
            cwd=None,
            log_path=log_path,
            env={"REVIEWSYS_CONFIG": str(Path(cfg.db_path).parent / "config.toml")},
        )
    except OSError as exc:
        finish_run(
            conn,
            cfg,
            run_id,
            RunStatus.FAILED,
            reason=f"spawn failed: {exc}",
            fail_kind=FailKind.INFRA,
        )
        return None
    with tx(conn):
        conn.execute("UPDATE runs SET pid=? WHERE id=?", (child.pid, run_id))
    log.info("spawned run %s pid %s for %s#%s", run_id, child.pid, head["repo"], head["number"])
    return run_id


def finish_run(
    conn: sqlite3.Connection,
    cfg: Config,
    run_id: int,
    status: RunStatus,
    *,
    reason: str = "",
    fail_kind: FailKind | None = None,
) -> None:
    """Terminal transition for a run, with head follow-up. Idempotent: no-op if already terminal."""
    ts = now()
    with tx(conn):
        run = conn.execute(
            "SELECT r.*, h.repo, h.number, h.status AS head_status FROM runs r JOIN heads h ON h.id=r.head_id WHERE r.id=?",
            (run_id,),
        ).fetchone()
        if run is None or RunStatus(run["status"]).terminal:
            return
        conn.execute(
            "UPDATE runs SET status=?, reason=?, fail_kind=?, finished_at=? WHERE id=?",
            (status.value, reason[:2000], fail_kind.value if fail_kind else None, ts, run_id),
        )
        event(
            conn,
            f"run.{status.value}",
            repo=run["repo"],
            number=run["number"],
            run_id=run_id,
            detail=reason[:500],
        )
        if run["head_status"] != HeadStatus.RUNNING.value:
            return  # head was superseded/closed while running; nothing more to do
        if status == RunStatus.DONE:
            conn.execute(
                "UPDATE heads SET status='done', finished_at=? WHERE id=?", (ts, run["head_id"])
            )
        elif status == RunStatus.CANCELLED:
            conn.execute(
                "UPDATE heads SET status='queued', eligible_at=? WHERE id=?", (ts, run["head_id"])
            )
        else:
            _requeue_or_fail(conn, cfg, run["head_id"], fail_kind or FailKind.INFRA, reason)


def _requeue_or_fail(
    conn: sqlite3.Connection, cfg: Config, head_id: int, kind: FailKind, reason: str
) -> None:
    head = conn.execute("SELECT * FROM heads WHERE id=?", (head_id,)).fetchone()
    attempts = int(head["attempts"])
    max_attempts = {FailKind.INFRA: cfg.max_attempts, FailKind.CONTRACT: 2, FailKind.FATAL: 0}[kind]
    ts = now()
    if attempts < max_attempts:
        backoff = (
            cfg.retry_backoff_minutes[min(attempts - 1, len(cfg.retry_backoff_minutes) - 1)]
            if cfg.retry_backoff_minutes
            else 5
        )
        eligible = fmt_ts(parse_ts(ts) + timedelta(minutes=backoff))
        conn.execute(
            "UPDATE heads SET status='queued', eligible_at=?, reason=? WHERE id=?",
            (eligible, reason[:500], head_id),
        )
        event(
            conn,
            "head.requeued",
            repo=head["repo"],
            number=head["number"],
            detail=f"attempt {attempts} failed ({kind}); retry in {backoff}m",
        )
    else:
        conn.execute(
            "UPDATE heads SET status='failed', finished_at=?, reason=? WHERE id=?",
            (ts, reason[:500], head_id),
        )
        event(
            conn,
            "head.failed",
            repo=head["repo"],
            number=head["number"],
            detail=f"after {attempts} attempt(s): {reason[:300]}",
        )


def request_cancel(conn: sqlite3.Connection, run_id: int, reason: str) -> None:
    with tx(conn):
        conn.execute(
            "UPDATE runs SET cancel_requested=1, reason=COALESCE(reason, ?) WHERE id=? AND status IN ('spawned','running')",
            (reason, run_id),
        )
        event(conn, "run.cancel_requested", run_id=run_id, detail=reason)


def apply_supersedes(conn: sqlite3.Connection, cfg: Config) -> int:
    """Runs whose head is no longer `running` (superseded/closed by ingest) get cancelled,
    unless they are already past verify2 (then they finish and publish at the old sha)."""
    rows = conn.execute(
        "SELECT r.id, r.phase, h.status AS head_status, h.reason FROM runs r JOIN heads h ON h.id=r.head_id WHERE r.status IN ('spawned','running') AND r.cancel_requested=0 AND h.status NOT IN ('running')"
    ).fetchall()
    n = 0
    for r in rows:
        if r["head_status"] == HeadStatus.SUPERSEDED.value and r["phase"] in ("verify2", "publish"):
            continue  # let it finish; publish will note new commits
        request_cancel(conn, r["id"], f"head {r['head_status']}: {r['reason'] or ''}")
        n += 1
    return n
