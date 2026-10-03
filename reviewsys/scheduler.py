"""Scheduler: claim eligible heads, spawn workers, retry/supersede policy.

The runs in flight are counted from the DB on every tick. Nothing is cached, so no crash
can leave a phantom run counted.
"""

from __future__ import annotations

import logging
import secrets
import sqlite3
import sys
from datetime import timedelta
from pathlib import Path

from . import audit, proc, progress
from .config import Config
from .db import event, fmt_ts, kv_get, now, parse_ts, tx
from .ingest import enqueue_head
from .models import FailKind, HeadStatus, RunStatus, Trigger

log = logging.getLogger(__name__)


def active_runs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT r.*, h.repo, h.number, h.sha, h.priority, h.queue FROM runs r JOIN heads h ON h.id=r.head_id WHERE r.status IN ('spawned','running')"
    ).fetchall()


def eligible_heads(conn: sqlite3.Connection, *, ts: str) -> list[sqlite3.Row]:
    """Live heads ready to start, in pick order. Audit heads are scheduled separately."""
    return conn.execute(
        "SELECT * FROM heads WHERE status='queued' AND queue='live' AND eligible_at<=? ORDER BY priority DESC, queued_at ASC",
        (ts,),
    ).fetchall()


def eligible_audits(conn: sqlite3.Connection, *, ts: str) -> list[sqlite3.Row]:
    """Audit heads ready to start, in the order the backfill was ranked (see audit.py)."""
    return conn.execute(
        "SELECT h.* FROM heads h LEFT JOIN audits a ON a.head_id=h.id WHERE h.status='queued' "
        "AND h.queue='audit' AND h.eligible_at<=? ORDER BY COALESCE(a.rank, 100), a.merged_at, h.queued_at",
        (ts,),
    ).fetchall()


def live_queued_count(conn: sqlite3.Connection) -> int:
    """Queued live heads: what the backlog rules and queue comments measure. Audit heads are
    left out so a backfill never pushes live reviews into backlog mode."""
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM heads WHERE status='queued' AND queue='live'"
        ).fetchone()[0]
    )


def worker_argv(run_id: int) -> list[str]:
    return [sys.executable, "-m", "reviewsys.cli", "worker", "--run-id", str(run_id)]


def schedule(conn: sqlite3.Connection, cfg: Config, *, spawn: bool = True) -> list[int]:
    """One scheduling pass. Returns the run ids started.

    Runs are not bounded by a review-slot count: every eligible live head starts (one run
    per PR), and what is limited is the models. Each lane waits for a slot in its model's
    machine-wide pool (lanepool.py), so a run whose next lane is on an idle model gets on
    with it instead of waiting for a run on a busy one to end. `[scheduling] max_runs` is
    only a safety cap on the runs in flight, live and audit together (0 = none).

    Live heads first, priority heads before the rest (`eligible_heads` order). Audit heads
    then start (up to `audit.concurrency`) only when no live head is left waiting, so a live
    review never queues behind an audit; in every pool an audit's lanes line up behind every
    live lane."""
    ts = now()
    started: list[int] = []
    active = active_runs(conn)
    active_prs = {(r["repo"], r["number"]) for r in active}
    in_flight = len(active)

    def at_cap() -> bool:
        return cfg.max_runs > 0 and in_flight >= cfg.max_runs

    live_waiting = False
    for head in eligible_heads(conn, ts=ts):
        if (head["repo"], head["number"]) in active_prs:
            continue  # single-flight per PR
        if at_cap():
            live_waiting = True
            break
        run_id = _start_run(conn, cfg, head, ts=ts, spawn=spawn)
        if run_id is None:
            live_waiting = True
            continue
        started.append(run_id)
        active_prs.add((head["repo"], head["number"]))
        in_flight += 1
    if live_waiting or not cfg.audit.enabled:
        return started
    audit_used = sum(1 for r in active if r["queue"] == "audit")
    audit_cap = audit.concurrency(conn, cfg)
    for head in eligible_audits(conn, ts=ts):
        if audit_used >= audit_cap or at_cap():
            break
        if (head["repo"], head["number"]) in active_prs:
            continue
        run_id = _start_run(conn, cfg, head, ts=ts, spawn=spawn)
        if run_id is None:
            continue
        started.append(run_id)
        active_prs.add((head["repo"], head["number"]))
        audit_used += 1
        in_flight += 1
    return started


def _start_run(
    conn: sqlite3.Connection, cfg: Config, head: sqlite3.Row, *, ts: str, spawn: bool
) -> int | None:
    attempt = int(head["attempts"]) + 1
    token = secrets.token_hex(8)
    deadline = fmt_ts(parse_ts(ts) + timedelta(minutes=cfg.run_timeout_minutes))
    with tx(conn):
        cur = conn.execute(
            "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at, path) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                head["id"],
                attempt,
                RunStatus.SPAWNED.value,
                token,
                ts,
                deadline,
                progress.path_of(head["queue"], head["priority"]),
            ),
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
            "SELECT r.*, h.repo, h.number, h.sha, h.status AS head_status FROM runs r JOIN heads h ON h.id=r.head_id WHERE r.id=?",
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
            # a conversation that lifted every finding of a later-round review asked for the
            # fresh Phase-2 audit that review skipped (worker._request_fresh_audit)
            key = f"rereview.after_run:{run_id}"
            if kv_get(conn, key):
                conn.execute("DELETE FROM kv WHERE key=?", (key,))
                enqueue_head(conn, cfg, run["repo"], run["number"], run["sha"], Trigger.MANUAL)
        elif status == RunStatus.CANCELLED:
            conn.execute(
                "UPDATE heads SET status='queued', eligible_at=? WHERE id=?", (ts, run["head_id"])
            )
        else:
            _requeue_or_fail(conn, cfg, run["head_id"], fail_kind or FailKind.INFRA, reason)


def retire_obsolete_head(
    conn: sqlite3.Connection, head_id: int, status: HeadStatus, reason: str
) -> bool:
    """End a running head whose run found it obsolete before reviewing anything: a push moved
    the PR's head past it (SUPERSEDED; ingest queues, or has queued, the new one) or the PR
    closed (CLOSED). The same end ingest gives a head it sees go stale, so no retry, no
    `head.failed` alert, and the gate comment is left to the newer head's writer. A head
    ingest already ended is left as it is. Caller holds the write transaction."""
    assert status in (HeadStatus.SUPERSEDED, HeadStatus.CLOSED)
    ts = now()
    head = conn.execute("SELECT * FROM heads WHERE id=?", (head_id,)).fetchone()
    n = conn.execute(
        "UPDATE heads SET status=?, superseded_at=?, finished_at=?, reason=? "
        "WHERE id=? AND status='running'",
        (
            status.value,
            ts if status == HeadStatus.SUPERSEDED else None,
            ts,
            reason[:500],
            head_id,
        ),
    ).rowcount
    if n:
        event(
            conn,
            f"head.{status.value}",
            repo=head["repo"],
            number=head["number"],
            detail=f"obsolete assigned head {head['sha'][:8]}: {reason[:300]}",
        )
    return bool(n)


def _requeue_or_fail(
    conn: sqlite3.Connection, cfg: Config, head_id: int, kind: FailKind, reason: str
) -> None:
    head = conn.execute("SELECT * FROM heads WHERE id=?", (head_id,)).fetchone()
    attempts = int(head["attempts"])
    # A push raced with the worker's initial metadata read.  The assigned SHA is obsolete;
    # this is normal queue churn, not a review failure and must not page the operator.
    if kind == FailKind.FATAL and reason.startswith("live base "):
        ts = now()
        conn.execute(
            "UPDATE heads SET status='queued', eligible_at=?, reason=? WHERE id=?",
            (ts, reason[:500], head_id),
        )
        event(
            conn,
            "head.requeued",
            repo=head["repo"],
            number=head["number"],
            detail=f"base moved while reviewing {head['sha'][:8]}: {reason[:300]}",
        )
        return
    # a worker since 2026-10 ends this case as cancelled (retire_obsolete_head) instead; kept
    # for one still on the older code during a deploy
    if kind == FailKind.FATAL and reason.startswith("live head "):
        ts = now()
        conn.execute(
            "UPDATE heads SET status='superseded', finished_at=?, reason=? WHERE id=?",
            (ts, reason[:500], head_id),
        )
        event(
            conn,
            "head.superseded",
            repo=head["repo"],
            number=head["number"],
            detail=f"obsolete assigned head {head['sha'][:8]}: {reason[:300]}",
        )
        return
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
        "SELECT r.id, r.phase, h.status AS head_status, h.reason FROM runs r JOIN heads h ON h.id=r.head_id WHERE r.status IN ('spawned','running') AND r.cancel_requested=0 AND h.status NOT IN ('running') AND h.queue='live'"
    ).fetchall()
    n = 0
    for r in rows:
        if r["head_status"] == HeadStatus.SUPERSEDED.value and r["phase"] in ("verify2", "publish"):
            continue  # let it finish; publish will note new commits
        request_cancel(conn, r["id"], f"head {r['head_status']}: {r['reason'] or ''}")
        n += 1
    return n
