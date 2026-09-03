"""Reaper: enforce heartbeat and deadlines; kill and fail dead/stuck runs."""

from __future__ import annotations

import logging
import sqlite3
from datetime import timedelta

from . import proc
from .config import Config
from .db import now_dt, parse_ts
from .models import FailKind, RunStatus
from .scheduler import finish_run

log = logging.getLogger(__name__)


def reap(conn: sqlite3.Connection, cfg: Config) -> list[tuple[int, str]]:
    """Returns [(run_id, action)] for every run acted on."""
    actions: list[tuple[int, str]] = []
    ts = now_dt()
    stale = timedelta(minutes=cfg.heartbeat_stale_minutes)
    for run in conn.execute("SELECT * FROM runs WHERE status IN ('spawned','running')").fetchall():
        rid, pid = int(run["id"]), run["pid"]
        hb = parse_ts(run["heartbeat_at"]) if run["heartbeat_at"] else None
        started = parse_ts(run["started_at"])
        deadline = parse_ts(run["deadline_at"])
        if ts > deadline:
            outcome = proc.kill_group(pid) if pid else "no-pid"
            finish_run(
                conn,
                cfg,
                rid,
                RunStatus.TIMED_OUT,
                reason=f"run deadline exceeded ({outcome})",
                fail_kind=FailKind.INFRA,
            )
            actions.append((rid, "timed_out"))
            continue
        if run["status"] == RunStatus.SPAWNED.value and hb is None:
            if ts - started > timedelta(minutes=3):
                outcome = proc.kill_group(pid) if pid else "no-pid"
                finish_run(
                    conn,
                    cfg,
                    rid,
                    RunStatus.FAILED,
                    reason=f"worker never heartbeat ({outcome})",
                    fail_kind=FailKind.INFRA,
                )
                actions.append((rid, "no_heartbeat"))
            continue
        if hb is not None and ts - hb > stale:
            outcome = proc.kill_group(pid) if pid else "no-pid"
            finish_run(
                conn,
                cfg,
                rid,
                RunStatus.FAILED,
                reason=f"heartbeat stale since {run['heartbeat_at']} ({outcome})",
                fail_kind=FailKind.INFRA,
            )
            actions.append((rid, "stale"))
            continue
        if pid and not proc.alive(pid):
            # process gone but DB not terminal: worker crashed without writing its status
            finish_run(
                conn,
                cfg,
                rid,
                RunStatus.FAILED,
                reason="worker process exited without terminal status",
                fail_kind=FailKind.INFRA,
            )
            actions.append((rid, "exited"))
            continue
        if run["cancel_requested"] and hb is not None and ts - hb > timedelta(seconds=90) and pid:
            outcome = proc.kill_group(pid)
            finish_run(
                conn,
                cfg,
                rid,
                RunStatus.CANCELLED,
                reason=f"hard-killed after cancel request ({outcome})",
            )
            actions.append((rid, "hard_cancel"))
    return actions
