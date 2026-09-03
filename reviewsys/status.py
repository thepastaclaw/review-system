"""Status snapshot and watchdog predicate."""

from __future__ import annotations

import sqlite3
from datetime import timedelta
from typing import Any

from .config import Config
from .db import fmt_ts, kv_get, now, now_dt, parse_ts


def snapshot(conn: sqlite3.Connection, cfg: Config) -> dict[str, Any]:
    ts = now()
    counts = {
        r["status"]: int(r["n"])
        for r in conn.execute("SELECT status, COUNT(*) AS n FROM heads GROUP BY status")
    }
    active = [
        dict(r)
        for r in conn.execute(
            "SELECT r.id, r.status, r.phase, r.pid, r.started_at, r.heartbeat_at, r.attempt, h.repo, h.number, h.sha FROM runs r JOIN heads h ON h.id=r.head_id WHERE r.status IN ('spawned','running')"
        )
    ]
    oldest = conn.execute("SELECT MIN(queued_at) AS q FROM heads WHERE status='queued'").fetchone()[
        "q"
    ]
    last_start = conn.execute("SELECT MAX(started_at) AS s FROM runs").fetchone()["s"]
    last_done = conn.execute(
        "SELECT MAX(finished_at) AS f FROM runs WHERE status='done'"
    ).fetchone()["f"]
    recent_done = conn.execute(
        "SELECT started_at, finished_at FROM runs WHERE status='done' ORDER BY finished_at DESC LIMIT 20"
    ).fetchall()
    durations = sorted(
        (parse_ts(r["finished_at"]) - parse_ts(r["started_at"])).total_seconds() / 60
        for r in recent_done
        if r["finished_at"]
    )
    median = durations[len(durations) // 2] if durations else None
    failed_24h = conn.execute(
        "SELECT COUNT(*) AS n FROM heads WHERE status='failed' AND finished_at>=?",
        (fmt_ts(parse_ts(ts) - timedelta(hours=24)),),
    ).fetchone()["n"]
    return {
        "ts": ts,
        "heads": counts,
        "active": active,
        "oldest_queued_at": oldest,
        "last_run_started_at": last_start,
        "last_run_done_at": last_done,
        "median_run_minutes": median,
        "failed_heads_24h": int(failed_24h),
        "ingest_last_at": kv_get(conn, "ingest.last_at"),
        "notify_last_at": kv_get(conn, "notify.last_at"),
        "ingest_error_streak": int(kv_get(conn, "ingest.error_streak", "0") or 0),
        "watchdog": watchdog(conn, cfg),
    }


def watchdog(conn: sqlite3.Connection, cfg: Config) -> dict[str, Any]:
    """True 'stuck' when eligible work exists, a slot is free, and nothing started recently."""
    ts = now_dt()
    eligible = conn.execute(
        "SELECT COUNT(*) AS n FROM heads WHERE status='queued' AND eligible_at<=?",
        (fmt_ts(ts),),
    ).fetchone()["n"]
    active = conn.execute(
        "SELECT COUNT(*) AS n FROM runs WHERE status IN ('spawned','running')"
    ).fetchone()["n"]
    last_start = conn.execute("SELECT MAX(started_at) AS s FROM runs").fetchone()["s"]
    idle_min = (ts - parse_ts(last_start)).total_seconds() / 60 if last_start else None
    stuck = (
        bool(eligible)
        and active < cfg.max_concurrent
        and (idle_min is None or idle_min > cfg.watchdog_minutes)
    )
    ingest_last = kv_get(conn, "ingest.last_at")
    ingest_stale = ingest_last is None or (ts - parse_ts(ingest_last)) > timedelta(
        seconds=cfg.ingest_interval_seconds * 4
    )
    return {
        "stuck": stuck,
        "eligible": int(eligible),
        "active": int(active),
        "idle_minutes": idle_min,
        "ingest_stale": ingest_stale,
    }
