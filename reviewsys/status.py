"""Status snapshot and watchdog predicate."""

from __future__ import annotations

import sqlite3
from datetime import timedelta
from typing import Any

from . import audit, degraded, lanepool, slots
from .config import Config
from .db import fmt_ts, kv_get, now, now_dt, parse_ts


def median_run_minutes(conn: sqlite3.Connection) -> float | None:
    """Median wall-clock minutes of the last 20 completed live runs, or None before the first
    one. Audit runs (forced Phase 2 plus a persistence check) would inflate every live ETA."""
    recent_done = conn.execute(
        "SELECT r.started_at, r.finished_at FROM runs r JOIN heads h ON h.id=r.head_id "
        "WHERE r.status='done' AND h.queue='live' ORDER BY r.finished_at DESC LIMIT 20"
    ).fetchall()
    durations = sorted(
        (parse_ts(r["finished_at"]) - parse_ts(r["started_at"])).total_seconds() / 60
        for r in recent_done
        if r["finished_at"]
    )
    return durations[len(durations) // 2] if durations else None


def slots_summary(cfg: Config, cap: slots.Capacity | None) -> str:
    """`runs: max N (0 = unlimited); lane pools: gpt ..., muse 8, ...` for `status` and
    `doctor`. `cap` None (no database at hand): the gpt pool as configured, per account."""
    if cap is None:
        gpt = (
            f"gpt {cfg.max_concurrent + cfg.priority_overflow} per usable OpenAI account, "
            f"up to {cfg.account_scale_max} accounts"
        )
    else:
        gpt = f"gpt {cap.ceiling} ({cap.reason})"
    pools = {lanepool.COMPARE_POOL: lanepool.COMPARE_POOL_SLOTS, **cfg.lane_pools}
    rest = ", ".join(f"{p} {n}" for p, n in sorted(pools.items()))
    return f"runs: max {cfg.max_runs} (0 = unlimited); lane pools: {gpt}, {rest}"


def snapshot(conn: sqlite3.Connection, cfg: Config) -> dict[str, Any]:
    ts = now()
    counts = {
        r["status"]: int(r["n"])
        for r in conn.execute(
            "SELECT status, COUNT(*) AS n FROM heads WHERE queue='live' GROUP BY status"
        )
    }
    active = [
        dict(r)
        for r in conn.execute(
            "SELECT r.id, r.status, r.phase, r.pid, r.started_at, r.heartbeat_at, r.attempt, h.repo, h.number, h.sha FROM runs r JOIN heads h ON h.id=r.head_id WHERE r.status IN ('spawned','running')"
        )
    ]
    oldest = conn.execute(
        "SELECT MIN(queued_at) AS q FROM heads WHERE status='queued' AND queue='live'"
    ).fetchone()["q"]
    last_start = conn.execute("SELECT MAX(started_at) AS s FROM runs").fetchone()["s"]
    last_done = conn.execute(
        "SELECT MAX(finished_at) AS f FROM runs WHERE status='done'"
    ).fetchone()["f"]
    median = median_run_minutes(conn)
    failed_24h = conn.execute(
        "SELECT COUNT(*) AS n FROM heads WHERE status='failed' AND queue='live' AND finished_at>=?",
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
        "degraded": degraded.snapshot(conn, cfg),
        "capacity": slots.capacity(conn, cfg).as_dict(),
        "audit": audit.summary(conn, cfg),
    }


def watchdog(conn: sqlite3.Connection, cfg: Config) -> dict[str, Any]:
    """True 'stuck' when a live head could start (eligible, its PR has no run in flight, and
    `max_runs` is not reached) yet nothing started recently. Runs are not bounded by a slot
    count any more (lanepool.py), so such a head starts on the next tick unless something is
    wrong."""
    ts = now_dt()
    eligible = conn.execute(
        "SELECT COUNT(*) AS n FROM heads WHERE status='queued' AND queue='live' AND eligible_at<=?",
        (fmt_ts(ts),),
    ).fetchone()["n"]
    # a head of a PR with a run in flight waits for it (single flight), not for the scheduler
    startable = conn.execute(
        "SELECT COUNT(*) AS n FROM heads h WHERE h.status='queued' AND h.queue='live' "
        "AND h.eligible_at<=? AND NOT EXISTS (SELECT 1 FROM runs r JOIN heads o ON o.id=r.head_id "
        "WHERE r.status IN ('spawned','running') AND o.repo=h.repo AND o.number=h.number)",
        (fmt_ts(ts),),
    ).fetchone()["n"]
    active = conn.execute(
        "SELECT COUNT(*) AS n FROM runs r JOIN heads h ON h.id=r.head_id "
        "WHERE r.status IN ('spawned','running') AND h.queue='live'"
    ).fetchone()["n"]
    in_flight = conn.execute(
        "SELECT COUNT(*) AS n FROM runs WHERE status IN ('spawned','running')"
    ).fetchone()["n"]
    last_start = conn.execute(
        "SELECT MAX(r.started_at) AS s FROM runs r JOIN heads h ON h.id=r.head_id WHERE h.queue='live'"
    ).fetchone()["s"]
    idle_min = (ts - parse_ts(last_start)).total_seconds() / 60 if last_start else None
    stuck = (
        bool(startable)
        and (cfg.max_runs == 0 or in_flight < cfg.max_runs)
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
