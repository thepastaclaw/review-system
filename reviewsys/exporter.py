"""Sanitized observability export for the public status dashboard."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Config
from .db import now, now_dt, parse_ts
from .status import snapshot


def _age(ts: str | None, at: datetime) -> int | None:
    return max(0, int((at - parse_ts(ts)).total_seconds())) if ts else None


def build_export(conn: sqlite3.Connection, cfg: Config) -> dict[str, Any]:
    at = now_dt()
    live = snapshot(conn, cfg)
    queued = []
    for row in conn.execute(
        "SELECT h.*, p.title, p.author FROM heads h LEFT JOIN prs p ON p.repo=h.repo AND p.number=h.number "
        "WHERE h.status='queued' ORDER BY h.priority DESC, h.eligible_at, h.queued_at"
    ):
        queued.append(
            {
                "repo": row["repo"],
                "number": row["number"],
                "sha": row["sha"],
                "title": row["title"] or "",
                "author": row["author"] or "",
                "priority": bool(row["priority"]),
                "trigger": row["trigger"],
                "queued_at": row["queued_at"],
                "eligible_at": row["eligible_at"],
                "age_seconds": _age(row["queued_at"], at),
            }
        )
    runs = []
    for row in conn.execute(
        "SELECT r.id,r.status,r.phase,r.attempt,r.started_at,r.heartbeat_at,r.deadline_at,h.repo,h.number,h.sha "
        "FROM runs r JOIN heads h ON h.id=r.head_id WHERE r.status IN ('spawned','running') ORDER BY r.started_at"
    ):
        runs.append(
            {
                **dict(row),
                "elapsed_seconds": _age(row["started_at"], at),
                "heartbeat_age_seconds": _age(row["heartbeat_at"], at),
            }
        )
    daily = [
        dict(r)
        for r in conn.execute(
            "SELECT substr(finished_at,1,10) day, COUNT(*) reviews, "
            "AVG((julianday(finished_at)-julianday(started_at))*86400) avg_seconds "
            "FROM runs WHERE status='done' AND finished_at IS NOT NULL GROUP BY day ORDER BY day DESC LIMIT 180"
        )
    ]
    findings = [
        dict(r)
        for r in conn.execute(
            "SELECT severity, COUNT(*) count FROM findings GROUP BY severity ORDER BY severity"
        )
    ]
    recent = [
        dict(r)
        for r in conn.execute(
            "SELECT ts,kind,repo,number,run_id,detail FROM events ORDER BY id DESC LIMIT 40"
        )
    ]
    return {
        "schema_version": 1,
        "generated_at": now(),
        "data_as_of": live["ts"],
        "live": {**live, "queued": queued, "active": runs},
        "history": {
            "daily": daily,
            "findings_by_severity": findings,
            "recent_events": recent,
            "totals": {
                "findings": conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0],
                "reviews": conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0],
                "runs": conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
            },
        },
    }


def write_export(conn: sqlite3.Connection, cfg: Config, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    target = output / "status.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(build_export(conn, cfg), indent=2, sort_keys=True) + "\n")
    tmp.replace(target)
    return target
