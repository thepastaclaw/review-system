"""Sanitized observability export for the public status dashboard."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Config
from .db import now, now_dt, parse_ts
from .queue_status import queued_order
from .status import snapshot


def _age(ts: str | None, at: datetime) -> int | None:
    return max(0, int((at - parse_ts(ts)).total_seconds())) if ts else None


def build_export(conn: sqlite3.Connection, cfg: Config) -> dict[str, Any]:
    at = now_dt()
    live = snapshot(conn, cfg)
    queued = []
    # the order the scheduler will actually pick them (shared with the queue comments): eligible
    # priority heads, then eligible normal heads by queued_at, then heads still in debounce or
    # backoff. Sorting by eligible_at would put a head that just finished its backoff ahead of
    # one queued hours earlier, which is not what happens.
    ts = now()
    for pos, h in enumerate(queued_order(conn, ts=ts), 1):
        row = conn.execute(
            "SELECT h.*, p.title, p.author FROM heads h LEFT JOIN prs p ON p.repo=h.repo AND p.number=h.number WHERE h.id=?",
            (h["id"],),
        ).fetchone()
        if row is None:
            continue
        comment = conn.execute(
            "SELECT value FROM kv WHERE key=?", (f"queue.comment_id:{row['repo']}#{row['number']}",)
        ).fetchone()
        queued.append(
            {
                "position": pos,
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
                "eligible": row["eligible_at"] <= ts,
                "reason": "waiting for debounce/backoff"
                if row["eligible_at"] > ts
                else "waiting for a review slot",
                "pr_url": f"https://github.com/{row['repo']}/pull/{row['number']}",
                "prioritize_url": f"https://github.com/{row['repo']}/issues/{row['number']}#issuecomment-{comment[0]}"
                if comment
                else f"https://github.com/{row['repo']}/pull/{row['number']}#issuecomment-new",
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
    token_totals = conn.execute(
        "SELECT COALESCE(SUM(tokens_in),0) tokens_in, COALESCE(SUM(tokens_out),0) tokens_out FROM lanes"
    ).fetchone()
    tokens_by_model = [
        dict(r)
        for r in conn.execute(
            "SELECT model,COALESCE(SUM(tokens_in),0) tokens_in,COALESCE(SUM(tokens_out),0) tokens_out,COUNT(*) lanes,"
            "ROUND(AVG(COALESCE(tokens_in,0)+COALESCE(tokens_out,0))) avg_tokens_per_lane FROM lanes GROUP BY model ORDER BY (tokens_in+tokens_out) DESC"
        )
    ]
    tokens_by_effort = [
        dict(r)
        for r in conn.execute(
            "SELECT COALESCE(effort,'unknown') effort,COALESCE(SUM(tokens_in),0) tokens_in,"
            "COALESCE(SUM(tokens_out),0) tokens_out,COUNT(*) lanes FROM lanes "
            "GROUP BY effort ORDER BY (tokens_in+tokens_out) DESC"
        )
    ]
    tokens_by_run = [
        dict(r)
        for r in conn.execute(
            "SELECT run_id,COALESCE(SUM(tokens_in),0) tokens_in,COALESCE(SUM(tokens_out),0) tokens_out FROM lanes GROUP BY run_id ORDER BY run_id DESC LIMIT 100"
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
    cap = live.pop("capacity")
    live["capacity"] = {
        "typical": cap["normal"],
        "priority_overflow": cap["priority"],
        "maximum": cap["ceiling"],
        "scale": cap["scale"],
        "usable_accounts": cap["usable"],
    }
    live["priority_queued"] = sum(1 for q in queued if q["priority"])
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
                "tokens_in": token_totals["tokens_in"],
                "tokens_out": token_totals["tokens_out"],
                "tokens_total": token_totals["tokens_in"] + token_totals["tokens_out"],
                "completed_reviews": conn.execute(
                    "SELECT COUNT(*) FROM runs WHERE status='done'"
                ).fetchone()[0],
                "avg_tokens_per_completed_review": round(
                    (token_totals["tokens_in"] + token_totals["tokens_out"])
                    / max(
                        1,
                        conn.execute("SELECT COUNT(*) FROM runs WHERE status='done'").fetchone()[0],
                    )
                ),
            },
            "tokens_by_model": tokens_by_model,
            "tokens_by_effort": tokens_by_effort,
            "tokens_by_run": tokens_by_run,
        },
    }


def write_export(conn: sqlite3.Connection, cfg: Config, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    target = output / "status.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(build_export(conn, cfg), indent=2, sort_keys=True) + "\n")
    tmp.replace(target)
    return target
