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


# runs a day's drill-down lists; a cancelled run is almost always a head superseded by a newer
# push, and hundreds of those would bury the reviews that actually happened
_DAY_STATUSES = ("done", "failed", "timed_out")


def _span(start: str | None, end: str | None) -> int | None:
    return max(0, int((parse_ts(end) - parse_ts(start)).total_seconds())) if start and end else None


_SEVERITIES = ("blocking", "suggestion", "nitpick")
# verification step -> the (phase, stage) its kept findings are recorded under
_VERIFY_PASSES = {
    "verify1": ("verify1", "verified"),
    "verify2": ("verify2", "verified"),
    "fresh_verify2": ("verify2", "verified-fresh"),
}
# `_backfill_posted` records GitHub's review state where a publication records its event
_STATE_EVENTS = {
    "APPROVED": "APPROVE",
    "CHANGES_REQUESTED": "REQUEST_CHANGES",
    "COMMENTED": "COMMENT",
}


def _severity_counts(hashes: dict[str, str] | None) -> dict[str, int]:
    values = list((hashes or {}).values())
    return {sev: values.count(sev) for sev in _SEVERITIES}


def _review_findings(
    passes: dict[tuple[str, str], dict[str, str]], last_verify: tuple[str, int | None] | None
) -> tuple[dict[str, int], dict[str, int]]:
    """(findings the review stands on, findings new to it), counted by severity.

    A review stands on the pass of its last verification step (an audit posts exactly that
    pass), carried-over blockers included: a re-review that only re-asserts an old blocker
    posts nothing new, yet requests changes. A last pass that kept nothing counts zero, never
    an earlier pass's findings (except on runs older than the `verified-fresh` stage, whose
    fresh pass shares the `verified` rows with the pass before it). `posted` rows are the
    findings the review added, except on a run that backfilled a publication an earlier
    attempt made (they then equal the pass). Hashes are keys, so a retried verifier counts
    each finding once."""
    step, kept = last_verify or (None, None)
    standing = passes.get(_VERIFY_PASSES[step]) if step else None
    if step == "fresh_verify2" and standing is None and kept != 0:
        # runs before the `verified-fresh` stage existed recorded the fresh pass as `verified`
        standing = passes.get(_VERIFY_PASSES["verify2"])
    new = passes.get(("final", "posted")) or passes.get(("preliminary", "posted"))
    return _severity_counts(standing), _severity_counts(new)


def build_day_runs(conn: sqlite3.Connection, days: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Every finished run of each UTC day in `days`, newest first, for the dashboard drill-down.

    Only public-safe fields: `runs.reason` is left out because failure text can carry proxy
    and account details, and findings are counted, never quoted."""
    if not days:
        return {}
    scope = (
        f"r.status IN ({','.join('?' * len(_DAY_STATUSES))}) AND r.finished_at IS NOT NULL "
        f"AND substr(r.finished_at,1,10) IN ({','.join('?' * len(days))})"
    )
    params = (*_DAY_STATUSES, *days)
    lanes: dict[int, list[dict[str, Any]]] = {}
    for row in conn.execute(
        "SELECT l.run_id,l.phase,l.role,l.model,l.effort,l.status,l.attempt,"
        "COALESCE(l.tokens_in,0) tokens_in,COALESCE(l.tokens_out,0) tokens_out "
        f"FROM lanes l JOIN runs r ON r.id=l.run_id WHERE {scope} ORDER BY l.id",
        params,
    ):
        lane = dict(row)
        lanes.setdefault(lane.pop("run_id"), []).append(lane)
    # one pass = the findings of one (phase, stage): hash -> severity
    passes: dict[int, dict[tuple[str, str], dict[str, str]]] = {}
    for row in conn.execute(
        "SELECT f.run_id,f.phase,f.stage,f.hash,f.severity FROM findings f "
        f"JOIN runs r ON r.id=f.run_id WHERE {scope} "
        "AND f.stage IN ('verified','verified-fresh','posted') ORDER BY f.id",
        params,
    ):
        run = passes.setdefault(row["run_id"], {})
        run.setdefault((row["phase"], row["stage"]), {})[row["hash"]] = row["severity"]
    # the step's detail says how many findings the pass kept: 0 tells an empty fresh pass
    # apart from a pre-`verified-fresh` run whose fresh findings sit under `verified`
    last_verify: dict[int, tuple[str, int | None]] = {}
    for row in conn.execute(
        f"SELECT s.run_id,s.name,s.detail FROM steps s JOIN runs r ON r.id=s.run_id WHERE {scope} "
        "AND s.status='ok' AND s.name IN ('verify1','verify2','fresh_verify2') "
        "ORDER BY s.finished_at, s.started_at",
        params,
    ):
        try:
            kept = json.loads(row["detail"] or "{}").get("findings")
        except (ValueError, AttributeError):
            kept = None
        last_verify[row["run_id"]] = (row["name"], kept if isinstance(kept, int) else None)
    verdicts: dict[int, str] = {}
    for row in conn.execute(
        f"SELECT v.run_id,v.event FROM reviews v JOIN runs r ON r.id=v.run_id WHERE {scope} "
        "ORDER BY v.id",
        params,
    ):
        verdicts[row["run_id"]] = _STATE_EVENTS.get(row["event"], row["event"])
    out: dict[str, list[dict[str, Any]]] = {day: [] for day in days}
    for row in conn.execute(
        "SELECT r.id,r.status,r.fail_kind,r.attempt,r.tier,r.degraded,r.blocker_count,r.review_url,"
        "r.started_at,r.finished_at,h.repo,h.number,h.sha,h.queue,h.trigger,h.priority,h.queued_at,"
        "h.eligible_at,COALESCE(p.title,a.title,'') title,a.comment_url audit_url,"
        # the head's queued_at/eligible_at are not reset by a retry, a backoff or an audit
        # escalation, so a later run waited from its predecessor's end, and only the head's
        # newest run can be measured against eligible_at
        "(SELECT MAX(o.finished_at) FROM runs o WHERE o.head_id=r.head_id AND o.id<r.id) prev_end,"
        "NOT EXISTS (SELECT 1 FROM runs o WHERE o.head_id=r.head_id AND o.id>r.id) newest "
        "FROM runs r JOIN heads h ON h.id=r.head_id "
        "LEFT JOIN prs p ON p.repo=h.repo AND p.number=h.number "
        "LEFT JOIN audits a ON a.run_id=r.id "
        f"WHERE {scope} ORDER BY r.finished_at DESC, r.id DESC",
        params,
    ):
        run_lanes = lanes.get(row["id"], [])
        findings, new_findings = _review_findings(
            passes.get(row["id"], {}), last_verify.get(row["id"])
        )
        pr_url = f"https://github.com/{row['repo']}/pull/{row['number']}"
        waited_from = max(filter(None, (row["queued_at"], row["prev_end"])), default=None)
        # the debounce/backoff ends at eligible_at; a run that starts later waited for a slot
        slot_wait = _span(row["eligible_at"], row["started_at"]) if row["newest"] else None
        out[row["finished_at"][:10]].append(
            {
                "id": row["id"],
                "status": row["status"],
                "fail_kind": row["fail_kind"],
                "attempt": row["attempt"],
                "tier": row["tier"],
                "degraded": bool(row["degraded"]),
                "repo": row["repo"],
                "number": row["number"],
                "sha": row["sha"],
                "title": row["title"],
                "queue": row["queue"],
                "trigger": row["trigger"],
                "priority": bool(row["priority"]),
                "queued_at": row["queued_at"],
                "started_at": row["started_at"],
                "finished_at": row["finished_at"],
                "wait_seconds": _span(waited_from, row["started_at"]),
                "slot_wait_seconds": slot_wait,
                "duration_seconds": _span(row["started_at"], row["finished_at"]),
                "tokens_in": sum(x["tokens_in"] for x in run_lanes),
                "tokens_out": sum(x["tokens_out"] for x in run_lanes),
                "lanes": run_lanes,
                "findings": findings,
                "new_findings": new_findings,
                "blockers": row["blocker_count"],
                "verdict": verdicts.get(row["id"]),
                "pr_url": pr_url,
                "review_url": row["review_url"] or row["audit_url"] or pr_url,
            }
        )
    return out


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
    # `queue` tells a post-merge audit (of a PR that is merged by definition) apart from a live
    # review; without it an audit reads as a review stuck on a PR that closed hours ago
    for row in conn.execute(
        "SELECT r.id,r.status,r.phase,r.attempt,r.started_at,r.heartbeat_at,r.deadline_at,h.repo,h.number,h.sha,"
        "h.queue,p.title FROM runs r JOIN heads h ON h.id=r.head_id "
        "LEFT JOIN prs p ON p.repo=h.repo AND p.number=h.number "
        "WHERE r.status IN ('spawned','running') ORDER BY r.started_at"
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
            "SELECT ts,kind,repo,number,run_id,detail FROM events WHERE kind NOT LIKE 'audit.%' "
            "ORDER BY id DESC LIMIT 40"
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


def _write_json(target: Path, text: str) -> None:
    tmp = target.with_suffix(".tmp")
    tmp.write_text(text + "\n")
    tmp.replace(target)


def write_export(conn: sqlite3.Connection, cfg: Config, output: Path) -> Path:
    """status.json plus one `days/<YYYY-MM-DD>.json` per day in the throughput list.

    The day files hold nothing relative to now, so a finished day's file never changes again
    and the per-minute publish only re-commits today's; a day that aged out is deleted."""
    output.mkdir(parents=True, exist_ok=True)
    export = build_export(conn, cfg)
    target = output / "status.json"
    _write_json(target, json.dumps(export, indent=2, sort_keys=True))
    days_dir = output / "days"
    days_dir.mkdir(exist_ok=True)
    days = [d["day"] for d in export["history"]["daily"]]
    for day, runs in build_day_runs(conn, days).items():
        # compact: a busy day is ~100 runs of ~9 lanes each, 250 KB indented vs ~110 KB
        body = {"schema_version": 1, "day": day, "runs": runs}
        _write_json(
            days_dir / f"{day}.json", json.dumps(body, separators=(",", ":"), sort_keys=True)
        )
    for stale in days_dir.iterdir():
        # a .tmp left by a crashed write too: the publisher stages this whole directory
        if stale.suffix != ".json" or stale.stem not in days:
            stale.unlink()
    return target
