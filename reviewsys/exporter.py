"""Sanitized observability export for the public status dashboard."""

from __future__ import annotations

import json
import sqlite3
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from . import lanepool, progress
from .config import Config
from .db import fmt_ts, now, now_dt, parse_ts
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


def _run_facts(
    conn: sqlite3.Connection, scope: str, params: tuple[Any, ...]
) -> tuple[
    dict[int, list[dict[str, Any]]],
    dict[int, dict[tuple[str, str], dict[str, str]]],
    dict[int, tuple[str, int | None]],
    dict[int, str],
]:
    """Finished lanes, findings passes, last verification step and verdict of every run
    matching `scope` (a WHERE clause over `runs r`)."""
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
    return lanes, passes, last_verify, verdicts


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
    lanes, passes, last_verify, verdicts = _run_facts(conn, scope, params)
    out: dict[str, list[dict[str, Any]]] = {day: [] for day in days}
    for row in conn.execute(
        "SELECT r.id,r.status,r.fail_kind,r.attempt,r.tier,r.degraded,r.blocker_count,r.review_url,"
        "r.started_at,r.finished_at,r.path,h.repo,h.number,h.sha,h.queue,h.trigger,h.priority,"
        "h.queued_at,h.eligible_at,COALESCE(p.title,a.title,'') title,a.comment_url audit_url,"
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
                "path": row["path"],  # the run's own; the head's fields can change after it
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


# event kinds whose detail is safe on the public page. Failure, degraded and lane-drop details
# quote lane errors (proxy and quota text), and any kind not listed here is withheld too.
_PUBLIC_EVENT_DETAIL = frozenset(
    {
        "head.queued",
        "head.superseded",
        "head.closed",
        "head.priority_requested",
        "run.spawned",
        "run.done",
        "phase1.skipped_backlog",
        "compare.selected",
        "compare.lanes",
    }
)
# step detail keys that are safe and useful on the public page; `error` and free-form failure
# text are not (they can carry proxy and account details)
_STEP_KEYS = (
    "roles",
    "selection",
    "tier",
    "method",
    "model",
    "effort",
    "reasoning",
    "admit_phase2",
    "phase2_effort",
    "blockers",
    "findings",
    "event",
    "posted",
    "fell_through",
)


def _step_info(status: str, detail: str | None) -> dict[str, Any]:
    try:
        raw = json.loads(detail or "{}")
    except ValueError:
        return {}
    if not isinstance(raw, dict):
        return {}
    info = {k: raw[k] for k in _STEP_KEYS if k in raw}
    if isinstance(info.get("reasoning"), str):
        info["reasoning"] = info["reasoning"][:400]
    if status == "skipped" and isinstance(raw.get("reason"), str):
        info["reason"] = raw["reason"][:200]  # e.g. "skipped for throughput: 22 PRs queued"
    return info


def _live_lanes(run_dir: str | None, slot_dir: Path, at: datetime) -> list[dict[str, Any]]:
    """The lanes of a run that are waiting for a pool slot or running right now: the ones
    with a lanepool state file and no lane-meta.json yet (that one is written when a lane
    ends, and the lane's row is in the DB from then on)."""
    if not run_dir:
        return []
    out = []
    lines: dict[str, list[str]] = {}
    root = Path(run_dir)
    # a repair lane runs in its parent lane's `repair/` subdirectory
    found = [
        *root.glob(f"*/*/{lanepool.LANE_STATE}"),
        *root.glob(f"attempts/*/repair/{lanepool.LANE_STATE}"),
    ]
    for state_file in sorted(found):
        lane_dir = state_file.parent
        meta = lane_dir / "lane-meta.json"
        try:
            # a stand-in retry of triage/selection reuses attempt-1, whose first lane-meta.json
            # is already there: only a state file older than it belongs to an ended lane
            if meta.exists() and meta.stat().st_mtime_ns >= state_file.stat().st_mtime_ns:
                continue
        except OSError:
            continue
        try:
            st = json.loads(state_file.read_text())
        except (OSError, ValueError):
            continue
        repair = lane_dir.name == "repair"
        # attempts/<phase>-<role>-<id> for phase lanes, <step>/attempt-N for selector/triage
        named = lane_dir.parent if repair else lane_dir
        phase = (
            named.name.split("-", 1)[0] if named.parent.name == "attempts" else named.parent.name
        )
        lane = {
            "phase": phase,
            "role": f"{st.get('role')} (repair)" if repair else st.get("role"),
            "model": st.get("model"),
            "effort": st.get("effort"),
            "pool": st.get("pool"),
            "state": st.get("state"),
            "since_seconds": _age(st.get("since"), at),
            "line_position": None,
        }
        if lane["state"] == "waiting":  # every lane that waits for a slot is in its pool's line
            pool = str(st.get("pool"))
            line = lines.setdefault(pool, lanepool.waiting_line(slot_dir, pool))
            mine = [i for i, t in enumerate(line) if t.endswith(f"-{st.get('waiter')}")]
            if mine:
                lane["line_position"], lane["line_length"] = mine[0] + 1, len(line)
        out.append(lane)
    return out


def _typical_minutes_by_tier(conn: sqlite3.Connection, at: datetime) -> dict[str, float]:
    """Median minutes of the week's completed runs per tier: what an active run is measured
    against."""
    by_tier: dict[str, list[float]] = {}
    for row in conn.execute(
        # live reviews only, as status.median_run_minutes: an audit forces Phase 2 and more
        "SELECT COALESCE(r.tier,'unknown') tier,r.started_at,r.finished_at FROM runs r "
        "JOIN heads h ON h.id=r.head_id WHERE r.status='done' AND h.queue='live' AND r.finished_at>=?",
        (fmt_ts(at - timedelta(days=7)),),
    ):
        minutes = (parse_ts(row["finished_at"]) - parse_ts(row["started_at"])).total_seconds()
        by_tier.setdefault(row["tier"], []).append(minutes / 60)
    return {t: round(sorted(v)[len(v) // 2], 1) for t, v in by_tier.items()}


# the window of the dashboard's priority-vs-normal timing
PATH_TIMING_DAYS = 7
# v0.23.0 (lanepool.StallClock) started pushing `runs.deadline_at` out by the time a run only
# waited for model slots; a run that started earlier has no such record (it waited for its slot
# before it started, which its wait to start already counts)
SLOT_WAIT_RECORDED_SINCE = "2026-10-01T19:13:32Z"


def _spread(values: list[int]) -> dict[str, int | None]:
    """How many samples, their median and their 90th percentile (seconds)."""
    if not values:
        return {"n": 0, "median_seconds": None, "p90_seconds": None}
    return {
        "n": len(values),
        "median_seconds": round(statistics.median(values)),
        "p90_seconds": round(statistics.quantiles(values, n=10, method="inclusive")[-1]),
    }


def build_path_timing(conn: sqlite3.Connection, cfg: Config, at: datetime) -> dict[str, Any]:
    """Wait to start, slot wait in the run and review time of the week's live reviews, per path
    (`runs.path`, recorded as the run starts: priority or normal, the day table's badge). A run
    whose path is not known (one from before the column whose head was queued again after it)
    is left out.

    Post-merge audits are left out: they take only the capacity live review leaves idle, so
    nobody waits on them. Conversations (a reply on a reviewed commit) are too: seconds long,
    they are not reviews. Every run that started in the window counts, whatever its outcome:
    - wait: as the day table's "Waited" column, from the head's queue time (or the end of the
      head's previous run, for a retry) to the run's start; the normal path's debounce is in
      it. A failed or cancelled run waited just as long before it started, so it counts too.
    - duration: start to finish of completed (`done`) runs only; a failure ends early.
    - slot_wait: of the same completed runs, the time the run only waited for model slots (at
      least one lane in a pool's line and none running): the stall credit added to its
      deadline, less the run timeout configured now (a changed `run_timeout_minutes` skews it
      for a week). Runs started before SLOT_WAIT_RECORDED_SINCE have no such record and are
      left out."""
    since = fmt_ts(at - timedelta(days=PATH_TIMING_DAYS))
    timeout = cfg.run_timeout_minutes * 60
    samples: dict[str, dict[str, list[int]]] = {
        path: {"wait": [], "duration": [], "slot_wait": []}
        for path in (progress.PRIORITY, progress.NORMAL)
    }
    for row in conn.execute(
        "SELECT r.status,r.started_at,r.finished_at,r.deadline_at,r.path,h.queued_at,"
        "(SELECT MAX(o.finished_at) FROM runs o WHERE o.head_id=r.head_id AND o.id<r.id) prev_end "
        "FROM runs r JOIN heads h ON h.id=r.head_id WHERE r.path IN (?,?) AND r.started_at>=? "
        "AND NOT EXISTS (SELECT 1 FROM steps c WHERE c.run_id=r.id AND c.name='converse')",
        (progress.PRIORITY, progress.NORMAL, since),
    ):
        mine = samples[row["path"]]
        waited_from = max(filter(None, (row["queued_at"], row["prev_end"])), default=None)
        if (wait := _span(waited_from, row["started_at"])) is not None:
            mine["wait"].append(wait)
        took = _span(row["started_at"], row["finished_at"])
        if row["status"] != "done" or took is None:
            continue
        mine["duration"].append(took)
        if row["started_at"] >= SLOT_WAIT_RECORDED_SINCE:
            credited = _span(row["started_at"], row["deadline_at"]) or 0
            mine["slot_wait"].append(max(0, credited - timeout))
    return {
        "window_days": PATH_TIMING_DAYS,
        "slot_wait_since": SLOT_WAIT_RECORDED_SINCE,
        "paths": {
            path: {k: _spread(v) for k, v in figures.items()} for path, figures in samples.items()
        },
    }


def build_active_runs(conn: sqlite3.Connection, cfg: Config, at: datetime) -> list[dict[str, Any]]:
    """Every run in flight with everything the page can say about it: steps so far, finished
    and live lanes, tokens and findings so far, and how long it waited to start."""
    scope = "r.status IN ('spawned','running')"
    lanes, passes, last_verify, _ = _run_facts(conn, scope, ())
    steps: dict[int, list[dict[str, Any]]] = {}
    for row in conn.execute(
        "SELECT s.run_id,s.name,s.status,s.started_at,s.finished_at,s.detail FROM steps s "
        f"JOIN runs r ON r.id=s.run_id WHERE {scope} ORDER BY s.started_at, s.rowid"
    ):
        steps.setdefault(row["run_id"], []).append(
            {
                "name": row["name"],
                "status": row["status"],
                "started_at": row["started_at"],
                "seconds": _span(row["started_at"], row["finished_at"])
                if row["finished_at"]
                else _age(row["started_at"], at),
                "info": _step_info(row["status"], row["detail"]),
            }
        )
    slot_dir = cfg.work_dir / "lane-slots"
    profiles: dict[tuple[str, str | None], dict[str, progress.StepStat]] = {}
    runs = []
    # `queue` tells a post-merge audit (of a PR that is merged by definition) apart from a live
    # review; without it an audit reads as a review stuck on a PR that closed hours ago
    for row in conn.execute(
        "SELECT r.id,r.status,r.phase,r.attempt,r.tier,r.degraded,r.started_at,r.heartbeat_at,"
        "r.deadline_at,r.run_dir,h.repo,h.number,h.sha,h.queue,h.trigger,h.priority,h.queued_at,"
        "p.title,p.author,"
        "(SELECT MAX(o.finished_at) FROM runs o WHERE o.head_id=r.head_id AND o.id<r.id) prev_end "
        "FROM runs r JOIN heads h ON h.id=r.head_id "
        "LEFT JOIN prs p ON p.repo=h.repo AND p.number=h.number "
        f"WHERE {scope} ORDER BY r.started_at"
    ):
        run_lanes = lanes.get(row["id"], [])
        findings, _ = _review_findings(passes.get(row["id"], {}), last_verify.get(row["id"]))
        waited_from = max(filter(None, (row["queued_at"], row["prev_end"])), default=None)
        run = dict(row)
        for internal in ("run_dir", "prev_end"):
            del run[internal]
        run.update(
            {
                "degraded": bool(row["degraded"]),
                "priority": bool(row["priority"]),
                "pr_url": f"https://github.com/{row['repo']}/pull/{row['number']}",
                "elapsed_seconds": _age(row["started_at"], at),
                "heartbeat_age_seconds": _age(row["heartbeat_at"], at),
                "deadline_in_seconds": int((parse_ts(row["deadline_at"]) - at).total_seconds())
                if row["deadline_at"]
                else None,
                "wait_seconds": _span(waited_from, row["started_at"]),
                "steps": steps.get(row["id"], []),
                "lanes": run_lanes,
                "live_lanes": _live_lanes(row["run_dir"], slot_dir, at),
                "tokens_in": sum(x["tokens_in"] for x in run_lanes),
                "tokens_out": sum(x["tokens_out"] for x in run_lanes),
                # the last finished verification's findings; None before any verifier ran
                "findings": findings if row["id"] in last_verify else None,
            }
        )
        for st, counts in zip(
            run["steps"], progress.step_lanes(conn, row["id"], run["steps"]), strict=True
        ):
            st.update(counts)
        est = progress.estimate(conn, row["id"], at, profiles)
        run["progress"] = est.progress if est else None
        run["remaining_seconds"] = est.remaining_seconds if est else None
        run["upcoming"] = est.upcoming if est else []
        run["overdue"] = bool(est and est.overdue)
        runs.append(run)
    return runs


def build_lane_pools(
    conn: sqlite3.Connection, cfg: Config, active: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Per model pool: its slot count now, lanes running and lanes waiting in line. A run that
    looks stalled is usually a run whose reviewers are waiting here."""
    slot_dir = cfg.work_dir / "lane-slots"
    live = [lane for run in active for lane in run["live_lanes"]]
    out = []
    for pool in sorted({"gpt", lanepool.COMPARE_POOL, *cfg.lane_pools}):
        mine = [x for x in live if x["pool"] == pool]
        out.append(
            {
                "pool": pool,
                "slots": lanepool.limit(conn, cfg, pool),
                "running": sum(1 for x in mine if x["state"] == "running"),
                # every waiting lane has a ticket in the pool's line; a waiting lane without
                # a place in it is one of a worker still on the code before the line covered
                # every lane (a deploy in progress)
                "waiting": len(lanepool.waiting_line(slot_dir, pool))
                + sum(1 for x in mine if x["state"] == "waiting" and x["line_position"] is None),
            }
        )
    return out


def _queue_reason(head: sqlite3.Row, ts: str) -> str:
    if head["eligible_at"] <= ts:
        return "waiting for a review slot"
    if head["attempts"]:
        return f"retry backoff after attempt {head['attempts']}"
    return "debounce (more pushes may follow)"


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
                "attempts": row["attempts"],
                "eligible_in_seconds": max(
                    0, int((parse_ts(row["eligible_at"]) - at).total_seconds())
                ),
                "reason": _queue_reason(row, ts),
                "pr_url": f"https://github.com/{row['repo']}/pull/{row['number']}",
                "prioritize_url": f"https://github.com/{row['repo']}/issues/{row['number']}#issuecomment-{comment[0]}"
                if comment
                else f"https://github.com/{row['repo']}/pull/{row['number']}#issuecomment-new",
            }
        )
    runs = build_active_runs(conn, cfg, at)
    today = ts[:10]
    today_row = conn.execute(
        "SELECT COUNT(*) FILTER (WHERE status='done') reviews,"
        "COUNT(*) FILTER (WHERE status IN ('failed','timed_out')) failed FROM runs "
        "WHERE finished_at>=?",
        (today,),
    ).fetchone()
    today_tokens = conn.execute(
        "SELECT COALESCE(SUM(COALESCE(tokens_in,0)+COALESCE(tokens_out,0)),0) FROM lanes l "
        "JOIN runs r ON r.id=l.run_id WHERE r.finished_at>=? AND r.status='done'",
        (today,),
    ).fetchone()[0]
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
    recent = []
    for r in conn.execute(
        "SELECT ts,kind,repo,number,run_id,detail FROM events WHERE kind NOT LIKE 'audit.%' "
        "ORDER BY id DESC LIMIT 40"
    ):
        event = dict(r)
        if event["kind"] not in _PUBLIC_EVENT_DETAIL:
            event["detail"] = None
        recent.append(event)
    cap = live.pop("capacity")
    # `typical`/`priority_overflow`/`maximum` are the gpt lane pool's stream budget now (runs
    # are bounded only by `max_runs`); the keys stay for the published page and its readers
    live["capacity"] = {
        "max_runs": cfg.max_runs,
        "typical": cap["normal"],
        "priority_overflow": cap["priority"],
        "maximum": cap["ceiling"],
        "scale": cap["scale"],
        "usable_accounts": cap["usable"],
        "reason": cap["reason"],
        "live_in_flight": sum(1 for r in runs if r["queue"] != "audit"),
    }
    live["priority_queued"] = sum(1 for q in queued if q["priority"])
    live["lane_pools"] = build_lane_pools(conn, cfg, runs)
    live["typical_minutes_by_tier"] = _typical_minutes_by_tier(conn, at)
    live["today"] = {
        "day": today,
        "reviews": today_row["reviews"],
        "failed": today_row["failed"],
        "tokens": today_tokens,
        "findings_posted": conn.execute(
            "SELECT COUNT(*) FROM posted_findings WHERE posted_at>=?", (today,)
        ).fetchone()[0],
    }
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
                # `findings` counts every lane's and verifier's raw rows; this is what reached GitHub
                "findings_posted": conn.execute("SELECT COUNT(*) FROM posted_findings").fetchone()[
                    0
                ],
                "reviews_imported": conn.execute(
                    "SELECT COUNT(*) FROM reviews WHERE imported=1"
                ).fetchone()[0],
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
            "path_timing": build_path_timing(conn, cfg, at),
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
