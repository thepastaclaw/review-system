"""How far along a running review is: a per-tier step profile from recent runs, and an
estimate of the time left for one run against it. Shared by the status export (dashboard
progress bars) and the worker (the live progress in the PR's gate comment).

The estimate is deliberately simple. Each step's typical duration is the median over the
tier's last `PROFILE_RUNS` completed live reviews, weighted by how often the step ran at all
(fresh Phase 2 runs on some reviews, Phase 1 is skipped on some repos). Time left = what the
running steps typically still need (the longest one, since steps of a single-stage review run
side by side) + the weighted durations of the steps after the furthest one the run reached.

A conversation (a reply on a reviewed commit) is measured against other conversations, never
a review profile; a review is not estimated at all until triage has set its tier, since the
tiers differ several-fold. A live review is measured against its own path too (`path_of`:
priority or normal), since a priority run's lanes go ahead of normal ones in every model
pool's line and so spend less of each step waiting for a slot; a path with fewer than
`MIN_PROFILE_RUNS` recent reviews of the tier borrows the tier's profile of both paths. A
running step well past its typical duration marks the estimate `overdue` ("longer than
usual") instead of letting the time left sit at the same figure for as long as it overruns.
"""

from __future__ import annotations

import sqlite3
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .db import parse_ts

# the order steps run in; a step before the furthest one the run reached that it never started
# was skipped (e.g. Phase 1 on a Phase-2-only review), not upcoming
ORDER = (
    "worktree",
    "select",
    "triage",
    "context",
    "phase1",
    "verify1",
    "gate",
    "phase2",
    "verify2",
    "fresh_phase2",
    "fresh_verify2",
    "publish",
    "persistence",
    "converse",
)
LABELS = {
    "worktree": "checkout",
    "select": "lanes chosen",
    "triage": "triage",
    "context": "context",
    "phase1": "Phase 1",
    "verify1": "verify 1",
    "gate": "gate",
    "phase2": "Phase 2",
    "verify2": "verify 2",
    "fresh_phase2": "fresh Phase 2",
    "fresh_verify2": "fresh verify",
    "publish": "publish",
    "persistence": "persistence",
    "converse": "conversation",
}
PROFILE_RUNS = 40
MIN_PROFILE_RUNS = 5  # below this a tier borrows the all-tier profile
UPCOMING_SHARE = 0.5  # a step shown as upcoming runs on at least half of the tier's reviews
OVERDUE_FACTOR = 1.5  # a running step this far past its median makes the estimate overdue
CONVERSATION = "conversation"  # the profile key of conversation runs
# the review paths, as the dashboard's badges name them: a live head with `priority` set (a
# mention, a review request, a ticked priority box, a reply under a finding, `enqueue`), any
# other live head, and a post-merge audit (`queue='audit'`, never priority)
PRIORITY, NORMAL, AUDIT = "priority", "normal", "audit"


def path_of(queue: str | None, priority: object) -> str:
    """The path of a run of a head from `queue` with `priority`: the same split the worker's
    lane-line rank uses (`lanepool.rank_of`). The scheduler records it as `runs.path` when the
    run starts; the dashboard's badges, path filter and timing read that."""
    if queue == "audit":
        return AUDIT
    return PRIORITY if priority else NORMAL


@dataclass(frozen=True, slots=True)
class StepStat:
    median_seconds: float
    share: float  # fraction of the profile's runs that ran the step


@dataclass(slots=True)
class Estimate:
    elapsed_seconds: int
    remaining_seconds: int | None  # None: no profile to measure against
    progress: float | None  # 0..0.99 while running
    upcoming: list[str] = field(default_factory=list)
    overdue: bool = False


def profile(conn: sqlite3.Connection, key: str, path: str | None = None) -> dict[str, StepStat]:
    """Typical duration and frequency of every step on recent completed live runs of `key`: a
    tier's reviews, or CONVERSATION. Reviews never include conversations (they are seconds long
    and would drag every review estimate down). `path` (PRIORITY or NORMAL) narrows a tier's
    reviews to that path while it has MIN_PROFILE_RUNS of them; None = both paths."""
    converse = "EXISTS (SELECT 1 FROM steps c WHERE c.run_id=r.id AND c.name='converse')"

    def load(run_filter: str, params: tuple[Any, ...]) -> tuple[int, dict[str, StepStat]]:
        runs = [
            r[0]
            for r in conn.execute(
                "SELECT r.id FROM runs r JOIN heads h ON h.id=r.head_id "
                f"WHERE r.status='done' AND h.queue='live' AND {run_filter} "
                "ORDER BY r.finished_at DESC LIMIT ?",
                (*params, PROFILE_RUNS),
            )
        ]
        if not runs:
            return 0, {}
        durations: dict[str, list[float]] = {}
        for row in conn.execute(
            "SELECT name,started_at,finished_at FROM steps WHERE status='ok' "
            f"AND finished_at IS NOT NULL AND run_id IN ({','.join('?' * len(runs))})",
            runs,
        ):
            seconds = (parse_ts(row[2]) - parse_ts(row[1])).total_seconds()
            durations.setdefault(row[0], []).append(max(0.0, seconds))
        return len(runs), {
            name: StepStat(statistics.median(v), len(v) / len(runs))
            for name, v in durations.items()
        }

    if key == CONVERSATION:
        return load(converse, ())[1]
    if path in (PRIORITY, NORMAL):
        n, prof = load(f"r.tier=? AND r.path=? AND NOT {converse}", (key, path))
        if n >= MIN_PROFILE_RUNS:
            return prof
    n, prof = load(f"r.tier=? AND NOT {converse}", (key,))
    if n < MIN_PROFILE_RUNS:
        prof = load(f"r.tier IS NOT NULL AND NOT {converse}", ())[1]
    return prof


def estimate(
    conn: sqlite3.Connection,
    run_id: int,
    at: datetime,
    profiles: dict[tuple[str, str | None], dict[str, StepStat]] | None = None,
) -> Estimate | None:
    """Time left for a running run; None when the run does not exist. `profiles` caches
    (tier, path) profiles across calls (the export estimates every active run)."""
    run = conn.execute(
        "SELECT r.started_at, r.tier, r.path, h.queue, h.priority FROM runs r "
        "LEFT JOIN heads h ON h.id=r.head_id WHERE r.id=?",
        (run_id,),
    ).fetchone()
    if run is None:
        return None
    elapsed = max(0, int((at - parse_ts(run[0])).total_seconds()))
    steps = conn.execute(
        "SELECT name,status,started_at FROM steps WHERE run_id=?", (run_id,)
    ).fetchall()
    seen = {s[0] for s in steps}
    key = CONVERSATION if "converse" in seen else run[1]
    if key is None:  # before triage: which tier's profile applies is not known yet
        return Estimate(elapsed, None, None)
    # an audit is measured against live reviews of both paths, as before the path split
    path = run[2] or path_of(run[3], run[4])  # a run started before `runs.path` existed
    cached = (key, None if key == CONVERSATION or path == AUDIT else path)
    if profiles is None:
        profiles = {}
    if cached not in profiles:
        profiles[cached] = profile(conn, *cached)
    prof = profiles[cached]
    if not prof:
        return Estimate(elapsed, None, None)
    in_step = {
        s[0]: max(0.0, (at - parse_ts(s[2])).total_seconds())
        for s in steps
        if s[1] == "running" and s[0] in prof
    }
    running_left = [prof[n].median_seconds - t for n, t in in_step.items()]
    overdue = any(
        t > OVERDUE_FACTOR * prof[n].median_seconds and t > 120 for n, t in in_step.items()
    )
    furthest = max((ORDER.index(s[0]) for s in steps if s[0] in ORDER), default=-1)
    later = [n for n in ORDER[furthest + 1 :] if n in prof and n not in seen]
    remaining = max([0.0, *running_left]) + sum(
        prof[n].median_seconds * prof[n].share for n in later
    )
    progress = min(0.99, elapsed / (elapsed + remaining)) if elapsed + remaining > 0 else 0.0
    return Estimate(
        elapsed_seconds=elapsed,
        remaining_seconds=int(remaining),
        progress=round(progress, 3),
        upcoming=[n for n in later if prof[n].share >= UPCOMING_SHARE],
        overdue=overdue,
    )


# reviewer steps -> the phase their lanes record
_REVIEWER_STEPS = {"phase1": "phase1", "phase2": "phase2", "fresh_phase2": "phase2"}


def step_lanes(
    conn: sqlite3.Connection, run_id: int, steps: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """For each of `steps` (dicts with `name`, `started_at`, `info`), how many of a reviewer
    step's lanes are done: `lanes_done`, and once the worker recorded the planned roles as the
    step started, `lanes_total` and `lanes_left` (the ones still going). {} for other steps.

    A lane's row is written when it ends, so the lanes a step finished are its phase's rows
    recorded after the step started (a fresh Phase 2 reruns the phase2 roles). A comparison
    run's Phase-2 twins carry the primary's role names; the second model, from the run's
    compare.selected event (as `reviewsys compare` reads it), tells them apart."""
    twin = conn.execute(
        "SELECT detail FROM events WHERE run_id=? AND kind='compare.selected' ORDER BY id DESC",
        (run_id,),
    ).fetchone()
    second = (
        dict(p.split("=", 1) for p in str(twin[0]).split() if "=" in p).get("second")
        if twin
        else None
    )
    done: list[tuple[str, str, str]] = []
    twins: list[tuple[str, str, str]] = []
    for r in conn.execute(
        "SELECT phase,role,started_at,model FROM lanes WHERE run_id=? "
        "AND status IN ('completed','repaired') AND phase IN ('phase1','phase2')",
        (run_id,),
    ):
        row = (str(r[0]), str(r[1]), str(r[2]))
        (twins if row[0] == "phase2" and second is not None and r[3] == second else done).append(
            row
        )
    out: list[dict[str, Any]] = []
    for st in steps:
        phase = _REVIEWER_STEPS.get(st["name"])
        if phase is None:
            out.append({})
            continue
        finished = {r for p, r, at in done if p == phase and at >= st["started_at"]}
        planned = (st.get("info") or {}).get("roles")
        if not isinstance(planned, list):
            out.append({"lanes_done": len(finished)})
            continue
        counts: dict[str, Any] = {
            "lanes_done": sum(1 for r in planned if r in finished),
            "lanes_total": len(planned),
            "lanes_left": [r for r in planned if r not in finished],
        }
        # the first Phase 2 of a comparison run waits (up to COMPARE_GRACE_MINUTES) for the
        # second model's twin of every role after its own lanes are done
        if st["name"] == "phase2" and second is not None:
            twin_done = {r for p, r, at in twins if at >= st["started_at"]}
            counts["comparison_left"] = sum(1 for r in planned if r not in twin_done)
        out.append(counts)
    return out
