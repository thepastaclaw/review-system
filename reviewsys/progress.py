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
tiers differ several-fold. A running step well past its typical duration marks the estimate
`overdue` instead of letting the time left sit at the same figure for as long as it overruns.
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


def profile(conn: sqlite3.Connection, key: str) -> dict[str, StepStat]:
    """Typical duration and frequency of every step on recent completed live runs of `key`: a
    tier's reviews, or CONVERSATION. Reviews never include conversations (they are seconds long
    and would drag every review estimate down)."""
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
    n, prof = load(f"r.tier=? AND NOT {converse}", (key,))
    if n < MIN_PROFILE_RUNS:
        prof = load(f"r.tier IS NOT NULL AND NOT {converse}", ())[1]
    return prof


def estimate(
    conn: sqlite3.Connection,
    run_id: int,
    at: datetime,
    profiles: dict[str, dict[str, StepStat]] | None = None,
) -> Estimate | None:
    """Time left for a running run; None when the run does not exist. `profiles` caches
    tier profiles across calls (the export estimates every active run)."""
    run = conn.execute("SELECT started_at, tier FROM runs WHERE id=?", (run_id,)).fetchone()
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
    if profiles is None:
        profiles = {}
    if key not in profiles:
        profiles[key] = profile(conn, key)
    prof = profiles[key]
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
