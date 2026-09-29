"""Model comparison report: on sampled runs every Phase-2 reviewer lane also ran on a second
model (see `ComparisonPolicy`). The final verifier weighed both sets under neutral labels, so
the findings it kept show what each model contributes.

A kept finding is credited to a model when one of that model's Phase-2 lanes raised it: the
same `finding_hash` (file + category + title), or, when the verifier retitled it, the same
file and category with overlapping lines. Kept = every finding the run's final verifier(s)
kept (`verify2` / `verified`), before cross-round dedupe.

Only runs that really compared the two count: finished, never degraded (stand-ins would be
credited to the primary), at least one comparison lane kept (event `compare.lanes`), and
every Phase-2 lane on one of the two models. A run the Phase-1 gate stopped has no Phase 2
and is left out.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

STATS = ("raised", "kept", "kept_blockers", "only", "only_blockers", "tokens_out")


@dataclass(frozen=True, slots=True)
class _Finding:
    hash: str
    file: str
    category: str
    lo: int | None
    hi: int | None
    severity: str

    def matches(self, other: _Finding) -> bool:
        if self.hash == other.hash:
            return True
        if not self.file or (self.file, self.category) != (other.file, other.category):
            return False
        if self.lo is None or other.lo is None:
            return False
        return self.lo <= (other.hi or other.lo) and other.lo <= (self.hi or self.lo)


@dataclass(slots=True)
class RunComparison:
    run_id: int
    repo: str
    number: int
    tier: str
    primary: str
    second: str
    kept: int = 0
    dropped_lanes: int = 0
    models: dict[str, dict[str, int]] = field(default_factory=dict)  # model -> STATS


def _findings(conn: sqlite3.Connection, run_id: int, phase: str, stage: str) -> list[_Finding]:
    """The run's findings at one phase/stage, one per `finding_hash`."""
    seen: dict[str, _Finding] = {}
    for r in conn.execute(
        "SELECT hash, file, category, line_start, line_end, severity FROM findings "
        "WHERE run_id=? AND phase=? AND stage=? ORDER BY id",
        (run_id, phase, stage),
    ):
        seen.setdefault(
            str(r["hash"]),
            _Finding(
                str(r["hash"]),
                r["file"] or "",
                r["category"] or "",
                r["line_start"],
                r["line_end"],
                r["severity"],
            ),
        )
    return list(seen.values())


def _fields(detail: str | None) -> dict[str, str]:
    """`k=v` pairs of an event detail."""
    return dict(p.split("=", 1) for p in str(detail or "").split() if "=" in p)


def _compared(conn: sqlite3.Connection, run_id: int, primary: str, second: str) -> bool:
    """Did this run's Phase 2 really run both models, and only those?"""
    ran = {
        str(r["model"])
        for r in conn.execute(
            "SELECT DISTINCT model FROM lanes WHERE run_id=? AND phase='phase2' "
            "AND status IN ('completed','repaired')",
            (run_id,),
        )
    }
    return second in ran and ran <= {primary, second}


def compare_runs(conn: sqlite3.Connection, *, since: str | None = None) -> list[RunComparison]:
    """Every comparison run that really compared the two models (newest first), optionally
    only those started at or after `since` (ISO timestamp)."""
    rows = conn.execute(
        "SELECT e.run_id, e.detail, r.tier, h.repo, h.number, "
        "(SELECT l.detail FROM events l WHERE l.run_id=e.run_id "
        "AND l.kind='compare.lanes' ORDER BY l.id LIMIT 1) lanes "
        "FROM events e JOIN runs r ON r.id=e.run_id JOIN heads h ON h.id=r.head_id "
        "WHERE e.kind='compare.selected' AND r.status='done' AND r.degraded=0 "
        "AND r.started_at >= ? ORDER BY e.run_id DESC",
        (since or "",),
    ).fetchall()
    out: list[RunComparison] = []
    for row in rows:
        detail = _fields(row["detail"])
        primary, second = detail.get("primary", "?"), detail.get("second", "?")
        lanes = _fields(row["lanes"])
        run_id = int(row["run_id"])
        if not int(lanes.get("kept", 0)) or not _compared(conn, run_id, primary, second):
            continue
        by_model = {
            primary: _findings(conn, run_id, "phase2", "lane"),
            second: _findings(conn, run_id, "phase2", "compare"),
        }
        kept = _findings(conn, run_id, "verify2", "verified")
        # which models raised each kept finding
        raisers = [{m for m, fs in by_model.items() if any(k.matches(f) for f in fs)} for k in kept]
        tokens = {
            str(r["model"]): int(r["tokens"] or 0)
            for r in conn.execute(
                "SELECT model, SUM(tokens_out) tokens FROM lanes "
                "WHERE run_id=? AND phase='phase2' GROUP BY model",
                (run_id,),
            )
        }
        models: dict[str, dict[str, int]] = {}
        for m, raised in by_model.items():
            hits = [k for k, r in zip(kept, raisers, strict=True) if m in r]
            solo = [k for k, r in zip(kept, raisers, strict=True) if r == {m}]
            models[m] = {
                "raised": len(raised),
                "kept": len(hits),
                "kept_blockers": sum(k.severity == "blocking" for k in hits),
                "only": len(solo),
                "only_blockers": sum(k.severity == "blocking" for k in solo),
                "tokens_out": tokens.get(m, 0),
            }
        out.append(
            RunComparison(
                run_id=run_id,
                repo=str(row["repo"]),
                number=int(row["number"]),
                tier=str(row["tier"] or ""),
                primary=primary,
                second=second,
                kept=len(kept),
                dropped_lanes=int(lanes.get("dropped", 0)),
                models=models,
            )
        )
    return out


def summarize(runs: list[RunComparison]) -> dict[str, Any]:
    """Totals per model over `runs`."""
    models: dict[str, dict[str, int]] = {}
    for c in runs:
        for model, stats in c.models.items():
            m = models.setdefault(model, dict.fromkeys(STATS, 0))
            for s in STATS:
                m[s] += stats[s]
    return {
        "runs": len(runs),
        "kept_findings": sum(c.kept for c in runs),
        "dropped_comparison_lanes": sum(c.dropped_lanes for c in runs),
        "models": models,
    }


def render(runs: list[RunComparison]) -> str:
    """A plain-text table: totals, then one line per run."""
    s = summarize(runs)
    lines = [
        f"{s['runs']} comparison runs, {s['kept_findings']} findings kept by the final verifier, "
        f"{s['dropped_comparison_lanes']} comparison lanes dropped",
        "",
        f"{'model':<28} {'raised':>7} {'kept':>6} {'kept🔴':>7} {'only':>6} {'only🔴':>7} {'tok out':>10}",
    ]
    for model, m in s["models"].items():
        lines.append(
            f"{model:<28} {m['raised']:>7} {m['kept']:>6} {m['kept_blockers']:>7} "
            f"{m['only']:>6} {m['only_blockers']:>7} {m['tokens_out']:>10}"
        )
    lines += ["", "only = kept findings no lane of the other model raised; 🔴 = blocking", ""]
    for c in runs:
        per = "  ".join(
            f"{m}: kept {st['kept']}/{st['raised']} only {st['only']}" for m, st in c.models.items()
        )
        lines.append(f"run {c.run_id} {c.repo}#{c.number} [{c.tier}] kept {c.kept}  {per}")
    return "\n".join(lines) + "\n"
