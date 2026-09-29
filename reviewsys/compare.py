"""Model comparison report: on sampled runs every Phase-2 reviewer lane also ran on a second
model (see `ComparisonPolicy`). The final verifier weighed both sets blind, so the findings
it kept show what each model contributes.

A kept finding is credited to a model when one of that model's Phase-2 lanes raised it: the
same `finding_hash` (file + category + title), or, when the verifier retitled it, the same
file with overlapping lines. Kept = every finding the final verifier(s) of the run kept
(`verify2` / `verified`), before cross-round dedupe.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class _Finding:
    hash: str
    file: str
    lo: int | None
    hi: int | None
    severity: str

    def matches(self, other: _Finding) -> bool:
        if self.hash == other.hash:
            return True
        if not self.file or self.file != other.file or self.lo is None or other.lo is None:
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
    raised: dict[str, int] = field(default_factory=dict)  # model -> Phase-2 findings raised
    kept_by: dict[str, int] = field(default_factory=dict)  # model -> kept findings it raised
    blockers_by: dict[str, int] = field(default_factory=dict)  # model -> kept blockers it raised
    kept: int = 0
    only: dict[str, int] = field(default_factory=dict)  # model -> kept findings only it raised
    blockers_only: dict[str, int] = field(default_factory=dict)
    tokens_out: dict[str, int] = field(default_factory=dict)
    dropped_lanes: int = 0


def _findings(conn: sqlite3.Connection, run_id: int, phase: str, stage: str) -> list[_Finding]:
    return [
        _Finding(str(r["hash"]), r["file"] or "", r["line_start"], r["line_end"], r["severity"])
        for r in conn.execute(
            "SELECT hash, file, line_start, line_end, severity FROM findings "
            "WHERE run_id=? AND phase=? AND stage=?",
            (run_id, phase, stage),
        )
    ]


def _unique(findings: list[_Finding]) -> list[_Finding]:
    seen: dict[str, _Finding] = {}
    for f in findings:
        seen.setdefault(f.hash, f)
    return list(seen.values())


def compare_runs(conn: sqlite3.Connection, *, since: str | None = None) -> list[RunComparison]:
    """Every finished comparison run (newest first), optionally only those started at or after
    `since` (ISO timestamp)."""
    rows = conn.execute(
        "SELECT e.run_id, e.detail, r.tier, h.repo, h.number FROM events e "
        "JOIN runs r ON r.id=e.run_id JOIN heads h ON h.id=r.head_id "
        "WHERE e.kind='compare.selected' AND r.status='done' AND r.started_at >= ? "
        "ORDER BY e.run_id DESC",
        (since or "",),
    ).fetchall()
    out: list[RunComparison] = []
    for row in rows:
        detail = dict(p.split("=", 1) for p in str(row["detail"]).split() if "=" in p)
        primary, second = detail.get("primary", "?"), detail.get("second", "?")
        run_id = int(row["run_id"])
        c = RunComparison(
            run_id=run_id,
            repo=str(row["repo"]),
            number=int(row["number"]),
            tier=str(row["tier"] or ""),
            primary=primary,
            second=second,
        )
        by_model = {
            primary: _unique(_findings(conn, run_id, "phase2", "lane")),
            second: _unique(_findings(conn, run_id, "phase2", "compare")),
        }
        kept = _unique(_findings(conn, run_id, "verify2", "verified"))
        c.kept = len(kept)
        for model, raised in by_model.items():
            c.raised[model] = len(raised)
            hits = [k for k in kept if any(k.matches(f) for f in raised)]
            c.kept_by[model] = len(hits)
            c.blockers_by[model] = sum(1 for k in hits if k.severity == "blocking")
        for model, other in ((primary, second), (second, primary)):
            solo = [
                k
                for k in kept
                if any(k.matches(f) for f in by_model[model])
                and not any(k.matches(f) for f in by_model[other])
            ]
            c.only[model] = len(solo)
            c.blockers_only[model] = sum(1 for k in solo if k.severity == "blocking")
        for lane in conn.execute(
            "SELECT model, SUM(tokens_out) tokens FROM lanes "
            "WHERE run_id=? AND phase='phase2' GROUP BY model",
            (run_id,),
        ):
            c.tokens_out[str(lane["model"])] = int(lane["tokens"] or 0)
        c.dropped_lanes = int(
            conn.execute(
                "SELECT COUNT(*) FROM events WHERE run_id=? AND kind='compare.lane_dropped'",
                (run_id,),
            ).fetchone()[0]
        )
        out.append(c)
    return out


def summarize(runs: list[RunComparison]) -> dict[str, Any]:
    """Totals per model over `runs`."""
    models: dict[str, dict[str, int]] = {}
    for c in runs:
        for model in (c.primary, c.second):
            m = models.setdefault(
                model,
                {
                    "raised": 0,
                    "kept": 0,
                    "kept_blockers": 0,
                    "only": 0,
                    "only_blockers": 0,
                    "tokens_out": 0,
                },
            )
            m["raised"] += c.raised.get(model, 0)
            m["kept"] += c.kept_by.get(model, 0)
            m["kept_blockers"] += c.blockers_by.get(model, 0)
            m["only"] += c.only.get(model, 0)
            m["only_blockers"] += c.blockers_only.get(model, 0)
            m["tokens_out"] += c.tokens_out.get(model, 0)
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
            f"{m}: kept {c.kept_by.get(m, 0)}/{c.raised.get(m, 0)} only {c.only.get(m, 0)}"
            for m in (c.primary, c.second)
        )
        lines.append(f"run {c.run_id} {c.repo}#{c.number} [{c.tier}] kept {c.kept}  {per}")
    return "\n".join(lines) + "\n"
