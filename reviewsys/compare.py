"""Model comparison report: on sampled runs every Phase-2 reviewer lane also ran on a second
model (see `ComparisonPolicy`). The final verifier weighed both sets under neutral labels, so
the findings it kept show what each model contributes.

The comparison is per role and first round only. A role counts when both its primary lane
and its comparison lane finished and the comparison lane was not dropped; a comparison lane
can be dropped (failed, out of grace, no slot) while its primary ran, and crediting that
role's findings to the primary alone would tilt the result. Only the first Phase 2 and its
verifier count: a fresh final pass (stages `fresh:*` / `verified-fresh`) re-runs the primary
alone.

A kept finding is credited to a model when one of that model's paired lanes raised it: the
same `finding_hash` (file + category + title), or, when the verifier retitled it, the same
file and category with overlapping lines. A kept finding only an unpaired primary role
raised is left out.

A run counts when it finished, never went degraded (stand-ins would be credited to the
primary), ran nothing but the two models in its first Phase 2, and has at least one paired
role. A run the Phase-1 gate stopped has no Phase 2 and is left out.
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
    roles: list[str] = field(default_factory=list)  # the paired roles
    unpaired: list[str] = field(default_factory=list)  # primary roles without a comparison
    kept: int = 0  # comparable findings the final verifier kept
    models: dict[str, dict[str, int]] = field(default_factory=dict)  # model -> STATS


def _findings(
    conn: sqlite3.Connection, run_id: int, phase: str, stages: list[str]
) -> list[_Finding]:
    """The run's findings at one phase in any of `stages`, one per `finding_hash`."""
    if not stages:
        return []
    seen: dict[str, _Finding] = {}
    for r in conn.execute(
        "SELECT hash, file, category, line_start, line_end, severity FROM findings "
        f"WHERE run_id=? AND phase=? AND stage IN ({','.join('?' * len(stages))}) ORDER BY id",
        (run_id, phase, *stages),
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


def _compare_run(conn: sqlite3.Connection, row: sqlite3.Row) -> RunComparison | None:
    run_id = int(row["run_id"])
    detail = _fields(row["detail"])
    primary, second = detail.get("primary", "?"), detail.get("second", "?")
    first_verify = conn.execute(
        "SELECT MIN(id) FROM lanes WHERE run_id=? AND phase='verify2'", (run_id,)
    ).fetchone()[0]
    if first_verify is None:
        return None
    done: dict[str, dict[str, int]] = {}  # model -> role -> tokens out (first Phase 2 only)
    for r in conn.execute(
        "SELECT role, model, tokens_out FROM lanes WHERE run_id=? AND phase='phase2' "
        "AND id < ? AND status IN ('completed','repaired','corrected')",
        (run_id, first_verify),
    ):
        done.setdefault(str(r["model"]), {})[str(r["role"])] = int(r["tokens_out"] or 0)
    if not set(done) <= {primary, second}:
        return None
    dropped = {
        str(d["detail"]).split(" ", 1)[0].removeprefix("phase2/").removesuffix("#2")
        for d in conn.execute(
            "SELECT detail FROM events WHERE run_id=? AND kind='compare.lane_dropped'", (run_id,)
        )
    }
    ran, twins = done.get(primary, {}), done.get(second, {})
    roles = sorted(r for r in ran if r in twins and r not in dropped)
    if not roles:
        return None
    unpaired = sorted(set(ran) - set(roles))
    by_model = {
        primary: _findings(conn, run_id, "phase2", [f"lane:{r}" for r in roles]),
        second: _findings(conn, run_id, "phase2", [f"compare:{r}" for r in roles]),
    }
    elsewhere = _findings(conn, run_id, "phase2", [f"lane:{r}" for r in unpaired])
    kept: list[_Finding] = []
    raisers: list[set[str]] = []
    for k in _findings(conn, run_id, "verify2", ["verified"]):
        who = {m for m, fs in by_model.items() if any(k.matches(f) for f in fs)}
        if any(k.matches(f) for f in elsewhere):
            if not who:
                continue  # only an unpaired role raised it: nothing to compare
            who.add(primary)  # the primary found it too, in a role the second model skipped
        kept.append(k)
        raisers.append(who)
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
            "tokens_out": sum(done.get(m, {}).get(r, 0) for r in roles),
        }
    return RunComparison(
        run_id=run_id,
        repo=str(row["repo"]),
        number=int(row["number"]),
        tier=str(row["tier"] or ""),
        primary=primary,
        second=second,
        roles=roles,
        unpaired=unpaired,
        kept=len(kept),
        models=models,
    )


def compare_runs(conn: sqlite3.Connection, *, since: str | None = None) -> list[RunComparison]:
    """Every comparison run with at least one paired role (newest first), optionally only
    those started at or after `since` (ISO timestamp)."""
    rows = conn.execute(
        "SELECT e.run_id, e.detail, r.tier, h.repo, h.number "
        "FROM events e JOIN runs r ON r.id=e.run_id JOIN heads h ON h.id=r.head_id "
        "WHERE e.kind='compare.selected' AND r.status='done' AND r.degraded=0 "
        "AND r.started_at >= ? ORDER BY e.run_id DESC",
        (since or "",),
    ).fetchall()
    return [c for row in rows if (c := _compare_run(conn, row)) is not None]


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
        "paired_roles": sum(len(c.roles) for c in runs),
        "unpaired_roles": sum(len(c.unpaired) for c in runs),
        "kept_findings": sum(c.kept for c in runs),
        "models": models,
    }


def render(runs: list[RunComparison]) -> str:
    """A plain-text table: totals, then one line per run."""
    s = summarize(runs)
    lines = [
        f"{s['runs']} comparison runs, {s['paired_roles']} paired reviewer roles "
        f"({s['unpaired_roles']} left out: comparison lane dropped), "
        f"{s['kept_findings']} comparable findings kept by the final verifier",
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
        lines.append(
            f"run {c.run_id} {c.repo}#{c.number} [{c.tier}] {len(c.roles)} roles, "
            f"kept {c.kept}  {per}"
        )
    return "\n".join(lines) + "\n"
