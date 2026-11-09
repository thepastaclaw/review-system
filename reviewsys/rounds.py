"""Review rounds per PR: how many head commits each PR needed the bot to fully review."""

from __future__ import annotations

import math
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from .db import fmt_ts, now_dt, parse_ts

TARGET = 4.0
CEILING = 5.0
MIN_N = 5
WATCH_AT = 5

DEFINITION = (
    "A round is a distinct head commit of a PR that got its own full final review from the bot: "
    "a `reviews` row with phase 'final' and imported=0, counted once per (repo, number, sha). "
    "Carried heads, CI updates and conversation follow-ups reuse a reviewed sha and add no round. "
    "Completed PRs are those no longer open, dated by the moment the poll saw them close "
    "(prs.updated_at, which a closed PR is never touched after). Only completed PRs with at least "
    "one round count. Rounds to first APPROVE is the position of the first round whose sha "
    "received an APPROVE verdict, among PRs that got one. "
    f"Target: mean <= {TARGET:g}, ceiling {CEILING:g}; a repo is flagged only at n >= {MIN_N}."
)


def _percentile(sorted_values: list[int], q: float) -> int:
    return sorted_values[max(0, math.ceil(q * len(sorted_values)) - 1)]


def _mean(values: list[int]) -> float | None:
    return round(statistics.fmean(values), 2) if values else None


def _median(values: list[int]) -> float | None:
    return round(statistics.median(values), 2) if values else None


def _stats(prs: list[dict[str, Any]]) -> dict[str, Any]:
    rounds = sorted(p["rounds"] for p in prs)
    to_approve = [p["to_approve"] for p in prs if p["to_approve"] is not None]
    n = len(rounds)
    mean = _mean(rounds)
    enough = n >= MIN_N
    return {
        "n": n,
        "mean": mean,
        "median": _median(rounds),
        "p90": _percentile(rounds, 0.9) if n else None,
        "max": rounds[-1] if n else None,
        "share_over_5": round(sum(1 for r in rounds if r > 5) / n, 3) if n else None,
        "approved_n": len(to_approve),
        "to_approve_mean": _mean(to_approve),
        "to_approve_median": _median(to_approve),
        "above_target": bool(enough and mean is not None and mean > TARGET),
        "above_ceiling": bool(enough and mean is not None and mean > CEILING),
    }


def _brief(prs: list[dict[str, Any]]) -> dict[str, Any]:
    rounds = [p["rounds"] for p in prs]
    return {"n": len(rounds), "mean": _mean(rounds), "median": _median(rounds)}


def _load(conn: sqlite3.Connection, repo: str | None) -> list[dict[str, Any]]:
    where = "AND r.repo=?" if repo else ""
    params: tuple[Any, ...] = (repo,) if repo else ()
    by_pr: dict[tuple[str, int], list[tuple[str, bool]]] = defaultdict(list)
    for row in conn.execute(
        "SELECT r.repo, r.number, r.sha, MIN(r.posted_at) first_at, "
        "MAX(r.event='APPROVE') approved FROM reviews r "
        f"WHERE r.phase='final' AND r.imported=0 {where} "
        "GROUP BY r.repo, r.number, r.sha ORDER BY first_at, MIN(r.id)",
        params,
    ):
        by_pr[(row["repo"], row["number"])].append((row["sha"], bool(row["approved"])))
    out = []
    for row in conn.execute("SELECT repo, number, author, state, updated_at FROM prs"):
        shas = by_pr.get((row["repo"], row["number"]))
        if not shas:
            continue
        first_ok = next((i for i, (_, ok) in enumerate(shas, 1) if ok), None)
        out.append(
            {
                "repo": row["repo"],
                "number": row["number"],
                "author": row["author"] or "",
                "open": row["state"] == "open",
                "completed_at": None if row["state"] == "open" else row["updated_at"],
                "rounds": len(shas),
                "to_approve": first_ok,
            }
        )
    return out


def _week_label(ts: datetime) -> str:
    year, week, _ = ts.isocalendar()
    return f"{year}-W{week:02d}"


def _repo_stats(prs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in prs:
        grouped[p["repo"]].append(p)
    rows = [{"repo": repo, **_stats(group)} for repo, group in grouped.items()]
    rows.sort(key=lambda r: (-r["n"], r["repo"]))
    return rows


def compute(
    conn: sqlite3.Connection,
    *,
    days: int = 28,
    weeks: int = 8,
    repo: str | None = None,
    at: datetime | None = None,
) -> dict[str, Any]:
    at = at or now_dt()
    prs = _load(conn, repo)
    done = [p for p in prs if not p["open"]]
    done_at = {id(p): parse_ts(p["completed_at"]) for p in done}

    def completed_since(since: datetime) -> list[dict[str, Any]]:
        return [p for p in done if since <= done_at[id(p)] <= at]

    window = completed_since(at - timedelta(days=days))

    monday = (at - timedelta(days=at.weekday())).replace(hour=0, minute=0, second=0)
    starts = [monday - timedelta(weeks=i) for i in range(weeks - 1, -1, -1)]
    week_prs: dict[str, list[dict[str, Any]]] = {_week_label(s): [] for s in starts}
    for p in done:
        label = _week_label(done_at[id(p)])
        if label in week_prs:
            week_prs[label].append(p)
    trend_repos = sorted({p["repo"] for group in week_prs.values() for p in group})
    trend = {
        "weeks": list(week_prs),
        "overall": [{"week": w, **_brief(g)} for w, g in week_prs.items()],
        "repos": [
            {
                "repo": r,
                "weeks": [
                    {"week": w, **_brief([p for p in g if p["repo"] == r])}
                    for w, g in week_prs.items()
                ],
            }
            for r in trend_repos
        ],
    }

    rolling: dict[str, Any] = {}
    for span in (7, 28):
        sel = completed_since(at - timedelta(days=span))
        rolling[f"{span}d"] = {
            "overall": _brief(sel),
            "repos": {
                r: _brief([p for p in sel if p["repo"] == r])
                for r in sorted({p["repo"] for p in sel})
            },
        }

    watch = sorted(
        (
            {"repo": p["repo"], "number": p["number"], "author": p["author"], "rounds": p["rounds"]}
            for p in prs
            if p["open"] and p["rounds"] >= WATCH_AT
        ),
        key=lambda w: (-w["rounds"], w["repo"], w["number"]),
    )

    return {
        "definition": DEFINITION,
        "generated_at": fmt_ts(at),
        "target": TARGET,
        "ceiling": CEILING,
        "min_n": MIN_N,
        "window_days": days,
        "overall": _stats(window),
        "repos": _repo_stats(window),
        "trend": trend,
        "rolling": rolling,
        "watch": watch,
    }


def _f(value: float | None, width: int = 4) -> str:
    return f"{value:.1f}".rjust(width) if value is not None else "-".rjust(width)


def _flag(row: dict[str, Any]) -> str:
    if row["above_ceiling"]:
        return "ABOVE CEILING"
    if row["above_target"]:
        return "ABOVE TARGET"
    return ""


def render(data: dict[str, Any]) -> str:
    rows = [{"repo": "ALL", **data["overall"]}, *data["repos"]]
    width = max(len(r["repo"]) for r in rows)
    lines = [
        f"Review rounds per PR - PRs completed in the last {data['window_days']}d "
        f"(target mean <= {data['target']:g}, ceiling {data['ceiling']:g})",
        f"{'repo'.ljust(width)}  {'n':>4} {'mean':>5} {'med':>4} {'p90':>4} {'max':>4} "
        f"{'>5':>4} {'1stOK':>5}  flag",
    ]
    for r in rows:
        share = f"{round(r['share_over_5'] * 100)}%" if r["share_over_5"] is not None else "-"
        lines.append(
            f"{r['repo'].ljust(width)}  {r['n']:>4} {_f(r['mean'], 5)} {_f(r['median'])} "
            f"{r['p90'] if r['p90'] is not None else '-':>4} "
            f"{r['max'] if r['max'] is not None else '-':>4} {share:>4} "
            f"{_f(r['to_approve_mean'], 5)}  {_flag(r)}".rstrip()
        )
    roll = data["rolling"]
    lines.append("")
    lines.append(
        "rolling mean (n): "
        + ", ".join(
            f"{k} {_f(v['overall']['mean'], 0).strip()} ({v['overall']['n']})"
            for k, v in roll.items()
        )
    )
    trend = data["trend"]
    if trend["weeks"]:
        lines.append("")
        lines.append("weekly trend by completion week, cells mean/median/n")
        cell = 9
        lines.append("".ljust(width) + "  " + "".join(w[-3:].rjust(cell) for w in trend["weeks"]))

        def row(label: str, weeks: list[dict[str, Any]]) -> str:
            cells = [
                (f"{w['mean']:.1f}/{w['median']:g}/{w['n']}" if w["n"] else "-").rjust(cell)
                for w in weeks
            ]
            return label.ljust(width) + "  " + "".join(cells)

        lines.append(row("ALL", trend["overall"]))
        lines.extend(row(r["repo"], r["weeks"]) for r in trend["repos"])
    lines.append("")
    if data["watch"]:
        lines.append(f"open PRs at >= {WATCH_AT} rounds")
        lines.extend(
            f"  {w['repo']}#{w['number']}  {w['author'] or '-'}  {w['rounds']} rounds"
            for w in data["watch"]
        )
    else:
        lines.append(f"no open PR at >= {WATCH_AT} rounds")
    return "\n".join(lines) + "\n"
