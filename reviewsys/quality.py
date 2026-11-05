"""`reviewsys quality`: does the review keep its word across rounds? Production metrics from the
issue ledger (ledger.py; run `reviewsys ledger backfill` first for history before the
write-through) plus `git diff` on the repos' mirrors. Definitions follow the v12 design
(DESIGN-opus.md §1 and Appendix A):

- M1 late-find rate: posted blockers raised in round >= 2 on lines (anchor ±3) unchanged since
  the previous reviewed head, over the checkable posted blockers (round 1 when its commit is in
  the mirror; later rounds when git can compare it with the previous head). n/a when no
  later-round blocker could be checked.
- M3 pushback: blockers with a later round whose last thread outcome is WITHDRAWN or
  INTENTIONALLY_DEFERRED.
- M4 re-raise: a withdrawn or deferred issue posted again (same hash or title, in any file;
  or, looser, the same file within ±15 lines) without its anchored lines having changed.
- M5 fate coverage: blockers with a later round whose fate is on record (a thread outcome, or
  carried by a later verifier set); the rest "vanished".
- M6 silent drop: vanished blockers whose file did not change by the next reviewed head.
- M9 contract failures: runs failed `contract`, by class (reconciliation, echo field, other).

A "later round" is a later done (non-audit) run at a different sha; "reviewed heads" are the
distinct shas with a published, non-imported review, in order of their first review."""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import ledger

LATE_MARGIN = 3  # anchor ± lines, M1 (Appendix A.3)
REVIVAL_MARGIN = 3  # anchor ± lines, M4 (the v12 settled-revival rule)
NEAR_LINES = 15  # M4 "by location"
# a maintainer's call against the finding: the outcomes that settle an issue that way, and the
# statuses (an explicit transition records the status itself)
_PUSHED_BACK = frozenset({"withdrawn", "deferred"})
_PUSHBACK = _PUSHED_BACK | {k for k, v in ledger.OUTCOME_STATUS.items() if v in _PUSHED_BACK}
_RECONCILIATION_RE = re.compile(r"reconciliation|STILL_VALID set|prior hash", re.I)
_ECHO_RE = re.compile(r"\b(review_phase|head_sha)=")


@dataclass(slots=True)
class RepoPaths:
    """Where a repo's git objects are: an explicit path, else `<mirrors_dir>/<owner>__<name>.git`."""

    mirrors_dir: Path | None = None
    overrides: dict[str, Path] = field(default_factory=dict)

    def path(self, repo: str) -> Path | None:
        if repo in self.overrides:
            return self.overrides[repo]
        if self.mirrors_dir is not None:
            p = self.mirrors_dir / f"{repo.replace('/', '__')}.git"
            if p.exists():
                return p
        return None


class _Git:
    """Cached git facts per repo: `git diff -U0` per (from, to, file), commits present; None
    (or False) when they cannot be had. Counts the checks skipped for want of a mirror."""

    def __init__(self, paths: RepoPaths) -> None:
        self.paths = paths
        self.cache: dict[tuple[str, str, str, str], str | None] = {}
        self.commits: dict[tuple[str, str], bool] = {}
        self.missing: Counter[str] = Counter()

    def _path(self, repo: str) -> Path | None:
        p = self.paths.path(repo)
        if p is None:
            self.missing[repo] += 1
        return p

    def has(self, repo: str, sha: str) -> bool:
        if (repo, sha) not in self.commits:
            p = self._path(repo)
            self.commits[(repo, sha)] = p is not None and bool(sha) and ledger.git_has(p, sha)
        return self.commits[(repo, sha)]

    def diff(self, repo: str, a: str, b: str, file: str) -> str | None:
        """The diff, or None when git cannot tell: no mirror, a commit missing, or a path on
        neither side (an empty diff then means a wrong path, not unchanged code)."""
        key = (repo, a, b, file)
        if key not in self.cache:
            p = self._path(repo)
            d = ledger.git_diff_u0(p, a, b, file) if p is not None else None
            if p is not None and d is not None and not d.strip() and not ledger.git_has(p, a, file):
                d = None
            self.cache[key] = d
        return self.cache[key]

    def anchor(
        self, repo: str, at: str, other: str, file: str, start: int | None, end: int | None
    ) -> str | None:
        """How the code at `file:start..end` as of `at` differs at `other`: `file_unchanged`,
        `new_file` (the file is missing on one side), `anchor_changed`, `anchor_unchanged`;
        None when git cannot tell."""
        d = self.diff(repo, at, other, file)
        if d is None:
            return None
        if not d.strip():
            return "file_unchanged"
        if any(x in ("--- /dev/null", "+++ /dev/null") for x in d.splitlines()):
            return "new_file"
        touched = ledger.anchor_touched(d, start, end, LATE_MARGIN)
        return "anchor_changed" if touched else "anchor_unchanged"

    def changed(
        self, repo: str, a: str, b: str, file: str, start: int | None, end: int | None
    ) -> bool | None:
        """`ledger.lines_changed` on the cached diff (the anchor is in `a`'s lines)."""
        if a == b:
            return False
        d = self.diff(repo, a, b, file)
        return None if d is None else ledger.diff_touches(d, start, end, REVIVAL_MARGIN)


def _rate(n: int, d: int) -> float | None:
    return round(100.0 * n / d, 1) if d else None


def _norm(title: Any) -> str:
    return " ".join(str(title or "").lower().split())


@dataclass(slots=True)
class _Post:
    repo: str
    number: int
    hash: str
    run_id: int | None
    sha: str
    ts: str
    file: str
    line_start: int | None
    line_end: int | None
    severity: str
    title: str


def _load(conn: sqlite3.Connection) -> dict[str, Any]:
    posts: list[_Post] = []
    marks: dict[tuple[str, int, str], list[tuple[str, str, str | None]]] = defaultdict(list)
    seen: dict[tuple[str, int, str], list[tuple[str, int | None]]] = defaultdict(list)
    for e in conn.execute(
        "SELECT repo, number, hash, kind, run_id, sha, ts, to_status, detail FROM issue_events "
        "ORDER BY ts, id"
    ):
        key = (str(e["repo"]), int(e["number"]), str(e["hash"]))
        d = json.loads(e["detail"] or "{}")
        if e["kind"] == "posted":
            posts.append(
                _Post(
                    *key,
                    run_id=e["run_id"],
                    sha=str(e["sha"] or ""),
                    ts=str(e["ts"]),
                    file=str(d.get("file") or ""),
                    line_start=d.get("line_start"),
                    line_end=d.get("line_end"),
                    severity=str(d.get("severity") or ""),
                    title=str(d.get("title") or ""),
                )
            )
        elif e["kind"] == "seen":
            seen[key].append((str(e["ts"]), e["run_id"]))
        elif e["kind"] == "outcome" and d.get("delivered"):
            marks[key].append((str(e["ts"]), str(d.get("outcome") or ""), e["sha"]))
        elif e["kind"] == "status" or (e["kind"] == "thread" and e["to_status"] == "resolved"):
            marks[key].append((str(e["ts"]), str(e["to_status"] or ""), e["sha"]))
    runs = {
        int(r["id"]): r
        for r in conn.execute(
            "SELECT r.id, r.status, r.started_at, h.repo, h.number, h.sha, h.queue FROM runs r "
            "JOIN heads h ON h.id=r.head_id"
        )
    }
    done: dict[tuple[str, int], list[tuple[str, str]]] = defaultdict(list)
    for r in runs.values():
        if r["status"] == "done" and r["queue"] == "live":
            done[(str(r["repo"]), int(r["number"]))].append((str(r["started_at"]), str(r["sha"])))
    heads: dict[tuple[str, int], list[tuple[str, str, bool]]] = defaultdict(list)
    for r in conn.execute(
        "SELECT v.repo, v.number, v.sha, min(v.posted_at) AS t, max(v.phase='final') AS final "
        "FROM reviews v LEFT JOIN runs r ON r.id=v.run_id LEFT JOIN heads h ON h.id=r.head_id "
        "WHERE v.imported=0 AND coalesce(h.queue, 'live')='live' GROUP BY 1, 2, 3 ORDER BY t"
    ):
        heads[(str(r["repo"]), int(r["number"]))].append(
            (str(r["sha"]), str(r["t"]), bool(r["final"]))
        )
    by_issue: dict[tuple[str, int, str], list[_Post]] = defaultdict(list)
    by_pr: dict[tuple[str, int], list[_Post]] = defaultdict(list)
    for p in posts:
        by_issue[(p.repo, p.number, p.hash)].append(p)
        by_pr[(p.repo, p.number)].append(p)
    return {
        "posts": posts,
        "by_issue": by_issue,
        "by_pr": by_pr,
        "marks": marks,
        "seen": seen,
        "runs": runs,
        "done": done,
        "heads": heads,
    }


def _fate(data: dict[str, Any], p: _Post) -> str:
    """The issue's fate after its first post (Appendix A.1): its last recorded outcome, else
    `carried` when a later verifier set (or a later publication) kept it, else `vanished`."""
    key = (p.repo, p.number, p.hash)
    later = [m for m in data["marks"].get(key, []) if m[0] >= p.ts]
    if later:
        return str(later[-1][1])
    if any(ts > p.ts and rid != p.run_id for ts, rid in data["seen"].get(key, [])):
        return "carried"
    if any(q.ts > p.ts and q.run_id != p.run_id for q in data["by_issue"][key]):
        return "carried"
    return "vanished"


def _later_round(data: dict[str, Any], p: _Post) -> bool:
    run = data["runs"].get(p.run_id) if p.run_id is not None else None
    start = str(run["started_at"]) if run else p.ts
    return any(t > start and sha != p.sha for t, sha in data["done"][(p.repo, p.number)])


def report(conn: sqlite3.Connection, *, since: str | None, paths: RepoPaths) -> dict[str, Any]:
    data = _load(conn)
    git = _Git(paths)
    # each issue by its first post
    blockers = [
        own[0]
        for own in data["by_issue"].values()
        if own[0].severity == "blocking" and (not since or own[0].ts >= since)
    ]

    # M1 late finds
    m1: Counter[str] = Counter()
    late_examples: list[str] = []
    for p in blockers:
        heads = data["heads"][(p.repo, p.number)]
        idx = next((i for i, h in enumerate(heads) if h[0] == p.sha), None)
        if idx is None:
            m1["unknown_round"] += 1
            continue
        if idx == 0:
            m1["round1"] += 1
            # checkable like a later round would be: the commit is there to diff
            m1["round1_unchecked"] += not git.has(p.repo, p.sha)
            continue
        m1["later_round"] += 1
        prev_sha, _, prev_final = heads[idx - 1]
        how = git.anchor(p.repo, p.sha, prev_sha, p.file, p.line_start, p.line_end)
        if how is None:
            m1["unchecked"] += 1
            continue
        m1[how] += 1
        if how in ("file_unchanged", "anchor_unchanged"):
            m1["late"] += 1
            m1["late_prev_final"] += prev_final
            m1["late_fixed"] += _fate(data, p) == "FIXED"
            if len(late_examples) < 20:
                late_examples.append(f"{p.repo}#{p.number} {p.hash} {p.file}:{p.line_start}")
    later_checked = m1["later_round"] - m1["unchecked"]
    checkable = m1["round1"] - m1["round1_unchecked"] + later_checked

    # M3, M5, M6 over blockers with a later round
    fates: Counter[str] = Counter()
    m6: Counter[str] = Counter()
    drops: list[str] = []
    with_later = [p for p in blockers if _later_round(data, p)]
    for p in with_later:
        fate = _fate(data, p)
        fates[fate] += 1
        if fate != "vanished":
            continue
        heads = data["heads"][(p.repo, p.number)]
        nxt = next((h for h in heads if h[1] > p.ts and h[0] != p.sha), None)
        if nxt is None:
            m6["no_next_review"] += 1
            continue
        how = git.anchor(p.repo, p.sha, nxt[0], p.file, p.line_start, p.line_end)
        if how is None:
            m6["unchecked"] += 1
        elif how == "file_unchanged":
            m6["silent_drop"] += 1
            if len(drops) < 20:
                drops.append(f"{p.repo}#{p.number} {p.hash} {p.file}:{p.line_start}")
        else:
            m6["file_changed"] += 1
    n_later = len(with_later)
    pushback = sum(n for f, n in fates.items() if f in _PUSHBACK)
    m6_checked = m6["silent_drop"] + m6["file_changed"]

    # M4 re-raise of settled issues
    m4: Counter[str] = Counter()
    reraises: list[str] = []
    for key, marks in data["marks"].items():
        settle = next((m for m in marks if m[1] in _PUSHBACK), None)
        own = data["by_issue"].get(key)
        if settle is None or not own or (since and settle[0] < since):
            continue
        m4["settled"] += 1
        anchor = [q for q in own if q.ts <= settle[0]] or own
        a = anchor[-1]
        # match kind -> what git says about the settled issue's own anchor by the time of each
        # match (a title match counts in any file); the worst case counts: one re-raise on
        # unchanged code is one
        states: dict[str, set[str]] = {"title": set(), "location": set()}
        for q in data["by_pr"][key[:2]]:
            if q.ts <= settle[0]:
                continue
            by_title = q.hash == a.hash or (bool(a.title) and _norm(q.title) == _norm(a.title))
            near = (
                q.file == a.file
                and q.line_start is not None
                and a.line_start is not None
                and abs(q.line_start - a.line_start) <= NEAR_LINES
            )
            if not (by_title or near):
                continue
            changed = git.changed(q.repo, a.sha, q.sha, a.file, a.line_start, a.line_end)
            state = "unchecked" if changed is None else "changed" if changed else "unchanged"
            if near:
                states["location"].add(state)
            if by_title:
                states["title"].add(state)
        for kind, found in states.items():
            if found:
                worst = next(x for x in ("unchanged", "unchecked", "changed") if x in found)
                m4[f"by_{kind}_{worst}"] += 1
        if "unchanged" in states["title"] | states["location"] and len(reraises) < 20:
            reraises.append(f"{key[0]}#{key[1]} {key[2]} {a.file}:{a.line_start}")

    # M9 contract failures
    m9: Counter[str] = Counter()
    for r in conn.execute(
        "SELECT status, fail_kind, reason FROM runs WHERE (?1 IS NULL OR started_at >= ?1)",
        (since,),
    ):
        m9["runs"] += 1
        m9["failed"] += r["status"] == "failed"
        if r["fail_kind"] != "contract":
            continue
        m9["contract"] += 1
        reason = str(r["reason"] or "")
        if _RECONCILIATION_RE.search(reason):
            m9["reconciliation"] += 1
        elif _ECHO_RE.search(reason):
            m9["echo_field"] += 1
        else:
            m9["other"] += 1

    notes: list[str] = []
    if not data["posts"]:
        notes.append("the ledger is empty: run `reviewsys ledger backfill` first")
    for repo, n in sorted(git.missing.items()):
        notes.append(f"no mirror for {repo}: {n} git checks skipped")
    settled = m4["settled"]
    return {
        "since": since,
        "posted_blockers": len(blockers),
        "m1_late_find": {
            "late": m1["late"],
            "checkable": checkable,
            # no later-round blocker checked: a rate over round 1 alone would read as 0%
            "rate_pct": _rate(m1["late"], checkable) if later_checked else None,
            "later_round_checked": later_checked,
            "later_round_rate_pct": _rate(m1["late"], later_checked),
            "round1": m1["round1"],
            "round1_unchecked": m1["round1_unchecked"],
            "later_round": m1["later_round"],
            "unchecked": m1["unchecked"],
            "unknown_round": m1["unknown_round"],
            "file_unchanged": m1["file_unchanged"],
            "anchor_unchanged": m1["anchor_unchanged"],
            "anchor_changed": m1["anchor_changed"],
            "new_file": m1["new_file"],
            "late_prev_final": m1["late_prev_final"],
            "late_fixed": m1["late_fixed"],
            "examples": late_examples,
        },
        "m3_pushback": {
            "pushed_back": pushback,
            "with_later_round": n_later,
            "rate_pct": _rate(pushback, n_later),
        },
        "m4_reraise": {
            "settled": settled,
            "by_title": m4["by_title_unchanged"],
            "by_location": m4["by_location_unchanged"],
            "revived_after_change": m4["by_title_changed"] + m4["by_location_changed"],
            "unchecked": m4["by_title_unchecked"] + m4["by_location_unchecked"],
            "examples": reraises,
        },
        "m5_fate_coverage": {
            "recorded": n_later - fates["vanished"],
            "with_later_round": n_later,
            "rate_pct": _rate(n_later - fates["vanished"], n_later),
            "fates": dict(fates.most_common()),
        },
        "m6_silent_drop": {
            "silent_drops": m6["silent_drop"],
            "vanished_checked": m6_checked,
            "rate_pct": _rate(m6["silent_drop"], m6_checked),
            "unchecked": m6["unchecked"],
            "no_next_review": m6["no_next_review"],
            "examples": drops,
        },
        "m9_contract": {
            "contract": m9["contract"],
            "failed_runs": m9["failed"],
            "runs": m9["runs"],
            "reconciliation": m9["reconciliation"],
            "echo_field": m9["echo_field"],
            "other": m9["other"],
        },
        "notes": notes,
    }


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v}%"


def render(rep: dict[str, Any]) -> str:
    m1, m3, m4 = rep["m1_late_find"], rep["m3_pushback"], rep["m4_reraise"]
    m5, m6, m9 = rep["m5_fate_coverage"], rep["m6_silent_drop"], rep["m9_contract"]
    fates = ", ".join(f"{k} {v}" for k, v in m5["fates"].items()) or "-"
    lines = [
        f"review quality since {rep['since'] or 'the start'}: {rep['posted_blockers']} posted blockers",
        f"M1 late-find rate: {m1['late']}/{m1['checkable']} checkable blockers "
        f"({_pct(m1['rate_pct'])}); {m1['late']}/{m1['later_round_checked']} of later-round "
        f"({_pct(m1['later_round_rate_pct'])})",
        f"   round 1 {m1['round1']} (unchecked {m1['round1_unchecked']}), later "
        f"{m1['later_round']} (unchecked {m1['unchecked']}): "
        f"file unchanged {m1['file_unchanged']}, anchor unchanged {m1['anchor_unchanged']}, "
        f"anchor changed {m1['anchor_changed']}, new file {m1['new_file']}; late after a final "
        f"review {m1['late_prev_final']}, late later FIXED {m1['late_fixed']}",
        f"M3 pushback: {m3['pushed_back']}/{m3['with_later_round']} blockers with a later round "
        f"({_pct(m3['rate_pct'])})",
        f"M4 re-raise: by title {m4['by_title']}/{m4['settled']} settled, by location "
        f"(±{NEAR_LINES}) {m4['by_location']}/{m4['settled']}; revived after a change "
        f"{m4['revived_after_change']}, unchecked {m4['unchecked']}",
        f"M5 fate coverage: {m5['recorded']}/{m5['with_later_round']} ({_pct(m5['rate_pct'])}): "
        f"{fates}",
        f"M6 silent drop: {m6['silent_drops']}/{m6['vanished_checked']} checkable vanished "
        f"({_pct(m6['rate_pct'])}); unchecked {m6['unchecked']}, no next review "
        f"{m6['no_next_review']}",
        f"M9 contract failures: {m9['contract']} of {m9['failed_runs']} failed runs "
        f"(reconciliation {m9['reconciliation']}, echo field {m9['echo_field']}, other "
        f"{m9['other']}); {m9['runs']} runs",
    ]
    lines += [f"note: {n}" for n in rep["notes"]]
    return "\n".join(lines) + "\n"
