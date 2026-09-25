"""Review pipeline v10: unanchored finders, triage against the issue ledger, one verifier per
new candidate group, one thread lane for every maintainer conversation, and a composer.

Every lane answers through a JSON Schema (`claude --json-schema`), so this module owns the
schemas, the parsers that turn a lane's answer into typed values, and the small pure rules the
harness applies between lanes (grouping, ranking, the severity gate, the withdrawn-issue
revival check). The orchestration that calls lanes lives in `pipeline_v10.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contract import CATEGORIES, SEVERITIES, SEVERITY_ALIASES, Finding
from .converse import ThreadOutcome
from .models import FailKind, ReviewError

MAX_CANDIDATES_PER_LANE = 8
VERDICTS = ("CONFIRMED", "PLAUSIBLE", "REFUTED")
THREAD_STATUSES = (
    "STILL_VALID",
    "FIXED",
    "OUTDATED",
    "WITHDRAWN",
    "INTENTIONALLY_DEFERRED",
    "NO_REPLY",
)
CLOSING_STATUSES = {"FIXED", "OUTDATED", "WITHDRAWN", "INTENTIONALLY_DEFERRED"}


# ---- schemas ----

_STR = {"type": "string"}
_LINE = {"type": ["integer", "null"]}

FINDER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": _STR,
        "candidates": {
            "type": "array",
            "maxItems": MAX_CANDIDATES_PER_LANE,
            "items": {
                "type": "object",
                "properties": {
                    "file": _STR,
                    "line_start": _LINE,
                    "line_end": _LINE,
                    "severity": {"type": "string", "enum": list(SEVERITIES)},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "title": _STR,
                    "failure_scenario": _STR,
                    "body": _STR,
                    "suggestion": {"type": ["string", "null"]},
                },
                "required": ["file", "severity", "category", "title", "failure_scenario", "body"],
            },
        },
    },
    "required": ["summary", "candidates"],
}

TRIAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "members": {"type": "array", "items": _STR, "minItems": 1},
                    "representative": _STR,
                    "match": _STR,
                    "reason": _STR,
                },
                "required": ["members", "representative", "match"],
            },
        }
    },
    "required": ["groups"],
}

VERIFIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "evidence": _STR,
        "confirm_by": _STR,
        "severity": {"type": "string", "enum": list(SEVERITIES)},
    },
    "required": ["verdict", "evidence", "severity"],
}

THREAD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "finding_hash": _STR,
                    "status": {"type": "string", "enum": list(THREAD_STATUSES)},
                    "reply": _STR,
                    "reasoning": _STR,
                },
                "required": ["finding_hash", "status", "reply"],
            },
        }
    },
    "required": ["issues"],
}

COMPOSER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": _STR,
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": _STR,
                    "title": _STR,
                    "body": _STR,
                    "suggestion": {"type": ["string", "null"]},
                },
                "required": ["id", "title", "body"],
            },
        },
        "coderabbit_extend": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"comment_id": {"type": "integer"}, "reply": _STR},
                "required": ["comment_id", "reply"],
            },
        },
    },
    "required": ["summary", "findings"],
}


# ---- typed values ----


@dataclass(slots=True)
class Candidate:
    """One finder claim, before verification. `id` is unique within a run phase."""

    id: str
    lane: str  # "phase:role", e.g. "phase2:scan"
    file: str
    line_start: int | None
    line_end: int | None
    severity: str
    category: str
    title: str
    failure_scenario: str
    body: str
    suggestion: str | None = None

    def to_finding(self, source: str = "") -> Finding:
        return Finding(
            file=self.file,
            title=self.title,
            body=self.body,
            severity=self.severity,
            category=self.category,
            line_start=self.line_start,
            line_end=self.line_end,
            suggestion=self.suggestion,
            source=source or self.lane,
            extra={"failure_scenario": self.failure_scenario},
        )

    def brief(self) -> dict[str, Any]:
        """What triage and the composer see: the claim, not the full body."""
        return {
            "id": self.id,
            "lane": self.lane,
            "file": self.file,
            "lines": [self.line_start, self.line_end],
            "severity": self.severity,
            "category": self.category,
            "title": self.title,
            "failure_scenario": self.failure_scenario,
        }


@dataclass(slots=True)
class Group:
    """Candidates triage judged to describe one defect, and how it relates to the ledger."""

    members: list[Candidate]
    representative: Candidate
    match: str  # "new" | "open:<hash>" | "closed:<hash>" | "coderabbit:<comment_id>"

    @property
    def kind(self) -> str:
        return self.match.split(":", 1)[0]

    @property
    def target(self) -> str:
        return self.match.split(":", 1)[1] if ":" in self.match else ""

    @property
    def lanes(self) -> list[str]:
        return sorted({m.lane for m in self.members})


@dataclass(slots=True)
class Verdict:
    verdict: str
    evidence: str
    severity: str
    confirm_by: str = ""


@dataclass(slots=True)
class Verified:
    """A group that survived verification, ready for the composer and publication."""

    group: Group
    verdict: Verdict
    severity: str  # after the severity gate
    finding: Finding = field(init=False)

    def __post_init__(self) -> None:
        rep = self.group.representative
        self.finding = rep.to_finding(source=", ".join(self.group.lanes))
        self.finding.severity = self.severity
        self.finding.extra.update(
            {
                "verdict": self.verdict.verdict,
                "evidence": self.verdict.evidence,
                "lanes": self.group.lanes,
            }
        )
        if self.verdict.confirm_by:
            self.finding.extra["confirm_by"] = self.verdict.confirm_by

    @property
    def refuted(self) -> bool:
        return self.verdict.verdict == "REFUTED"


# ---- parsers ----


def _str(v: Any) -> str:
    return str(v).strip() if isinstance(v, str | int | float) and not isinstance(v, bool) else ""


def _line(v: Any) -> int | None:
    return int(v) if isinstance(v, int | float) and not isinstance(v, bool) and v > 0 else None


def _severity(v: Any) -> str:
    s = _str(v).lower()
    s = SEVERITY_ALIASES.get(s, s)
    if s not in SEVERITIES:
        raise ReviewError(FailKind.CONTRACT, f"invalid severity {v!r}")
    return s


def parse_candidates(
    raw: dict[str, Any], *, lane: str, start: int = 1, prefix: str = ""
) -> list[Candidate]:
    """A finder's candidates, numbered `c{prefix}-{start}`, `c{prefix}-{start + 1}`, ... so
    ids stay unique across the lanes of one phase."""
    rows = raw.get("candidates")
    if not isinstance(rows, list):
        raise ReviewError(FailKind.CONTRACT, f"{lane}: candidates must be a list")
    out: list[Candidate] = []
    for i, r in enumerate(rows[:MAX_CANDIDATES_PER_LANE]):
        if not isinstance(r, dict):
            raise ReviewError(FailKind.CONTRACT, f"{lane}: candidate {i} is not an object")
        title = _str(r.get("title"))
        if not title:
            raise ReviewError(FailKind.CONTRACT, f"{lane}: candidate {i} has no title")
        cat = _str(r.get("category")).lower() or "general"
        ls = _line(r.get("line_start"))
        out.append(
            Candidate(
                id=f"c{prefix}-{start + i}",
                lane=lane,
                file=_str(r.get("file")),
                line_start=ls,
                line_end=_line(r.get("line_end")) or ls,
                severity=_severity(r.get("severity")),
                category=cat if cat in CATEGORIES else "general",
                title=title,
                failure_scenario=_str(r.get("failure_scenario")),
                body=_str(r.get("body")),
                suggestion=_str(r.get("suggestion")) or None,
            )
        )
    return out


def parse_groups(
    raw: dict[str, Any],
    candidates: list[Candidate],
    *,
    open_hashes: set[str],
    closed_hashes: set[str],
    coderabbit_ids: set[str],
) -> list[Group]:
    """Triage's grouping. Every candidate lands in exactly one group: ids the model left out
    become singleton `new` groups (never silently dropped), unknown ids are ignored, and a
    match naming something that does not exist falls back to `new`."""
    by_id = {c.id: c for c in candidates}
    rows = raw.get("groups")
    if not isinstance(rows, list):
        raise ReviewError(FailKind.CONTRACT, "triage: groups must be a list")
    placed: set[str] = set()
    groups: list[Group] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        members = [
            by_id[m]
            for m in (r.get("members") or [])
            if isinstance(m, str) and m in by_id and m not in placed
        ]
        if not members:
            continue
        placed.update(m.id for m in members)
        rep_id = _str(r.get("representative"))
        rep = next((m for m in members if m.id == rep_id), None) or _most_concrete(members)
        match = _normalize_match(
            _str(r.get("match")),
            open_hashes=open_hashes,
            closed_hashes=closed_hashes,
            coderabbit_ids=coderabbit_ids,
        )
        groups.append(Group(members, rep, match))
    for c in candidates:
        if c.id not in placed:
            groups.append(Group([c], c, "new"))
    return groups


def _normalize_match(
    m: str, *, open_hashes: set[str], closed_hashes: set[str], coderabbit_ids: set[str]
) -> str:
    kind, _, target = m.partition(":")
    kind = kind.strip().lower()
    target = target.strip()
    if kind in {"open", "same_as_open"} and target in open_hashes:
        return f"open:{target}"
    if kind in {"closed", "same_as_closed"} and target in closed_hashes:
        return f"closed:{target}"
    if kind in {"coderabbit", "restates_coderabbit"} and target in coderabbit_ids:
        return f"coderabbit:{target}"
    return "new"


def _most_concrete(members: list[Candidate]) -> Candidate:
    return max(members, key=lambda c: (_SEV_RANK[c.severity], len(c.failure_scenario)))


def parse_verdict(raw: dict[str, Any]) -> Verdict:
    v = _str(raw.get("verdict")).upper()
    if v not in VERDICTS:
        raise ReviewError(FailKind.CONTRACT, f"verifier: invalid verdict {raw.get('verdict')!r}")
    evidence = _str(raw.get("evidence"))
    if not evidence:
        raise ReviewError(FailKind.CONTRACT, "verifier: a verdict needs evidence")
    return Verdict(
        verdict=v,
        evidence=evidence,
        severity=_severity(raw.get("severity")),
        confirm_by=_str(raw.get("confirm_by")),
    )


def parse_thread_decisions(raw: dict[str, Any], *, expected: set[str]) -> dict[str, ThreadOutcome]:
    """One outcome per expected issue. A closing status needs a reply: it is what the thread
    shows when the finding is retired, and an outcome that never reaches the thread must not
    move the ledger."""
    rows = raw.get("issues")
    if not isinstance(rows, list):
        raise ReviewError(FailKind.CONTRACT, "thread lane: issues must be a list")
    out: dict[str, ThreadOutcome] = {}
    for r in rows:
        if not isinstance(r, dict):
            raise ReviewError(FailKind.CONTRACT, "thread lane: row is not an object")
        h = _str(r.get("finding_hash"))
        st = _str(r.get("status")).upper()
        reply = _str(r.get("reply"))
        if h not in expected:
            raise ReviewError(FailKind.CONTRACT, f"thread lane: unexpected issue {h!r}")
        if h in out:
            raise ReviewError(FailKind.CONTRACT, f"thread lane: duplicate row for {h!r}")
        if st not in THREAD_STATUSES:
            raise ReviewError(FailKind.CONTRACT, f"thread lane: {h}: bad status {st!r}")
        if st == "NO_REPLY":
            reply = ""
        elif st in CLOSING_STATUSES and not reply:
            raise ReviewError(FailKind.CONTRACT, f"thread lane: {h}: {st} without a reply")
        out[h] = ThreadOutcome(h, st, reply, _str(r.get("reasoning"))[:1000])
    missing = expected - set(out)
    if missing:
        raise ReviewError(FailKind.CONTRACT, f"thread lane: missing issues {sorted(missing)}")
    return out


# ---- harness rules ----

_SEV_RANK = {"blocking": 3, "suggestion": 2, "nitpick": 1}
_VERDICT_RANK = {"CONFIRMED": 2, "PLAUSIBLE": 1, "REFUTED": 0}


def rank_groups(groups: list[Group]) -> list[Group]:
    """Most severe first, then the ones more lanes independently found."""
    return sorted(
        groups,
        key=lambda g: (-max(_SEV_RANK[m.severity] for m in g.members), -len(g.lanes)),
    )


def gated_severity(verdict: Verdict) -> str:
    """Blocking requires CONFIRMED: a PLAUSIBLE blocker is published as a suggestion that says
    what would confirm it (decision D1, 2026-09-22)."""
    if verdict.severity == "blocking" and verdict.verdict != "CONFIRMED":
        return "suggestion"
    return verdict.severity


def rank_verified(items: list[Verified]) -> list[Verified]:
    return sorted(
        items,
        key=lambda v: (
            -_SEV_RANK[v.severity],
            -_VERDICT_RANK[v.verdict.verdict],
            -len(v.group.lanes),
        ),
    )


def lines_changed(diff_text: str, line_start: int | None, line_end: int | None) -> bool:
    """Whether a unified diff (old side = the code an issue was judged against) touches the
    issue's anchored lines. No anchor: any change to the file counts."""
    if not diff_text.strip():
        return False
    if not line_start:
        return True
    lo, hi = line_start, line_end or line_start
    for line in diff_text.splitlines():
        if not line.startswith("@@"):
            continue
        # @@ -a,b +c,d @@
        try:
            old = line.split()[1]
            start, _, count = old[1:].partition(",")
            s, n = int(start), int(count) if count else 1
        except (IndexError, ValueError):
            return True
        if n == 0:
            if lo <= s + 1 and s <= hi:
                return True
        elif s <= hi and s + n - 1 >= lo:
            return True
    return False
