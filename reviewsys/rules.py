"""Policy v12, pure: the finder candidate contract and the settled list the finders are told
about (v12 design §2.1). No I/O.

Identity is the harness's, never a model's: finder candidates get ids at parse time
(`p2-general-3`)."""

from __future__ import annotations

from typing import Any

from .contract import (
    CATEGORIES,
    SEVERITIES,
    SEVERITY_ALIASES,
    Finding,
    ReviewerOutput,
)
from .ledger import Issue
from .models import FailKind, ReviewError

SETTLED_LIST_MAX = 10  # withdrawn / deferred issues the finders are told about
FAILURE_SCENARIO = "failure_scenario"


def _str(v: Any) -> str:
    return str(v).strip() if isinstance(v, str | int | float) and not isinstance(v, bool) else ""


def _line(v: Any) -> int | None:
    return int(v) if isinstance(v, int | float) and not isinstance(v, bool) and v > 0 else None


def _severity(v: Any) -> str | None:
    s = _str(v).lower()
    s = SEVERITY_ALIASES.get(s, s)
    return s if s in SEVERITIES else None


# ---- finders: the candidate contract (§2.1) ----


def candidate_prefix(phase: str, key: str, *, fresh: bool = False) -> str:
    """The id prefix of a finder lane's candidates: `p1-general`, `p2-security-auditor`, `cf-general`
    for the confirmation pass. `key` is the lane's output key (a comparison twin's blinded
    `<role>/a|b`)."""
    head = "cf" if fresh else ("p1" if phase == "phase1" else "p2")
    return f"{head}-{key}"


def parse_finder_output(
    raw: dict[str, Any],
    *,
    id_prefix: str,
    source: str,
    expected_phase: str,
    head_sha: str,
) -> ReviewerOutput:
    """A finder lane's answer: `summary` plus `candidates`, each numbered `<id_prefix>-<n>`. A
    lane that answered under the old key (`findings`) is read the same way, noted. Every
    problem is reported at once, so one correction turn can fix them together. The output's
    `raw` is the shape a v11 verifier reads (`findings`), so a v11 verifier can weigh
    unanchored finders too."""
    notes: list[str] = []
    rows = raw.get("candidates")
    if rows is None and isinstance(raw.get("findings"), list):
        rows = raw["findings"]
        notes.append("candidates read from `findings`")
    if not isinstance(rows, list):
        raise ReviewError(FailKind.CONTRACT, "finder output: candidates must be a list")
    problems: list[str] = []
    findings: list[Finding] = []
    for i, r in enumerate(rows, 1):
        if not isinstance(r, dict):
            problems.append(f"candidate {i} is not an object")
            continue
        title = _str(r.get("title"))
        sev = _severity(r.get("severity"))
        if not title:
            problems.append(f"candidate {i} has no title")
        if sev is None:
            problems.append(f"candidate {i} has invalid severity {r.get('severity')!r}")
        if not title or sev is None:
            continue
        cat = _str(r.get("category")).lower()
        ls = _line(r.get("line_start"))
        findings.append(
            Finding(
                file=_str(r.get("file")),
                title=title,
                body=_str(r.get("body")),
                severity=sev,
                category=cat if cat in CATEGORIES else "general",
                line_start=ls,
                line_end=_line(r.get("line_end")) or ls,
                suggestion=_str(r.get("suggestion")) or None,
                source=source,
                extra={FAILURE_SCENARIO: _str(r.get(FAILURE_SCENARIO))},
                candidate_id=f"{id_prefix}-{i}",
            )
        )
    if problems:
        raise ReviewError(FailKind.CONTRACT, "finder output: " + "; ".join(problems))
    summary = _str(raw.get("summary"))
    return ReviewerOutput(
        summary=summary,
        findings=findings,
        out_of_scope=[],
        review_phase=expected_phase,
        head_sha=head_sha,
        prior_reconciliation=[],
        raw={
            "summary": summary,
            "findings": [f.to_dict() for f in findings],
            "out_of_scope_findings": [],
        },
        normalized=notes,
    )


def _lines(i: Issue) -> list[int | None]:
    return [i.line_start, i.line_end]


def settled_list(issues: dict[str, Issue], reasons: dict[str, str]) -> list[dict[str, Any]]:
    """The finders' settled list (§2.1): withdrawn or deferred issues, newest first, at most
    SETTLED_LIST_MAX, each with its location and the latest human word on it."""
    rows = sorted(
        (i for i in issues.values() if i.status in {"withdrawn", "deferred"}),
        key=lambda i: i.updated_at,
        reverse=True,
    )
    return [
        {
            "file": i.file,
            "lines": _lines(i),
            "title": i.title,
            "status": i.status,
            "reason": (reasons.get(i.hash) or i.closed_reason or "")[:300],
        }
        for i in rows[:SETTLED_LIST_MAX]
    ]
