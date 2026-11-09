"""Later review rounds under policy v11: where each verified finding sits relative to the code
the PR already had reviewed, and what that means for a new finding there.

Every push re-reviews the whole PR, so code nobody touched since the previous final review
gets a fresh chance to draw a false finding every round. A finding's code status is decided
mechanically, from git in the run's worktree, by its file:

- `A` (CHANGED): the file changed since the previous full-strength final-reviewed head;
- `B` (IN_REVIEW): the file is unchanged since that head, but changed since the PR's first
  review;
- `C` (SINCE_FIRST): the file is unchanged since the PR's first review;
- `U` (UNKNOWN): no anchor, a file the head does not have, or git cannot tell. Treated as `A`.

A file renamed without a content change is unchanged. New findings on B/C code (`split`):
nitpicks are dropped, suggestions are listed once in the review body as follow-up candidates
(never an inline thread, never restated), and blockers stand only when an independent
confirmation lane (`parse_decisions`) finds a concrete reachable failure. Findings already
raised on the PR, and everything on A code, take the normal path."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contract import Finding
from .ledger import _git, git_has
from .models import FailKind, ReviewError

CHANGED, IN_REVIEW, SINCE_FIRST, UNKNOWN = "A", "B", "C", "U"
UNCHANGED = frozenset({IN_REVIEW, SINCE_FIRST})

VERDICTS = ("blocking", "non_blocking", "invalid")

# what became of a gated finding: the `findings.stage` its row is kept under
STAGE_NIT = "late-nit"
STAGE_FOLLOWUP = "late-followup"
STAGE_REPEAT = "late-repeat"
STAGE_UNCONFIRMED = "late-unconfirmed"
STAGE_INVALID = "late-invalid"
# a follow-up candidate a published review listed: what later rounds never restate
STAGE_LISTED = "followup"
# a confirmation verdict that is remembered for the finding: (stage, verdict)
REMEMBERED = {STAGE_UNCONFIRMED: "non_blocking", STAGE_INVALID: "invalid"}


def changed_paths(name_status: str) -> set[str]:
    """The head-side paths a `git diff --name-status -M -z` changed: added, modified, retyped
    or renamed with edits. A rename with no content change (`R100`) is not a change."""
    tokens = name_status.split("\0")
    out: set[str] = set()
    i = 0
    while i < len(tokens) and tokens[i]:
        kind = tokens[i]
        if kind[:1] in ("R", "C"):
            if i + 2 < len(tokens) and kind != "R100":
                out.add(tokens[i + 2])
            i += 3
            continue
        if kind[:1] != "D" and i + 1 < len(tokens):
            out.add(tokens[i + 1])
        i += 2
    return out


@dataclass(slots=True)
class CodeHistory:
    """The code statuses of one run: `head` against the previous full-strength final-reviewed
    head `prev` and the PR's first reviewed head `first`, in the worktree `repo`."""

    repo: Path
    head: str
    prev: str
    first: str | None = None
    _changed: dict[str, set[str] | None] = field(default_factory=dict)

    def status(self, f: Finding) -> str:
        if not f.file or not f.line_start or not git_has(self.repo, self.head, f.file):
            return UNKNOWN
        since_prev = self.changed_since(self.prev, f.file)
        if since_prev is None:
            return UNKNOWN
        if since_prev:
            return CHANGED
        if not self.first or self.first == self.prev:
            return SINCE_FIRST
        return SINCE_FIRST if self.changed_since(self.first, f.file) is False else IN_REVIEW

    def changed_since(self, since: str, path: str) -> bool | None:
        """Whether the head's `path` changed since `since`; None when git cannot tell."""
        if since == self.head:
            return False
        paths = self.changed(since)
        return None if paths is None else path in paths

    def changed(self, since: str) -> set[str] | None:
        """Every head path that changed since `since` (`changed_paths`), once per commit."""
        if since not in self._changed:
            res = _git(
                self.repo, "diff", "--name-status", "-M", "-z", "--no-color", since, self.head
            )
            ok = res is not None and res.returncode == 0
            self._changed[since] = changed_paths(res.stdout) if res is not None and ok else None
        return self._changed[since]


@dataclass(slots=True)
class Split:
    """A later round's verified findings, by what happens to them."""

    keep: list[Finding] = field(default_factory=list)  # the normal path
    nits: list[Finding] = field(default_factory=list)  # new B/C nitpicks: dropped
    followups: list[Finding] = field(default_factory=list)  # new B/C suggestions, first listing
    repeats: list[Finding] = field(default_factory=list)  # ... already listed by a review
    candidates: list[Finding] = field(default_factory=list)  # new B/C blockers: to confirm

    @property
    def changed(self) -> bool:
        return bool(self.nits or self.followups or self.repeats or self.candidates)


def is_listed(f: Finding, listed: set[str]) -> bool:
    """`f` was listed as a follow-up candidate: `listed` holds hashes and dedupe keys."""
    return f.hash in listed or f.dedupe_key in listed


def split(
    findings: Iterable[Finding], *, is_new: Callable[[Finding], bool], listed: set[str]
) -> Split:
    """`findings` (each with `extra["code_status"]` set) by what happens to them: only new
    findings on code unchanged since the previous final review leave the normal path."""
    out = Split()
    for f in findings:
        if f.extra.get("code_status") not in UNCHANGED or not is_new(f):
            out.keep.append(f)
        elif f.severity == "nitpick":
            out.nits.append(f)
        elif f.severity == "blocking":
            out.candidates.append(f)
        elif is_listed(f, listed):
            out.repeats.append(f)
        else:
            out.followups.append(f)
    return out


@dataclass(frozen=True, slots=True)
class Decision:
    verdict: str  # one of VERDICTS
    failure: str = ""  # the concrete reachable failure, for `blocking`
    reason: str = ""


def candidate_ids(n: int) -> list[str]:
    return [f"c{i}" for i in range(1, n + 1)]


def candidate_briefs(findings: list[Finding]) -> list[dict[str, Any]]:
    """What the confirmation lane is shown of each candidate: the claim, never the verifier's
    reasoning, confidence or sources."""
    return [
        {
            "id": cid,
            "file": f.file,
            "line_start": f.line_start,
            "line_end": f.line_end,
            "category": f.category,
            "title": f.title,
            "body": f.body,
            **({"suggested_fix": f.suggestion} if f.suggestion else {}),
        }
        for cid, f in zip(candidate_ids(len(findings)), findings, strict=True)
    ]


def parse_decisions(raw: dict[str, Any], ids: list[str]) -> dict[str, Decision]:
    """The confirmation lane's `decisions[]`: exactly one per id, a known verdict, and a
    `failure` for every `blocking` one. Breaches are CONTRACT errors (correctable)."""
    rows = raw.get("decisions")
    if not isinstance(rows, list):
        raise ReviewError(FailKind.CONTRACT, "confirm: decisions must be a list")
    out: dict[str, Decision] = {}
    problems: list[str] = []
    named: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            problems.append("a decision is not an object")
            continue
        cid = str(row.get("id") or "").strip()
        verdict = re.sub(r"[\s\-]+", "_", str(row.get("verdict") or "").strip().lower())
        failure = str(row.get("failure") or "").strip()
        if cid not in ids:
            problems.append(f"unknown id {cid!r}")
            continue
        if cid in named:
            problems.append(f"{cid} decided twice")
        elif verdict not in VERDICTS:
            problems.append(f"{cid}: verdict {verdict!r} is not one of {', '.join(VERDICTS)}")
        elif verdict == "blocking" and not failure:
            problems.append(f"{cid}: a blocking verdict needs a concrete `failure`")
        else:
            out[cid] = Decision(verdict, failure, str(row.get("reason") or "").strip())
        named.add(cid)
    missing = [i for i in ids if i not in named]
    if missing:
        problems.append(f"no decision for {', '.join(missing)}")
    if problems:
        raise ReviewError(FailKind.CONTRACT, "; ".join(problems))
    return out
