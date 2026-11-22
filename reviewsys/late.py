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
raised on the PR, and everything on A code, take the normal path.

A finding an earlier full review judged FIXED or OUTDATED (`STAGE_FIXED`) and that this round
raises again on lines unchanged since that judgement (`split`'s `revived`) is a new blocker
for the gate: it stands only when the confirmation lane confirms it, and otherwise stays
settled. A non-blocking one stays settled without asking."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contract import Finding
from .ledger import _git, _hunks, _map_line, diff_touches, git_diff_u0, git_has, lines_changed
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
STAGE_SETTLED = "late-settled"  # judged fixed earlier, raised again on unchanged lines
# a follow-up candidate a published review listed: what later rounds never restate
STAGE_LISTED = "followup"
# a confirmation verdict that is remembered for the finding: (stage, verdict)
REMEMBERED = {STAGE_UNCONFIRMED: "non_blocking", STAGE_INVALID: "invalid"}
# a prior finding a full review judged fixed, its body the status and the head it was judged
# at (two lines); later rounds read it until a review carries or posts the finding again
STAGE_FIXED = "judged-fixed"
JUDGED_FIXED = frozenset({"FIXED", "OUTDATED"})
# lines around a listed follow-up candidate that still count as its place
LISTED_MARGIN = 3

Spot = tuple[str, int, int]  # (file, line_start, line_end)
# what the gate sets on a finding, stripped from a verifier's output so only the gate does
ROUTING_KEYS = ("posted_severity", "raised_as")


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
        """By the finding's own lines: a file an author keeps editing is changed in every
        round, which let half the late blockers on untouched lines pass as findings on new
        code. Lines git cannot place count as changed."""
        if not f.file or not f.line_start or not git_has(self.repo, self.head, f.file):
            return UNKNOWN
        since_prev = self.changed_since(self.prev, f.file)
        if since_prev is None:
            return UNKNOWN
        if since_prev and self.moved(self.prev, f):
            return CHANGED
        if not self.first or self.first == self.prev:
            return SINCE_FIRST
        if self.changed_since(self.first, f.file) is False or not self.moved(self.first, f):
            return SINCE_FIRST
        return IN_REVIEW

    def moved(self, since: str, f: Finding) -> bool:
        """The lines under `f` (as the head reads them) changed since `since`, or git cannot
        tell."""
        changed = lines_changed(self.repo, self.head, since, f.file, f.line_start, f.line_end)
        return changed is not False

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
    candidates: list[Finding] = field(default_factory=list)  # new blockers: to confirm
    settled: list[Finding] = field(default_factory=list)  # revived below blocking: dropped

    @property
    def changed(self) -> bool:
        return bool(self.nits or self.followups or self.repeats or self.candidates or self.settled)


def hold_severity(findings: Iterable[Finding], posted: Mapping[str, str]) -> list[Finding]:
    """A carried finding on code unchanged since the previous final review is not raised to
    blocking: nothing under it changed, its thread still states the severity it was posted
    with (`posted`, by prior hash), and no confirmation lane looks at a carried finding. On
    dashpay/dash#8158 a suggestion open for five rounds came back as a blocker from a pass
    that was not shown its thread. Returns the findings held at their posted severity, each
    marked `extra["raised_as"]` (it runs twice in a round: the mark stays)."""
    held = []
    for f in findings:
        was = posted.get(f.prior_hash or "")
        if was and was != "blocking" and f.severity == "blocking" and on_unchanged(f):
            f.severity = was
            f.extra["raised_as"] = "blocking"
            held.append(f)
    return held


def is_listed(f: Finding, listed: set[str], spots: Iterable[Spot] = ()) -> bool:
    """`f` was listed as a follow-up candidate: `listed` holds hashes and dedupe keys, `spots`
    where the listed ones are at this head (`listed_spots`). One on an overlapping or adjoining
    range of the same file is that candidate again, whatever its title or severity."""
    if f.hash in listed or f.dedupe_key in listed:
        return True
    if not f.file or not f.line_start:
        return False
    start, end = f.line_start, f.line_end or f.line_start
    return any(
        path == f.file and start <= e + LISTED_MARGIN and end >= s - LISTED_MARGIN
        for path, s, e in spots
    )


def listed_spots(
    repo: Path, head: str, listed: Iterable[tuple[str, int | None, int | None, str]]
) -> list[Spot]:
    """Where each listed follow-up candidate (file, lines, the head it was listed at) is at
    `head`: one whose lines have not changed since, re-located by the diff between the two
    heads (`ledger.lines_changed` and `ledger.map_lines` on one diff per head and file). One
    git cannot place, or whose code changed, has no place."""
    out: list[Spot] = []
    diffs: dict[tuple[str, str], str | None] = {}
    for path, start, end, sha in listed:
        if not path or not start:
            continue
        if (sha, path) not in diffs:
            diffs[sha, path] = git_diff_u0(repo, sha, head, path) if sha != head else ""
        diff = diffs[sha, path]
        if diff is None or "+++ /dev/null" in diff.splitlines():
            continue
        if not diff.strip():
            if sha == head or git_has(repo, head, path):
                out.append((path, start, end or start))
            continue
        hunks = _hunks(diff)
        if diff_touches(diff, start, end) or hunks is None:
            continue
        out.append(
            (path, _map_line(hunks, start, end=False), _map_line(hunks, end or start, end=True))
        )
    return out


def revived_sentence(judged: Mapping[str, Any], failure: str) -> str:
    """What a confirmed revival of a finding judged fixed says on its thread."""
    return (
        f"Judged fixed at `{judged['sha'][:8]}`; raised again and confirmed by an independent "
        f"pass: {failure}"
    )


def on_unchanged(f: Finding) -> bool:
    """`f` is on code unchanged since the previous final review."""
    return f.extra.get("code_status") in UNCHANGED


def split(
    findings: Iterable[Finding],
    *,
    posted: Callable[[Finding], str | None],
    listed: set[str],
    spots: Iterable[Spot] = (),
    revived: Callable[[Finding], dict[str, Any] | None] = lambda f: None,
) -> Split:
    """`findings` (each with `extra["code_status"]` set) by what happens to them. `posted`:
    the severity an earlier round posted the finding with, None when it is new to the PR.

    A later round's blocker that no earlier round posted as one is confirmed before it
    blocks, wherever it is: a new finding, or one carried or restated from below blocking
    (`extra["posted_severity"]`), which would otherwise undo an earlier confirmation's
    verdict a round later. The other new findings leave the normal path only on code
    unchanged since the previous final review.

    `revived`: the judgement (`hash`, `status`, `sha`) of a finding an earlier review judged
    fixed that `f` raises again on lines unchanged since, else None. Such a blocker is
    confirmed like a new one (`extra["judged_fixed"]`), and so is one held at the severity it
    was posted with (`hold_severity`): unconfirmed, it is that carried finding again
    (`extra["posted_severity"]`). One the round raises below blocking stays settled."""
    out = Split()
    spots = list(spots)
    for f in findings:
        f.extra.pop("revived_note", None)
        judged = revived(f)
        if judged:
            f.extra["judged_fixed"] = judged
            f.extra.pop("posted_severity", None)
            if f.extra.get("raised_as") == "blocking":
                f.extra["posted_severity"], f.severity = f.severity, "blocking"
            (out.candidates if f.severity == "blocking" else out.settled).append(f)
            continue
        f.extra.pop("judged_fixed", None)
        was = posted(f)
        if f.severity == "blocking" and was != "blocking":
            if was:
                f.extra["posted_severity"] = was
            out.candidates.append(f)
        elif was is not None or not on_unchanged(f):
            out.keep.append(f)
        elif f.severity == "nitpick":
            out.nits.append(f)
        elif is_listed(f, listed, spots):
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
            **(
                {
                    "judged_fixed_at": judged["sha"],
                    "lines_unchanged_since": True,
                    **({"replies_since": judged["replies"]} if judged.get("replies") else {}),
                }
                if (judged := f.extra.get("judged_fixed"))
                else {}
            ),
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
