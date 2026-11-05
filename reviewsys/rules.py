"""Policy v12, pure: the finder candidate contract, the verify+match answer, the code rules that
turn it into issues (v12 design §2.3), the ledger verdict (§3), and what the lanes are told from
the ledger. No I/O: every git question comes in as a callable, so the worker decides how to ask
it and the tests can answer it.

Identity is the harness's, never a model's: finder candidates get ids at parse time
(`p2-general-3`), the verifier groups them by those ids and names ledger issues by their stored
`finding_hash`, and a new issue's hash is computed once from its composed title."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from . import ledger
from .contract import (
    CATEGORIES,
    SEVERITIES,
    SEVERITY_ALIASES,
    Finding,
    ReviewerOutput,
    _expand,
)
from .ledger import Issue
from .models import FailKind, ReviewError
from .schemas import VERDICTS

# the verifier's batch: above this many candidates a phase's verify+match is split into shards
SHARD_SIZE = 12
MAX_SHARDS = 3
# the thread lane's batch: issues, and characters of their inputs (thread, code, diff)
THREAD_SHARD_SIZE = 12
THREAD_SHARD_CHARS = 60_000
SETTLED_LIST_MAX = 10  # withdrawn / deferred issues the finders are told about
# statuses as v12 reads them. A thread a human resolved by hand still counts toward the verdict
# (only an explicit outcome lifts a finding), so a candidate restating it corroborates it
VERDICT_STANDING = ledger.COUNTING
MATCH_SETTLED = frozenset({"withdrawn", "deferred", "outdated"})
MATCH_FIXED = frozenset({"fixed"})
_SEV_RANK = {"blocking": 3, "suggestion": 2, "nitpick": 1}
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


def brief(f: Finding) -> dict[str, Any]:
    """A candidate as the verifier sees it: its claim, without its lane (ids name the role)."""
    out: dict[str, Any] = {
        "id": f.candidate_id,
        "file": f.file,
        "line_start": f.line_start,
        "line_end": f.line_end,
        "severity": f.severity,
        "category": f.category,
        "title": f.title,
        FAILURE_SCENARIO: f.extra.get(FAILURE_SCENARIO) or "",
        "body": f.body,
    }
    if f.suggestion:
        out["suggestion"] = f.suggestion
    gate = f.extra.get("gate_verdict")
    if gate:
        out["gate_verdict"] = gate
    return out


def shards(
    cands: list[Finding], size: int = SHARD_SIZE, most: int = MAX_SHARDS
) -> list[list[Finding]]:
    """The candidates in at most `most` verifier batches, split only above `size`, a file's
    candidates always together (duplicates across lanes sit in the same file, and the batch
    that groups them must see them all). Largest files first onto the lightest batch; each
    batch keeps the candidates' original order."""
    if len(cands) <= size:
        return [list(cands)]
    n = min(most, math.ceil(len(cands) / size))
    by_file: dict[str, list[int]] = {}
    for i, c in enumerate(cands):
        by_file.setdefault(c.file, []).append(i)
    bins: list[list[int]] = [[] for _ in range(n)]
    for idx in sorted(by_file.values(), key=len, reverse=True):
        min(bins, key=len).extend(idx)
    return [[cands[i] for i in sorted(b)] for b in bins if b]


def thread_shards(
    sizes: dict[str, int],
    size: int = THREAD_SHARD_SIZE,
    chars: int = THREAD_SHARD_CHARS,
    most: int = MAX_SHARDS,
) -> list[list[str]]:
    """The thread lane's work set (hash -> characters of its input) in at most `most`
    batches: one unless it holds more than `size` issues or `chars` characters; the biggest
    issues first onto the lightest batch; each batch keeps the work set's order."""
    order = list(sizes)
    total = sum(sizes.values())
    n = min(most, max(math.ceil(len(order) / size), math.ceil(total / chars), 1))
    if n <= 1:
        return [order] if order else []
    bins: list[list[str]] = [[] for _ in range(n)]
    load = [0] * n
    for h in sorted(order, key=lambda h: sizes[h], reverse=True):
        k = min(range(n), key=lambda i: (load[i], len(bins[i])))
        bins[k].append(h)
        load[k] += sizes[h]
    return [sorted(b, key=order.index) for b in bins if b]


# ---- the verifier's answer (§2.2) ----


@dataclass(slots=True)
class Group:
    members: list[str]
    kind: str  # new | open | settled | fixed | coderabbit
    target: str  # the issue hash / CodeRabbit comment id; "" for new
    verdict: str
    evidence: str
    confirm_by: str
    severity: str
    title: str
    body: str
    suggestion: str | None

    @property
    def match(self) -> str:
        return f"{self.kind}:{self.target}" if self.target else self.kind


@dataclass(slots=True)
class VerifyAnswer:
    summary: str
    groups: list[Group]
    raw: dict[str, Any]
    normalized: list[str] = field(default_factory=list)


def issue_kind(issue: Issue) -> str:
    """How a ledger issue is matched: `open` while it counts toward the verdict, `settled` once
    a call closed it that the code must respect, `fixed` once fixed."""
    if issue.status in VERDICT_STANDING:
        return "open"
    if issue.status in MATCH_FIXED:
        return "fixed"
    return "settled"


def _match(
    value: Any,
    *,
    issues: dict[str, Issue],
    extra_open: set[str],
    coderabbit_ids: set[int],
    notes: list[str],
    problems: list[str],
    label: str,
) -> tuple[str, str] | None:
    text = _str(value)
    kind, _, target = text.partition(":")
    kind, target = kind.strip().lower(), target.strip().lower()
    if kind == "new" and not target:
        return "new", ""
    if kind == "coderabbit":
        try:
            cid = int(target)
        except ValueError:
            cid = -1
        if cid in coderabbit_ids:
            return "coderabbit", str(cid)
        problems.append(f"{label}: match {text!r} names no CodeRabbit comment listed")
        return None
    if kind in {"open", "settled", "fixed", "closed"}:
        known = set(issues) | extra_open
        full = _expand(target, known) if target else None
        if full is None:
            problems.append(f"{label}: match {text!r} names no issue listed")
            return None
        if full != target:
            notes.append(f"{label}: hash {target!r} -> {full!r}")
        actual = "open" if full in extra_open else issue_kind(issues[full])
        if actual != kind:
            notes.append(f"{label}: match {kind}:{full} -> {actual}:{full} (its ledger status)")
        return actual, full
    problems.append(
        f"{label}: match {text!r} is not new | open:<hash> | settled:<hash> | fixed:<hash> | coderabbit:<id>"
    )
    return None


def parse_verify(
    raw: dict[str, Any],
    *,
    candidate_ids: list[str],
    issues: dict[str, Issue],
    coderabbit_ids: Iterable[int] = (),
    extra_open: Iterable[str] = (),
) -> VerifyAnswer:
    """The verify+match answer, checked whole: every candidate id in exactly one group, every
    match naming something listed (a hash prefix is expanded; a match kind that contradicts the
    issue's ledger status is corrected to it, noted), a verdict with evidence, a severity and
    a title. `extra_open`: hashes the lane was shown as open besides the ledger's (this round's
    own new issues, for the confirmation pass). Raises CONTRACT naming every problem."""
    if "@coderabbitai review" in json.dumps(raw).lower():
        raise ReviewError(
            FailKind.CONTRACT, "verifier output contains a forbidden CodeRabbit retrigger"
        )
    rows = raw.get("groups")
    if not isinstance(rows, list):
        raise ReviewError(FailKind.CONTRACT, "verifier: groups must be a list")
    known = set(candidate_ids)
    cr = set(coderabbit_ids)
    xo = set(extra_open)
    placed: dict[str, int] = {}
    problems: list[str] = []
    notes: list[str] = []
    groups: list[Group] = []
    for gi, r in enumerate(rows, 1):
        label, before = f"group {gi}", len(problems)
        if not isinstance(r, dict):
            problems.append(f"{label} is not an object")
            continue
        members = r.get("members")
        if not isinstance(members, list) or not members:
            problems.append(f"{label}: members must be a non-empty list of candidate ids")
            continue
        ids: list[str] = []
        for m in members:
            mid = _str(m)
            if mid not in known:
                problems.append(f"{label}: {mid!r} is not a candidate id")
            elif mid in placed:
                problems.append(f"{label}: {mid} is already in group {placed[mid]}")
            else:
                placed[mid] = gi
                ids.append(mid)
        match = _match(
            r.get("match"),
            issues=issues,
            extra_open=xo,
            coderabbit_ids=cr,
            notes=notes,
            problems=problems,
            label=label,
        )
        verdict = _str(r.get("verdict")).upper()
        if verdict not in VERDICTS:
            problems.append(f"{label}: verdict {r.get('verdict')!r} not in {' | '.join(VERDICTS)}")
        evidence = _str(r.get("evidence"))
        if not evidence:
            problems.append(f"{label}: a verdict needs `evidence` (quoted lines)")
        refuted = verdict == "REFUTED"  # dropped: it needs no final wording
        sev = _severity(r.get("severity")) or ("nitpick" if refuted else None)
        if sev is None:
            problems.append(
                f"{label}: severity {r.get('severity')!r} not in {' | '.join(SEVERITIES)}"
            )
        title = _str(r.get("title")) or (f"refuted {', '.join(ids)}" if refuted else "")
        if not title:
            problems.append(f"{label}: no title")
        if len(problems) > before or match is None or sev is None:
            continue  # every problem of the group is listed; it is read again once fixed
        groups.append(
            Group(
                members=ids,
                kind=match[0],
                target=match[1],
                verdict=verdict,
                evidence=evidence,
                confirm_by=_str(r.get("confirm_by")),
                severity=sev,
                title=title,
                body=_str(r.get("body")),
                suggestion=_str(r.get("suggestion")) or None,
            )
        )
    missing = [c for c in candidate_ids if c not in placed]
    if missing:
        problems.append(f"candidates in no group: {missing}")
    if problems:
        raise ReviewError(FailKind.CONTRACT, "; ".join(problems))
    return VerifyAnswer(summary=_str(raw.get("summary")), groups=groups, raw=raw, normalized=notes)


# ---- the code rules (§2.3) ----


@dataclass(slots=True)
class Outcome:
    """What a verify+match answer means once the code rules ran over it."""

    new: list[Finding] = field(default_factory=list)  # to post: new issues, revivals, regressions
    # ledger issue hash -> the candidate ids that restated it (it stays open; no new thread)
    corroborated: dict[str, list[str]] = field(default_factory=dict)
    # ledger issue hash -> the strongest verdict a restatement of it got (CONFIRMED wins)
    refound: dict[str, str] = field(default_factory=dict)
    # an open non-blocking issue the verifier now CONFIRMED as blocking: raised in place
    upgrades: dict[str, Finding] = field(default_factory=dict)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    reactions: list[dict[str, Any]] = field(default_factory=list)  # on CodeRabbit comments
    groups: list[Group] = field(default_factory=list)  # every group, as answered
    # groups that were not REFUTED, each with the finding the code composed from it
    kept: list[tuple[Group, Finding]] = field(default_factory=list)


OUT_OF_SCOPE = "out of scope:"


def _dropped(g: Group, reason: str) -> dict[str, Any]:
    """A dropped group. A REFUTED group whose evidence starts with `Out of scope:` is an
    important adjacent problem the review publishes as a non-blocking follow-up suggestion
    (`follow_up`); every other v12 drop says explicitly that it is not one, so a refutation that
    merely quotes "outside" never reaches that list."""
    d: dict[str, Any] = {"original_title": g.title, "members": g.members, "reason": reason}
    ev = g.evidence.strip()
    d["follow_up"] = g.verdict == "REFUTED" and ev.lower().startswith(OUT_OF_SCOPE)
    if d["follow_up"]:
        body = ev[len(OUT_OF_SCOPE) :].strip()
        refuted_default = g.title.startswith("refuted ")
        d["title"] = (body.split(". ")[0][:120] or "Follow-up") if refuted_default else g.title
        d["body"] = body
    return d


def _representative(members: list[Finding]) -> Finding:
    return max(members, key=lambda c: (_SEV_RANK.get(c.severity, 0), -members.index(c)))


def _plausible_note(confirm_by: str) -> str:
    what = confirm_by or "the verifier did not say; treat it as unconfirmed"
    return f"\n\n**Not confirmed as blocking.** What would confirm it: {what}"


def apply_rules(
    answer: VerifyAnswer,
    candidates: dict[str, Finding],
    issues: dict[str, Issue],
    *,
    changed_since_closed: Callable[[Issue], bool | None],
    source_of: Callable[[list[Finding]], str],
    thread_url: Callable[[Issue], str] = lambda i: "",
    extra_open: Iterable[str] = (),
) -> Outcome:
    """The harness's rules over one answer:

    - REFUTED is dropped (a CodeRabbit comment it restated gets `disagree` with the evidence).
    - Blocking requires CONFIRMED: a PLAUSIBLE blocker is a suggestion saying what would
      confirm it.
    - `open:H` corroborates H, which stays open; no new thread. A non-blocking H now CONFIRMED
      as blocking is raised in place (`upgrades`).
    - `settled:H` is dropped unless the code at H's anchor changed since H was closed
      (`changed_since_closed`; unknown counts as unchanged): then it is new, its body saying
      what H was.
    - `fixed:H` refound is a regression: a new issue whose body links H (retitled if its title
      would carry H's identity).
    - `coderabbit:<id>`: posted as ours, as v11 kept an agent finding over the CodeRabbit one,
      and the comment gets `agree`.
    - A "new" issue whose composed title gives a ledger hash (or the hash of an issue this
      round already raised, `extra_open`) is that issue, matched as above.
    Each new issue's identity is the hash of its composed title, computed here once."""
    out = Outcome(groups=list(answer.groups))
    xo = set(extra_open)
    posted: set[str] = set()
    for g in answer.groups:
        members = [candidates[m] for m in g.members if m in candidates]
        rep = _representative(members) if members else None
        if g.verdict == "REFUTED":
            out.dropped.append(_dropped(g, f"REFUTED: {g.evidence}"))
            if g.kind == "coderabbit":
                out.reactions.append(
                    {"comment_id": int(g.target), "action": "disagree", "reply": g.evidence}
                )
            continue
        sev, body = g.severity, g.body
        if sev == "blocking" and g.verdict != "CONFIRMED":
            sev, body = "suggestion", body + _plausible_note(g.confirm_by)
        suggestion = g.suggestion
        if suggestion and len({(m.file, m.line_start, m.line_end) for m in members}) > 1:
            # the group's members point at different lines: replacement code committed onto
            # the representative's lines could land on the wrong ones, so it is shown, not
            # offered as a one-click change
            body += (
                "\n\n**Suggested change** (the reports point at different lines, so it is not "
                f"offered as a one-click suggestion):\n```\n{suggestion}\n```"
            )
            suggestion = None
        f = Finding(
            file=rep.file if rep else "",
            title=g.title,
            body=body,
            severity=sev,
            category=rep.category if rep else "general",
            line_start=rep.line_start if rep else None,
            line_end=rep.line_end if rep else None,
            suggestion=suggestion,
            source=source_of(members),
            extra={"verdict": g.verdict, "evidence": g.evidence, "members": g.members},
            candidate_id=g.members[0],
        )
        out.kept.append((g, f))
        kind, target = g.kind, g.target
        if kind == "new" and f.hash in xo:
            kind, target = "open", f.hash
        elif kind == "new" and f.hash in issues:
            kind, target = issue_kind(issues[f.hash]), f.hash
        if kind == "coderabbit":
            out.reactions.append({"comment_id": int(target), "action": "agree"})
            kind = "new"
        if kind == "open" and target in xo:
            out.dropped.append(_dropped(g, f"DUPLICATE: raised this round as {target}"))
            continue
        if kind == "open":
            out.corroborated.setdefault(target, []).extend(g.members)
            if out.refound.get(target) != "CONFIRMED":
                out.refound[target] = g.verdict
            h = issues[target]
            if h.severity != "blocking" and sev == "blocking":
                out.upgrades[target] = f
            continue
        if kind == "settled":
            h = issues[target]
            if changed_since_closed(h) is not True:
                out.dropped.append(
                    _dropped(
                        g,
                        f"SETTLED: matches {h.status} issue {target}; the code at it has not changed since",
                    )
                )
                continue
            f.body += (
                f"\n\n_This was raised before and {h.status} at `{(h.closed_sha or '?')[:8]}`"
                + (f" ({thread_url(h)})" if thread_url(h) else "")
                + "; the code at that location has changed since._"
            )
            f.extra["revives"] = target
        if kind == "fixed":
            h = issues[target]
            if f.hash == target:
                f.title += " (regression)"
            link = thread_url(h) or f"`{target}`"
            f.body += (
                f"\n\n_This looks like a regression of an earlier finding that was fixed: {link}._"
            )
            f.extra["regression_of"] = target
        if f.hash in posted:
            out.dropped.append(_dropped(g, f"DUPLICATE: same title as another group ({f.hash})"))
            continue
        posted.add(f.hash)
        out.new.append(f)
    return out


def within_budget(new: list[Finding], budget: int) -> tuple[list[Finding], list[Finding]]:
    """The comment budget, in code (v11 asked the verifier to keep to it): every blocker is
    posted; then suggestions before nitpicks, CONFIRMED before PLAUSIBLE, and those more lanes
    found first, up to `budget` findings in all. Returns (posted, over budget); each keeps its
    order. A budget of 0 or less posts everything."""
    if budget <= 0 or len(new) <= budget:
        return list(new), []

    def rank(f: Finding) -> tuple[int, int, int]:
        confirmed = f.extra.get("verdict") == "CONFIRMED"
        return (-_SEV_RANK.get(f.severity, 0), -int(confirmed), -len(f.extra.get("members") or []))

    keep = {id(f) for f in new if f.severity == "blocking"}
    for f in sorted((f for f in new if f.severity != "blocking"), key=rank):
        if len(keep) >= budget:
            break
        keep.add(id(f))
    return [f for f in new if id(f) in keep], [f for f in new if id(f) not in keep]


# ---- what the lanes are told from the ledger ----


def thread_url(repo: str, number: int, issue: Issue) -> str:
    if not issue.comment_id:
        return ""
    return f"https://github.com/{repo}/pull/{number}#discussion_r{issue.comment_id}"


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


def ledger_summary(
    issues: dict[str, Issue],
    *,
    thread_status: dict[str, str],
    last_message: dict[str, str],
    raised_now: Iterable[Finding] = (),
) -> dict[str, list[dict[str, Any]]]:
    """The verifier's compact view of the ledger: open issues (with what the thread lane
    decided this round and the latest message on the thread), settled ones (a candidate that
    restates one is dropped unless the code changed) and fixed ones (refound = regression).
    `raised_now`: this round's verified new issues, listed as open for the confirmation pass."""
    out: dict[str, list[dict[str, Any]]] = {"open": [], "settled": [], "fixed": []}
    for h, i in issues.items():
        row: dict[str, Any] = {
            "hash": h,
            "severity": i.severity,
            "file": i.file,
            "lines": _lines(i),
            "title": i.title,
            "status": i.status,
        }
        kind = issue_kind(i)
        if kind == "open":
            if h in thread_status:
                row["thread_lane"] = thread_status[h]
            if last_message.get(h):
                row["last_message"] = last_message[h][:600]
        else:
            row["closed_at"] = (i.closed_sha or "")[:12]
            row["closing_note"] = (i.closed_reason or "")[:200]
        out[kind].append(row)
    for f in raised_now:
        out["open"].append(
            {
                "hash": f.hash,
                "severity": f.severity,
                "file": f.file,
                "lines": [f.line_start, f.line_end],
                "title": f.title,
                "status": "raised this round",
            }
        )
    return out


# ---- the ledger verdict (§3) ----


def standing(
    issues: dict[str, Issue],
    *,
    reconciled: bool = True,
    pending: Iterable[str] = (),
    last_reviewed_sha: str | None = None,
    open_threads: Iterable[str] = (),
) -> list[Issue]:
    """The issues that count toward the verdict. A PR's backfilled history holds blockers that
    vanished without an outcome: until its ledger was reconciled, and for the issues the
    reconcile left to the thread lane (`pending`) until the lane has judged them, only those
    last seen at the previous reviewed head or with an open thread on the PR count."""
    threads, doubtful = set(open_threads), set(pending)
    rows = [i for i in issues.values() if i.status in VERDICT_STANDING]
    return [
        i
        for i in rows
        if (reconciled and i.hash not in doubtful)
        or i.hash in threads
        or (last_reviewed_sha and i.last_seen_sha == last_reviewed_sha)
    ]


def carried(issues: Iterable[Issue], url: Callable[[Issue], str]) -> list[Finding]:
    """Standing issues as the findings a round carries (never re-posted)."""
    return [
        Finding(
            file=i.file,
            title=i.title,
            body=i.body,
            severity=i.severity,
            category=i.category,
            line_start=i.line_start,
            line_end=i.line_end,
            source="carried",
            prior_hash=i.hash,
            extra={"thread_url": url(i), "status": i.status},
        )
        for i in issues
    ]


def verdict(*, blockers: int, suggestions: int, full_strength: bool, confirmed: bool) -> str:
    """REQUEST_CHANGES iff a blocker stands; APPROVE only with nothing above a nitpick standing,
    on a full-strength run whose confirmation pass ran when the policy asks for one; else
    COMMENT. (The head being live is checked when publishing.)"""
    if blockers:
        return "REQUEST_CHANGES"
    if not suggestions and full_strength and confirmed:
        return "APPROVE"
    return "COMMENT"


def project(
    issues: dict[str, Issue], statuses: dict[str, str], upgrades: Iterable[str] = ()
) -> dict[str, Issue]:
    """`issues` with this round's pending status changes (`statuses`: hash -> ledger status)
    and severity upgrades applied, for decisions taken before they are recorded (the gate, a
    dry run). Copies; the ledger rows are untouched."""
    out = dict(issues)
    for h, st in statuses.items():
        if h in out:
            out[h] = dataclasses.replace(out[h], status=st)
    for h in upgrades:
        if h in out:
            out[h] = dataclasses.replace(out[h], severity="blocking")
    return out
