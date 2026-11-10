"""Render and post the GitHub review. Port of review_poster.py + diff_position_mapper.py.

`render()` is pure: given a ReviewModel it returns the exact body and comments.
Network happens only in `publish()`.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from textwrap import indent
from typing import Any

from . import github
from .contract import Finding, VerifierOutput
from .dedupe import (
    FINDING_MARKER_RE,
    THREAD_STATUS_MARKER,
    THREAD_STATUS_RE,
    ExistingComment,
    Suppressed,
    collapse_same_root,
    find_duplicate,
    has_same_root_duplicates,
    normalize_root_id,
    thread_has_resolution_reply,
)
from .gh import Gh
from .models import FailKind, ReviewError

SEVERITY_ICONS = {"blocking": "\U0001f534", "suggestion": "\U0001f7e1", "nitpick": "\U0001f4ac"}
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_SUMMARY_SOURCE_LINE_RE = re.compile(
    r"^\s*(?:>\s*)?(?:[-*+]\s+)?(?:#{1,6}\s+)?(?:\*{1,2}|_{1,2})?Source\s*:(?:\*{1,2}|_{1,2})?\s*.*$",
    re.IGNORECASE,
)


# ---- diff position mapping ----


def parse_diff(diff_text: str) -> dict[str, list[dict[str, int]]]:
    files: dict[str, list[dict[str, int]]] = {}
    current: str | None = None
    rename_map: dict[str, str] = {}
    pending_from: str | None = None
    binary = False
    for line in diff_text.splitlines():
        if line.startswith("rename from "):
            pending_from = line[len("rename from ") :]
            continue
        if line.startswith("rename to ") and pending_from:
            rename_map[pending_from] = line[len("rename to ") :]
            pending_from = None
            continue
        if line.startswith("Binary files "):
            binary = True
            continue
        if line.startswith("diff --git "):
            binary = False
            pending_from = None
            parts = line.split(" b/", 1)
            if len(parts) == 2:
                current = parts[1].rstrip("\t")
                files.setdefault(current, [])
            continue
        if line.startswith("+++ b/"):
            # git appends a tab after paths that contain spaces
            current = line[len("+++ b/") :].rstrip("\t")
            files.setdefault(current, [])
            continue
        if binary or current is None:
            continue
        m = HUNK_RE.match(line)
        if m:
            start = int(m.group(3))
            count = int(m.group(4)) if m.group(4) is not None else 1
            files[current].append(
                {
                    "new_start": start,
                    "new_count": count,
                    "new_end": start + count - 1 if count > 0 else start - 1,
                }
            )
    for old, new in rename_map.items():
        if new in files and old not in files:
            files[old] = files[new]
    return files


def line_in_diff(parsed: dict[str, list[dict[str, int]]], path: str, line: int) -> bool:
    return any(h["new_start"] <= line <= h["new_end"] for h in parsed.get(path, []))


# the verifier's `source` labels; `codex`/`claude` are the names older templates used
LEGACY_SOURCE_PHASE = {
    "phase1": "phase1",
    "phase2": "phase2",
    "codex": "phase1",
    "claude": "phase2",
}


def attribute_sources(
    verified: VerifierOutput,
    reviewers: list[dict[str, Any]],
    lane_findings: dict[str, list[Finding]],
) -> None:
    """Rewrite each verified finding's `source` from the verifier's labels (`phase1` or the older `codex` =
    Phase-1 lanes, `phase2` or `claude` = Phase-2 lanes, `coderabbit`) into the model ids and agents that
    actually raised it. Within a phase the lanes whose own output carries the same
    `finding_hash` are named; if the verifier retitled or merged, every lane of that phase is.
    `reviewers` is the run's completed-lane provenance, `lane_findings` maps `"phase:role"`
    (or `"phase:fresh:role"`) to that lane's findings."""
    lanes = [r for r in reviewers if r.get("status") == "completed"]
    phases_run = {str(r["phase"]) for r in lanes}

    def raised_by(reviewer: dict[str, Any], f: Finding) -> bool:
        phase = str(reviewer["phase"])
        key = str(reviewer.get("key") or reviewer["role"])  # `<role>#2` for a comparison lane
        prefix = "fresh:" if reviewer.get("fresh") else ""
        return any(lf.hash == f.hash for lf in lane_findings.get(f"{phase}:{prefix}{key}", []))

    for f in verified.findings:
        labels = _source_labels(f.source)
        phases = {LEGACY_SOURCE_PHASE[x] for x in labels if x in LEGACY_SOURCE_PHASE}
        if not labels:  # verifier gave nothing usable: go by the hash, else name every lane
            phases = {
                p
                for p in phases_run
                if any(str(r["phase"]) == p and raised_by(r, f) for r in lanes)
            }
            phases = phases or phases_run
        parts: list[str] = []
        for phase in sorted(phases & phases_run):
            in_phase = [r for r in lanes if r["phase"] == phase]
            parts += name_lanes([r for r in in_phase if raised_by(r, f)] or in_phase)
        parts += sorted(x for x in labels if x not in LEGACY_SOURCE_PHASE)
        if parts:
            f.source = "; ".join(parts)


def name_lanes(lanes: Iterable[dict[str, Any]]) -> list[str]:
    """Reviewer lanes as a finding's source names them: one "`model` (agent: roles)" per
    model and agent, in the order they first appear."""
    groups: dict[tuple[str, str], list[str]] = {}
    for r in lanes:
        roles = groups.setdefault((str(r["model"]), str(r["agent"])), [])
        if str(r["role"]) not in roles:
            roles.append(str(r["role"]))
    return [f"`{m}` ({a}: {', '.join(rs)})" for (m, a), rs in groups.items()]


def _source_labels(raw: str) -> list[str]:
    """`['claude', 'codex']`, `claude`, `unknown` → the labels the verifier meant."""
    out = [x.strip().strip("'\"").lower() for x in raw.strip().strip("[]").split(",")]
    return [x for x in out if x and x != "unknown"]


def _follow_up(dropped: dict[str, Any]) -> bool:
    """Whether a dropped finding is published as an out-of-scope follow-up suggestion: policy
    v12 marks it (`follow_up`, rules.py); the v11 verifier's drops go by their reason's wording."""
    if "follow_up" in dropped:
        return bool(dropped["follow_up"])
    return any(k in str(dropped.get("reason", "")).lower() for k in ("out of scope", "outside"))


def comment_body(f: Finding) -> str:
    marker = f"<!-- thepastaclaw-review v1 finding={f.hash} dedupe={f.dedupe_key}"
    root = normalize_root_id(f.root_id)
    if root:
        marker += f" root={root}"
    marker += " -->"
    parts = [
        marker,
        f"**{SEVERITY_ICONS.get(f.severity, SEVERITY_ICONS['nitpick'])} {f.severity.capitalize()}: {f.title}**",
        "",
        f.body,
    ]
    if f.suggestion:
        parts += ["", "```suggestion", f.suggestion, "```"]
    parts.append(f"\n<sub>source: {f.source}</sub>")
    return "\n".join(parts)


def map_comment(f: Finding, parsed: dict[str, list[dict[str, int]]]) -> dict[str, Any] | None:
    if not f.file or not f.line_end or not line_in_diff(parsed, f.file, f.line_end):
        return None
    c: dict[str, Any] = {
        "path": f.file,
        "line": f.line_end,
        "side": "RIGHT",
        "body": comment_body(f),
    }
    start = f.line_start or f.line_end
    if start != f.line_end:
        if not line_in_diff(parsed, f.file, start):
            for h in parsed.get(f.file, []):
                if h["new_start"] <= f.line_end <= h["new_end"]:
                    start = h["new_start"]
                    break
        if start != f.line_end:
            c["start_line"] = start
            c["start_side"] = "RIGHT"
    return c


# ---- body rendering ----


@dataclass(slots=True)
class Provenance:
    reviewers: list[dict[str, Any]]  # {model, agent, role, effort?, status, phase}
    verifier: dict[str, Any]  # {model, agent, role}
    policy_fingerprint: str
    triage: dict[str, Any] | None = None  # {tier, model, effort, method, reasoning, error}
    phase2_skipped: str | None = None  # set when a final review was published from Phase 1 only
    phase1_skipped: str | None = (
        None  # set when the run went straight to Phase 2 (backlog / failure)
    )
    phase1_choice: dict[str, Any] | None = None  # {model, reason, skipped:[{model, reason}]}
    adhoc: bool = False  # repo has no skills entry: generic guidance, all specialists offered
    fresh_final: bool = False  # an independent Phase-2 gate ran after iterative reconciliation
    conversation: bool = False  # no reviewer lanes ran: a discussion outcome moved the verdict
    # stand-in models were in use (primary models out of quota): {reason, source, since,
    # label, substitutes:{primary: stand_in}, phase1_effort_cap}
    degraded: dict[str, Any] | None = None
    # the second model the Phase-2 reviewers also ran on for a model comparison
    compare_model: str | None = None
    # Phase 1 ran beside Phase 2 with no blocker gate (a `single_stage` tier)
    single_stage: bool = False
    # policy v12: the confirmation pass that ran on this head {roles, new}
    confirmation: dict[str, Any] | None = None
    # the first full review ran every Phase-2 reviewer a second time, independently
    fresh_first: bool = False
    # a later round's new findings on code unchanged since the previous final review
    # (late.py): {since, nits, followups, repeats, candidates, confirmed, unconfirmed,
    # invalid, failed}
    late: dict[str, Any] | None = None


@dataclass(slots=True)
class ReviewModel:
    repo: str
    number: int
    head_sha: str
    phase: str  # preliminary|final
    verified: VerifierOutput
    provenance: Provenance
    kept: list[Finding]
    skipped: list[Finding]
    suppressed: list[Suppressed]
    comments: list[dict[str, Any]]
    body_only: bool = False
    superseded_note: str | None = None
    extra_lines: list[str] = field(default_factory=list)
    rereview: bool = False
    # policy v11: an approval with non-blocking findings says they can be fixed forward
    fix_forward: bool = False
    # prior findings this review settled as DISPUTED (each with `extra["thread_url"]`): shown
    # once, on the head where the dispute was settled
    disputed: list[Finding] = field(default_factory=list)

    @property
    def canonical_event(self) -> str:
        return verdict_event(self.verified, self.provenance)


def _location(f: Finding) -> str:
    if not f.line_start:
        return f.file or "(no file)"
    if f.line_start == f.line_end or not f.line_end:
        return f"{f.file}:{f.line_start}"
    return f"{f.file}:{f.line_start}-{f.line_end}"


AI_PROMPT_GUIDANCE = (
    "These findings are from an automated code review (PastaClaw). How to work through them:",
    "- Verify each finding against the code; fix only what is real, as narrowly as the finding.",
    "- If a finding is wrong, decline it on its thread in one or two sentences; the bot answers "
    "once and stops.",
    "- To defer one, link a tracking issue on its thread (or reply asking the bot to file one).",
    "- Suggestions don't block the merge, but most are worth fixing in this PR; judge each on "
    "its merits.",
    "- Keep the PR's scope; don't grow it to satisfy hardening suggestions.",
    "- Carried items were raised in an earlier round: read the thread first, and don't re-fix "
    "what you already fixed, declined or deferred there. Disputed and deferred items are "
    "already handled and are not listed.",
)
_AWAITING_NOTE = (
    "awaiting a tracking link: link an issue on its thread, or ask there and PastaClaw files "
    "it; no code change needed"
)


def _awaiting(f: Finding) -> bool:
    """A prior finding whose deferral is held until a tracking issue is linked."""
    return bool(f.extra.get("awaiting_tracking"))


def _ai_prompt(
    findings: list[Finding], suppressed: list[Suppressed], carried: list[Finding] | None = None
) -> str | None:
    """The agent prompt: the guidance, then every finding to act on by file. New findings come
    with their text; a carried one (folded onto its existing thread, or policy v12's standing
    issues) is listed with its thread only, so it never reads as new."""
    # the thread a carried finding is on ("" when unknown); None marks a new finding
    entries: list[tuple[Finding, str | None]] = [(f, None) for f in findings]
    entries += [
        (s.finding, s.html_url or "") for s in suppressed if s.action != "deduped_same_batch_root"
    ]
    entries += [(f, str(f.extra.get("thread_url") or "")) for f in carried or []]
    if not entries:
        return None
    lines = [*AI_PROMPT_GUIDANCE, ""]
    by_file: dict[str, list[tuple[Finding, str | None]]] = {}
    for f, thread in entries:
        by_file.setdefault(f.file or "(no file)", []).append((f, thread))
    for path, items in by_file.items():
        lines.append(f"In `{path}`:")
        for f, thread in items:
            lines.append(f"- [{f.severity.upper()}] {_location(f)}: {f.title}")
            link = f"; thread: {thread}" if thread else ""
            if _awaiting(f):
                lines.append(f"  ({_AWAITING_NOTE}{link})")
            elif thread is not None:
                lines.append(f"  (carried{link})")
            else:
                if f.extra.get("thread_url"):
                    lines.append(f"  (existing thread: {f.extra['thread_url']})")
                if f.body:
                    lines.append(indent(f.body, "  "))
        lines.append("")
    return "\n".join(lines).rstrip()


def _fence(content: str) -> str:
    longest = max((len(m.group(0)) for m in re.finditer(r"`+", content)), default=0)
    return "`" * max(3, longest + 1)


def _model_label(lane: dict[str, Any]) -> str:
    """A lane's model, naming the primary it stood in for when it was a degraded-mode swap."""
    sub = lane.get("substitute_for")
    return f"`{lane['model']}` (standing in for `{sub}`)" if sub else f"`{lane['model']}`"


def _source_line(p: Provenance) -> str:
    if p.conversation:
        return (
            f"Source: conversation lane {_model_label(p.verifier)} "
            f"(agent: `{p.verifier['agent']}`); no reviewer or verifier lanes ran for this follow-up"
        )
    parts = [
        f"reviewer {i}: {_model_label(r)} (agent: `{r['agent']}`, role: `{r['role']}`)"
        for i, r in enumerate(p.reviewers, 1)
    ]
    parts.append(
        f"final verifier: {_model_label(p.verifier)} "
        f"(agent: `{p.verifier['agent']}`, role: `{p.verifier['role']}`)"
    )
    return "Source: " + "; ".join(parts)


DEGRADED_BADGE = github.DEGRADED_BADGE  # one badge for reviews, gate and queue comments


def _substitutions(d: dict[str, Any]) -> str:
    """`primary → stand-in` pairs from a degraded disclosure; "" when it carries none."""
    return ", ".join(f"`{k}` → `{v}`" for k, v in sorted((d.get("substitutes") or {}).items()))


def degraded_banner(d: dict[str, Any]) -> str:
    """The visible warning at the top of anything published while stand-ins were in use."""
    subs = _substitutions(d)
    cap = d.get("phase1_effort_cap")
    return (
        f"> **{DEGRADED_BADGE} review.** The primary review models were unavailable "
        f"({d.get('reason') or 'quota exhausted'}), so this review ran on stand-in models"
        + (f": {subs}" if subs else "")
        + ". Both review phases and the independent verifiers still ran, but on weaker models"
        + (f", with Phase 1 capped at `{cap}` effort" if cap else "")
        + ". Treat the verdict as provisional; a full-strength re-review will run on the next "
        "push once the primary models are back."
    )


def _degraded_line(d: dict[str, Any]) -> str:
    subs = _substitutions(d)
    line = f"- **Degraded mode**: {d.get('reason') or 'primary models unavailable'} (detected by {d.get('source') or 'probe'}"
    if d.get("since"):
        line += f", since {d['since']}"
    line += ")"
    if subs:
        line += f"; stand-ins {subs}"
    if d.get("phase1_effort_cap"):
        line += f"; Phase 1 effort capped at `{d['phase1_effort_cap']}`"
    return line


def _triage_line(t: dict[str, Any]) -> str:
    if str(t.get("method", "")).startswith("llm:"):
        how = f"{_model_label(t)} (effort {t['effort']})"
    else:
        how = f"fallback after triage failure ({t.get('error') or 'unknown'})"
    why = f" — {t['reasoning']}" if t.get("reasoning") else ""
    return f"- Triage: `{t['tier']}` by {how}{why}"


def _phase1_choice_line(c: dict[str, Any]) -> str:
    line = f"- Phase 1 model: `{c['model']}` — {c['reason']}"
    skipped = [f"`{s['model']}` ({s['reason']})" for s in c.get("skipped") or []]
    if skipped:
        line += "; passed over " + ", ".join(skipped)
    return line


def _late_line(d: dict[str, Any]) -> str:
    """What a later round did with new findings on code unchanged since the last final review."""
    parts: list[str] = []
    n = d.get("candidates", 0)
    if n and d.get("failed"):
        parts.append(
            f"{n} new blocker(s) stand as raised: the independent confirmation pass failed"
        )
    elif n:
        outcome = ", ".join(
            f"{d[k]} {label}"
            for k, label in (
                ("confirmed", "confirmed"),
                ("unconfirmed", "not confirmed (now follow-up candidates)"),
                ("invalid", "judged invalid (dropped)"),
            )
            if d.get(k)
        )
        earlier = (
            f" ({d['remembered']} as an earlier confirmation decided, the file unchanged since)"
            if d.get("remembered")
            else ""
        )
        parts.append(
            f"{n} new blocker(s) went to an independent confirmation pass (the final "
            f"verifier's model in a fresh session, without its reasoning): {outcome}{earlier}"
        )
    for key, label in (
        ("nits", "new nitpick(s) dropped"),
        ("followups", "new suggestion(s) listed as follow-up candidates, not posted inline"),
        ("repeats", "follow-up candidate(s) already listed by an earlier review, not repeated"),
    ):
        if d.get(key):
            parts.append(f"{d[key]} {label}")
    return f"- Code unchanged since the last final review (`{d.get('since', '')}`): " + (
        "; ".join(parts) or "nothing new"
    )


def _provenance_lines(p: Provenance, phase: str) -> list[str]:
    if p.conversation:
        return ([_degraded_line(p.degraded)] if p.degraded else []) + [
            "- Verdict moved because the inline discussion withdrew or deferred findings of the "
            "standing review of this commit; the code was not re-reviewed"
        ]

    def fmt(r: dict[str, Any]) -> str:
        status = r["status"] + (f", effort {r['effort']}" if r.get("effort") else "")
        return f"{_model_label(r)} — {r['role']} ({status}); agent `{r['agent']}`"

    p1 = [fmt(r) for r in p.reviewers if r.get("phase") == "phase1"]
    p2 = [fmt(r) for r in p.reviewers if r.get("phase") == "phase2"]
    lines: list[str] = []
    if p.degraded:
        lines.append(_degraded_line(p.degraded))
    if p.adhoc:
        lines.append(
            "- Ad hoc review: this repository has no PastaClaw review skill; reviewers used "
            "generic guidance and the repository's own conventions"
        )
    if p.triage:
        lines.append(_triage_line(p.triage))
    if p.phase1_skipped:
        state = "failed" if p.phase1_skipped == github.PHASE1_FAILED else "not run"
        p1_line = f"- Phase 1 reviewers: **{state} ({p.phase1_skipped})**"
    else:
        p1_line = "- Phase 1 reviewers: " + (", ".join(p1) if p1 else "provenance missing")
    lines.append(p1_line)
    if p.phase1_choice and not p.phase1_skipped:
        lines.append(_phase1_choice_line(p.phase1_choice))
    if p.single_stage:
        lines.append(
            "- Single stage: Phase 1 and Phase 2 reviewed this head side by side, with no "
            "blocker gate between them (triage tier)"
        )
    if p.fresh_first:
        lines.append(
            "- Second Phase-2 pass: on this first full review every Phase-2 reviewer also ran a "
            "second, independent time, and the verifier weighed both sets"
        )
    if p.fresh_final:
        lines.append(
            "- Fresh final gate: an independent Phase-2 review ran after iterative findings were reconciled"
        )
    if p.late:
        lines.append(_late_line(p.late))
    if p.confirmation:
        roles = ", ".join(f"`{r}`" for r in p.confirmation.get("roles") or [])
        lines.append(
            f"- Confirmation pass: {roles} reviewed the whole change again, independently, "
            f"after the review found no blocker ({p.confirmation.get('new', 0)} new finding(s))"
        )
    lines += [
        f"- Fresh verifier: {_model_label(p.verifier)} — {p.verifier['role']}; agent `{p.verifier['agent']}`",
    ]
    if phase == "preliminary":
        lines.append(
            f"- Phase 2 reviewers: **not run ({p.phase2_skipped or 'deferred by blocker gate'})**"
        )
    elif p.phase2_skipped:
        lines.append(
            f"- Phase 2 reviewers: **not run ({p.phase2_skipped})**; this review comments and never approves"
        )
    else:
        lines.append(
            "- Phase 2 reviewers: " + (", ".join(p2) if p2 else "no successful evidence recorded")
        )
        if p.compare_model:
            lines.append(
                f"- Model comparison: every Phase-2 reviewer also ran on `{p.compare_model}`; "
                "the verifier weighed both sets without knowing which model wrote which"
            )
    return lines


_CARRIED_STATUS = {
    "deferral_pending": " (a deferral was proposed; it needs a maintainer's call)",
    "resolved": " (thread resolved by hand; it still counts until withdrawn or fixed)",
}
# dedupe folded these onto a thread a human resolved
_FOLDED_ON_RESOLVED = frozenset(
    {
        "deduped_existing_resolved_thread",
        "replied_to_resolved_thread",
        "would_reply_to_resolved_thread",
        "failed_to_reply_to_resolved_thread",
    }
)
# a second finding of this batch on a root another one already speaks for
_BATCH_ALIASES = frozenset({"deduped_same_batch_root", "deduped_same_batch_existing_thread"})


def _folded(s: Suppressed) -> Finding:
    """A finding dedupe folded onto its earlier thread, as `_carried_line` shows it."""
    extra = {**s.finding.extra, "thread_url": s.html_url or s.finding.extra.get("thread_url")}
    if s.action in _FOLDED_ON_RESOLVED:
        extra["status"] = "resolved"
    return dataclasses.replace(s.finding, extra=extra)


def _awaiting_line(held: list[Finding]) -> str:
    """The one compact line for prior findings whose deferral waits for a tracking issue."""
    named = "; ".join(
        _block_text(f.title)
        + (f" ([thread]({f.extra['thread_url']}))" if f.extra.get("thread_url") else "")
        for f in held
    )
    return (
        f"⏸️ **Awaiting tracking link** ({len(held)}): {named}. Link an issue on its thread, "
        "or ask there and PastaClaw files it."
    )


def _carried_line(f: Finding) -> str:
    """One standing finding from an earlier round in the body: not re-posted or re-described,
    linked to its thread."""
    icon = SEVERITY_ICONS.get(f.severity, SEVERITY_ICONS["nitpick"])
    url = f.extra.get("thread_url")
    link = f" ([thread]({url}))" if url else ""
    status = _CARRIED_STATUS.get(str(f.extra.get("status") or ""), "")
    return f"- {icon} **{f.severity.capitalize()}**: {f.title} — `{_location(f)}`{link}{status}"


def _finding_block(f: Finding) -> str:
    icon = SEVERITY_ICONS.get(f.severity, SEVERITY_ICONS["nitpick"])
    lines = [
        f"**{icon} {f.severity.capitalize()}: {f.title}**",
        f"`{_location(f)}`",
        "",
        f.body or "_No details provided._",
    ]
    if f.suggestion:
        lines += ["", "```suggestion", f.suggestion, "```"]
    lines += ["", f"<sub>source: {f.source}</sub>", ""]
    return "\n".join(lines)


def _counts_line(
    findings: list[Finding], *, include_zero: bool = False, blocker_count: int | None = None
) -> str:
    counts = {"blocking": 0, "suggestion": 0, "nitpick": 0}
    for f in findings:
        counts[f.severity if f.severity in counts else "nitpick"] += 1
    if blocker_count is not None:
        counts["blocking"] = blocker_count
    return " | ".join(
        f"{SEVERITY_ICONS[severity]} {counts[severity]} {label}"
        for severity, label in (
            ("blocking", "blocking"),
            ("suggestion", "suggestion(s)"),
            ("nitpick", "nitpick(s)"),
        )
        if counts[severity] or include_zero
    )


def _followups_block(followups: list[Finding]) -> list[str]:
    """The collapsed "Follow-up candidates (optional)" section, or nothing."""
    if not followups:
        return []
    return [
        "",
        "<details>",
        f"<summary>Follow-up candidates (optional) ({len(followups)})</summary>",
        "",
        "New non-blocking observations on code unchanged since the last full review. "
        "No inline threads are opened for them and later reviews do not repeat them: fix "
        "forward or track separately if they are worth it.",
        "",
        *(_finding_block(f) for f in followups),
        "</details>",
    ]


def render(m: ReviewModel) -> str:
    if m.phase == "preliminary":
        title = "Preliminary review — Phase 1 blocker gate"
    elif m.provenance.phase2_skipped:
        tier = (m.provenance.triage or {}).get("tier", "trivial")
        title = f"Final review — Phase 1 only ({tier} change)"
    elif m.provenance.phase1_skipped:
        title = f"Final validation — Phase 2 only ({github.phase2_only_label(m.provenance.phase1_skipped)})"
    else:
        title = "Final validation — Phase 1 + Phase 2"
    summary = "\n".join(
        line for line in m.verified.summary.splitlines() if not _SUMMARY_SOURCE_LINE_RE.match(line)
    ).strip()
    if m.rereview:
        title = f"Re-review — {title}"
    if m.provenance.degraded:
        title = f"{DEGRADED_BADGE} — {title}"
    parts = [
        github.REVIEW_MARKER,
        f"<!-- thepastaclaw-review-phase v1 phase={m.phase} sha={m.head_sha} policy={m.provenance.policy_fingerprint[:16]} -->",
        f"## {title}",
        "",
    ]
    if m.provenance.degraded:
        parts += [degraded_banner(m.provenance.degraded), ""]
    parts += [summary, ""]
    if m.phase == "preliminary":
        parts += [
            "Validated blockers were found by the Phase-1 review and confirmed by a fresh verifier. Phase 2 is deferred until a fresh same-head revalidation clears the blocker gate."
            # phase2_skipped on a preliminary: the points gate deferred on suggestions alone
            if m.verified.blocker_count or not m.provenance.phase2_skipped
            else "The Phase-1 findings confirmed by a fresh verifier are above the gate budget. Phase 2 is deferred until a fresh same-head revalidation brings them under it.",
            "",
        ]
    if m.superseded_note:
        parts += [m.superseded_note, ""]
    counts_line = _counts_line(
        [*m.verified.findings, *m.verified.carried],
        blocker_count=m.verified.blocker_count if m.phase == "preliminary" else None,
    )
    if counts_line or m.rereview:
        if not counts_line:
            counts_line = "🔴 0 blocking | 🟡 0 suggestion(s) | 💬 0 nitpick(s)"
        parts.append(counts_line)
    optional = m.verified.followups or any(
        f.severity != "blocking" for f in (*m.verified.findings, *m.verified.carried)
    )
    if m.fix_forward and optional and m.canonical_event == "APPROVE":
        parts += [
            "",
            "**Approved:** nothing here blocks the merge. The non-blocking findings are "
            "optional: fix them forward, in this PR or a follow-up, without waiting for "
            "another review round.",
        ]
    if m.skipped:
        # findings GitHub cannot take inline still have to reach the developer in full
        why = (
            "GitHub refused the PR diff as too large"
            if m.body_only
            else "the lines are not part of this PR's diff"
        )
        parts += [
            "",
            f"### {len(m.skipped)} finding(s) not shown inline ({why})",
            "",
        ]
        parts += [_finding_block(f) for f in m.skipped]
    folded = [_folded(s) for s in m.suppressed if s.action not in _BATCH_ALIASES]
    earlier = [*folded, *m.verified.carried]
    still_open = [f for f in earlier if not _awaiting(f)]
    if still_open:
        parts += ["", f"### Still open from earlier rounds ({len(still_open)})", ""]
        parts += [_carried_line(f) for f in still_open]
    if m.disputed:
        parts += ["", f"### Disputed — left to maintainers ({len(m.disputed)})", ""]
        parts += [_carried_line(f) for f in m.disputed]
    held = [f for f in [*earlier, *m.kept] if _awaiting(f)]
    if held:
        parts += ["", _awaiting_line(held)]
    if m.verified.noted:
        parts += [
            "",
            "<details>",
            f"<summary>Also noticed ({len(m.verified.noted)}): non-blocking, on code unchanged "
            "since the last full review</summary>",
            "",
            *(_finding_block(f) for f in m.verified.noted),
            "</details>",
        ]
    parts += _followups_block(m.verified.followups)
    parts += [
        "",
        "<details>",
        "<summary>Review provenance</summary>",
        "",
        _source_line(m.provenance),
        "",
        *_provenance_lines(m.provenance, m.phase),
        "",
        "</details>",
    ]
    prompt = _ai_prompt(m.kept, m.suppressed, m.verified.carried)
    if m.rereview and not prompt:
        prompt = (
            "These findings are from an automated code review. Verify the current code and "
            "confirm that no unresolved issues remain.\n\nNo unresolved findings remain from "
            "the prior review on this head."
        )
    if prompt:
        fence = _fence(prompt)
        parts += [
            "",
            "<details>",
            "<summary>🤖 Prompt for all review comments with AI agents</summary>",
            "",
            fence,
            prompt,
            fence,
            "</details>",
        ]
    oos = m.verified.out_of_scope + [d for d in m.verified.dropped if _follow_up(d)]
    if oos:
        parts += [
            "",
            "<details>",
            f"<summary>Out-of-scope follow-up suggestions ({len(oos)})</summary>",
            "",
            "These are valid observations, but they are outside this PR's scope and should be handled in separate issues or author/maintainer-requested PRs rather than blocking this review.",
            "",
        ]
        for o in oos:
            t = o.get("title") or o.get("original_title") or "Follow-up"
            b = o.get("body") or o.get("reason") or "Worth tracking separately."
            fu = (
                o.get("suggested_followup")
                or "Consider creating a separate issue or author/maintainer-requested PR for this."
            )
            parts.append(f"- **{t}** — {b}\n  - Follow-up: {fu}")
        parts += ["", "</details>"]
    parts += m.extra_lines
    return "\n".join(parts)


# ---- assembly + posting ----


@dataclass(slots=True)
class PublishResult:
    posted: bool
    event: str
    transport_event: str
    body: str
    review_id: int | None = None
    review_url: str | None = None
    comments: list[dict[str, Any]] = field(default_factory=list)
    suppressed: list[Suppressed] = field(default_factory=list)
    skipped_reason: str | None = None


THREAD_REUSE_MARKER = "<!-- thepastaclaw-thread-reuse v1 root={root} sha={sha} -->"


def _thread_reuse_marker(root: ExistingComment, head_sha: str) -> str:
    """Marker for the one reply used when reusing an already resolved finding thread."""
    return THREAD_REUSE_MARKER.format(root=root.id, sha=head_sha)


def _thread_has_marker(thread: dict[str, Any], marker: str) -> bool:
    return any(
        marker in str(c.get("body") or "")
        for c in (thread.get("comments") or {}).get("nodes") or []
    )


def dedupe_against_github(
    gh: Gh,
    repo: str,
    number: int,
    head_sha: str,
    findings: list[Finding],
    bot_login: str,
    *,
    dry_run: bool,
    fresh: Callable[[Finding, ExistingComment], bool] | None = None,
    settled_hashes: set[str] | None = None,
) -> tuple[list[Finding], list[Suppressed]]:
    """`fresh(f, comment)`: `f` is never folded onto that comment (policy v12: a regression or a
    revival gets a new thread instead of disappearing into a closed one)."""
    records = [
        ExistingComment.from_github(c)
        for c in github.inline_comments(gh, repo, number)
        if (c.get("user") or {}).get("login") == bot_login
    ]
    if not records:
        return list(findings), []
    kept: list[Finding] = []
    sup: list[Suppressed] = []
    # Several verifier findings can fuzzy-match the same historical root.  A root can only
    # receive one carry-forward reply for a reviewed head; otherwise aliases turn into
    # duplicate GitHub comments even though each finding was considered once.
    seen_matches: set[tuple[str, str]] = set()
    settled_hashes = settled_hashes or set()
    for f in findings:
        # A finding explicitly reintroduced after a prior lift is new evidence, not a carry
        # onto the old resolved root. Give it a fresh inline comment so the old withdrawal is
        # never silently turned back into a live thread; a reworded one whose own comment is
        # already up (an earlier head) is deduped onto that as usual.
        marker = f.extra.get("reintroduced_finding_hash")
        if marker in settled_hashes and (
            f.hash == marker or not any(r.finding_hash == f.hash for r in records)
        ):
            kept.append(f)
            continue
        candidates = records if fresh is None else [r for r in records if not fresh(f, r)]
        match, reason = find_duplicate(f, candidates)
        if not match:
            kept.append(f)
            continue
        thread = github.thread_for_comment(gh, match.node_id) or {}
        base = Suppressed(
            f,
            "",
            detail=reason or "",
            comment_id=match.id,
            html_url=match.html_url,
            comment_hash=match.finding_hash,
        )
        match_key = (match.node_id or match.id, head_sha)
        if match_key in seen_matches:
            base.action = "deduped_same_batch_existing_thread"
            sup.append(base)
            continue
        seen_matches.add(match_key)
        if thread_has_resolution_reply(thread, bot_login):
            base.action = "deduped_maintainer_addressed"
        elif not thread.get("isResolved"):
            base.action = "deduped_existing_open_thread"
        else:
            marker = _thread_reuse_marker(match, head_sha)
            already = any(
                (c.get("author") or {}).get("login") == bot_login
                and head_sha[:8] in (c.get("body") or "")
                for c in (thread.get("comments") or {}).get("nodes") or []
            )
            if already or _thread_has_marker(thread, marker):
                base.action = "deduped_existing_resolved_thread"
            elif dry_run:
                base.action = "would_reply_to_resolved_thread"
            else:
                try:
                    github.post_reply(
                        gh,
                        repo,
                        number,
                        int(match.id),
                        f"{marker}\n⚠️ Reusing this existing thread because the same finding still applies on `{head_sha[:8]}`. If it has already been addressed, please mark the thread resolved.",
                    )
                    base.action = "replied_to_resolved_thread"
                except (ReviewError, ValueError):
                    base.action = "failed_to_reply_to_resolved_thread"
        sup.append(base)
    return kept, sup


def build(
    gh: Gh,
    *,
    repo: str,
    number: int,
    head_sha: str,
    phase: str,
    verified: VerifierOutput,
    provenance: Provenance,
    bot_login: str,
    diff_text: str | None,
    dry_run: bool,
    superseded_note: str | None = None,
    rereview: bool = False,
    fresh: Callable[[Finding, ExistingComment], bool] | None = None,
    settled_hashes: set[str] | None = None,
) -> ReviewModel:
    findings, collapsed = collapse_same_root(list(verified.findings))
    kept, sup = dedupe_against_github(
        gh,
        repo,
        number,
        head_sha,
        findings,
        bot_login,
        dry_run=dry_run,
        fresh=fresh,
        settled_hashes=settled_hashes,
    )
    suppressed = list(collapsed) + list(sup)
    if has_same_root_duplicates(kept):
        raise ReviewError(
            FailKind.CONTRACT, "same-root duplicates survived collapse; refusing to publish"
        )
    body_only = diff_text is None
    comments: list[dict[str, Any]] = []
    skipped: list[Finding] = []
    if body_only:
        skipped = list(findings)
    else:
        parsed = parse_diff(diff_text or "")
        for f in kept:
            c = map_comment(f, parsed)
            if c:
                comments.append(c)
            else:
                skipped.append(f)
    return ReviewModel(
        repo=repo,
        number=number,
        head_sha=head_sha,
        phase=phase,
        verified=verified,
        provenance=provenance,
        kept=kept,
        skipped=skipped,
        suppressed=suppressed,
        comments=comments,
        body_only=body_only,
        superseded_note=superseded_note,
        rereview=rereview,
    )


def publish(
    gh: Gh,
    m: ReviewModel,
    *,
    bot_login: str,
    dry_run: bool,
    force_comment: bool = False,
) -> PublishResult:
    body = render(m)
    event = m.canonical_event
    transport = "COMMENT" if force_comment else event
    if not force_comment and event in {"APPROVE", "REQUEST_CHANGES"}:
        author = github.pr_meta(gh, m.repo, m.number).author
        if author.lower() == bot_login.lower():
            transport = "COMMENT"
            body += f"\n\n_Canonical verifier result: `{event}`. GitHub does not allow authors to approve or request changes on their own pull requests, so this review was submitted using `COMMENT` transport. The findings and blocker status above are unchanged._"
    elif force_comment and event != "COMMENT":
        body += f"\n\n_Canonical verifier result: `{event}`; this re-review summary is informational and does not change the standing review state._"
    payload = {"commit_id": m.head_sha, "body": body, "event": transport, "comments": m.comments}
    if dry_run:
        return PublishResult(
            posted=False,
            event=event,
            transport_event=transport,
            body=body,
            comments=m.comments,
            suppressed=m.suppressed,
            skipped_reason="dry_run",
        )
    resp = github.post_review(gh, m.repo, m.number, payload)
    return PublishResult(
        posted=True,
        event=event,
        transport_event=transport,
        body=body,
        review_id=int(resp["id"]),
        review_url=str(resp.get("html_url") or ""),
        comments=m.comments,
        suppressed=m.suppressed,
    )


def post_coderabbit_reactions(
    gh: Gh, repo: str, number: int, reactions: list[dict[str, Any]], bot_login: str
) -> list[dict[str, Any]]:
    content = {"agree": "+1", "disagree": "-1", "extend": "+1"}
    replied: set[int] | None = None
    results = []
    for r in reactions:
        cid, action = int(r["comment_id"]), str(r["action"])
        item: dict[str, Any] = {"comment_id": cid, "action": action, "ok": True}
        try:
            github.react(gh, repo, cid, content[action])
        except ReviewError as exc:
            item.update(ok=False, error=str(exc))
            results.append(item)
            continue
        reply = str(r.get("reply") or "").strip()
        if action in {"disagree", "extend"} and reply:
            if replied is None:
                replied = {
                    int(c["in_reply_to_id"])
                    for c in github.inline_comments(gh, repo, number)
                    if (c.get("user") or {}).get("login") == bot_login and c.get("in_reply_to_id")
                }
            if cid not in replied:
                try:
                    github.post_reply(gh, repo, number, cid, reply)
                    replied.add(cid)
                    item["replied"] = True
                except ReviewError as exc:
                    item.update(ok=False, error=str(exc))
        results.append(item)
    return results


# ---- answering human replies on prior finding threads ----

THREAD_ANSWER_MARKER = (
    "<!-- thepastaclaw-thread-answer v1 sha={sha} reply={reply} finding={finding} -->"
)
_REASON_MAX = 1200
_ANSWER_LEAD = {
    "WITHDRAWN": "Withdrawn",
    "FIXED": "Resolved",
    "OUTDATED": "No longer applies",
    "INTENTIONALLY_DEFERRED": "Deferred",
    "STILL_VALID": "Still applies",
}
# leads only a full review's answers use (the policy v12 thread lane never states these)
_REVIEW_ANSWER_LEAD = {
    "DISPUTED": "Disputed",
    "DEFERRAL_NEEDS_TRACKING": "Needs a tracking issue",
}
_RESOLVING = {"WITHDRAWN", "FIXED", "OUTDATED"}
# status -> the quiet trailing note render_conversation_answer appends (absent: no note)
_CONVERSATION_NOTE = {
    "WITHDRAWN": "\n\n_Withdrawing this finding._",
    "DISPUTED": (
        "\n\n_Disputed: left to the maintainers; this finding no longer blocks this pull request._"
    ),
    "FIX_PENDING": (
        "\n\n_The finding stands on this commit until that change is pushed; the push gets a "
        "full review._"
    ),
    "INTENTIONALLY_DEFERRED": (
        "\n\n_Deferred on a maintainer's call: this finding no longer blocks this pull request._"
    ),
    "DEFERRAL_PENDING": (
        "\n\n_Deferring this needs a maintainer's call; until then it still blocks this pull "
        "request._"
    ),
    "DEFERRAL_NEEDS_TRACKING": (
        "\n\n_Not deferred yet: a deferral needs a tracking issue for the follow-up; until "
        "then the finding stands._"
    ),
}


# A lifted finding whose thread stays open (the bot cannot resolve threads without write
# access, and a deferral is never resolved) still reads as an open 🔴 to anyone skimming the
# PR. The only thing the bot can change there is its own root comment: this line goes between
# the finding marker and the bold headline, which `github._finding_title` keeps parsing (a
# quoted line is not a headline; dedupe strips it). One line per thread, replaced on change.
_THREAD_STATUS_LINE = {
    "WITHDRAWN": "✅ **Withdrawn**",
    "FIXED": "✅ **Resolved**",
    "OUTDATED": "✅ **No longer applies**",
    "INTENTIONALLY_DEFERRED": "⏭️ **Deferred** (maintainer decision; not blocking)",
    "DISPUTED": "⚖️ **Disputed** (left to the maintainers; not blocking)",
    # not a lift: the finding stands, but its root says what the deferral is waiting for
    "DEFERRAL_NEEDS_TRACKING": "⏸️ **Needs a tracking issue** (not yet deferred)",
}


def with_thread_status(
    body: str, *, status: str | None, head_sha: str, tracking_url: str = ""
) -> str:
    """`body` (a bot finding's root comment) with its status line set for `status`, or removed
    when `status` is None or has no line. A deferral's line names its tracking issue."""
    body = THREAD_STATUS_RE.sub("", body, count=1)
    line = _THREAD_STATUS_LINE.get(status or "")
    if not line:
        return body
    if status == "INTENTIONALLY_DEFERRED" and tracking_url:
        line = line.replace("not blocking)", f"not blocking; tracked in {tracking_url})")
    lines = body.split("\n")
    i = 0
    while i < len(lines) and lines[i].lstrip().startswith("<!--"):
        i += 1
    block = [THREAD_STATUS_MARKER, f"> {line} at `{head_sha[:8]}`; see the replies below.", ""]
    return "\n".join(lines[:i] + block + lines[i:])


def _set_thread_status(
    gh: Gh,
    repo: str,
    *,
    comment_id: Any,
    root_body: str,
    status: str | None,
    head_sha: str,
    tracking_url: str = "",
) -> bool | str:
    """Edit our own finding root to show `status`. True when the root now shows it (or already
    did), else the failure; only ever touches a comment carrying our finding marker."""
    if not FINDING_MARKER_RE.search(root_body):
        return "failed: root carries no finding marker"
    new = with_thread_status(root_body, status=status, head_sha=head_sha, tracking_url=tracking_url)
    if new == root_body:
        return True
    try:
        github.edit_review_comment(gh, repo, int(comment_id), new)
    except (ReviewError, ValueError) as exc:
        return f"failed: {exc}"
    return True


def scrub_reply(text: str, limit: int) -> str:
    """Bound and defuse model-written text before it reaches GitHub: no CodeRabbit retriggers,
    no echoed bot markers (they would corrupt the once-per-reply bookkeeping), no live
    @-mentions (a bot must never page people). Truncation happens before defusing so a
    mention is never cut in half, and a code fence left open by the cut is closed."""
    low = text.lower()
    if "@coderabbitai" in low or "<!-- thepastaclaw" in low:
        return ""
    text = text[:limit].rstrip()
    if text.count("```") % 2:
        text += "\n```"
    return text.replace("@", "@\u200b")


def _lead(status: str) -> str:
    return _ANSWER_LEAD.get(status) or _REVIEW_ANSWER_LEAD.get(status, status)


def render_thread_answer(*, marker: str, status: str, head_sha: str, reply: str) -> str:
    return f"{marker}\n**{_lead(status)}** (re-reviewed at `{head_sha[:8]}`): {reply}"


def render_conversation_answer(
    *, marker: str, status: str, reply: str, tracking_url: str = "", filed: bool = False
) -> str:
    """A conversational answer: the model's prose, with the status as a quiet trailing note
    rather than a bold lead, so the thread reads like a discussion, not a verdict log. A
    deferral's note links the issue that tracks the follow-up (`filed`: the bot just filed
    it on a maintainer's request)."""
    note = _CONVERSATION_NOTE.get(status, "")
    if status == "INTENTIONALLY_DEFERRED" and tracking_url:
        lead = f"Filed {tracking_url} for the follow-up" if filed else f"Tracked in {tracking_url}"
        note = (
            f"\n\n_{lead}. Deferred on a maintainer's call: this finding no longer blocks this "
            "pull request._"
        )
    return f"{marker}\n{reply}{note}"


def _deliver_answer(
    gh: Gh,
    repo: str,
    number: int,
    *,
    comment_id: Any,
    body: str,
    thread_id: Any,
    resolve: bool,
    item: dict[str, Any],
    root_body: str,
    root_status: str | None,
    head_sha: str,
    tracking_url: str = "",
) -> None:
    """Post one answer on a finding thread and, when the outcome retires the finding, resolve
    the thread. Records what happened on `item` (`action`, and `resolved` when we tried).

    `root_status` is the outcome when it lifts the finding (or says what a deferral waits
    for). If the thread then stays open, the root comment gets that status line (`marked`);
    any other answer clears a stale one."""
    try:
        github.post_reply(gh, repo, number, int(comment_id), body)
        item["action"] = "replied"
    except (ReviewError, ValueError) as exc:
        item["action"] = f"reply_failed: {exc}"
        return
    if resolve and thread_id:
        try:
            github.resolve_thread(gh, str(thread_id))
            item["resolved"] = True
        except ReviewError as exc:
            item["resolved"] = f"failed: {exc}"
    if not root_body:
        return
    status = None if item.get("resolved") is True else root_status
    if status or THREAD_STATUS_MARKER in root_body:
        item["marked"] = _set_thread_status(
            gh,
            repo,
            comment_id=comment_id,
            root_body=root_body,
            status=status,
            head_sha=head_sha,
            tracking_url=tracking_url,
        )


def answer_conversation(
    gh: Gh,
    repo: str,
    number: int,
    head_sha: str,
    *,
    threads: dict[str, dict[str, Any]],
    outcomes: dict[str, Any],
    settled: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Post the conversation lane's replies: one per replied thread with something to say,
    exactly once per (human reply, head), resolving threads the model conceded. `outcomes` maps finding_hash -> ThreadOutcome. Returns what was done.

    `settled` (hash -> outcome) are threads an earlier answer already lifted on this commit:
    the lift stands whatever this answer says, so their root keeps its status line (a
    restated deferral without a tracking link would otherwise drop the link it shows)."""
    posted = [str(c.get("body") or "") for c in github.inline_comments(gh, repo, number)]
    done: list[dict[str, Any]] = []
    seen_roots: set[tuple[str, str]] = set()  # same alias guard as answer_replied_threads
    for h, t in threads.items():
        cid = t.get("comment_id")
        o = outcomes.get(h)
        if not cid or o is None:
            continue
        item: dict[str, Any] = {
            "finding_hash": h,
            "status": o.status,
            "comment_id": cid,
            "mode": "conversation",
        }
        root_key = (str(t.get("thread_id") or ""), str(cid))
        if root_key in seen_roots:
            item.update(status="DUPLICATE_THREAD", action="duplicate_thread_skipped")
            done.append(item)
            continue
        seen_roots.add(root_key)
        marker = THREAD_ANSWER_MARKER.format(
            sha=head_sha, reply=t.get("latest_reply_id"), finding=h
        )
        if any(marker in b for b in posted):
            item["action"] = "already_answered"
            done.append(item)
            continue
        if o.status == "NO_REPLY":
            item["action"] = "no_reply"
            done.append(item)
            continue
        reply = scrub_reply(o.reply, 2500)
        if not reply:
            item["action"] = "suppressed_reply"
            done.append(item)
            continue
        if o.tracking_url:
            item["tracking"] = o.tracking_url
        # a settled thread's root keeps the line its lift put there unless this answer lifts
        # it again (a re-deferral by a maintainer that names its tracking refreshes the line)
        relift = o.status in _RESOLVING or (
            o.status == "INTENTIONALLY_DEFERRED" and bool(o.tracking_url)
        )
        _deliver_answer(
            gh,
            repo,
            number,
            comment_id=cid,
            body=render_conversation_answer(
                marker=marker,
                # a settled finding stays lifted: no note claiming it stands
                status=""
                if h in (settled or {}) and o.status == "DEFERRAL_NEEDS_TRACKING"
                else o.status,
                reply=reply,
                tracking_url=o.tracking_url,
                filed=o.tracking_filed,
            ),
            thread_id=t.get("thread_id"),
            resolve=o.status in _RESOLVING,
            item=item,
            root_body="" if h in (settled or {}) and not relift else str(t.get("body") or ""),
            root_status=o.status if o.status in _THREAD_STATUS_LINE else None,
            head_sha=head_sha,
            tracking_url=o.tracking_url,
        )
        done.append(item)
    return done


def render_thread_lane_answer(
    *,
    marker: str,
    status: str,
    head_sha: str,
    reply: str,
    same_sha: bool,
    tracking_url: str = "",
) -> str:
    """A policy v12 thread-lane answer: conversational on a same-sha reply (a deferral's note
    links its tracking issue); on a new push a settled status leads (as v11's re-review
    answers did), the pending ones end in a note."""
    if not same_sha and status in _ANSWER_LEAD:
        return render_thread_answer(marker=marker, status=status, head_sha=head_sha, reply=reply)
    return render_conversation_answer(
        marker=marker, status=status, reply=reply, tracking_url=tracking_url
    )


def shown_status(body: str) -> str:
    """The outcome a posted thread answer shows: its bold lead (`render_thread_answer`), else
    its trailing note (`render_conversation_answer`), else STILL_VALID (the one outcome a
    conversational answer states only in prose)."""
    for status, lead in _ANSWER_LEAD.items():
        if re.search(rf"\*\*{re.escape(lead)}\*\* \(re-reviewed at", body):
            return status
    for status, note in _CONVERSATION_NOTE.items():
        if body.rstrip().endswith(note.strip()):
            return status
    return "STILL_VALID"


def answer_thread_lane(
    gh: Gh,
    repo: str,
    number: int,
    head_sha: str,
    *,
    outcomes: dict[str, Any],
    threads: dict[str, dict[str, Any]],
    same_sha: bool,
    keep: set[str] | frozenset[str] = frozenset(),
    settled: set[str] | frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Post the policy v12 thread lane's outcomes (`outcomes`: finding_hash -> ThreadOutcome):
    exactly once per (human reply, head) on each open thread, resolving the thread when the
    outcome retires the issue and marking its root comment where it stays open. An issue with
    no open thread (`threads`: the PR's unresolved bot finding threads) gets no reply: its
    outcome stands as is (`no_thread`). `keep`: issues this round restated, whose root must
    not go on showing an earlier head's "✅ Withdrawn". `settled`: issues already deferred;
    a deferral restated there without a link leaves the root's line (and link) as it is. Returns what was done, one item per outcome (the
    `thread.answered` events and ledger outcomes are recorded from these)."""
    posted = [str(c.get("body") or "") for c in github.inline_comments(gh, repo, number)]
    done: list[dict[str, Any]] = []
    seen_roots: set[tuple[str, str]] = set()
    mode = "conversation" if same_sha else "thread"
    for h, o in outcomes.items():
        item: dict[str, Any] = {"finding_hash": h, "status": o.status, "mode": mode}
        if o.tracking_url:
            item["tracking"] = o.tracking_url
        t = threads.get(h)
        if t is None or not t.get("comment_id"):
            item["action"] = "no_thread"
            done.append(item)
            continue
        cid = t["comment_id"]
        item["comment_id"] = cid
        root_key = (str(t.get("thread_id") or ""), str(cid))
        if root_key in seen_roots:
            item.update(status="DUPLICATE_THREAD", action="duplicate_thread_skipped")
            done.append(item)
            continue
        seen_roots.add(root_key)
        marker = THREAD_ANSWER_MARKER.format(
            sha=head_sha, reply=t.get("latest_reply_id"), finding=h
        )
        earlier = next((b for b in posted if marker in b), None)
        if earlier is not None:
            # answered on this head before (a retried run): what counts is what the thread
            # shows, which may not be what this attempt decided
            shown = shown_status(earlier)
            if shown != o.status:
                item.update(status=shown, decided=o.status)
            item["action"] = "already_answered"
            done.append(item)
            continue
        if not t.get("awaiting_answer") and _has_answer_for_status(t, o.status):
            item["action"] = "already_answered"
            done.append(item)
            continue
        if not o.reply:
            item["action"] = "no_reply"
            done.append(item)
            continue
        reply = scrub_reply(o.reply, 2500)
        if not reply:
            item["action"] = "suppressed_reply"
            done.append(item)
            continue
        # a deferral restated without its link keeps the line (and link) the root shows; any
        # other answer rewrites or clears it (the lane may re-open a deferred issue)
        keep_root = h in settled and o.status == "INTENTIONALLY_DEFERRED" and not o.tracking_url
        _deliver_answer(
            gh,
            repo,
            number,
            comment_id=cid,
            body=render_thread_lane_answer(
                marker=marker,
                status=o.status,
                head_sha=head_sha,
                reply=reply,
                same_sha=same_sha,
                tracking_url=o.tracking_url,
            ),
            thread_id=t.get("thread_id"),
            resolve=o.status in _RESOLVING,
            item=item,
            root_body="" if keep_root else str(t.get("body") or ""),
            root_status=o.status if o.status in _THREAD_STATUS_LINE else None,
            head_sha=head_sha,
            tracking_url=o.tracking_url,
        )
        done.append(item)
    for h in sorted(keep):
        t = threads.get(h)
        body = str((t or {}).get("body") or "")
        if t is None or h in outcomes or THREAD_STATUS_MARKER not in body:
            continue
        done.append(
            {
                "finding_hash": h,
                "status": "STILL_VALID",
                "comment_id": t.get("comment_id"),
                "mode": mode,
                "action": "status_cleared",
                "marked": _set_thread_status(
                    gh,
                    repo,
                    comment_id=t.get("comment_id"),
                    root_body=body,
                    status=None,
                    head_sha=head_sha,
                ),
            }
        )
    return done


def _answer_outcome(
    finding_hash: str, row: dict[str, Any], kept_hashes: set[str]
) -> tuple[str, str, bool]:
    """(status, reason, explicit) to post on one replied finding thread.

    The verifier's kept set is canonical: if the reviewer kept a finding the verifier dropped
    (or nobody reconciled it), the outcome is that the finding does not stand on this head.
    `explicit` is False when that outcome was defaulted rather than stated by a reviewer; the
    caller answers but never resolves the thread on a default. A reviewer-written reason only
    survives when it explains the status we actually publish, and is bounded and scrubbed of
    bot retriggers before it reaches GitHub.
    """
    stated = str(row.get("status") or "")
    if row.get("needs_tracking"):  # a deferral held standing until it is tracked
        reason = scrub_reply(str(row.get("reason") or "").strip(), _REASON_MAX)
        return "DEFERRAL_NEEDS_TRACKING", reason or "This deferral needs a tracking issue.", True
    if finding_hash in kept_hashes:
        status, explicit = "STILL_VALID", True
    elif stated and stated != "STILL_VALID":
        status, explicit = stated, True
    else:
        status, explicit = "WITHDRAWN", False
    reason = ""
    if stated == status:
        reason = scrub_reply(str(row.get("reason") or "").strip(), _REASON_MAX)
    if not reason and status == "STILL_VALID":
        reason = "The verifier kept this finding on the current head; see the updated review."
    elif not reason:
        reason = "This finding did not survive verification on the current head."
    return status, reason, explicit


def _has_answer_for_status(thread: dict[str, Any], status: str) -> bool:
    """Whether this finding already has a bot answer with the same outcome.

    Open-thread reconciliation runs on every new head.  The answer marker is intentionally
    head-specific for human replies, but an open thread with no newer human reply must not get
    the same status comment on every head merely because GitHub failed to show it as resolved.
    """
    lead = _lead(status)
    # a conversation's answer states these two in its note, not a bold lead
    note = _CONVERSATION_NOTE.get(status, "").strip() if status in _REVIEW_ANSWER_LEAD else ""
    for answer in thread.get("bot_answers") or []:
        body = str(answer.get("body") or "")
        if "<!-- thepastaclaw-thread-answer v1" not in body:
            continue
        if re.search(rf"\*\*{re.escape(lead)}\*\*", body) or (note and note in body):
            return True
    return False


def answer_replied_threads(
    gh: Gh,
    repo: str,
    number: int,
    head_sha: str,
    *,
    threads: dict[str, dict[str, Any]],
    reconciliation: dict[str, dict[str, Any]],
    verified: VerifierOutput,
    open_threads: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Reply once per (human reply, head) on each replied prior finding thread with the
    reconciliation outcome, and resolve threads whose finding a reviewer explicitly withdrew,
    fixed or marked outdated. Other open bot threads (`open_threads`, nobody replied) get the
    same note and resolution when a reviewer explicitly withdrew their finding, so a dropped
    finding never lingers as an open comment; likewise, once, a finding settled as DISPUTED,
    a deferral held for want of tracking (`needs_tracking`: the ask to link or have the bot
    file it) and one the orchestrator settled without a reply to answer (`announce`). A lift
    that leaves the thread open, or a held deferral, is also shown on the thread's root
    comment. Returns what was done."""
    # a verifier that echoes the carried finding without its finding_hash must not turn a
    # kept finding into "withdrawn": fall back to the finding's own hash as well
    kept_hashes = {f.prior_hash for f in verified.findings if f.prior_hash} | {
        f.hash for f in verified.findings
    }
    posted = [str(c.get("body") or "") for c in github.inline_comments(gh, repo, number)]
    done: list[dict[str, Any]] = []
    todo = dict(threads)
    for h, t in (open_threads or {}).items():
        row = reconciliation.get(h) or {}
        if h in todo or (h in kept_hashes and not row.get("needs_tracking")):
            continue
        stated = str(row.get("status") or "")
        if (
            stated in {"WITHDRAWN", "FIXED", "OUTDATED", "DISPUTED"}
            or row.get("needs_tracking")
            or row.get("announce")
        ):
            todo[h] = {**t, "latest_reply_id": None}
    # A malformed/legacy thread index can contain the same GitHub thread under more than
    # one finding identity (for example after a marker migration).  Replies are addressed to
    # the thread root, so posting once for each mapping would create duplicate answers on the
    # same commit.  Keep the first deterministic entry and report the aliases as suppressed.
    seen_roots: set[tuple[str, str]] = set()
    for h, t in todo.items():
        cid = t.get("comment_id")
        if not cid:
            continue
        root_key = (str(t.get("thread_id") or ""), str(cid))
        if root_key in seen_roots:
            done.append(
                {
                    "finding_hash": h,
                    "status": "DUPLICATE_THREAD",
                    "comment_id": cid,
                    "action": "duplicate_thread_skipped",
                }
            )
            continue
        seen_roots.add(root_key)
        marker = THREAD_ANSWER_MARKER.format(
            sha=head_sha, reply=t.get("latest_reply_id"), finding=h
        )
        status, reason, explicit = _answer_outcome(h, reconciliation.get(h) or {}, kept_hashes)
        item: dict[str, Any] = {"finding_hash": h, "status": status, "comment_id": cid}
        # For an open thread with no newer human reply, a prior answer with the same outcome is
        # already sufficient.  Do not retry the resolve mutation either: GitHub may leave the
        # thread open when the mutation is rejected or races, and repeating it on every head is
        # noisy and does not change the review state.
        if not t.get("awaiting_answer") and _has_answer_for_status(t, status):
            item["action"] = "already_answered"
            done.append(item)
            continue
        if any(marker in b for b in posted):
            item["action"] = "already_answered"
            done.append(item)
            continue
        resolve = explicit and status in _RESOLVING
        marks = resolve or (status in _THREAD_STATUS_LINE and status not in _RESOLVING)
        _deliver_answer(
            gh,
            repo,
            number,
            comment_id=cid,
            body=render_thread_answer(
                marker=marker, status=status, head_sha=head_sha, reply=reason
            ),
            thread_id=t.get("thread_id"),
            resolve=resolve,
            item=item,
            root_body=str(t.get("body") or ""),
            root_status=status if marks else None,
            head_sha=head_sha,
        )
        done.append(item)
    # a finding this review keeps again must not go on reading "✅ Withdrawn" from an earlier
    # head just because nobody replied on its thread this time
    for h, t in (open_threads or {}).items():
        body = str(t.get("body") or "")
        if h in kept_hashes and h not in todo and THREAD_STATUS_MARKER in body:
            item = {"finding_hash": h, "status": "STILL_VALID", "comment_id": t.get("comment_id")}
            item["action"] = "status_cleared"
            item["marked"] = _set_thread_status(
                gh,
                repo,
                comment_id=t.get("comment_id"),
                root_body=body,
                status=None,
                head_sha=head_sha,
            )
            done.append(item)
    return done


# ---- the standing review after discussion ----

# A finding the discussion lifted while other blockers stand moves no verdict, so no follow-up
# review is posted, and the standing review kept saying "🔴 2 blocking" and listing the
# deferred finding as [BLOCKING] in its agent prompt (dashpay/dash#7892). The bot can edit its
# own review body: a status block under the title, struck-through counts, and a retagged prompt
# entry, all derived from the original text so a re-edit is idempotent and reversible.
REVIEW_STATUS_MARKER = "<!-- thepastaclaw-review-status v1 -->"
_REVIEW_STATUS_RE = re.compile(re.escape(REVIEW_STATUS_MARKER) + r"\n(?:>[^\n]*\n)*\n?")
_COUNT_LABELS = {"blocking": "blocking", "suggestion(s)": "suggestion", "nitpick(s)": "nitpick"}
_COUNT_SEGMENT = r"(\S+) (?:~~(\d+)~~ )?(\d+) (blocking|suggestion\(s\)|nitpick\(s\))"
_COUNT_SEGMENT_RE = re.compile(_COUNT_SEGMENT)
_COUNTS_LINE_RE = re.compile(rf"^{_COUNT_SEGMENT}(?: \| {_COUNT_SEGMENT})*$")
_PROMPT_ITEM_RE = re.compile(r"^- \[([A-Z]+)(?: → [^\]]*)?\] (.*)$")
# a `_carried_line`, less the settled note `with_lifted_findings` may have added
_CARRIED_ITEM_RE = re.compile(
    r"^(- \S+ \*\*\w+\*\*: (.*) — `([^`]*)`.*?)(?: — \*\*settled:\*\* .*)?$"
)
# status -> (status-block label, agent-prompt tag)
_LIFTED_LABEL = {
    "WITHDRAWN": ("✅ Withdrawn", "WITHDRAWN in discussion; do not fix"),
    "INTENTIONALLY_DEFERRED": (
        "⏭️ Deferred (maintainer decision)",
        "DEFERRED by a maintainer; not blocking, do not fix in this PR",
    ),
    "DISPUTED": (
        "⚖️ Disputed (left to the maintainers)",
        "DISPUTED; left to the maintainers, do not fix",
    ),
}
_LIFTED_DEFAULT = ("Settled in discussion", "SETTLED in discussion; do not fix")


@dataclass(frozen=True, slots=True)
class LiftedFinding:
    hash: str
    severity: str
    file: str
    title: str
    status: str  # the conversation outcome that lifted it ("" when not on record)
    tracking: str = ""  # a deferral's tracking issue, when on record


def _recount(line: str, lifted: list[LiftedFinding]) -> tuple[str, int | None]:
    """The counts line with `lifted` struck off the original numbers, and the blocking count
    it now shows (None when the line has no blocking segment)."""
    by_sev: dict[str, int] = {}
    for f in lifted:
        by_sev[f.severity] = by_sev.get(f.severity, 0) + 1
    blocking: int | None = None

    def seg(m: re.Match[str]) -> str:
        nonlocal blocking
        icon, orig, label = m.group(1), int(m.group(2) or m.group(3)), m.group(4)
        now = max(orig - by_sev.get(_COUNT_LABELS[label], 0), 0)
        if label == "blocking":
            blocking = now
        return f"{icon} ~~{orig}~~ {now} {label}" if now != orig else f"{icon} {orig} {label}"

    return _COUNT_SEGMENT_RE.sub(seg, line), blocking


def _block_text(title: str) -> str:
    """A title as one inert line: a newline would end the quoted status block early (and the
    next edit could not find its end), an HTML comment could fake a marker."""
    return " ".join(title.split()).replace("<!--", "&lt;!--")


def with_lifted_findings(body: str, lifted: list[LiftedFinding]) -> str:
    """`body` (the bot's standing review) annotated for the findings the discussion lifted:
    a status block under the title, the counts struck through, its line in the compact list of
    carried findings marked settled, and the agent-prompt entries retagged. Only a finding this review lists in its agent prompt (by file, location and
    exact title; each entry at most once) is annotated, so a finding the review never counted
    (a legacy thread) moves no number, and a same-titled finding elsewhere keeps its tag.
    With nothing lifted every annotation is removed again."""
    lines = _REVIEW_STATUS_RE.sub("", body, count=1).split("\n")
    todo = list(lifted)
    matched: list[LiftedFinding] = []
    path = ""
    for i, line in enumerate(lines):
        if line.startswith("In `") and line.endswith("`:"):
            path = line[4:-2]
            continue
        m = _PROMPT_ITEM_RE.match(line)
        if not m:
            continue
        sev, rest = m.group(1), m.group(2)
        hit = next(
            (
                f
                for f in todo
                if (f.file or "(no file)") == path
                and rest.endswith(f": {f.title}")
                and _is_location(rest[: -len(f.title) - 2], path)
            ),
            None,
        )
        tag = ""
        if hit:
            todo.remove(hit)
            matched.append(hit)
            tag = f" → {_LIFTED_LABEL.get(hit.status, _LIFTED_DEFAULT)[1]}"
        lines[i] = f"- [{sev}{tag}] {rest}"
    # the counts line sits between the summary and the first <details>; the last full-line
    # match there, so a summary sentence shaped like a count is never the one rewritten
    details = next((i for i, ln in enumerate(lines) if ln == "<details>"), len(lines))
    counts = [i for i in range(details) if _COUNTS_LINE_RE.match(lines[i])]
    left: int | None = None
    if counts:
        lines[counts[-1]], left = _recount(lines[counts[-1]], matched)
    for i in range(details):  # the compact list of findings carried from earlier rounds
        c = _CARRIED_ITEM_RE.match(lines[i])
        if not c:
            continue
        hit = next(
            (
                f
                for f in matched
                if f.title == c.group(2) and _is_location(c.group(3), f.file or "(no file)")
            ),
            None,
        )
        lines[i] = c.group(1) + (
            f" — **settled:** {_LIFTED_LABEL.get(hit.status, _LIFTED_DEFAULT)[0]}" if hit else ""
        )
    if not matched:
        return "\n".join(lines)
    if left is None:
        tail = "."
    elif left == 0:
        tail = "; no blocking finding remains."
    else:
        tail = f"; {left} blocking finding(s) still stand."
    block = [
        REVIEW_STATUS_MARKER,
        f"> **Updated after discussion:** {len(matched)} finding(s) of this review were settled "
        f"in the inline threads{tail}",
        *(
            f"> - {_LIFTED_LABEL.get(f.status, _LIFTED_DEFAULT)[0]}: "
            f"{SEVERITY_ICONS.get(f.severity, '')} {_block_text(f.title)}"
            + (f" (tracked in {_block_text(f.tracking)})" if f.tracking else "")
            for f in matched
        ),
        "",
    ]
    title = next((i for i, ln in enumerate(lines) if ln.startswith("## ")), None)
    if title is None:  # no heading: after the leading markers
        title = next((i for i, ln in enumerate(lines) if not ln.startswith("<!--")), len(lines)) - 1
    at = title + 1
    if at < len(lines) and not lines[at].strip():
        at += 1
    return "\n".join(lines[:at] + block + lines[at:])


def _is_location(loc: str, path: str) -> bool:
    """`loc` is what `_location` renders for a finding in `path`."""
    return loc == path or (
        loc.startswith(f"{path}:") and loc[len(path) + 1 :].replace("-", "").isdigit()
    )


def tracking_issue_body(
    body: str,
    *,
    repo: str,
    number: int,
    comment_url: str,
    asker: str,
    severity: str,
    title: str,
    finding: str,
) -> str:
    """The body of a tracking issue the bot files for a deferred finding: the lane's
    description of the follow-up, then where it came from (the PR, the thread, who asked) and
    a short quote of the finding. Mentions are defused: the bot never pages anyone, and posts
    as itself with no attribution footer."""
    quote = "\n".join(f"> {ln}" for ln in finding.strip()[:600].splitlines()) or "> (no text)"
    origin = [
        "### Origin",
        "",
        f"Deferred from https://github.com/{repo}/pull/{number} "
        f"(review thread: {comment_url}) at the request of maintainer `{asker}`.",
        "",
        f"PastaClaw finding ({severity}): **{_block_text(title)}**",
        "",
        quote,
    ]
    described = scrub_reply(body, 4000)
    return (described + "\n\n" if described else "") + "\n".join(origin).replace("@", "@\u200b")


# ---- same-sha verdict update ----

UPDATE_MARKER = "<!-- thepastaclaw-review-update v1 phase={phase} sha={sha} -->"


def render_verdict_update(
    *,
    head_sha: str,
    phase: str,
    verified: VerifierOutput,
    provenance: Provenance,
    previous_event: str,
    new_event: str,
    lifted_blockers: list[str],
    note: str = "",
) -> str:
    n = verified.blocker_count
    summary = "\n".join(
        line for line in verified.summary.splitlines() if not _SUMMARY_SOURCE_LINE_RE.match(line)
    ).strip()
    title = f"Re-review after discussion — commit {head_sha[:8]}"
    if provenance.degraded:
        title = f"{DEGRADED_BADGE} — {title}"
    parts = [
        UPDATE_MARKER.format(phase=phase, sha=head_sha),
        f"## {title}",
        "",
    ]
    if provenance.degraded:
        parts += [degraded_banner(provenance.degraded), ""]
    parts += [
        f"Standing review was `{previous_event}`; this re-review is `{new_event}`: "
        + ("no blocking findings remain." if not n else f"{n} blocking finding(s) now stand."),
        "",
    ]
    if note:
        parts += [note, ""]
    if not provenance.conversation:  # it re-reviewed nothing, so it has nothing to count
        parts += [_counts_line([*verified.findings, *verified.carried], include_zero=True), ""]
    if lifted_blockers:
        parts += ["No longer blocking:", *(f"- {t}" for t in lifted_blockers), ""]
    if summary:
        parts += [summary, ""]
    if verified.followups:
        parts += [*_followups_block(verified.followups)[1:], ""]
    parts += [
        "_Same commit as the standing review; the inline threads above carry the per-finding "
        "outcome. This follow-up exists only to correct the verdict._",
        "",
        "<details>",
        "<summary>Review provenance</summary>",
        "",
        _source_line(provenance),
        "",
        *_provenance_lines(provenance, phase),
        "",
        "</details>",
    ]
    prompt = _ai_prompt(list(verified.findings), [])
    if prompt:
        fence = _fence(prompt)
        parts += [
            "",
            "<details>",
            "<summary>🤖 Prompt for all review comments with AI agents</summary>",
            "",
            fence,
            prompt,
            fence,
            "</details>",
        ]
    return "\n".join(parts)


def standing_verdict(
    reviews: list[dict[str, Any]], head_sha: str, phase: str, bot_login: str
) -> str | None:
    """The bot's most recent review state for this (sha, phase), counting follow-up updates, or
    None when the bot has not reviewed that head/phase. Dismissed reviews count as dismissed."""
    marker = f"phase={phase} sha={head_sha}"
    latest: str | None = None
    for r in reviews:
        if (r.get("user") or {}).get("login") != bot_login:
            continue
        if marker in str(r.get("body") or ""):
            latest = str(r.get("state") or "")
    return latest


def verdict_event(verified: VerifierOutput, provenance: Provenance) -> str:
    """Blockers always REQUEST_CHANGES. APPROVE only when the full-strength pipeline ran:
    a Phase-1-only verdict (the second-round reviewers did not run) and a degraded-mode
    verdict (stand-in models) both stop at COMMENT."""
    if verified.blocker_count:
        return "REQUEST_CHANGES"
    if (
        verified.review_action == "APPROVE"
        and not provenance.phase2_skipped
        and not provenance.degraded
    ):
        return "APPROVE"
    return "COMMENT"


def transport_event(event: str, *, own_pr: bool) -> str:
    """GitHub refuses APPROVE/REQUEST_CHANGES from the PR author, so a bot-authored PR gets
    COMMENT transport and the canonical verdict goes in the body."""
    return "COMMENT" if own_pr and event in {"APPROVE", "REQUEST_CHANGES"} else event


EVENT_STATE = {
    "REQUEST_CHANGES": "CHANGES_REQUESTED",
    "APPROVE": "APPROVED",
    "COMMENT": "COMMENTED",
}
STATE_EVENT = {v: k for k, v in EVENT_STATE.items()}


def canonical_event(value: str) -> str:
    """Normalise a `reviews.event` value: canonical events pass through, GitHub review states
    (as backfilled from an existing review) map back to the event that produced them."""
    return STATE_EVENT.get(value, value)


def publish_verdict_update(
    gh: Gh,
    *,
    repo: str,
    number: int,
    head_sha: str,
    phase: str,
    verified: VerifierOutput,
    provenance: Provenance,
    previous_event: str,
    lifted_blockers: list[str],
    bot_login: str,
    note: str = "",
    dry_run: bool = False,
) -> PublishResult:
    event = verdict_event(verified, provenance)
    body = render_verdict_update(
        head_sha=head_sha,
        phase=phase,
        verified=verified,
        provenance=provenance,
        previous_event=previous_event,
        new_event=event,
        lifted_blockers=lifted_blockers,
        note=note,
    )
    own = github.pr_meta(gh, repo, number).author.lower() == bot_login.lower()
    transport = transport_event(event, own_pr=own)
    if transport != event:
        body += f"\n\n_Canonical verifier result: `{event}`; submitted as `COMMENT` because GitHub does not allow authors to approve or request changes on their own pull requests._"
    if dry_run:
        return PublishResult(
            posted=False,
            event=event,
            transport_event=transport,
            body=body,
            skipped_reason="dry_run",
        )
    resp = github.post_review(
        gh, repo, number, {"commit_id": head_sha, "body": body, "event": transport, "comments": []}
    )
    return PublishResult(
        posted=True,
        event=event,
        transport_event=transport,
        body=body,
        review_id=int(resp["id"]),
        review_url=str(resp.get("html_url") or "") or None,
    )


# ---- CI after the verdict (ci.py) ----

# beside UPDATE_MARKER (which makes `standing_verdict` count the follow-up): says it is the
# head's CI update, of which there is at most one per commit
CI_UPDATE_MARKER = "<!-- thepastaclaw-ci-update v1 sha={sha} -->"
_CI_LEAD = {
    "yes": "checks that failed after the review look related to this change.",
    "unclear": "checks that failed after the review may relate to this change, and the "
    "evidence does not rule it out.",
}
_CI_REASON = {"yes": "Why they look related", "unclear": "Why it is unclear"}
_CI_REASON_MAX = 1200


def _ci_check_line(c: dict[str, Any]) -> str:
    name = str(c.get("name") or "?").replace("`", "")
    label = f"[`{name}`]({c['url']})" if c.get("url") else f"`{name}`"
    return f"- {label} ({c.get('conclusion') or 'failed'})"


def render_ci_update(
    *,
    head_sha: str,
    failing: list[dict[str, Any]],
    related: str,
    reason: str,
    lane: dict[str, Any],
    judged: bool,
    logs_read: int,
    degraded: dict[str, Any] | None = None,
) -> str:
    """The follow-up review that moves an APPROVE to COMMENT when CI failed after it.
    `judged` False: the CI-check lane gave no answer, so the failure counts as unclear."""
    title = f"CI failed after approval — commit {head_sha[:8]}"
    if degraded:
        title = f"{DEGRADED_BADGE} — {title}"
    why = (
        scrub_reply(reason, _CI_REASON_MAX)
        if judged
        else "the CI-check lane could not judge the failure, so it is treated as possibly related."
    )
    parts = [
        UPDATE_MARKER.format(phase="final", sha=head_sha),
        CI_UPDATE_MARKER.format(sha=head_sha),
        f"## {title}",
        "",
        "The standing review approved this commit; this follow-up moves the verdict to "
        f"`COMMENT`: {_CI_LEAD.get(related, _CI_LEAD['unclear'])}",
        "",
        "Failing checks:",
        *(_ci_check_line(c) for c in failing),
        "",
    ]
    if why:
        parts += [f"{_CI_REASON.get(related, _CI_REASON['unclear'])}: {why}", ""]
    parts += [
        "_Nothing was re-reviewed: the findings of the standing review are unchanged. The next "
        "push gets a full review; a re-run that passes does not restore the approval._",
        "",
        "<details>",
        "<summary>Review provenance</summary>",
        "",
        f"Source: CI-check lane {_model_label(lane)} (agent: `{lane['agent']}`); no reviewer "
        "or verifier lanes ran for this follow-up",
        "",
        *([_degraded_line(degraded)] if degraded else []),
        "- Verdict moved because CI failed on this commit after the review; failure logs read: "
        f"{logs_read} GitHub Actions job(s)",
        "",
        "</details>",
    ]
    return "\n".join(parts)


def publish_ci_update(
    gh: Gh,
    *,
    repo: str,
    number: int,
    head_sha: str,
    failing: list[dict[str, Any]],
    related: str,
    reason: str,
    lane: dict[str, Any],
    judged: bool,
    logs_read: int,
    degraded: dict[str, Any] | None = None,
) -> PublishResult:
    """Post the CI follow-up review: always COMMENT, so it needs no own-PR transport rule."""
    body = render_ci_update(
        head_sha=head_sha,
        failing=failing,
        related=related,
        reason=reason,
        lane=lane,
        judged=judged,
        logs_read=logs_read,
        degraded=degraded,
    )
    resp = github.post_review(
        gh, repo, number, {"commit_id": head_sha, "body": body, "event": "COMMENT", "comments": []}
    )
    return PublishResult(
        posted=True,
        event="COMMENT",
        transport_event="COMMENT",
        body=body,
        review_id=int(resp["id"]),
        review_url=str(resp.get("html_url") or "") or None,
    )
