"""Render and post the GitHub review. Port of review_poster.py + diff_position_mapper.py.

`render()` is pure: given a ReviewModel it returns the exact body and comments.
Network happens only in `publish()`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from textwrap import indent
from typing import Any

from . import github
from .contract import Finding, VerifierOutput
from .dedupe import (
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
                current = parts[1]
                files.setdefault(current, [])
            continue
        if line.startswith("+++ b/"):
            current = line[len("+++ b/") :]
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
    phase1_skipped: str | None = None  # set when the run went straight to Phase 2 (queue backlog)
    adhoc: bool = False  # repo has no skills entry: generic guidance, all specialists offered


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

    @property
    def canonical_event(self) -> str:
        if self.verified.blocker_count:
            return "REQUEST_CHANGES"
        if self.provenance.phase2_skipped:
            # a Phase-1-only verdict never approves: the second-round reviewers did not run
            return "COMMENT"
        return "APPROVE" if self.verified.review_action == "APPROVE" else "COMMENT"


def _location(f: Finding) -> str:
    if not f.line_start:
        return f.file or "(no file)"
    if f.line_start == f.line_end or not f.line_end:
        return f"{f.file}:{f.line_start}"
    return f"{f.file}:{f.line_start}-{f.line_end}"


def _ai_prompt(findings: list[Finding], suppressed: list[Suppressed]) -> str | None:
    entries: list[tuple[Finding, Suppressed | None]] = [(f, None) for f in findings]
    entries += [(s.finding, s) for s in suppressed if s.action != "deduped_same_batch_root"]
    if not entries:
        return None
    lines = [
        "These findings are from an automated code review. Verify each finding against the current code and only fix it if needed.",
        "",
    ]
    by_file: dict[str, list[tuple[Finding, Suppressed | None]]] = {}
    for f, s in entries:
        by_file.setdefault(f.file or "(no file)", []).append((f, s))
    for path, items in by_file.items():
        lines.append(f"In `{path}`:")
        for f, s in items:
            lines.append(f"- [{f.severity.upper()}] {_location(f)}: {f.title}")
            if s and s.html_url:
                lines.append(f"  (existing thread: {s.html_url})")
            if f.body:
                lines.append(indent(f.body, "  "))
        lines.append("")
    return "\n".join(lines).rstrip()


def _fence(content: str) -> str:
    longest = max((len(m.group(0)) for m in re.finditer(r"`+", content)), default=0)
    return "`" * max(3, longest + 1)


def _source_line(p: Provenance) -> str:
    parts = [
        f"reviewer {i}: `{r['model']}` (agent: `{r['agent']}`, role: `{r['role']}`)"
        for i, r in enumerate(p.reviewers, 1)
    ]
    parts.append(
        f"final verifier: `{p.verifier['model']}` (agent: `{p.verifier['agent']}`, role: `{p.verifier['role']}`)"
    )
    return "Source: " + "; ".join(parts)


def _triage_line(t: dict[str, Any]) -> str:
    if str(t.get("method", "")).startswith("llm:"):
        how = f"`{t['model']}` (effort {t['effort']})"
    else:
        how = f"fallback after triage failure ({t.get('error') or 'unknown'})"
    why = f" — {t['reasoning']}" if t.get("reasoning") else ""
    return f"- Triage: `{t['tier']}` by {how}{why}"


def _provenance_lines(p: Provenance, phase: str) -> list[str]:
    def fmt(r: dict[str, Any]) -> str:
        status = r["status"] + (f", effort {r['effort']}" if r.get("effort") else "")
        return f"`{r['model']}` — {r['role']} ({status}); agent `{r['agent']}`"

    p1 = [fmt(r) for r in p.reviewers if r.get("phase") == "phase1"]
    p2 = [fmt(r) for r in p.reviewers if r.get("phase") == "phase2"]
    lines = ["### Review provenance"]
    if p.adhoc:
        lines.append(
            "- Ad hoc review: this repository has no PastaClaw review skill; reviewers used "
            "generic guidance and the repository's own conventions"
        )
    if p.triage:
        lines.append(_triage_line(p.triage))
    if p.phase1_skipped:
        p1_line = f"- Phase 1 reviewers: **not run ({p.phase1_skipped})**"
    else:
        p1_line = "- Phase 1 reviewers: " + (", ".join(p1) if p1 else "provenance missing")
    lines += [
        p1_line,
        f"- Fresh verifier: `{p.verifier['model']}` — {p.verifier['role']}; agent `{p.verifier['agent']}`",
    ]
    if phase == "preliminary":
        lines.append("- Phase 2 reviewers: **not run (deferred by blocker gate)**")
    elif p.phase2_skipped:
        lines.append(
            f"- Phase 2 reviewers: **not run ({p.phase2_skipped})**; this review comments and never approves"
        )
    else:
        lines.append(
            "- Phase 2 reviewers: " + (", ".join(p2) if p2 else "no successful evidence recorded")
        )
    return lines


def render(m: ReviewModel) -> str:
    counts = {"blocking": 0, "suggestion": 0, "nitpick": 0}
    for f in m.skipped if m.body_only else m.kept:
        counts[f.severity if f.severity in counts else "nitpick"] += 1
    if m.phase == "preliminary":
        counts["blocking"] = m.verified.blocker_count
    if m.phase == "preliminary":
        title = "Preliminary review — Phase 1 blocker gate"
    elif m.provenance.phase2_skipped:
        tier = (m.provenance.triage or {}).get("tier", "trivial")
        title = f"Final review — Phase 1 only ({tier} change)"
    elif m.provenance.phase1_skipped:
        title = "Final validation — Phase 2 only (queue backlog)"
    else:
        title = "Final validation — Phase 1 + Phase 2"
    summary = "\n".join(
        line for line in m.verified.summary.splitlines() if not _SUMMARY_SOURCE_LINE_RE.match(line)
    ).strip()
    parts = [
        github.REVIEW_MARKER,
        f"<!-- thepastaclaw-review-phase v1 phase={m.phase} sha={m.head_sha} policy={m.provenance.policy_fingerprint[:16]} -->",
        f"## {title}",
        "",
        summary,
        "",
        _source_line(m.provenance),
        "",
    ]
    if m.phase == "preliminary":
        parts += [
            "Validated blockers were found by the Phase-1 review and confirmed by a fresh verifier. Phase 2 is deferred until a fresh same-head revalidation clears the blocker gate.",
            "",
        ]
    if m.superseded_note:
        parts += [m.superseded_note, ""]
    parts += _provenance_lines(m.provenance, m.phase)
    parts.append("")
    counts_line = " | ".join(
        f"{SEVERITY_ICONS[sev]} {counts[sev]} {label}"
        for sev, label in (
            ("blocking", "blocking"),
            ("suggestion", "suggestion(s)"),
            ("nitpick", "nitpick(s)"),
        )
        if counts[sev]
    )
    if counts_line:
        parts.append(counts_line)
    if m.skipped:
        if m.body_only:
            parts.append(
                f"\n_{len(m.skipped)} finding(s) omitted from inline comments because GitHub refused the PR diff as too large; listed below._"
            )
            parts += ["", f"### {len(m.skipped)} unmapped finding(s)", ""]
            for i, f in enumerate(m.skipped, 1):
                parts.append(
                    f"### {i}. [{f.severity}] {f.title}\n`{_location(f)}`\n\n{f.body or '_No details provided._'}"
                )
        else:
            parts.append(f"\n_{len(m.skipped)} additional finding(s) omitted (not in diff)._")
    if m.suppressed:
        parts.append(
            f"\n_{len(m.suppressed)} carried-forward finding(s) already raised on this PR; not re-posting as new inline comments._"
        )
    prompt = _ai_prompt(m.kept, m.suppressed)
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
    oos = m.verified.out_of_scope + [
        d
        for d in m.verified.dropped
        if any(k in str(d.get("reason", "")).lower() for k in ("out of scope", "outside"))
    ]
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


def dedupe_against_github(
    gh: Gh,
    repo: str,
    number: int,
    head_sha: str,
    findings: list[Finding],
    bot_login: str,
    *,
    dry_run: bool,
) -> tuple[list[Finding], list[Suppressed]]:
    records = [
        ExistingComment.from_github(c)
        for c in github.inline_comments(gh, repo, number)
        if (c.get("user") or {}).get("login") == bot_login
    ]
    if not records:
        return list(findings), []
    kept: list[Finding] = []
    sup: list[Suppressed] = []
    for f in findings:
        match, reason = find_duplicate(f, records)
        if not match:
            kept.append(f)
            continue
        thread = github.thread_for_comment(gh, match.node_id) or {}
        base = Suppressed(f, "", detail=reason or "", comment_id=match.id, html_url=match.html_url)
        if thread_has_resolution_reply(thread, bot_login):
            base.action = "deduped_maintainer_addressed"
        elif not thread.get("isResolved"):
            base.action = "deduped_existing_open_thread"
        else:
            already = any(
                (c.get("author") or {}).get("login") == bot_login
                and head_sha[:8] in (c.get("body") or "")
                for c in (thread.get("comments") or {}).get("nodes") or []
            )
            if already:
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
                        f"⚠️ Reusing this existing thread because the same finding still applies on `{head_sha[:8]}`. If it has already been addressed, please mark the thread resolved.",
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
) -> ReviewModel:
    findings, collapsed = collapse_same_root(list(verified.findings))
    kept, sup = dedupe_against_github(
        gh, repo, number, head_sha, findings, bot_login, dry_run=dry_run
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
    )


def publish(gh: Gh, m: ReviewModel, *, bot_login: str, dry_run: bool) -> PublishResult:
    body = render(m)
    event = m.canonical_event
    if (
        m.verified.findings
        and not m.kept
        and not m.skipped
        and event != "APPROVE"
        and m.phase != "preliminary"
    ):
        return PublishResult(
            posted=False,
            event=event,
            transport_event=event,
            body=body,
            suppressed=m.suppressed,
            skipped_reason="already_reviewed",
        )
    transport = event
    if event in {"APPROVE", "REQUEST_CHANGES"}:
        author = github.pr_meta(gh, m.repo, m.number).author
        if author.lower() == bot_login.lower():
            transport = "COMMENT"
            body += f"\n\n_Canonical verifier result: `{event}`. GitHub does not allow authors to approve or request changes on their own pull requests, so this review was submitted using `COMMENT` transport. The findings and blocker status above are unchanged._"
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
    if finding_hash in kept_hashes:
        status, explicit = "STILL_VALID", True
    elif stated and stated != "STILL_VALID":
        status, explicit = stated, True
    else:
        status, explicit = "WITHDRAWN", False
    reason = ""
    if stated == status:
        reason = str(row.get("reason") or "").strip()
        if "@coderabbitai" in reason.lower():
            reason = ""
        reason = reason.replace("@", "@\u200b")[:_REASON_MAX]
    if not reason and status == "STILL_VALID":
        reason = "The verifier kept this finding on the current head; see the updated review."
    elif not reason:
        reason = "This finding did not survive verification on the current head."
    return status, reason, explicit


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
    finding never lingers as an open comment. Returns what was done."""
    # a verifier that echoes the carried finding without its finding_hash must not turn a
    # kept finding into "withdrawn": fall back to the finding's own hash as well
    kept_hashes = {f.prior_hash for f in verified.findings if f.prior_hash} | {
        f.hash for f in verified.findings
    }
    posted = [str(c.get("body") or "") for c in github.inline_comments(gh, repo, number)]
    done: list[dict[str, Any]] = []
    todo = dict(threads)
    for h, t in (open_threads or {}).items():
        if h in todo or h in kept_hashes:
            continue
        stated = str((reconciliation.get(h) or {}).get("status") or "")
        if stated in {"WITHDRAWN", "FIXED", "OUTDATED"}:
            todo[h] = {**t, "latest_reply_id": None}
    for h, t in todo.items():
        cid = t.get("comment_id")
        if not cid:
            continue
        marker = THREAD_ANSWER_MARKER.format(
            sha=head_sha, reply=t.get("latest_reply_id"), finding=h
        )
        status, reason, explicit = _answer_outcome(h, reconciliation.get(h) or {}, kept_hashes)
        item: dict[str, Any] = {"finding_hash": h, "status": status, "comment_id": cid}
        if any(marker in b for b in posted):
            item["action"] = "already_answered"
            done.append(item)
            continue
        body = (
            f"{marker}\n"
            f"**{_ANSWER_LEAD.get(status, status)}** (re-reviewed at `{head_sha[:8]}`): {reason}"
        )
        try:
            github.post_reply(gh, repo, number, int(cid), body)
            item["action"] = "replied"
        except (ReviewError, ValueError) as exc:
            item["action"] = f"reply_failed: {exc}"
            done.append(item)
            continue
        if explicit and status in {"WITHDRAWN", "FIXED", "OUTDATED"} and t.get("thread_id"):
            try:
                github.resolve_thread(gh, str(t["thread_id"]))
                item["resolved"] = True
            except ReviewError as exc:
                item["resolved"] = f"failed: {exc}"
        done.append(item)
    return done


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
    withdrawn_blockers: list[str],
) -> str:
    n = verified.blocker_count
    summary = "\n".join(
        line for line in verified.summary.splitlines() if not _SUMMARY_SOURCE_LINE_RE.match(line)
    ).strip()
    parts = [
        UPDATE_MARKER.format(phase=phase, sha=head_sha),
        f"## Re-review after discussion — commit {head_sha[:8]}",
        "",
        f"Standing review was `{previous_event}`; this re-review is `{new_event}`: "
        + ("no blocking findings remain." if not n else f"{n} blocking finding(s) now stand."),
        "",
    ]
    if withdrawn_blockers:
        parts += ["Withdrawn blocking finding(s):", *(f"- {t}" for t in withdrawn_blockers), ""]
    if summary:
        parts += [summary, ""]
    parts += [
        "_Same commit as the standing review; the inline threads above carry the per-finding "
        "outcome. This follow-up exists only to correct the verdict._",
        "",
        _source_line(provenance),
        "",
        *_provenance_lines(provenance, phase),
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
    if verified.blocker_count:
        return "REQUEST_CHANGES"
    if verified.review_action == "APPROVE" and not provenance.phase2_skipped:
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
    withdrawn_blockers: list[str],
    bot_login: str,
) -> PublishResult:
    event = verdict_event(verified, provenance)
    body = render_verdict_update(
        head_sha=head_sha,
        phase=phase,
        verified=verified,
        provenance=provenance,
        previous_event=previous_event,
        new_event=event,
        withdrawn_blockers=withdrawn_blockers,
    )
    own = github.pr_meta(gh, repo, number).author.lower() == bot_login.lower()
    transport = transport_event(event, own_pr=own)
    if transport != event:
        body += f"\n\n_Canonical verifier result: `{event}`; submitted as `COMMENT` because GitHub does not allow authors to approve or request changes on their own pull requests._"
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
