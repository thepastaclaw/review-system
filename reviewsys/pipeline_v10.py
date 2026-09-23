"""The v10 review flow (policy `pipeline` block present).

    finders -> group (dedupe + ledger match) -> one verifier per new group   (per phase)
    Phase-1 gate
    threads (every open ledger issue) -> CodeRabbit leftovers -> composer -> publish

Finders never see earlier findings; the group lane decides what is new, what is an open
ledger issue re-found, and what was already closed. The thread lane is the only lane that
reads maintainers' replies and the only one that moves an issue's status. Every lane answers
through a JSON Schema. See REVIEWSYS_REVIEW_QUALITY_PLAN.md (r3) for the reasoning.

This module orchestrates; the schemas, parsers and pure rules live in `v10.py`, and the
lane runner, degraded mode, the Phase-1 ladder and publication are the worker's.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import converse, github, ledger, publish
from . import worker as w
from .config import LaneModel, PipelineV10, min_effort
from .contract import Finding, VerifierOutput
from .converse import ThreadOutcome
from .db import connect, event, kv_get, kv_set, tx
from .dedupe import _span, _spans_overlap
from .lane import TurnCapReached
from .ledger import Issue
from .models import FailKind, ReviewError, RunStatus, StepName
from .prompts import read_template, skill_texts
from .steps import worktree as wt
from .v10 import (
    COMPOSER_SCHEMA,
    FINDER_SCHEMA,
    THREAD_SCHEMA,
    TRIAGE_SCHEMA,
    VERIFIER_SCHEMA,
    Candidate,
    Group,
    Verified,
    gated_severity,
    lines_changed,
    parse_candidates,
    parse_groups,
    parse_thread_decisions,
    parse_verdict,
    rank_groups,
    rank_verified,
)
from .worker import RunContext

log = logging.getLogger(__name__)

REPO_RULE_FILES = ("CLAUDE.md", "AGENTS.md")
REPO_RULES_MAX = 12_000
DISCUSSION_MAX = 20_000
CR_BODY_MAX = 1500
REPLY_MAX = 1500
KEEPS_OPEN = {"STILL_VALID", "NO_REPLY"}
LIFTS = {"WITHDRAWN", "FIXED"}


@dataclasses.dataclass(slots=True)
class PhaseResult:
    """One review phase: what verification kept, the open ledger issues finders re-found
    (evidence for the thread lane), and what did not get a verdict."""

    verified: list[Verified]
    refound: dict[str, list[Candidate]]
    overflow: int = 0  # groups past `verify_cap`
    unverified: int = 0  # groups whose verifier lane failed
    unfinished: list[str] = dataclasses.field(default_factory=list)  # finders past their turn cap

    @property
    def blockers(self) -> int:
        return sum(1 for v in self.verified if v.severity == "blocking")


# ---- small helpers ----

_FIELD_RE = re.compile(r"\{([a-z_]+)\}")
# publication renders the severity itself ("🔴 Blocking: <title>")
_SEVERITY_LABEL_RE = re.compile(r"^\W*(blocking|suggestion|nitpick)\W*[:\-]\s*", re.IGNORECASE)


def _fill(template: str, values: dict[str, str]) -> str:
    """Substitute the named fields in one pass: text substituted in (skills, PR descriptions)
    is never itself scanned for fields, and unknown `{...}` is left alone."""
    return _FIELD_RE.sub(lambda m: values.get(m.group(1), m.group(0)), template)


def _pipe(ctx: RunContext) -> PipelineV10:
    p = ctx.cfg.policy.pipeline
    assert p is not None
    return p


def _repo_rules(ctx: RunContext) -> str:
    """The reviewed repository's own agent rules, for the conventions check in `scan`."""
    if not ctx.worktree:
        return ""
    parts = []
    for name in REPO_RULE_FILES:
        path = ctx.worktree / name
        if path.is_file():
            text = path.read_text(errors="replace")[:REPO_RULES_MAX]
            parts.append(f"### The repository's own {name}\n\n{text}")
    return "\n\n".join(parts)


def _base_values(ctx: RunContext) -> dict[str, str]:
    assert ctx.meta
    project_skill, review_skill = skill_texts(ctx.cfg, ctx.repo)
    return {
        "repo": ctx.repo,
        "pr_number": str(ctx.number),
        "pr_title": ctx.meta.title,
        "pr_description": (ctx.meta.body or "(no description)")[:12_000],
        "base_branch": ctx.meta.base_ref,
        "head_sha": ctx.sha,
        "coverage_from": ctx.coverage_from,
        "project_skill": project_skill,
        "review_skill": review_skill,
    }


def _cr_bodies(ctx: RunContext) -> dict[str, str]:
    return {
        str(f["comment_id"]): str(f.get("body") or "")[:CR_BODY_MAX]
        for f in ctx.coderabbit.get("findings", [])
    }


def _parallel[T, R](ctx: RunContext, fn: Callable[[RunContext, T], R], items: list[T]) -> list[R]:
    """`fn(ctx_i, item)` for every item on up to `verify_concurrency` threads. SQLite handles
    are per-thread, so each worker gets a shallow copy of the context with its own connection
    (lane rows, events and a mid-run degraded flip are written through it). A degraded flip in
    one worker is copied back so the rest of the run sees it."""
    if not items:
        return []

    def work(item: T) -> tuple[R, Any]:
        conn = connect(ctx.cfg.db_path)
        try:
            local = dataclasses.replace(ctx, conn=conn)
            return fn(local, item), local.degraded
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=_pipe(ctx).verify_concurrency) as pool:
        pairs = list(pool.map(work, items))
    for _, state in pairs:
        if state is not None and state.active and not ctx.is_degraded:
            ctx.degraded = state
    return [r for r, _ in pairs]


# ---- finders ----


def _finder_specs(ctx: RunContext, phase: str) -> list[tuple[str, str, str]]:
    """(role, method text, lane title) for every finder lane this phase runs."""
    p = _pipe(ctx)
    specs: list[tuple[str, str, str]] = []
    if phase == "phase1":
        method = "\n\n".join(read_template(ctx.cfg, rel) for rel in p.phase1_general)
        specs.append(("general", method, "general review"))
    else:
        for name, rel in p.general_lanes.items():
            specs.append((name, read_template(ctx.cfg, rel), f"{name} method"))
    for sid in ctx.selection:
        spec_rel = p.specialist_prompts.get(sid)
        if spec_rel:
            specs.append((sid, read_template(ctx.cfg, spec_rel), f"{sid} specialist"))
    return specs


def run_finders(
    ctx: RunContext,
    *,
    phase: str,
    specs: list[tuple[str, str, str]],
    lm: LaneModel,
    fallback: Callable[[RunContext, LaneModel, ReviewError], LaneModel | None] | None = None,
) -> tuple[list[Candidate], list[str]]:
    """All finder lanes of one phase, sequentially (each is a long agentic session; the proxy,
    not this loop, is the bottleneck). Returns the candidates and the roles that ran out of
    turns (those contribute nothing; the review says so). A Phase-1 lane that dies moves down
    the model ladder for it and the lanes after it, as in v9."""
    p = _pipe(ctx)
    frame = read_template(ctx.cfg, p.finder_frame)
    base = {
        **_base_values(ctx),
        "discussion": json.dumps(
            {
                "issue_comments": ctx.evidence.get("issue_comments") or [],
                "review_threads": ctx.evidence.get("review_threads") or [],
            },
            indent=1,
        )[:DISCUSSION_MAX],
        "repo_rules": _repo_rules(ctx),
    }
    turns = p.max_turns.get(phase) or None
    out: list[Candidate] = []
    unfinished: list[str] = []
    for role, method, title in specs:
        # the method's own fields ({repo}, {head_sha} in a specialist's commands) first; the
        # method text then goes into the frame unscanned
        prompt = _fill(frame, {**base, "lane_title": title, "method": "\x00M\x00"})
        prompt = prompt.replace("\x00M\x00", _fill(method, base))
        while True:
            cap = p.specialist_effort_cap.get(role)
            lane_lm = dataclasses.replace(lm, effort=min_effort(lm.effort, cap)) if cap else lm
            try:
                raw = w._run_lane(
                    ctx,
                    phase=phase,
                    role=role,
                    lm=lane_lm,
                    prompt=prompt,
                    is_verifier=False,
                    schema=FINDER_SCHEMA,
                    max_turns=turns,
                )
                break
            except TurnCapReached:
                raw = None
                break
            except ReviewError as exc:
                nxt = fallback(ctx, lm, exc) if fallback else None
                if nxt is None:
                    raise
                lm = nxt
        if raw is None:
            unfinished.append(role)
            continue
        cands = parse_candidates(raw, lane=f"{phase}:{role}", start=len(out) + 1, prefix=phase[-1])
        w._record_findings(ctx, phase, "lane", [c.to_finding() for c in cands])
        out.extend(cands)
    if unfinished:
        with tx(ctx.conn):
            event(
                ctx.conn,
                "finder.turn_cap",
                repo=ctx.repo,
                number=ctx.number,
                run_id=ctx.run_id,
                detail=f"{phase}: {', '.join(unfinished)} ran out of turns ({turns})",
            )
    return out, unfinished


# ---- grouping against the ledger ----


def group(
    ctx: RunContext,
    *,
    phase: str,
    cands: list[Candidate],
    issues: dict[str, Issue],
    lm: LaneModel,
) -> list[Group]:
    """Dedupe candidates and match them to the ledger. A failed group lane is not worth the
    review: every candidate becomes its own `new` group, and publication's duplicate check
    against existing threads is the safety net."""
    if not cands:
        return []
    p = _pipe(ctx)
    open_h = {h for h, i in issues.items() if i.is_open}
    closed_h = set(issues) - open_h
    cr = _cr_bodies(ctx)
    if len(cands) == 1 and not issues and not cr:
        return [Group(cands, cands[0], "new")]
    ledger_view = [
        i.summary(_last_message(ctx.open_threads.get(h) or ctx.resolved_threads.get(h)))
        for h, i in issues.items()
    ]
    cr_view = [
        {
            "comment_id": f["comment_id"],
            "path": f.get("path"),
            "line": f.get("line"),
            "body": cr[str(f["comment_id"])],
        }
        for f in ctx.coderabbit.get("findings", [])
    ]
    prompt = _fill(
        read_template(ctx.cfg, p.triage_prompt),
        {
            "repo": ctx.repo,
            "pr_number": str(ctx.number),
            "head_sha": ctx.sha,
            "candidates": json.dumps([c.brief() for c in cands], indent=1),
            "ledger": json.dumps(ledger_view, indent=1),
            "coderabbit": json.dumps(cr_view, indent=1),
        },
    )
    try:
        raw = w._run_lane(
            ctx,
            phase=phase,
            role="group",
            lm=dataclasses.replace(lm, effort=p.triage_effort),
            prompt=prompt,
            is_verifier=True,
            schema=TRIAGE_SCHEMA,
        )
    except ReviewError as exc:
        log.warning("group lane failed, verifying every candidate on its own: %s", exc)
        raw = {"groups": []}
    return parse_groups(
        raw, cands, open_hashes=open_h, closed_hashes=closed_h, coderabbit_ids=set(cr)
    )


def _last_message(thread: dict[str, Any] | None) -> str:
    tr = (thread or {}).get("transcript") or []
    return str(tr[-1].get("body") or "") if tr else ""


def _revive(ctx: RunContext, g: Group, issues: dict[str, Issue]) -> bool:
    """A group matched to a closed issue: verify it again only when that is warranted. A fixed
    issue that reappears is a regression. A withdrawn, deferred, outdated or hand-resolved one
    (a decision) comes back only when the lines it points at changed since it was closed."""
    issue = issues.get(g.target)
    if issue is None or issue.status not in ledger.STICKY:
        return True
    if not ctx.worktree or not issue.closed_sha or not issue.file:
        return False
    diff = wt.diff_file(ctx.worktree, issue.closed_sha, ctx.sha, issue.file, context=0)
    if diff is None:
        return False  # cannot tell (history rewritten): the decision stands
    return lines_changed(diff, issue.line_start, issue.line_end)


# ---- verification ----


def _verify_one(
    ctx: RunContext,
    g: Group,
    *,
    phase: str,
    lm: LaneModel,
    values: dict[str, str],
    template: str,
    closed: Issue | None,
) -> Verified | None:
    rep = g.representative
    cand = {**rep.brief(), "body": rep.body, "suggestion": rep.suggestion}
    others = [m.brief() for m in g.members if m.id != rep.id]
    notes = []
    if others:
        notes.append(
            "Other reviewers raised the same point independently:\n\n```json\n"
            + json.dumps(others, indent=1)
            + "\n```"
        )
    if closed is not None:
        thread = ctx.open_threads.get(closed.hash) or ctx.resolved_threads.get(closed.hash) or {}
        notes.append(
            f"This was raised before and closed as `{closed.status}`"
            + (f" ({closed.note})" if closed.note else "")
            + ". Its discussion, so you do not repeat an argument already settled:\n\n```json\n"
            + json.dumps(converse._thread_for_prompt(closed.hash, thread), indent=1)[:8000]
            + "\n```"
        )
    prompt = _fill(
        template,
        {**values, "candidate": json.dumps(cand, indent=1), "context_note": "\n\n".join(notes)},
    )
    try:
        raw = w._run_lane(
            ctx,
            phase=phase,
            role=f"verify-{rep.id}",
            lm=lm,
            prompt=prompt,
            is_verifier=True,
            schema=VERIFIER_SCHEMA,
        )
        verdict = parse_verdict(raw)
    except ReviewError as exc:
        log.warning("verifier for %s failed: %s", rep.id, exc)
        with tx(ctx.conn):
            event(
                ctx.conn,
                "verify.candidate_failed",
                repo=ctx.repo,
                number=ctx.number,
                run_id=ctx.run_id,
                detail=f"{rep.id} [{rep.severity}] {rep.title[:120]}: {exc}"[:1000],
            )
        return None
    return Verified(g, verdict, gated_severity(verdict))


def verify(
    ctx: RunContext, *, phase: str, groups: list[Group], issues: dict[str, Issue], lm: LaneModel
) -> PhaseResult:
    """One verifier lane per group that needs one. A blocking candidate whose verifier fails
    fails the run (INFRA, retried by the scheduler): publishing without it would be a verdict
    nobody checked. A failed non-blocking one is dropped and disclosed."""
    p = _pipe(ctx)
    refound: dict[str, list[Candidate]] = {}
    todo: list[Group] = []
    for g in groups:
        if g.kind == "open":
            refound.setdefault(g.target, []).extend(g.members)
        elif g.kind != "closed" or _revive(ctx, g, issues):
            todo.append(g)
    ranked = rank_groups(todo)
    batch = ranked[: p.verify_cap]
    values = _base_values(ctx)
    template = read_template(ctx.cfg, p.verifier_prompt)
    results = _parallel(
        ctx,
        lambda c, g: _verify_one(
            c,
            g,
            phase=phase,
            lm=lm,
            values=values,
            template=template,
            closed=issues.get(g.target) if g.kind == "closed" else None,
        ),
        batch,
    )
    failed = [g for g, r in zip(batch, results, strict=True) if r is None]
    if any(m.severity == "blocking" for g in failed for m in g.members):
        raise ReviewError(
            FailKind.INFRA,
            f"{phase}: the verifier for a blocking candidate failed; not publishing an unchecked verdict",
        )
    kept = [r for r in results if r is not None and not r.refuted]
    w._record_findings(ctx, phase, "verified", [v.finding for v in kept])
    overflow = len(ranked) - len(batch)
    if overflow or failed:
        with tx(ctx.conn):
            event(
                ctx.conn,
                "verify.capped",
                repo=ctx.repo,
                number=ctx.number,
                run_id=ctx.run_id,
                detail=f"{phase}: verified {len(batch)} of {len(ranked)} groups (cap {p.verify_cap}); {len(failed)} verifier lane(s) failed",
            )
    return PhaseResult(
        verified=rank_verified(kept), refound=refound, overflow=overflow, unverified=len(failed)
    )


def review_phase(
    ctx: RunContext,
    *,
    phase: str,
    finder_lm: LaneModel | Callable[[], LaneModel],
    verifier_lm: LaneModel,
    issues: dict[str, Issue],
    fallback: Callable[[RunContext, LaneModel, ReviewError], LaneModel | None] | None = None,
) -> PhaseResult:
    n = phase[-1]
    find, grp, ver = StepName(f"phase{n}"), StepName(f"group{n}"), StepName(f"verify{n}")
    w._step_start(ctx, find)
    specs = _finder_specs(ctx, phase)
    lm = finder_lm() if callable(finder_lm) else finder_lm
    cands, unfinished = run_finders(ctx, phase=phase, specs=specs, lm=lm, fallback=fallback)
    w._step_end(
        ctx,
        find,
        "ok",
        {"roles": [s[0] for s in specs], "candidates": len(cands), "turn_cap": unfinished},
    )
    w._step_start(ctx, grp)
    groups = group(ctx, phase=phase, cands=cands, issues=issues, lm=verifier_lm)
    w._step_end(
        ctx,
        grp,
        "ok",
        {"groups": len(groups), "matches": {g.representative.id: g.match for g in groups}},
    )
    w._step_start(ctx, ver)
    res = verify(ctx, phase=f"verify{n}", groups=groups, issues=issues, lm=verifier_lm)
    res.unfinished = unfinished
    w._step_end(
        ctx,
        ver,
        "ok",
        {
            "kept": len(res.verified),
            "blockers": res.blockers,
            "overflow": res.overflow,
            "unverified": res.unverified,
            "refound": sorted(res.refound),
        },
    )
    return res


# ---- the thread lane ----


def _issue_for_threads(ctx: RunContext, issue: Issue, refound: list[Candidate]) -> dict[str, Any]:
    thread = ctx.open_threads.get(issue.hash) or {
        "severity": issue.severity,
        "title": issue.title,
        "path": issue.file,
        "line": issue.line_start,
        "body": issue.body,
    }
    d = converse._thread_for_prompt(issue.hash, thread)
    d["ledger_status"] = issue.status
    if ctx.worktree and issue.file:
        d["code_at_head"] = _code_excerpt(ctx, issue)
        since = issue.closed_sha or issue.opened_sha
        if since and since != ctx.sha:
            diff = wt.diff_file(ctx.worktree, since, ctx.sha, issue.file)
            d["diff_since_last_judged"] = (
                "(unavailable)" if diff is None else diff[:6000] or "(no change)"
            )
    if refound:
        d["refound_this_round"] = [c.brief() for c in refound]
    return d


def _code_excerpt(ctx: RunContext, issue: Issue, pad: int = 12) -> str:
    assert ctx.worktree
    path = ctx.worktree / issue.file
    if not path.is_file():
        return "(file no longer exists at head)"
    lines = path.read_text(errors="replace").splitlines()
    if not issue.line_start:
        return "\n".join(lines[:80])
    lo = max(1, issue.line_start - pad)
    hi = min(len(lines), (issue.line_end or issue.line_start) + pad)
    return "\n".join(f"{n:>5}  {lines[n - 1]}" for n in range(lo, hi + 1))


def _check_live_head(ctx: RunContext) -> None:
    """Nothing is posted against an obsolete commit (same guard as `step_publish`)."""
    live = github.pr_meta(ctx.gh, ctx.repo, ctx.number)
    if live.head_sha != ctx.sha:
        raise ReviewError(
            FailKind.FATAL, f"live head {live.head_sha[:8]} != assigned {ctx.sha[:8]}"
        )


def threads(
    ctx: RunContext,
    *,
    issues: dict[str, Issue],
    refound: dict[str, list[Candidate]],
    new_code: bool,
) -> tuple[dict[str, ThreadOutcome], set[str]]:
    """Decide every open ledger issue (and any issue with a human waiting), post what the lane
    wrote, and move the ledger. `new_code`: the head changed since the issues were raised, so
    every open issue needs a look; on a reply-only run only replied threads do.

    Returns the decisions and the issues lifted (withdrawn or fixed, with the reply on the
    thread). A failed thread lane decides nothing: every open issue stays as it was."""
    waiting = set(w._threads_to_answer(ctx))
    todo = {
        h: i
        for h, i in issues.items()
        if h in waiting or (i.is_open and (new_code or h in refound))
    }
    w._step_start(ctx, StepName.THREADS)
    if not todo:
        w._step_end(ctx, StepName.THREADS, "ok", {"issues": 0})
        return {}, set()
    assert ctx.meta
    replied = {h: ctx.open_threads[h] for h in todo if h in waiting}
    fetched, unfetched = w._fetch_linked_commits(ctx, replied)
    project_skill, review_skill = skill_texts(ctx.cfg, ctx.repo)
    standing = publish.standing_verdict(
        github.reviews(ctx.gh, ctx.repo, ctx.number), ctx.sha, "final", ctx.cfg.bot_login
    )
    prompt = _fill(
        read_template(ctx.cfg, _pipe(ctx).threads_prompt),
        {
            "repo": ctx.repo,
            "pr_number": str(ctx.number),
            "head_sha": ctx.sha,
            "pr_title": ctx.meta.title,
            "pr_author": ctx.meta.author,
            "standing_verdict": standing or "none",
            "pr_description": (ctx.meta.body or "")[:6000],
            "project_skill": project_skill,
            "review_skill": review_skill,
            "issues": json.dumps(
                [_issue_for_threads(ctx, i, refound.get(h, [])) for h, i in todo.items()], indent=1
            ),
            "linked_commits": converse.linked_commits_block(fetched, unfetched, ctx.sha),
            "discussion": json.dumps(ctx.evidence.get("issue_comments") or [], indent=1)[:12_000],
        },
    )
    lm = ctx.lane_model(ctx.cfg.policy.conversation_lane)
    out: dict[str, ThreadOutcome] = {}
    last = ""
    for _ in range(2):
        try:
            raw = w._run_lane(
                ctx,
                phase="threads",
                role="threads",
                lm=lm,
                prompt=prompt + (f"\n\nYour previous answer was rejected: {last}" if last else ""),
                is_verifier=True,
                schema=THREAD_SCHEMA,
            )
            out = parse_thread_decisions(raw, expected=set(todo))
            break
        except ReviewError as exc:
            last = exc.message
    if not out:
        log.warning("thread lane failed; leaving every issue as it was: %s", last)
        w._step_end(ctx, StepName.THREADS, "failed", {"issues": len(todo), "error": last[:500]})
        return {}, set()
    (ctx.run_dir / "threads.json").write_text(
        json.dumps({h: dataclasses.asdict(d) for h, d in out.items()}, indent=1)
    )
    postable = {h: ctx.open_threads[h] for h in out if h in ctx.open_threads}
    answered: list[dict[str, Any]] = []
    if not ctx.dry_run and postable:
        _check_live_head(ctx)
        answered = publish.answer_conversation(
            ctx.gh,
            ctx.repo,
            ctx.number,
            ctx.sha,
            threads=postable,
            outcomes={h: o for h, o in out.items() if h in postable},
        )
    # an outcome reached its thread when we replied, or already had on an earlier attempt
    reached = {
        a["finding_hash"] for a in answered if a.get("action") in {"replied", "already_answered"}
    }
    lifted = {h for h in reached if out[h].status in LIFTS}
    if not ctx.dry_run:
        with tx(ctx.conn):
            for a in answered:
                event(
                    ctx.conn,
                    "thread.answered",
                    repo=ctx.repo,
                    number=ctx.number,
                    run_id=ctx.run_id,
                    detail=json.dumps(a),
                )
            for h, o in out.items():
                # the ledger moves when the reply reached the thread, or when there is no
                # thread to post on (the issue was never inline, or its thread is gone)
                if h in reached or h not in ctx.open_threads:
                    ledger.record_outcome(
                        ctx.conn, ctx.repo, ctx.number, ctx.sha, h, o.status, o.reply or o.reasoning
                    )
                if o.status == "NO_REPLY" and h in replied:
                    kv_set(ctx.conn, w._silent_key(ctx, h), str(replied[h].get("latest_reply_id")))
            w._record_conceded(ctx, "final", lifted)
    w._step_end(
        ctx,
        StepName.THREADS,
        "ok",
        {
            "issues": len(todo),
            "outcomes": {h: o.status for h, o in out.items()},
            "posted": len(reached),
        },
    )
    return out, lifted


# ---- CodeRabbit, composer, publication ----


def _cr_judged_key(ctx: RunContext, cid: str) -> str:
    return f"v10.cr_judged:{ctx.repo}#{ctx.number}:{cid}"


def coderabbit_reactions(
    ctx: RunContext, *, matched: dict[str, Verified], lm: LaneModel
) -> list[dict[str, Any]]:
    """A reaction for every open CodeRabbit comment not judged in an earlier round: `agree`
    when one of our verified findings makes the same point, otherwise that comment verified on
    its own (capped by `verify_cap`, like finder groups)."""
    cr = _cr_bodies(ctx)
    fresh = [cid for cid in cr if kv_get(ctx.conn, _cr_judged_key(ctx, cid)) is None]
    if not fresh:
        return []
    w._step_start(ctx, StepName.CODERABBIT)
    p = _pipe(ctx)
    todo = [cid for cid in fresh if cid not in matched][: p.verify_cap]
    by_id = {str(f["comment_id"]): f for f in ctx.coderabbit.get("findings", [])}
    groups = []
    for cid in todo:
        f = by_id[cid]
        c = Candidate(
            id=f"cr-{cid}",
            lane="coderabbit",
            file=str(f.get("path") or ""),
            line_start=f.get("line"),
            line_end=f.get("line"),
            severity="suggestion",
            category="general",
            title="CodeRabbit comment",
            failure_scenario="(as argued in the comment)",
            body=str(f.get("body") or "")[:6000],
        )
        groups.append(Group([c], c, f"coderabbit:{cid}"))
    values = _base_values(ctx)
    template = read_template(ctx.cfg, p.verifier_prompt)
    results = _parallel(
        ctx,
        lambda c, g: _verify_one(
            c, g, phase="coderabbit", lm=lm, values=values, template=template, closed=None
        ),
        groups,
    )
    reactions = [{"comment_id": int(cid), "action": "agree"} for cid in fresh if cid in matched]
    for g, r in zip(groups, results, strict=True):
        if r is None:
            continue  # verifier failed: say nothing rather than guess
        if r.refuted:
            reply = publish.scrub_reply(r.verdict.evidence, REPLY_MAX)
            reactions.append({"comment_id": int(g.target), "action": "disagree", "reply": reply})
        else:
            reactions.append({"comment_id": int(g.target), "action": "agree"})
    w._step_end(ctx, StepName.CODERABBIT, "ok", {"reactions": len(reactions)})
    return reactions


def _mark_cr_judged(ctx: RunContext, reactions: list[dict[str, Any]]) -> None:
    if ctx.dry_run or not reactions:
        return
    with tx(ctx.conn):
        for r in reactions:
            kv_set(ctx.conn, _cr_judged_key(ctx, str(r["comment_id"])), str(r["action"]))


def compose(
    ctx: RunContext,
    *,
    verified: list[Verified],
    outcomes: dict[str, ThreadOutcome],
    extendable: dict[str, Verified],
    lm: LaneModel,
) -> tuple[str, list[dict[str, Any]]]:
    """Titles, bodies and the summary in one voice. Falls back to the finders' own text when
    the composer lane fails: wording is never worth failing a review over."""
    w._step_start(ctx, StepName.COMPOSE)
    p = _pipe(ctx)
    items = [
        {
            "id": v.group.representative.id,
            "severity": v.severity,
            "verdict": v.verdict.verdict,
            "title": v.finding.title,
            "file": v.finding.file,
            "lines": [v.finding.line_start, v.finding.line_end],
            "failure_scenario": v.group.representative.failure_scenario,
            "reviewer_body": v.finding.body[:4000],
            "verifier_evidence": v.verdict.evidence[:3000],
            "confirm_by": v.verdict.confirm_by,
            "suggestion": v.finding.suggestion,
        }
        for v in verified
    ]
    cr = _cr_bodies(ctx)
    prompt = _fill(
        read_template(ctx.cfg, p.composer_prompt),
        {
            "repo": ctx.repo,
            "pr_number": str(ctx.number),
            "head_sha": ctx.sha,
            "findings": json.dumps(items, indent=1),
            "coderabbit": json.dumps(
                [{"comment_id": int(cid), "body": cr.get(cid, "")} for cid in extendable], indent=1
            ),
            "thread_outcomes": json.dumps(
                [{"finding_hash": h, "status": o.status} for h, o in outcomes.items()], indent=1
            ),
        },
    )
    extend: list[dict[str, Any]] = []
    summary = ""
    try:
        raw = w._run_lane(
            ctx,
            phase="compose",
            role="composer",
            lm=dataclasses.replace(lm, effort=p.composer_effort),
            prompt=prompt,
            is_verifier=True,
            schema=COMPOSER_SCHEMA,
        )
        by_id = {str(r.get("id")): r for r in raw.get("findings") or [] if isinstance(r, dict)}
        for v in verified:
            r = by_id.get(v.group.representative.id)
            if not r:
                continue
            title = _SEVERITY_LABEL_RE.sub("", str(r.get("title") or "").strip())
            body = str(r.get("body") or "").strip()
            if title:
                v.finding.title = title
            if body:
                v.finding.body = body
            if "suggestion" in r:
                sugg = r["suggestion"]
                v.finding.suggestion = sugg if isinstance(sugg, str) and sugg.strip() else None
        summary = str(raw.get("summary") or "").strip()
        for e in raw.get("coderabbit_extend") or []:
            if not isinstance(e, dict) or str(e.get("comment_id")) not in extendable:
                continue
            reply = publish.scrub_reply(str(e.get("reply") or ""), REPLY_MAX)
            if reply:
                extend.append(
                    {"comment_id": int(e["comment_id"]), "action": "extend", "reply": reply}
                )
        w._step_end(ctx, StepName.COMPOSE, "ok", {"findings": len(verified)})
    except ReviewError as exc:
        log.warning("composer failed, publishing the finders' wording: %s", exc)
        w._step_end(ctx, StepName.COMPOSE, "failed", {"error": str(exc)[:500]})
        for v in verified:  # keep D1's promise: a downgraded blocker says what would confirm it
            if v.verdict.confirm_by and v.verdict.confirm_by not in v.finding.body:
                v.finding.body += f"\n\nWhat would confirm it: {v.verdict.confirm_by}"
    return summary, extend


def to_verifier_output(
    ctx: RunContext,
    *,
    phase: str,
    summary: str,
    result: PhaseResult,
    carried: list[Finding],
    reactions: list[dict[str, Any]],
) -> VerifierOutput:
    """The v10 result in the shape publication consumes. New findings are the verified ones
    (capped by `comment_budget`, most severe first); `carried` are open ledger issues the
    thread lane did not close: they count toward the verdict but are never posted again.

    The review approves (subject to `verdict_event`'s full-strength and non-degraded rules)
    only when nothing is outstanding and every candidate got a verdict."""
    budget = ctx.cfg.comment_budget
    new = [v.finding for v in result.verified]
    over = max(0, len(new) - budget)
    new = new[:budget]
    notes = []
    if not summary:
        summary = (
            f"{len(new)} verified finding(s) on this head."
            if new
            else "No verified findings on this head."
        )
    if over or result.overflow:
        notes.append(
            f"_{over + result.overflow} further candidate finding(s) were not published (verification cap / comment budget)._"
        )
    if result.unverified:
        notes.append(
            f"_{result.unverified} non-blocking candidate(s) could not be verified and were not published._"
        )
    if result.unfinished:
        notes.append(
            f"_Reviewer lane(s) {', '.join(result.unfinished)} ran out of turns and contributed nothing._"
        )
    clean = (
        not new
        and not carried
        and not (over or result.overflow or result.unverified or result.unfinished)
    )
    return VerifierOutput(
        summary="\n\n".join([summary, *notes]).strip(),
        review_action="APPROVE" if phase == "final" and clean else "COMMENT",
        findings=[*new, *carried],
        dropped=[],
        out_of_scope=[],
        coderabbit_reactions=reactions,
        prerequisite_adjudications=[],
        adjudication_complete=True,
        review_phase=phase,
        raw={"pipeline": "v10"},
    )


def _carried(
    issues: dict[str, Issue],
    outcomes: dict[str, ThreadOutcome],
    open_threads: dict[str, dict[str, Any]],
) -> list[Finding]:
    """Open issues the thread lane did not close (or did not get to) stay in the verdict."""
    out = []
    for h, i in issues.items():
        o = outcomes.get(h)
        if i.is_open and (o is None or o.status in KEEPS_OPEN):
            f = i.as_finding()
            f.extra["has_thread"] = h in open_threads
            out.append(f)
    return out


# ---- the run ----


def run(ctx: RunContext, *, reply_only: bool) -> RunStatus:
    """Everything after the context step, for policies with a `pipeline` block."""
    threads_all = github.review_threads(ctx.gh, ctx.repo, ctx.number)
    ctx.resolved_threads = {
        h: t
        for h, t in github.finding_threads(
            threads_all, ctx.cfg.bot_login, include_resolved=True
        ).items()
        if t.get("is_resolved")
    }
    issues = ledger.sync(ctx.conn, ctx.repo, ctx.number, ctx.open_threads, dry_run=ctx.dry_run)
    if reply_only:
        # a human replied on a commit we already reviewed: no new code, so no finders; the
        # thread lane answers, and a verdict follow-up is posted if every blocker went away
        _, lifted = threads(ctx, issues=issues, refound={}, new_code=False)
        remaining = w._open_blockers(ctx, "final", lifted=lifted)
        if not ctx.dry_run:
            w._conversation_verdict_update(
                ctx,
                "final",
                ctx.lane_model(ctx.cfg.policy.conversation_lane),
                github.reviews(ctx.gh, ctx.repo, ctx.number),
                remaining=remaining,
                lifted=lifted,
            )
        w._gate_comment(ctx, "done", phase="final", blocker_count=remaining)
        return RunStatus.DONE
    result, phase, phase2_skipped = _review(ctx, issues)
    ctx.check_cancel()
    outcomes, _ = threads(ctx, issues=issues, refound=result.refound, new_code=True)
    issues = ledger.load(ctx.conn, ctx.repo, ctx.number) if not ctx.dry_run else issues
    carried = _carried(issues, outcomes, ctx.open_threads)
    pol = ctx.cfg.policy
    verifier_lm = ctx.lane_model(
        pol.phase1_verifier if phase == "preliminary" else pol.phase2_verifier
    )
    matched_cr = {v.group.target: v for v in result.verified if v.group.kind == "coderabbit"}
    reactions = coderabbit_reactions(ctx, matched=matched_cr, lm=verifier_lm)
    summary, extend = compose(
        ctx, verified=result.verified, outcomes=outcomes, extendable=matched_cr, lm=verifier_lm
    )
    out = to_verifier_output(
        ctx,
        phase=phase,
        summary=summary,
        result=result,
        carried=carried,
        reactions=reactions + extend,
    )
    w._record_findings(
        ctx, "verify1" if phase == "preliminary" else "verify2", "published", out.findings
    )
    # resolved at publish time, so a mid-run switch to a stand-in is what provenance names
    verifier_lm = ctx.lane_model(
        pol.phase1_verifier if phase == "preliminary" else pol.phase2_verifier
    )
    res = w._publish_step(
        ctx, phase=phase, verified=out, verifier_lm=verifier_lm, phase2_skipped=phase2_skipped
    )
    if res.posted and not ctx.dry_run:
        _mark_cr_judged(ctx, reactions + extend)
        with tx(ctx.conn):
            ledger.record_published(ctx.conn, ctx.repo, ctx.number, ctx.sha, res.published)
    return RunStatus.DONE


def _review(ctx: RunContext, issues: dict[str, Issue]) -> tuple[PhaseResult, str, str | None]:
    """The two review phases and the gate between them: (result, publication phase, why
    Phase 2 was skipped). A standing blocker Phase 1 re-found closes the gate like a new one:
    Phase 2 reviews a PR whose known blockers are gone, as in v9."""
    pol = ctx.cfg.policy
    effort = pol.tier_effort(ctx.tier)

    def phase2() -> PhaseResult:
        return review_phase(
            ctx,
            phase="phase2",
            finder_lm=ctx.lane_model(w._with_effort(pol.phase2_reviewer, effort.phase2)),
            verifier_lm=ctx.lane_model(pol.phase2_verifier),
            issues=issues,
        )

    if w._backlog_skips_phase1(ctx, phase2_effort=effort.phase2):
        return phase2(), "final", None
    r1 = review_phase(
        ctx,
        phase="phase1",
        finder_lm=lambda: w._choose_phase1(ctx, effort.phase1),
        verifier_lm=ctx.lane_model(pol.phase1_verifier),
        issues=issues,
        fallback=w._phase1_fallback,
    )
    standing = sum(1 for h in r1.refound if issues[h].severity == "blocking")
    w._step_start(ctx, StepName.GATE)
    admit = pol.phase2_enabled and effort.phase2 is not None and not (r1.blockers or standing)
    w._step_end(
        ctx,
        StepName.GATE,
        "ok",
        {
            "admit_phase2": admit,
            "tier": ctx.tier,
            "phase2_effort": effort.phase2,
            "standing_blockers_refound": standing,
        },
    )
    if admit:
        return _merge(r1, phase2()), "final", None
    if r1.blockers or standing or not pol.phase2_enabled:
        return r1, "preliminary", None
    return r1, "final", f"triage rated this change {ctx.tier}"


def _merge(a: PhaseResult, b: PhaseResult) -> PhaseResult:
    """Phase 1 found no blockers and Phase 2 ran: publish both phases' verified findings. The
    same defect found in both phases is one finding: Phase 2's (the stronger reviewers) wins
    when a Phase-1 finding has the same file, category, severity and overlapping lines.
    Publication's same-root collapse catches the rest."""
    kept = list(b.verified)
    for v in a.verified:
        f = v.finding
        if not any(
            k.finding.file == f.file
            and k.finding.category == f.category
            and k.severity == v.severity
            and _spans_overlap(_span(k.finding), _span(f))
            for k in b.verified
        ):
            kept.append(v)
    return PhaseResult(
        verified=rank_verified(kept),
        refound={
            h: [*a.refound.get(h, []), *b.refound.get(h, [])] for h in {*a.refound, *b.refound}
        },
        overflow=a.overflow + b.overflow,
        unverified=a.unverified + b.unverified,
        unfinished=[*a.unfinished, *b.unfinished],
    )
