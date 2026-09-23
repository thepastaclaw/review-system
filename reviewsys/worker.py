"""Worker: runs one review (one `runs` row) end to end.

Steps: worktree -> select -> triage -> context -> phase1 -> verify1 -> gate -> phase2 -> verify2 -> publish.
Every step is recorded in `steps`. Any ReviewError ends the run as failed with
its classification; the scheduler decides on retry. Cooperative cancellation
is checked on every heartbeat.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import shutil
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import converse, degraded, github, labels, publish, quota
from .config import Config, DegradedPolicy, LaneModel, min_effort
from .contract import (
    Finding,
    ReviewerOutput,
    VerifierOutput,
    parse_json_object,
    parse_reviewer_output,
    parse_verifier_output,
)
from .db import event, kv_get, kv_set, now, tx
from .degraded import Prober
from .gate import admit_phase2
from .gh import Gh
from .lane import (
    LaneResult,
    LaneRunner,
    LaneSpec,
    TurnCapReached,
    lane_output,
    prompt_sha,
    run_claude_lane,
)
from .models import FailKind, ReviewError, RunStatus, StepName, Trigger
from .prompts import REPAIR_PROMPT, prior_for_prompt, reviewer_prompt, skill_texts, verifier_prompt
from .scheduler import finish_run
from .select import Selection, select, write_selection
from .steps import worktree as wt
from .triage import Triage, triage, write_triage

log = logging.getLogger(__name__)


class Cancelled(Exception):
    pass


@dataclass(slots=True)
class RunContext:
    cfg: Config
    conn: sqlite3.Connection
    gh: Gh
    run_id: int
    head_id: int
    repo: str
    number: int
    sha: str
    token: str
    run_dir: Path
    trigger: str = ""  # heads.trigger: what queued this head
    worktree: Path | None = None
    mirror: Path | None = None
    meta: github.PrMeta | None = None
    base_sha: str = ""
    selection: list[str] = field(default_factory=list)
    files: list[dict[str, Any]] = field(default_factory=list)
    triage: Triage | None = None
    tier: str = "normal"
    phase1_skipped: str | None = (
        None  # reason when the backlog rule sent this run straight to Phase 2
    )
    phase1_choice: quota.Choice | None = None  # which Phase-1 ladder rung ran, and why
    phase1_effort: str | None = None  # the tier's Phase-1 effort, re-capped per rung
    quota_reader: quota.QuotaReader | None = None  # test injection; None = proxy lookup
    degraded: degraded.State | None = None  # stand-in models in use (see degraded.py)
    prober: Prober | None = None  # test injection; None = probe the proxy
    evidence: dict[str, Any] = field(default_factory=dict)
    coderabbit: dict[str, Any] = field(default_factory=dict)
    coderabbit_ids: list[int] = field(default_factory=list)
    prior: list[dict[str, Any]] = field(default_factory=list)
    prior_sha: str | None = None
    has_prior_review: bool = False
    fresh_final: bool = False
    # finding_hash -> {comment_id, thread_id, replies} for prior findings with human replies
    prior_threads: dict[str, dict[str, Any]] = field(default_factory=dict)
    # finding_hash -> thread facts for every unresolved bot finding thread on the PR
    open_threads: dict[str, dict[str, Any]] = field(default_factory=dict)
    # v10: the same for resolved ones (their discussion, for issues that come back)
    resolved_threads: dict[str, dict[str, Any]] = field(default_factory=dict)
    coverage_from: str = ""
    phase1_outputs: dict[str, ReviewerOutput] = field(default_factory=dict)
    phase2_outputs: dict[str, ReviewerOutput] = field(default_factory=dict)
    verify1: VerifierOutput | None = None
    verify2: VerifierOutput | None = None
    reviewers: list[dict[str, Any]] = field(default_factory=list)
    lane_runner: LaneRunner = run_claude_lane
    dry_run: bool = False
    cancel_flag: threading.Event = field(default_factory=threading.Event)

    def check_cancel(self) -> None:
        if self.cancel_flag.is_set():
            raise Cancelled()

    @property
    def adhoc(self) -> bool:
        """This repo has no skills entry, so it is reviewed on generic guidance alone."""
        return self.cfg.repo(self.repo) is None

    @property
    def degraded_policy(self) -> DegradedPolicy | None:
        """The degraded policy while this run is actually running on stand-ins, else None."""
        pol = self.cfg.policy.degraded
        return pol if pol and self.degraded and self.degraded.active else None

    @property
    def is_degraded(self) -> bool:
        return self.degraded_policy is not None

    def lane_model(self, lm: LaneModel) -> LaneModel:
        """`lm`, or its degraded-mode stand-in while the primary models are unavailable."""
        pol = self.degraded_policy
        return pol.resolve(lm) if pol else lm

    def model_name(self, model: str) -> str:
        """Same, for a lane built from a bare model name (selector, repair)."""
        pol = self.degraded_policy
        sub = pol.substitutes.get(model) if pol else None
        return sub.model if sub else model


# ---- heartbeat ----


def _heartbeat_loop(ctx: RunContext, stop: threading.Event) -> None:
    conn = sqlite3.connect(str(ctx.cfg.db_path), timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=10000")
    while not stop.wait(ctx.cfg.heartbeat_seconds):
        try:
            row = conn.execute(
                "SELECT token, cancel_requested, status FROM runs WHERE id=?", (ctx.run_id,)
            ).fetchone()
            if (
                row is None
                or row["token"] != ctx.token
                or row["cancel_requested"]
                or RunStatus(row["status"]).terminal
            ):
                ctx.cancel_flag.set()
                continue
            conn.execute(
                "UPDATE runs SET heartbeat_at=?, status='running' WHERE id=? AND status IN ('spawned','running')",
                (now(), ctx.run_id),
            )
        except sqlite3.Error as exc:
            log.warning("heartbeat error: %s", exc)
    conn.close()


# ---- step bookkeeping ----


def _step_start(ctx: RunContext, name: StepName) -> None:
    with tx(ctx.conn):
        ctx.conn.execute(
            "INSERT OR REPLACE INTO steps (run_id, name, status, started_at) VALUES (?,?,?,?)",
            (ctx.run_id, name.value, "running", now()),
        )
        ctx.conn.execute("UPDATE runs SET phase=? WHERE id=?", (name.value, ctx.run_id))


def _step_end(
    ctx: RunContext, name: StepName, status: str, detail: dict[str, Any] | None = None
) -> None:
    with tx(ctx.conn):
        ctx.conn.execute(
            "UPDATE steps SET status=?, finished_at=?, detail=? WHERE run_id=? AND name=?",
            (status, now(), json.dumps(detail or {})[:4000], ctx.run_id, name.value),
        )


def _lane_row(
    ctx: RunContext,
    *,
    phase: str,
    role: str,
    lm: LaneModel,
    attempt: int,
    attempt_id: str,
    status: str,
    res: LaneResult | None,
    artifact_dir: Path,
    psha: str,
    reason: str = "",
) -> None:
    with tx(ctx.conn):
        ctx.conn.execute(
            "INSERT INTO lanes (run_id, phase, role, agent, model, effort, attempt, attempt_id, status, exit_code, tokens_in, tokens_out, started_at, finished_at, artifact_dir, prompt_sha, reason) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                ctx.run_id,
                phase,
                role,
                lm.agent,
                lm.model,
                lm.effort,
                attempt,
                attempt_id,
                status,
                res.exit_code if res else None,
                res.tokens_in if res else None,
                res.tokens_out if res else None,
                now(),
                now(),
                str(artifact_dir),
                psha,
                reason[:500],
            ),
        )


_FINDINGS_INSERT = "INSERT INTO findings (run_id, phase, stage, hash, file, line_start, line_end, severity, confidence, category, title, body) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"


def _insert_findings(ctx: RunContext, phase: str, stage: str, findings: list[Finding]) -> None:
    """Write finding rows. The caller must already hold the write transaction."""
    ctx.conn.executemany(
        _FINDINGS_INSERT,
        [
            (
                ctx.run_id,
                phase,
                stage,
                f.hash,
                f.file,
                f.line_start,
                f.line_end,
                f.severity,
                f.confidence,
                f.category,
                f.title,
                f.body[:8000],
            )
            for f in findings
        ],
    )


def _record_findings(ctx: RunContext, phase: str, stage: str, findings: list[Finding]) -> None:
    with tx(ctx.conn):
        _insert_findings(ctx, phase, stage, findings)


def _set_run_review(
    ctx: RunContext, *, blocker_count: int, review_id: int | None, review_url: str | None
) -> None:
    """Record the published review on the run row. Caller holds the write transaction."""
    ctx.conn.execute(
        "UPDATE runs SET blocker_count=?, review_id=?, review_url=? WHERE id=?",
        (blocker_count, review_id, review_url, ctx.run_id),
    )


def _gate_comment(ctx: RunContext, status: str, **kw: Any) -> None:
    if ctx.dry_run:
        return
    if ctx.triage is not None:
        kw.setdefault("tier", ctx.tier)
    if ctx.phase1_skipped:
        kw.setdefault("phase1_skipped", ctx.phase1_skipped)
    if ctx.adhoc:
        kw.setdefault("adhoc", True)
    if ctx.is_degraded:
        kw.setdefault("degraded", True)
    try:
        github.upsert_gate_comment(
            ctx.gh, ctx.repo, ctx.number, ctx.cfg.bot_login, github.gate_body(status, ctx.sha, **kw)
        )
    except ReviewError as exc:
        log.warning("gate comment update failed: %s", exc)


# ---- steps ----


def step_worktree(ctx: RunContext) -> None:
    meta = github.pr_meta(ctx.gh, ctx.repo, ctx.number)
    if meta.state != "open" or meta.merged:
        raise ReviewError(FailKind.FATAL, f"PR is {meta.state}{' (merged)' if meta.merged else ''}")
    if meta.head_sha != ctx.sha:
        raise ReviewError(
            FailKind.FATAL, f"live head {meta.head_sha[:8]} != assigned {ctx.sha[:8]}"
        )
    ctx.meta = meta
    ctx.base_sha = meta.base_sha
    ctx.mirror = wt.ensure_mirror(ctx.cfg.mirrors_dir, ctx.repo)
    wt.fetch_head(ctx.mirror, ctx.number, ctx.sha)
    ctx.worktree = wt.create_worktree(
        ctx.mirror,
        ctx.cfg.worktrees_dir,
        f"{ctx.repo.replace('/', '-')}-{ctx.number}-{ctx.run_id}",
        ctx.sha,
    )
    base = wt.merge_base(ctx.worktree, meta.base_ref, ctx.sha) if meta.base_ref else None
    ctx.coverage_from = base or f"{ctx.sha}~1"
    with tx(ctx.conn):
        ctx.conn.execute(
            "UPDATE runs SET worktree=?, run_dir=? WHERE id=?",
            (str(ctx.worktree), str(ctx.run_dir), ctx.run_id),
        )


def step_select(ctx: RunContext) -> None:
    assert ctx.meta and ctx.worktree
    files_raw = (
        ctx.gh.api(f"repos/{ctx.repo}/pulls/{ctx.number}/files?per_page=100", paginate=True) or []
    )
    ctx.files = [f for f in files_raw if isinstance(f, dict)]
    files = [str(f.get("filename")) for f in ctx.files]
    meta, worktree = ctx.meta, ctx.worktree

    def attempt() -> Selection:
        return select(
            ctx.cfg,
            repo=ctx.repo,
            title=meta.title,
            body=meta.body,
            files=files,
            run_dir=ctx.run_dir,
            worktree=worktree,
            runner=ctx.lane_runner,
            model_for=ctx.model_name,
        )

    sel = attempt()
    if sel.error and _degrade_on_side_lane_failure(
        ctx, "select", ctx.cfg.policy.selector_model, sel.error
    ):
        sel = attempt()  # once more, on the stand-ins
    write_selection(ctx.run_dir, sel)
    ctx.selection = sel.selected
    if sel.error:
        with tx(ctx.conn):
            event(
                ctx.conn,
                "select.degraded",
                repo=ctx.repo,
                number=ctx.number,
                run_id=ctx.run_id,
                detail=f"method={sel.method} error={sel.error}",
            )


def step_triage(ctx: RunContext) -> None:
    """Rate the PR so reviewer effort can scale; no-op unless the policy configures triage."""
    assert ctx.meta and ctx.worktree
    lane = ctx.cfg.policy.triage
    if lane is None:
        return
    meta, worktree = ctx.meta, ctx.worktree

    def attempt() -> Triage:
        return triage(
            ctx.cfg,
            repo=ctx.repo,
            base_ref=meta.base_ref,
            title=meta.title,
            body=meta.body,
            files=ctx.files,
            run_dir=ctx.run_dir,
            worktree=worktree,
            runner=ctx.lane_runner,
            lane=ctx.lane_model(lane),
        )

    t = attempt()
    if t.error and _degrade_on_side_lane_failure(ctx, "triage", lane.model, t.error):
        t = attempt()  # once more, on the stand-in
    write_triage(ctx.run_dir, t)
    ctx.triage = t
    ctx.tier = t.tier
    with tx(ctx.conn):
        ctx.conn.execute("UPDATE runs SET tier=? WHERE id=?", (t.tier, ctx.run_id))
        if t.error:
            event(
                ctx.conn,
                "triage.degraded",
                repo=ctx.repo,
                number=ctx.number,
                run_id=ctx.run_id,
                detail=f"tier={t.tier} method={t.method} error={t.error}",
            )
    _gate_comment(ctx, "in_progress")


def step_context(ctx: RunContext) -> None:
    assert ctx.meta
    threads = github.review_threads(ctx.gh, ctx.repo, ctx.number)
    ctx.evidence = github.evidence_bundle(
        ctx.gh, ctx.repo, ctx.number, ctx.meta, ctx.cfg.bot_login, include_coderabbit=False
    )
    ctx.coderabbit = github.coderabbit_context(threads)
    ctx.coderabbit_ids = [
        int(f["comment_id"]) for f in ctx.coderabbit["findings"] if f.get("comment_id")
    ]
    # Any earlier published review means this head is an iterative pass.  Once the
    # historical findings have been reconciled, the run must earn approval through
    # a fresh Phase-2 final audit of the complete current diff.
    local_prior = ctx.conn.execute(
        "SELECT 1 FROM reviews WHERE repo=? AND number=? LIMIT 1", (ctx.repo, ctx.number)
    ).fetchone()
    github_prior = any(
        (r.get("user") or {}).get("login", "").lower() == ctx.cfg.bot_login.lower()
        and github.REVIEW_MARKER in str(r.get("body") or "")
        for r in github.reviews(ctx.gh, ctx.repo, ctx.number)
    )
    ctx.has_prior_review = bool(local_prior or github_prior)
    # prior findings: last posted set for this PR from our DB, plus any human replies on
    # their inline threads (the reviewer must engage with pushback, not re-raise past it)
    ctx.open_threads = github.finding_threads(threads, ctx.cfg.bot_login)
    replied = {h: t for h, t in ctx.open_threads.items() if t["awaiting_answer"]}
    rows = ctx.conn.execute(
        "SELECT pf.hash, pf.sha, f.file, f.line_start, f.line_end, f.severity, f.category, f.title, f.body FROM posted_findings pf LEFT JOIN findings f ON f.hash=pf.hash AND f.stage='posted' WHERE pf.repo=? AND pf.number=? ORDER BY pf.posted_at DESC, f.id DESC",
        (ctx.repo, ctx.number),
    ).fetchall()
    seen: set[str] = set()
    prior: list[Finding] = []
    for r in rows:
        if r["hash"] in seen or not r["title"]:
            continue
        seen.add(r["hash"])
        ctx.prior_sha = ctx.prior_sha or r["sha"]
        thread = replied.get(r["hash"])
        prior.append(
            Finding(
                file=r["file"] or "",
                title=r["title"],
                body=r["body"] or "",
                severity=r["severity"] or "nitpick",
                category=r["category"] or "general",
                line_start=r["line_start"],
                line_end=r["line_end"],
                extra={"thread_replies": thread["transcript"]} if thread else {},
            )
        )
    for h, t in replied.items():
        if h in seen or not t.get("title"):
            continue
        # a finding this database never recorded (posted by the legacy pipeline) that a human
        # replied to: reconstruct it from the comment so the reply still gets adjudicated
        seen.add(h)
        prior.append(
            Finding(
                file=t.get("path") or "",
                title=t["title"],
                body=t["body"],
                severity=t["severity"],
                line_start=t.get("line"),
                line_end=t.get("line"),
                prior_hash=h,
                extra={"thread_replies": t["transcript"]},
            )
        )
    ctx.prior = prior_for_prompt(prior, ctx.prior_sha or "")
    ctx.prior_threads = {h: t for h, t in replied.items() if h in seen}
    (ctx.run_dir / "evidence.json").write_text(
        json.dumps(
            {"evidence": ctx.evidence, "coderabbit": ctx.coderabbit, "prior": ctx.prior}, indent=1
        )
    )


def _run_lane(
    ctx: RunContext,
    *,
    phase: str,
    role: str,
    lm: LaneModel,
    prompt: str,
    is_verifier: bool,
    fresh: bool = False,
    schema: dict[str, Any] | None = None,
    max_turns: int | None = None,
) -> dict[str, Any]:
    """Run a lane with bounded retries and one cheap JSON repair. Returns parsed output.

    With `schema` the answer is enforced by Claude Code itself (retried in-conversation), so
    there is no repair lane: a lane that still fails the contract is retried whole.

    A lane on a primary model that dies on a quota failure flips the run into degraded mode
    (when the policy has a stand-in for that model) and gets its two attempts again on the
    stand-in, so one exhausted pool does not fail the review."""
    assert ctx.worktree
    lm = ctx.lane_model(lm)
    last: str = ""
    attempt, budget = 0, 2
    while attempt < budget:
        attempt += 1
        ctx.check_cancel()
        attempt_id = uuid.uuid4().hex[:12]
        art = ctx.run_dir / "attempts" / f"{phase}-{role}-{attempt_id}"
        spec = LaneSpec(
            role=role,
            agent=lm.agent,
            model=lm.model,
            effort=lm.effort,
            prompt=prompt,
            cwd=ctx.worktree,
            add_dir=ctx.run_dir,
            timeout_seconds=ctx.cfg.lane_timeout_minutes * 60,
            claude_bin=ctx.cfg.claude_bin,
            max_budget_usd=ctx.cfg.lane_budget_usd,
            json_schema=schema,
            max_turns=max_turns,
            clean_context=ctx.cfg.policy.pipeline is not None,
        )
        res = ctx.lane_runner(spec, art, ctx.worktree)
        psha = prompt_sha(prompt)
        try:
            out = lane_output(res)
        except ReviewError as exc:
            if isinstance(exc, TurnCapReached):
                _lane_row(
                    ctx,
                    phase=phase,
                    role=role,
                    lm=lm,
                    attempt=attempt,
                    attempt_id=attempt_id,
                    status="turn_cap",
                    res=res,
                    artifact_dir=art,
                    psha=psha,
                    reason=str(exc),
                )
                raise
            if (
                schema is None
                and exc.kind == FailKind.CONTRACT
                and res.ok
                and res.result_text.strip()
            ):
                repaired = _repair(ctx, res.result_text, art)
                if repaired is not None:
                    _lane_row(
                        ctx,
                        phase=phase,
                        role=role,
                        lm=lm,
                        attempt=attempt,
                        attempt_id=attempt_id,
                        status="repaired",
                        res=res,
                        artifact_dir=art,
                        psha=psha,
                    )
                    return repaired
            _lane_row(
                ctx,
                phase=phase,
                role=role,
                lm=lm,
                attempt=attempt,
                attempt_id=attempt_id,
                status="failed",
                res=res,
                artifact_dir=art,
                psha=psha,
                reason=str(exc),
            )
            last = str(exc)
            switched = _degrade_on_quota_failure(ctx, lm, exc, res)
            if switched is not None:
                lm, budget = switched, attempt + 2
            continue
        _lane_row(
            ctx,
            phase=phase,
            role=role,
            lm=lm,
            attempt=attempt,
            attempt_id=attempt_id,
            status="completed",
            res=res,
            artifact_dir=art,
            psha=psha,
        )
        if not is_verifier:
            ctx.reviewers.append(
                {
                    "model": lm.model,
                    "agent": lm.agent,
                    "role": role,
                    "effort": lm.effort,
                    "status": "completed",
                    "phase": phase,
                    "fresh": fresh,
                    "attempt_id": attempt_id,
                    "substitute_for": lm.substitute_for,
                }
            )
        return out
    raise ReviewError(
        FailKind.INFRA if "timed out" in last or "exit" in last else FailKind.CONTRACT,
        f"{phase}/{role} lane failed {'twice' if attempt == 2 else f'{attempt} times'}: {last}",
    )


def _repair(ctx: RunContext, raw: str, art: Path) -> dict[str, Any] | None:
    if len(raw) > 200_000 or not ctx.worktree:
        return None
    spec = LaneSpec(
        role="repair",
        agent="repair",
        model=ctx.model_name(ctx.cfg.policy.repair_model),
        effort="low",
        prompt=REPAIR_PROMPT.format(raw=raw),
        cwd=ctx.worktree,
        add_dir=ctx.run_dir,
        timeout_seconds=300,
        claude_bin=ctx.cfg.claude_bin,
    )
    try:
        res = ctx.lane_runner(spec, art / "repair", ctx.worktree)
        if not res.ok:
            return None
        return parse_json_object(res.result_text)
    except ReviewError:
        return None


def _reviewer_lane(
    ctx: RunContext, *, phase: str, role: str, lm: LaneModel, prompt: str, fresh: bool = False
) -> tuple[dict[str, Any], LaneModel]:
    """One reviewer lane, retried down the Phase-1 ladder when its rung dies. Returns the
    raw output and the model that produced it, so the remaining roles stay on that rung."""
    while True:
        try:
            raw = _run_lane(
                ctx, phase=phase, role=role, lm=lm, prompt=prompt, is_verifier=False, fresh=fresh
            )
            return raw, lm
        except ReviewError as exc:
            nxt = _phase1_fallback(ctx, lm, exc) if phase == "phase1" else None
            if nxt is None:
                raise
            lm = nxt


def _reviewer_lanes(
    ctx: RunContext,
    *,
    phase: str,
    lm: LaneModel,
    expected_phase: str,
    fresh: bool = False,
) -> dict[str, ReviewerOutput]:
    assert ctx.meta
    roles = ["general", *ctx.selection]
    outputs: dict[str, ReviewerOutput] = {}
    review_prior = [] if fresh else ctx.prior
    review_prior_sha = None if fresh else ctx.prior_sha
    prior_hashes = {str(p["finding_hash"]) for p in review_prior}
    for role in roles:
        prompt = reviewer_prompt(
            ctx.cfg,
            repo=ctx.repo,
            number=ctx.number,
            head_sha=ctx.sha,
            phase=expected_phase,
            role=role,
            meta=ctx.meta.as_dict(),
            coverage_from=ctx.coverage_from,
            evidence=ctx.evidence,
            prior=review_prior,
            prior_sha=review_prior_sha,
            fresh=fresh,
        )
        raw, lm = _reviewer_lane(ctx, phase=phase, role=role, lm=lm, prompt=prompt, fresh=fresh)
        outputs[role] = parse_reviewer_output(
            raw,
            expected_phase=expected_phase,
            head_sha=ctx.sha,
            source=f"{phase}:{role}",
            prior_hashes=prior_hashes,
        )
        _record_findings(ctx, phase, "lane", outputs[role].findings)
    return outputs


def _verifier_lane(
    ctx: RunContext,
    *,
    phase: str,
    lm: LaneModel,
    expected_phase: str,
    phase2_outputs: dict[str, ReviewerOutput] | None = None,
    fresh_final: bool = False,
) -> VerifierOutput:
    all_phase2 = phase2_outputs if phase2_outputs is not None else ctx.phase2_outputs
    prompt = verifier_prompt(
        ctx.cfg,
        repo=ctx.repo,
        number=ctx.number,
        head_sha=ctx.sha,
        phase=expected_phase,
        phase1_outputs={r: o.raw for r, o in ctx.phase1_outputs.items()},
        phase2_outputs={r: o.raw for r, o in all_phase2.items()},
        coderabbit=ctx.coderabbit,
        coderabbit_ids=ctx.coderabbit_ids,
        evidence=ctx.evidence,
        prior=ctx.prior,
        prior_sha=ctx.prior_sha,
        phase1_skipped=ctx.phase1_skipped,
        fresh_final=fresh_final,
    )
    raw = _run_lane(ctx, phase=phase, role="verifier", lm=lm, prompt=prompt, is_verifier=True)
    out = parse_verifier_output(
        raw, expected_phase=expected_phase, expected_coderabbit_ids=ctx.coderabbit_ids
    )
    lane_findings = {
        f"{p}:{role}": o.findings
        for p, outputs in (("phase1", ctx.phase1_outputs), ("phase2", ctx.phase2_outputs))
        for role, o in outputs.items()
    }
    if phase2_outputs is not None:
        lane_findings.update(
            {
                f"phase2:fresh:{role.removeprefix('fresh:')}": o.findings
                for role, o in phase2_outputs.items()
                if role.startswith("fresh:")
            }
        )
    publish.attribute_sources(
        out,
        ctx.reviewers,
        lane_findings,
    )
    _record_findings(ctx, phase, "verified", out.findings)
    return out


def _backfill_posted(
    ctx: RunContext, phase: str, existing: dict[str, Any], verified: VerifierOutput
) -> None:
    review_id = int(existing.get("id") or 0) or None
    ts = now()
    with tx(ctx.conn):
        if not ctx.conn.execute(
            "SELECT 1 FROM reviews WHERE repo=? AND number=? AND sha=? AND phase=?",
            (ctx.repo, ctx.number, ctx.sha, phase),
        ).fetchone():
            ctx.conn.execute(
                "INSERT INTO reviews (run_id, repo, number, sha, phase, github_review_id, event, posted_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    ctx.run_id,
                    ctx.repo,
                    ctx.number,
                    ctx.sha,
                    phase,
                    review_id,
                    str(existing.get("state") or "UNKNOWN"),
                    ts,
                ),
            )
        ctx.conn.executemany(
            "INSERT INTO posted_findings (repo, number, hash, sha, review_id, posted_at) VALUES (?,?,?,?,?,?) ON CONFLICT(repo, number, hash) DO NOTHING",
            [(ctx.repo, ctx.number, f.hash, ctx.sha, review_id, ts) for f in verified.findings],
        )
        _insert_findings(ctx, phase, "posted", verified.findings)
        _set_run_review(
            ctx,
            blocker_count=verified.blocker_count,
            review_id=review_id,
            review_url=str(existing.get("html_url") or "") or None,
        )


def step_publish(
    ctx: RunContext,
    *,
    phase: str,
    verified: VerifierOutput,
    verifier_lm: LaneModel,
    phase2_skipped: str | None = None,
) -> publish.PublishResult:
    # The review may have taken long enough for a push to land after the initial
    # worktree check. Never publish an approval (or any verdict) against an
    # obsolete head; the scheduler will supersede this run and ingest will queue
    # the live commit.
    live = github.pr_meta(ctx.gh, ctx.repo, ctx.number)
    if live.state != "open" or live.merged:
        raise ReviewError(FailKind.FATAL, f"PR is {live.state}{' (merged)' if live.merged else ''}")
    if live.head_sha != ctx.sha:
        raise ReviewError(
            FailKind.FATAL, f"live head {live.head_sha[:8]} != assigned {ctx.sha[:8]}"
        )
    if ctx.base_sha and live.base_sha and live.base_sha != ctx.base_sha:
        raise ReviewError(
            FailKind.FATAL,
            f"live base {live.base_sha[:8]} != assigned {ctx.base_sha[:8]}",
        )
    existing = github.existing_review_for_sha(
        ctx.gh, ctx.repo, ctx.number, ctx.sha, phase, ctx.cfg.bot_login
    )
    if existing is None and phase == "preliminary":
        # a same-sha re-review that now finds blockers after a FINAL review stood: correct the
        # final verdict with a follow-up rather than stacking a second full review
        existing = github.existing_review_for_sha(
            ctx.gh, ctx.repo, ctx.number, ctx.sha, "final", ctx.cfg.bot_login
        )
        if existing:
            phase = "final"
    if existing:
        # Either a prior attempt posted but died before recording it, or this is a reply-
        # triggered re-review of an already-reviewed commit. Backfill so future rounds still
        # see these findings as "prior", never re-post the review for this sha/phase, answer the
        # threads that were replied to, and, if the verdict moved (a blocker withdrawn or a new
        # one found), post a short follow-up review so the standing verdict is not left stale.
        _backfill_posted(ctx, phase, existing, verified)
        _answer_threads(ctx, phase, verified)
        update = _verdict_update(ctx, phase, verified, verifier_lm, phase2_skipped=phase2_skipped)
        if update is not None:
            return update
        model = _build_review(
            ctx,
            phase=phase,
            verified=verified,
            verifier_lm=verifier_lm,
            phase2_skipped=phase2_skipped,
            rereview=True,
        )
        result = publish.publish(
            ctx.gh,
            model,
            bot_login=ctx.cfg.bot_login,
            dry_run=ctx.dry_run,
            force_comment=True,
        )
        (ctx.run_dir / f"review-{phase}.md").write_text(result.body)
        _record_publication(ctx, phase, model, result, verified)
        return result
    model = _build_review(
        ctx,
        phase=phase,
        verified=verified,
        verifier_lm=verifier_lm,
        phase2_skipped=phase2_skipped,
        rereview=ctx.has_prior_review,
    )
    (ctx.run_dir / f"review-{phase}.md").write_text(publish.render(model))
    result = publish.publish(
        ctx.gh,
        model,
        bot_login=ctx.cfg.bot_login,
        dry_run=ctx.dry_run,
    )
    _record_publication(ctx, phase, model, result, verified)
    if result.posted and verified.coderabbit_reactions and not ctx.dry_run:
        publish.post_coderabbit_reactions(
            ctx.gh, ctx.repo, ctx.number, verified.coderabbit_reactions, ctx.cfg.bot_login
        )
    _answer_threads(ctx, phase, verified)
    return result


def _answer_threads(ctx: RunContext, phase: str, verified: VerifierOutput) -> None:
    if not ctx.open_threads or ctx.dry_run or ctx.cfg.policy.pipeline is not None:
        return  # v10: the thread lane already answered every thread it had something for
    answered = publish.answer_replied_threads(
        ctx.gh,
        ctx.repo,
        ctx.number,
        ctx.sha,
        threads=ctx.prior_threads,
        open_threads=ctx.open_threads,
        reconciliation=_reconciliation(ctx, phase, verified),
        verified=verified,
    )
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


def _record_verdict(ctx: RunContext, phase: str, verified: VerifierOutput, new_event: str) -> None:
    """Keep our own record of the verdict current when nothing is posted: on a bot-authored PR
    GitHub records every review as COMMENTED, so a canonical APPROVE <-> REQUEST_CHANGES move
    changes no transport state and gets no follow-up review, yet the verdict label reads
    `reviews.event` and must see it."""
    with tx(ctx.conn):
        prev = labels.latest_verdict(ctx.conn, ctx.repo, ctx.number, ctx.sha)
        if prev == new_event:
            return
        ctx.conn.execute(
            "INSERT INTO reviews (run_id, repo, number, sha, phase, github_review_id, event, posted_at) VALUES (?,?,?,?,?,NULL,?,?)",
            (ctx.run_id, ctx.repo, ctx.number, ctx.sha, phase, new_event, now()),
        )
        _set_run_review(ctx, blocker_count=verified.blocker_count, review_id=None, review_url=None)
        event(
            ctx.conn,
            "review.verdict_recorded",
            repo=ctx.repo,
            number=ctx.number,
            run_id=ctx.run_id,
            detail=f"{prev} -> {new_event} on {ctx.sha[:8]} (not posted: transport state unchanged)",
        )


def _verdict_update(
    ctx: RunContext,
    phase: str,
    verified: VerifierOutput,
    verifier_lm: LaneModel,
    *,
    phase2_skipped: str | None,
) -> publish.PublishResult | None:
    """On a same-sha re-review, post a follow-up review when the verdict moved.

    Compares against the bot's LATEST review state for this (sha, phase), which includes
    earlier follow-ups, so the same correction is never posted twice. A review a maintainer
    dismissed is left dismissed: we only ever move from a state we set ourselves."""
    if ctx.dry_run:
        return None
    prov = publish.Provenance(
        reviewers=ctx.reviewers,
        verifier=_lane_provenance(verifier_lm, "final-verifier"),
        policy_fingerprint=ctx.cfg.policy.fingerprint,
        triage=_triage_provenance(ctx),
        phase2_skipped=phase2_skipped,
        phase1_skipped=ctx.phase1_skipped,
        phase1_choice=_phase1_choice_provenance(ctx),
        adhoc=ctx.adhoc,
        fresh_final=ctx.fresh_final,
        degraded=_degraded_provenance(ctx),
    )
    state = publish.standing_verdict(
        github.reviews(ctx.gh, ctx.repo, ctx.number), ctx.sha, phase, ctx.cfg.bot_login
    )
    if state not in {"CHANGES_REQUESTED", "COMMENTED", "APPROVED"}:
        return None  # dismissed (or unknown): a human overrode us; do not re-assert
    # compare what GitHub would record (transport), not the canonical event: on a bot-authored
    # PR both collapse to COMMENTED, and comparing the canonical event would repost forever
    assert ctx.meta
    own = ctx.meta.author.lower() == ctx.cfg.bot_login.lower()
    new_event = publish.verdict_event(verified, prov)
    if publish.EVENT_STATE[publish.transport_event(new_event, own_pr=own)] == state:
        _record_verdict(ctx, phase, verified, new_event)
        return None
    if prov.degraded and state == "APPROVED" and not verified.blocker_count:
        # a stand-in model found nothing new; that is not grounds to retract a full-strength
        # approval, and the downgrade to COMMENT is only our own degraded-mode caution
        with tx(ctx.conn):
            event(
                ctx.conn,
                "review.verdict_kept",
                repo=ctx.repo,
                number=ctx.number,
                run_id=ctx.run_id,
                detail=f"APPROVED stands on {ctx.sha[:8]}: degraded re-review found no blockers",
            )
        return None
    withdrawn = [
        str(r.get("finding_hash"))
        for r in _reconciliation(ctx, phase, verified).values()
        if r.get("status") in {"WITHDRAWN", "FIXED", "OUTDATED"}
    ]
    titles = [
        t["title"]
        for h, t in ctx.open_threads.items()
        if h in withdrawn and t.get("title") and t.get("severity") == "blocking"
    ]
    result = publish.publish_verdict_update(
        ctx.gh,
        repo=ctx.repo,
        number=ctx.number,
        head_sha=ctx.sha,
        phase=phase,
        verified=verified,
        provenance=prov,
        previous_event=state,
        withdrawn_blockers=titles,
        bot_login=ctx.cfg.bot_login,
    )
    with tx(ctx.conn):
        ctx.conn.execute(
            "INSERT INTO reviews (run_id, repo, number, sha, phase, github_review_id, event, posted_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                ctx.run_id,
                ctx.repo,
                ctx.number,
                ctx.sha,
                phase,
                result.review_id,
                result.event,
                now(),
            ),
        )
        _set_run_review(
            ctx,
            blocker_count=verified.blocker_count,
            review_id=result.review_id,
            review_url=result.review_url,
        )
        event(
            ctx.conn,
            "review.verdict_updated",
            repo=ctx.repo,
            number=ctx.number,
            run_id=ctx.run_id,
            detail=f"{state} -> {result.event} on {ctx.sha[:8]}",
        )
    return result


def _reconciliation(
    ctx: RunContext, phase: str, verified: VerifierOutput | None = None
) -> dict[str, dict[str, Any]]:
    """The `prior_finding_reconciliation` rows for the phase being published, one per hash.

    The verifier is the canonical source of truth, so its rows win outright: it is the lane
    that actually read the human's reply against the code and decided FIXED / WITHDRAWN, and
    its reason is the one that belongs on the thread. Reviewer lanes fill in only the hashes
    the verifier did not reconcile; among those, the first row with a reason wins. (Before
    this, the verifier's rows were ignored and a reviewer's STILL_VALID reason was silently
    replaced by "did not survive verification" whenever the verifier overruled it.)
    """
    merged: dict[str, dict[str, Any]] = {}
    for row in verified.prior_reconciliation if verified else []:
        merged.setdefault(str(row["finding_hash"]), row)
    outputs = ctx.phase1_outputs if phase == "preliminary" else ctx.phase2_outputs
    outputs = outputs or ctx.phase1_outputs or ctx.phase2_outputs
    for out in outputs.values():
        for row in out.prior_reconciliation:
            h = str(row.get("finding_hash") or "")
            if not h:
                continue
            if h not in merged or (row.get("reason") and not merged[h].get("reason")):
                merged[h] = row
    return merged


def _build_review(
    ctx: RunContext,
    *,
    phase: str,
    verified: VerifierOutput,
    verifier_lm: LaneModel,
    phase2_skipped: str | None = None,
    rereview: bool = False,
) -> publish.ReviewModel:
    diff = github.pr_diff(ctx.gh, ctx.repo, ctx.number)
    prov = publish.Provenance(
        reviewers=ctx.reviewers,
        verifier=_lane_provenance(
            verifier_lm, "verifier" if phase == "preliminary" else "final-verifier"
        ),
        policy_fingerprint=ctx.cfg.policy.fingerprint,
        triage=_triage_provenance(ctx),
        phase2_skipped=phase2_skipped,
        phase1_skipped=ctx.phase1_skipped,
        phase1_choice=_phase1_choice_provenance(ctx),
        adhoc=ctx.adhoc,
        fresh_final=ctx.fresh_final,
        degraded=_degraded_provenance(ctx),
    )
    note = None
    head_row = ctx.conn.execute("SELECT status FROM heads WHERE id=?", (ctx.head_id,)).fetchone()
    if head_row and head_row["status"] == "superseded":
        live = github.pr_meta(ctx.gh, ctx.repo, ctx.number).head_sha
        note = f"_This review was completed for commit `{ctx.sha[:8]}`; the PR has since moved to `{live[:8]}`. A fresh review of the new head is queued._"
    return publish.build(
        ctx.gh,
        repo=ctx.repo,
        number=ctx.number,
        head_sha=ctx.sha,
        phase=phase,
        verified=verified,
        provenance=prov,
        bot_login=ctx.cfg.bot_login,
        diff_text=diff,
        dry_run=ctx.dry_run,
        superseded_note=note,
        rereview=rereview,
    )


def _lane_provenance(lm: LaneModel, role: str) -> dict[str, Any]:
    """How a verifier/conversation lane is disclosed in the published provenance."""
    return {
        "model": lm.model,
        "agent": lm.agent,
        "role": role,
        "substitute_for": lm.substitute_for,
    }


def _degraded_provenance(ctx: RunContext) -> dict[str, Any] | None:
    """Disclosed on every publication made while stand-in models were in use."""
    pol = ctx.degraded_policy
    if pol is None or ctx.degraded is None:
        return None
    return {
        "reason": ctx.degraded.reason,
        "source": ctx.degraded.source,
        "since": ctx.degraded.since,
        "label": pol.label,
        "substitutes": {k: v.model for k, v in pol.substitutes.items()},
        "phase1_effort_cap": pol.phase1_effort_cap,
    }


def _enter_degraded(ctx: RunContext, state: degraded.State, kind: str, detail: str) -> None:
    ctx.degraded = state
    with tx(ctx.conn):
        ctx.conn.execute("UPDATE runs SET degraded=1 WHERE id=?", (ctx.run_id,))
        event(ctx.conn, kind, repo=ctx.repo, number=ctx.number, run_id=ctx.run_id, detail=detail)


def _flip_to_degraded(
    ctx: RunContext, model: str, *, error: str, upstream: str, detail: str
) -> bool:
    """`model` just failed with `error` (a lane's stderr, never model output): when that says
    the pool is out of quota and the policy has a stand-in, switch this run into degraded
    mode and record the failure with a hold, so the next runs start degraded too. False when
    the mode is already on, no stand-in applies, or the error is not about quota.

    The published reason is a sanitised version of `upstream`; the full `detail` only reaches
    the events table."""
    pol = ctx.cfg.policy.degraded
    if (
        pol is None
        or ctx.is_degraded
        or model not in pol.substitutes
        or not degraded.looks_like_quota_failure(error)
    ):
        return False
    reason = degraded.publishable_reason(model, upstream)
    degraded.record_probe(ctx.conn, quota_exhausted=True, reason=reason, hold=True)
    _enter_degraded(
        ctx,
        degraded.State(True, reason, "lane", since=now()),
        "degraded.entered_midrun",
        detail[:1000],
    )
    log.warning("lane on %s hit a quota failure; switching to stand-in models", model)
    return True


def _degrade_on_quota_failure(
    ctx: RunContext, lm: LaneModel, exc: ReviewError, res: LaneResult
) -> LaneModel | None:
    """A reviewer/verifier lane died: when it was on a primary model and the failure is a
    quota/cooldown one, flip the run into degraded mode and return the lane on its stand-in.
    None otherwise.

    Only an infrastructure failure whose *stderr* says quota counts. A contract failure is
    the model's own output and may legitimately talk about rate limits (a PR touching quota
    code), so it must never flip the whole system."""
    pol = ctx.cfg.policy.degraded
    if pol is None or lm.substitute_for is not None or exc.kind != FailKind.INFRA:
        return None
    if not _flip_to_degraded(
        ctx,
        lm.model,
        error=res.stderr or "",
        upstream=res.first_stderr_line,
        detail=f"{lm.model} lane: {exc.message}",
    ):
        return None
    return pol.resolve(lm)


def _degrade_on_side_lane_failure(ctx: RunContext, step: str, model: str, error: str) -> bool:
    """The selector/triage lanes fall back on their own (heuristic / fallback tier), which
    hides the earliest sign that the primary pool is dry. A quota-shaped failure there flips
    the run into degraded mode (with the same hold as a reviewer lane) so the caller can try
    once more on the stand-in and the rest of the run does not walk into the same 429.
    `error` is the lane's exit status plus its first stderr line (never model output)."""
    return _flip_to_degraded(
        ctx,
        model,
        error=error,
        upstream=error.split(": ", 1)[-1],
        detail=f"{step} lane: {error}",
    )


def _detect_degraded(ctx: RunContext) -> None:
    """Decide the run's mode up front (probe cached across runs) and record it."""
    if ctx.cfg.policy.degraded is None:
        return
    state = degraded.detect(ctx.conn, ctx.cfg, prober=ctx.prober or degraded.probe)
    if state.active:
        _enter_degraded(ctx, state, "degraded.run", f"{state.reason} ({state.source})")
    else:
        ctx.degraded = state


def _phase1_choice_provenance(ctx: RunContext) -> dict[str, Any] | None:
    """Only worth a line when there was a ladder to choose from."""
    if ctx.phase1_choice is None or not ctx.cfg.policy.has_phase1_ladder:
        return None
    return ctx.phase1_choice.as_dict()


def _triage_provenance(ctx: RunContext) -> dict[str, Any] | None:
    t, lm = ctx.triage, ctx.cfg.policy.triage
    if t is None or lm is None:
        return None
    ran = ctx.lane_model(lm)
    return {
        "tier": t.tier,
        "model": ran.model,
        "effort": ran.effort,
        "substitute_for": ran.substitute_for,
        "method": t.method,
        "reasoning": t.reasoning,
        "error": t.error,
    }


def _record_publication(
    ctx: RunContext,
    phase: str,
    model: publish.ReviewModel,
    result: publish.PublishResult,
    verified: VerifierOutput,
) -> None:
    with tx(ctx.conn):
        if result.posted:
            ctx.conn.execute(
                "INSERT INTO reviews (run_id, repo, number, sha, phase, github_review_id, event, posted_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    ctx.run_id,
                    ctx.repo,
                    ctx.number,
                    ctx.sha,
                    phase,
                    result.review_id,
                    result.event,
                    now(),
                ),
            )
            ctx.conn.executemany(
                "INSERT INTO posted_findings (repo, number, hash, sha, review_id, posted_at) VALUES (?,?,?,?,?,?) ON CONFLICT(repo, number, hash) DO UPDATE SET sha=excluded.sha, review_id=excluded.review_id, posted_at=excluded.posted_at",
                [
                    (ctx.repo, ctx.number, f.hash, ctx.sha, result.review_id, now())
                    for f in model.kept
                ],
            )
            _insert_findings(ctx, phase, "posted", model.kept)
        _set_run_review(
            ctx,
            blocker_count=verified.blocker_count,
            review_id=result.review_id,
            review_url=result.review_url,
        )


# ---- orchestration ----


def _reviewer_step(
    ctx: RunContext,
    *,
    step: StepName,
    phase: str,
    lm: LaneModel | Callable[[], LaneModel],
    expected_phase: str,
    fresh: bool = False,
) -> dict[str, ReviewerOutput]:
    _step_start(ctx, step)
    if callable(lm):
        lm = lm()  # Phase 1 picks its model inside the step, so a slow lookup shows there
    outputs = _reviewer_lanes(ctx, phase=phase, lm=lm, expected_phase=expected_phase, fresh=fresh)
    _step_end(ctx, step, "ok", {"roles": list(outputs)})
    return outputs


def _verify_step(
    ctx: RunContext,
    *,
    step: StepName,
    phase: str,
    lm: LaneModel,
    expected_phase: str,
    phase2_outputs: dict[str, ReviewerOutput] | None = None,
    fresh_final: bool = False,
) -> VerifierOutput:
    _step_start(ctx, step)
    out = _verifier_lane(
        ctx,
        phase=phase,
        lm=lm,
        expected_phase=expected_phase,
        phase2_outputs=phase2_outputs,
        fresh_final=fresh_final,
    )
    _step_end(ctx, step, "ok", {"blockers": out.blocker_count, "findings": len(out.findings)})
    return out


def _publish_step(
    ctx: RunContext,
    *,
    phase: str,
    verified: VerifierOutput,
    verifier_lm: LaneModel,
    phase2_skipped: str | None = None,
) -> publish.PublishResult:
    _step_start(ctx, StepName.PUBLISH)
    res = step_publish(
        ctx, phase=phase, verified=verified, verifier_lm=verifier_lm, phase2_skipped=phase2_skipped
    )
    _step_end(
        ctx,
        StepName.PUBLISH,
        "ok",
        {"posted": res.posted, "event": res.event, "skipped": res.skipped_reason},
    )
    _gate_comment(
        ctx,
        "done",
        phase=phase,
        blocker_count=verified.blocker_count,
        phase2_skipped=phase2_skipped,
    )
    return res


def _fresh_final_if_needed(
    ctx: RunContext, *, verified: VerifierOutput, effort: Any
) -> VerifierOutput:
    """Run the independent final gate after an iterative review has cleared blockers.

    Prior findings and discussion remain available to the verifier, while the fresh
    Phase-2 reviewer lanes receive no prior-finding checklist and inspect the complete
    current merge-base range independently.
    """
    if (
        not ctx.has_prior_review
        or verified.blocker_count
        or not ctx.cfg.policy.phase2_enabled
        or effort.phase2 is None
    ):
        return verified
    ctx.fresh_final = True
    fresh_outputs = _reviewer_step(
        ctx,
        step=StepName.FRESH_PHASE2,
        phase="phase2",
        lm=ctx.lane_model(_with_effort(ctx.cfg.policy.phase2_reviewer, effort.phase2)),
        expected_phase="final",
        fresh=True,
    )
    combined = dict(ctx.phase2_outputs)
    combined.update({f"fresh:{role}": output for role, output in fresh_outputs.items()})
    return _verify_step(
        ctx,
        step=StepName.FRESH_VERIFY2,
        phase="verify2",
        lm=ctx.lane_model(ctx.cfg.policy.phase2_verifier),
        expected_phase="final",
        phase2_outputs=combined,
        fresh_final=True,
    )


def _with_effort(lm: LaneModel, effort: str | None) -> LaneModel:
    return dataclasses.replace(lm, effort=effort) if effort else lm


def _rung(ctx: RunContext, choice: quota.Choice, kind: str) -> LaneModel:
    """Adopt `choice`: record it, and clamp the tier's Phase-1 effort to the rung's cap."""
    ctx.phase1_choice = choice
    if ctx.cfg.policy.has_phase1_ladder:
        with tx(ctx.conn):
            event(
                ctx.conn,
                kind,
                repo=ctx.repo,
                number=ctx.number,
                run_id=ctx.run_id,
                detail=choice.log_line(),
            )
    cap = choice.model.effort
    return ctx.lane_model(_with_effort(choice.model, min_effort(ctx.phase1_effort or cap, cap)))


def _choose_phase1(ctx: RunContext, effort: str) -> LaneModel:
    """Pick the Phase-1 model from the policy's ladder by remaining subscription quota
    (see `quota.py`); recorded on the run and disclosed in the review provenance."""
    pol = ctx.cfg.policy
    dp = ctx.degraded_policy
    if dp is not None:
        # keep the included-quota rungs eligible (GLM is passed over above `high`) rather
        # than sending every Phase 1 to the paid last rung while Phase 2 is already there
        effort = dp.phase1_effort(effort)
    ctx.phase1_effort = effort
    reader = ctx.quota_reader or quota.cached_reader(ctx.conn)
    choice = quota.choose(pol.phase1_candidates, pol.quota_reserve, reader=reader, effort=effort)
    return _rung(ctx, choice, "phase1.model_selected")


def _phase1_fallback(ctx: RunContext, failed: LaneModel, exc: ReviewError) -> LaneModel | None:
    """A Phase-1 lane died on its rung (rate limit, dead upstream, malformed output twice):
    move down the ladder for this and the remaining roles. None when there is nowhere to go."""
    pol, choice = ctx.cfg.policy, ctx.phase1_choice
    if choice is None:
        return None
    lower = choice.remaining(pol.phase1_candidates)
    if not lower:
        return None
    log.warning("phase1 lane on %s failed (%s); falling down the ladder", failed.model, exc)
    skipped = (*choice.skipped, quota.Skipped(failed.model, "lane failed", str(exc)[:600]))
    reader = ctx.quota_reader or quota.cached_reader(ctx.conn)
    nxt = quota.choose(
        lower, pol.quota_reserve, reader=reader, skipped=skipped, effort=ctx.phase1_effort
    )
    return _rung(ctx, nxt, "phase1.model_fallback")


def _backlog_skips_phase1(ctx: RunContext, *, phase2_effort: str | None) -> bool:
    """Throughput rule: with a deep queue, skip the slow Phase-1 reviewers and go straight to Phase 2.

    Only when Phase 2 is enabled and the tier would have run it (a trivial tier has no Phase 2 to
    fall through to, so it keeps its Phase-1-only path). Recorded as a `phase1` step with status
    `skipped`, an event, and disclosed in the review provenance and the gate comment.
    """
    limit = ctx.cfg.backlog_skip_phase1_above
    if limit <= 0 or not ctx.cfg.policy.phase2_enabled or phase2_effort is None:
        return False
    dp = ctx.degraded_policy
    if dp is not None and not dp.backlog_skip_phase1:
        # opt-in: keep both phases in degraded mode so the cross-model check survives, at the
        # cost of the slow Phase-1 rungs. Off by default -- a deep queue in degraded mode is
        # the case that can least afford them.
        return False
    queued = ctx.conn.execute("SELECT COUNT(*) AS n FROM heads WHERE status='queued'").fetchone()[
        "n"
    ]
    if queued <= limit:
        return False
    reason = f"skipped for throughput: {queued} PRs queued, above the {limit} limit"
    ctx.phase1_skipped = reason
    _step_start(ctx, StepName.PHASE1)
    _step_end(ctx, StepName.PHASE1, "skipped", {"reason": reason, "queued": queued, "limit": limit})
    with tx(ctx.conn):
        event(
            ctx.conn,
            "phase1.skipped_backlog",
            repo=ctx.repo,
            number=ctx.number,
            run_id=ctx.run_id,
            detail=reason,
        )
    _gate_comment(ctx, "in_progress")
    return True


# ---- conversation mode: answer replies on an already-reviewed commit ----

LINKED_COMMIT_LIMIT = 5


def _standing_review(ctx: RunContext) -> dict[str, Any] | None:
    """The bot's FINAL review of this exact commit, or None. A preliminary review (blockers
    found, Phase 2 deferred) does not qualify: a reply that talks a blocker down there must
    re-run the pipeline so Phase 2 and the final verdict still happen."""
    r = github.existing_review_for_sha(
        ctx.gh, ctx.repo, ctx.number, ctx.sha, "final", ctx.cfg.bot_login
    )
    return {**r, "phase": "final"} if r else None


def _fetch_linked_commits(
    ctx: RunContext, threads: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """(fetched, unfetched) commits humans linked in the replied threads. Only commits from a
    fork of this repository are fetched (the delta against the mirror is small); anything else
    is reported to the model as unavailable with the reason. Fetched ones land in the shared
    object store, so `git show <sha>` works inside the run's worktree."""
    fetched: list[dict[str, str]] = []
    unfetched: list[dict[str, str]] = []
    if not ctx.mirror:
        return fetched, unfetched
    own_name = ctx.repo.split("/", 1)[1].lower()
    for i, c in enumerate(converse.linked_commits(threads)):
        if i >= LINKED_COMMIT_LIMIT:
            unfetched.append({**c, "reason": f"more than {LINKED_COMMIT_LIMIT} commits linked"})
        elif c["repo"].lower() != own_name:
            unfetched.append({**c, "reason": "not a fork of this repository"})
        elif wt.fetch_commit(ctx.mirror, c["url"], c["sha"]):
            fetched.append(c)
        else:
            unfetched.append({**c, "reason": "fetch failed; private or deleted fork?"})
    return fetched, unfetched


def _silent_key(ctx: RunContext, h: str) -> str:
    return f"converse.silent:{ctx.repo}#{ctx.number}:{h}"


def _threads_to_answer(ctx: RunContext) -> dict[str, dict[str, Any]]:
    """Threads with a human reply newer than our last answer, minus those where we already
    considered that exact reply and chose silence (recorded in kv per finding)."""
    out: dict[str, dict[str, Any]] = {}
    for h, t in ctx.open_threads.items():
        if not t.get("awaiting_answer"):
            continue
        if kv_get(ctx.conn, _silent_key(ctx, h)) == str(t.get("latest_reply_id")):
            continue
        out[h] = t
    return out


def _run_conversation_lane(
    ctx: RunContext, lm: LaneModel, threads: dict[str, dict[str, Any]], **prompt_kw: Any
) -> converse.ConversationOutput:
    """Two semantic attempts (each `_run_lane` already retries transport/JSON failures once);
    the second attempt is told what was wrong with the first."""
    last = ""
    for _ in range(2):
        prompt = converse.prompt(threads=threads, previous_error=last, **prompt_kw)
        raw = _run_lane(
            ctx, phase="converse", role="conversation", lm=lm, prompt=prompt, is_verifier=True
        )
        try:
            return converse.parse(raw, expected=set(threads))
        except ReviewError as exc:
            last = exc.message
    raise ReviewError(FailKind.CONTRACT, f"conversation lane output invalid twice: {last}")


def step_converse(ctx: RunContext, standing: dict[str, Any]) -> dict[str, Any]:
    assert ctx.meta and ctx.worktree
    phase = str(standing["phase"])
    threads = _threads_to_answer(ctx)
    reviews = github.reviews(ctx.gh, ctx.repo, ctx.number)
    if not threads:
        # the reply came from another bot, landed on a resolved thread, or was already
        # considered: nothing to say, restore the gate comment we overwrote at run start
        remaining = _open_blockers(ctx, phase)
        _gate_comment(ctx, "done", phase=phase, blocker_count=remaining)
        return {"answered": 0, "reason": "no thread awaiting an answer"}
    fetched, unfetched = _fetch_linked_commits(ctx, threads)
    project_skill, review_skill = skill_texts(ctx.cfg, ctx.repo)
    lm = ctx.lane_model(ctx.cfg.policy.conversation_lane)
    out = _run_conversation_lane(
        ctx,
        lm,
        threads,
        repo=ctx.repo,
        number=ctx.number,
        head_sha=ctx.sha,
        meta=ctx.meta.as_dict(),
        project_skill=project_skill,
        review_skill=review_skill,
        fetched_commits=fetched,
        unfetched_commits=unfetched,
        standing_verdict=publish.standing_verdict(reviews, ctx.sha, phase, ctx.cfg.bot_login),
        evidence=ctx.evidence,
    )
    (ctx.run_dir / "conversation.json").write_text(
        json.dumps({h: dataclasses.asdict(o) for h, o in out.outcomes.items()}, indent=1)
    )
    # same guard as step_publish: a push may have landed while the lane ran, and a verdict
    # follow-up must never be posted against an obsolete commit (the scheduler supersedes
    # this head and ingest queues the live one)
    live = github.pr_meta(ctx.gh, ctx.repo, ctx.number)
    if live.head_sha != ctx.sha:
        raise ReviewError(
            FailKind.FATAL, f"live head {live.head_sha[:8]} != assigned {ctx.sha[:8]}"
        )
    answered: list[dict[str, Any]] = []
    if not ctx.dry_run:
        answered = publish.answer_conversation(
            ctx.gh, ctx.repo, ctx.number, ctx.sha, threads=threads, outcomes=out.outcomes
        )
    # Posting happens before the bookkeeping below is committed. If the worker dies in
    # between, the reply is public but no `conceded` row exists, and the requeued run will not
    # revisit the thread (our own comment is now its newest). That blocker then keeps counting
    # until a fresh review of a new push resets the standing set; accepted rather than risking
    # the reverse (a recorded concession that never reached the thread).
    posted_ok = {a["finding_hash"] for a in answered if a.get("action") == "replied"}
    considered = {a["finding_hash"] for a in answered}
    # only an outcome that actually reached the thread lifts a blocker or counts as silence
    lifted = {h for h in posted_ok if out.outcomes[h].status in {"WITHDRAWN", "FIXED"}}
    silent = {h for h, o in out.outcomes.items() if o.status == "NO_REPLY" and h in considered}
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
        for h in silent:
            kv_set(ctx.conn, _silent_key(ctx, h), str(threads[h].get("latest_reply_id")))
        _record_conceded(ctx, phase, lifted)
    remaining = _open_blockers(ctx, phase, lifted=lifted)
    update = None
    if not ctx.dry_run:
        update = _conversation_verdict_update(
            ctx, phase, lm, reviews, remaining=remaining, lifted=lifted
        )
    _gate_comment(ctx, "done", phase=phase, blocker_count=remaining)
    return {
        "answered": len(posted_ok),
        "outcomes": {h: o.status for h, o in out.outcomes.items()},
        "linked_commits": {
            "fetched": [c["sha"] for c in fetched],
            "unfetched": [c["sha"] for c in unfetched],
        },
        "blockers_remaining": remaining,
        "verdict_updated": bool(update and update.posted),
    }


def _record_conceded(ctx: RunContext, phase: str, lifted: set[str]) -> None:
    """A finding the conversation withdrew or confirmed fixed gets a `conceded` findings row
    for this run (and so this sha), which `_open_blockers` honours on every later run. Caller
    holds the write transaction."""
    if not lifted:
        return
    rows = []
    for h in lifted:
        t = ctx.open_threads.get(h) or {}
        rows.append(
            (
                ctx.run_id,
                phase,
                "conceded",
                h,
                t.get("path") or "",
                t.get("line"),
                t.get("line"),
                t.get("severity") or "nitpick",
                None,
                "general",
                t.get("title") or h,
                "",
            )
        )
    ctx.conn.executemany(_FINDINGS_INSERT, rows)


def _open_blockers(ctx: RunContext, phase: str, *, lifted: set[str] | None = None) -> int:
    """Blocking findings on this commit's `phase` publication that still stand.

    Standing set: the blockers in the most recent `phase` publication for this sha (the latest
    run's `posted`-stage rows, which a same-sha re-review refreshes to the verifier's kept
    set), plus unresolved blocking threads on GitHub this database has no row for (legacy
    findings). Minus: findings a conversation on this sha already conceded (`conceded` rows),
    and those lifted by the current run. A blocking thread a maintainer resolved by hand,
    without the bot conceding, still counts: only an explicit outcome or a fresh review lifts
    a verdict. Scoped to the phase whose verdict is being moved, so a preliminary publication
    stacked on the same sha by a manual re-review never leaks into the final accounting.
    """
    rows = ctx.conn.execute(
        "SELECT f.run_id, f.hash, f.severity, f.stage FROM findings f JOIN runs r ON r.id=f.run_id "
        "JOIN heads h ON h.id=r.head_id WHERE h.repo=? AND h.number=? AND h.sha=? AND f.phase=? "
        "AND f.stage IN ('posted','conceded')",
        (ctx.repo, ctx.number, ctx.sha, phase),
    ).fetchall()
    known = {str(r["hash"]) for r in rows}
    conceded = {str(r["hash"]) for r in rows if r["stage"] == "conceded"}
    posted = [r for r in rows if r["stage"] == "posted"]
    latest_posted_run = max((int(r["run_id"]) for r in posted), default=None)
    standing = {
        str(r["hash"])
        for r in posted
        if int(r["run_id"]) == latest_posted_run and r["severity"] == "blocking"
    }
    standing |= {
        h for h, t in ctx.open_threads.items() if t.get("severity") == "blocking" and h not in known
    }
    return len(standing - conceded - (lifted or set()))


def _conversation_verdict_update(
    ctx: RunContext,
    phase: str,
    lm: LaneModel,
    reviews: list[dict[str, Any]],
    *,
    remaining: int,
    lifted: set[str],
) -> publish.PublishResult | None:
    """When the discussion withdrew or resolved every blocking finding on this commit, the
    standing REQUEST_CHANGES is stale: post the same short follow-up review a re-review
    would, disclosing that no code was re-reviewed. A conversation never approves and never
    adds blockers, so this is the only direction it can move a verdict."""
    if remaining or not lifted:
        return None
    state = publish.standing_verdict(reviews, ctx.sha, phase, ctx.cfg.bot_login)
    if state != "CHANGES_REQUESTED":
        return None
    prov = publish.Provenance(
        reviewers=[],
        verifier=_lane_provenance(lm, "conversation"),
        policy_fingerprint=ctx.cfg.policy.fingerprint,
        adhoc=ctx.adhoc,
        conversation=True,
        degraded=_degraded_provenance(ctx),
    )
    verified = VerifierOutput(
        summary="",
        review_action="COMMENT",
        findings=[],
        dropped=[],
        out_of_scope=[],
        coderabbit_reactions=[],
        prerequisite_adjudications=[],
        adjudication_complete=True,
        review_phase=phase,
        raw={},
    )
    titles = [
        t["title"]
        for h, t in ctx.open_threads.items()
        if h in lifted and t.get("severity") == "blocking" and t.get("title")
    ]
    result = publish.publish_verdict_update(
        ctx.gh,
        repo=ctx.repo,
        number=ctx.number,
        head_sha=ctx.sha,
        phase=phase,
        verified=verified,
        provenance=prov,
        previous_event=state,
        withdrawn_blockers=titles,
        bot_login=ctx.cfg.bot_login,
    )
    with tx(ctx.conn):
        ctx.conn.execute(
            "INSERT INTO reviews (run_id, repo, number, sha, phase, github_review_id, event, posted_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                ctx.run_id,
                ctx.repo,
                ctx.number,
                ctx.sha,
                phase,
                result.review_id,
                result.event,
                now(),
            ),
        )
        _set_run_review(
            ctx, blocker_count=0, review_id=result.review_id, review_url=result.review_url
        )
        event(
            ctx.conn,
            "review.verdict_updated",
            repo=ctx.repo,
            number=ctx.number,
            run_id=ctx.run_id,
            detail=f"{state} -> {result.event} on {ctx.sha[:8]} (conversation: every blocker withdrawn or resolved)",
        )
    return result


def run(ctx: RunContext) -> RunStatus:
    pol = ctx.cfg.policy
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    _detect_degraded(ctx)
    _gate_comment(ctx, "in_progress")
    standing = _standing_review(ctx) if ctx.trigger == Trigger.REVIEW_REPLY else None
    for name, fn in (
        (StepName.WORKTREE, step_worktree),
        (StepName.SELECT, step_select),
        (StepName.TRIAGE, step_triage),
        (StepName.CONTEXT, step_context),
    ):
        if name == StepName.TRIAGE and pol.triage is None:
            continue
        if standing and name in (StepName.SELECT, StepName.TRIAGE):
            continue  # a conversation needs no specialist selection or effort triage
        ctx.check_cancel()
        _step_start(ctx, name)
        fn(ctx)
        detail: dict[str, Any] | None = None
        if name == StepName.SELECT:
            detail = {"selection": ctx.selection}
        elif name == StepName.TRIAGE and ctx.triage:
            detail = dataclasses.asdict(ctx.triage)
        _step_end(ctx, name, "ok", detail)
    if pol.pipeline is not None:
        from . import pipeline_v10

        return pipeline_v10.run(ctx, reply_only=standing is not None)
    if standing:
        # a human replied on a commit we already reviewed: the code did not change, the
        # discussion did. Answer the threads; never re-run the review pipeline for that.
        _step_start(ctx, StepName.CONVERSE)
        detail = step_converse(ctx, standing)
        _step_end(ctx, StepName.CONVERSE, "ok", detail)
        return RunStatus.DONE
    effort = pol.tier_effort(ctx.tier)
    if _backlog_skips_phase1(ctx, phase2_effort=effort.phase2):
        ctx.phase2_outputs = _reviewer_step(
            ctx,
            step=StepName.PHASE2,
            phase="phase2",
            lm=ctx.lane_model(_with_effort(pol.phase2_reviewer, effort.phase2)),
            expected_phase="final",
        )
        ctx.verify2 = _verify_step(
            ctx,
            step=StepName.VERIFY2,
            phase="verify2",
            lm=ctx.lane_model(pol.phase2_verifier),
            expected_phase="final",
        )
        ctx.verify2 = _fresh_final_if_needed(ctx, verified=ctx.verify2, effort=effort)
        _publish_step(
            ctx,
            phase="final",
            verified=ctx.verify2,
            verifier_lm=ctx.lane_model(pol.phase2_verifier),
        )
        return RunStatus.DONE
    ctx.phase1_outputs = _reviewer_step(
        ctx,
        step=StepName.PHASE1,
        phase="phase1",
        lm=lambda: _choose_phase1(ctx, effort.phase1),
        expected_phase="preliminary",
    )
    ctx.verify1 = _verify_step(
        ctx,
        step=StepName.VERIFY1,
        phase="verify1",
        lm=ctx.lane_model(pol.phase1_verifier),
        expected_phase="preliminary",
    )
    _step_start(ctx, StepName.GATE)
    tier_allows = effort.phase2 is not None
    admit = admit_phase2(ctx.verify1, phase2_enabled=pol.phase2_enabled, tier_allows=tier_allows)
    _step_end(
        ctx,
        StepName.GATE,
        "ok",
        {"admit_phase2": admit, "tier": ctx.tier, "phase2_effort": effort.phase2},
    )
    if not admit:
        ctx.check_cancel()
        verifier1 = ctx.lane_model(pol.phase1_verifier)
        if ctx.verify1.blocker_count or not pol.phase2_enabled:
            _publish_step(ctx, phase="preliminary", verified=ctx.verify1, verifier_lm=verifier1)
        else:
            # no blockers and the tier says a second round adds nothing: final from Phase 1
            _publish_step(
                ctx,
                phase="final",
                verified=ctx.verify1,
                verifier_lm=verifier1,
                phase2_skipped=f"triage rated this change {ctx.tier}",
            )
        return RunStatus.DONE
    ctx.phase2_outputs = _reviewer_step(
        ctx,
        step=StepName.PHASE2,
        phase="phase2",
        lm=ctx.lane_model(_with_effort(pol.phase2_reviewer, effort.phase2)),
        expected_phase="final",
    )
    ctx.verify2 = _verify_step(
        ctx,
        step=StepName.VERIFY2,
        phase="verify2",
        lm=ctx.lane_model(pol.phase2_verifier),
        expected_phase="final",
    )
    ctx.verify2 = _fresh_final_if_needed(ctx, verified=ctx.verify2, effort=effort)
    _publish_step(
        ctx, phase="final", verified=ctx.verify2, verifier_lm=ctx.lane_model(pol.phase2_verifier)
    )
    return RunStatus.DONE


def cleanup(ctx: RunContext, status: RunStatus) -> None:
    if ctx.worktree and ctx.mirror and status == RunStatus.DONE:
        wt.remove_worktree(ctx.mirror, ctx.worktree)
        shutil.rmtree(ctx.worktree, ignore_errors=True)


def main(
    cfg: Config,
    conn: sqlite3.Connection,
    run_id: int,
    *,
    gh: Gh | None = None,
    lane_runner: LaneRunner | None = None,
    dry_run: bool = False,
    heartbeat: bool = True,
    quota_reader: quota.QuotaReader | None = None,
    prober: Prober | None = None,
) -> RunStatus:
    row = conn.execute(
        "SELECT r.*, h.repo, h.number, h.sha, h.trigger FROM runs r JOIN heads h ON h.id=r.head_id WHERE r.id=?",
        (run_id,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"run {run_id} not found")
    if RunStatus(row["status"]).terminal:
        return RunStatus(row["status"])
    ctx = RunContext(
        cfg=cfg,
        conn=conn,
        gh=gh or Gh(cfg.gh_bin),
        run_id=run_id,
        head_id=int(row["head_id"]),
        repo=str(row["repo"]),
        number=int(row["number"]),
        sha=str(row["sha"]),
        token=str(row["token"]),
        run_dir=cfg.runs_dir / f"run-{run_id}",
        trigger=str(row["trigger"] or ""),
        dry_run=dry_run,
    )
    if lane_runner:
        ctx.lane_runner = lane_runner
    ctx.quota_reader = quota_reader
    ctx.prober = prober
    with tx(conn):
        claimed = conn.execute(
            "UPDATE runs SET heartbeat_at=?, status='running' WHERE id=? AND token=? AND status IN ('spawned','running')",
            (now(), run_id, ctx.token),
        ).rowcount
    if not claimed:
        # reaped or replaced between our read and this write; never resurrect a terminal run
        return RunStatus(
            conn.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()["status"]
        )
    stop = threading.Event()
    hb = (
        threading.Thread(target=_heartbeat_loop, args=(ctx, stop), daemon=True)
        if heartbeat
        else None
    )
    if hb:
        hb.start()
    status = RunStatus.FAILED
    reason, kind = "", FailKind.INFRA
    t0 = time.monotonic()
    try:
        status = run(ctx)
        reason = f"ok in {int(time.monotonic() - t0)}s"
        kind = FailKind.INFRA
    except Cancelled:
        status, reason = RunStatus.CANCELLED, "cancel requested"
    except ReviewError as exc:
        status, reason, kind = RunStatus.FAILED, exc.message, exc.kind
        current = ctx.conn.execute("SELECT phase FROM runs WHERE id=?", (run_id,)).fetchone()
        if current and current["phase"]:
            with contextlib.suppress(ValueError):
                _step_end(ctx, StepName(current["phase"]), "failed", {"error": exc.message[:1000]})
        _gate_comment(ctx, "failed", reason=exc.message[:200])
    except Exception as exc:
        status, reason, kind = RunStatus.FAILED, f"{type(exc).__name__}: {exc}", FailKind.INFRA
        log.exception("worker crashed")
    finally:
        stop.set()
        if hb:
            hb.join(timeout=5)
        finish_run(
            conn,
            cfg,
            run_id,
            status,
            reason=reason,
            fail_kind=None if status == RunStatus.DONE else kind,
        )
        cleanup(ctx, status)
    return status
