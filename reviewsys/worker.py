"""Worker: runs one review (one `runs` row) end to end.

Steps: worktree -> select -> context -> phase1 -> verify1 -> gate -> phase2 -> verify2 -> publish.
Every step is recorded in `steps`. Any ReviewError ends the run as failed with
its classification; the scheduler decides on retry. Cooperative cancellation
is checked on every heartbeat.
"""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import github, publish
from .config import Config, LaneModel
from .contract import (
    Finding,
    ReviewerOutput,
    VerifierOutput,
    parse_json_object,
    parse_reviewer_output,
    parse_verifier_output,
)
from .db import event, now, tx
from .gate import admit_phase2
from .gh import Gh
from .lane import LaneResult, LaneRunner, LaneSpec, lane_output, prompt_sha, run_claude_lane
from .models import FailKind, ReviewError, RunStatus, StepName
from .prompts import REPAIR_PROMPT, prior_for_prompt, reviewer_prompt, verifier_prompt
from .scheduler import finish_run
from .select import select, write_selection
from .steps import worktree as wt

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
    attempt: int
    token: str
    run_dir: Path
    worktree: Path | None = None
    mirror: Path | None = None
    meta: github.PrMeta | None = None
    selection: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    coderabbit: dict[str, Any] = field(default_factory=dict)
    coderabbit_ids: list[int] = field(default_factory=list)
    prior: list[dict[str, Any]] = field(default_factory=list)
    prior_sha: str | None = None
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
            "INSERT INTO lanes (run_id, phase, role, agent, model, attempt, attempt_id, status, exit_code, tokens_in, tokens_out, started_at, finished_at, artifact_dir, prompt_sha, reason) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                ctx.run_id,
                phase,
                role,
                lm.agent,
                lm.model,
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


def _record_findings(ctx: RunContext, phase: str, stage: str, findings: list[Finding]) -> None:
    with tx(ctx.conn):
        for f in findings:
            ctx.conn.execute(
                "INSERT INTO findings (run_id, phase, stage, hash, file, line_start, line_end, severity, confidence, category, title, body) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
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
                ),
            )


def _gate_comment(ctx: RunContext, status: str, **kw: Any) -> None:
    if ctx.dry_run:
        return
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
    files = [str(f.get("filename")) for f in files_raw if isinstance(f, dict)]
    sel = select(
        ctx.cfg,
        repo=ctx.repo,
        title=ctx.meta.title,
        body=ctx.meta.body,
        files=files,
        run_dir=ctx.run_dir,
        worktree=ctx.worktree,
        runner=ctx.lane_runner,
    )
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
    # prior findings: last posted set for this PR from our DB
    rows = ctx.conn.execute(
        "SELECT pf.hash, pf.sha, f.file, f.line_start, f.line_end, f.severity, f.category, f.title FROM posted_findings pf LEFT JOIN findings f ON f.hash=pf.hash AND f.stage='posted' WHERE pf.repo=? AND pf.number=? ORDER BY pf.posted_at DESC",
        (ctx.repo, ctx.number),
    ).fetchall()
    seen: set[str] = set()
    prior: list[Finding] = []
    for r in rows:
        if r["hash"] in seen or not r["title"]:
            continue
        seen.add(r["hash"])
        ctx.prior_sha = ctx.prior_sha or r["sha"]
        prior.append(
            Finding(
                file=r["file"] or "",
                title=r["title"],
                body="",
                severity=r["severity"] or "nitpick",
                category=r["category"] or "general",
                line_start=r["line_start"],
                line_end=r["line_end"],
            )
        )
    ctx.prior = prior_for_prompt(prior, ctx.prior_sha or "") if prior else []
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
    expected_phase: str,
    is_verifier: bool,
) -> dict[str, Any]:
    """Run a lane with bounded retries and one cheap JSON repair. Returns parsed output."""
    assert ctx.worktree
    last: str = ""
    for attempt in (1, 2):
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
        )
        res = ctx.lane_runner(spec, art, ctx.worktree)
        psha = prompt_sha(prompt)
        try:
            out = lane_output(res)
        except ReviewError as exc:
            if exc.kind == FailKind.CONTRACT and res.ok and res.result_text.strip():
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
                    "status": "completed",
                    "phase": phase,
                    "attempt_id": attempt_id,
                }
            )
        return out
    raise ReviewError(
        FailKind.INFRA if "timed out" in last or "exit" in last else FailKind.CONTRACT,
        f"{phase}/{role} lane failed twice: {last}",
    )


def _repair(ctx: RunContext, raw: str, art: Path) -> dict[str, Any] | None:
    if len(raw) > 200_000 or not ctx.worktree:
        return None
    spec = LaneSpec(
        role="repair",
        agent="repair",
        model=ctx.cfg.policy.repair_model,
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


def _reviewer_lanes(
    ctx: RunContext, *, phase: str, lm: LaneModel, expected_phase: str
) -> dict[str, ReviewerOutput]:
    assert ctx.meta
    roles = ["general", *ctx.selection]
    outputs: dict[str, ReviewerOutput] = {}
    prior_hashes = {str(p["finding_hash"]) for p in ctx.prior}
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
            prior=ctx.prior,
            prior_sha=ctx.prior_sha,
        )
        raw = _run_lane(
            ctx,
            phase=phase,
            role=role,
            lm=lm,
            prompt=prompt,
            expected_phase=expected_phase,
            is_verifier=False,
        )
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
    p1: dict[str, ReviewerOutput],
    p2: dict[str, ReviewerOutput],
) -> VerifierOutput:
    prompt = verifier_prompt(
        ctx.cfg,
        repo=ctx.repo,
        number=ctx.number,
        head_sha=ctx.sha,
        phase=expected_phase,
        phase1_outputs={r: o.raw for r, o in p1.items()},
        phase2_outputs={r: o.raw for r, o in p2.items()},
        coderabbit=ctx.coderabbit,
        coderabbit_ids=ctx.coderabbit_ids,
        evidence=ctx.evidence,
        prior=ctx.prior,
        prior_sha=ctx.prior_sha,
    )
    raw = _run_lane(
        ctx,
        phase=phase,
        role="verifier",
        lm=lm,
        prompt=prompt,
        expected_phase=expected_phase,
        is_verifier=True,
    )
    out = parse_verifier_output(
        raw, expected_phase=expected_phase, expected_coderabbit_ids=ctx.coderabbit_ids
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
        for f in verified.findings:
            ctx.conn.execute(
                "INSERT INTO posted_findings (repo, number, hash, sha, review_id, posted_at) VALUES (?,?,?,?,?,?) ON CONFLICT(repo, number, hash) DO NOTHING",
                (ctx.repo, ctx.number, f.hash, ctx.sha, review_id, ts),
            )
            ctx.conn.execute(
                "INSERT INTO findings (run_id, phase, stage, hash, file, line_start, line_end, severity, confidence, category, title, body) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ctx.run_id,
                    phase,
                    "posted",
                    f.hash,
                    f.file,
                    f.line_start,
                    f.line_end,
                    f.severity,
                    f.confidence,
                    f.category,
                    f.title,
                    f.body[:8000],
                ),
            )
        ctx.conn.execute(
            "UPDATE runs SET blocker_count=?, review_id=?, review_url=? WHERE id=?",
            (
                verified.blocker_count,
                review_id,
                str(existing.get("html_url") or "") or None,
                ctx.run_id,
            ),
        )


def step_publish(
    ctx: RunContext, *, phase: str, verified: VerifierOutput, verifier_lm: LaneModel
) -> publish.PublishResult:
    existing = github.existing_review_for_sha(
        ctx.gh, ctx.repo, ctx.number, ctx.sha, phase, ctx.cfg.bot_login
    )
    if existing:
        # A prior attempt posted but died before recording it. Backfill so future rounds
        # still see these findings as "prior" and never post again for this sha/phase.
        _backfill_posted(ctx, phase, existing, verified)
        return publish.PublishResult(
            False,
            "COMMENT",
            "COMMENT",
            int(existing.get("id") or 0) or None,
            str(existing.get("html_url") or "") or None,
            "",
            [],
            [],
            skipped_reason="already_published_for_sha",
        )
    diff = github.pr_diff(ctx.gh, ctx.repo, ctx.number)
    prov = publish.Provenance(
        reviewers=ctx.reviewers,
        verifier={
            "model": verifier_lm.model,
            "agent": verifier_lm.agent,
            "role": "verifier" if phase == "preliminary" else "final-verifier",
        },
        policy_fingerprint=ctx.cfg.policy.fingerprint,
    )
    note = None
    head_row = ctx.conn.execute("SELECT status FROM heads WHERE id=?", (ctx.head_id,)).fetchone()
    if head_row and head_row["status"] == "superseded":
        live = github.pr_meta(ctx.gh, ctx.repo, ctx.number).head_sha
        note = f"_This review was completed for commit `{ctx.sha[:8]}`; the PR has since moved to `{live[:8]}`. A fresh review of the new head is queued._"
    model = publish.build(
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
    )
    (ctx.run_dir / f"review-{phase}.md").write_text(publish.render(model))
    result = publish.publish(ctx.gh, model, bot_login=ctx.cfg.bot_login, dry_run=ctx.dry_run)
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
            for f in model.kept:
                ctx.conn.execute(
                    "INSERT INTO posted_findings (repo, number, hash, sha, review_id, posted_at) VALUES (?,?,?,?,?,?) ON CONFLICT(repo, number, hash) DO UPDATE SET sha=excluded.sha, review_id=excluded.review_id, posted_at=excluded.posted_at",
                    (ctx.repo, ctx.number, f.hash, ctx.sha, result.review_id, now()),
                )
            for f in model.kept:
                ctx.conn.execute(
                    "INSERT INTO findings (run_id, phase, stage, hash, file, line_start, line_end, severity, confidence, category, title, body) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        ctx.run_id,
                        phase,
                        "posted",
                        f.hash,
                        f.file,
                        f.line_start,
                        f.line_end,
                        f.severity,
                        f.confidence,
                        f.category,
                        f.title,
                        f.body[:8000],
                    ),
                )
        ctx.conn.execute(
            "UPDATE runs SET blocker_count=?, review_id=?, review_url=? WHERE id=?",
            (verified.blocker_count, result.review_id, result.review_url, ctx.run_id),
        )
    if result.posted and verified.coderabbit_reactions and not ctx.dry_run:
        publish.post_coderabbit_reactions(
            ctx.gh, ctx.repo, ctx.number, verified.coderabbit_reactions, ctx.cfg.bot_login
        )
    return result


# ---- orchestration ----


def run(ctx: RunContext) -> RunStatus:
    pol = ctx.cfg.policy
    steps: list[tuple[StepName, Any]] = []
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    _gate_comment(ctx, "in_progress")
    try:
        for name, fn in (
            (StepName.WORKTREE, step_worktree),
            (StepName.SELECT, step_select),
            (StepName.CONTEXT, step_context),
        ):
            ctx.check_cancel()
            _step_start(ctx, name)
            fn(ctx)
            _step_end(
                ctx, name, "ok", {"selection": ctx.selection} if name == StepName.SELECT else None
            )
        _step_start(ctx, StepName.PHASE1)
        ctx.phase1_outputs = _reviewer_lanes(
            ctx, phase="phase1", lm=pol.phase1_reviewer, expected_phase="preliminary"
        )
        _step_end(ctx, StepName.PHASE1, "ok", {"roles": list(ctx.phase1_outputs)})
        _step_start(ctx, StepName.VERIFY1)
        ctx.verify1 = _verifier_lane(
            ctx,
            phase="verify1",
            lm=pol.phase1_verifier,
            expected_phase="preliminary",
            p1=ctx.phase1_outputs,
            p2={},
        )
        _step_end(
            ctx,
            StepName.VERIFY1,
            "ok",
            {"blockers": ctx.verify1.blocker_count, "findings": len(ctx.verify1.findings)},
        )
        _step_start(ctx, StepName.GATE)
        admit = admit_phase2(ctx.verify1, phase2_enabled=pol.phase2_enabled)
        _step_end(ctx, StepName.GATE, "ok", {"admit_phase2": admit})
        if not admit:
            ctx.check_cancel()
            _step_start(ctx, StepName.PUBLISH)
            res = step_publish(
                ctx, phase="preliminary", verified=ctx.verify1, verifier_lm=pol.phase1_verifier
            )
            _step_end(
                ctx,
                StepName.PUBLISH,
                "ok",
                {"posted": res.posted, "event": res.event, "skipped": res.skipped_reason},
            )
            _gate_comment(ctx, "done", phase="preliminary", blocker_count=ctx.verify1.blocker_count)
            steps.append((StepName.PUBLISH, res))
            return RunStatus.DONE
        _step_start(ctx, StepName.PHASE2)
        ctx.phase2_outputs = _reviewer_lanes(
            ctx, phase="phase2", lm=pol.phase2_reviewer, expected_phase="final"
        )
        _step_end(ctx, StepName.PHASE2, "ok", {"roles": list(ctx.phase2_outputs)})
        _step_start(ctx, StepName.VERIFY2)
        ctx.verify2 = _verifier_lane(
            ctx,
            phase="verify2",
            lm=pol.phase2_verifier,
            expected_phase="final",
            p1=ctx.phase1_outputs,
            p2=ctx.phase2_outputs,
        )
        _step_end(
            ctx,
            StepName.VERIFY2,
            "ok",
            {"blockers": ctx.verify2.blocker_count, "findings": len(ctx.verify2.findings)},
        )
        _step_start(ctx, StepName.PUBLISH)
        res = step_publish(
            ctx, phase="final", verified=ctx.verify2, verifier_lm=pol.phase2_verifier
        )
        _step_end(
            ctx,
            StepName.PUBLISH,
            "ok",
            {"posted": res.posted, "event": res.event, "skipped": res.skipped_reason},
        )
        _gate_comment(ctx, "done", phase="final", blocker_count=ctx.verify2.blocker_count)
        return RunStatus.DONE
    finally:
        pass


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
) -> RunStatus:
    row = conn.execute(
        "SELECT r.*, h.repo, h.number, h.sha FROM runs r JOIN heads h ON h.id=r.head_id WHERE r.id=?",
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
        attempt=int(row["attempt"]),
        token=str(row["token"]),
        run_dir=cfg.runs_dir / f"run-{run_id}",
        dry_run=dry_run,
    )
    if lane_runner:
        ctx.lane_runner = lane_runner
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
