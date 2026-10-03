"""Worker: runs one review (one `runs` row) end to end.

Steps: worktree -> select + triage (one prep lane) -> context -> phase1 -> verify1 -> gate -> phase2 -> verify2 -> publish.
Every step is recorded in `steps`. Any ReviewError ends the run as failed with
its classification; the scheduler decides on retry. Cooperative cancellation
is checked on every heartbeat.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import logging
import random
import shutil
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from . import audit, converse, degraded, github, labels, lanepool, publish, quota
from . import prep as prep_mod
from . import progress as progress_mod
from .config import Config, DegradedPolicy, LaneModel, TierEffort, min_effort
from .contract import (
    Finding,
    ReviewerOutput,
    VerifierOutput,
    parse_json_object,
    parse_reviewer_output,
    parse_verifier_output,
)
from .db import connect_existing, event, kv_get, kv_set, now, now_dt, tx
from .degraded import Prober
from .gate import gate_score, phase1_blocks
from .gh import Gh
from .lane import (
    LaneResult,
    LaneRunner,
    LaneSpec,
    exec_deny_profile,
    lane_output,
    new_session_id,
    prompt_sha,
    run_claude_lane,
    sandboxed,
)
from .models import FailKind, HeadStatus, ReviewError, RunStatus, StepName, Trigger
from .prompts import (
    correction_prompt,
    prior_for_prompt,
    repair_prompt,
    reviewer_prompt,
    skill_texts,
    verifier_prompt,
)
from .scheduler import finish_run, live_queued_count, retire_obsolete_head
from .select import Selection, select, write_selection
from .steps import worktree as wt
from .triage import Triage, triage, write_triage

log = logging.getLogger(__name__)


class Cancelled(Exception):
    pass


class HeadObsolete(Exception):
    """The run's head is no longer worth reviewing: the PR's live head moved past it (a push
    landed after it was queued) or the PR closed. Not a failure: the run ends `cancelled` and
    the head `superseded` / `closed`, as ingest ends a head it sees go stale (see `main`)."""

    def __init__(self, status: HeadStatus, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


def _require_live(ctx: RunContext, live: github.PrMeta) -> None:
    """Raise HeadObsolete unless the PR is open and `live` is still the run's head."""
    if live.state != "open" or live.merged:
        raise HeadObsolete(
            HeadStatus.CLOSED, f"PR is {live.state}{' (merged)' if live.merged else ''}"
        )
    if live.head_sha != ctx.sha:
        raise HeadObsolete(
            HeadStatus.SUPERSEDED, f"live head {live.head_sha[:8]} != assigned {ctx.sha[:8]}"
        )


class LaneStopped(Exception):
    """A lane stopped before it finished because its phase is being abandoned (a sibling lane
    failed). Never a failure of its own: the sibling's error is what the run reports.
    `started` False: it never ran (stopped while it waited for a pool slot)."""

    def __init__(self, *, started: bool = True) -> None:
        super().__init__()
        self.started = started


class LaneFailed(ReviewError):
    """A lane failed for good (every attempt it had). `cause` is the last attempt's own
    error, without the "phase/role lane failed twice" framing: what a stopped sibling's
    reason quotes."""

    def __init__(self, kind: FailKind, message: str, cause: str) -> None:
        super().__init__(kind, message)
        self.cause = cause


class StopSignal:
    """A threading.Event that remembers why it was set (the first reason wins), so a lane
    stopped because its phase is being abandoned can say which lane failed and how."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self.reason = ""

    def set(self, reason: str = "") -> None:
        with self._lock:
            if not self._event.is_set():
                self.reason = reason
                self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True, slots=True)
class OutputCheck[T]:
    """How a lane's JSON answer becomes its typed output (`validate`, raising a CONTRACT
    ReviewError when it breaks the contract), and what a correction turn or the repair lane is
    told about it. `kind` "reviewer" / "verifier" lanes get correction turns in their own
    session (`[lanes] correction_turns`); "" (persistence, conversation) only the repair lane,
    as before."""

    validate: Callable[[dict[str, Any]], T]
    kind: str = ""
    expected_phase: str = ""
    prior: list[dict[str, Any]] = field(default_factory=list)  # prior findings, as prompted
    coderabbit_ids: list[int] = field(default_factory=list)


def _as_is(raw: dict[str, Any]) -> dict[str, Any]:
    return raw


PLAIN_OUTPUT = OutputCheck(validate=_as_is)


@dataclass(slots=True)
class _Verdict:
    """One answer judged against its OutputCheck: `output` when `error` is None."""

    output: Any = None
    error: ReviewError | None = None
    # no JSON object could be read from an answer that did finish: what the repair lane fixes
    unparsed: bool = False


def _validated[T](raw: dict[str, Any], check: OutputCheck[T]) -> _Verdict:
    try:
        return _Verdict(output=check.validate(raw))
    except ReviewError as exc:
        return _Verdict(error=exc)
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        # a value of the wrong type the parser did not expect (`int("abc")` for a comment id)
        # is the model's contract breach too, and as correctable; logged with its traceback,
        # since a bug in the parser itself would look the same
        log.exception("lane output rejected by %s", getattr(check.validate, "__name__", "?"))
        msg = f"output rejected: {type(exc).__name__}: {exc}"
        return _Verdict(error=ReviewError(FailKind.CONTRACT, msg))


def _evaluate[T](turn: LaneResult, check: OutputCheck[T]) -> _Verdict:
    """A finished turn's answer: an INFRA error when the turn itself failed (crash, timeout,
    upstream error), CONTRACT when its answer has no JSON object or breaks the contract."""
    try:
        raw = lane_output(turn)
    except ReviewError as exc:
        unparsed = exc.kind is FailKind.CONTRACT and turn.ok and bool(turn.result_text.strip())
        return _Verdict(error=exc, unparsed=unparsed)
    return _validated(raw, check)


@dataclass(slots=True)
class RunContext:
    cfg: Config
    main_conn: sqlite3.Connection  # the worker thread's; lane threads get their own (`conn`)
    gh: Gh
    run_id: int
    head_id: int
    repo: str
    number: int
    sha: str
    token: str
    run_dir: Path
    trigger: str = ""  # heads.trigger: what queued this head
    audit: dict[str, Any] | None = None  # the audits row when this is a post-merge audit run
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
    # the second model Phase-2 reviewer lanes also run on for this run (see ComparisonPolicy)
    compare_model: str | None = None
    phase1_choice: quota.Choice | None = None  # which Phase-1 ladder rung ran, and why
    phase1_lm: LaneModel | None = None  # the lane model of that rung; Phase-1 lanes start on it
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
    # Phase 1 ran beside Phase 2 with no blocker gate between them (a `single_stage` tier)
    single_stage: bool = False
    # a Phase 2 is planned for this run (the tier has one and it is enabled): only then does
    # Phase 1 slim down to `phase1_specialists`; a Phase-1-only review keeps every specialist
    phase2_follows: bool = True
    # finding_hash -> {comment_id, thread_id, replies} for prior findings with human replies
    prior_threads: dict[str, dict[str, Any]] = field(default_factory=dict)
    # finding_hash -> thread facts for every unresolved bot finding thread on the PR
    open_threads: dict[str, dict[str, Any]] = field(default_factory=dict)
    coverage_from: str = ""
    phase1_outputs: dict[str, ReviewerOutput] = field(default_factory=dict)
    phase2_outputs: dict[str, ReviewerOutput] = field(default_factory=dict)
    verify1: VerifierOutput | None = None
    verify2: VerifierOutput | None = None
    reviewers: list[dict[str, Any]] = field(default_factory=list)
    lane_runner: LaneRunner = run_claude_lane
    dry_run: bool = False
    cancel_flag: threading.Event = field(default_factory=threading.Event)
    # the reviewer lanes of a phase run in parallel threads: this guards the run-wide
    # decisions they can race on (falling down the Phase-1 ladder, flipping to degraded)
    state_lock: threading.Lock = field(default_factory=threading.Lock)
    lane_conns: threading.local = field(default_factory=threading.local)
    owner_thread: int = field(default_factory=threading.get_ident)  # the one using main_conn
    # live progress in the PR's gate comment: while "in progress" is the latest status posted,
    # `gate_live_kw` holds its arguments and the heartbeat re-renders it (see
    # `_refresh_gate_progress`). `gate_lock` orders those edits against a final status, so a
    # refresh can never overwrite "done" or "failed".
    gate_lock: threading.Lock = field(default_factory=threading.Lock)
    gate_live_kw: dict[str, Any] | None = None
    gate_comment_id: int | None = None
    gate_steps_seen: tuple[tuple[str, str, str], ...] = ()  # steps + lane notes last shown
    gate_refreshed_at: float = 0.0  # time.monotonic() of the last progress edit
    gate_body_seen: str = ""  # the in-progress body last written, to skip edits that change nothing
    # a conversation run (answers replies on a reviewed commit); None until run() knows. The
    # progress estimate measures it against conversations from the start (see progress.py)
    conversation: bool | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        """This thread's database connection: the worker's own on the main thread, a private
        one on a parallel lane thread (closed by `close_lane_conn` when the lane ends)."""
        if threading.get_ident() == self.owner_thread:
            return self.main_conn
        c: sqlite3.Connection | None = getattr(self.lane_conns, "conn", None)
        if c is None:
            assert str(self.cfg.db_path) != ":memory:", "lane threads need a database file"
            c = self.lane_conns.conn = connect_existing(self.cfg.db_path)
        return c

    def close_lane_conn(self) -> None:
        c = getattr(self.lane_conns, "conn", None)
        if c is not None:
            c.close()
            self.lane_conns.conn = None

    @property
    def is_audit(self) -> bool:
        return self.audit is not None

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
            continue
        _refresh_gate_progress(ctx, conn, stop)
    conn.close()


# a step change shows within a heartbeat or two; otherwise the bar and time left move every
# 10 minutes (edits notify nobody, but each one is an API write)
GATE_PROGRESS_EVERY_SECONDS = 600
GATE_PROGRESS_MIN_GAP_SECONDS = 30


def _progress_steps(conn: sqlite3.Connection, run_id: int) -> list[tuple[str, str, str]]:
    """(name, status, note) per step; the note says how many lanes a reviewer step finished."""
    rows = [
        {"name": r[0], "status": r[1], "started_at": r[2], "info": _json_or_empty(r[3])}
        for r in conn.execute(
            "SELECT name,status,started_at,detail FROM steps WHERE run_id=? "
            "ORDER BY started_at, rowid",
            (run_id,),
        )
    ]
    notes = []
    for st, n in zip(rows, progress_mod.step_lanes(conn, run_id, rows), strict=True):
        note = ""
        if st["status"] == "running" and "lanes_total" in n:
            note = f"{n['lanes_done']}/{n['lanes_total']} lanes"
            if n.get("comparison_left") and n["lanes_done"] == n["lanes_total"]:
                note += f", {n['comparison_left']} comparison still going"
        notes.append((st["name"], st["status"], note))
    return notes


def _json_or_empty(text: str | None) -> dict[str, Any]:
    try:
        value = json.loads(text or "{}")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _gate_progress(ctx: RunContext, conn: sqlite3.Connection) -> dict[str, Any] | None:
    """The progress block of the in-progress gate comment; None if it cannot be estimated.
    Never raises: the gate comment is bookkeeping, not the review."""
    try:
        at = now_dt()
        est = progress_mod.estimate(conn, ctx.run_id, at, conversation=ctx.conversation)
        if est is None:
            return None
        return {
            "basis": est.basis,
            "fraction": est.progress,
            "overdue": est.overdue,
            "remaining_seconds": est.remaining_seconds,
            "elapsed_seconds": est.elapsed_seconds,
            "steps": [
                *_progress_steps(conn, ctx.run_id),
                *((n, "upcoming", "") for n in est.upcoming),
            ],
            "run_id": ctx.run_id,
            "updated_at": at.strftime("%H:%M"),
        }
    except Exception as exc:
        log.warning("gate progress estimate failed: %s", exc)
        return None


def _gate_edit_due(ctx: RunContext, body: str, since: float) -> bool:
    """Whether a re-rendered in-progress `body` is worth a GitHub edit: never when nothing but
    the update time moved; at once (past the min gap) when the status line, a step chip or the
    estimate's basis changed; otherwise (only the bar and time left moved) every
    GATE_PROGRESS_EVERY_SECONDS. A step change alone is not enough: steps the comment does not
    show (a finished checkout) used to re-post an identical body two or three times a run."""
    seen = ctx.gate_body_seen
    if github.without_update_time(body) == github.without_update_time(seen):
        return False
    return github.gate_digest(body) != github.gate_digest(seen) or (
        since >= GATE_PROGRESS_EVERY_SECONDS
    )


def _refresh_gate_progress(
    ctx: RunContext, conn: sqlite3.Connection, stop: threading.Event | None = None
) -> None:
    """Re-render the in-progress gate comment when the run's steps changed, or every
    GATE_PROGRESS_EVERY_SECONDS, and edit it when that changed what it shows
    (`_gate_edit_due`). Runs on the heartbeat thread, with its own connection.

    Never waits: if the main thread is writing the gate comment (it holds `gate_lock` across a
    GitHub call), this pass is skipped, so a slow GitHub never delays the heartbeat or its
    cancel check. Stops for good once the head is no longer `running`: a superseded head's
    comment belongs to the queue now (queue_status.py writes the new commit's status there)."""
    if ctx.gate_live_kw is None or ctx.gate_comment_id is None:
        return
    try:
        steps = tuple(_progress_steps(conn, ctx.run_id))
        head = conn.execute("SELECT status FROM heads WHERE id=?", (ctx.head_id,)).fetchone()
    except sqlite3.Error:
        return
    if head is None or head[0] != "running":
        return
    since = time.monotonic() - ctx.gate_refreshed_at
    if since < GATE_PROGRESS_MIN_GAP_SECONDS or (
        steps == ctx.gate_steps_seen and since < GATE_PROGRESS_EVERY_SECONDS
    ):
        return
    if not ctx.gate_lock.acquire(blocking=False):
        return
    try:
        kw = ctx.gate_live_kw
        # a final status landed meanwhile, or the worker is shutting down
        if kw is None or ctx.gate_comment_id is None or (stop is not None and stop.is_set()):
            return
        body = github.gate_body(
            "in_progress", ctx.sha, **_gate_defaults(ctx, kw), progress=_gate_progress(ctx, conn)
        )
        ctx.gate_steps_seen = steps  # rendered: re-render on the next change or interval
        since = time.monotonic() - ctx.gate_refreshed_at  # the main thread may have just edited
        if not _gate_edit_due(ctx, body, since):
            if github.without_update_time(body) == github.without_update_time(ctx.gate_body_seen):
                # nothing at all moved (no history, so no bar): wait out another interval
                # rather than re-render on every heartbeat
                ctx.gate_refreshed_at = time.monotonic()
            return
        ctx.gate_refreshed_at = time.monotonic()
        try:
            ctx.gh.api(
                f"repos/{ctx.repo}/issues/comments/{ctx.gate_comment_id}",
                method="PATCH",
                body={"body": body},
                timeout=30,
            )
            ctx.gate_body_seen = body
        except ReviewError as exc:
            log.warning("gate progress update failed: %s", exc)
    finally:
        ctx.gate_lock.release()


# ---- step bookkeeping ----


def _step_start(ctx: RunContext, name: StepName, detail: dict[str, Any] | None = None) -> None:
    """Record a step as running, with what is known up front (`detail`, replaced by
    `_step_end`'s). `runs.phase` (the step a failure is charged to, and what status shows)
    only follows the worker's own thread: on a single-stage run Phase 1 runs on a second
    thread beside Phase 2, which owns it."""
    with tx(ctx.conn):
        ctx.conn.execute(
            "INSERT OR REPLACE INTO steps (run_id, name, status, started_at, detail) "
            "VALUES (?,?,?,?,?)",
            (
                ctx.run_id,
                name.value,
                "running",
                now(),
                json.dumps(detail)[:4000] if detail else None,
            ),
        )
        if threading.get_ident() == ctx.owner_thread:
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
    """One row per lane attempt. Its correction turns (`res.followups`) ran in the same
    session and slot, so their tokens are added to the row rather than given rows of their
    own: a role still has one row per attempt, which is what progress and compare count."""
    turns = [res, *res.followups] if res else []
    tokens_in = [t.tokens_in for t in turns if t.tokens_in is not None]
    tokens_out = [t.tokens_out for t in turns if t.tokens_out is not None]
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
                sum(tokens_in) if tokens_in else None,
                sum(tokens_out) if tokens_out else None,
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


def _gate_defaults(ctx: RunContext, kw: dict[str, Any]) -> dict[str, Any]:
    """`kw` plus what the run knows by now (tier, Phase-2-only, ad hoc, degraded)."""
    kw = dict(kw)
    if ctx.triage is not None:
        kw.setdefault("tier", ctx.tier)
    if ctx.phase1_skipped:
        kw.setdefault("phase1_skipped", ctx.phase1_skipped)
    if ctx.adhoc:
        kw.setdefault("adhoc", True)
    if ctx.is_degraded:
        kw.setdefault("degraded", True)
    return kw


def _gate_comment(ctx: RunContext, status: str, **kw: Any) -> None:
    if ctx.dry_run or ctx.is_audit:
        return
    with ctx.gate_lock:
        live = status == "in_progress"
        ctx.gate_live_kw = kw if live else None
        full = _gate_defaults(ctx, kw)
        if live:
            full["progress"] = _gate_progress(ctx, ctx.conn)
        body = github.gate_body(status, ctx.sha, **full)
        # an in-progress status this worker already shows (same status line, chips, basis) is
        # left to the heartbeat's interval refresh; the first one always replaces the queue's
        if not live or ctx.gate_comment_id is None or _gate_edit_due(ctx, body, 0.0):
            try:
                cid = github.upsert_gate_comment(
                    ctx.gh, ctx.repo, ctx.number, ctx.cfg.bot_login, body
                )
                ctx.gate_comment_id = cid or ctx.gate_comment_id
                ctx.gate_body_seen = body if live else ""
            except ReviewError as exc:
                log.warning("gate comment update failed: %s", exc)
            if live:
                ctx.gate_refreshed_at = time.monotonic()
        if live:
            with contextlib.suppress(sqlite3.Error):
                ctx.gate_steps_seen = tuple(_progress_steps(ctx.conn, ctx.run_id))


def _gate_closed(ctx: RunContext, reason: str) -> None:
    """The PR closed or merged before this run could review it: an existing gate comment says
    so instead of a stale "queued" / "in progress" status that nothing else would replace (no
    newer head will be queued). Never creates a comment on a closed PR just to say that."""
    if ctx.dry_run or ctx.is_audit:
        return
    with ctx.gate_lock:
        ctx.gate_live_kw = None
        try:
            existing = github.find_gate_comment(ctx.gh, ctx.repo, ctx.number, ctx.cfg.bot_login)
            if existing:
                ctx.gh.api(
                    f"repos/{ctx.repo}/issues/comments/{existing['id']}",
                    method="PATCH",
                    body={"body": github.gate_body("closed", ctx.sha, reason=reason)},
                )
        except ReviewError as exc:
            log.warning("gate comment update failed: %s", exc)


# ---- steps ----


def step_worktree(ctx: RunContext) -> None:
    meta = github.pr_meta(ctx.gh, ctx.repo, ctx.number)
    if ctx.is_audit:
        _audit_worktree(ctx, meta)
        return
    _require_live(ctx, meta)
    ctx.meta = meta
    ctx.base_sha = meta.base_sha
    # the first "in progress" status, once the head is known to be live (an obsolete run
    # leaves the comment to the newer head's writer): the checkout shows as the running step
    _gate_comment(ctx, "in_progress")
    ctx.worktree = _checkout_head(ctx)
    base = wt.merge_base(ctx.worktree, meta.base_ref, ctx.sha) if meta.base_ref else None
    ctx.coverage_from = base or f"{ctx.sha}~1"
    _record_worktree(ctx)


def _worktree_name(ctx: RunContext, suffix: str = "") -> str:
    return f"{ctx.repo.replace('/', '-')}-{ctx.number}-{ctx.run_id}{suffix}"


def _checkout_head(ctx: RunContext) -> Path:
    """Fetch the assigned head into the repo's mirror and check it out in a fresh worktree."""
    ctx.mirror = wt.ensure_mirror(ctx.cfg.mirrors_dir, ctx.repo)
    wt.fetch_head(ctx.mirror, ctx.number, ctx.sha)
    return wt.create_worktree(ctx.mirror, ctx.cfg.worktrees_dir, _worktree_name(ctx), ctx.sha)


def _record_worktree(ctx: RunContext) -> None:
    with tx(ctx.conn):
        ctx.conn.execute(
            "UPDATE runs SET worktree=?, run_dir=? WHERE id=?",
            (str(ctx.worktree), str(ctx.run_dir), ctx.run_id),
        )


def _audit_worktree(ctx: RunContext, meta: github.PrMeta) -> None:
    """The merged head, reviewed against the base branch as it stood just before the merge:
    the first parent of the merge commit (for a squash or rebase merge, the parent of the
    first landed commit is the same thing). The PR's own diff, as merged, not today's base."""
    assert ctx.audit is not None
    if not meta.merged:
        raise ReviewError(FailKind.FATAL, f"audit of a PR that is not merged (state {meta.state})")
    ctx.meta = meta
    ctx.worktree = _checkout_head(ctx)
    merge = str(ctx.audit.get("merge_commit") or "")
    base = wt.pre_merge_base(ctx.worktree, meta.base_ref, merge, ctx.sha) if merge else None
    if base is None and meta.base_ref:
        base = wt.merge_base(ctx.worktree, meta.base_ref, ctx.sha)
    if not base or base == ctx.sha:
        # the merge base with *today's* base is the head itself after a true merge: that range
        # is empty and would report a false clean. Better no answer than a wrong one.
        raise ReviewError(
            FailKind.INFRA if not base else FailKind.FATAL,
            f"cannot determine the pre-merge base of {ctx.sha[:8]} (merge {merge[:8] or '?'})",
        )
    ctx.coverage_from = base
    ctx.base_sha = base
    _record_worktree(ctx)


def step_prep(ctx: RunContext) -> None:
    """Which specialists review the PR (`select` step) and which effort tier it is (`triage`
    step). With triage configured and specialists to choose from, one prep lane answers both
    (prep.py): the two step rows start and end together, so the progress profile (steps by
    name, see progress.py) reads old runs, where they ran one after the other, and new ones
    alike, and a running pair counts once (the longer one). Otherwise the selector (no
    triage: a light audit, or a policy without triage) or triage (nothing to choose: the
    selection comes from config) runs alone, as before the merge.

    Every half keeps its own fallback, artifact (selector.json / triage.json) and event
    (`select.degraded` / `triage.degraded`)."""
    assert ctx.meta and ctx.worktree
    _step_start(ctx, StepName.SELECT)
    files_raw = (
        ctx.gh.api(f"repos/{ctx.repo}/pulls/{ctx.number}/files?per_page=100", paginate=True) or []
    )
    ctx.files = [f for f in files_raw if isinstance(f, dict)]
    lane = ctx.cfg.policy.triage
    if lane is not None and ctx.audit and ctx.audit["mode"] == audit.MODE_LIGHT:
        ctx.tier = LIGHT_AUDIT_TIER  # one cheap Phase-1 pass; escalated on a blocker
        lane = None
    if lane is not None and any(not s.always_run for s in ctx.cfg.specialists_for(ctx.repo)):
        _step_start(ctx, StepName.TRIAGE)
        p = _prep(ctx, lane)
        _end_select(ctx, p.selection)
        _end_triage(ctx, p.triage)
    else:
        _end_select(ctx, _select(ctx))
        if lane is None:
            return
        ctx.check_cancel()
        _step_start(ctx, StepName.TRIAGE)
        _end_triage(ctx, _triage(ctx, lane))
    _gate_comment(ctx, "in_progress")


def _prep(ctx: RunContext, lane: LaneModel) -> prep_mod.Prep:
    assert ctx.meta and ctx.worktree
    meta, worktree = ctx.meta, ctx.worktree

    def attempt(prior: prep_mod.Prep | None = None) -> prep_mod.Prep:
        return prep_mod.prep(
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
            prior=prior,
        )

    p = attempt()
    if (
        (p.selection.error or p.triage.error)
        and p.infra_error
        and _degrade_on_side_lane_failure(ctx, "prep", lane.model, p.infra_error)
    ):
        p = attempt(p)  # once more on the stand-in, for the half (or both) that fell back
    return p


def _select(ctx: RunContext) -> Selection:
    assert ctx.meta and ctx.worktree
    meta, worktree = ctx.meta, ctx.worktree
    files = [str(f.get("filename")) for f in ctx.files]

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
    return sel


def _triage(ctx: RunContext, lane: LaneModel) -> Triage:
    assert ctx.meta and ctx.worktree
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
    return t


def _end_select(ctx: RunContext, sel: Selection) -> None:
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
    _step_end(ctx, StepName.SELECT, "ok", {"selection": ctx.selection})


def _end_triage(ctx: RunContext, t: Triage) -> None:
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
    _step_end(ctx, StepName.TRIAGE, "ok", dataclasses.asdict(t))


def step_context(ctx: RunContext) -> None:
    assert ctx.meta
    threads = github.review_threads(ctx.gh, ctx.repo, ctx.number)
    ctx.evidence = github.evidence_bundle(
        ctx.gh, ctx.repo, ctx.number, ctx.meta, ctx.cfg.bot_login, include_coderabbit=False
    )
    ctx.evidence["ci"] = github.ci_checks(ctx.gh, ctx.repo, ctx.sha)
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
) -> dict[str, Any]:
    """A lane whose JSON answer its caller checks itself (persistence, conversation): retries
    and the repair lane as below, no correction turns."""
    return _run_checked_lane(
        ctx,
        phase=phase,
        role=role,
        lm=lm,
        prompt=prompt,
        is_verifier=is_verifier,
        check=PLAIN_OUTPUT,
    )


def _run_checked_lane[T](
    ctx: RunContext,
    *,
    phase: str,
    role: str,
    lm: LaneModel,
    prompt: str,
    is_verifier: bool,
    check: OutputCheck[T],
    fresh: bool = False,
    should_stop: Callable[[], bool] | None = None,
    comparison: bool = False,
    stop_reason: Callable[[], str] | None = None,
) -> T:
    """Run a lane with bounded retries until its answer passes `check`. Returns the output.

    An answer that breaks the contract (no JSON object in it, or `check.validate` rejects it
    after the deterministic normalization in contract.py) is first corrected in the lane's
    own session: up to `[lanes] correction_turns` follow-up turns tell the model the exact
    errors and the values it must echo (`lane.run_with_corrections`, inside the lane's pool
    slot). An answer that still has no JSON object after that (the session could not be
    resumed, or the corrections ran out) goes to the context-free repair lane as a last resort,
    told the values it cannot know; its output is normalized and validated like any other. An
    attempt still failing after that is a failed attempt, retried once: so it happens before
    the Phase-1 ladder (`_reviewer_lane`) moves a role down for a contract failure. A lane told
    to stop between its turns is stopped, never a failed attempt. Every correction is an
    event (`lane.corrected` / `lane.correction_failed`), every normalized answer one too
    (`lane.output_normalized`); the row is `corrected` / `repaired` / `completed`.

    A lane on a primary model that dies on a quota failure flips the run into degraded mode
    (when the policy has a stand-in for that model) and gets its two attempts again on the
    stand-in, so one exhausted pool does not fail the review.

    `should_stop` (default: the run was cancelled) is polled while the lane runs; a stopped
    lane raises Cancelled when the run was cancelled, else LaneStopped, its row saying why
    (`stop_reason`). A `comparison` lane (keyed `<role>#2`) gets one attempt, takes its slot in
    the `compare` pool (never a production one), never runs on a stand-in and never flips the
    run: it fails instead."""
    assert ctx.worktree
    stop = should_stop or ctx.cancel_flag.is_set
    key = f"{role}{COMPARE_SUFFIX}" if comparison else role
    pool = lanepool.COMPARE_POOL if comparison else None
    corrections = ctx.cfg.lane_correction_turns if check.kind else 0
    last = cause = ""
    attempt, budget = 0, 1 if comparison else 2
    while attempt < budget:
        attempt += 1
        if stop():
            ctx.check_cancel()  # a cancelled run is Cancelled, never just a stopped lane
            raise LaneStopped(started=attempt > 1)
        if comparison and ctx.is_degraded:
            raise ReviewError(FailKind.INFRA, f"{phase}/{key}: the run went degraded")
        # re-resolved per attempt: a parallel sibling may have flipped the run to degraded
        lm = lm if comparison else ctx.lane_model(lm)
        if phase == "phase1" and not is_verifier and ctx.phase1_lm not in (None, lm):
            # ... or moved it down the Phase-1 ladder: no second try on the rung it left
            raise ReviewError(FailKind.INFRA, f"{phase}/{role}: the run left {lm.model}")
        attempt_id = uuid.uuid4().hex[:12]
        art = ctx.run_dir / "attempts" / f"{phase}-{role}-{attempt_id}"
        asked: list[str] = []  # the error each correction turn was asked to fix

        def ask(turn: LaneResult, asked: list[str] = asked) -> str | None:
            verdict = _evaluate(turn, check)
            if verdict.error is None or verdict.error.kind is not FailKind.CONTRACT:
                return None
            asked.append(verdict.error.message)
            return correction_prompt(
                errors=verdict.error.message.split("; "),
                kind=check.kind,
                expected_phase=check.expected_phase,
                head_sha=ctx.sha,
                prior=check.prior,
                coderabbit_ids=check.coderabbit_ids,
                turn=len(asked),
                turns=corrections,
            )

        session = new_session_id() if corrections else ""
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
            should_stop=stop,
            parallel=not is_verifier and not comparison,
            pool=pool,
            session_id=session,
            check=ask if session else None,
            corrections=corrections,
        )
        res = ctx.lane_runner(spec, art, ctx.worktree)
        psha = prompt_sha(prompt)
        turns = [res, *res.followups]
        row = functools.partial(
            _lane_row,
            ctx,
            phase=phase,
            role=role,
            lm=lm,
            attempt=attempt,
            attempt_id=attempt_id,
            res=res,
            artifact_dir=art,
            psha=psha,
        )

        def stopped(started: bool, row: functools.partial[None] = row) -> LaneStopped:
            """Record a lane stopped before it finished (or between its turns) and say so."""
            if started:  # a lane stopped while it waited for a slot never ran
                if ctx.cancel_flag.is_set():
                    why = "run cancelled"
                elif comparison:
                    why = "stopped: comparison lane"
                else:
                    why = (stop_reason() if stop_reason else "") or (
                        "another lane of this phase failed"
                    )
                row(status="cancelled", reason=why)
            ctx.check_cancel()
            return LaneStopped(started=started)

        if any(t.cancelled for t in turns):  # stopped, before it finished or mid-correction
            raise stopped(res.started)
        # the newest turn that finished: a correction that could not resume the session (or
        # died on its own) leaves the answer before it standing
        basis = next((t for t in reversed(turns) if t.ok), res)
        failed_turn = turns[-1] if res.followups and not turns[-1].ok else None
        verdict = _evaluate(basis, check)
        if verdict.error is not None and stop():
            # told to stop between turns (the phase was abandoned, the run cancelled): not a
            # failed attempt, so never a retry or a fall down the ladder for a stopped lane
            raise stopped(True)
        status = "completed" if basis is res else "corrected"  # when it passes
        if verdict.error is not None and verdict.unparsed:
            # still no JSON object: the session could not be resumed, the corrections ran
            # out, or there were none. The context-free repair lane is the last resort.
            repaired = _repair(ctx, basis.result_text, art, stop, pool, check)
            if repaired is not None:
                fixed = _validated(repaired, check)
                if fixed.error is None:
                    verdict, status = fixed, "repaired"
                else:
                    msg = f"repaired output: {fixed.error.message}"
                    verdict = _Verdict(error=ReviewError(FailKind.CONTRACT, msg))
        if res.followups:
            _correction_event(
                ctx,
                f"{phase}/{key} {lm.model} turns={len(res.followups)}",
                asked,
                verdict.error.message
                if verdict.error
                else ("rescued by the repair lane" if status == "repaired" else None),
                failed_turn,
            )
        if verdict.error is None:
            notes: list[str] = getattr(verdict.output, "normalized", None) or []
            reason = (
                f"corrected ({len(res.followups)} turns): {asked[0]}"
                if asked and status == "corrected"
                else ""
            )
            if notes:
                reason = "; ".join(filter(None, (reason, "normalized: " + "; ".join(notes))))
                _lane_event(
                    ctx, "lane.output_normalized", f"{phase}/{key} {lm.model}: {'; '.join(notes)}"
                )
            row(status=status, reason=reason)
            if not is_verifier:
                _record_reviewer(ctx, phase, role, key, lm, fresh, attempt_id)
            return cast(T, verdict.output)
        exc = verdict.error
        after = f" (after {len(res.followups)} correction turns)" if res.followups else ""
        row(status="failed", reason=f"{exc}{after}")
        last, cause = f"{exc}{after}", exc.message
        # a correction turn that died on a dry pool says so as surely as a first turn would
        quota_turn, quota_exc = failed_turn or res, exc
        if failed_turn is not None:  # did not finish: lane_output names its infra error
            try:
                lane_output(failed_turn)
            except ReviewError as turn_exc:
                quota_exc = turn_exc
        switched = None if comparison else _degrade_on_quota_failure(ctx, lm, quota_exc, quota_turn)
        if switched is not None:
            lm, budget = switched, attempt + 2
    raise LaneFailed(
        FailKind.INFRA if "timed out" in last or "exit" in last else FailKind.CONTRACT,
        f"{phase}/{role} lane failed {'twice' if attempt == 2 else f'{attempt} times'}: {last}",
        cause,
    )


def _lane_event(ctx: RunContext, kind: str, detail: str) -> None:
    with tx(ctx.conn):
        event(
            ctx.conn,
            kind,
            repo=ctx.repo,
            number=ctx.number,
            run_id=ctx.run_id,
            detail=detail[:1000],
        )


def _correction_event(
    ctx: RunContext,
    head: str,
    asked: list[str],
    failure: str | None,
    failed_turn: LaneResult | None,
) -> None:
    """`lane.corrected` / `lane.correction_failed`, one per lane attempt that had correction
    turns: `head` (phase/role, model, turns), what the first turn was asked to fix, and for a
    failure why (still invalid, or the session could not be resumed) and how it ended (the
    error, or that the repair lane rescued it). `failure` None: the correction worked."""
    first = f"first error: {asked[0][:400]}" if asked else ""
    if failure is None:
        _lane_event(ctx, "lane.corrected", f"{head}: {first}")
        return
    if failed_turn is None:
        how = "still invalid"
    elif failed_turn.timed_out:
        how = "correction turn timed out"
    else:
        said = failed_turn.first_stderr_line or f"exit {failed_turn.exit_code}"
        how = f"resume failed: {said[:200]}"
    _lane_event(ctx, "lane.correction_failed", f"{head} ({how}): {failure[:400]}; {first}")


def _record_reviewer(
    ctx: RunContext,
    phase: str,
    role: str,
    key: str,
    lm: LaneModel,
    fresh: bool,
    attempt_id: str,
) -> None:
    """A reviewer lane's provenance entry. Lanes of a phase finish on parallel threads, and
    a dropped comparison lane takes its entry back out, so both go through the state lock."""
    with ctx.state_lock:
        ctx.reviewers.append(
            {
                "model": lm.model,
                "agent": lm.agent,
                "role": role,
                "key": key,
                "effort": lm.effort,
                "status": "completed",
                "phase": phase,
                "fresh": fresh,
                "attempt_id": attempt_id,
                "substitute_for": lm.substitute_for,
            }
        )


def _repair[T](
    ctx: RunContext,
    raw: str,
    art: Path,
    stop: Callable[[], bool],
    pool: str | None,
    check: OutputCheck[T],
) -> dict[str, Any] | None:
    """The context-free repair lane: turns an answer with no readable JSON object into one.
    It is told the echo fields and prior hashes it could not know (it used to guess them, and
    44% of its outputs then failed the contract); its output still goes through the same
    normalization and validation. `pool`: a comparison lane's repair stays in its pool."""
    if len(raw) > 200_000 or not ctx.worktree:
        return None
    spec = LaneSpec(
        role="repair",
        agent="repair",
        model=ctx.model_name(ctx.cfg.policy.repair_model),
        effort="low",
        prompt=repair_prompt(
            raw,
            kind=check.kind,
            expected_phase=check.expected_phase,
            head_sha=ctx.sha,
            prior_hashes=[str(p["finding_hash"]) for p in check.prior],
        ),
        cwd=ctx.worktree,
        add_dir=ctx.run_dir,
        timeout_seconds=300,
        claude_bin=ctx.cfg.claude_bin,
        should_stop=stop,
        pool=pool,
    )
    try:
        res = ctx.lane_runner(spec, art / "repair", ctx.worktree)
        if not res.ok:
            return None
        return parse_json_object(res.result_text)
    except ReviewError:
        return None


def _reviewer_lane(
    ctx: RunContext,
    *,
    phase: str,
    role: str,
    lm: LaneModel,
    prompt: str,
    check: OutputCheck[ReviewerOutput],
    fresh: bool = False,
    should_stop: Callable[[], bool] | None = None,
    comparison: bool = False,
    stop_reason: Callable[[], str] | None = None,
) -> ReviewerOutput:
    """One reviewer lane, retried down the Phase-1 ladder when its rung dies. A Phase-1 lane
    starts (and restarts) on the run's current rung, so a lane that starts after a sibling
    fell down the ladder does not walk into the same dead rung. Its output contract is checked
    (and corrected) per attempt inside `_run_checked_lane`, so a contract failure reaches the
    ladder only after the corrections and the retry."""
    while True:
        if phase == "phase1" and ctx.phase1_lm is not None:
            lm = ctx.phase1_lm
        try:
            return _run_checked_lane(
                ctx,
                phase=phase,
                role=role,
                lm=lm,
                prompt=prompt,
                is_verifier=False,
                check=check,
                fresh=fresh,
                should_stop=should_stop,
                comparison=comparison,
                stop_reason=stop_reason,
            )
        except ReviewError as exc:
            if should_stop is not None and should_stop():
                # the phase is being abandoned (or the run cancelled) meanwhile: no ladder
                # fall for it; the failure that stopped it is what the run reports
                ctx.check_cancel()
                raise LaneStopped() from exc
            nxt = _phase1_fallback(ctx, lm, exc) if phase == "phase1" else None
            if nxt is None:
                raise
            lm = nxt


def _phase_roles(ctx: RunContext, phase: str) -> list[str]:
    """The reviewer roles of a phase: `general` plus the selected specialists. Phase 1 runs
    only the specialists the policy lists for it (`phase1_specialists`), when it lists any."""
    allowed = ctx.cfg.policy.phase1_specialists
    picked = ctx.selection
    if phase == "phase1" and allowed is not None and ctx.phase2_follows:
        picked = [s for s in picked if s in allowed]
    return ["general", *picked]


def _reviewer_lanes(
    ctx: RunContext,
    *,
    phase: str,
    lm: LaneModel,
    expected_phase: str,
    fresh: bool = False,
    abort: StopSignal | None = None,
) -> dict[str, ReviewerOutput]:
    """The general reviewer and the phase's specialists (`_phase_roles`), side by side (at
    most `phase_parallelism` at once, all of them when it is 0; each lane also waits for a
    slot in its model's pool, see lanepool.py). The reviewers of a phase never read each
    other's output, so only the verifier after them has to wait for all of them. `abort`,
    when set by the caller, stops every lane (a single-stage run whose other phase failed).

    One lane failing for good fails the phase, exactly as it did when they ran one after
    another: its siblings are stopped rather than left to spend quota on a run that will be
    retried from scratch, their rows naming the lane that failed and how ("stopped:
    phase1/rust-quality output failed the contract (...)").

    On a comparison run (`ctx.compare_model`) every Phase-2 role also runs on that model,
    keyed `<role>#2`, on a pool of its own so the primary lanes keep their parallelism. Not on
    the fresh final pass (it would double an already large verifier prompt). A comparison lane
    never holds the review up or fails it: it takes its slot in a pool of its own (see
    lanepool.py), one that fails is dropped with an event, and one still running
    `COMPARE_GRACE_MINUTES` after the last primary lane finished is stopped and dropped the
    same way."""
    assert ctx.meta
    meta = ctx.meta.as_dict()
    if abort is None:
        abort = StopSignal()
    roles = _phase_roles(ctx, phase)
    review_prior = [] if fresh else ctx.prior
    review_prior_sha = None if fresh else ctx.prior_sha
    prior_hashes = {str(p["finding_hash"]) for p in review_prior}
    prompts = {
        role: reviewer_prompt(
            ctx.cfg,
            repo=ctx.repo,
            number=ctx.number,
            head_sha=ctx.sha,
            phase=expected_phase,
            role=role,
            meta=meta,
            coverage_from=ctx.coverage_from,
            evidence=ctx.evidence,
            prior=review_prior,
            prior_sha=review_prior_sha,
            fresh=fresh,
        )
        for role in roles
    }
    # (output key, role, lane model, comparison?)
    lanes: list[tuple[str, str, LaneModel, bool]] = [(r, r, lm, False) for r in roles]
    if phase == "phase2" and ctx.compare_model and not fresh:
        twin = dataclasses.replace(lm, model=ctx.compare_model, substitute_for=None)
        lanes += [(f"{r}{COMPARE_SUFFIX}", r, twin, True) for r in roles]
    keys = [k for k, *_ in lanes]
    abandon = StopSignal()  # the phase failed: every lane stops, told which lane failed
    twins_stop = threading.Event()  # the comparison lanes ran out of grace

    def stopped() -> bool:
        return ctx.cancel_flag.is_set() or abandon.is_set() or abort.is_set()

    def twin_stopped() -> bool:
        return stopped() or twins_stop.is_set()

    def drop(key: str, model: str, why: str) -> None:
        """A comparison lane is out: its provenance entry (if it got that far) goes too."""
        log.warning("comparison lane %s on %s dropped: %s", key, model, why)
        with ctx.state_lock:
            ctx.reviewers[:] = [
                r for r in ctx.reviewers if not (r["phase"] == phase and r["key"] == key)
            ]
        try:
            with tx(ctx.conn):
                event(
                    ctx.conn,
                    "compare.lane_dropped",
                    repo=ctx.repo,
                    number=ctx.number,
                    run_id=ctx.run_id,
                    detail=f"{phase}/{key} {model}: {why[:400]}",
                )
        except sqlite3.Error as exc:  # bookkeeping for a comparison must not fail the review
            log.warning("could not record dropped comparison lane %s: %s", key, exc)

    def stop_reason() -> str:
        # the caller's abort is the root cause when it is set (a single-stage Phase 2 failed)
        return abort.reason or abandon.reason

    def review(key: str, role: str, lane_lm: LaneModel, comparison: bool) -> ReviewerOutput | None:
        def validate(raw: dict[str, Any]) -> ReviewerOutput:
            return parse_reviewer_output(
                raw,
                expected_phase=expected_phase,
                head_sha=ctx.sha,
                source=f"{phase}:{key}",
                prior_hashes=prior_hashes,
            )

        check = OutputCheck(
            validate=validate, kind="reviewer", expected_phase=expected_phase, prior=review_prior
        )
        try:
            return _reviewer_lane(
                ctx,
                phase=phase,
                role=role,
                lm=lane_lm,
                prompt=prompts[role],
                check=check,
                fresh=fresh,
                should_stop=twin_stopped if comparison else stopped,
                comparison=comparison,
                stop_reason=stop_reason,
            )
        except Exception as exc:  # a comparison lane never fails the phase, whatever broke
            if not comparison or abandon.is_set() or isinstance(exc, Cancelled):
                raise  # (a comparison lane stopped because the phase failed says nothing)
            if isinstance(exc, ReviewError):
                why = exc.message
            elif isinstance(exc, LaneStopped):
                why = (
                    f"still running {COMPARE_GRACE_MINUTES} min after the primary lanes finished"
                    if exc.started
                    else f"never started within {COMPARE_GRACE_MINUTES} min of the primary lanes "
                    "(no compare slot, or no idle production slot)"
                )
            else:
                why = f"{type(exc).__name__}: {exc}"
            drop(key, lane_lm.model, why)
            return None
        finally:
            ctx.close_lane_conn()

    results: dict[str, ReviewerOutput | None] = {}
    errors: list[BaseException] = []
    workers = min(len(roles), ctx.cfg.phase_parallelism or len(roles))
    with contextlib.ExitStack() as stack:
        pools = [
            stack.enter_context(
                ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"{phase}-{name}")
            )
            for name in ("lane", "compare")
        ]
        futures = {pools[lane[3]].submit(review, *lane): lane[0] for lane in lanes}
        primaries = {f for f, k in futures.items() if not k.endswith(COMPARE_SUFFIX)}
        pending = set(futures)
        grace_until: float | None = None  # set once every primary lane is done
        try:
            while pending:
                timeout = (
                    None
                    if grace_until is None or twins_stop.is_set()
                    else max(0.0, grace_until - time.monotonic())
                )
                done, pending = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
                for fut in done:
                    try:
                        results[futures[fut]] = fut.result()
                    except BaseException as exc:  # Cancelled included: it must stop the siblings
                        errors.append(exc)
                        abandon.set(_stop_note(phase, futures[fut], exc))
                        for f in futures:
                            f.cancel()  # roles not started yet never start
                if grace_until is None and primaries.isdisjoint(pending):
                    grace_until = time.monotonic() + COMPARE_GRACE_MINUTES * 60
                if grace_until is not None and time.monotonic() >= grace_until:
                    twins_stop.set()
        finally:
            if len(results) < len(lanes):
                # however we leave (an interrupt too), no lane outlives the phase
                abandon.set(f"stopped: {phase} was abandoned")
    if errors:
        # a cancelled run wins; else the failure that stopped the others, never a stopped lane
        raise next(
            (e for e in errors if isinstance(e, Cancelled)),
            next((e for e in errors if not isinstance(e, LaneStopped)), errors[0]),
        )
    if len(results) < len(lanes):
        ctx.check_cancel()
        raise ReviewError(FailKind.INFRA, f"{phase}: a reviewer lane never ran")
    # provenance and finding rows in lane order, as if the lanes had run one after another.
    # Only this call's own entries move: on a single-stage run the other phase is appending
    # to the same list at the same time.
    with ctx.state_lock:
        mine = [
            i
            for i, r in enumerate(ctx.reviewers)
            if r["phase"] == phase and r["fresh"] == fresh and r["key"] in keys
        ]
        ordered = sorted((ctx.reviewers[i] for i in mine), key=lambda r: keys.index(r["key"]))
        for i, r in zip(mine, ordered, strict=True):
            ctx.reviewers[i] = r
    outputs = {k: out for k in keys if (out := results[k]) is not None}
    twins = [k for k in keys if k.endswith(COMPARE_SUFFIX)]
    if twins:
        kept = sum(k in outputs for k in twins)
        with tx(ctx.conn):
            event(
                ctx.conn,
                "compare.lanes",
                repo=ctx.repo,
                number=ctx.number,
                run_id=ctx.run_id,
                detail=f"phase={phase} kept={kept} dropped={len(twins) - kept}",
            )
    with tx(ctx.conn):
        for k, output in outputs.items():
            # `<kind>:<role>` so `reviewsys compare` can pair each role's primary and
            # comparison lane, first round only; nothing else reads these stages
            if k.endswith(COMPARE_SUFFIX):
                stage = f"compare:{k.removesuffix(COMPARE_SUFFIX)}"
            else:
                stage = f"{'fresh' if fresh else 'lane'}:{k}"
            _insert_findings(ctx, phase, stage, output.findings)
    return outputs


def _brief(text: str, n: int = 160) -> str:
    """`text` on one line, at most `n` characters."""
    one = " ".join(text.split())
    return one if len(one) <= n else one[: n - 1] + "…"


def _stop_note(phase: str, key: str, exc: BaseException) -> str:
    """Why a phase's other lanes are being stopped, for their rows: the lane that failed and
    a short reason (model output never reaches the public page; lane rows are private)."""
    if isinstance(exc, Cancelled):
        return "run cancelled"
    if isinstance(exc, LaneStopped):  # stopped by the caller's abort, which says why
        return f"stopped: {phase} was abandoned"
    if isinstance(exc, ReviewError):
        cause = exc.cause if isinstance(exc, LaneFailed) else exc.message
        what = "output failed the contract" if exc.kind is FailKind.CONTRACT else "failed"
        return f"stopped: {phase}/{key} {what} ({_brief(cause)})"
    return f"stopped: {phase}/{key} crashed ({type(exc).__name__})"


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
        phase2_outputs=_blind({r: o.raw for r, o in all_phase2.items()}, seed=ctx.run_id),
        coderabbit=ctx.coderabbit,
        coderabbit_ids=ctx.coderabbit_ids,
        evidence=ctx.evidence,
        prior=ctx.prior,
        prior_sha=ctx.prior_sha,
        phase1_skipped=ctx.phase1_skipped,
        fresh_final=fresh_final,
        single_stage=ctx.single_stage,
    )
    prior_hashes = {str(p["finding_hash"]) for p in ctx.prior}

    def validate(raw: dict[str, Any]) -> VerifierOutput:
        return parse_verifier_output(
            raw,
            expected_phase=expected_phase,
            expected_coderabbit_ids=ctx.coderabbit_ids,
            prior_hashes=prior_hashes,
        )

    out = _run_checked_lane(
        ctx,
        phase=phase,
        role="verifier",
        lm=lm,
        prompt=prompt,
        is_verifier=True,
        check=OutputCheck(
            validate=validate,
            kind="verifier",
            expected_phase=expected_phase,
            prior=ctx.prior,
            coderabbit_ids=ctx.coderabbit_ids,
        ),
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
    _record_findings(ctx, phase, "verified-fresh" if fresh_final else "verified", out.findings)
    return out


def _blind(outputs: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """Phase-2 outputs as the verifier sees them on a comparison run: each role's pair gets
    the neutral labels `<role>/a` and `<role>/b`, which one is the comparison model decided
    per run, so neither the `#2` key nor the order gives the second model away. Unchanged
    when no comparison lane ran. Attribution does not depend on these labels (it matches
    each lane's own findings by hash)."""
    twins = {k.removesuffix(COMPARE_SUFFIX) for k in outputs if k.endswith(COMPARE_SUFFIX)}
    if not twins:
        return outputs
    second_is_a = random.Random(seed).random() < 0.5
    out: dict[str, Any] = {}
    for key, value in outputs.items():
        base = key.removesuffix(COMPARE_SUFFIX)
        if base in twins:
            key = f"{base}/{'a' if key.endswith(COMPARE_SUFFIX) == second_is_a else 'b'}"
        out[key] = value
    return dict(sorted(out.items()))


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
    # obsolete head; the run ends superseded (HeadObsolete) and ingest queues the
    # live commit.
    live = github.pr_meta(ctx.gh, ctx.repo, ctx.number)
    _require_live(ctx, live)
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
    if not ctx.open_threads or ctx.dry_run:
        return
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
    lifted = [
        str(r.get("finding_hash"))
        for r in _reconciliation(ctx, phase, verified).values()
        if r.get("status") in {"WITHDRAWN", "FIXED", "OUTDATED", "INTENTIONALLY_DEFERRED"}
    ]
    titles = [
        t["title"]
        for h, t in ctx.open_threads.items()
        if h in lifted and t.get("title") and t.get("severity") == "blocking"
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
        lifted_blockers=titles,
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
    for key, out in outputs.items():
        if key.endswith(COMPARE_SUFFIX):
            continue  # a comparison lane never answers or resolves a thread
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
    # comparison lanes are dropped on failure: disclose the comparison only when one finished
    compared = any(r["key"].endswith(COMPARE_SUFFIX) for r in ctx.reviewers)
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
        compare_model=ctx.compare_model if compared else None,
        single_stage=ctx.single_stage and not ctx.phase1_skipped,
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
    ctx: RunContext,
    model: str,
    *,
    error: str,
    upstream: str,
    detail: str,
    already_counts: bool = False,
) -> bool:
    """`model` just failed with `error` (a lane's stderr, never model output): when that says
    the pool is out of quota and the policy has a stand-in, switch this run into degraded
    mode and record the failure with a hold, so the next runs start degraded too. False when
    the mode is already on, no stand-in applies, or the error is not about quota.

    The published reason is a sanitised version of `upstream`; the full `detail` only reaches
    the events table. `already_counts`: True as well when a parallel sibling lane flipped the
    run first (this lane was already running on the primary and died on the same dry pool)."""
    pol = ctx.cfg.policy.degraded
    with ctx.state_lock:  # parallel lanes on the same dry pool must flip the run only once
        if (
            pol is None
            or model not in pol.substitutes
            or not degraded.looks_like_quota_failure(error)
        ):
            return False
        if ctx.is_degraded:
            return already_counts
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
        error=res.infra_error,
        upstream=res.first_stderr_line,
        detail=f"{lm.model} lane: {exc.message}",
        already_counts=True,
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
    # the lane that answered: a run that went degraded after the primary rated it (a later
    # lane hit the dry pool, or only the selection half was retried on the stand-in) still
    # names the primary
    ran = lm if t.method == f"llm:{lm.model}" else ctx.lane_model(lm)
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
    abort: StopSignal | None = None,
) -> dict[str, ReviewerOutput]:
    # the planned lanes, so a phase that is still going can say how many of them are done
    _step_start(ctx, step, {"roles": _phase_roles(ctx, phase)})
    if callable(lm):
        lm = lm()  # Phase 1 picks its model inside the step, so a slow lookup shows there
    outputs = _reviewer_lanes(
        ctx, phase=phase, lm=lm, expected_phase=expected_phase, fresh=fresh, abort=abort
    )
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
    gate_deferred: bool = False,
) -> None:
    """`gate_deferred`: the points gate held Phase 2 back; the gate comment shows the score."""
    if ctx.is_audit:
        _audit_publish(
            ctx,
            phase=phase,
            verified=verified,
            verifier_lm=verifier_lm,
            phase2_skipped=phase2_skipped,
        )
        return
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
    score = gate_score(verified, ctx.cfg.policy.phase1_gate) if gate_deferred else {}
    _gate_comment(
        ctx,
        "done",
        phase=phase,
        blocker_count=verified.blocker_count,
        phase2_skipped=phase2_skipped,
        **score,
    )


def _audit_publish(
    ctx: RunContext,
    *,
    phase: str,
    verified: VerifierOutput,
    verifier_lm: LaneModel,
    phase2_skipped: str | None = None,
) -> None:
    """An audit never posts a review: the verdict goes to the report (and, for a merge the
    sweep caught live, to a post-merge comment plus one issue while blockers are still
    present). A light audit that finds a blocker is re-queued as a full one instead."""
    assert ctx.audit is not None and ctx.meta is not None
    if ctx.audit["mode"] == audit.MODE_LIGHT and verified.blocker_count:
        audit.escalate(ctx.conn, int(ctx.audit["id"]), ctx.head_id)
        return
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
    # dry_run: the dedupe pass must not reply on the merged PR's old threads
    model = publish.build(
        ctx.gh,
        repo=ctx.repo,
        number=ctx.number,
        head_sha=ctx.sha,
        phase=phase,
        verified=verified,
        provenance=prov,
        bot_login=ctx.cfg.bot_login,
        diff_text=None,
        dry_run=True,
    )
    body = publish.render(model)
    (ctx.run_dir / f"audit-review-{phase}.md").write_text(body)
    persistence, tip_ref, tip_sha = step_persistence(ctx, verified)
    outcome = audit.Outcome(
        audit=ctx.audit,
        verdict=publish.verdict_event(verified, prov),
        findings=list(verified.findings),
        persistence=persistence,
        tip_ref=tip_ref,
        tip_sha=tip_sha,
        review_body=body,
        degraded=ctx.is_degraded,
        run_id=ctx.run_id,
    )
    _step_start(ctx, StepName.PUBLISH)
    audit.write_report(ctx.cfg, outcome)
    comment_url = issue_url = None
    if ctx.audit["source"] == audit.SOURCE_LIVE and ctx.cfg.audit.post_live and not ctx.dry_run:
        aid = int(ctx.audit["id"])
        comment_url, issue_url = audit.post_live(
            ctx.gh,
            outcome,
            bot_login=ctx.cfg.bot_login,
            persist=lambda col, url: audit.persist_url(ctx.conn, aid, col, url),
        )
    with tx(ctx.conn):
        audit.record(ctx.conn, outcome, comment_url=comment_url, issue_url=issue_url)
        _insert_findings(ctx, phase, "audit-posted", verified.findings)
        _set_run_review(
            ctx, blocker_count=verified.blocker_count, review_id=None, review_url=comment_url
        )
    _step_end(
        ctx,
        StepName.PUBLISH,
        "ok",
        {
            "audit": True,
            "verdict": outcome.verdict,
            "blockers": len(outcome.blockers),
            "still_present": len(outcome.open_blockers),
            "comment": comment_url,
            "issue": issue_url,
        },
    )


def step_persistence(
    ctx: RunContext, verified: VerifierOutput
) -> tuple[dict[str, dict[str, Any]], str, str]:
    """Audit: for every blocker, is it still present on the base branch tip today? One lane in
    a worktree at the tip, read-only. No blockers: nothing to check. A failed lane leaves every
    blocker UNKNOWN (counted as open), never FIXED."""
    assert ctx.audit is not None and ctx.meta is not None and ctx.mirror is not None
    blockers = [f for f in verified.findings if f.severity == "blocking"]
    tip_ref = ctx.meta.base_ref or str(ctx.audit.get("base_ref") or "")
    hashes = [f.hash for f in blockers]
    if not blockers:
        return {}, tip_ref, ""
    _step_start(ctx, StepName.PERSISTENCE)
    tip_sha = wt.fetch_branch(ctx.mirror, tip_ref) if tip_ref else None
    if tip_sha is None:
        # the base branch is gone (a feature branch deleted after it merged on): check against
        # where the merge landed instead
        tip_sha = str(ctx.audit.get("merge_commit") or "") or ctx.sha
        tip_ref = f"{tip_ref or 'base'} (deleted; merge commit)"
    tip_wt = None
    try:
        wt.ensure_commit(ctx.mirror, tip_sha)
        tip_wt = wt.create_worktree(
            ctx.mirror, ctx.cfg.worktrees_dir, _worktree_name(ctx, "-tip"), tip_sha
        )
        wt.ensure_commit(tip_wt, ctx.sha)  # `git show <head>:path` must work from the tip
        prompt = audit.persistence_prompt(
            repo=ctx.repo,
            number=ctx.number,
            title=ctx.meta.title,
            merged_at=str(ctx.audit.get("merged_at") or ""),
            head_sha=ctx.sha,
            tip_ref=tip_ref,
            tip_sha=tip_sha,
            blockers=blockers,
        )
        lm = dataclasses.replace(ctx.cfg.policy.conversation_lane, agent="audit-persistence")
        saved, ctx.worktree = ctx.worktree, tip_wt
        try:
            raw = _run_lane(
                ctx, phase="persistence", role="persistence", lm=lm, prompt=prompt, is_verifier=True
            )
        finally:
            ctx.worktree = saved
        result = audit.parse_persistence(raw, hashes)
        status = "ok"
    except ReviewError as exc:
        if ctx.audit["source"] == audit.SOURCE_LIVE and exc.kind is FailKind.INFRA:
            # a live audit posts an issue from this answer: never on a lane that merely died
            raise
        log.warning("persistence check failed: %s", exc)
        result = audit.parse_persistence({}, hashes)
        status = "failed"
    finally:
        if tip_wt is not None:
            wt.remove_worktree(ctx.mirror, tip_wt)
    (ctx.run_dir / "persistence.json").write_text(json.dumps(result, indent=1))
    _step_end(
        ctx,
        StepName.PERSISTENCE,
        status,
        {
            "tip": f"{tip_ref}@{tip_sha[:8]}",
            "statuses": {h: r["status"] for h, r in result.items()},
        },
    )
    return result, tip_ref, tip_sha


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
    ctx.phase1_lm = ctx.lane_model(
        _with_effort(choice.model, min_effort(ctx.phase1_effort or cap, cap))
    )
    return ctx.phase1_lm


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
    with ctx.state_lock:
        if ctx.phase1_lm is not None and ctx.phase1_lm != failed:
            # a parallel sibling already moved the run off this rung: follow it there
            return ctx.phase1_lm
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
    if ctx.is_audit or limit <= 0 or not ctx.cfg.policy.phase2_enabled or phase2_effort is None:
        return False
    dp = ctx.degraded_policy
    if dp is not None and not dp.backlog_skip_phase1:
        # opt-in: keep both phases in degraded mode so the cross-model check survives, at the
        # cost of the slow Phase-1 rungs. Off by default -- a deep queue in degraded mode is
        # the case that can least afford them.
        return False
    queued = live_queued_count(ctx.conn)
    if queued <= limit:
        return False
    reason = f"{github.PHASE1_BACKLOG} {queued} PRs queued, above the {limit} limit"
    _skip_phase1(ctx, reason, "phase1.skipped_backlog", queued=queued, limit=limit)
    return True


def _skip_phase1(ctx: RunContext, reason: str, kind: str, **detail: Any) -> None:
    """Record a Phase 1 that will not run: a `skipped` phase1 step, an event, and the gate
    comment; `ctx.phase1_skipped` carries the reason into the verifier prompt and the review."""
    ctx.phase1_skipped = reason
    _step_start(ctx, StepName.PHASE1)
    _step_end(ctx, StepName.PHASE1, "skipped", {"reason": reason, **detail})
    with tx(ctx.conn):
        event(ctx.conn, kind, repo=ctx.repo, number=ctx.number, run_id=ctx.run_id, detail=reason)
    _gate_comment(ctx, "in_progress")


def _choose_comparison(ctx: RunContext) -> None:
    """Sample this run for a model comparison (see ComparisonPolicy). Never on an audit or in
    degraded mode (both sets would run on the same stand-in)."""
    cmp = ctx.cfg.policy.comparison
    if (
        cmp is None
        or ctx.is_audit
        or ctx.is_degraded
        or not cmp.selects(ctx.repo, ctx.number, ctx.sha, ctx.tier)
    ):
        return
    ctx.compare_model = cmp.model
    with tx(ctx.conn):
        event(
            ctx.conn,
            "compare.selected",
            repo=ctx.repo,
            number=ctx.number,
            run_id=ctx.run_id,
            detail=f"tier={ctx.tier} primary={ctx.cfg.policy.phase2_reviewer.model} second={cmp.model}",
        )


COMPARE_SUFFIX = "#2"  # output key of a comparison lane: `<role>#2`
# how long comparison lanes may still run once every primary lane of their phase is done
COMPARE_GRACE_MINUTES = 20
LIGHT_AUDIT_TIER = "trivial"  # the policy tier a light audit reviews at (Phase 1 only)
PHASE1_FAILED = github.PHASE1_FAILED
PHASE1_REPO_OFF = github.PHASE1_REPO_OFF
# the preliminary review's Phase-2 note when the gate deferred on suggestions alone
PHASE2_DEFERRED = "deferred by the Phase-1 gate"
# Phase-2 effort for a tier without a Phase 2 (trivial) on a repo that has no Phase 1
NO_PHASE1_TRIVIAL_EFFORT = "low"


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
    # follow-up must never be posted against an obsolete commit (the run ends superseded and
    # ingest queues the live one)
    _require_live(ctx, github.pr_meta(ctx.gh, ctx.repo, ctx.number))
    out = converse.accept_deferrals(out, threads)
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
    lifted = {h for h in posted_ok if out.outcomes[h].status in converse.LIFTING_STATUSES}
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
    kept = _standing_findings(ctx, phase, lifted=lifted)
    update = None
    if not ctx.dry_run:
        update = _conversation_verdict_update(ctx, phase, lm, kept, lifted=lifted)
    _gate_comment(ctx, "done", phase=phase, blocker_count=kept.blockers)
    return {
        "answered": len(posted_ok),
        "outcomes": {h: o.status for h, o in out.outcomes.items()},
        "linked_commits": {
            "fetched": [c["sha"] for c in fetched],
            "unfetched": [c["sha"] for c in unfetched],
        },
        "blockers_remaining": kept.blockers,
        "verdict_updated": bool(update and update.posted),
    }


def _record_conceded(ctx: RunContext, phase: str, lifted: set[str]) -> None:
    """A finding the conversation withdrew or accepted as deferred gets a `conceded` findings
    row for this run (and so this sha), which `_standing_findings` honours on every later run. Caller holds the write transaction."""
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


@dataclass(slots=True)
class _Standing:
    """What still stands of this commit's `phase` publication."""

    findings: dict[str, str]  # finding hash -> severity, lifted ones removed
    run_id: int | None  # the run that published it (None: no recorded findings)
    verified: bool  # that run's final verifier set is on record (an approval needs it)
    lifted_any: bool  # at least one of that set's non-nitpick findings has been lifted

    @property
    def blockers(self) -> int:
        return sum(1 for sev in self.findings.values() if sev == "blocking")


def _identity(file: Any, title: Any) -> tuple[str, str]:
    return str(file or ""), str(title or "").strip().lower()


def _standing_findings(ctx: RunContext, phase: str, *, lifted: set[str] | None = None) -> _Standing:
    """Findings on this commit's `phase` publication that still stand.

    The publication is the latest run that posted findings for this sha (a same-sha re-review
    refreshes it). Its standing set is that run's final verifier set (`verify2` rows, the
    fresh audit's when it ran), which includes findings that dedupe carried onto an existing
    thread instead of posting; runs without those rows fall back to their `posted` rows. Plus
    unresolved bot finding threads this database never posted (legacy findings); threads from
    earlier heads were re-adjudicated by that run. Minus: findings a conversation on this sha
    conceded after that run (`conceded` rows) and those lifted by the current run, matched by
    hash or by file and title (a carried finding may be re-hashed). A thread a maintainer
    resolved by hand, without the bot conceding it, still counts: only an explicit outcome or
    a fresh review lifts a verdict. Scoped to the phase whose verdict is being moved, so a
    preliminary publication stacked on the same sha never leaks into the final accounting.
    """
    rows = ctx.conn.execute(
        "SELECT f.run_id, f.hash, f.severity, f.stage, f.file, f.title FROM findings f "
        "JOIN runs r ON r.id=f.run_id JOIN heads h ON h.id=r.head_id "
        "WHERE h.repo=? AND h.number=? AND h.sha=? AND f.phase=? AND f.stage IN ('posted','conceded')",
        (ctx.repo, ctx.number, ctx.sha, phase),
    ).fetchall()
    run_id = max((int(r["run_id"]) for r in rows if r["stage"] == "posted"), default=None)
    pub: dict[str, tuple[str, tuple[str, str]]] = {}
    verified = False
    if run_id is not None:
        final = ctx.conn.execute(
            "SELECT hash, severity, stage, file, title FROM findings WHERE run_id=? "
            "AND phase='verify2' AND stage IN ('verified','verified-fresh')",
            (run_id,),
        ).fetchall()
        fresh = [r for r in final if r["stage"] == "verified-fresh"]
        if phase == "final" and final:
            verified = True
            for r in fresh or final:
                pub[str(r["hash"])] = (str(r["severity"]), _identity(r["file"], r["title"]))
        for r in rows:
            if r["stage"] == "posted" and int(r["run_id"]) == run_id:
                pub.setdefault(
                    str(r["hash"]), (str(r["severity"]), _identity(r["file"], r["title"]))
                )
    ever_posted = {
        str(r["hash"])
        for r in ctx.conn.execute(
            "SELECT hash FROM posted_findings WHERE repo=? AND number=?", (ctx.repo, ctx.number)
        )
    } | {str(r["hash"]) for r in rows}
    for h, t in ctx.open_threads.items():
        if h not in pub and h not in ever_posted:
            pub[h] = (str(t.get("severity") or "nitpick"), _identity(t.get("path"), t.get("title")))
    # a later full review of this sha re-adjudicated everything: older concessions are void
    gone = {
        str(r["hash"]): _identity(r["file"], r["title"])
        for r in rows
        if r["stage"] == "conceded" and (run_id is None or int(r["run_id"]) > run_id)
    }
    for h in lifted or set():
        t = ctx.open_threads.get(h) or {}
        gone[h] = _identity(t.get("path"), t.get("title"))
    gone_ids = {i for i in gone.values() if i[1]}
    lifted_hashes = {h for h, (_, ident) in pub.items() if h in gone or ident in gone_ids}
    return _Standing(
        findings={h: sev for h, (sev, _) in pub.items() if h not in lifted_hashes},
        run_id=run_id,
        verified=verified,
        lifted_any=any(pub[h][0] != "nitpick" for h in lifted_hashes),
    )


def _open_blockers(ctx: RunContext, phase: str, *, lifted: set[str] | None = None) -> int:
    """Blocking findings on this commit's `phase` publication that still stand."""
    return _standing_findings(ctx, phase, lifted=lifted).blockers


def _review_strength(ctx: RunContext, run_id: int | None) -> str:
    """Whether the run that published a commit's standing review is one an APPROVE may rest
    on, by the pipeline's own bar: Phase 2 completed on the primary models (`verdict_event`),
    and on a later review round the fresh Phase-2 audit ran too (`_fresh_final_if_needed`,
    which skips it while blockers stand). "ok", "needs_fresh" or "weak"."""
    if run_id is None:
        return "weak"
    row = ctx.conn.execute(
        "SELECT r.degraded, "
        "(SELECT status FROM steps WHERE run_id=r.id AND name=?) AS phase2, "
        "(SELECT status FROM steps WHERE run_id=r.id AND name=?) AS fresh, "
        "EXISTS (SELECT 1 FROM reviews WHERE repo=? AND number=? AND run_id<r.id) AS later "
        "FROM runs r WHERE r.id=?",
        (StepName.PHASE2.value, StepName.FRESH_PHASE2.value, ctx.repo, ctx.number, run_id),
    ).fetchone()
    if not row or row["degraded"] or row["phase2"] != "ok":
        return "weak"
    return "needs_fresh" if row["later"] and row["fresh"] != "ok" else "ok"


def _request_fresh_audit(ctx: RunContext) -> bool:
    """Ask for one same-sha re-review once this conversation run ends, so the fresh Phase-2
    audit the standing review skipped can run and approve. Once per commit: a re-review that
    re-raises a finding is answered in its threads like any other. True if requested now."""
    guard = f"converse.fresh_audit:{ctx.repo}#{ctx.number}:{ctx.sha}"
    with tx(ctx.conn):
        if kv_get(ctx.conn, guard):
            return False
        kv_set(ctx.conn, guard, str(ctx.run_id))
        # read by scheduler.finish_run once this run is DONE (its head is RUNNING until then)
        kv_set(ctx.conn, f"rereview.after_run:{ctx.run_id}", "1")
        event(
            ctx.conn,
            "review.fresh_audit_requested",
            repo=ctx.repo,
            number=ctx.number,
            run_id=ctx.run_id,
            detail=f"{ctx.sha[:8]}: every finding lifted in discussion; the approval needs the fresh Phase-2 audit",
        )
    return True


def _conversation_verdict_update(
    ctx: RunContext, phase: str, lm: LaneModel, standing: _Standing, *, lifted: set[str]
) -> publish.PublishResult | None:
    """Move the verdict on this commit after a conversation lifted findings of its review.

    The verdict is the commit's full review minus what the discussion withdrew or deferred.
    When nothing above a nitpick is left of the verifier's set, something of it was lifted,
    and that review is one the pipeline itself would approve on (`_review_strength`), the
    follow-up APPROVEs: a COMMENT cannot clear a REQUEST_CHANGES on GitHub, so anything less
    leaves a ready pull request showing a blocking review that no longer exists
    (dashpay/dash#7778). A later-round review that skipped the fresh audit gets that audit
    queued instead. Otherwise, with every blocker gone, a standing REQUEST_CHANGES becomes
    COMMENT. A conversation never adds blockers, and posts nothing when the verdict would not
    move."""
    if standing.blockers or not lifted:
        return None
    # the lane ran for minutes: a push or a dismissal since then must not get an approval
    _require_live(ctx, github.pr_meta(ctx.gh, ctx.repo, ctx.number))
    reviews = github.reviews(ctx.gh, ctx.repo, ctx.number)
    state = publish.standing_verdict(reviews, ctx.sha, phase, ctx.cfg.bot_login)
    if state not in {"CHANGES_REQUESTED", "COMMENTED"}:
        return None  # approved already, or dismissed: a human overrode us; do not re-assert
    left = sum(1 for sev in standing.findings.values() if sev != "nitpick")
    strength = _review_strength(ctx, standing.run_id)
    clear = standing.verified and standing.lifted_any and not left
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
        review_action="APPROVE" if clear and strength == "ok" else "COMMENT",
        findings=[],
        dropped=[],
        out_of_scope=[],
        coderabbit_reactions=[],
        prerequisite_adjudications=[],
        adjudication_complete=True,
        review_phase=phase,
        raw={},
    )
    new_event = publish.verdict_event(verified, prov)
    source = f"the full review of `{ctx.sha[:8]}`" + (
        f" (run {standing.run_id})" if standing.run_id else ""
    )
    if new_event == "APPROVE":
        note = (
            f"Approved: the discussion withdrew or deferred every finding {source} raised, "
            "and the approval rests on that review."
        )
    elif left:
        note = f"Not approved: {left} non-blocking finding(s) from {source} still stand."
    elif prov.degraded:
        note = "Not approved: this follow-up ran on a stand-in model (degraded mode)."
    elif clear and strength == "needs_fresh":
        queued = _request_fresh_audit(ctx)
        note = (
            "Not approved yet: on a later review round an approval needs the fresh Phase-2 "
            "audit of the whole change, which the review of this commit skipped because it "
            "had found a blocker. "
            + (
                "That audit is queued for this commit and approves if it finds nothing."
                if queued
                else "It was already requested once for this commit; push or ask for a re-review."
            )
        )
    elif not standing.verified:
        note = (
            "Not approved: the verifier record an approval after discussion rests on is not "
            "available for the standing review of this commit."
        )
    elif not standing.lifted_any:
        note = "Not approved: the findings settled here are not the ones the standing review of this commit raised."
    else:
        note = (
            "Not approved: the standing review of this commit did not run at full strength "
            "(Phase 2 or the primary models were missing), so it cannot carry an approval."
        )
    assert ctx.meta
    own = ctx.meta.author.lower() == ctx.cfg.bot_login.lower()
    if publish.EVENT_STATE[publish.transport_event(new_event, own_pr=own)] == state:
        _record_verdict(ctx, phase, verified, new_event)
        return None
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
        lifted_blockers=titles,
        bot_login=ctx.cfg.bot_login,
        note=note,
    )
    what = "every finding" if result.event == "APPROVE" else "every blocker"
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
            detail=f"{state} -> {result.event} on {ctx.sha[:8]} (conversation: {what} withdrawn or deferred)",
        )
    return result


def run(ctx: RunContext) -> RunStatus:
    pol = ctx.cfg.policy
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    _detect_degraded(ctx)
    standing = _standing_review(ctx) if ctx.trigger == Trigger.REVIEW_REPLY else None
    ctx.conversation = standing is not None
    ctx.check_cancel()
    _step_start(ctx, StepName.WORKTREE)
    step_worktree(ctx)
    _step_end(ctx, StepName.WORKTREE, "ok")
    if not standing:  # a conversation needs no specialist selection or effort triage
        ctx.check_cancel()
        step_prep(ctx)
    ctx.check_cancel()
    _step_start(ctx, StepName.CONTEXT)
    step_context(ctx)
    _step_end(ctx, StepName.CONTEXT, "ok")
    if standing:
        # a human replied on a commit we already reviewed: the code did not change, the
        # discussion did. Answer the threads; never re-run the review pipeline for that.
        _step_start(ctx, StepName.CONVERSE)
        detail = step_converse(ctx, standing)
        _step_end(ctx, StepName.CONVERSE, "ok", detail)
        return RunStatus.DONE
    effort = pol.tier_effort(ctx.tier)
    ctx.phase2_follows = pol.phase2_enabled and effort.phase2 is not None
    _choose_comparison(ctx)
    if _repo_skips_phase1(ctx):
        return _phase2_only(ctx, _without_phase1(effort))
    if _backlog_skips_phase1(ctx, phase2_effort=effort.phase2):
        return _phase2_only(ctx, effort)
    if effort.single_stage and pol.phase2_enabled and effort.phase2 and not ctx.is_audit:
        return _single_stage(ctx, effort)
    try:
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
    except ReviewError as exc:
        if not _phase1_failure_falls_through(ctx, exc, phase2_effort=effort.phase2):
            raise
        return _phase2_only(ctx, effort)
    _step_start(ctx, StepName.GATE)
    tier_allows = effort.phase2 is not None
    # the gate only defers a Phase 2 that would run; a head whose final review already stands
    # (a same-sha re-review) is never held back, nor its verdict retracted, by suggestions alone
    blocks = (
        pol.phase2_enabled
        and tier_allows
        and phase1_blocks(ctx.verify1, pol.phase1_gate)
        and not (ctx.verify1.blocker_count == 0 and _final_review_stands(ctx))
    )
    admit = pol.phase2_enabled and tier_allows and not blocks
    if ctx.audit and ctx.audit["mode"] != audit.MODE_LIGHT:
        # nobody will push a fix and re-run the gate on a merged PR: an audit needs the
        # complete finding set, so Phase 2 runs whether or not Phase 1 found blockers
        admit = pol.phase2_enabled and tier_allows
    _step_end(
        ctx,
        StepName.GATE,
        "ok",
        {
            "admit_phase2": admit,
            "tier": ctx.tier,
            "phase2_effort": effort.phase2,
            "blockers": ctx.verify1.blocker_count,
            **gate_score(ctx.verify1, pol.phase1_gate),
        },
    )
    if not admit:
        ctx.check_cancel()
        verifier1 = ctx.lane_model(pol.phase1_verifier)
        if blocks or ctx.verify1.blocker_count or not pol.phase2_enabled:
            # a preliminary review never approves: Phase 2 has not seen this head. Held back
            # by suggestions alone, it would otherwise carry the gate verifier's APPROVE.
            _publish_step(
                ctx,
                phase="preliminary",
                verified=ctx.verify1,
                verifier_lm=verifier1,
                phase2_skipped=PHASE2_DEFERRED
                if blocks and not ctx.verify1.blocker_count
                else None,
                gate_deferred=blocks,
            )
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
    return _phase2_only(ctx, effort)


def _phase2_only(ctx: RunContext, effort: TierEffort) -> RunStatus:
    """Phase 2 and the final verifier on their own, when Phase 1 did not run (a deep queue)
    or did not finish (every rung of its ladder failed)."""
    _phase2_reviewers(ctx, effort)
    return _final_review(ctx, effort)


def _phase2_reviewers(ctx: RunContext, effort: TierEffort, abort: StopSignal | None = None) -> None:
    ctx.phase2_outputs = _reviewer_step(
        ctx,
        step=StepName.PHASE2,
        phase="phase2",
        lm=ctx.lane_model(_with_effort(ctx.cfg.policy.phase2_reviewer, effort.phase2)),
        expected_phase="final",
        abort=abort,
    )


def _final_review(ctx: RunContext, effort: TierEffort) -> RunStatus:
    """The final verifier over the Phase-2 (and any Phase-1) output, then the final review."""
    pol = ctx.cfg.policy
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


def _final_review_stands(ctx: RunContext) -> bool:
    """A final review for this exact head is already on the PR (a same-sha re-review)."""
    if ctx.dry_run:
        return False
    try:
        return (
            github.existing_review_for_sha(
                ctx.gh, ctx.repo, ctx.number, ctx.sha, "final", ctx.cfg.bot_login
            )
            is not None
        )
    except ReviewError as exc:  # unknown: gate as usual; step_publish re-checks before posting
        log.warning("could not look up a standing final review: %s", exc)
        return False


def _repo_skips_phase1(ctx: RunContext) -> bool:
    """The repo is configured without Phase 1 (`phase1: false` in its skills entry): its
    reviews go straight to Phase 2, recorded and disclosed like the backlog rule. Not for an
    audit (it keeps its own flow) or when there is no Phase 2 to go to."""
    repo = ctx.cfg.repo(ctx.repo)
    if repo is None or repo.phase1 or ctx.is_audit or not ctx.cfg.policy.phase2_enabled:
        return False
    _skip_phase1(ctx, PHASE1_REPO_OFF, "phase1.skipped_repo")
    return True


def _without_phase1(effort: TierEffort) -> TierEffort:
    """The tier's efforts for a run that has no Phase 1: a tier that skips Phase 2 (trivial)
    reviews in Phase 2 at the lowest effort instead, since Phase 2 is all there is."""
    if effort.phase2 is not None:
        return effort
    return dataclasses.replace(effort, phase2=NO_PHASE1_TRIVIAL_EFFORT)


def _single_stage(ctx: RunContext, effort: TierEffort) -> RunStatus:
    """One stage, no blocker gate (a tier with `single_stage`): the Phase-1 reviewers on their
    ladder model run beside the Phase-2 reviewers, and the final verifier weighs both sets,
    so the review gets both models' view in the time of the slower phase. Phase 1 is extra
    coverage here: when it fails on every rung its output is dropped and disclosed, never the
    review. When Phase 2 fails, Phase 1 is stopped and the run fails as it would have."""
    ctx.single_stage = True
    abort = StopSignal()
    p1: dict[str, Any] = {}

    def phase1() -> None:
        try:
            p1["out"] = _reviewer_step(
                ctx,
                step=StepName.PHASE1,
                phase="phase1",
                lm=lambda: _choose_phase1(ctx, effort.phase1),
                expected_phase="preliminary",
                abort=abort,
            )
        except BaseException as exc:  # handed to the main thread below
            p1["exc"] = exc
            stopped = isinstance(exc, (Cancelled, LaneStopped)) or abort.is_set()
            try:  # a failed Phase 2 is charged to `phase2`; this step must not stay running
                if stopped or not isinstance(exc, Exception):
                    _step_end(ctx, StepName.PHASE1, "cancelled" if stopped else "failed")
                else:
                    # recorded the moment it fails (step error + event), not hours later when
                    # Phase 2 is done and its output is dropped
                    _record_phase1_failure(
                        ctx,
                        _failure_text(exc),
                        StepName.PHASE1,
                        "phase1.failed_single_stage",
                        dropped=True,
                    )
            except sqlite3.Error as db_exc:
                log.warning("could not close the single-stage phase1 step: %s", db_exc)
        finally:
            ctx.close_lane_conn()

    t = threading.Thread(target=phase1, name="single-stage-phase1", daemon=True)
    t.start()
    try:
        _phase2_reviewers(ctx, effort)
    except BaseException as p2_exc:
        abort.set(f"stopped: phase2 failed ({_brief(_failure_text(p2_exc))})")
        raise
    finally:
        t.join()
    exc = p1.get("exc")
    if exc is None:
        ctx.phase1_outputs = p1["out"]
    elif isinstance(exc, Cancelled) or not isinstance(exc, Exception):
        raise exc
    else:
        _drop_single_stage_phase1(ctx, exc)
    ctx.check_cancel()
    return _final_review(ctx, effort)


def _failure_text(exc: BaseException) -> str:
    return exc.message if isinstance(exc, ReviewError) else f"{type(exc).__name__}: {exc}"


def _drop_single_stage_phase1(ctx: RunContext, exc: Exception) -> None:
    """Phase 1 of a single-stage run failed while Phase 2 finished: review on Phase 2 alone.
    The failure itself (step error, `phase1.failed_single_stage`) was recorded when it
    happened, on the Phase-1 thread; only its output is dropped here."""
    log.warning(
        "single-stage phase1 failed (%s); publishing from Phase 2 alone", _failure_text(exc)
    )
    _discard_phase1(ctx)


def _drop_phase1(ctx: RunContext, msg: str, step: StepName, kind: str, **detail: Any) -> None:
    """Phase 1 failed and the run goes on without it: whatever it produced is dropped (never
    handed to the final verifier unverified), the failed step and an event keep the error, and
    the review only says Phase 1 failed (`PHASE1_FAILED`)."""
    _discard_phase1(ctx)
    _record_phase1_failure(ctx, msg, step, kind, **detail)


def _discard_phase1(ctx: RunContext) -> None:
    ctx.phase1_skipped = PHASE1_FAILED
    ctx.phase1_outputs, ctx.verify1 = {}, None
    with ctx.state_lock:
        ctx.reviewers[:] = [r for r in ctx.reviewers if r["phase"] != "phase1"]


def _record_phase1_failure(
    ctx: RunContext, msg: str, step: StepName, kind: str, **detail: Any
) -> None:
    """The failed step with its error (the status page shows it, sanitized) and the event."""
    _step_end(ctx, step, "failed", {"error": msg[:1000], **detail})
    with tx(ctx.conn):
        event(ctx.conn, kind, repo=ctx.repo, number=ctx.number, run_id=ctx.run_id, detail=msg[:500])


def _phase1_failure_falls_through(
    ctx: RunContext, exc: ReviewError, *, phase2_effort: str | None
) -> bool:
    """Phase 1 failed (a reviewer lane died on every rung of the ladder, the gate verifier
    died twice, or either returned output that breaks the contract): rather than failing the
    run and retrying it from scratch, review with Phase 2 alone, exactly like the backlog rule
    does. Whatever Phase 1 produced is dropped: unverified, it must not reach the final
    verifier as if it had been gated. The error text stays in the step and the event; the
    review only says Phase 1 failed.

    Not when there is no Phase 2 to fall through to (disabled, or the tier skips it), not for
    an audit (it needs the complete finding set and has no one waiting on it: it fails and is
    retried, like the backlog rule leaves audits alone), and not for a FATAL error (a broken
    prompt template would only fail Phase 2 the same way, after spending it)."""
    ctx.check_cancel()
    if (
        not ctx.cfg.policy.phase2_enabled
        or phase2_effort is None
        or ctx.is_audit
        or exc.kind is FailKind.FATAL
    ):
        return False
    log.warning("phase1 failed (%s); continuing with Phase 2 only", exc)
    row = ctx.conn.execute("SELECT phase FROM runs WHERE id=?", (ctx.run_id,)).fetchone()
    step = StepName(row["phase"]) if row and row["phase"] else StepName.PHASE1
    _drop_phase1(ctx, exc.message, step, "phase1.failed_fallthrough", fell_through=True)
    _gate_comment(ctx, "in_progress")
    return True


def cleanup(ctx: RunContext, status: RunStatus) -> None:
    if ctx.worktree and ctx.mirror and (status == RunStatus.DONE or ctx.is_audit):
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
        "SELECT r.*, h.repo, h.number, h.sha, h.trigger, h.queue, h.priority FROM runs r JOIN heads h ON h.id=r.head_id WHERE r.id=?",
        (run_id,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"run {run_id} not found")
    if RunStatus(row["status"]).terminal:
        return RunStatus(row["status"])
    ctx = RunContext(
        cfg=cfg,
        main_conn=conn,
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
    if row["queue"] == "audit":
        ctx.audit = audit.load(conn, ctx.head_id)
        if ctx.audit is None:
            raise SystemExit(f"run {run_id}: audit head without an audits row")
    # every lane of this run, parallel reviewers and side lanes alike, lines up for a slot in
    # its model's machine-wide pool (lanepool.py)
    ctx.lane_runner = lanepool.gated(
        sandboxed(lane_runner or run_claude_lane, exec_deny_profile(cfg.lane_deny_exec)),
        lambda: ctx.conn,
        cfg,
        run_stopped=ctx.cancel_flag.is_set,
        queue="audit" if ctx.is_audit else "priority" if row["priority"] else "live",
        run_id=run_id,
    )
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
    except HeadObsolete as exc:
        # nothing to review: end the head as ingest would have, with no failure, retry, alert
        # or failure gate comment (the newer head's queue comment takes the PR's comment over)
        status, reason = RunStatus.CANCELLED, f"head {exc.status.value}: {exc.reason}"
        current = ctx.conn.execute("SELECT phase FROM runs WHERE id=?", (run_id,)).fetchone()
        with tx(ctx.conn):
            if current and current["phase"]:
                ctx.conn.execute(
                    "UPDATE steps SET status='cancelled', finished_at=?, detail=? "
                    "WHERE run_id=? AND name=? AND status='running'",
                    (now(), json.dumps({"reason": reason}), run_id, current["phase"]),
                )
            retire_obsolete_head(ctx.conn, ctx.head_id, exc.status, exc.reason)
        if exc.status == HeadStatus.CLOSED:
            _gate_closed(ctx, exc.reason)
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
        with ctx.gate_lock:  # no progress edit may follow the run's end (requeue, cancel)
            ctx.gate_live_kw = None
        stop.set()
        if hb:
            hb.join(timeout=5)
        finish_run(
            conn,
            cfg,
            run_id,
            status,
            reason=reason,
            fail_kind=kind if status == RunStatus.FAILED else None,  # a cancel is no failure
        )
        cleanup(ctx, status)
    return status
