"""Worker: runs one review (one `runs` row) end to end.

Steps: worktree -> select + triage (one prep lane) -> context -> phase1 -> verify1 -> gate -> phase2 -> verify2 -> publish.
Policy v12 (`review_model_policy.pipeline`) runs inside the same steps: finder lanes on the
candidate contract, a thread lane beside them (`_start_thread_lane`), verify+match with the code
rules (`_verify12`), a confirmation pass (`_confirm12`) and a verdict from the issue ledger
(`_publish12`); a reply on a reviewed commit goes to `step_threads`.
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
import os
import random
import re
import shutil
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

from . import (
    audit,
    ci,
    converse,
    degraded,
    drift,
    github,
    labels,
    lanepool,
    late,
    ledger,
    publish,
    quota,
    rules,
    schemas,
    sister,
)
from . import prep as prep_mod
from . import progress as progress_mod
from .config import Config, DegradedPolicy, LaneModel, TierEffort, min_effort
from .contract import (
    Finding,
    ReviewerOutput,
    VerifierOutput,
    finding_dedupe_key,
    normalize_finding_text,
    parse_json_object,
    parse_reviewer_output,
    parse_verifier_output,
)
from .db import connect_existing, event, fmt_ts, kv_get, kv_set, now, now_dt, parse_ts, tx
from .dedupe import (
    FINDING_MARKER_RE,
    ExistingComment,
    carry_forward_tokens,
    content_tokens,
    find_duplicate,
    title_similarity,
)
from .degraded import Prober
from .gate import gate_score, phase1_blocks
from .gh import Gh
from .ingest import reviewed_sha
from .lane import (
    LaneResult,
    LaneRunner,
    LaneSpec,
    SchemaRetriesExhausted,
    exec_deny_profile,
    isolated,
    lane_output,
    new_session_id,
    prompt_sha,
    run_claude_lane,
    sandboxed,
)
from .models import FailKind, HeadStatus, ReviewError, RunStatus, StepName, Trigger
from .prompts import (
    V12_KINDS,
    confirmation_prompt,
    correction_prompt,
    finder_prompt,
    prior_for_prompt,
    repair_prompt,
    repo_guidance,
    reviewer_prompt,
    skill_texts,
    thread_prompt,
    verifier_prompt,
    verify_prompt,
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


class UncheckedBlockers(ReviewError):
    """A verify+match shard failed for good holding blocking candidates (v12 design §2.2): the
    run fails (INFRA) rather than publish without checking them, the confirmation pass's own
    verification included."""


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
    # policy v12 kinds (`finder`, `verify`, `thread`): the ids the answer must cover, re-told
    # in a correction turn (the verifier's candidate ids, the thread lane's hashes)
    ids: list[str] = field(default_factory=list)
    # an answer with no JSON object may go to the context-free repair lane
    repairable: bool = True


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
        unparsed = (
            exc.kind is FailKind.CONTRACT and turn.answered and bool(turn.result_text.strip())
        )
        return _Verdict(error=exc, unparsed=unparsed)
    return _validated(raw, check)


@dataclass(slots=True)
class _Confirmation:
    """The policy v12 confirmation pass a round owed (`_confirm12`): the roles it runs, whether
    it ran and what it added, or why not."""

    roles: list[str]
    ran: bool = False
    new: int = 0
    skipped: str | None = None


@dataclass(slots=True)
class _ThreadLane:
    """The policy v12 thread lane of a new-sha round, running beside the finders on its own
    thread (`_start_thread_lane`); its outcomes are read where they are first needed
    (`_thread_outcomes`) and applied when the round publishes."""

    stop: StopSignal
    thread: threading.Thread | None = None
    out: converse.ConversationOutput | None = None
    error: BaseException | None = None
    joined: bool = False


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
    merge_base: str | None = None  # of the head with the live base branch (not on an audit)
    diff_print: str | None = None  # the PR's own diff (worktree.diff_print; not on an audit)
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
    # the same prior findings in full, by the hash they are reconciled under (`prior` is what
    # the prompt sees, without most bodies)
    prior_findings: dict[str, Finding] = field(default_factory=dict)
    # identities explicitly lifted by the latest adjudication; a later head must not silently
    # turn one back into a blocker, while an explicitly marked reintroduction remains possible
    settled_hashes: set[str] = field(default_factory=set)
    prior_sha: str | None = None
    has_prior_review: bool = False
    # the head of the PR's previous final review (a same-sha re-review: this head), whether it
    # had blockers, and the PR's first reviewed head; None before any final review
    prev_final_sha: str | None = None
    prev_final_blockers: bool = False
    first_reviewed_sha: str | None = None
    code_history: late.CodeHistory | None = None
    # what this later round did with new findings on unchanged code (publish.Provenance.late)
    late_report: dict[str, Any] | None = None
    fresh_first: bool = False  # the first full review ran its second Phase-2 pass
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
    # the standing verdict a degraded same-sha re-review left in place (`_verdict_update`):
    # recorded again after its informational summary, whose canonical event is only COMMENT
    kept_verdict: str = ""
    reviewers: list[dict[str, Any]] = field(default_factory=list)
    lane_runner: LaneRunner = run_claude_lane
    dry_run: bool = False
    # `reviewsys replay` only: `replay` (its gh is pinned and refuses writes, so the dry-run
    # shortcuts that stand in for GitHub reads are not taken), run Phase 2 whatever the gate
    # decides (the decision is still recorded), and review from this merge base instead of
    # one with the base branch as of now
    replay: bool = False
    force_phase2: bool = False
    coverage_base: str | None = None
    # a replay's review context as production recorded it for the source run (`sister_repos`,
    # `base_drift` from its evidence.json): the live gathering is skipped for a replay
    replay_context: dict[str, Any] | None = None
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
    # deferrals recorded before this instant were accepted without tracking (see
    # converse.tracking_required_since); set by `main`
    deferral_cutoff: str = ""
    # a conversation run (answers replies on a reviewed commit); None until run() knows. The
    # progress estimate measures it against conversations from the start (see progress.py)
    conversation: bool | None = None
    # ---- policy v12 (`review_model_policy.pipeline`); untouched under v11 ----
    issues: dict[str, ledger.Issue] = field(default_factory=dict)  # the PR's ledger at context
    # the ledger was reconciled before this round (`ledger.reconcile`); until it is, only
    # issues with an open thread or last seen at the previous reviewed head count
    reconciled: bool = True
    reconcile_pending: set[str] = field(default_factory=set)  # left for the thread lane to judge
    # finding_hash -> thread facts for every bot finding thread, resolved ones included
    thread_index: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_reviewed_sha: str | None = None  # the newest other head the bot published a review on
    last_final_sha: str | None = None  # ... a final review on (late_nonblocking)
    finder_evidence: dict[str, Any] = field(default_factory=dict)  # evidence minus our threads
    settled: list[dict[str, Any]] = field(default_factory=list)  # the finders' settled list
    thread_work: dict[str, list[str]] = field(default_factory=dict)  # hash -> why it is judged
    thread_lane: _ThreadLane | None = None
    thread_failed: bool = False
    # a maintainer-driven concession met a blind re-find of the same issue this round
    thread_conflict: bool = False
    verify_incomplete: bool = False  # a verifier shard without blockers was dropped
    # the confirmation pass on this round; None when the policy does not ask for one here
    confirmation: _Confirmation | None = None

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

    @property
    def is_ci_check(self) -> bool:
        """A check failed after this head's APPROVE: the light CI-check run (ci.py), which
        reviews nothing and leaves the review's gate comment as it is."""
        return self.trigger == Trigger.CI_FAILED

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


def _fail_run_progress(ctx: RunContext, reason: str) -> None:
    """Close unfinished steps on either an expected failure or an unexpected crash."""
    with tx(ctx.conn):
        ctx.conn.execute(
            "UPDATE steps SET status='failed', finished_at=?, detail=? "
            "WHERE run_id=? AND status='running'",
            (now(), json.dumps({"error": reason[:1000]}), ctx.run_id),
        )
    _gate_comment(ctx, "failed", reason=reason[:200])


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
    own: a role still has one row per attempt, which is what progress and compare count.

    Written when the attempt ends: `finished_at` is now, `started_at` when its process started
    (now less the whole seconds its turns ran, never earlier than that), so the span is the
    time it held its slot, not the time it waited for one. Progress counts a lane done by its
    `finished_at` (progress.step_lanes)."""
    turns = [res, *res.followups] if res else []
    tokens_in = [t.tokens_in for t in turns if t.tokens_in is not None]
    tokens_out = [t.tokens_out for t in turns if t.tokens_out is not None]
    finished = now_dt()
    ran = sum(max(0.0, t.duration_s or 0.0) for t in turns)
    started = fmt_ts(finished - timedelta(seconds=int(ran)))
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
                started,
                fmt_ts(finished),
                str(artifact_dir),
                psha,
                reason[:500],
            ),
        )


_FINDINGS_INSERT = "INSERT INTO findings (run_id, phase, stage, hash, file, line_start, line_end, severity, confidence, category, title, body, candidate_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"
_FINDINGS_INSERT_STATUS = "INSERT INTO findings (run_id, phase, stage, hash, file, line_start, line_end, severity, confidence, category, title, body, candidate_id, code_status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"


def _insert_findings(ctx: RunContext, phase: str, stage: str, findings: list[Finding]) -> None:
    """Write finding rows (with the policy v12 candidate id and a later round's code status,
    late.py). The caller must already hold the write transaction."""
    ctx.conn.executemany(
        _FINDINGS_INSERT_STATUS,
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
                f.candidate_id,
                f.extra.get("code_status"),
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
    if ctx.dry_run or ctx.is_audit or ctx.is_ci_check:
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
    if ctx.dry_run or ctx.is_audit or ctx.is_ci_check:
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
    # a replay reviews the range production reviewed (`coverage_base`), never the live base
    base = ctx.coverage_base or (
        wt.merge_base(ctx.worktree, meta.base_ref, ctx.sha) if meta.base_ref else None
    )
    ctx.merge_base = base
    ctx.coverage_from = base or f"{ctx.sha}~1"
    # what the review of this commit reads; a conversation reviews nothing and records none
    if base and not ctx.conversation:
        ctx.diff_print = wt.diff_print(ctx.worktree, base, ctx.sha)
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
        if ctx.diff_print:
            ctx.conn.execute(
                "UPDATE heads SET diff_print=? WHERE id=?", (ctx.diff_print, ctx.head_id)
            )


# heads nobody asked a review of: only these may stand on the review of an earlier commit
CARRYABLE = frozenset({Trigger.NEW_PR, Trigger.NEW_PUSH})
CARRIED_VERDICTS = {
    "APPROVE": "approved",
    "REQUEST_CHANGES": "changes requested",
    "COMMENT": "commented",
}


def _carry(ctx: RunContext) -> bool:
    """A push that left the PR's own diff as it was (an empty commit, a merge of the base, a
    rebase onto it) changes nothing a reviewer read: the review of the head before it stands
    for this commit and no lane runs. dashpay's PR Hygiene controller applies the same rule
    (`carried_heads`) and already counts that final review for this commit. Only an automatic
    head (a person asking gets a review: ingest.enqueue_head), and only from the head right
    before it, by its diff as its own run found it: a predecessor superseded before a run
    never recorded one, so the commit after it starts over. Never past a reply under a
    finding that no run has read (`_unread_reply`): the review of this commit answers it.
    True when carried (run done)."""
    if ctx.trigger not in CARRYABLE or not ctx.diff_print:
        return False
    prev = ctx.conn.execute(
        "SELECT sha, diff_print FROM heads WHERE repo=? AND number=? AND queue='live' AND id<? "
        "ORDER BY id DESC LIMIT 1",
        (ctx.repo, ctx.number, ctx.head_id),
    ).fetchone()
    if prev is None or prev["diff_print"] != ctx.diff_print:
        return False
    source = reviewed_sha(ctx.conn, ctx.repo, ctx.number, str(prev["sha"]))
    verdict = _carryable_verdict(ctx, source)
    if verdict is None:
        return False
    with tx(ctx.conn):
        head = ctx.conn.execute(
            "SELECT trigger, status FROM heads WHERE id=?", (ctx.head_id,)
        ).fetchone()
        if head["status"] != HeadStatus.RUNNING:  # superseded meanwhile: no word for the newer
            raise Cancelled()
        if head["trigger"] not in CARRYABLE:  # someone asked for a review during the checkout
            ctx.trigger = str(head["trigger"])
            return False
        if _unread_reply(ctx):
            return False
        ctx.conn.execute("UPDATE heads SET carried_from=? WHERE id=?", (source, ctx.head_id))
        event(
            ctx.conn,
            "head.carried",
            repo=ctx.repo,
            number=ctx.number,
            run_id=ctx.run_id,
            detail=f"{ctx.sha[:8]} stands on {source[:8]} ({verdict}): diff {ctx.diff_print[:12]}",
        )
    _step_start(ctx, StepName.CARRY)
    _gate_comment(
        ctx,
        "carried",
        carried_from=source,
        reason=f"{CARRIED_VERDICTS[verdict]}; `@{ctx.cfg.bot_login} review` asks for a review "
        "of this commit",
    )
    if verdict == "APPROVE":
        _watch_ci(ctx, known=[])
    _step_end(ctx, StepName.CARRY, "ok", {"carried_from": source, "verdict": verdict})
    return True


# a reply this close before a run's context read may not have been listed to it yet
REPLY_READ_LAG = timedelta(minutes=1)


def _unread_reply(ctx: RunContext) -> bool:
    """A reply under one of the bot's findings on this PR that no run has answered since: none
    of the PR's runs that went on to finish gathered its context (`step_context`, which reads
    the threads) after it was written. A run that failed, was cancelled or was superseded
    after reading answered nothing. Rows the router has not judged yet count; those it ignored
    (not ours, untrusted, another bot) do not. Caller holds the transaction."""
    last = ctx.conn.execute(
        "SELECT MAX(s.started_at) FROM steps s JOIN runs r ON r.id=s.run_id "
        "JOIN heads h ON h.id=r.head_id WHERE h.repo=? AND h.number=? AND s.name=? "
        "AND r.status=?",
        (ctx.repo, ctx.number, StepName.CONTEXT.value, RunStatus.DONE.value),
    ).fetchone()[0]
    since = fmt_ts(parse_ts(str(last)) - REPLY_READ_LAG) if last else ""
    return (
        ctx.conn.execute(
            "SELECT 1 FROM inbox WHERE kind='review_reply' AND repo=? AND number=? "
            "AND occurred_at>=? AND (action IS NULL OR action NOT LIKE 'ignored%') LIMIT 1",
            (ctx.repo, ctx.number, since),
        ).fetchone()
        is not None
    )


def _carryable_verdict(ctx: RunContext, sha: str) -> str | None:
    """The verdict a carry may pass on: the newest review of `sha`, when it is a final review
    (a preliminary one is no report to the hygiene controller, and its deferred Phase 2 is
    owed) that ran on the primary models and that no CI update has weakened since (a rebase is
    exactly the push that may win either back). None otherwise."""
    row = ctx.conn.execute(
        "SELECT rv.event, rv.phase, (SELECT MAX(r.degraded) FROM reviews o JOIN runs r "
        "ON r.id=o.run_id WHERE o.repo=rv.repo AND o.number=rv.number AND o.sha=rv.sha) degraded "
        "FROM reviews rv WHERE rv.repo=? AND rv.number=? AND rv.sha=? "
        "ORDER BY rv.posted_at DESC, rv.id DESC LIMIT 1",
        (ctx.repo, ctx.number, sha),
    ).fetchone()
    if row is None or row["phase"] != "final" or row["degraded"]:
        return None
    if kv_get(ctx.conn, ci.updated_key(ctx.repo, ctx.number, sha)):
        return None
    verdict = publish.canonical_event(str(row["event"]))
    return verdict if verdict in CARRIED_VERDICTS else None


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
    _fetch_files(ctx)
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


def _fetch_files(ctx: RunContext) -> None:
    files_raw = (
        ctx.gh.api(f"repos/{ctx.repo}/pulls/{ctx.number}/files?per_page=100", paginate=True) or []
    )
    ctx.files = [f for f in files_raw if isinstance(f, dict)]


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


def _latest_settlement(ctx: RunContext, finding_hash: str) -> str | None:
    """Return the latest explicit lift for a finding, unless a later full review re-kept it.

    Full reviews write verified/posted rows; conversation outcomes write conceded rows. Ordering
    by run id makes a later independently evidenced reintroduction active again, while a later
    withdrawal remains a tombstone for every subsequent head.
    """
    row = ctx.conn.execute(
        "SELECT f.stage, f.body FROM findings f JOIN runs r ON r.id=f.run_id "
        "JOIN heads h ON h.id=r.head_id WHERE h.repo=? AND h.number=? AND f.hash=? "
        "AND f.stage IN ('conceded','verified','verified-fresh','posted') "
        "ORDER BY r.id DESC, f.id DESC LIMIT 1",
        (ctx.repo, ctx.number, finding_hash),
    ).fetchone()
    if not row or row["stage"] != "conceded":
        return None
    status = str(row["body"] or "").split("\n", 1)[0]
    # judgement lifts only: FIXED/OUTDATED rest on the code, which a later head may revert
    return status if status in converse.LIFTING_STATUSES else None


def _prior_extra(ctx: RunContext, h: str, thread: dict[str, Any] | None) -> dict[str, Any]:
    """A prior finding's `extra`: its thread's replies, and the settled status if any."""
    extra: dict[str, Any] = {"thread_replies": thread["transcript"]} if thread else {}
    settled = _latest_settlement(ctx, h)
    if settled:
        extra["prior_status"] = settled
        ctx.settled_hashes.add(h)
    return extra


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
    # Any earlier published review means this head is an iterative pass; the previous
    # full-strength final review decides which code counts as already reviewed and whether
    # the fresh final gate is due (`_review_history`, `_verify2`).
    local_prior = ctx.conn.execute(
        "SELECT 1 FROM reviews WHERE repo=? AND number=? LIMIT 1", (ctx.repo, ctx.number)
    ).fetchone()
    ours = [
        r
        for r in github.reviews(ctx.gh, ctx.repo, ctx.number)
        if (r.get("user") or {}).get("login", "").lower() == ctx.cfg.bot_login.lower()
        and github.REVIEW_MARKER in str(r.get("body") or "")
    ]
    ctx.has_prior_review = bool(local_prior or ours)
    _review_history(ctx, ours)
    # prior findings: last posted set for this PR from our DB, plus any human replies on
    # their inline threads (the reviewer must engage with pushback, not re-raise past it)
    ctx.open_threads = github.finding_threads(threads, ctx.cfg.bot_login)
    _ledger_write(
        ctx,
        "threads",
        lambda c: ledger.sync_threads(
            c,
            ctx.repo,
            ctx.number,
            run_id=ctx.run_id,
            sha=ctx.sha,
            threads=threads,
            bot_login=ctx.cfg.bot_login,
        ),
    )
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
                extra=_prior_extra(ctx, str(r["hash"]), thread),
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
                extra=_prior_extra(ctx, h, t),
            )
        )
    ctx.prior = prior_for_prompt(prior, ctx.prior_sha or "")
    ctx.prior_findings = {f.prior_hash or f.hash: f for f in prior}
    ctx.prior_threads = {h: t for h, t in replied.items() if h in seen}
    # the conversation lane is given none of it. A replay never gathers it (the sister
    # checkouts' default tips and the base drift are live state, and sister checkouts write the
    # mirrors); it gets what production recorded for the source run instead
    if not ctx.conversation and not ctx.replay:
        _review_context(ctx)
    elif not ctx.conversation and ctx.replay_context:
        _replayed_context(ctx, ctx.replay_context)
    (ctx.run_dir / "evidence.json").write_text(
        json.dumps(
            {"evidence": ctx.evidence, "coderabbit": ctx.coderabbit, "prior": ctx.prior}, indent=1
        )
    )
    pol = ctx.cfg.policy
    if pol.unanchored and not ctx.is_audit:  # (the v12 verifier needs unanchored finders)
        _context12(ctx, threads)


_REVIEW_PHASE_RE = re.compile(r"thepastaclaw-review-phase v1 phase=(\w+) sha=([0-9a-f]{7,40})")


def _pr_reviews(ctx: RunContext, before_run: int | None = None) -> list[sqlite3.Row]:
    """The reviews of this PR on our record, oldest first (before run `before_run`, when
    given), less a conversation's verdict follow-up and a CI update: neither reviewed code.
    `full`: a final review at full strength (Phase 2 ran, not degraded)."""
    return ctx.conn.execute(
        "SELECT rv.sha, rv.phase, rv.event, r.blocker_count, rv.phase='final' AND "
        "COALESCE(r.degraded, 1)=0 AND EXISTS (SELECT 1 FROM steps p WHERE p.run_id=rv.run_id "
        "AND p.name=? AND p.status='ok') AS full FROM reviews rv "
        "LEFT JOIN runs r ON r.id=rv.run_id WHERE rv.repo=? AND rv.number=? "
        "AND (? IS NULL OR rv.run_id<?) "
        "AND NOT EXISTS (SELECT 1 FROM steps s WHERE s.run_id=rv.run_id AND s.name IN (?, ?)) "
        "ORDER BY rv.posted_at, rv.id",
        (
            StepName.PHASE2.value,
            ctx.repo,
            ctx.number,
            before_run,
            before_run,
            StepName.CONVERSE.value,
            StepName.CI_CHECK.value,
        ),
    ).fetchall()


def _had_blockers(review: sqlite3.Row) -> bool:
    return bool(review["blocker_count"]) or (
        publish.canonical_event(str(review["event"])) == "REQUEST_CHANGES"
    )


def _review_history(ctx: RunContext, ours: list[dict[str, Any]]) -> None:
    """The PR's previous full-strength final review on our record (its head, whether it had
    blockers) and its first reviewed head (from the bot's review markers on GitHub, `ours`,
    when our record has none). A final review that is not on our record, or did not run at
    full strength, is no basis for treating code as already reviewed."""
    rows = _pr_reviews(ctx)
    full = [r for r in rows if r["full"]]
    if full:
        ctx.prev_final_sha = str(full[-1]["sha"])
        ctx.prev_final_blockers = _had_blockers(full[-1])
    marked = (_REVIEW_PHASE_RE.search(str(r.get("body") or "")) for r in ours)
    reviewed = [str(r["sha"]) for r in rows] or [m.group(2) for m in marked if m]
    ctx.first_reviewed_sha = reviewed[0] if reviewed else None


# ---- policy v12: what a round knows from the ledger ----


def _lines_changed(
    ctx: RunContext, from_sha: str, to_sha: str, file: str, ls: int | None, le: int | None
) -> bool | None:
    """`ledger.lines_changed` in this run's worktree (its mirror holds every reviewed head);
    None when git cannot tell."""
    if ctx.worktree is None or not file:
        return None
    return ledger.lines_changed(ctx.worktree, from_sha, to_sha, file, ls, le)


def _issue_moved(ctx: RunContext, i: ledger.Issue) -> bool | None:
    """`ledger.anchor_moved` in this run's worktree."""
    return ledger.anchor_moved(
        i, ctx.sha, lambda a, b, file, ls, le: _lines_changed(ctx, a, b, file, ls, le)
    )


def _map_lines(
    ctx: RunContext, from_sha: str, to_sha: str, file: str, ls: int | None, le: int | None
) -> tuple[int | None, int | None] | None:
    """`ledger.map_lines` in this run's worktree; None when git cannot tell."""
    if ctx.worktree is None or not file:
        return None
    return ledger.map_lines(ctx.worktree, from_sha, to_sha, file, ls, le)


def _closed_changed(ctx: RunContext, i: ledger.Issue) -> bool | None:
    """Whether the code at a settled issue changed since it was closed (the revival rule):
    its anchored lines, re-located at the commit it was closed at (`_map_lines`: the anchor is
    numbered in `anchor_sha`, and a typical OUTDATED issue's lines were rewritten before it
    closed), changed between that commit and this head. Not some other edit in the file. None
    when git cannot tell, which never revives anything."""
    since = i.closed_sha or i.last_seen_sha
    anchor = i.anchor_sha or i.opened_sha
    if not since or not anchor:
        return None
    if since == ctx.sha:
        return False
    lines = _map_lines(ctx, anchor, since, i.file, i.line_start, i.line_end)
    if lines is None:
        return None
    return _lines_changed(ctx, since, ctx.sha, i.file, *lines)


def _issue_url(ctx: RunContext, i: ledger.Issue) -> str:
    return rules.thread_url(ctx.repo, ctx.number, i)


def _last_human(t: dict[str, Any]) -> str:
    replies = [c for c in t.get("transcript") or [] if not c.get("is_bot")]
    return converse.strip_marker(str(replies[-1].get("body") or ""))[:600] if replies else ""


def _context12(ctx: RunContext, threads: list[dict[str, Any]]) -> None:
    """Policy v12 additions to the context step: every bot finding thread (resolved ones too),
    the previous reviewed heads, the ledger (reconciled once per PR first), the finders'
    evidence without our own threads and their settled list, and the thread lane's work set."""
    pol, bot = ctx.cfg.policy, ctx.cfg.bot_login
    ctx.thread_index = github.finding_threads([{**t, "is_resolved": False} for t in threads], bot)
    rows = ctx.conn.execute(
        "SELECT sha, phase FROM reviews WHERE repo=? AND number=? AND sha!=? "
        "ORDER BY posted_at DESC, rowid DESC",
        (ctx.repo, ctx.number, ctx.sha),
    ).fetchall()
    ctx.last_reviewed_sha = str(rows[0]["sha"]) if rows else None
    ctx.last_final_sha = next((str(r["sha"]) for r in rows if r["phase"] == "final"), None)
    if pol.verifier_v12:
        _reconcile12(ctx)
    ctx.issues = ledger.load(ctx.conn, ctx.repo, ctx.number)
    if pol.unanchored:
        # open issues belong to the thread lane: finders see neither our findings nor the
        # discussion under them, only the settled list
        ctx.finder_evidence = {
            **ctx.evidence,
            "review_threads": [
                t
                for t in ctx.evidence.get("review_threads") or []
                if not github._is_bot((t.get("comments") or [{}])[0], bot)
            ],
        }
        reasons = {h: _last_human(t) for h, t in ctx.thread_index.items()}
        ctx.settled = rules.settled_list(ctx.issues, reasons)
    if pol.thread_lane and not ctx.conversation:
        ctx.thread_work = _thread_work(ctx)
    (ctx.run_dir / "ledger.json").write_text(
        json.dumps(
            {
                "reconciled": ctx.reconciled,
                "pending": sorted(ctx.reconcile_pending),
                "last_reviewed_sha": ctx.last_reviewed_sha,
                "issues": {h: dataclasses.asdict(i) for h, i in ctx.issues.items()},
                "settled": ctx.settled,
                "thread_work": ctx.thread_work,
            },
            indent=1,
        )
    )


def _reconcile12(ctx: RunContext) -> None:
    """The PR's one-shot ledger reconcile (`ledger.reconcile`) before its first v12 round. Git
    is asked before the write transaction, never inside it."""
    rec = ledger.reconciled(ctx.conn, ctx.repo, ctx.number)
    if rec is None and not ctx.dry_run:
        moved = ledger.moved_anchors(
            ledger.load(ctx.conn, ctx.repo, ctx.number),
            ctx.sha,
            lambda a, b, file, ls, le: _lines_changed(ctx, a, b, file, ls, le),
            open_threads=ctx.open_threads,
        )
        ledger.safe_write(
            ctx.conn,
            "reconcile",
            lambda c: ledger.reconcile(
                c,
                ctx.repo,
                ctx.number,
                pr_open=True,
                open_threads=set(ctx.open_threads),
                moved=moved,
                run_id=ctx.run_id,
                sha=ctx.sha,
            ),
            repo=ctx.repo,
            number=ctx.number,
            run_id=ctx.run_id,
        )
        rec = ledger.reconciled(ctx.conn, ctx.repo, ctx.number)
    ctx.reconciled = rec is not None
    ctx.reconcile_pending = set((rec or {}).get("pending") or [])


def _thread_work(ctx: RunContext) -> dict[str, list[str]]:
    """The thread lane's work set on a new-sha round (v12 design §2.4), computed by code: open
    threads with a human reply newer than our last answer (and not already met with silence),
    standing issues the reconcile left to judge, and standing issues whose code changed since
    they were last judged. Every other open issue stays open with no model call."""
    work: dict[str, list[str]] = {}
    for h in _threads_to_answer(ctx):
        work.setdefault(h, []).append("reply")
    for h, i in ctx.issues.items():
        if i.status not in rules.VERDICT_STANDING:
            continue
        if h in ctx.reconcile_pending:
            work.setdefault(h, []).append("reconcile")
            continue
        moved = _issue_moved(ctx, i)
        # an issue git cannot place (no file, line or anchor commit) waits for a reply:
        # "unknown" would otherwise send it again on every push
        anchored = bool(i.file and i.line_start and (i.anchor_sha or i.opened_sha))
        if moved is True or (moved is None and anchored):
            work.setdefault(h, []).append("code_changed")
    return work


# ---- policy v12: the thread lane (converse.py grown) ----


def _code_at(
    worktree: Path | None, file: str, ls: int | None, le: int | None, around: int = 20
) -> str:
    """The issue's file at the reviewed head around its anchor, numbered."""
    if worktree is None or not file:
        return ""
    path = (worktree / file).resolve()
    if not path.is_relative_to(worktree.resolve()) or not path.is_file():
        return "(the file does not exist at this head)"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    lo = max(1, (ls or 1) - around)
    hi = min(len(lines), (le or ls or 1) + around)
    return "\n".join(f"{n}: {lines[n - 1]}" for n in range(lo, hi + 1))[:8000]


def _thread_item(ctx: RunContext, h: str, why: list[str], same_sha: bool) -> dict[str, Any]:
    """One issue as the thread lane sees it: the finding, its whole thread, the code at head
    around its anchor and the diff of its file since it was last judged."""
    i = ctx.issues.get(h)
    t = ctx.open_threads.get(h) or ctx.thread_index.get(h) or {}
    file = i.file if i else str(t.get("path") or "")
    ls = i.line_start if i else t.get("line")
    le = i.line_end if i else t.get("line")
    item: dict[str, Any] = {
        "finding_hash": h,
        "why": why,
        "status": i.status if i else "open",
        "severity": i.severity if i else t.get("severity"),
        "title": i.title if i else t.get("title"),
        "file": file,
        "lines": [ls, le],
        "anchored_at": ((i.anchor_sha or "") if i else "")[:12],
        "finding_body": converse.strip_marker(i.body if i and i.body else str(t.get("body") or "")),
        "thread": "open"
        if h in ctx.open_threads
        else ("resolved by a human" if t else "none (no inline comment to reply on)"),
        "exchange": converse.exchange(t.get("transcript") or []),
        "awaiting_your_answer": bool(t.get("awaiting_answer")) and h in ctx.open_threads,
        "code_at_head": _code_at(ctx.worktree, file, ls, le),
    }
    since = (i.last_seen_sha or i.anchor_sha or i.opened_sha) if i else None
    if not same_sha and since and since != ctx.sha and ctx.worktree is not None and file:
        item["last_judged_at"] = since[:12]
        item["diff_since_last_judged"] = (
            ledger.git_diff(ctx.worktree, since, ctx.sha, file)
            or "(no change to this file, or git could not compare the commits)"
        )
    return item


def _run_thread_lane(
    ctx: RunContext,
    work: dict[str, list[str]],
    *,
    same_sha: bool,
    standing_verdict: str | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[converse.ConversationOutput, LaneModel]:
    """The thread lane over `work` (hash -> why), in at most three shards by issue count and
    input size (`rules.thread_shards`), side by side: their outcomes after the code's own
    rules (`converse.parse_threads`: every hash exactly once, FIXED / OUTDATED where nothing
    changed is FIX_PENDING; `converse.accept_deferrals`). A shard that fails for good leaves
    only its own issues as they were (`threads.failed`) and the round cannot approve
    (`ctx.thread_failed`); when every shard fails the error is raised. `same_sha`: a reply
    to the reviewed commit."""
    assert ctx.meta
    pol = ctx.cfg.policy
    hashes = list(work)
    facts = {h: ctx.open_threads.get(h) or ctx.thread_index.get(h) or {} for h in hashes}
    fetched, unfetched = _fetch_linked_commits(
        ctx, {h: t for h, t in facts.items() if t}, should_stop=should_stop
    )
    linked = "\n".join(
        [
            *(
                f"- `{c['sha']}` from {c['owner']}/{c['repo']}: available in this checkout "
                f"(`git show {c['sha']}`)"
                for c in fetched
            ),
            *(
                f"- `{c['sha']}` from {c['owner']}/{c['repo']}: could NOT be fetched "
                f"({c.get('reason') or 'unavailable'})"
                for c in unfetched
            ),
        ]
    )
    fix_pending = (
        set(hashes)
        if same_sha
        else {
            h for h in hashes if (i := ctx.issues.get(h)) is not None and i.last_seen_sha == ctx.sha
        }
    )
    items = {h: _thread_item(ctx, h, work[h], same_sha) for h in hashes}
    shards = rules.thread_shards({h: len(json.dumps(items[h])) for h in hashes})
    lm = ctx.lane_model(pol.conversation_lane)
    base = "conversation" if same_sha else "thread"
    roles = (
        [base]
        if len(shards) == 1
        else [f"{base}-{n}of{len(shards)}" for n in range(1, len(shards) + 1)]
    )

    def one(n: int) -> converse.ConversationOutput:
        batch = shards[n]
        prompt = thread_prompt(
            ctx.cfg,
            repo=ctx.repo,
            number=ctx.number,
            head_sha=ctx.sha,
            meta=meta,
            issues=[items[h] for h in batch],
            linked_commits=linked,
            evidence=ctx.evidence,
            standing_verdict=standing_verdict,
            same_sha=same_sha,
            structured=pol.structured_output,
        )
        try:
            return _run_checked_lane(
                ctx,
                phase="converse" if same_sha else "threads",
                role=roles[n],
                lm=lm,
                prompt=prompt,
                is_verifier=True,
                check=OutputCheck(
                    validate=lambda raw: converse.parse_threads(
                        raw, hashes=batch, fix_pending=fix_pending
                    ),
                    kind="thread",
                    ids=batch,
                ),
                should_stop=should_stop,
                schema=schemas.thread_schema(batch),
            )
        finally:
            if len(shards) > 1:
                ctx.close_lane_conn()

    meta = ctx.meta.as_dict()
    results: list[converse.ConversationOutput | BaseException] = []
    if len(shards) == 1:
        results = [one(0)]
    else:
        with ThreadPoolExecutor(
            max_workers=len(shards), thread_name_prefix=f"{base}-shard"
        ) as pool:
            futures = [pool.submit(one, n) for n in range(len(shards))]
            for fut in futures:
                try:
                    results.append(fut.result())
                except BaseException as exc:  # sorted out below, after every shard ended
                    results.append(exc)
    errors = [r for r in results if isinstance(r, BaseException)]
    fatal = next((e for e in errors if not isinstance(e, ReviewError)), None)
    if fatal is not None:
        raise fatal  # Cancelled, a stopped lane, a crash: not a shard's contract
    if errors and len(errors) == len(results):
        raise errors[0]
    outcomes: dict[str, converse.ThreadOutcome] = {}
    for n, r in enumerate(results):
        if isinstance(r, BaseException):
            ctx.thread_failed = True
            _lane_event(
                ctx,
                "threads.failed",
                f"{roles[n]} ({len(shards[n])} issue(s) left as they were): "
                + _failure_text(r)[:800],
            )
        else:
            outcomes.update(r.outcomes)
    out = _accept_thread_deferrals(ctx, outcomes, facts)
    (ctx.run_dir / ("conversation.json" if same_sha else "threads.json")).write_text(
        json.dumps(
            {
                "work": work,
                "shards": shards,
                "outcomes": {h: dataclasses.asdict(o) for h, o in out.outcomes.items()},
            },
            indent=1,
        )
    )
    return out, lm


def _start_thread_lane(ctx: RunContext) -> None:
    """Start the thread lane beside the finders (v12 design §2.4: it adds no latency): only
    on a policy with it, never on an audit, and only when its work set is non-empty."""
    if not ctx.cfg.policy.thread_lane or ctx.is_audit or not ctx.thread_work:
        return
    tl = _ThreadLane(stop=StopSignal())

    def work() -> None:
        try:
            tl.out, _ = _run_thread_lane(
                ctx,
                ctx.thread_work,
                same_sha=False,
                should_stop=lambda: ctx.cancel_flag.is_set() or tl.stop.is_set(),
            )
        except BaseException as exc:  # handed to whoever reads the outcomes
            tl.error = exc
        finally:
            ctx.close_lane_conn()

    tl.thread = threading.Thread(target=work, name="thread-lane", daemon=True)
    ctx.thread_lane = tl
    tl.thread.start()


THREAD_LANE_STOP_SECONDS = 120  # a stopped lane's process is killed well within this


def _stop_thread_lane(ctx: RunContext) -> None:
    """Tell the thread lane to stop (its lanes and fetches poll for it) and wait for it,
    bounded: a lane that does not end in time is left to its daemon thread and logged."""
    tl = ctx.thread_lane
    if tl is not None and tl.thread is not None:
        tl.stop.set("the run ended")
        tl.thread.join(timeout=THREAD_LANE_STOP_SECONDS)
        if tl.thread.is_alive():
            log.warning(
                "thread lane still running %ss after the run ended", THREAD_LANE_STOP_SECONDS
            )


def _thread_outcomes(ctx: RunContext) -> dict[str, converse.ThreadOutcome]:
    """The thread lane's outcomes, waiting for it if it is still running. A thread lane that
    failed for good does not fail the review: every issue it was to judge stays as it was
    (open issues keep counting), recorded as `threads.failed`, and is judged again next round.
    A cancelled run is cancelled."""
    tl = ctx.thread_lane
    if tl is None:
        return {}
    if not tl.joined:
        if tl.thread is not None:
            tl.thread.join()
        tl.joined = True
        exc = tl.error
        if isinstance(exc, Cancelled) or (exc is not None and not isinstance(exc, Exception)):
            raise exc
        if exc is not None:
            ctx.thread_failed = True
            log.warning("thread lane failed: %s", exc)
            _lane_event(ctx, "threads.failed", _failure_text(exc)[:1000])
    return dict(tl.out.outcomes) if tl.out is not None else {}


def _accept_thread_deferrals(
    ctx: RunContext,
    outcomes: dict[str, converse.ThreadOutcome],
    facts: dict[str, dict[str, Any]],
) -> converse.ConversationOutput:
    """`converse.accept_deferrals` for the thread lane: a deferral needs a maintainer's call
    and, above a nitpick, a tracking issue or PR visible on the PR (the one its reply cites,
    or one the maintainer points to). The lane cannot have one filed, so an untracked
    deferral is asked for a link. An issue the ledger already holds deferred is settled: a
    restatement is not a new deferral to vet."""
    pr_text = _pr_text(ctx)
    threads = {
        h: {
            "transcript": facts[h].get("transcript") or [],
            "severity": i.severity if (i := ctx.issues.get(h)) else facts[h].get("severity"),
        }
        for h in outcomes
    }

    def cite(h: str, o: converse.ThreadOutcome) -> converse.ThreadOutcome:
        if o.status != "INTENTIONALLY_DEFERRED" or o.tracking_url:
            return o
        t = threads[h]
        link = converse.cited_reference(
            o.reply,
            said=converse.thread_text(t, humans_only=True),
            visible=converse.thread_text(t) + "\n" + pr_text,
            repo=ctx.repo,
            number=ctx.number,
        )
        return dataclasses.replace(o, tracking_url=link)

    return converse.accept_deferrals(
        converse.ConversationOutput(outcomes={h: cite(h, o) for h, o in outcomes.items()}),
        threads,
        settled=_ledger_deferred(ctx, outcomes),
        repo=ctx.repo,
        number=ctx.number,
        pr_text=pr_text,
        can_file=False,
    )


def _ledger_deferred(ctx: RunContext, hashes: Iterable[str]) -> set[str]:
    """Those of `hashes` the ledger already holds deferred: a restated deferral there is not
    a new one to vet, and its thread keeps the status line (and tracking link) it shows."""
    return {h for h in hashes if (i := ctx.issues.get(h)) is not None and i.status == "deferred"}


def _pending_statuses(
    outs: dict[str, converse.ThreadOutcome], corroborated: set[str]
) -> dict[str, str]:
    """hash -> the ledger status this round's thread outcomes leave, a FIXED / OUTDATED of an
    issue a finder just restated suppressed (it stays open: v12 design §2.3)."""
    out: dict[str, str] = {}
    for h, o in outs.items():
        if o.status in {"FIXED", "OUTDATED"} and h in corroborated:
            continue
        st = ledger.OUTCOME_STATUS.get(o.status)
        if st is not None:
            out[h] = st
    return out


def _projected_lifted(outs: dict[str, converse.ThreadOutcome], corroborated: set[str]) -> set[str]:
    """The hashes this round's thread outcomes would close were every one delivered: what a
    replay, which delivers nothing, decides on in place of `_record_thread_answers`' set."""
    return {h for h, st in _pending_statuses(outs, corroborated).items() if st in ledger.CLOSED}


def _review_context(ctx: RunContext) -> None:
    """Evidence from beyond the PR's own range: the sister repos it builds against, checked
    out for the lanes (`sister_repos`, sister.py), and how the base branch moved since it
    branched (`base_drift`, drift.py; not on an audit, whose PR is in the base by now). Best
    effort: what cannot be had is recorded in the evidence and an event, never a failure."""
    assert ctx.meta and ctx.worktree
    try:
        pins = sister.lock_pins(ctx.cfg, ctx.repo, ctx.worktree)
    except Exception as exc:  # evidence only; the configured sister repos still come
        pins = {}
        _context_event(ctx, "context.sister_failed", f"Cargo.lock: {type(exc).__name__}: {exc}")
    try:
        repos = sister.gather(
            ctx.cfg,
            ctx.conn,
            repo=ctx.repo,
            run_id=ctx.run_id,
            pins=pins,
            should_stop=ctx.cancel_flag.is_set,
        )
    except Exception as exc:  # it records its own failures; this is a bug's safety net
        repos = []
        _context_event(ctx, "context.sister_failed", f"{type(exc).__name__}: {exc}")
    ctx.check_cancel()
    for r in repos:
        if r.get("error"):
            _context_event(ctx, "context.sister_failed", f"{r['repo']}: {r['error']}")
    if repos:
        ctx.evidence["sister_repos"] = repos
    if ctx.is_audit or not ctx.merge_base or not ctx.meta.base_ref:
        return
    try:
        bd = drift.base_drift(ctx.worktree, ctx.meta.base_ref, ctx.merge_base, ctx.sha)
    except Exception as exc:  # evidence only, never the run's problem
        bd = {"base_branch": ctx.meta.base_ref, "error": f"{type(exc).__name__}: {exc}"[:300]}
    ctx.evidence["base_drift"] = bd
    if bd.get("error"):
        _context_event(ctx, "context.base_drift_failed", bd["error"])


def _replayed_context(ctx: RunContext, recorded: dict[str, Any]) -> None:
    """The source run's `sister_repos` and `base_drift`, so a replay's lanes see the evidence
    the production lanes saw. A checkout slot that is gone since gets a null path (the lanes'
    "not checked out"); one still there is touched, as live reuse does (gc goes by mtime; the
    replay also holds a lease on them, `sister.lease`)."""
    repos = []
    for r in recorded.get("sister_repos") or []:
        if not isinstance(r, dict):
            continue
        r = dict(r)
        if r.get("also_pinned"):
            r["also_pinned"] = [dict(s) for s in r["also_pinned"] if isinstance(s, dict)]
        for key in ("pinned", "default"):
            if isinstance(r.get(key), dict):
                r[key] = dict(r[key])
        gone = False
        for slot in (*sister._pin_slots(r), r.get("default")):
            if not isinstance(slot, dict) or not slot.get("path"):
                continue
            p = Path(str(slot["path"]))
            if p.is_dir():
                with contextlib.suppress(OSError):
                    os.utime(p)
            else:
                slot["path"] = None
                gone = True
        if gone:
            r["skipped"] = "a checkout production used is no longer present for the replay"
        repos.append(r)
    if repos:
        ctx.evidence["sister_repos"] = repos
    bd = recorded.get("base_drift")
    if isinstance(bd, dict) and not ctx.is_audit:
        ctx.evidence["base_drift"] = bd


def _context_event(ctx: RunContext, kind: str, detail: str) -> None:
    with tx(ctx.conn):
        event(ctx.conn, kind, repo=ctx.repo, number=ctx.number, run_id=ctx.run_id, detail=detail)


def _run_lane(
    ctx: RunContext,
    *,
    phase: str,
    role: str,
    lm: LaneModel,
    prompt: str,
    is_verifier: bool,
    schema: dict[str, Any] | None = None,
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
        schema=schema,
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
    schema: dict[str, Any] | None = None,
) -> T:
    """Run a lane with bounded retries until its answer passes `check`. Returns the output.

    `schema` (schemas.py) is the lane kind's answer schema. Under policy v12 with
    `pipeline.structured_output` the lane answers through it (`--json-schema`), its
    correction turns too, and its envelope's `structured_output` is read before its text; a
    lane whose structured-output retries ran out failed its contract. Otherwise it is
    ignored. The schema enforces shape only: the answer still goes through `check`, and
    the correction turns below remain the fallback for everything else.

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
    json_schema = schema if ctx.cfg.policy.structured_output else None
    last = cause = ""
    off_schema = False  # the last failure was Claude Code giving up on the schema
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
                stamp_echo=ctx.cfg.policy.stamps_echo_fields,
                ids=check.ids,
                structured=json_schema is not None,
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
            extra_dirs=sister.lane_dirs(ctx.evidence),
            timeout_seconds=ctx.cfg.lane_timeout_minutes * 60,
            claude_bin=ctx.cfg.claude_bin,
            max_budget_usd=ctx.cfg.lane_budget_usd,
            should_stop=stop,
            parallel=not is_verifier and not comparison,
            pool=pool,
            session_id=session,
            check=ask if session else None,
            corrections=corrections,
            json_schema=json_schema,
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
        basis = next((t for t in reversed(turns) if t.answered), res)
        failed_turn = turns[-1] if res.followups and not turns[-1].answered else None
        verdict = _evaluate(basis, check)
        if verdict.error is not None and stop():
            # told to stop between turns (the phase was abandoned, the run cancelled): not a
            # failed attempt, so never a retry or a fall down the ladder for a stopped lane
            raise stopped(True)
        status = "completed" if basis is res else "corrected"  # when it passes
        if verdict.error is not None and verdict.unparsed and check.repairable:
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
        off_schema = isinstance(exc, SchemaRetriesExhausted)
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
        FailKind.INFRA
        if not off_schema and ("timed out" in last or "exit" in last)
        else FailKind.CONTRACT,
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
            prior_hashes=check.ids
            if check.kind in V12_KINDS
            else [str(p["finding_hash"]) for p in check.prior],
            stamp_echo=ctx.cfg.policy.stamps_echo_fields,
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
    schema: dict[str, Any] | None = None,
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
                schema=schema,
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


def _blind_label(key: str, *, seed: int) -> str:
    """A Phase-2 lane's output key under its neutral comparison label: `<role>/a` or
    `<role>/b`, which one is the comparison model decided per run (`seed`)."""
    second_is_a = random.Random(seed).random() < 0.5
    base = key.removesuffix(COMPARE_SUFFIX)
    return f"{base}/{'a' if key.endswith(COMPARE_SUFFIX) == second_is_a else 'b'}"


def _blinded_key(ctx: RunContext, key: str) -> str:
    """A Phase-2 lane's output key as `_blind` shows it to the verifier on a comparison run,
    for its candidate ids; unchanged otherwise."""
    return _blind_label(key, seed=ctx.run_id) if ctx.compare_model else key


def _lane_title(ctx: RunContext, role: str) -> str:
    if role == "general":
        return "general review"
    spec = next((s for s in ctx.cfg.specialists if s.id == role), None)
    return f"{role} ({spec.description})" if spec and spec.description else role


def _phase_roles(ctx: RunContext, phase: str) -> list[str]:
    """The reviewer roles of a phase: `general` plus the selected specialists. Phase 1 runs
    only the specialists the policy lists for it (`phase1_specialists`), when it lists any."""
    allowed = ctx.cfg.policy.phase1_specialists
    picked = ctx.selection
    if phase == "phase1" and allowed is not None and ctx.phase2_follows:
        picked = [s for s in picked if s in allowed]
    return ["general", *picked]


def _lane_prompts(
    ctx: RunContext,
    roles: list[str],
    *,
    meta: dict[str, Any],
    expected_phase: str,
    guidance: str,
    prior: list[dict[str, Any]],
    prior_sha: str | None,
    fresh: bool,
) -> dict[str, str]:
    """Each role's prompt: a policy v12 finder (`finder_prompt`), else the v11 reviewer."""
    pol = ctx.cfg.policy
    if pol.unanchored:
        return {
            role: finder_prompt(
                ctx.cfg,
                repo=ctx.repo,
                number=ctx.number,
                head_sha=ctx.sha,
                role=role,
                lane_title=_lane_title(ctx, role),
                meta=meta,
                coverage_from=ctx.coverage_from,
                evidence=ctx.finder_evidence or ctx.evidence,
                guidance=guidance,
                last_reviewed_sha=ctx.last_reviewed_sha,
                settled=ctx.settled,
                structured=pol.structured_output,
            )
            for role in roles
        }
    return {
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
            prior=prior,
            prior_sha=prior_sha,
            fresh=fresh,
            stamp_echo=pol.stamps_echo_fields,
            guidance=guidance,
        )
        for role in roles
    }


def _reviewer_lanes(
    ctx: RunContext,
    *,
    phase: str,
    lm: LaneModel,
    expected_phase: str,
    fresh: bool = False,
    abort: StopSignal | None = None,
    roles: list[str] | None = None,
) -> dict[str, ReviewerOutput]:
    """The general reviewer and the phase's specialists (`_phase_roles`; `roles` instead, for
    the policy v12 confirmation pass), side by side (at
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
    roles = roles if roles is not None else _phase_roles(ctx, phase)
    review_prior = [] if fresh else ctx.prior
    review_prior_sha = None if fresh else ctx.prior_sha
    prior_hashes = {str(p["finding_hash"]) for p in review_prior}
    stamp = ctx.cfg.policy.stamps_echo_fields
    guidance = repo_guidance(ctx.worktree) if ctx.cfg.policy.isolates_lane_env else ""
    # policy v12: finder lanes on the candidate contract (`rules.parse_finder_output`)
    unanchored = ctx.cfg.policy.unanchored
    prompts = _lane_prompts(
        ctx,
        roles,
        meta=meta,
        expected_phase=expected_phase,
        guidance=guidance,
        prior=review_prior,
        prior_sha=review_prior_sha,
        fresh=fresh,
    )
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
                stamp_echo=stamp,
                settled_hashes=ctx.settled_hashes,
            )

        def validate_candidates(raw: dict[str, Any]) -> ReviewerOutput:
            label = _blinded_key(ctx, key) if phase == "phase2" and not fresh else key
            return rules.parse_finder_output(
                raw,
                id_prefix=rules.candidate_prefix(phase, label, fresh=fresh),
                source=f"{phase}:{'fresh:' if fresh else ''}{key}",
                expected_phase=expected_phase,
                head_sha=ctx.sha,
            )

        check = (
            OutputCheck(validate=validate_candidates, kind="finder", expected_phase=expected_phase)
            if unanchored
            else OutputCheck(
                validate=validate,
                kind="reviewer",
                expected_phase=expected_phase,
                prior=review_prior,
            )
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
                schema=schemas.FINDER_SCHEMA if unanchored else None,
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
    if ctx.cfg.policy.verifier_v12:
        return _verify12(
            ctx,
            phase=phase,
            lm=lm,
            expected_phase=expected_phase,
            candidates=_phase_candidates(ctx, phase),
            note=_phase_note(ctx, phase),
        )
    all_phase2 = phase2_outputs if phase2_outputs is not None else ctx.phase2_outputs
    stamp = ctx.cfg.policy.stamps_echo_fields
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
        stamp_echo=stamp,
        guidance=repo_guidance(ctx.worktree) if ctx.cfg.policy.isolates_lane_env else "",
    )
    prior_hashes = {str(p["finding_hash"]) for p in ctx.prior}

    def validate(raw: dict[str, Any]) -> VerifierOutput:
        return parse_verifier_output(
            raw,
            expected_phase=expected_phase,
            expected_coderabbit_ids=ctx.coderabbit_ids,
            prior_hashes=prior_hashes,
            stamp_echo=stamp,
            settled_hashes=ctx.settled_hashes,
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
    _hold_untracked_deferrals(ctx, out, expected_phase)
    _record_findings(ctx, phase, "verified-fresh" if fresh_final else "verified", out.findings)
    return out


def _hold_untracked_deferrals(ctx: RunContext, out: VerifierOutput, expected_phase: str) -> None:
    """A prior finding (above a nitpick) this review would lift as INTENTIONALLY_DEFERRED
    keeps standing unless a tracking issue or PR is visible on the PR: the row's `tracking`
    when the thread, description or PR comments show it, else a reference in the finding's
    own thread (resolved or not) that its humans point to as tracking
    (`converse.review_deferral_link`), else one already on record for it
    (`_known_tracking`), unless the deferral predates the rule (`_grandfathered`). Held, the
    finding is carried again as it
    was, and the verifier's row for it becomes STILL_VALID (flagged `needs_tracking`) with a
    reason asking for the issue, which is what its thread is answered with. This overrides
    the verifier on purpose: a deferral nobody tracks is a finding nobody fixes. In place,
    like `publish.attribute_sources`. Post-merge audits are left alone: nobody acts on a
    merged PR's threads, and the audit reports its own findings."""
    if ctx.is_audit:
        return
    recon = _reconciliation(ctx, expected_phase, out)
    kept = {f.prior_hash for f in out.findings if f.prior_hash} | {f.hash for f in out.findings}
    rows = {str(r["finding_hash"]): r for r in out.prior_reconciliation}
    pr_text = _pr_text(ctx)
    said = _finding_thread_texts(ctx)
    held, kept_old = [], []
    for h, merged in recon.items():
        # the verifier's own row decides; a reviewer's only where the verifier has none
        row = rows.get(h, merged)
        prior = ctx.prior_findings.get(h)
        if (
            row.get("status") != "INTENTIONALLY_DEFERRED"
            or prior is None
            or h in kept
            or not converse.needs_tracking(prior.severity)
        ):
            continue
        if converse.review_deferral_link(
            row.get("tracking"), said.get(h, ""), pr_text=pr_text, repo=ctx.repo, number=ctx.number
        ) or _known_tracking(ctx, h):
            continue
        if _grandfathered(ctx, h):
            kept_old.append(h)
            continue
        out.findings.append(
            dataclasses.replace(
                prior, prior_hash=h, source="prior finding (deferral not tracked)", extra={}
            )
        )
        rows[h] = {
            **row,
            "status": "STILL_VALID",
            "reason": converse.UNTRACKED_DEFERRAL_REASON,
            "needs_tracking": True,
        }
        held.append(h)
    if kept_old:
        _defer_event(
            ctx,
            "defer.grandfathered",
            f"{expected_phase}: deferred before {ctx.deferral_cutoff}: {', '.join(kept_old)}",
        )
    if held:
        out.prior_reconciliation = list(rows.values())
        # the verifier approved with these lifted; standing again (above a nitpick), they
        # leave nothing to approve, and an APPROVE would also be watched for CI (ci.py)
        if out.review_action == "APPROVE":
            out.review_action = "COMMENT"
        _defer_event(ctx, "defer.needs_tracking", f"{expected_phase}: held {', '.join(held)}")


def _finding_thread_texts(ctx: RunContext) -> dict[str, str]:
    """finding hash -> what the humans said under that finding's thread: our own answers are
    no evidence of tracking (an issue the bot filed is on record, `_known_tracking`). A
    thread a maintainer resolved (often right after deferring) is only in the evidence, which
    keeps the human replies alone."""
    out = {}
    for t in ctx.evidence.get("review_threads") or []:
        cs = t.get("comments") or []
        m = FINDING_MARKER_RE.search(str(cs[0].get("body") or "")) if cs else None
        if m:
            out[m.group(1)] = "\n".join(str(c.get("body") or "") for c in cs[1:])
    out.update({h: converse.thread_text(t, humans_only=True) for h, t in ctx.open_threads.items()})
    return out


def _known_tracking(ctx: RunContext, h: str) -> str:
    """A tracking link already on record for finding `h` on this PR: an issue the bot filed
    (or linked) for it, or the link an earlier conversation's deferral rested on (the second
    line of a `conceded` row's body)."""
    filed = kv_get(ctx.conn, f"defer.issue:{ctx.repo}#{ctx.number}:{h}")
    if filed:
        return filed
    rows = ctx.conn.execute(
        "SELECT f.body FROM findings f JOIN runs r ON r.id=f.run_id JOIN heads hd ON "
        "hd.id=r.head_id WHERE hd.repo=? AND hd.number=? AND f.stage='conceded' AND f.hash=? "
        "ORDER BY f.id DESC",
        (ctx.repo, ctx.number, h),
    ).fetchall()
    return next((link for r in rows if (link := str(r["body"] or "").partition("\n")[2])), "")


def _grandfathered(ctx: RunContext, h: str) -> bool:
    """Finding `h` was deferred on this PR before deferrals needed tracking
    (`ctx.deferral_cutoff`) and has not stood in a review since, so it keeps lifting without
    one. Evidence of the old deferral, any of: a conversation's INTENTIONALLY_DEFERRED
    concession from a run that started before it (rows older than the outcome column have
    an empty body); an INTENTIONALLY_DEFERRED thread answer before it (event, or our answer
    on the open thread), from either path; or the last review published before it no longer
    carrying the finding it had posted earlier (a verifier's deferral on a thread nobody
    replied to leaves no other trace; a finding withdrawn or fixed then and called deferred
    now is let through too). A finding a review carried after the cutoff was standing
    again, so deferring it now is a new deferral."""
    cutoff = ctx.deferral_cutoff
    if not cutoff or _stood_since(ctx, h, cutoff):
        return False
    pr = (ctx.repo, ctx.number)
    if ctx.conn.execute(
        "SELECT 1 FROM findings f JOIN runs r ON r.id=f.run_id JOIN heads hd ON hd.id=r.head_id "
        "WHERE hd.repo=? AND hd.number=? AND f.hash=? AND f.stage='conceded' "
        "AND (f.body LIKE 'INTENTIONALLY_DEFERRED%' OR f.body='') AND r.started_at < ? LIMIT 1",
        (*pr, h, cutoff),
    ).fetchone():
        return True
    for r in ctx.conn.execute(
        "SELECT detail FROM events WHERE kind='thread.answered' AND repo=? AND number=? "
        "AND ts < ? AND detail LIKE ?",
        (*pr, cutoff, f"%{h}%"),
    ):
        try:
            a = json.loads(r["detail"] or "{}")
        except json.JSONDecodeError:
            continue
        if a.get("finding_hash") == h and a.get("status") == "INTENTIONALLY_DEFERRED":
            return True
    for a in (ctx.open_threads.get(h) or {}).get("bot_answers") or []:
        body = str(a.get("body") or "")
        if str(a.get("created_at") or "~") < cutoff and (
            "**Deferred**" in body or "_Deferred on a maintainer's call" in body
        ):
            return True
    last = ctx.conn.execute(
        "SELECT rv.run_id, r.started_at FROM reviews rv JOIN runs r ON r.id=rv.run_id "
        "WHERE rv.repo=? AND rv.number=? AND r.started_at < ? AND rv.github_review_id IS NOT NULL "
        f"AND {_NOT_A_FOLLOW_UP} ORDER BY r.started_at DESC, rv.id DESC LIMIT 1",
        (*pr, cutoff, StepName.CONVERSE.value, StepName.CI_CHECK.value),
    ).fetchone()
    if last is None:
        return False
    posted_before = ctx.conn.execute(
        "SELECT 1 FROM posted_findings WHERE repo=? AND number=? AND hash=? AND posted_at < ?",
        (*pr, h, last["started_at"]),
    ).fetchone()
    return bool(posted_before) and not _carried_by(ctx, h, "r.id=?", (last["run_id"],))


# a `reviews` row of a run that reviewed the code, not a conversation's verdict follow-up
# or a CI update (ci.py), neither of which records a verifier set; `rv.run_id` must be in
# scope, the two step names bound
_NOT_A_FOLLOW_UP = (
    "NOT EXISTS (SELECT 1 FROM steps s WHERE s.run_id=rv.run_id AND s.name IN (?, ?))"
)


def _stood_since(ctx: RunContext, h: str, cutoff: str) -> bool:
    """A review of this PR other than this run, started at or after `cutoff`, kept `h`."""
    return _carried_by(ctx, h, "r.started_at >= ? AND r.id <> ?", (cutoff, ctx.run_id), pr=True)


def _carried_by(
    ctx: RunContext, h: str, where: str, params: tuple[Any, ...], *, pr: bool = False
) -> bool:
    """Some run matching `where` (over `runs r`) kept finding `h` in its verifier set or
    posted it, by hash or, since a carried finding may be re-hashed, by file and title."""
    ident = ctx.conn.execute(
        "SELECT f.file, f.title FROM findings f JOIN runs r ON r.id=f.run_id JOIN heads hd ON "
        "hd.id=r.head_id WHERE hd.repo=? AND hd.number=? AND f.hash=? AND f.stage='posted' "
        "LIMIT 1",
        (ctx.repo, ctx.number, h),
    ).fetchone()
    file, title = (ident["file"] or "", str(ident["title"] or "")) if ident else ("", "")
    scope = "hd.repo=? AND hd.number=? AND " if pr else ""
    return bool(
        ctx.conn.execute(
            "SELECT 1 FROM findings f JOIN runs r ON r.id=f.run_id JOIN heads hd ON "
            f"hd.id=r.head_id WHERE {scope}{where} "
            "AND f.stage IN ('verified', 'verified-fresh', 'posted') "
            "AND (f.hash=? OR (? <> '' AND f.file=? AND lower(trim(f.title))=lower(trim(?)))) "
            "LIMIT 1",
            (*((ctx.repo, ctx.number) if pr else ()), *params, h, title, file, title),
        ).fetchone()
    )


def _blind(outputs: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """Phase-2 outputs as the verifier sees them on a comparison run: each role's pair gets
    the neutral labels `<role>/a` and `<role>/b`, which one is the comparison model decided
    per run, so neither the `#2` key nor the order gives the second model away. Unchanged
    when no comparison lane ran. Attribution does not depend on these labels (it matches
    each lane's own findings by hash)."""
    twins = {k.removesuffix(COMPARE_SUFFIX) for k in outputs if k.endswith(COMPARE_SUFFIX)}
    if not twins:
        return outputs
    out: dict[str, Any] = {}
    for key, value in outputs.items():
        if key.removesuffix(COMPARE_SUFFIX) in twins:
            key = _blind_label(key, seed=seed)
        out[key] = value
    return dict(sorted(out.items()))


# ---- policy v12: verify+match, code rules (rules.py) ----


def _candidates_of(outputs: dict[str, ReviewerOutput]) -> list[Finding]:
    return [f for o in outputs.values() for f in o.findings if f.candidate_id]


def _gate_kept(v1: VerifierOutput) -> list[Finding]:
    """What the gate verifier kept, as candidates for the final one (`v1-<n>`): each group's
    claim as the gate composed it (before the code's PLAUSIBLE demotion), with its gate verdict
    shown. The final verifier re-judges them."""
    out: list[Finding] = []
    for n, (g, f) in enumerate(_outcome(v1).kept, 1):
        out.append(
            Finding(
                file=f.file,
                title=g.title,
                body=g.body,
                severity=g.severity,
                category=f.category,
                line_start=f.line_start,
                line_end=f.line_end,
                suggestion=g.suggestion,
                source=f.source,
                extra={
                    rules.FAILURE_SCENARIO: g.evidence,
                    "gate_verdict": f"{g.verdict} at the gate, matched {g.match}",
                },
                candidate_id=f"v1-{n}",
            )
        )
    return out


def _phase_candidates(ctx: RunContext, phase: str) -> list[Finding]:
    """The candidates a phase's verify+match weighs: Phase 1's at the gate; at the final one
    Phase 2's plus what the gate kept, or plus Phase 1's own on a single-stage run."""
    if phase == "verify1":
        return _candidates_of(ctx.phase1_outputs)
    p2 = _candidates_of(ctx.phase2_outputs)
    if ctx.single_stage:
        return _candidates_of(ctx.phase1_outputs) + p2
    if ctx.verify1 is not None:
        return p2 + _gate_kept(ctx.verify1)
    return p2


def _phase_note(ctx: RunContext, phase: str) -> str:
    if phase == "verify1":
        return (
            "This is the gate verification of the Phase-1 candidates (`p1-*`). The orchestrator "
            "decides from your verdicts whether Phase 2 runs before anything is published."
        )
    if ctx.single_stage:
        return (
            "This is the final verification. Phase 1 (`p1-*`) and Phase 2 (`p2-*`) reviewed "
            "this head side by side with no gate between them; weigh every candidate the same."
        )
    if ctx.verify1 is not None:
        return (
            "This is the final verification of the Phase-2 candidates (`p2-*`). The `v1-*` "
            "candidates are what the gate verifier kept from Phase 1 (its verdict is shown as "
            "`gate_verdict`): judge them again, as final authority."
        )
    return "This is the final verification of the Phase-2 candidates (`p2-*`)."


def _sources12(ctx: RunContext, members: list[Finding]) -> str:
    """Who raised a verified group: the models and agents of its members' lanes, named as
    `publish.attribute_sources` names them. A gate-kept candidate already carries its own."""
    parts: list[str] = []
    lanes: list[dict[str, Any]] = []
    for m in members:
        phase, _, key = m.source.partition(":")
        fresh = key.startswith("fresh:")
        key = key.removeprefix("fresh:")
        own = [
            r
            for r in ctx.reviewers
            if r["phase"] == phase and r["key"] == key and bool(r.get("fresh")) == fresh
        ]
        if own:
            lanes += own
        elif m.source and m.source not in parts:
            parts.append(m.source)
    named = publish.name_lanes(lanes)
    return "; ".join(named + [p for p in parts if p not in named]) or "unknown"


def _last_messages(ctx: RunContext) -> dict[str, str]:
    out: dict[str, str] = {}
    for h, t in ctx.thread_index.items():
        last = (t.get("transcript") or [])[-1:] or []
        if last:
            who = "you" if last[0].get("is_bot") else str(last[0].get("author") or "")
            out[h] = f"{who}: {converse.strip_marker(str(last[0].get('body') or ''))}"
    return out


def _verify_shard(
    ctx: RunContext,
    *,
    phase: str,
    role: str,
    lm: LaneModel,
    batch: list[Finding],
    summary: dict[str, Any],
    raised_now: set[str],
    note: str,
) -> rules.VerifyAnswer:
    assert ctx.meta
    ids = [str(f.candidate_id) for f in batch]
    prompt = verify_prompt(
        ctx.cfg,
        repo=ctx.repo,
        number=ctx.number,
        head_sha=ctx.sha,
        meta=ctx.meta.as_dict(),
        coverage_from=ctx.coverage_from,
        candidates=[rules.brief(f) for f in batch],
        coderabbit=ctx.coderabbit,
        ledger_summary=summary,
        evidence=ctx.evidence,
        guidance=repo_guidance(ctx.worktree) if ctx.cfg.policy.isolates_lane_env else "",
        phase_note=note,
        structured=ctx.cfg.policy.structured_output,
    )

    def validate(raw: dict[str, Any]) -> rules.VerifyAnswer:
        return rules.parse_verify(
            raw,
            candidate_ids=ids,
            issues=ctx.issues,
            coderabbit_ids=ctx.coderabbit_ids,
            extra_open=raised_now,
        )

    return _run_checked_lane(
        ctx,
        phase=phase,
        role=role,
        lm=lm,
        prompt=prompt,
        is_verifier=True,
        check=OutputCheck(validate=validate, kind="verify", ids=ids),
        schema=schemas.verifier_schema(ids),
    )


def _verify12(
    ctx: RunContext,
    *,
    phase: str,
    lm: LaneModel,
    expected_phase: str,
    candidates: list[Finding],
    note: str,
    raised_now: list[Finding] | None = None,
    stage: str = "verified",
) -> VerifierOutput:
    """One policy v12 verify+match per phase (v12 design §2.2): every candidate, the CodeRabbit
    comments and the ledger summary to one lane, split into at most three shards by file above
    twelve candidates (`_verify_shards`). No candidates: no lane. The code rules
    (`rules.apply_rules`) turn the answer into this round's issues; the ledger's standing
    issues ride along as `carried`."""
    raised = list(raised_now or [])
    raised_hashes = {f.hash for f in raised}
    outs = _thread_outcomes(ctx)
    summary = rules.ledger_summary(
        ctx.issues,
        thread_status={h: o.status for h, o in outs.items()},
        last_message=_last_messages(ctx),
        raised_now=raised,
    )
    answer = _merge_answers(
        ctx,
        phase,
        _verify_shards(
            ctx,
            phase=phase,
            lm=lm,
            candidates=candidates,
            summary=summary,
            raised=raised_hashes,
            note=note,
        ),
        empty=not candidates,
    )
    outcome = rules.apply_rules(
        answer,
        {str(f.candidate_id): f for f in candidates},
        ctx.issues,
        changed_since_closed=lambda i: _closed_changed(ctx, i),
        source_of=lambda ms: _sources12(ctx, ms),
        thread_url=lambda i: _issue_url(ctx, i),
        extra_open=raised_hashes,
    )
    post, noted = _late_split(ctx, outcome.new)
    carried = _carried12(
        ctx,
        rules.project(
            ctx.issues, _pending_statuses(outs, set(outcome.corroborated)), outcome.upgrades
        ),
    )
    standing_sev = [f.severity for f in (*post, *carried)]
    out = VerifierOutput(
        summary=answer.summary,
        # provisional (an audit's report reads it); a review's verdict is decided from the
        # ledger when it publishes (`_verdict12`)
        review_action=rules.verdict(
            blockers=standing_sev.count("blocking"),
            suggestions=standing_sev.count("suggestion"),
            full_strength=True,
            confirmed=True,
        ),
        findings=post,
        dropped=outcome.dropped,
        out_of_scope=[],
        coderabbit_reactions=_dedupe_reactions(outcome.reactions),
        prerequisite_adjudications=[],
        adjudication_complete=True,
        review_phase=expected_phase,
        raw=answer.raw,
        normalized=answer.normalized,
        carried=carried,
        noted=noted,
        rules=outcome,
    )
    _record_findings(ctx, phase, stage, out.findings)
    _write_verify_artifact(ctx, phase, stage, candidates, answer, out)
    return out


def _verify_shards(
    ctx: RunContext,
    *,
    phase: str,
    lm: LaneModel,
    candidates: list[Finding],
    summary: dict[str, Any],
    raised: set[str],
    note: str,
) -> list[rules.VerifyAnswer]:
    """The verify+match lanes over `candidates` (`rules.shards`), side by side. A shard that
    fails for good (after the retry inside `_run_checked_lane`) has its candidates dropped
    when none of them is blocking (the run can no longer approve), and fails the run (INFRA,
    `UncheckedBlockers`) when a blocker would go unchecked."""
    batches = rules.shards(candidates) if candidates else []
    k = len(batches)
    roles = ["verifier"] if k == 1 else [f"verifier-{n}of{k}" for n in range(1, k + 1)]
    notes = (
        [note]
        if k == 1
        else [
            f"{note}\n\nThis is batch {n} of {k}: the other batches hold the candidates of other "
            "files."
            for n in range(1, k + 1)
        ]
    )

    def one(n: int) -> rules.VerifyAnswer | None:
        batch = batches[n]
        try:
            return _verify_shard(
                ctx,
                phase=phase,
                role=roles[n],
                lm=lm,
                batch=batch,
                summary=summary,
                raised_now=raised,
                note=notes[n],
            )
        except ReviewError as exc:
            ctx.check_cancel()
            blocking = [f.candidate_id for f in batch if f.severity == "blocking"]
            if blocking:
                raise UncheckedBlockers(
                    FailKind.INFRA,
                    f"{phase}/{roles[n]} failed with blocking candidates unchecked "
                    f"({', '.join(map(str, blocking))}): {exc.message}",
                ) from exc
            ctx.verify_incomplete = True
            _lane_event(
                ctx,
                "verify.shard_dropped",
                f"{phase}/{roles[n]}: {len(batch)} non-blocking candidate(s) dropped: {exc.message}",
            )
            return None
        finally:
            ctx.close_lane_conn()

    if k <= 1:
        res = [one(0)] if k else []
    else:
        with ThreadPoolExecutor(max_workers=k, thread_name_prefix=f"{phase}-shard") as pool:
            res = list(pool.map(one, range(k)))
    return [a for a in res if a is not None]


def _merge_answers(
    ctx: RunContext, phase: str, answers: list[rules.VerifyAnswer], *, empty: bool
) -> rules.VerifyAnswer:
    """The shards' answers as one (`empty`: there was nothing to verify)."""
    text = "\n\n".join(a.summary for a in answers if a.summary) or (
        "The reviewers raised nothing to verify on this head." if empty else ""
    )
    answer = rules.VerifyAnswer(
        summary=text,
        groups=[g for a in answers for g in a.groups],
        raw={"summary": text, "groups": [a.raw.get("groups") for a in answers]},
        normalized=[n for a in answers for n in a.normalized],
    )
    if answer.normalized:
        _lane_event(
            ctx,
            "lane.output_normalized",
            f"{phase}/verifier: {'; '.join(answer.normalized)}"[:1000],
        )
    return answer


def _dedupe_reactions(reactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One reaction per CodeRabbit comment; agreeing wins over disagreeing."""
    out: dict[int, dict[str, Any]] = {}
    for r in reactions:
        if r["comment_id"] not in out or r["action"] == "agree":
            out[r["comment_id"]] = r
    return list(out.values())


def _write_verify_artifact(
    ctx: RunContext,
    phase: str,
    stage: str,
    candidates: list[Finding],
    answer: rules.VerifyAnswer,
    out: VerifierOutput,
) -> None:
    outcome = out.rules
    assert outcome is not None
    (ctx.run_dir / f"{phase}{'-fresh' if stage != 'verified' else ''}.json").write_text(
        json.dumps(
            {
                "candidates": [rules.brief(f) for f in candidates],
                "groups": [dataclasses.asdict(g) for g in answer.groups],
                "new": [f.to_dict() for f in out.findings],
                "noted": [f.to_dict() for f in out.noted],
                "corroborated": outcome.corroborated,
                "upgrades": sorted(outcome.upgrades),
                "dropped": outcome.dropped,
                "carried": [f.prior_hash for f in out.carried],
            },
            indent=1,
        )
    )


def _stand12(
    ctx: RunContext, issues: dict[str, ledger.Issue], judged: Iterable[str] = ()
) -> list[ledger.Issue]:
    """The issues that count toward this round's verdict (`rules.standing`): the reconcile's
    leftovers only once the thread lane has judged them (`judged`)."""
    return rules.standing(
        issues,
        reconciled=ctx.reconciled,
        pending=ctx.reconcile_pending - set(judged),
        last_reviewed_sha=ctx.last_reviewed_sha,
        open_threads=ctx.open_threads,
    )


def _carried12(ctx: RunContext, issues: dict[str, ledger.Issue]) -> list[Finding]:
    stand = _stand12(ctx, issues, _thread_outcomes(ctx))
    return rules.carried(stand, lambda i: _issue_url(ctx, i))


def _late_split(ctx: RunContext, new: list[Finding]) -> tuple[list[Finding], list[Finding]]:
    """`late_nonblocking` (owner decision D2): this round's non-blocking new issues on lines
    unchanged since the last final review are tagged `late`; under "note" they are collapsed
    into "Also noticed" (not posted, not in the ledger, never holding APPROVE). Their count is
    recorded either way, for the eval. Revivals and regressions are never late."""
    pol = ctx.cfg.policy.pipeline
    since = ctx.last_final_sha
    if pol is None or not since or since == ctx.sha:
        return new, []
    late: list[Finding] = []
    for f in new:
        if f.severity == "blocking" or "revives" in f.extra or "regression_of" in f.extra:
            continue
        # the anchor is in this head's numbering: diff from it (lines_changed is symmetric)
        if _lines_changed(ctx, ctx.sha, since, f.file, f.line_start, f.line_end) is False:
            f.extra["late"] = True
            late.append(f)
    if late:
        _lane_event(
            ctx,
            "v12.late_nonblocking",
            f"{len(late)} late non-blocking finding(s) since {since[:8]} ({pol.late_nonblocking})",
        )
    if pol.late_nonblocking != "note":
        return new, []
    return [f for f in new if not f.extra.get("late")], late


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

    def record(c: sqlite3.Connection) -> None:
        # nothing new went on GitHub: the findings the PR already had are carried, and only
        # those it never had (a prior attempt posted them, then died) are recorded as posted
        have = ledger.known(c, ctx.repo, ctx.number)
        posted = [f for f in verified.findings if (f.prior_hash or f.hash) not in have]
        ledger.record_published(
            c,
            ctx.repo,
            ctx.number,
            run_id=ctx.run_id,
            sha=ctx.sha,
            posted=posted,
            kept=verified.findings,
        )

    _ledger_write(ctx, "publication", record)


def _record_lifted_reconciliations(ctx: RunContext, phase: str, verified: VerifierOutput) -> None:
    """Persist full-review lifts as tombstones for later PR heads.

    Conversation concessions already use this table. Full reviews used to answer and publish a
    withdrawal but leave only the original ``posted_findings`` identity, so the next head fed it
    back as an ordinary active blocker. A later full review may explicitly reintroduce the same
    identity; its marker is evidence that the new finding is intentional and keeps the tombstone
    from winning over the new publication.
    """
    # a hash the verifier published (carried, held or reintroduced) stands, whatever a
    # reviewer lane's row merged in by `_reconciliation` says
    kept = {
        h
        for f in verified.findings
        for h in (f.hash, f.prior_hash, f.extra.get("reintroduced_finding_hash"))
        if h
    }
    rows = []
    for h, outcome in _reconciliation(ctx, phase, verified).items():
        status = str(outcome.get("status") or "")
        prior = ctx.prior_findings.get(h)
        if status not in converse.LIFTING_STATUSES or h in kept or prior is None:
            continue
        # a deferral keeps the tracking link it was checked against on a second line, as
        # `_record_conceded` does, so `_known_tracking` still finds it on later heads
        tracking = ""
        if status == "INTENTIONALLY_DEFERRED":
            tracking = converse.review_deferral_link(
                outcome.get("tracking"),
                _finding_thread_texts(ctx).get(h, ""),
                pr_text=_pr_text(ctx),
                repo=ctx.repo,
                number=ctx.number,
            )
        rows.append(
            (
                ctx.run_id,
                phase,
                "conceded",
                h,
                prior.file,
                prior.line_start,
                prior.line_end,
                prior.severity,
                None,
                prior.category,
                prior.title,
                status + (f"\n{tracking}" if tracking else ""),
                None,
            )
        )
    if rows:
        with tx(ctx.conn):
            ctx.conn.executemany(_FINDINGS_INSERT, rows)


def step_publish(
    ctx: RunContext,
    *,
    phase: str,
    verified: VerifierOutput,
    verifier_lm: LaneModel,
    phase2_skipped: str | None = None,
) -> publish.PublishResult:
    if ctx.cfg.policy.verifier_v12:
        return _publish12(
            ctx,
            phase=phase,
            verified=verified,
            verifier_lm=verifier_lm,
            phase2_skipped=phase2_skipped,
        )
    _require_live_base(ctx)
    existing, phase = _existing_review(ctx, phase)
    if existing:
        # Either a prior attempt posted but died before recording it, or this is a reply-
        # triggered re-review of an already-reviewed commit. Backfill so future rounds still
        # see these findings as "prior", never re-post the review for this sha/phase, answer the
        # threads that were replied to, and, if the verdict moved (a blocker withdrawn or a new
        # one found), post a short follow-up review so the standing verdict is not left stale.
        _backfill_posted(ctx, phase, existing, verified)
        _record_lifted_reconciliations(ctx, phase, verified)
        _answer_threads(ctx, phase, verified)
        update = _verdict_update(ctx, phase, verified, verifier_lm, phase2_skipped=phase2_skipped)
        if update is not None:
            return update
        return _publish_review(
            ctx,
            phase=phase,
            verified=verified,
            verifier_lm=verifier_lm,
            phase2_skipped=phase2_skipped,
            rereview=True,
        )[1]
    result = _publish_review(
        ctx,
        phase=phase,
        verified=verified,
        verifier_lm=verifier_lm,
        phase2_skipped=phase2_skipped,
        rereview=False,
    )[1]
    _answer_threads(ctx, phase, verified)
    return result


def _require_live_base(ctx: RunContext) -> None:
    """The review may have taken long enough for a push to land after the initial worktree
    check. Never publish an approval (or any verdict) against an obsolete head; the run ends
    superseded (HeadObsolete) and ingest queues the live commit. A moved base is fatal."""
    live = github.pr_meta(ctx.gh, ctx.repo, ctx.number)
    _require_live(ctx, live)
    if ctx.base_sha and live.base_sha and live.base_sha != ctx.base_sha:
        raise ReviewError(
            FailKind.FATAL,
            f"live base {live.base_sha[:8]} != assigned {ctx.base_sha[:8]}",
        )


def _existing_review(ctx: RunContext, phase: str) -> tuple[dict[str, Any] | None, str]:
    """The bot's review of this commit and `phase` already on the PR, and the phase it is: a
    same-sha re-review that now finds blockers after a FINAL review stood corrects the final
    verdict with a follow-up rather than stacking a second full review."""
    existing = github.existing_review_for_sha(
        ctx.gh, ctx.repo, ctx.number, ctx.sha, phase, ctx.cfg.bot_login
    )
    if existing is None and phase == "preliminary":
        existing = github.existing_review_for_sha(
            ctx.gh, ctx.repo, ctx.number, ctx.sha, "final", ctx.cfg.bot_login
        )
        if existing:
            phase = "final"
    return existing, phase


def _publish_review(
    ctx: RunContext,
    *,
    phase: str,
    verified: VerifierOutput,
    verifier_lm: LaneModel,
    phase2_skipped: str | None,
    rereview: bool,
) -> tuple[publish.ReviewModel, publish.PublishResult]:
    """Build, post and record the review. `rereview`: a same-sha re-review whose review
    already stands, posted as an informational COMMENT; otherwise the review proper, with its
    CodeRabbit reactions. Either way its body is kept as `review-<phase>.md`."""
    model = _build_review(
        ctx,
        phase=phase,
        verified=verified,
        verifier_lm=verifier_lm,
        phase2_skipped=phase2_skipped,
        rereview=rereview or ctx.has_prior_review,
    )
    if rereview:
        # the artifact goes first, as on the normal path: a failed write after the post would
        # fail the run and its retry would post the COMMENT again
        (ctx.run_dir / f"review-{phase}.md").write_text(publish.render(model))
        result = publish.publish(
            ctx.gh,
            model,
            bot_login=ctx.cfg.bot_login,
            dry_run=ctx.dry_run,
            force_comment=True,
        )
        with contextlib.suppress(OSError):  # the posted body (with its transport note)
            (ctx.run_dir / f"review-{phase}.md").write_text(result.body)
        _record_publication(ctx, phase, model, result, verified)
        if ctx.kept_verdict:
            # the APPROVE still stands on GitHub: the label and the CI watch (ci.py) read
            # our record, which the summary's COMMENT just overwrote
            _record_verdict(ctx, phase, verified, ctx.kept_verdict)
        return model, result
    (ctx.run_dir / f"review-{phase}.md").write_text(publish.render(model))
    result = publish.publish(
        ctx.gh,
        model,
        bot_login=ctx.cfg.bot_login,
        dry_run=ctx.dry_run,
    )
    _record_publication(ctx, phase, model, result, verified)
    _record_lifted_reconciliations(ctx, phase, verified)
    if result.posted and verified.coderabbit_reactions and not ctx.dry_run:
        publish.post_coderabbit_reactions(
            ctx.gh, ctx.repo, ctx.number, verified.coderabbit_reactions, ctx.cfg.bot_login
        )
    return model, result


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
        _record_answered_events(ctx, answered)
    _ledger_write(
        ctx,
        "answers",
        lambda c: ledger.record_answers(
            c,
            ctx.repo,
            ctx.number,
            run_id=ctx.run_id,
            sha=ctx.sha,
            answered=answered,
            source="verifier",
        ),
    )


def _record_answered_events(ctx: RunContext, answered: list[dict[str, Any]]) -> None:
    """One `thread.answered` event per item an answering pass returned. Caller holds the
    write transaction."""
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
    lifted_titles: list[str] | None = None,
) -> publish.PublishResult | None:
    """On a same-sha re-review, post a follow-up review when the verdict moved.
    `lifted_titles` (policy v12): the blockers this round lifted, from the thread lane.

    Compares against the bot's LATEST review state for this (sha, phase), which includes
    earlier follow-ups, so the same correction is never posted twice. A review a maintainer
    dismissed is left dismissed: we only ever move from a state we set ourselves."""
    if ctx.dry_run and not ctx.replay:
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
        confirmation=_confirmation_provenance(ctx),
        fresh_first=ctx.fresh_first,
        late=ctx.late_report,
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
        ctx.kept_verdict = "APPROVE"
        return None
    if lifted_titles is not None:
        titles = lifted_titles
    else:
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
        dry_run=ctx.dry_run,
    )
    if not result.posted:
        _record_dry_publication(ctx, phase, result, verified.findings, verdict_update=True)
        return result
    with tx(ctx.conn):
        _insert_findings(ctx, phase, late.STAGE_LISTED, verified.followups)
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
        confirmation=_confirmation_provenance(ctx),
        fresh_first=ctx.fresh_first,
        late=ctx.late_report,
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
        rereview=rereview,
        fresh=_fresh_threads(ctx),
        settled_hashes=ctx.settled_hashes,
    )
    model.fix_forward = ctx.cfg.policy.pipeline is None
    held = [
        ctx.prior_findings[str(r["finding_hash"])].title
        for r in verified.prior_reconciliation
        if r.get("needs_tracking") and str(r["finding_hash"]) in ctx.prior_findings
    ]
    if held:
        model.extra_lines.append(publish.untracked_deferrals_line(held))
    return model


def _fresh_threads(ctx: RunContext) -> Callable[[Finding, Any], bool] | None:
    """Policy v12: a regression or a revival never folds onto the thread of a closed issue (it
    would block from inside a resolved thread nobody reads); it gets a new thread, whose body
    links the old one. Only a standing issue's thread can take it (a retry of this round)."""
    if not ctx.cfg.policy.verifier_v12:
        return None
    standing = {h for h, i in ctx.issues.items() if i.status in rules.VERDICT_STANDING}

    def fresh(f: Finding, comment: Any) -> bool:
        reopens = bool(f.extra.get("revives") or f.extra.get("regression_of"))
        return reopens and comment.finding_hash not in standing

    return fresh


def _lane_provenance(lm: LaneModel, role: str) -> dict[str, Any]:
    """How a verifier/conversation lane is disclosed in the published provenance."""
    return {
        "model": lm.model,
        "agent": lm.agent,
        "role": role,
        "substitute_for": lm.substitute_for,
    }


def _confirmation_provenance(ctx: RunContext) -> dict[str, Any] | None:
    """The policy v12 confirmation pass, disclosed when it ran on this head."""
    c = ctx.confirmation
    return {"roles": c.roles, "new": c.new} if c is not None and c.ran else None


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


def _finding_record(f: Finding) -> dict[str, Any]:
    return {**f.to_dict(), "hash": f.hash}


def _inline_record(comments: list[dict[str, Any]], kept: list[Finding]) -> list[dict[str, Any]]:
    titles = {f.hash: f.title for f in kept}
    out = []
    for c in comments:
        m = FINDING_MARKER_RE.search(str(c.get("body") or ""))
        h = m.group(1) if m else ""
        line = c.get("line")
        out.append(
            {
                "hash": h,
                "file": c.get("path"),
                "line_start": c.get("start_line") or line,
                "line_end": line,
                "title": titles.get(h, ""),
            }
        )
    return out


def _record_dry_publication(
    ctx: RunContext,
    phase: str,
    result: publish.PublishResult,
    kept: list[Finding],
    *,
    verdict_update: bool = False,
) -> None:
    """A dry run posts nothing and writes no `posted` rows: keep what would have been posted
    as `publication-<phase>.json` (read by `reviewsys replay`'s report). A same-sha verdict
    follow-up stands on the verifier's findings (`verdict_update`)."""
    (ctx.run_dir / f"publication-{phase}.json").write_text(
        json.dumps(
            {
                "phase": phase,
                "event": result.event,
                "transport_event": result.transport_event,
                "verdict_update": verdict_update,
                "kept": [_finding_record(f) for f in kept],
                "suppressed": len(result.suppressed),
                # the inline comments it would open (a new thread each): the eval counts those
                # landing on an issue that already has an open thread
                "inline": _inline_record(result.comments, kept),
                "body": result.body,
            },
            indent=1,
        )
    )


def _record_publication(
    ctx: RunContext,
    phase: str,
    model: publish.ReviewModel,
    result: publish.PublishResult,
    verified: VerifierOutput,
) -> None:
    if ctx.dry_run:
        _record_dry_publication(ctx, phase, result, model.kept)
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
            _insert_findings(ctx, phase, late.STAGE_LISTED, verified.followups)
        _set_run_review(
            ctx,
            blocker_count=verified.blocker_count,
            review_id=result.review_id,
            review_url=result.review_url,
        )
    if result.posted:
        _ledger_write(
            ctx,
            "publication",
            lambda c: ledger.record_published(
                c,
                ctx.repo,
                ctx.number,
                run_id=ctx.run_id,
                sha=ctx.sha,
                posted=model.kept,
                kept=verified.findings,
                deduped_comment_ids=[s.comment_id for s in model.suppressed if s.comment_id],
            ),
        )


def _ledger_write(ctx: RunContext, what: str, fn: Callable[[sqlite3.Connection], object]) -> None:
    """Write-through to the issue ledger (ledger.py), after the step's own bookkeeping has
    committed. Write-only under policy v11: nothing reads it, and a failed write is logged and
    recorded (`ledger.write_failed`), never raised. Audits and dry runs are not part of a PR's
    review history."""
    if ctx.is_audit or ctx.dry_run:
        return
    ledger.safe_write(ctx.conn, what, fn, repo=ctx.repo, number=ctx.number, run_id=ctx.run_id)


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
    roles: list[str] | None = None,
) -> dict[str, ReviewerOutput]:
    # the planned lanes, so a phase that is still going can say how many of them are done
    _step_start(ctx, step, {"roles": roles if roles is not None else _phase_roles(ctx, phase)})
    if callable(lm):
        lm = lm()  # Phase 1 picks its model inside the step, so a slow lookup shows there
    outputs = _reviewer_lanes(
        ctx,
        phase=phase,
        lm=lm,
        expected_phase=expected_phase,
        fresh=fresh,
        abort=abort,
        roles=roles,
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
    # the verifier weighed the checks that had failed when the context was read
    _watch_ci(ctx, known=ci.failing_names(ctx.evidence.get("ci"), ctx.cfg.ci.ignore_re))


def _watch_ci(ctx: RunContext, *, known: list[str]) -> None:
    """Once an APPROVE stands on this head, watch its CI if checks are still running or
    failed since `known` (ci.register). Best effort: the verdict is already published."""
    if ctx.dry_run or ctx.is_audit:
        return
    try:
        ci.register(
            ctx.conn,
            ctx.cfg,
            ctx.gh,
            repo=ctx.repo,
            number=ctx.number,
            sha=ctx.sha,
            run_id=ctx.run_id,
            known=known,
        )
    except Exception as exc:
        log.warning("CI watch for %s#%s not registered: %s", ctx.repo, ctx.number, exc)
        with contextlib.suppress(sqlite3.Error), tx(ctx.conn):
            event(
                ctx.conn,
                "ci.watch_failed",
                repo=ctx.repo,
                number=ctx.number,
                run_id=ctx.run_id,
                detail=str(exc)[:500],
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


def _verify2(ctx: RunContext, effort: TierEffort) -> VerifierOutput:
    """The final verification under policy v11, in three parts:

    - A second Phase-2 pass, so a PR gets as much as possible in one round: on the first full
      review every Phase-2 reviewer runs a second, independent time before the final verifier,
      which weighs both sets (`_fresh_first_due`; best effort: `_fresh_first`). On a later
      round the fresh final gate runs after the verifier (`_fresh_final_due`), and only once
      the blockers of the previous final review are gone, since they may have hidden others.
    - On a later round, new findings on code unchanged since the previous final review leave
      the normal path (`_gate_late`): nitpicks are dropped, suggestions become follow-up
      candidates, and blockers need an independent confirmation.
    - With nothing blocking, the verdict is an approval (`_approve_when_clear`)."""
    pol = ctx.cfg.policy
    combined = _fresh_first(ctx, effort) if _fresh_first_due(ctx, effort) else None
    verified = _verify_step(
        ctx,
        step=StepName.VERIFY2,
        phase="verify2",
        lm=ctx.lane_model(pol.phase2_verifier),
        expected_phase="final",
        phase2_outputs=combined,
    )
    step, stage = StepName.VERIFY2, "verified"
    if _fresh_final_due(ctx, verified, effort):
        ctx.fresh_final = True
        combined = {**ctx.phase2_outputs, **_fresh_reviewers(ctx, effort)}
        verified = _verify_step(
            ctx,
            step=StepName.FRESH_VERIFY2,
            phase="verify2",
            lm=ctx.lane_model(pol.phase2_verifier),
            expected_phase="final",
            phase2_outputs=combined,
            fresh_final=True,
        )
        step, stage = StepName.FRESH_VERIFY2, "verified-fresh"
    split = _classify_late(ctx, verified)
    if split is not None:
        _gate_late(ctx, verified, split, step=step, stage=stage)
    _approve_when_clear(ctx, verified)
    return verified


def _converges(ctx: RunContext) -> bool:
    """The round-converging rules of `_verify2` apply: a policy v11 review. An audit (nobody
    acts on a merged PR's review rounds) and a partial policy v12 pipeline keep the earlier
    behaviour."""
    return ctx.cfg.policy.pipeline is None and not ctx.is_audit


def _fresh_reviewers(ctx: RunContext, effort: TierEffort) -> dict[str, ReviewerOutput]:
    """The Phase-2 reviewers once more, without the prior-finding checklist, keyed
    `fresh:<role>` for the verifier."""
    outputs = _reviewer_step(
        ctx,
        step=StepName.FRESH_PHASE2,
        phase="phase2",
        lm=ctx.lane_model(_with_effort(ctx.cfg.policy.phase2_reviewer, effort.phase2)),
        expected_phase="final",
        fresh=True,
    )
    return {f"fresh:{role}": output for role, output in outputs.items()}


# tiers whose first review stays one Phase-2 pass: their changes are small enough
FRESH_FIRST_SKIP_TIERS = frozenset({"trivial", "low"})


def _fresh_first_due(ctx: RunContext, effort: TierEffort) -> bool:
    """The first full review of a PR runs the second Phase-2 pass, where Phase 2 runs at all.
    Not on a comparison run (its second model already doubles every Phase-2 reviewer), a
    single-stage tier (Phase 1 already ran beside Phase 2) or a fast tier."""
    pol = ctx.cfg.policy
    return (
        ctx.prev_final_sha is None
        and _converges(ctx)
        and not ctx.compare_model
        and not ctx.single_stage
        and ctx.tier not in FRESH_FIRST_SKIP_TIERS
        and pol.phase2_enabled
        and effort.phase2 is not None
    )


def _fresh_first(ctx: RunContext, effort: TierEffort) -> dict[str, ReviewerOutput] | None:
    """The first review's second Phase-2 pass, beside the first pass's outputs. Best effort:
    when it fails, the review goes on with the first pass alone (`fresh_first.failed`)."""
    try:
        fresh = _fresh_reviewers(ctx, effort)
    except ReviewError as exc:
        ctx.check_cancel()
        _step_end(ctx, StepName.FRESH_PHASE2, "failed", {"error": exc.message[:1000]})
        _lane_event(ctx, "fresh_first.failed", exc.message)
        with ctx.state_lock:
            ctx.reviewers[:] = [r for r in ctx.reviewers if not r["fresh"]]
        return None
    ctx.fresh_first = True
    return {**ctx.phase2_outputs, **fresh}


def _fresh_final_due(ctx: RunContext, verified: VerifierOutput, effort: TierEffort) -> bool:
    """The fresh final gate on a later round: when the previous final review had blockers and
    none stands now, so what they masked gets an unanchored look. Never after a clean
    approval, and not while any blocker stands (a new one on unchanged code included: it goes
    to its confirmation instead). Without `_converges`, the earlier rule: any re-review
    without blockers."""
    pol = ctx.cfg.policy
    if not pol.phase2_enabled or effort.phase2 is None or verified.blocker_count:
        return False
    if not _converges(ctx):
        return ctx.has_prior_review
    return ctx.prev_final_sha is not None and ctx.prev_final_blockers


def _classify_late(ctx: RunContext, verified: VerifierOutput) -> late.Split | None:
    """Each verified finding's code status (`extra["code_status"]`, late.py) and the split of
    a later round's findings; None on a first full review (or without `_converges`)."""
    if not _converges(ctx) or ctx.prev_final_sha is None or ctx.worktree is None:
        return None
    if ctx.code_history is None:
        ctx.code_history = late.CodeHistory(
            ctx.worktree, ctx.sha, ctx.prev_final_sha, ctx.first_reviewed_sha
        )
    for f in verified.findings:
        f.extra["code_status"] = ctx.code_history.status(f)
    records = _raised_records(ctx)
    raised = {
        str(r["hash"])
        for r in ctx.conn.execute(
            "SELECT hash FROM posted_findings WHERE repo=? AND number=?", (ctx.repo, ctx.number)
        )
    }
    raised |= set(ctx.prior_findings) | set(ctx.open_threads)

    def is_new(f: Finding) -> bool:
        if f.prior_hash or f.hash in raised:
            return False
        return find_duplicate(f, records)[0] is None

    return late.split(verified.findings, is_new=is_new, listed=_listed_followups(ctx))


def _raised_record(
    h: str, file: Any, line: Any, category: Any, title: Any, body: Any
) -> ExistingComment:
    file, title, body = str(file or ""), str(title or ""), str(body or "")
    return ExistingComment(
        id="",
        node_id="",
        finding_hash=h,
        dedupe_key=finding_dedupe_key(file, str(category or "general"), title, body),
        root_id=None,
        cf_tokens=carry_forward_tokens(title, body),
        path=file,
        line=line if isinstance(line, int) else None,
        norm_title=normalize_finding_text(title),
        norm_body=normalize_finding_text(body),
        html_url="",
    )


def _raised_records(ctx: RunContext) -> list[ExistingComment]:
    """The findings already raised on the PR, as `find_duplicate` matches against: a finding
    restating one of them is not new."""
    out = [
        _raised_record(h, p.file, p.line_end, p.category, p.title, p.body)
        for h, p in ctx.prior_findings.items()
    ]
    out += [
        _raised_record(h, t.get("path"), t.get("line"), "general", t.get("title"), t.get("body"))
        for h, t in ctx.open_threads.items()
        if h not in ctx.prior_findings
    ]
    return out


def _listed_followups(ctx: RunContext) -> set[str]:
    """The hashes and dedupe keys of every follow-up candidate a review of this PR listed."""
    rows = ctx.conn.execute(
        "SELECT f.hash, f.file, f.category, f.title, f.body FROM findings f "
        "JOIN runs r ON r.id=f.run_id JOIN heads h ON h.id=r.head_id "
        "WHERE h.repo=? AND h.number=? AND h.queue='live' AND f.stage=?",
        (ctx.repo, ctx.number, late.STAGE_LISTED),
    ).fetchall()
    keys = {str(r["hash"]) for r in rows}
    return keys | {
        finding_dedupe_key(r["file"] or "", r["category"] or "general", r["title"], r["body"])
        for r in rows
    }


def _gate_late(
    ctx: RunContext, verified: VerifierOutput, split: late.Split, *, step: StepName, stage: str
) -> None:
    """A later round's new findings on code unchanged since the previous final review: nits
    dropped, suggestions listed once as follow-up candidates (`verified.followups`), blockers
    kept only when the confirmation lane finds a concrete reachable failure (unconfirmed ones
    become follow-up candidates, invalid ones are dropped; a failed lane keeps them as raised).
    The run's final verifier rows (`stage`) are rewritten to what stands, with every finding's
    code status, each gated finding is kept under its `late-*` stage, and the step's counts
    are updated to match."""
    raw_findings, raw_blockers = len(verified.findings), verified.blocker_count
    remembered = _remembered_decisions(ctx, split.candidates)
    ask = [f for f in split.candidates if id(f) not in remembered]
    asked = _confirm_late(ctx, ask) if ask else {}
    unchecked = ask if asked is None else []
    decisions = {**remembered, **(asked or {})}
    confirmed, unconfirmed, invalid = _apply_decisions(split.candidates, decisions)
    listed = _listed_followups(ctx)
    demoted = [
        dataclasses.replace(f, severity="suggestion", extra={**f.extra, "demoted_from": "blocking"})
        for f in unconfirmed
    ]
    repeats = split.repeats + [f for f in demoted if late.is_listed(f, listed)]
    stands = {id(f) for f in (*split.keep, *confirmed, *unchecked)}
    verified.findings = [f for f in verified.findings if id(f) in stands]
    verified.followups = split.followups + [f for f in demoted if not late.is_listed(f, listed)]
    since = ctx.prev_final_sha or ""
    report: dict[str, Any] = {
        "since": since[:8],
        "nits": len(split.nits),
        "followups": len(split.followups),
        "repeats": len(repeats),
        "candidates": len(split.candidates),
        "confirmed": len(confirmed),
        "unconfirmed": len(unconfirmed),
        "invalid": len(invalid),
        "remembered": len(remembered),
        "failed": asked is None,
    }
    if split.changed:
        ctx.late_report = report
    left_out = {
        late.STAGE_NIT: split.nits,
        late.STAGE_FOLLOWUP: split.followups,
        late.STAGE_REPEAT: split.repeats,
        late.STAGE_UNCONFIRMED: unconfirmed,
        late.STAGE_INVALID: invalid,
    }
    with tx(ctx.conn):
        ctx.conn.execute(
            "DELETE FROM findings WHERE run_id=? AND phase='verify2' AND stage=?",
            (ctx.run_id, stage),
        )
        _insert_findings(ctx, "verify2", stage, verified.findings)
        for late_stage, findings in left_out.items():
            _insert_findings(ctx, "verify2", late_stage, findings)
        row = ctx.conn.execute(
            "SELECT detail FROM steps WHERE run_id=? AND name=?", (ctx.run_id, step.value)
        ).fetchone()
        detail = {
            **_json_or_empty(row["detail"] if row else None),
            "blockers": verified.blocker_count,
            "findings": len(verified.findings),
            "raw_blockers": raw_blockers,
            "raw_findings": raw_findings,
        }
        ctx.conn.execute(
            "UPDATE steps SET detail=? WHERE run_id=? AND name=?",
            (json.dumps(detail)[:4000], ctx.run_id, step.value),
        )
        if split.changed:
            event(
                ctx.conn,
                "late.gated",
                repo=ctx.repo,
                number=ctx.number,
                run_id=ctx.run_id,
                detail=json.dumps(report),
            )
    gated = {**left_out, "confirmed": confirmed, "unchecked": unchecked}
    (ctx.run_dir / "late.json").write_text(
        json.dumps(
            {
                **report,
                "since": since,
                "first": ctx.first_reviewed_sha,
                "statuses": {f.hash: f.extra.get("code_status") for f in verified.findings},
                "gated": {k: [_finding_record(f) for f in v] for k, v in gated.items()},
            },
            indent=1,
        )
    )


def _apply_decisions(
    candidates: list[Finding], decisions: dict[int, late.Decision]
) -> tuple[list[Finding], list[Finding], list[Finding]]:
    """(confirmed, unconfirmed, invalid) candidates by their confirmation verdicts (keyed by
    `id(finding)`); a confirmed one's body says so. A candidate without one is in none."""
    confirmed: list[Finding] = []
    unconfirmed: list[Finding] = []
    invalid: list[Finding] = []
    for f in candidates:
        d = decisions.get(id(f))
        if d is None:
            continue
        f.extra["confirmation"] = {"verdict": d.verdict, "failure": d.failure, "reason": d.reason}
        if d.verdict == "blocking":
            f.body += (
                "\n\n_Raised on code unchanged since the last review, and confirmed by an "
                f"independent pass: {d.failure}_"
            )
            confirmed.append(f)
        elif d.verdict == "invalid":
            invalid.append(f)
        else:
            unconfirmed.append(f)
    return confirmed, unconfirmed, invalid


def _remembered_decisions(ctx: RunContext, candidates: list[Finding]) -> dict[int, late.Decision]:
    """Candidates an earlier round's confirmation already judged not blocking (by hash), whose
    file has not changed since that round: its verdict stands, no lane asks again."""
    if not candidates or ctx.code_history is None:
        return {}
    marks = ",".join("?" * len(late.REMEMBERED))
    rows = ctx.conn.execute(
        "SELECT f.hash, f.stage, h.sha FROM findings f JOIN runs r ON r.id=f.run_id "
        "JOIN heads h ON h.id=r.head_id WHERE h.repo=? AND h.number=? AND h.queue='live' "
        f"AND f.run_id<>? AND f.stage IN ({marks}) ORDER BY f.id",
        (ctx.repo, ctx.number, ctx.run_id, *late.REMEMBERED),
    ).fetchall()
    earlier = {str(r["hash"]): (str(r["stage"]), str(r["sha"])) for r in rows}
    out: dict[int, late.Decision] = {}
    for f in candidates:
        stage, sha = earlier.get(f.hash, ("", ""))
        if stage and ctx.code_history.changed_since(sha, f.file) is False:
            out[id(f)] = late.Decision(late.REMEMBERED[stage], reason=f"decided at {sha[:8]}")
    return out


def _confirm_late(ctx: RunContext, candidates: list[Finding]) -> dict[int, late.Decision] | None:
    """One confirmation lane for all new blockers on unchanged code of this round: the final
    verifier's model and effort in a fresh session, shown the claims, the code and the PR
    diff, never the verifier's reasoning. Decisions by `id(finding)`. None when the lane
    failed for good: the blockers then stand as raised, so a failure never costs a real one.
    Its answer is never rewritten by the context-free repair lane (`repairable`)."""
    assert ctx.meta is not None
    ids = late.candidate_ids(len(candidates))
    _step_start(ctx, StepName.CONFIRM, {"candidates": len(candidates)})
    prompt = confirmation_prompt(
        ctx.cfg,
        repo=ctx.repo,
        number=ctx.number,
        head_sha=ctx.sha,
        meta=ctx.meta.as_dict(),
        coverage_from=ctx.coverage_from,
        reported=late.candidate_briefs(candidates),
        guidance=repo_guidance(ctx.worktree) if ctx.cfg.policy.isolates_lane_env else "",
    )

    def validate(raw: dict[str, Any]) -> dict[str, late.Decision]:
        return late.parse_decisions(raw, ids)

    try:
        decisions = _run_checked_lane(
            ctx,
            phase="confirm",
            role="confirm",
            lm=ctx.lane_model(ctx.cfg.policy.phase2_verifier),
            prompt=prompt,
            is_verifier=True,
            check=OutputCheck(validate=validate, kind="confirm", ids=ids, repairable=False),
        )
    except ReviewError as exc:
        ctx.check_cancel()
        _step_end(ctx, StepName.CONFIRM, "failed", {"error": exc.message[:1000]})
        _lane_event(ctx, "late.confirm_failed", f"{len(ids)} blocker(s) kept: {exc.message}")
        return None
    verdicts = [d.verdict for d in decisions.values()]
    _step_end(
        ctx,
        StepName.CONFIRM,
        "ok",
        {"candidates": len(ids), **{v: verdicts.count(v) for v in late.VERDICTS}},
    )
    return {id(f): decisions[cid] for cid, f in zip(ids, candidates, strict=True)}


def _approve_when_clear(ctx: RunContext, verified: VerifierOutput) -> None:
    """Nothing blocks: the verdict is an approval, whatever non-blocking findings come with it
    (they can be fixed forward). `publish.verdict_event` still stops a review that did not run
    at full strength (Phase 2 skipped, degraded mode) at COMMENT."""
    if not _converges(ctx) or verified.blocker_count or verified.review_action == "APPROVE":
        return
    _lane_event(
        ctx,
        "verdict.approve_nonblocking",
        f"verifier said {verified.review_action}; {len(verified.findings)} non-blocking "
        f"finding(s), {len(verified.followups)} follow-up candidate(s)",
    )
    verified.review_action = "APPROVE"


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
TRACKING_ISSUE_MATCH = 0.6  # title overlap at which an open issue already tracks a follow-up
TRACKING_ISSUE_PENDING = "pending"  # kv: `gh issue create` started, its outcome not yet known


def _standing_review(ctx: RunContext) -> dict[str, Any] | None:
    """The bot's FINAL review of this exact commit, or None. A preliminary review (blockers
    found, Phase 2 deferred) does not qualify: a reply that talks a blocker down there must
    re-run the pipeline so Phase 2 and the final verdict still happen."""
    r = github.existing_review_for_sha(
        ctx.gh, ctx.repo, ctx.number, ctx.sha, "final", ctx.cfg.bot_login
    )
    return {**r, "phase": "final"} if r else None


def _fetch_linked_commits(
    ctx: RunContext,
    threads: dict[str, dict[str, Any]],
    *,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """(fetched, unfetched) commits humans linked in the replied threads. Only commits from a
    fork of this repository are fetched (the delta against the mirror is small); anything else
    is reported to the model as unavailable with the reason. Fetched ones land in the shared
    object store, so `git show <sha>` works inside the run's worktree. `should_stop` (the
    policy v12 thread lane) is polled between fetches."""
    fetched: list[dict[str, str]] = []
    unfetched: list[dict[str, str]] = []
    if not ctx.mirror:
        return fetched, unfetched
    own_name = ctx.repo.split("/", 1)[1].lower()
    for i, c in enumerate(converse.linked_commits(threads)):
        if should_stop is not None and should_stop():
            unfetched.append({**c, "reason": "not fetched: the run is stopping"})
        elif i >= LINKED_COMMIT_LIMIT:
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
        kept = _standing_findings(ctx, phase)
        _gate_comment(ctx, "done", phase=phase, blocker_count=kept.blockers)
        return {
            "answered": 0,
            "reason": "no thread awaiting an answer",
            "review_summary": _mark_lifted_in_review(ctx, phase, kept, reviews),
        }
    fetched, unfetched = _fetch_linked_commits(ctx, threads)
    project_skill, review_skill = skill_texts(ctx.cfg, ctx.repo)
    lm = ctx.lane_model(ctx.cfg.policy.conversation_lane)
    before = _standing_findings(ctx, phase)
    settled = before.settled(threads)
    # a deferral from before the tracking rule, of a finding this commit's review does not
    # stand on, is settled: restating it is no new deferral to vet (the lane is told so),
    # and a reply must not say the finding stands when it does not
    settled |= {
        h: "INTENTIONALLY_DEFERRED"
        for h in threads
        if h not in settled and h not in before.findings and _grandfathered(ctx, h)
    }
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
        still_blocking=[
            before.titles.get(h) or h for h, sev in before.findings.items() if sev == "blocking"
        ],
        settled=settled,
    )
    (ctx.run_dir / "conversation.json").write_text(
        json.dumps({h: dataclasses.asdict(o) for h, o in out.outcomes.items()}, indent=1)
    )
    # same guard as step_publish: a push may have landed while the lane ran, and a verdict
    # follow-up must never be posted against an obsolete commit (the run ends superseded and
    # ingest queues the live one)
    _require_live(ctx, github.pr_meta(ctx.gh, ctx.repo, ctx.number))
    out = converse.accept_deferrals(
        out,
        threads,
        settled=set(settled),
        repo=ctx.repo,
        number=ctx.number,
        pr_text=_pr_text(ctx),
    )
    answered: list[dict[str, Any]] = []
    if not ctx.dry_run:
        out = _file_tracking_issues(ctx, out, threads, settled)
        answered = publish.answer_conversation(
            ctx.gh,
            ctx.repo,
            ctx.number,
            ctx.sha,
            threads=threads,
            outcomes=out.outcomes,
            settled=settled,
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
        _record_answered_events(ctx, answered)
        for h in silent:
            kv_set(ctx.conn, _silent_key(ctx, h), str(threads[h].get("latest_reply_id")))
        _record_conceded(ctx, phase, {h: out.outcomes[h] for h in lifted})
    _ledger_write(
        ctx,
        "conversation",
        lambda c: ledger.record_answers(
            c,
            ctx.repo,
            ctx.number,
            run_id=ctx.run_id,
            sha=ctx.sha,
            answered=answered,
            source="conversation",
            conceded=lifted,
        ),
    )
    kept = _standing_findings(ctx, phase, lifted=lifted)
    summary = _mark_lifted_in_review(ctx, phase, kept, reviews)
    update = None
    if not ctx.dry_run:
        update = _conversation_verdict_update(ctx, phase, lm, kept, lifted=lifted)
    _gate_comment(ctx, "done", phase=phase, blocker_count=kept.blockers)
    return {
        "answered": len(posted_ok),
        "outcomes": {h: o.status for h, o in out.outcomes.items()},
        "tracking": {h: o.tracking_url for h, o in out.outcomes.items() if o.tracking_url},
        "linked_commits": {
            "fetched": [c["sha"] for c in fetched],
            "unfetched": [c["sha"] for c in unfetched],
        },
        "blockers_remaining": kept.blockers,
        "verdict_updated": bool(update and update.posted),
        "review_summary": summary,
    }


def _pr_text(ctx: RunContext) -> str:
    """What a reader of the PR sees besides the review threads: its description and the PR
    comments (where a deferral's tracking issue may be linked)."""
    comments = [str(c.get("body") or "") for c in ctx.evidence.get("issue_comments") or []]
    return "\n".join([ctx.meta.body if ctx.meta else "", *comments])


def _file_tracking_issues(
    ctx: RunContext,
    out: converse.ConversationOutput,
    threads: dict[str, dict[str, Any]],
    settled: dict[str, str],
) -> converse.ConversationOutput:
    """File the tracking issue for every deferral a maintainer asked the bot to file. Filed
    (or an open issue that already matches) -> the deferral stands with that link; not
    filable -> DEFERRAL_NEEDS_TRACKING, with a reply asking the maintainer to file it (a
    `settled` thread stays lifted either way, and its reply says nothing else)."""
    outcomes = dict(out.outcomes)
    for h, o in out.outcomes.items():
        req = o.file_request
        if o.status != "INTENTIONALLY_DEFERRED" or req is None:
            continue
        url, filed = _tracking_issue(ctx, h, req, threads[h])
        if url:
            outcomes[h] = dataclasses.replace(
                o, tracking_url=url, tracking_filed=filed, file_request=None
            )
        else:
            outcomes[h] = dataclasses.replace(
                o,
                status=converse.DEFERRAL_NEEDS_TRACKING,
                reply=converse.filing_failed_reply(
                    req.repo, allowed=ctx.cfg.deferral.may_file(req.repo), settled=h in settled
                ),
                file_request=None,
            )
    return converse.ConversationOutput(outcomes=outcomes)


def _tracking_issue(
    ctx: RunContext, h: str, req: converse.FileRequest, thread: dict[str, Any]
) -> tuple[str, bool]:
    """(url, filed now) of the issue tracking finding `h`'s follow-up, or ("", False) when it
    cannot be had. Never files twice: kv holds the answer, and `pending` from just before
    `gh issue create` until its URL is known. A create that may have gone through (a
    timeout, a server error, no URL printed, or a worker that died) leaves `pending`, and
    every later attempt asks the maintainer to check and link it instead; a create GitHub
    refused outright (permissions, validation) clears it. An open issue whose title clearly
    matches is linked instead of filing another; a search that fails files nothing."""
    key = f"defer.issue:{ctx.repo}#{ctx.number}:{h}"
    known = kv_get(ctx.conn, key)
    if known == TRACKING_ISSUE_PENDING:
        _defer_event(ctx, "defer.file_failed", f"{h}: an earlier attempt may have filed it")
        return "", False
    if known:
        return known, False
    if not ctx.cfg.deferral.may_file(req.repo):
        _defer_event(
            ctx, "defer.file_failed", f"{h}: {req.repo} is not in [deferral] file_owners/file_repos"
        )
        return "", False
    try:
        url = _matching_issue(ctx, req)
    except ReviewError as exc:
        _defer_event(ctx, "defer.file_failed", f"{h}: search in {req.repo}: {exc}")
        return "", False
    filed = not url
    if filed:
        with tx(ctx.conn):
            kv_set(ctx.conn, key, TRACKING_ISSUE_PENDING)
        try:
            url = github.create_issue(ctx.gh, req.repo, req.title, _issue_body(ctx, req, thread))
        except ReviewError as exc:
            with tx(ctx.conn):
                if exc.kind is FailKind.CONTRACT:  # refused: nothing was created
                    ctx.conn.execute("DELETE FROM kv WHERE key=?", (key,))
                event(
                    ctx.conn,
                    "defer.file_failed",
                    repo=ctx.repo,
                    number=ctx.number,
                    run_id=ctx.run_id,
                    detail=f"{h}: create in {req.repo}: {exc}",
                )
            return "", False
    with tx(ctx.conn):
        kv_set(ctx.conn, key, url)
        event(
            ctx.conn,
            "defer.issue_filed" if filed else "defer.issue_linked",
            repo=ctx.repo,
            number=ctx.number,
            run_id=ctx.run_id,
            detail=f"{h}: {url}",
        )
    return url, filed


def _issue_body(ctx: RunContext, req: converse.FileRequest, thread: dict[str, Any]) -> str:
    asker = converse.maintainer_reply(thread) or {}
    cid = asker.get("id") or thread.get("comment_id")
    return publish.tracking_issue_body(
        req.body,
        repo=ctx.repo,
        number=ctx.number,
        comment_url=f"https://github.com/{ctx.repo}/pull/{ctx.number}#discussion_r{cid}",
        asker=str(asker.get("author") or "a maintainer"),
        severity=str(thread.get("severity") or "nitpick"),
        title=str(thread.get("title") or ""),
        finding=converse.finding_text(str(thread.get("body") or "")),
    )


def _matching_issue(ctx: RunContext, req: converse.FileRequest) -> str:
    """An open issue in the target repo whose title clearly matches the one to file, or ""."""
    # GitHub ANDs search terms: the three most distinctive words, so a reworded title that
    # still passes the overlap threshold below is found
    terms = sorted(content_tokens(req.title), key=len, reverse=True)[:3]
    if not terms:
        return ""
    rows = github.open_issues_matching(ctx.gh, req.repo, " ".join(terms))
    score, url = max(
        (
            (title_similarity(req.title, str(r.get("title") or "")), str(r.get("url") or ""))
            for r in rows
        ),
        default=(0.0, ""),
    )
    return url if score >= TRACKING_ISSUE_MATCH else ""


def _defer_event(ctx: RunContext, kind: str, detail: str) -> None:
    with tx(ctx.conn):
        event(ctx.conn, kind, repo=ctx.repo, number=ctx.number, run_id=ctx.run_id, detail=detail)


def _record_conceded(
    ctx: RunContext, phase: str, lifted: dict[str, converse.ThreadOutcome]
) -> None:
    """A finding the conversation withdrew or accepted as deferred gets a `conceded` findings
    row for this run (and so this sha), which `_standing_findings` honours on every later run.
    `lifted` maps finding hash -> outcome; the row's body keeps the status, plus a deferral's
    tracking link on a second line. Caller holds the write transaction."""
    if not lifted:
        return
    rows = []
    for h, o in lifted.items():
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
                o.status + (f"\n{o.tracking_url}" if o.tracking_url else ""),
                None,
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
    titles: dict[str, str] = field(default_factory=dict)  # hash -> title, lifted ones too
    lifted: list[publish.LiftedFinding] = field(default_factory=list)

    def settled(self, threads: dict[str, dict[str, Any]]) -> dict[str, str]:
        """thread hash -> the outcome that already lifted its finding on this commit, matched
        the way concessions are (by hash, or by file and title for a re-hashed carry)."""
        by_hash = {f.hash: f.status or "SETTLED" for f in self.lifted}
        by_id = {_identity(f.file, f.title): f.status or "SETTLED" for f in self.lifted}
        out = {}
        for h, t in threads.items():
            st = by_hash.get(h) or by_id.get(_identity(t.get("path"), t.get("title")))
            if st:
                out[h] = st
        return out

    @property
    def blockers(self) -> int:
        return sum(1 for sev in self.findings.values() if sev == "blocking")


def _identity(file: Any, title: Any) -> tuple[str, str]:
    return str(file or ""), str(title or "").strip().lower()


def _standing_findings(ctx: RunContext, phase: str, *, lifted: set[str] | None = None) -> _Standing:
    """Findings on this commit's `phase` publication that still stand.

    The publication is the latest run that posted findings or published the review for this
    sha (a same-sha re-review refreshes it). Its standing set is that run's final verifier set (`verify2` rows, the
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
        "SELECT f.run_id, f.hash, f.severity, f.stage, f.file, f.title, f.body FROM findings f "
        "JOIN runs r ON r.id=f.run_id JOIN heads h ON h.id=r.head_id "
        "WHERE h.repo=? AND h.number=? AND h.sha=? AND f.phase=? AND f.stage IN ('posted','conceded')",
        (ctx.repo, ctx.number, ctx.sha, phase),
    ).fetchall()
    # a full review that carried every finding onto existing threads posts none, so it has
    # no `posted` rows, but it is still this commit's publication (dashpay/dash#7861: missing
    # it left "the verifier record ... is not available" and a COMMENT that never cleared the
    # REQUEST_CHANGES it had just posted). Every run that writes a `reviews` row posted or
    # recorded that review, except a conversation's own verdict follow-up and a CI update
    # (ci.py), neither of which reviewed anything.
    published = ctx.conn.execute(
        "SELECT MAX(rv.run_id) FROM reviews rv WHERE rv.repo=? AND rv.number=? AND rv.sha=? "
        "AND rv.phase=? AND NOT EXISTS (SELECT 1 FROM steps s WHERE s.run_id=rv.run_id "
        "AND s.name IN (?, ?))",
        (
            ctx.repo,
            ctx.number,
            ctx.sha,
            phase,
            StepName.CONVERSE.value,
            StepName.CI_CHECK.value,
        ),
    ).fetchone()[0]
    run_id = max(
        [int(r["run_id"]) for r in rows if r["stage"] == "posted"]
        + ([int(published)] if published is not None else []),
        default=None,
    )
    pub: dict[str, tuple[str, tuple[str, str]]] = {}
    titles: dict[str, str] = {}
    verified = False
    if run_id is not None:
        final = ctx.conn.execute(
            "SELECT hash, severity, stage, file, title FROM findings WHERE run_id=? "
            "AND phase='verify2' AND stage IN ('verified','verified-fresh')",
            (run_id,),
        ).fetchall()
        fresh = [r for r in final if r["stage"] == "verified-fresh"]
        if not fresh and ctx.cfg.policy.pipeline is None and _fresh_kept_nothing(ctx, run_id):
            final = []  # the fresh pass is the final set: never the first pass's findings
        if phase == "final" and final:
            verified = True
            for r in fresh or final:
                pub[str(r["hash"])] = (str(r["severity"]), _identity(r["file"], r["title"]))
                titles[str(r["hash"])] = str(r["title"] or "")
        for r in rows:
            if r["stage"] == "posted" and int(r["run_id"]) == run_id:
                pub.setdefault(
                    str(r["hash"]), (str(r["severity"]), _identity(r["file"], r["title"]))
                )
                titles.setdefault(str(r["hash"]), str(r["title"] or ""))
    ever_posted = {
        str(r["hash"])
        for r in ctx.conn.execute(
            "SELECT hash FROM posted_findings WHERE repo=? AND number=?", (ctx.repo, ctx.number)
        )
    } | {str(r["hash"]) for r in rows}
    for h, t in ctx.open_threads.items():
        if h not in pub and h not in ever_posted:
            pub[h] = (str(t.get("severity") or "nitpick"), _identity(t.get("path"), t.get("title")))
            titles[h] = str(t.get("title") or "")
    # a later full review of this sha re-adjudicated everything: older concessions are void.
    # Each concession row's body is the outcome that lifted it (empty on rows older than that)
    gone: dict[str, tuple[str, str]] = {}
    outcome: dict[tuple[str, str] | str, str] = {}
    tracking: dict[tuple[str, str] | str, str] = {}
    for r in rows:
        if r["stage"] == "conceded" and (run_id is None or int(r["run_id"]) > run_id):
            h, ident = str(r["hash"]), _identity(r["file"], r["title"])
            gone[h] = ident
            status, _, link = str(r["body"] or "").partition("\n")
            outcome[h] = outcome[ident] = status
            # a later answer on a settled thread may not restate the link: keep the one known
            tracking[h] = link or tracking.get(h, "")
            tracking[ident] = link or tracking.get(ident, "")
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
        titles=titles,
        lifted=[
            publish.LiftedFinding(
                hash=h,
                severity=sev,
                file=ident[0],
                title=titles.get(h) or h,
                status=outcome.get(h) or outcome.get(ident) or "",
                tracking=tracking.get(h) or tracking.get(ident) or "",
            )
            for h, (sev, ident) in sorted(pub.items())
            if h in lifted_hashes
        ],
    )


def _fresh_kept_nothing(ctx: RunContext, run_id: int) -> bool:
    """The run's fresh final gate ran and kept no finding (its step's `findings` count is 0)."""
    row = ctx.conn.execute(
        "SELECT detail FROM steps WHERE run_id=? AND name=? AND status='ok'",
        (run_id, StepName.FRESH_VERIFY2.value),
    ).fetchone()
    return row is not None and _json_or_empty(row["detail"]).get("findings") == 0


def _open_blockers(ctx: RunContext, phase: str, *, lifted: set[str] | None = None) -> int:
    """Blocking findings on this commit's `phase` publication that still stand."""
    return _standing_findings(ctx, phase, lifted=lifted).blockers


def _strength_row(ctx: RunContext, run_id: int) -> sqlite3.Row | None:
    """What makes a run's review strong enough to carry an approval: degraded mode, its Phase
    2 and second-pass steps, whether it was a later round, and (policy v12) dropped verifier
    shards and why its confirmation pass did not run."""
    row: sqlite3.Row | None = ctx.conn.execute(
        "SELECT r.degraded, "
        "(SELECT status FROM steps WHERE run_id=r.id AND name=?) AS phase2, "
        "(SELECT status FROM steps WHERE run_id=r.id AND name=?) AS fresh_phase2, "
        "(SELECT status FROM steps WHERE run_id=r.id AND name=?) AS fresh_verify2, "
        "EXISTS (SELECT 1 FROM reviews WHERE repo=? AND number=? AND run_id<r.id) AS later, "
        "EXISTS (SELECT 1 FROM events WHERE run_id=r.id AND kind='verify.shard_dropped') "
        "AS dropped, "
        "EXISTS (SELECT 1 FROM events WHERE run_id=r.id AND kind='threads.conflict') "
        "AS conflict, "
        "(SELECT kind || ':' || detail FROM events WHERE run_id=r.id AND kind IN "
        "('confirmation.skipped', 'confirmation.failed') ORDER BY id DESC LIMIT 1) AS skipped "
        "FROM runs r WHERE r.id=?",
        (
            StepName.PHASE2.value,
            StepName.FRESH_PHASE2.value,
            StepName.FRESH_VERIFY2.value,
            ctx.repo,
            ctx.number,
            run_id,
        ),
    ).fetchone()
    return row


def _mark_lifted_in_review(
    ctx: RunContext, phase: str, standing: _Standing, reviews: list[dict[str, Any]]
) -> str:
    """Bring the body of this commit's standing `phase` review in line with what the
    discussion lifted (`publish.with_lifted_findings`). Runs after every conversation, so a
    body left stale by an earlier run (or an edit that failed) is repaired by the next one.
    Returns what happened; a failed edit never fails the run."""
    if ctx.dry_run:
        return "dry_run"
    marker = f"thepastaclaw-review-phase v1 phase={phase} sha={ctx.sha}"
    ours = [
        r
        for r in reviews
        if (r.get("user") or {}).get("login") == ctx.cfg.bot_login
        and marker in str(r.get("body") or "")
    ]
    # the publication the standing set describes; the newest marked review when the run that
    # published it recorded no review id (adopted an existing one, or a legacy import)
    row = ctx.conn.execute(
        "SELECT github_review_id FROM reviews WHERE run_id=? AND phase=? "
        "AND github_review_id IS NOT NULL ORDER BY id DESC LIMIT 1",
        (standing.run_id, phase),
    ).fetchone()
    review = next(
        (r for r in ours if row and str(r.get("id")) == str(row["github_review_id"])),
        ours[-1] if ours else None,
    )
    if review is None:
        return "no_review"
    body = str(review.get("body") or "")
    new = publish.with_lifted_findings(body, standing.lifted)
    if new == body:
        return "unchanged"
    try:
        github.edit_review(ctx.gh, str(review["node_id"]), new)
    except (ReviewError, ValueError, KeyError) as exc:
        with tx(ctx.conn):
            event(
                ctx.conn,
                "review.summary_update_failed",
                repo=ctx.repo,
                number=ctx.number,
                run_id=ctx.run_id,
                detail=f"review {review.get('id')}: {exc}",
            )
        return f"failed: {exc}"
    return "updated"


def _review_strength(ctx: RunContext, run_id: int | None) -> str:
    """Whether the run that published a commit's standing review is one an APPROVE may rest
    on, by the pipeline's own bar: Phase 2 completed on the primary models (`verdict_event`),
    and on a later review round the fresh Phase-2 audit ran too where it was due
    (`_fresh_was_due`; it is skipped while blockers stand). "ok", "needs_fresh" or "weak"."""
    if run_id is None:
        return "weak"
    row = _strength_row(ctx, run_id)
    if not row or row["degraded"] or row["phase2"] != "ok":
        return "weak"
    if not row["later"] or row["fresh_phase2"] == "ok":
        return "ok"
    return "needs_fresh" if _fresh_was_due(ctx, run_id) else "ok"


def _fresh_was_due(ctx: RunContext, run_id: int) -> bool:
    """Whether an approval resting on run `run_id` (a later round that ran no fresh gate)
    needs it: always under a policy v12 pipeline; under v11 (`_fresh_final_due`) when that run
    had blockers, now lifted, or the full-strength final review before it had (or there was
    none: then it was the first full review, whose second pass did not run)."""
    if ctx.cfg.policy.pipeline is not None:
        return True
    own = ctx.conn.execute("SELECT blocker_count FROM runs WHERE id=?", (run_id,)).fetchone()
    finals = [r for r in _pr_reviews(ctx, before_run=run_id) if r["full"]]
    return bool(own and own["blocker_count"]) or not finals or _had_blockers(finals[-1])


def _fresh_audit_guard(ctx: RunContext) -> str:
    return f"converse.fresh_audit:{ctx.repo}#{ctx.number}:{ctx.sha}"


def _request_fresh_audit(ctx: RunContext) -> bool:
    """Ask for one same-sha re-review once this conversation run ends, so the fresh Phase-2
    audit the standing review skipped can run and approve. Once per commit: a re-review that
    re-raises a finding is answered in its threads like any other. True if requested now."""
    guard = _fresh_audit_guard(ctx)
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
    When no blocker is left of the verifier's set (policy v12: nothing above a nitpick),
    something above a nitpick of it was lifted, and that review is one the pipeline itself
    would approve on (`_review_strength`), the
    follow-up APPROVEs: a COMMENT cannot clear a REQUEST_CHANGES on GitHub, so anything less
    leaves a ready pull request showing a blocking review that no longer exists
    (dashpay/dash#7778). A later-round review that skipped the fresh audit gets that audit
    queued instead. Otherwise, with every blocker gone, a standing REQUEST_CHANGES becomes
    COMMENT. A conversation never adds blockers, and posts nothing when the verdict would not
    move."""
    if standing.blockers or not lifted or not standing.lifted_any:
        # nothing of the publication's own set was lifted (it was clean, or only nitpicks
        # went): the discussion removed nothing the verdict rests on, so it stays
        return None
    state = _live_verdict_state(ctx, phase)
    if state is None:
        return None
    left = sum(1 for sev in standing.findings.values() if sev != "nitpick")
    # policy v11 approves with non-blocking findings standing (`_approve_when_clear`)
    fix_forward = ctx.cfg.policy.pipeline is None
    strength = _review_strength(ctx, standing.run_id)
    clear = standing.verified and standing.lifted_any and (fix_forward or not left)
    # a code check failed after this commit's approval and looked related (ci.py): only a
    # review that sees that failure may approve the commit again, not a discussion
    ci_hold = kv_get(ctx.conn, ci.updated_key(ctx.repo, ctx.number, ctx.sha)) is not None
    prov = _conversation_provenance(ctx, lm)
    verified = _verdict_only(
        phase, "APPROVE" if clear and strength == "ok" and not ci_hold else "COMMENT"
    )
    new_event = publish.verdict_event(verified, prov)
    source = f"the full review of `{ctx.sha[:8]}`" + (
        f" (run {standing.run_id})" if standing.run_id else ""
    )
    if new_event == "APPROVE" and left:
        note = (
            f"Approved: the discussion withdrew or deferred every blocker {source} raised, and "
            f"the approval rests on that review. The {left} non-blocking finding(s) still "
            "standing are optional: fix them forward."
        )
    elif new_event == "APPROVE":
        note = (
            f"Approved: the discussion withdrew or deferred every finding {source} raised, "
            "and the approval rests on that review."
        )
    elif left and not fix_forward:
        note = f"Not approved: {left} non-blocking finding(s) from {source} still stand."
    elif prov.degraded:
        note = "Not approved: this follow-up ran on a stand-in model (degraded mode)."
    elif clear and ci_hold:
        note = (
            "Not approved: CI failed on this commit after its review, in checks that looked "
            "related (see the CI follow-up); the next push gets a full review."
        )
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
    else:
        note = (
            "Not approved: the standing review of this commit did not run at full strength "
            "(Phase 2 or the primary models were missing), so it cannot carry an approval."
        )
    titles = [
        t["title"]
        for h, t in ctx.open_threads.items()
        if h in lifted and t.get("severity") == "blocking" and t.get("title")
    ]
    return _post_conversation_update(
        ctx, phase, verified, prov, state=state, new_event=new_event, titles=titles, note=note
    )


def _live_verdict_state(ctx: RunContext, phase: str) -> str | None:
    """The bot's standing verdict on this commit's `phase` review when a conversation may move
    it, else None (approved already, or dismissed: a human overrode us; do not re-assert). The
    lane ran for minutes: a push or a dismissal since then must not get an approval, so the
    head is checked live first."""
    _require_live(ctx, github.pr_meta(ctx.gh, ctx.repo, ctx.number))
    reviews = github.reviews(ctx.gh, ctx.repo, ctx.number)
    state = publish.standing_verdict(reviews, ctx.sha, phase, ctx.cfg.bot_login)
    return state if state in {"CHANGES_REQUESTED", "COMMENTED"} else None


def _conversation_provenance(ctx: RunContext, lm: LaneModel) -> publish.Provenance:
    return publish.Provenance(
        reviewers=[],
        verifier=_lane_provenance(lm, "conversation"),
        policy_fingerprint=ctx.cfg.policy.fingerprint,
        adhoc=ctx.adhoc,
        conversation=True,
        degraded=_degraded_provenance(ctx),
    )


def _verdict_only(phase: str, action: str) -> VerifierOutput:
    """A verdict with nothing else: what a conversation's follow-up review stands on."""
    return VerifierOutput(
        summary="",
        review_action=action,
        findings=[],
        dropped=[],
        out_of_scope=[],
        coderabbit_reactions=[],
        prerequisite_adjudications=[],
        adjudication_complete=True,
        review_phase=phase,
        raw={},
    )


def _post_conversation_update(
    ctx: RunContext,
    phase: str,
    verified: VerifierOutput,
    prov: publish.Provenance,
    *,
    state: str,
    new_event: str,
    titles: list[str],
    note: str,
) -> publish.PublishResult | None:
    """Post a conversation's verdict follow-up on the commit and record it; nothing when
    GitHub would record the state it already shows (then only our own record moves)."""
    assert ctx.meta
    own = ctx.meta.author.lower() == ctx.cfg.bot_login.lower()
    if publish.EVENT_STATE[publish.transport_event(new_event, own_pr=own)] == state:
        approved = new_event == "APPROVE" and (
            labels.latest_verdict(ctx.conn, ctx.repo, ctx.number, ctx.sha) != "APPROVE"
        )
        _record_verdict(ctx, phase, verified, new_event)
        if approved:  # a bot-authored PR: the APPROVE exists only in our record
            _watch_ci(ctx, known=[])
        return None
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
    if result.event == "APPROVE":
        # nothing weighed CI for this approval: the discussion only lifted findings
        _watch_ci(ctx, known=[])
    return result


# ---- policy v12: confirmation pass, publishing from the ledger, same-sha thread lane ----


def _confirmation_due(ctx: RunContext) -> bool:
    """Whether the policy asks for a confirmation pass on this round (v12 design §2.5):
    "no_blockers" on every round, "rereview" on re-reviews only (today's fresh-final rounds),
    "off" never, and never on an audit."""
    trigger = ctx.cfg.policy.confirmation_trigger
    if trigger == "off" or ctx.is_audit:
        return False
    return trigger == "no_blockers" or ctx.has_prior_review


def _confirm12(ctx: RunContext, verified: VerifierOutput, effort: TierEffort) -> VerifierOutput:
    """The confirmation pass (replaces fresh-final under the v12 verifier): when no blocker
    stands after the code rules on a full-strength run, the confirmation roles (general and
    security-auditor in policy v12) review the whole change once more, unanchored, and only
    their candidates are verified, against everything already verified this round. Skipped
    while blockers stand, in degraded mode (which never approves) or when verification was
    incomplete. Recorded on the run (`ctx.confirmation`); an approval needs it when the policy
    asks for it. Its lanes failing for good costs the approval, never the review; a blocker
    its verification could not check fails the run, as in the main pass."""
    if not _confirmation_due(ctx):
        return verified
    pol = ctx.cfg.policy
    assert pol.pipeline is not None
    wanted = pol.pipeline.confirmation.roles
    conf = ctx.confirmation = _Confirmation(
        roles=[r for r in _phase_roles(ctx, "phase2") if wanted is None or r in wanted]
    )
    conf.skipped = _confirmation_skip(ctx, verified, effort, conf.roles)
    if conf.skipped:
        _lane_event(ctx, "confirmation.skipped", conf.skipped)
        return verified
    step = StepName.FRESH_PHASE2
    try:
        outs = _reviewer_step(
            ctx,
            step=step,
            phase="phase2",
            lm=ctx.lane_model(_with_effort(pol.phase2_reviewer, effort.phase2)),
            expected_phase="final",
            fresh=True,
            roles=conf.roles,
        )
        step = StepName.FRESH_VERIFY2
        _step_start(ctx, step)
        delta = _verify12(
            ctx,
            phase="verify2",
            lm=ctx.lane_model(pol.phase2_verifier),
            expected_phase="final",
            candidates=_candidates_of(outs),
            note=(
                "These candidates (`cf-*`) come from a second, independent pass over the same "
                "head, run because the first found no blocker. The ledger's open list includes "
                "the issues this round already raised (status `raised this round`): a candidate "
                "restating one of them matches it as `open:<hash>`."
            ),
            raised_now=[*verified.findings, *verified.noted],
            stage="verified-fresh",
        )
        _step_end(ctx, step, "ok", {"blockers": delta.blocker_count, "new": len(delta.findings)})
    except UncheckedBlockers:
        raise
    except ReviewError as exc:
        ctx.check_cancel()
        _step_end(ctx, step, "failed", {"error": exc.message[:1000]})
        conf.skipped = f"failed: {exc.message[:200]}"
        _lane_event(ctx, "confirmation.failed", exc.message)
        return verified
    conf.ran, conf.new = True, len(delta.findings)
    _merge_confirmation(verified, delta)
    return verified


def _confirmation_skip(
    ctx: RunContext, verified: VerifierOutput, effort: TierEffort, roles: list[str]
) -> str | None:
    """Why the confirmation pass cannot run on this round, or None when it can."""
    pol = ctx.cfg.policy
    if verified.blocker_count:
        return "blockers stand"
    if ctx.is_degraded:
        return "degraded mode"
    if not pol.phase2_enabled or effort.phase2 is None:
        return "no Phase 2"
    if ctx.verify_incomplete:
        return "verification incomplete"
    if not roles:
        return "no confirmation role selected"
    return None


def _merge_confirmation(verified: VerifierOutput, delta: VerifierOutput) -> None:
    """The confirmation pass's verified delta, added to the round's verified set."""
    verified.findings += delta.findings
    verified.noted += delta.noted
    verified.dropped += delta.dropped
    have = {x["comment_id"] for x in verified.coderabbit_reactions}
    verified.coderabbit_reactions += [
        x for x in delta.coderabbit_reactions if x["comment_id"] not in have
    ]
    r, d = verified.rules, delta.rules
    if r is None or d is None:
        return
    for h, ms in d.corroborated.items():
        r.corroborated.setdefault(h, []).extend(ms)
    for h, v in d.refound.items():
        if r.refound.get(h) != "CONFIRMED":
            r.refound[h] = v
    for h, f in d.upgrades.items():
        r.upgrades.setdefault(h, f)


def _verdict12(
    ctx: RunContext, phase: str, verified: VerifierOutput, phase2_skipped: str | None
) -> str:
    """The round's verdict from the ledger (v12 design §3): any standing blocker requests
    changes; APPROVE needs nothing above a nitpick standing, a final review on the full
    pipeline (Phase 2 ran, primary models, every candidate verified) and the confirmation pass
    when the policy asks for one; otherwise COMMENT. The model's own `review_action` is never
    asked for."""
    blockers = verified.blocker_count
    suggestions = sum(
        1 for f in (*verified.findings, *verified.carried) if f.severity == "suggestion"
    )
    full = (
        phase == "final"
        and not phase2_skipped
        and not ctx.is_degraded
        and not ctx.verify_incomplete
        and not ctx.thread_failed  # someone may be waiting on a thread nobody read
        and not ctx.thread_conflict  # a concession and a re-find disagree; a human decides
        and not ctx.is_audit
    )
    confirmed = ctx.confirmation is None or ctx.confirmation.ran
    return rules.verdict(
        blockers=blockers, suggestions=suggestions, full_strength=full, confirmed=confirmed
    )


def _record_thread_answers(
    ctx: RunContext,
    answered: list[dict[str, Any]],
    outs: dict[str, converse.ThreadOutcome],
    *,
    source: str,
    phase: str,
) -> set[str]:
    """Book what the thread lane's outcomes did: `thread.answered` events, silence memory for a
    reply met with NO_REPLY, `conceded` rows for what a delivered outcome withdrew or deferred,
    the ledger outcomes (an outcome moves its issue only once it is on the thread, or the issue
    has none), the affirmed issues' `last_seen`, and the reconcile's pending list. Returns the
    hashes the delivered outcomes closed."""
    delivered = {
        str(a["finding_hash"]): str(a["status"])
        for a in answered
        if a.get("action") in ledger.DELIVERED_ACTIONS
    }
    considered = {str(a["finding_hash"]) for a in answered}
    lifted = {h for h, st in delivered.items() if ledger.OUTCOME_STATUS.get(st) in ledger.CLOSED}
    conceded = {h for h in lifted if delivered[h] in converse.LIFTING_STATUSES}
    silent = {
        h
        for h, o in outs.items()
        if not o.reply
        and h in considered
        and (ctx.open_threads.get(h) or {}).get("awaiting_answer")
    }
    affirmed = {h for h, o in outs.items() if o.status in {"STILL_VALID", "FIX_PENDING"}}
    with tx(ctx.conn):
        _record_answered_events(ctx, answered)
        for h in silent:
            kv_set(ctx.conn, _silent_key(ctx, h), str(ctx.open_threads[h].get("latest_reply_id")))
        _record_conceded(ctx, phase, {h: outs[h] for h in conceded})

    def record(c: sqlite3.Connection) -> None:
        ledger.record_answers(
            c,
            ctx.repo,
            ctx.number,
            run_id=ctx.run_id,
            sha=ctx.sha,
            answered=answered,
            source=source,
            conceded=conceded,
        )
        ledger.record_seen(
            c, ctx.repo, ctx.number, run_id=ctx.run_id, sha=ctx.sha, hashes=affirmed, source=source
        )
        ledger.settle_pending(c, ctx.repo, ctx.number, outs)

    _ledger_write(ctx, "thread outcomes", record)
    return lifted


def _apply_thread_outcomes(ctx: RunContext, verified: VerifierOutput, phase: str) -> set[str]:
    """Deliver the new-sha thread lane's outcomes on their threads, a FIXED / OUTDATED of an
    issue a finder just restated suppressed (it stays open, unanswered), and book them."""
    outs = _thread_outcomes(ctx)
    if ctx.dry_run:
        # nothing goes on a thread, nothing is booked; a replay still decides on what the
        # outcomes would lift (the titles of a same-sha verdict follow-up)
        return _projected_lifted(outs, _refound(verified)) if ctx.replay else set()
    corroborated = _refound(verified)
    answered: list[dict[str, Any]] = []
    deliver: dict[str, converse.ThreadOutcome] = {}
    for h, o in outs.items():
        if o.status in {"FIXED", "OUTDATED"} and h in corroborated:
            answered.append(
                {
                    "finding_hash": h,
                    "status": o.status,
                    "action": "suppressed_refound",
                    "mode": "thread",
                }
            )
        else:
            deliver[h] = o
    if deliver or corroborated:
        answered += publish.answer_thread_lane(
            ctx.gh,
            ctx.repo,
            ctx.number,
            ctx.sha,
            outcomes=deliver,
            threads=ctx.open_threads,
            same_sha=False,
            keep=corroborated,
            settled=_ledger_deferred(ctx, deliver),
        )
    if not outs and not answered:
        return set()
    return _record_thread_answers(ctx, answered, outs, source="thread", phase=phase)


def _resolve_conflicts(ctx: RunContext, verified: VerifierOutput) -> None:
    """A maintainer-driven concession (WITHDRAWN, or a deferral a maintainer made) wins over a
    finder restating the same issue this round: the finders never saw the reply. The issue is
    not raised in place, the disagreement is recorded (`threads.conflict`, for the eval), and
    the round cannot approve. A FIXED / OUTDATED outcome a finder contradicts is the other way
    round (the issue stays open: `_apply_thread_outcomes`)."""
    outcome = _outcome(verified)
    outs = _thread_outcomes(ctx)
    conflicts = {
        h: {"finding_hash": h, "outcome": outs[h].status, "verdict": outcome.refound.get(h, "")}
        for h in outcome.corroborated
        if h in outs and outs[h].status in converse.LIFTING_STATUSES
    }
    for h in conflicts:
        outcome.upgrades.pop(h, None)
    _record_conflicts(ctx, list(conflicts.values()))


def _record_conflicts(ctx: RunContext, conflicts: list[dict[str, Any]]) -> None:
    """Concessions this round that a finder contradicted (`threads.conflict`, for the eval):
    the round cannot approve."""
    if not conflicts:
        return
    ctx.thread_conflict = True
    with tx(ctx.conn):
        for c in conflicts:
            event(
                ctx.conn,
                "threads.conflict",
                repo=ctx.repo,
                number=ctx.number,
                run_id=ctx.run_id,
                detail=json.dumps(c),
            )


def _drop_settled_refolds(ctx: RunContext, verified: VerifierOutput) -> None:
    """A new issue that publish-time dedupe would fold onto a settled issue's thread (the
    verifier saw a new title; dedupe sees the same lines and a similar title) is that issue
    raised again, so it gets the rule `rules.apply_rules` gives a `settled:` match: dropped
    unless the code at it changed since it was settled (`_closed_changed`). One conceded on its
    thread this very round is settled too, and the re-find is a conflict (H1: the concession
    wins). Runs before the verdict, so a dropped one blocks nothing."""
    if not verified.findings:
        return
    records = [
        ExistingComment.from_github(c)
        for c in github.inline_comments(ctx.gh, ctx.repo, ctx.number)
        if (c.get("user") or {}).get("login") == ctx.cfg.bot_login
    ]
    if not records:
        return
    fresh = _fresh_threads(ctx)
    outs = _thread_outcomes(ctx)
    by_comment = {str(i.comment_id): i for i in ctx.issues.values() if i.comment_id}
    keep: list[Finding] = []
    conflicts: list[dict[str, Any]] = []
    for f in verified.findings:
        candidates = records if fresh is None else [r for r in records if not fresh(f, r)]
        match, _ = find_duplicate(f, candidates)
        issue = (ctx.issues.get(match.finding_hash) or by_comment.get(match.id)) if match else None
        o = outs.get(issue.hash) if issue is not None else None
        conceded = o is not None and o.status in converse.LIFTING_STATUSES
        settled = issue is not None and issue.status in rules.MATCH_SETTLED
        if issue is None or not (conceded or settled):
            keep.append(f)
            continue
        if conceded:
            conflicts.append(
                {
                    "finding_hash": issue.hash,
                    "outcome": outs[issue.hash].status,
                    "verdict": str(f.extra.get("verdict") or ""),
                }
            )
        elif _closed_changed(ctx, issue) is True:
            keep.append(f)  # revived: publishing folds it onto its own thread, re-opening it
            continue
        verified.dropped.append(
            {
                "original_title": f.title,
                "members": f.extra.get("members") or [],
                "reason": f"SETTLED: restates {issue.status if settled else 'conceded'} issue "
                f"{issue.hash} on its thread; the code at it has not changed since",
                "follow_up": False,
            }
        )
        _lane_event(ctx, "v12.settled_refold", f"{f.title[:200]} -> {issue.hash}")
    verified.findings = keep
    _record_conflicts(ctx, conflicts)


def _apply_upgrades(ctx: RunContext, verified: VerifierOutput) -> None:
    """An open non-blocking issue a candidate restated and the verifier CONFIRMED as blocking
    is raised in place: the ledger's severity, and a note on its thread (never a new thread)."""
    outcome = _outcome(verified)
    if not outcome.upgrades or ctx.dry_run:
        return
    for h, f in outcome.upgrades.items():
        t = ctx.open_threads.get(h)
        if t and t.get("comment_id"):
            note = publish.scrub_reply(
                f"Raising this to **blocking** at `{ctx.sha[:8]}`: {f.extra.get('evidence') or f.body}",
                2500,
            )
            if note:
                try:
                    github.post_reply(ctx.gh, ctx.repo, ctx.number, int(t["comment_id"]), note)
                except (ReviewError, ValueError) as exc:
                    log.warning("severity note on %s failed: %s", h, exc)

    def record(c: sqlite3.Connection) -> None:
        for h, f in outcome.upgrades.items():
            ledger.raise_severity(
                c,
                ctx.repo,
                ctx.number,
                h,
                "blocking",
                run_id=ctx.run_id,
                sha=ctx.sha,
                source="verifier",
                reason=str(f.extra.get("evidence") or ""),
            )

    _ledger_write(ctx, "severity", record)


def _ledger_after(ctx: RunContext, verified: VerifierOutput) -> dict[str, ledger.Issue]:
    """The ledger after this round's outcomes: re-read once they are recorded; projected in a
    dry run, where nothing is."""
    if not ctx.dry_run:
        return ledger.load(ctx.conn, ctx.repo, ctx.number)
    corroborated = _refound(verified)
    upgrades = _outcome(verified).upgrades
    return rules.project(
        ctx.issues, _pending_statuses(_thread_outcomes(ctx), corroborated), upgrades
    )


def _publish12(
    ctx: RunContext,
    *,
    phase: str,
    verified: VerifierOutput,
    verifier_lm: LaneModel,
    phase2_skipped: str | None,
) -> publish.PublishResult:
    """Publish a policy v12 round from the ledger: the thread lane's outcomes go on their
    threads first, then the verdict is computed over what the ledger now holds plus this
    round's new issues (`_verdict12`), and the review posts the new issues inline and lists
    the carried ones under "Still open from earlier rounds" (never re-posted)."""
    _require_live_base(ctx)
    before = dict(ctx.issues)
    _drop_settled_refolds(ctx, verified)
    _apply_budget(ctx, verified)
    _resolve_conflicts(ctx, verified)
    lifted = _apply_thread_outcomes(ctx, verified, phase)
    _apply_upgrades(ctx, verified)
    verified.carried = _carried12(ctx, _ledger_after(ctx, verified))
    verified.review_action = _verdict12(ctx, phase, verified, phase2_skipped)
    corroborated = _refound(verified)
    if corroborated:
        _ledger_write(
            ctx,
            "corroborated",
            lambda c: ledger.record_seen(
                c,
                ctx.repo,
                ctx.number,
                run_id=ctx.run_id,
                sha=ctx.sha,
                hashes=corroborated,
                source="verifier",
            ),
        )
    existing, phase = _existing_review(ctx, phase)
    if existing:
        # a same-sha re-review: as v11, never a second full review for the commit; a verdict
        # that moved gets a follow-up
        _backfill_posted(ctx, phase, existing, verified)
        update = _verdict_update(
            ctx,
            phase,
            verified,
            verifier_lm,
            phase2_skipped=phase2_skipped,
            lifted_titles=_lifted_blocker_titles(before, lifted),
        )
        if update is not None:
            return update
    model, result = _publish_review(
        ctx,
        phase=phase,
        verified=verified,
        verifier_lm=verifier_lm,
        phase2_skipped=phase2_skipped,
        rereview=bool(existing),
    )
    _record_folds(ctx, model)
    _record_revivals(ctx, verified)
    return result


def _apply_budget(ctx: RunContext, verified: VerifierOutput) -> None:
    """The comment budget (`rules.within_budget`): what is over it is dropped, and the summary
    says how many."""
    verified.findings, over = rules.within_budget(verified.findings, ctx.cfg.comment_budget)
    if not over:
        return
    verified.dropped += [
        {"original_title": f.title, "reason": "BUDGET: over the comment budget"} for f in over
    ]
    verified.summary += (
        f"\n\n_{len(over)} more non-blocking finding(s) over the comment budget of "
        f"{ctx.cfg.comment_budget} are not posted on this round._"
    )


def _outcome(v: VerifierOutput) -> rules.Outcome:
    """The code rules' account of a v12 verifier answer (an empty one for any other)."""
    return v.rules if isinstance(v.rules, rules.Outcome) else rules.Outcome()


def _blockers(issues: Iterable[ledger.Issue]) -> int:
    return sum(1 for i in issues if i.severity == "blocking")


def _refound(verified: VerifierOutput) -> set[str]:
    """The ledger issues a candidate restated this round."""
    return set(_outcome(verified).corroborated)


def _lifted_blocker_titles(before: dict[str, ledger.Issue], lifted: set[str]) -> list[str]:
    return [before[h].title for h in lifted if h in before and before[h].severity == "blocking"]


def _record_folds(ctx: RunContext, model: publish.ReviewModel) -> None:
    """Every new issue publishing folded onto an existing finding comment becomes that
    comment's ledger issue (`ledger.attach_folded`): affirmed, re-opened, or recorded, so it
    is listed, judged and withdrawable instead of blocking from nowhere."""
    folds = [s for s in model.suppressed if s.comment_id]
    if not folds:
        return

    def reopens(i: ledger.Issue) -> bool:
        # a settled issue only comes back on code that changed since (as `settled:` does)
        return i.status not in rules.MATCH_SETTLED or _closed_changed(ctx, i) is True

    def record(c: sqlite3.Connection) -> None:
        for s in folds:
            ledger.attach_folded(
                c,
                ctx.repo,
                ctx.number,
                run_id=ctx.run_id,
                sha=ctx.sha,
                finding=s.finding,
                comment_hash=s.comment_hash,
                comment_id=s.comment_id,
                reopens=reopens,
            )

    _ledger_write(ctx, "folds", record)


def _record_revivals(ctx: RunContext, verified: VerifierOutput) -> None:
    """A settled issue raised again on changed code under its own title keeps its identity:
    publishing dedupes it onto its old thread, so the ledger re-opens it explicitly (under a
    new title it is a new issue, and the old one stays settled)."""
    revived = sorted(
        {
            str(f.extra["revives"])
            for f in verified.findings
            if f.extra.get("revives") and f.hash == f.extra["revives"]
        }
    )
    if not revived:
        return

    def record(c: sqlite3.Connection) -> None:
        for h in revived:
            ledger.transition(
                c,
                ctx.repo,
                ctx.number,
                h,
                ledger.OPEN,
                run_id=ctx.run_id,
                sha=ctx.sha,
                source="verifier",
                reason="raised again: the code at it changed since it was settled",
            )

    _ledger_write(ctx, "revival", record)


def _ledger_blockers(ctx: RunContext) -> int:
    stand = _stand12(ctx, ledger.load(ctx.conn, ctx.repo, ctx.number))
    return _blockers(stand)


def step_threads(ctx: RunContext, standing: dict[str, Any]) -> dict[str, Any]:
    """A reply on a reviewed commit under policy v12: the thread lane in same-sha mode over
    the threads awaiting an answer (FIXED there is FIX_PENDING), its outcomes delivered and
    recorded in the ledger, then the verdict restated over the ledger (#38,
    `_conversation_verdict12`). A conversation never adds blockers."""
    assert ctx.meta and ctx.worktree
    phase = str(standing["phase"])
    threads = _threads_to_answer(ctx)
    reviews = github.reviews(ctx.gh, ctx.repo, ctx.number)
    if not threads:
        # nothing to say; a review body an earlier run left stale is repaired (as under v11)
        _gate_comment(ctx, "done", phase=phase, blocker_count=_ledger_blockers(ctx))
        return {
            "answered": 0,
            "reason": "no thread awaiting an answer",
            "review_summary": _mark_lifted_in_review(
                ctx, phase, _lifted_standing12(ctx, phase), reviews
            ),
        }
    out, lm = _run_thread_lane(
        ctx,
        {h: ["reply"] for h in threads},
        same_sha=True,
        standing_verdict=publish.standing_verdict(reviews, ctx.sha, phase, ctx.cfg.bot_login),
    )
    _require_live(ctx, github.pr_meta(ctx.gh, ctx.repo, ctx.number))
    before = ledger.load(ctx.conn, ctx.repo, ctx.number)
    answered: list[dict[str, Any]] = []
    if not ctx.dry_run:
        answered = publish.answer_thread_lane(
            ctx.gh,
            ctx.repo,
            ctx.number,
            ctx.sha,
            outcomes=out.outcomes,
            threads=ctx.open_threads,
            same_sha=True,
            settled=_ledger_deferred(ctx, out.outcomes),
        )
    lifted = _record_thread_answers(ctx, answered, out.outcomes, source="conversation", phase=phase)
    # the standing review's body says what the discussion lifted, as under v11 (its counts
    # and agent prompt would otherwise go on listing a withdrawn blocker)
    summary = _mark_lifted_in_review(ctx, phase, _lifted_standing12(ctx, phase), reviews)
    issues = (
        ledger.load(ctx.conn, ctx.repo, ctx.number)
        if not ctx.dry_run
        else rules.project(before, _pending_statuses(out.outcomes, set()))
    )
    stand = _stand12(ctx, issues, out.outcomes)
    blockers = _blockers(stand)
    update = None
    replayed: dict[str, Any] = {}
    if not ctx.dry_run:
        update = _conversation_verdict12(ctx, phase, lm, stand, lifted=lifted, before=before)
    elif ctx.replay:
        # a replay delivers nothing (its gh is read-only), so it decides on what the outcomes
        # would lift and records the follow-up it would post (the eval's S8 reads it)
        lifted = _projected_lifted(out.outcomes, set())
        replayed["would_post"] = _would_post12(ctx, phase, lm, stand, lifted=lifted, before=before)
        path = ctx.run_dir / "conversation.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), **replayed}, indent=1))
    _gate_comment(ctx, "done", phase=phase, blocker_count=blockers)
    return {
        "answered": sum(1 for a in answered if a.get("action") == "replied"),
        "outcomes": {h: o.status for h, o in out.outcomes.items()},
        "blockers_remaining": blockers,
        "verdict_updated": bool(update and update.posted),
        "review_summary": summary,
        **replayed,
    }


# ledger statuses a discussion lifts an issue to -> the outcome the review body names
_LIFTED12 = {"withdrawn": "WITHDRAWN", "deferred": "INTENTIONALLY_DEFERRED"}


def _lifted_standing12(ctx: RunContext, phase: str) -> _Standing:
    """What the discussion lifted on this commit, from the ledger, for `_mark_lifted_in_review`:
    every issue withdrawn or deferred on it, those its review raised and those it carried from
    earlier rounds alike (`publish.with_lifted_findings` marks only what the review lists),
    with a deferral's tracking link from its concession row."""
    links: dict[str, str] = {}
    for r in ctx.conn.execute(
        "SELECT f.hash, f.body FROM findings f JOIN runs r ON r.id=f.run_id "
        "JOIN heads h ON h.id=r.head_id WHERE h.repo=? AND h.number=? AND h.sha=? "
        "AND f.stage='conceded' ORDER BY f.id",
        (ctx.repo, ctx.number, ctx.sha),
    ):
        link = str(r["body"] or "").partition("\n")[2]
        if link:
            links[str(r["hash"])] = link
    lifted = [
        publish.LiftedFinding(
            hash=h,
            severity=i.severity,
            file=i.file,
            title=i.title,
            status=_LIFTED12[i.status],
            tracking=links.get(h, ""),
        )
        for h, i in sorted(ledger.load(ctx.conn, ctx.repo, ctx.number).items())
        if i.status in _LIFTED12 and i.status_sha == ctx.sha
    ]
    return _Standing(
        findings={},
        run_id=_standing_run12(ctx, phase),
        verified=True,
        lifted_any=bool(lifted),
        lifted=lifted,
    )


def _standing_run12(ctx: RunContext, phase: str) -> int | None:
    """The full-pipeline run whose `phase` review of this commit stands: the latest one that
    published it (a conversation's own follow-ups do not count)."""
    row = ctx.conn.execute(
        "SELECT run_id FROM reviews WHERE repo=? AND number=? AND sha=? AND phase=? "
        "AND run_id IN (SELECT run_id FROM steps WHERE name=? AND status='ok') "
        "ORDER BY posted_at DESC, rowid DESC LIMIT 1",
        (ctx.repo, ctx.number, ctx.sha, phase, StepName.PUBLISH.value),
    ).fetchone()
    return int(row["run_id"]) if row and row["run_id"] is not None else None


# why a run's confirmation pass did not run, as the #38 note says it (`_confirm12` reasons)
_SKIP_WORDS = {"blockers stand": "it had found a blocker"}


def _review_strength12(ctx: RunContext, run_id: int | None) -> tuple[str, str]:
    """`_review_strength` under policy v12, with why: ("ok", ""), ("weak", reason) when the
    run cannot carry an approval (Phase 2 missing, stand-in models, candidates it dropped
    unverified, or a concession its verifier contradicted, whatever the confirmation trigger), or ("needs_fresh", reason) when the policy
    asked that run for a confirmation pass and it did not run, `reason` being why it did not."""
    if run_id is None:
        return "weak", "no full review of this commit is on record"
    row = _strength_row(ctx, run_id)
    if not row or row["degraded"] or row["phase2"] != "ok":
        return "weak", "Phase 2 or the primary models were missing"
    if row["dropped"]:
        return "weak", "some of its candidates were never verified"
    if row["conflict"]:
        # that run refused to approve over it (`_verdict12`); a reply must not approve either
        return "weak", "a finding the discussion withdrew was confirmed again by the verifier"
    trigger = ctx.cfg.policy.confirmation_trigger
    required = trigger == "no_blockers" or (trigger == "rereview" and bool(row["later"]))
    if not required or row["fresh_verify2"] == "ok":
        return "ok", ""
    kind, _, reason = str(row["skipped"] or "").partition(":")
    if kind == "confirmation.failed":
        why = "its lanes failed"  # the error itself is never published
    else:
        why = _SKIP_WORDS.get(reason, reason or "it did not run")
    return "needs_fresh", why


@dataclass(slots=True)
class _Verdict12:
    """What a v12 conversation's #38 follow-up would say (`_decide_conversation12`)."""

    state: str  # the bot's standing verdict it moves from
    event: str  # the canonical event (`publish.verdict_event`)
    verified: VerifierOutput
    prov: publish.Provenance
    titles: list[str]
    note: str


def _conversation_verdict12(
    ctx: RunContext,
    phase: str,
    lm: LaneModel,
    stand: list[ledger.Issue],
    *,
    lifted: set[str],
    before: dict[str, ledger.Issue],
) -> publish.PublishResult | None:
    """#38 restated over the ledger (`_decide_conversation12`), posted: nothing is when the
    verdict would not move, or a human overrode it."""
    d = _decide_conversation12(
        ctx, phase, lm, stand, lifted=lifted, before=before, queue=lambda: _request_fresh_audit(ctx)
    )
    if d is None:
        return None
    return _post_conversation_update(
        ctx,
        phase,
        d.verified,
        d.prov,
        state=d.state,
        new_event=d.event,
        titles=d.titles,
        note=d.note,
    )


def _would_post12(
    ctx: RunContext,
    phase: str,
    lm: LaneModel,
    stand: list[ledger.Issue],
    *,
    lifted: set[str],
    before: dict[str, ledger.Issue],
) -> dict[str, Any] | None:
    """A replay's account of the #38 follow-up a live run would post (it posts nothing,
    queues nothing and records no verdict): its event, note and the lifted set it stands on;
    `posts` is False when GitHub would record the state it already shows. None: no follow-up."""
    guard = _fresh_audit_guard(ctx)
    d = _decide_conversation12(
        ctx,
        phase,
        lm,
        stand,
        lifted=lifted,
        before=before,
        queue=lambda: not kv_get(ctx.conn, guard),
    )
    if d is None:
        return None
    assert ctx.meta
    own = ctx.meta.author.lower() == ctx.cfg.bot_login.lower()
    return {
        "event": d.event,
        "note": d.note,
        "lifted": sorted(lifted),
        "titles": d.titles,
        "from": d.state,
        "posts": publish.EVENT_STATE[publish.transport_event(d.event, own_pr=own)] != d.state,
    }


def _decide_conversation12(
    ctx: RunContext,
    phase: str,
    lm: LaneModel,
    stand: list[ledger.Issue],
    *,
    lifted: set[str],
    before: dict[str, ledger.Issue],
    queue: Callable[[], bool],
) -> _Verdict12 | None:
    """#38 restated over the ledger: after a same-sha thread pass lifted something above a
    nitpick, with no blocker left standing, APPROVE when nothing above a nitpick stands and
    the standing review ran at full strength with its confirmation pass; COMMENT otherwise,
    queuing one same-sha re-review (`queue`: True if queued now) when only the confirmation
    pass is missing. None when there is nothing to say, or a human overrode the verdict."""
    lifted_any = any(h in before and before[h].severity != "nitpick" for h in lifted)
    if _blockers(stand) or not lifted_any:
        return None
    state = _live_verdict_state(ctx, phase)
    if state is None:
        return None
    left = sum(1 for i in stand if i.severity != "nitpick")
    run_id = _standing_run12(ctx, phase)
    strength, why = _review_strength12(ctx, run_id)
    prov = _conversation_provenance(ctx, lm)
    # a code check failed after this commit's approval and looked related (ci.py): only a
    # review that sees that failure may approve the commit again, not a discussion
    ci_hold = kv_get(ctx.conn, ci.updated_key(ctx.repo, ctx.number, ctx.sha)) is not None
    approve = not left and strength == "ok" and not ctx.thread_failed and not ci_hold
    verified = _verdict_only(phase, "APPROVE" if approve else "COMMENT")
    new_event = publish.verdict_event(verified, prov)
    source = f"the full review of `{ctx.sha[:8]}`" + (f" (run {run_id})" if run_id else "")
    if new_event == "APPROVE":
        note = (
            f"Approved: the discussion withdrew or deferred every finding {source} left "
            "standing, and the approval rests on that review."
        )
    elif left:
        note = f"Not approved: {left} non-blocking finding(s) still stand on this PR."
    elif ctx.thread_failed:
        note = (
            "Not approved: some of the threads could not be read this round; they are read "
            "again on the next reply or push."
        )
    elif prov.degraded:
        note = "Not approved: this follow-up ran on a stand-in model (degraded mode)."
    elif ci_hold:
        note = (
            "Not approved: CI failed on this commit after its review, in checks that looked "
            "related (see the CI follow-up); the next push gets a full review."
        )
    elif strength == "needs_fresh":
        queued = queue()
        note = (
            "Not approved yet: an approval needs the confirmation pass over the whole change, "
            f"which the review of this commit did not run ({why}). "
            + (
                "It is queued for this commit and approves if it finds nothing."
                if queued
                else "It was already requested once for this commit; push or ask for a re-review."
            )
        )
    else:
        note = (
            "Not approved: the standing review of this commit did not run at full strength "
            f"({why}), so it cannot carry an approval."
        )
    return _Verdict12(
        state=state,
        event=new_event,
        verified=verified,
        prov=prov,
        titles=_lifted_blocker_titles(before, lifted),
        note=note,
    )


# ---- CI after the verdict: a check failed after this head's APPROVE (ci.py) ----


def step_ci_check(ctx: RunContext) -> dict[str, Any]:
    """The light run the `ci_watch` daemon task queues when a code check failed after this
    head's APPROVE: no checkout, no review. Re-reads CI and the standing verdict, asks one
    cheap lane whether the failure relates to the change, and when it does (or the lane
    cannot tell) posts one follow-up review that moves the APPROVE to COMMENT. Unrelated: an
    event, nothing posted. Never touches a REQUEST_CHANGES or COMMENT, never posts twice for a
    head. Returns the step detail."""
    live = github.pr_meta(ctx.gh, ctx.repo, ctx.number)
    _require_live(ctx, live)
    ctx.meta = live
    state = ci.read(ctx.gh, ctx.cfg, ctx.repo, ctx.sha)
    if state.error is not None:
        raise ReviewError(FailKind.INFRA, f"CI checks unreadable: {state.error}")
    w = ci.load(ctx.conn, ctx.repo, ctx.number, ctx.sha) or {}
    failing = state.unjudged(w.get("known") or [])
    names = [str(c["name"]) for c in failing]
    pending = bool(state.pending)
    if not failing:  # re-run and passed since the daemon looked
        _ci_done(ctx, "ci.recovered", "no failing check left to judge", pending=pending)
        return {"outcome": "recovered"}
    reviews = github.reviews(ctx.gh, ctx.repo, ctx.number)
    if held := _ci_held(ctx, reviews, {}):
        return held
    _fetch_files(ctx)
    logs = ci.job_logs(ctx.gh, ctx.repo, failing)
    text = ci.prompt(
        repo=ctx.repo,
        number=ctx.number,
        base_ref=live.base_ref,
        title=live.title,
        body=live.body,
        files=ctx.files,
        failing=failing,
        logs=logs,
        review=_ci_review_summary(ctx, reviews),
    )
    j = _ci_judge(ctx, text, names)
    ctx.check_cancel()  # a lane stopped by a cancel is no "could not judge"
    detail = {"outcome": j.related, "checks": names, "method": j.method, "related": j.checks}
    (ctx.run_dir / "ci_check.json").write_text(
        json.dumps({**dataclasses.asdict(j), "failing": failing, "logs": logs}, indent=1)
    )
    if j.related == "no":
        why = f"{', '.join(names)}: {j.reason}"
        _ci_done(ctx, "ci.unrelated", why, judged=names, pending=pending)
        return detail
    # the lane ran for a minute or more: a push, a dismissal or another verdict since then
    _require_live(ctx, github.pr_meta(ctx.gh, ctx.repo, ctx.number))
    if held := _ci_held(ctx, github.reviews(ctx.gh, ctx.repo, ctx.number), detail):
        return held
    ctx.check_cancel()
    if ctx.dry_run:
        return {**detail, "posted": False}
    result = publish.publish_ci_update(
        ctx.gh,
        repo=ctx.repo,
        number=ctx.number,
        head_sha=ctx.sha,
        failing=failing,
        related=j.related,
        reason=j.reason,
        lane=_lane_provenance(ctx.lane_model(ctx.cfg.policy.ci_check_lane), "ci-check"),
        judged=j.judged,
        logs_read=sum(1 for lg in logs if lg.get("tail")),
        degraded=_degraded_provenance(ctx),
    )
    with tx(ctx.conn):
        _ci_record_update(ctx, result.review_id, result.event)
        _set_run_review(
            ctx, blocker_count=0, review_id=result.review_id, review_url=result.review_url
        )
        event(
            ctx.conn,
            "review.verdict_updated",
            repo=ctx.repo,
            number=ctx.number,
            run_id=ctx.run_id,
            detail=f"APPROVED -> {result.event} on {ctx.sha[:8]} (CI failed after the review: {', '.join(names)}; related={j.related})",
        )
        why = f"{j.related}: {', '.join(names)}: {j.reason or j.error or ''}"
        _ci_end(ctx, "ci.related", why, pending=False)
    return {**detail, "posted": True, "event": result.event}


def _ci_held(
    ctx: RunContext, reviews: list[dict[str, Any]], detail: dict[str, Any]
) -> dict[str, Any] | None:
    """The step detail when the CI check has nothing to change (the watch ends, with an event
    saying why), else None: while the bot's APPROVE stands on this exact head and the head
    has no CI update yet. A COMMENT or REQUEST_CHANGES is never touched, a dismissal never
    re-asserted, and a head gets one CI update (one posted just before a crash, which this
    database missed, is recorded now)."""
    bot = ctx.cfg.bot_login
    kind, why = "ci.already_updated", ""
    marker = publish.CI_UPDATE_MARKER.format(sha=ctx.sha)
    posted = next(
        (
            r
            for r in reviews
            if (r.get("user") or {}).get("login") == bot and marker in str(r.get("body") or "")
        ),
        None,
    )
    if kv_get(ctx.conn, ci.updated_key(ctx.repo, ctx.number, ctx.sha)):
        why = "a CI update was already posted for this head"
    elif posted is not None:
        with tx(ctx.conn):
            _ci_record_update(ctx, posted.get("id"), "COMMENT")
        why = f"a CI update was already posted for this head (review {posted.get('id')})"
    else:
        state = publish.standing_verdict(reviews, ctx.sha, "final", bot)
        assert ctx.meta
        own = ctx.meta.author.lower() == bot.lower()
        if state == "APPROVED" or (
            # a bot-authored PR: GitHub shows COMMENTED, our record says APPROVE
            own and labels.latest_verdict(ctx.conn, ctx.repo, ctx.number, ctx.sha) == "APPROVE"
        ):
            return None
        kind, why = "ci.verdict_moved", f"the standing verdict is {state or 'none'}"
    _ci_done(ctx, kind, why, pending=False)
    return {**detail, "outcome": "no_approve", "reason": why}


def _ci_record_update(ctx: RunContext, review_id: Any, event_name: str) -> None:
    """Record the head's CI update: its verdict (the label follows) and the once-per-head
    guard. Caller holds the write transaction."""
    ctx.conn.execute(
        "INSERT INTO reviews (run_id, repo, number, sha, phase, github_review_id, event, posted_at) VALUES (?,?,?,?,?,?,?,?)",
        (ctx.run_id, ctx.repo, ctx.number, ctx.sha, "final", review_id, event_name, now()),
    )
    kv_set(
        ctx.conn,
        ci.updated_key(ctx.repo, ctx.number, ctx.sha),
        json.dumps({"review_id": review_id, "run_id": ctx.run_id, "at": now()}),
    )


def _ci_end(
    ctx: RunContext, kind: str, why: str, *, pending: bool, judged: list[str] | None = None
) -> None:
    """The CI check is done with the head: say why, and watch on while checks still run.
    Caller holds the write transaction."""
    event(
        ctx.conn,
        kind,
        repo=ctx.repo,
        number=ctx.number,
        run_id=ctx.run_id,
        detail=f"{ctx.sha[:8]}: {why}",
    )
    ci.release(ctx.conn, ctx.repo, ctx.number, ctx.sha, judged=judged or [], pending=pending)


def _ci_done(
    ctx: RunContext, kind: str, why: str, *, pending: bool, judged: list[str] | None = None
) -> None:
    with tx(ctx.conn):
        _ci_end(ctx, kind, why, pending=pending, judged=judged)


def _ci_review_summary(ctx: RunContext, reviews: list[dict[str, Any]]) -> str:
    """The body of the review the APPROVE came from, as the lane reads it."""
    for r in reversed(reviews):
        body = str(r.get("body") or "")
        if (r.get("user") or {}).get("login") == ctx.cfg.bot_login and (
            f"phase=final sha={ctx.sha}" in body and github.REVIEW_MARKER in body
        ):
            return ci.review_summary(body)
    return ""


def _ci_judge(ctx: RunContext, text: str, names: list[str]) -> ci.Judgement:
    lane = ctx.cfg.policy.ci_check_lane

    def attempt(first: int = 1) -> ci.Judgement:
        return ci.judge(
            ctx.cfg,
            lane=ctx.lane_model(lane),
            text=text,
            names=names,
            run_dir=ctx.run_dir,
            runner=ctx.lane_runner,
            should_stop=ctx.cancel_flag.is_set,
            first_attempt=first,
        )

    j = attempt()
    if j.infra_error and _degrade_on_side_lane_failure(ctx, "ci_check", lane.model, j.infra_error):
        j = attempt(3)  # once more, on the stand-in (its artifacts numbered on)
    if j.error:
        with tx(ctx.conn):
            event(
                ctx.conn,
                "ci.check_degraded",
                repo=ctx.repo,
                number=ctx.number,
                run_id=ctx.run_id,
                detail=f"related={j.related} method={j.method} error={j.error}",
            )
    return j


def run(ctx: RunContext) -> RunStatus:
    pol = ctx.cfg.policy
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    _detect_degraded(ctx)
    if ctx.is_ci_check:
        _step_start(ctx, StepName.CI_CHECK)
        _step_end(ctx, StepName.CI_CHECK, "ok", step_ci_check(ctx))
        return RunStatus.DONE
    standing = _standing_review(ctx) if ctx.trigger == Trigger.REVIEW_REPLY else None
    ctx.conversation = standing is not None
    ctx.check_cancel()
    _step_start(ctx, StepName.WORKTREE)
    step_worktree(ctx)
    _step_end(ctx, StepName.WORKTREE, "ok")
    if not standing and _carry(ctx):
        return RunStatus.DONE
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
        detail = step_threads(ctx, standing) if pol.thread_lane else step_converse(ctx, standing)
        _step_end(ctx, StepName.CONVERSE, "ok", detail)
        return RunStatus.DONE
    effort = pol.tier_effort(ctx.tier)
    ctx.phase2_follows = pol.phase2_enabled and effort.phase2 is not None
    _choose_comparison(ctx)
    # policy v12: the thread lane runs beside whichever finders run first (a no-op otherwise)
    _start_thread_lane(ctx)
    try:
        return _review(ctx, effort)
    finally:
        _stop_thread_lane(ctx)


def _review(ctx: RunContext, effort: TierEffort) -> RunStatus:
    """The review pipeline of a new-sha round (or a same-sha re-review): Phase 1, the gate and
    Phase 2, or the paths that skip parts of it."""
    pol = ctx.cfg.policy
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
    forced = ctx.force_phase2 and blocks and not admit and not ctx.is_audit
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
            **({"forced_phase2": True} if forced else {}),
        },
    )
    if forced:
        _record_gate_would_publish(ctx, gate_deferred=not ctx.verify1.blocker_count)
        return _phase2_only(ctx, effort)
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


def _record_gate_would_publish(ctx: RunContext, *, gate_deferred: bool) -> None:
    """A replay forcing Phase 2 past the gate (`force_phase2`): write the preliminary review
    the gate would have published instead (`gate-would-publish.json`), so the eval can score
    both what the gate published and what the whole pipeline finds. Not under policy v12:
    its publication is decided by `_publish12`'s post-rules pass (budget, settled re-folds,
    thread outcomes, the ledger verdict), which writes events and the ledger as it goes and so
    cannot be run a second time for a publication that never happens; a preliminary built
    from the gate verifier's raw set would not be what v12 posts, so none is recorded."""
    assert ctx.verify1 is not None
    if ctx.cfg.policy.pipeline is not None:
        with tx(ctx.conn):
            event(
                ctx.conn,
                "replay.phase2_forced",
                repo=ctx.repo,
                number=ctx.number,
                run_id=ctx.run_id,
                detail=f"gate held Phase 2 ({ctx.verify1.blocker_count} blockers); running it "
                "anyway (policy v12: no gate-would-publish record)",
            )
        return
    out: dict[str, Any] = {
        "phase": "preliminary",
        "gate_deferred": gate_deferred,
        "blockers": ctx.verify1.blocker_count,
        "findings": [_finding_record(f) for f in ctx.verify1.findings],
    }
    try:
        model = _build_review(
            ctx,
            phase="preliminary",
            verified=ctx.verify1,
            verifier_lm=ctx.lane_model(ctx.cfg.policy.phase1_verifier),
            phase2_skipped=PHASE2_DEFERRED if gate_deferred else None,
            rereview=ctx.has_prior_review,
        )
        out.update(
            event=model.canonical_event,
            kept=[_finding_record(f) for f in model.kept],
            body=publish.render(model),
        )
    except ReviewError as exc:  # evidence for the eval only, never the run's problem
        out["error"] = exc.message
    (ctx.run_dir / "gate-would-publish.json").write_text(json.dumps(out, indent=1))
    with tx(ctx.conn):
        event(
            ctx.conn,
            "replay.phase2_forced",
            repo=ctx.repo,
            number=ctx.number,
            run_id=ctx.run_id,
            detail=f"gate held Phase 2 ({ctx.verify1.blocker_count} blockers); running it anyway",
        )


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
    if pol.verifier_v12:
        ctx.verify2 = _verify_step(
            ctx,
            step=StepName.VERIFY2,
            phase="verify2",
            lm=ctx.lane_model(pol.phase2_verifier),
            expected_phase="final",
        )
        ctx.verify2 = _confirm12(ctx, ctx.verify2, effort)
    else:
        ctx.verify2 = _verify2(ctx, effort)
    _publish_step(
        ctx, phase="final", verified=ctx.verify2, verifier_lm=ctx.lane_model(pol.phase2_verifier)
    )
    return RunStatus.DONE


def _final_review_stands(ctx: RunContext) -> bool:
    """A final review for this exact head is already on the PR (a same-sha re-review)."""
    if ctx.dry_run and not ctx.replay:
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
    replay: bool = False,
    force_phase2: bool = False,
    coverage_base: str | None = None,
    replay_context: dict[str, Any] | None = None,
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
        replay=replay,
        force_phase2=force_phase2,
        coverage_base=coverage_base,
        replay_context=replay_context,
    )
    ctx.deferral_cutoff = converse.tracking_required_since(
        conn, cfg.deferral.tracking_required_since, record=not dry_run
    )
    if row["queue"] == "audit":
        ctx.audit = audit.load(conn, ctx.head_id)
        if ctx.audit is None:
            raise SystemExit(f"run {run_id}: audit head without an audits row")
    # every lane of this run, parallel reviewers and side lanes alike, lines up for a slot in
    # its model's machine-wide pool (lanepool.py); under policy v12 it may run isolated. A
    # replay (policy eval) lines up behind every production lane, whatever it replays
    queue = "audit" if ctx.is_audit else "priority" if row["priority"] else "live"
    if replay:
        queue = "replay"
    ctx.lane_runner = lanepool.gated(
        sandboxed(
            isolated(lane_runner or run_claude_lane, cfg.policy.isolates_lane_env),
            exec_deny_profile(cfg.lane_deny_exec),
        ),
        lambda: ctx.conn,
        cfg,
        run_stopped=ctx.cancel_flag.is_set,
        queue=queue,
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
        _fail_run_progress(ctx, reason)
    except Exception as exc:
        status, reason, kind = RunStatus.FAILED, f"{type(exc).__name__}: {exc}", FailKind.INFRA
        log.exception("worker crashed")
        _fail_run_progress(ctx, reason)
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
