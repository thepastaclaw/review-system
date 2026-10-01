"""Machine-wide lane slots per model pool, shared by every worker process.

The scheduler does not bound reviews by a run count any more (only a runaway guard,
`[scheduling] max_runs`): a run starts as soon as its head is eligible, and the models are
what is limited. The owner's rule (2026-10): "instead of limits for PRs, only limit for
models; if we have open model slots, do it. That way we get Muse out of the way ASAP; and
prioritize verify over finder jobs." So a run whose Phase 1 is on Muse or GLM no longer
waits behind runs that hold the gpt budget; its cheap lanes start while gpt is busy, and it
only waits where its next lane's model is actually saturated. No role changes model to get
around a busy pool.

Every lane a worker starts takes a slot in its model's pool first: the pool is the model
family (`gpt`, `muse`, `glm`, `gemini`), which is one provider quota. A slot is an exclusive
`flock` on one of N files under `work/lane-slots`, held for as long as the lane runs. The
kernel drops the lock when the process dies, so a crashed or hard-killed worker can never
ghost a slot, and no daemon bookkeeping is involved.

`gpt` lanes are bounded by the account-scaled stream budget (`slots.capacity().ceiling`:
`max_concurrent + priority_overflow` streams per usable OpenAI account, see slots.py). The
other pools take their bound from `[scheduling] lane_pools`; a family with no entry is not
gated.

Every lane that takes a slot lines up first: one ticket file per waiter in
`{pool}.line/`, named `{rank}-{run_id:012d}-{time_ns:020d}-{pid}-{tid}`, so the lexical
order is the rank, then the older run, then arrival. The ranks:

  1 priority-run side lane   2 live-run side lane     (verifiers, selector, triage,
  3 priority-run reviewer    4 live-run reviewer       repair, conversation: `parallel`
  5 audit side lane          6 audit reviewer          False; reviewers: True)

Side lanes are short and on some run's critical path (a verifier finishes a review a
reviewer has already paid for), so they go first; audits only ever use what live review
leaves idle. Older runs first means a run that has started finishes before a newer one
starts more lanes on the same model. The take rules (n = the pool's slot count; its top
slot, index n-1, is kept for side lanes when n > 1):
- The head of the line may take any slot its kind allows: a side lane the top slot first,
  then any other; a reviewer only one below the top.
- A side lane that is not the head, with no other side lane ahead of it (only reviewers
  are), may take the top slot and only that one: an express lane, so a verifier never
  waits behind a long reviewer queue or behind a head reviewer stuck waiting for a
  reviewer slot while the top one is free.
Without the line, a run whose lane just finished would take the freed slot straight back
for its next lane before any other run's waiter woke up. A ticket whose process died is
pruned by the next waiter that reads the line.

The directory is new (`.line`, not the `.wait` of the reviewer-only line it replaces):
workers still running the old code during a deploy parse every ticket name in `.wait` as
`rank-time-pid-tid` and must never meet this format. Both generations take the same slot
files, so the pool bound holds across a deploy; only the ordering between them is loose.

Comparison lanes (a second model beside Phase 2, see `ComparisonPolicy`) take slots in a
pool of their own, `compare`, never the model family's: a slot is held for the whole lane
and cannot be taken back, so a comparison lane in a production pool could make a primary
lane (its own run's retry included) wait behind it. They still draw on the same OpenAI
accounts, whose stream budget the `gpt` ceiling is, so one only starts while no lane is in
the `gpt` line and that pool has an idle reviewer slot at that moment (it does not take it),
and the `compare` pool (`[scheduling] lane_pools.compare`, default COMPARE_POOL_SLOTS) bounds
how far they can go over the budget when production picks up after they started.

A run's deadline (`runs.deadline_at`, enforced by reaper.py) must not count the time it
spends waiting for slots, now that runs are admitted without a model-aware cap: `gated`
tracks each run's lanes and pushes the deadline forward by every stretch in which at least
one lane waited and none ran (see `StallClock`), up to STALL_CREDIT_MAX_FACTOR run timeouts
in all, so a line that never moves still ends in a timeout.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

from . import slots
from .config import Config
from .db import fmt_ts, now, parse_ts, tx
from .lane import LaneResult, LaneRunner, LaneSpec

log = logging.getLogger(__name__)

POLL_SECONDS = 1.0

# a lane's place in its pool's line (see the module docstring)
RANK_PRIORITY_SIDE, RANK_LIVE_SIDE = 1, 2
RANK_PRIORITY_REVIEWER, RANK_LIVE_REVIEWER = 3, 4
RANK_AUDIT_SIDE, RANK_AUDIT_REVIEWER = 5, 6
SIDE_RANKS = frozenset({RANK_PRIORITY_SIDE, RANK_LIVE_SIDE, RANK_AUDIT_SIDE})
_RANKS = {
    ("priority", True): RANK_PRIORITY_SIDE,
    ("live", True): RANK_LIVE_SIDE,
    ("priority", False): RANK_PRIORITY_REVIEWER,
    ("live", False): RANK_LIVE_REVIEWER,
    ("audit", True): RANK_AUDIT_SIDE,
    ("audit", False): RANK_AUDIT_REVIEWER,
}
LINE_SUFFIX = ".line"
# a stall that has gone on this long is credited to the run's deadline while it lasts, so a
# run waiting for longer than its whole deadline is not reaped before the stall ends
STALL_FLUSH_SECONDS = 60.0
# at most this many run timeouts of slot waiting are credited to one run's deadline
STALL_CREDIT_MAX_FACTOR = 2

COMPARE_POOL = "compare"
COMPARE_POOL_SLOTS = 2
# what a lane is doing right now, for the public status export (exporter.py): `waiting` for a
# pool slot or `running`. Written into the lane's artifact dir; the runner's lane-meta.json
# supersedes it once the lane ends.
LANE_STATE = "lane-state.json"


def pool_of(model: str) -> str:
    return model.split("-", 1)[0]


def rank_of(queue: str, *, side: bool) -> int:
    """The line rank of a lane of a run from `queue` (`priority`, `live` or `audit`)."""
    return _RANKS[(queue, side)]


def limit(conn: sqlite3.Connection, cfg: Config, pool: str) -> int | None:
    """The pool's slot count right now; None = not gated."""
    if pool == "gpt":
        return slots.capacity(conn, cfg).ceiling
    if pool == COMPARE_POOL:  # always bounded, even when config.toml lists other pools only
        return cfg.lane_pools.get(pool, COMPARE_POOL_SLOTS)
    return cfg.lane_pools.get(pool)


def _try_slots(slot_dir: Path, pool: str, indexes: range) -> int | None:
    for i in indexes:
        fd = os.open(slot_dir / f"{pool}-{i}.lock", os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            continue
        return fd
    return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _fields(name: str) -> tuple[int, int] | None:
    """(rank, pid) of a ticket name, or None when it is not one."""
    parts = name.split("-")
    if len(parts) != 5:
        return None
    try:
        return int(parts[0]), int(parts[3])
    except ValueError:
        return None


def _line(line_dir: Path, prune: bool = True) -> list[str]:
    """The waiting tickets in order, skipping (and with `prune`, deleting) those of processes
    that died waiting."""
    line = []
    for name in sorted(os.listdir(line_dir)):
        fields = _fields(name)
        if fields is None:
            continue
        pid = fields[1]
        if pid != os.getpid() and not _alive(pid):
            if prune:
                (line_dir / name).unlink(missing_ok=True)
            continue
        line.append(name)
    return line


def line_dir(slot_dir: Path, pool: str) -> Path:
    return slot_dir / f"{pool}{LINE_SUFFIX}"


def waiting_line(slot_dir: Path, pool: str) -> list[str]:
    """The pool's waiting tickets in order, without `_line`'s cleanup: for readers that are
    not waiters themselves (the status export)."""
    try:
        return _line(line_dir(slot_dir, pool), prune=False)
    except OSError:
        return []


def waiter_key() -> str:
    """The `<pid>-<thread>` suffix of the ticket `acquire` takes on this thread."""
    return f"{os.getpid()}-{threading.get_ident()}"


def ticket_name(rank: int, run_id: int) -> str:
    return f"{rank}-{run_id:012d}-{time.time_ns():020d}-{waiter_key()}"


def _may_take(line: list[str], ticket: str, n: int, side: bool) -> range | None:
    """The slot indexes `ticket` may try now (in order), or None while it must wait."""
    pos = line.index(ticket)
    if pos == 0:
        if side:
            return range(n - 1, -1, -1)  # the reserved top slot first, then any
        return range(max(1, n - 1))  # never the top slot while the pool has more than one
    if side and n > 1:
        ahead = (_fields(t) for t in line[:pos])
        if not any(f is not None and f[0] in SIDE_RANKS for f in ahead):
            return range(n - 1, n)  # the express lane: the top slot only
    return None


def acquire(
    slot_dir: Path,
    pool: str,
    count: Callable[[], int],
    should_stop: Callable[[], bool],
    rank: int = RANK_LIVE_SIDE,
    run_id: int = 0,
    admit: Callable[[], bool] | None = None,
    on_wait: Callable[[], None] | None = None,
) -> int | None:
    """Line up in the pool's line at `rank` (for run `run_id`), block until the take rules
    let this lane have a slot and return its locked fd, or None once `should_stop()` says it
    is no longer wanted. `admit`, when given, must also say yes before it tries; `on_wait` is
    called on every pass that did not get a slot. `count` is re-read on every pass, so a
    capacity change applies to lanes still waiting. The ticket is removed on every way out."""
    slot_dir.mkdir(parents=True, exist_ok=True)
    wait_dir = line_dir(slot_dir, pool)
    wait_dir.mkdir(exist_ok=True)
    side = rank in SIDE_RANKS
    ticket = wait_dir / ticket_name(rank, run_id)
    ticket.touch()
    try:
        while True:
            line = _line(wait_dir)
            if ticket.name not in line:
                ticket.touch()  # removed under us (an operator clearing the dir): line up again
                line = _line(wait_dir)
            indexes = _may_take(line, ticket.name, max(1, count()), side)
            if indexes is not None and (admit is None or admit()):
                fd = _try_slots(slot_dir, pool, indexes)
                if fd is not None:
                    return fd
            if should_stop():
                return None
            if on_wait is not None:
                on_wait()
            time.sleep(POLL_SECONDS)
    finally:
        ticket.unlink(missing_ok=True)


def _mark(artifact_dir: Path, spec: LaneSpec, pool: str, state: str | None) -> None:
    """Record the lane's state (None: it never started, forget it). Best effort: a lane never
    fails over its own bookkeeping."""
    target = artifact_dir / LANE_STATE
    try:
        if state is None:
            target.unlink(missing_ok=True)
            return
        artifact_dir.mkdir(parents=True, exist_ok=True)
        tmp = artifact_dir / f"{LANE_STATE}.tmp"
        tmp.write_text(
            json.dumps(
                {
                    "state": state,
                    "since": now(),
                    "pool": pool,
                    "role": spec.role,
                    "model": spec.model,
                    "effort": spec.effort,
                    "parallel": spec.parallel,
                    "waiter": waiter_key(),
                }
            )
        )
        tmp.replace(target)
    except Exception:  # bookkeeping for the status page, never the lane's problem
        pass


def release(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


class StallClock:
    """One run's lanes, for its deadline: the run is stalled while at least one of its lanes
    waits for a slot and none is running. Each transition returns the stalled seconds it
    ends (0.0 when it ends none); `flush` hands out a stall still going on in slices, so a
    very long one is credited before the reaper's deadline passes. Thread-safe: a run's
    reviewer lanes wait and run on parallel threads."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._waiting = 0
        self._running = 0
        self._since: float | None = None  # start of the stall (or of its uncredited part)

    def _change(self, waiting: int, running: int) -> float:
        with self._lock:
            self._waiting += waiting
            self._running += running
            stalled = self._waiting > 0 and self._running == 0
            if stalled and self._since is None:
                self._since = self._clock()
            elif not stalled and self._since is not None:
                ended, self._since = self._clock() - self._since, None
                return ended
            return 0.0

    def wait_started(self) -> float:
        return self._change(+1, 0)

    def wait_ended(self, *, started: bool) -> float:
        return self._change(-1, +1 if started else 0)

    def run_started(self) -> float:  # a lane that never waits (its pool is not gated)
        return self._change(0, +1)

    def run_ended(self) -> float:
        return self._change(0, -1)

    def flush(self, min_seconds: float = STALL_FLUSH_SECONDS) -> float:
        with self._lock:
            if self._since is None:
                return 0.0
            at = self._clock()
            if at - self._since < min_seconds:
                return 0.0
            ended, self._since = at - self._since, at
            return ended


def extend_deadline(conn: sqlite3.Connection, run_id: int, seconds: float) -> None:
    """Push the run's deadline forward by `seconds` (whole seconds, as it is stored)."""
    with tx(conn):
        row = conn.execute(
            "SELECT deadline_at FROM runs WHERE id=? AND status IN ('spawned','running')",
            (run_id,),
        ).fetchone()
        if row is None or not row["deadline_at"]:
            return
        later = parse_ts(row["deadline_at"]) + timedelta(seconds=round(seconds))
        conn.execute("UPDATE runs SET deadline_at=? WHERE id=?", (fmt_ts(later), run_id))


def gated(
    runner: LaneRunner,
    conn: Callable[[], sqlite3.Connection],
    cfg: Config,
    run_stopped: Callable[[], bool],
    queue: str,
    run_id: int,
    clock: Callable[[], float] = time.monotonic,
) -> LaneRunner:
    """`runner`, taking a pool slot around every lane. A lane stopped while it waits for a
    slot returns a result that never `started`. `conn` returns the calling thread's
    connection (lanes run on parallel threads); `queue` (`priority`, `live` or `audit`) and
    `run_id` are the run's place in every pool's line. The run's deadline is pushed past the
    time its lanes only waited for slots (`StallClock`)."""
    slot_dir = cfg.work_dir / "lane-slots"
    stall = StallClock(clock)
    # stalled seconds not yet written (a write that failed is retried), and the room left
    # under the cap: a line that never moves (a ticket whose pid was reused) must still end
    # in a timeout, not keep a run alive forever
    owed = [0.0]
    room = [cfg.run_timeout_minutes * 60.0 * STALL_CREDIT_MAX_FACTOR]
    owed_lock = threading.Lock()

    def credit(seconds: float) -> None:
        with owed_lock:
            owed[0] += seconds
            if owed[0] < 1 or room[0] < 1:
                return
            due = min(owed[0], room[0])
            owed[0], room[0] = 0.0, room[0] - due
        try:
            extend_deadline(conn(), run_id, due)
        except Exception as exc:  # the deadline is bookkeeping; never fail a lane over it
            log.warning("run %s: could not extend the deadline by %.0fs: %s", run_id, due, exc)
            with owed_lock:
                owed[0] += due
                room[0] += due

    def run(spec: LaneSpec, artifact_dir: Path, worktree: Path) -> LaneResult:
        pool = spec.pool or pool_of(spec.model)
        try:
            return _run(spec, artifact_dir, worktree, pool)
        except BaseException:
            # a lane that raised never writes lane-meta.json: without this the status page
            # would show it running for the rest of the run
            _mark(artifact_dir, spec, pool, None)
            raise

    def _running(spec: LaneSpec, artifact_dir: Path, worktree: Path, pool: str) -> LaneResult:
        try:
            _mark(artifact_dir, spec, pool, "running")
            return runner(spec, artifact_dir, worktree)
        finally:
            credit(stall.run_ended())

    def _run(spec: LaneSpec, artifact_dir: Path, worktree: Path, pool: str) -> LaneResult:
        if limit(conn(), cfg, pool) is None:
            credit(stall.run_started())
            return _running(spec, artifact_dir, worktree, pool)

        def gpt_idle() -> bool:
            """No production lane is waiting for gpt and one of its reviewer slots is free
            right now (probed, not kept): a waiter may be about to take the free slot."""
            gpt_line = line_dir(slot_dir, "gpt")
            if gpt_line.is_dir() and _line(gpt_line):
                return False
            n = limit(conn(), cfg, "gpt") or 1
            fd = _try_slots(slot_dir, "gpt", range(max(1, n - 1)))
            if fd is None:
                return False
            release(fd)
            return True

        _mark(artifact_dir, spec, pool, "waiting")
        stop = spec.should_stop or run_stopped
        credit(stall.wait_started())
        fd: int | None = None
        try:
            fd = acquire(
                slot_dir,
                pool,
                lambda: limit(conn(), cfg, pool) or 1,
                stop,
                rank=rank_of(queue, side=not spec.parallel),
                run_id=run_id,
                admit=gpt_idle if pool == COMPARE_POOL else None,
                on_wait=lambda: credit(stall.flush()),
            )
            if fd is not None and stop():
                release(fd)  # stopped while it waited: a lane no longer wanted must not start
                fd = None
        finally:
            credit(stall.wait_ended(started=fd is not None))
        if fd is None:
            _mark(artifact_dir, spec, pool, None)
            return LaneResult(
                exit_code=None, stdout="", stderr="", duration_s=0, cancelled=True, started=False
            )
        try:
            return _running(spec, artifact_dir, worktree, pool)
        finally:
            release(fd)

    return run
