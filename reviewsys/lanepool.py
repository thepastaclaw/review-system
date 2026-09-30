"""Machine-wide lane slots per model pool, shared by every worker process.

The reviewer lanes of one phase run in parallel (worker `_reviewer_lanes`), so the number of
model sessions in flight is no longer the number of runs. Every lane a worker starts takes a
slot in its model's pool first: the pool is the model family (`gpt`, `muse`, `glm`,
`gemini`), which is one provider quota. A slot is an exclusive `flock` on one of N files
under `work/lane-slots`, held for as long as the lane runs. The kernel drops the lock when
the process dies, so a crashed or hard-killed worker can never ghost a slot, and no daemon
bookkeeping is involved.

`gpt` lanes are bounded by the run ceiling (`slots.capacity`): the run slots are sized as
~3 streams per usable OpenAI account, and before lanes ran in parallel a run never held more
than one stream, so the ceiling is exactly the stream budget the accounts were measured at.
The other pools take their bound from `[scheduling] lane_pools`; a family with no entry is
not gated.

Two rules keep a busy run from starving the others:
- The top slot of a pool (when it has more than one) is reserved for lanes that are not
  parallel reviewers: verifiers, triage, selector, repair, conversation. They are short and
  on some run's critical path, so they never queue behind another run's reviewers.
- Parallel reviewer lanes wait in line: a ticket file per waiter, and only the head of the
  line may take a slot. Priority runs line up before live runs, live runs before audits
  (audits only ever use capacity live review leaves idle), then first come first served.
  Without the line, a run whose lane just finished takes the freed slot straight back for
  its next reviewer, before any other run's waiter wakes up.

Comparison lanes (a second model beside Phase 2, see `ComparisonPolicy`) take slots in a
pool of their own, `compare`, never the model family's: a slot is held for the whole lane
and cannot be taken back, so a comparison lane in a production pool could make a primary
lane (its own run's retry included) wait behind it. They still draw on the same OpenAI
accounts, whose stream budget the `gpt` ceiling is, so one only starts while the `gpt` pool
has an idle reviewer slot at that moment (it does not take it), and the `compare` pool
(`[scheduling] lane_pools.compare`, default COMPARE_POOL_SLOTS) bounds how far they can go
over the budget when production picks up after they started.
"""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path

from . import slots
from .config import Config
from .db import now
from .lane import LaneResult, LaneRunner, LaneSpec

POLL_SECONDS = 1.0

# a reviewer lane's place in line; lanes that are not parallel reviewers do not line up
RANK_PRIORITY, RANK_LIVE, RANK_AUDIT = 1, 2, 3

COMPARE_POOL = "compare"
COMPARE_POOL_SLOTS = 2
# what a lane is doing right now, for the public status export (exporter.py): `waiting` for a
# pool slot or `running`. Written into the lane's artifact dir; the runner's lane-meta.json
# supersedes it once the lane ends.
LANE_STATE = "lane-state.json"


def pool_of(model: str) -> str:
    return model.split("-", 1)[0]


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


def _line(wait_dir: Path, prune: bool = True) -> list[str]:
    """The waiting tickets in order, skipping (and with `prune`, deleting) those of processes
    that died waiting."""
    line = []
    for name in sorted(os.listdir(wait_dir)):
        try:
            pid = int(name.split("-")[2])
        except (IndexError, ValueError):
            continue
        if pid != os.getpid() and not _alive(pid):
            if prune:
                (wait_dir / name).unlink(missing_ok=True)
            continue
        line.append(name)
    return line


def waiting_line(slot_dir: Path, pool: str) -> list[str]:
    """The pool's waiting tickets in order, without `_line`'s cleanup: for readers that are
    not waiters themselves (the status export)."""
    try:
        return _line(slot_dir / f"{pool}.wait", prune=False)
    except OSError:
        return []


def waiter_key() -> str:
    """The `<pid>-<thread>` suffix of the ticket `acquire` takes on this thread."""
    return f"{os.getpid()}-{threading.get_ident()}"


def acquire(
    slot_dir: Path,
    pool: str,
    count: Callable[[], int],
    should_stop: Callable[[], bool],
    rank: int | None = None,
    admit: Callable[[], bool] | None = None,
) -> int | None:
    """Block until the lane may start and return its locked slot fd, or None once
    `should_stop()` says it is no longer wanted. `rank` None: not a parallel reviewer, may
    take any slot (the reserved top one first) without lining up; `admit` (rank None only)
    must also say yes before it tries. `count` is re-read on every pass, so a capacity
    change applies to lanes still waiting."""
    slot_dir.mkdir(parents=True, exist_ok=True)
    if rank is None:
        while True:
            n = max(1, count())
            if admit is None or admit():
                fd = _try_slots(slot_dir, pool, range(n - 1, -1, -1))
                if fd is not None:
                    return fd
            if should_stop():
                return None
            time.sleep(POLL_SECONDS)
    wait_dir = slot_dir / f"{pool}.wait"
    wait_dir.mkdir(exist_ok=True)
    ticket = wait_dir / f"{rank}-{time.time_ns():020d}-{waiter_key()}"
    ticket.touch()
    try:
        while True:
            if _line(wait_dir)[:1] == [ticket.name]:
                n = max(1, count())
                fd = _try_slots(slot_dir, pool, range(max(1, n - 1)))
                if fd is not None:
                    return fd
            if should_stop():
                return None
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


def gated(
    runner: LaneRunner,
    conn: Callable[[], sqlite3.Connection],
    cfg: Config,
    run_stopped: Callable[[], bool],
    reviewer_rank: int,
) -> LaneRunner:
    """`runner`, taking a pool slot around every lane. A lane stopped while it waits for a
    slot returns a result that never `started`. `conn` returns the calling thread's
    connection (lanes run on parallel threads); `reviewer_rank` is this run's place in line
    for its parallel reviewer lanes."""
    slot_dir = cfg.work_dir / "lane-slots"

    def run(spec: LaneSpec, artifact_dir: Path, worktree: Path) -> LaneResult:
        pool = spec.pool or pool_of(spec.model)
        try:
            return _run(spec, artifact_dir, worktree, pool)
        except BaseException:
            # a lane that raised never writes lane-meta.json: without this the status page
            # would show it running for the rest of the run
            _mark(artifact_dir, spec, pool, None)
            raise

    def _run(spec: LaneSpec, artifact_dir: Path, worktree: Path, pool: str) -> LaneResult:
        if limit(conn(), cfg, pool) is None:
            _mark(artifact_dir, spec, pool, "running")
            return runner(spec, artifact_dir, worktree)

        def gpt_idle() -> bool:
            """No production reviewer is waiting and one of their slots is free right now
            (probed, not kept): a waiter may be about to take the free slot."""
            wait_dir = slot_dir / "gpt.wait"
            if wait_dir.is_dir() and _line(wait_dir):
                return False
            n = limit(conn(), cfg, "gpt") or 1
            fd = _try_slots(slot_dir, "gpt", range(max(1, n - 1)))
            if fd is None:
                return False
            release(fd)
            return True

        _mark(artifact_dir, spec, pool, "waiting")
        fd = acquire(
            slot_dir,
            pool,
            lambda: limit(conn(), cfg, pool) or 1,
            spec.should_stop or run_stopped,
            rank=reviewer_rank if spec.parallel else None,
            admit=gpt_idle if pool == COMPARE_POOL else None,
        )
        stop = spec.should_stop or run_stopped
        if fd is not None and stop():
            release(fd)  # stopped while it waited: a lane no longer wanted must not start
            fd = None
        if fd is None:
            _mark(artifact_dir, spec, pool, None)
            return LaneResult(
                exit_code=None, stdout="", stderr="", duration_s=0, cancelled=True, started=False
            )
        try:
            _mark(artifact_dir, spec, pool, "running")
            return runner(spec, artifact_dir, worktree)
        finally:
            release(fd)

    return run
