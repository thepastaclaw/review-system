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
lane (its own run's retry included) wait behind it. The pool bounds the extra load those
lanes put on the provider (`[scheduling] lane_pools.compare`, default COMPARE_POOL_SLOTS).
"""

from __future__ import annotations

import fcntl
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path

from . import slots
from .config import Config
from .lane import LaneResult, LaneRunner, LaneSpec

POLL_SECONDS = 1.0

# a reviewer lane's place in line; lanes that are not parallel reviewers do not line up
RANK_PRIORITY, RANK_LIVE, RANK_AUDIT = 1, 2, 3

COMPARE_POOL = "compare"
COMPARE_POOL_SLOTS = 4


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


def _line(wait_dir: Path) -> list[str]:
    """The waiting tickets in order, dropping those of processes that died waiting."""
    line = []
    for name in sorted(os.listdir(wait_dir)):
        try:
            pid = int(name.split("-")[2])
        except (IndexError, ValueError):
            continue
        if pid != os.getpid() and not _alive(pid):
            (wait_dir / name).unlink(missing_ok=True)
            continue
        line.append(name)
    return line


def acquire(
    slot_dir: Path,
    pool: str,
    count: Callable[[], int],
    should_stop: Callable[[], bool],
    rank: int | None = None,
) -> int | None:
    """Block until the lane may start and return its locked slot fd, or None once
    `should_stop()` says it is no longer wanted. `rank` None: not a parallel reviewer, may
    take any slot (the reserved top one first) without lining up. `count` is re-read on
    every pass, so a capacity change applies to lanes still waiting."""
    slot_dir.mkdir(parents=True, exist_ok=True)
    if rank is None:
        while True:
            n = max(1, count())
            fd = _try_slots(slot_dir, pool, range(n - 1, -1, -1))
            if fd is not None:
                return fd
            if should_stop():
                return None
            time.sleep(POLL_SECONDS)
    wait_dir = slot_dir / f"{pool}.wait"
    wait_dir.mkdir(exist_ok=True)
    ticket = wait_dir / f"{rank}-{time.time_ns():020d}-{os.getpid()}-{threading.get_ident()}"
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
        if limit(conn(), cfg, pool) is None:
            return runner(spec, artifact_dir, worktree)
        fd = acquire(
            slot_dir,
            pool,
            lambda: limit(conn(), cfg, pool) or 1,
            spec.should_stop or run_stopped,
            rank=reviewer_rank if spec.parallel else None,
        )
        if fd is None:
            return LaneResult(
                exit_code=None, stdout="", stderr="", duration_s=0, cancelled=True, started=False
            )
        try:
            return runner(spec, artifact_dir, worktree)
        finally:
            release(fd)

    return run
