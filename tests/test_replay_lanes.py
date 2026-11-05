"""Replay (policy eval) lanes in the machine's pools: they wait in their own directory, which
production (v0.24.0 through v0.28.0 included) never reads, take a slot only while no production lane waits,
never the side-lane slot, and at most floor((n - 1) * share) of a pool."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import ModuleType

import pytest

from reviewsys import lanepool, slots
from reviewsys.db import connect, connect_existing, kv_set, now, tx
from reviewsys.lane import LaneResult, LaneSpec
from reviewsys.lanepool import RANK_LIVE_REVIEWER, RANK_REPLAY, replay_slots
from reviewsys.models import ReviewError

V024 = Path(__file__).parent / "golden" / "lanepool_v0_24_0.py.txt"


def _t(rank: int, run: int = 1, n: int = 0) -> str:
    return f"{rank}-{run:012d}-{n:020d}-{os.getpid()}-1"


@pytest.fixture
def v024() -> ModuleType:
    """The production daemon's lanepool (v0.24.0, `git show 72ff47d:reviewsys/lanepool.py`;
    byte-identical in v0.28.0, 4e75920fe), loaded beside this one as a module of the same
    package."""
    from importlib.machinery import SourceFileLoader

    name = "reviewsys._lanepool_v0_24_0"
    loader = SourceFileLoader(name, str(V024))
    spec = importlib.util.spec_from_loader(name, loader)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    mod.POLL_SECONDS = 0.005  # type: ignore[attr-defined]
    yield mod
    sys.modules.pop(name, None)


def _waiting_replay(slot_dir: Path, pool: str, stop: threading.Event) -> threading.Thread:
    """A replay lane waiting for `pool` (held back by a busy share) until `stop`."""
    t = threading.Thread(
        target=lanepool._acquire_replay,
        args=(slot_dir, pool, lambda: range(0), stop.is_set, 9, None),
        daemon=True,
    )
    t.start()
    deadline = time.monotonic() + 5
    while not list(lanepool.replay_dir(slot_dir, pool).glob("7-*")):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    return t


def _spec(tmp_path: Path, model: str, pool: str | None = None) -> LaneSpec:
    return LaneSpec(
        role="r",
        agent="a",
        model=model,
        effort="high",
        prompt="p",
        cwd=tmp_path,
        add_dir=tmp_path,
        timeout_seconds=5,
        claude_bin="x",
        pool=pool,
    )


def _ok(spec, art, wt) -> LaneResult:
    return LaneResult(exit_code=0, stdout="", stderr="", duration_s=0)


def test_the_v024_daemon_sees_nothing_of_waiting_replays(cfg, conn, tmp_path, v024):
    """H1: v0.24.0 starts a comparison lane only while `gpt.line` is empty, and its status
    export counts that line: a replay waiting for gpt must not show up in either."""
    slot_dir = cfg.work_dir / "lane-slots"
    stop = threading.Event()
    t = _waiting_replay(slot_dir, "gpt", stop)
    try:
        assert v024.waiting_line(slot_dir, "gpt") == []
        assert lanepool.waiting_line(slot_dir, "gpt") == []
        done = threading.Event()
        gated = v024.gated(
            _ok, lambda: connect_existing(cfg.db_path), cfg, lambda: False, "live", 1
        )
        spec = _spec(tmp_path, "gpt-6-astra", pool=v024.COMPARE_POOL)
        runner = threading.Thread(
            target=lambda: (gated(spec, tmp_path / "art", tmp_path), done.set()), daemon=True
        )
        runner.start()
        runner.join(timeout=10)
        assert done.is_set()  # the comparison lane started at once
    finally:
        stop.set()
        t.join(timeout=5)
    assert not list(lanepool.replay_dir(slot_dir, "gpt").iterdir())  # ticket removed


def test_the_replay_rank_and_shares():
    assert lanepool.rank_of("replay", side=True) == RANK_REPLAY
    assert lanepool.rank_of("replay", side=False) == RANK_REPLAY
    # the box's pools at the default share 0.5: gpt 12, compare 2, gemini 6, glm 6, muse 20
    assert list(replay_slots(12, 0.5)) == [10, 9, 8, 7, 6]
    assert list(replay_slots(6, 0.5)) == [4, 3]
    assert list(replay_slots(20, 0.5)) == list(range(18, 9, -1))
    assert list(replay_slots(2, 0.5)) == [] and list(replay_slots(1, 0.5)) == []
    assert list(replay_slots(6, 0.0)) == []
    assert list(replay_slots(2, 0.5, cap=1)) == [0]  # an explicit per-pool override
    assert list(replay_slots(4, 1.0)) == [2, 1, 0]  # never the top slot
    assert list(replay_slots(4, 0.5, cap=9)) == [2, 1, 0]


def test_a_replay_waits_while_a_production_ticket_is_in_line(tmp_path):
    line = lanepool.line_dir(tmp_path, "glm")
    line.mkdir(parents=True)
    prod = line / _t(RANK_LIVE_REVIEWER)
    prod.touch()
    polls = {"n": 0}

    def stop() -> bool:
        polls["n"] += 1
        return polls["n"] > 3

    got = lanepool._acquire_replay(tmp_path, "glm", lambda: replay_slots(6, 0.5), stop, 9, None)
    assert got is None
    assert prod.exists()  # never pruned or touched by a replay
    prod.unlink()
    got = lanepool._acquire_replay(
        tmp_path, "glm", lambda: replay_slots(6, 0.5), lambda: False, 9, None
    )
    assert got is not None and got[1] == 4
    fd = got[0]
    try:  # slot 4, just below the top: slots 0-3 and the top stay free for production
        assert lanepool._try_slots(tmp_path, "glm", range(4, 5)) is None
        free = lanepool._try_slots(tmp_path, "glm", range(5, 6))
        assert free is not None
        lanepool.release(free)
    finally:
        lanepool.release(fd)
    assert list(line.iterdir()) == []  # a replay never wrote to the production line


def test_replays_hold_at_most_their_share(tmp_path):
    held = []
    for _ in range(2):
        got = lanepool._acquire_replay(
            tmp_path, "glm", lambda: replay_slots(6, 0.5), lambda: False, 1, None
        )
        assert got is not None
        held.append(got[0])
    polls = {"n": 0}

    def stop() -> bool:
        polls["n"] += 1
        return polls["n"] > 3

    try:
        third = lanepool._acquire_replay(
            tmp_path, "glm", lambda: replay_slots(6, 0.5), stop, 1, None
        )
        assert third is None
        # production still gets every other slot
        prod = lanepool.acquire(tmp_path, "glm", lambda: 6, lambda: False, rank=RANK_LIVE_REVIEWER)
        assert prod is not None
        lanepool.release(prod)
    finally:
        for fd in held:
            lanepool.release(fd)


def test_a_pool_without_a_replay_share_fails_the_lane_at_once(cfg, conn, tmp_path):
    small = dataclasses.replace(cfg, lane_pools={**cfg.lane_pools, "glm": 2})
    gated = lanepool.gated(_ok, lambda: conn, small, lambda: False, "replay", 1)
    with pytest.raises(ReviewError, match="no share for replay lanes"):
        gated(_spec(tmp_path, "glm-5.3-flash"), tmp_path / "art", tmp_path)
    opened = dataclasses.replace(small, replay_pool_slots={"glm": 1})
    gated = lanepool.gated(_ok, lambda: conn, opened, lambda: False, "replay", 1)
    assert gated(_spec(tmp_path, "glm-5.3-flash"), tmp_path / "art2", tmp_path).ok


def test_a_replay_lane_runs_through_gated_in_its_own_directory(cfg, conn, tmp_path):
    seen: list[list[str]] = []
    slot_dir = cfg.work_dir / "lane-slots"

    def runner(spec, art, wt) -> LaneResult:
        seen.append(sorted(p.name for p in slot_dir.iterdir()))
        return _ok(spec, art, wt)

    gated = lanepool.gated(runner, lambda: conn, cfg, lambda: False, "replay", 1)
    assert gated(_spec(tmp_path, "glm-5.3-flash"), tmp_path / "art", tmp_path).ok
    assert "glm.replay" in seen[0] and "glm.line" not in seen[0]


def test_the_gpt_pool_is_sized_from_the_capacity_database(cfg, tmp_path):
    live = connect(tmp_path / "live.db")
    accounts = [
        dataclasses.asdict(slots.Account(name=f"a{i}", usable=True, reason="ok")) for i in range(2)
    ]
    with tx(live):
        kv_set(live, slots.KV_ACCOUNTS, json.dumps({"at": now(), "accounts": accounts}))
    scratch = connect(tmp_path / "scratch.db")
    base = cfg.max_concurrent + cfg.priority_overflow
    assert lanepool.limit(scratch, cfg, "gpt") == base  # a scratch database has no reading
    replay_cfg = dataclasses.replace(cfg, lane_capacity_db=tmp_path / "live.db")
    assert lanepool.limit(scratch, replay_cfg, "gpt") == 2 * base
    gone = dataclasses.replace(cfg, lane_capacity_db=tmp_path / "missing.db")
    assert lanepool.limit(scratch, gone, "gpt") == base


def test_production_lanepool_is_v024_apart_from_the_replay_path():
    """The production take rules, line and status export are v0.24.0's, unchanged."""
    import inspect

    src = V024.read_text()
    for fn in (lanepool._may_take, lanepool.acquire, lanepool.waiting_line, lanepool._line):
        assert inspect.getsource(fn) in src, fn.__name__


def test_a_slot_outside_a_share_that_shrank_is_let_go(tmp_path):
    """M7: the share is read again once the slot is locked; the pool shrank between the two
    reads (an OpenAI account parked), so the slot taken is no longer a replay's: let go,
    and the lane takes one inside the new share."""
    reads = iter([range(4, 2, -1), range(1, 0, -1)])  # n=6 share 2, then n=3 share 1

    def indexes() -> range:
        return next(reads, range(1, 0, -1))

    got = lanepool._acquire_replay(tmp_path, "glm", indexes, lambda: False, 1, None)
    assert got is not None and got[1] == 1
    try:
        free = lanepool._try_slots(tmp_path, "glm", range(4, 5))
        assert free is not None  # slot 4 was released, not kept
        lanepool.release(free)
    finally:
        lanepool.release(got[0])


def test_a_correction_turn_moves_off_a_slot_the_share_no_longer_covers(
    cfg, conn, tmp_path, monkeypatch
):
    """M7: a replay lane keeps its slot between turns only while the share still covers it;
    after the pool shrank, the correction turn waits for a slot inside the new share."""
    import dataclasses as dc

    slot_dir = cfg.work_dir / "lane-slots"
    size = {"glm": 6}
    held: list[list[int]] = []

    def runner(spec, art, wt) -> LaneResult:
        taken = []
        for i in range(6):
            fd = lanepool._try_slots(slot_dir, "glm", range(i, i + 1))
            if fd is None:
                taken.append(i)
            else:
                lanepool.release(fd)
        held.append(taken)
        size["glm"] = 3  # an account parked while the first turn ran
        return LaneResult(exit_code=0, stdout="", stderr="", duration_s=0, result_text="x")

    real = lanepool.limit
    monkeypatch.setattr(
        lanepool, "limit", lambda c, cf, pool: size[pool] if pool == "glm" else real(c, cf, pool)
    )
    gated = lanepool.gated(runner, lambda: conn, cfg, lambda: False, "replay", 1)
    spec = dc.replace(
        _spec(tmp_path, "glm-5.3-flash"),
        session_id="11111111-1111-1111-1111-111111111111",
        corrections=1,
        check=lambda res: "again" if len(held) == 1 else None,
    )
    assert gated(spec, tmp_path / "art", tmp_path).ok
    # first turn on slot 4 (n=6: slots 4 and 3 are the replays'); once the pool is 3 the
    # replays have slot 1 only, and the correction turn ran there, not on slot 4
    assert held == [[4], [1]]
