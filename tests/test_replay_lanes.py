"""Replay (policy eval) lanes in the machine's pools: behind every production lane, never the
side-lane slot, at most a share of each pool; production unchanged without them."""

from __future__ import annotations

import dataclasses
import json
import os
import threading

from reviewsys import lanepool, slots
from reviewsys.db import connect, kv_set, now, tx
from reviewsys.lanepool import (
    RANK_AUDIT_REVIEWER,
    RANK_LIVE_REVIEWER,
    RANK_LIVE_SIDE,
    RANK_REPLAY,
    _may_take,
    replay_slots,
)


def _t(rank: int, run: int = 1, n: int = 0) -> str:
    return f"{rank}-{run:012d}-{n:020d}-{os.getpid()}-1"


def test_the_replay_rank_is_last():
    assert lanepool.rank_of("replay", side=True) == lanepool.rank_of("replay", side=False)
    assert RANK_REPLAY > RANK_AUDIT_REVIEWER
    assert sorted([_t(RANK_REPLAY, 1), _t(RANK_AUDIT_REVIEWER, 99)])[0].startswith("6-")


def test_replay_slots_are_a_share_just_below_the_top():
    assert list(replay_slots(6, 0.5)) == [4, 3, 2]
    assert list(replay_slots(4, 0.5)) == [2, 1]
    assert list(replay_slots(2, 0.5)) == [0]
    assert list(replay_slots(1, 0.5)) == [0]
    assert list(replay_slots(12, 0.0)) == [10]  # at least one
    assert list(replay_slots(3, 1.0)) == [1, 0]  # never the top slot


def test_a_replay_lane_takes_a_slot_only_with_no_production_lane_waiting():
    replay = _t(RANK_REPLAY, 3)
    assert _may_take([replay], replay, 6, side=False) == replay_slots(6, 0.5)
    for prod in (_t(RANK_LIVE_SIDE), _t(RANK_LIVE_REVIEWER), _t(RANK_AUDIT_REVIEWER)):
        line = sorted([prod, replay])
        assert line[-1] == replay
        assert _may_take(line, replay, 6, side=True) is None  # not even the express lane
        assert _may_take(line, replay, 6, side=False) is None
    # a second replay waits behind the first
    other = _t(RANK_REPLAY, 4)
    assert _may_take(sorted([replay, other]), other, 6, side=False) is None


def test_production_takes_are_unchanged_by_replays_in_the_line():
    replay, reviewer, side = _t(RANK_REPLAY, 1), _t(RANK_LIVE_REVIEWER, 2), _t(RANK_LIVE_SIDE, 3)
    for line in ([reviewer, side], [reviewer, side, replay]):
        line = sorted(line)
        assert _may_take(line, side, 6, side=True) == range(5, -1, -1)  # head: top first
        assert _may_take(line, reviewer, 6, side=False) is None
    line = sorted([reviewer, replay])
    assert _may_take(line, reviewer, 6, side=False) == range(5)


def test_the_status_export_does_not_count_replay_waiters(tmp_path):
    d = lanepool.line_dir(tmp_path, "gpt")
    d.mkdir()
    for name in (_t(RANK_LIVE_REVIEWER), _t(RANK_REPLAY, 2)):
        (d / name).touch()
    assert lanepool.waiting_line(tmp_path, "gpt") == [_t(RANK_LIVE_REVIEWER)]


def test_a_replay_waiter_never_overtakes_a_production_waiter(tmp_path):
    """Real flocks: with a production ticket in the line a replay lane does not get a slot
    even though slots are free; once the line is clear it takes one below the top."""
    d = lanepool.line_dir(tmp_path, "glm")
    d.mkdir(parents=True)
    prod = d / _t(RANK_LIVE_REVIEWER, 1)
    prod.touch()
    polls = {"n": 0}

    def stop() -> bool:
        polls["n"] += 1
        return polls["n"] > 3

    assert lanepool.acquire(tmp_path, "glm", lambda: 6, stop, rank=RANK_REPLAY, run_id=9) is None
    prod.unlink()
    fd = lanepool.acquire(tmp_path, "glm", lambda: 6, lambda: False, rank=RANK_REPLAY, run_id=9)
    assert fd is not None
    try:  # it holds slot 4 (just below the top), so the top stays free for production
        top = lanepool._try_slots(tmp_path, "glm", range(5, 6))
        assert top is not None
        lanepool.release(top)
        assert lanepool._try_slots(tmp_path, "glm", range(4, 5)) is None
    finally:
        lanepool.release(fd)


def test_replays_hold_at_most_their_share_of_a_pool(tmp_path):
    held: list[int] = []
    for _ in range(3):
        fd = lanepool.acquire(tmp_path, "glm", lambda: 6, lambda: False, rank=RANK_REPLAY)
        assert fd is not None
        held.append(fd)
    polls = {"n": 0}

    def stop() -> bool:
        polls["n"] += 1
        return polls["n"] > 3

    try:
        assert lanepool.acquire(tmp_path, "glm", lambda: 6, stop, rank=RANK_REPLAY) is None
        # production still has slots 0, 1 and the top one
        prod = lanepool.acquire(tmp_path, "glm", lambda: 6, lambda: False, rank=RANK_LIVE_SIDE)
        assert prod is not None
        lanepool.release(prod)
    finally:
        for fd in held:
            lanepool.release(fd)


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


def test_a_comparison_lane_is_not_held_back_by_waiting_replays(cfg, conn, tmp_path):
    """`gpt_idle` (comparison lanes start only while no gpt lane waits) ignores replay
    tickets: a replay waiting for gpt changes nothing for production."""
    from reviewsys.lane import LaneResult

    slot_dir = cfg.work_dir / "lane-slots"
    gpt_line = lanepool.line_dir(slot_dir, "gpt")
    gpt_line.mkdir(parents=True)
    (gpt_line / _t(RANK_REPLAY, 5)).touch()
    ran: list[str] = []

    def runner(spec, art, wt):
        ran.append(spec.role)
        return LaneResult(exit_code=0, stdout="", stderr="", duration_s=0)

    gated = lanepool.gated(runner, lambda: conn, cfg, lambda: False, "live", 1)
    from reviewsys.lane import LaneSpec

    spec = LaneSpec(
        role="cmp",
        agent="a",
        model="gpt-6-astra",
        effort="high",
        prompt="p",
        cwd=tmp_path,
        add_dir=tmp_path,
        timeout_seconds=5,
        claude_bin="x",
        pool=lanepool.COMPARE_POOL,
    )
    done = threading.Event()
    t = threading.Thread(
        target=lambda: (gated(spec, tmp_path / "art", tmp_path), done.set()), daemon=True
    )
    t.start()
    t.join(timeout=10)
    assert done.is_set() and ran == ["cmp"]
