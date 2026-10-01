"""The reviewer lanes of a phase run side by side, bounded per run and per model pool."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from reviewsys import lane as lane_mod
from reviewsys import lanepool, worker
from reviewsys.db import tx
from reviewsys.ingest import enqueue_head
from reviewsys.lane import LaneResult, LaneSpec
from reviewsys.models import RunStatus, Trigger
from reviewsys.scheduler import schedule
from reviewsys.steps import worktree as wt

HEAD = "a" * 40
REVIEWER_ROLES = {"general", "always-on", "security-auditor"}


@pytest.fixture(autouse=True)
def fake_git(monkeypatch, tmp_path):
    monkeypatch.setattr(wt, "ensure_mirror", lambda mirrors, repo: tmp_path / "mirror")
    monkeypatch.setattr(wt, "fetch_head", lambda mirror, number, sha: None)

    def create(mirror, wts, name, sha):
        p = wts / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(wt, "create_worktree", create)
    monkeypatch.setattr(wt, "remove_worktree", lambda mirror, path: None)
    monkeypatch.setattr(wt, "merge_base", lambda worktree, base, sha: "b" * 40)


def _verifier():
    return {
        "summary": "ok",
        "review_action": "COMMENT",
        "findings": [],
        "dropped_findings": [],
        "out_of_scope_findings": [],
        "coderabbit_reactions": [],
        "prerequisite_adjudications": [],
        "adjudication_complete": True,
    }


class Overlap:
    """Wraps FakeLanes: reviewer lanes wait for each other at a barrier, so the test only
    passes when they are really in flight together, and the peak is recorded."""

    def __init__(self, inner, *, barrier: int = 0, hold: float = 0.0) -> None:
        self.inner, self.hold = inner, hold
        self.barrier = threading.Barrier(barrier) if barrier else None
        self.lock = threading.Lock()
        self.live = self.peak = 0

    def __call__(self, spec: LaneSpec, art: Path, worktree: Path) -> LaneResult:
        if spec.role not in REVIEWER_ROLES:
            return self.inner(spec, art, worktree)
        with self.lock:
            self.live += 1
            self.peak = max(self.peak, self.live)
        try:
            if self.barrier is not None:
                self.barrier.wait(timeout=10)
            time.sleep(self.hold)
            return self.inner(spec, art, worktree)
        finally:
            with self.lock:
                self.live -= 1


def _run(cfg, conn, gh, runner):
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.MENTION)
    (rid,) = schedule(conn, cfg, spawn=False)
    return rid, worker.main(cfg, conn, rid, gh=gh, lane_runner=runner, heartbeat=False)


def test_reviewers_of_a_phase_run_side_by_side(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier()
    # all three reviewers must be inside the runner at once, in both phases, or this deadlocks;
    # a ceiling of 5 leaves Phase 2 four gpt reviewer slots (the top one is the verifiers')
    runner = Overlap(lanes, barrier=3)
    rid, status = _run(dataclasses.replace(cfg, max_concurrent=4), conn, gh, runner)
    assert status == RunStatus.DONE
    assert runner.peak == 3
    # provenance and finding rows stay in role order, as when the lanes ran one by one
    body = gh.posted_reviews[0]["body"]
    assert (
        body.index("— general (") < body.index("— always-on (") < body.index("— security-auditor (")
    )
    phases = [
        r["phase"]
        for r in conn.execute("SELECT phase FROM lanes WHERE run_id=? ORDER BY id", (rid,))
        if r["phase"] in ("phase1", "phase2")
    ]
    assert phases == ["phase1"] * 3 + ["phase2"] * 3, "a phase waits for all its lanes"


def test_phase_parallelism_caps_lanes_per_run(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier()
    runner = Overlap(lanes, hold=0.05)
    _, status = _run(dataclasses.replace(cfg, phase_parallelism=2), conn, gh, runner)
    assert status == RunStatus.DONE
    assert runner.peak == 2


def test_parallelism_one_is_the_old_sequential_flow(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier()
    runner = Overlap(lanes, hold=0.02)
    _, status = _run(dataclasses.replace(cfg, phase_parallelism=1), conn, gh, runner)
    assert status == RunStatus.DONE
    assert runner.peak == 1


def test_a_failed_lane_stops_its_siblings_and_fails_the_phase(cfg, conn, gh, lanes):
    """One reviewer failing for good fails the phase; the lanes still running are stopped
    (and recorded as cancelled) instead of spending quota on a phase that is abandoned. Phase 2
    runs the same way, so a failing lane there fails the run."""
    # straight to Phase 2, with gpt slots for all three reviewers at once
    cfg = dataclasses.replace(cfg, backlog_skip_phase1_above=1, max_concurrent=4)
    with tx(conn):  # a queue deeper than the limit once the head under test has started
        for n in (98, 99):
            enqueue_head(conn, cfg, "dashpay/platform", n, f"{n}" * 20, Trigger.NEW_PR)
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier()
    lanes.timeout_roles = {"general"}
    stopped: list[str] = []
    running = threading.Semaphore(0)  # general fails only once both siblings are running

    def runner(spec: LaneSpec, art: Path, worktree: Path) -> LaneResult:
        if spec.role == "general" and spec.should_stop is not None:
            for _ in range(2):
                assert running.acquire(timeout=10)
            running.release(2)  # a retry of general must not wait again
        if spec.role in ("always-on", "security-auditor"):
            running.release()
            # a long lane: runs until the phase is abandoned
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if spec.should_stop and spec.should_stop():
                    stopped.append(spec.role)
                    return LaneResult(
                        exit_code=None, stdout="", stderr="", duration_s=1, cancelled=True
                    )
                time.sleep(0.01)
            raise AssertionError(f"{spec.role} was never told to stop")
        return lanes(spec, art, worktree)

    rid, status = _run(cfg, conn, gh, runner)
    assert status == RunStatus.FAILED
    assert sorted(stopped) == ["always-on", "security-auditor"]
    reason = conn.execute("SELECT reason FROM runs WHERE id=?", (rid,)).fetchone()["reason"]
    assert "phase2/general lane failed twice" in reason, "the real failure, not a stopped lane"
    rows = {
        r["role"]: r["status"]
        for r in conn.execute(
            "SELECT role, status FROM lanes WHERE run_id=? AND phase='phase2'", (rid,)
        )
    }
    assert rows == {"general": "failed", "always-on": "cancelled", "security-auditor": "cancelled"}
    # a stopped lane names the lane that failed and how, not just "another lane failed"
    reasons = {
        r["reason"]
        for r in conn.execute(
            "SELECT reason FROM lanes WHERE run_id=? AND status='cancelled'", (rid,)
        )
    }
    assert reasons == {"stopped: phase2/general failed (lane timed out)"}
    assert not gh.posted_reviews


def test_pool_slots_bound_lanes_across_workers(tmp_path):
    """The per-model cap holds across processes: flock slots in one directory."""
    got: list[int] = []
    for _ in range(2):
        fd = lanepool.acquire(tmp_path, "muse", lambda: 2, lambda: False)
        assert fd is not None
        got.append(fd)
    # both slots held (as another worker would hold them): a third lane waits, then gives up
    assert lanepool.acquire(tmp_path, "muse", lambda: 2, lambda: True) is None
    # another process sees the same slots taken
    probe = (
        "import fcntl, os, sys\n"
        "free = 0\n"
        "for i in range(2):\n"
        "    fd = os.open(os.path.join(sys.argv[1], f'muse-{i}.lock'), os.O_RDWR)\n"
        "    try:\n"
        "        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); free += 1\n"
        "    except BlockingIOError:\n"
        "        pass\n"
        "print(free)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe, str(tmp_path)], capture_output=True, text=True
    )
    assert out.stdout.strip() == "0"
    lanepool.release(got.pop())
    fd = lanepool.acquire(tmp_path, "muse", lambda: 2, lambda: True)
    assert fd is not None, "a released slot is free again"
    for f in (fd, *got):
        lanepool.release(f)


def test_gated_runner_waits_for_a_slot_and_honours_cancel(cfg, conn, tmp_path):
    cfg = dataclasses.replace(cfg, lane_pools={"muse": 1})
    slot_dir = cfg.work_dir / "lane-slots"
    held = lanepool.acquire(slot_dir, "muse", lambda: 1, lambda: False)
    ran: list[str] = []
    art = tmp_path / "art"
    states: list[dict] = []

    def inner(spec, art, worktree):
        ran.append(spec.role)
        states.append(json.loads((art / lanepool.LANE_STATE).read_text()))
        return LaneResult(exit_code=0, stdout="", stderr="", duration_s=1)

    stop = threading.Event()
    gated = lanepool.gated(
        inner, lambda: conn, cfg, run_stopped=stop.is_set, queue="live", run_id=0
    )
    spec = LaneSpec(
        role="general",
        agent="a",
        model="muse-spark-1.3-contributor",
        effort="high",
        prompt="p",
        cwd=Path("."),
        add_dir=Path("."),
        timeout_seconds=5,
        claude_bin="claude",
    )
    stop.set()  # the run is cancelled while the lane waits for the only slot
    res = gated(spec, art, tmp_path)
    assert res.cancelled and not res.started and ran == []
    assert not (art / lanepool.LANE_STATE).exists(), "a lane that never started leaves no state"
    lanepool.release(held)
    stop.clear()
    assert not gated(spec, art, tmp_path).cancelled and ran == ["general"]
    assert states[-1]["state"] == "running" and states[-1]["pool"] == "muse"
    assert (states[-1]["model"], states[-1]["effort"]) == ("muse-spark-1.3-contributor", "high")
    # an ungated family never touches the slot files, but still says it is running
    assert gated(dataclasses.replace(spec, model="other-model"), tmp_path / "b", tmp_path).ok
    assert states[-1]["state"] == "running" and states[-1]["pool"] == "other"


def _muse_spec(tmp_path, **kw):
    return LaneSpec(
        agent="a",
        model="muse-spark-1.3-contributor",
        prompt="p",
        cwd=tmp_path,
        add_dir=tmp_path,
        timeout_seconds=5,
        claude_bin="claude",
        **kw,
    )


def test_gated_lane_that_raises_leaves_no_running_state(cfg, conn, tmp_path):
    """A runner that raises never writes lane-meta.json; its state file must not keep saying
    `running` for the rest of the run."""

    def boom(spec, art, worktree):
        raise OSError("claude binary missing")

    gated = lanepool.gated(boom, lambda: conn, cfg, lambda: False, "live", 0)
    with pytest.raises(OSError):
        gated(_muse_spec(tmp_path, role="triage", effort="low"), tmp_path / "art", tmp_path)
    assert not (tmp_path / "art" / lanepool.LANE_STATE).exists()


def test_waiting_lane_says_so_with_its_place_in_line(cfg, conn, tmp_path):
    """While a reviewer lane waits for a slot its state file says `waiting`, and its waiter key
    finds its ticket in the pool's line (what the status export shows as the place in line)."""
    cfg = dataclasses.replace(cfg, lane_pools={"muse": 2})
    slot_dir = cfg.work_dir / "lane-slots"
    held = lanepool.acquire(
        slot_dir, "muse", lambda: 2, lambda: False, rank=lanepool.RANK_LIVE_REVIEWER
    )
    art = tmp_path / "art"
    spec = _muse_spec(tmp_path, role="general", effort="high", parallel=True)
    gated = lanepool.gated(
        lambda *_: LaneResult(exit_code=0, stdout="", stderr="", duration_s=1),
        lambda: conn,
        cfg,
        run_stopped=lambda: False,
        queue="live",
        run_id=0,
    )
    t = threading.Thread(target=gated, args=(spec, art, tmp_path))
    t.start()
    for _ in range(50):
        if lanepool.waiting_line(slot_dir, "muse"):
            break
        time.sleep(0.05)
    state = json.loads((art / lanepool.LANE_STATE).read_text())
    assert state["state"] == "waiting" and state["parallel"] is True
    line = lanepool.waiting_line(slot_dir, "muse")
    assert len(line) == 1 and line[0].endswith("-" + state["waiter"])
    lanepool.release(held)
    t.join(5)
    assert json.loads((art / lanepool.LANE_STATE).read_text())["state"] == "running"


def test_gpt_pool_follows_the_review_slot_ceiling(cfg, conn):
    assert lanepool.limit(conn, cfg, "gpt") == cfg.max_concurrent + cfg.priority_overflow
    assert lanepool.limit(conn, cfg, "muse") == 8
    assert lanepool.limit(conn, cfg, "unknown") is None


def test_running_claude_lane_stops_when_asked(tmp_path):
    """The real runner: a lane asked to stop is terminated mid-run, not left to finish."""
    script = tmp_path / "claude"
    script.write_text("#!/bin/sh\ncat >/dev/null\nexec sleep 30\n")
    script.chmod(0o755)
    flag = threading.Event()
    spec = LaneSpec(
        role="general",
        agent="a",
        model="m",
        effort="high",
        prompt="p",
        cwd=tmp_path,
        add_dir=tmp_path,
        timeout_seconds=60,
        claude_bin=str(script),
        should_stop=flag.is_set,
    )
    threading.Timer(0.3, flag.set).start()
    t0 = time.monotonic()
    old = lane_mod.STOP_POLL_SECONDS
    lane_mod.STOP_POLL_SECONDS = 0.1
    try:
        res = lane_mod.run_claude_lane(spec, tmp_path / "art", tmp_path)
    finally:
        lane_mod.STOP_POLL_SECONDS = old
    assert res.cancelled and not res.timed_out and not res.ok
    assert time.monotonic() - t0 < 10
    meta = json.loads((tmp_path / "art" / "lane-meta.json").read_text())
    assert meta["cancelled"] is True


def test_gpt_reviewers_leave_the_top_slot_to_verifiers(cfg, conn, gh, lanes):
    """At the static ceiling (3 gpt streams) Phase 2 runs two reviewers at a time: the third
    slot stays free for verifiers and side lanes, which are on some run's critical path."""
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier()
    runner = Overlap(lanes, hold=0.3)

    def spy(spec, art, worktree):
        # count only the gpt (Phase-2) reviewers; Phase 1 runs on glm with its own pool
        return (
            runner(spec, art, worktree)
            if spec.model.startswith("gpt")
            else lanes(spec, art, worktree)
        )

    _, status = _run(cfg, conn, gh, spy)
    assert status == RunStatus.DONE
    assert runner.peak == 2
    slot_dir = cfg.work_dir / "lane-slots"
    # with both reviewer slots held elsewhere, a verifier still gets the reserved one ...
    reviewer = lanepool.RANK_LIVE_REVIEWER
    held = [
        lanepool.acquire(slot_dir, "gpt", lambda: 3, lambda: False, rank=reviewer) for _ in range(2)
    ]
    fd = lanepool.acquire(slot_dir, "gpt", lambda: 3, lambda: True, rank=lanepool.RANK_LIVE_SIDE)
    assert fd is not None
    # ... and a third reviewer does not
    assert lanepool.acquire(slot_dir, "gpt", lambda: 3, lambda: True, rank=reviewer) is None
    for f in (fd, *held):
        lanepool.release(f)


def _ticket(line: Path, rank: int, run_id: int, pid: int, tid: int, at: int = 0) -> Path:
    t = line / f"{rank}-{run_id:012d}-{at:020d}-{pid}-{tid}"
    t.touch()
    return t


def test_every_lane_lines_up_side_lanes_first_priority_first_older_run_first(tmp_path):
    """One line per pool for every lane: side lanes (verifiers, triage, ...) before reviewers,
    priority before live, an older run before a newer one, then arrival; audits last."""
    assert [lanepool.rank_of(q, side=True) for q in ("priority", "live", "audit")] == [1, 2, 5]
    assert [lanepool.rank_of(q, side=False) for q in ("priority", "live", "audit")] == [3, 4, 6]
    line = tmp_path / "glm.line"
    line.mkdir()
    pid = os.getpid()
    expected = [
        _ticket(line, lanepool.RANK_PRIORITY_SIDE, 99, pid, 1, at=9),
        _ticket(line, lanepool.RANK_LIVE_SIDE, 50, pid, 2, at=8),
        _ticket(line, lanepool.RANK_PRIORITY_REVIEWER, 70, pid, 3, at=7),
        _ticket(line, lanepool.RANK_LIVE_REVIEWER, 3, pid, 4, at=6),  # older run first ...
        _ticket(line, lanepool.RANK_LIVE_REVIEWER, 9, pid, 5, at=1),  # ... then arrival
        _ticket(line, lanepool.RANK_LIVE_REVIEWER, 9, pid, 6, at=2),
        _ticket(line, lanepool.RANK_AUDIT_SIDE, 1, pid, 7, at=0),
        _ticket(line, lanepool.RANK_AUDIT_REVIEWER, 0, pid, 8, at=0),
    ]
    (line / "2-00000000001-1234-1").touch()  # not a ticket of this format: ignored
    assert lanepool.waiting_line(tmp_path, "glm") == [t.name for t in expected]
    # a fresh ticket sorts by the same rules
    assert lanepool.ticket_name(lanepool.RANK_LIVE_SIDE, 7) < expected[2].name


def test_take_rules_head_any_express_top_only(tmp_path):
    """The head takes any slot its kind allows (a reviewer never the top one while the pool
    has more); a side lane with only reviewers ahead may take the top slot and nothing else;
    everyone else waits."""
    r1, r2 = "4-000000000001-00000000000000000001-1-1", "4-000000000002-00000000000000000002-1-2"
    s1, s2 = "5-000000000003-00000000000000000003-1-3", "5-000000000004-00000000000000000004-1-4"
    may = lanepool._may_take
    assert list(may([s1], s1, 3, True) or []) == [2, 1, 0], "side head: the top slot first"
    assert list(may([r1], r1, 3, False) or []) == [0, 1], "reviewer head: never the top"
    assert list(may([r1], r1, 1, False) or []) == [0], "a one-slot pool has no reserved slot"
    assert list(may([r1, r2, s1], s1, 3, True) or []) == [2], "express lane: the top only"
    assert may([r1, s1, s2], s2, 3, True) is None, "a side lane ahead goes first"
    assert may([r1, r2], r2, 3, False) is None, "a reviewer waits for its turn"
    assert may([r1, s1], s1, 1, True) is None, "no express lane without a reserved slot"


def test_a_verifier_passes_a_head_reviewer_stuck_waiting(tmp_path):
    """The head of the gpt line is another worker's reviewer, waiting because every reviewer
    slot is taken; a verifier behind it takes the free top slot, a reviewer behind it waits."""
    line = tmp_path / "gpt.line"
    line.mkdir()
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        _ticket(line, lanepool.RANK_LIVE_REVIEWER, 1, sleeper.pid, 1)
        reviewer = lanepool.RANK_LIVE_REVIEWER
        held = [lanepool._try_slots(tmp_path, "gpt", range(i, i + 1)) for i in (0, 1)]
        polls = iter(range(3))

        def give_up_soon() -> bool:
            return next(polls, None) is None

        lanepool.POLL_SECONDS, old = 0.01, lanepool.POLL_SECONDS
        try:
            fd = lanepool.acquire(
                tmp_path, "gpt", lambda: 3, lambda: True, rank=lanepool.RANK_AUDIT_SIDE, run_id=9
            )
            assert fd is not None, "the express lane"
            assert lanepool.acquire(tmp_path, "gpt", lambda: 3, give_up_soon, rank=reviewer) is None
        finally:
            lanepool.POLL_SECONDS = old
        assert [f.name.split("-")[0] for f in line.iterdir()] == ["4"], "waiters took their tickets"
        for f in (fd, *held):
            assert f is not None
            lanepool.release(f)
    finally:
        sleeper.kill()
        sleeper.wait()


def test_a_ticket_left_by_a_dead_process_is_pruned(tmp_path):
    line = tmp_path / "glm.line"
    line.mkdir()
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    stale = _ticket(line, lanepool.RANK_PRIORITY_SIDE, 1, gone.pid, 1)
    assert lanepool.waiting_line(tmp_path, "glm") == [], "readers skip it without deleting it"
    assert stale.exists()
    fd = lanepool.acquire(
        tmp_path, "glm", lambda: 4, lambda: True, rank=lanepool.RANK_AUDIT_REVIEWER
    )
    assert fd is not None and not stale.exists()
    lanepool.release(fd)
    assert list(line.iterdir()) == []


def test_stall_clock_counts_only_time_with_lanes_waiting_and_none_running():
    t = [0.0]
    c = lanepool.StallClock(lambda: t[0])
    assert c.wait_started() == 0.0  # stalled from 0
    t[0] = 10
    assert c.wait_ended(started=True) == 10  # a lane runs: the stall ends
    t[0] = 15
    assert c.wait_started() == 0.0  # waiting beside a running lane is not a stall
    t[0] = 40
    assert c.run_ended() == 0.0  # nothing runs any more, one waits: stalled from 40
    t[0] = 100
    assert c.flush() == 60  # a long stall is credited while it lasts ...
    t[0] = 130
    assert c.flush() == 0.0  # ... in slices of at least a minute
    t[0] = 150
    assert c.wait_ended(started=False) == 50  # given up: nothing waits, the rest is credited
    assert c.run_started() == 0.0 and c.run_ended() == 0.0


def test_slot_waits_push_the_run_deadline_out(cfg, conn, tmp_path, monkeypatch):
    """A run admitted without a run cap may wait long for model slots; that time must not
    count against its deadline. Time a lane spends running does."""
    from reviewsys.db import parse_ts

    monkeypatch.setattr(lanepool, "POLL_SECONDS", 0.01)
    cfg = dataclasses.replace(cfg, lane_pools={"muse": 1})
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.MENTION)
    (rid,) = schedule(conn, cfg, spawn=False)

    def deadline():
        row = conn.execute("SELECT deadline_at FROM runs WHERE id=?", (rid,)).fetchone()
        return parse_ts(row["deadline_at"])

    before = deadline()
    slot_dir = cfg.work_dir / "lane-slots"
    t = [1000.0]

    def inner(spec, art, worktree):
        t[0] += 300  # five minutes running: not a stall
        return LaneResult(exit_code=0, stdout="", stderr="", duration_s=300)

    gated = lanepool.gated(inner, lambda: conn, cfg, lambda: False, "live", rid, clock=lambda: t[0])
    spec = _muse_spec(tmp_path, role="general", effort="high", parallel=True)

    def lane_waits(seconds: float) -> None:
        """Run the lane while another run holds the only slot for `seconds` (fake clock)."""
        held = lanepool.acquire(slot_dir, "muse", lambda: 1, lambda: False)

        def other_run_finishes():
            # only once our lane is in line: a fixed delay raced a slow runner, which let the
            # clock jump before the lane started waiting (no stall, no credit)
            deadline_at = time.monotonic() + 10
            while not lanepool.waiting_line(slot_dir, "muse") and time.monotonic() < deadline_at:
                time.sleep(0.005)
            t[0] += seconds
            lanepool.release(held)

        other = threading.Thread(target=other_run_finishes)
        other.start()
        assert gated(spec, tmp_path, tmp_path).ok
        other.join()

    lane_waits(600)  # ten minutes in which the lane only waited
    assert (deadline() - before).total_seconds() == 600
    # the credit is capped (two run timeouts): a line that never moves still times out
    cap = cfg.run_timeout_minutes * 60 * lanepool.STALL_CREDIT_MAX_FACTOR
    lane_waits(10 * cap)
    assert (deadline() - before).total_seconds() == cap


def test_parallel_lanes_on_a_dry_pool_flip_the_run_once(
    conn, gh, lanes, skills_dir, tmp_path, monkeypatch
):
    """Three Phase-2 lanes on astra die on the same 429 together: the run enters degraded
    mode once and every lane finishes on the stand-in."""
    import test_degraded as td

    c = dataclasses.replace(td._cfg_with(skills_dir, tmp_path), max_concurrent=4)
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier()
    dead = (
        "API Error: Request rejected (429) · All credentials for model gpt-6-astra are cooling down"
    )
    barrier = threading.Barrier(3)

    def runner(spec, art, worktree):
        if spec.model == "gpt-6-astra" and spec.role in REVIEWER_ROLES:
            barrier.wait(timeout=10)  # all three are on the dry pool at the same moment
            return LaneResult(exit_code=1, stdout="", stderr=dead, duration_s=1)
        return lanes(spec, art, worktree)

    with tx(conn):
        enqueue_head(conn, c, "dashpay/platform", 1, HEAD, Trigger.MENTION)
    (rid,) = schedule(conn, c, spawn=False)
    status = worker.main(c, conn, rid, gh=gh, lane_runner=runner, heartbeat=False, prober=td.FINE)
    assert status == RunStatus.DONE
    kinds = [r["kind"] for r in conn.execute("SELECT kind FROM events WHERE run_id=?", (rid,))]
    assert kinds.count("degraded.entered_midrun") == 1
    done = conn.execute(
        "SELECT role, model FROM lanes WHERE run_id=? AND phase='phase2' AND status='completed'",
        (rid,),
    ).fetchall()
    assert sorted(r["role"] for r in done) == sorted(REVIEWER_ROLES)
    assert {r["model"] for r in done} == {td.MUSE}
