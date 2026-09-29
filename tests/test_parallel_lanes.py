"""The reviewer lanes of a phase run side by side, bounded per run and per model pool."""

from __future__ import annotations

import dataclasses
import json
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


def test_a_failed_lane_stops_its_siblings_and_fails_the_run(cfg, conn, gh, lanes):
    """One reviewer failing for good fails the phase, as before; the lanes still running are
    stopped (and recorded as cancelled) instead of spending quota on a run that is retried."""
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier()
    lanes.timeout_roles = {"general"}
    stopped: list[str] = []

    def runner(spec: LaneSpec, art: Path, worktree: Path) -> LaneResult:
        if spec.role in ("always-on", "security-auditor"):
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
    assert "phase1/general lane failed twice" in reason, "the real failure, not a stopped lane"
    rows = {
        r["role"]: r["status"]
        for r in conn.execute(
            "SELECT role, status FROM lanes WHERE run_id=? AND phase='phase1'", (rid,)
        )
    }
    assert rows == {"general": "failed", "always-on": "cancelled", "security-auditor": "cancelled"}
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


def test_gated_runner_waits_for_a_slot_and_honours_cancel(cfg, conn):
    cfg = dataclasses.replace(cfg, lane_pools={"muse": 1})
    slot_dir = cfg.work_dir / "lane-slots"
    held = lanepool.acquire(slot_dir, "muse", lambda: 1, lambda: False)
    ran: list[str] = []

    def inner(spec, art, worktree):
        ran.append(spec.role)
        return LaneResult(exit_code=0, stdout="", stderr="", duration_s=1)

    stop = threading.Event()
    gated = lanepool.gated(
        inner, lambda: conn, cfg, run_stopped=stop.is_set, reviewer_rank=lanepool.RANK_LIVE
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
    res = gated(spec, Path("."), Path("."))
    assert res.cancelled and not res.started and ran == []
    lanepool.release(held)
    stop.clear()
    assert not gated(spec, Path("."), Path(".")).cancelled and ran == ["general"]
    # an ungated family never touches the slot files
    assert gated(dataclasses.replace(spec, model="other-model"), Path("."), Path(".")).ok


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
    held = [lanepool.acquire(slot_dir, "gpt", lambda: 3, lambda: False, rank=2) for _ in range(2)]
    fd = lanepool.acquire(slot_dir, "gpt", lambda: 3, lambda: True)
    assert fd is not None
    # ... and a third reviewer does not
    assert lanepool.acquire(slot_dir, "gpt", lambda: 3, lambda: True, rank=2) is None
    for f in (fd, *held):
        lanepool.release(f)


def test_reviewer_lanes_wait_in_line(tmp_path):
    """Only the head of the line takes a slot, and priority lines up before live and live
    before audits; a ticket left by a dead process does not block the line."""
    wait_dir = tmp_path / "glm.wait"
    wait_dir.mkdir()
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        ahead = wait_dir / f"{lanepool.RANK_PRIORITY}-{0:020d}-{sleeper.pid}-1"
        ahead.touch()
        polls = iter(range(5))

        def give_up_soon() -> bool:
            return next(polls, None) is None

        # a free slot, but a priority lane from another worker is ahead in line
        assert lanepool.acquire(tmp_path, "glm", lambda: 4, give_up_soon, rank=2) is None
        assert list(wait_dir.iterdir()) == [ahead], "the waiter took its ticket with it"
        # an audit ticket queued long before still goes after a live lane
        audit_ticket = wait_dir / f"{lanepool.RANK_AUDIT}-{0:020d}-{sleeper.pid}-2"
        audit_ticket.touch()
        ahead.unlink()
        fd = lanepool.acquire(tmp_path, "glm", lambda: 4, lambda: True, rank=2)
        assert fd is not None
        lanepool.release(fd)
    finally:
        sleeper.kill()
        sleeper.wait()
    # its process is gone: the stale ticket is dropped and the line moves
    stale = wait_dir / f"{lanepool.RANK_PRIORITY}-{0:020d}-{sleeper.pid}-3"
    stale.touch()
    fd = lanepool.acquire(tmp_path, "glm", lambda: 4, lambda: True, rank=2)
    assert fd is not None and not stale.exists()
    lanepool.release(fd)


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
