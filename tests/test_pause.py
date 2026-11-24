"""Pause mode: a degraded block without stand-ins pauses reviewing while the primary models
are out, instead of running any lane on another model."""

from __future__ import annotations

import contextlib
import json
import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest
import test_degraded as td

from reviewsys import config as cfg_mod
from reviewsys import db as db_mod
from reviewsys import degraded, lanepool
from reviewsys.daemon import Daemon
from reviewsys.db import fmt_ts, kv_get, kv_set, now_dt, parse_ts, tx
from reviewsys.ingest import enqueue_head
from reviewsys.lane import LaneResult, LaneSpec
from reviewsys.models import RunStatus, Trigger
from reviewsys.publish import DEGRADED_BADGE
from reviewsys.scheduler import schedule

PAUSE_BLOCK = {"sentinel": "gpt-6-astra", "label": "paused"}
QUOTA = "API Error: Request rejected (429) · The usage limit has been reached"
fake_git = td.fake_git


@pytest.fixture
def paused_cfg(skills_dir, tmp_path):
    return td._cfg_with(skills_dir, tmp_path, block=PAUSE_BLOCK)


@pytest.fixture(autouse=True)
def fast_poll(monkeypatch):
    monkeypatch.setattr(lanepool, "POLL_SECONDS", 0.01)


def _pause(conn):
    degraded.record_probe(conn, quota_exhausted=True, reason="`gpt-6-astra` unavailable: quota")


def _resume(conn):
    degraded.record_probe(conn, quota_exhausted=False, reason="gpt-6-astra answered")


def _spec(tmp_path, model, **kw):
    return LaneSpec(
        role="verifier",
        agent="a",
        model=model,
        effort="high",
        prompt="p",
        cwd=tmp_path,
        add_dir=tmp_path,
        timeout_seconds=5,
        claude_bin="claude",
        **kw,
    )


def _conns(cfg):
    """A connection per thread, as the worker gives its lanes."""
    local = threading.local()

    def conn():
        if not hasattr(local, "c"):
            local.c = db_mod.connect(cfg.db_path)
        return local.c

    return conn


def _ok(*_):
    return LaneResult(exit_code=0, stdout="", stderr="", duration_s=1)


def _until(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.005)


def _state(art: Path) -> str | None:
    try:
        return json.loads((art / lanepool.LANE_STATE).read_text())["state"]
    except (OSError, ValueError):
        return None


# ---- config, snapshot, wording ----


def test_a_sentinel_only_block_is_pause_mode(cfg, conn, paused_cfg, skills_dir, tmp_path):
    d = paused_cfg.policy.degraded
    assert d is not None and d.pauses and d.substitutes == {} and d.label == "paused"
    lm = cfg_mod.LaneModel(agent="phase2-reviewer", model="gpt-6-astra", effort="max")
    assert d.resolve(lm) is lm, "no stand-in, ever"
    c = td._cfg_with(skills_dir, tmp_path, block={"sentinel": "gpt-6-astra", "substitutes": {}})
    assert c.policy.degraded is not None and c.policy.degraded.pauses
    assert not td._cfg_with(skills_dir, tmp_path).policy.degraded.pauses
    for bad in (
        {"label": "paused"},  # no sentinel
        {"sentinel": "gpt-6-astra", "phase1_effort_cap": "ultra"},
        {"sentinel": "gpt-6-astra", "substitutes": {"gpt-5.6-sol": "muse-x"}},
    ):
        with pytest.raises(ValueError):
            td._cfg_with(skills_dir, tmp_path, block=bad)
    snap = degraded.snapshot(conn, paused_cfg)
    assert snap["mode"] == "pause" and "substitutes" not in snap and not snap["active"]
    assert degraded.snapshot(conn, td._cfg_with(skills_dir, tmp_path))["mode"] == "standin"


def test_paused_reads_the_stored_state_and_never_probes(conn, paused_cfg, skills_dir, tmp_path):
    assert not degraded.paused(conn, paused_cfg), "no reading yet"
    _pause(conn)
    assert degraded.paused(conn, paused_cfg) and degraded.snapshot(conn, paused_cfg)["active"]
    assert not degraded.paused(conn, td._cfg_with(skills_dir, tmp_path)), "stand-in mode"
    stale = fmt_ts(now_dt() - degraded.SNAPSHOT_MAX_AGE - timedelta(minutes=1))
    with tx(conn):
        kv_set(conn, degraded.KV_PROBE, json.dumps({"at": stale, "quota_exhausted": True}))
    assert not degraded.paused(conn, paused_cfg), "a reading too old to trust"
    degraded.force(conn, "on")
    assert degraded.paused(conn, paused_cfg)
    _pause(conn)
    degraded.force(conn, "off")
    assert not degraded.paused(conn, paused_cfg)


def test_transition_says_reviews_are_paused(conn, gh, notifier, paused_cfg):
    on = degraded.State(True, "`gpt-6-astra` unavailable: quota", "probe", pause=True)
    off = degraded.State(False, "gpt-6-astra answered", "probe", pause=True)
    text = degraded.transition(conn, on) or ""
    assert text.startswith("PAUSING reviews: `gpt-6-astra` unavailable: quota")
    assert "paused until the primary models answer again" in text and "stand-in" not in text
    assert on.describe() == "paused (`gpt-6-astra` unavailable: quota; probe)"
    assert "resuming reviews" in (degraded.transition(conn, off) or "")
    d = Daemon(paused_cfg, conn, gh=gh, notifier=notifier, spawn=False, prober=td.EXHAUSTED)
    assert d.t_degraded()["pause"] is True
    assert "PAUSING reviews" in notifier.sent[-1][1]


# ---- the lane gate ----


def test_a_primary_lane_waits_while_paused_and_others_start_at_once(conn, paused_cfg, tmp_path):
    ran: list[str] = []
    gated = lanepool.gated(
        lambda spec, a, w: (ran.append(spec.model), _ok())[1],
        _conns(paused_cfg),
        paused_cfg,
        run_stopped=lambda: False,
        queue="live",
        run_id=0,
    )
    _pause(conn)
    art = tmp_path / "gpt"
    out: list[LaneResult] = []
    t = threading.Thread(
        target=lambda: out.append(gated(_spec(tmp_path, "gpt-5.6-sol"), art, tmp_path))
    )
    t.start()
    _until(lambda: _state(art) == "waiting")
    assert gated(_spec(tmp_path, "glm-5.3-flash"), tmp_path / "glm", tmp_path).ok
    time.sleep(0.05)
    assert ran == ["glm-5.3-flash"], "the gpt lane has not started"
    _resume(conn)
    t.join(10)
    assert out[0].ok and ran == ["glm-5.3-flash", "gpt-5.6-sol"]


def test_the_pause_wait_stops_on_cancel(conn, paused_cfg, tmp_path):
    stop = threading.Event()
    gated = lanepool.gated(
        _ok, _conns(paused_cfg), paused_cfg, run_stopped=stop.is_set, queue="live", run_id=0
    )
    _pause(conn)
    art = tmp_path / "art"
    out: list[LaneResult] = []
    t = threading.Thread(
        target=lambda: out.append(gated(_spec(tmp_path, "gpt-6-astra"), art, tmp_path))
    )
    t.start()
    _until(lambda: _state(art) == "waiting")
    stop.set()
    t.join(2)
    assert not t.is_alive() and out[0].cancelled and not out[0].started
    assert not (art / lanepool.LANE_STATE).exists()
    own = threading.Event()
    own.set()
    res = gated(_spec(tmp_path, "gpt-6-astra", should_stop=own.is_set), art, tmp_path)
    assert res.cancelled and not res.started, "the lane's own stop too"


def test_the_pause_wait_extends_the_deadline_past_the_stall_cap(conn, paused_cfg, tmp_path):
    with tx(conn):
        enqueue_head(conn, paused_cfg, "dashpay/platform", 1, td.HEAD, Trigger.MENTION)
    (rid,) = schedule(conn, paused_cfg, spawn=False)

    def deadline():
        row = conn.execute("SELECT deadline_at FROM runs WHERE id=?", (rid,)).fetchone()
        return parse_ts(row["deadline_at"])

    before = deadline()
    clock = [1000.0]
    gated = lanepool.gated(
        _ok, _conns(paused_cfg), paused_cfg, lambda: False, "live", rid, clock=lambda: clock[0]
    )
    _pause(conn)
    art = tmp_path / "art"
    cap = paused_cfg.run_timeout_minutes * 60 * lanepool.STALL_CREDIT_MAX_FACTOR

    def outage_ends():
        _until(lambda: _state(art) == "waiting")
        clock[0] += 10 * cap
        with contextlib.closing(db_mod.connect(paused_cfg.db_path)) as c:
            _resume(c)

    other = threading.Thread(target=outage_ends)
    other.start()
    assert gated(_spec(tmp_path, "gpt-6-astra"), art, tmp_path).ok
    other.join()
    assert (deadline() - before).total_seconds() == 10 * cap


# ---- runs ----


@pytest.fixture
def outage(monkeypatch, lanes, paused_cfg):
    """The pool comes back the first time a lane sleeps while the system is paused: what was
    dead answers and the stored state clears. Returns the lane calls, in order, with `back`
    at that moment, and the runner that records them."""
    log: list[str] = []
    real_sleep = time.sleep

    def sleep(seconds):
        if "back" not in log:
            c = db_mod.connect(paused_cfg.db_path)  # lanes sleep on their own threads
            try:
                if degraded.paused(c, paused_cfg):
                    log.append("back")
                    lanes.dead_models.clear()
                    _resume(c)
            finally:
                c.close()
        real_sleep(0.001)

    monkeypatch.setattr(lanepool.time, "sleep", sleep)

    def runner(spec, art, wt):
        log.append(f"{spec.role}:{spec.model}")
        return lanes(spec, art, wt)

    return log, runner


def test_a_quota_failure_pauses_and_retries_on_the_same_model(conn, gh, lanes, paused_cfg, outage):
    log, runner = outage
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = td._verifier("APPROVE")
    lanes.dead_models = {"gpt-5.6-sol"}
    lanes.dead_stderr = QUOTA
    rid, status = td._run(paused_cfg, conn, gh, runner, prober=td.FINE)
    assert status == RunStatus.DONE
    gate = [i for i, e in enumerate(log) if e == "verifier:gpt-5.6-sol"]
    assert len(gate) == 2, "failed once, then once more on the same model"
    assert gate[0] < log.index("back") < gate[1], "the retry waited for the pool"
    assert td.MUSE not in {s.model for s in lanes.calls}, "no stand-in"
    assert conn.execute("SELECT degraded FROM runs WHERE id=?", (rid,)).fetchone()[0] == 0
    kinds = [r["kind"] for r in conn.execute("SELECT kind FROM events WHERE run_id=?", (rid,))]
    assert "degraded.paused_midrun" in kinds
    assert "degraded.entered_midrun" not in kinds and "degraded.run" not in kinds
    review = gh.posted_reviews[0]
    assert review["event"] == "APPROVE" and DEGRADED_BADGE not in review["body"]
    assert "Phase 2 only (Phase 1 failed)" not in review["body"]


def test_a_side_lane_quota_failure_retries_instead_of_falling_back(
    conn, gh, lanes, paused_cfg, outage
):
    log, runner = outage
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = td._verifier()
    lanes.dead_models = {"gpt-6-astra"}
    lanes.dead_stderr = QUOTA
    rid, status = td._run(paused_cfg, conn, gh, runner, prober=td.FINE)
    assert status == RunStatus.DONE
    prep = [i for i, e in enumerate(log) if e.startswith("prep:")]
    assert [log[i] for i in prep] == ["prep:gpt-6-astra"] * 3, "its two attempts, then a third"
    assert prep[1] < log.index("back") < prep[2]
    kinds = [r["kind"] for r in conn.execute("SELECT kind FROM events WHERE run_id=?", (rid,))]
    assert "triage.degraded" not in kinds and "select.degraded" not in kinds
    assert "degraded.paused_midrun" in kinds
    assert conn.execute("SELECT degraded FROM runs WHERE id=?", (rid,)).fetchone()[0] == 0


# ---- the scheduler ----


def test_the_scheduler_holds_the_queue_while_paused(conn, paused_cfg):
    with tx(conn):
        for n in (1, 2):
            enqueue_head(conn, paused_cfg, "dashpay/platform", n, str(n) * 40, Trigger.MENTION)
    _pause(conn)
    assert schedule(conn, paused_cfg, spawn=False) == []
    assert schedule(conn, paused_cfg, spawn=False) == []
    queued = conn.execute("SELECT COUNT(*) FROM heads WHERE status='queued'").fetchone()[0]
    assert queued == 2 and conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    kinds = [r["kind"] for r in conn.execute("SELECT kind FROM events")]
    assert kinds.count("schedule.paused") == 1, "one event per pause, not per tick"
    _resume(conn)
    assert len(schedule(conn, paused_cfg, spawn=False)) == 2
    assert kv_get(conn, "schedule.paused_at") is None
    assert [r["kind"] for r in conn.execute("SELECT kind FROM events")].count(
        "schedule.resumed"
    ) == 1
