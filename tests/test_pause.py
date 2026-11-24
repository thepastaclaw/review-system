"""Pause mode: a degraded block without stand-ins pauses reviewing while the primary models
are out, instead of running any lane on another model."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest
import test_degraded as td

from reviewsys import config as cfg_mod
from reviewsys import db as db_mod
from reviewsys import degraded, lanepool, status, worker
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
    monkeypatch.setattr(lanepool, "PAUSE_READ_SECONDS", 0.0)


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


def test_a_lane_in_line_when_the_pause_begins_waits_it_out(conn, paused_cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(lanepool, "limit", lambda *_: 1)
    ran: list[str] = []
    first_may_end = threading.Event()

    def runner(spec, a, w):
        ran.append(a.name)
        if a.name == "first":
            first_may_end.wait(10)
        return _ok()

    gated = lanepool.gated(
        runner,
        _conns(paused_cfg),
        paused_cfg,
        run_stopped=lambda: False,
        queue="live",
        run_id=0,
    )
    threads = [
        threading.Thread(
            target=lambda n=n: gated(_spec(tmp_path, "gpt-5.6-sol"), tmp_path / n, tmp_path)
        )
        for n in ("first", "second")
    ]
    threads[0].start()
    _until(lambda: ran == ["first"])
    threads[1].start()
    _until(lambda: _state(tmp_path / "second") == "waiting")
    time.sleep(0.05)
    _pause(conn)
    first_may_end.set()
    threads[0].join(10)
    time.sleep(0.2)
    assert ran == ["first"], "the queued lane took the freed slot during the pause"
    _resume(conn)
    threads[1].join(10)
    assert ran == ["first", "second"]


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
    c = dataclasses.replace(paused_cfg, lane_pools={"muse": 1})
    gated = lanepool.gated(_ok, _conns(c), c, lambda: False, "live", rid, clock=lambda: clock[0])
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
    # the capped allowance for slot waits is still all there
    slot_dir = c.work_dir / "lane-slots"
    held = lanepool.acquire(slot_dir, "muse", lambda: 1, lambda: False)

    def slot_frees():
        _until(lambda: bool(lanepool.waiting_line(slot_dir, "muse")))
        clock[0] += 600
        lanepool.release(held)

    other = threading.Thread(target=slot_frees)
    other.start()
    assert gated(_spec(tmp_path, td.MUSE), tmp_path / "muse", tmp_path).ok
    other.join()
    assert (deadline() - before).total_seconds() == 10 * cap + 600


def test_a_pause_wait_beside_a_running_lane_is_not_credited(conn, paused_cfg, tmp_path):
    """Phase 1 still running while the thread lane waits for the paused pool: the run is
    working, so its deadline stays where it is."""
    with tx(conn):
        enqueue_head(conn, paused_cfg, "dashpay/platform", 1, td.HEAD, Trigger.MENTION)
    (rid,) = schedule(conn, paused_cfg, spawn=False)
    before = conn.execute("SELECT deadline_at FROM runs WHERE id=?", (rid,)).fetchone()[0]
    clock = [1000.0]
    running, finish = threading.Event(), threading.Event()

    def inner(spec, art, wt):
        if spec.model == "glm-5.3-flash":
            running.set()
            finish.wait(10)
        return _ok()

    gated = lanepool.gated(
        inner, _conns(paused_cfg), paused_cfg, lambda: False, "live", rid, clock=lambda: clock[0]
    )
    phase1 = threading.Thread(
        target=gated, args=(_spec(tmp_path, "glm-5.3-flash"), tmp_path / "glm", tmp_path)
    )
    phase1.start()
    running.wait(10)
    _pause(conn)
    art = tmp_path / "gpt"
    lane = threading.Thread(target=gated, args=(_spec(tmp_path, "gpt-6-astra"), art, tmp_path))
    lane.start()
    _until(lambda: _state(art) == "waiting")
    clock[0] += 3600
    _resume(conn)
    lane.join(10)
    finish.set()
    phase1.join(10)
    after = conn.execute("SELECT deadline_at FROM runs WHERE id=?", (rid,)).fetchone()[0]
    assert after == before


def test_stall_clock_keeps_pause_stretches_apart_and_counts_each_once():
    t = [0.0]
    c = lanepool.StallClock(lambda: t[0])
    assert c.pause_started() == 0.0
    t[0] = 100
    assert c.flush() == 0.0 and c.uncapped() == 100, "a pause is credited uncapped"
    t[0] = 150
    assert c.run_started() == 0.0 and c.uncapped() == 50
    t[0] = 400
    assert c.run_ended() == 0.0 and c.uncapped() == 0.0, "nothing while a lane runs"
    t[0] = 450
    assert c.wait_started() == 0.0, "a slot wait inside a pause stretch is not counted twice"
    t[0] = 500
    assert c.pause_ended() == 0.0 and c.uncapped() == 100
    t[0] = 530
    assert c.wait_ended(started=False) == 30 and c.uncapped() == 0.0


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


def test_the_watchdog_does_not_call_a_paused_queue_stuck(conn, paused_cfg):
    with tx(conn):
        enqueue_head(conn, paused_cfg, "dashpay/platform", 1, td.HEAD, Trigger.MENTION)
    assert status.watchdog(conn, paused_cfg)["stuck"]
    _pause(conn)
    assert not status.watchdog(conn, paused_cfg)["stuck"]


# ---- the stored state over an outage ----


def test_the_pause_has_no_gap_when_a_lane_hold_ends(conn, paused_cfg, monkeypatch):
    """The daemon cannot re-probe while a hold is in force: the reading must stay trusted
    after the hold ends until that re-probe replaces it, and then follow the re-probe."""
    t0 = now_dt()
    at = [t0]
    monkeypatch.setattr(degraded, "now_dt", lambda: at[0])

    def when(minutes):
        at[0] = t0 + timedelta(minutes=minutes)

    for prober, still_out in ((td.EXHAUSTED, True), (td.FINE, False)):
        when(0)
        _pause(conn)
        when(0.5)
        degraded.record_probe(conn, quota_exhausted=True, reason="lane: spent", hold=True)
        for m in (10, 19, 20.6, 21):  # the hold ends at 20.5
            when(m)
            assert degraded.paused(conn, paused_cfg), m
            assert degraded.snapshot(conn, paused_cfg)["active"], m
        probes: list[str] = []
        degraded.detect(
            conn,
            paused_cfg,
            prober=lambda m, seen=probes, ask=prober: (seen.append(m), ask(m))[1],
            refresh=True,
        )
        assert probes == ["gpt-6-astra"], "the daemon's next pass probes"
        for m in (21, 22, 30):
            when(m)
            assert degraded.paused(conn, paused_cfg) is still_out, m


def _ctx(cfg, conn, tmp_path):
    return worker.RunContext(
        cfg=cfg,
        main_conn=conn,
        gh=None,
        run_id=1,
        head_id=1,
        repo="dashpay/platform",
        number=1,
        sha=td.HEAD,
        token="t",
        run_dir=tmp_path,
    )


def test_only_a_spent_quota_holds_the_pause(conn, paused_cfg, tmp_path, monkeypatch):
    ctx = _ctx(paused_cfg, conn, tmp_path)
    fail = functools.partial(
        worker._pause_on_quota_failure, ctx, "gpt-5.6-sol", upstream="429", detail="lane"
    )
    blip = "lane exit 1: API Error: 429 rate_limit_error"

    def stored():
        return json.loads(kv_get(conn, degraded.KV_PROBE))

    assert fail(error=blip)
    assert stored()["quota_exhausted"] and "hold_until" not in stored()
    t0 = now_dt()
    monkeypatch.setattr(degraded, "now_dt", lambda: t0 + degraded.PROBE_CACHE * 3)
    assert degraded.paused(conn, paused_cfg), "a short rate limit pauses until the next probe"
    degraded.detect(conn, paused_cfg, prober=td.FINE, refresh=True)
    assert not degraded.paused(conn, paused_cfg), "... and the probe lifts it"
    monkeypatch.setattr(degraded, "now_dt", now_dt)
    assert fail(error=QUOTA)
    hold = stored()["hold_until"]
    assert fail(error=blip) and stored()["hold_until"] == hold, "a blip keeps the spent hold"
    degraded.detect(conn, paused_cfg, prober=td.FINE, refresh=True)
    assert degraded.paused(conn, paused_cfg), "a passing probe cannot lift a spent quota's hold"
    kinds = [r["kind"] for r in conn.execute("SELECT kind FROM events")]
    assert kinds.count("degraded.paused_midrun") == 3
    assert not worker._pause_on_quota_failure(
        ctx, "glm-5.3-flash", error=QUOTA, upstream="429", detail="lane"
    ), "another family never pauses"
    assert not fail(error="lane exit 1: segfault"), "nor does a failure that is not quota"
    degraded.force(conn, "off")
    assert not fail(error=QUOTA), "forced off: the failure is an ordinary one"


def test_a_quota_failure_on_another_family_does_not_pause(conn, gh, lanes, paused_cfg):
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = td._verifier()
    lanes.dead_models = {"glm-5.3-flash"}
    lanes.dead_stderr = QUOTA
    rid, run_status = td._run(paused_cfg, conn, gh, lanes, prober=td.FINE)
    assert run_status == RunStatus.DONE
    kinds = [r["kind"] for r in conn.execute("SELECT kind FROM events WHERE run_id=?", (rid,))]
    assert "degraded.paused_midrun" not in kinds
    assert not degraded.paused(conn, paused_cfg)


def test_forced_off_a_primary_quota_failure_fails_the_lane_as_usual(conn, gh, lanes, paused_cfg):
    degraded.force(conn, "off")
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = td._verifier()
    lanes.dead_models = {"gpt-5.6-sol"}
    lanes.dead_stderr = QUOTA
    rid, run_status = td._run(paused_cfg, conn, gh, lanes, prober=td.FINE)
    assert run_status == RunStatus.DONE
    assert [s.model for s in lanes.calls if s.role == "verifier"].count("gpt-5.6-sol") == 2
    assert "Phase 2 only (Phase 1 failed)" in gh.posted_reviews[0]["body"]
    kinds = [r["kind"] for r in conn.execute("SELECT kind FROM events WHERE run_id=?", (rid,))]
    assert "degraded.paused_midrun" not in kinds


def test_a_triage_quota_failure_retries_instead_of_falling_back(
    conn, gh, lanes, paused_cfg, outage, skills_dir, tmp_path
):
    log, runner = outage
    raw = json.loads((skills_dir / "config.json").read_text())
    raw["specialists"] = [s for s in raw["specialists"] if s.get("always_run")]
    (skills_dir / "config.json").write_text(json.dumps(raw))
    c = cfg_mod.load(tmp_path / "config.toml")
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = td._verifier()
    lanes.triage = {"tier": "low", "reasoning": "small"}
    lanes.dead_models = {"gpt-6-astra"}
    lanes.dead_stderr = QUOTA
    rid, run_status = td._run(c, conn, gh, runner, prober=td.FINE)
    assert run_status == RunStatus.DONE
    tri = [i for i, e in enumerate(log) if e.startswith("triage:")]
    assert [log[i] for i in tri] == ["triage:gpt-6-astra"] * 3
    assert tri[1] < log.index("back") < tri[2]
    kinds = [r["kind"] for r in conn.execute("SELECT kind FROM events WHERE run_id=?", (rid,))]
    assert "triage.degraded" not in kinds
    assert conn.execute("SELECT tier FROM runs WHERE id=?", (rid,)).fetchone()["tier"] == "low"
