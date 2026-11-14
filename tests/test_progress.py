"""Live progress: the per-tier step profile and time-left estimate used by the dashboard."""

from __future__ import annotations

from datetime import timedelta

from reviewsys import github, progress, worker
from reviewsys.db import fmt_ts, now_dt, tx


def _head(conn, number, status="done", *, priority=False, queue="live"):
    return conn.execute(
        "INSERT INTO heads (repo, number, sha, trigger, priority, status, queued_at, eligible_at, "
        "queue) VALUES ('dashpay/platform', ?, ?, 'new_push', ?, ?, 't', 't', ?)",
        (number, f"{number:040x}", int(priority), status, queue),
    ).lastrowid


def _run(conn, head_id, *, status, tier, started, path=None):
    return conn.execute(
        "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at, finished_at, "
        "tier, path) VALUES (?,1,?,?,?,?,?,?,?)",
        (
            head_id,
            status,
            f"t{head_id}",
            fmt_ts(started),
            "t",
            fmt_ts(started + timedelta(hours=1)) if status == "done" else None,
            tier,
            path,
        ),
    ).lastrowid


def _step(conn, run_id, name, start, seconds=None, status="ok"):
    conn.execute(
        "INSERT INTO steps (run_id, name, status, started_at, finished_at) VALUES (?,?,?,?,?)",
        (
            run_id,
            name,
            status,
            fmt_ts(start),
            fmt_ts(start + timedelta(seconds=seconds)) if seconds is not None else None,
        ),
    )


def _history(conn, n=6):
    """n completed normal-tier reviews: Phase 1 10 min, verify 1 min, publish 10 s; a fresh
    Phase 2 of 15 min on every other one."""
    t0 = now_dt() - timedelta(days=1)
    for i in range(n):
        run = _run(conn, _head(conn, 100 + i), status="done", tier="normal", started=t0)
        _step(conn, run, "phase1", t0, 600)
        _step(conn, run, "verify1", t0, 60)
        if i % 2:
            _step(conn, run, "fresh_phase2", t0, 900)
        _step(conn, run, "publish", t0, 10)


def test_estimate_weighs_steps_by_how_often_they_run(conn):
    at = now_dt()
    with tx(conn):
        _history(conn)
        run = _run(
            conn,
            _head(conn, 1, "running"),
            status="running",
            tier="normal",
            started=at - timedelta(minutes=6),
        )
        _step(conn, run, "triage", at - timedelta(minutes=6), 20)
        _step(conn, run, "phase1", at - timedelta(minutes=5), status="running")
    est = progress.estimate(conn, run, at)
    assert est is not None and est.elapsed_seconds == 360
    # Phase 1 still needs ~300 s, then verify 60 + half a fresh Phase 2 (450) + publish 10
    assert est.remaining_seconds == 300 + 60 + 450 + 10
    assert est.progress == round(360 / (360 + 820), 3)
    assert est.upcoming == ["verify1", "fresh_phase2", "publish"]


def test_steps_before_the_furthest_one_reached_are_not_upcoming(conn):
    """A Phase-2-only review never ran Phase 1; it is skipped, not still to come."""
    at = now_dt()
    with tx(conn):
        _history(conn)
        run = _run(
            conn,
            _head(conn, 2, "running"),
            status="running",
            tier="normal",
            started=at - timedelta(minutes=2),
        )
        _step(conn, run, "fresh_phase2", at - timedelta(minutes=2), status="running")
    est = progress.estimate(conn, run, at)
    assert est is not None and "phase1" not in est.upcoming and "verify1" not in est.upcoming
    assert est.upcoming == ["publish"]


def test_no_history_means_no_estimate(conn):
    at = now_dt()
    with tx(conn):
        run = _run(
            conn,
            _head(conn, 3, "running"),
            status="running",
            tier="normal",
            started=at - timedelta(minutes=1),
        )
    est = progress.estimate(conn, run, at)
    assert est is not None and est.progress is None and est.remaining_seconds is None


def test_in_progress_gate_body_is_static_and_links_to_the_run():
    body = github.gate_body(
        "in_progress",
        "a" * 40,
        run_id=42,
        tier="normal",
        degraded=True,
        phase1_skipped="skipped for throughput: 12 PRs queued",
    )
    assert body == (
        f"{github.GATE_MARKER}\n🔍 Review in progress (commit aaaaaaaa) · "
        f"[live progress]({github.DASHBOARD_URL}#run=42)"
    )
    assert "%" not in body and "min left" not in body and "updated " not in body


def test_before_triage_every_tier_counts_and_conversations_use_their_own_profile(conn):
    at = now_dt()
    with tx(conn):
        _history(conn)
        t0 = at - timedelta(days=1)
        for i in range(3):  # conversations: seconds long, never part of a review profile
            chat = _run(conn, _head(conn, 200 + i), status="done", tier=None, started=t0)
            _step(conn, chat, "converse", t0, 20)
        fresh = _run(
            conn,
            _head(conn, 5, "running"),
            status="running",
            tier=None,
            started=at - timedelta(seconds=30),
        )
        _step(conn, fresh, "worktree", at - timedelta(seconds=30), status="running")
        reply = _run(
            conn,
            _head(conn, 6, "running"),
            status="running",
            tier=None,
            started=at - timedelta(seconds=10),
        )
        _step(conn, reply, "converse", at - timedelta(seconds=5), status="running")
        # queued by a reply: may yet be a conversation, so no review estimate before triage
        maybe = _run(
            conn,
            _head(conn, 8, "running"),
            status="running",
            tier=None,
            started=at - timedelta(seconds=30),
        )
        conn.execute("UPDATE heads SET trigger='review_reply' WHERE number=8")
        _step(conn, maybe, "worktree", at - timedelta(seconds=30), status="running")
    est = progress.estimate(conn, fresh, at)
    # no tier yet: measured against reviews of every tier, and it says so
    assert est is not None and est.basis == "reviews" and est.progress is not None
    assert est.remaining_seconds == 600 + 60 + 450 + 10
    assert est.upcoming == ["phase1", "verify1", "fresh_phase2", "publish"]
    est = progress.estimate(conn, maybe, at)
    assert est is not None and est.progress is None and est.basis is None
    assert progress.estimate(conn, maybe, at, conversation=False).basis == "reviews"
    est = progress.estimate(conn, maybe, at, conversation=True)
    assert est is not None and est.basis == "conversation" and est.remaining_seconds == 20
    est = progress.estimate(conn, reply, at)
    assert est is not None and est.remaining_seconds == 15 and est.upcoming == []
    assert est.basis == "conversation"
    assert "converse" not in progress.profile(conn, "normal")
    assert "converse" not in progress.profile(conn, progress.ALL_TIERS)
    with tx(conn):
        conn.execute("UPDATE runs SET tier='normal' WHERE id=?", (fresh,))
    assert progress.estimate(conn, fresh, at).basis == "tier"


def test_a_step_far_past_its_median_is_overdue(conn):
    at = now_dt()
    with tx(conn):
        _history(conn)
        run = _run(
            conn,
            _head(conn, 7, "running"),
            status="running",
            tier="normal",
            started=at - timedelta(minutes=40),
        )
        _step(conn, run, "phase1", at - timedelta(minutes=40), status="running")  # median 10
    est = progress.estimate(conn, run, at)
    assert est is not None and est.overdue


def _phase1_history(conn, *, priority, minutes, n=6, first=300):
    """n completed normal-tier reviews on one path whose Phase 1 took `minutes`. The heads all
    say normal: a run's path is the one recorded as it started, whatever its head says now (a
    reply re-queues a reviewed head as priority)."""
    t0 = now_dt() - timedelta(days=1)
    for i in range(n):
        run = _run(
            conn,
            _head(conn, first + i),
            status="done",
            tier="normal",
            started=t0,
            path="priority" if priority else "normal",
        )
        _step(conn, run, "phase1", t0, minutes * 60)


def _in_phase1(conn, number, at, minutes, **head):
    run = _run(
        conn,
        _head(conn, number, "running", **head),
        status="running",
        tier="normal",
        started=at - timedelta(minutes=minutes),
    )
    _step(conn, run, "phase1", at - timedelta(minutes=minutes), status="running")
    return run


def test_longer_than_usual_compares_a_run_with_its_own_path(conn):
    """Priority lanes go first in every pool's line, so a priority Phase 1 is short; 6 min of
    it is long for a priority review and quick for a normal one."""
    at = now_dt()
    with tx(conn):
        _phase1_history(conn, priority=True, minutes=3, first=300)
        _phase1_history(conn, priority=False, minutes=20, first=400)
        fast = _in_phase1(conn, 8, at, 6, priority=True)
        slow = _in_phase1(conn, 9, at, 6)
        audit = _in_phase1(conn, 10, at, 6, queue="audit")
    profiles: dict = {}
    assert progress.estimate(conn, fast, at, profiles).overdue  # 6 > 1.5 x 3
    est = progress.estimate(conn, slow, at, profiles)
    assert not est.overdue and est.remaining_seconds == 14 * 60
    # an audit (never priority) is measured against both paths: median of 3s and 20s = 11.5
    assert progress.estimate(conn, audit, at, profiles).remaining_seconds == int(5.5 * 60)
    assert set(profiles) == {("normal", "priority"), ("normal", "normal"), ("normal", None)}
    assert progress.path_of("live", 1) == "priority" and progress.path_of("audit", 0) == "audit"


def test_a_path_with_too_few_runs_borrows_the_tiers_profile_of_both_paths(conn):
    at = now_dt()
    with tx(conn):
        _phase1_history(conn, priority=True, minutes=3, n=progress.MIN_PROFILE_RUNS - 1)
        _phase1_history(conn, priority=False, minutes=20, n=7, first=400)
        run = _in_phase1(conn, 11, at, 6, priority=True)
    est = progress.estimate(conn, run, at)
    # 4 priority runs of 3 min are not enough: the tier's 11 runs say 20 min
    assert est is not None and not est.overdue and est.remaining_seconds == 14 * 60


def test_step_lanes_count_a_phases_planned_lanes_without_comparison_twins(conn):
    at = now_dt()
    t = lambda m: fmt_ts(at - timedelta(minutes=m))  # noqa: E731
    with tx(conn):
        run = _run(
            conn,
            _head(conn, 9, "running"),
            status="running",
            tier="critical",
            started=at - timedelta(hours=2),
        )
        conn.execute(
            "INSERT INTO events (ts, kind, run_id, detail) VALUES (?,?,?,?)",
            (
                t(100),
                "compare.selected",
                run,
                "tier=critical primary=gpt-6.1-sol second=gpt-6-astra",
            ),
        )
        for phase, role, model, minutes in (
            ("phase1", "general", "glm-5.3-flash", 70),
            ("phase1", "rust-quality", "glm-5.3-flash", 65),
            ("phase2", "general", "gpt-6-astra", 20),  # the comparison twin finished first
            ("phase2", "general", "gpt-6.1-sol", 50),  # first Phase 2, before the fresh pass
        ):
            conn.execute(
                "INSERT INTO lanes (run_id, phase, role, agent, model, attempt, attempt_id, "
                "status, started_at, finished_at) VALUES (?,?,?,?,?,1,'x','completed',?,?)",
                (run, phase, role, "a", model, t(minutes + 5), t(minutes)),
            )
    steps = [
        {
            "name": "phase1",
            "started_at": t(90),
            "info": {"roles": ["general", "rust-quality", "ffi"]},
        },
        {"name": "fresh_phase2", "started_at": t(30), "info": {"roles": ["general", "ffi"]}},
        {"name": "phase2", "started_at": t(60), "info": {}},  # an older worker: no plan
        {"name": "verify1", "started_at": t(80), "info": {}},
    ]
    p1, fresh, old, verify = progress.step_lanes(conn, run, steps)
    assert p1 == {"lanes_done": 2, "lanes_total": 3, "lanes_left": ["ffi"]}
    assert fresh == {"lanes_done": 0, "lanes_total": 2, "lanes_left": ["general", "ffi"]}
    assert old == {"lanes_done": 1} and verify == {}


def test_a_lane_counts_by_when_it_finished_not_when_it_started(conn):
    """A lane's `started_at` is its finish less the seconds its turns ran, and timestamps are
    whole seconds: a lane that started 0.4 s after its step can read a second before it. It
    still counts as one of the step's finished lanes; one that finished before the step
    started (an earlier pass of the phase) does not."""
    at = now_dt()
    step_at = at - timedelta(minutes=10)
    with tx(conn):
        run = _run(conn, _head(conn, 11, "running"), status="running", tier="normal", started=at)
        for role, started, finished in (
            ("general", step_at - timedelta(seconds=1), at),  # the sub-second boundary
            ("ffi", step_at - timedelta(minutes=20), step_at - timedelta(seconds=1)),
        ):
            conn.execute(
                "INSERT INTO lanes (run_id, phase, role, agent, model, attempt, attempt_id, "
                "status, started_at, finished_at) VALUES (?,'phase2',?,'a','m',1,'x',"
                "'completed',?,?)",
                (run, role, fmt_ts(started), fmt_ts(finished)),
            )
    step = {"name": "phase2", "started_at": fmt_ts(step_at), "info": {"roles": ["general", "ffi"]}}
    (counts,) = progress.step_lanes(conn, run, [step])
    assert counts == {"lanes_done": 1, "lanes_total": 2, "lanes_left": ["ffi"]}


def test_comparison_twins_still_going_after_the_primary_lanes(conn):
    """A comparison run's Phase 2 waits for the second model's twins once its own lanes are
    done; the count says so instead of reading as a finished phase."""
    at = now_dt()
    with tx(conn):
        run = _run(
            conn,
            _head(conn, 10, "running"),
            status="running",
            tier="critical",
            started=at - timedelta(hours=1),
        )
        conn.execute(
            "INSERT INTO events (ts, kind, run_id, detail) VALUES (?,?,?,?)",
            (
                fmt_ts(at),
                "compare.selected",
                run,
                "tier=critical primary=gpt-6.1-sol second=gpt-6-astra",
            ),
        )
        for role, model in (
            ("general", "gpt-6.1-sol"),
            ("ffi", "gpt-6.1-sol"),
            ("general", "gpt-6-astra"),
        ):
            conn.execute(
                "INSERT INTO lanes (run_id, phase, role, agent, model, attempt, attempt_id, "
                "status, started_at, finished_at) VALUES (?,'phase2',?,'a',?,1,'x','completed',?,?)",
                (run, role, model, fmt_ts(at - timedelta(minutes=10)), fmt_ts(at)),
            )
    step = {
        "name": "phase2",
        "started_at": fmt_ts(at - timedelta(minutes=30)),
        "info": {"roles": ["general", "ffi"]},
    }
    (counts,) = progress.step_lanes(conn, run, [step])
    assert counts == {"lanes_done": 2, "lanes_total": 2, "lanes_left": [], "comparison_left": 1}


def _ctx(cfg, conn, gh, tmp_path, run, hid, number):
    return worker.RunContext(
        cfg=cfg,
        main_conn=conn,
        gh=gh,
        run_id=run,
        head_id=hid,
        repo="dashpay/platform",
        number=number,
        sha=f"{number}" * 40,
        token="t",
        run_dir=tmp_path,
        conversation=False,
    )


def test_worker_writes_in_progress_once_and_reuses_the_static_body(cfg, conn, gh, tmp_path):
    at = now_dt()
    with tx(conn):
        hid = _head(conn, 9, "running")
        run = _run(conn, hid, status="running", tier=None, started=at)
    ctx = _ctx(cfg, conn, gh, tmp_path, run, hid, 9)
    worker._gate_comment(ctx, "in_progress")
    worker._gate_comment(ctx, "in_progress", tier="critical")
    assert len(gh.gate_bodies) == 1
    assert f"#run={run}" in gh.gate_bodies[0]
    assert "%" not in gh.gate_bodies[0] and "min left" not in gh.gate_bodies[0]
    worker._gate_comment(ctx, "done", phase="final", blocker_count=0)
    assert len(gh.gate_bodies) == 2
    assert "Final review complete" in gh.gate_bodies[-1]


def test_an_audit_without_a_tier_is_not_estimated(conn):
    """A light audit never records a tier; measured against reviews of every tier it would
    show an hour left for a ten-minute pass."""
    at = now_dt()
    with tx(conn):
        _history(conn)
        hid = _head(conn, 12, "running")
        conn.execute("UPDATE heads SET queue='audit' WHERE id=?", (hid,))
        run = _run(conn, hid, status="running", tier=None, started=at - timedelta(minutes=1))
        _step(conn, run, "phase1", at - timedelta(minutes=1), status="running")
    est = progress.estimate(conn, run, at)
    assert est is not None and est.progress is None and est.basis is None
