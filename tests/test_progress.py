"""Live progress: the per-tier step profile, the time-left estimate, and the gate comment
that carries it (refreshed from the worker's heartbeat until a final status replaces it)."""

from __future__ import annotations

from datetime import timedelta

from reviewsys import github, progress, worker
from reviewsys.db import fmt_ts, now_dt, tx


def _head(conn, number, status="done"):
    return conn.execute(
        "INSERT INTO heads (repo, number, sha, trigger, status, queued_at, eligible_at) "
        "VALUES ('dashpay/platform', ?, ?, 'new_push', ?, 't', 't')",
        (number, f"{number:040x}", status),
    ).lastrowid


def _run(conn, head_id, *, status, tier, started):
    return conn.execute(
        "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at, finished_at, "
        "tier) VALUES (?,1,?,?,?,?,?,?)",
        (
            head_id,
            status,
            f"t{head_id}",
            fmt_ts(started),
            "t",
            fmt_ts(started + timedelta(hours=1)) if status == "done" else None,
            tier,
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


def test_gate_body_carries_progress_below_the_unchanged_status_line():
    body = github.gate_body(
        "in_progress",
        "a" * 40,
        tier="normal",
        progress={
            "fraction": 0.45,
            "remaining_seconds": 1500,
            "elapsed_seconds": 3900,
            "steps": [
                ("worktree", "ok"),
                ("triage", "ok"),
                ("phase1", "running"),
                ("verify1", "upcoming"),
            ],
            "run_id": 42,
            "updated_at": "12:34",
        },
    )
    lines = body.splitlines()
    assert lines[1] == github.gate_body("in_progress", "a" * 40, tier="normal").splitlines()[1]
    assert lines[2].startswith("`█████████░░░░░░░░░░░` **45%**")
    assert "about 25 min left" in lines[2] and "running for 1 h 5 min" in lines[2]
    assert lines[3] == "✅ triage → ⏳ **Phase 1** → ▫️ verify 1", "setup steps left out"
    assert f"{github.DASHBOARD_URL}#run=42" in lines[4]


def test_heartbeat_refreshes_in_progress_comment_until_a_final_status(cfg, conn, gh, tmp_path):
    at = now_dt()
    with tx(conn):
        _history(conn)
        hid = _head(conn, 4, "running")
        run = _run(conn, hid, status="running", tier="normal", started=at - timedelta(minutes=3))
        _step(conn, run, "phase1", at - timedelta(minutes=3), status="running")
    ctx = worker.RunContext(
        cfg=cfg,
        main_conn=conn,
        gh=gh,
        run_id=run,
        head_id=hid,
        repo="dashpay/platform",
        number=4,
        sha="4" * 40,
        token="t",
        run_dir=tmp_path,
    )
    worker._gate_comment(ctx, "in_progress")
    assert f"#run={run}" in gh.gate_bodies[-1] and ctx.gate_comment_id == 77
    posted = len(gh.gate_bodies)
    worker._refresh_gate_progress(ctx, conn)
    assert len(gh.gate_bodies) == posted, "nothing changed and it was just written"
    with tx(conn):
        conn.execute(
            "UPDATE steps SET status='ok', finished_at=? WHERE run_id=?", (fmt_ts(at), run)
        )
        _step(conn, run, "verify1", at, status="running")
    ctx.gate_refreshed_at -= worker.GATE_PROGRESS_MIN_GAP_SECONDS + 1
    worker._refresh_gate_progress(ctx, conn)
    assert len(gh.gate_bodies) == posted + 1 and "⏳ **verify 1**" in gh.gate_bodies[-1]
    worker._gate_comment(ctx, "done", phase="final", blocker_count=0)
    final = len(gh.gate_bodies)
    ctx.gate_refreshed_at -= worker.GATE_PROGRESS_EVERY_SECONDS + 1
    worker._refresh_gate_progress(ctx, conn)
    assert len(gh.gate_bodies) == final, "a final status is never overwritten"
    assert "Final review complete" in gh.gate_bodies[-1]


def test_no_estimate_before_triage_and_conversations_use_their_own_profile(conn):
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
    est = progress.estimate(conn, fresh, at)
    assert est is not None and est.progress is None, "no tier yet: no guess"
    est = progress.estimate(conn, reply, at)
    assert est is not None and est.remaining_seconds == 15 and est.upcoming == []
    assert "converse" not in progress.profile(conn, "normal")


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
    assert "taking longer than usual" in github.gate_body(
        "in_progress",
        "a" * 40,
        progress={
            "fraction": est.progress,
            "overdue": True,
            "remaining_seconds": 5,
            "elapsed_seconds": 2400,
            "steps": [],
            "run_id": run,
            "updated_at": "x",
        },
    )


def test_refresh_stops_for_a_superseded_head_and_never_waits_for_the_lock(cfg, conn, gh, tmp_path):
    at = now_dt()
    with tx(conn):
        _history(conn)
        hid = _head(conn, 8, "running")
        run = _run(conn, hid, status="running", tier="normal", started=at - timedelta(minutes=3))
        _step(conn, run, "phase1", at - timedelta(minutes=3), status="running")
    ctx = worker.RunContext(
        cfg=cfg,
        main_conn=conn,
        gh=gh,
        run_id=run,
        head_id=hid,
        repo="dashpay/platform",
        number=8,
        sha="8" * 40,
        token="t",
        run_dir=tmp_path,
    )
    worker._gate_comment(ctx, "in_progress")
    posted = len(gh.gate_bodies)
    ctx.gate_refreshed_at -= worker.GATE_PROGRESS_EVERY_SECONDS + 1
    with ctx.gate_lock:  # the main thread is mid-write: the heartbeat skips, it does not block
        worker._refresh_gate_progress(ctx, conn)
    assert len(gh.gate_bodies) == posted
    with tx(conn):
        conn.execute("UPDATE heads SET status='queued' WHERE id=?", (hid,))  # a new push landed
    worker._refresh_gate_progress(ctx, conn)
    assert len(gh.gate_bodies) == posted, "the queue owns the comment of a superseded head"
