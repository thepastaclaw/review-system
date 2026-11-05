"""`lanes.started_at`: when the lane's process started, not when its row was written."""

from __future__ import annotations

import types
from datetime import UTC, datetime
from pathlib import Path

import test_worker_e2e as e2e

from reviewsys import worker
from reviewsys.db import parse_ts, tx
from reviewsys.lane import LaneResult

fake_git = e2e.fake_git  # the e2e module's autouse fixture, for the scenario run here


def test_lane_rows_span_the_time_the_lane_ran(cfg, conn, gh, lanes):
    e2e.test_two_phase_final_review_posts_once(cfg, conn, gh, lanes)
    rows = conn.execute("SELECT started_at, finished_at FROM lanes").fetchall()
    assert rows
    for r in rows:  # FakeLanes report one second each
        assert (parse_ts(r["finished_at"]) - parse_ts(r["started_at"])).total_seconds() == 1


def test_lane_row_started_at_counts_its_correction_turns(conn, monkeypatch):
    with tx(conn):
        hid = conn.execute(
            "INSERT INTO heads (repo, number, sha, trigger, queued_at, eligible_at) "
            "VALUES ('o/r', 1, 's', 'manual', 'x', 'x')"
        ).lastrowid
        rid = conn.execute(
            "INSERT INTO runs (head_id, attempt, token, started_at, deadline_at) "
            "VALUES (?, 1, 't', 'x', 'x')",
            (hid,),
        ).lastrowid
    end = datetime(2026, 10, 4, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(worker, "now_dt", lambda: end)
    res = LaneResult(exit_code=0, stdout="", stderr="", duration_s=100.2)
    res.followups.append(LaneResult(exit_code=0, stdout="", stderr="", duration_s=19.9))
    ctx = types.SimpleNamespace(conn=conn, run_id=rid)
    lm = types.SimpleNamespace(agent="a", model="m", effort="high")
    worker._lane_row(
        ctx,  # type: ignore[arg-type]
        phase="phase2",
        role="general",
        lm=lm,  # type: ignore[arg-type]
        attempt=1,
        attempt_id="x",
        status="completed",
        res=res,
        artifact_dir=Path("/x"),
        psha="p",
    )
    row = conn.execute("SELECT started_at, finished_at FROM lanes").fetchone()
    assert (row["started_at"], row["finished_at"]) == (
        "2026-10-04T11:58:00Z",
        "2026-10-04T12:00:00Z",
    )
