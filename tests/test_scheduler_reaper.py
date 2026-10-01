import os
import subprocess
import sys
import time
from dataclasses import replace
from datetime import timedelta

from reviewsys.db import now, parse_ts, tx
from reviewsys.ingest import enqueue_head
from reviewsys.models import FailKind, RunStatus, Trigger
from reviewsys.reaper import reap
from reviewsys.scheduler import apply_supersedes, finish_run, schedule
from reviewsys.status import watchdog


def queue(conn, cfg, n: int, *, priority=False, ready=True):
    ids = []
    with tx(conn):
        for i in range(n):
            enqueue_head(
                conn,
                cfg,
                "dashpay/platform",
                100 + i,
                f"{i:040x}",
                Trigger.MENTION if priority else Trigger.NEW_PR,
            )
            if ready:
                conn.execute("UPDATE heads SET eligible_at=queued_at WHERE number=?", (100 + i,))
            ids.append(
                conn.execute("SELECT id FROM heads WHERE number=?", (100 + i,)).fetchone()["id"]
            )
    return ids


def _in_flight(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM runs WHERE status='spawned'").fetchone()[0]


def test_every_eligible_head_starts_without_a_run_slot_count(cfg, conn):
    """Reviews are bounded by model slots (lanepool.py), not a run count: a backlog far past
    the old 2 + 1 starts at once, and priority work arriving later starts too."""
    queue(conn, cfg, 12)
    assert len(schedule(conn, cfg, spawn=False)) == 12
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 200, "e" * 40, Trigger.REVIEW_REQUESTED)
    assert len(schedule(conn, cfg, spawn=False)) == 1
    assert schedule(conn, cfg, spawn=False) == []
    assert _in_flight(conn) == 13


def test_max_runs_is_a_safety_cap_priority_first(cfg, conn):
    cfg = replace(cfg, max_runs=3)
    queue(conn, cfg, 4)
    assert len(schedule(conn, cfg, spawn=False)) == 3
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 200, "e" * 40, Trigger.REVIEW_REQUESTED)
    assert schedule(conn, cfg, spawn=False) == [], "the cap holds for priority work too"
    with tx(conn):
        conn.execute("UPDATE runs SET status='done' WHERE id=(SELECT MIN(id) FROM runs)")
    (rid,) = schedule(conn, cfg, spawn=False)
    picked = conn.execute(
        "SELECT h.number FROM runs r JOIN heads h ON h.id=r.head_id WHERE r.id=?", (rid,)
    ).fetchone()[0]
    assert picked == 200, "the priority head, though normal heads were queued before it"
    assert _in_flight(conn) == 3


def test_max_runs_zero_means_no_cap(cfg, conn):
    cfg = replace(cfg, max_runs=0)
    queue(conn, cfg, 40)
    assert len(schedule(conn, cfg, spawn=False)) == 40


def test_debounce_blocks_until_eligible(cfg, conn):
    queue(conn, cfg, 1, ready=False)
    assert schedule(conn, cfg, spawn=False) == []
    with tx(conn):
        conn.execute("UPDATE heads SET eligible_at=?", (now(),))
    assert len(schedule(conn, cfg, spawn=False)) == 1


def test_single_flight_per_pr(cfg, conn):
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 1, "a" * 40, Trigger.MENTION)
    schedule(conn, cfg, spawn=False)
    with tx(conn):
        # new push while running: old head superseded, new head queued
        enqueue_head(conn, cfg, "dashpay/platform", 1, "b" * 40, Trigger.MENTION)
    assert schedule(conn, cfg, spawn=False) == []  # same PR already has an active run
    assert apply_supersedes(conn, cfg) == 1
    run = conn.execute("SELECT * FROM runs").fetchone()
    assert run["cancel_requested"] == 1


def test_finish_run_retry_then_fail(cfg, conn):
    (hid,) = queue(conn, cfg, 1)
    for attempt in range(1, 4):
        (rid,) = schedule(conn, cfg, spawn=False)
        finish_run(conn, cfg, rid, RunStatus.FAILED, reason="boom", fail_kind=FailKind.INFRA)
        head = conn.execute("SELECT * FROM heads WHERE id=?", (hid,)).fetchone()
        if attempt < cfg.max_attempts:
            assert head["status"] == "queued" and head["attempts"] == attempt
            expected_backoff = cfg.retry_backoff_minutes[attempt - 1]
            assert parse_ts(head["eligible_at"]) - parse_ts(now()) > timedelta(
                minutes=expected_backoff - 1
            )
            with tx(conn):
                conn.execute("UPDATE heads SET eligible_at=? WHERE id=?", (now(), hid))
        else:
            assert head["status"] == "failed"
    assert conn.execute("SELECT COUNT(*) FROM events WHERE kind='head.failed'").fetchone()[0] == 1
    # finish_run is idempotent
    finish_run(conn, cfg, rid, RunStatus.DONE)
    assert (
        conn.execute("SELECT status FROM runs WHERE id=?", (rid,)).fetchone()["status"] == "failed"
    )


def test_fatal_never_retries(cfg, conn):
    queue(conn, cfg, 1)
    (rid,) = schedule(conn, cfg, spawn=False)
    finish_run(conn, cfg, rid, RunStatus.FAILED, reason="PR closed", fail_kind=FailKind.FATAL)
    assert conn.execute("SELECT status FROM heads").fetchone()["status"] == "failed"


def test_live_base_move_requeues_same_head(cfg, conn):
    (hid,) = queue(conn, cfg, 1)
    (rid,) = schedule(conn, cfg, spawn=False)
    finish_run(
        conn,
        cfg,
        rid,
        RunStatus.FAILED,
        reason="live base deadbeef != assigned cafebabe",
        fail_kind=FailKind.FATAL,
    )
    head = conn.execute("SELECT * FROM heads WHERE id=?", (hid,)).fetchone()
    assert head["status"] == "queued"
    assert head["sha"] == "0000000000000000000000000000000000000000"
    assert conn.execute("SELECT kind FROM events WHERE kind='head.requeued'").fetchone()


def test_reaper_kills_stale_and_no_heartbeat(cfg, conn):
    queue(conn, cfg, 2)
    r1, r2 = schedule(conn, cfg, spawn=False)
    long_lived = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    with tx(conn):
        # r1: running with stale heartbeat, real pid
        conn.execute(
            "UPDATE runs SET status='running', pid=?, heartbeat_at=? WHERE id=?",
            (
                long_lived.pid,
                (parse_ts(now()) - timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),
                r1,
            ),
        )
        # r2: spawned 5 minutes ago, never heartbeat, pid already gone
        conn.execute(
            "UPDATE runs SET pid=999999, started_at=? WHERE id=?",
            ((parse_ts(now()) - timedelta(minutes=5)).isoformat().replace("+00:00", "Z"), r2),
        )
    actions = dict(reap(conn, cfg))
    assert actions == {r1: "stale", r2: "no_heartbeat"}
    time.sleep(0.3)
    assert long_lived.poll() is not None, "stale worker's process group was killed"
    assert {r["status"] for r in conn.execute("SELECT status FROM runs")} == {"failed"}
    assert {r["status"] for r in conn.execute("SELECT status FROM heads")} == {
        "queued"
    }  # requeued for retry
    assert schedule(conn, cfg, spawn=False) == []  # backoff not elapsed
    long_lived.wait(timeout=5)


def test_reaper_deadline(cfg, conn):
    queue(conn, cfg, 1)
    (rid,) = schedule(conn, cfg, spawn=False)
    with tx(conn):
        conn.execute(
            "UPDATE runs SET status='running', heartbeat_at=?, deadline_at=? WHERE id=?",
            (
                now(),
                (parse_ts(now()) - timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
                rid,
            ),
        )
    assert dict(reap(conn, cfg)) == {rid: "timed_out"}


def test_reaper_process_exited_without_status(cfg, conn):
    queue(conn, cfg, 1)
    (rid,) = schedule(conn, cfg, spawn=False)
    p = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    p.wait()
    with tx(conn):
        conn.execute(
            "UPDATE runs SET status='running', pid=?, heartbeat_at=? WHERE id=?",
            (p.pid, now(), rid),
        )
    assert dict(reap(conn, cfg)) == {rid: "exited"}


def test_no_slot_can_be_ghosted(cfg, conn):
    """The property the old system lacked: a dead run never occupies a slot for more than one reap."""
    queue(conn, cfg, 3)
    rids = schedule(conn, cfg, spawn=False)
    with tx(conn):
        for rid in rids:
            conn.execute(
                "UPDATE runs SET status='running', pid=999998, heartbeat_at=? WHERE id=?",
                (now(), rid),
            )
    reap(conn, cfg)
    assert (
        conn.execute("SELECT COUNT(*) FROM runs WHERE status IN ('spawned','running')").fetchone()[
            0
        ]
        == 0
    )
    with tx(conn):
        conn.execute("UPDATE heads SET eligible_at=? WHERE status='queued'", (now(),))
    assert len(schedule(conn, cfg, spawn=False)) == 3


def test_watchdog(cfg, conn):
    w = watchdog(conn, cfg)
    assert not w["stuck"]
    queue(conn, cfg, 1)
    w = watchdog(conn, cfg)
    assert w["stuck"] and w["eligible"] == 1  # eligible work, under the cap, nothing started


def test_watchdog_is_quiet_at_the_run_cap_and_behind_single_flight(cfg, conn):
    cfg = replace(cfg, max_runs=1, watchdog_minutes=0)
    queue(conn, cfg, 2)
    (rid,) = schedule(conn, cfg, spawn=False)
    with tx(conn):  # the run started long ago; a head is still waiting
        conn.execute("UPDATE runs SET started_at='2026-01-01T00:00:00Z' WHERE id=?", (rid,))
    assert not watchdog(conn, cfg)["stuck"], "waiting for max_runs is not stuck"
    assert watchdog(conn, replace(cfg, max_runs=0))["stuck"]
    # a head of a PR whose run is still going waits for that run, not for the scheduler
    running = conn.execute(
        "SELECT h.number FROM runs r JOIN heads h ON h.id=r.head_id WHERE r.id=?", (rid,)
    ).fetchone()[0]
    with tx(conn):
        conn.execute("DELETE FROM heads WHERE id NOT IN (SELECT head_id FROM runs)")
        enqueue_head(conn, cfg, "dashpay/platform", running, "f" * 40, Trigger.MENTION)
        conn.execute("UPDATE heads SET eligible_at=queued_at WHERE status='queued'")
    w = watchdog(conn, replace(cfg, max_runs=0))
    assert w["eligible"] == 1 and not w["stuck"]


def test_spawn_real_worker_process(cfg, conn, tmp_path, monkeypatch):
    """schedule() with spawn=True launches a detached process; we don't run a real review here."""
    queue(conn, cfg, 1)
    monkeypatch.setattr(
        "reviewsys.scheduler.worker_argv",
        lambda run_id: [sys.executable, "-c", "import time; time.sleep(0.2)"],
    )
    (rid,) = schedule(conn, cfg, spawn=True)
    pid = conn.execute("SELECT pid FROM runs WHERE id=?", (rid,)).fetchone()["pid"]
    assert pid and (cfg.logs_dir / f"run-{rid}.log").exists()
    for _ in range(50):
        try:
            os.kill(pid, 0)
            time.sleep(0.05)
        except ProcessLookupError:
            break


def test_shadow_daemon_creates_no_runs_and_does_not_alert_stuck(cfg, conn, gh, notifier):
    from reviewsys.daemon import Daemon

    queue(conn, cfg, 2)
    d = Daemon(cfg, conn, gh=gh, notifier=notifier, spawn=False)
    res = d.tick(force=True)
    assert "schedule" not in res and "reap" not in res
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert not [s for s in notifier.sent if s[0] == "alert" and "stuck=True" in s[1]]


def test_daemon_singleton_lock(tmp_path):
    import pytest

    from reviewsys.daemon import acquire_singleton_lock

    held = acquire_singleton_lock(tmp_path / "d.lock")
    with pytest.raises(SystemExit):
        acquire_singleton_lock(tmp_path / "d.lock")
    held.close()
    again = acquire_singleton_lock(tmp_path / "d.lock")
    again.close()
