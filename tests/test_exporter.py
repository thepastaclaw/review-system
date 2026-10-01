"""The public status export must show the queue in the order the scheduler drains it."""

from __future__ import annotations

import json
import os
from datetime import timedelta

from reviewsys import degraded, lanepool
from reviewsys.db import MIGRATIONS, event, fmt_ts, now_dt, parse_ts, tx
from reviewsys.exporter import (
    _live_lanes,
    build_day_runs,
    build_export,
    build_path_timing,
    write_export,
)
from reviewsys.ingest import enqueue_head
from reviewsys.models import Trigger
from reviewsys.progress import path_of
from reviewsys.queue_status import queued_order
from reviewsys.scheduler import eligible_heads


def _queue(conn, cfg, number, *, queued_ago_min, eligible_in_min, trigger=Trigger.NEW_PR):
    at = now_dt()
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", number, f"{number:040x}", trigger)
        conn.execute(
            "UPDATE heads SET queued_at=?, eligible_at=? WHERE repo='dashpay/platform' AND number=?",
            (
                fmt_ts(at - timedelta(minutes=queued_ago_min)),
                fmt_ts(at + timedelta(minutes=eligible_in_min)),
                number,
            ),
        )


def test_export_queue_order_matches_the_scheduler(cfg, conn):
    # oldest head, long eligible; it must be first even though its eligible_at is not the
    # smallest (the dashboard used to sort by eligible_at and showed it 10th)
    _queue(conn, cfg, 10, queued_ago_min=600, eligible_in_min=-5)
    # queued later but its backoff ended earlier
    _queue(conn, cfg, 11, queued_ago_min=300, eligible_in_min=-30)
    # priority head queued last of all: goes first
    _queue(conn, cfg, 12, queued_ago_min=1, eligible_in_min=-1, trigger=Trigger.MENTION)
    # still in debounce: after every eligible head, whatever its queued_at
    _queue(conn, cfg, 13, queued_ago_min=900, eligible_in_min=20)
    export = build_export(conn, cfg)
    numbers = [q["number"] for q in export["live"]["queued"]]
    assert numbers == [12, 10, 11, 13]
    assert [q["position"] for q in export["live"]["queued"]] == [1, 2, 3, 4]
    assert [q["number"] for q in queued_order(conn)] == numbers, "same order as the queue comments"
    assert [h["number"] for h in eligible_heads(conn, ts=fmt_ts(now_dt()))] == [12, 10, 11], (
        "same order as the scheduler picks"
    )
    assert export["live"]["queued"][-1]["reason"] == "debounce (more pushes may follow)"
    assert export["live"]["queued"][0]["priority"] is True
    assert export["live"]["degraded"] == {"configured": False, "active": False}


def test_export_carries_degraded_state(cfg, conn, skills_dir, tmp_path):
    import json

    from reviewsys import config as cfg_mod

    raw = json.loads((skills_dir / "config.json").read_text())
    raw["review_model_policy"]["degraded"] = {
        "sentinel": "gpt-6-astra",
        "substitutes": {"gpt-6-astra": "muse-spark-1.3-contributor"},
    }
    (skills_dir / "config.json").write_text(json.dumps(raw))
    c = cfg_mod.load(tmp_path / "config.toml")
    degraded.force(conn, "on")
    d = build_export(conn, c)["live"]["degraded"]
    assert d["active"] and d["forced"] == "on"
    assert d["substitutes"] == {"gpt-6-astra": "muse-spark-1.3-contributor"}


def test_export_marks_audit_runs(cfg, conn):
    """A post-merge audit is always of a merged PR: the dashboard must say so, or a healthy
    audit hours into Phase 1 looks like a live review stuck on a PR that closed long ago."""
    ts = fmt_ts(now_dt())
    with tx(conn):
        conn.execute(
            "INSERT INTO prs (repo, number, head_sha, title, state, updated_at) VALUES (?,?,?,?,?,?)",
            ("dashpay/platform", 7, "7" * 40, "feat: merged earlier", "closed", ts),
        )
        for number, queue in ((7, "audit"), (8, "live")):
            hid = conn.execute(
                "INSERT INTO heads (repo, number, sha, trigger, status, queued_at, eligible_at, queue) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    "dashpay/platform",
                    number,
                    f"{number}" * 40,
                    "audit" if queue == "audit" else "new_pr",
                    "running",
                    ts,
                    ts,
                    queue,
                ),
            ).lastrowid
            conn.execute(
                "INSERT INTO runs (head_id, attempt, status, token, started_at, heartbeat_at, deadline_at, phase) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (hid, 1, "running", f"t{number}", ts, ts, ts, "phase1"),
            )
    active = {a["number"]: a for a in build_export(conn, cfg)["live"]["active"]}
    assert active[7]["queue"] == "audit" and active[7]["title"] == "feat: merged earlier"
    assert active[8]["queue"] == "live" and active[8]["title"] is None


def _finished_run(conn, number, *, status="done", finished="2026-09-30T12:00:00Z"):
    done = status == "done"
    hid = conn.execute(
        "INSERT INTO heads (repo, number, sha, trigger, status, queued_at, eligible_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (
            "dashpay/platform",
            number,
            f"{number}" * 40,
            "new_push",
            "done",
            "2026-09-30T09:00:00Z",
            "2026-09-30T09:30:00Z",
        ),
    ).lastrowid
    return conn.execute(
        "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at, finished_at, "
        "fail_kind, reason, tier, blocker_count, review_url) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            hid,
            1,
            status,
            f"t{number}",
            "2026-09-30T10:00:00Z",
            "2026-09-30T20:00:00Z",
            finished,
            None if done else "infra",
            None if done else "proxy said no to someone@example.com",
            "normal",
            1 if done else None,
            f"https://github.com/dashpay/platform/pull/{number}#pullrequestreview-9"
            if done
            else None,
        ),
    ).lastrowid


def _step(conn, run_id, name, finished, kept):
    conn.execute(
        "INSERT INTO steps (run_id, name, status, started_at, finished_at, detail) "
        "VALUES (?,?,?,?,?,?)",
        (run_id, name, "ok", finished, finished, json.dumps({"findings": kept})),
    )


def _finding(conn, run_id, phase, stage, hash_, severity):
    conn.execute(
        "INSERT INTO findings (run_id, phase, stage, hash, severity, title, body) "
        "VALUES (?,?,?,?,?,?,?)",
        (run_id, phase, stage, hash_, severity, "secret title", "secret body"),
    )


def test_day_runs_carry_timing_tokens_findings_and_links(conn):
    with tx(conn):
        run = _finished_run(conn, 20)
        for phase, tin, tout in (("phase1", 1000, 100), ("verify2", 500, 50)):
            conn.execute(
                "INSERT INTO lanes (run_id, phase, role, agent, model, effort, attempt, attempt_id, "
                "status, tokens_in, tokens_out, started_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (run, phase, "general", "a", "glm", "high", 1, "x", "completed", tin, tout, "t"),
            )
        # a carried-over blocker the verifier re-asserted but the review did not post again,
        # a verifier retry repeating one hash, and one new suggestion
        _finding(conn, run, "verify1", "verified", "old", "blocking")
        for _ in range(2):
            _finding(conn, run, "verify2", "verified", "b1", "blocking")
        _finding(conn, run, "verify2", "verified", "s1", "suggestion")
        _finding(conn, run, "final", "posted", "s1", "suggestion")
        _step(conn, run, "verify1", "2026-09-30T10:30:00Z", 1)
        _step(conn, run, "verify2", "2026-09-30T11:30:00Z", 2)
        conn.execute(
            "INSERT INTO reviews (run_id, repo, number, sha, phase, github_review_id, event, posted_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (run, "dashpay/platform", 20, "2" * 40, "final", 9, "REQUEST_CHANGES", "t"),
        )
        _finished_run(conn, 21, status="failed", finished="2026-09-30T13:00:00Z")
        _finished_run(conn, 22, status="cancelled")
        _finished_run(conn, 23, finished="2026-09-29T23:59:59Z")
    days = build_day_runs(conn, ["2026-09-30"])
    runs = days["2026-09-30"]
    assert [r["number"] for r in runs] == [21, 20], "newest first; cancelled runs left out"
    done = runs[1]
    assert done["wait_seconds"] == 3600 and done["slot_wait_seconds"] == 1800
    assert done["duration_seconds"] == 7200
    assert (done["tokens_in"], done["tokens_out"], len(done["lanes"])) == (1500, 150, 2)
    assert done["findings"] == {"blocking": 1, "suggestion": 1, "nitpick": 0}
    assert done["new_findings"] == {"blocking": 0, "suggestion": 1, "nitpick": 0}
    assert done["verdict"] == "REQUEST_CHANGES"
    assert done["review_url"].endswith("#pullrequestreview-9")
    failed = runs[0]
    assert failed["review_url"] == failed["pr_url"] == "https://github.com/dashpay/platform/pull/21"
    assert failed["fail_kind"] == "infra"
    blob = json.dumps(days)
    assert "example.com" not in blob and "secret" not in blob, "no reasons or finding text"


def test_write_export_writes_day_files_and_drops_aged_out_days(cfg, conn, tmp_path):
    with tx(conn):
        _finished_run(conn, 30)
    out = tmp_path / "out"
    (out / "days").mkdir(parents=True)
    (out / "days" / "2020-01-01.json").write_text("{}")
    write_export(conn, cfg, out)
    assert sorted(p.name for p in (out / "days").iterdir()) == ["2026-09-30.json"]
    day = json.loads((out / "days" / "2026-09-30.json").read_text())
    assert day["day"] == "2026-09-30" and [r["number"] for r in day["runs"]] == [30]


def test_day_runs_count_an_empty_last_pass_as_clean(conn):
    """Phase 2 dropped everything verify1 kept: the final review is clean, and must not show
    the preliminary pass's findings."""
    with tx(conn):
        run = _finished_run(conn, 40)
        _finding(conn, run, "verify1", "verified", "a", "suggestion")
        _finding(conn, run, "verify2", "verified", "b", "blocking")
        _step(conn, run, "verify1", "2026-09-30T10:10:00Z", 1)
        _step(conn, run, "verify2", "2026-09-30T10:20:00Z", 1)
        _step(conn, run, "fresh_verify2", "2026-09-30T10:40:00Z", 0)  # kept nothing
        # before the `verified-fresh` stage: the fresh pass's findings sit under `verified`
        legacy = _finished_run(conn, 41)
        _finding(conn, legacy, "verify2", "verified", "c", "blocking")
        _step(conn, legacy, "verify2", "2026-09-30T10:20:00Z", 0)
        _step(conn, legacy, "fresh_verify2", "2026-09-30T10:40:00Z", 1)
    by_number = {r["number"]: r for r in build_day_runs(conn, ["2026-09-30"])["2026-09-30"]}
    assert by_number[40]["findings"] == {"blocking": 0, "suggestion": 0, "nitpick": 0}
    assert by_number[41]["findings"] == {"blocking": 1, "suggestion": 0, "nitpick": 0}


def test_day_runs_measure_a_retry_from_its_predecessor_and_normalize_backfilled_verdicts(conn):
    with tx(conn):
        first = _finished_run(conn, 50, status="failed", finished="2026-09-30T10:30:00Z")
        head = conn.execute("SELECT head_id FROM runs WHERE id=?", (first,)).fetchone()[0]
        retry = conn.execute(
            "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at, "
            "finished_at) VALUES (?,?,?,?,?,?,?)",
            (head, 2, "done", "t50b", "2026-09-30T10:45:00Z", "t", "2026-09-30T11:00:00Z"),
        ).lastrowid
        conn.execute("UPDATE heads SET eligible_at='2026-09-30T10:35:00Z' WHERE id=?", (head,))
        # a retry that found its publication already on GitHub records the review *state*
        conn.execute(
            "INSERT INTO reviews (run_id, repo, number, sha, phase, event, posted_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (retry, "dashpay/platform", 50, "5" * 40, "final", "CHANGES_REQUESTED", "t"),
        )
    by_id = {r["id"]: r for r in build_day_runs(conn, ["2026-09-30"])["2026-09-30"]}
    assert by_id[retry]["wait_seconds"] == 900, "from the failed attempt's end, not queued_at"
    assert by_id[retry]["slot_wait_seconds"] == 600
    assert by_id[first]["wait_seconds"] == 3600
    assert by_id[first]["slot_wait_seconds"] is None, "eligible_at now belongs to the retry"
    assert by_id[retry]["verdict"] == "REQUEST_CHANGES"


def test_active_runs_carry_steps_lanes_and_live_lane_state(cfg, conn, tmp_path):
    """An active run shows its steps (safe detail only), finished lanes and tokens so far, and
    the lanes waiting or running right now from their lanepool state files."""
    ts = fmt_ts(now_dt() - timedelta(minutes=30))
    run_dir = tmp_path / "run-60"
    with tx(conn):
        hid = conn.execute(
            "INSERT INTO heads (repo, number, sha, trigger, status, queued_at, eligible_at) "
            "VALUES ('dashpay/platform', 60, ?, 'new_push', 'running', ?, ?)",
            ("6" * 40, fmt_ts(now_dt() - timedelta(minutes=45)), ts),
        ).lastrowid
        run = conn.execute(
            "INSERT INTO runs (head_id, attempt, status, token, started_at, heartbeat_at, "
            "deadline_at, phase, tier, run_dir) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                hid,
                1,
                "running",
                "t60",
                ts,
                ts,
                fmt_ts(now_dt() + timedelta(hours=5)),
                "phase1",
                "critical",
                str(run_dir),
            ),
        ).lastrowid
        for name, status, detail in (
            ("triage", "ok", {"tier": "critical", "reasoning": "big diff", "error": "a@b.c"}),
            ("phase1", "running", {}),
            ("context", "failed", {"error": "token for someone@example.com"}),
            (
                "phase2",
                "failed",
                {
                    "error": "phase2/general lane failed twice: [infra] lane exit 1: API Error: "
                    '429 {"error": {"message": "limit for a@b.c"}} see https://proxy.example/x '
                    "in /Users/claw/.reviewsys/work/run-60",
                    "dropped": True,
                },
            ),
            (
                "verify1",
                "failed",
                {
                    "error": "phase1/general lane failed twice: [contract] model output is not a "
                    "JSON object: 'Here is my review of the private code and what it does'"
                },
            ),
        ):
            conn.execute(
                "INSERT INTO steps (run_id, name, status, started_at, detail) VALUES (?,?,?,?,?)",
                (run, name, status, ts, json.dumps(detail)),
            )
        conn.execute(
            "INSERT INTO lanes (run_id, phase, role, agent, model, effort, attempt, attempt_id, "
            "status, tokens_in, tokens_out, started_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (run, "triage", "triage", "a", "gpt-6.1-sol", "low", 1, "x", "completed", 900, 100, ts),
        )
    slot_dir = cfg.work_dir / "lane-slots"
    (slot_dir / "glm.line").mkdir(parents=True)
    me, verifier = f"{os.getpid()}-7", f"{os.getpid()}-8"
    for name in (
        f"2-{run:012d}-{1:020d}-{verifier}",  # a side lane lines up before the reviewers
        f"4-{run:012d}-{2:020d}-{os.getpid()}-1",
        f"4-{run:012d}-{3:020d}-{me}",
    ):
        (slot_dir / "glm.line" / name).touch()

    def lane(dirname, **state):
        d = run_dir / "attempts" / dirname
        d.mkdir(parents=True)
        (d / lanepool.LANE_STATE).write_text(json.dumps({"since": ts, "pool": "glm", **state}))
        return d

    lane("phase1-general-aa", state="running", role="general", model="glm-5.3-flash", effort="max")
    lane(
        "phase1-ffi-bb",
        state="waiting",
        role="ffi",
        model="glm-5.3-flash",
        effort="max",
        parallel=True,
        waiter=me,
    )
    lane(
        "verify1-verifier-dd",
        state="waiting",
        role="verifier",
        model="glm-5.3-flash",
        effort="high",
        parallel=False,
        waiter=verifier,
    )
    done = lane("phase1-old-cc", state="running", role="old", model="glm-5.3-flash", effort="max")
    (done / "lane-meta.json").write_text("{}")  # ended: its row is in the DB, not live

    export = build_export(conn, cfg)
    (active,) = export["live"]["active"]
    assert active["wait_seconds"] == 900 and active["tokens_in"] == 900
    steps = {s["name"]: s for s in active["steps"]}
    assert list(steps) == ["triage", "phase1", "context", "phase2", "verify1"]
    assert steps["triage"]["info"] == {"tier": "critical", "reasoning": "big diff"}
    # a failed step says why on the public page, sanitized: no account, URL, path, upstream
    # body or model output
    assert steps["context"]["info"] == {"error": "token for <account>"}
    assert steps["phase2"]["info"] == {
        "error": "phase2/general lane failed twice: lane exit 1: API Error: 429 {…} see <url> "
        "in <path>"
    }
    assert steps["verify1"]["info"] == {
        "error": "phase1/general lane failed twice: model output is not a JSON object: '…'"
    }
    text = json.dumps(export)
    assert "example.com" not in text and "a@b.c" not in text and "private code" not in text
    live = {x["role"]: x for x in active["live_lanes"]}
    assert set(live) == {"general", "ffi", "verifier"}
    assert live["general"]["state"] == "running" and live["general"]["since_seconds"] >= 1800
    assert (live["ffi"]["line_position"], live["ffi"]["line_length"]) == (3, 3)
    # a waiting side lane has its place in line too (every lane lines up)
    assert (live["verifier"]["line_position"], live["verifier"]["line_length"]) == (1, 3)
    assert "run_dir" not in active
    pools = {p["pool"]: p for p in export["live"]["lane_pools"]}
    assert (pools["glm"]["running"], pools["glm"]["waiting"]) == (1, 3)
    cap = export["live"]["capacity"]
    assert cap["max_runs"] == cfg.max_runs and cap["maximum"] == 3, "the gpt pool's budget"


def test_queue_says_why_a_head_waits(cfg, conn):
    _queue(conn, cfg, 70, queued_ago_min=5, eligible_in_min=25)
    _queue(conn, cfg, 71, queued_ago_min=60, eligible_in_min=10)
    _queue(conn, cfg, 72, queued_ago_min=60, eligible_in_min=-1)
    with tx(conn):
        conn.execute("UPDATE heads SET attempts=1 WHERE number=71")
    queued = {q["number"]: q for q in build_export(conn, cfg)["live"]["queued"]}
    assert queued[70]["reason"].startswith("debounce")
    assert 1400 <= queued[70]["eligible_in_seconds"] <= 1500
    assert queued[71]["reason"] == "retry backoff after attempt 1"
    assert queued[72]["reason"] == "waiting for a review slot"
    assert queued[72]["eligible_in_seconds"] == 0


def test_events_withhold_details_that_can_quote_lane_errors(cfg, conn):
    with tx(conn):
        event(conn, "run.done", repo="dashpay/platform", number=1, detail="ok in 970s")
        event(conn, "run.failed", repo="dashpay/platform", number=1, detail="503 auth_unavailable")
        event(conn, "degraded.transition", detail="degraded (usage_limit_reached)")
        event(conn, "some.future_kind", detail="anything")
    details = {e["kind"]: e["detail"] for e in build_export(conn, cfg)["history"]["recent_events"]}
    assert details["run.done"] == "ok in 970s"
    assert details["run.failed"] is None
    assert details["degraded.transition"] is None and details["some.future_kind"] is None


def test_live_lanes_see_stand_in_retries_and_repairs(tmp_path):
    """A stand-in retry of triage reuses attempt-1 (its first lane-meta.json already there),
    and a repair lane runs in its parent's `repair/` dir: both are live while they run."""
    ts = fmt_ts(now_dt())
    triage = tmp_path / "triage" / "attempt-1"
    triage.mkdir(parents=True)
    (triage / "lane-meta.json").write_text("{}")
    state = triage / lanepool.LANE_STATE
    state.write_text(json.dumps({"state": "running", "role": "triage", "since": ts}))
    meta_ns = (triage / "lane-meta.json").stat().st_mtime_ns
    os.utime(state, ns=(meta_ns + 10**9, meta_ns + 10**9))
    repair = tmp_path / "attempts" / "phase1-general-aa" / "repair"
    repair.mkdir(parents=True)
    (repair.parent / "lane-meta.json").write_text("{}")
    (repair / lanepool.LANE_STATE).write_text(
        json.dumps({"state": "waiting", "role": "general", "since": ts, "pool": "gpt"})
    )
    lanes = {x["role"]: x for x in _live_lanes(str(tmp_path), tmp_path / "slots", now_dt())}
    assert lanes["triage"]["phase"] == "triage"
    assert lanes["general (repair)"]["phase"] == "phase1"
    os.utime(state, ns=(meta_ns - 10**9, meta_ns - 10**9))  # the lane-meta.json is newer: ended
    assert "triage" not in {x["role"] for x in _live_lanes(str(tmp_path), tmp_path, now_dt())}


_AT = parse_ts("2026-10-20T12:00:00Z")  # after SLOT_WAIT_RECORDED_SINCE, whatever today is


def _path_run(
    conn,
    number,
    *,
    priority=False,
    queue="live",
    status="done",
    waited_min=10,
    took_min=30,
    slot_wait_min=0,
    days_ago=1,
    converse=False,
):
    """A head queued `waited_min` before its one run started `days_ago` days before _AT; the
    run took `took_min` and only waited for slots for `slot_wait_min` (its deadline credit)."""
    started = _AT - timedelta(days=days_ago)
    hid = conn.execute(
        "INSERT INTO heads (repo, number, sha, trigger, priority, status, queued_at, eligible_at, "
        "queue) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            "dashpay/platform",
            number,
            f"{number:040x}",
            "mention" if priority else "new_push",
            int(priority),
            "done",
            fmt_ts(started - timedelta(minutes=waited_min)),
            fmt_ts(started),
            queue,
        ),
    ).lastrowid
    run = conn.execute(
        "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at, finished_at, "
        "path) VALUES (?,1,?,?,?,?,?,?)",
        (
            hid,
            status,
            f"t{number}",
            fmt_ts(started),
            fmt_ts(started + timedelta(minutes=360 + slot_wait_min)),  # run_timeout_minutes 360
            fmt_ts(started + timedelta(minutes=took_min)) if status != "running" else None,
            path_of(queue, priority),
        ),
    ).lastrowid
    if converse:
        conn.execute(
            "INSERT INTO steps (run_id, name, status, started_at) VALUES (?,?,?,?)",
            (run, "converse", "ok", fmt_ts(started)),
        )
    return run


def test_path_timing_is_empty_without_runs(cfg, conn):
    timing = build_path_timing(conn, cfg, _AT)
    empty = {"n": 0, "median_seconds": None, "p90_seconds": None}
    assert timing["window_days"] == 7
    assert timing["paths"] == {
        path: {"wait": empty, "duration": empty, "slot_wait": empty}
        for path in ("priority", "normal")
    }
    assert build_export(conn, cfg)["history"]["path_timing"]["paths"]["normal"]["wait"] == empty


def test_path_timing_of_a_single_run_is_that_run(cfg, conn):
    with tx(conn):
        _path_run(conn, 1, waited_min=12, took_min=40, slot_wait_min=5)
    normal = build_path_timing(conn, cfg, _AT)["paths"]["normal"]
    assert normal["wait"] == {"n": 1, "median_seconds": 720, "p90_seconds": 720}
    assert normal["duration"] == {"n": 1, "median_seconds": 2400, "p90_seconds": 2400}
    assert normal["slot_wait"] == {"n": 1, "median_seconds": 300, "p90_seconds": 300}


def test_path_timing_splits_priority_from_normal_and_leaves_audits_out(cfg, conn):
    with tx(conn):
        for i in range(10):  # normal: waits 31..40 min (debounce), reviews 60 min
            _path_run(conn, 10 + i, waited_min=31 + i, took_min=60, slot_wait_min=i)
        for i in range(3):  # priority: waits 1..3 min, reviews 20, 30, 40 min
            _path_run(conn, 30 + i, priority=True, waited_min=1 + i, took_min=20 + 10 * i)
        # left out: audits, conversations, runs before the window, and still-running runs'
        # review time (their wait counts: it ended when they started)
        _path_run(conn, 40, queue="audit", waited_min=900, took_min=900)
        _path_run(conn, 41, priority=True, converse=True, waited_min=0, took_min=1)
        _path_run(conn, 42, days_ago=8, waited_min=999, took_min=999)
        _path_run(conn, 43, priority=True, status="running", waited_min=4)
        # a failed run waited too, but its short life is no review time
        _path_run(conn, 44, priority=True, status="failed", waited_min=5, took_min=2)
    paths = build_path_timing(conn, cfg, _AT)["paths"]
    normal, priority = paths["normal"], paths["priority"]
    # p90 interpolates between the 9th and 10th of the sorted samples: 39.1 min
    assert normal["wait"] == {"n": 10, "median_seconds": 35 * 60 + 30, "p90_seconds": 2346}
    assert normal["duration"] == {"n": 10, "median_seconds": 3600, "p90_seconds": 3600}
    assert normal["slot_wait"]["n"] == 10 and normal["slot_wait"]["median_seconds"] == 270
    assert priority["wait"]["n"] == 5 and priority["wait"]["median_seconds"] == 180
    assert priority["duration"] == {"n": 3, "median_seconds": 1800, "p90_seconds": 2280}
    assert priority["slot_wait"] == {"n": 3, "median_seconds": 0, "p90_seconds": 0}


def test_path_timing_measures_a_retry_from_its_predecessor(cfg, conn):
    with tx(conn):
        first = _path_run(conn, 50, status="failed", waited_min=30, took_min=10)
        head, started = conn.execute(
            "SELECT head_id, started_at FROM runs WHERE id=?", (first,)
        ).fetchone()
        retry_start = parse_ts(started) + timedelta(minutes=25)  # 15 min after the failure
        conn.execute(
            "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at, "
            "finished_at, path) VALUES (?,2,'done','t50b',?,?,?,'normal')",
            (
                head,
                fmt_ts(retry_start),
                fmt_ts(retry_start + timedelta(minutes=360)),
                fmt_ts(retry_start + timedelta(minutes=45)),
            ),
        )
    normal = build_path_timing(conn, cfg, _AT)["paths"]["normal"]
    assert normal["wait"]["n"] == 2 and normal["wait"]["median_seconds"] == (30 + 15) * 60 // 2
    assert normal["duration"]["n"] == 1 and normal["duration"]["median_seconds"] == 45 * 60


def test_path_timing_has_no_slot_wait_for_runs_before_it_was_recorded(cfg, conn):
    """Before v0.23 a run waited for its slot before it started (its wait to start counts
    that) and its deadline was never extended: a 0 there would be a made-up figure."""
    with tx(conn):
        _path_run(conn, 60, days_ago=19.5, took_min=30)  # 2026-10-01T00:00Z
    paths = build_path_timing(conn, cfg, _AT + timedelta(days=-13))["paths"]
    assert paths["normal"]["duration"]["n"] == 1
    assert paths["normal"]["slot_wait"]["n"] == 0


def test_path_timing_keeps_a_runs_path_when_a_reply_requeues_its_head_as_priority(cfg, conn):
    """A reply under a finding re-queues the reviewed head with `priority` set and a new
    `queued_at`; the normal review that ran before must stay normal, with its own wait."""
    with tx(conn):
        run = _path_run(conn, 70, waited_min=45, took_min=50)
        conn.execute(
            "UPDATE heads SET priority=1, trigger='review_reply', queued_at=? "
            "WHERE id=(SELECT head_id FROM runs WHERE id=?)",
            (fmt_ts(_AT - timedelta(hours=1)), run),
        )
    paths = build_path_timing(conn, cfg, _AT)["paths"]
    assert paths["priority"]["wait"]["n"] == 0 and paths["priority"]["duration"]["n"] == 0
    assert paths["normal"]["duration"]["median_seconds"] == 50 * 60
    with tx(conn):
        conn.execute("UPDATE runs SET path=NULL WHERE id=?", (run,))
    assert build_path_timing(conn, cfg, _AT)["paths"]["normal"]["wait"]["n"] == 0, (
        "a run whose path is not known is left out"
    )


def test_runs_from_before_the_path_column_take_it_from_their_head_when_it_is_known(conn):
    with tx(conn):
        kept = _path_run(conn, 80, priority=True)
        requeued = _path_run(conn, 81)
        audited = _path_run(conn, 82, queue="audit")
        merged = _path_run(conn, 83)
        conn.execute("UPDATE runs SET path=NULL")
        # queued again after their run started: by a reply (priority now) and as an audit
        conn.execute("UPDATE heads SET priority=1, queued_at=? WHERE number=81", (fmt_ts(_AT),))
        conn.execute("UPDATE heads SET queue='audit', queued_at=? WHERE number=83", (fmt_ts(_AT),))
        backfill = [x for x in MIGRATIONS[5].split(";") if "UPDATE" in x]
        conn.execute(backfill[0])
    paths = dict(conn.execute("SELECT id, path FROM runs").fetchall())
    assert paths == {kept: "priority", requeued: None, audited: "audit", merged: None}
