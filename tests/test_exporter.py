"""The public status export must show the queue in the order the scheduler drains it."""

from __future__ import annotations

import json
import os
from datetime import timedelta

from reviewsys import degraded, lanepool
from reviewsys.db import event, fmt_ts, now_dt, tx
from reviewsys.exporter import _live_lanes, build_day_runs, build_export, write_export
from reviewsys.ingest import enqueue_head
from reviewsys.models import Trigger
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
    assert [s["name"] for s in active["steps"]] == ["triage", "phase1", "context"]
    assert active["steps"][0]["info"] == {"tier": "critical", "reasoning": "big diff"}
    assert active["steps"][2]["info"] == {}
    assert "example.com" not in json.dumps(export) and "a@b.c" not in json.dumps(export)
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
