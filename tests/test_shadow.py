"""The live v12 shadow (shadow.py): head selection, sampling, the launch gates and the quota
guard, the daily cap, concurrency, comparison, outcome scoring, the public export, and
process cleanup."""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from reviewsys import config as cfg_mod
from reviewsys import degraded, exporter, replay, replayfake, shadow, slots
from reviewsys.db import connect, fmt_ts, kv_set, now_dt, tx
from reviewsys.proc import alive

REPO = "dashpay/platform"
SECRET_TITLE = "Private overflow in secret_module"
SECRET_BODY = "quoted model output about a private PR"


# ---- a production database ----


def _ago(**kw: float) -> str:
    return fmt_ts(now_dt() - timedelta(**kw))


class Prod:
    """A production database to select heads from (written here only: tests stand in for the
    daemon)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.conn = connect(path)
        self.n = 0
        accounts = [dataclasses.asdict(slots.Account(name=f"a{i}", usable=True, reason=""))
                    for i in range(3)]  # fmt: skip
        with tx(self.conn):
            kv_set(self.conn, slots.KV_ACCOUNTS,
                   json.dumps({"at": fmt_ts(now_dt()), "accounts": accounts}))  # fmt: skip

    def run(
        self,
        number: int,
        *,
        sha: str | None = None,
        finished: str | None = None,
        started: str | None = None,
        queue: str = "live",
        path: str | None = None,
        status: str = "done",
        degraded_: int = 0,
        review: str | None = "final",
        imported: int = 0,
        event: str = "REQUEST_CHANGES",
        steps: tuple[str, ...] = (),
        pr_head: bool = True,
        findings: list[dict[str, Any]] | None = None,
        lanes: list[tuple[str, int, int]] = (),
    ) -> int:
        self.n += 1
        sha = sha or f"{self.n:040x}"
        finished = finished or _ago(minutes=5)
        started = started or fmt_ts(_parse(finished) - timedelta(hours=1))
        c = self.conn
        with tx(c):
            hid = c.execute(
                "INSERT INTO heads (repo, number, sha, trigger, status, queued_at, eligible_at, "
                "queue) VALUES (?,?,?,?,?,?,?,?)",
                (REPO, number, sha, "new_push", "done", started, started, queue),
            ).lastrowid
            rid = int(
                c.execute(
                    "INSERT INTO runs (head_id, attempt, status, token, started_at, finished_at, "
                    "deadline_at, tier, degraded, path) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (hid, 1, status, "t", started, finished, started, "normal", degraded_, path),
                ).lastrowid
                or 0
            )
            if review:
                c.execute(
                    "INSERT INTO reviews (run_id, repo, number, sha, phase, event, posted_at, "
                    "imported) VALUES (?,?,?,?,?,?,?,?)",
                    (rid, REPO, number, sha, review, event, finished, imported),
                )
            for st in steps:
                c.execute(
                    "INSERT INTO steps (run_id, name, status, started_at) VALUES (?,?,?,?)",
                    (rid, st, "ok", started),
                )
            if pr_head:
                c.execute(
                    "INSERT INTO prs (repo, number, head_sha, title, state, updated_at) "
                    "VALUES (?,?,?,?,?,?) ON CONFLICT(repo, number) DO UPDATE SET "
                    "head_sha=excluded.head_sha",
                    (REPO, number, sha, f"PR {number}", "open", finished),
                )
            for f in findings or []:
                self.finding(rid, **f)
            for model, tin, tout in lanes:
                c.execute(
                    "INSERT INTO lanes (run_id, phase, role, agent, model, attempt, attempt_id, "
                    "status, tokens_in, tokens_out, started_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (rid, "phase2", "general", "a", model, 1, "x", "completed", tin, tout, started),
                )
        return rid

    def finding(
        self,
        run_id: int,
        *,
        hash: str,
        title: str,
        file: str = "f.rs",
        line: int = 10,
        severity: str = "blocking",
        stage: str = "posted",
        phase: str = "final",
        body: str = SECRET_BODY,
        category: str = "logic",
    ) -> None:
        self.conn.execute(
            "INSERT INTO findings (run_id, phase, stage, hash, file, line_start, line_end, "
            "severity, category, title, body) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, phase, stage, hash, file, line, line, severity, category, title, body),
        )


def _parse(ts: str):
    from reviewsys.db import parse_ts

    return parse_ts(ts)


def _opts(tmp_path: Path, prod: Path, **kw: Any) -> shadow.Options:
    skills = tmp_path / "skills-v12"
    skills.mkdir(parents=True, exist_ok=True)
    return shadow.Options(
        root=tmp_path / "shadow",
        skills=skills,
        prod_db=prod,
        min_free_gb=0,
        tick_seconds=0.05,
        **kw,
    )


GO = lambda running: ("go", "", "")  # noqa: E731


class Fake:
    """A replay that never ends until told to (a launcher handle)."""

    def __init__(self, rc: int | None = None) -> None:
        self.pid = 0
        self.rc = rc

    def poll(self) -> int | None:
        return self.rc


def _runner(opts: shadow.Options, **kw: Any) -> shadow.Runner:
    kw.setdefault("gate", GO)
    kw.setdefault("launcher", lambda o, row: Fake())
    r = shadow.Runner(opts, shadow.connect(opts.db_path), **kw)
    r.set_watermark(_ago(days=2))
    return r


def _rows(r: shadow.Runner) -> dict[int, sqlite3.Row]:
    return {int(x["source_run"]): x for x in r.conn.execute("SELECT * FROM shadows")}


@pytest.fixture
def prod(tmp_path: Path) -> Prod:
    return Prod(tmp_path / "prod.db")


# ---- selection ----


def test_selection_takes_finished_published_live_reviews_only(tmp_path, prod):
    ok = prod.run(1)
    excluded = {
        "audit": prod.run(2, queue="audit", path="audit"),
        "audit (before runs.path)": prod.run(12, queue="audit"),
        "conversation": prod.run(3, steps=("converse",)),
        "carried": prod.run(4, steps=("carry",)),
        "ci re-check": prod.run(14, steps=("ci_check",)),
        "degraded": prod.run(5, degraded_=1),
        "failed": prod.run(6, status="failed"),
        "no review": prod.run(7, review=None),
        "preliminary only": prod.run(8, review="preliminary"),
        "imported": prod.run(9, imported=1),
        "too old": prod.run(10, finished=_ago(hours=30)),
    }
    old_head = prod.run(11)
    prod.run(11, review=None, status="running")  # a newer push since
    # a live review whose head an audit took over after it finished (an unclean merge)
    taken_over = prod.run(13, queue="audit", path="normal")
    r = _runner(_opts(tmp_path, prod.path))
    r.discover()
    rows = _rows(r)
    assert rows[ok]["status"] == "queued" and rows[taken_over]["status"] == "queued"
    assert rows[old_head]["status"] == "skipped" and rows[old_head]["reason"] == "superseded"
    for why, rid in excluded.items():
        assert rid not in rows, why


def test_a_carried_head_is_left_out_by_its_carry_run_not_its_column(tmp_path, prod):
    # main's schema has heads.carried_from; the query never names it either way
    cols = {r[1] for r in prod.conn.execute("PRAGMA table_info(heads)")}
    if "carried_from" not in cols:
        with tx(prod.conn):
            prod.conn.execute("ALTER TABLE heads ADD COLUMN carried_from TEXT")
    carried = prod.run(1, steps=("carry",))
    asked = prod.run(2)  # someone asked for a review of a carried commit: a real review
    with tx(prod.conn):
        prod.conn.execute("UPDATE heads SET carried_from='abc'")
    r = _runner(_opts(tmp_path, prod.path))
    r.discover()
    assert set(_rows(r)) == {asked} and carried not in _rows(r)


def test_the_watermark_keeps_history_out_unless_asked(tmp_path, prod):
    before = prod.run(1, finished=_ago(hours=2))
    opts = _opts(tmp_path, prod.path)
    r = shadow.Runner(opts, shadow.connect(opts.db_path), gate=GO)
    r.set_watermark(None)  # first start: now
    r.discover()
    assert before not in _rows(r)
    assert r.set_watermark(None) == shadow.kv_get(r.conn, "watermark")  # kept on restart
    r.set_watermark(_ago(hours=3))
    r.discover()
    assert _rows(r)[before]["status"] == "queued"


def test_sampling_is_deterministic_by_run_id(tmp_path, prod):
    picks = [shadow.sampled(i, 0.3) for i in range(2000)]
    assert picks == [shadow.sampled(i, 0.3) for i in range(2000)]
    assert 0.25 < sum(picks) / len(picks) < 0.35
    assert all(shadow.sampled(i, 1.0) for i in range(50))
    assert not any(shadow.sampled(i, 0.0) for i in range(50))
    rid = prod.run(1)
    r = _runner(_opts(tmp_path, prod.path, sample_rate=0.0))
    r.discover()
    assert _rows(r)[rid]["reason"] == "not sampled"


def test_a_run_is_never_shadowed_twice(tmp_path, prod):
    rid = prod.run(1)
    r = _runner(_opts(tmp_path, prod.path), launcher=lambda o, row: Fake(rc=1))
    r.discover()
    r.discover()
    assert r.conn.execute("SELECT COUNT(*) FROM shadows").fetchone()[0] == 1
    r.launch()
    r.reap()
    assert _rows(r)[rid]["status"] == "failed"
    r.tick()
    assert r.conn.execute("SELECT COUNT(*) FROM shadows").fetchone()[0] == 1
    assert _rows(r)[rid]["attempts"] == 1


# ---- launching ----


def test_concurrency_cap_and_newest_first(tmp_path, prod):
    ids = [prod.run(n, finished=_ago(minutes=60 - n)) for n in range(1, 6)]
    launched: list[int] = []

    def launcher(o, row):
        launched.append(int(row["source_run"]))
        return Fake()

    r = _runner(_opts(tmp_path, prod.path, concurrency=2), launcher=launcher)
    r.tick()
    r.tick()
    assert launched == [ids[4], ids[3]]
    assert len(r.running) == 2


def test_heads_superseded_or_aged_out_while_queued_are_skipped_at_launch(tmp_path, prod):
    a = prod.run(1)
    b = prod.run(2)
    r = _runner(_opts(tmp_path, prod.path, max_age_hours=1))
    r.discover()
    prod.run(1, review=None, status="running")
    r.clock = lambda: now_dt() + timedelta(minutes=10)
    r.launch()
    rows = _rows(r)
    assert rows[a]["reason"] == "superseded" and rows[b]["status"] == "running"
    c = prod.run(3)
    r.discover()
    r.running.clear()
    r.clock = lambda: now_dt() + timedelta(hours=2)
    r.launch()
    assert _rows(r)[c]["reason"] == "aged out"


def test_degraded_skips_and_records(tmp_path, prod):
    a, b = prod.run(1), prod.run(2)
    r = _runner(_opts(tmp_path, prod.path), gate=lambda n: ("skip", "degraded", "forced"))
    r.tick()
    rows = _rows(r)
    assert {rows[a]["reason"], rows[b]["reason"]} == {"degraded"}
    assert not r.running


@pytest.mark.parametrize("reason", ["accounts", "backlog", "disk"])
def test_a_production_hold_waits_without_skipping(tmp_path, prod, reason):
    rid = prod.run(1)
    verdict = {"v": ("wait", reason, "why")}
    r = _runner(_opts(tmp_path, prod.path), gate=lambda n: verdict["v"])
    r.tick()
    row = _rows(r)[rid]
    assert (row["status"], row["reason"]) == ("waiting", reason) and not r.running
    verdict["v"] = ("go", "", "")
    r.tick()
    assert _rows(r)[rid]["status"] == "running"


def test_the_production_gate_reads_the_production_database(tmp_path, prod):
    opts = _opts(tmp_path, prod.path)
    gate = shadow.production_gate(opts)
    assert gate(0) == ("go", "", "")
    with tx(prod.conn):
        kv_set(prod.conn, degraded.KV_FORCE, "on")
    assert gate(0)[:2] == ("skip", "degraded")
    with tx(prod.conn):
        prod.conn.execute("DELETE FROM kv WHERE key=?", (degraded.KV_FORCE,))
        accounts = [dataclasses.asdict(slots.Account(name="a", usable=False, reason=""))]
        kv_set(prod.conn, slots.KV_ACCOUNTS,
               json.dumps({"at": fmt_ts(now_dt()), "accounts": accounts}))  # fmt: skip
    assert gate(0)[:2] == ("wait", "accounts")
    disk = shadow.production_gate(dataclasses.replace(opts, min_free_gb=10**9,
                                                      min_usable_accounts=0))  # fmt: skip
    assert disk(0)[:2] == ("wait", "disk")


def test_the_quota_guard_holds_launches_and_logs_its_line(tmp_path, prod):
    rid = prod.run(1)
    flag = tmp_path / "allow"
    cmd = (f"if [ -f {flag} ]; then echo '{{\"ok\": true}}'; "
           f"else echo '{{\"ok\": false, \"why\": \"low\"}}'; exit 3; fi")  # fmt: skip
    r = _runner(_opts(tmp_path, prod.path, guard_cmd=cmd))
    r.tick()
    row = _rows(r)[rid]
    assert (row["status"], row["reason"]) == ("waiting", "quota guard") and not r.running
    ev = r.conn.execute("SELECT kind, detail FROM events ORDER BY id DESC").fetchone()
    assert ev["kind"] == "guard.refused" and json.loads(ev["detail"]) == {"ok": False, "why": "low"}
    assert json.loads(shadow.kv_get(r.conn, "runner.state"))["reason"] == "quota guard"
    flag.touch()
    r.tick()
    assert _rows(r)[rid]["status"] == "running"


def test_the_guard_runs_before_every_launch(tmp_path, prod):
    for n in range(3):
        prod.run(n + 1)
    calls = []

    def guard():
        calls.append(1)
        return len(calls) < 2, "{}"

    r = _runner(_opts(tmp_path, prod.path, concurrency=3), guard=guard)
    r.tick()
    assert len(calls) == 2 and len(r.running) == 1


def test_stop_on_guard_terminates_running_replays_after_two_failures(tmp_path, prod):
    rid = prod.run(1)
    allowed = {"v": True}
    procs: list[subprocess.Popen] = []

    def launcher(o, row):
        p = subprocess.Popen(["sleep", "600"], start_new_session=True)
        procs.append(p)
        return p

    opts = _opts(tmp_path, prod.path, stop_on_guard=True)
    r = _runner(opts, launcher=launcher, guard=lambda: (allowed["v"], "{}"), kill_grace=2)
    r.tick()
    assert _rows(r)[rid]["status"] == "running"
    allowed["v"] = False
    r.tick()
    assert r.running and procs[0].poll() is None  # one failure: still running
    r.tick()
    assert not r.running
    procs[0].wait(timeout=10)
    row = _rows(r)[rid]
    assert (row["status"], row["reason"], row["attempts"]) == ("queued", "quota guard", 0)


def test_without_stop_on_guard_running_replays_keep_going(tmp_path, prod):
    prod.run(1)
    prod.run(2)
    allowed = {"v": True}
    r = _runner(_opts(tmp_path, prod.path, concurrency=1), guard=lambda: (allowed["v"], "{}"))
    r.tick()
    allowed["v"] = False
    for _ in range(3):
        r.tick()
    assert len(r.running) == 1


def test_the_daily_cap_holds_launches(tmp_path, prod):
    ids = [prod.run(n + 1) for n in range(3)]
    r = _runner(_opts(tmp_path, prod.path, concurrency=5, max_heads_per_day=2),
                launcher=lambda o, row: Fake(rc=1))  # fmt: skip
    r.tick()
    r.reap()
    r.tick()
    rows = _rows(r)
    assert sum(rows[i]["status"] == "failed" for i in ids) == 2
    held = [rows[i] for i in ids if rows[i]["status"] == "waiting"]
    assert len(held) == 1 and held[0]["reason"] == "daily cap"
    # the oldest is the one held: newest first
    assert held[0]["source_run"] == ids[0]


def test_a_launch_error_fails_only_that_head(tmp_path, prod):
    rid = prod.run(1)

    def boom(o, row):
        raise OSError("no python")

    r = _runner(_opts(tmp_path, prod.path), launcher=boom)
    r.tick()
    assert _rows(r)[rid]["reason"] == "launch error"


def test_replay_argv_is_a_v12_replay_into_the_root(tmp_path, prod):
    opts = _opts(tmp_path, prod.path, config="/c.toml", fake_lanes=True)
    argv = shadow.replay_argv(opts, 42)
    assert argv[argv.index("replay") + 1] == "42"
    assert argv[argv.index("--out") + 1] == str(opts.root / "replays")
    assert argv[argv.index("--policy-name") + 1] == "v12"
    assert "--force-phase2" not in argv and "--fake-lanes" in argv


# ---- reaping ----


def _report(**kw: Any) -> dict[str, Any]:
    rep = {
        "status": "done",
        "invalid": None,
        "published": [],
        "publications": {"final": {"event": "APPROVE"}},
        "gpt": {"lanes": 4, "tokens_in": 1000, "tokens_out": 100},
        "gpt_excl_reconcile": {"lanes": 3, "tokens_in": 900, "tokens_out": 100},
        "tokens_in": 2000,
        "tokens_out": 200,
        "wall_clock_s": 7200.0,
        "active_s": 3600.0,
        "tier": "normal",
        "policy": "v12",
        "v12": {"threads_conflict": 0},
    }
    return rep | kw


def _finish_with(tmp_path, prod, rc: int, report: dict[str, Any] | None):
    rid = prod.run(1)
    r = _runner(_opts(tmp_path, prod.path), launcher=lambda o, row: Fake(rc=rc))
    r.discover()
    r.launch()
    if report is not None:
        p = r.opts.report(rid)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report))
        (p.parent / "work" / "worktrees" / "x").mkdir(parents=True)
    r.reap()
    return r, _rows(r)[rid]


def test_reap_books_each_outcome(tmp_path, prod):
    _, row = _finish_with(tmp_path / "a", Prod(tmp_path / "a.db"), 75, None)
    assert (row["status"], row["reason"]) == ("skipped", "degraded")
    _, row = _finish_with(tmp_path / "b", Prod(tmp_path / "b.db"), 1, None)
    assert (row["status"], row["reason"]) == ("failed", "no report")
    _, row = _finish_with(tmp_path / "c", Prod(tmp_path / "c.db"), 75,
                          _report(invalid="quota failure mid-run"))  # fmt: skip
    assert row["reason"] == "invalid: quota"
    _, row = _finish_with(tmp_path / "d", Prod(tmp_path / "d.db"), 1,
                          _report(status="failed", fail_kind="contract"))  # fmt: skip
    assert row["reason"] == "run failed"
    r, row = _finish_with(tmp_path / "e", Prod(tmp_path / "e.db"), 0, _report())
    assert row["status"] == "done" and row["shadow_verdict"] == "APPROVE"
    assert not (r.opts.replays / f"run-{row['source_run']}" / "work" / "worktrees").exists()


def test_a_replay_past_its_timeout_is_killed(tmp_path, prod):
    rid = prod.run(1)
    procs = []

    def launcher(o, row):
        p = subprocess.Popen(["sleep", "600"], start_new_session=True)
        procs.append(p)
        return p

    r = _runner(_opts(tmp_path, prod.path, replay_timeout_hours=1), launcher=launcher,
                kill_grace=2)  # fmt: skip
    r.tick()
    r.clock = lambda: now_dt() + timedelta(hours=2)
    r.reap()
    procs[0].wait(timeout=10)
    assert _rows(r)[rid]["reason"] == "timed out"


# ---- comparison ----


def F(hash: str, title: str, line: int = 10, severity: str = "blocking", file: str = "f.rs"):
    return {"hash": hash, "title": title, "file": file, "line_start": line, "line_end": line,
            "severity": severity, "category": "logic"}  # fmt: skip


def test_compare_findings_matches_one_to_one():
    prod = [F("p1", "Overflow in parse_len"), F("p2", "Race on cache map", 80)]
    sh = [F("s1", "parse_len overflow on long input", 14), F("s2", "Unchecked unwrap", 200,
          "suggestion")]  # fmt: skip
    pairs, prod_only, shadow_only = shadow.compare_findings(prod, sh)
    assert [(p["hash"], s["hash"]) for p, s in pairs] == [("p1", "s1")]
    assert [f["hash"] for f in prod_only] == ["p2"]
    assert [f["hash"] for f in shadow_only] == ["s2"]


def test_compare_stores_verdicts_counts_and_cost(tmp_path, prod):
    rid = prod.run(
        1,
        event="CHANGES_REQUESTED",
        findings=[dict(hash="p1", title="Overflow in parse_len"),
                  dict(hash="p2", title="Race on cache map", line=80, severity="suggestion"),
                  dict(hash="old", title="x", phase="preliminary")],
        lanes=[("gpt-6.1-sol", 500, 50), ("gpt-6.1-sol", 300, 30), ("glm-5.3-flash", 100, 10)],
    )  # fmt: skip
    r = _runner(_opts(tmp_path, prod.path))
    r.discover()
    sid = int(_rows(r)[rid]["id"])
    rep = _report(published=[F("s1", "parse_len overflow", 12), F("s2", "Leak", 300, "nitpick")])
    row = shadow.compare(r.conn, shadow.connect_ro(prod.path), sid, rid, rep)
    assert row["prod_verdict"] == "REQUEST_CHANGES" and row["shadow_verdict"] == "APPROVE"
    assert (row["prod_blocking"], row["prod_suggestion"], row["prod_nitpick"]) == (1, 1, 0)
    assert (row["shadow_blocking"], row["shadow_nitpick"]) == (1, 1)
    assert (row["matched"], row["prod_only"], row["shadow_only"]) == (1, 1, 1)
    assert (row["prod_gpt_lanes"], row["prod_gpt_tokens"], row["prod_tokens"]) == (2, 880, 990)
    # the shadow's cost leaves out the work only its own one-shot reconcile added
    assert (row["shadow_gpt_lanes"], row["shadow_gpt_tokens"]) == (3, 1000)
    assert row["prod_wall_s"] == 3600.0
    sides = {(f["side"], f["match"]) for f in r.conn.execute("SELECT * FROM findings")}
    assert sides == {("prod", "matched"), ("shadow", "matched"), ("prod", "only"),
                     ("shadow", "only")}  # fmt: skip


# ---- outcomes ----


def test_outcome_scoring_on_a_seeded_production_database(tmp_path, prod):
    src = prod.run(
        1,
        sha="a" * 40,
        finished=_ago(hours=5),
        findings=[dict(hash="fa", title="Bogus lock order", line=10),
                  dict(hash="ok", title="Missing bounds check", line=40),
                  dict(hash="both", title="Shared overflow", line=70)],
    )  # fmt: skip
    r = _runner(_opts(tmp_path, prod.path))
    r.discover()
    sid = int(_rows(r)[src]["id"])
    rep = _report(published=[F("sh", "Unchecked index in decode", 100),
                             F("both2", "Shared overflow here", 71)])  # fmt: skip
    shadow.compare(r.conn, shadow.connect_ro(prod.path), sid, src, rep)
    r._set(sid, status="done")
    # later: a conversation withdraws `fa`, the author fixes `ok`, production raises the
    # shadow's finding on the next push, and the PR merges
    later = prod.run(1, sha="b" * 40, finished=_ago(hours=1), started=_ago(hours=2),
                     findings=[dict(hash="later", title="decode: unchecked index", line=103)])  # fmt: skip
    with tx(prod.conn):
        prod.finding(later, hash="fa", title="Bogus lock order", stage="conceded", body="WITHDRAWN")
        prod.conn.execute(
            "INSERT INTO events (ts, kind, repo, number, run_id, detail) VALUES (?,?,?,?,?,?)",
            (_ago(hours=1), "thread.answered", REPO, 1, later,
             json.dumps({"finding_hash": "ok", "status": "FIXED", "action": "replied"})),
        )  # fmt: skip
        prod.conn.execute(
            "INSERT INTO audits (repo, number, sha, source, mode, queued_at, merged_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (REPO, 1, "b" * 40, "sync", "full", _ago(minutes=30), _ago(minutes=30)),
        )
    res = shadow.score(r.conn, prod.path)
    assert res == {"caught_earlier": 1, "prod_false_alarms": 1, "prod_confirmed": 1, "heads": 1}
    out = {(o["kind"], o["hash"]): o for o in r.conn.execute("SELECT * FROM outcomes")}
    assert out[("caught_earlier", "sh")]["later_run"] == later
    assert out[("prod_false_alarm", "fa")]["in_shadow"] == 0
    assert out[("prod_confirmed", "ok")]["status"] == "FIXED"
    assert out[("pr_closed", "")]["status"] == "merged"
    row = _rows(r)[src]
    assert (row["pr_state"], row["caught_earlier"]) == ("merged", 1)
    # idempotent, and an outcome that no longer holds goes away
    first_seen = out[("caught_earlier", "sh")]["observed_at"]
    with tx(prod.conn):
        prod.finding(later + 0, hash="fa", title="Bogus lock order", stage="posted")
    shadow.score(r.conn, prod.path)
    out = {(o["kind"], o["hash"]): o for o in r.conn.execute("SELECT * FROM outcomes")}
    assert ("prod_false_alarm", "fa") not in out
    assert out[("caught_earlier", "sh")]["observed_at"] == first_seen
    rep = shadow.summary(r.conn)
    assert rep["outcomes"]["shadow_caught_earlier"] == 1
    assert rep["outcomes"]["prod_confirmed_blockers"] == 1
    assert (rep["outcomes"]["prs_closed"], rep["outcomes"]["prs_merged_known"]) == (1, 1)
    assert "Shared overflow" not in json.dumps(rep)


def test_ledger_statuses_count_when_the_ledger_has_them(tmp_path, prod):
    src = prod.run(1, findings=[dict(hash="d", title="Deferred thing")])
    r = _runner(_opts(tmp_path, prod.path))
    r.discover()
    sid = int(_rows(r)[src]["id"])
    shadow.compare(r.conn, shadow.connect_ro(prod.path), sid, src, _report())
    r._set(sid, status="done")
    with tx(prod.conn):
        prod.conn.execute(
            "INSERT INTO issues (repo, number, hash, severity, title, status, status_source, "
            "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (REPO, 1, "d", "blocking", "Deferred thing", "deferred", "thread", _ago(minutes=1),
             _ago(minutes=1)),
        )  # fmt: skip
    assert shadow.score(r.conn, prod.path)["prod_false_alarms"] == 1


# ---- report and public export ----


def _populated(tmp_path, prod) -> shadow.Runner:
    rid = prod.run(1, findings=[dict(hash="p", title=SECRET_TITLE)])
    skipped = prod.run(2)
    r = _runner(_opts(tmp_path, prod.path))
    r.discover()
    rows = _rows(r)
    shadow.compare(r.conn, shadow.connect_ro(prod.path), int(rows[rid]["id"]), rid,
                   _report(published=[F("s", SECRET_TITLE + " too", 11)]))  # fmt: skip
    r._set(int(rows[rid]["id"]), status="done", finished_at=fmt_ts(now_dt()))
    r._set(int(rows[skipped]["id"]), status="failed", reason="run failed",
           detail=f"contract: {SECRET_BODY}")  # fmt: skip
    return r


def test_report_summary_and_text(tmp_path, prod):
    r = _populated(tmp_path, prod)
    s = shadow.summary(r.conn)
    assert s["heads"]["shadowed_total"] == 1 and s["heads"]["shadowed_24h"] == 1
    assert s["confusion"]["REQUEST_CHANGES"]["APPROVE"] == 1
    assert s["verdict_agreement"] == 0.0
    assert s["findings"] == {"matched": 1, "prod_only": 0, "shadow_only": 0, "agreement": 1.0}
    assert s["heads"]["failed_by_reason"] == {"run failed": 1}
    text = shadow.render(s, shadow.runner_state(r.conn, r.opts.root))
    assert "verdict agreement: 0.0%" in text and "REQUEST_CHANGES" in text


def test_public_export_carries_no_finding_text_or_failure_text(tmp_path, prod, monkeypatch):
    r = _populated(tmp_path, prod)
    r._state("waiting", "quota guard", '{"accounts": ["acct-secret"]}')
    block = shadow.public_export(r.opts.db_path)
    text = json.dumps(block)
    for secret in (SECRET_TITLE, SECRET_BODY, "contract:", "acct-secret"):
        assert secret not in text
    assert block["runner"]["reason"] == "quota guard"
    head = next(h for h in block["recent"] if h["status"] == "done")
    assert head["disagree"] and head["prod"]["blocking"] == 1 and head["matched"] == 1
    assert head["pr_url"] == f"https://github.com/{REPO}/pull/1"
    # a reason outside the vocabulary is never shown as written
    sid = r.conn.execute(
        "SELECT id FROM shadows WHERE source_run=?", (head["source_run"],)
    ).fetchone()[0]
    r._set(sid, reason="lane said: " + SECRET_BODY)
    assert SECRET_BODY not in json.dumps(shadow.public_export(r.opts.db_path))
    # and through the exporter's file
    monkeypatch.setenv(shadow.ROOT_ENV, str(r.opts.root))
    assert exporter.build_shadow() is not None
    monkeypatch.setenv(shadow.ROOT_ENV, str(tmp_path / "nowhere"))
    assert exporter.build_shadow() is None


def test_write_export_adds_the_shadow_block(cfg, conn, tmp_path, monkeypatch, prod):
    r = _populated(tmp_path, prod)
    monkeypatch.setenv(shadow.ROOT_ENV, str(r.opts.root))
    out = tmp_path / "export"
    exporter.write_export(conn, cfg, out)
    data = json.loads((out / "status.json").read_text())
    assert data["shadow"]["summary"]["heads"]["shadowed_total"] == 1
    assert SECRET_TITLE not in (out / "status.json").read_text()
    monkeypatch.setenv(shadow.ROOT_ENV, str(tmp_path / "none"))
    exporter.write_export(conn, cfg, out)
    assert "shadow" not in json.loads((out / "status.json").read_text())


# ---- process lifecycle ----


def test_one_runner_per_root(tmp_path, prod):
    a = _runner(_opts(tmp_path, prod.path))
    b = _runner(_opts(tmp_path, prod.path))
    a.lock()
    try:
        with pytest.raises(SystemExit):
            b.lock()
    finally:
        a.unlock()
    b.lock()
    b.unlock()


def test_stop_terminates_replay_groups_and_requeues(tmp_path, prod):
    rid = prod.run(1)
    procs = []

    def launcher(o, row):
        # a child of its own (the lane, in a real replay) dies with the group too
        p = subprocess.Popen(["sh", "-c", "sleep 600 & wait"], start_new_session=True)
        procs.append(p)
        return p

    r = _runner(_opts(tmp_path, prod.path), launcher=launcher, kill_grace=2)
    stop = threading.Event()
    t = threading.Thread(target=r.run_forever, args=(stop,))
    t.start()
    for _ in range(100):
        if r.running:
            break
        time.sleep(0.05)
    pgid = procs[0].pid
    stop.set()
    t.join(timeout=30)
    assert not t.is_alive()
    procs[0].wait(timeout=10)
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)
    row = _rows(r)[rid]
    assert (row["status"], row["reason"], row["attempts"]) == ("queued", "interrupted", 0)


def _ps_works() -> bool:
    try:
        return subprocess.run(["ps", "-o", "command=", "-p", str(os.getpid())],
                              capture_output=True, check=False).returncode == 0  # fmt: skip
    except OSError:
        return False


@pytest.mark.skipif(not _ps_works(), reason="ps is not usable here (sandboxed)")
def test_is_replay_reads_the_command_line():
    p = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", "reviewsys", "replay", "17"],
        start_new_session=True,
    )
    try:
        time.sleep(0.3)
        assert shadow._is_replay(p.pid, 17)
        assert not shadow._is_replay(p.pid, 1)
    finally:
        p.kill()
        p.wait()


def test_a_stop_books_a_replay_that_already_finished(tmp_path, prod):
    rid = prod.run(1)
    r = _runner(_opts(tmp_path, prod.path), kill_grace=2)
    r.discover()
    p = subprocess.Popen(["true"], start_new_session=True)
    r.launcher = lambda o, row: p
    r.launch()
    p.wait()
    r.opts.report(rid).parent.mkdir(parents=True, exist_ok=True)
    r.opts.report(rid).write_text(json.dumps(_report()))
    r.stop_all()
    assert _rows(r)[rid]["status"] == "done"


def test_a_failed_booking_never_strands_a_head(tmp_path, prod, monkeypatch):
    rid = prod.run(1)
    r = _runner(_opts(tmp_path, prod.path), launcher=lambda o, row: Fake(rc=0))
    r.tick()
    monkeypatch.setattr(r, "finish", lambda row, rc: 1 / 0)
    r.reap()
    row = _rows(r)[rid]
    assert (row["status"], row["reason"]) == ("failed", "compare error") and not r.running


def test_the_daily_cap_counts_stopped_attempts(tmp_path, prod):
    prod.run(1)
    r = _runner(_opts(tmp_path, prod.path, max_heads_per_day=1), kill_grace=1)
    procs = []

    def launcher(o, row):
        procs.append(subprocess.Popen(["sleep", "600"], start_new_session=True))
        return procs[-1]

    r.launcher = launcher
    r.tick()
    r.stop_all()  # back in the queue, but the attempt spent quota
    procs[0].wait(timeout=10)
    r.tick()
    assert not r.running and r.launched_today() == 1


def test_an_unknown_verdict_never_breaks_the_report(tmp_path, prod):
    r = _populated(tmp_path, prod)
    r.conn.execute("UPDATE shadows SET prod_verdict='DISMISSED' WHERE status='done'")
    assert shadow.summary(r.conn)["confusion"]["none"]["APPROVE"] == 1


def test_scoring_degraded_heads_are_left_out_of_the_numbers(tmp_path, prod):
    r = _populated(tmp_path, prod)
    r.conn.execute("UPDATE shadows SET scoring_degraded=1 WHERE status='done'")
    s = shadow.summary(r.conn)
    assert s["heads"]["shadowed_total"] == 1 and s["heads"]["scoring_degraded"] == 1
    assert s["verdict_agreement"] is None


def test_recover_after_a_killed_runner(tmp_path, prod, monkeypatch):
    rid = prod.run(1)
    other = prod.run(2)
    r = _runner(_opts(tmp_path, prod.path, max_attempts=2), kill_grace=2)
    # what `ps` says on the box (test_is_replay_reads_the_command_line covers the real one)
    monkeypatch.setattr(shadow, "_is_replay", lambda pid, run_id: pid == leftover.pid)
    r.discover()
    # a leftover replay of `rid` (its command line names it), and a pid that is no replay
    leftover = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(600)", "reviewsys", "replay", str(rid)],
        start_new_session=True,
    )
    bystander = subprocess.Popen(["sleep", "600"], start_new_session=True)
    rows = _rows(r)
    r._set(int(rows[rid]["id"]), status="running", pid=leftover.pid, attempts=1)
    r._set(int(rows[other]["id"]), status="running", pid=bystander.pid, attempts=2)
    (r.opts.replays / f"run-{rid}" / "work" / "mirrors" / "x").mkdir(parents=True)
    try:
        assert shadow.Runner(r.opts, r.conn, gate=GO, kill_grace=2).recover() == 2
        leftover.wait(timeout=10)
        assert alive(bystander.pid)  # never signalled: not a replay
        rows = _rows(r)
        assert (rows[rid]["status"], rows[rid]["reason"]) == ("queued", "interrupted")
        assert (rows[other]["status"], rows[other]["reason"]) == ("failed", "interrupted")
        assert not (r.opts.replays / f"run-{rid}" / "work" / "mirrors").exists()
    finally:
        for p in (leftover, bystander):
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()


def test_recover_never_runs_an_unverifiable_replay_twice_and_books_a_finished_one(
    tmp_path, prod, monkeypatch
):
    unknown, finished = prod.run(1), prod.run(2)
    r = _runner(_opts(tmp_path, prod.path), kill_grace=2)
    r.discover()
    live = subprocess.Popen(["sleep", "600"], start_new_session=True)
    monkeypatch.setattr(shadow, "_is_replay", lambda pid, run_id: None)  # `ps` cannot say
    rows = _rows(r)
    r._set(int(rows[unknown]["id"]), status="running", pid=live.pid, attempts=1)
    r._set(int(rows[finished]["id"]), status="running", pid=999999, attempts=1)
    rep = r.opts.report(finished)
    rep.parent.mkdir(parents=True)
    rep.write_text(json.dumps(_report()))
    try:
        r.recover()
        rows = _rows(r)
        assert (rows[unknown]["status"], rows[unknown]["reason"]) == ("failed", "interrupted")
        assert alive(live.pid)  # never signalled
        assert rows[finished]["status"] == "done" and rows[finished]["compared_at"]
    finally:
        for p in (live,):
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()


def test_prune_keeps_reports(tmp_path, prod):
    rid = prod.run(1)
    r = _runner(_opts(tmp_path, prod.path, keep_artifacts_days=1))
    r.discover()
    d = r.opts.replays / f"run-{rid}"
    (d / "work" / "runs").mkdir(parents=True)
    (d / "report.json").write_text("{}")
    r._set(int(_rows(r)[rid]["id"]), status="done", finished_at=_ago(days=3))
    assert r.prune() == 1
    assert (d / "report.json").exists() and not (d / "work").exists()


# ---- end to end, through replay() ----


class _NoBroker:
    def __init__(self, pin, real) -> None:
        self.path = "/nonexistent/socket"

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


def test_one_head_end_to_end_through_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "Broker", _NoBroker)
    monkeypatch.setattr(replay, "check_shim", lambda shim, profile: "")
    fx = replayfake.build_fixture(tmp_path / "fx")
    run1 = fx["source_runs"][0]
    db = Path(fx["db"])
    pc = connect(db)
    with tx(pc):
        # production published a final review of the fixture head with one of the blockers
        # the fake lanes find and one they do not
        pc.execute(
            "INSERT INTO reviews (run_id, repo, number, sha, phase, event, posted_at) "
            "SELECT r.id, h.repo, h.number, h.sha, 'final', 'REQUEST_CHANGES', r.finished_at "
            "FROM runs r JOIN heads h ON h.id=r.head_id WHERE r.id=?",
            (run1,),
        )
        for h, title, line in (("x1", "Overflow in f3", 3), ("x2", "Unrelated in f18", 18)):
            pc.execute(
                "INSERT INTO findings (run_id, phase, stage, hash, file, line_start, line_end, "
                "severity, category, title, body) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (run1, "final", "posted", h, "f.rs", line, line, "blocking", "logic", title, "b"),
            )
    pc.close()
    skills = tmp_path / "skills-v12"
    subprocess.run(["git", "clone", "-q", fx["skills_repo"], str(skills)], check=True)
    subprocess.run(["git", "-C", str(skills), "checkout", "-q", "--detach", fx["v12"]], check=True)
    cfg = cfg_mod.load(Path(fx["config"]))
    opts = shadow.Options(root=tmp_path / "shadow", skills=skills, prod_db=db,
                          max_age_hours=0, min_free_gb=0)  # fmt: skip

    class Done:
        pid = 0

        def __init__(self, rc: int) -> None:
            self.rc = rc

        def poll(self) -> int:
            return self.rc

    def in_process(o: shadow.Options, row: sqlite3.Row) -> Done:
        rep = replay.replay(cfg, run_id=int(row["source_run"]), out=o.replays, skills=o.skills,
                            policy_name="v12", gh_token="ro",
                            lane_runner=replayfake.FakeLanes())  # fmt: skip
        return Done(0 if rep["status"] == "done" else 1)

    r = shadow.Runner(opts, shadow.connect(opts.db_path), launcher=in_process, gate=GO)
    r.set_watermark("2026-01-01T00:00:00Z")
    r.tick()
    r.reap()
    row = _rows(r)[run1]
    assert row["status"] == "done", (row["reason"], row["detail"])
    assert row["prod_verdict"] == "REQUEST_CHANGES" and row["shadow_verdict"]
    assert row["matched"] >= 1 and row["prod_only"] >= 1
    assert row["policy"] and json.loads(row["v12"])
    # PR 2's run has no final review: never selected
    assert fx["source_runs"][1] not in _rows(r)
    assert not (opts.replays / f"run-{run1}" / "work" / "worktrees").exists()
    calls = Path(fx["root"]) / "gh-calls.log"
    if calls.exists():
        for c in (json.loads(x) for x in calls.read_text().splitlines()):
            assert "--method" not in c or c[c.index("--method") + 1] == "GET"


# ---- deploy ----

DEPLOY = Path(__file__).resolve().parent.parent / "deploy"

FAKE_RUNNER = r"""#!/bin/sh
# a stand-in for `reviewsys ... shadow run|status`: holds the lock and the pidfile until TERM
root=""; prev=""
for a in "$@"; do [ "$prev" = "--root" ] && root="$a"; prev="$a"; done
case " $* " in *" status "*) echo "fake status"; exit 0;; esac
echo "$*" > "$root/argv"
exec /usr/bin/lockf -s -k -t 0 "$root/shadow.lock" /bin/sh -c '
  echo $$ > "$1/shadow.pid"
  trap "rm -f \"$1/shadow.pid\"; exit 0" TERM
  while :; do sleep 0.2; done' sh "$root"
"""


def test_the_launch_agent_is_low_priority_and_separate():
    import plistlib

    p = plistlib.loads((DEPLOY / "ai.thepastaclaw.reviewsys-shadow.plist").read_bytes())
    assert p["Label"] == "ai.thepastaclaw.reviewsys-shadow"
    assert p["Nice"] == 10 and p["ProcessType"] == "Background"
    args = p["ProgramArguments"]
    assert args[args.index("shadow") + 1] == "run"
    assert args[args.index("--skills") + 1] == "/Users/claw/Projects/skills-v12"
    assert "quota_guard.py" in args[args.index("--guard-cmd") + 1]


@pytest.mark.skipif(
    not Path("/usr/bin/lockf").exists(), reason="deploy/shadow.sh runs on the macOS box (lockf)"
)
def test_shadow_sh_starts_only_when_enabled_and_stops_cleanly(tmp_path):
    home = tmp_path / "home"
    venv = home / ".reviewsys" / "venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").symlink_to(sys.executable)
    (venv / "reviewsys").write_text(FAKE_RUNNER)
    (venv / "reviewsys").chmod(0o755)
    skills = home / "Projects" / "skills-v12"
    skills.mkdir(parents=True)
    (skills / "config.json").write_text("{}")
    root = home / ".reviewsys-shadow"
    env = {**os.environ, "HOME": str(home), "REVIEWSYS_SHADOW_GUARD": "echo ok"}
    env.pop("REVIEWSYS_SHADOW_ROOT", None)

    def sh(*a: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["bash", str(DEPLOY / "shadow.sh"), *a], env=env,
                              capture_output=True, text=True, timeout=60, check=False)  # fmt: skip

    assert sh().returncode == 2
    assert "not enabled" in sh("start").stdout
    root.mkdir()
    (root / "enabled").touch()
    assert sh("start").returncode == 0
    for _ in range(50):
        if (root / "shadow.pid").exists():
            break
        time.sleep(0.1)
    pid = int((root / "shadow.pid").read_text())
    argv = (root / "argv").read_text()
    assert "shadow run" in argv and f"--skills {skills}" in argv and "--guard-cmd echo ok" in argv
    sh("start")  # a second start is a no-op
    assert int((root / "shadow.pid").read_text()) == pid
    st = sh("status")
    assert f"runner pid: {pid}" in st.stdout and "enabled: yes" in st.stdout
    assert "stopped" in sh("stop").stdout
    assert not (root / "enabled").exists() and not alive(pid)
    assert "not enabled" in sh("start").stdout
    # a stale pidfile naming someone else's live process: never signalled, never trusted
    other = subprocess.Popen(["sleep", "600"], start_new_session=True)
    try:
        (root / "shadow.pid").write_text(f"{other.pid}\n")
        (root / "enabled").touch()
        assert "not running" in sh("pause").stdout and alive(other.pid)
        assert "started" in sh("start").stdout
        for _ in range(50):
            if (root / "shadow.pid").read_text().strip() != str(other.pid):
                break
            time.sleep(0.1)
        new = int((root / "shadow.pid").read_text())
        assert new != other.pid
        assert "stopped" in sh("pause").stdout and (root / "enabled").exists()
        assert alive(other.pid) and not alive(new)
    finally:
        other.kill()
        other.wait()


def test_a_head_is_shadowed_once_whatever_reviews_it_again(tmp_path, prod):
    first = prod.run(7, finished=_ago(minutes=20))
    c = prod.conn
    hid, started = c.execute("SELECT head_id, started_at FROM runs WHERE id=?", (first,)).fetchone()
    with tx(c):
        again = int(
            c.execute(
                "INSERT INTO runs (head_id, attempt, status, token, started_at, finished_at, "
                "deadline_at, tier) VALUES (?,?,?,?,?,?,?,?)",
                (hid, 1, "done", "t2", started, _ago(minutes=5), started, "normal"),
            ).lastrowid
            or 0
        )
        c.execute(
            "INSERT INTO reviews (run_id, repo, number, sha, phase, event, posted_at) "
            "SELECT ?, repo, number, sha, 'final', 'COMMENT', ? FROM heads WHERE id=?",
            (again, _ago(minutes=5), hid),
        )
    r = _runner(_opts(tmp_path, prod.path))
    r.discover()
    rows = _rows(r)
    assert rows[first]["status"] == "queued"
    assert rows[again]["status"] == "skipped" and rows[again]["reason"] == "head already shadowed"
