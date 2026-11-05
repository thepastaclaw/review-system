"""`reviewsys replay-set` (resumable, arm order alternating, degraded abort) and
`reviewsys replay-score` (matcher, paired per-set metrics, noise floor, blinded judge list)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from reviewsys import replayscore, replayset
from reviewsys.replay import EXIT_ABORTED
from reviewsys.replayscore import Label, Sample, jaccard, matches, score
from reviewsys.replayset import GB, Gate, Job, Policy, SetItem, load_set, plan, run_set


def _policies(tmp_path: Path) -> list[Policy]:
    return [Policy("v11", tmp_path / "s11"), Policy("v12", tmp_path / "s12")]


def _report(job: Job, status: str = "done", **kw: Any) -> dict[str, Any]:
    return {
        "source_run": job.item.run_id,
        "repo": "o/r",
        "number": job.item.run_id,
        "sha": f"{job.item.run_id:040d}",
        "status": status,
        "invalid": None,
        "published": [],
        **kw,
    }


def _write(job: Job, out: Path, report: dict[str, Any]) -> None:
    job.dir(out).mkdir(parents=True, exist_ok=True)
    (job.dir(out) / "report.json").write_text(json.dumps(report))


# ---- replay-set ----


def test_set_file_forms(tmp_path):
    p = tmp_path / "set.jsonl"
    p.write_text('{"run_id": 5, "set": "S1", "samples": 2}\n\n{"run_id": 6}\n')
    assert load_set(p) == [SetItem(5, "S1", 2), SetItem(6, "", 1)]
    p.write_text(json.dumps({"heads": [7, {"run_id": 8, "set": "S4", "note": "fp"}]}))
    assert load_set(p) == [SetItem(7), SetItem(8, "S4", 1, "fp")]
    p.write_text("[5, 5]")
    with pytest.raises(SystemExit, match="twice"):
        load_set(p)
    # eval/set.json: `source_run_id`, head ids, and `repeats` (two samples of those heads)
    p.write_text(
        json.dumps(
            {
                "heads": [
                    {"id": "dash-1-aa", "set": "S1", "source_run_id": 11, "why": "late blocker"},
                    {"id": "dash-2-bb", "set": "S4", "source_run_id": 12},
                ],
                "repeats": ["dash-2-bb"],
            }
        )
    )
    assert load_set(p) == [
        SetItem(11, "S1", 1, "late blocker", "dash-1-aa"),
        SetItem(12, "S4", 2, "", "dash-2-bb"),
    ]


def test_plan_runs_every_arm_per_head_and_alternates_which_goes_first(tmp_path):
    jobs = plan([SetItem(1), SetItem(2, samples=2), SetItem(3)], _policies(tmp_path))
    assert [(j.item.run_id, j.sample, j.policy.name) for j in jobs] == [
        (1, 0, "v11"),
        (1, 0, "v12"),
        (2, 0, "v12"),
        (2, 0, "v11"),
        (2, 1, "v11"),
        (2, 1, "v12"),
        (3, 0, "v12"),
        (3, 0, "v11"),
    ]
    assert [j.name for j in jobs[4:6]] == ["run-2-r1", "run-2-r1"]


def test_run_set_resumes_skipping_completed_replays(tmp_path):
    out, pols = tmp_path / "out", _policies(tmp_path)
    items = [SetItem(1), SetItem(2)]
    done = Job(items[0], pols[0], 0)
    _write(done, out, _report(done))
    failed = Job(items[1], pols[1], 0)
    _write(failed, out, _report(failed, status="failed"))
    ran: list[str] = []

    def runner(job: Job, out_: Path) -> int:
        ran.append(f"{job.policy.name}/{job.name}")
        _write(job, out_, _report(job))
        return 0

    res = run_set(items, pols, out, concurrency=2, runner=runner, echo=lambda s: None)
    assert sorted(ran) == ["v11/run-2", "v12/run-1"]  # done and failed ones kept as they are
    assert res["skipped"] == 2 and res["ok"] == 2 and not res["aborted"]
    ran.clear()
    res = run_set(items, pols, out, concurrency=2, runner=runner, echo=lambda s: None)
    assert ran == [] and res["skipped"] == 4
    seen_stale: list[bool] = []

    def retry(job: Job, out_: Path) -> int:
        seen_stale.append((job.dir(out_) / "report.json").exists())
        return runner(job, out_)

    _write(failed, out, _report(failed, status="failed"))
    ran.clear()
    res = run_set(
        items, pols, out, concurrency=2, runner=retry, retry_failed=True, echo=lambda s: None
    )
    assert ran == ["v12/run-2"] and seen_stale == [False]  # the old report goes first
    manifest = json.loads((out / replayset.MANIFEST).read_text())
    assert [p["name"] for p in manifest["policies"]] == ["v11", "v12"]


def test_run_set_respects_concurrency(tmp_path):
    out, pols = tmp_path / "out", _policies(tmp_path)
    live, peak = [0], [0]
    lock = threading.Lock()
    gate = threading.Event()

    def runner(job: Job, out_: Path) -> int:
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        gate.wait(0.05)
        with lock:
            live[0] -= 1
        _write(job, out_, _report(job))
        return 0

    run_set(
        [SetItem(i) for i in range(1, 6)],
        pols,
        out,
        concurrency=3,
        runner=runner,
        echo=lambda s: None,
    )
    assert peak[0] == 3


def test_the_manifest_keeps_arms_run_separately(tmp_path):
    out, (v11, v12) = tmp_path / "out", _policies(tmp_path)
    run_set([SetItem(1)], [v11], out, concurrency=1, runner=lambda j, o: 0, echo=lambda s: None)
    run_set([SetItem(2)], [v12], out, concurrency=1, runner=lambda j, o: 0, echo=lambda s: None)
    manifest = json.loads((out / replayset.MANIFEST).read_text())
    assert [p["name"] for p in manifest["policies"]] == ["v11", "v12"]
    assert sorted(h["run_id"] for h in manifest["heads"]) == [1, 2]


def test_an_abort_stops_the_set_even_over_a_stale_report(tmp_path):
    out, pols = tmp_path / "out", _policies(tmp_path)
    stale = Job(SetItem(1), pols[0], 0)
    _write(stale, out, _report(stale, status="failed"))
    res = run_set(
        [SetItem(1), SetItem(2)],
        pols,
        out,
        concurrency=1,
        runner=lambda j, o: EXIT_ABORTED,
        retry_failed=True,
        echo=lambda s: None,
    )
    assert res["aborted"] and res["ran"] == 1 and res["not_started"] == 3
    assert EXIT_ABORTED == 75


def test_a_degraded_abort_stops_the_set(tmp_path):
    out, pols = tmp_path / "out", _policies(tmp_path)
    ran: list[str] = []

    def runner(job: Job, out_: Path) -> int:
        ran.append(job.name)
        return EXIT_ABORTED  # refused to start: no report

    res = run_set(
        [SetItem(i) for i in range(1, 4)],
        pols,
        out,
        concurrency=1,
        runner=runner,
        echo=lambda s: None,
    )
    assert ran == ["run-1"] and res["aborted"] and res["not_started"] == 5


def test_policy_argument(tmp_path):
    (tmp_path / "s").mkdir()
    with pytest.raises(SystemExit, match=r"config\.json"):
        replayset.parse_policy(f"v12={tmp_path / 's'}")
    (tmp_path / "s" / "config.json").write_text("{}")
    assert replayset.parse_policy(f"v12={tmp_path / 's'}").name == "v12"
    with pytest.raises(SystemExit, match="NAME=SKILLS_DIR"):
        replayset.parse_policy("v12")


def test_replay_set_cli_runs_replays_as_subprocesses(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(argv, **kw):
        calls.append(argv)
        out = Path(argv[argv.index("--out") + 1])
        rid = int(argv[argv.index("replay") + 1])
        sample = int(argv[argv.index("--sample") + 1])
        d = out / replayset.run_dir_name(rid, sample)
        d.mkdir(parents=True)
        (d / "report.json").write_text(json.dumps({"status": "done"}))

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(replayset.subprocess, "run", fake_run)
    monkeypatch.setattr(replayset, "free_bytes", lambda p: 10**15)
    for name in ("s11", "s12"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.json").write_text("{}")
    setfile = tmp_path / "set.json"
    setfile.write_text(json.dumps([{"run_id": 9, "samples": 2}]))
    from reviewsys.cli import main

    rc = main(
        [
            "replay-set",
            str(setfile),
            "--policy",
            f"v11={tmp_path / 's11'}",
            "--policy",
            f"v12={tmp_path / 's12'}",
            "--out",
            str(tmp_path / "out"),
            "--concurrency",
            "1",
            "--force-phase2",
            "--pin-phase1-model",
            "gemini-3.5-pro",
            "--no-production-guard",
        ]
    )
    assert rc == 0 and len(calls) == 4
    argv = calls[0]
    assert argv[1:4] == ["-m", "reviewsys.cli", "replay"] or "replay" in argv
    assert (
        "--force-phase2" in argv and argv[argv.index("--pin-phase1-model") + 1] == "gemini-3.5-pro"
    )
    assert {Path(a[a.index("--out") + 1]).name for a in calls} == {"v11", "v12"}
    assert (tmp_path / "out" / "v12" / "run-9-r1" / "report.json").exists()
    assert "--fake-lanes" not in argv
    # a repository the box has no mirror of goes to one place for both arms, never production
    assert {a[a.index("--eval-mirrors") + 1] for a in calls} == {str(tmp_path / "out" / ".mirrors")}


def test_replay_set_cli_limit_fake_lanes_and_disk_stop(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(replayset.subprocess, "run", lambda argv, **kw: calls.append(argv))
    monkeypatch.setattr(replayset, "free_bytes", lambda p: 3 * replayset.GB)
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "config.json").write_text("{}")
    setfile = tmp_path / "set.json"
    setfile.write_text(json.dumps([{"run_id": i, "set": "S1"} for i in range(1, 4)]))
    from reviewsys.cli import main

    args = ["replay-set", str(setfile), "--policy", f"v11={tmp_path / 's'}"]
    args += ["--out", str(tmp_path / "out"), "--limit", "1", "--fake-lanes"]
    args += ["--no-production-guard", "--disk-wait-minutes", "0", "--disk-per-job-gb", "6"]
    assert main([*args, "--min-free-gb", "5"]) == replayset.EXIT_DISK and calls == []
    monkeypatch.setattr(
        replayset.subprocess,
        "run",
        lambda argv, **kw: calls.append(argv) or type("R", (), {"returncode": 0})(),
    )
    assert main([*args, "--min-free-gb", "2"]) == 1  # ran, but wrote no report
    assert len(calls) == 1 and "--fake-lanes" in calls[0]
    manifest = json.loads((tmp_path / "out" / replayset.MANIFEST).read_text())
    assert manifest["options"]["limit"] == 1 and manifest["options"]["fake_lanes"] is True


def test_limit_takes_heads_round_robin_over_the_sets():
    items = [
        SetItem(1, "S1"),
        SetItem(2, "S1"),
        SetItem(3, "S2"),
        SetItem(4, "S3"),
        SetItem(5, "S1"),
    ]
    assert [i.run_id for i in replayset.limit_items(items, 3)] == [1, 3, 4]
    assert [i.run_id for i in replayset.limit_items(items, 4)] == [1, 2, 3, 4]
    assert replayset.limit_items(items, None) == items and replayset.limit_items(items, 9) == items


def test_a_short_disk_holds_launches_until_running_replays_free_it(tmp_path):
    out, pols = tmp_path / "out", _policies(tmp_path)[:1]
    events: list[str] = []
    # what the disk check reads, call by call: room for the first replay, then short while it
    # runs (its worktree), then room again once it ended
    readings = iter([30, 10, 30, 30])

    def runner(job: Job, out_: Path) -> int:
        events.append(f"start {job.name}")
        _write(job, out_, _report(job))
        events.append(f"end {job.name}")
        return 0

    echoes: list[str] = []
    gate = Gate(out, min_free_gb=25, job_gb=0, disk_free=lambda p: next(readings, 30) * GB)
    res = run_set(
        [SetItem(1), SetItem(2)], pols, out, concurrency=2, runner=runner,
        echo=echoes.append, gate=gate,
    )  # fmt: skip
    assert res["ok"] == 2 and not res["aborted"]
    assert events == ["start run-1", "end run-1", "start run-2", "end run-2"]
    assert any("holding" in e for e in echoes) and any("again" in e for e in echoes)
    free = {"gb": 10}  # short from the start, and nothing running to free it
    slept: list[float] = []
    gate = Gate(
        out, min_free_gb=25, disk_wait_minutes=2, disk_free=lambda p: free["gb"] * GB,
        sleep=slept.append,
    )  # fmt: skip
    res = run_set(
        [SetItem(1), SetItem(2), SetItem(3)], pols, tmp_path / "out2", concurrency=1,
        runner=runner, echo=echoes.append, gate=gate,
    )  # fmt: skip
    assert res["aborted"].startswith("disk:") and res["not_started"] == 3
    assert replayset.stopped_exit(res["aborted"]) == replayset.EXIT_DISK
    assert slept == [replayset.WAIT_POLL_SECONDS, replayset.WAIT_POLL_SECONDS]


def test_each_running_replay_counts_against_the_floor(tmp_path):
    """M4: 50 GB free, floor 35, 6 GB per running replay: two may run, not three, though the
    free space has not moved yet (their worktrees are still being checked out)."""
    gate = Gate(tmp_path, min_free_gb=35, job_gb=6, disk_free=lambda p: 50 * GB)
    assert gate.check(0) == ("", "") and gate.check(2) == ("", "")
    why, kind = gate.check(3)
    assert kind == "disk" and "3 running" in why


def test_a_disk_that_recovers_while_waiting_resumes_launching(tmp_path):
    out, pols = tmp_path / "out", _policies(tmp_path)[:1]
    free = {"gb": 10}

    def sleep(s: float) -> None:
        free["gb"] = 40  # production's gc freed space meanwhile

    gate = Gate(
        out, min_free_gb=25, disk_wait_minutes=5, disk_free=lambda p: free["gb"] * GB, sleep=sleep
    )
    res = run_set(
        [SetItem(1)], pols, out, concurrency=1,
        runner=lambda j, o: (_write(j, o, _report(j)), 0)[1], echo=lambda s: None, gate=gate,
    )  # fmt: skip
    assert res["ok"] == 1 and not res["aborted"]


# ---- the production guard ----


def _prod_db(tmp_path: Path, *, usable: int = 3, at: str | None = None, queued: int = 0) -> Path:
    import dataclasses as dc

    from reviewsys import slots
    from reviewsys.db import connect, kv_set, now, tx

    db = tmp_path / "prod.db"
    conn = connect(db)
    accounts = [
        dc.asdict(slots.Account(name=f"a{i}", usable=i < usable, reason="")) for i in range(3)
    ]
    with tx(conn):
        kv_set(conn, slots.KV_ACCOUNTS, json.dumps({"at": at or now(), "accounts": accounts}))
        for i in range(queued):
            conn.execute(
                "INSERT INTO heads (repo, number, sha, trigger, status, queued_at, eligible_at) "
                "VALUES ('o/r', ?, ?, 'new_push', 'queued', ?, ?)",
                (i, f"{i:040d}", now(), now()),
            )
    conn.close()
    return db


def test_production_guard(tmp_path):
    from reviewsys import degraded
    from reviewsys.db import connect, kv_set, tx

    guard = replayset.production_guard
    db = _prod_db(tmp_path)
    assert guard(db, min_accounts=2, backlog_limit=10) == ("", False)
    db = _prod_db(tmp_path / "a", usable=1)
    assert guard(db, min_accounts=2, backlog_limit=10) == (
        "1 usable OpenAI account(s), fewer than 2",
        True,
    )
    assert guard(db, min_accounts=0, backlog_limit=10) == ("", False)
    db = _prod_db(tmp_path / "b", at="2026-01-01T00:00:00Z")
    why, stop = guard(db, min_accounts=2, backlog_limit=10)
    assert stop and "stale" in why
    db = _prod_db(tmp_path / "c", queued=5)
    why, stop = guard(db, min_accounts=2, backlog_limit=10)
    assert not stop and "5 live heads queued" in why
    db = _prod_db(tmp_path / "d")
    conn = connect(db)
    with tx(conn):
        kv_set(conn, degraded.KV_STATE, json.dumps({"active": True}))
    conn.close()
    assert guard(db, min_accounts=2, backlog_limit=10) == ("degraded mode is active", True)
    with tx(connect(db)) as c:
        kv_set(c, degraded.KV_FORCE, "on")
    assert guard(db, min_accounts=2, backlog_limit=10)[1] is True
    assert guard(tmp_path / "none.db", min_accounts=2, backlog_limit=10)[1] is True


def test_production_degrading_stops_the_set_cleanly(tmp_path):
    out, pols = tmp_path / "out", _policies(tmp_path)[:1]
    states = iter([("", False), ("degraded mode is active", True)])
    gate = Gate(out, guard=lambda: next(states, ("degraded mode is active", True)))
    res = run_set(
        [SetItem(1), SetItem(2), SetItem(3)], pols, out, concurrency=1,
        runner=lambda j, o: (_write(j, o, _report(j)), 0)[1], echo=lambda s: None, gate=gate,
    )  # fmt: skip
    assert res["ok"] == 1 and res["not_started"] == 2
    assert res["aborted"] == "stopped: degraded mode is active"
    assert replayset.stopped_exit(res["aborted"]) == EXIT_ABORTED
    # a busy production queue holds launches, then the wait runs out: also a clean stop
    slept: list[float] = []
    gate = Gate(
        out, guard=lambda: ("9 live heads queued", False), busy_wait_minutes=1, sleep=slept.append
    )
    res = run_set(
        [SetItem(4)], pols, out, concurrency=1, runner=lambda j, o: 0, echo=lambda s: None,
        gate=gate,
    )  # fmt: skip
    assert res["aborted"] == "stopped: 9 live heads queued" and slept == [60.0]


def test_the_replay_set_cli_reads_the_configured_database_before_launching(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(replayset.subprocess, "run", lambda argv, **kw: calls.append(argv))
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "config.json").write_text("{}")
    db = _prod_db(tmp_path, usable=1)
    from reviewsys.config import DEFAULT_TOML

    config = tmp_path / "config.toml"
    config.write_text(DEFAULT_TOML.replace('db = "~/.reviewsys/review.db"', f'db = "{db}"'))
    setfile = tmp_path / "set.json"
    setfile.write_text(json.dumps([{"run_id": 1}]))
    from reviewsys.cli import main

    rc = main(
        ["--config", str(config), "replay-set", str(setfile), "--policy",
         f"v11={tmp_path / 's'}", "--out", str(tmp_path / "out"), "--min-free-gb", "0"]
    )  # fmt: skip
    assert rc == EXIT_ABORTED and calls == []


# ---- matcher ----


def _f(
    title: str, line: int = 10, file: str = "a.rs", sev: str = "blocking", **kw: Any
) -> dict[str, Any]:
    return {
        "file": file,
        "line_start": line,
        "line_end": line,
        "title": title,
        "severity": sev,
        "hash": kw.pop("hash", title),
        **kw,
    }


def test_matcher_needs_location_and_title_overlap():
    label = {
        "file": "a.rs",
        "line_start": 20,
        "line_end": 22,
        "title": "Fee estimate omits drainage cost",
        "hash": "",
        "category": "",
    }
    assert matches(_f("Fee estimation omits drainage costs", 12), label)  # 8 lines away
    assert not matches(_f("Fee estimation omits drainage costs", 9), label)  # 11 lines away
    assert not matches(_f("Unrelated null deref", 21), label)  # right place, other issue
    assert not matches(_f("Fee estimate omits drainage cost", 21, file="b.rs"), label)
    assert jaccard("the a", "the a") == 0.0  # stopwords only: no evidence


def test_matcher_falls_back_to_compare_identity():
    label = {
        "file": "a.rs",
        "line_start": 40,
        "line_end": 40,
        "title": "x",
        "hash": "h1",
        "category": "logic",
    }
    assert matches(_f("Something reworded", 90, hash="h1"), label)  # same finding hash
    label["title"] = "Fee estimation omits drainage costs on change outputs"
    # same file, category and lines, a little title overlap (Jaccard 1/8, under 0.2)
    assert matches(_f("Wrong fee", 40, hash="h2", category="logic"), label)
    assert not matches(_f("Wrong fee", 40, hash="h2", category="style"), label)
    assert not matches(_f("Unrelated thing", 40, hash="h2", category="logic"), label)  # no overlap
    nocat = {**label, "hash": "", "category": ""}
    assert not matches(_f("Wrong fee", 40, hash=""), nocat)  # no hash/category luck


def test_matching_is_one_to_one():
    labels = [
        {
            "file": "a.rs",
            "line_start": 10,
            "line_end": 10,
            "title": "Overflow in fee calc",
            "hash": "",
            "category": "",
        },
        {
            "file": "a.rs",
            "line_start": 12,
            "line_end": 12,
            "title": "Overflow in fee path",
            "hash": "",
            "category": "",
        },
    ]
    got, rest = replayscore.assign([_f("Overflow in fee calc", 11)], labels)
    assert list(got) == [0] and rest == []  # the closer title wins; one label only
    got, rest = replayscore.assign(
        [_f("Overflow in fee calc", 11), _f("Overflow in the fee calc", 10, hash="x")], labels[:1]
    )
    assert len(got) == 1 and len(rest) == 1  # the second finding is not the same label again


# ---- scoring ----


def _label(n: int, title: str, label: str, set_: str, line: int = 10, **kw: Any) -> Label:
    return Label("o/r", n, f"{n:040d}", "a.rs", line, line, title, label, set_, **kw)


def _sample(
    policy: str,
    n: int,
    findings: list[dict[str, Any]],
    set_: str = "S1",
    k: int = 0,
    samples: int = 1,
    **rep: Any,
) -> Sample:
    item = SetItem(n, set_, samples)
    report = {
        "repo": "o/r",
        "number": n,
        "sha": f"{n:040d}",
        "status": "done",
        "invalid": None,
        "published": findings,
        "wall_clock_s": 100.0,
        "gpt": {"lanes": 4, "tokens_in": 1000, "tokens_out": 100},
        "tokens_in": 5000,
        "tokens_out": 500,
        "contract": {"run_failed": False, "failed_attempts": 0},
        **rep,
    }
    return Sample(policy, item, k, report)


def test_scorer_paired_metrics_per_set():
    items = [
        SetItem(1, "S1"),
        SetItem(2, "S1"),
        SetItem(3, "S4"),
        SetItem(4, "S5"),
        SetItem(5, "S1"),
    ]
    labels = [
        _label(1, "Overflow in fee calc", "tp", "S1"),
        _label(2, "Race in wallet unlock", "tp", "S1"),
        _label(3, "FindDevNetGenesisBlock unused", "fp", "S4"),
        _label(4, "Deferred migration rename", "deferred", "S5"),
        _label(4, "Withdrawn but code changed", "fp", "S5", line=50, anchor_changed=True),
    ]
    v11 = [
        _sample("v11", 1, [_f("Overflow in the fee calc", sev="suggestion")]),
        _sample("v11", 2, []),
        _sample("v11", 3, [_f("FindDevNetGenesisBlock is unused")], set_="S4"),
        _sample(
            "v11",
            4,
            [_f("Deferred migration rename", sev="nitpick")],
            set_="S5",
            wall_clock_s=300.0,
        ),
    ]
    v12 = [
        _sample("v12", 1, [_f("Overflow in fee calc")]),
        _sample("v12", 2, [_f("Race in wallet unlock path"), _f("Brand new blocker", line=200)]),
        _sample("v12", 3, [], set_="S4"),
        _sample(
            "v12",
            4,
            [_f("Withdrawn but code changed", line=50, sev="suggestion")],
            set_="S5",
            gpt={"lanes": 6, "tokens_in": 2000, "tokens_out": 200},
        ),
        _sample("v12", 5, []),  # v11 has no usable sample of head 5: left out (unpaired)
    ]
    v11.append(_sample("v11", 5, [], status="failed"))
    result, to_judge, key = score(items, labels, v11 + v12, ["v11", "v12"])
    assert result["heads"] == {
        "in_set": 5,
        "paired": 4,
        "unpaired": [5],
        "thread_resolution_as_of_now": [],
    }
    s1 = result["sets"]["S1"]
    assert s1["v11"]["recall"] == 0.5 and s1["v12"]["recall"] == 1.0
    assert s1["v11"]["blocker_recall"] == 0.0 and s1["v12"]["blocker_recall"] == 1.0
    assert s1["delta_vs_v11"]["v12"]["recall"] == 50.0
    assert s1["v12"]["new_blockers"] == 1 and s1["v11"]["new_blockers"] == 0
    s4 = result["sets"]["S4"]
    assert (s4["v11"]["fp_blockers"], s4["v12"]["fp_blockers"]) == (1, 0)
    assert s4["v11"]["heads_with_blocker_rate"] == 1.0
    s5 = result["sets"]["S5"]
    # a settled issue re-raised at any severity on an unchanged anchor counts; a changed one not
    assert (s5["v11"]["reraised_unchanged"], s5["v12"]["reraised_unchanged"]) == (1, 0)
    assert s5["v11"]["fp_blockers"] == 0
    arms = result["arms"]
    assert arms["v11"]["wall_clock_p50_s"] == 100.0 and arms["v11"]["failed"] == 0
    assert arms["v12"]["gpt_lanes_per_run"] == 4.5
    assert result["ratios_vs_baseline"]["v12"]["gpt_lanes_per_run"] == 1.125
    assert len(to_judge) == 1 and to_judge[0]["title"] == "Brand new blocker"
    assert "v12" not in json.dumps(to_judge)  # blinded
    assert key[to_judge[0]["id"]]["raised_by"] == [("v12", 0)]


def test_scorer_counts_duplicates_and_contract_failures():
    items = [SetItem(1, "S6")]
    dup = [_f("A", 10, hash="h1"), _f("B", 12, hash="h2"), _f("C", 30, hash="h3")]
    samples = [
        _sample("v11", 1, dup, set_="S6", contract={"run_failed": True, "failed_attempts": 2}),
        _sample("v12", 1, dup[:1], set_="S6"),
    ]
    result, _, _ = score(items, [], samples, ["v11", "v12"])
    assert result["arms"]["v11"]["duplicates_per_run"] == 1.0
    assert result["arms"]["v12"]["duplicates_per_run"] == 0.0
    assert result["arms"]["v11"]["contract_failures"]["runs_failed"] == 1
    assert result["arms"]["v11"]["contract_failures"]["lane_attempts_failed"] == 2


def test_noise_floor_from_repeat_samples():
    items = [SetItem(1, "S1", samples=2), SetItem(2, "S1", samples=2)]
    labels = [
        _label(1, "Overflow in fee calc", "tp", "S1"),
        _label(2, "Race in unlock", "tp", "S1"),
    ]
    hit1 = [_f("Overflow in fee calc")]
    hit2 = [_f("Race in unlock")]
    samples = [
        _sample("v11", 1, hit1, k=0, samples=2),
        _sample("v11", 1, [], k=1, samples=2),  # disagrees with itself
        _sample("v11", 2, hit2, k=0, samples=2),
        _sample("v11", 2, hit2, k=1, samples=2),
        _sample("v12", 1, hit1, k=0, samples=2),
        _sample("v12", 1, hit1, k=1, samples=2),
        _sample("v12", 2, [], k=0, samples=2),
        _sample("v12", 2, [], k=1, samples=2),
    ]
    result, _, _ = score(items, labels, samples, ["v11", "v12"])
    # v11 found 2 of 2 in sample 0 and 1 of 2 in sample 1; v12 1 of 2 in both
    assert result["noise"]["per_arm"]["v11"]["recall_delta_pp"] == 50.0
    assert result["noise"]["per_arm"]["v11"]["label_flips"] == 1
    assert result["noise"]["per_arm"]["v12"]["recall_delta_pp"] == 0.0
    assert result["noise"]["floor_pp"] == 50.0
    # recall averages the samples of a head: v11 found label 1 in half its samples
    assert result["sets"]["S1"]["v11"]["recall"] == 0.75


def test_per_head_means_over_samples_for_counts():
    """A known FP raised as a blocker in one of two samples counts 0.5, not 1 (or 2)."""
    items = [SetItem(3, "S4", samples=2)]
    labels = [_label(3, "FindDevNetGenesisBlock unused", "fp", "S4")]
    fp = [_f("FindDevNetGenesisBlock is unused")]
    samples = [
        _sample("v11", 3, fp, set_="S4", k=0, samples=2),
        _sample("v11", 3, [], set_="S4", k=1, samples=2),
        _sample("v12", 3, [_f("New thing", 90)], set_="S4", k=0, samples=2),
        _sample("v12", 3, [_f("New thing", 90)], set_="S4", k=1, samples=2),
    ]
    result, _, _ = score(items, labels, samples, ["v11", "v12"])
    s4 = result["sets"]["S4"]
    assert s4["v11"]["fp_blockers"] == 0.5 and s4["v11"]["heads_with_blocker_rate"] == 0.5
    assert s4["v12"]["new_blockers"] == 1.0
    assert s4["delta_vs_v11"]["v12"]["fp_blockers"] == -0.5


def test_subsets_pooled_recall_bootstrap_and_thread_outcomes():
    items = [SetItem(i, s) for i, s in ((1, "S1"), (2, "S2"), (3, "S5"), (4, "S5"), (5, "S8"))]
    labels = [
        _label(1, "Overflow in fee calc", "tp", "S1"),
        _label(2, "Race in wallet unlock", "tp", "S2"),
        _label(3, "Settled rename", "fp", "S5", round_kind="settling_round"),
        _label(4, "Settled migration", "fp", "S5", round_kind="post_settle"),
        _label(
            5, "HMAC erase", "deferred", "S8", hash="a77", thread_outcome="INTENTIONALLY_DEFERRED"
        ),
    ]
    # conversation.json as each arm writes it: v11 flat, v12 under "outcomes" (+ would_post)
    conv = {
        "work": {"a77": ["reply"]},
        "shards": [["a77"]],
        "outcomes": {"a77": {"status": "DEFERRAL_PENDING"}},
        "would_post": None,
    }
    samples = [
        _sample("v11", 1, []),
        _sample("v11", 2, [], set_="S2"),
        _sample("v11", 3, [_f("Settled rename", sev="suggestion")], set_="S5"),
        _sample("v11", 4, [], set_="S5"),
        _sample("v11", 5, [], set_="S8", conversation={"a77": {"status": "STILL_VALID"}}),
        _sample("v12", 1, [_f("Overflow in fee calc")]),
        _sample("v12", 2, [_f("Race in wallet unlock")], set_="S2"),
        _sample("v12", 3, [], set_="S5"),
        _sample("v12", 4, [_f("Settled migration")], set_="S5"),
        _sample("v12", 5, [], set_="S8", conversation=conv),
    ]
    result, _, _ = score(items, labels, samples, ["v11", "v12"])
    sets = result["sets"]
    assert sets["S1+S2"]["v11"]["recall"] == 0.0 and sets["S1+S2"]["v12"]["recall"] == 1.0
    assert sets["S1+S2"]["v12"]["tp_labels"] == 2
    ci = sets["S1+S2"]["delta_vs_v11"]["v12"]["recall_ci95_pp"]
    assert ci == [100.0, 100.0]
    assert sets["S5:settling_round"]["v11"]["reraised_unchanged"] == 1.0
    assert sets["S5:settling_round"]["v12"]["reraised_unchanged"] == 0.0
    assert sets["S5:post_settle"]["v12"]["reraised_unchanged"] == 1.0
    assert sets["S5"]["v12"]["fp_blockers"] == 1.0
    # a pending deferral reaches an expected deferral
    assert sets["S8"]["v11"]["thread_outcomes_reached"] == 0.0
    assert sets["S8"]["v12"]["thread_outcomes_reached"] == 1.0
    # the #38 follow-up decided on: v12 records it (none here), v11 never does
    assert sets["S8"]["v11"]["would_post"] == {"not_recorded": 1.0}
    assert sets["S8"]["v12"]["would_post"] == {"none": 1.0}
    assert "would_post" not in sets["S1"]["v12"]


def test_s8_reads_the_v12_follow_up_it_would_post():
    items = [SetItem(6, "S8", samples=2)]
    labels = [_label(6, "Fee drainage", "tp", "S8", hash="f33", thread_outcome="WITHDRAWN")]

    def conv(status: str, event: str | None) -> dict[str, Any]:
        wp = {"event": event, "note": "-", "lifted": ["f33"], "posts": True} if event else None
        return {"outcomes": {"f33": {"status": status}}, "would_post": wp}

    flat = {"f33": {"status": "WITHDRAWN"}}
    samples = [
        _sample("v11", 6, [], set_="S8", k=0, samples=2, conversation=flat),
        _sample("v11", 6, [], set_="S8", k=1, samples=2, conversation=flat),
        _sample("v12", 6, [], set_="S8", k=0, samples=2, conversation=conv("WITHDRAWN", "APPROVE")),
        _sample("v12", 6, [], set_="S8", k=1, samples=2, conversation=conv("STILL_VALID", None)),
    ]
    result, _, _ = score(items, labels, samples, ["v11", "v12"])
    s8 = result["sets"]["S8"]
    assert s8["v11"]["thread_outcomes_reached"] == 1.0
    assert s8["v12"]["thread_outcomes_reached"] == 0.5
    assert s8["v12"]["would_post"] == {"APPROVE": 0.5, "none": 0.5}
    assert s8["v11"]["would_post"] == {"not_recorded": 1.0}


EVAL = Path("/Users/pasta/workspace/reviewsys-v12-design/eval")


def test_the_eval_set_and_labels_format(tmp_path):
    """eval/set.json and eval/labels.jsonl as the eval-set builder writes them (a sample of
    their rows here; the real files are scored too when they are on this machine)."""
    setfile = tmp_path / "set.json"
    labels = tmp_path / "labels.jsonl"
    setfile.write_text(
        json.dumps(
            {
                "targets": {"S1": 1},
                "heads": [
                    {
                        "id": "yappr-484-87bdb83b",
                        "set": "S1",
                        "source_run_id": 775,
                        "repo": "o/r",
                        "number": 484,
                        "sha": "8" * 40,
                        "cutoff": "2026-09-16T09:23:12Z",
                    },
                    {
                        "id": "bls-132-7cded921",
                        "set": "S4",
                        "source_run_id": 2885,
                        "repo": "o/r",
                        "number": 132,
                        "sha": "7" * 40,
                    },
                    {
                        "id": "bls-132-7cded921-reply",
                        "set": "S8",
                        "source_run_id": 2926,
                        "repo": "o/r",
                        "number": 132,
                        "sha": "7" * 40,
                    },
                    {
                        "id": "dash-7107-5885d7ba",
                        "set": "S6",
                        "source_run_id": 3248,
                        "repo": "o/r",
                        "number": 7107,
                        "sha": "5" * 40,
                    },
                ],
                "repeats": ["yappr-484-87bdb83b"],
                "missing_locally": [],
            }
        )
    )
    rows = [
        {
            "repo": "o/r",
            "number": 484,
            "sha": "8" * 40,
            "file": "lib/x.ts",
            "line_start": 145,
            "line_end": 151,
            "title": "Prevent background contract revalidation from recursively triggering recovery",
            "severity": "blocking",
            "label": "tp",
            "set": "S1",
            "source": "late_unchanged_fixed",
            "head_id": "yappr-484-87bdb83b",
            "strength": "TP-strong",
            "finding_hash": "2feb0f040eab",
        },
        {
            "repo": "o/r",
            "number": 132,
            "sha": "7" * 40,
            "file": "src/hmac.c",
            "line_start": 77,
            "line_end": 77,
            "title": "BLST MSVC sources",
            "severity": "blocking",
            "label": "fp",
            "set": "S4",
            "source": "withdrawn",
            "head_id": "bls-132-7cded921",
            "human_reply": {"user": "knst", "body": "no"},
        },
        {
            "repo": "o/r",
            "number": 132,
            "sha": "7" * 40,
            "file": "include/util.hpp",
            "line_start": 77,
            "line_end": 77,
            "title": "Explicitly erase the HMAC allocation before releasing it",
            "severity": "blocking",
            "label": "deferred",
            "set": "S8",
            "source": "reply_outcome",
            "head_id": "bls-132-7cded921-reply",
            "finding_hash": "a772484d6d1a",
            "thread_outcome": "INTENTIONALLY_DEFERRED",
            "production_outcome": "STILL_VALID",
        },
        {
            "repo": "o/r",
            "number": 7107,
            "sha": "5" * 40,
            "file": None,
            "line_start": None,
            "line_end": None,
            "title": "long-tail head (checks only)",
            "severity": None,
            "label": "none",
            "set": "S6",
            "source": "long_tail",
        },
    ]
    labels.write_text("".join(json.dumps(r) + "\n" for r in rows))
    items = load_set(setfile)
    assert [(i.run_id, i.samples) for i in items] == [(775, 2), (2885, 1), (2926, 1), (3248, 1)]
    out = tmp_path / "out"
    run_set(
        items, _policies(tmp_path), out, concurrency=1, runner=lambda j, o: 0, echo=lambda s: None
    )
    by_run = {
        i.run_id: next(
            json.loads(json.dumps(h))
            for h in json.loads(setfile.read_text())["heads"]
            if h["source_run_id"] == i.run_id
        )
        for i in items
    }
    for item in items:
        for p in ("v11", "v12"):
            for k in range(item.samples):
                job = Job(item, Policy(p, Path()), k)
                h = by_run[item.run_id]
                found = (
                    [
                        _f(
                            "Prevent contract revalidation from triggering recovery",
                            146,
                            file="lib/x.ts",
                        )
                    ]
                    if item.run_id == 775 and p == "v12"
                    else []
                )
                _write(
                    job,
                    out,
                    _report(
                        job,
                        number=h["number"],
                        sha=h["sha"],
                        published=found,
                        conversation={"a772484d6d1a": {"status": "STILL_VALID"}}
                        if item.run_id == 2926
                        else None,
                    ),
                )
    from reviewsys.cli import main

    assert main(["replay-score", str(setfile), str(labels), "--out", str(out)]) == 0
    result = json.loads((out / "score.json").read_text())
    assert result["heads"]["paired"] == 4 and result["labels"]["on_paired_heads"] == 4
    assert (
        result["sets"]["S1"]["v12"]["recall"] == 1.0
        and result["sets"]["S1"]["v11"]["recall"] == 0.0
    )
    # the S4 and S8 heads share a commit: each label scores only on its own head
    assert (
        result["sets"]["S4"]["v11"]["heads"] == 1
        and result["sets"]["S8"]["v11"]["thread_outcome_labels"] == 1
    )
    assert result["sets"]["S6"]["v11"]["tp_labels"] == 0
    real_set, real_labels = EVAL / "set.json", EVAL / "labels.jsonl"
    if real_set.exists() and real_labels.exists():
        assert len(load_set(real_set)) == len(json.loads(real_set.read_text())["heads"])
        assert len(replayscore.load_labels(real_labels)) == len(
            real_labels.read_text().splitlines()
        )


def test_judge_list_clusters_one_issue_raised_by_both_arms():
    items = [SetItem(1, "S7")]
    samples = [
        _sample("v11", 1, [_f("Unchecked index into peers vector", 40)], set_="S7"),
        _sample("v12", 1, [_f("Unchecked peers vector index", 42, hash="other")], set_="S7"),
    ]
    _, to_judge, key = score(items, [], samples, ["v11", "v12"])
    assert len(to_judge) == 1
    assert key[to_judge[0]["id"]]["raised_by"] == [("v11", 0), ("v12", 0)]


def test_replay_score_cli_writes_outputs(tmp_path, capsys):
    out, pols = tmp_path / "out", _policies(tmp_path)
    items = [SetItem(1, "S1")]
    setfile = tmp_path / "set.json"
    setfile.write_text(json.dumps([{"run_id": 1, "set": "S1"}]))
    run_set(items, pols, out, concurrency=1, runner=lambda j, o: 0, echo=lambda s: None)
    for p, found in (("v11", []), ("v12", [_f("Overflow in fee calc")])):
        job = Job(items[0], Policy(p, Path()), 0)
        _write(job, out, _report(job, published=found, number=1, sha="1" * 40))
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "repo": "o/r",
                "number": 1,
                "sha": "1" * 40,
                "file": "a.rs",
                "line_start": 10,
                "line_end": 10,
                "title": "Overflow in fee calc",
                "label": "tp",
                "set": "S1",
                "source": "fixed later",
            }
        )
        + "\n"
    )
    from reviewsys.cli import main

    assert main(["replay-score", str(setfile), str(labels), "--out", str(out)]) == 0
    result = json.loads((out / "score.json").read_text())
    assert result["sets"]["S1"]["v12"]["recall"] == 1.0
    assert (out / "to_judge.jsonl").read_text() == ""
    assert "S1" in capsys.readouterr().out


def test_malformed_labels_are_rejected(tmp_path):
    p = tmp_path / "l.json"
    p.write_text(json.dumps([{"repo": "o/r", "number": 1, "sha": "x", "label": "maybe"}]))
    with pytest.raises(SystemExit, match="must be one of"):
        replayscore.load_labels(p)
    p.write_text(json.dumps([{"repo": "o/r", "label": "tp"}]))
    with pytest.raises(SystemExit, match="malformed"):
        replayscore.load_labels(p)


def test_s5_is_scored_with_and_without_heads_whose_thread_state_is_as_of_now():
    """L4: on head 2 the view could not rewind a thread's resolution; S5 counts both heads,
    S5:exact_threads only head 1."""
    items = [SetItem(1, "S5"), SetItem(2, "S5")]
    labels = [_label(1, "Settled thing", "fp", "S5"), _label(2, "Settled thing", "fp", "S5")]
    raised = [_f("Settled thing")]
    samples = [
        _sample(p, n, raised, set_="S5", thread_resolution_as_of_now=int(n == 2))
        for p in ("v11", "v12")
        for n in (1, 2)
    ]
    result, _, _ = score(items, labels, samples, ["v11", "v12"])
    assert result["heads"]["thread_resolution_as_of_now"] == [2]
    assert result["sets"]["S5"]["v12"]["heads"] == 2
    assert result["sets"]["S5"]["v12"]["reraised_unchanged"] == 2.0
    exact = result["sets"]["S5:exact_threads"]["v12"]
    assert exact["heads"] == 1 and exact["reraised_unchanged"] == 1.0


def test_wall_clock_is_reported_with_and_without_slot_waits():
    """L8: replays wait behind production for lane slots; the active time is what is left."""
    items = [SetItem(1, "S1"), SetItem(2, "S1")]
    samples = [
        _sample("v11", 1, [], wall_clock_s=100.0, slot_wait_s=0.0, active_s=100.0),
        _sample("v11", 2, [], wall_clock_s=300.0, slot_wait_s=200.0, active_s=100.0),
        _sample("v12", 1, [], wall_clock_s=200.0, slot_wait_s=80.0, active_s=120.0),
        _sample("v12", 2, [], wall_clock_s=120.0, slot_wait_s=0.0, active_s=120.0),
    ]
    result, _, _ = score(items, [], samples, ["v11", "v12"])
    assert result["arms"]["v11"]["wall_clock_p50_s"] == 200.0
    assert result["arms"]["v11"]["active_p50_s"] == 100.0
    assert result["arms"]["v12"]["slot_wait_p50_s"] == 40.0
    assert result["ratios_vs_baseline"]["v12"]["active_p50_s"] == 1.2
