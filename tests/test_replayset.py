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
from reviewsys.replayset import Job, Policy, SetItem, load_set, plan, run_set


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
    res = run_set(
        items, pols, out, concurrency=2, runner=runner, retry_failed=True, echo=lambda s: None
    )
    assert ran == ["v12/run-2"]  # the failed one, on request
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
    assert matches(_f("Reworded entirely", 40, hash="h2", category="logic"), label)
    assert not matches(_f("Reworded entirely", 40, hash="h2", category="style"), label)
    nocat = {**label, "hash": "", "category": ""}
    assert not matches(_f("Reworded entirely", 40, hash=""), nocat)  # no hash/category luck


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
    assert result["heads"] == {"in_set": 5, "paired": 4, "unpaired": [5]}
    s1 = result["sets"]["S1"]
    assert s1["v11"]["recall"] == 0.5 and s1["v12"]["recall"] == 1.0
    assert s1["v11"]["blocker_recall"] == 0.0 and s1["v12"]["blocker_recall"] == 1.0
    assert s1["delta_vs_v11"]["v12"]["recall"] == 50.0
    assert s1["v12"]["new_blockers"] == 1 and s1["v11"]["new_blockers"] == 0
    s4 = result["sets"]["S4"]
    assert (s4["v11"]["fp_blockers"], s4["v12"]["fp_blockers"]) == (1, 0)
    assert s4["v11"]["runs_with_blocker_rate"] == 1.0
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
    assert result["noise"]["per_arm"]["v11"]["flip_rate"] == 0.5
    assert result["noise"]["per_arm"]["v12"]["flip_rate"] == 0.0
    assert result["noise"]["floor_pp"] == 25.0
    # recall averages the samples of a head: v11 found label 1 in half its samples
    assert result["sets"]["S1"]["v11"]["recall"] == 0.75


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
