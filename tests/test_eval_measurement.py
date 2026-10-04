"""What the v12 eval measures, measured right: the one-shot reconcile production already ran,
the reconcile-only thread work a replay adds, blinded judge items, the v12 signals, delivered
thread answers, and the small worker and config guards the final review asked for."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from reviewsys import config as cfg_mod
from reviewsys import ledger, publish, replay, replayscore, rules, worker
from reviewsys.db import connect, event, kv_set, tx
from reviewsys.replayscore import ARM_PHRASES, LABEL_MATCH, NEW_BLOCKER, Sample, score
from reviewsys.replayset import SetItem
from reviewsys.steps import worktree as wt

REPO, NUMBER = "o/r", 7
CUTOFF = "2026-09-20T12:00:00Z"


def _db(tmp_path: Path, name: str = "src.db") -> sqlite3.Connection:
    return connect(tmp_path / name)


def _record(conn, rec: dict[str, Any]) -> None:
    with tx(conn):
        kv_set(conn, f"ledger.reconciled:{REPO}#{NUMBER}", json.dumps(rec))


def _answered(conn, ts: str, h: str) -> None:
    with tx(conn):
        conn.execute(
            "INSERT INTO events (ts, kind, repo, number, detail) VALUES (?,?,?,?,?)",
            (ts, "thread.answered", REPO, NUMBER, json.dumps({"finding_hash": h})),
        )


# ---- 1: the reconcile production recorded ----


def test_a_reconcile_recorded_before_the_cutoff_is_copied_as_of_it(tmp_path):
    src = _db(tmp_path)
    _record(
        src,
        {"run": 3, "sha": "s", "pending": ["c"], "at": "2026-09-20T10:00:00Z",
         "pending_at_reconcile": ["a", "b", "c"]},
    )  # fmt: skip
    _answered(src, "2026-09-20T11:00:00Z", "a")  # judged before the cutoff: gone by then
    _answered(src, "2026-09-20T13:00:00Z", "b")  # judged after it: still pending at it
    rec = replay.recorded_reconcile(src, REPO, NUMBER, CUTOFF)
    assert rec == {"run": 3, "sha": "s", "at": "2026-09-20T10:00:00Z", "pending": ["b", "c"],
                   "exact": True}  # fmt: skip
    scratch = _db(tmp_path, "scratch.db")
    pin = SimpleNamespace(repo=REPO, number=NUMBER, sha="h", cutoff=CUTOFF)
    got = replay.reconcile_as_of(
        scratch,
        src,
        pin=pin,
        gh=None,
        bot_login="b",
        git_dir=tmp_path,  # type: ignore[arg-type]
    )
    assert got == {"source": "recorded", "at": "2026-09-20T10:00:00Z", "pending": 2,
                   "pending_exact": True}  # fmt: skip
    assert ledger.reconciled(scratch, REPO, NUMBER)["pending"] == ["b", "c"]  # type: ignore[index]


def test_a_reconcile_after_the_cutoff_or_none_means_the_replay_runs_it(tmp_path):
    src = _db(tmp_path)
    assert replay.recorded_reconcile(src, REPO, NUMBER, CUTOFF) is None
    _record(src, {"run": None, "sha": "s", "pending": [], "at": "2026-09-21T00:00:00Z"})
    assert replay.recorded_reconcile(src, REPO, NUMBER, CUTOFF) is None


def test_an_older_record_is_dated_by_its_run(tmp_path):
    src = _db(tmp_path)
    with tx(src):
        hid = src.execute(
            "INSERT INTO heads (repo, number, sha, trigger, status, queued_at, eligible_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (REPO, NUMBER, "s", "manual", "done", CUTOFF, CUTOFF),
        ).lastrowid
        rid = src.execute(
            "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at) "
            "VALUES (?, 1, 'done', 't', '2026-09-19T00:00:00Z', '2026-09-19T00:00:00Z')",
            (hid,),
        ).lastrowid
    _record(src, {"run": rid, "sha": "s", "pending": ["x"]})
    rec = replay.recorded_reconcile(src, REPO, NUMBER, CUTOFF)
    assert rec is not None and rec["at"] == "2026-09-19T00:00:00Z" and rec["exact"] is False


def _lane(role: str, model: str, tin: int, tout: int) -> dict[str, Any]:
    return {"role": role, "model": model, "tokens_in": tin, "tokens_out": tout}


def test_reconcile_only_thread_work_is_counted_apart():
    threads = {
        "work": {
            "a": ["reconcile"],
            "b": ["reconcile"],
            "c": ["reply"],
            "d": ["reconcile", "reply"],
        },
        "shards": [["a", "b"], ["c", "d"]],
    }
    lanes = [
        _lane("thread-1of2", "gpt-6.1-sol", 1000, 100),
        _lane("thread-2of2", "gpt-6.1-sol", 500, 50),
        _lane("general", "gpt-6.1-sol", 9000, 900),
    ]
    rw = replay.reconcile_work(threads, lanes, counted=True)
    assert rw["issues"] == 2 and rw["shards"] == 1
    assert rw["gpt_lanes"] == 1.0 and rw["gpt_tokens_in"] == 1000 and rw["gpt_tokens_out"] == 100
    assert replay.reconcile_work(threads, lanes, counted=False)["gpt_lanes"] == 0.0


def _sample(policy: str, n: int, **rep: Any) -> Sample:
    report = {
        "repo": "o/r", "number": n, "sha": f"{n:040d}", "status": "done", "invalid": None,
        "published": [], "wall_clock_s": 1.0, "gpt": {"lanes": 10, "tokens_in": 1000,
        "tokens_out": 0}, **rep,
    }  # fmt: skip
    return Sample(policy, SetItem(n, rep.pop("set_", "S1")), 0, report)


def test_the_cost_ratio_has_a_figure_without_the_reconcile_work():
    items = [SetItem(1, "S1")]
    v11 = _sample("v11", 1)
    v12 = _sample(
        "v12", 1, gpt={"lanes": 12, "tokens_in": 1300, "tokens_out": 0},
        gpt_excl_reconcile={"lanes": 10, "tokens_in": 1000, "tokens_out": 0},
    )  # fmt: skip
    result, _, _ = score(items, [], [v11, v12], ["v11", "v12"])
    ratios = result["ratios_vs_baseline"]["v12"]
    assert ratios["gpt_tokens_per_run"] == 1.3
    assert ratios["gpt_tokens_per_run_excl_reconcile"] == 1.0  # what the 1.15x gate reads
    assert ratios["gpt_lanes_per_run_excl_reconcile"] == 1.0


# ---- 3: blinding ----

V12_BODY = (
    "The fee estimate omits the drainage term.\n\n"
    "_This was raised before and withdrawn at `5a06c55a` "
    "(https://github.com/o/r/pull/7#discussion_r123); the code at that location has changed "
    "since._\n\n_This looks like a regression of an earlier finding that was fixed: "
    "https://github.com/o/r/pull/7#discussion_r456._\n\n"
    "**Not confirmed as blocking.** What would confirm it: a failing test\n\n"
    "**Suggested change** (the reports point at different lines, so it is not offered as a "
    "one-click suggestion):\n```\nfee += drain\n```"
)


def test_the_arm_phrases_are_the_ones_rules_writes():
    import inspect

    src = inspect.getsource(rules) + inspect.getsource(publish)
    for phrase in ARM_PHRASES:
        assert phrase in src or phrase == "#discussion_r", phrase


def test_judge_items_from_either_arm_carry_no_arm_framing():
    from reviewsys import judge

    items = [SetItem(1, "S4")]
    v12 = {
        "file": "f.rs",
        "line_start": 3,
        "line_end": 4,
        "severity": "blocking",
        "title": "Fee estimate omits drainage (regression)",
        "body": V12_BODY,
        "hash": "x",
    }
    v11 = {"file": "f.rs", "line_start": 40, "line_end": 40, "severity": "blocking",
           "title": "Unchecked index into peers", "body": "peers[i] is read unchecked.",
           "hash": "y"}  # fmt: skip
    label = replayscore.Label("o/r", 1, f"{1:040d}", "f.rs", 5, 5, "Wraps around", "fp", "S4")
    samples = [_sample("v12", 1, published=[v12], set_="S4"),
               _sample("v11", 1, published=[v11], set_="S4")]  # fmt: skip
    _, todo, _ = score(items, [label], samples, ["v11", "v12"])
    assert {t["kind"] for t in todo} == {NEW_BLOCKER, LABEL_MATCH}
    for t in todo:
        text = json.dumps(t) + judge.prompt(t)
        for phrase in ARM_PHRASES:
            assert phrase not in text, (phrase, t["title"])
    fee = next(t for t in todo if t["title"].startswith("Fee"))
    assert fee["title"] == "Fee estimate omits drainage"
    assert "omits the drainage term" in fee["body"] and "fee += drain" in fee["body"]


def test_a_verdict_on_a_blinded_item_still_applies_to_the_raw_finding():
    items = [SetItem(1, "S7")]
    raw = {"file": "f.rs", "line_start": 3, "line_end": 3, "severity": "blocking",
           "title": "Fee estimate omits drainage (regression)", "body": V12_BODY, "hash": "x"}  # fmt: skip
    samples = [_sample("v11", 1, set_="S7"), _sample("v12", 1, published=[raw], set_="S7")]
    _, todo, _ = score(items, [], samples, ["v11", "v12"])
    judged = [{**todo[0], "verdict": {"real": "yes"}}]
    result, again, _ = score(items, [], samples, ["v11", "v12"], judged=judged)
    assert again == [] and result["arms"]["v12"]["precision"]["new_blocker_precision"] == 1.0


# ---- 4: v12 signals ----


def test_v12_signals_from_the_run_events_and_verify_artifacts(tmp_path):
    scratch = _db(tmp_path)
    with tx(scratch):
        for kind, detail in (
            ("threads.conflict", "{}"),
            ("threads.failed", "thread-1of2 (2 issue(s) left as they were): boom"),
            ("v12.late_nonblocking", "3 late non-blocking finding(s) since abcdef12 (post)"),
            ("v12.settled_refold", "x -> y"),
            ("threads.conflict", "{}"),
        ):
            event(scratch, kind, repo=REPO, number=NUMBER, run_id=1, detail=detail)
        event(scratch, "threads.conflict", repo=REPO, number=NUMBER, run_id=2, detail="{}")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "phase2.json").write_text(
        json.dumps({"candidates": [{}] * 8, "corroborated": {"h1": ["c1", "c2"], "h2": ["c3"]}})
    )
    (run_dir / "evidence.json").write_text("{}")
    sig = replay.v12_signals(scratch, 1, run_dir)
    assert sig == {"threads_conflict": 2, "threads_failed": 1, "late_nonblocking": 3,
                   "settled_refold": 1, "candidates": 8, "candidates_matched_open": 3,
                   "match_share": 0.375}  # fmt: skip


def test_scoring_reads_the_signals_duplicate_threads_and_degraded_runs():
    items = [SetItem(1, "S5"), SetItem(2, "S5")]
    prior = [{"hash": "p", "file": "f.rs", "line_start": 10, "line_end": 10,
              "title": "Fee estimate omits drainage"}]  # fmt: skip
    dup = {"hash": "q", "file": "f.rs", "line_start": 11, "line_end": 11,
           "title": "Fee estimate omits the drainage"}  # fmt: skip
    v12 = {"threads_conflict": 1, "late_nonblocking": 2, "settled_refold": 0,
           "candidates": 4, "candidates_matched_open": 1}  # fmt: skip
    samples = [
        _sample("v11", 1, set_="S5", prior_open_threads=prior, inline=[]),
        _sample("v12", 1, set_="S5", prior_open_threads=prior, inline=[dup], v12=v12),
        _sample("v11", 2, set_="S5"),
        _sample("v12", 2, set_="S5", degraded_for_scoring="a thread-lane shard failed"),
    ]
    result, _, _ = score(items, [], samples, ["v11", "v12"])
    assert result["heads"]["paired"] == 1  # head 2's v12 sample does not count
    a = result["arms"]["v12"]
    assert a["degraded_for_scoring"] == 1 and a["conflicts_per_run"] == 1.0
    assert a["late_nonblocking_per_run"] == 2.0 and a["match_share"] == 0.25
    assert a["duplicate_threads_per_run"] == 1.0
    assert result["sets"]["S5"]["v12"]["duplicate_threads"] == 1.0  # the S5 gate wants 0
    assert result["sets"]["S5"]["v11"]["duplicate_threads"] == 0.0


# ---- 5: what reached a thread ----


def test_only_delivered_answers_lift(tmp_path):
    src = _db(tmp_path)
    for i, (h, action) in enumerate(
        (("a", "replied"), ("b", "suppressed_refound"), ("c", "reply_failed"),
         ("d", "already_answered"), ("e", None))
    ):  # fmt: skip
        detail = {
            "finding_hash": h,
            "status": "WITHDRAWN",
            **({"action": action} if action else {}),
        }
        with tx(src):
            src.execute(
                "INSERT INTO events (ts, kind, repo, number, detail) VALUES (?,?,?,?,?)",
                (f"2026-09-20T0{i}:00:00Z", "thread.answered", REPO, NUMBER, json.dumps(detail)),
            )
    assert replay.lifted_before(src, REPO, NUMBER, CUTOFF) == {"a": "WITHDRAWN", "d": "WITHDRAWN"}


# ---- 6: no gate-would-publish under v12 ----


def test_v12_records_no_gate_publication_and_says_so(tmp_path):
    conn = _db(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ctx = SimpleNamespace(
        verify1=SimpleNamespace(blocker_count=2),
        cfg=SimpleNamespace(policy=SimpleNamespace(pipeline=object())),
        conn=conn, repo=REPO, number=NUMBER, run_id=5, run_dir=run_dir,
    )  # fmt: skip
    worker._record_gate_would_publish(ctx, gate_deferred=True)  # type: ignore[arg-type]
    assert not (run_dir / "gate-would-publish.json").exists()
    detail = conn.execute("SELECT detail FROM events WHERE kind='replay.phase2_forced'").fetchone()
    assert "policy v12: no gate-would-publish record" in detail[0]
    assert "cannot be rerun" in replay.V12_GATE_NOTE


# ---- 8: eval-only settings never stop the daemon ----


@pytest.mark.parametrize(
    ("share", "slots", "want_share", "want_slots"),
    [
        ('"lots"', '{ gpt = "two", glm = 2 }', 0.5, {"glm": 2}),
        ("0.25", '"gpt=2"', 0.25, {}),
        ("2", "{ gpt = -3, muse = true }", 1.0, {"gpt": 0}),
    ],
)
def test_bad_replay_pool_settings_fall_back_with_a_warning(
    tmp_path, skills_dir, caplog, share, slots, want_share, want_slots
):
    toml = cfg_mod.DEFAULT_TOML.replace(
        'skills = "~/Projects/skills"', f'skills = "{skills_dir}"'
    ).replace(
        "replay_pool_share = 0.5\nreplay_pool_slots = {}",
        f"replay_pool_share = {share}\nreplay_pool_slots = {slots}",
    )
    p = tmp_path / "config.toml"
    p.write_text(toml)
    cfg = cfg_mod.load(p)
    assert cfg.replay_pool_share == want_share and cfg.replay_pool_slots == want_slots
    if want_slots != {"gpt": 0}:
        assert "replay_pool" in caplog.text


# ---- 9, 10, 11: worker ----


def test_events_have_a_run_index(tmp_path):
    conn = _db(tmp_path)
    names = {r[1] for r in conn.execute("PRAGMA index_list(events)")}
    assert "events_run" in names


def test_the_sha_fallback_fetch_is_bounded_tighter(monkeypatch, tmp_path):
    calls: list[tuple[tuple[str, ...], int]] = []

    def fake_git(*args: str, cwd=None, timeout: int = 600) -> str:
        calls.append((args, timeout))
        if args[0] == "cat-file" and len([c for c in calls if c[0][0] == "cat-file"]) == 1:
            raise wt.ReviewError(wt.FailKind.CONTRACT, "missing")
        return ""

    monkeypatch.setattr(wt, "_git", fake_git)
    wt.fetch_head(tmp_path, 1, "d" * 40)
    fetches = [(a, t) for a, t in calls if a[0] == "fetch"]
    assert fetches[0][1] == 1800 and fetches[1] == (("fetch", "--no-tags", "origin", "d" * 40), 600)
    assert wt.SHA_FETCH_TIMEOUT == 600


# ---- 7: the eval's own cap on a pool ----


def test_replay_pool_slots_on_the_command_line(cfg, monkeypatch):
    import argparse

    from reviewsys import cli

    monkeypatch.setattr(cli, "_cfg", lambda args: cfg)
    args = argparse.Namespace(replay_pool_slots=["gpt=2", "glm=1"])
    assert cli._replay_cfg(args).replay_pool_slots == {"gpt": 2, "glm": 1}
    with pytest.raises(SystemExit, match="POOL=N"):
        cli._replay_cfg(argparse.Namespace(replay_pool_slots=["gpt"]))
    assert cli._replay_cfg(argparse.Namespace(replay_pool_slots=None)) is cfg


def test_replay_set_passes_the_pool_cap_to_every_replay(tmp_path, monkeypatch):
    from reviewsys import replayset
    from reviewsys.cli import main

    calls: list[list[str]] = []
    monkeypatch.setattr(replayset.subprocess, "run", lambda argv, **kw: calls.append(argv))
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "config.json").write_text("{}")
    setfile = tmp_path / "set.json"
    setfile.write_text(json.dumps([{"run_id": 1}]))
    main(
        ["replay-set", str(setfile), "--policy", f"v11={tmp_path / 's'}", "--out",
         str(tmp_path / "out"), "--no-production-guard", "--min-free-gb", "0",
         "--replay-pool-slots", "gpt=2"]
    )  # fmt: skip
    assert calls and calls[0][calls[0].index("--replay-pool-slots") + 1] == "gpt=2"


def test_eval_run_defaults_the_gpt_cap_to_two():
    text = (Path(__file__).resolve().parent.parent / "deploy" / "eval-run.sh").read_text()
    assert "POOL_SLOTS=(gpt=2)" in text and "--replay-pool-slots" in text
