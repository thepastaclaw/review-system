"""`reviewsys replay`: a past run's exact head, its history as of the run's start, a pinned
read-only GitHub, lanes that see the same; degraded mode refuses to start."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeGh, FakeLanes

from reviewsys import degraded, replay
from reviewsys import worker as worker_mod
from reviewsys.config import DegradedPolicy, Substitute
from reviewsys.contract import finding_hash
from reviewsys.db import kv_set, tx
from reviewsys.lane import LaneResult, LaneSpec
from reviewsys.models import ReviewError, RunStatus
from reviewsys.steps import worktree as wt

REPO, NUMBER = "dashpay/platform", 1
CUTOFF = "2026-09-20T12:00:00Z"
BEFORE, AFTER = "2026-09-20T10:00:00Z", "2026-09-20T13:00:00Z"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def mirror(cfg, tmp_path) -> dict[str, str]:
    """The repo's bare mirror where the replay looks for it: base on main, the PR's head."""
    src = tmp_path / "src"
    src.mkdir()
    _git(src, "init", "-q", "-b", "main")
    _git(src, "config", "user.email", "t@example.com")
    _git(src, "config", "user.name", "T")
    (src / "f.rs").write_text("".join(f"fn f{i}() {{}}\n" for i in range(20)))
    _git(src, "add", ".")
    _git(src, "commit", "-qm", "base")
    base = _git(src, "rev-parse", "HEAD")
    _git(src, "checkout", "-qb", "pr")
    (src / "f.rs").write_text("".join(f"fn f{i}() {{ {i} }}\n" for i in range(20)))
    _git(src, "commit", "-qam", "change")
    head = _git(src, "rev-parse", "HEAD")
    cfg.mirrors_dir.mkdir(parents=True)
    subprocess.run(
        ["git", "clone", "-q", "--bare", str(src), str(cfg.mirrors_dir / "dashpay__platform.git")],
        check=True,
    )
    return {"base": base, "head": head}


def _source_run(conn, sha: str, *, started: str, finished: str | None, status: str = "done") -> int:
    with tx(conn):
        row = conn.execute(
            "SELECT id FROM heads WHERE repo=? AND number=? AND sha=?", (REPO, NUMBER, sha)
        ).fetchone()
        hid = (
            row["id"]
            if row
            else conn.execute(
                "INSERT INTO heads (repo, number, sha, trigger, status, queued_at, eligible_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (REPO, NUMBER, sha, "new_push", "done", started, started),
            ).lastrowid
        )
        return int(
            conn.execute(
                "INSERT INTO runs (head_id, attempt, status, token, started_at, finished_at, "
                "deadline_at, tier) VALUES (?,?,?,?,?,?,?,?)",
                (hid, 1, status, "t", started, finished, started, "normal"),
            ).lastrowid
            or 0
        )


def _finding(conn, run_id: int, h: str, stage: str, title: str, phase: str = "final") -> None:
    with tx(conn):
        conn.execute(
            "INSERT INTO findings (run_id, phase, stage, hash, file, line_start, line_end, "
            "severity, category, title, body) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, phase, stage, h, "f.rs", 3, 4, "blocking", "logic", title, "b"),
        )


@pytest.fixture
def history(conn, mirror) -> dict[str, Any]:
    """One earlier round (finished before the replayed run), the replayed run, and a later
    round whose findings and events must not leak into the replay."""
    old_sha = "1" * 40
    early = _source_run(conn, old_sha, started="2026-09-19T10:00:00Z", finished=BEFORE)
    _finding(conn, early, "h-early", "posted", "Early blocker")
    _finding(conn, early, "h-early", "verified", "Early blocker", phase="verify2")
    src = _source_run(conn, mirror["head"], started=CUTOFF, finished=AFTER)
    _finding(conn, src, "h-src", "posted", "What the replayed run posted")
    straddle = _source_run(conn, old_sha, started=BEFORE, finished=AFTER)  # still running
    _finding(conn, straddle, "h-straddle", "posted", "Posted after the cutoff")
    later = _source_run(conn, "2" * 40, started=AFTER, finished="2026-09-20T15:00:00Z")
    _finding(conn, later, "h-later", "posted", "Later blocker")
    _finding(conn, later, "h-early", "conceded", "Early blocker")
    with tx(conn):
        conn.executemany(
            "INSERT INTO posted_findings (repo, number, hash, sha, review_id, posted_at) "
            "VALUES (?,?,?,?,?,?)",
            [
                (REPO, NUMBER, "h-early", old_sha, 11, BEFORE),
                (REPO, NUMBER, "h-later", "2" * 40, 13, AFTER),
            ],
        )
        conn.executemany(
            "INSERT INTO reviews (run_id, repo, number, sha, phase, github_review_id, event, "
            "posted_at) VALUES (?,?,?,?,?,?,?,?)",
            [
                (early, REPO, NUMBER, old_sha, "final", 11, "REQUEST_CHANGES", BEFORE),
                (later, REPO, NUMBER, "2" * 40, "final", 13, "APPROVE", AFTER),
            ],
        )
        conn.executemany(
            "INSERT INTO events (ts, kind, repo, number, run_id, detail) VALUES (?,?,?,?,?,?)",
            [
                (BEFORE, "thread.answered", REPO, NUMBER, early, '{"finding_hash": "h-early"}'),
                (AFTER, "thread.answered", REPO, NUMBER, later, '{"finding_hash": "h-later"}'),
                (BEFORE, "run.done", REPO, NUMBER, early, ""),
            ],
        )
    return {"early": early, "src": src, "later": later, "old_sha": old_sha, **mirror}


def _evidence(cfg, run_id: int, base: str, *, ci: dict[str, Any] | None = None) -> None:
    d = cfg.runs_dir / f"run-{run_id}"
    d.mkdir(parents=True)
    ev = {
        "pr": {
            "title": "Title then",
            "body": "Body then",
            "baseRefName": "main",
            "baseRefOid": base,
        },
        "ci": ci
        or {
            "fetched_at": CUTOFF,
            "checks": [
                {"name": "lint", "status": "completed", "conclusion": "success", "url": "u"}
            ],
        },
    }
    (d / "evidence.json").write_text(json.dumps({"evidence": ev}))


# ---- seeding ----


def test_seeding_copies_history_from_before_the_cutoff_only(conn, history, tmp_path):
    from reviewsys.db import connect

    scratch = connect(tmp_path / "scratch.db")
    counts = replay.seed_history(
        scratch,
        conn,
        repo=REPO,
        number=NUMBER,
        sha=history["head"],
        cutoff=CUTOFF,
        exclude_run=history["src"],
    )
    hashes = {r["hash"]: r["stage"] for r in scratch.execute("SELECT hash, stage FROM findings")}
    assert set(hashes) == {"h-early"}  # not the replayed run's, the straddler's or later ones
    assert {r["stage"] for r in scratch.execute("SELECT stage FROM findings")} == {
        "posted",
        "verified",
    }
    assert [
        (r["hash"], r["sha"]) for r in scratch.execute("SELECT hash, sha FROM posted_findings")
    ] == [("h-early", history["old_sha"])]
    assert [r["event"] for r in scratch.execute("SELECT event FROM reviews")] == ["REQUEST_CHANGES"]
    answered = scratch.execute("SELECT detail FROM events WHERE kind='thread.answered'").fetchall()
    assert [json.loads(r["detail"])["finding_hash"] for r in answered] == ["h-early"]
    assert not scratch.execute("SELECT 1 FROM events WHERE kind='run.done'").fetchone()
    assert counts["runs"] == 1 and counts["thread_answered"] == 1
    # the seeded run sits on its own commit, so per-commit lookups (standing verdict) work
    (sha,) = scratch.execute("SELECT h.sha FROM runs r JOIN heads h ON h.id=r.head_id").fetchone()
    assert sha == history["old_sha"]


def test_a_finding_reposted_after_the_cutoff_keeps_its_earlier_posting(conn, history, tmp_path):
    from reviewsys.db import connect

    with tx(conn):  # the later round re-posted h-early: the live row says so
        conn.execute(
            "UPDATE posted_findings SET sha=?, posted_at=? WHERE hash='h-early'", ("2" * 40, AFTER)
        )
    scratch = connect(tmp_path / "scratch.db")
    replay.seed_history(
        scratch, conn, repo=REPO, number=NUMBER, sha=history["head"], cutoff=CUTOFF, exclude_run=0
    )
    row = scratch.execute("SELECT sha, posted_at FROM posted_findings").fetchone()
    assert (row["sha"], row["posted_at"]) == (history["old_sha"], BEFORE)


def test_ledger_seeding_hook_is_called_with_the_cutoff(cfg, conn, history, tmp_path, monkeypatch):
    calls: list[tuple[str, int, str]] = []
    monkeypatch.setattr(
        replay,
        "seed_ledger",
        lambda s, p, repo, number, cutoff: calls.append((repo, number, cutoff)),
    )
    _evidence(cfg, history["src"], history["base"])
    _replay(cfg, history, tmp_path, FakeLanes(history["head"]))
    assert calls == [(REPO, NUMBER, CUTOFF)]


# ---- the replay ----


def _lanes_ok(lanes: FakeLanes, findings: list[dict[str, Any]]) -> None:
    """Lanes that report `findings` and reconcile the one seeded prior finding as FIXED."""
    prior = [
        {
            "finding_hash": finding_hash("f.rs", "logic", "Early blocker"),
            "status": "FIXED",
            "reason": "rewritten",
        }
    ]
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": findings,
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": prior,
    }
    verifier = {
        "prior_finding_reconciliation": prior,
        "summary": "s",
        "review_action": "COMMENT",
        "findings": findings,
        "dropped_findings": [],
        "out_of_scope_findings": [],
        "coderabbit_reactions": [],
        "prerequisite_adjudications": [],
        "adjudication_complete": True,
    }
    lanes.verifier["default"] = verifier


def _blocker(title: str = "Overflow in f3") -> dict[str, Any]:
    return {
        "file": "f.rs",
        "line_start": 3,
        "line_end": 4,
        "severity": "blocking",
        "confidence": 0.9,
        "category": "logic",
        "title": title,
        "body": "x",
    }


def _replay(cfg, history, tmp_path, lanes, **kw) -> dict[str, Any]:
    return replay.replay(
        cfg,
        run_id=history["src"],
        out=tmp_path / "out",
        gh_runner=kw.pop("gh_runner", FakeGh().runner),
        lane_runner=lanes,
        **kw,
    )


class _FakeBroker:
    """The broker binds a Unix socket, which a sandboxed test host may refuse; test_ghshim
    runs the real one. Lanes here are fakes and never call the shim."""

    def __init__(self, pin, real) -> None:
        self.path = "/nonexistent/socket"

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


@pytest.fixture(autouse=True)
def _no_broker(monkeypatch):
    monkeypatch.setattr(replay, "Broker", _FakeBroker)
    monkeypatch.setattr(replay, "check_shim", lambda shim, profile: "")


def test_replay_reviews_the_source_head_with_pinned_lanes_and_posts_nothing(
    cfg, conn, history, tmp_path
):
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [_blocker()])
    gh = FakeGh()
    gh.pr = {**gh.pr, "state": "closed", "merged": True, "title": "Retitled since"}
    report = _replay(cfg, history, tmp_path, lanes, gh_runner=gh.runner, force_phase2=True)
    assert report["status"] == "done", report["reason"]
    assert not gh.posted_reviews and not gh.gate_bodies and not gh.replies
    assert all(c[:2] != ["pr", "diff"] for c in gh.calls)  # the diff came from git
    work = tmp_path / "out" / f"run-{history['src']}"
    pin = json.loads((work / "pin.json").read_text())
    assert (pin["sha"], pin["merge_base"], pin["cutoff"]) == (
        history["head"],
        history["base"],
        CUTOFF,
    )
    # every lane runs with the shim first on PATH
    bin_dir = str(work / "bin")
    assert lanes.calls and all(dict(s.env)["PATH"].startswith(bin_dir) for s in lanes.calls)
    assert all(dict(s.env)["CLAUDE_ENV_FILE"].startswith(bin_dir) for s in lanes.calls)
    # no web tools, and no GitHub token of the developer's own (none was given)
    assert all(set(s.disallowed_tools) == {"WebFetch", "WebSearch"} for s in lanes.calls)
    assert all(dict(s.env)["GH_TOKEN"] == "" == dict(s.env)["GITHUB_TOKEN"] for s in lanes.calls)
    # lane slots are the machine's, shared by every replay
    assert (work / "work" / "lane-slots").resolve() == (cfg.work_dir / "lane-slots").resolve()
    assert report["options"]["gh_token"].startswith("none")
    reviewer = next(s for s in lanes.calls if s.role == "general")
    assert f"`{history['base']}..{history['head']}`" in reviewer.prompt
    assert "Title then" in reviewer.prompt and "Retitled" not in reviewer.prompt
    # Phase 2 ran past the gate; the gate's own publication is kept for scoring
    assert report["gate"]["forced_phase2"] is True
    assert report["gate"]["would_publish"]["blockers"] == 1
    assert {x["phase"] for x in report["lanes"]} >= {"phase1", "verify1", "phase2", "verify2"}
    assert [(f["title"], f["severity"]) for f in report["published"]] == [
        ("Overflow in f3", "blocking")
    ]
    assert report["published_from"] == "publication-final"
    assert "phase=final" in report["review_body"]
    assert report["historical_posted"][0]["hash"] == "h-src"
    assert report["seeded"]["runs"] == 1
    assert report["gpt"]["lanes"] == sum(x["model"].startswith("gpt") for x in report["lanes"])
    assert all(x["started_at"] and x["finished_at"] for x in report["lanes"])
    assert report["pin"]["ci_snapshot"] is True
    assert json.loads((work / "report.json").read_text())["source_run"] == history["src"]
    assert not (work / "work" / "worktrees").exists()  # disposable worktrees removed


def test_replay_never_touches_the_source_database(cfg, conn, history, tmp_path):
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [])
    before = (
        conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0],
    )
    _replay(cfg, history, tmp_path, lanes)
    after = (
        conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0],
    )
    assert before == after


def test_replay_pins_the_phase1_model_and_never_degrades(cfg, conn, history, tmp_path):
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [])
    report = _replay(cfg, history, tmp_path, lanes, pin_phase1_model="gemini-3.5-pro")
    p1 = {x["model"] for x in report["lanes"] if x["phase"] == "phase1"}
    assert p1 == {"gemini-3.5-pro"}
    assert report["options"]["pin_phase1_model"] == "gemini-3.5-pro"


def test_replay_refuses_to_start_in_degraded_mode(cfg, conn, history, tmp_path):
    with tx(conn):
        kv_set(conn, degraded.KV_FORCE, "on")
    lanes = FakeLanes(history["head"])
    with pytest.raises(replay.ReplayAborted, match="forced on"):
        _replay(cfg, history, tmp_path, lanes)
    assert lanes.calls == [] and not (tmp_path / "out").exists()


def test_degraded_probe_aborts_a_policy_with_stand_ins(cfg, conn):
    import dataclasses

    pol = dataclasses.replace(
        cfg.policy,
        degraded=DegradedPolicy(
            sentinel="gpt-6-astra", substitutes={"gpt-6-astra": Substitute("muse-spark")}
        ),
    )
    assert "out of quota" in replay.degraded_problem(pol, conn, lambda m: (True, "429"))
    assert replay.degraded_problem(pol, conn, lambda m: (False, "ok")) == ""
    assert replay.replay_policy(pol, None).degraded is None


def test_a_quota_failure_mid_replay_marks_the_report_invalid(cfg, conn, history, tmp_path):
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [])
    lanes.dead_models = {"gpt-6-astra"}
    lanes.dead_stderr = "API Error: 429 usage limit has been reached"
    report = _replay(cfg, history, tmp_path, lanes)
    assert report["status"] == "failed" and report["invalid"] == "quota failure mid-run"


def test_without_a_ci_snapshot_lanes_are_told_ci_is_unavailable(cfg, conn, history, tmp_path):
    _evidence(cfg, history["src"], history["base"], ci={"error": "rate limited"})
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [])
    report = _replay(cfg, history, tmp_path, lanes)
    assert report["pin"]["ci_snapshot"] is False
    evidence = json.loads((Path(report["run_dir"]) / "evidence.json").read_text())["evidence"]["ci"]
    assert "CI unavailable in this replay" in evidence["error"]


def test_a_conversation_run_is_replayed_as_one(cfg, conn, history, tmp_path, monkeypatch):
    with tx(conn):
        conn.execute(
            "INSERT INTO steps (run_id, name, status, started_at) VALUES (?,?,?,?)",
            (history["src"], "converse", "ok", CUTOFF),
        )
    seen: list[str] = []
    real_main = worker_mod.main

    def spy(cfg_, conn_, run_id, **kw):
        seen.append(
            conn_.execute(
                "SELECT h.trigger FROM runs r JOIN heads h ON h.id=r.head_id WHERE r.id=?",
                (run_id,),
            ).fetchone()[0]
        )
        return real_main(cfg_, conn_, run_id, **kw)

    monkeypatch.setattr(worker_mod, "main", spy)
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [])
    report = _replay(cfg, history, tmp_path, lanes)
    assert seen == ["review_reply"] and report["source"]["trigger_replayed_as"] == "review_reply"


def test_rerunning_a_replay_replaces_its_directory(cfg, conn, history, tmp_path):
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [])
    _replay(cfg, history, tmp_path, lanes)
    stale = tmp_path / "out" / f"run-{history['src']}" / "stale.txt"
    stale.write_text("x")
    report = _replay(cfg, history, tmp_path, lanes, sample=0)
    assert report["status"] == "done" and not stale.exists()


def test_lane_env_reaches_the_spawned_process(tmp_path, monkeypatch):
    """`LaneSpec.env` is set in the real lane's environment (run_claude_lane)."""
    from reviewsys import lane as lane_mod

    script = tmp_path / "claude"
    script.write_text('#!/bin/sh\nprintf \'{"result": "%s"}\' "$REPLAY_PROBE"\n')
    script.chmod(0o755)
    spec = LaneSpec(
        role="general",
        agent="a",
        model="m",
        effort="high",
        prompt="p",
        cwd=tmp_path,
        add_dir=tmp_path,
        timeout_seconds=30,
        claude_bin=str(script),
    )
    run = lane_mod.with_overrides(
        lane_mod.run_claude_lane, env={"REPLAY_PROBE": "pinned"}, disallowed_tools=("WebFetch",)
    )
    res: LaneResult = run(spec, tmp_path / "art", tmp_path)
    assert res.result_text == "pinned"
    argv = json.loads((tmp_path / "art" / "lane-meta.json").read_text())["argv"]
    assert argv[argv.index("--disallowedTools") + 1] == "WebFetch"


def test_status_is_reported_for_failed_replays(cfg, conn, history, tmp_path):
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [])
    lanes.timeout_roles = {"general"}
    report = _replay(cfg, history, tmp_path, lanes)
    assert report["status"] == RunStatus.FAILED.value and report["invalid"] is None


def test_fetch_head_keeps_going_when_the_commit_is_already_there(cfg, mirror):
    """Two arms of one head fetch the PR ref into the shared mirror at once (one loses the
    ref lock), or the PR has moved on: the replayed commit is what counts."""
    m = cfg.mirrors_dir / "dashpay__platform.git"
    wt.fetch_head(m, 1, mirror["head"])  # there is no refs/pull/1/head to fetch
    with pytest.raises(ReviewError, match="fetch"):
        wt.fetch_head(m, 1, "d" * 40)


def test_a_finding_posted_before_the_cutoff_by_a_run_still_going_is_seeded(conn, history, tmp_path):
    """The straddling run posted h-straddle before the cutoff and finished after it: its
    findings row is recovered, so the worker's prior list keeps the finding."""
    from reviewsys.db import connect

    with tx(conn):
        conn.execute(
            "INSERT INTO posted_findings (repo, number, hash, sha, posted_at) VALUES (?,?,?,?,?)",
            (REPO, NUMBER, "h-straddle", history["old_sha"], "2026-09-20T11:30:00Z"),
        )
    scratch = connect(tmp_path / "scratch.db")
    counts = replay.seed_history(
        scratch, conn, repo=REPO, number=NUMBER, sha=history["head"], cutoff=CUTOFF, exclude_run=0
    )
    assert counts["posted_rows_recovered"] == 1
    row = scratch.execute(
        "SELECT pf.hash, f.title FROM posted_findings pf JOIN findings f "
        "ON f.hash=pf.hash AND f.stage='posted' WHERE pf.hash='h-straddle'"
    ).fetchone()
    assert row["title"] == "Posted after the cutoff"


def test_lifted_statuses_as_of_the_cutoff(conn, history):
    with tx(conn):
        for ts, h, status, action in (
            (BEFORE, "h1", "WITHDRAWN", "replied"),
            (BEFORE, "h2", "FIXED", "replied"),
            ("2026-09-20T11:00:00Z", "h2", "STILL_VALID", "replied"),  # reopened before cutoff
            (BEFORE, "h3", "WITHDRAWN", "no_reply"),  # never reached the thread
            (AFTER, "h4", "WITHDRAWN", "replied"),  # after the cutoff
        ):
            conn.execute(
                "INSERT INTO events (ts, kind, repo, number, detail) VALUES (?,?,?,?,?)",
                (
                    ts,
                    "thread.answered",
                    REPO,
                    NUMBER,
                    json.dumps({"finding_hash": h, "status": status, "action": action}),
                ),
            )
    got = replay.lifted_before(conn, REPO, NUMBER, CUTOFF)
    assert got == {"h1": "WITHDRAWN"}


def test_a_same_sha_rereview_records_its_verdict_update_without_posting(
    cfg, conn, history, tmp_path
):
    """With the pinned gh a replay takes the paths a dry run skips: a final review of this
    head already stands, so the verdict follow-up is computed and recorded, never posted."""
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [_blocker()])
    gh = FakeGh()
    gh.posted_reviews = [
        {
            "id": 9,
            "body": f"<!-- thepastaclaw-review v1 phase=final sha={history['head']} -->",
            "event": "COMMENT",
        }
    ]
    report = _replay(cfg, history, tmp_path, lanes, gh_runner=gh.runner)
    assert report["status"] == "done", report["reason"]
    assert len(gh.posted_reviews) == 1  # nothing posted
    assert report["publications"]["final"]["verdict_update"] is True
    assert report["publications"]["final"]["event"] == "REQUEST_CHANGES"
    assert [f["title"] for f in report["published"]] == ["Overflow in f3"]


def test_lane_output_reading_github_directly_marks_the_replay_invalid(cfg, conn, history, tmp_path):
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [])

    def leaky(spec, art, wt_):
        res = lanes(spec, art, wt_)
        if spec.role == "general":
            (art / "stdout.json").write_text(
                "fetched https://api.github.com/repos/dashpay/platform/pulls/1/files"
            )
        return res

    report = _replay(cfg, history, tmp_path, leaky)
    assert report["invalid"] == "lane reached live GitHub"
    assert report["live_github"]["lane_output_hits"]


def test_a_read_only_token_reaches_the_lanes(cfg, conn, history, tmp_path):
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [])
    report = _replay(cfg, history, tmp_path, lanes, gh_token="ro-token")
    assert {dict(s.env)["GH_TOKEN"] for s in lanes.calls} == {"ro-token"}
    assert report["options"]["gh_token"] == "read-only token"


def test_an_audit_queue_source_run_is_replayed_as_a_live_review(cfg, conn, history, tmp_path):
    with tx(conn):
        conn.execute(
            "UPDATE heads SET queue='audit' WHERE id=(SELECT head_id FROM runs WHERE id=?)",
            (history["src"],),
        )
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [])
    report = _replay(cfg, history, tmp_path, lanes)
    assert report["status"] == "done" and report["source"]["queue"] == "audit"
