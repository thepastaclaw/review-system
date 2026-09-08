"""End-to-end: daemon tick -> spawned run -> worker with fake gh + fake lanes -> posted review."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reviewsys import worker
from reviewsys.daemon import Daemon
from reviewsys.db import tx
from reviewsys.ingest import enqueue_head
from reviewsys.models import RunStatus, Trigger
from reviewsys.scheduler import schedule
from reviewsys.steps import worktree as wt

HEAD = "a" * 40
GOLDEN = Path(__file__).parent / "golden" / "final-review.md"


@pytest.fixture(autouse=True)
def fake_git(monkeypatch, tmp_path):
    """Replace git operations with a local directory so no network/git is needed."""
    monkeypatch.setattr(wt, "ensure_mirror", lambda mirrors, repo: tmp_path / "mirror")
    monkeypatch.setattr(wt, "fetch_head", lambda mirror, number, sha: None)

    def create(mirror, wts, name, sha):
        p = wts / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(wt, "create_worktree", create)
    monkeypatch.setattr(wt, "remove_worktree", lambda mirror, path: None)
    monkeypatch.setattr(wt, "merge_base", lambda worktree, base, sha: "b" * 40)


def _blocking(title="Fee estimation omits drainage costs"):
    return {
        "file": "f.rs",
        "line_start": 11,
        "line_end": 12,
        "severity": "blocking",
        "confidence": 0.95,
        "category": "logic",
        "title": title,
        "body": "Estimation must be >= actual.",
    }


def _sugg(title="Add a regression test", file="f.rs"):
    return {
        "file": file,
        "line_start": 12,
        "line_end": 12,
        "severity": "suggestion",
        "confidence": 0.8,
        "category": "test-coverage",
        "title": title,
        "body": "No test covers the new branch.",
        "suggestion": "assert!(x);",
    }


def _verifier(findings):
    return {
        "summary": "The change is sound overall.\nSource: model-authored line to strip",
        "review_action": "COMMENT",
        "findings": findings,
        "dropped_findings": [],
        "out_of_scope_findings": [],
        "coderabbit_reactions": [],
        "prerequisite_adjudications": [],
        "adjudication_complete": True,
    }


def _run(cfg, conn, gh, lanes, *, dry_run=False):
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.MENTION)
    (rid,) = schedule(conn, cfg, spawn=False)
    status = worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, dry_run=dry_run, heartbeat=False)
    return rid, status


def test_two_phase_final_review_posts_once(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [_sugg()],
        "out_of_scope_findings": [],
    }
    lanes.verifier["preliminary"] = _verifier([])  # no blockers -> phase 2 admitted
    lanes.verifier["final"] = _verifier([_sugg()])
    rid, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    roles = [(s.role, s.model, s.effort) for s in lanes.calls]
    # selector, triage, phase1 general+always-on+security (GLM @max), verifier (Sol),
    # phase2 x3 (astra @high for the `normal` tier), final verifier (astra, fixed high)
    assert roles[0][0] == "selector"
    assert roles[1] == ("triage", "gpt-6-astra", "low")
    assert roles[2:5] == [
        ("general", "glm-5.3-flash", "max"),
        ("always-on", "glm-5.3-flash", "max"),
        ("security-auditor", "glm-5.3-flash", "max"),
    ]
    assert roles[5] == ("verifier", "gpt-5.6-sol", "high")
    assert [r[1:] for r in roles[6:9]] == [("gpt-6-astra", "high")] * 3
    assert roles[9] == ("verifier", "gpt-6-astra", "high")
    assert conn.execute("SELECT tier FROM runs WHERE id=?", (rid,)).fetchone()["tier"] == "normal"
    assert {r["effort"] for r in conn.execute("SELECT effort FROM lanes WHERE phase='phase2'")} == {
        "high"
    }
    assert len(gh.posted_reviews) == 1
    review = gh.posted_reviews[0]
    assert review["event"] == "COMMENT" and review["commit_id"] == HEAD
    assert len(review["comments"]) == 1 and review["comments"][0]["line"] == 12
    body = review["body"]
    assert "phase=final" in body and body.count("Source: ") == 1 and "model-authored" not in body
    assert "🟡 1 suggestion(s)" in body
    steps = {
        r["name"]: r["status"]
        for r in conn.execute("SELECT name, status FROM steps WHERE run_id=?", (rid,))
    }
    assert steps == {
        "worktree": "ok",
        "select": "ok",
        "triage": "ok",
        "context": "ok",
        "phase1": "ok",
        "verify1": "ok",
        "gate": "ok",
        "phase2": "ok",
        "verify2": "ok",
        "publish": "ok",
    }
    assert conn.execute("SELECT status FROM heads").fetchone()["status"] == "done"
    assert conn.execute("SELECT COUNT(*) FROM posted_findings").fetchone()[0] == 1
    assert conn.execute("SELECT SUM(tokens_in) FROM lanes").fetchone()[0] == 100 * 8
    # gate comment went in_progress -> done
    assert gh.gate_bodies[0].splitlines()[1].startswith("🔍 Review in progress") and gh.gate_bodies[
        -1
    ].splitlines()[1].startswith("✅ Final review complete")
    # golden body
    if GOLDEN.exists():
        assert body == GOLDEN.read_text()
    else:
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(body)


def test_blocker_gate_publishes_preliminary_request_changes(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [_blocking()],
        "out_of_scope_findings": [],
    }
    lanes.verifier["preliminary"] = _verifier([_blocking()])
    rid, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert len(gh.posted_reviews) == 1 and gh.posted_reviews[0]["event"] == "REQUEST_CHANGES"
    assert "phase=preliminary" in gh.posted_reviews[0]["body"]
    assert all(
        s.model == "glm-5.3-flash" or s.role in ("verifier", "selector", "triage")
        for s in lanes.calls
    ), "phase 2 must not run"
    assert not conn.execute(
        "SELECT 1 FROM steps WHERE run_id=? AND name='phase2'", (rid,)
    ).fetchone()
    assert (
        conn.execute("SELECT blocker_count FROM runs WHERE id=?", (rid,)).fetchone()[
            "blocker_count"
        ]
        == 1
    )


def test_own_pr_downgrades_to_comment(cfg, conn, gh, lanes):
    gh.pr = {**gh.pr, "user": {"login": "thepastaclaw"}}
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [_blocking()],
        "out_of_scope_findings": [],
    }
    lanes.verifier["preliminary"] = _verifier([_blocking()])
    _run(cfg, conn, gh, lanes)
    r = gh.posted_reviews[0]
    assert r["event"] == "COMMENT" and "Canonical verifier result: `REQUEST_CHANGES`" in r["body"]


def test_broken_json_is_repaired_once(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    lanes.broken_once = {"general"}
    _rid, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert any(s.role == "repair" for s in lanes.calls)
    assert (
        conn.execute("SELECT status FROM lanes WHERE role='general' AND phase='phase1'").fetchone()[
            "status"
        ]
        == "repaired"
    )


def test_lane_timeout_fails_run_as_infra_and_requeues(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.timeout_roles = {"general"}
    rid, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.FAILED
    run = conn.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
    assert run["fail_kind"] == "infra" and "timed out" in run["reason"]
    assert conn.execute("SELECT status, attempts FROM heads").fetchone()["status"] == "queued"
    assert (
        conn.execute(
            "SELECT status FROM steps WHERE run_id=? AND name='phase1'", (rid,)
        ).fetchone()["status"]
        == "failed"
    )
    assert gh.gate_bodies[-1].splitlines()[1].startswith("⚠️ Automated review could not complete")
    assert not gh.posted_reviews


def test_closed_pr_is_fatal(cfg, conn, gh, lanes):
    gh.pr = {**gh.pr, "state": "closed"}
    rid, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.FAILED
    assert (
        conn.execute("SELECT fail_kind FROM runs WHERE id=?", (rid,)).fetchone()["fail_kind"]
        == "fatal"
    )
    assert conn.execute("SELECT status FROM heads").fetchone()["status"] == "failed"


def test_head_moved_is_fatal(cfg, conn, gh, lanes):
    gh.pr = {**gh.pr, "head": {"sha": "c" * 40}}
    rid, status = _run(cfg, conn, gh, lanes)
    assert (
        status == RunStatus.FAILED
        and "live head"
        in conn.execute("SELECT reason FROM runs WHERE id=?", (rid,)).fetchone()["reason"]
    )


def test_cross_round_dedupe_suppresses_existing_thread(cfg, conn, gh, lanes):
    s = _sugg()
    from reviewsys.contract import Finding

    fh = Finding.from_dict(s).hash
    gh.inline = [
        {
            "id": 900,
            "node_id": "N1",
            "user": {"login": "thepastaclaw"},
            "path": "f.rs",
            "line": 12,
            "html_url": "https://gh/900",
            "body": f"<!-- thepastaclaw-review v1 finding={fh} dedupe=x -->\n**🟡 Suggestion: {s['title']}**\n\nbody",
        }
    ]
    lanes.reviewer["default"] = {"summary": "ok", "findings": [s], "out_of_scope_findings": []}
    lanes.verifier["preliminary"] = _verifier([])
    lanes.verifier["final"] = _verifier([s])
    _rid, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    # every finding was a duplicate -> no new review round
    assert not gh.posted_reviews


def test_already_published_for_sha_is_skipped(cfg, conn, gh, lanes):
    gh.posted_reviews.append(
        {"id": 1, "body": f"<!-- thepastaclaw-review-phase v1 phase=final sha={HEAD} policy=x -->"}
    )
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    rid, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE and len(gh.posted_reviews) == 1
    assert (
        "already_published_for_sha"
        in conn.execute(
            "SELECT detail FROM steps WHERE run_id=? AND name='publish'", (rid,)
        ).fetchone()["detail"]
    )


def test_dry_run_posts_nothing_but_writes_body(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [_sugg()],
        "out_of_scope_findings": [],
    }
    lanes.verifier["default"] = _verifier([_sugg()])
    rid, status = _run(cfg, conn, gh, lanes, dry_run=True)
    assert status == RunStatus.DONE and not gh.posted_reviews and not gh.gate_bodies
    assert (cfg.runs_dir / f"run-{rid}" / "review-final.md").exists()


def test_coderabbit_reactions_posted(cfg, conn, gh, lanes):
    gh.threads = [
        {
            "id": "T1",
            "isResolved": False,
            "isOutdated": False,
            "path": "f.rs",
            "line": 12,
            "comments": {
                "nodes": [
                    {
                        "databaseId": 4242,
                        "author": {"login": "coderabbitai[bot]"},
                        "body": "Consider bounds",
                        "createdAt": "x",
                        "authorAssociation": "NONE",
                    }
                ]
            },
        }
    ]
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    v = _verifier([])
    v["coderabbit_reactions"] = [
        {"comment_id": 4242, "action": "disagree", "reply": "Bounds are enforced upstream."}
    ]
    lanes.verifier["default"] = v
    _rid, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert (
        gh.reactions == [{"content": "-1"}]
        and gh.replies[-1]["body"] == "Bounds are enforced upstream."
    )


def test_selector_failure_is_recorded_not_silent(cfg, conn, gh, lanes):
    lanes.selector = "not json at all"
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    rid, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert (
        conn.execute("SELECT COUNT(*) FROM events WHERE kind='select.degraded'").fetchone()[0] == 1
    )
    sel = json.loads((cfg.runs_dir / f"run-{rid}" / "selector.json").read_text())
    assert sel["method"] == "heuristic" and "always-on" in sel["selected"]


def test_daemon_tick_shadow_mode(cfg, conn, gh, notifier):
    gh.open_prs["dashpay/platform"] = [
        {
            "number": 3,
            "headRefOid": "d" * 40,
            "isDraft": False,
            "title": "t",
            "updatedAt": "2026-09-01T00:00:00Z",
            "author": {"login": "dev"},
        }
    ]
    d = Daemon(cfg, conn, gh=gh, notifier=notifier, spawn=False)
    res = d.tick(force=True)
    assert res["ingest"]["dashpay/platform"]["created"] == 1
    assert "schedule" not in res  # shadow mode observes only
    with tx(conn):
        conn.execute("UPDATE heads SET eligible_at=queued_at")
    d.tick(force=True)
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    # a live daemon on the same DB schedules it
    live = Daemon(cfg, conn, gh=gh, notifier=notifier, spawn=True)
    res = live.tick(force=True)
    assert len(res["schedule"]) == 1
    assert conn.execute("SELECT status FROM runs").fetchone()["status"] == "spawned"


def _reviewer_calls(lanes):
    return [s for s in lanes.calls if s.role not in ("selector", "triage", "verifier", "repair")]


def test_tier_scales_effort_and_is_disclosed(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    lanes.triage = {"tier": "critical", "reasoning": "touches consensus"}
    rid, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    efforts = {(s.model, s.effort) for s in _reviewer_calls(lanes)}
    assert efforts == {("glm-5.3-flash", "max"), ("gpt-6-astra", "xhigh")}
    body = gh.posted_reviews[0]["body"]
    assert "- Triage: `critical` by `gpt-6-astra` (effort low) — touches consensus" in body
    assert "general (completed, effort xhigh); agent `phase2-reviewer`" in body
    assert conn.execute("SELECT tier FROM runs WHERE id=?", (rid,)).fetchone()["tier"] == "critical"
    gate = json.loads(
        conn.execute("SELECT detail FROM steps WHERE run_id=? AND name='gate'", (rid,)).fetchone()[
            "detail"
        ]
    )
    assert gate == {"admit_phase2": True, "tier": "critical", "phase2_effort": "xhigh"}


def test_low_tier_uses_medium_phase2(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    lanes.triage = {"tier": "low", "reasoning": "small"}
    _run(cfg, conn, gh, lanes)
    assert {(s.model, s.effort) for s in _reviewer_calls(lanes)} == {
        ("glm-5.3-flash", "high"),
        ("gpt-6-astra", "medium"),
    }


def test_trivial_tier_publishes_final_from_phase1(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [_sugg()],
        "out_of_scope_findings": [],
    }
    lanes.verifier["preliminary"] = {**_verifier([_sugg()]), "review_action": "APPROVE"}
    lanes.triage = {"tier": "trivial", "reasoning": "typo fix"}
    rid, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert all(s.model == "glm-5.3-flash" for s in _reviewer_calls(lanes)), "phase 2 must not run"
    assert not conn.execute(
        "SELECT 1 FROM steps WHERE run_id=? AND name='phase2'", (rid,)
    ).fetchone()
    review = gh.posted_reviews[0]
    # a Phase-1-only verdict is published as COMMENT even when the verifier said APPROVE
    assert review["event"] == "COMMENT" and "phase=final" in review["body"]
    assert "## Final review — Phase 1 only (trivial change)" in review["body"]
    assert "- Phase 2 reviewers: **not run (triage rated this change trivial)**" in review["body"]
    assert "never approves" in review["body"]
    assert "Validated blockers were found" not in review["body"]
    gate = gh.gate_bodies[-1]
    assert "Final review complete — Phase 1 only" in gate and "triage: trivial" in gate


def test_trivial_tier_with_blockers_stays_preliminary(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [_blocking()],
        "out_of_scope_findings": [],
    }
    lanes.verifier["preliminary"] = _verifier([_blocking()])
    lanes.triage = {"tier": "trivial", "reasoning": "looks small"}
    _run(cfg, conn, gh, lanes)
    review = gh.posted_reviews[0]
    assert review["event"] == "REQUEST_CHANGES" and "phase=preliminary" in review["body"]
    assert "deferred by blocker gate" in review["body"]


def test_triage_failure_falls_back_and_is_recorded(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    lanes.triage = "garbage"
    rid, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert conn.execute("SELECT tier FROM runs WHERE id=?", (rid,)).fetchone()["tier"] == "normal"
    assert (
        conn.execute("SELECT COUNT(*) FROM events WHERE kind='triage.degraded'").fetchone()[0] == 1
    )
    t = json.loads((cfg.runs_dir / f"run-{rid}" / "triage.json").read_text())
    assert t["method"] == "fallback" and t["tier"] == "normal" and t["error"]
    assert "- Triage: `normal` by fallback after triage failure" in gh.posted_reviews[0]["body"]


def test_policy_without_triage_block_runs_single_tier(cfg, conn, gh, lanes, skills_dir, tmp_path):
    from reviewsys import config as cfg_mod

    raw = json.loads((skills_dir / "config.json").read_text())
    del raw["review_model_policy"]["triage"]
    (skills_dir / "config.json").write_text(json.dumps(raw))
    cfg2 = cfg_mod.load(tmp_path / "config.toml")  # written by the cfg fixture
    assert cfg2.policy.triage is None and list(cfg2.policy.tiers) == ["normal"]
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    rid, status = _run(cfg2, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert not any(s.role == "triage" for s in lanes.calls)
    assert {(s.model, s.effort) for s in _reviewer_calls(lanes)} == {
        ("glm-5.3-flash", "max"),
        ("gpt-6-astra", "high"),
    }
    assert conn.execute("SELECT tier FROM runs WHERE id=?", (rid,)).fetchone()["tier"] is None
    assert "- Triage:" not in gh.posted_reviews[0]["body"]


def _queue_extra_heads(conn, cfg, n):
    with tx(conn):
        for i in range(n):
            enqueue_head(conn, cfg, "dashpay/platform", 100 + i, f"{i:040x}", Trigger.NEW_PR)


def test_deep_backlog_skips_phase1_and_discloses_it(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [_sugg()],
        "out_of_scope_findings": [],
    }
    lanes.verifier["final"] = _verifier([_sugg()])
    _queue_extra_heads(conn, cfg, cfg.backlog_skip_phase1_above + 1)  # + the head under test
    rid, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    calls = _reviewer_calls(lanes)
    assert calls and all(s.model == "gpt-6-astra" for s in calls), "no GLM lane may run"
    assert not any(
        s.role == "verifier" and "must be `preliminary`" in s.prompt for s in lanes.calls
    )
    verifier = [s for s in lanes.calls if s.role == "verifier"]
    assert len(verifier) == 1 and "Phase-1 reviewer lanes did not run" in verifier[0].prompt
    step = conn.execute(
        "SELECT status, detail FROM steps WHERE run_id=? AND name='phase1'", (rid,)
    ).fetchone()
    assert step["status"] == "skipped" and "queued" in json.loads(step["detail"])
    assert not conn.execute(
        "SELECT 1 FROM steps WHERE run_id=? AND name IN ('verify1','gate')", (rid,)
    ).fetchone()
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM events WHERE kind='phase1.skipped_backlog' AND run_id=?", (rid,)
        ).fetchone()[0]
        == 1
    )
    review = gh.posted_reviews[0]
    assert "phase=final" in review["body"]
    assert "## Final validation — Phase 2 only (queue backlog)" in review["body"]
    assert "- Phase 1 reviewers: **not run (skipped for throughput: " in review["body"]
    assert "- Phase 2 reviewers: `gpt-6-astra`" in review["body"]
    assert "Phase 2 only (queue backlog)" in gh.gate_bodies[-1]


def test_backlog_at_or_below_limit_runs_both_phases(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    _queue_extra_heads(conn, cfg, cfg.backlog_skip_phase1_above - 1)  # exactly at the limit
    rid, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert {s.model for s in _reviewer_calls(lanes)} == {"glm-5.3-flash", "gpt-6-astra"}
    assert (
        conn.execute(
            "SELECT status FROM steps WHERE run_id=? AND name='phase1'", (rid,)
        ).fetchone()["status"]
        == "ok"
    )
    assert "Phase 1 + Phase 2" in gh.posted_reviews[0]["body"]


def test_backlog_skip_does_not_apply_to_trivial_tier(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["preliminary"] = _verifier([])
    lanes.triage = {"tier": "trivial", "reasoning": "typo"}
    _queue_extra_heads(conn, cfg, cfg.backlog_skip_phase1_above + 5)
    _run(cfg, conn, gh, lanes)
    assert all(s.model == "glm-5.3-flash" for s in _reviewer_calls(lanes))
    assert "Phase 1 only (trivial change)" in gh.posted_reviews[0]["body"]
