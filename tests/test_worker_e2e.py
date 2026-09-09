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


def _run_repo(cfg, conn, gh, lanes, repo, number, trigger=Trigger.MENTION):
    with tx(conn):
        enqueue_head(conn, cfg, repo, number, HEAD, trigger)
    (rid,) = schedule(conn, cfg, spawn=False)
    status = worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False)
    return rid, status


def test_adhoc_review_of_unlisted_repo(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [_sugg()],
        "out_of_scope_findings": [],
    }
    lanes.verifier["default"] = _verifier([_sugg()])
    lanes.selector = {"selected": ["security-auditor"], "reasoning": "auth code"}
    _rid, status = _run_repo(cfg, conn, gh, lanes, "dashpay/quorum-list-server", 14)
    assert status == RunStatus.DONE
    sel = lanes.calls[0]
    assert sel.role == "selector" and "security-auditor" in sel.prompt
    assert "always-on" not in sel.prompt, "always-run specialists are repo-specific"
    roles = [s.role for s in _reviewer_calls(lanes)]
    assert roles == ["general", "security-auditor"] * 2
    general = next(s for s in _reviewer_calls(lanes) if s.role == "general")
    assert "ad hoc review" in general.prompt and "PROJECT SKILL" not in general.prompt
    body = gh.posted_reviews[0]["body"]
    assert "- Ad hoc review: this repository has no PastaClaw review skill" in body
    assert "ad hoc (no repo skill)" in gh.gate_bodies[-1]


def _gql_comment(c):
    return {
        "databaseId": c["id"],
        "author": {"login": c["author"]},
        "body": c["body"],
        "createdAt": c.get("created_at"),
        "authorAssociation": c.get("association"),
    }


def _prior_thread(fh, *, replies):
    """Raw GraphQL reviewThreads node (what FakeGh serves) for one bot finding plus replies."""
    root = {
        "id": 900,
        "author": "thepastaclaw",
        "body": f"<!-- thepastaclaw-review v1 finding={fh} dedupe=x -->\n**🟡 Suggestion: T**\n\nold body",
        "created_at": "2026-09-07T00:00:00Z",
        "association": "NONE",
    }
    return {
        "id": "PRRT_1",
        "isResolved": False,
        "isOutdated": False,
        "path": "f.rs",
        "line": 12,
        "comments": {"nodes": [_gql_comment(c) for c in (root, *replies)]},
    }


def _seed_prior(conn, finding, *, body="old body"):
    from reviewsys.contract import Finding

    f = Finding.from_dict(finding)
    with tx(conn):
        hid = conn.execute(
            "INSERT INTO heads (repo, number, sha, trigger, priority, status, queued_at, eligible_at) VALUES (?,?,?,?,?,?,?,?)",
            ("dashpay/platform", 1, "b" * 40, "new_pr", 0, "done", "2026-09-07", "2026-09-07"),
        ).lastrowid
        rid = conn.execute(
            "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at) VALUES (?,1,'done','t','2026-09-07T00:00:00Z','2026-09-07T00:00:00Z')",
            (hid,),
        ).lastrowid
        conn.execute(
            "INSERT INTO findings (run_id, phase, stage, hash, file, line_start, line_end, severity, category, title, body) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                rid,
                "verify2",
                "posted",
                f.hash,
                f.file,
                f.line_start,
                f.line_end,
                f.severity,
                f.category,
                f.title,
                body,
            ),
        )
        conn.execute(
            "INSERT INTO posted_findings (repo, number, hash, sha, review_id, posted_at) VALUES (?,?,?,?,?,?)",
            ("dashpay/platform", 1, f.hash, "b" * 40, 1, "2026-09-07T00:00:00Z"),
        )
    return f.hash


def test_reply_on_prior_finding_reaches_reviewer_and_is_answered(cfg, conn, gh, lanes):
    s = _sugg()
    fh = _seed_prior(conn, s)
    reply = {
        "id": 901,
        "author": "knst",
        "body": "The pool has no priorities; this does not apply.",
        "created_at": "2026-09-08T20:17:36Z",
        "association": "COLLABORATOR",
    }
    gh.threads = [_prior_thread(fh, replies=[reply])]
    withdrawn = {
        "summary": "ok",
        "findings": [],
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": [
            {
                "finding_hash": fh,
                "status": "WITHDRAWN",
                "reason": "You are right: the pool has no priorities, so the ordering concern does not arise.",
            }
        ],
    }
    lanes.reviewer["default"] = withdrawn
    lanes.verifier["default"] = _verifier([])
    _rid, status = _run_repo(cfg, conn, gh, lanes, "dashpay/platform", 1, Trigger.REVIEW_REPLY)
    assert status == RunStatus.DONE
    prompt = next(c.prompt for c in _reviewer_calls(lanes))
    assert "thread_replies" in prompt and "no priorities" in prompt and "old body" in prompt
    # the reviewer also sees our own root comment inside the evidence thread
    assert "finding=" + fh in prompt.split("Structured review-thread state", 1)[1]
    assert gh.replies and gh.replies[0]["body"].startswith(
        "<!-- thepastaclaw-thread-answer v1 sha=" + HEAD
    )
    assert "**Withdrawn** (re-reviewed at `aaaaaaaa`): You are right" in gh.replies[0]["body"]
    assert any("resolveReviewThread" in " ".join(c) for c in gh.calls)
    ev = conn.execute("SELECT detail FROM events WHERE kind='thread.answered'").fetchone()
    assert '"status": "WITHDRAWN"' in ev["detail"] and '"resolved": true' in ev["detail"]


def test_still_valid_reply_is_answered_without_resolving(cfg, conn, gh, lanes):
    s = _sugg()
    fh = _seed_prior(conn, s)
    gh.threads = [
        _prior_thread(
            fh,
            replies=[
                {
                    "id": 901,
                    "author": "knst",
                    "body": "disagree",
                    "created_at": "x",
                    "association": "COLLABORATOR",
                }
            ],
        )
    ]
    carried = {**s, "finding_hash": fh, "body": "Still applies because the queue is FIFO."}
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [carried],
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": [
            {
                "finding_hash": fh,
                "status": "STILL_VALID",
                "reason": "The FIFO queue still serialises warmers ahead of aggregation.",
            }
        ],
    }
    lanes.verifier["default"] = _verifier([carried])
    # the existing inline comment makes the carried finding a cross-round duplicate (no new review)
    gh.inline = [
        {
            "id": 900,
            "node_id": "N1",
            "user": {"login": "thepastaclaw"},
            "path": "f.rs",
            "line": 12,
            "html_url": "https://gh/900",
            "body": gh.threads[0]["comments"]["nodes"][0]["body"],
        }
    ]
    _rid, status = _run_repo(cfg, conn, gh, lanes, "dashpay/platform", 1, Trigger.REVIEW_REPLY)
    assert status == RunStatus.DONE
    assert not gh.posted_reviews  # every finding was a duplicate: no new review round
    # ...but the human asked a question on the thread, and the answer is the only thing
    # that reaches them, so it is posted there, and the thread stays open
    assert len(gh.replies) == 1
    assert "**Still applies** (re-reviewed at `aaaaaaaa`): The FIFO queue" in gh.replies[0]["body"]
    assert not any("resolveReviewThread" in " ".join(c) for c in gh.calls)


def test_thread_answer_is_posted_once_per_reply(cfg, conn, gh, lanes):
    from reviewsys.contract import parse_verifier_output
    from reviewsys.publish import answer_replied_threads

    verified = parse_verifier_output(
        {**_verifier([]), "review_phase": "final"},
        expected_phase="final",
        expected_coderabbit_ids=[],
    )
    threads = {
        "abc": {
            "comment_id": 900,
            "thread_id": "PRRT_1",
            "latest_reply_id": 901,
            "replies": [{"id": 901, "author": "knst", "body": "x"}],
        }
    }
    recon = {
        "abc": {"finding_hash": "abc", "status": "FIXED", "reason": "Addressed by the new guard."}
    }
    out = answer_replied_threads(
        gh, "dashpay/platform", 1, HEAD, threads=threads, reconciliation=recon, verified=verified
    )
    assert out[0]["action"] == "replied" and out[0]["resolved"] is True
    gh.inline = [
        {
            "id": 902,
            "in_reply_to_id": 900,
            "user": {"login": "thepastaclaw"},
            "body": gh.replies[0]["body"],
        }
    ]
    out = answer_replied_threads(
        gh, "dashpay/platform", 1, HEAD, threads=threads, reconciliation=recon, verified=verified
    )
    assert out[0]["action"] == "already_answered" and len(gh.replies) == 1
    # a newer human reply on the same thread and head is a new question: answer again
    threads["abc"]["latest_reply_id"] = 903
    out = answer_replied_threads(
        gh, "dashpay/platform", 1, HEAD, threads=threads, reconciliation=recon, verified=verified
    )
    assert out[0]["action"] == "replied" and len(gh.replies) == 2


def test_defaulted_withdrawal_answers_but_never_resolves(cfg, conn, gh, lanes):
    """A verifier that drops the finding without any reviewer stating a status: the bot says the
    finding did not survive, but does not close the human's thread on a default."""
    from reviewsys.contract import parse_verifier_output
    from reviewsys.publish import answer_replied_threads

    verified = parse_verifier_output(
        {**_verifier([]), "review_phase": "final"},
        expected_phase="final",
        expected_coderabbit_ids=[],
    )
    threads = {
        "abc": {"comment_id": 900, "thread_id": "PRRT_1", "latest_reply_id": 901, "replies": []}
    }
    out = answer_replied_threads(
        gh, "dashpay/platform", 1, HEAD, threads=threads, reconciliation={}, verified=verified
    )
    assert (
        out[0]["status"] == "WITHDRAWN"
        and out[0]["action"] == "replied"
        and "resolved" not in out[0]
    )
    assert not any("resolveReviewThread" in " ".join(c) for c in gh.calls)


def test_verifier_kept_finding_without_hash_still_counts_as_still_valid(cfg, conn, gh, lanes):
    from reviewsys.contract import Finding, parse_verifier_output
    from reviewsys.publish import answer_replied_threads

    s = _sugg()
    fh = Finding.from_dict(s).hash
    verified = parse_verifier_output(
        {**_verifier([s]), "review_phase": "final"},
        expected_phase="final",
        expected_coderabbit_ids=[],
    )
    assert verified.findings[0].prior_hash is None  # the verifier forgot to echo finding_hash
    threads = {
        fh: {"comment_id": 900, "thread_id": "PRRT_1", "latest_reply_id": 901, "replies": []}
    }
    recon = {fh: {"finding_hash": fh, "status": "WITHDRAWN", "reason": "a lane thought so"}}
    out = answer_replied_threads(
        gh, "dashpay/platform", 1, HEAD, threads=threads, reconciliation=recon, verified=verified
    )
    assert out[0]["status"] == "STILL_VALID"
    assert (
        "**Still applies**" in gh.replies[0]["body"]
        and "a lane thought so" not in gh.replies[0]["body"]
    )
    assert not any("resolveReviewThread" in " ".join(c) for c in gh.calls)


def test_reason_is_scrubbed_of_mentions_and_retriggers(cfg, conn, gh, lanes):
    from reviewsys.publish import _answer_outcome

    status, reason, explicit = _answer_outcome(
        "h", {"status": "WITHDRAWN", "reason": "cc @someone; also @coderabbitai review"}, set()
    )
    assert (status, explicit) == (
        "WITHDRAWN",
        True,
    ) and reason == "This finding did not survive verification on the current head."
    status, reason, _ = _answer_outcome(
        "h", {"status": "WITHDRAWN", "reason": "Agreed with @knst here. " + "x" * 5000}, set()
    )
    assert "@\u200bknst" in reason and len(reason) <= 1200


def test_resolved_and_already_answered_threads_are_left_alone(cfg, conn, gh, lanes):
    from reviewsys.github import replied_finding_threads

    human = {
        "id": 901,
        "author": "knst",
        "body": "no",
        "created_at": "2026-09-08T20:00:00Z",
        "association": "COLLABORATOR",
    }
    bot_answer = {
        "id": 902,
        "author": "thepastaclaw",
        "body": "answer",
        "created_at": "2026-09-08T21:00:00Z",
        "association": "NONE",
    }
    root = {
        "id": 900,
        "author": "thepastaclaw",
        "body": "<!-- thepastaclaw-review v1 finding=abc -->\n**🟡 Suggestion: T**",
        "created_at": "x",
        "association": "NONE",
    }

    def parsed(comments, resolved=False):
        return {
            "thread_id": "t",
            "is_resolved": resolved,
            "is_outdated": False,
            "path": "f.rs",
            "line": 1,
            "comments": comments,
        }

    assert replied_finding_threads([parsed([root, human], resolved=True)], "thepastaclaw") == {}
    assert replied_finding_threads([parsed([root, human, bot_answer])], "thepastaclaw") == {}
    again = {**human, "id": 903, "created_at": "2026-09-08T22:00:00Z"}
    out = replied_finding_threads([parsed([root, human, bot_answer, again])], "thepastaclaw")
    assert out["abc"]["latest_reply_id"] == 903 and [r["id"] for r in out["abc"]["replies"]] == [
        901,
        903,
    ]


def test_reply_rerun_on_reviewed_commit_answers_thread_without_new_review(cfg, conn, gh, lanes):
    s = _sugg()
    fh = _seed_prior(conn, s)
    gh.threads = [
        _prior_thread(
            fh,
            replies=[
                {
                    "id": 901,
                    "author": "knst",
                    "body": "wrong",
                    "created_at": "x",
                    "association": "COLLABORATOR",
                }
            ],
        )
    ]
    gh.posted_reviews.append(
        {
            "id": 4242,
            "html_url": "https://gh/r/4242",
            "body": f"<!-- thepastaclaw-review-phase v1 phase=final sha={HEAD} policy=x -->",
        }
    )
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [],
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": [
            {"finding_hash": fh, "status": "WITHDRAWN", "reason": "Agreed, the guard covers it."}
        ],
    }
    lanes.verifier["default"] = _verifier([])
    with tx(conn):
        conn.execute("UPDATE heads SET sha=?, status='done'", (HEAD,))
        assert (
            enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.REVIEW_REPLY) == "requeued"
        )
    (rid,) = schedule(conn, cfg, spawn=False)
    status = worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False)
    assert status == RunStatus.DONE
    assert len(gh.posted_reviews) == 1, "same commit: no second review"
    assert len(gh.replies) == 1 and "**Withdrawn**" in gh.replies[0]["body"]
    assert any("resolveReviewThread" in " ".join(c) for c in gh.calls)


def test_reply_on_legacy_finding_without_db_row_is_adjudicated(cfg, conn, gh, lanes):
    # the thread root carries a marker hash this database has never seen
    gh.threads = [
        _prior_thread(
            "c5669452cde4",
            replies=[
                {
                    "id": 901,
                    "author": "knst",
                    "body": "does not apply",
                    "created_at": "x",
                    "association": "COLLABORATOR",
                }
            ],
        )
    ]
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [],
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": [
            {"finding_hash": "c5669452cde4", "status": "WITHDRAWN", "reason": "Agreed."}
        ],
    }
    lanes.verifier["default"] = _verifier([])
    _rid, status = _run_repo(cfg, conn, gh, lanes, "dashpay/platform", 1, Trigger.REVIEW_REPLY)
    assert status == RunStatus.DONE
    prompt = next(c.prompt for c in _reviewer_calls(lanes))
    assert '"finding_hash": "c5669452cde4"' in prompt and '"original_title": "T"' in prompt
    assert '"severity": "suggestion"' in prompt and "old body" in prompt
    assert len(gh.replies) == 1 and "**Withdrawn**" in gh.replies[0]["body"]
