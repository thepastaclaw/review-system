"""End-to-end: daemon tick -> spawned run -> worker with fake gh + fake lanes -> posted review."""

from __future__ import annotations

import dataclasses
import json
import threading
from pathlib import Path

import pytest

from reviewsys import labels, publish, worker
from reviewsys import lane as lane_mod
from reviewsys.daemon import Daemon
from reviewsys.db import kv_get, kv_set, tx
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
    monkeypatch.setattr(wt, "fetch_commit", lambda mirror, url, sha: not sha.startswith("dead"))


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


def _open_pr(conn, sha=HEAD, updated_at="2026-01-01T00:00:00Z"):
    """The `prs` row ingest would have written for an open PR at `sha`."""
    with tx(conn):
        conn.execute(
            "INSERT INTO prs (repo, number, head_sha, state, updated_at) VALUES (?,?,?,?,?)",
            ("dashpay/platform", 1, sha, "open", updated_at),
        )


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
    # prep (selection + triage in one lane, on the triage model), phase1
    # general+always-on+security (GLM @max, in parallel), verifier (Sol), phase2 x3 (astra
    # @high for the `normal` tier), final verifier (astra, fixed high)
    assert roles[0] == ("prep", "gpt-6-astra", "low")
    assert sorted(roles[1:4]) == [
        ("always-on", "glm-5.3-flash", "max"),
        ("general", "glm-5.3-flash", "max"),
        ("security-auditor", "glm-5.3-flash", "max"),
    ]
    assert roles[4] == ("verifier", "gpt-5.6-sol", "high")
    assert [r[1:] for r in roles[5:8]] == [("gpt-6-astra", "high")] * 3
    assert roles[8] == ("verifier", "gpt-6-astra", "high")
    assert conn.execute("SELECT tier FROM runs WHERE id=?", (rid,)).fetchone()["tier"] == "normal"
    assert {r["effort"] for r in conn.execute("SELECT effort FROM lanes WHERE phase='phase2'")} == {
        "high"
    }
    assert len(gh.posted_reviews) == 1
    review = gh.posted_reviews[0]
    assert review["event"] == "COMMENT" and review["commit_id"] == HEAD
    assert len(review["comments"]) == 1 and review["comments"][0]["line"] == 12
    # the inline footer names the lanes that raised the finding, not the verifier's legacy labels
    assert (
        "<sub>source: `glm-5.3-flash` (phase1-reviewer: general, always-on, security-auditor); "
        "`gpt-6-astra` (phase2-reviewer: general, always-on, security-auditor)</sub>"
    ) in review["comments"][0]["body"]
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


def test_final_review_publishes_with_binary_diff(cfg, conn, gh, lanes, binary_diff_gh):
    # Use the actual subprocess/decoder for the diff, and the usual fake for GitHub writes.
    runner = gh.runner
    gh.runner = lambda argv, stdin, timeout: (
        binary_diff_gh.runner([binary_diff_gh.bin, *argv[1:]], stdin, timeout)
        if list(argv[1:3]) == ["pr", "diff"]
        else runner(argv, stdin, timeout)
    )
    blocker = _blocking("Preserve café 東京 🦀 in the review")
    unmappable = _sugg("Binary proof needs coverage — café", file="proof.bin")
    unmappable["line_start"] = unmappable["line_end"] = 1
    findings = [blocker, unmappable]
    lanes.reviewer["default"] = {"summary": "ok", "findings": findings, "out_of_scope_findings": []}
    lanes.verifier["preliminary"] = _verifier([])
    lanes.verifier["final"] = _verifier(findings)
    rid, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    (review,) = gh.posted_reviews
    assert review["event"] == "REQUEST_CHANGES" and review["commit_id"] == HEAD
    (inline,) = review["comments"]
    assert inline["path"] == "f.rs" and inline["line"] == 12
    assert blocker["title"] in inline["body"]
    assert unmappable["title"] in review["body"]
    assert unmappable["body"] in review["body"]
    assert "1 finding(s) not shown inline" in review["body"]
    steps = {
        r["name"]: r["status"] for r in conn.execute("SELECT * FROM steps WHERE run_id=?", (rid,))
    }
    assert steps["verify2"] == steps["publish"] == "ok"
    assert conn.execute("SELECT blocker_count FROM runs WHERE id=?", (rid,)).fetchone()[0] == 1


@pytest.mark.parametrize("expected", [False, True])
def test_terminal_publish_error_closes_progress(cfg, conn, gh, lanes, monkeypatch, expected):
    from reviewsys.models import FailKind, ReviewError

    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])

    def crash(*args, **kwargs):
        if expected:
            raise ReviewError(FailKind.CONTRACT, "publication rejected")
        raise UnicodeDecodeError("utf-8", b"\xa1", 0, 1, "invalid start byte")

    monkeypatch.setattr(worker, "step_publish", crash)
    rid, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.FAILED
    step = conn.execute("SELECT * FROM steps WHERE run_id=? AND name='publish'", (rid,)).fetchone()
    assert step["status"] == "failed" and step["finished_at"]
    error = json.loads(step["detail"])["error"]
    assert ("publication rejected" if expected else "UnicodeDecodeError") in error
    assert not conn.execute(
        "SELECT 1 FROM steps WHERE run_id=? AND status='running'", (rid,)
    ).fetchone()
    assert (
        conn.execute(
            "SELECT status FROM steps WHERE run_id=? AND name='verify2'", (rid,)
        ).fetchone()[0]
        == "ok"
    )
    assert gh.gate_bodies[-1].splitlines()[1].startswith("⚠️ Automated review could not complete")
    assert not gh.posted_reviews


def test_considered_dismissed_reaches_the_verifier_but_is_not_published(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [_sugg()],
        "out_of_scope_findings": [],
        "considered_dismissed": [
            {
                "area": "fee estimation",
                "concern": "drainage costs missing from the estimate",
                "why_dismissed": "the v2 caller adds them before the check",
            },
            "not an object",
        ],
    }
    lanes.verifier["preliminary"] = _verifier([])
    lanes.verifier["final"] = _verifier([_sugg()])
    _, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    verifiers = [s for s in lanes.calls if s.role == "verifier"]
    assert len(verifiers) == 2
    for v in verifiers:  # Phase-1 and Phase-2 reviewer outputs alike
        assert "drainage costs missing from the estimate" in v.prompt
        assert "the v2 caller adds them before the check" in v.prompt
        assert "not an object" not in v.prompt
    review = gh.posted_reviews[0]
    published = review["body"] + "".join(c["body"] for c in review["comments"])
    assert "considered_dismissed" not in published and "drainage costs" not in published


def test_incremental_review_requires_fresh_final_phase2_gate(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [],
        "out_of_scope_findings": [],
    }
    lanes.verifier["preliminary"] = _verifier([])
    lanes.verifier["final"] = _verifier([])

    first_rid, first_status = _run(cfg, conn, gh, lanes)
    assert first_status == RunStatus.DONE
    first_call_count = len(lanes.calls)

    # Re-open the same reviewed head as a manual/reply-triggered iterative pass.
    with tx(conn):
        assert enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.MANUAL) == "requeued"
    (second_rid,) = schedule(conn, cfg, spawn=False)
    assert (
        worker.main(cfg, conn, second_rid, gh=gh, lane_runner=lanes, heartbeat=False)
        == RunStatus.DONE
    )

    second_steps = {
        row["name"]: row["status"]
        for row in conn.execute("SELECT name, status FROM steps WHERE run_id=?", (second_rid,))
    }
    assert second_steps["fresh_phase2"] == "ok"
    assert second_steps["fresh_verify2"] == "ok"
    fresh_calls = lanes.calls[first_call_count:]
    assert sum(1 for call in fresh_calls if call.role == "verifier") == 3
    fresh_prompts = [
        call.prompt
        for call in fresh_calls
        if call.role == "general" and "fresh final review" in call.prompt
    ]
    assert fresh_prompts
    assert "Prior findings requiring cumulative adjudication" not in fresh_prompts[0]
    assert any(
        call.role == "verifier" and "This is the fresh final gate" in call.prompt
        for call in fresh_calls
    )
    assert (
        conn.execute("SELECT COUNT(*) FROM reviews WHERE run_id=?", (first_rid,)).fetchone()[0] == 1
    )


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
        s.model == "glm-5.3-flash" or s.role in ("verifier", "selector", "triage", "prep")
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
    """With correction turns off the context-free repair lane is the only fix."""
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    lanes.broken_once = {"general"}
    _rid, status = _run(dataclasses.replace(cfg, lane_correction_turns=0), conn, gh, lanes)
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


def _obsolete_run_ends_quietly(conn, gh, rid, status, head_status):
    """A run whose head went stale ends cancelled, not failed: no retry, no `head.failed`
    alert, no failure gate comment; the head ends as ingest would have ended it."""
    assert status == RunStatus.CANCELLED
    run = conn.execute("SELECT status, fail_kind, reason FROM runs WHERE id=?", (rid,)).fetchone()
    assert run["status"] == "cancelled" and run["fail_kind"] is None
    head = conn.execute("SELECT status, attempts FROM heads").fetchone()
    assert head["status"] == head_status and head["attempts"] == 1
    kinds = [r[0] for r in conn.execute("SELECT kind FROM events")]
    assert f"head.{head_status}" in kinds and "run.cancelled" in kinds
    assert "head.failed" not in kinds and "head.requeued" not in kinds and "run.failed" not in kinds
    assert not any("could not complete" in b for b in gh.gate_bodies)
    assert not gh.posted_reviews
    return run["reason"]


@pytest.mark.parametrize("merged", [False, True])
def test_closed_pr_ends_the_head_closed(cfg, conn, gh, lanes, merged):
    gh.pr = {**gh.pr, "state": "closed", "merged": merged}
    rid, status = _run(cfg, conn, gh, lanes)
    reason = _obsolete_run_ends_quietly(conn, gh, rid, status, "closed")
    assert reason == f"head closed: PR is closed{' (merged)' if merged else ''}"
    assert not gh.gate_bodies, "nothing posted for a PR that closed before the run started"
    assert not lanes.calls


def test_closed_pr_replaces_a_stale_queue_comment(cfg, conn, gh, lanes):
    """No newer head will take the comment over: it says the PR closed instead of "queued"."""
    gh.issue_comments = [
        {
            "id": 77,
            "user": {"login": "thepastaclaw"},
            "body": "<!-- thepastaclaw-gate v1 -->\n🕓 Queued",
        }
    ]
    gh.pr = {**gh.pr, "state": "closed", "merged": True}
    rid, status = _run(cfg, conn, gh, lanes)
    _obsolete_run_ends_quietly(conn, gh, rid, status, "closed")
    assert gh.gate_bodies == [
        f"<!-- thepastaclaw-gate v1 -->\n⏹️ Not reviewed — PR is closed (merged) (commit {HEAD[:8]})"
    ]


def test_head_moved_before_the_run_is_superseded_not_failed(cfg, conn, gh, lanes):
    """dash-wallet#1578 run 2879: a push landed between queueing and start."""
    gh.pr = {**gh.pr, "head": {"sha": "c" * 40}}
    rid, status = _run(cfg, conn, gh, lanes)
    reason = _obsolete_run_ends_quietly(conn, gh, rid, status, "superseded")
    assert reason == f"head superseded: live head cccccccc != assigned {HEAD[:8]}"
    assert not gh.gate_bodies, "the newer head's queue comment owns the PR's comment"
    step = conn.execute("SELECT status, detail FROM steps WHERE run_id=?", (rid,)).fetchone()
    assert step["status"] == "cancelled" and "live head" in step["detail"]
    assert not lanes.calls


def test_head_moves_before_publish_and_never_receives_verdict(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [],
        "out_of_scope_findings": [],
    }
    lanes.verifier["preliminary"] = _verifier([])
    lanes.verifier["final"] = _verifier([])

    def racing_lane(spec, artifact_dir, worktree):
        result = lanes(spec, artifact_dir, worktree)
        if spec.role == "verifier" and "must be `final`" in spec.prompt:
            gh.pr = {**gh.pr, "head": {"sha": "c" * 40}}
        return result

    rid, status = _run(cfg, conn, gh, racing_lane)
    reason = _obsolete_run_ends_quietly(conn, gh, rid, status, "superseded")
    assert "live head" in reason
    publish = conn.execute("SELECT status FROM steps WHERE run_id=? AND name='publish'", (rid,))
    assert publish.fetchone()["status"] == "cancelled"


def test_cross_round_dedupe_posts_visible_summary(cfg, conn, gh, lanes):
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
    # every finding was a duplicate, but the review summary is still posted
    assert len(gh.posted_reviews) == 1
    assert "Review provenance" in gh.posted_reviews[0]["body"]
    assert "🟡 1 suggestion(s)" in gh.posted_reviews[0]["body"]
    assert "Prompt for all review comments with AI agents" in gh.posted_reviews[0]["body"]


def test_already_published_for_sha_posts_rereview_summary(cfg, conn, gh, lanes):
    gh.posted_reviews.append(
        {"id": 1, "body": f"<!-- thepastaclaw-review-phase v1 phase=final sha={HEAD} policy=x -->"}
    )
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    rid, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE and len(gh.posted_reviews) == 2
    assert "## Re-review" in gh.posted_reviews[1]["body"]
    assert "Review provenance" in gh.posted_reviews[1]["body"]
    assert "🔴 0 blocking" in gh.posted_reviews[1]["body"]
    assert "Prompt for all review comments with AI agents" in gh.posted_reviews[1]["body"]
    assert (
        '"posted": true'
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
    return [
        s for s in lanes.calls if s.role not in ("selector", "triage", "prep", "verifier", "repair")
    ]


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
    assert gate == {
        "admit_phase2": True,
        "tier": "critical",
        "phase2_effort": "xhigh",
        "blockers": 0,
        "points": None,
        "block_above": None,
    }


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
    assert len(verifier) == 1 and "There is no Phase-1 evidence" in verifier[0].prompt
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
    assert sel.role == "prep" and "security-auditor" in sel.prompt
    assert "always-on" not in sel.prompt, "always-run specialists are repo-specific"
    roles = [s.role for s in _reviewer_calls(lanes)]
    # each phase runs its two lanes in parallel, in either order
    assert sorted(roles[:2]) == sorted(roles[2:]) == ["general", "security-auditor"]
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
        row = conn.execute(
            "SELECT id FROM heads WHERE repo=? AND number=? AND sha=?",
            ("dashpay/platform", 1, "b" * 40),
        ).fetchone()
        hid = (
            row["id"]
            if row
            else conn.execute(
                "INSERT INTO heads (repo, number, sha, trigger, priority, status, queued_at, eligible_at) VALUES (?,?,?,?,?,?,?,?)",
                ("dashpay/platform", 1, "b" * 40, "new_pr", 0, "done", "2026-09-07", "2026-09-07"),
            ).lastrowid
        )
        # one run per seeded head: every prior finding belongs to the same publication,
        # as it would in a real review round
        run_row = conn.execute("SELECT id FROM runs WHERE head_id=?", (hid,)).fetchone()
        rid = (
            run_row["id"]
            if run_row
            else conn.execute(
                "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at) VALUES (?,1,'done','t','2026-09-07T00:00:00Z','2026-09-07T00:00:00Z')",
                (hid,),
            ).lastrowid
        )
        conn.execute(
            "INSERT INTO findings (run_id, phase, stage, hash, file, line_start, line_end, severity, category, title, body) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                rid,
                "final",  # what a real publication records (see _record_publication)
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


def _seed_withdrawn_prior(conn, finding):
    fh = _seed_prior(conn, finding)
    with tx(conn):
        row = conn.execute(
            "SELECT r.id AS run_id FROM runs r JOIN heads h ON h.id=r.head_id "
            "WHERE h.repo=? AND h.number=? AND h.sha=?",
            ("dashpay/platform", 1, "b" * 40),
        ).fetchone()
        f = worker.Finding.from_dict(finding)
        conn.execute(
            "INSERT INTO findings (run_id, phase, stage, hash, file, line_start, line_end, "
            "severity, category, title, body) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                row["run_id"],
                "final",
                "conceded",
                fh,
                f.file,
                f.line_start,
                f.line_end,
                f.severity,
                f.category,
                f.title,
                "WITHDRAWN",
            ),
        )
    return fh


def _queue_new_head(cfg, conn, gh, sha, lanes=None):
    gh.pr = {**gh.pr, "head": {**gh.pr["head"], "sha": sha}}
    if lanes is not None:
        lanes.head = sha
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 1, sha, Trigger.MANUAL)
    (rid,) = schedule(conn, cfg, spawn=False)
    return rid


def test_withdrawn_finding_stays_settled_on_a_later_full_review_head(cfg, conn, gh, lanes):
    """A conversation lift is input to the next head, not forgotten history."""
    finding = _blocking("Old blocker that was withdrawn")
    fh = _seed_withdrawn_prior(conn, finding)
    gh.threads = [_prior_thread(fh, replies=[])]
    lanes.reviewer["default"] = {
        "summary": "The old concern remains settled.",
        "findings": [],
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": [
            {"finding_hash": fh, "status": "WITHDRAWN", "reason": "It remains inapplicable."}
        ],
    }
    lanes.verifier["default"] = _verifier([])
    rid = _queue_new_head(cfg, conn, gh, "c" * 40, lanes)
    assert worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False) == RunStatus.DONE
    reviewer_prompt = next(c.prompt for c in _reviewer_calls(lanes))
    assert '"prior_status": "WITHDRAWN"' in reviewer_prompt
    review = gh.posted_reviews[-1]
    assert review["event"] == "COMMENT" and finding["title"] not in review["body"]
    assert (
        conn.execute(
            "SELECT stage,body FROM findings WHERE run_id=? AND hash=? ORDER BY id DESC LIMIT 1",
            (rid, fh),
        ).fetchone()["body"]
        == "WITHDRAWN"
    )


def test_reintroduced_settled_finding_requires_new_marker_and_gets_new_thread(cfg, conn, gh, lanes):
    finding = _blocking("Old blocker that was withdrawn")
    fh = _seed_withdrawn_prior(conn, finding)
    gh.threads = [_prior_thread(fh, replies=[])]
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
    reintroduced = {**finding, "reintroduced_finding_hash": fh}
    lanes.reviewer["default"] = {
        "summary": "The same defect is independently present again.",
        "findings": [reintroduced],
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": [
            {
                "finding_hash": fh,
                "status": "WITHDRAWN",
                "reason": "The old instance remains settled.",
            }
        ],
    }
    lanes.verifier["default"] = _verifier([reintroduced])
    rid = _queue_new_head(cfg, conn, gh, "d" * 40, lanes)
    assert worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False) == RunStatus.DONE
    assert len(gh.posted_reviews[-1]["comments"]) == 1
    assert gh.posted_reviews[-1]["comments"][0]["body"].find(finding["title"]) >= 0
    assert (
        conn.execute(
            "SELECT stage FROM findings WHERE run_id=? AND hash=? ORDER BY id DESC LIMIT 1",
            (rid, fh),
        ).fetchone()["stage"]
        == "posted"
    )


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
    # the existing inline comment makes the carried finding a cross-round duplicate; the
    # re-review summary remains visible even though no new inline finding is needed.
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
    assert len(gh.posted_reviews) == 1
    assert "Review provenance" in gh.posted_reviews[0]["body"]
    # ...but the human asked a question on the thread, and the answer is the only thing
    # that reaches them, so it is posted there, and the thread stays open
    assert len(gh.replies) == 1
    assert "**Still applies** (re-reviewed at `aaaaaaaa`): The FIFO queue" in gh.replies[0]["body"]
    assert not any("resolveReviewThread" in " ".join(c) for c in gh.calls)


def test_verifier_overruling_a_reviewer_posts_the_verifier_reason(cfg, conn, gh, lanes):
    """The general lane keeps insisting STILL_VALID; the verifier, having read the author's
    'fixed in <sha>' reply against the code, says FIXED with a reason. The thread must get the
    verifier's reason, not the blank 'did not survive verification' (dash#7107 run 811)."""
    s = _sugg()
    fh = _seed_prior(conn, s)
    gh.threads = [
        _prior_thread(
            fh,
            replies=[
                {
                    "id": 901,
                    "author": "PastaPastaPasta",
                    "body": "Fixed in 5885d7ba12: the cache is now keyed by the owning manager.",
                    "created_at": "x",
                    "association": "MEMBER",
                }
            ],
        )
    ]
    carried = {**s, "finding_hash": fh}
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [carried],
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": [
            {
                "finding_hash": fh,
                "status": "STILL_VALID",
                "reason": "A new manager can reuse the address.",
            }
        ],
    }
    lanes.verifier["default"] = {
        **_verifier([]),
        "prior_finding_reconciliation": [
            {
                "finding_hash": fh,
                "status": "FIXED",
                "reason": "5885d7ba12 keys the snapshot cache by manager and tip, so a destroyed chainstate's snapshot cannot be reused.",
            }
        ],
    }
    _rid, status = _run_repo(cfg, conn, gh, lanes, "dashpay/platform", 1, Trigger.REVIEW_REPLY)
    assert status == RunStatus.DONE
    assert len(gh.replies) == 1
    body = gh.replies[0]["body"]
    assert "**Resolved** (re-reviewed at `aaaaaaaa`): 5885d7ba12 keys the snapshot cache" in body
    assert "did not survive verification" not in body
    assert any("resolveReviewThread" in " ".join(c) for c in gh.calls)


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


def test_thread_answer_skips_duplicate_thread_alias(cfg, conn, gh, lanes):
    """A legacy/migrated thread index may expose one root under two finding hashes.
    Only one answer may be posted for that root and reviewed head."""
    from reviewsys.contract import parse_verifier_output
    from reviewsys.publish import answer_replied_threads

    verified = parse_verifier_output(
        {**_verifier([]), "review_phase": "final"},
        expected_phase="final",
        expected_coderabbit_ids=[],
    )
    threads = {
        "first": {
            "comment_id": 900,
            "thread_id": "PRRT_1",
            "latest_reply_id": 901,
        },
        # Same GitHub root, stale alias from the old marker format.
        "stale-alias": {
            "comment_id": 900,
            "thread_id": "PRRT_1",
            "latest_reply_id": 901,
        },
    }
    recon = {
        "first": {"status": "WITHDRAWN", "reason": "Not reproducible."},
        "stale-alias": {"status": "WITHDRAWN", "reason": "Not reproducible."},
    }
    out = answer_replied_threads(
        gh,
        "dashpay/platform",
        1,
        HEAD,
        threads=threads,
        reconciliation=recon,
        verified=verified,
    )
    assert len(gh.replies) == 1
    assert [x["action"] for x in out] == ["replied", "duplicate_thread_skipped"]


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
    assert "@\u200bknst" in reason and len(reason.replace("\u200b", "")) <= 1200


def test_resolved_and_already_answered_threads_are_left_alone(cfg, conn, gh, lanes):
    from reviewsys.github import finding_threads

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

    assert finding_threads([parsed([root, human], resolved=True)], "thepastaclaw") == {}
    quiet = finding_threads([parsed([root, human, bot_answer])], "thepastaclaw")
    assert quiet["abc"]["awaiting_answer"] is False  # known open thread, nobody waiting
    again = {**human, "id": 903, "created_at": "2026-09-08T22:00:00Z"}
    out = finding_threads([parsed([root, human, bot_answer, again])], "thepastaclaw")
    assert out["abc"]["awaiting_answer"] is True
    assert out["abc"]["latest_reply_id"] == 903 and [r["id"] for r in out["abc"]["replies"]] == [
        901,
        903,
    ]


def test_reply_on_reviewed_commit_runs_conversation_lane_not_a_review(cfg, conn, gh, lanes):
    """A reply on a commit we already reviewed is a conversation, not a re-review: no reviewer
    or verifier lane runs, the conversation lane sees the whole thread and any linked commit,
    and its prose is what gets posted."""
    s = _sugg()
    fh = _seed_prior(conn, s)
    earlier_answer = {
        "id": 902,
        "author": "thepastaclaw",
        "body": "<!-- thepastaclaw-thread-answer v1 sha=x reply=901 finding="
        + fh
        + " -->\n**Still applies** (re-reviewed at `x`): The queue is FIFO.",
        "created_at": "2026-09-08T21:00:00Z",
        "association": "NONE",
    }
    gh.threads = [
        _prior_thread(
            fh,
            replies=[
                {
                    "id": 901,
                    "author": "knst",
                    "body": "wrong",
                    "created_at": "2026-09-08T20:00:00Z",
                    "association": "COLLABORATOR",
                },
                earlier_answer,
                {
                    "id": 903,
                    "author": "UdjinM6",
                    "body": "How about https://github.com/UdjinM6/platform/commit/7c19184c849c382d6c72d31baeced0bd4631c616 ?",
                    "created_at": "2026-09-08T22:00:00Z",
                    "association": "NONE",
                },
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
    lanes.conversation = {
        "threads": [
            {
                "finding_hash": fh,
                "status": "FIXED",
                "reply": "Yes, 7c19184c does it: pausing the worker under the lock makes the batch deterministic, which was the whole concern. Thanks @UdjinM6.",
                "reasoning": "the linked commit adds the synchronisation point",
            }
        ]
    }
    with tx(conn):
        conn.execute("UPDATE heads SET sha=?, status='done'", (HEAD,))
        assert (
            enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.REVIEW_REPLY) == "requeued"
        )
    (rid,) = schedule(conn, cfg, spawn=False)
    status = worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False)
    assert status == RunStatus.DONE
    roles = [c.role for c in lanes.calls]
    assert roles == ["conversation"], roles  # no selector, triage, reviewer or verifier lanes
    prompt = lanes.calls[0].prompt
    # the whole exchange, our own earlier answer included, and the linked commit
    assert "you (PastaClaw)" in prompt and "The queue is FIFO." in prompt
    assert "7c19184c849c382d6c72d31baeced0bd4631c616" in prompt and "git show" in prompt
    assert len(gh.posted_reviews) == 1, "same commit: no second review"
    assert len(gh.replies) == 1
    body = gh.replies[0]["body"]
    assert body.startswith(
        f"<!-- thepastaclaw-thread-answer v1 sha={HEAD} reply=903 finding={fh} -->"
    )
    # the linked commit is not on the head: a pending fix, not a resolved finding
    assert "Yes, 7c19184c does it" in body and "until that change is pushed" in body
    assert (
        "**Still applies**" not in body and "**Resolved**" not in body
    )  # prose, not a verdict log
    assert "@\u200bUdjinM6" in body  # mentions defused
    assert not any("resolveReviewThread" in " ".join(c) for c in gh.calls)
    ev = conn.execute("SELECT detail FROM events WHERE kind='thread.answered'").fetchone()
    assert '"status": "FIX_PENDING"' in ev["detail"] and '"mode": "conversation"' in ev["detail"]
    steps = [
        r["name"]
        for r in conn.execute("SELECT name FROM steps WHERE run_id=? ORDER BY rowid", (rid,))
    ]
    assert steps == ["worktree", "context", "converse"]
    assert gh.gate_bodies[-1].splitlines()[1].startswith("✅ Final review complete")
    assert (Path(cfg.runs_dir) / f"run-{rid}" / "conversation.json").exists()


def test_conversation_stays_silent_when_it_has_nothing_to_add(cfg, conn, gh, lanes):
    s = _sugg()
    fh = _seed_prior(conn, s)
    gh.threads = [
        _prior_thread(
            fh,
            replies=[
                {
                    "id": 901,
                    "author": "PastaPastaPasta",
                    "body": "@UdjinM6 can you break the tie?",
                    "created_at": "x",
                    "association": "MEMBER",
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
    lanes.conversation = {
        "threads": [
            {
                "finding_hash": fh,
                "status": "NO_REPLY",
                "reply": "",
                "reasoning": "asked someone else",
            }
        ]
    }
    with tx(conn):
        conn.execute("UPDATE heads SET sha=?, status='done'", (HEAD,))
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.REVIEW_REPLY)
    (rid,) = schedule(conn, cfg, spawn=False)
    assert worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False) == RunStatus.DONE
    assert gh.replies == [] and len(gh.posted_reviews) == 1
    ev = conn.execute("SELECT detail FROM events WHERE kind='thread.answered'").fetchone()
    assert '"action": "no_reply"' in ev["detail"]
    # silence is remembered per (finding, reply): a later conversation on this PR does not
    # re-ask about the same human message, so no late second opinion can ever post
    assert kv_get(conn, f"converse.silent:dashpay/platform#1:{fh}") == "901"
    lanes.conversation = {
        "threads": [{"finding_hash": fh, "status": "STILL_VALID", "reply": "late"}]
    }
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.REVIEW_REPLY)
    (rid2,) = schedule(conn, cfg, spawn=False)
    assert worker.main(cfg, conn, rid2, gh=gh, lane_runner=lanes, heartbeat=False) == RunStatus.DONE
    assert gh.replies == [] and [c.role for c in lanes.calls] == ["conversation"]
    # ...until someone speaks again on that thread
    gh.threads[0]["comments"]["nodes"].append(
        _gql_comment(
            {
                "id": 905,
                "author": "UdjinM6",
                "body": "here is my take",
                "created_at": "y",
                "association": "NONE",
            }
        )
    )
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.REVIEW_REPLY)
    (rid3,) = schedule(conn, cfg, spawn=False)
    assert worker.main(cfg, conn, rid3, gh=gh, lane_runner=lanes, heartbeat=False) == RunStatus.DONE
    assert len(gh.replies) == 1 and "reply=905" in gh.replies[0]["body"]


def test_conversation_conceding_every_blocker_lifts_request_changes(cfg, conn, gh, lanes):
    blocker = _blocking()
    bh = _seed_prior(conn, blocker, body="blocker body")
    human = {
        "id": 901,
        "author": "knst",
        "body": "this is latency, not correctness; see the bench in the description",
        "created_at": "x",
        "association": "COLLABORATOR",
    }
    t = _prior_thread(bh, replies=[human])
    t["comments"]["nodes"][0]["body"] = (
        f"<!-- thepastaclaw-review v1 finding={bh} dedupe=x -->\n**🔴 Blocking: {blocker['title']}**\n\nblocker body"
    )
    gh.threads = [t]
    gh.posted_reviews.append(
        {
            "id": 4242,
            "event": "REQUEST_CHANGES",
            "html_url": "https://gh/r/4242",
            "body": f"<!-- thepastaclaw-review-phase v1 phase=final sha={HEAD} policy=x -->",
        }
    )
    lanes.conversation = {
        "threads": [
            {
                "finding_hash": bh,
                "status": "WITHDRAWN",
                "reply": "You are right, I misread the bench: the estimate is an upper bound, so this is latency only. Withdrawing.",
            }
        ]
    }
    with tx(conn):
        conn.execute("UPDATE heads SET sha=?, status='done'", (HEAD,))
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.REVIEW_REPLY)
    (rid,) = schedule(conn, cfg, spawn=False)
    assert worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False) == RunStatus.DONE
    assert "_Withdrawing this finding._" in gh.replies[0]["body"]
    assert len(gh.posted_reviews) == 2
    upd = gh.posted_reviews[1]
    assert upd["event"] == "COMMENT" and upd["comments"] == []
    assert "Standing review was `CHANGES_REQUESTED`; this re-review is `COMMENT`" in upd["body"]
    assert f"- {blocker['title']}" in upd["body"]
    assert "conversation lane" in upd["body"] and "the code was not re-reviewed" in upd["body"]
    # the seeded publication has no verifier record: no approval can rest on it
    assert "Not approved: the verifier record an approval after discussion rests on" in upd["body"]
    ev = conn.execute("SELECT detail FROM events WHERE kind='review.verdict_updated'").fetchone()
    assert "CHANGES_REQUESTED -> COMMENT" in ev["detail"] and "conversation" in ev["detail"]
    assert gh.gate_bodies[-1].splitlines()[1].startswith("✅ Final review complete — no blockers")


def test_conversation_accepting_a_deferral_lifts_request_changes(cfg, conn, gh, lanes):
    """dashpay/dash#7769: the author showed the blocker is a pre-existing race outside the PR's
    narrower guarantee, and the lane answered "I will not press it further". Keeping
    REQUEST_CHANGES after that contradicts our own reply, and a full re-review would already
    drop a deferred finding from the kept set."""
    blocker = _blocking()
    bh = _seed_prior(conn, blocker, body="blocker body")
    human = {
        "id": 901,
        "author": "knst",
        "body": "the race predates this PR; draining the queue here would stall mainnet. "
        "Tracked in dashpay/platform#4321.",
        "created_at": "x",
        "association": "COLLABORATOR",
    }
    t = _prior_thread(bh, replies=[human])
    t["comments"]["nodes"][0]["body"] = (
        f"<!-- thepastaclaw-review v1 finding={bh} dedupe=x -->\n**🔴 Blocking: {blocker['title']}**\n\nblocker body"
    )
    gh.threads = [t]
    gh.posted_reviews.append(
        {
            "id": 4242,
            "event": "REQUEST_CHANGES",
            "html_url": "https://gh/r/4242",
            "body": f"<!-- thepastaclaw-review-phase v1 phase=final sha={HEAD} policy=x -->",
        }
    )
    lanes.conversation = {
        "threads": [
            {
                "finding_hash": bh,
                "status": "INTENTIONALLY_DEFERRED",
                "reply": "Agreed, the race predates this PR; leaving it for a follow-up.",
                "tracking": {"url": "dashpay/platform#4321"},
            }
        ]
    }
    with tx(conn):
        conn.execute("UPDATE heads SET sha=?, status='done'", (HEAD,))
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.REVIEW_REPLY)
    (rid,) = schedule(conn, cfg, spawn=False)
    assert worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False) == RunStatus.DONE
    assert gh.replies[0]["body"].endswith(
        "_Tracked in https://github.com/dashpay/platform/issues/4321. Deferred on a "
        "maintainer's call: this finding no longer blocks this pull request._"
    )
    # the thread stays open as the record of the deferral; only the verdict moves
    assert not any("resolveReviewThread" in " ".join(c) for c in gh.calls)
    assert len(gh.posted_reviews) == 2
    upd = gh.posted_reviews[1]
    assert upd["event"] == "COMMENT" and f"- {blocker['title']}" in upd["body"]
    assert gh.gate_bodies[-1].splitlines()[1].startswith("✅ Final review complete — no blockers")


def test_conversation_deferral_by_an_outside_contributor_keeps_the_blocker(cfg, conn, gh, lanes):
    """A contributor declining to fix a blocker is not a maintainer decision to defer it: the
    reply is posted, but REQUEST_CHANGES stands."""
    blocker = _blocking()
    bh = _seed_prior(conn, blocker, body="blocker body")
    human = {
        "id": 901,
        "author": "drive-by",
        "body": "out of scope, will do it in a follow-up",
        "created_at": "x",
        "association": "CONTRIBUTOR",
    }
    t = _prior_thread(bh, replies=[human])
    t["comments"]["nodes"][0]["body"] = (
        f"<!-- thepastaclaw-review v1 finding={bh} dedupe=x -->\n**🔴 Blocking: {blocker['title']}**\n\nblocker body"
    )
    gh.threads = [t]
    gh.posted_reviews.append(
        {
            "id": 4242,
            "event": "REQUEST_CHANGES",
            "html_url": "https://gh/r/4242",
            "body": f"<!-- thepastaclaw-review-phase v1 phase=final sha={HEAD} policy=x -->",
        }
    )
    lanes.conversation = {
        "threads": [{"finding_hash": bh, "status": "INTENTIONALLY_DEFERRED", "reply": "ok"}]
    }
    with tx(conn):
        conn.execute("UPDATE heads SET sha=?, status='done'", (HEAD,))
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.REVIEW_REPLY)
    (rid,) = schedule(conn, cfg, spawn=False)
    assert worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False) == RunStatus.DONE
    assert len(gh.replies) == 1 and len(gh.posted_reviews) == 1
    assert gh.replies[0]["body"].endswith("it still blocks this pull request._")
    assert not conn.execute("SELECT 1 FROM findings WHERE stage='conceded'").fetchone()
    assert gh.gate_bodies[-1].splitlines()[1].startswith("⛔ Final review complete — 1 blocking")


def _standing_body(*titles):
    fence = "```"
    items = "\n".join(
        f"- [BLOCKING] f.rs:11-12: {t}\n  Estimation must be >= actual." for t in titles
    )
    return (
        "<!-- thepastaclaw-review v1 -->\n"
        f"<!-- thepastaclaw-review-phase v1 phase=final sha={HEAD} policy=x -->\n"
        "## Re-review — Final validation — Phase 1 + Phase 2\n\n"
        "Two synchronization gaps remain.\n\n"
        f"🔴 {len(titles)} blocking\n\n"
        "<details>\n<summary>🤖 Prompt for all review comments with AI agents</summary>\n\n"
        f"{fence}\nThese findings are from an automated code review.\n\nIn `f.rs`:\n{items}\n{fence}\n"
        "</details>"
    )


def test_deferring_one_of_two_blockers_corrects_the_standing_review(cfg, conn, gh, lanes):
    """dashpay/dash#7892: a maintainer deferred one of two blockers. REQUEST_CHANGES rightly
    stayed for the other, so no follow-up review was posted, and the standing review kept
    saying "2 blocking" with the deferred finding still [BLOCKING] in its agent prompt. The
    maintainer pointed that out twice and got two more acknowledgements. The bot edits its own
    review body instead; a later run with nothing new leaves it alone."""
    b1, b2 = _blocking("Synchronize the second session"), _blocking("Synchronize secondary types")
    h1 = _seed_prior(conn, b1, body="one")
    h2 = _seed_prior(conn, b2, body="two")
    gh.can_resolve = False
    gh.threads = [
        _finding_thread(h1, b1["title"], 800, [KNST]),
        _finding_thread(h2, b2["title"], 810, []),
    ]
    gh.posted_reviews.append(
        {
            "id": 4242,
            "event": "REQUEST_CHANGES",
            "body": _standing_body(b1["title"], b2["title"]),
        }
    )
    gh.pr["body"] = "B\n\nThe second session's sync is tracked in #4321."
    defer = {
        "status": "INTENTIONALLY_DEFERRED",
        "reply": "Fine as a follow-up.",
        "tracking": {"url": "#4321"},
    }
    tracked = "https://github.com/dashpay/platform/issues/4321"
    rid = _converse(cfg, conn, gh, lanes, [{"finding_hash": h1, **defer}])
    assert len(gh.posted_reviews) == 1  # the other blocker still stands: no verdict update
    assert [e[0] for e in gh.review_edits] == [4242]
    body = gh.posted_reviews[0]["body"]
    assert "🔴 ~~2~~ 1 blocking" in body
    assert (
        "**Updated after discussion:** 1 finding(s)" in body
        and "1 blocking finding(s) still stand" in body
    )
    assert f"> - ⏭️ Deferred (maintainer decision): 🔴 {b1['title']} (tracked in {tracked})" in body
    assert (
        f"- [BLOCKING → DEFERRED by a maintainer; not blocking, do not fix in this PR] f.rs:11-12: {b1['title']}"
        in body
    )
    assert f"- [BLOCKING] f.rs:11-12: {b2['title']}" in body
    # markers and heading stay where every reader of the body looks for them
    assert body.startswith("<!-- thepastaclaw-review v1 -->\n<!-- thepastaclaw-review-phase")
    assert body.split("\n")[2].startswith("## Re-review")
    detail = json.loads(
        conn.execute(
            "SELECT detail FROM steps WHERE run_id=? AND name='converse'", (rid,)
        ).fetchone()["detail"]
    )
    assert detail["review_summary"] == "updated" and detail["blockers_remaining"] == 1
    assert gh.gate_bodies[-1].splitlines()[1].startswith("⛔ Final review complete — 1 blocking")

    # the maintainer complains the summary is stale; the lane has the state and stays quiet
    # this time. The body is already right: no second edit.
    complaint = {**KNST, "id": 902, "body": "IT IS STILL HERE"}
    gh.threads[0] = _finding_thread(h1, b1["title"], 800, [KNST, complaint])
    _converse(
        cfg, conn, gh, lanes, [{"finding_hash": h1, "status": "NO_REPLY", "reply": ""}], first=False
    )
    assert len(gh.review_edits) == 1
    prompt = lanes.calls[-1].prompt
    standing = prompt.split("## Where your review stands")[1].split("\n## ")[0]
    assert f"- {b2['title']}" in standing and b1["title"] not in standing
    assert '"already_settled_as": "INTENTIONALLY_DEFERRED"' in prompt

    # the PR author (not a maintainer) re-asks; the lane restates the deferral. That is not
    # a new deferral to vet: no "still blocks" note, and the root keeps its Deferred line
    ours = {
        "id": 950,
        "author": "thepastaclaw",
        "body": f"<!-- thepastaclaw-thread-answer v1 sha={HEAD} reply=902 finding={h1} -->\nok",
        "created_at": "y",
        "association": "NONE",
    }
    author = {**KNST, "id": 904, "association": "CONTRIBUTOR", "body": "still says blocking"}
    gh.threads[0] = _finding_thread(h1, b1["title"], 800, [KNST, complaint, ours, author])
    root = gh.threads[0]["comments"]["nodes"][0]
    root["body"] = publish.with_thread_status(
        root["body"], status="INTENTIONALLY_DEFERRED", head_sha=HEAD, tracking_url=tracked
    )
    edits = len(gh.comment_edits)
    _converse(cfg, conn, gh, lanes, [{"finding_hash": h1, **defer}], first=False)
    assert gh.replies[-1]["body"].endswith("no longer blocks this pull request._")
    assert "⏭️ **Deferred**" in root["body"] and len(gh.comment_edits) == edits
    # even an answer that would not lift it leaves the lift (and the root's line) alone
    author2 = {**author, "id": 905, "body": "so is it blocking or not?"}
    gh.threads[0] = _finding_thread(h1, b1["title"], 800, [KNST, complaint, ours, author2])
    gh.threads[0]["comments"]["nodes"][0] = root
    still = {"status": "STILL_VALID", "reply": "The concern is real, but it is deferred."}
    _converse(cfg, conn, gh, lanes, [{"finding_hash": h1, **still}], first=False)
    assert "⏭️ **Deferred**" in root["body"] and len(gh.comment_edits) == edits
    assert "🔴 ~~2~~ 1 blocking" in gh.posted_reviews[0]["body"]

    # withdrawing the second blocker clears the verdict and the summary counts both
    gh.threads[1] = _finding_thread(h2, b2["title"], 810, [{**KNST, "id": 903}])
    _converse(cfg, conn, gh, lanes, [{"finding_hash": h2, **WITHDRAW}], first=False)
    body = gh.posted_reviews[0]["body"]
    assert "🔴 ~~2~~ 0 blocking" in body and "no blocking finding remains" in body
    assert f"> - ✅ Withdrawn: 🔴 {b2['title']}" in body
    assert body.count("thepastaclaw-review-status v1") == 1
    assert len(gh.posted_reviews) == 2 and gh.posted_reviews[1]["event"] == "COMMENT"


def test_conversation_concession_is_remembered_across_runs(cfg, conn, gh, lanes):
    """Two blockers, conceded in two separate conversations: the second run must see the first
    concession (a `conceded` findings row) or the verdict could never be lifted."""
    b1, b2 = _blocking("First blocker"), _blocking("Second blocker")
    h1 = _seed_prior(conn, b1, body="one")
    h2 = _seed_prior(conn, b2, body="two")
    human = {
        "id": 901,
        "author": "knst",
        "body": "no",
        "created_at": "x",
        "association": "COLLABORATOR",
    }

    def thread(h, title, root_id, replies):
        t = _prior_thread(h, replies=replies)
        t["id"] = f"PRRT_{root_id}"
        t["comments"]["nodes"][0]["databaseId"] = root_id
        t["comments"]["nodes"][0]["body"] = (
            f"<!-- thepastaclaw-review v1 finding={h} dedupe=x -->\n**🔴 Blocking: {title}**\n\nbody"
        )
        return t

    gh.threads = [thread(h1, "First blocker", 800, [human]), thread(h2, "Second blocker", 810, [])]
    gh.posted_reviews.append(
        {
            "id": 4242,
            "event": "REQUEST_CHANGES",
            "html_url": "https://gh/r/4242",
            "body": f"<!-- thepastaclaw-review-phase v1 phase=final sha={HEAD} policy=x -->",
        }
    )
    lanes.conversation = {
        "threads": [{"finding_hash": h1, "status": "WITHDRAWN", "reply": "fair, withdrawn"}]
    }
    with tx(conn):
        conn.execute("UPDATE heads SET sha=?, status='done'", (HEAD,))
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.REVIEW_REPLY)
    (rid,) = schedule(conn, cfg, spawn=False)
    assert worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False) == RunStatus.DONE
    assert len(gh.posted_reviews) == 1  # one blocker still stands
    assert gh.gate_bodies[-1].splitlines()[1].startswith("⛔ Final review complete — 1 blocking")
    assert conn.execute(
        "SELECT stage FROM findings WHERE hash=? AND stage='conceded'", (h1,)
    ).fetchone()
    # the first thread is now resolved on GitHub (gone from open threads); the second gets a reply
    gh.threads = [thread(h2, "Second blocker", 810, [{**human, "id": 902}])]
    lanes.conversation = {
        "threads": [
            {"finding_hash": h2, "status": "WITHDRAWN", "reply": "right, the guard covers it"}
        ]
    }
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.REVIEW_REPLY)
    (rid2,) = schedule(conn, cfg, spawn=False)
    assert worker.main(cfg, conn, rid2, gh=gh, lane_runner=lanes, heartbeat=False) == RunStatus.DONE
    assert len(gh.posted_reviews) == 2
    upd = gh.posted_reviews[1]
    assert (
        upd["event"] == "COMMENT"
        and "- Second blocker" in upd["body"]
        and "- First blocker" not in upd["body"]
    )
    assert gh.gate_bodies[-1].splitlines()[1].startswith("✅ Final review complete — no blockers")


def test_conversation_never_lifts_a_verdict_on_a_hand_resolved_blocker(cfg, conn, gh, lanes):
    """A maintainer resolved the blocker thread themselves; the bot never conceded it. A reply
    on some other thread must not turn that into a lifted REQUEST_CHANGES."""
    blocker, sugg = _blocking(), _sugg()
    _seed_prior(conn, blocker, body="blocker body")
    sh = _seed_prior(conn, sugg)
    human = {
        "id": 901,
        "author": "knst",
        "body": "?",
        "created_at": "x",
        "association": "COLLABORATOR",
    }
    gh.threads = [_prior_thread(sh, replies=[human])]  # blocker thread resolved: not listed
    gh.posted_reviews.append(
        {
            "id": 4242,
            "event": "REQUEST_CHANGES",
            "html_url": "https://gh/r/4242",
            "body": f"<!-- thepastaclaw-review-phase v1 phase=final sha={HEAD} policy=x -->",
        }
    )
    lanes.conversation = {
        "threads": [{"finding_hash": sh, "status": "WITHDRAWN", "reply": "ok, dropping this"}]
    }
    with tx(conn):
        conn.execute("UPDATE heads SET sha=?, status='done'", (HEAD,))
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.REVIEW_REPLY)
    (rid,) = schedule(conn, cfg, spawn=False)
    assert worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False) == RunStatus.DONE
    assert len(gh.posted_reviews) == 1
    assert gh.gate_bodies[-1].splitlines()[1].startswith("⛔ Final review complete — 1 blocking")


def _full_strength_prior(conn, *, extra_verified=()):
    """Make the seeded prior publication a full-strength final review: Phase 2 ran on the
    primary models and its final verifier set (`verify2` rows: everything posted, plus
    `extra_verified` findings dedupe carried onto other threads) is on record."""
    from reviewsys.contract import Finding

    with tx(conn):
        (rid,) = conn.execute(
            "SELECT DISTINCT run_id FROM findings WHERE stage='posted'"
        ).fetchone()
        conn.execute(
            "INSERT INTO steps (run_id, name, status, started_at) VALUES (?, 'phase2', 'ok', 'x')",
            (rid,),
        )
        conn.execute(
            "INSERT INTO findings (run_id, phase, stage, hash, file, line_start, line_end, "
            "severity, confidence, category, title, body) SELECT run_id, 'verify2', 'verified', "
            "hash, file, line_start, line_end, severity, confidence, category, title, body "
            "FROM findings WHERE run_id=? AND stage='posted'",
            (rid,),
        )
        for raw in extra_verified:
            f = Finding.from_dict(raw)
            conn.execute(
                worker._FINDINGS_INSERT,
                (
                    rid,
                    "verify2",
                    "verified",
                    f.hash,
                    f.file,
                    f.line_start,
                    f.line_end,
                    f.severity,
                    f.confidence,
                    f.category,
                    f.title,
                    f.body,
                ),
            )
    return rid


def _finding_thread(h, title, root_id, replies, *, label="🔴 Blocking"):
    t = _prior_thread(h, replies=replies)
    t["id"] = f"PRRT_{root_id}"
    t["comments"]["nodes"][0]["databaseId"] = root_id
    t["comments"]["nodes"][0]["body"] = (
        f"<!-- thepastaclaw-review v1 finding={h} dedupe=x -->\n**{label}: {title}**\n\nbody"
    )
    return t


def _standing_final(gh):
    gh.posted_reviews.append(
        {
            "id": 4242,
            "event": "REQUEST_CHANGES",
            "html_url": "https://gh/r/4242",
            "body": f"<!-- thepastaclaw-review-phase v1 phase=final sha={HEAD} policy=x -->",
        }
    )


def _converse(cfg, conn, gh, lanes, threads, *, first=True):
    lanes.conversation = {"threads": threads}
    with tx(conn):
        if first:
            conn.execute("UPDATE heads SET sha=?, status='done'", (HEAD,))
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.REVIEW_REPLY)
    (rid,) = schedule(conn, cfg, spawn=False)
    assert worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False) == RunStatus.DONE
    return rid


KNST = {
    "id": 901,
    "author": "knst",
    "body": "can't happen",
    "created_at": "x",
    "association": "COLLABORATOR",
}
WITHDRAW = {"status": "WITHDRAWN", "reply": "Right: allowReSign is false."}


def _updates(conn):
    return [
        r["detail"]
        for r in conn.execute(
            "SELECT detail FROM events WHERE kind='review.verdict_updated' ORDER BY id"
        )
    ]


def test_conversation_withdrawing_the_only_finding_approves(cfg, conn, gh, lanes):
    """dashpay/dash#7778: the full review's only finding was a blocker the author argued away.
    A COMMENT cannot clear REQUEST_CHANGES on GitHub, so the follow-up must APPROVE; and since
    the bot cannot resolve threads there, its own finding comment says the finding is withdrawn.
    A leftover nitpick does not hold an approval back."""
    blocker = _blocking()
    bh = _seed_prior(conn, blocker, body="blocker body")
    _seed_prior(conn, {**_sugg("Tidy the helper"), "severity": "nitpick"})
    posting_run = _full_strength_prior(conn)
    gh.can_resolve = False
    gh.threads = [_finding_thread(bh, blocker["title"], 800, [KNST])]
    _standing_final(gh)
    _converse(cfg, conn, gh, lanes, [{"finding_hash": bh, **WITHDRAW}])
    assert len(gh.posted_reviews) == 2
    upd = gh.posted_reviews[1]
    assert upd["event"] == "APPROVE" and upd["comments"] == []
    assert "Standing review was `CHANGES_REQUESTED`; this re-review is `APPROVE`" in upd["body"]
    assert f"the full review of `{HEAD[:8]}` (run {posting_run})" in upd["body"]
    assert "the code was not re-reviewed" in upd["body"] and "suggestion" not in upd["body"]
    assert "CHANGES_REQUESTED -> APPROVE" in _updates(conn)[0]
    ((cid, body),) = gh.comment_edits
    assert cid == 800 and f"> ✅ **Withdrawn** at `{HEAD[:8]}`" in body
    assert body.splitlines()[0].startswith(f"<!-- thepastaclaw-review v1 finding={bh}")
    assert gh.gate_bodies[-1].splitlines()[1].startswith("✅ Final review complete — no blockers")


def test_conversation_approves_only_once_no_suggestion_is_left(cfg, conn, gh, lanes):
    """Blocker withdrawn but a suggestion stands: COMMENT, saying why it is not an approval.
    Withdrawing the suggestion in a later conversation then approves from that COMMENT."""
    blocker, sugg = _blocking(), _sugg()
    bh = _seed_prior(conn, blocker, body="blocker body")
    sh = _seed_prior(conn, sugg)
    _full_strength_prior(conn)
    gh.threads = [
        _finding_thread(bh, blocker["title"], 800, [KNST]),
        _finding_thread(sh, sugg["title"], 810, [], label="🟡 Suggestion"),
    ]
    _standing_final(gh)
    _converse(cfg, conn, gh, lanes, [{"finding_hash": bh, **WITHDRAW}])
    assert [r["event"] for r in gh.posted_reviews[1:]] == ["COMMENT"]
    assert "Not approved: 1 non-blocking finding(s)" in gh.posted_reviews[1]["body"]
    # the blocker thread is resolved now (gone); the suggestion gets a reply and is withdrawn
    gh.threads = [
        _finding_thread(sh, sugg["title"], 810, [{**KNST, "id": 902}], label="🟡 Suggestion")
    ]
    _converse(cfg, conn, gh, lanes, [{"finding_hash": sh, **WITHDRAW}], first=False)
    assert [r["event"] for r in gh.posted_reviews[1:]] == ["COMMENT", "APPROVE"]
    assert "COMMENTED -> APPROVE" in _updates(conn)[-1]


def _carried_publication(conn, gh, blocker, *, earlier_round):
    """The prior run published review 4242 on HEAD but carried its findings onto existing
    threads: `verify2` rows, a `reviews` row and a publish step, no `posted` rows. With
    `earlier_round`, an older head's review (where the threads came from) precedes it."""
    rows = []
    if earlier_round:
        with tx(conn):
            hid = conn.execute(
                "INSERT INTO heads (repo, number, sha, trigger, priority, status, queued_at, "
                "eligible_at) VALUES ('dashpay/platform', 1, ?, 'new_pr', 0, 'superseded', "
                "'x', 'x')",
                ("d" * 40,),
            ).lastrowid
            old = conn.execute(
                "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at) "
                "VALUES (?, 1, 'done', 't', 'x', 'x')",
                (hid,),
            ).lastrowid
        rows.append((old, "d" * 40, 4100))
    bh = _seed_prior(conn, blocker, body="blocker body")
    rid = _full_strength_prior(conn)
    rows.append((rid, HEAD, 4242))
    with tx(conn):
        conn.execute("DELETE FROM findings WHERE stage='posted'")
        conn.execute(
            "INSERT INTO steps (run_id, name, status, started_at) VALUES (?, 'publish', 'ok', 'x')",
            (rid,),
        )
        for run, sha, review in rows:
            conn.execute(
                "INSERT INTO reviews (run_id, repo, number, sha, phase, github_review_id, "
                "event, posted_at) VALUES (?, 'dashpay/platform', 1, ?, 'final', ?, "
                "'REQUEST_CHANGES', 'x')",
                (run, sha, review),
            )
    gh.can_resolve = False
    gh.threads = [_finding_thread(bh, blocker["title"], 800, [KNST])]
    gh.posted_reviews.append(
        {"id": 4242, "event": "REQUEST_CHANGES", "body": _standing_body(blocker["title"])}
    )
    return bh


def test_a_publication_that_carried_every_finding_is_still_the_publication(cfg, conn, gh, lanes):
    """dashpay/dash#7861: the review of this commit carried all its findings onto threads, so
    it posted none and wrote no `posted` rows. Withdrawing its only blocker then found "no
    verifier record" and posted a COMMENT, which leaves the REQUEST_CHANGES it had just posted
    standing. Its `reviews` row makes it the publication: on a first round that approves."""
    blocker = _blocking()
    bh = _carried_publication(conn, gh, blocker, earlier_round=False)
    _converse(cfg, conn, gh, lanes, [{"finding_hash": bh, **WITHDRAW}])
    assert [r["event"] for r in gh.posted_reviews] == ["REQUEST_CHANGES", "APPROVE"]
    assert "verifier record" not in gh.posted_reviews[1]["body"]
    assert "🔴 ~~1~~ 0 blocking" in gh.posted_reviews[0]["body"]


def test_a_carried_publication_on_a_later_round_queues_the_fresh_audit(cfg, conn, gh, lanes):
    """#7861 as it really was: the threads came from an earlier head's review, so this is a
    later round whose review skipped the fresh audit (it had a blocker). The follow-up says
    so and queues the audit, which approves if it finds nothing."""
    blocker = _blocking()
    bh = _carried_publication(conn, gh, blocker, earlier_round=True)
    with tx(conn):  # only the seeded head becomes the reviewed commit; the older one stays
        conn.execute("UPDATE heads SET sha=?, status='done' WHERE sha=?", (HEAD, "b" * 40))
    rid = _converse(cfg, conn, gh, lanes, [{"finding_hash": bh, **WITHDRAW}], first=False)
    assert [r["event"] for r in gh.posted_reviews] == ["REQUEST_CHANGES", "COMMENT"]
    body = gh.posted_reviews[1]["body"]
    assert "verifier record" not in body and "That audit is queued" in body
    assert conn.execute(
        "SELECT 1 FROM events WHERE kind='review.fresh_audit_requested' AND run_id=?", (rid,)
    ).fetchone()


def test_a_finding_carried_onto_a_resolved_thread_still_blocks(cfg, conn, gh, lanes):
    """The verifier kept a blocker that dedupe carried onto a thread a maintainer had resolved
    by hand, so it was never posted for this commit and has no open thread. Withdrawing the
    one finding that was posted must not approve, or lift anything, over it."""
    nit = {**_sugg("Rename the helper"), "severity": "suggestion"}
    nh = _seed_prior(conn, nit)
    _full_strength_prior(conn, extra_verified=[_blocking("Carried blocker")])
    gh.threads = [_finding_thread(nh, nit["title"], 810, [KNST], label="🟡 Suggestion")]
    _standing_final(gh)
    _converse(cfg, conn, gh, lanes, [{"finding_hash": nh, **WITHDRAW}])
    assert len(gh.posted_reviews) == 1 and _updates(conn) == []
    assert gh.gate_bodies[-1].splitlines()[1].startswith("⛔ Final review complete — 1 blocking")


def test_an_open_thread_from_an_earlier_head_does_not_hold_back_approval(cfg, conn, gh, lanes):
    """dashpay/dash: the bot cannot resolve threads, so a suggestion fixed on an earlier head
    keeps an open thread forever. The review of this commit re-adjudicated it (not in its
    verifier set), so it must not count as a finding that still stands."""
    blocker = _blocking()
    bh = _seed_prior(conn, blocker, body="blocker body")
    _full_strength_prior(conn)
    old = _sugg("Old suggestion fixed on an earlier head")
    from reviewsys.contract import Finding

    oh = Finding.from_dict(old).hash
    with tx(conn):
        conn.execute(
            "INSERT INTO posted_findings (repo, number, hash, sha, review_id, posted_at) VALUES (?,?,?,?,?,?)",
            ("dashpay/platform", 1, oh, "c" * 40, 1, "2026-09-01T00:00:00Z"),
        )
    gh.threads = [
        _finding_thread(bh, blocker["title"], 800, [KNST]),
        _finding_thread(oh, old["title"], 820, [], label="🟡 Suggestion"),
    ]
    _standing_final(gh)
    _converse(cfg, conn, gh, lanes, [{"finding_hash": bh, **WITHDRAW}])
    assert [r["event"] for r in gh.posted_reviews[1:]] == ["APPROVE"]


def test_a_later_round_without_the_fresh_audit_queues_it_instead_of_approving(cfg, conn, gh, lanes):
    """On a later review round the pipeline approves only after a fresh Phase-2 audit, which it
    skips while a blocker stands. Withdrawing that blocker must not approve around the audit:
    COMMENT now, and the commit is re-queued once (after this run ends) so the audit runs."""
    blocker = _blocking()
    bh = _seed_prior(conn, blocker, body="blocker body")
    posting_run = _full_strength_prior(conn)
    with tx(conn):  # an earlier round: a review posted by an older run
        hid = conn.execute("SELECT head_id FROM runs WHERE id=?", (posting_run,)).fetchone()[0]
        older = conn.execute(
            "INSERT INTO runs (id, head_id, attempt, status, token, started_at, deadline_at) VALUES (?,?,1,'done','t','x','x')",
            (posting_run - 1 if posting_run > 1 else 0, hid),
        ).lastrowid
        conn.execute(
            "INSERT INTO reviews (run_id, repo, number, sha, phase, github_review_id, event, posted_at) VALUES (?,?,?,?,?,?,?,?)",
            (older, "dashpay/platform", 1, "c" * 40, "final", 1, "REQUEST_CHANGES", "x"),
        )
    gh.threads = [_finding_thread(bh, blocker["title"], 800, [KNST])]
    _standing_final(gh)
    rid = _converse(cfg, conn, gh, lanes, [{"finding_hash": bh, **WITHDRAW}])
    assert [r["event"] for r in gh.posted_reviews[1:]] == ["COMMENT"]
    assert "That audit is queued for this commit" in gh.posted_reviews[1]["body"]
    head = conn.execute(
        "SELECT status, trigger, priority FROM heads WHERE repo='dashpay/platform' AND number=1 AND sha=?",
        (HEAD,),
    ).fetchone()
    assert (head["status"], head["trigger"], head["priority"]) == ("queued", "manual", 1)
    assert kv_get(conn, f"rereview.after_run:{rid}") is None
    # once per commit: a second conversation that lifts again does not queue another audit
    assert kv_get(conn, f"converse.fresh_audit:dashpay/platform#1:{HEAD}") == str(rid)


def test_a_promised_fix_lifts_nothing(cfg, conn, gh, lanes):
    """The #7778 misstep: the lane called a change the author had not pushed "FIXED" and the
    blocker was lifted on code that still had it. Now the reply goes out, nothing else moves."""
    blocker = _blocking()
    bh = _seed_prior(conn, blocker, body="blocker body")
    _full_strength_prior(conn)
    gh.threads = [_finding_thread(bh, blocker["title"], 800, [KNST])]
    _standing_final(gh)
    _converse(
        cfg,
        conn,
        gh,
        lanes,
        [{"finding_hash": bh, "status": "FIXED", "reply": "Resending ehf_tx resolves it."}],
    )
    assert len(gh.replies) == 1 and "until that change is pushed" in gh.replies[0]["body"]
    assert len(gh.posted_reviews) == 1 and gh.comment_edits == []
    assert not conn.execute("SELECT 1 FROM findings WHERE stage='conceded'").fetchone()
    assert gh.gate_bodies[-1].splitlines()[1].startswith("⛔ Final review complete — 1 blocking")


def test_reply_on_preliminary_review_still_runs_the_pipeline(cfg, conn, gh, lanes):
    """Blockers found, Phase 2 deferred, then a maintainer argues the blocker down: the
    pipeline must run so Phase 2 and a final review can still happen; a conversation would
    strand the PR without one."""
    blocker = _blocking()
    bh = _seed_prior(conn, blocker, body="blocker body")
    human = {
        "id": 901,
        "author": "knst",
        "body": "not a bug",
        "created_at": "x",
        "association": "COLLABORATOR",
    }
    gh.threads = [_prior_thread(bh, replies=[human])]
    gh.posted_reviews.append(
        {
            "id": 4242,
            "event": "REQUEST_CHANGES",
            "html_url": "https://gh/r/4242",
            "body": f"<!-- thepastaclaw-review-phase v1 phase=preliminary sha={HEAD} policy=x -->",
        }
    )
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [],
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": [
            {"finding_hash": bh, "status": "WITHDRAWN", "reason": "Agreed."}
        ],
    }
    lanes.verifier["default"] = _verifier([])
    with tx(conn):
        conn.execute("UPDATE heads SET sha=?, status='done'", (HEAD,))
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.REVIEW_REPLY)
    (rid,) = schedule(conn, cfg, spawn=False)
    assert worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False) == RunStatus.DONE
    roles = [c.role for c in lanes.calls]
    assert "conversation" not in roles and "verifier" in roles
    assert len(gh.posted_reviews) == 2  # the final review got published
    assert any("**Withdrawn**" in r["body"] for r in gh.replies)


def test_other_bots_replies_never_await_an_answer(cfg, conn, gh, lanes):
    from reviewsys.github import finding_threads

    root = {
        "id": 900,
        "author": "thepastaclaw",
        "body": "<!-- thepastaclaw-review v1 finding=abc -->\n**🟡 Suggestion: T**",
        "created_at": "x",
        "association": "NONE",
    }
    cr = {
        "id": 901,
        "author": "coderabbitai[bot]",
        "body": "I agree",
        "created_at": "y",
        "association": "NONE",
    }
    parsed = {
        "thread_id": "t",
        "is_resolved": False,
        "is_outdated": False,
        "path": "f",
        "line": 1,
        "comments": [root, cr],
    }
    out = finding_threads([parsed], "thepastaclaw")
    assert out["abc"]["awaiting_answer"] is False and out["abc"]["replies"] == []
    assert [c["author"] for c in out["abc"]["transcript"]] == ["coderabbitai[bot]"]


def test_conversation_keeping_a_blocker_leaves_the_verdict_alone(cfg, conn, gh, lanes):
    blocker = _blocking()
    bh = _seed_prior(conn, blocker, body="blocker body")
    human = {
        "id": 901,
        "author": "knst",
        "body": "no",
        "created_at": "x",
        "association": "COLLABORATOR",
    }
    t = _prior_thread(bh, replies=[human])
    t["comments"]["nodes"][0]["body"] = (
        f"<!-- thepastaclaw-review v1 finding={bh} dedupe=x -->\n**🔴 Blocking: {blocker['title']}**\n\nblocker body"
    )
    gh.threads = [t]
    gh.posted_reviews.append(
        {
            "id": 4242,
            "event": "REQUEST_CHANGES",
            "html_url": "https://gh/r/4242",
            "body": f"<!-- thepastaclaw-review-phase v1 phase=final sha={HEAD} policy=x -->",
        }
    )
    lanes.conversation = {
        "threads": [
            {
                "finding_hash": bh,
                "status": "STILL_VALID",
                "reply": "Concretely: with `n=0` the estimate at line 11 is below the actual fee, so the transaction is rejected, not delayed.",
            }
        ]
    }
    with tx(conn):
        conn.execute("UPDATE heads SET sha=?, status='done'", (HEAD,))
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.REVIEW_REPLY)
    (rid,) = schedule(conn, cfg, spawn=False)
    assert worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False) == RunStatus.DONE
    assert len(gh.posted_reviews) == 1 and len(gh.replies) == 1
    assert gh.replies[0]["body"].endswith("not delayed.")  # no status suffix for STILL_VALID
    assert not any("resolveReviewThread" in " ".join(c) for c in gh.calls)
    assert gh.gate_bodies[-1].splitlines()[1].startswith("⛔ Final review complete — 1 blocking")


def test_conversation_lane_bad_output_fails_the_run_after_two_tries(cfg, conn, gh, lanes):
    s = _sugg()
    fh = _seed_prior(conn, s)
    gh.threads = [
        _prior_thread(
            fh,
            replies=[
                {
                    "id": 901,
                    "author": "knst",
                    "body": "?",
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
    lanes.conversation = {
        "threads": [{"finding_hash": "nope", "status": "STILL_VALID", "reply": "x"}]
    }
    with tx(conn):
        conn.execute("UPDATE heads SET sha=?, status='done'", (HEAD,))
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.REVIEW_REPLY)
    (rid,) = schedule(conn, cfg, spawn=False)
    assert (
        worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False) == RunStatus.FAILED
    )
    assert [c.role for c in lanes.calls] == ["conversation", "conversation"]
    assert gh.replies == []


def test_reply_on_unreviewed_commit_still_runs_a_full_review(cfg, conn, gh, lanes):
    """No standing review for this commit (a push landed since the last review): a reply is a
    normal priority review with the thread in context, exactly as before."""
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
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [],
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": [
            {"finding_hash": fh, "status": "WITHDRAWN", "reason": "Agreed."}
        ],
    }
    lanes.verifier["default"] = _verifier([])
    _rid, status = _run_repo(cfg, conn, gh, lanes, "dashpay/platform", 1, Trigger.REVIEW_REPLY)
    assert status == RunStatus.DONE
    assert "conversation" not in [c.role for c in lanes.calls]
    assert len(gh.posted_reviews) == 1 and "**Withdrawn**" in gh.replies[0]["body"]


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


def test_same_sha_rereview_withdraws_blocker_and_updates_verdict(cfg, conn, gh, lanes):
    """Author pushes back on one finding; the re-review withdraws a *different* blocking
    finding nobody replied to. The blocker thread gets a note and is resolved, and a follow-up
    review corrects the standing REQUEST_CHANGES verdict on the same commit."""
    blocker = _blocking()
    sugg = _sugg()
    bh = _seed_prior(conn, blocker, body="blocker body")
    sh = _seed_prior(conn, sugg)
    human = {
        "id": 901,
        "author": "knst",
        "body": "disagree",
        "created_at": "2026-09-08T20:00:00Z",
        "association": "COLLABORATOR",
    }
    blocker_thread = _prior_thread(bh, replies=[])
    blocker_thread["id"], blocker_thread["comments"]["nodes"][0]["databaseId"] = "PRRT_B", 800
    blocker_thread["comments"]["nodes"][0]["body"] = (
        f"<!-- thepastaclaw-review v1 finding={bh} dedupe=x -->\n**🔴 Blocking: {blocker['title']}**\n\nblocker body"
    )
    gh.threads = [blocker_thread, _prior_thread(sh, replies=[human])]
    gh.posted_reviews.append(
        {
            "id": 4242,
            "event": "REQUEST_CHANGES",
            "html_url": "https://gh/r/4242",
            "body": f"<!-- thepastaclaw-review-phase v1 phase=final sha={HEAD} policy=x -->",
        }
    )
    carried = {**sugg, "finding_hash": sh}
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [carried],
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": [
            {
                "finding_hash": bh,
                "status": "WITHDRAWN",
                "reason": "Latency, not a correctness failure.",
            },
            {
                "finding_hash": sh,
                "status": "STILL_VALID",
                "reason": "FIFO still serialises warmers.",
            },
        ],
    }
    lanes.verifier["default"] = _verifier([carried])
    with tx(conn):
        conn.execute("UPDATE heads SET sha=?, status='done'", (HEAD,))
        assert enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.MANUAL) == "requeued"
    (rid,) = schedule(conn, cfg, spawn=False)
    status = worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False)
    assert status == RunStatus.DONE
    texts = [r["body"] for r in gh.replies]
    assert any("**Still applies**" in t and "FIFO" in t for t in texts)
    assert any("**Withdrawn**" in t and "Latency" in t and "reply=None" in t for t in texts)
    assert sum("resolveReviewThread" in " ".join(c) for c in gh.calls) == 1
    # follow-up review corrects the verdict without re-posting inline comments
    assert len(gh.posted_reviews) == 2
    upd = gh.posted_reviews[1]
    assert upd["event"] == "COMMENT" and upd["comments"] == []
    assert "## Re-review after discussion" in upd["body"]
    assert "Standing review was `CHANGES_REQUESTED`; this re-review is `COMMENT`" in upd["body"]
    assert f"- {blocker['title']}" in upd["body"]
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM reviews WHERE event='COMMENT' AND run_id=?", (rid,)
        ).fetchone()[0]
        == 1
    )
    ev = conn.execute("SELECT detail FROM events WHERE kind='review.verdict_updated'").fetchone()
    assert ev and "CHANGES_REQUESTED -> COMMENT" in ev["detail"]
    assert gh.gate_bodies[-1].splitlines()[1].startswith("✅ Final review complete — no blockers")


def test_same_sha_rereview_with_unchanged_verdict_posts_no_update(cfg, conn, gh, lanes):
    s = _sugg()
    fh = _seed_prior(conn, s)
    human = {
        "id": 901,
        "author": "knst",
        "body": "?",
        "created_at": "x",
        "association": "COLLABORATOR",
    }
    gh.threads = [_prior_thread(fh, replies=[human])]
    gh.posted_reviews.append(
        {
            "id": 4242,
            "event": "COMMENT",
            "html_url": "https://gh/r/4242",
            "body": f"<!-- thepastaclaw-review-phase v1 phase=final sha={HEAD} policy=x -->",
        }
    )
    carried = {**s, "finding_hash": fh}
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [carried],
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": [
            {"finding_hash": fh, "status": "STILL_VALID", "reason": "Yes."}
        ],
    }
    lanes.verifier["default"] = _verifier([carried])
    with tx(conn):
        conn.execute("UPDATE heads SET sha=?, status='done'", (HEAD,))
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.MANUAL)
    (rid,) = schedule(conn, cfg, spawn=False)
    assert worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False) == RunStatus.DONE
    assert len(gh.posted_reviews) == 2 and len(gh.replies) == 1
    assert "## Re-review" in gh.posted_reviews[1]["body"]
    assert (
        conn.execute("SELECT COUNT(*) FROM events WHERE kind='review.verdict_updated'").fetchone()[
            0
        ]
        == 0
    )


def test_verdict_update_is_not_reposted_and_respects_dismissal(cfg, conn, gh, lanes):
    from reviewsys.contract import parse_verifier_output
    from reviewsys.publish import Provenance, standing_verdict, verdict_event

    prov = Provenance(
        reviewers=[], verifier={"model": "m", "agent": "a", "role": "r"}, policy_fingerprint="x"
    )
    clean = parse_verifier_output(
        {**_verifier([]), "review_phase": "final"},
        expected_phase="final",
        expected_coderabbit_ids=[],
    )
    original = {
        "user": {"login": "thepastaclaw"},
        "state": "CHANGES_REQUESTED",
        "body": f"<!-- thepastaclaw-review-phase v1 phase=final sha={HEAD} -->",
    }
    update = {
        "user": {"login": "thepastaclaw"},
        "state": "COMMENTED",
        "body": f"<!-- thepastaclaw-review-update v1 phase=final sha={HEAD} -->",
    }
    assert standing_verdict([original], HEAD, "final", "thepastaclaw") == "CHANGES_REQUESTED"
    # after one follow-up the standing verdict is the follow-up's state: no second post
    assert standing_verdict([original, update], HEAD, "final", "thepastaclaw") == "COMMENTED"
    assert verdict_event(clean, prov) == "COMMENT"
    dismissed = {**original, "state": "DISMISSED"}
    assert standing_verdict([dismissed], HEAD, "final", "thepastaclaw") == "DISMISSED"
    other = {**original, "user": {"login": "someone"}}
    assert standing_verdict([other], HEAD, "final", "thepastaclaw") is None
    # a Phase-1-only re-review never approves, even via the update path
    approving = parse_verifier_output(
        {**_verifier([]), "review_phase": "final", "review_action": "APPROVE"},
        expected_phase="final",
        expected_coderabbit_ids=[],
    )
    assert verdict_event(approving, prov) == "APPROVE"
    assert (
        verdict_event(
            approving,
            Provenance(
                reviewers=[],
                verifier=prov.verifier,
                policy_fingerprint="x",
                phase2_skipped="trivial",
            ),
        )
        == "COMMENT"
    )


def test_second_withdrawn_finding_at_same_sha_still_gets_its_note(cfg, conn, gh, lanes):
    from reviewsys.contract import parse_verifier_output
    from reviewsys.publish import answer_replied_threads

    verified = parse_verifier_output(
        {**_verifier([]), "review_phase": "final"},
        expected_phase="final",
        expected_coderabbit_ids=[],
    )
    open_threads = {
        "aaa": {
            "comment_id": 900,
            "thread_id": "T1",
            "awaiting_answer": False,
            "latest_reply_id": None,
            "replies": [],
        },
        "bbb": {
            "comment_id": 910,
            "thread_id": "T2",
            "awaiting_answer": False,
            "latest_reply_id": None,
            "replies": [],
        },
    }
    out = answer_replied_threads(
        gh,
        "dashpay/platform",
        1,
        HEAD,
        threads={},
        open_threads=open_threads,
        reconciliation={"aaa": {"finding_hash": "aaa", "status": "WITHDRAWN", "reason": "first"}},
        verified=verified,
    )
    assert [o["action"] for o in out] == ["replied"] and "finding=aaa" in gh.replies[0]["body"]
    # a later run at the same sha withdraws the second finding: the first is already answered,
    # the second must still be noted and resolved
    gh.inline = [
        {
            "id": 901,
            "in_reply_to_id": 900,
            "user": {"login": "thepastaclaw"},
            "body": gh.replies[0]["body"],
        }
    ]
    out = answer_replied_threads(
        gh,
        "dashpay/platform",
        1,
        HEAD,
        threads={},
        open_threads=open_threads,
        reconciliation={
            "aaa": {"finding_hash": "aaa", "status": "WITHDRAWN", "reason": "first"},
            "bbb": {"finding_hash": "bbb", "status": "WITHDRAWN", "reason": "second"},
        },
        verified=verified,
    )
    assert {o["finding_hash"]: o["action"] for o in out} == {
        "aaa": "already_answered",
        "bbb": "replied",
    }
    assert sum("resolveReviewThread" in " ".join(c) for c in gh.calls) == 2


def test_open_thread_with_same_answer_is_not_replied_again_on_new_head(cfg, conn, gh, lanes):
    """A failed or delayed resolve must not turn every new head into another status comment."""
    from reviewsys.contract import parse_verifier_output
    from reviewsys.publish import answer_replied_threads

    verified = parse_verifier_output(
        {**_verifier([]), "review_phase": "final"},
        expected_phase="final",
        expected_coderabbit_ids=[],
    )
    prior_answer = (
        "<!-- thepastaclaw-thread-answer v1 sha=old reply=None finding=aaa -->\n"
        "**Resolved** (re-reviewed at `old`): fixed"
    )
    out = answer_replied_threads(
        gh,
        "dashpay/platform",
        1,
        "b" * 40,
        threads={},
        open_threads={
            "aaa": {
                "comment_id": 900,
                "thread_id": "T1",
                "awaiting_answer": False,
                "latest_reply_id": None,
                "bot_answers": [{"id": 901, "body": prior_answer}],
                "replies": [],
            }
        },
        reconciliation={"aaa": {"finding_hash": "aaa", "status": "FIXED", "reason": "fixed"}},
        verified=verified,
    )
    assert out == [
        {"finding_hash": "aaa", "status": "FIXED", "comment_id": 900, "action": "already_answered"}
    ]
    assert not gh.replies


def test_same_sha_rereview_that_finds_a_blocker_updates_final_verdict(cfg, conn, gh, lanes):
    """The standing review is a clean FINAL; the re-review finds a blocker at Phase 1. Instead of
    stacking a full preliminary review on the same commit, the final verdict gets a follow-up."""
    s = _sugg()
    fh = _seed_prior(conn, s)
    human = {
        "id": 901,
        "author": "knst",
        "body": "are you sure?",
        "created_at": "x",
        "association": "COLLABORATOR",
    }
    gh.threads = [_prior_thread(fh, replies=[human])]
    gh.posted_reviews.append(
        {
            "id": 4242,
            "event": "COMMENT",
            "html_url": "https://gh/r/4242",
            "body": f"<!-- thepastaclaw-review-phase v1 phase=final sha={HEAD} policy=x -->",
        }
    )
    b = _blocking()
    lanes.reviewer["default"] = {
        "summary": "found a real problem",
        "findings": [b],
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": [
            {"finding_hash": fh, "status": "FIXED", "reason": "Addressed."}
        ],
    }
    lanes.verifier["default"] = _verifier([b])
    with tx(conn):
        conn.execute("UPDATE heads SET sha=?, status='done'", (HEAD,))
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.MANUAL)
    (rid,) = schedule(conn, cfg, spawn=False)
    assert worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False) == RunStatus.DONE
    assert len(gh.posted_reviews) == 2
    upd = gh.posted_reviews[1]
    assert upd["event"] == "REQUEST_CHANGES" and upd["comments"] == []
    assert "Standing review was `COMMENTED`; this re-review is `REQUEST_CHANGES`" in upd["body"]
    assert "1 blocking finding(s) now stand" in upd["body"]
    assert any("**Resolved**" in r["body"] for r in gh.replies)


def test_verdict_update_converges_on_bot_authored_pr(cfg, conn, gh, lanes):
    """On the bot's own PR GitHub records every review as COMMENTED, so the idempotency check
    must compare the transport state, or a blocking re-review would repost every run."""
    from reviewsys.publish import EVENT_STATE, transport_event

    assert transport_event("REQUEST_CHANGES", own_pr=True) == "COMMENT"
    assert transport_event("REQUEST_CHANGES", own_pr=False) == "REQUEST_CHANGES"
    assert EVENT_STATE[transport_event("APPROVE", own_pr=True)] == "COMMENTED"
    s = _sugg()
    fh = _seed_prior(conn, s)
    human = {
        "id": 901,
        "author": "knst",
        "body": "?",
        "created_at": "x",
        "association": "COLLABORATOR",
    }
    gh.threads = [_prior_thread(fh, replies=[human])]
    gh.pr = {**gh.pr, "user": {"login": "thepastaclaw"}}
    # standing: a COMMENT-transport review (state COMMENTED) on the bot's own PR
    gh.posted_reviews.append(
        {
            "id": 4242,
            "event": "COMMENT",
            "html_url": "https://gh/r/4242",
            "body": f"<!-- thepastaclaw-review-phase v1 phase=final sha={HEAD} policy=x -->",
        }
    )
    b = _blocking()
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [b],
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": [
            {"finding_hash": fh, "status": "FIXED", "reason": "Done."}
        ],
    }
    lanes.verifier["default"] = _verifier([b])
    with tx(conn):
        conn.execute("UPDATE heads SET sha=?, status='done'", (HEAD,))
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.MANUAL)
    (rid,) = schedule(conn, cfg, spawn=False)
    assert worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False) == RunStatus.DONE
    # blockers found, but on an own PR the transport is COMMENT == standing COMMENTED;
    # a separate re-review summary is still posted for visible provenance.
    assert len(gh.posted_reviews) == 2
    assert (
        conn.execute("SELECT COUNT(*) FROM events WHERE kind='review.verdict_updated'").fetchone()[
            0
        ]
        == 0
    )
    # ... yet our own record (and so the verdict label) carries the canonical REQUEST_CHANGES
    _open_pr(conn)
    rows = [r["event"] for r in conn.execute("SELECT event FROM reviews ORDER BY id")]
    assert rows == ["COMMENTED", "REQUEST_CHANGES", "REQUEST_CHANGES"]
    assert labels.wanted(conn, "dashpay/platform", 1) == "pastaclaw:changes-requested"
    ev = conn.execute("SELECT detail FROM events WHERE kind='review.verdict_recorded'").fetchone()
    assert ev and "COMMENT -> REQUEST_CHANGES" in ev["detail"]
    # recording the same verdict again is a no-op
    from reviewsys.contract import parse_verifier_output

    ctx = worker.RunContext.__new__(worker.RunContext)
    ctx.main_conn, ctx.owner_thread = conn, threading.get_ident()
    ctx.repo, ctx.number, ctx.sha, ctx.run_id = "dashpay/platform", 1, HEAD, rid
    out = parse_verifier_output(
        {**_verifier([b]), "review_phase": "final"},
        expected_phase="final",
        expected_coderabbit_ids=[],
    )
    worker._record_verdict(ctx, "final", out, "REQUEST_CHANGES")
    assert conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 3


def _reconcile(conn, gh, repos=None):
    return labels.reconcile(conn, gh, repos={} if repos is None else repos)


def _run_blocking(cfg, conn, gh, lanes):
    """One review of an open PR at HEAD that lands on REQUEST_CHANGES."""
    lanes.reviewer["default"] = {
        "summary": "x",
        "findings": [_blocking()],
        "out_of_scope_findings": [],
    }
    lanes.verifier["preliminary"] = _verifier([_blocking()])
    _open_pr(conn)
    return _run(cfg, conn, gh, lanes)


def test_verdict_label_follows_review_push_and_close(cfg, conn, gh, lanes):
    """The label is reconciled from DB state: a posted verdict sets it, a push (new head, from
    ingest or the router) clears it before the re-review starts, a revert or a follow-up review
    swaps it, closing the PR clears it and reopening restores it."""
    gh.labels = ["pastaclaw:approved", "bug"]  # stale from an earlier commit
    rid, status = _run_blocking(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE and gh.posted_reviews[0]["event"] == "REQUEST_CHANGES"
    assert gh.labels == ["pastaclaw:approved", "bug"]  # the worker itself never touches labels
    assert _reconcile(conn, gh) == 1
    assert gh.labels == ["bug", "pastaclaw:changes-requested"]
    assert gh.label_calls == [
        ("DELETE", "pastaclaw:approved"),
        ("POST", "pastaclaw:changes-requested"),
    ]
    ev = conn.execute("SELECT detail FROM events WHERE kind='label.synced'").fetchone()
    assert ev and ev["detail"] == "pastaclaw:changes-requested"
    # idempotent: re-running changes nothing
    gh.label_calls.clear()
    assert _reconcile(conn, gh) == 0 and gh.label_calls == []

    # a new commit arrives via the router (mention on the new head) before ingest has seen it:
    # label cleared at once, the standing review describes a commit that is no longer live
    with tx(conn):
        conn.execute("UPDATE heads SET status='done'")
        assert (
            enqueue_head(conn, cfg, "dashpay/platform", 1, "b" * 40, Trigger.MENTION) == "created"
        )
    assert _reconcile(conn, gh) == 1
    assert gh.labels == ["bug"] and gh.label_calls == [("DELETE", "pastaclaw:changes-requested")]

    # ingest catches up and the review for the new head is recorded (as a run would): set again
    with tx(conn):
        conn.execute(
            "UPDATE prs SET head_sha=?, updated_at=? WHERE number=1",
            ("b" * 40, "2099-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO reviews (run_id, repo, number, sha, phase, event, posted_at) VALUES (?,?,?,?,?,?,?)",
            (rid, "dashpay/platform", 1, "b" * 40, "final", "APPROVE", "2099-01-01T00:00:00Z"),
        )
    assert _reconcile(conn, gh) == 1 and gh.labels == ["bug", "pastaclaw:approved"]

    # a revert force-push back to the first, already-reviewed commit creates no head row (the
    # sha is known) but ingest moves prs.head_sha: no verdict stands for the live commit
    with tx(conn):
        conn.execute(
            "UPDATE prs SET head_sha=?, updated_at=? WHERE number=1", (HEAD, "2099-01-01T12:00:00Z")
        )
    assert _reconcile(conn, gh) == 1 and gh.labels == ["bug"]
    with tx(conn):
        conn.execute(
            "UPDATE prs SET head_sha=?, updated_at=? WHERE number=1",
            ("b" * 40, "2099-01-01T13:00:00Z"),
        )
    assert _reconcile(conn, gh) == 1 and gh.labels == ["bug", "pastaclaw:approved"]

    # a same-sha follow-up that moved the verdict (as _verdict_update records it)
    with tx(conn):
        conn.execute(
            "INSERT INTO reviews (run_id, repo, number, sha, phase, event, posted_at) VALUES (?,?,?,?,?,?,?)",
            (rid, "dashpay/platform", 1, "b" * 40, "final", "COMMENT", "2099-01-02T00:00:00Z"),
        )
    assert _reconcile(conn, gh) == 1 and gh.labels == ["bug", "pastaclaw:commented"]

    # PR closes: cleared once; reopened at the same commit: the standing verdict is restored
    with tx(conn):
        conn.execute(
            "UPDATE prs SET state='closed', updated_at=? WHERE number=1", ("2099-01-03T00:00:00Z",)
        )
    assert _reconcile(conn, gh) == 1 and gh.labels == ["bug"]
    with tx(conn):
        conn.execute(
            "UPDATE prs SET state='open', updated_at=? WHERE number=1", ("2099-01-03T01:00:00Z",)
        )
    assert _reconcile(conn, gh) == 1 and gh.labels == ["bug", "pastaclaw:commented"]
    # once the cursor is past every timestamp the PR is not even looked at
    with tx(conn):
        conn.execute("UPDATE kv SET value='2099-01-04T00:00:00Z' WHERE key='labels.reconciled_at'")
    gh.calls.clear()
    assert _reconcile(conn, gh) == 0 and not any("/labels" in " ".join(c) for c in gh.calls)


def test_verdict_label_skips_repos_without_labels_and_dismissed_reviews(cfg, conn, gh, lanes):
    gh.labels_defined = False
    _run_blocking(cfg, conn, gh, lanes)
    repos: dict[str, bool] = {}
    assert _reconcile(conn, gh, repos) == 0
    assert repos == {"dashpay/platform": False} and gh.labels == []
    # the repo was never written to: no add that would have auto-created the label
    assert gh.label_calls == []
    assert (
        conn.execute("SELECT COUNT(*) FROM events WHERE kind='label.sync_failed'").fetchone()[0]
        == 1
    )
    # disabled repos are not retried, even when dirty again
    with tx(conn):
        conn.execute("UPDATE heads SET status='done'")
        enqueue_head(conn, cfg, "dashpay/platform", 1, "c" * 40, Trigger.MENTION)
        conn.execute("UPDATE prs SET head_sha=? WHERE number=1", ("c" * 40,))
    gh.calls.clear()
    assert _reconcile(conn, gh, repos) == 0
    assert not any("/labels" in " ".join(c) for c in gh.calls)

    # a review whose recorded event is a backfilled/dismissed state carries no label
    with tx(conn):
        conn.execute(
            "INSERT INTO reviews (run_id, repo, number, sha, phase, event, posted_at) VALUES (NULL,?,?,?,?,?,?)",
            ("dashpay/platform", 1, "c" * 40, "final", "DISMISSED", "2099-01-01T00:00:00Z"),
        )
    assert labels.wanted(conn, "dashpay/platform", 1) is None
    with tx(conn):
        conn.execute(
            "INSERT INTO reviews (run_id, repo, number, sha, phase, event, posted_at) VALUES (NULL,?,?,?,?,?,?)",
            ("dashpay/platform", 1, "c" * 40, "final", "APPROVED", "2099-01-02T00:00:00Z"),
        )
    assert labels.wanted(conn, "dashpay/platform", 1) == "pastaclaw:approved"
    with tx(conn):
        conn.execute(
            "INSERT INTO reviews (run_id, repo, number, sha, phase, event, posted_at, imported) VALUES (NULL,?,?,?,?,?,?,1)",
            ("dashpay/platform", 1, "c" * 40, "final", "IMPORTED", "2099-01-03T00:00:00Z"),
        )
    assert labels.wanted(conn, "dashpay/platform", 1) is None


def test_verdict_label_transient_failure_holds_cursor(cfg, conn, gh, lanes):
    _run_blocking(cfg, conn, gh, lanes)
    gh.fail_next = ["HTTP 502 Bad Gateway"]
    repos: dict[str, bool] = {}
    assert _reconcile(conn, gh, repos) == 0
    assert repos == {} and kv_get(conn, "labels.reconciled_at") is None
    # backed off: the next pass inside the window does nothing at all
    gh.calls.clear()
    assert _reconcile(conn, gh, repos) == 0 and gh.calls == []
    with tx(conn):
        conn.execute("DELETE FROM kv WHERE key='labels.reconciled_at.retry_at'")
    assert _reconcile(conn, gh, repos) == 1 and gh.labels == ["pastaclaw:changes-requested"]
    assert repos == {"dashpay/platform": True}
    assert kv_get(conn, "labels.reconciled_at")


def test_daemon_runs_label_reconciliation_only_when_live(cfg, conn, gh, notifier):
    shadow = Daemon(cfg, conn, gh=gh, notifier=notifier, spawn=False)
    assert "labels" not in shadow.tick(force=True)
    live = Daemon(cfg, conn, gh=gh, notifier=notifier, spawn=True)
    assert live.tick(force=True)["labels"] == 0


LADDER = [
    {
        "model": "gemini-3.8-flash-high",
        "reasoning": "high",
        "quota": {"provider": "antigravity", "group": "Gemini Models"},
    },
    {"model": "glm-5.3-flash", "reasoning": "max", "quota": {"provider": "zai"}},
    {"model": "muse-spark-1.3-contributor", "agent": "muse-reviewer", "reasoning": "xhigh"},
]


def _ladder_cfg(skills_dir, tmp_path):
    from reviewsys import config as cfg_mod

    raw = json.loads((skills_dir / "config.json").read_text())
    raw["review_model_policy"]["phase1"]["candidates"] = LADDER
    (skills_dir / "config.json").write_text(json.dumps(raw))
    return cfg_mod.load(tmp_path / "config.toml")


def _quota_reader(table):
    from reviewsys import quota

    def read(source):
        v = table[source.provider]
        if isinstance(v, Exception):
            raise v
        return quota.QuotaStatus("acct", (quota.Window("5h", v[0]), quota.Window("weekly", v[1])))

    return read


def _run_ladder(cfg, conn, gh, lanes, reader):
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.MENTION)
    (rid,) = schedule(conn, cfg, spawn=False)
    status = worker.main(
        cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False, quota_reader=reader
    )
    return rid, status


def test_phase1_ladder_uses_first_rung_with_quota(cfg, conn, gh, lanes, skills_dir, tmp_path):
    cfg2 = _ladder_cfg(skills_dir, tmp_path)
    assert [c.model for c in cfg2.policy.phase1_candidates] == [c["model"] for c in LADDER]
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    lanes.triage = {"tier": "critical", "reasoning": "touches consensus"}
    rid, status = _run_ladder(cfg2, conn, gh, lanes, _quota_reader({"antigravity": (0.9, 0.6)}))
    assert status == RunStatus.DONE
    p1 = [(s.model, s.agent, s.effort) for s in _reviewer_calls(lanes) if s.model != "gpt-6-astra"]
    # the critical tier asks for max; the gemini rung caps at high
    assert p1 == [("gemini-3.8-flash-high", "phase1-reviewer", "high")] * 3
    body = gh.posted_reviews[0]["body"]
    assert (
        "- Phase 1 model: `gemini-3.8-flash-high` — antigravity quota: 5h 90% left, weekly 60% left"
        in body
    )
    assert "passed over" not in body
    rows = conn.execute(
        "SELECT model, effort FROM lanes WHERE run_id=? AND phase='phase1'", (rid,)
    ).fetchall()
    assert {(r["model"], r["effort"]) for r in rows} == {("gemini-3.8-flash-high", "high")}
    # adding a ladder must not move the tier table, which is seeded by phase1.reviewer
    assert cfg2.policy.phase1_reviewer.model == "glm-5.3-flash"
    assert (
        cfg2.policy.tier_effort("normal").phase1 == cfg.policy.tier_effort("normal").phase1 == "max"
    )
    ev = conn.execute(
        "SELECT detail FROM events WHERE kind='phase1.model_selected' AND run_id=?", (rid,)
    ).fetchone()
    assert ev["detail"] == "gemini-3.8-flash-high: antigravity quota: 5h 90% left, weekly 60% left"


def test_phase1_ladder_falls_to_paid_rung_and_discloses_why(
    cfg, conn, gh, lanes, skills_dir, tmp_path
):
    from reviewsys import quota

    cfg2 = _ladder_cfg(skills_dir, tmp_path)
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    reader = _quota_reader({"antigravity": (0.99, 0.05), "zai": quota.QuotaError("proxy down")})
    _, status = _run_ladder(cfg2, conn, gh, lanes, reader)
    assert status == RunStatus.DONE
    p1 = {(s.model, s.agent, s.effort) for s in _reviewer_calls(lanes) if s.model != "gpt-6-astra"}
    # the muse rung carries its own agent; the `normal` tier asks max, the rung caps at xhigh
    assert p1 == {("muse-spark-1.3-contributor", "muse-reviewer", "xhigh")}
    body = gh.posted_reviews[0]["body"]
    assert (
        "- Phase 1 model: `muse-spark-1.3-contributor` — not quota-gated; passed over "
        "`gemini-3.8-flash-high` (antigravity below 15% reserve: 5h 99% left, weekly 5% left), "
        "`glm-5.3-flash` (zai quota lookup failed)"
    ) in body
    assert "proxy down" not in body  # error text stays in the event, never in the review
    assert "reviewer 1: `muse-spark-1.3-contributor` (agent: `muse-reviewer`" in body
    ev = conn.execute("SELECT detail FROM events WHERE kind='phase1.model_selected'").fetchone()
    assert "glm-5.3-flash (zai quota lookup failed: proxy down)" in ev["detail"]


def test_phase1_lane_failure_falls_down_the_ladder(cfg, conn, gh, lanes, skills_dir, tmp_path):
    cfg2 = _ladder_cfg(skills_dir, tmp_path)
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    lanes.dead_models = {"gemini-3.8-flash-high"}
    reader = _quota_reader({"antigravity": (0.9, 0.9), "zai": (0.9, 0.9)})
    rid, status = _run_ladder(cfg2, conn, gh, lanes, reader)
    assert status == RunStatus.DONE
    p1 = [(s.role, s.model) for s in _reviewer_calls(lanes) if s.model != "gpt-6-astra"]
    # the lanes start side by side on gemini; the first to die there twice moves the run down
    # the ladder, lanes still on gemini follow when they die, and lanes that had not started
    # yet go straight to glm. Every role finishes on glm, exactly once.
    roles = ["always-on", "general", "security-auditor"]
    # (a lane whose first try failed after a sibling had already moved the run makes no
    # second try on the rung the run left)
    on_gemini = [r for r, m in p1 if m == "gemini-3.8-flash-high"]
    assert 2 <= len(on_gemini) <= 6
    assert all(on_gemini.count(r) <= 2 for r in roles), "at most two tries per lane on a rung"
    assert sorted(r for r, m in p1 if m == "glm-5.3-flash") == roles
    assert {m for _, m in p1} == {"gemini-3.8-flash-high", "glm-5.3-flash"}
    body = gh.posted_reviews[0]["body"]
    assert (
        "- Phase 1 model: `glm-5.3-flash` — zai quota: 5h 90% left, weekly 90% left; "
        "passed over `gemini-3.8-flash-high` (lane failed)"
    ) in body
    assert "429" not in body
    kinds = [
        r["kind"]
        for r in conn.execute(
            "SELECT kind FROM events WHERE run_id=? AND kind LIKE 'phase1.model%' ORDER BY id",
            (rid,),
        )
    ]
    assert kinds == ["phase1.model_selected", "phase1.model_fallback"]
    rows = conn.execute(
        "SELECT model, status FROM lanes WHERE run_id=? AND phase='phase1' ORDER BY id", (rid,)
    ).fetchall()
    assert {r["status"] for r in rows if r["model"] == "gemini-3.8-flash-high"} == {"failed"}
    assert [r["status"] for r in rows if r["model"] == "glm-5.3-flash"] == ["completed"] * 3


def test_fallback_below_a_dead_rung_honours_the_ceiling(cfg, conn, gh, lanes, skills_dir, tmp_path):
    """Gemini (cap high) dies at tier max; the fallback must judge GLM against the tier's
    effort (max), not the clamped effort the dead rung ran at (high), and skip it."""
    from reviewsys import config as cfg_mod

    raw = json.loads((skills_dir / "config.json").read_text())
    ladder = json.loads(json.dumps(LADDER))
    ladder[1]["use_up_to"] = "high"
    raw["review_model_policy"]["phase1"]["candidates"] = ladder
    (skills_dir / "config.json").write_text(json.dumps(raw))
    cfg2 = cfg_mod.load(tmp_path / "config.toml")
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    lanes.dead_models = {"gemini-3.8-flash-high"}
    reader = _quota_reader({"antigravity": (0.9, 0.9), "zai": (0.9, 0.9)})
    _, status = _run_ladder(cfg2, conn, gh, lanes, reader)
    assert status == RunStatus.DONE
    models = [s.model for s in _reviewer_calls(lanes) if s.model != "gpt-6-astra"]
    assert models[0] == "gemini-3.8-flash-high"
    assert set(models) == {"gemini-3.8-flash-high", "muse-spark-1.3-contributor"}
    assert models.count("muse-spark-1.3-contributor") == 3  # every role, once, on muse
    body = gh.posted_reviews[0]["body"]
    assert "`glm-5.3-flash` (not used above high effort; tier asks max)" in body


def test_phase1_failure_on_last_rung_falls_through_to_phase_2(
    cfg, conn, gh, lanes, skills_dir, tmp_path
):
    """Every Phase-1 rung is out: the run does not fail and retry, it reviews with Phase 2
    alone and says so, like the backlog rule."""
    cfg2 = _ladder_cfg(skills_dir, tmp_path)
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    lanes.dead_models = {"muse-spark-1.3-contributor"}
    reader = _quota_reader({"antigravity": (0.0, 0.0), "zai": (0.0, 0.0)})
    rid, status = _run_ladder(cfg2, conn, gh, lanes, reader)
    assert status == RunStatus.DONE
    body = gh.posted_reviews[0]["body"]
    assert "## Final validation — Phase 2 only (Phase 1 failed)" in body
    assert "- Phase 1 reviewers: **failed (failed on every model; its output was dropped)**" in body
    assert "429" not in body, "the error text stays in the event"
    assert "Phase 2 only (Phase 1 failed)" in gh.gate_bodies[-1]
    step = conn.execute(
        "SELECT status, detail FROM steps WHERE run_id=? AND name='phase1'", (rid,)
    ).fetchone()
    assert step["status"] == "failed" and '"fell_through": true' in step["detail"]
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM events WHERE kind='phase1.failed_fallthrough' AND run_id=?",
            (rid,),
        ).fetchone()[0]
        == 1
    )
    verifier = next(s for s in lanes.calls if s.role == "verifier")
    assert "no Phase-1 evidence for this head (Phase 1 failed on every model" in verifier.prompt


def test_phase1_failure_on_a_trivial_tier_still_fails_the_run(
    cfg, conn, gh, lanes, skills_dir, tmp_path
):
    """A trivial tier has no Phase 2 to fall through to."""
    cfg2 = _ladder_cfg(skills_dir, tmp_path)
    lanes.triage = {"tier": "trivial", "reasoning": "typo"}
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    lanes.dead_models = {"muse-spark-1.3-contributor"}
    reader = _quota_reader({"antigravity": (0.0, 0.0), "zai": (0.0, 0.0)})
    _, status = _run_ladder(cfg2, conn, gh, lanes, reader)
    assert status == RunStatus.FAILED
    assert gh.posted_reviews == []


def test_phase1_ladder_survives_a_reader_that_raises_anything(
    cfg, conn, gh, lanes, skills_dir, tmp_path
):
    cfg2 = _ladder_cfg(skills_dir, tmp_path)
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])

    def reader(_):
        raise TypeError("'NoneType' object is not iterable")

    _, status = _run_ladder(cfg2, conn, gh, lanes, reader)
    assert status == RunStatus.DONE
    models = {s.model for s in _reviewer_calls(lanes) if s.model != "gpt-6-astra"}
    assert models == {"muse-spark-1.3-contributor"}


def test_ladder_config_validation(cfg, skills_dir, tmp_path):
    from reviewsys import config as cfg_mod

    pristine = (skills_dir / "config.json").read_text()

    def load_with(**phase1_extra):
        raw = json.loads(pristine)
        raw["review_model_policy"]["phase1"].update(phase1_extra)
        (skills_dir / "config.json").write_text(json.dumps(raw))
        return cfg_mod.load(tmp_path / "config.toml")

    with pytest.raises(ValueError, match="not the last rung"):
        load_with(candidates=[{"model": "muse"}, {"model": "glm", "quota": {"provider": "zai"}}])
    with pytest.raises(ValueError, match="quota provider"):
        load_with(candidates=[{"model": "x", "quota": {"provider": "openai"}}])
    with pytest.raises(ValueError, match="quota_reserve"):
        load_with(quota_reserve=15)
    with pytest.raises(ValueError, match="effort"):
        load_with(candidates=[{"model": "x", "reasoning": "ultra"}])
    with pytest.raises(ValueError, match="use_up_to"):
        load_with(
            candidates=[
                {"model": "x", "use_up_to": "ultra", "quota": {"provider": "zai"}},
                {"model": "y"},
            ]
        )
    with pytest.raises(ValueError, match="last rung"):
        load_with(candidates=[{"model": "x", "use_up_to": "high"}])
    assert (
        load_with(
            candidates=[
                {"model": "x", "use_up_to": "high", "quota": {"provider": "zai"}},
                {"model": "y"},
            ]
        )
        .policy.phase1_candidates[0]
        .use_up_to
        == "high"
    )


def test_glm_rung_is_passed_over_when_the_tier_asks_max(cfg, conn, gh, lanes, skills_dir, tmp_path):
    """The live policy shape: Gemini out of quota, GLM capped by `use_up_to: high`, a normal
    tier asking max. Phase 1 still runs (on Muse at xhigh); the skip is disclosed."""
    from reviewsys import config as cfg_mod

    raw = json.loads((skills_dir / "config.json").read_text())
    ladder = json.loads(json.dumps(LADDER))
    ladder[1]["use_up_to"] = "high"
    raw["review_model_policy"]["phase1"]["candidates"] = ladder
    (skills_dir / "config.json").write_text(json.dumps(raw))
    cfg2 = cfg_mod.load(tmp_path / "config.toml")
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    zai_calls: list[str] = []

    def reader(source):
        zai_calls.append(source.provider)
        return _quota_reader({"antigravity": (0.99, 0.05), "zai": (1.0, 1.0)})(source)

    _, status = _run_ladder(cfg2, conn, gh, lanes, reader)
    assert status == RunStatus.DONE
    assert "zai" not in zai_calls  # never even looked at GLM's quota
    p1 = {(s.model, s.effort) for s in _reviewer_calls(lanes) if s.model != "gpt-6-astra"}
    assert p1 == {("muse-spark-1.3-contributor", "xhigh")}
    body = gh.posted_reviews[0]["body"]
    assert "`glm-5.3-flash` (not used above high effort; tier asks max)" in body
    # a low tier asks high: GLM is back in play (second PR, same repo)
    lanes.calls.clear()
    lanes.triage = {"tier": "low", "reasoning": "small"}
    gh.posted_reviews.clear()
    with tx(conn):
        enqueue_head(conn, cfg2, "dashpay/platform", 2, HEAD, Trigger.MENTION)
    (rid,) = schedule(conn, cfg2, spawn=False)
    status = worker.main(
        cfg2, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False, quota_reader=reader
    )
    assert status == RunStatus.DONE
    p1 = {(s.model, s.effort) for s in _reviewer_calls(lanes) if s.model != "gpt-6-astra"}
    assert p1 == {("glm-5.3-flash", "high")}


def test_phase1_ladder_is_silent_when_backlog_skips_phase1(
    cfg, conn, gh, lanes, skills_dir, tmp_path
):
    cfg2 = _ladder_cfg(skills_dir, tmp_path)
    _queue_extra_heads(conn, cfg2, cfg2.backlog_skip_phase1_above + 1)
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])

    def reader(_):
        raise AssertionError("quota must not be consulted when Phase 1 is skipped")

    _, status = _run_ladder(cfg2, conn, gh, lanes, reader)
    assert status == RunStatus.DONE
    assert "- Phase 1 model:" not in gh.posted_reviews[0]["body"]
    assert (
        conn.execute("SELECT 1 FROM events WHERE kind='phase1.model_selected'").fetchone() is None
    )


def test_policy_without_candidates_has_single_rung_and_no_choice_line(cfg, conn, gh, lanes):
    assert cfg.policy.phase1_candidates == (cfg.policy.phase1_reviewer,)
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])

    def reader(_):
        raise AssertionError("single rung must not be quota-checked")

    _, status = _run_ladder(cfg, conn, gh, lanes, reader)
    assert status == RunStatus.DONE
    assert "- Phase 1 model:" not in gh.posted_reviews[0]["body"]


def test_triage_prompt_makes_normal_the_default_and_critical_conjunctive():
    from reviewsys.triage import _prompt

    p = _prompt("dashpay/dash", "develop", "t", "b", [], ["trivial", "low", "normal", "critical"])
    assert "`normal`: the default" in p
    assert "BOTH large or intricate AND the diff itself changes a critical surface" in p
    assert "Size alone never qualifies" in p
    assert "When unsure between two tiers pick the lower one" in p
    assert "pick the higher" not in p


def test_contract_breaking_phase1_output_falls_through(cfg, conn, gh, lanes):
    """A Phase-1 reviewer whose output breaks the contract (wrong head_sha) drops Phase 1."""
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    real = lanes.__call__

    def runner(spec, art, worktree):
        res = real(spec, art, worktree)
        # every Phase-1 general turn, its correction turns included, names another commit
        if spec.role == "general" and spec.model == "glm-5.3-flash":
            text = res.result_text.replace(HEAD, "f" * 40)
            res.result_text, res.stdout = text, json.dumps({"result": text})
        return res

    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.MENTION)
    (rid,) = schedule(conn, cfg, spawn=False)
    assert worker.main(cfg, conn, rid, gh=gh, lane_runner=runner, heartbeat=False) == (
        RunStatus.DONE
    )
    assert "Phase 2 only (Phase 1 failed)" in gh.posted_reviews[0]["body"]


def test_fatal_phase1_error_still_fails_the_run(cfg, conn, gh, lanes, monkeypatch):
    """A broken prompt template would break Phase 2 the same way: fail now, spend nothing."""
    from reviewsys.models import FailKind, ReviewError

    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])

    def broken(*a, **k):
        raise ReviewError(FailKind.FATAL, "verifier template missing")

    monkeypatch.setattr(worker, "verifier_prompt", broken)
    rid, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.FAILED
    assert not [s for s in lanes.calls if s.model == "gpt-6-astra" and s.role != "prep"]
    step = conn.execute(
        "SELECT name, status FROM steps WHERE run_id=? AND name='verify1'", (rid,)
    ).fetchone()
    assert step["status"] == "failed"


def test_a_later_full_review_voids_older_concessions(conn):
    """Conversation run 1 conceded X; full review run 2 of the same sha re-posted X. A later
    concession of Y must not count X as lifted: the full review re-adjudicated it."""
    from types import SimpleNamespace

    with tx(conn):
        hid = conn.execute(
            "INSERT INTO heads (repo, number, sha, trigger, priority, status, queued_at, eligible_at) VALUES (?,?,?,?,?,?,?,?)",
            ("dashpay/platform", 1, HEAD, "new_pr", 0, "done", "2026-09-07", "2026-09-07"),
        ).lastrowid
        runs = [
            conn.execute(
                "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at) VALUES (?,1,'done','t','2026-09-07T00:00:00Z','2026-09-07T00:00:00Z')",
                (hid,),
            ).lastrowid
            for _ in range(3)
        ]
        rows = [
            (runs[0], "posted", "xxx"),
            (runs[0], "posted", "yyy"),
            (runs[1], "conceded", "xxx"),
            (runs[2], "posted", "xxx"),
            (runs[2], "posted", "yyy"),
        ]
        for rid, stage, h in rows:
            conn.execute(
                worker._FINDINGS_INSERT,
                (rid, "final", stage, h, "f.rs", 1, 1, "blocking", None, "general", h, ""),
            )
    ctx = SimpleNamespace(conn=conn, repo="dashpay/platform", number=1, sha=HEAD, open_threads={})
    assert worker._open_blockers(ctx, "final", lifted={"yyy"}) == 1  # type: ignore[arg-type]
    # a concession made after that full review counts again
    with tx(conn):
        rid = conn.execute(
            "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at) VALUES (?,1,'done','t','2026-09-07T00:00:00Z','2026-09-07T00:00:00Z')",
            (hid,),
        ).lastrowid
        conn.execute(
            worker._FINDINGS_INSERT,
            (rid, "final", "conceded", "xxx", "f.rs", 1, 1, "blocking", None, "general", "x", ""),
        )
    assert worker._open_blockers(ctx, "final", lifted={"yyy"}) == 0  # type: ignore[arg-type]


def test_reviewers_and_verifiers_are_told_to_review_statically_from_ci(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    _run(cfg, conn, gh, lanes)
    graded = [s for s in lanes.calls if s.role not in ("selector", "triage", "prep")]
    if lane_mod.exec_deny_profile(cfg.lane_deny_exec):  # darwin: main() wraps every lane
        assert all(s.sandbox_profile for s in lanes.calls)
    assert graded and all("This is a static review." in s.prompt for s in graded)
    assert all('"ci": {' in s.prompt and "--log-failed" in s.prompt for s in graded)


# ---- deferrals need a tracking issue; the bot files it when a maintainer asks ----

FILE_IT = {
    "status": "INTENTIONALLY_DEFERRED",
    "reply": "Agreed, the drainage costs predate this PR; deferring it.",
    "tracking": {
        "file": {
            "repo": "dashpay/platform",
            "title": "Account for drainage costs in fee estimation",
            "body": "Fee estimation leaves out drainage costs (f.rs:11).",
        }
    },
}


def _deferral_thread(gh, conn, body="Deferred by decision; go ahead and file it."):
    blocker = _blocking()
    bh = _seed_prior(conn, blocker, body="Estimation must be >= actual.")
    gh.threads = [_finding_thread(bh, blocker["title"], 900, [{**KNST, "body": body}])]
    _standing_final(gh)
    return bh


def _issue_calls(gh, verb):
    return [c for c in gh.calls if c[:2] == ["issue", verb]]


def test_conversation_files_the_tracking_issue_a_maintainer_asks_for(cfg, conn, gh, lanes):
    bh = _deferral_thread(gh, conn)
    gh.open_issues = [
        {"number": 7, "title": "Flaky test", "url": "https://github.com/dashpay/platform/issues/7"}
    ]
    _converse(cfg, conn, gh, lanes, [{"finding_hash": bh, **FILE_IT}])
    (issue,) = gh.created_issues
    assert issue["repo"] == "dashpay/platform"
    assert issue["title"] == "Account for drainage costs in fee estimation"
    body = issue["body"]
    assert body.startswith("Fee estimation leaves out drainage costs (f.rs:11).")
    assert "Deferred from https://github.com/dashpay/platform/pull/1 " in body
    assert "https://github.com/dashpay/platform/pull/1#discussion_r901" in body
    assert "maintainer `knst`" in body and body.endswith(
        "Fee estimation omits drainage costs**\n\n> body"
    )
    assert "Posted autonomously" not in body
    url = "https://github.com/dashpay/platform/issues/9000"
    reply = gh.replies[-1]["body"]
    assert reply.endswith(
        f"_Filed {url} for the follow-up. Deferred on a maintainer's call: this finding no "
        "longer blocks this pull request._"
    )
    assert kv_get(conn, f"defer.issue:dashpay/platform#1:{bh}") == url
    row = conn.execute("SELECT body FROM findings WHERE stage='conceded'").fetchone()
    assert row["body"] == f"INTENTIONALLY_DEFERRED\n{url}"
    assert len(gh.posted_reviews) == 2  # the blocker is lifted: the verdict moves
    assert conn.execute("SELECT 1 FROM events WHERE kind='defer.issue_filed'").fetchone()
    # a requeued run (the worker died after filing, before the reply and concession were
    # recorded) never files twice, and does not even search again
    with tx(conn):
        conn.execute("DELETE FROM findings WHERE stage='conceded'")
    _converse(cfg, conn, gh, lanes, [{"finding_hash": bh, **FILE_IT}], first=False)
    assert len(gh.created_issues) == 1 and len(_issue_calls(gh, "list")) == 1
    assert url in gh.replies[-1]["body"]


def test_conversation_links_an_open_issue_that_already_tracks_the_follow_up(cfg, conn, gh, lanes):
    bh = _deferral_thread(gh, conn)
    existing = "https://github.com/dashpay/platform/issues/42"
    gh.open_issues = [
        {"number": 42, "title": "Fee estimation should account for drainage costs", "url": existing}
    ]
    _converse(cfg, conn, gh, lanes, [{"finding_hash": bh, **FILE_IT}])
    assert not gh.created_issues and not _issue_calls(gh, "create")
    (search,) = _issue_calls(gh, "list")
    assert search[search.index("-R") + 1] == "dashpay/platform"
    assert "in:title" in search[search.index("--search") + 1]
    assert f"_Tracked in {existing}. Deferred" in gh.replies[-1]["body"]
    assert kv_get(conn, f"defer.issue:dashpay/platform#1:{bh}") == existing
    assert conn.execute("SELECT 1 FROM events WHERE kind='defer.issue_linked'").fetchone()


@pytest.mark.parametrize(
    "repo, error, says",
    [
        (
            "dashpay/platform",
            "HTTP 403: Resource not accessible by integration",
            "I tried to file the tracking issue in `dashpay/platform`, but could not confirm",
        ),
        ("bitcoin/bitcoin", "", "I can't file issues in `bitcoin/bitcoin`."),
    ],
)
def test_conversation_asks_the_maintainer_to_file_when_the_bot_cannot(
    cfg, conn, gh, lanes, repo, error, says
):
    bh = _deferral_thread(gh, conn)
    gh.issue_create_error = error
    ask = {**FILE_IT, "tracking": {"file": {**FILE_IT["tracking"]["file"], "repo": repo}}}
    _converse(cfg, conn, gh, lanes, [{"finding_hash": bh, **ask}])
    reply = gh.replies[-1]["body"]
    assert says in reply and "link it here?" in reply
    assert reply.endswith("until then the finding stands._")
    if repo == "bitcoin/bitcoin":  # not allowlisted: GitHub is never asked
        assert not _issue_calls(gh, "list") and not _issue_calls(gh, "create")
    failed = conn.execute("SELECT detail FROM events WHERE kind='defer.file_failed'").fetchone()
    assert failed and bh in failed["detail"]
    assert kv_get(conn, f"defer.issue:dashpay/platform#1:{bh}") is None
    assert not conn.execute("SELECT 1 FROM findings WHERE stage='conceded'").fetchone()
    assert len(gh.posted_reviews) == 1
    assert gh.gate_bodies[-1].splitlines()[1].startswith("⛔ Final review complete — 1 blocking")


def test_conversation_deferral_without_tracking_keeps_the_blocker_and_asks(cfg, conn, gh, lanes):
    """platform#5258: a MEMBER deferred "to a change of its own", the lane accepted it, and
    the finding was buried under an APPROVE with no issue ever filed."""
    bh = _deferral_thread(gh, conn, body="deferred by decision, to a change of its own")
    defer = {"status": "INTENTIONALLY_DEFERRED", "reply": "Understood, deferring."}
    _converse(cfg, conn, gh, lanes, [{"finding_hash": bh, **defer}])
    reply = gh.replies[-1]["body"]
    assert "Before I mark this deferred it needs a tracking issue" in reply
    assert "I'll file it in `dashpay/platform`" in reply and "Understood" not in reply
    root = gh.threads[0]["comments"]["nodes"][0]["body"]
    assert "⏸️ **Needs a tracking issue** (not yet deferred)" in root
    assert not gh.created_issues and len(gh.posted_reviews) == 1
    assert not conn.execute("SELECT 1 FROM findings WHERE stage='conceded'").fetchone()


def _full_review_deferral(gh, conn, lanes, reply_body):
    blocker = _blocking()
    bh = _seed_prior(conn, blocker, body="blocker body")
    t = _finding_thread(bh, blocker["title"], 900, [{**KNST, "body": reply_body}])
    gh.threads = [t]
    deferred = {
        "finding_hash": bh,
        "status": "INTENTIONALLY_DEFERRED",
        "reason": "Pre-existing; fine to defer.",
    }
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [],
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": [deferred],
    }
    lanes.verifier["default"] = {**_verifier([]), "prior_finding_reconciliation": [deferred]}
    return bh, blocker


def test_full_review_keeps_a_deferred_finding_standing_until_it_is_tracked(cfg, conn, gh, lanes):
    """The verifier lifting a prior finding as INTENTIONALLY_DEFERRED is the same deferral: with
    no tracking reference visible on the PR it does not lift, and the review says why."""
    bh, blocker = _full_review_deferral(gh, conn, lanes, "out of scope here, a follow-up later")
    _rid, status = _run_repo(cfg, conn, gh, lanes, "dashpay/platform", 1, Trigger.REVIEW_REPLY)
    assert status == RunStatus.DONE
    (review,) = gh.posted_reviews
    assert review["event"] == "REQUEST_CHANGES"  # the blocker stands: Phase 2 deferred
    assert "phase=preliminary" in review["body"]
    assert (
        f"_⏸️ Needs a tracking issue before it can be deferred (1): {blocker['title']}. "
        "Link one in the finding's thread, or ask there and PastaClaw files it._"
    ) in review["body"]
    (answer,) = gh.replies
    assert (
        "**Still applies** (re-reviewed at `aaaaaaaa`): Deferring this needs a tracking issue"
        in answer["body"]
    )
    held = conn.execute("SELECT detail FROM events WHERE kind='defer.needs_tracking'").fetchone()
    assert held and bh in held["detail"]
    kept = conn.execute(
        "SELECT severity FROM findings WHERE phase='verify1' AND stage='verified' AND hash=?",
        (bh,),
    ).fetchone()
    assert kept["severity"] == "blocking"
    prompt = next(c.prompt for c in _reviewer_calls(lanes))
    assert "give the row a `tracking` field" in prompt


def test_a_held_suggestion_is_not_approved_over_nor_watched_for_ci(cfg, conn, gh, lanes):
    """The verifier approved with a suggestion lifted as deferred; held for tracking, the
    suggestion stands again, so the verdict is COMMENT, and no CI watch (ci.py) is kept for
    an APPROVE that was never published."""
    from reviewsys import ci

    s = _sugg()
    sh = _seed_prior(conn, s)
    gh.threads = [
        _finding_thread(sh, s["title"], 900, [{**KNST, "body": "later"}], label="🟡 Suggestion")
    ]
    deferred = {"finding_hash": sh, "status": "INTENTIONALLY_DEFERRED", "reason": "Later."}
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [],
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": [deferred],
    }
    approve = {
        **_verifier([]),
        "review_action": "APPROVE",
        "prior_finding_reconciliation": [deferred],
    }
    lanes.verifier["default"] = approve
    lanes.verifier["final"] = {**approve, "review_phase": "final"}
    gh.routes[f"repos/dashpay/platform/commits/{HEAD}/check-runs?per_page=100"] = {
        "total_count": 1,
        "check_runs": [{"name": "Tests", "status": "in_progress", "conclusion": None}],
    }
    _rid, status = _run_repo(cfg, conn, gh, lanes, "dashpay/platform", 1, Trigger.REVIEW_REPLY)
    assert status == RunStatus.DONE
    review = gh.posted_reviews[-1]
    assert "phase=final" in review["body"] and review["event"] == "COMMENT"
    assert "Needs a tracking issue before it can be deferred (1)" in review["body"]
    assert ci.load(conn, "dashpay/platform", 1, HEAD) is None


def test_full_review_lifts_a_deferral_tracked_in_its_thread(cfg, conn, gh, lanes):
    _full_review_deferral(
        gh, conn, lanes, "pre-existing; tracked in https://github.com/dashpay/platform/issues/4321"
    )
    _rid, status = _run_repo(cfg, conn, gh, lanes, "dashpay/platform", 1, Trigger.REVIEW_REPLY)
    assert status == RunStatus.DONE
    review = gh.posted_reviews[-1]
    assert review["event"] != "REQUEST_CHANGES" and "phase=final" in review["body"]
    assert "Needs a tracking issue" not in review["body"]
    assert "**Deferred** (re-reviewed at `aaaaaaaa`): Pre-existing" in gh.replies[0]["body"]
    assert not conn.execute("SELECT 1 FROM events WHERE kind='defer.needs_tracking'").fetchone()


@pytest.mark.parametrize("source", ["resolved_thread", "filed_by_the_bot"])
def test_full_review_finds_tracking_on_resolved_threads_and_on_record(cfg, conn, gh, lanes, source):
    """A maintainer often resolves the thread right after deferring, and the issue the bot
    filed is linked only in its own reply: neither may make a tracked deferral stand again."""
    reply = "pre-existing; tracked in #4321" if source == "resolved_thread" else "defer it"
    bh, _ = _full_review_deferral(gh, conn, lanes, reply)
    if source == "resolved_thread":
        gh.threads[0]["isResolved"] = True
    else:
        with tx(conn):
            kv_set(conn, f"defer.issue:dashpay/platform#1:{bh}", "https://x/issues/9")
    _rid, status = _run_repo(cfg, conn, gh, lanes, "dashpay/platform", 1, Trigger.REVIEW_REPLY)
    assert status == RunStatus.DONE
    review = gh.posted_reviews[-1]
    assert review["event"] != "REQUEST_CHANGES" and "Needs a tracking issue" not in review["body"]
    assert not conn.execute("SELECT 1 FROM events WHERE kind='defer.needs_tracking'").fetchone()


def test_a_create_that_may_have_gone_through_is_never_retried(cfg, conn, gh, lanes):
    """A timeout, a server error or no printed URL may still have created the issue: the
    pending marker makes every later attempt ask the maintainer instead of filing again. A
    clean refusal (403) clears it, so a later attempt may file."""
    bh = _deferral_thread(gh, conn)
    key = f"defer.issue:dashpay/platform#1:{bh}"
    gh.issue_create_error = "HTTP 502: Bad Gateway"
    _converse(cfg, conn, gh, lanes, [{"finding_hash": bh, **FILE_IT}])
    assert kv_get(conn, key) == worker.TRACKING_ISSUE_PENDING
    gh.issue_create_error = ""
    _converse(cfg, conn, gh, lanes, [{"finding_hash": bh, **FILE_IT}], first=False)
    assert len(_issue_calls(gh, "create")) == 1 and not gh.created_issues
    assert "could not confirm it was filed" in gh.replies[-1]["body"]
    with tx(conn):
        conn.execute("DELETE FROM kv WHERE key=?", (key,))
    gh.issue_create_error = "HTTP 403: Resource not accessible by integration"
    _converse(cfg, conn, gh, lanes, [{"finding_hash": bh, **FILE_IT}], first=False)
    assert kv_get(conn, key) is None
    gh.issue_create_error = ""
    _converse(cfg, conn, gh, lanes, [{"finding_hash": bh, **FILE_IT}], first=False)
    assert len(gh.created_issues) == 1 and kv_get(conn, key).endswith("/issues/9000")


def test_full_review_reads_the_verifiers_own_row_first(cfg, conn, gh, lanes):
    """A reviewer's row with a reason must not stand in for the verifier's own decision: the
    verifier deferred (untracked) while a reviewer said STILL_VALID."""
    bh, blocker = _full_review_deferral(gh, conn, lanes, "out of scope here")
    lanes.reviewer["default"] = {
        **lanes.reviewer["default"],
        "findings": [{**blocker, "finding_hash": bh}],
        "prior_finding_reconciliation": [
            {"finding_hash": bh, "status": "STILL_VALID", "reason": "Still there."}
        ],
    }
    lanes.verifier["default"] = {
        **_verifier([]),
        "prior_finding_reconciliation": [{"finding_hash": bh, "status": "INTENTIONALLY_DEFERRED"}],
    }
    _rid, status = _run_repo(cfg, conn, gh, lanes, "dashpay/platform", 1, Trigger.REVIEW_REPLY)
    assert status == RunStatus.DONE
    assert gh.posted_reviews[-1]["event"] == "REQUEST_CHANGES"
    assert conn.execute("SELECT 1 FROM events WHERE kind='defer.needs_tracking'").fetchone()


def test_a_settled_deferral_can_still_have_its_issue_filed(cfg, conn, gh, lanes):
    """A deferral lifted before tracking was required stays lifted; a maintainer may still
    ask the bot to file its issue, and a failure there says nothing about the finding
    standing (it does not)."""
    bh = _deferral_thread(gh, conn, body="out of scope")
    with tx(conn):
        conn.execute("UPDATE heads SET sha=?, status='done'", (HEAD,))
        # an earlier conversation's concession, from before tracking was required (a run
        # after the publication, so it counts)
        conn.execute(
            worker._FINDINGS_INSERT,
            (
                _new_run(conn),
                "final",
                "conceded",
                bh,
                "f.rs",
                11,
                11,
                "blocking",
                None,
                "general",
                _blocking()["title"],
                "INTENTIONALLY_DEFERRED",
            ),
        )
    gh.issue_create_error = "HTTP 502: Bad Gateway"
    _converse(cfg, conn, gh, lanes, [{"finding_hash": bh, **FILE_IT}], first=False)
    assert len(_issue_calls(gh, "create")) == 1
    reply = gh.replies[-1]["body"]
    assert "could not confirm it was filed" in reply
    assert "finding stands" not in reply and "Not deferred yet" not in reply
    assert gh.gate_bodies[-1].splitlines()[1].startswith("✅ Final review complete — no blockers")


def _new_run(conn, at="2026-09-08T00:00:00Z"):
    (hid,) = conn.execute("SELECT id FROM heads LIMIT 1").fetchone()
    return conn.execute(
        "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at) "
        "VALUES (?,2,'done',?,?,?)",
        (hid, f"t-{at}", at, at),
    ).lastrowid


# ---- deferrals accepted under the old rule are grandfathered ----

CUTOFF_KEY = "deferral.tracking_required_since"


def _old_deferral(conn, gh, bh, source, *, at="2026-09-08T12:00:00Z"):
    """Evidence that `bh` was deferred at `at`, as each path left it."""
    with tx(conn):
        if source == "concession":  # a conversation's lift, old body format (no link)
            conn.execute(
                worker._FINDINGS_INSERT,
                (
                    _new_run(conn, at),
                    "final",
                    "conceded",
                    bh,
                    "f.rs",
                    11,
                    11,
                    "blocking",
                    None,
                    "general",
                    _blocking()["title"],
                    "INTENTIONALLY_DEFERRED",
                ),
            )
        elif source == "event":  # a thread answer either path recorded
            detail = json.dumps(
                {"finding_hash": bh, "status": "INTENTIONALLY_DEFERRED", "action": "replied"}
            )
            conn.execute(
                "INSERT INTO events (ts, kind, repo, number, detail) VALUES (?,?,?,?,?)",
                (at, "thread.answered", "dashpay/platform", 1, detail),
            )
        elif source == "verifier_set":  # a published review that no longer carried it
            conn.execute(
                "INSERT INTO reviews (run_id, repo, number, sha, phase, github_review_id, "
                "event, posted_at) VALUES (?,?,?,?,?,?,?,?)",
                (_new_run(conn, at), "dashpay/platform", 1, "c" * 40, "final", 7, "APPROVE", at),
            )
    if source == "bot_answer":  # our own answer on the still-open thread
        answer = {
            "id": 950,
            "author": "thepastaclaw",
            "body": f"<!-- thepastaclaw-thread-answer v1 sha=x reply=901 finding={bh} -->\n"
            "**Deferred** (re-reviewed at `cccccccc`): Pre-existing.",
            "created_at": at,
            "association": "NONE",
        }
        gh.threads[0]["comments"]["nodes"].append(_gql_comment(answer))


@pytest.mark.parametrize("source", ["concession", "event", "verifier_set", "bot_answer"])
def test_a_deferral_from_before_the_cutoff_stays_lifted_on_the_next_push(
    cfg, conn, gh, lanes, source
):
    bh, _ = _full_review_deferral(gh, conn, lanes, "out of scope here")
    _old_deferral(conn, gh, bh, source)
    with tx(conn):
        kv_set(conn, CUTOFF_KEY, "2026-10-01T00:00:00Z")
    _rid, status = _run_repo(cfg, conn, gh, lanes, "dashpay/platform", 1, Trigger.REVIEW_REPLY)
    assert status == RunStatus.DONE
    review = gh.posted_reviews[-1]
    assert review["event"] != "REQUEST_CHANGES" and "Needs a tracking issue" not in review["body"]
    assert not conn.execute("SELECT 1 FROM events WHERE kind='defer.needs_tracking'").fetchone()
    old = conn.execute("SELECT detail FROM events WHERE kind='defer.grandfathered'").fetchone()
    assert old and bh in old["detail"] and "2026-10-01T00:00:00Z" in old["detail"]


@pytest.mark.parametrize("source", ["concession", "event", "verifier_set", "bot_answer"])
def test_an_untracked_deferral_after_the_cutoff_is_held(cfg, conn, gh, lanes, source):
    bh, _ = _full_review_deferral(gh, conn, lanes, "out of scope here")
    _old_deferral(conn, gh, bh, source)
    with tx(conn):
        kv_set(conn, CUTOFF_KEY, "2026-09-01T00:00:00Z")
    _rid, status = _run_repo(cfg, conn, gh, lanes, "dashpay/platform", 1, Trigger.REVIEW_REPLY)
    assert status == RunStatus.DONE
    assert gh.posted_reviews[-1]["event"] == "REQUEST_CHANGES"
    held = conn.execute("SELECT detail FROM events WHERE kind='defer.needs_tracking'").fetchone()
    assert held and bh in held["detail"]
    assert not conn.execute("SELECT 1 FROM events WHERE kind='defer.grandfathered'").fetchone()


def test_a_grandfathered_deferral_restated_in_conversation_is_not_re_vetted(cfg, conn, gh, lanes):
    """On a new head the old deferral was let through; a maintainer asking about it gets the
    restatement, not a demand for tracking that would claim the finding stands."""
    bh = _deferral_thread(gh, conn, body="is this still deferred?")
    _old_deferral(conn, gh, bh, "event")
    with tx(conn):
        kv_set(conn, CUTOFF_KEY, "2026-10-01T00:00:00Z")
        conn.execute("UPDATE heads SET sha=?, status='done'", (HEAD,))
        # this commit's full review (after the cutoff) let the old deferral through
        _publish(conn, _new_run(conn, "2026-10-05T00:00:00Z"), HEAD)
    defer = {"status": "INTENTIONALLY_DEFERRED", "reply": "Yes, still deferred."}
    _converse(cfg, conn, gh, lanes, [{"finding_hash": bh, **defer}], first=False)
    reply = gh.replies[-1]["body"]
    assert reply.startswith(f"<!-- thepastaclaw-thread-answer v1 sha={HEAD}")
    assert "Yes, still deferred." in reply and "Before I mark this deferred" not in reply

    assert "_Deferred on a maintainer's call" in reply


def _publish(conn, run_id, sha, *, step=None):
    """A `reviews` row for `run_id`; `step`: a follow-up's own step (`converse`: a
    conversation's verdict follow-up, `ci_check`: a CI update)."""
    conn.execute(
        "INSERT INTO reviews (run_id, repo, number, sha, phase, github_review_id, event, "
        "posted_at) VALUES (?,?,?,?,?,?,?,?)",
        (run_id, "dashpay/platform", 1, sha, "final", 8, "COMMENT", "x"),
    )
    if step:
        conn.execute(
            "INSERT INTO steps (run_id, name, status, started_at) VALUES (?, ?, 'ok', 'x')",
            (run_id, step),
        )


@pytest.mark.parametrize("case", ["conversation_follow_up", "ci_update", "stood_after_cutoff"])
def test_old_evidence_does_not_grandfather_a_new_deferral(cfg, conn, gh, lanes, case):
    """The last review before the cutoff being a conversation's verdict follow-up or a CI
    update (ci.py; neither has a verifier set) proves nothing; and a finding deferred long ago that a review carried
    again after the cutoff was standing, so deferring it now is a new deferral."""
    bh, blocker = _full_review_deferral(gh, conn, lanes, "out of scope here")
    if case == "stood_after_cutoff":
        _old_deferral(conn, gh, bh, "event")
    with tx(conn):
        kv_set(conn, CUTOFF_KEY, "2026-10-01T00:00:00Z")
        if case in ("conversation_follow_up", "ci_update"):
            step = "converse" if case == "conversation_follow_up" else "ci_check"
            _publish(conn, _new_run(conn, "2026-09-20T00:00:00Z"), "c" * 40, step=step)
        else:
            f = _blocking()  # carried again, re-hashed (another category): matched by title
            conn.execute(
                worker._FINDINGS_INSERT,
                (
                    _new_run(conn, "2026-10-05T00:00:00Z"),
                    "verify2",
                    "verified",
                    "0123456789ab",
                    f["file"],
                    11,
                    12,
                    "blocking",
                    None,
                    "bug",
                    blocker["title"],
                    "still",
                ),
            )
    _rid, status = _run_repo(cfg, conn, gh, lanes, "dashpay/platform", 1, Trigger.REVIEW_REPLY)
    assert status == RunStatus.DONE
    assert gh.posted_reviews[-1]["event"] == "REQUEST_CHANGES"
    assert not conn.execute("SELECT 1 FROM events WHERE kind='defer.grandfathered'").fetchone()
