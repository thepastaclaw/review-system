"""Batching a review's summary, inline comments, and thread replies into one GitHub review."""

from __future__ import annotations

import json
import subprocess

import pytest

from reviewsys import github, publish, worker
from reviewsys.contract import Finding
from reviewsys.db import tx
from reviewsys.gh import Gh, read_only_runner
from reviewsys.ingest import enqueue_head
from reviewsys.models import FailKind, ReviewError, RunStatus, Trigger
from reviewsys.scheduler import schedule
from reviewsys.steps import worktree as wt

HEAD = "a" * 40


@pytest.fixture(autouse=True)
def fake_git(monkeypatch, tmp_path):
    monkeypatch.setattr(wt, "ensure_mirror", lambda mirrors, repo: tmp_path / "mirror")
    monkeypatch.setattr(wt, "fetch_head", lambda mirror, number, sha: None)

    def create(mirror, wts, name, sha):
        p = wts / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(wt, "create_worktree", create)
    monkeypatch.setattr(wt, "remove_worktree", lambda mirror, path: None)
    monkeypatch.setattr(wt, "merge_base", lambda worktree, base, sha: "b" * 40)
    monkeypatch.setattr(wt, "fetch_commit", lambda mirror, url, sha: True)


def _thread(
    tid: str,
    cid: int,
    h: str,
    title: str,
    *,
    severity: str = "blocking",
    reply: str | None = None,
    resolved: bool = False,
) -> dict:
    icon = "🔴 Blocking" if severity == "blocking" else "🟡 Suggestion"
    nodes = [
        {
            "databaseId": cid,
            "author": {"login": "thepastaclaw"},
            "body": f"<!-- thepastaclaw-review v1 finding={h} dedupe=dd{cid} -->\n**{icon}: {title}**\n\nbody",
            "createdAt": "2026-01-01T00:00:00Z",
            "authorAssociation": "NONE",
        }
    ]
    if reply is not None:
        nodes.append(
            {
                "databaseId": cid + 1,
                "author": {"login": "someone"},
                "body": reply,
                "createdAt": "2026-01-01T01:00:00Z",
                "authorAssociation": "MEMBER",
            }
        )
    return {
        "id": tid,
        "isResolved": resolved,
        "isOutdated": False,
        "path": "f.rs",
        "line": 12,
        "comments": {"nodes": nodes},
    }


def _verifier(findings=None, recon=None, action="COMMENT", cr=None):
    return {
        "summary": "Summary.",
        "review_action": action,
        "findings": findings or [],
        "dropped_findings": [],
        "out_of_scope_findings": [],
        "coderabbit_reactions": cr or [],
        "prior_finding_reconciliation": recon or [],
        "prerequisite_adjudications": [],
        "adjudication_complete": True,
    }


def _run(cfg, conn, gh, lanes, *, trigger=Trigger.MENTION):
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, trigger)
    (rid,) = schedule(conn, cfg, spawn=False)
    status = worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False)
    return rid, status


def test_full_review_batches_findings_thread_answers_and_coderabbit_in_one_review(
    cfg, conn, gh, lanes
):
    """A review with new inline findings, 3 prior-thread answers (including a resolved thread),
    and a CodeRabbit reply produces exactly 1 GitHub review and 0 standalone reply calls."""
    old1 = {
        "file": "f.rs",
        "line_start": 11,
        "line_end": 12,
        "severity": "blocking",
        "confidence": 0.95,
        "category": "logic",
        "title": "Old bug 1",
        "body": "Old body 1.",
    }
    old2 = {
        "file": "f.rs",
        "line_start": 11,
        "line_end": 12,
        "severity": "blocking",
        "confidence": 0.95,
        "category": "logic",
        "title": "Old bug 2",
        "body": "Old body 2.",
    }
    old3 = {
        "file": "f.rs",
        "line_start": 11,
        "line_end": 12,
        "severity": "blocking",
        "confidence": 0.95,
        "category": "logic",
        "title": "Old bug 3",
        "body": "Old body 3.",
    }
    f1, f2, f3 = Finding(**old1), Finding(**old2), Finding(**old3)
    # Round 1 on an earlier commit posts the 3 blockers.
    prev_sha = "0" * 40
    gh.pr["head"]["sha"] = prev_sha
    lanes.head = prev_sha
    lanes.reviewer["default"] = {
        "summary": "issues",
        "findings": [old1, old2, old3],
        "out_of_scope_findings": [],
    }
    lanes.verifier["preliminary"] = _verifier([old1, old2, old3], action="REQUEST_CHANGES")
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 1, prev_sha, Trigger.MENTION)
    (rid0,) = schedule(conn, cfg, spawn=False)
    assert worker.main(cfg, conn, rid0, gh=gh, lane_runner=lanes, heartbeat=False) == RunStatus.DONE
    assert len(gh.submitted_reviews) == 1

    # Round 2 on HEAD: threads from round 1 exist (1 replied, 1 unreplied, 1 replied+resolved),
    # plus a CodeRabbit comment.
    gh.pr["head"]["sha"] = HEAD
    lanes.head = HEAD
    gh.threads = [
        _thread("PRRT_1", 101, f1.hash, f1.title, reply="Fixed in abc"),
        _thread("PRRT_2", 102, f2.hash, f2.title),
        _thread("PRRT_3", 103, f3.hash, f3.title, reply="Already handled", resolved=True),
        {
            "id": "PRRT_cr",
            "isResolved": False,
            "isOutdated": False,
            "path": "f.rs",
            "line": 11,
            "comments": {
                "nodes": [
                    {
                        "databaseId": 201,
                        "author": {"login": "coderabbitai[bot]"},
                        "body": "CodeRabbit comment",
                        "createdAt": "2026-01-01T00:00:00Z",
                        "authorAssociation": "NONE",
                    }
                ]
            },
        },
    ]
    new_finding = {
        "file": "f.rs",
        "line_start": 11,
        "line_end": 12,
        "severity": "suggestion",
        "confidence": 0.9,
        "category": "logic",
        "title": "New suggestion",
        "body": "Details.",
    }
    recon = [
        {"finding_hash": f1.hash, "status": "FIXED", "reason": "Verified fixed."},
        {"finding_hash": f2.hash, "status": "WITHDRAWN", "reason": "False positive."},
        {"finding_hash": f3.hash, "status": "FIXED", "reason": "Confirmed."},
    ]
    cr = [{"comment_id": 201, "action": "disagree", "reply": "This branch is unreachable."}]
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [new_finding],
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": recon,
    }
    lanes.verifier["preliminary"] = _verifier([], recon=recon)
    lanes.verifier["final"] = _verifier([new_finding], recon=recon, action="APPROVE", cr=cr)

    before = len(gh.submitted_reviews)
    _, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert len(gh.submitted_reviews) == before + 1
    assert len(gh.posted_reviews) == before + 1
    assert gh.standalone_replies == []
    assert gh.pending_reviews == {}

    rev = gh.submitted_reviews[-1]
    assert rev["event"] == "APPROVE"
    assert len(rev["comments"]) == 1
    assert {r["thread_id"] for r in rev["replies"]} == {"PRRT_1", "PRRT_2", "PRRT_3", "PRRT_cr"}
    assert len(gh.reactions) == 1
    # Only the 2 unresolved threads are resolved; PRRT_3 was already resolved.
    resolve_calls = [c for c in gh.calls if any("resolveReviewThread" in a for a in c)]
    assert len(resolve_calls) == 2


def test_review_without_replies_uses_rest_fast_path(cfg, conn, gh, lanes):
    """When there are no replies to bundle, publish uses the single REST POST without GraphQL."""
    lanes.reviewer["default"] = {"summary": "clean", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["preliminary"] = _verifier([])
    lanes.verifier["final"] = _verifier([], action="APPROVE")

    _, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert len(gh.submitted_reviews) == 1
    assert gh.standalone_replies == []
    assert not any(c[:2] == ["api", "graphql"] and "--input" in c for c in gh.calls)


def test_conversation_batches_multiple_replies_and_verdict_into_one_review(cfg, conn, gh, lanes):
    """A conversation answering 2 threads AND moving REQUEST_CHANGES -> APPROVE posts 1 review."""
    b1 = {
        "file": "f.rs",
        "line_start": 11,
        "line_end": 12,
        "severity": "blocking",
        "confidence": 0.95,
        "category": "logic",
        "title": "First blocker",
        "body": "Body 1.",
    }
    b2 = {
        "file": "f.rs",
        "line_start": 11,
        "line_end": 12,
        "severity": "blocking",
        "confidence": 0.95,
        "category": "logic",
        "title": "Second blocker",
        "body": "Body 2.",
    }
    f1, f2 = Finding(**b1), Finding(**b2)
    lanes.reviewer["default"] = {
        "summary": "issues",
        "findings": [b1, b2],
        "out_of_scope_findings": [],
    }
    lanes.verifier["preliminary"] = _verifier([])
    lanes.verifier["final"] = _verifier([b1, b2], action="REQUEST_CHANGES")

    _, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert len(gh.submitted_reviews) == 1

    # Now human replies to both blockers; conversation withdraws both and approves.
    gh.threads = [
        _thread("PRRT_b1", 301, f1.hash, f1.title, reply="Not a bug."),
        _thread("PRRT_b2", 302, f2.hash, f2.title, reply="Also not a bug."),
    ]
    lanes.conversation = {
        "threads": [
            {"finding_hash": f1.hash, "status": "WITHDRAWN", "reply": "Good catch, withdrawing."},
            {"finding_hash": f2.hash, "status": "WITHDRAWN", "reply": "Agreed, withdrawing."},
        ]
    }
    before = len(gh.submitted_reviews)
    _, status = _run(cfg, conn, gh, lanes, trigger=Trigger.REVIEW_REPLY)
    assert status == RunStatus.DONE
    assert len(gh.submitted_reviews) == before + 1
    assert gh.standalone_replies == []
    rev = gh.submitted_reviews[-1]
    assert rev["event"] == "APPROVE"
    assert {r["thread_id"] for r in rev["replies"]} == {"PRRT_b1", "PRRT_b2"}


def test_conversation_without_verdict_move_batches_multiple_replies_into_one_review(
    cfg, conn, gh, lanes
):
    """A conversation answering 2 threads with STILL_VALID (no verdict change) posts 1 review."""
    b1 = {
        "file": "f.rs",
        "line_start": 11,
        "line_end": 12,
        "severity": "blocking",
        "confidence": 0.95,
        "category": "logic",
        "title": "First blocker",
        "body": "Body 1.",
    }
    b2 = {
        "file": "f.rs",
        "line_start": 11,
        "line_end": 12,
        "severity": "blocking",
        "confidence": 0.95,
        "category": "logic",
        "title": "Second blocker",
        "body": "Body 2.",
    }
    f1, f2 = Finding(**b1), Finding(**b2)
    lanes.reviewer["default"] = {
        "summary": "issues",
        "findings": [b1, b2],
        "out_of_scope_findings": [],
    }
    lanes.verifier["preliminary"] = _verifier([])
    lanes.verifier["final"] = _verifier([b1, b2], action="REQUEST_CHANGES")
    _, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE

    gh.threads = [
        _thread("PRRT_b1", 301, f1.hash, f1.title, reply="Why?"),
        _thread("PRRT_b2", 302, f2.hash, f2.title, reply="Why?"),
    ]
    lanes.conversation = {
        "threads": [
            {"finding_hash": f1.hash, "status": "STILL_VALID", "reply": "Still applies (1)."},
            {"finding_hash": f2.hash, "status": "STILL_VALID", "reply": "Still applies (2)."},
        ]
    }
    before = len(gh.submitted_reviews)
    _, status = _run(cfg, conn, gh, lanes, trigger=Trigger.REVIEW_REPLY)
    assert status == RunStatus.DONE
    assert len(gh.submitted_reviews) == before + 1
    assert len(gh.posted_reviews) == 1  # no verdict summary review added
    assert gh.standalone_replies == []
    rev = gh.submitted_reviews[-1]
    assert rev["event"] == "COMMENT" and rev["body"] == ""
    assert {r["thread_id"] for r in rev["replies"]} == {"PRRT_b1", "PRRT_b2"}


def test_stale_pending_review_is_deleted_before_starting_new_batch(gh):
    """A leftover PENDING review from a crashed prior run is deleted before opening a new one."""
    gh.pending_reviews["PRR_stale_99"] = {"commit_id": HEAD, "comments": [], "replies": []}
    batch = publish.ReviewBatch(gh, "dashpay/platform", 1, "thepastaclaw", commit_oid=HEAD)
    batch.queue_reply(comment_id=10, thread_id="PRRT_1", body="hello")
    resp = batch.submit_review(commit_id=HEAD, event="COMMENT", body="review body", comments=[])
    assert resp["id"]
    assert "PRR_stale_99" in gh.deleted_pending_reviews
    assert gh.pending_reviews == {}
    assert len(gh.submitted_reviews) == 1
    assert gh.submitted_reviews[0]["replies"] == [{"thread_id": "PRRT_1", "body": "hello"}]


def test_failed_submit_aborts_pending_review_and_skips_post_actions(gh, monkeypatch):
    """If submitPullRequestReview fails, the pending review is deleted and no post-submit
    thread resolutions or root edits run."""
    batch = publish.ReviewBatch(gh, "dashpay/platform", 1, "thepastaclaw", commit_oid=HEAD)
    post_ran = []
    batch.queue_reply(
        comment_id=10,
        thread_id="PRRT_1",
        body="reply",
        post_submit=lambda: post_ran.append(True),
    )

    def boom(*_args, **_kwargs):
        raise ReviewError(FailKind.INFRA, "502 bad gateway")

    monkeypatch.setattr(github, "submit_pending_review", boom)
    with pytest.raises(ReviewError, match="502"):
        batch.submit_review(commit_id=HEAD, event="COMMENT", body="summary", comments=[])

    assert gh.pending_reviews == {}
    assert len(gh.deleted_pending_reviews) == 1
    assert post_ran == []


def test_partial_reply_failure_still_submits_review_with_remaining_replies(gh, monkeypatch):
    """One failing thread reply records reply_failed without blocking the review or other replies."""
    batch = publish.ReviewBatch(gh, "dashpay/platform", 1, "thepastaclaw", commit_oid=HEAD)
    statuses: dict[str, str] = {}
    batch.queue_reply(
        comment_id=10,
        thread_id="PRRT_bad",
        body="first",
        on_ok=lambda: statuses.__setitem__("bad", "ok"),
        on_error=lambda err: statuses.__setitem__("bad", f"err:{err}"),
    )
    batch.queue_reply(
        comment_id=11,
        thread_id="PRRT_good",
        body="second",
        on_ok=lambda: statuses.__setitem__("good", "ok"),
        on_error=lambda err: statuses.__setitem__("good", f"err:{err}"),
    )

    real_add = github.add_pending_reply

    def flaky_add(gh_, rid, tid, body):
        if tid == "PRRT_bad":
            raise ReviewError(FailKind.CONTRACT, "thread locked")
        return real_add(gh_, rid, tid, body)

    monkeypatch.setattr(github, "add_pending_reply", flaky_add)
    batch.submit_review(commit_id=HEAD, event="COMMENT", body="summary", comments=[])
    assert statuses["bad"].startswith("err:")
    assert statuses["good"] == "ok"
    assert len(gh.submitted_reviews) == 1
    assert gh.submitted_reviews[0]["replies"] == [{"thread_id": "PRRT_good", "body": "second"}]


def test_all_replies_failing_in_reply_only_batch_aborts_empty_pending_review(gh, monkeypatch):
    """If every reply fails to attach and there is no review body, abort instead of submitting."""
    batch = publish.ReviewBatch(gh, "dashpay/platform", 1, "thepastaclaw", commit_oid=HEAD)
    batch.queue_reply(comment_id=10, thread_id="PRRT_bad", body="first")
    monkeypatch.setattr(
        github,
        "add_pending_reply",
        lambda *_a, **_k: (_ for _ in ()).throw(ReviewError(FailKind.CONTRACT, "gone")),
    )
    assert batch.submit_replies_only() is None
    assert gh.submitted_reviews == []
    assert gh.pending_reviews == {}
    assert len(gh.deleted_pending_reviews) == 1


def test_graphql_structured_input_uses_stdin_and_is_blocked_by_read_only_runner():
    """Structured GraphQL variables travel on stdin (`--input -`) and stay blocked in replay."""
    seen: list[tuple[list[str], str | None]] = []

    def runner(argv, stdin, timeout):
        seen.append((list(argv), stdin))
        return subprocess.CompletedProcess(argv, 0, json.dumps({"data": {"ok": True}}), "")

    gh = Gh(runner=runner)
    res = gh.graphql("mutation($input: X!) { m(input: $input) }", {"input": {"a": [1, 2]}})
    assert res == {"ok": True}
    assert seen[0][0] == ["gh", "api", "graphql", "--input", "-"]
    assert json.loads(seen[0][1] or "{}")["variables"] == {"input": {"a": [1, 2]}}

    ro = Gh(runner=read_only_runner(runner))
    with pytest.raises(ReviewError, match="read-only gh"):
        ro.graphql("mutation($input: X!) { m(input: $input) }", {"input": {"a": [1, 2]}})


def test_dedupe_reuse_reply_batches_into_the_same_review(cfg, conn, gh, lanes):
    """A resolved-thread reuse note from dedupe_against_github goes into the same GitHub review."""
    reused = {
        "file": "f.rs",
        "line_start": 11,
        "line_end": 12,
        "severity": "blocking",
        "confidence": 0.95,
        "category": "logic",
        "title": "Existing resolved bug",
        "body": "Still broken.",
    }
    fresh = {
        "file": "f.rs",
        "line_start": 11,
        "line_end": 12,
        "severity": "suggestion",
        "confidence": 0.85,
        "category": "logic",
        "title": "Brand new suggestion",
        "body": "Add a test.",
    }
    rf = Finding(**reused)
    gh.inline = [
        {
            "id": 501,
            "node_id": "PRRC_501",
            "path": "f.rs",
            "line": 12,
            "html_url": "https://github.com/dashpay/platform/pull/1#discussion_r501",
            "user": {"login": "thepastaclaw"},
            "body": (
                f"<!-- thepastaclaw-review v1 finding={rf.hash} dedupe={rf.dedupe_key} -->\n"
                f"**🔴 Blocking: {rf.title}**\n\nold body"
            ),
        }
    ]
    gh.routes["graphql:node"] = None
    orig_dispatch = gh._dispatch

    def _with_resolved_node(args, stdin):
        if args[:2] == ["api", "graphql"] and any("node(id:" in a for a in args):
            return {
                "data": {
                    "node": {
                        "pullRequestReviewThread": {
                            "id": "PRRT_reused",
                            "isResolved": True,
                            "comments": {"nodes": []},
                        }
                    }
                }
            }
        return orig_dispatch(args, stdin)

    gh._dispatch = _with_resolved_node
    lanes.reviewer["default"] = {
        "summary": "issues",
        "findings": [reused, fresh],
        "out_of_scope_findings": [],
    }
    lanes.verifier["preliminary"] = _verifier([reused, fresh], action="REQUEST_CHANGES")

    _, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert len(gh.submitted_reviews) == 1
    assert gh.standalone_replies == []
    rev = gh.submitted_reviews[0]
    assert len(rev["comments"]) == 1
    assert len(rev["replies"]) == 1
    assert rev["replies"][0]["thread_id"] == "PRRT_reused"
    assert "Reusing this existing thread" in rev["replies"][0]["body"]


def test_v12_publish_and_threads_batch_replies_with_review(
    cfg, skills_dir, conn, gh, lanes, monkeypatch
):
    """Under policy v12, both a new-head review with thread answers and a same-sha conversation
    that moves the verdict produce 1 GitHub review each and 0 standalone replies."""
    from tests.v12_support import (
        A,
        B,
        Git,
        cand,
        publish_threads,
        refute_gate,
        reply,
        run_round,
        v12_cfg,
    )

    c12 = v12_cfg(cfg, skills_dir)
    g = Git()
    monkeypatch.setattr(worker, "_lines_changed", g)
    monkeypatch.setattr(worker, "_map_lines", g.map)
    lanes.verify12 = refute_gate

    # Round 1 on head A: 2 blockers.
    c1 = cand("First v12 blocker", line=11)
    c2 = cand("Second v12 blocker", line=12)
    lanes.candidates["default"] = [c1, c2]
    run_round(c12, conn, gh, lanes, A)
    assert len(gh.submitted_reviews) == 1
    assert gh.standalone_replies == []
    ids = publish_threads(gh)
    h1, h2 = list(ids)

    # Round 2 on head B: thread lane marks both FIXED while finder finds 1 new suggestion.
    # Both thread replies + the new inline comment + the review summary land in 1 review.
    g.change("f.rs", 11, at=B)
    g.change("f.rs", 12, at=B)
    lanes.thread_out[h1] = {"status": "FIXED", "reply": "Fixed in B."}
    lanes.thread_out[h2] = {"status": "FIXED", "reply": "Also fixed in B."}
    lanes.candidates["default"] = [cand("New v12 suggestion", severity="suggestion", line=11)]
    before = len(gh.submitted_reviews)
    run_round(c12, conn, gh, lanes, B)
    assert len(gh.submitted_reviews) == before + 1
    assert gh.standalone_replies == []
    rev = gh.submitted_reviews[-1]
    assert len(rev["comments"]) == 1
    assert len(rev["replies"]) == 2

    # Round 3 (conversation on B): author replies to the suggestion; thread lane withdraws it
    # and the verdict moves COMMENT -> APPROVE — reply + APPROVE in 1 single review.
    ids_b = publish_threads(gh)
    (h_sugg,) = [h for h in ids_b if h not in {h1, h2}]
    reply(gh, ids_b[h_sugg], "Actually unnecessary.", rid=901)
    lanes.thread_out[h_sugg] = {"status": "WITHDRAWN", "reply": "Fair point, withdrawing."}
    before = len(gh.submitted_reviews)
    run_round(c12, conn, gh, lanes, B, trigger=Trigger.REVIEW_REPLY)
    assert len(gh.submitted_reviews) == before + 1
    assert gh.standalone_replies == []
    rev = gh.submitted_reviews[-1]
    assert rev["event"] == "APPROVE"
    assert len(rev["replies"]) == 1
