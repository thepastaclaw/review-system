"""A v11 policy's published output is pinned end to end across rounds: a first review, a
re-review of a new push that reconciles a human reply, and a same-sha conversation. The golden
(golden/v11-e2e-rounds.json) is compared as copied text: under a v11 policy nothing the bot
posts may differ from production's. Everything but `ledger` was written by main at v0.28.0
(4e75920fe) running this scenario; `ledger` (which main does not have) by the v12 branch
before its review behavior landed (feat/review-quality-v12 @736cab9)."""

from __future__ import annotations

import json
from pathlib import Path

from reviewsys import ledger, worker
from reviewsys.db import tx
from reviewsys.ingest import enqueue_head
from reviewsys.models import RunStatus, Trigger
from reviewsys.scheduler import schedule
from tests.test_worker_e2e import (  # fake_git: autouse, fakes the mirror and worktree
    _blocking,
    _gql_comment,
    _sugg,
    _verifier,
    fake_git,  # noqa: F401
)

GOLDEN = Path(__file__).parent / "golden" / "v11-e2e-rounds.json"
A, C = "a" * 40, "c" * 40
REPO = "dashpay/platform"


def _round(cfg, conn, gh, lanes, sha, trigger):
    lanes.head = sha
    gh.pr["head"]["sha"] = sha
    with tx(conn):
        enqueue_head(conn, cfg, REPO, 1, sha, trigger)
    (rid,) = schedule(conn, cfg, spawn=False)
    status = worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False)
    assert status == RunStatus.DONE
    return rid


def _thread(root_id, body, replies, *, resolved=False):
    root = {
        "id": root_id,
        "author": "thepastaclaw",
        "body": body,
        "created_at": "2026-09-07T00:00:00Z",
        "association": "NONE",
    }
    return {
        "id": f"PRRT_{root_id}",
        "isResolved": resolved,
        "isOutdated": False,
        "path": "f.rs",
        "line": 12,
        "comments": {"nodes": [_gql_comment(c) for c in (root, *replies)]},
    }


def run_scenario(cfg, conn, gh, lanes) -> dict:
    blocker, sugg = _blocking(), _sugg()
    # round 1 (A): Phase 1 finds a suggestion, Phase 2 adds a blocker
    lanes.reviewer["default"] = {"summary": "ok", "findings": [sugg], "out_of_scope_findings": []}
    lanes.verifier["preliminary"] = _verifier([])
    lanes.verifier["final"] = _verifier([blocker, sugg])
    _round(cfg, conn, gh, lanes, A, Trigger.MENTION)
    first = gh.posted_reviews[0]
    bh = next(
        r["hash"]
        for r in conn.execute("SELECT hash, title FROM findings WHERE stage='posted'")
        if r["title"] == blocker["title"]
    )
    sh = next(
        r["hash"]
        for r in conn.execute("SELECT hash, title FROM findings WHERE stage='posted'")
        if r["title"] == sugg["title"]
    )
    ids = {bh: 800, sh: 810}
    by_hash = {
        h: next(c for c in first["comments"] if f"finding={h}" in c["body"]) for h in (bh, sh)
    }
    gh.inline = [
        {
            "id": ids[h],
            "node_id": f"N{ids[h]}",
            "user": {"login": "thepastaclaw"},
            "body": c["body"],
            "path": c["path"],
            "line": c["line"],
            "html_url": f"https://github.com/{REPO}/pull/1#discussion_r{ids[h]}",
        }
        for h, c in by_hash.items()
    ]
    reply = {
        "id": 901,
        "author": "knst",
        "body": "The drainage cost is folded in by the caller.",
        "created_at": "2026-09-08T00:00:00Z",
        "association": "COLLABORATOR",
    }
    gh.threads = [
        _thread(800, by_hash[bh]["body"], [reply]),
        _thread(810, by_hash[sh]["body"], []),
    ]
    # round 2 (C, a new push): the blocker still holds, the suggestion was fixed
    recon = [
        {"finding_hash": bh, "status": "STILL_VALID", "reason": "The caller does not add it."},
        {"finding_hash": sh, "status": "FIXED", "reason": "The test was added."},
    ]
    carried = {**blocker, "finding_hash": bh}
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [carried],
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": recon,
    }
    # the gate verifier drops the carried blocker, so Phase 2 runs and the review is final
    lanes.verifier["preliminary"] = {**_verifier([]), "prior_finding_reconciliation": recon}
    lanes.verifier["final"] = {**_verifier([carried]), "prior_finding_reconciliation": recon}
    _round(cfg, conn, gh, lanes, C, Trigger.MENTION)
    # round 3 (C again): a same-sha reply withdraws the blocker in conversation
    gh.threads = [
        _thread(
            800,
            by_hash[bh]["body"],
            [reply, {**reply, "id": 902, "created_at": "2026-09-09T00:00:00Z", "body": "See L40."}],
        )
    ]
    lanes.conversation = {
        "threads": [{"finding_hash": bh, "status": "WITHDRAWN", "reply": "Right, L40 adds it."}]
    }
    _round(cfg, conn, gh, lanes, C, Trigger.REVIEW_REPLY)
    issues = ledger.load(conn, REPO, 1)
    return {
        "reviews": gh.posted_reviews,
        "replies": gh.replies,
        "comment_edits": gh.comment_edits,
        "review_edits": gh.review_edits,
        "reactions": gh.reactions,
        "ledger": {h: [i.status, i.severity, i.title] for h, i in sorted(issues.items())},
    }


def test_v11_rounds_publish_exactly_what_they_did(cfg, conn, gh, lanes):
    assert cfg.policy.pipeline is None
    got = json.loads(json.dumps(run_scenario(cfg, conn, gh, lanes)))
    assert got == json.loads(GOLDEN.read_text(encoding="utf-8"))
