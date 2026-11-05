"""Policy v12 end to end, through the worker with fake lanes and a simulated PR: a first round,
a re-review after a maintainer withdrawal, a same-sha reply (#38), and the critical tier's
single stage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reviewsys.models import Trigger
from tests.conftest import group
from tests.test_worker_e2e import fake_git  # noqa: F401  (autouse: fake mirror/worktree)
from tests.v12_support import (
    A,
    B,
    by_title,
    cand,
    confirm_matches,
    publish_threads,
    refuted,
    reply,
    resolve,
    resolve_with_answers,
    run_round,
    v12_cfg,
)

pytestmark = pytest.mark.usefixtures("wide_diff")


def test_first_round(v12, conn, gh, lanes, git):
    lanes.candidates["default"] = []

    def verify(label, cands, prompt):
        if label == "verify1":  # the gate refutes Phase 1's claims
            return refuted(cands)
        return {
            "summary": "Two problems.",
            "groups": [
                group(
                    c,
                    verdict="CONFIRMED" if "Fee" in c["title"] else "PLAUSIBLE",
                    confirm_by="a loom test",
                )
                for c in cands
            ],
        }

    lanes.candidates["general"] = [cand("Fee estimate omits drainage"), cand("Maybe racy", line=12)]
    lanes.verify12 = verify
    rid = run_round(v12, conn, gh, lanes, A)
    review = gh.posted_reviews[-1]
    assert review["event"] == "REQUEST_CHANGES"
    bodies = [c["body"] for c in review["comments"]]
    assert any("Blocking: Fee estimate omits drainage" in b for b in bodies)
    racy = next(b for b in bodies if "Maybe racy" in b)
    assert "Suggestion: Maybe racy" in racy and "What would confirm it: a loom test" in racy
    led = by_title(conn)
    assert {t: (i.status, i.severity) for t, i in led.items()} == {
        "Fee estimate omits drainage": ("open", "blocking"),
        "Maybe racy": ("open", "suggestion"),
    }
    ids = [
        r[0]
        for r in conn.execute(
            "SELECT candidate_id FROM findings WHERE run_id=? AND stage LIKE 'lane:%'", (rid,)
        )
    ]
    assert ids and all(i.startswith(("p1-", "p2-")) for i in ids)
    finder = next(c for c in lanes.calls if "## Output: candidates" in c.prompt)
    for absent in ("prior_finding_reconciliation", "finding_hash", "confidence", "review_phase"):
        assert absent not in finder.prompt.split("## Output: candidates")[0]
    assert "last reviewed" not in finder.prompt  # round 1: no pointer
    assert not any(k == "thread" for k, *_ in lanes.v12_calls)  # nothing to judge


def test_rereview_after_a_maintainer_withdrawal(v12, conn, gh, lanes, git):
    lanes.verify12 = confirm_matches
    lanes.candidates["default"] = []
    lanes.candidates["general"] = [
        cand("Fee estimate omits drainage"),
        cand("Add a regression test", severity="suggestion", line=20),
        cand("Name the constant", severity="suggestion", line=30),
    ]
    run_round(v12, conn, gh, lanes, A)
    assert gh.posted_reviews[-1]["event"] == "REQUEST_CHANGES"
    ids = publish_threads(gh)
    fee, test_, name = (
        by_title(conn)[t].hash
        for t in ("Fee estimate omits drainage", "Add a regression test", "Name the constant")
    )
    reply(gh, ids[fee], "The caller adds the drainage cost; see wallet.rs:40.", rid=901)

    # round 2 (new push B, nothing near the anchors changed): the thread lane judges only the
    # replied issue; the finders restate the two open suggestions and nothing else
    lanes.thread_out[fee] = {"status": "WITHDRAWN", "reply": "You are right: the caller adds it."}
    lanes.candidates["general"] = [
        cand(f"Add a regression test [open:{test_}]", severity="suggestion", line=20),
        cand(f"Name the constant [open:{name}]", severity="suggestion", line=30),
    ]
    before = len(gh.posted_reviews)
    run_round(v12, conn, gh, lanes, B)
    threads_calls = [ids_ for kind, _, ids_ in lanes.v12_calls if kind == "thread"]
    assert threads_calls == [[fee]], "only the replied issue is judged; unchanged ones are not"
    (answer,) = [r for r in gh.replies if f"finding={fee}" in r["body"]]
    assert "**Withdrawn** (re-reviewed at `cccccccc`): You are right" in answer["body"]
    assert any("resolveReviewThread" in " ".join(c) for c in gh.calls)
    review = gh.posted_reviews[before]
    assert review["event"] == "COMMENT" and review["comments"] == []  # nothing re-posted
    assert "### Still open from earlier rounds (2)" in review["body"]
    assert f"#discussion_r{ids[test_]}" in review["body"]
    assert (
        "Fee estimate"
        not in review["body"].split("Still open from earlier rounds")[1].split("<details>")[0]
    )
    led = by_title(conn)
    assert led["Fee estimate omits drainage"].status == "withdrawn"
    assert led["Add a regression test"].status == "open"
    assert led["Add a regression test"].last_seen_sha == B  # corroborated this round
    finder = next(c for c in reversed(lanes.calls) if "## Output: candidates" in c.prompt)
    assert f"last reviewed at `{A}`" in finder.prompt

    # round 3 (C): a finder raises the withdrawn blocker again on unchanged lines: dropped,
    # never re-posted; the finders were told it is settled. The suggestion's code changed
    # and the thread lane says it is fixed.
    resolve(gh, ids[fee])  # the bot resolved it with its answer
    git.change("f.rs", 20, at="d" * 40)
    lanes.thread_out = {test_: {"status": "FIXED", "reply": "The test now covers it."}}
    lanes.candidates["general"] = [
        cand(f"Fee estimate omits drainage [settled:{fee}]"),
        cand(f"Name the constant [open:{name}]", severity="suggestion", line=30),
    ]
    before, calls = len(gh.posted_reviews), len(lanes.v12_calls)
    rid = run_round(v12, conn, gh, lanes, "d" * 40)
    assert [i for k, _, i in lanes.v12_calls[calls:] if k == "thread"] == [[test_]]
    run_dir = Path(conn.execute("SELECT run_dir FROM runs WHERE id=?", (rid,)).fetchone()[0])
    dropped = json.loads((run_dir / "verify2.json").read_text())["dropped"]
    assert [d["reason"].split(":")[0] for d in dropped] == ["SETTLED"]
    finder = next(c for c in reversed(lanes.calls) if "## Output: candidates" in c.prompt)
    settled = finder.prompt.split("### Settled in earlier rounds", 1)[1]
    assert "Fee estimate omits drainage" in settled and "wallet.rs:40" in settled
    review = gh.posted_reviews[before]
    assert review["comments"] == [] and review["event"] == "COMMENT"
    led = by_title(conn)
    assert led["Fee estimate omits drainage"].status == "withdrawn"  # not re-opened
    assert led["Add a regression test"].status == "fixed"
    assert led["Name the constant"].status == "open"


def test_a_refound_issue_is_not_fixed_by_the_thread_lane(v12, conn, gh, lanes, git):
    """The thread lane calls a blocker FIXED, but a finder restated it on this head: it stays
    open, the FIXED reply is never posted, and the review still requests changes."""
    lanes.verify12 = confirm_matches
    lanes.candidates["default"] = []
    lanes.candidates["general"] = [cand("Fee estimate omits drainage")]
    run_round(v12, conn, gh, lanes, A)
    publish_threads(gh)
    fee = by_title(conn)["Fee estimate omits drainage"].hash
    git.change("f.rs", 11, at=B)
    lanes.thread_out[fee] = {"status": "FIXED", "reply": "Fixed by the new branch."}
    lanes.candidates["general"] = [cand(f"Fee estimate omits drainage [open:{fee}]")]
    before = len(gh.posted_reviews)
    run_round(v12, conn, gh, lanes, B)
    assert not [r for r in gh.replies if f"finding={fee}" in r["body"]]
    assert by_title(conn)["Fee estimate omits drainage"].status == "open"
    review = gh.posted_reviews[before]
    assert review["event"] == "REQUEST_CHANGES" and review["comments"] == []
    assert "Still open from earlier rounds (1)" in review["body"]
    ev = [
        r["detail"] for r in conn.execute("SELECT detail FROM events WHERE kind='thread.answered'")
    ]
    assert any("suppressed_refound" in d for d in ev)


def test_same_sha_reply_withdrawing_the_last_finding_approves(v12, conn, gh, lanes, git):
    """#38 over the ledger: a full-strength review leaves one suggestion; a maintainer argues
    it away on the same commit, and the follow-up approves."""
    lanes.verify12 = confirm_matches
    lanes.candidates["default"] = []
    lanes.candidates["general"] = [cand("Add a regression test", severity="suggestion", line=20)]
    run_round(v12, conn, gh, lanes, A)
    assert gh.posted_reviews[-1]["event"] == "COMMENT"
    ids = publish_threads(gh)
    sugg = by_title(conn)["Add a regression test"].hash
    reply(gh, ids[sugg], "There is a test in tests/fee.rs already.", rid=902)
    lanes.thread_out[sugg] = {"status": "WITHDRAWN", "reply": "Right, tests/fee.rs covers it."}
    before = len(gh.posted_reviews)
    run_round(v12, conn, gh, lanes, A, Trigger.REVIEW_REPLY)
    assert [k for k, role, _ in lanes.v12_calls if k == "thread"] == ["thread"]
    upd = gh.posted_reviews[before]
    assert upd["event"] == "APPROVE" and upd["comments"] == []
    assert "Standing review was `COMMENTED`; this re-review is `APPROVE`" in upd["body"]
    assert by_title(conn)["Add a regression test"].status == "withdrawn"
    (answer,) = [r for r in gh.replies if f"finding={sugg}" in r["body"]]
    assert answer["body"].endswith("_Withdrawing this finding._")
    # the standing review's own body says what the discussion settled (#41)
    ((rid, body),) = gh.review_edits
    assert rid == gh.posted_reviews[0]["id"]
    assert "**Updated after discussion:** 1 finding(s) of this review were settled" in body
    assert "✅ Withdrawn: 🟡 Add a regression test" in body


def test_a_carried_issue_lifted_in_discussion_is_marked_in_the_review(v12, conn, gh, lanes, git):
    """The standing review of a later round lists the issues it carried from earlier rounds;
    a same-sha reply that withdraws one updates that review's body too (#41), though no
    findings row of that review names it (the ledger does)."""
    lanes.verify12 = confirm_matches
    lanes.candidates["default"] = []
    lanes.candidates["general"] = [cand("Fee estimate omits drainage", severity="suggestion")]
    run_round(v12, conn, gh, lanes, A)
    ids = publish_threads(gh)
    fee = by_title(conn)["Fee estimate omits drainage"].hash
    lanes.candidates["general"] = [
        cand(f"Fee estimate omits drainage [open:{fee}]", severity="suggestion")
    ]
    run_round(v12, conn, gh, lanes, B)
    review_b = gh.posted_reviews[-1]
    reply(gh, ids[fee], "The caller adds the drainage cost; see wallet.rs:40.", rid=901)
    lanes.thread_out[fee] = {"status": "WITHDRAWN", "reply": "You are right: the caller adds it."}
    run_round(v12, conn, gh, lanes, B, Trigger.REVIEW_REPLY)
    assert by_title(conn)["Fee estimate omits drainage"].status == "withdrawn"
    ((rid, body),) = gh.review_edits
    assert rid == review_b["id"]
    assert "✅ Withdrawn: 🟡 Fee estimate omits drainage" in body
    assert "WITHDRAWN in discussion; do not fix" in body


def test_no_same_sha_approval_after_a_related_ci_failure(v12, conn, gh, lanes, git):
    """As under v11 (ci.py): once CI failed after this commit's approval in checks that looked
    related, a discussion alone never approves it again."""
    from reviewsys import ci
    from reviewsys.db import kv_set, tx

    lanes.verify12 = confirm_matches
    lanes.candidates["default"] = []
    lanes.candidates["general"] = [cand("Add a regression test", severity="suggestion", line=20)]
    run_round(v12, conn, gh, lanes, A)
    ids = publish_threads(gh)
    sugg = by_title(conn)["Add a regression test"].hash
    reply(gh, ids[sugg], "There is a test in tests/fee.rs already.", rid=902)
    lanes.thread_out[sugg] = {"status": "WITHDRAWN", "reply": "Right, tests/fee.rs covers it."}
    with tx(conn):
        kv_set(conn, ci.updated_key("dashpay/platform", 1, A), json.dumps({"review_id": 1}))
    run_round(v12, conn, gh, lanes, A, Trigger.REVIEW_REPLY)
    assert by_title(conn)["Add a regression test"].status == "withdrawn"
    assert all(r["event"] != "APPROVE" for r in gh.posted_reviews)


def test_critical_tier_is_single_stage_with_the_thread_lane_beside(
    cfg, skills_dir, conn, gh, lanes, git
):
    v12 = v12_cfg(cfg, skills_dir, tiers={"critical": {"single_stage": True}})
    lanes.triage = {"tier": "critical", "reasoning": "consensus"}
    lanes.candidates["default"] = []
    lanes.candidates["general"] = [cand("Fee estimate omits drainage")]
    run_round(v12, conn, gh, lanes, A)
    assert [k for k, label, _ in lanes.v12_calls if k == "verify"] == ["verify"]
    (ids_,) = [i for k, label, i in lanes.v12_calls if k == "verify"]
    assert {i.split("-")[0] for i in ids_} == {"p1", "p2"}  # both phases, one verifier
    assert ("verify", "verify1", ids_) not in lanes.v12_calls
    ids = publish_threads(gh)
    fee = by_title(conn)["Fee estimate omits drainage"].hash
    reply(gh, ids[fee], "Not reachable: amounts are validated upstream.", rid=904)
    lanes.thread_out[fee] = {"status": "WITHDRAWN", "reply": "Right, validated upstream."}
    lanes.candidates["general"] = []
    lanes.v12_calls.clear()
    before = len(gh.posted_reviews)
    run_round(v12, conn, gh, lanes, B)
    kinds = [k for k, *_ in lanes.v12_calls]
    assert kinds.count("thread") == 1 and kinds.count("verify") == 0  # nothing to verify
    review = gh.posted_reviews[before]
    assert review["event"] == "APPROVE"
    assert by_title(conn)["Fee estimate omits drainage"].status == "withdrawn"
    assert "Single stage" in review["body"]


def test_a_settled_issue_comes_back_only_when_its_code_changed(v12, conn, gh, lanes, git):
    lanes.verify12 = confirm_matches
    lanes.candidates["default"] = []
    lanes.candidates["general"] = [cand("Fee estimate omits drainage")]
    run_round(v12, conn, gh, lanes, A)
    ids = publish_threads(gh)
    fee = by_title(conn)["Fee estimate omits drainage"].hash
    reply(gh, ids[fee], "The caller adds it.", rid=905)
    lanes.thread_out[fee] = {"status": "WITHDRAWN", "reply": "Right."}
    run_round(v12, conn, gh, lanes, B)
    resolve(gh, ids[fee])
    assert by_title(conn)["Fee estimate omits drainage"].status == "withdrawn"
    # the lines under it changed, and a finder raises it again: it stands again
    git.change("f.rs", 11, at="d" * 40)
    lanes.thread_out = {}
    lanes.candidates["general"] = [cand(f"Fee estimate omits drainage [settled:{fee}]")]
    before = len(gh.posted_reviews)
    run_round(v12, conn, gh, lanes, "d" * 40)
    review = gh.posted_reviews[before]
    assert review["event"] == "REQUEST_CHANGES"
    assert by_title(conn)["Fee estimate omits drainage"].status == "open"


def test_a_maintainer_concession_wins_over_a_blind_refind(v12, conn, gh, lanes, git):
    """The thread lane withdraws a suggestion after a maintainer's reply while a finder, who
    never saw the reply, restates it as a CONFIRMED blocker. The concession stands: no
    "raising this to blocking" note on the thread it just resolved, no upgrade, the
    disagreement recorded for the eval, and no approval this round."""
    lanes.verify12 = confirm_matches
    lanes.candidates["default"] = []
    lanes.candidates["general"] = [cand("Add a regression test", severity="suggestion", line=12)]
    run_round(v12, conn, gh, lanes, A)
    ids = publish_threads(gh)
    sugg = by_title(conn)["Add a regression test"].hash
    reply(gh, ids[sugg], "tests/fee.rs covers it already.", rid=906)
    lanes.thread_out[sugg] = {"status": "WITHDRAWN", "reply": "Right, tests/fee.rs covers it."}
    lanes.candidates["general"] = [cand(f"Add a regression test [open:{sugg}]", line=12)]
    before = len(gh.posted_reviews)
    rid = run_round(v12, conn, gh, lanes, B)
    bodies = [
        r["body"] for r in gh.replies if f"finding={sugg}" in r["body"] or "Raising" in r["body"]
    ]
    assert len(bodies) == 1 and "Withdrawn" in bodies[0]
    issue = by_title(conn)["Add a regression test"]
    assert (issue.status, issue.severity) == ("withdrawn", "suggestion")
    (detail,) = [
        json.loads(r[0])
        for r in conn.execute(
            "SELECT detail FROM events WHERE kind='threads.conflict' AND run_id=?", (rid,)
        )
    ]
    assert detail == {"finding_hash": sugg, "outcome": "WITHDRAWN", "verdict": "CONFIRMED"}
    assert gh.posted_reviews[before]["event"] == "COMMENT"


def _fixed_blocker(v12, conn, gh, lanes, git, *, status="FIXED"):
    """Round 1 raises a blocker on line 11; round 2 (its code changed) the thread lane closes
    it, and its thread is resolved. Returns (hash, root comment id)."""
    lanes.verify12 = confirm_matches
    lanes.candidates["default"] = []
    lanes.candidates["general"] = [cand("Fee estimate omits drainage")]
    run_round(v12, conn, gh, lanes, A)
    ids = publish_threads(gh)
    fee = by_title(conn)["Fee estimate omits drainage"].hash
    reply(gh, ids[fee], "Changed it.", rid=907)
    git.change("f.rs", 11, at=B)
    lanes.thread_out[fee] = {"status": status, "reply": "Agreed."}
    lanes.candidates["general"] = []
    run_round(v12, conn, gh, lanes, B)
    resolve(gh, ids[fee])
    lanes.thread_out = {}
    return fee, ids[fee]


def test_a_regression_gets_a_thread_of_its_own(v12, conn, gh, lanes, git):
    fee, _ = _fixed_blocker(v12, conn, gh, lanes, git)
    assert by_title(conn)["Fee estimate omits drainage"].status == "fixed"
    lanes.candidates["general"] = [cand(f"Fee estimate omits drainage [fixed:{fee}]")]
    before = len(gh.posted_reviews)
    run_round(v12, conn, gh, lanes, "d" * 40)
    review = gh.posted_reviews[before]
    assert review["event"] == "REQUEST_CHANGES"
    (comment,) = review["comments"]  # posted, not folded into the old resolved thread
    assert "Fee estimate omits drainage (regression)" in comment["body"]
    assert "regression of an earlier finding that was fixed" in comment["body"]
    led = by_title(conn)
    assert led["Fee estimate omits drainage (regression)"].status == "open"
    assert led["Fee estimate omits drainage"].status == "fixed"


def test_a_revival_under_a_new_title_gets_a_thread_of_its_own(v12, conn, gh, lanes, git):
    fee, _ = _fixed_blocker(v12, conn, gh, lanes, git, status="WITHDRAWN")
    assert by_title(conn)["Fee estimate omits drainage"].status == "withdrawn"
    git.change("f.rs", 11, at="d" * 40)  # its code changed again after it was withdrawn
    lanes.candidates["general"] = [
        cand(f"Fee estimate still omits the drainage cost [settled:{fee}]")
    ]
    before = len(gh.posted_reviews)
    run_round(v12, conn, gh, lanes, "d" * 40)
    review = gh.posted_reviews[before]
    (comment,) = review["comments"]
    assert "Fee estimate still omits the drainage cost" in comment["body"]
    assert "raised before and withdrawn" in comment["body"]
    led = by_title(conn)
    assert led["Fee estimate still omits the drainage cost"].status == "open"


def test_a_new_issue_folded_onto_a_closed_thread_reopens_that_issue(v12, conn, gh, lanes, git):
    """Dedupe folds a new blocker onto the thread of a fixed issue (same lines, similar
    title): the review requests changes without an inline comment, so the folded issue must
    be the ledger's, open on that thread, where it can be judged and withdrawn."""
    _, root = _fixed_blocker(v12, conn, gh, lanes, git)
    lanes.candidates["general"] = [cand("Fee estimate still omits the drainage cost")]
    before = len(gh.posted_reviews)
    run_round(v12, conn, gh, lanes, "d" * 40)
    review = gh.posted_reviews[before]
    assert review["event"] == "REQUEST_CHANGES" and review["comments"] == []
    issue = by_title(conn)["Fee estimate omits drainage"]
    assert issue.status == "open" and issue.comment_id == root
    before = len(gh.posted_reviews)
    lanes.candidates["general"] = []
    run_round(v12, conn, gh, lanes, "e" * 40)
    assert "Still open from earlier rounds (1)" in gh.posted_reviews[before]["body"]


def _withdrawn_blocker(v12, conn, gh, lanes):
    """Round 1 (A) raises a blocker on line 11; round 2 (B) a maintainer argues it away, the
    thread lane withdraws it, and the bot resolves its thread. Returns (hash, root id)."""
    lanes.verify12 = confirm_matches
    lanes.candidates["default"] = []
    lanes.candidates["general"] = [cand("Fee estimate omits drainage")]
    run_round(v12, conn, gh, lanes, A)
    ids = publish_threads(gh)
    fee = by_title(conn)["Fee estimate omits drainage"].hash
    reply(gh, ids[fee], "The caller adds the drainage cost.", rid=909)
    lanes.thread_out[fee] = {"status": "WITHDRAWN", "reply": "Right, the caller adds it."}
    lanes.candidates["general"] = []
    run_round(v12, conn, gh, lanes, B)
    resolve_with_answers(gh, ids[fee])
    lanes.thread_out = {}
    assert by_title(conn)["Fee estimate omits drainage"].status == "withdrawn"
    return fee, ids[fee]


def test_a_reworded_refind_of_a_withdrawn_issue_on_unchanged_code_does_not_block(
    v12, conn, gh, lanes, git
):
    """The reviewer's probe: a finder repeats a withdrawn blocker under a reworded title on
    lines nobody touched, and publish-time dedupe would fold it onto the withdrawn thread. It
    is dropped like a `settled:` match on unchanged code: no block, the issue stays withdrawn,
    and the next push can approve; nothing reads as resolved by hand."""
    _withdrawn_blocker(v12, conn, gh, lanes)
    lanes.candidates["general"] = [cand("Fee estimate still omits the drainage cost")]
    before = len(gh.posted_reviews)
    rid = run_round(v12, conn, gh, lanes, "d" * 40)
    review = gh.posted_reviews[before]
    assert review["event"] != "REQUEST_CHANGES" and review["comments"] == []
    assert "carried-forward" not in review["body"]
    assert by_title(conn)["Fee estimate omits drainage"].status == "withdrawn"
    (refold,) = conn.execute(
        "SELECT detail FROM events WHERE kind='v12.settled_refold' AND run_id=?", (rid,)
    ).fetchone()
    assert refold.endswith(by_title(conn)["Fee estimate omits drainage"].hash)
    lanes.candidates["general"] = []
    before = len(gh.posted_reviews)
    run_round(v12, conn, gh, lanes, "e" * 40)
    review = gh.posted_reviews[before]
    assert review["event"] == "APPROVE"
    assert "resolved by hand" not in review["body"]
    assert by_title(conn)["Fee estimate omits drainage"].status == "withdrawn"


def test_a_reworded_refind_on_changed_code_revives_the_issue_on_its_own_thread(
    v12, conn, gh, lanes, git
):
    _, root = _withdrawn_blocker(v12, conn, gh, lanes)
    git.change("f.rs", 11, at="d" * 40)
    lanes.candidates["general"] = [cand("Fee estimate still omits the drainage cost")]
    before = len(gh.posted_reviews)
    run_round(v12, conn, gh, lanes, "d" * 40)
    review = gh.posted_reviews[before]
    assert review["event"] == "REQUEST_CHANGES" and review["comments"] == []  # its own thread
    issue = by_title(conn)["Fee estimate omits drainage"]
    assert (issue.status, issue.comment_id) == ("open", root)
    lanes.candidates["general"] = []
    before = len(gh.posted_reviews)
    run_round(v12, conn, gh, lanes, "e" * 40)
    review = gh.posted_reviews[before]
    assert review["event"] == "REQUEST_CHANGES"
    assert "resolved by hand" not in review["body"]  # the bot resolved it, then it came back
    assert by_title(conn)["Fee estimate omits drainage"].status == "open"


def test_a_reworded_refind_of_an_issue_conceded_this_round_is_a_conflict(v12, conn, gh, lanes, git):
    """H1 through dedupe: the thread lane withdraws a blocker while a finder, who never saw
    the reply, repeats it under another title that dedupe folds onto its thread. The
    concession wins: the re-find is dropped, the issue stays withdrawn, the disagreement is
    recorded and the round does not approve."""
    lanes.verify12 = confirm_matches
    lanes.candidates["default"] = []
    lanes.candidates["general"] = [cand("Fee estimate omits drainage")]
    run_round(v12, conn, gh, lanes, A)
    ids = publish_threads(gh)
    fee = by_title(conn)["Fee estimate omits drainage"].hash
    reply(gh, ids[fee], "The caller adds the drainage cost.", rid=910)
    lanes.thread_out[fee] = {"status": "WITHDRAWN", "reply": "Right, the caller adds it."}
    lanes.candidates["general"] = [cand("Fee estimate still omits the drainage cost")]
    before = len(gh.posted_reviews)
    rid = run_round(v12, conn, gh, lanes, B)
    review = gh.posted_reviews[before]
    assert review["event"] == "COMMENT" and review["comments"] == []
    assert by_title(conn)["Fee estimate omits drainage"].status == "withdrawn"
    (detail,) = conn.execute(
        "SELECT detail FROM events WHERE kind='threads.conflict' AND run_id=?", (rid,)
    ).fetchone()
    assert json.loads(detail) == {
        "finding_hash": fee,
        "outcome": "WITHDRAWN",
        "verdict": "CONFIRMED",
    }
