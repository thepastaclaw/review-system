"""A thread's discussion settles once: a dispute on the merits is answered one time and then
left to the maintainers (DISPUTED), an untracked deferral is asked for tracking one time, and
carried findings are listed compactly instead of being described again on every head."""

from __future__ import annotations

import json

import pytest

from reviewsys import converse, publish, worker
from reviewsys.contract import Finding, VerifierOutput, parse_verifier_output
from reviewsys.db import kv_get, kv_set, tx
from reviewsys.dedupe import Suppressed
from reviewsys.models import ReviewError, RunStatus
from tests.test_worker_e2e import (  # fake_git: autouse, fakes the mirror and worktree
    HEAD,
    KNST,
    _blocking,
    _converse,
    _finding_thread,
    _gql_comment,
    _queue_new_head,
    _reviewer_calls,
    _seed_prior,
    _standing_final,
    _sugg,
    _verifier,
    fake_git,  # noqa: F401
)

C, D = "c" * 40, "d" * 40
URL = "https://github.com/dashpay/platform/pull/1#discussion_r900"
DISPUTE = {**KNST, "body": "Not a bug: the caller clamps it, as in #5000 and #5100."}
ENGAGED = "The clamp at L40 runs after the estimate is returned, so the shortfall still escapes."


def _thread(gh, h, title, replies, *, label="🔴 Blocking"):
    gh.threads = [_finding_thread(h, title, 900, replies, label=label)]
    gh.inline = [
        {
            "id": 900,
            "node_id": "N900",
            "user": {"login": "thepastaclaw"},
            "path": "f.rs",
            "line": 12,
            "html_url": URL,
            "body": gh.threads[0]["comments"]["nodes"][0]["body"],
        }
    ]


def _bot(body, *, id=950, at="y"):
    """Our earlier answer on the thread (after KNST's "x", before a later "z")."""
    return {"id": id, "author": "thepastaclaw", "body": body, "created_at": at}


def _lanes(lanes, h, finding, status, *, reason="r", disputed=False, findings=None):
    row = {"finding_hash": h, "status": status, "reason": reason}
    if disputed:
        row["disputed"] = True
    carried = [{**finding, "finding_hash": h}] if status == "STILL_VALID" else []
    found = carried + list(findings or [])
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": found,
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": [row],
    }
    lanes.verifier["default"] = {**_verifier(found), "prior_finding_reconciliation": [row]}


def _review(cfg, conn, gh, lanes, sha):
    rid = _queue_new_head(cfg, conn, gh, sha, lanes)
    assert worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False) == (
        RunStatus.DONE
    )
    return rid


def _verifier_prompts(lanes):
    return [c.prompt for c in lanes.calls if c.role == "verifier"]


# ---- disputes: answered once, then DISPUTED ----


def _disputed_round_one(cfg, conn, gh, lanes):
    """A maintainer disputes a blocker; the review answers their argument on the thread."""
    blocker = _blocking()
    h = _seed_prior(conn, blocker)
    _thread(gh, h, blocker["title"], [DISPUTE])
    _lanes(lanes, h, blocker, "STILL_VALID", reason=ENGAGED, disputed=True)
    _review(cfg, conn, gh, lanes, HEAD)
    return h, blocker


def test_a_dispute_is_answered_on_its_argument_then_left_to_the_maintainers(cfg, conn, gh, lanes):
    """platform#5529: an author declined a blocker with precedents; the bot answered "Still
    applies" and kept it blocking for 24 reviews."""
    h, blocker = _disputed_round_one(cfg, conn, gh, lanes)
    prompt = _verifier_prompts(lanes)[0]
    # the verifier reads the exchange as the conversation lane does, and the dispute rule
    assert '"author": "knst"' in prompt and '"association": "COLLABORATOR"' in prompt
    assert "set `disputed: true` on its row and answer that argument" in prompt
    (answer,) = gh.replies
    assert f"**Still applies** (re-reviewed at `aaaaaaaa`): {ENGAGED}" in answer["body"]
    assert gh.posted_reviews[-1]["event"] == "REQUEST_CHANGES"
    memo = json.loads(kv_get(conn, f"dispute.answered:dashpay/platform#1:{h}"))
    assert memo == {"reply": "901", "standing": True, "sha": HEAD}

    # next push: nobody replied since the answer and the review would carry it again as is
    gh.threads[0]["comments"]["nodes"].append(_gql_comment(_bot(answer["body"])))
    _lanes(lanes, h, blocker, "STILL_VALID", reason="Still applies.")
    _review(cfg, conn, gh, lanes, C)
    review = gh.posted_reviews[-1]
    assert review["event"] == "COMMENT" and "phase=final" in review["body"]
    assert "🔴 1 blocking" not in review["body"]
    body = review["body"]
    assert "### Disputed — left to maintainers (1)" in body
    assert f"- 🔴 **Blocking**: {blocker['title']} — `f.rs:11-12` ([thread]({URL}))" in body
    assert "Still open from earlier rounds" not in body
    prompt_block = body.split("Prompt for all review comments")[-1]
    assert blocker["title"] not in prompt_block
    closing = gh.replies[-1]["body"]
    assert closing.startswith("<!-- thepastaclaw-thread-answer v1 sha=" + C)
    assert f"**Disputed** (re-reviewed at `cccccccc`): {converse.DISPUTE_REPLY}" in closing
    root = gh.threads[0]["comments"]["nodes"][0]["body"]
    assert "⚖️ **Disputed** (left to the maintainers; not blocking)" in root
    conceded = conn.execute(
        "SELECT body FROM findings WHERE stage='conceded' AND hash=? ORDER BY id DESC", (h,)
    ).fetchone()
    assert conceded["body"] == "DISPUTED"

    # the push after that: settled, so it is neither restated nor answered again
    gh.threads[0]["comments"]["nodes"].append(_gql_comment(_bot(closing, id=951, at="y2")))
    replies = len(gh.replies)
    _lanes(lanes, h, blocker, "DISPUTED", reason="Left to the maintainers.")
    _review(cfg, conn, gh, lanes, D)
    assert '"prior_status": "DISPUTED"' in _verifier_prompts(lanes)[-1]
    body = gh.posted_reviews[-1]["body"]
    assert gh.posted_reviews[-1]["event"] != "REQUEST_CHANGES"
    assert blocker["title"] not in body and "Disputed — left to maintainers" not in body
    assert len(gh.replies) == replies


def test_a_maintainer_who_still_disagrees_gets_one_closing_reply(cfg, conn, gh, lanes):
    h, blocker = _disputed_round_one(cfg, conn, gh, lanes)
    again = {**KNST, "id": 902, "created_at": "z", "body": "No: #5000 does exactly this."}
    gh.threads[0]["comments"]["nodes"] += [
        _gql_comment(_bot(gh.replies[0]["body"])),
        _gql_comment(again),
    ]
    _lanes(lanes, h, blocker, "STILL_VALID", reason="Yet another restatement.", disputed=True)
    _review(cfg, conn, gh, lanes, C)
    assert '"dispute_answered": true' in _verifier_prompts(lanes)[-1]
    closing = gh.replies[-1]["body"]
    assert "reply=902" in closing and converse.DISPUTE_REPLY in closing
    assert "restatement" not in closing
    assert gh.posted_reviews[-1]["event"] == "COMMENT"


def test_a_dispute_from_outside_the_pr_is_answered_but_not_settled(cfg, conn, gh, lanes):
    """Only a maintainer or the PR author speaks for the pull request."""
    blocker = _blocking()
    h = _seed_prior(conn, blocker)
    _thread(gh, h, blocker["title"], [{**DISPUTE, "author": "drive-by", "association": "NONE"}])
    _lanes(lanes, h, blocker, "STILL_VALID", reason=ENGAGED, disputed=True)
    _review(cfg, conn, gh, lanes, HEAD)
    assert kv_get(conn, f"dispute.answered:dashpay/platform#1:{h}") is None


def test_the_pr_author_counts_as_a_disputer(cfg, conn, gh, lanes):
    blocker = _blocking()
    h = _seed_prior(conn, blocker)
    author = {**DISPUTE, "author": "someone", "association": "CONTRIBUTOR"}
    _thread(gh, h, blocker["title"], [author])
    _lanes(lanes, h, blocker, "STILL_VALID", reason=ENGAGED, disputed=True)
    _review(cfg, conn, gh, lanes, HEAD)
    assert json.loads(kv_get(conn, f"dispute.answered:dashpay/platform#1:{h}"))["standing"]


def test_a_reintroduction_revives_a_disputed_finding(cfg, conn, gh, lanes):
    h, blocker = _disputed_round_one(cfg, conn, gh, lanes)
    gh.threads[0]["comments"]["nodes"].append(_gql_comment(_bot(gh.replies[0]["body"])))
    _lanes(lanes, h, blocker, "STILL_VALID")
    _review(cfg, conn, gh, lanes, C)
    assert gh.posted_reviews[-1]["event"] == "COMMENT"
    back = {
        **blocker,
        "body": "New evidence: the clamp was removed.",
        "reintroduced_finding_hash": h,
    }
    _lanes(lanes, h, blocker, "DISPUTED", findings=[back])
    rid = _review(cfg, conn, gh, lanes, D)
    review = gh.posted_reviews[-1]
    assert review["event"] == "REQUEST_CHANGES"
    assert any("New evidence" in c["body"] for c in review["comments"])
    posted = conn.execute(
        "SELECT 1 FROM findings WHERE run_id=? AND hash=? AND stage='posted'", (rid, h)
    ).fetchone()
    assert posted
    # the revived finding's discussion starts over: its old answered dispute is forgotten
    assert kv_get(conn, f"dispute.answered:dashpay/platform#1:{h}") is None


def test_a_push_that_changes_the_disputed_code_is_new_evidence(cfg, conn, gh, lanes, monkeypatch):
    h, blocker = _disputed_round_one(cfg, conn, gh, lanes)
    gh.threads[0]["comments"]["nodes"].append(_gql_comment(_bot(gh.replies[0]["body"])))
    moved = []
    monkeypatch.setattr(
        worker, "_lines_changed", lambda ctx, a, b, file, ls, le: moved.append((a, b)) or True
    )
    _lanes(lanes, h, blocker, "STILL_VALID", reason="The partial fix still misses the clamp.")
    _review(cfg, conn, gh, lanes, C)
    assert moved and moved[0] == (C, HEAD)  # this head's lines, against the answered head
    assert gh.posted_reviews[-1]["event"] == "REQUEST_CHANGES"
    assert not conn.execute("SELECT 1 FROM findings WHERE stage='conceded'").fetchone()


def _second_thread(gh, h, title, root_id, replies, *, label="🟡 Suggestion"):
    gh.threads.append(_finding_thread(h, title, root_id, replies, label=label))
    gh.inline.append(
        {
            **gh.inline[0],
            "id": root_id,
            "node_id": f"N{root_id}",
            "html_url": f"https://github.com/dashpay/platform/pull/1#discussion_r{root_id}",
            "body": gh.threads[-1]["comments"]["nodes"][0]["body"],
        }
    )


def _run_conversation(cfg, conn, gh, lanes, sha, threads):
    from reviewsys.ingest import enqueue_head
    from reviewsys.models import Trigger
    from reviewsys.scheduler import schedule

    lanes.conversation = {"threads": threads}
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 1, sha, Trigger.REVIEW_REPLY)
    (rid,) = schedule(conn, cfg, spawn=False)
    assert worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False) == (
        RunStatus.DONE
    )


def test_discussion_never_approves_over_a_disputed_blocker(cfg, conn, gh, lanes):
    """The full review left a blocker DISPUTED; a later deferral of the last suggestion on the
    same commit lifts everything else, and still does not approve."""
    blocker, sugg = _blocking(), _sugg()
    bh, sh = _seed_prior(conn, blocker), _seed_prior(conn, sugg)
    _thread(gh, bh, blocker["title"], [DISPUTE])
    _second_thread(gh, sh, sugg["title"], 910, [])
    rows = [
        {"finding_hash": bh, "status": "STILL_VALID", "reason": ENGAGED, "disputed": True},
        {"finding_hash": sh, "status": "STILL_VALID", "reason": "Still missing."},
    ]
    both = [{**blocker, "finding_hash": bh}, {**sugg, "finding_hash": sh}]
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": both,
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": rows,
    }
    lanes.verifier["default"] = {**_verifier(both), "prior_finding_reconciliation": rows}
    _review(cfg, conn, gh, lanes, HEAD)
    gh.threads[0]["comments"]["nodes"].append(_gql_comment(_bot(gh.replies[0]["body"])))
    _review(cfg, conn, gh, lanes, C)
    assert gh.posted_reviews[-1]["event"] == "COMMENT"
    assert "### Disputed — left to maintainers (1)" in gh.posted_reviews[-1]["body"]
    tracked = "deferring; tracked in https://github.com/dashpay/platform/issues/4321"
    gh.threads[1]["comments"]["nodes"].append(_gql_comment({**KNST, "id": 911, "body": tracked}))
    defer = {
        "finding_hash": sh,
        "status": "INTENTIONALLY_DEFERRED",
        "reply": "Deferring it.",
        "tracking": {"url": "https://github.com/dashpay/platform/issues/4321"},
    }
    _run_conversation(cfg, conn, gh, lanes, C, [defer])
    assert "Tracked in https://github.com/dashpay/platform/issues/4321" in gh.replies[-1]["body"]
    assert all(r["event"] != "APPROVE" for r in gh.posted_reviews)
    last = conn.execute("SELECT event FROM reviews ORDER BY id DESC LIMIT 1").fetchone()
    assert last["event"] != "APPROVE"


def test_a_reply_on_a_disputed_thread_keeps_it_settled(cfg, conn, gh, lanes):
    h, _blocker = _disputed_round_one(cfg, conn, gh, lanes)
    gh.threads[0]["comments"]["nodes"].append(_gql_comment(_bot(gh.replies[0]["body"])))
    _lanes(lanes, h, _blocker, "STILL_VALID")
    _review(cfg, conn, gh, lanes, C)
    closing = gh.replies[-1]["body"]
    gh.threads[0]["comments"]["nodes"] += [
        _gql_comment(_bot(closing, id=951, at="y2")),
        _gql_comment({**KNST, "id": 903, "created_at": "z", "body": "Thanks, we'll decide."}),
    ]
    row = {"finding_hash": h, "status": "STILL_VALID", "reply": "Happy to help.", "disputed": True}
    _run_conversation(cfg, conn, gh, lanes, C, [row])
    prompt = next(c.prompt for c in lanes.calls if c.role == "conversation")
    assert '"already_settled_as": "DISPUTED"' in prompt
    assert "Happy to help." in gh.replies[-1]["body"]
    assert converse.DISPUTE_REPLY not in gh.replies[-1]["body"]  # not settled a second time
    root = gh.threads[0]["comments"]["nodes"][0]["body"]
    assert "⚖️ **Disputed** (left to the maintainers; not blocking)" in root


def test_the_contract_keeps_disputed_for_settled_disputes():
    raw = {
        **_verifier([]),
        "prior_finding_reconciliation": [{"finding_hash": "abc", "status": "DISPUTED"}],
    }
    with pytest.raises(ReviewError, match="DISPUTED is only for a prior finding already settled"):
        parse_verifier_output(raw, expected_phase="final", expected_coderabbit_ids=[])
    out = parse_verifier_output(
        raw, expected_phase="final", expected_coderabbit_ids=[], settled_hashes={"abc"}
    )
    assert out.prior_reconciliation[0]["status"] == "DISPUTED"


# ---- the conversation lane settles disputes the same way ----


def _conversation_dispute(gh, conn, *, answered: bool):
    blocker = _blocking()
    h = _seed_prior(conn, blocker, body="Estimation must be >= actual.")
    replies = [DISPUTE]
    if answered:
        replies += [
            _bot("<!-- thepastaclaw-thread-answer v1 sha=x reply=901 finding=y -->\nThe clamp."),
            {**KNST, "id": 902, "created_at": "z", "body": "Still not a bug."},
        ]
        with tx(conn):
            kv_set(conn, f"dispute.answered:dashpay/platform#1:{h}", '{"reply": "901"}')
    gh.threads = [_finding_thread(h, blocker["title"], 900, replies)]
    _standing_final(gh)
    return h


def test_conversation_records_the_first_answer_to_a_dispute(cfg, conn, gh, lanes):
    h = _conversation_dispute(gh, conn, answered=False)
    row = {"finding_hash": h, "status": "STILL_VALID", "reply": ENGAGED, "disputed": True}
    _converse(cfg, conn, gh, lanes, [row])
    assert ENGAGED in gh.replies[-1]["body"]
    assert json.loads(kv_get(conn, f"dispute.answered:dashpay/platform#1:{h}"))["reply"] == "901"
    assert len(gh.posted_reviews) == 1  # still blocking


def test_conversation_marks_a_dispute_answered_before_disputed(cfg, conn, gh, lanes):
    h = _conversation_dispute(gh, conn, answered=True)
    _converse(cfg, conn, gh, lanes, [{"finding_hash": h, "status": "NO_REPLY", "disputed": True}])
    prompt = next(c.prompt for c in lanes.calls if c.role == "conversation")
    assert '"dispute_answered": true' in prompt
    reply = gh.replies[-1]["body"]
    assert reply.startswith("<!-- thepastaclaw-thread-answer v1") and "reply=902" in reply
    assert converse.DISPUTE_REPLY in reply and reply.endswith(
        "_Disputed: left to the maintainers; this finding no longer blocks this pull request._"
    )
    row = conn.execute("SELECT body FROM findings WHERE stage='conceded'").fetchone()
    assert row["body"] == "DISPUTED"
    update = gh.posted_reviews[-1]
    assert update["event"] == "COMMENT"  # REQUEST_CHANGES lifted, but not approved
    assert "are disputed and left to the maintainers" in update["body"]


# ---- untracked deferrals: asked once ----


def _deferral_round_one(cfg, conn, gh, lanes, finding, *, label, reply):
    h = _seed_prior(conn, finding)
    _thread(gh, h, finding["title"], [reply], label=label)
    _lanes(lanes, h, finding, "INTENTIONALLY_DEFERRED", reason="Fine to do later.")
    _review(cfg, conn, gh, lanes, HEAD)
    (ask,) = gh.replies
    assert "**Needs a tracking issue** (re-reviewed at `aaaaaaaa`)" in ask["body"]
    assert "ask here and I'll file it" in ask["body"]
    assert kv_get(conn, f"defer.asked:dashpay/platform#1:{h}")
    gh.threads[0]["comments"]["nodes"].append(_gql_comment(_bot(ask["body"])))
    return h


@pytest.mark.parametrize(
    "restated, says",
    [
        ("STILL_VALID", converse.UNTRACKED_SUGGESTION_REPLY),  # the verifier forgot the deferral
        ("INTENTIONALLY_DEFERRED", "Fine to do later."),  # the verifier deferred it again
    ],
)
def test_a_suggestion_deferred_without_a_link_is_accepted_after_one_ask(
    cfg, conn, gh, lanes, restated, says
):
    """platform#5749: a suggestion deferred without a link was restated in ten summaries."""
    sugg = _sugg()
    author = {**KNST, "author": "someone", "association": "CONTRIBUTOR", "body": "later"}
    h = _deferral_round_one(cfg, conn, gh, lanes, sugg, label="🟡 Suggestion", reply=author)
    _lanes(lanes, h, sugg, restated, reason="Fine to do later.")
    _review(cfg, conn, gh, lanes, C)
    body = gh.posted_reviews[-1]["body"]
    assert sugg["title"] not in body and "Awaiting tracking link" not in body
    deferred = gh.replies[-1]["body"]
    assert f"**Deferred** (re-reviewed at `cccccccc`): {says}" in deferred
    root = gh.threads[0]["comments"]["nodes"][0]["body"]
    assert "⏭️ **Deferred**" in root and "Needs a tracking issue" not in root
    row = conn.execute(
        "SELECT body FROM findings WHERE stage='conceded' AND hash=? ORDER BY id DESC", (h,)
    ).fetchone()
    assert row["body"] == "INTENTIONALLY_DEFERRED"

    # settled: the next head restates nothing, holds nothing and asks nothing
    gh.threads[0]["comments"]["nodes"].append(_gql_comment(_bot(deferred, id=951, at="y2")))
    replies = len(gh.replies)
    _lanes(lanes, h, sugg, "INTENTIONALLY_DEFERRED", reason="Deferred.")
    _review(cfg, conn, gh, lanes, D)
    body = gh.posted_reviews[-1]["body"]
    assert sugg["title"] not in body and len(gh.replies) == replies
    assert not conn.execute(
        "SELECT 1 FROM events WHERE kind='defer.needs_tracking' AND run_id=?",
        (conn.execute("SELECT MAX(id) FROM runs").fetchone()[0],),
    ).fetchone()


def test_a_blocker_deferred_without_a_link_is_asked_once_then_listed_compactly(
    cfg, conn, gh, lanes
):
    """platform#5810: an untracked deferral got no thread reply and kept blocking."""
    blocker = _blocking()
    h = _deferral_round_one(
        cfg, conn, gh, lanes, blocker, label="🔴 Blocking", reply={**KNST, "body": "later"}
    )
    assert "⏸️ **Awaiting tracking link** (1)" in gh.posted_reviews[-1]["body"]
    _lanes(lanes, h, blocker, "STILL_VALID", reason="Still applies.")
    _review(cfg, conn, gh, lanes, C)
    review = gh.posted_reviews[-1]
    assert review["event"] == "REQUEST_CHANGES" and len(gh.replies) == 1  # not asked again
    body = review["body"]
    assert (
        f"⏸️ **Awaiting tracking link** (1): {blocker['title']} ([thread]({URL})). Link an "
        "issue on its thread, or ask there and PastaClaw files it."
    ) in body
    assert "Still open from earlier rounds" not in body
    assert body.count(blocker["title"]) == 2  # the line above and the agent-prompt entry
    assert "Estimation must be >= actual." not in body


def test_a_blocker_whose_tracking_turns_up_later_is_deferred(cfg, conn, gh, lanes):
    blocker = _blocking()
    h = _deferral_round_one(
        cfg, conn, gh, lanes, blocker, label="🔴 Blocking", reply={**KNST, "body": "later"}
    )
    gh.pr["body"] = "Follow-up tracked in https://github.com/dashpay/platform/issues/77"
    with tx(conn):
        kv_set(conn, f"defer.issue:dashpay/platform#1:{h}", "https://x/issues/77")
    _lanes(lanes, h, blocker, "STILL_VALID")
    _review(cfg, conn, gh, lanes, C)
    assert gh.posted_reviews[-1]["event"] != "REQUEST_CHANGES"
    assert "Tracked in https://x/issues/77; deferring this." in gh.replies[-1]["body"]


def test_conversation_accepts_a_suggestion_deferral_after_one_ask(cfg, conn, gh, lanes):
    s = _sugg()
    h = _seed_prior(conn, s)
    author = {**KNST, "author": "someone", "association": "CONTRIBUTOR", "body": "no, later"}
    gh.threads = [_finding_thread(h, s["title"], 900, [author], label="🟡 Suggestion")]
    _standing_final(gh)
    with tx(conn):
        kv_set(conn, f"defer.asked:dashpay/platform#1:{h}", '{"reply": "800"}')
    ask = {"finding_hash": h, "status": "DEFERRAL_NEEDS_TRACKING", "reply": "Link one?"}
    _converse(cfg, conn, gh, lanes, [ask])
    reply = gh.replies[-1]["body"]
    assert converse.UNTRACKED_SUGGESTION_REPLY in reply and "Link one?" not in reply
    row = conn.execute("SELECT body FROM findings WHERE stage='conceded'").fetchone()
    assert row["body"] == "INTENTIONALLY_DEFERRED"


# ---- the review body ----


def _model(kept, suppressed, **kw) -> publish.ReviewModel:
    verified = VerifierOutput(
        summary="New on this head.",
        review_action="COMMENT",
        findings=[*kept, *(s.finding for s in suppressed)],
        dropped=[],
        out_of_scope=[],
        coderabbit_reactions=[],
        prerequisite_adjudications=[],
        adjudication_complete=True,
        review_phase="final",
        raw={},
    )
    prov = publish.Provenance(
        reviewers=[],
        verifier={"model": "m", "agent": "a", "role": "final-verifier"},
        policy_fingerprint="f" * 64,
    )
    return publish.ReviewModel(
        repo="dashpay/platform",
        number=1,
        head_sha=HEAD,
        phase="final",
        verified=verified,
        provenance=prov,
        kept=kept,
        skipped=[],
        suppressed=suppressed,
        comments=[],
        rereview=True,
        **kw,
    )


def test_carried_findings_are_listed_compactly_not_described_again():
    old = Finding.from_dict({**_blocking(), "body": "OLD PROSE"})
    old.prior_hash = old.hash
    new = Finding.from_dict({**_sugg("Fresh one"), "body": "FRESH PROSE"})
    held = Finding.from_dict({**_blocking("Held"), "body": "HELD PROSE"})
    held.extra["awaiting_tracking"] = True
    gone = Finding.from_dict(_blocking("Gone"))
    gone.extra["thread_url"] = "https://t/9"
    m = _model(
        [new],
        [
            Suppressed(old, "deduped_existing_open_thread", html_url=URL),
            Suppressed(held, "deduped_existing_open_thread", html_url="https://t/2"),
        ],
        disputed=[gone],
    )
    body = publish.render(m)
    assert "OLD PROSE" not in body and "HELD PROSE" not in body and "FRESH PROSE" in body
    assert (
        f"### Still open from earlier rounds (1)\n\n- 🔴 **Blocking**: {old.title} — "
        f"`f.rs:11-12` ([thread]({URL}))"
    ) in body
    assert "### Disputed — left to maintainers (1)\n\n- 🔴 **Blocking**: Gone" in body
    assert "⏸️ **Awaiting tracking link** (1): Held ([thread](https://t/2))." in body
    assert "carried-forward finding(s)" not in body
    prompt = body.split("```\n", 1)[1]
    assert f"- [BLOCKING] f.rs:11-12: {old.title}\n  (carried; thread: {URL})" in prompt
    assert "- [BLOCKING] f.rs:11-12: Held\n  (awaiting a tracking link" in prompt
    assert "Gone" not in prompt  # disputed: not a finding to act on

    # a later lift marks its compact line, and the edit is undone with it
    lifted = publish.LiftedFinding(
        hash=old.hash, severity="blocking", file="f.rs", title=old.title, status="WITHDRAWN"
    )
    edited = publish.with_lifted_findings(body, [lifted])
    assert f"([thread]({URL})) — **settled:** ✅ Withdrawn" in edited
    assert publish.with_lifted_findings(edited, []) == body


def test_the_agent_prompt_guides_author_agents():
    f = Finding.from_dict(_sugg())
    prompt = publish._ai_prompt([f], [])
    assert prompt is not None
    guidance = prompt.split("\n\n", 1)[0].splitlines()
    assert guidance == list(publish.AI_PROMPT_GUIDANCE) and len(guidance) <= 10
    text = "\n".join(guidance)
    for needle in (
        "fix only what is real, as narrowly as the finding",
        "decline it on its thread in one or two sentences; the bot answers once and stops",
        "link a tracking issue on its thread (or reply asking the bot to file one)",
        "Suggestions are optional: fix them forward in a later PR",
        "one push: self-review the full diff first, and don't push piecemeal",
        "Keep the PR's scope; don't grow it to satisfy hardening suggestions",
        "Disputed and deferred items are already handled",
    ):
        assert needle in text
    assert prompt.endswith(f"  {f.body}")


def test_the_verifier_is_told_to_leave_carried_findings_out_of_its_summary(cfg, conn, gh, lanes):
    _disputed_round_one(cfg, conn, gh, lanes)
    assert "do not describe them again" in _verifier_prompts(lanes)[0]
    assert not any("do not describe them again" in c.prompt for c in _reviewer_calls(lanes))


def test_a_withdrawal_after_a_dispute_no_longer_holds_the_approval(cfg, conn, gh, lanes):
    """A blocker settled DISPUTED that a later conversation on the same commit withdraws stops
    counting: only each finding's newest settlement matters."""
    blocker, sugg = _blocking(), _sugg()
    bh, sh = _seed_prior(conn, blocker), _seed_prior(conn, sugg)
    _thread(gh, bh, blocker["title"], [DISPUTE])
    _second_thread(gh, sh, sugg["title"], 910, [])
    rows = [
        {"finding_hash": bh, "status": "STILL_VALID", "reason": ENGAGED, "disputed": True},
        {"finding_hash": sh, "status": "STILL_VALID", "reason": "Still missing."},
    ]
    both = [{**blocker, "finding_hash": bh}, {**sugg, "finding_hash": sh}]
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": both,
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": rows,
    }
    lanes.verifier["default"] = {**_verifier(both), "prior_finding_reconciliation": rows}
    _review(cfg, conn, gh, lanes, HEAD)
    gh.threads[0]["comments"]["nodes"].append(_gql_comment(_bot(gh.replies[0]["body"])))
    _review(cfg, conn, gh, lanes, C)
    assert gh.posted_reviews[-1]["event"] == "COMMENT"
    gh.threads[0]["comments"]["nodes"] += [
        _gql_comment(_bot(gh.replies[-1]["body"], id=951, at="y2")),
        _gql_comment(
            {**KNST, "id": 912, "created_at": "z", "body": "Proof the path is unreachable: #5200."}
        ),
    ]
    tracked = "deferring; tracked in https://github.com/dashpay/platform/issues/4321"
    gh.threads[1]["comments"]["nodes"].append(_gql_comment({**KNST, "id": 911, "body": tracked}))
    outcomes = [
        {"finding_hash": bh, "status": "WITHDRAWN", "reply": "You're right; withdrawn."},
        {
            "finding_hash": sh,
            "status": "INTENTIONALLY_DEFERRED",
            "reply": "Deferring it.",
            "tracking": {"url": "https://github.com/dashpay/platform/issues/4321"},
        },
    ]
    _run_conversation(cfg, conn, gh, lanes, C, outcomes)
    last = conn.execute("SELECT event FROM reviews ORDER BY id DESC LIMIT 1").fetchone()
    assert last["event"] == "APPROVE"
