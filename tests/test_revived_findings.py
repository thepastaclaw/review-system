"""A finding the review judged fixed does not come back on unchanged code unconfirmed.

dashpay/dash-evo-tool#1112: a blocker all three passes judged FIXED came back 15 days later as
a carried blocker from a newer model, on byte-identical code, and stayed for five rounds.
#1213: a finding judged FIXED and resolved came back STILL_VALID after a reply on its resolved
thread. #1173 and #1165: a follow-up candidate was listed again under a reworded title, on the
same lines or on lines unrelated inserts had shifted."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from reviewsys import late, prompts, publish, worker
from reviewsys.contract import Finding
from reviewsys.db import kv_set, tx
from tests.test_late import _f, _lines, _round, _stages, _with, repo, statuses  # noqa: F401
from tests.test_ledger import _commit, _git
from tests.test_settle_disputes import _thread
from tests.test_worker_e2e import (  # fake_git: autouse, fakes the mirror and worktree
    KNST,
    _blocking,
    _gql_comment,
    _reviewer_calls,
    _run,
    _seed_prior,
    _seed_withdrawn_prior,
    _sugg,
    _verifier,
    fake_git,  # noqa: F401
)

C, D, E = "c" * 40, "d" * 40, "e" * 40
B = {**_blocking("Race between the refresh and the wallet lock"), "category": "bug"}
S = {**_sugg("Cover the refresh race in a test"), "line_start": 20, "line_end": 20}
STILL = "The refresh still races the lock."


@pytest.fixture
def unchanged(monkeypatch, statuses):  # noqa: F811
    """The lines under every finding unchanged since any earlier head (a flag flips it)."""
    moved = {"value": False}
    monkeypatch.setattr(late.CodeHistory, "moved", lambda self, since, f: moved["value"])
    return moved


def _fixed(cfg, conn, gh, lanes, finding=B, *, thread=True):
    """Round 1 raises `finding` at HEAD; round 2 (at C) judges it FIXED."""
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["preliminary"] = _verifier([])
    lanes.verifier["final"] = _verifier([finding])
    _run(cfg, conn, gh, lanes)
    h = Finding.from_dict(finding).hash
    if thread:
        _thread(gh, h, finding["title"], [])
    _round(cfg, conn, gh, lanes, C, [], [{"finding_hash": h, "status": "FIXED", "reason": "ok"}])
    return h


def _raised_again(cfg, conn, gh, lanes, h, sha, finding=B, *, reason=STILL, **row):
    recon = [{"finding_hash": h, "status": "STILL_VALID", "reason": reason, **row}]
    return _round(cfg, conn, gh, lanes, sha, [{**finding, "finding_hash": h}], recon)


def _judged_rows(conn, h):
    return [
        (r["run_id"], r["body"])
        for r in conn.execute(
            "SELECT run_id, body FROM findings WHERE hash=? AND stage=? ORDER BY id",
            (h, late.STAGE_FIXED),
        )
    ]


def _prompts(lanes, role):
    return [c.prompt for c in lanes.calls if c.role == role]


def test_a_fixed_judgement_is_kept_and_shown_to_the_next_round(cfg, conn, gh, lanes, unchanged):
    h = _fixed(cfg, conn, gh, lanes)
    (fixed_run,) = [r for (r,) in conn.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 1")]
    assert _judged_rows(conn, h) == [(fixed_run, f"FIXED\n{C}")]
    # an old reader (the previous code) sees no settlement in it
    ctx = SimpleNamespace(conn=conn, repo="dashpay/platform", number=1)
    assert worker._latest_settlement(ctx, h) is None
    assert worker._judged_fixed(ctx, h) == {"status": "FIXED", "sha": C}

    calls = len(lanes.calls)
    _round(cfg, conn, gh, lanes, D, [], [{"finding_hash": h, "status": "FIXED", "reason": "ok"}])
    reviewer = _reviewer_calls(SimpleNamespace(calls=lanes.calls[calls:]))[0].prompt
    for p in (reviewer, _prompts(lanes, "verifier")[-1]):
        assert '"judged_fixed": {' in p and f'"at": "{C}"' in p
        assert prompts.JUDGED_FIXED_RULE.replace("\n", "") in p.replace("\n", "")
    # judged again at D: the newer head is the one it was judged at
    assert _judged_rows(conn, h)[-1][1] == f"FIXED\n{D}"


def test_raised_again_on_unchanged_lines_it_needs_confirmation_and_stays_settled_without(
    cfg, conn, gh, lanes, unchanged
):
    """#1112: judged FIXED at C, raised again at D as a carried blocker on lines unchanged
    since: the confirmation lane is asked, and without a reachable failure it does not count,
    is not carried, and is listed once as a follow-up candidate."""
    h = _fixed(cfg, conn, gh, lanes)
    lanes.confirm = {B["title"]: "non_blocking"}
    rid = _raised_again(cfg, conn, gh, lanes, h, D)
    (confirm,) = _prompts(lanes, "confirm")
    assert f'"judged_fixed_at": "{C}"' in confirm and '"lines_unchanged_since": true' in confirm
    assert "raises again one that an earlier review of this pull request judged fixed" in confirm
    review = gh.posted_reviews[-1]
    assert review["event"] == "APPROVE" and not review["comments"]
    followups = review["body"].split("Follow-up candidates (optional) (1)", 1)[1]
    assert B["title"] in followups
    assert (
        "1 finding(s) judged fixed earlier and raised again on unchanged lines kept settled"
        in (review["body"])
    )
    stages = _stages(conn, rid)
    assert ("verified", B["title"]) not in stages
    assert ("late-unconfirmed", B["title"]) in stages and ("followup", B["title"]) in stages
    # it stays judged fixed at C, recorded again by this run
    assert _judged_rows(conn, h)[-1] == (rid, f"FIXED\n{C}")

    # raised again at E: the earlier verdict stands, it is not listed again, nothing blocks
    calls = len(lanes.calls)
    rid = _raised_again(cfg, conn, gh, lanes, h, E)
    assert f'"at": "{C}"' in _prompts(lanes, "verifier")[-1]
    assert not [c for c in lanes.calls[calls:] if c.role == "confirm"]
    review = gh.posted_reviews[-1]
    assert review["event"] == "APPROVE" and "Follow-up candidates" not in review["body"]
    assert "1 follow-up candidate(s) already listed by an earlier review" in review["body"]
    assert _judged_rows(conn, h)[-1] == (rid, f"FIXED\n{C}")


def test_confirmed_it_stands_and_its_thread_says_why(cfg, conn, gh, lanes, unchanged):
    h = _fixed(cfg, conn, gh, lanes)
    replies = len(gh.replies)
    rid = _raised_again(cfg, conn, gh, lanes, h, D)
    sentence = (
        "Judged fixed at `cccccccc`; raised again and confirmed by an independent pass: "
        "a reachable failure"
    )
    assert gh.posted_reviews[-1]["event"] == "REQUEST_CHANGES"
    (answer,) = gh.replies[replies:]
    assert f"**Still applies** (re-reviewed at `dddddddd`): {sentence}" in answer["body"]
    assert STILL in answer["body"]
    assert _stages(conn, rid)[("verified", B["title"])][0] == "blocking"
    body = conn.execute(
        "SELECT body FROM findings WHERE run_id=? AND stage='verified' AND hash=?", (rid, h)
    ).fetchone()["body"]
    assert body.endswith(f"_{sentence}_")
    ctx = SimpleNamespace(conn=conn, repo="dashpay/platform", number=1)
    assert worker._judged_fixed(ctx, h) is None  # standing again: an ordinary prior finding


def test_on_lines_changed_since_it_takes_the_normal_path(cfg, conn, gh, lanes, unchanged):
    h = _fixed(cfg, conn, gh, lanes)
    unchanged["value"] = True
    rid = _raised_again(cfg, conn, gh, lanes, h, D)
    assert not _prompts(lanes, "confirm")
    assert gh.posted_reviews[-1]["event"] == "REQUEST_CHANGES"
    assert _stages(conn, rid)[("verified", B["title"])][0] == "blocking"


def test_a_failed_confirmation_keeps_it_as_raised(cfg, conn, gh, lanes, unchanged):
    """As for any new blocker of a later round: a failed (or degraded) confirmation lane never
    costs a real one."""
    h = _fixed(cfg, conn, gh, lanes)
    lanes.confirm = lambda reported: {"decisions": [{"id": "c1", "verdict": "blocking"}]}
    rid = _raised_again(cfg, conn, gh, lanes, h, D)
    review = gh.posted_reviews[-1]
    assert review["event"] == "REQUEST_CHANGES"
    assert (
        "1 new blocker(s) stand as raised: the independent confirmation pass failed"
        in (review["body"])
    )
    assert ("verified", B["title"]) in _stages(conn, rid)


def test_a_reply_on_its_resolved_thread_is_answered_and_reopens_nothing_unconfirmed(
    cfg, conn, gh, lanes, unchanged
):
    """#1213: judged FIXED and resolved; the author replies on the resolved thread and the next
    round says STILL_VALID on unchanged lines. The reply is answered, the finding stays
    settled, and an earlier answered dispute does not turn it into a DISPUTED blocker."""
    h = _fixed(cfg, conn, gh, lanes)
    gh.threads[0]["isResolved"] = True
    reply = {**KNST, "body": "Per maintainer decision, not changing this."}
    gh.threads[0]["comments"]["nodes"].append(_gql_comment(reply))
    with tx(conn):
        memo = {"reply": "901", "standing": True, "sha": C}
        kv_set(conn, f"dispute.answered:dashpay/platform#1:{h}", json.dumps(memo))
    lanes.confirm = {B["title"]: "non_blocking"}
    replies = len(gh.replies)
    _raised_again(cfg, conn, gh, lanes, h, D, disputed=True)
    review = gh.posted_reviews[-1]
    assert review["event"] == "APPROVE" and "Disputed" not in review["body"]
    (answer,) = gh.replies[replies:]
    assert "**Resolved**" in answer["body"]
    assert (
        "Judged fixed at `cccccccc`, and the lines it cites have not changed since"
        in (answer["body"])
    )
    assert "so it stays settled: read the code" in answer["body"]
    assert not conn.execute(
        "SELECT 1 FROM findings WHERE hash=? AND stage='conceded'", (h,)
    ).fetchone()


def test_the_conversation_lane_answers_a_fixed_finding_as_settled(cfg, conn, gh, lanes, unchanged):
    """A reply on the resolved thread of a finding judged fixed, answered on the same commit:
    the answer neither re-counts it nor takes its resolved status off the thread."""
    from tests.test_settle_disputes import _run_conversation

    h = _fixed(cfg, conn, gh, lanes)
    root = gh.threads[0]["comments"]["nodes"][0]
    root["body"] = publish.with_thread_status(root["body"], status="FIXED", head_sha=C)
    gh.inline[0]["body"] = root["body"]
    reviews = len(gh.posted_reviews)
    gh.threads[0]["isResolved"] = True
    gh.threads[0]["comments"]["nodes"].append(
        _gql_comment({**KNST, "body": "Per maintainer decision, not changing this."})
    )
    lanes.calls.clear()
    still = {"finding_hash": h, "status": "STILL_VALID", "reply": "The race is still there."}
    _run_conversation(cfg, conn, gh, lanes, C, [still])
    (prompt,) = _prompts(lanes, "conversation")
    assert '"already_settled_as": "FIXED"' in prompt
    assert "The race is still there." in gh.replies[-1]["body"]
    assert "✅ **Resolved**" in root["body"]
    assert len(gh.posted_reviews) == reviews  # no verdict moved
    ctx = SimpleNamespace(conn=conn, repo="dashpay/platform", number=1)
    assert worker._judged_fixed(ctx, h) == {"status": "FIXED", "sha": C}


def test_a_non_blocking_finding_raised_again_is_not_carried(cfg, conn, gh, lanes, unchanged):
    h = _fixed(cfg, conn, gh, lanes, S)
    rid = _raised_again(cfg, conn, gh, lanes, h, D, S)
    assert not _prompts(lanes, "confirm")
    review = gh.posted_reviews[-1]
    assert review["event"] == "APPROVE" and S["title"] not in review["body"]
    assert ("late-settled", S["title"]) in _stages(conn, rid)
    assert _judged_rows(conn, h)[-1] == (rid, f"FIXED\n{C}")


def test_without_a_fixed_judgement_on_record_nothing_changes(cfg, conn, gh, lanes, unchanged):
    """A database from before: a carried blocker on unchanged lines is carried as it was."""
    h = _seed_prior(conn, B)
    _thread(gh, h, B["title"], [])
    rid = _raised_again(cfg, conn, gh, lanes, h, D)
    assert not _prompts(lanes, "confirm")
    assert "judged_fixed" not in _prompts(lanes, "verifier")[-1]
    assert gh.posted_reviews[-1]["event"] == "REQUEST_CHANGES"
    assert _stages(conn, rid)[("verified", B["title"])][0] == "blocking"


def test_a_withdrawal_is_not_overwritten_by_a_later_fixed(cfg, conn, gh, lanes, unchanged):
    """A WITHDRAWN tombstone stays the finding's settlement when a later round calls it FIXED:
    a judged-fixed row is written only for a finding no judgement lift settled."""
    h = _seed_withdrawn_prior(conn, B)
    _thread(gh, h, B["title"], [])
    _round(cfg, conn, gh, lanes, D, [], [{"finding_hash": h, "status": "FIXED", "reason": "ok"}])
    assert _judged_rows(conn, h) == []
    ctx = SimpleNamespace(conn=conn, repo="dashpay/platform", number=1)
    assert worker._latest_settlement(ctx, h) == "WITHDRAWN"


def test_the_split_confirms_a_revived_blocker_and_settles_the_rest():
    judged = {"hash": "h", "status": "FIXED", "sha": C}
    blocker = _with(_f("f.rs", 3, title="blocker", severity="blocking"), late.SINCE_FIRST)
    sugg = _with(_f("f.rs", 4, title="sugg", severity="suggestion"), late.SINCE_FIRST)
    other = _with(_f("f.rs", 9, title="other", severity="blocking"), late.SINCE_FIRST)
    other.extra["judged_fixed"] = judged  # what a lane wrote is never taken as the record
    split = late.split(
        [blocker, sugg, other],
        posted=lambda f: "blocking",
        listed=set(),
        revived=lambda f: judged if f.title != "other" else None,
    )
    assert split.candidates == [blocker] and split.settled == [sugg] and split.keep == [other]
    assert blocker.extra["judged_fixed"] == judged and "judged_fixed" not in other.extra
    (brief,) = late.candidate_briefs([blocker])
    assert brief["judged_fixed_at"] == C and brief["lines_unchanged_since"] is True


def test_the_reuse_note_on_a_resolved_thread_says_it_was_judged_fixed(gh):
    f = Finding.from_dict(B)
    f.extra["revived_note"] = "Judged fixed at `cccccccc`; raised again and confirmed: boom"
    gh.inline = [
        {
            "id": 900,
            "node_id": "N900",
            "user": {"login": "thepastaclaw"},
            "path": "f.rs",
            "line": 12,
            "html_url": "u",
            "body": f"<!-- thepastaclaw-review v1 finding={f.hash} dedupe={f.dedupe_key} -->\n"
            f"**🔴 Blocking: {f.title}**\n\nold body",
        }
    ]
    original = gh._dispatch

    def resolved(args, stdin):
        if args[:2] == ["api", "graphql"] and any("node(id:" in a for a in args):
            node = {"id": "PRRT_1", "isResolved": True, "comments": {"nodes": []}}
            return {"data": {"node": {"pullRequestReviewThread": node}}}
        return original(args, stdin)

    gh._dispatch = resolved
    kept, sup = publish.dedupe_against_github(
        gh, "dashpay/platform", 1, D, [f], "thepastaclaw", dry_run=False
    )
    assert not kept and sup[0].action == "replied_to_resolved_thread"
    body = gh.replies[-1]["body"]
    assert "Reusing this existing thread on `dddddddd`. Judged fixed at `cccccccc`" in body
    assert "please mark the thread resolved" not in body


def test_the_rule_is_shown_only_when_a_prior_finding_was_judged_fixed():
    prior = [{"finding_hash": "h", "original_title": "t"}]
    assert prompts.JUDGED_FIXED_RULE not in prompts.prior_findings_block(prior, C)
    judged = [{**prior[0], "judged_fixed": {"status": "FIXED", "at": C}}]
    assert prompts.JUDGED_FIXED_RULE in prompts.prior_findings_block(judged, C)


# ---- follow-up candidates: a repeat by its place, whatever its title ----


def test_a_listed_follow_up_is_recognised_by_its_place(repo):  # noqa: F811
    listed_at = _commit(repo, {"m.rs": "\n".join(_lines(60)) + "\n"})
    # #1165: unrelated inserts above it shift its lines
    shifted = ["inserted line, unrelated to the follow-up"] * 6
    head = _commit(repo, {"m.rs": "\n".join(shifted + _lines(60)) + "\n"})
    spots = late.listed_spots(repo, head, [("m.rs", 30, 34, listed_at)])
    assert spots == [("m.rs", 36, 40)]
    # #1173: the same lines, reworded
    same = _f("m.rs", 36, 40, title="Exercise the refusal through the real stdin entry point")
    assert late.is_listed(same, set(), spots)
    # shifted and a little wider
    assert late.is_listed(_f("m.rs", 38, 44, title="Reworded"), set(), spots)
    # a different follow-up elsewhere in the file, or in another file, is not a repeat
    assert not late.is_listed(_f("m.rs", 5, 8, title="Something else"), set(), spots)
    assert not late.is_listed(_f("n.rs", 36, 40, title="Reworded"), set(), spots)
    # the listed lines themselves were rewritten: no longer the place it was listed at
    lines = _lines(60)
    lines[31] = "rewritten since the listing"
    rewritten = _commit(repo, {"m.rs": "\n".join(lines) + "\n"})
    assert late.listed_spots(repo, rewritten, [("m.rs", 30, 34, listed_at)]) == []
    # git cannot place it
    assert late.listed_spots(repo, head, [("m.rs", 30, 34, "0" * 40)]) == []
    assert _git(repo, "rev-parse", "HEAD") == rewritten


def test_a_listed_suggestion_raised_again_as_a_blocker_goes_to_confirmation():
    """The severity does not make it new: it is confirmed first (#79's rule for any new
    blocker), and if the confirmation demotes it, it is not listed a second time."""
    spots = [("f.rs", 10, 12)]
    raised = _with(_f("f.rs", 11, title="Now a blocker", severity="blocking"), late.SINCE_FIRST)
    split = late.split([raised], posted=lambda f: None, listed=set(), spots=spots)
    assert split.candidates == [raised] and not split.repeats
    demoted = _with(_f("f.rs", 11, title="Now a blocker", severity="suggestion"), late.SINCE_FIRST)
    assert late.is_listed(demoted, set(), spots)
    sugg = _with(_f("f.rs", 12, title="Reworded again", severity="suggestion"), late.SINCE_FIRST)
    assert late.split([sugg], posted=lambda f: None, listed=set(), spots=spots).repeats == [sugg]
