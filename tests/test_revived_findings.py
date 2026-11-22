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
from reviewsys.models import RunStatus
from tests.test_late import _f, _lines, _round, _stages, _with, repo, statuses  # noqa: F401
from tests.test_ledger import _commit, _git
from tests.test_settle_disputes import _thread
from tests.test_worker_e2e import (  # fake_git: autouse, fakes the mirror and worktree
    KNST,
    _blocking,
    _gql_comment,
    _queue_new_head,
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
    monkeypatch.setattr(late.CodeHistory, "changed_since", lambda self, since, path: moved["value"])
    return moved


def _judged(conn, h):
    ctx = SimpleNamespace(conn=conn, repo="dashpay/platform", number=1)
    judged = worker._judged_fixed(ctx, h)
    return judged and {k: v for k, v in judged.items() if k != "since"}


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
    assert _judged(conn, h) == {"status": "FIXED", "sha": C}

    calls = len(lanes.calls)
    _round(cfg, conn, gh, lanes, D, [], [{"finding_hash": h, "status": "FIXED", "reason": "ok"}])
    reviewer = _reviewer_calls(SimpleNamespace(calls=lanes.calls[calls:]))[0].prompt
    for p in (reviewer, _prompts(lanes, "verifier")[-1]):
        assert '"judged_fixed": {' in p and f'"at": "{C}"' in p
        assert prompts.JUDGED_FIXED_RULE.replace("\n", "") in p.replace("\n", "")
    # judged again at D: the newer head is the one it was judged at
    assert _judged_rows(conn, h)[-1][1] == f"FIXED\n{D}"
    # judged again on the same head: nothing new to write
    rows = len(_judged_rows(conn, h))
    _round(cfg, conn, gh, lanes, D, [], [{"finding_hash": h, "status": "FIXED", "reason": "ok"}])
    assert len(_judged_rows(conn, h)) == rows


def test_a_judged_fixed_prior_is_shown_by_an_excerpt():
    f = Finding.from_dict({**B, "body": "x" * 5000})
    f.extra["judged_fixed"] = {"status": "FIXED", "sha": C, "since": "t"}
    (d,) = prompts.prior_for_prompt([f], C)
    assert d["judged_fixed"] == {"status": "FIXED", "at": C}
    assert len(d["body"]) == prompts.JUDGED_FIXED_EXCERPT
    f.extra["thread_replies"] = [{"author": "knst", "body": "no"}]
    assert len(prompts.prior_for_prompt([f], C)[0]["body"]) == 4000


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
    # it stays judged fixed at C, written again with C by this run
    assert _judged_rows(conn, h)[-1] == (rid, f"FIXED\n{C}")
    assert _judged(conn, h) == {"status": "FIXED", "sha": C}

    # raised again at E: the earlier verdict stands, it is not listed again, nothing blocks
    calls = len(lanes.calls)
    rid = _raised_again(cfg, conn, gh, lanes, h, E)
    assert f'"at": "{C}"' in _prompts(lanes, "verifier")[-1]
    assert not [c for c in lanes.calls[calls:] if c.role == "confirm"]
    review = gh.posted_reviews[-1]
    assert review["event"] == "APPROVE" and "Follow-up candidates" not in review["body"]
    assert "1 follow-up candidate(s) already listed by an earlier review" in review["body"]
    assert _judged(conn, h) == {"status": "FIXED", "sha": C}


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
    assert _judged(conn, h) is None  # standing again: an ordinary prior finding


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


def test_a_reply_saying_it_is_not_fixed_gets_it_checked_again(cfg, conn, gh, lanes, unchanged):
    """A wrong FIXED, and a revival the confirmation lane did not confirm: a human replies on
    the resolved thread with a reproduction. The conversation lane's answer keeps it resolved
    and says the next full review re-checks it; that review asks the lane afresh instead of
    reusing its earlier verdict, and shows it the reply."""
    from tests.test_settle_disputes import _run_conversation

    h = _fixed(cfg, conn, gh, lanes)
    lanes.confirm = {B["title"]: "non_blocking"}
    _raised_again(cfg, conn, gh, lanes, h, D)
    assert len(_prompts(lanes, "confirm")) == 1
    root = gh.threads[0]["comments"]["nodes"][0]
    root["body"] = publish.with_thread_status(root["body"], status="FIXED", head_sha=C)
    gh.inline[0]["body"] = root["body"]
    gh.threads[0]["isResolved"] = True
    repro = "Not fixed: run `det-cli refresh & det-cli lock` twice and the wallet deadlocks."
    gh.threads[0]["comments"]["nodes"].append(_gql_comment({**KNST, "body": repro}))
    reviews = len(gh.posted_reviews)
    still = {"finding_hash": h, "status": "STILL_VALID", "reply": "The race is still there."}
    calls = len(lanes.calls)
    _run_conversation(cfg, conn, gh, lanes, D, [still])
    (prompt,) = [c.prompt for c in lanes.calls[calls:] if c.role == "conversation"]
    assert '"already_settled_as": "FIXED"' in prompt
    answer = gh.replies[-1]["body"]
    assert answer.rstrip().endswith(worker.STAYS_RESOLVED)
    assert "The race is still there." in answer
    assert "✅ **Resolved**" in root["body"]
    assert len(gh.posted_reviews) == reviews  # no verdict moved
    assert _judged(conn, h) == {"status": "FIXED", "sha": C}

    lanes.confirm = {}
    calls = len(lanes.calls)
    _raised_again(cfg, conn, gh, lanes, h, E)
    (confirm,) = [c.prompt for c in lanes.calls[calls:] if c.role == "confirm"]
    assert "replies_since" in confirm and "det-cli refresh" in confirm
    assert gh.posted_reviews[-1]["event"] == "REQUEST_CHANGES"


def test_a_same_head_re_review_asks_the_confirmation_lane_again(cfg, conn, gh, lanes, unchanged):
    h = _fixed(cfg, conn, gh, lanes)
    lanes.confirm = {B["title"]: "non_blocking"}
    _raised_again(cfg, conn, gh, lanes, h, D)
    _raised_again(cfg, conn, gh, lanes, h, E)
    assert len(_prompts(lanes, "confirm")) == 1  # a new head, nothing said since: reused
    _raised_again(cfg, conn, gh, lanes, h, E)
    assert len(_prompts(lanes, "confirm")) == 2


def test_a_non_blocking_finding_raised_again_is_not_carried(cfg, conn, gh, lanes, unchanged):
    h = _fixed(cfg, conn, gh, lanes, S)
    rid = _raised_again(cfg, conn, gh, lanes, h, D, S)
    assert not _prompts(lanes, "confirm")
    review = gh.posted_reviews[-1]
    assert review["event"] == "APPROVE" and S["title"] not in review["body"]
    assert ("late-settled", S["title"]) in _stages(conn, rid)
    assert _judged(conn, h) == {"status": "FIXED", "sha": C}


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


# ---- what may supersede the memory ----


def _unfinished_run_keeping(conn, finding, sha, status="cancelled"):
    """A run of a later head that wrote its final verifier set (carrying `finding`) and then
    ended `status` without publishing: a push superseded it during the confirmation lane."""
    f = Finding.from_dict(finding)
    with tx(conn):
        hid = conn.execute(
            "INSERT INTO heads (repo, number, sha, trigger, priority, status, queued_at, "
            "eligible_at) VALUES ('dashpay/platform', 1, ?, 'push', 0, 'superseded', 't', 't')",
            (sha,),
        ).lastrowid
        rid = conn.execute(
            "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at) "
            "VALUES (?, 1, ?, 't', '2099-01-01T00:00:00Z', 't')",
            (hid, status),
        ).lastrowid
        for phase in ("verify1", "verify2"):
            conn.execute(
                "INSERT INTO findings (run_id, phase, stage, hash, file, line_start, line_end, "
                "severity, category, title, body) VALUES (?,?,'verified',?,?,?,?,?,?,?,'b')",
                (rid, phase, f.hash, f.file, 11, 12, f.severity, f.category, f.title),
            )
    return rid


def test_a_run_that_never_finished_does_not_erase_a_fixed_judgement(
    cfg, conn, gh, lanes, unchanged
):
    """#1112 again otherwise: a run writes its verifier set before the confirmation lane, and
    a push that supersedes it there would leave a STILL_VALID carry as the newest record."""
    h = _fixed(cfg, conn, gh, lanes)
    _unfinished_run_keeping(conn, B, D)
    assert _judged(conn, h) == {"status": "FIXED", "sha": C}
    lanes.confirm = {B["title"]: "non_blocking"}
    _raised_again(cfg, conn, gh, lanes, h, E)
    assert len(_prompts(lanes, "confirm")) == 1
    assert gh.posted_reviews[-1]["event"] == "APPROVE"


def test_a_run_that_never_finished_does_not_lift_a_withdrawal(conn):
    h = _seed_withdrawn_prior(conn, B)
    ctx = SimpleNamespace(conn=conn, repo="dashpay/platform", number=1)
    _unfinished_run_keeping(conn, B, D, status="failed")
    assert worker._latest_settlement(ctx, h) == "WITHDRAWN"
    # a finished run's Phase-1 pass whose final review dropped it re-keeps nothing either
    rid = _unfinished_run_keeping(conn, B, E, status="done")
    with tx(conn):
        conn.execute("DELETE FROM findings WHERE run_id=? AND phase='verify2'", (rid,))
    assert worker._latest_settlement(ctx, h) == "WITHDRAWN"
    with tx(conn):
        conn.execute(
            "INSERT INTO findings (run_id, phase, stage, hash, severity, title, body) "
            "VALUES (?, 'verify2', 'verified', ?, 'blocking', 't', 'b')",
            (rid, h),
        )
    assert worker._latest_settlement(ctx, h) is None  # a finished final review kept it


# ---- severity, phases, modes ----


def test_a_suggestion_judged_fixed_and_raised_as_a_blocker_is_confirmed_or_stays_visible(
    cfg, conn, gh, lanes, unchanged, monkeypatch
):
    """`late.hold_severity` holds a carried suggestion raised to blocking on unchanged code at
    its posted severity; judged fixed earlier, it still goes to the confirmation lane, and
    unconfirmed it is the carried suggestion it was, not dropped."""
    from tests import test_late

    monkeypatch.setitem(test_late.STATUS, S["title"], late.SINCE_FIRST)
    h = _fixed(cfg, conn, gh, lanes, S)
    lanes.confirm = {S["title"]: "non_blocking"}
    rid = _raised_again(cfg, conn, gh, lanes, h, D, {**S, "severity": "blocking"})
    assert len(_prompts(lanes, "confirm")) == 1
    review = gh.posted_reviews[-1]
    assert review["event"] != "REQUEST_CHANGES" and S["title"] in review["body"]
    assert _stages(conn, rid)[("verified", S["title"])][0] == "suggestion"
    assert _judged(conn, h) is None  # standing again, as the suggestion it was


def test_a_suggestion_judged_fixed_and_confirmed_as_a_blocker_blocks(
    cfg, conn, gh, lanes, unchanged, monkeypatch
):
    from tests import test_late

    monkeypatch.setitem(test_late.STATUS, S["title"], late.SINCE_FIRST)
    h = _fixed(cfg, conn, gh, lanes, S)
    rid = _raised_again(cfg, conn, gh, lanes, h, D, {**S, "severity": "blocking"})
    assert len(_prompts(lanes, "confirm")) == 1
    assert gh.posted_reviews[-1]["event"] == "REQUEST_CHANGES"
    assert _stages(conn, rid)[("verified", S["title"])][0] == "blocking"


def test_a_preliminary_review_still_settles_an_answered_dispute(cfg, conn, gh, lanes, unchanged):
    """The Phase-1 verifier has no confirmation gate after it: a carried finding judged fixed
    whose dispute was already answered still becomes DISPUTED there, instead of being posted
    as an unconfirmed blocker by a preliminary review."""
    h = _fixed(cfg, conn, gh, lanes)
    gh.threads[0]["comments"]["nodes"].append(_gql_comment({**KNST, "body": "Not a bug."}))
    with tx(conn):
        memo = {"reply": "901", "standing": True, "sha": C}
        kv_set(conn, f"dispute.answered:dashpay/platform#1:{h}", json.dumps(memo))
    row = {"finding_hash": h, "status": "STILL_VALID", "reason": STILL, "disputed": True}
    carried = {**B, "finding_hash": h}
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [carried],
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": [row],
    }
    lanes.verifier["preliminary"] = {
        **_verifier([carried]),
        "prior_finding_reconciliation": [row],
    }
    lanes.verifier["final"] = {**_verifier([carried]), "prior_finding_reconciliation": [row]}
    lanes.confirm = {B["title"]: "non_blocking"}
    rid = _queue_new_head(cfg, conn, gh, C, lanes)
    assert worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False) == (
        RunStatus.DONE
    )
    phases = [r["phase"] for r in conn.execute("SELECT phase FROM reviews WHERE run_id=?", (rid,))]
    assert "preliminary" not in phases
    assert all(r["event"] != "REQUEST_CHANGES" for r in gh.posted_reviews[-1:])


def test_in_degraded_mode_it_stands_as_raised(cfg, conn, gh, lanes, unchanged, monkeypatch):
    h = _fixed(cfg, conn, gh, lanes)
    monkeypatch.setattr(worker.RunContext, "is_degraded", property(lambda self: True))
    rid = _raised_again(cfg, conn, gh, lanes, h, D)
    assert not _prompts(lanes, "confirm")
    assert gh.posted_reviews[-1]["event"] == "REQUEST_CHANGES"
    assert ("verified", B["title"]) in _stages(conn, rid)
    assert conn.execute(
        "SELECT 1 FROM events WHERE kind='late.confirm_skipped' AND run_id=?", (rid,)
    ).fetchone()


def test_judged_fixed_again_in_between_moves_the_head(cfg, conn, gh, lanes, unchanged):
    h = _fixed(cfg, conn, gh, lanes)
    _round(cfg, conn, gh, lanes, D, [], [{"finding_hash": h, "status": "FIXED", "reason": "ok"}])
    assert _judged(conn, h) == {"status": "FIXED", "sha": D}
    lanes.confirm = {B["title"]: "non_blocking"}
    _raised_again(cfg, conn, gh, lanes, h, E)
    (confirm,) = _prompts(lanes, "confirm")
    assert f'"judged_fixed_at": "{D}"' in confirm


def test_the_confirmation_lane_is_not_told_the_failure_must_be_one_the_review_missed(cfg):
    f = Finding.from_dict(B)
    f.extra["judged_fixed"] = {"sha": C, "replies": [{"author": "knst", "body": "repro"}]}
    p = prompts.confirmation_prompt(
        cfg,
        repo="dashpay/platform",
        number=1,
        head_sha=D,
        meta={"title": "t", "body": "b"},
        coverage_from="b" * 40,
        reported=late.candidate_briefs([f]),
    )
    assert "whether that review missed it or code around it has changed since" in p
    assert '"replies_since"' in p and '"body": "repro"' in p


# ---- on a real repository ----


def _later_round(worktree, head, prev):
    return SimpleNamespace(
        cfg=SimpleNamespace(policy=SimpleNamespace(pipeline=None)),
        is_audit=False,
        sha=head,
        prev_final_sha=prev,
        first_reviewed_sha=None,
        worktree=worktree,
        code_history=None,
    )


def test_unchanged_lines_as_git_reads_them(repo):  # noqa: F811
    base = _commit(repo, {"w.rs": "\n".join(_lines(40, "w")) + "\n", "x.rs": "x\n"})
    _git(repo, "checkout", "-q", "-b", "feature")
    fixed = _commit(repo, {"w.rs": "\n".join(_lines(40, "w")) + "\nfixed tail\n"})
    f = _f("w.rs", 10, 12)
    # lines changed since the head it was judged fixed at: the normal path
    edited = _lines(40, "w")
    edited[10] = "edited after the fix"
    head = _commit(repo, {"w.rs": "\n".join(edited) + "\nfixed tail\n"})
    assert not worker._unchanged_since(_later_round(repo, head, fixed), fixed, f)
    # the head it was judged fixed at is gone (a force-push): the normal path
    assert not worker._unchanged_since(_later_round(repo, head, fixed), "0" * 40, f)
    _git(repo, "reset", "-q", "--hard", fixed)
    # renamed without a content change: unchanged, under its new name
    _git(repo, "mv", "w.rs", "v.rs")
    _git(repo, "commit", "-q", "-m", "rename")
    renamed = _git(repo, "rev-parse", "HEAD")
    assert worker._unchanged_since(_later_round(repo, renamed, fixed), fixed, _f("v.rs", 10, 12))
    _git(repo, "reset", "-q", "--hard", fixed)
    # the base branch merged in, touching other files and other lines: unchanged
    _git(repo, "checkout", "-q", "-")
    upstream = _lines(40, "w")
    upstream[35] = "changed on the base branch"
    _commit(repo, {"w.rs": "\n".join(upstream) + "\n", "x.rs": "y\n"})
    _git(repo, "checkout", "-q", "feature")
    _git(repo, "merge", "-q", "--no-edit", "-")
    merged = _git(repo, "rev-parse", "HEAD")
    assert merged != fixed and base
    assert worker._unchanged_since(_later_round(repo, merged, fixed), fixed, f)
    assert not worker._unchanged_since(_later_round(repo, merged, fixed), fixed, _f("w.rs", 35))


def test_listed_places_read_one_diff_per_head_and_file(repo, monkeypatch):  # noqa: F811
    listed_at = _commit(repo, {"m.rs": "\n".join(_lines(60)) + "\n"})
    head = _commit(repo, {"m.rs": "\n".join(["new"] * 4 + _lines(60)) + "\n"})
    calls = []
    real = late.git_diff_u0
    monkeypatch.setattr(late, "git_diff_u0", lambda *a: calls.append(a) or real(*a))
    rows = [("m.rs", 10, 12, listed_at), ("m.rs", 40, 41, listed_at), ("m.rs", 50, 50, head)]
    assert late.listed_spots(repo, head, rows) == [
        ("m.rs", 14, 16),
        ("m.rs", 44, 45),
        ("m.rs", 50, 50),
    ]
    assert len(calls) == 1


def test_a_run_that_published_and_then_failed_still_supersedes_a_settlement(conn):
    """A publication happened whatever came after it: a reintroduced finding it posted stands
    over an older withdrawal."""
    h = _seed_withdrawn_prior(conn, B)
    rid = _unfinished_run_keeping(conn, B, D, status="failed")
    with tx(conn):
        conn.execute(
            "UPDATE findings SET stage='posted' WHERE run_id=? AND phase='verify2'", (rid,)
        )
    ctx = SimpleNamespace(conn=conn, repo="dashpay/platform", number=1)
    assert worker._latest_settlement(ctx, h) is None


def test_a_revival_kept_settled_on_a_same_head_re_review_keeps_its_judgement(
    cfg, conn, gh, lanes, unchanged
):
    """FIXED at C while a blocker stands; a re-review of C asked for runs the fresh final gate
    before the late gate, so its raw verifier rows carry the revived suggestion. Kept settled,
    its judgement is written again with C, and the next round still sees it."""
    x = {**_blocking("Fee sum overflows"), "line_start": 30, "line_end": 31}
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["preliminary"] = _verifier([])
    lanes.verifier["final"] = _verifier([S, x])
    _run(cfg, conn, gh, lanes)
    hs, hx = Finding.from_dict(S).hash, Finding.from_dict(x).hash
    carried_x = {**x, "finding_hash": hx}
    _round(
        cfg,
        conn,
        gh,
        lanes,
        C,
        [carried_x],
        [
            {"finding_hash": hs, "status": "FIXED", "reason": "ok"},
            {"finding_hash": hx, "status": "STILL_VALID", "reason": "still"},
        ],
    )
    assert gh.posted_reviews[-1]["event"] == "REQUEST_CHANGES"
    rid = _round(
        cfg,
        conn,
        gh,
        lanes,
        C,
        [{**S, "finding_hash": hs}],
        [
            {"finding_hash": hs, "status": "STILL_VALID", "reason": "still"},
            {"finding_hash": hx, "status": "WITHDRAWN", "reason": "wrong"},
        ],
    )
    steps = {r["name"] for r in conn.execute("SELECT name FROM steps WHERE run_id=?", (rid,))}
    assert "fresh_phase2" in steps
    assert conn.execute(
        "SELECT 1 FROM findings WHERE run_id=? AND phase='verify2' AND stage='verified' AND hash=?",
        (rid, hs),
    ).fetchone()
    assert _judged_rows(conn, hs)[-1] == (rid, f"FIXED\n{C}")
    assert _judged(conn, hs) == {"status": "FIXED", "sha": C}


def test_another_bots_reply_is_not_a_human_reply():
    thread = {
        "transcript": [
            {"author": "coderabbitai[bot]", "is_bot": False, "created_at": "z", "body": "nit"},
            {"author": "coderabbitai", "is_bot": False, "created_at": "z", "body": "nit"},
            {"author": "thepastaclaw", "is_bot": True, "created_at": "z", "body": "ours"},
            {"author": "knst", "is_bot": False, "created_at": "z", "body": "repro"},
            {"author": "knst", "is_bot": False, "created_at": "a", "body": "older"},
        ]
    }
    ctx = SimpleNamespace(open_threads={"h": thread})
    assert [c["body"] for c in worker._human_replies_since(ctx, "h", "m")] == ["repro"]


def test_a_verifier_cannot_set_the_gates_routing_keys(cfg, conn, gh, lanes, statuses):  # noqa: F811
    """A new blocker on unchanged code whose verifier finding claims `posted_severity:
    blocking` is not "demoted" to blocking when the confirmation lane does not confirm it."""
    from tests.test_late import FALSE, _first_round

    _first_round(cfg, conn, gh, lanes)
    lanes.confirm = {FALSE["title"]: "non_blocking"}
    rid = _round(
        cfg, conn, gh, lanes, C, [{**FALSE, "posted_severity": "blocking", "raised_as": "x"}]
    )
    assert gh.posted_reviews[-1]["event"] == "APPROVE"
    assert ("verified", FALSE["title"]) not in _stages(conn, rid)
    assert ("followup", FALSE["title"]) in _stages(conn, rid)


def _issue_calls(gh):
    return [c for c in gh.calls if c[:1] == ["issue"]]


def test_a_deferral_on_a_finding_judged_fixed_files_no_issue(cfg, conn, gh, lanes, unchanged):
    """A finding a full review judged fixed needs confirmation if it is back, not a tracking
    issue: a maintainer deferring it in the conversation, or a verifier reconciling it as
    deferred, files nothing."""
    from tests.test_settle_disputes import _run_conversation

    h = _fixed(cfg, conn, gh, lanes)
    lanes.confirm = {B["title"]: "non_blocking"}
    _raised_again(cfg, conn, gh, lanes, h, D)
    gh.threads[0]["comments"]["nodes"].append(
        _gql_comment({**KNST, "body": "Out of scope here, let's defer it."})
    )
    defer = {"finding_hash": h, "status": "INTENTIONALLY_DEFERRED", "reply": "Deferring it."}
    _run_conversation(cfg, conn, gh, lanes, D, [defer])
    assert not _issue_calls(gh) and not gh.created_issues
    assert "Before I mark this deferred" in gh.replies[-1]["body"]
    assert _judged(conn, h) == {"status": "FIXED", "sha": C}

    recon = [{"finding_hash": h, "status": "INTENTIONALLY_DEFERRED", "reason": "Later."}]
    _round(cfg, conn, gh, lanes, E, [], recon)
    assert not _issue_calls(gh) and not gh.created_issues
