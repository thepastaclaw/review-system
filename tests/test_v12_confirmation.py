"""Policy v12 confirmation pass (B4): when it runs (`confirmation.trigger`), which roles,
the delta verification against the round's own issues, the skips (blockers, degraded mode,
incomplete verification), and the #38 same-sha approval that needs it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reviewsys import worker
from reviewsys.contract import Finding
from reviewsys.models import Trigger
from tests.conftest import group
from tests.test_pipeline_policy import load_with
from tests.test_worker_e2e import (
    fake_git,  # noqa: F401  (autouse: fake mirror/worktree)
)
from tests.v12_support import (
    REPO,
    A,
    B,
    by_title,
    cand,
    confirm_matches,
    issues,
    publish_threads,
    refuted,
    reply,
    run_round,
    shard_setup,
    try_round,
    v12_cfg,
)

pytestmark = pytest.mark.usefixtures("wide_diff")


@pytest.mark.parametrize("trigger", ["off", "rereview", "no_blockers"])
def test_confirmation_trigger(cfg, skills_dir, conn, gh, lanes, git, trigger):
    v12 = v12_cfg(cfg, skills_dir, confirmation={"trigger": trigger})
    lanes.candidates["default"] = []
    rid = run_round(v12, conn, gh, lanes, A)
    steps = {r[0] for r in conn.execute("SELECT name FROM steps WHERE run_id=?", (rid,))}
    assert ("fresh_phase2" in steps) is (trigger == "no_blockers")
    assert gh.posted_reviews[-1]["event"] == "APPROVE"  # round 1: only no_blockers needs it
    rid2 = run_round(v12, conn, gh, lanes, B)
    steps = {r[0] for r in conn.execute("SELECT name FROM steps WHERE run_id=?", (rid2,))}
    assert ("fresh_phase2" in steps) is (trigger != "off")


def test_confirmation_is_skipped_in_degraded_mode(
    cfg, skills_dir, conn, gh, lanes, git, monkeypatch
):
    v12 = v12_cfg(cfg, skills_dir)
    monkeypatch.setattr(worker.RunContext, "is_degraded", property(lambda self: True))
    lanes.candidates["default"] = []
    rid = run_round(v12, conn, gh, lanes, A)
    assert (
        conn.execute(
            "SELECT detail FROM events WHERE kind='confirmation.skipped' AND run_id=?", (rid,)
        ).fetchone()["detail"]
        == "degraded mode"
    )
    assert gh.posted_reviews[-1]["event"] == "COMMENT"


def test_a_confirmation_finding_is_verified_against_this_rounds_issues(
    cfg, skills_dir, conn, gh, lanes, git
):
    """The second pass restates this round's suggestion (dropped as a duplicate) and finds one
    new suggestion: the delta verifier sees the round's issues as open, and the new one is
    published beside them; still a suggestion standing, so no approval."""
    v12 = v12_cfg(cfg, skills_dir)
    lanes.candidates["default"] = []
    lanes.candidates["general"] = [cand("Add a test", severity="suggestion", line=12)]
    calls = iter(range(10))
    # security-auditor runs in Phase 1, Phase 2 and then the confirmation pass
    lanes.candidates["security-auditor"] = lambda: (
        [cand("Log the retry", severity="suggestion", line=14)] if next(calls) == 2 else []
    )
    raised = Finding(file="f.rs", title="Add a test", body="", category="logic").hash

    def verify(label, cands, prompt):
        if label == "verify1":
            return refuted(cands)
        if label == "confirm":
            assert '"status": "raised this round"' in prompt and raised in prompt
            assert {c["id"].split("-")[0] for c in cands} == {"cf"}
            return {
                "summary": "-",
                "groups": [
                    group(c, match=f"open:{raised}") if c["title"] == "Add a test" else group(c)
                    for c in cands
                ],
            }
        return {"summary": "ok", "groups": [group(c) for c in cands]}

    lanes.verify12 = verify
    rid = run_round(v12, conn, gh, lanes, A)
    review = gh.posted_reviews[-1]
    titles = sorted(c["body"].split(": ", 1)[1].split("**")[0] for c in review["comments"])
    assert titles == ["Add a test", "Log the retry"] and review["event"] == "COMMENT"
    assert "Confirmation pass: `general`, `security-auditor`" in review["body"]
    assert "(1 new finding(s))" in review["body"]
    fresh = conn.execute(
        "SELECT title FROM findings WHERE run_id=? AND stage='verified-fresh'", (rid,)
    ).fetchall()
    assert [r[0] for r in fresh] == ["Log the retry"]


def test_same_sha_reply_without_the_confirmation_pass_queues_it(v12, conn, gh, lanes, git):
    """The standing review had a blocker, so its confirmation pass never ran: withdrawing the
    blocker in discussion comments and queues one same-sha re-review, which runs the pass
    and then approves."""
    from reviewsys.db import kv_get

    lanes.verify12 = confirm_matches
    lanes.candidates["default"] = []
    lanes.candidates["general"] = [cand("Fee estimate omits drainage")]
    run_round(v12, conn, gh, lanes, A)
    ids = publish_threads(gh)
    fee = by_title(conn)["Fee estimate omits drainage"].hash
    reply(gh, ids[fee], "The caller adds it.", rid=903)
    lanes.thread_out[fee] = {"status": "WITHDRAWN", "reply": "Agreed, the caller adds it."}
    before = len(gh.posted_reviews)
    rid = run_round(v12, conn, gh, lanes, A, Trigger.REVIEW_REPLY)
    upd = gh.posted_reviews[before]
    assert upd["event"] == "COMMENT" and "confirmation pass" in upd["body"]
    assert "which the review of this commit did not run (it had found a blocker)" in upd["body"]
    assert "converse.fresh_audit" in str(
        conn.execute("SELECT key FROM kv WHERE key LIKE 'converse.fresh_audit:%'").fetchall()[0][0]
    )
    assert kv_get(conn, f"rereview.after_run:{rid}") is None  # consumed: the head is queued
    # the queued same-sha re-review: nothing stands, so the confirmation pass runs and approves
    gh.threads = []
    lanes.thread_out = {}
    lanes.candidates["general"] = []
    from reviewsys.scheduler import schedule

    (rid2,) = schedule(conn, v12, spawn=False)
    assert worker.main(v12, conn, rid2, gh=gh, lane_runner=lanes, heartbeat=False).value == "done"
    assert gh.posted_reviews[-1]["event"] == "APPROVE"
    steps = dict(conn.execute("SELECT name, status FROM steps WHERE run_id=?", (rid2,)).fetchall())
    assert steps["fresh_verify2"] == "ok"
    assert "Confirmation pass: `general`, `security-auditor`" in gh.posted_reviews[-1]["body"]


def test_the_pass_runs_the_configured_roles_only(v12, conn, gh, lanes, git):
    lanes.verify12 = confirm_matches
    lanes.candidates["default"] = []
    lanes.candidates["general"] = [cand("Add a regression test", severity="suggestion", line=20)]
    rid = run_round(v12, conn, gh, lanes, A)
    steps = {
        r["name"]: (r["status"], r["detail"])
        for r in conn.execute("SELECT name, status, detail FROM steps WHERE run_id=?", (rid,))
    }
    assert steps["fresh_phase2"][0] == "ok" and steps["fresh_verify2"][0] == "ok"
    # always-on ran in Phase 2 but is not a confirmation role
    assert '"roles": ["general", "security-auditor"]' in steps["fresh_phase2"][1]
    assert [label for k, label, _ in lanes.v12_calls if k == "verify"][-1] == "confirm"
    assert gh.posted_reviews[-1]["event"] == "COMMENT"  # the suggestion stands


def test_blockers_skip_the_pass(v12, conn, gh, lanes, git):
    lanes.verify12 = confirm_matches
    lanes.candidates["default"] = []
    lanes.candidates["general"] = [cand("Fee estimate omits drainage")]
    rid = run_round(v12, conn, gh, lanes, A)
    assert (
        conn.execute(
            "SELECT detail FROM events WHERE kind='confirmation.skipped' AND run_id=?", (rid,)
        ).fetchone()["detail"]
        == "blockers stand"
    )


def test_an_incomplete_verification_skips_the_pass(cfg, skills_dir, conn, gh, lanes, git):
    v12 = v12_cfg(cfg, skills_dir)
    shard_setup(lanes, blocking_in_failing=False)
    run_round(v12, conn, gh, lanes, A)
    assert (
        conn.execute("SELECT detail FROM events WHERE kind='confirmation.skipped'").fetchone()[
            "detail"
        ]
        == "verification incomplete"
    )


def test_no_blockers_trigger_needs_the_v12_verifier(cfg, skills_dir):
    with pytest.raises(ValueError, match=r'"no_blockers" needs verifier\.contract "v12"'):
        load_with(cfg, skills_dir, {"version": "v12", "confirmation": {"trigger": "no_blockers"}})


@pytest.mark.parametrize("trigger", ["off", "no_blockers"])
def test_no_same_sha_approval_over_a_review_that_dropped_candidates(
    cfg, skills_dir, conn, gh, lanes, git, trigger
):
    """The standing review dropped a verifier shard (its candidates were never verified): a
    discussion lifting its last blocker comments, whatever the confirmation trigger."""
    v12 = v12_cfg(cfg, skills_dir, confirmation={"trigger": trigger})
    gh.diff = "diff --git a/b.rs b/b.rs\n+++ b/b.rs\n@@ -1,3 +1,60 @@\n+x\n"
    shard_setup(lanes, blocking_in_failing=False)
    lanes.candidates["general"][-1]["severity"] = "blocking"  # in the shard that verifies
    run_round(v12, conn, gh, lanes, A)
    assert gh.posted_reviews[-1]["event"] == "REQUEST_CHANGES"
    ids = publish_threads(gh)
    blocker = next(h for h, i in issues(conn).items() if i.severity == "blocking")
    reply(gh, ids[blocker], "Not reachable.", rid=908)
    lanes.thread_out[blocker] = {"status": "WITHDRAWN", "reply": "Right."}
    before = len(gh.posted_reviews)
    run_round(v12, conn, gh, lanes, A, Trigger.REVIEW_REPLY)
    upd = gh.posted_reviews[before]
    assert upd["event"] == "COMMENT"
    assert (
        "did not run at full strength (some of its candidates were never verified)" in upd["body"]
    )
    assert not conn.execute("SELECT 1 FROM kv WHERE key LIKE 'rereview.after_run:%'").fetchone()


def test_an_unchecked_blocker_in_the_confirmation_pass_fails_the_run(
    cfg, skills_dir, conn, gh, lanes, git
):
    """§2.2 holds in the confirmation pass too: its verify+match failing with a blocking
    candidate fails the run (INFRA) instead of publishing an approval nobody checked."""
    from reviewsys.models import RunStatus

    v12 = v12_cfg(cfg, skills_dir)
    lanes.candidates["default"] = []
    calls = iter(range(10))
    # security-auditor: Phase 1, Phase 2, then the confirmation pass raises a blocker
    lanes.candidates["security-auditor"] = lambda: [cand("Unchecked")] if next(calls) == 2 else []

    def verify(label, cands, prompt):
        if label == "confirm":
            return {"summary": "x", "groups": "never valid"}
        return {"summary": "ok", "groups": [group(c) for c in cands]}

    lanes.verify12 = verify
    rid, status = try_round(v12, conn, gh, lanes, A)
    assert status == RunStatus.FAILED
    row = conn.execute("SELECT reason, fail_kind FROM runs WHERE id=?", (rid,)).fetchone()
    assert row["fail_kind"] == "infra" and "blocking candidates unchecked" in row["reason"]
    assert not gh.posted_reviews


def _withdrawn_blocker(cfg, conn, gh, lanes) -> str:
    """A full review of A raising one blocker, then a maintainer's reply the thread lane
    answers WITHDRAWN: the finding hash."""
    lanes.verify12 = confirm_matches
    lanes.candidates["default"] = []
    lanes.candidates["general"] = [cand("Fee estimate omits drainage")]
    run_round(cfg, conn, gh, lanes, A)
    ids = publish_threads(gh)
    fee = by_title(conn)["Fee estimate omits drainage"].hash
    reply(gh, ids[fee], "The caller adds it.", rid=903)
    lanes.thread_out[fee] = {"status": "WITHDRAWN", "reply": "Agreed, the caller adds it."}
    return fee


def _converse_detail(conn, rid: int) -> dict:
    row = conn.execute("SELECT detail FROM steps WHERE run_id=? AND name='converse'", (rid,))
    return json.loads(row.fetchone()["detail"])


@pytest.mark.parametrize(
    ("trigger", "event", "note"),
    [
        ("off", "APPROVE", "Approved: the discussion withdrew or deferred every finding"),
        ("no_blockers", "COMMENT", "It is queued for this commit and approves if it finds"),
    ],
)
def test_a_replayed_same_sha_reply_records_the_follow_up_it_would_post(
    cfg, skills_dir, conn, gh, lanes, git, trigger, event, note
):
    """A replay delivers nothing (its gh is read-only), yet decides on what the outcomes
    would lift: the maintainer's withdrawal lifts the only blocker, so the follow-up is an
    APPROVE over a full-strength standing review, or a COMMENT queuing the confirmation pass
    the standing review skipped. It is recorded (conversation.json and the step), and nothing
    is posted, queued or moved in the ledger."""
    v12 = v12_cfg(cfg, skills_dir, confirmation={"trigger": trigger})
    fee = _withdrawn_blocker(v12, conn, gh, lanes)
    posted, replies = len(gh.posted_reviews), len(gh.replies)
    reviews = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    rid = run_round(v12, conn, gh, lanes, A, Trigger.REVIEW_REPLY, dry_run=True, replay=True)
    conv = json.loads((v12.runs_dir / f"run-{rid}" / "conversation.json").read_text())
    wp = conv["would_post"]
    assert wp["event"] == event and note in wp["note"]
    assert wp["lifted"] == [fee] and wp["titles"] == ["Fee estimate omits drainage"]
    assert wp["from"] == "CHANGES_REQUESTED" and wp["posts"] is True
    assert conv["outcomes"][fee]["status"] == "WITHDRAWN"  # the lane's outcomes stay as written
    detail = _converse_detail(conn, rid)
    assert detail["would_post"] == wp and detail["verdict_updated"] is False
    assert len(gh.posted_reviews) == posted and len(gh.replies) == replies
    assert conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == reviews
    assert not conn.execute("SELECT 1 FROM kv WHERE key LIKE 'converse.fresh_audit:%'").fetchone()
    assert not conn.execute("SELECT 1 FROM kv WHERE key LIKE 'rereview.after_run:%'").fetchone()
    assert issues(conn)[fee].status == "open"  # never delivered, never booked


def test_a_plain_dry_run_of_a_same_sha_reply_records_no_follow_up(v12, conn, gh, lanes, git):
    fee = _withdrawn_blocker(v12, conn, gh, lanes)
    posted = len(gh.posted_reviews)
    rid = run_round(v12, conn, gh, lanes, A, Trigger.REVIEW_REPLY, dry_run=True)
    conv = json.loads((v12.runs_dir / f"run-{rid}" / "conversation.json").read_text())
    assert "would_post" not in conv and conv["outcomes"][fee]["status"] == "WITHDRAWN"
    assert "would_post" not in _converse_detail(conn, rid)
    assert len(gh.posted_reviews) == posted
    assert issues(conn)[fee].status == "open"


def test_a_replayed_same_sha_rereview_names_the_blocker_its_threads_would_lift(
    v12, conn, gh, lanes, git
):
    """The full path's thread lane (a same-sha re-review) delivers nothing in a replay either:
    the verdict follow-up it would post still lists the blocker the withdrawal would lift."""
    _withdrawn_blocker(v12, conn, gh, lanes)
    lanes.candidates["general"] = []
    posted = len(gh.posted_reviews)
    rid = run_round(v12, conn, gh, lanes, A, Trigger.MANUAL, dry_run=True, replay=True)
    pub = json.loads((v12.runs_dir / f"run-{rid}" / "publication-final.json").read_text())
    assert pub["verdict_update"] is True
    assert "No longer blocking:\n- Fee estimate omits drainage" in pub["body"]
    assert len(gh.posted_reviews) == posted


def test_no_same_sha_approval_over_a_round_that_had_a_conflict(v12, conn, gh, lanes, git):
    """The standing round refused to approve because a concession and the verifier disagreed
    (`threads.conflict`): a same-sha reply lifting the last finding comments, saying why."""
    lanes.verify12 = confirm_matches
    lanes.candidates["default"] = []
    lanes.candidates["general"] = [
        cand("Add a regression test", severity="suggestion", line=12),
        cand("Name the constant", severity="suggestion", line=14),
    ]
    run_round(v12, conn, gh, lanes, A)
    ids = publish_threads(gh)
    test_ = by_title(conn)["Add a regression test"].hash
    name = by_title(conn)["Name the constant"].hash
    reply(gh, ids[test_], "tests/fee.rs covers it.", rid=911)
    lanes.thread_out[test_] = {"status": "WITHDRAWN", "reply": "Right."}
    lanes.candidates["general"] = [cand(f"Add a regression test [open:{test_}]", line=12)]
    rid = run_round(v12, conn, gh, lanes, B)
    assert conn.execute(
        "SELECT 1 FROM events WHERE kind='threads.conflict' AND run_id=?", (rid,)
    ).fetchone()
    reply(
        gh, ids[name], "The constant is named in the next PR.", rid=912, at="2026-10-05T00:00:00Z"
    )
    lanes.thread_out = {name: {"status": "WITHDRAWN", "reply": "Fine, then it goes there."}}
    lanes.candidates["general"] = []
    before = len(gh.posted_reviews)
    run_round(v12, conn, gh, lanes, B, Trigger.REVIEW_REPLY)
    assert len(gh.posted_reviews) == before  # no approval: it stays COMMENT
    assert by_title(conn)["Name the constant"].status == "withdrawn"
    ctx = worker.RunContext(
        cfg=v12,
        main_conn=conn,
        gh=gh,
        run_id=0,
        head_id=0,
        repo=REPO,
        number=1,
        sha=B,
        token="t",
        run_dir=Path("."),
    )
    assert worker._review_strength12(ctx, rid) == (
        "weak",
        "a finding the discussion withdrew was confirmed again by the verifier",
    )
