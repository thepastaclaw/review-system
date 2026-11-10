"""Stepping back from a PR that keeps drawing new blockers (stepback.py): when a round is due
one, what its lane must answer, how the review shows it, and the pause of automatic re-review
that follows."""

from __future__ import annotations

import json

import pytest

from reviewsys import checks, stepback
from reviewsys import config as cfg_mod
from reviewsys.db import tx
from reviewsys.ingest import carried_from, enqueue_head
from reviewsys.lane import LaneResult
from reviewsys.models import ReviewError, RunStatus, Trigger
from reviewsys.prompts import stepback_prompt
from reviewsys.scheduler import schedule
from tests.test_worker_e2e import (  # fake_git: autouse, fakes the mirror and worktree
    _blocking,
    _sugg,
    _verifier,
    fake_git,  # noqa: F401
)
from tests.test_worker_e2e import worker as worker_mod

REPO = "dashpay/platform"
S1, S2, S3 = "1" * 40, "2" * 40, "3" * 40
C, D, E, F = "c" * 40, "d" * 40, "e" * 40, "f" * 40
PAST = "2020-01-01T00:00:00Z"  # queued long enough ago that the push debounce has passed

# ---- when a round is due one ----


def _round(sha, blockers, suggestions=0):
    return stepback.Round(
        sha,
        tuple(
            {"hash": f"{sha[:4]}{i}", "severity": sev, "file": "f.rs", "title": "t", "body": "b"}
            for i, sev in enumerate(["blocking"] * blockers + ["suggestion"] * suggestions)
        ),
    )


def _due(earlier, new, *, stood=False):
    return stepback.due(earlier, new, min_round=4, min_blockers=6, min_recent=3, stood=stood)


def test_the_first_step_back_needs_a_late_round_many_blockers_and_two_rounds_of_new_ones():
    three = [_round(S1, 3), _round(S2, 1), _round(S3, 2)]
    assert _due(three, 1) == "round 4, 7 blockers so far, 3 new in the last two rounds"
    # one new blocker after one is what convergence looks like
    assert _due([_round(S1, 4), _round(S2, 2), _round(S3, 1)], 1) is None
    assert _due(three, 0) is None  # nothing new to explain
    assert _due(three[:2], 5) is None  # round 3
    assert _due([_round(S1, 1), _round(S2, 1), _round(S3, 1)], 2) is None  # 5 blockers in all
    assert _due([_round(S1, 4), _round(S2, 2), _round(S3, 0, 3)], 2) is None  # last round clean
    assert _due([], 9) is None


def test_once_a_step_back_stands_every_round_with_a_new_blocker_steps_back_again():
    assert _due([_round(S1, 1)], 1, stood=True) == "causes were already named on this PR"
    assert _due([_round(S1, 1)], 0, stood=True) is None


# ---- the lane's contract ----

IDS = ["p1", "p2", "p3", "n1", "n2"]


def _answer(**kw):
    return {
        "diagnosis": "foundational_gap",
        "themes": [
            {
                "name": "No durable record of an in-flight payment",
                "root_cause": "Payment state lives in three stores with no single owner.",
                "finding_ids": ["p1", "p3", "n1"],
                "tracked_issue": "https://github.com/dashpay/platform/issues/4270",
            }
        ],
        "recommendation": {
            "kind": "land_prerequisite_first",
            "text": "Land the payment journal first, or drop redelivery from this PR.",
        },
        **kw,
    }


def test_an_answer_that_names_causes_is_read_with_its_themes_and_decision():
    a = stepback.parse_analysis(_answer(diagnosis="Foundational-Gap"), IDS, previous=False)
    assert a.stands and a.diagnosis == "foundational_gap" and a.kind == "land_prerequisite_first"
    (theme,) = a.themes
    assert theme.finding_ids == ("p1", "p3", "n1")
    assert theme.tracked_issue.endswith("/issues/4270") and a.decision_made == ""
    # a tracked issue that is not a link to one is left out, never shown
    loose = _answer()
    loose["themes"][0]["tracked_issue"] = "see the roadmap [x](javascript:1)"
    assert stepback.parse_analysis(loose, IDS, previous=False).themes[0].tracked_issue == ""


def test_words_that_only_look_like_finding_ids_are_text():
    answer = _answer(recommendation={"kind": "split_pr", "text": "Keep p95 under n0 on p256."})
    answer["themes"][0]["root_cause"] = "The O(n10) scan over secp256k1 keys is p2p-visible."
    assert stepback.parse_analysis(answer, IDS, previous=False).text.startswith("Keep p95")


def test_converging_carries_no_themes_and_asks_for_nothing():
    a = stepback.parse_analysis(
        {
            "diagnosis": "converging",
            "themes": [{"name": "ignored"}],
            "recommendation": {"kind": "continue_normal", "text": ""},
        },
        IDS,
        previous=False,
    )
    assert not a.stands and a.themes == () and a.text == ""


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"diagnosis": "reviewer_noise"}, "diagnosis 'reviewer_noise' is not one of"),
        ({"recommendation": {"kind": "rewrite", "text": "x"}}, "recommendation.kind 'rewrite'"),
        (
            {"recommendation": {"kind": "continue_normal", "text": ""}},
            "`converging` goes with `continue_normal`",
        ),
        ({"diagnosis": "converging"}, "`converging` goes with `continue_normal`"),
        (
            {"recommendation": {"kind": "split_pr", "text": " "}},
            "recommendation.text must say what to decide",
        ),
        ({"themes": []}, "themes must list 1 to 4 underlying causes"),
        (
            {"themes": [{"name": "n", "root_cause": "c", "finding_ids": ["p1", "zz"]}]},
            "theme 1 names unknown finding ids zz",
        ),
        (
            {"themes": [{"name": "n", "root_cause": "c", "finding_ids": ["p1", "p1"]}]},
            "theme 1 needs at least 2 finding ids",
        ),
        (
            {"themes": [{"name": "", "root_cause": "c", "finding_ids": ["p1", "p2"]}]},
            "theme 1 needs a `name` and a `root_cause`",
        ),
        (
            {
                "themes": [
                    {"name": "n", "root_cause": "p3 and n2 fix it", "finding_ids": ["p1", "p2"]}
                ]
            },
            r"theme 1 refers to findings by id \(p3, n2\)",
        ),
        (
            {"recommendation": {"kind": "split_pr", "text": "Do what n1 says, over p2p."}},
            r"recommendation.text refers to findings by id \(n1\)",
        ),
    ],
)
def test_a_breach_of_the_contract_is_a_correctable_error(change, message):
    with pytest.raises(ReviewError, match=message):
        stepback.parse_analysis(_answer(**change), IDS, previous=False)


def test_with_an_earlier_step_back_the_answer_says_whether_its_decision_was_made():
    assert stepback.parse_analysis(
        _answer(decision_made="Partly"), IDS, previous=True
    ).decision_made == ("partly")
    with pytest.raises(ReviewError, match="decision_made '' is not one of yes, partly, no"):
        stepback.parse_analysis(_answer(), IDS, previous=True)


def test_only_a_second_naming_that_covers_the_rounds_new_blockers_pauses():
    first = stepback.parse_analysis(_answer(), IDS, previous=False)
    assert not stepback.holds(first, ["n1"], told=False)  # the first naming is a note
    again = stepback.parse_analysis(_answer(decision_made="no"), IDS, previous=True)
    assert stepback.holds(again, ["n1"], told=True) and stepback.holds(
        again, ["n1", "n2"], told=True
    )
    assert not stepback.holds(again, ["n2"], told=True) and not stepback.holds(again, [], told=True)
    made = stepback.parse_analysis(_answer(decision_made="yes"), IDS, previous=True)
    assert not stepback.holds(made, ["n1"], told=True)  # what is left is carrying it out
    assert not stepback.holds(stepback.Analysis(stepback.CONVERGING), ["n1"], told=True)


def test_the_prompt_shows_every_finding_and_the_step_back_that_stands(cfg):
    items = stepback.raised(
        [_round(S1, 1, 1), _round(S2, 1)],
        [{"severity": "blocking", "file": "g.rs", "title": "New one", "body": "x" * 900}],
        {f"{S1[:4]}0"},
    )
    assert [(r.id, r.round, r.state) for r in items] == [
        ("p1", 1, "open"),
        ("p2", 1, "closed"),
        ("p3", 2, "closed"),
        ("n1", 3, "new"),
    ]
    assert len(items[-1].brief()["body"]) < 720 and items[-1].brief()["body"].endswith("[...]")
    kw = {
        "repo": REPO,
        "number": 1,
        "head_sha": C,
        "meta": {"title": "Private payments", "body": "Adds redelivery."},
        "coverage_from": "b" * 40,
        "first_sha": S1,
        "round_now": 3,
        "raised": [r.brief() for r in items],
    }
    p = stepback_prompt(cfg, **kw)
    assert '"id": "n1"' in p and "Adds redelivery." in p and "round (round 3)" in p
    assert f"git diff --stat {S1}..{C}" in p and "decision_made" not in p
    assert "New blocking findings per round so far: 1, 1, and 1 in this round." in p
    again = stepback_prompt(cfg, **kw, previous={"posted_at_commit": S2, "round": 2})
    assert "What an earlier review already said" in again
    assert '"decision_made": "yes|partly|no"' in again


# ---- the review ----


def _outcome(*, hold, previous=None, answer=None):
    a = stepback.parse_analysis(_answer(**(answer or {})), IDS, previous=previous is not None)
    items = stepback.raised(
        [_round(S1, 2), _round(S2, 1)],
        [
            {
                "severity": "blocking",
                "file": "f.rs",
                "title": "Reservation leaks on @restart <details>",
                "body": "",
            },
            {"severity": "blocking", "file": "f.rs", "title": "Other", "body": ""},
        ],
        set(),
    )
    return stepback.Outcome(
        analysis=a,
        raised={r.id: r for r in items},
        round=3,
        blockers=5,
        hold=hold,
        previous=previous,
    )


def test_the_first_naming_is_a_note_beside_the_findings():
    out = _outcome(hold=False)
    text = "\n".join(stepback.render(out, "thepastaclaw"))
    assert text.startswith("### A pattern across this PR's review rounds")
    assert (
        "review round 3, and the PR has drawn 5 blocking findings so far. 3 of them share one"
        in text
    )
    assert (
        "**1. No durable record of an in-flight payment** (1 new in this round, 2 earlier finding(s))"
        in text
    )
    assert "Already tracked: https://github.com/dashpay/platform/issues/4270" in text
    # model-written and finding text never pages anyone and never opens HTML
    assert "In this round: _Reservation leaks on @\u200brestart &lt;details>_" in text
    assert "**Suggested decision (land the prerequisite first):** Land the payment journal" in text
    assert (
        "If a later review again finds new blockers from this cause, it will ask for that" in text
    )
    assert "Automatic re-review is paused" not in text and "Step back" not in text
    out.may_hold = False  # the bot's own PR, or `hold = false`
    assert "pause automatic re-review" not in "\n".join(stepback.render(out, "thepastaclaw"))
    note = stepback.agent_note(out)
    assert note.startswith("Several findings on this PR share an underlying cause")
    assert "No durable record of an in-flight payment" in note and "fix the cause" in note


def test_the_second_naming_asks_for_the_decision_and_says_re_review_is_paused():
    previous = stepback.Record(
        sha=S2, run_id=1, round=2, diagnosis="patch_cascade", kind="split_pr", text="t"
    )
    out = _outcome(hold=True, previous=previous, answer={"decision_made": "no"})
    text = "\n".join(stepback.render(out, "thepastaclaw"))
    assert text.startswith("### Step back: this PR keeps drawing new blockers")
    assert (
        f"The review of `{S2[:8]}` named this cause and the decision that would settle it; "
        "this round found 1 more finding(s) from the same place." in text
    )
    assert "**Requested decision (land the prerequisite first):**" in text
    assert "Automatic re-review is paused until then" in text
    assert "Comment `@thepastaclaw review` once the decision is made: that resumes" in text
    assert stepback.agent_note(out).startswith(
        'Before fixing anything, read the review\'s "Step back"'
    )
    converging = stepback.Outcome(stepback.Analysis(stepback.CONVERGING), {}, 3, 5, False)
    assert stepback.render(converging, "thepastaclaw") == []


def test_the_stepback_section_is_configurable(tmp_path):
    assert cfg_mod.StepbackConfig() == cfg_mod._stepback({})
    off = cfg_mod._stepback(
        {"enabled": False, "hold": False, "min_round": 1, "min_blockers": 0, "min_recent": 0}
    )
    assert not off.enabled and not off.hold
    assert (off.min_round, off.min_blockers, off.min_recent) == (2, 1, 2)
    assert "[stepback]" in cfg_mod.DEFAULT_TOML


# ---- end to end ----

NEW1 = {**_blocking("Reservation survives a failed broadcast"), "category": "bug"}
NEW2 = {**_blocking("Journal entry is not fenced to the wallet"), "line_start": 14, "line_end": 14}
EARLIER = [
    (S1, ["Reservation leaks on restart", "Scan progress is not invalidated"]),
    (S2, ["Journal replay double-spends a note", "Redrive ignores the wallet generation"]),
    (S3, ["Stream completion is inferred", "Reservation is released twice"]),
]


def _seed_rounds(conn, rounds=EARLIER):
    """Earlier review rounds as the database keeps them: a reviewed head, its final review
    and the blockers it posted (none of them an open prior of the next round)."""
    with tx(conn):
        for i, (sha, titles) in enumerate(rounds, 1):
            at = f"2026-11-0{i}T00:00:00Z"
            hid = conn.execute(
                "INSERT INTO heads (repo, number, sha, trigger, priority, status, queued_at, "
                "eligible_at) VALUES (?,1,?,'new_push',0,'done',?,?)",
                (REPO, sha, at, at),
            ).lastrowid
            rid = conn.execute(
                "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at) "
                "VALUES (?,1,'done','t',?,?)",
                (hid, at, at),
            ).lastrowid
            conn.execute(
                "INSERT INTO reviews (run_id, repo, number, sha, phase, github_review_id, event, "
                "posted_at) VALUES (?,?,1,?,'final',?,'REQUEST_CHANGES',?)",
                (rid, REPO, sha, 900 + i, at),
            )
            for n, title in enumerate(titles):
                conn.execute(
                    "INSERT INTO findings (run_id, phase, stage, hash, file, line_start, "
                    "line_end, severity, category, title, body) "
                    "VALUES (?,'final','posted',?,'f.rs',1,1,'blocking','bug',?,'earlier body')",
                    (rid, f"{sha[:6]}{n:06d}", title),
                )


def _push(conn, cfg, gh, lanes, sha, trigger, final=(), *, gate=(), runner=None, **kw):
    """The PR's head moves to `sha`, a head is queued for it by `trigger`, and its run goes
    through with `final` as the final verifier's findings (`gate`: the Phase-1 verifier's)."""
    gh.pr = {**gh.pr, "head": {**gh.pr["head"], "sha": sha}}
    lanes.head = sha
    fixed = [  # what earlier runs of this test posted is fixed by now
        {"finding_hash": str(r["hash"]), "status": "FIXED", "reason": "ok"}
        for r in conn.execute("SELECT hash FROM posted_findings WHERE repo=?", (REPO,))
    ]
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [],
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": fixed,
    }
    lanes.verifier["preliminary"] = _verifier(list(gate))
    lanes.verifier["final"] = _verifier(list(final))
    with tx(conn):
        conn.execute(
            "INSERT INTO prs (repo, number, head_sha, state, updated_at) VALUES (?,1,?,'open',?) "
            "ON CONFLICT(repo, number) DO UPDATE SET head_sha=excluded.head_sha",
            (REPO, sha, PAST),
        )
        queued = enqueue_head(conn, cfg, REPO, 1, sha, trigger, ts=PAST)
    (rid,) = schedule(conn, cfg, spawn=False)
    status = worker_mod.main(
        cfg, conn, rid, gh=gh, lane_runner=runner or lanes, heartbeat=False, **kw
    )
    assert status == RunStatus.DONE
    return rid, queued


def _themed(briefs):
    """The lane's answer: one cause over the first earlier finding and every new blocker."""
    new = [b["id"] for b in briefs if b["state"] == "new" and b["severity"] == "blocking"]
    return _answer(
        themes=[{**_answer()["themes"][0], "finding_ids": ["p1", *new]}], decision_made="no"
    )


def _events(conn, kind):
    return [
        str(r["detail"]) for r in conn.execute("SELECT detail FROM events WHERE kind=?", (kind,))
    ]


def _lane_calls(lanes, role="stepback"):
    return [s for s in lanes.calls if s.role == role]


AGAIN = {**_blocking("Redrive races the wallet wipe"), "line_start": 20, "line_end": 20}
THIRD = {**_blocking("Unrelated overflow in the fee sum"), "line_start": 30, "line_end": 30}
G = "9" * 40


def _told(conn, cfg, gh, lanes):
    """Rounds 1-3 as seeded, then round 4 at C: the review that first names the causes."""
    _seed_rounds(conn)
    lanes.stepback = _themed
    return _push(conn, cfg, gh, lanes, C, Trigger.NEW_PUSH, [NEW1, NEW2, _sugg()])[0]


def _paused(conn, cfg, gh, lanes):
    """... and round 5 at D, which draws a blocker from them again: the step-back, paused."""
    _told(conn, cfg, gh, lanes)
    return _push(conn, cfg, gh, lanes, D, Trigger.NEW_PUSH, [AGAIN])[0]


def test_the_first_round_that_names_causes_notes_them_and_changes_nothing_else(
    cfg, conn, gh, lanes
):
    rid = _told(conn, cfg, gh, lanes)
    (lane,) = _lane_calls(lanes)
    assert lane.model == "gpt-6-astra" and lane.effort == "high"  # the final verifier's
    for title in ("Reservation leaks on restart", NEW1["title"], NEW2["title"]):
        assert title in lane.prompt
    assert "round (round 4)" in lane.prompt and '"id": "p6"' in lane.prompt
    assert '"id": "n3"' in lane.prompt  # the suggestion is shown too
    assert "New blocking findings per round so far: 2, 2, 2, and 2 in this round." in lane.prompt
    review = gh.posted_reviews[-1]
    assert review["event"] == "REQUEST_CHANGES"
    inline = {c["body"].split("**")[1] for c in review["comments"]}
    assert f"🔴 Blocking: {NEW1['title']}" in inline and f"🔴 Blocking: {NEW2['title']}" in inline
    body = review["body"]
    assert stepback.NOTE_HEADING in body and stepback.HEADING not in body
    assert "review round 4, and the PR has drawn 8 blocking findings so far. 3 of them" in body
    assert "(2 new in this round, 1 earlier finding(s))" in body
    assert body.index(stepback.NOTE_HEADING) < body.index("🔴 2 blocking")
    assert "it will ask for that decision and pause automatic re-review" in body
    assert "Several findings on this PR share an underlying cause" in body
    rec = stepback.load(conn, REPO, 1)
    assert rec is not None and rec.sha == C and not rec.hold and rec.round == 4
    assert rec.run_id == rid and rec.diagnosis == "foundational_gap"
    assert rec.themes[0]["tracked_issue"].endswith("4270")
    assert _events(conn, "stepback.posted") == [
        f"{C[:8]} foundational_gap -> land_prerequisite_first"
    ]
    step = conn.execute(
        "SELECT status, detail FROM steps WHERE run_id=? AND name='stepback'", (rid,)
    ).fetchone()
    assert step["status"] == "ok" and json.loads(step["detail"])["hold"] is False
    # the next push is reviewed as always, and one without a new blocker asks no lane
    calls = len(lanes.calls)
    _push(conn, cfg, gh, lanes, D, Trigger.NEW_PUSH)
    assert len(lanes.calls) > calls and carried_from(conn, REPO, 1, D) is None
    assert len(_lane_calls(lanes)) == 1 and gh.posted_reviews[-1]["event"] == "APPROVE"


def test_new_blockers_from_named_causes_again_step_back_and_pause_re_review(cfg, conn, gh, lanes):
    rid = _paused(conn, cfg, gh, lanes)
    second = _lane_calls(lanes)[-1]
    assert "What an earlier review already said" in second.prompt
    assert f'"posted_at_commit": "{C}"' in second.prompt
    review = gh.posted_reviews[-1]
    assert review["event"] == "REQUEST_CHANGES"
    body = review["body"]
    # the finding is posted as always (here outside the diff, so in the body)
    assert f"🔴 Blocking: {AGAIN['title']}" in body or any(
        AGAIN["title"] in c["body"] for c in review["comments"]
    )
    assert stepback.HEADING in body and stepback.NOTE_HEADING not in body
    assert f"The review of `{C[:8]}` named this cause and the decision that would" in body
    assert "review round 5, and the PR has drawn 9 blocking findings so far" in body
    assert "**Requested decision (land the prerequisite first):**" in body
    assert "Automatic re-review is paused until then" in body
    assert 'Before fixing anything, read the review\'s "Step back" section.' in body
    rec = stepback.load(conn, REPO, 1)
    assert rec is not None and rec.sha == D and rec.hold and rec.round == 5 and rec.run_id == rid
    assert _events(conn, "stepback.posted")[-1] == (
        f"{D[:8]} foundational_gap -> land_prerequisite_first (automatic re-review paused)"
    )


def test_a_push_after_a_step_back_keeps_its_verdict_until_someone_asks(cfg, conn, gh, lanes):
    _paused(conn, cfg, gh, lanes)
    reviews, calls = len(gh.posted_reviews), len(lanes.calls)

    held, _ = _push(conn, cfg, gh, lanes, E, Trigger.NEW_PUSH)
    assert len(lanes.calls) == calls and len(gh.posted_reviews) == reviews  # no lane, no review
    assert carried_from(conn, REPO, 1, E) == D and stepback.is_held(conn, REPO, 1, E)
    assert "Automatic re-review is paused" in gh.gate_bodies[-1]
    assert f"commit {D[:8]} stepped back" in gh.gate_bodies[-1]
    assert "`@thepastaclaw review` asks for a review of this commit" in gh.gate_bodies[-1]
    steps = [r["name"] for r in conn.execute("SELECT name FROM steps WHERE run_id=?", (held,))]
    assert sorted(steps) == ["carry", "worktree"]
    assert _events(conn, "head.held") == [
        f"{E[:8]} keeps the step-back of {D[:8]} (foundational_gap)"
    ]
    note = checks._stands_note(conn, REPO, 1, E, D)
    assert "stepped back and asked for a decision" in note and "was not reviewed" in note

    # a second push is held on the same review; a plain re-queue of a held commit is a no-op
    _push(conn, cfg, gh, lanes, F, Trigger.NEW_PUSH)
    assert carried_from(conn, REPO, 1, F) == D and len(lanes.calls) == calls
    with tx(conn):
        assert enqueue_head(conn, cfg, REPO, 1, F, Trigger.NEW_PUSH, ts=PAST) == "noop"

    # someone asks: the commit gets its review, and one without new blockers ends the hold
    asked = len(_lane_calls(lanes))
    _, queued = _push(conn, cfg, gh, lanes, F, Trigger.MENTION)
    assert queued == "requeued" and len(lanes.calls) > calls
    assert len(gh.posted_reviews) == reviews + 1 and len(_lane_calls(lanes)) == asked
    rec = stepback.load(conn, REPO, 1)
    assert rec is not None and not rec.hold and rec.sha == D
    assert _events(conn, "stepback.released") == [f"the review of {F[:8]} ends the hold of {D[:8]}"]
    calls = len(lanes.calls)
    _push(conn, cfg, gh, lanes, G, Trigger.NEW_PUSH)
    assert len(lanes.calls) > calls and carried_from(conn, REPO, 1, G) is None


def test_a_requested_review_that_draws_blockers_again_keeps_the_pause_on_itself(
    cfg, conn, gh, lanes
):
    _paused(conn, cfg, gh, lanes)
    _push(conn, cfg, gh, lanes, E, Trigger.MENTION, [THIRD])
    body = gh.posted_reviews[-1]["body"]
    assert stepback.HEADING in body and f"The review of `{D[:8]}` named this cause" in body
    rec = stepback.load(conn, REPO, 1)
    assert rec is not None and rec.sha == E and rec.hold and rec.round == 6
    calls = len(lanes.calls)
    _push(conn, cfg, gh, lanes, F, Trigger.NEW_PUSH)
    assert len(lanes.calls) == calls and carried_from(conn, REPO, 1, F) == E


def test_a_decision_that_was_made_or_rounds_that_converge_end_the_pause(cfg, conn, gh, lanes):
    _paused(conn, cfg, gh, lanes)
    lanes.stepback = lambda briefs: {**_themed(briefs), "decision_made": "yes"}
    _push(conn, cfg, gh, lanes, E, Trigger.MENTION, [THIRD])
    body = gh.posted_reviews[-1]["body"]
    assert stepback.NOTE_HEADING in body and "Automatic re-review is paused" not in body
    rec = stepback.load(conn, REPO, 1)
    assert rec is not None and rec.sha == E and not rec.hold

    lanes.stepback = None  # converging
    last = {**_blocking("One last narrow defect"), "line_start": 50, "line_end": 50}
    _push(conn, cfg, gh, lanes, F, Trigger.NEW_PUSH, [last])
    body = gh.posted_reviews[-1]["body"]
    assert stepback.NOTE_HEADING not in body and stepback.HEADING not in body
    rec = stepback.load(conn, REPO, 1)
    assert rec is not None and rec.sha == E  # what was said stands; nothing new was


def test_converging_rounds_get_the_usual_review(cfg, conn, gh, lanes):
    _seed_rounds(conn)
    _push(conn, cfg, gh, lanes, C, Trigger.NEW_PUSH, [NEW1])
    assert len(_lane_calls(lanes)) == 1
    body = gh.posted_reviews[-1]["body"]
    assert stepback.NOTE_HEADING not in body and stepback.HEADING not in body
    assert stepback.load(conn, REPO, 1) is None and not _events(conn, "stepback.posted")
    calls = len(lanes.calls)
    _push(conn, cfg, gh, lanes, D, Trigger.NEW_PUSH)
    assert len(lanes.calls) > calls  # reviewed as always


def test_causes_that_miss_the_new_blockers_are_noted_without_a_pause(cfg, conn, gh, lanes):
    _told(conn, cfg, gh, lanes)
    # p1, p3 and n1 of three new blockers
    lanes.stepback = {**_answer(), "decision_made": "no"}
    _push(conn, cfg, gh, lanes, D, Trigger.NEW_PUSH, [AGAIN, THIRD, {**THIRD, "title": "A third"}])
    body = gh.posted_reviews[-1]["body"]
    assert stepback.NOTE_HEADING in body and "Automatic re-review is paused" not in body
    rec = stepback.load(conn, REPO, 1)
    assert rec is not None and rec.sha == D and not rec.hold
    calls = len(lanes.calls)
    _push(conn, cfg, gh, lanes, E, Trigger.NEW_PUSH)
    assert len(lanes.calls) > calls and carried_from(conn, REPO, 1, E) is None


def test_a_failed_lane_or_a_stand_in_model_leaves_the_review_as_it_was(
    cfg, conn, gh, lanes, monkeypatch
):
    _seed_rounds(conn)
    lanes.stepback = "there is a pattern here, I think"
    rid, _ = _push(conn, cfg, gh, lanes, C, Trigger.NEW_PUSH, [NEW1])
    review = gh.posted_reviews[-1]
    assert review["event"] == "REQUEST_CHANGES" and stepback.NOTE_HEADING not in review["body"]
    assert not _lane_calls(lanes, "repair") and stepback.load(conn, REPO, 1) is None
    assert len(_events(conn, "stepback.failed")) == 1
    step = conn.execute(
        "SELECT status FROM steps WHERE run_id=? AND name='stepback'", (rid,)
    ).fetchone()
    assert step["status"] == "failed"

    lanes.stepback = _themed
    asked = len(_lane_calls(lanes))
    monkeypatch.setattr(worker_mod.RunContext, "is_degraded", property(lambda self: True))
    _push(conn, cfg, gh, lanes, D, Trigger.NEW_PUSH, [AGAIN, THIRD])
    assert len(_lane_calls(lanes)) == asked
    assert stepback.NOTE_HEADING not in gh.posted_reviews[-1]["body"]
    assert len(_events(conn, "stepback.skipped")) == 1


def test_an_early_round_and_a_switched_off_step_back_never_ask_the_lane(cfg, conn, gh, lanes):
    _seed_rounds(conn, EARLIER[:2])
    lanes.stepback = _themed
    _push(conn, cfg, gh, lanes, C, Trigger.NEW_PUSH, [NEW1, NEW2])  # round 3
    assert not _lane_calls(lanes)
    off = cfg_mod.dataclasses.replace(cfg, stepback=cfg_mod.StepbackConfig(enabled=False))
    _push(conn, off, gh, lanes, D, Trigger.NEW_PUSH, [{**NEW1, "title": "Another new one"}])
    assert not _lane_calls(lanes) and stepback.NOTE_HEADING not in gh.posted_reviews[-1]["body"]


def test_a_hold_ends_when_its_review_no_longer_requests_changes(cfg, conn, gh, lanes):
    _paused(conn, cfg, gh, lanes)
    with tx(conn):  # a conversation lifted the blockers: the newest review of D is a COMMENT
        conn.execute(
            "INSERT INTO reviews (run_id, repo, number, sha, phase, github_review_id, event, "
            "posted_at) VALUES (NULL,?,1,?,'final',999,'COMMENT','2030-01-01T00:00:00Z')",
            (REPO, D),
        )
    calls = len(lanes.calls)
    _push(conn, cfg, gh, lanes, E, Trigger.NEW_PUSH)
    assert len(lanes.calls) > calls and carried_from(conn, REPO, 1, E) is None


# ---- what must not hold, carry or count ----


def test_a_pr_the_bot_authored_is_never_held(cfg, conn, gh, lanes):
    """Mentions on its own PRs go to the bot's agent, not to the queue: nothing could ask."""
    gh.pr = {**gh.pr, "user": {"login": "thepastaclaw"}}
    _paused(conn, cfg, gh, lanes)
    bodies = [r["body"] for r in gh.posted_reviews[-2:]]
    assert all(stepback.NOTE_HEADING in b for b in bodies)
    assert all("pause automatic re-review" not in b and "is paused" not in b for b in bodies)
    rec = stepback.load(conn, REPO, 1)
    assert rec is not None and rec.sha == D and not rec.hold
    calls = len(lanes.calls)
    _push(conn, cfg, gh, lanes, E, Trigger.NEW_PUSH)
    assert len(lanes.calls) > calls and carried_from(conn, REPO, 1, E) is None


def test_a_rebase_of_a_held_commit_never_carries_the_step_backs_verdict(
    cfg, conn, gh, lanes, monkeypatch
):
    """`_carry` compares diffs with the head before; that head's verdict is another diff's."""
    from reviewsys.steps import worktree as wt

    prints = {C: "diff-0", D: "diff-1", E: "diff-2", F: "diff-2"}
    monkeypatch.setattr(wt, "diff_print", lambda worktree, base, sha: prints.get(sha))
    _paused(conn, cfg, gh, lanes)
    _push(conn, cfg, gh, lanes, E, Trigger.NEW_PUSH)
    assert carried_from(conn, REPO, 1, E) == D
    off = cfg_mod.dataclasses.replace(cfg, stepback=cfg_mod.StepbackConfig(hold=False))
    calls = len(lanes.calls)
    _push(conn, off, gh, lanes, F, Trigger.NEW_PUSH)  # the same diff as E, which nobody reviewed
    assert len(lanes.calls) > calls and carried_from(conn, REPO, 1, F) is None
    assert not _events(conn, "head.carried")


def _reply(conn, at="2031-01-01T00:00:00Z"):
    """A reply under a finding that no run has read: the next push is reviewed, not held."""
    with tx(conn):
        conn.execute(
            "INSERT INTO inbox (source_id, kind, repo, number, actor, body, occurred_at, seen_at) "
            "VALUES ('r1','review_reply',?,1,'someone','Declined: by design.',?,?)",
            (REPO, at, at),
        )


def test_a_review_nobody_asked_for_leaves_the_pause_unless_it_is_final_and_answered(
    cfg, conn, gh, lanes
):
    _paused(conn, cfg, gh, lanes)
    _reply(conn)
    gate = {**_blocking("Phase 1 finds a new one"), "line_start": 40, "line_end": 40}
    _push(conn, cfg, gh, lanes, E, Trigger.NEW_PUSH, gate=[gate])  # stopped at the Phase-1 gate
    assert "Preliminary review" in gh.posted_reviews[-1]["body"]
    rec = stepback.load(conn, REPO, 1)
    assert rec is not None and rec.hold and rec.sha == D and not _events(conn, "stepback.released")

    lanes.stepback = "no JSON here"  # F: a full review whose step-back lane fails
    _push(conn, cfg, gh, lanes, F, Trigger.NEW_PUSH, [THIRD])
    rec = stepback.load(conn, REPO, 1)
    assert rec is not None and rec.hold and rec.sha == D
    with tx(conn):
        conn.execute("DELETE FROM inbox")
    calls = len(lanes.calls)
    _push(conn, cfg, gh, lanes, G, Trigger.NEW_PUSH)
    assert len(lanes.calls) == calls and carried_from(conn, REPO, 1, G) == D


def test_a_review_someone_asked_for_ends_the_pause_whatever_it_finds(cfg, conn, gh, lanes):
    _paused(conn, cfg, gh, lanes)
    gate = {**_blocking("Phase 1 finds a new one"), "line_start": 40, "line_end": 40}
    _push(conn, cfg, gh, lanes, E, Trigger.MENTION, gate=[gate])  # stopped at the Phase-1 gate
    rec = stepback.load(conn, REPO, 1)
    assert rec is not None and not rec.hold and rec.sha == D
    assert _events(conn, "stepback.released") == [f"the review of {E[:8]} ends the hold of {D[:8]}"]
    calls = len(lanes.calls)
    _push(conn, cfg, gh, lanes, F, Trigger.NEW_PUSH)
    assert len(lanes.calls) > calls and carried_from(conn, REPO, 1, F) is None


def test_asking_on_the_paused_commit_itself_resumes_re_review(cfg, conn, gh, lanes):
    """The author decides, then comments before pushing: the commit is reviewed already."""
    _paused(conn, cfg, gh, lanes)
    with tx(conn):
        assert enqueue_head(conn, cfg, REPO, 1, D, Trigger.MENTION, ts=PAST) == "noop"
    rec = stepback.load(conn, REPO, 1)
    assert rec is not None and not rec.hold and rec.sha == D
    assert _events(conn, "stepback.released") == [
        f"asked for on {D[:8]}: the next push is reviewed"
    ]
    with tx(conn):  # asking again, or on a PR that is not paused, changes nothing
        assert enqueue_head(conn, cfg, REPO, 1, D, Trigger.REVIEW_REQUESTED, ts=PAST) == "noop"
    assert len(_events(conn, "stepback.released")) == 1
    calls = len(lanes.calls)
    _push(conn, cfg, gh, lanes, E, Trigger.NEW_PUSH)
    assert len(lanes.calls) > calls and carried_from(conn, REPO, 1, E) is None


def test_a_retry_of_an_asked_review_keeps_the_pause_it_posted(cfg, conn, gh, lanes):
    """Asked for on E, the review paused again and its run died before recording it."""
    _paused(conn, cfg, gh, lanes)
    _posted_already(gh, E)
    _push(conn, cfg, gh, lanes, E, Trigger.MENTION, [THIRD])
    rec = stepback.load(conn, REPO, 1)
    assert rec is not None and rec.sha == E and rec.hold
    assert not _events(conn, "stepback.released")
    calls = len(lanes.calls)
    _push(conn, cfg, gh, lanes, F, Trigger.NEW_PUSH)
    assert len(lanes.calls) == calls and carried_from(conn, REPO, 1, F) == E


def test_a_dismissal_is_seen_by_a_review_someone_asked_for_too(cfg, conn, gh, lanes):
    _paused(conn, cfg, gh, lanes)
    gh.posted_reviews[-1]["state"] = "DISMISSED"
    gh.posted_reviews[-1]["user"] = {"login": "thepastaclaw"}
    _push(conn, cfg, gh, lanes, E, Trigger.MENTION, [THIRD])
    body = gh.posted_reviews[-1]["body"]
    assert stepback.NOTE_HEADING in body and "pause automatic re-review" not in body
    rec = stepback.load(conn, REPO, 1)
    assert rec is not None and rec.sha == E and not rec.hold and rec.overruled


def test_a_paused_record_never_holds_the_bots_own_pr(cfg, conn, gh, lanes):
    _paused(conn, cfg, gh, lanes)
    gh.pr = {**gh.pr, "user": {"login": "thepastaclaw"}}
    calls = len(lanes.calls)
    _push(conn, cfg, gh, lanes, E, Trigger.NEW_PUSH)
    assert len(lanes.calls) > calls and carried_from(conn, REPO, 1, E) is None


def test_a_dismissed_step_back_ends_the_pause_and_later_reviews_never_pause(cfg, conn, gh, lanes):
    _paused(conn, cfg, gh, lanes)
    dismissed = gh.posted_reviews[-1]
    dismissed["state"] = "DISMISSED"
    dismissed["user"] = {"login": "thepastaclaw"}
    # a same-sha summary posted since does not bring the dismissed request back
    gh.posted_reviews.append({**dismissed, "id": 4243, "state": "COMMENTED", "comments": []})
    _push(conn, cfg, gh, lanes, E, Trigger.NEW_PUSH, [THIRD])  # reviewed, and themed again
    body = gh.posted_reviews[-1]["body"]
    assert stepback.NOTE_HEADING in body and stepback.HEADING not in body
    assert "pause automatic re-review" not in body
    rec = stepback.load(conn, REPO, 1)
    assert rec is not None and rec.sha == E and not rec.hold and rec.overruled
    assert _events(conn, "stepback.released") == [
        f"a maintainer dismissed the review of {D[:8]} ends the hold of {D[:8]}"
    ]
    calls = len(lanes.calls)
    _push(conn, cfg, gh, lanes, F, Trigger.NEW_PUSH)
    assert len(lanes.calls) > calls and carried_from(conn, REPO, 1, F) is None


def test_a_quota_failure_in_the_lane_never_moves_the_run_onto_stand_ins(
    conn, gh, lanes, skills_dir, tmp_path
):
    from tests.test_degraded import FINE, _cfg_with

    c = _cfg_with(skills_dir, tmp_path)
    _seed_rounds(conn)

    def runner(spec, artifact_dir, worktree):
        if spec.role == "stepback":
            lanes.calls.append(spec)
            return LaneResult(
                exit_code=1,
                stdout="",
                stderr="API Error: Request rejected (429) · All credentials for model "
                "gpt-6-astra are cooling down",
                duration_s=1,
            )
        return lanes(spec, artifact_dir, worktree)

    rid, _ = _push(conn, c, gh, lanes, C, Trigger.NEW_PUSH, [NEW1], runner=runner, prober=FINE)
    assert [s.model for s in _lane_calls(lanes)] == ["gpt-6-astra", "gpt-6-astra"]
    assert conn.execute("SELECT degraded FROM runs WHERE id=?", (rid,)).fetchone()[0] == 0
    review = gh.posted_reviews[-1]
    assert review["event"] == "REQUEST_CHANGES" and "DEGRADED" not in review["body"]
    assert stepback.NOTE_HEADING not in review["body"]
    assert len(_events(conn, "stepback.failed")) == 1


def test_a_finding_recorded_again_under_another_hash_is_not_a_new_one(conn):
    _seed_rounds(conn, EARLIER[:2])
    with tx(conn):  # a same-sha re-review of round 2 kept round 1's finding, re-categorised
        rid = conn.execute(
            "SELECT r.id FROM runs r JOIN heads h ON h.id=r.head_id WHERE h.sha=?", (S2,)
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO findings (run_id, phase, stage, hash, file, line_start, line_end, "
            "severity, category, title, body) "
            "VALUES (?,'final','posted','ffffffffffff','F.rs',1,1,'blocking','logic',?,'b')",
            (rid, "Reservation  leaks on RESTART"),
        )
    first, second = stepback.rounds(conn, REPO, 1, before=C)
    assert (first.sha, first.blockers, second.sha, second.blockers) == (S1, 2, S2, 2)
    assert stepback.rounds(conn, REPO, 1, before=S2) == [first]


def _posted_already(gh, sha):
    """A paused step-back review of `sha` on GitHub that the database knows nothing of."""
    gh.posted_reviews.append(
        {
            "id": 4242,
            "user": {"login": "thepastaclaw"},
            "state": "CHANGES_REQUESTED",
            "commit_id": sha,
            "body": "<!-- thepastaclaw-review v1 -->\n<!-- thepastaclaw-review-phase v1 "
            f"phase=final sha={sha} policy=x -->\n## Final validation\n\n{stepback.HEADING}\n\n"
            f"_{stepback.held_note('thepastaclaw')}_\n\n🔴 1 blocking",
            "comments": [],
        }
    )
    return len(gh.posted_reviews)


@pytest.mark.parametrize("retry", ["same", "decided", "fails"])
def test_a_retry_that_finds_its_review_posted_records_what_it_says(cfg, conn, gh, lanes, retry):
    """The first attempt posted the review and died before recording it. Whatever the
    retry's lane answers, the pause the posted review announced is what stands."""
    _told(conn, cfg, gh, lanes)
    posted = _posted_already(gh, D)
    if retry == "decided":
        lanes.stepback = lambda briefs: {**_themed(briefs), "decision_made": "yes"}
    elif retry == "fails":
        lanes.stepback = "no JSON here"
    _push(conn, cfg, gh, lanes, D, Trigger.NEW_PUSH, [AGAIN])
    rec = stepback.load(conn, REPO, 1)
    assert rec is not None and rec.sha == D and rec.hold
    assert rec.themes and rec.diagnosis == "foundational_gap"  # the retry's, or the note's
    assert all(stepback.HEADING not in r["body"] for r in gh.posted_reviews[posted:])
    calls = len(lanes.calls)
    _push(conn, cfg, gh, lanes, E, Trigger.NEW_PUSH)
    assert len(lanes.calls) == calls and carried_from(conn, REPO, 1, E) == D


def test_what_a_record_knew_of_held_commits_survives_the_next_one(cfg, conn, gh, lanes):
    _paused(conn, cfg, gh, lanes)
    _push(conn, cfg, gh, lanes, E, Trigger.NEW_PUSH)  # held on D
    _push(conn, cfg, gh, lanes, F, Trigger.MENTION, [THIRD])  # steps back again, on F
    rec = stepback.load(conn, REPO, 1)
    assert rec is not None and rec.sha == F and rec.hold and rec.held == [E]
    assert stepback.is_held(conn, REPO, 1, E)


def test_a_note_over_non_blocking_findings_does_not_count_blockers():
    out = _outcome(hold=False)
    out.raised = {
        k: stepback.Raised(r.id, r.round, "suggestion", r.file, r.title, r.body, r.state)
        for k, r in out.raised.items()
    }
    text = "\n".join(stepback.render(out, "thepastaclaw"))
    assert "so far. Several of its findings share one underlying cause." in text


def test_a_record_this_code_cannot_read_is_no_record(conn):
    def stored(value):
        with tx(conn):
            stepback.kv_set(conn, stepback.key(REPO, 1), value)
        return stepback.load(conn, REPO, 1)

    assert stepback.load(conn, REPO, 1) is None
    for bad in ("not json", "[1]", '{"sha": 5}', '{"sha": "c", "hold": true}'):
        assert stored(bad) is None  # the last one lacks the fields a record needs
    rec = stored(
        json.dumps(
            {
                "sha": C,
                "run_id": 7,
                "round": 4,
                "diagnosis": "patch_cascade",
                "kind": "split_pr",
                "text": "t",
                "themes": None,
                "hold": "yes",
                "held": None,
                "overruled": 1,
                "added_by_a_later_version": 1,
            }
        )
    )
    assert rec is not None and rec.sha == C and rec.held == [] and rec.themes == []
    assert rec.hold is False and rec.overruled is False
    assert not stepback.is_held(conn, REPO, 1, D)
