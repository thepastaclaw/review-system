"""Later review rounds (late.py): code status of each finding against the previous final
review, the split of new findings on unchanged code, the independent confirmation of their
blockers, and how a round publishes them."""

from __future__ import annotations

import json

import pytest

from reviewsys import late, publish
from reviewsys.contract import Finding, VerifierOutput
from reviewsys.db import tx
from reviewsys.models import ReviewError, RunStatus
from reviewsys.prompts import confirmation_prompt
from tests.test_ledger import _commit, _git
from tests.test_worker_e2e import (  # fake_git: autouse, fakes the mirror and worktree
    HEAD,
    _blocking,
    _queue_new_head,
    _run,
    _sugg,
    _verifier,
    fake_git,  # noqa: F401
)
from tests.test_worker_e2e import worker as worker_mod

# ---- code status, on a real repository ----


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    return r


def _f(file, ls, le=None, **kw):
    return Finding(
        file=file, title=kw.pop("title", "t"), body="b", line_start=ls, line_end=le, **kw
    )


def _lines(n, prefix="line"):
    return [
        f"{prefix} {i} of the original code, long enough to be tracked" for i in range(1, n + 1)
    ]


def test_status_is_by_file_and_follows_renames(repo):
    files = {
        "a.rs": "\n".join(_lines(20, "a")) + "\n",
        "b.rs": "\n".join(_lines(20, "b")) + "\n",
        "c.rs": "\n".join(_lines(20, "c")) + "\n",
        "g.rs": "\n".join(_lines(10, "g")) + "\n",
        "sp ace.rs": "x\n",
    }
    first = _commit(repo, files)
    prev = _commit(repo, {"b.rs": files["b.rs"] + "added during the review\n"})
    _commit(repo, {"a.rs": "changed since the last review\n" + files["a.rs"]})
    _git(repo, "mv", "g.rs", "h.rs")
    _git(repo, "commit", "-q", "-m", "rename")
    _commit(repo, {"new.rs": "brand new\n", "sp ace.rs": "y\n"})
    head = _git(repo, "rev-parse", "HEAD")
    hist = late.CodeHistory(repo, head, prev, first)
    # any change to the file since the previous final review: the file's findings are new code
    assert hist.status(_f("a.rs", 18)) == late.CHANGED
    assert hist.status(_f("new.rs", 1)) == late.CHANGED  # did not exist then
    assert hist.status(_f("sp ace.rs", 1)) == late.CHANGED
    assert hist.status(_f("b.rs", 3)) == late.IN_REVIEW  # changed during review, not since
    assert hist.status(_f("c.rs", 3)) == late.SINCE_FIRST  # round-1 code, never changed
    assert hist.status(_f("h.rs", 5)) == late.SINCE_FIRST  # renamed, content unchanged
    assert hist.status(_f("c.rs", None)) == late.UNKNOWN  # no anchor
    assert hist.status(_f("", 3)) == late.UNKNOWN
    assert hist.status(_f("gone.rs", 3)) == late.UNKNOWN  # not in the head
    # the same head as the previous final review (a same-sha re-review): nothing changed
    assert late.CodeHistory(repo, prev, prev, first).status(_f("b.rs", 3)) == late.IN_REVIEW
    # a commit git does not have: cannot tell
    assert late.CodeHistory(repo, head, "0" * 40).status(_f("c.rs", 3)) == late.UNKNOWN


def test_changed_paths_reads_name_status_z():
    raw = "M\0a.rs\0A\0b c.rs\0D\0gone.rs\0R100\0old.rs\0new.rs\0R087\0x.rs\0y.rs\0T\0l\0"
    assert late.changed_paths(raw) == {"a.rs", "b c.rs", "y.rs", "l"}
    assert late.changed_paths("") == set()


# ---- the split and the confirmation contract ----


def _with(f, status):
    f.extra["code_status"] = status
    return f


def test_split_sends_only_new_findings_on_unchanged_code_off_the_normal_path():
    nit = _with(_f("f.rs", 1, title="nit", severity="nitpick"), late.SINCE_FIRST)
    sugg = _with(_f("f.rs", 2, title="sugg", severity="suggestion"), late.IN_REVIEW)
    listed = _with(_f("f.rs", 3, title="listed", severity="suggestion"), late.SINCE_FIRST)
    blocker = _with(_f("f.rs", 4, title="blocker", severity="blocking"), late.SINCE_FIRST)
    old = _with(_f("f.rs", 5, title="old", severity="blocking"), late.SINCE_FIRST)
    changed = _with(_f("f.rs", 6, title="changed", severity="nitpick"), late.CHANGED)
    unknown = _with(_f("f.rs", 7, title="unknown", severity="blocking"), late.UNKNOWN)
    split = late.split(
        [nit, sugg, listed, blocker, old, changed, unknown],
        is_new=lambda f: f.title != "old",
        listed={listed.dedupe_key},
    )
    assert split.nits == [nit] and split.followups == [sugg] and split.repeats == [listed]
    assert split.candidates == [blocker] and split.keep == [old, changed, unknown]
    assert split.changed
    assert not late.split([old, changed], is_new=lambda f: False, listed=set()).changed


def test_parse_decisions_needs_one_known_verdict_per_id_and_a_failure_to_block():
    ids = ["c1", "c2"]
    out = late.parse_decisions(
        {
            "decisions": [
                {"id": "c1", "verdict": "Blocking", "failure": "x reaches y", "reason": "r"},
                {"id": "c2", "verdict": "non-blocking"},
            ]
        },
        ids,
    )
    assert out["c1"] == late.Decision("blocking", "x reaches y", "r")
    assert out["c2"].verdict == "non_blocking"
    for raw, msg in (
        ({"decisions": [{"id": "c1", "verdict": "blocking"}]}, "c1: a blocking verdict needs"),
        ({"decisions": [{"id": "c1", "verdict": "maybe"}]}, "verdict 'maybe'"),
        ({"decisions": [{"id": "c9", "verdict": "invalid"}]}, "unknown id 'c9'"),
        ({"decisions": "no"}, "decisions must be a list"),
    ):
        with pytest.raises(ReviewError, match=msg):
            late.parse_decisions(raw, ids)
    with pytest.raises(ReviewError, match="no decision for c2"):
        late.parse_decisions({"decisions": [{"id": "c1", "verdict": "invalid"}]}, ids)


def test_the_confirmation_prompt_shows_the_claim_never_the_verifiers_reasoning(cfg):
    f = Finding.from_dict({**_blocking(), "confidence": 0.95, "source": "phase2:general"})
    briefs = late.candidate_briefs([f])
    assert briefs == [
        {
            "id": "c1",
            "file": "f.rs",
            "line_start": 11,
            "line_end": 12,
            "category": "logic",
            "title": f.title,
            "body": f.body,
        }
    ]
    p = confirmation_prompt(
        cfg,
        repo="dashpay/platform",
        number=1,
        head_sha=HEAD,
        meta={"title": "Fee fix", "body": "Fixes fees."},
        coverage_from="b" * 40,
        reported=briefs,
    )
    assert f"git diff {'b' * 40}..{HEAD}" in p and "Fixes fees." in p
    assert '"id": "c1"' in p and "confidence" not in p and "phase2:general" not in p
    assert "`blocking` verdict requires `failure`" in p


# ---- rendering ----


def _model(verified, *, fix_forward=True, late_report=None):
    prov = publish.Provenance(
        reviewers=[],
        verifier={"model": "m", "agent": "a", "role": "final-verifier"},
        policy_fingerprint="f" * 64,
        late=late_report,
    )
    return publish.ReviewModel(
        repo="r",
        number=1,
        head_sha=HEAD,
        phase="final",
        verified=verified,
        provenance=prov,
        kept=list(verified.findings),
        skipped=[],
        suppressed=[],
        comments=[],
        fix_forward=fix_forward,
    )


def _verified(findings, followups=(), action="APPROVE"):
    return VerifierOutput(
        summary="s",
        review_action=action,
        findings=list(findings),
        dropped=[],
        out_of_scope=[],
        coderabbit_reactions=[],
        prerequisite_adjudications=[],
        adjudication_complete=True,
        review_phase="final",
        raw={},
        followups=list(followups),
    )


def test_an_approval_with_suggestions_says_they_are_optional_and_lists_follow_ups():
    sugg = Finding.from_dict(_sugg())
    later = Finding.from_dict(_sugg("Name the constant"))
    body = publish.render(
        _model(
            _verified([sugg], [later]),
            late_report={"since": "abcd1234", "nits": 1, "followups": 1, "candidates": 0},
        )
    )
    assert "**Approved:** nothing here blocks the merge" in body
    assert "<summary>Follow-up candidates (optional) (1)</summary>" in body
    assert "Name the constant" in body and body.index("Follow-up candidates") > body.index(
        "🟡 1 suggestion(s)"
    )
    assert (
        "- Code unchanged since the last final review (`abcd1234`): 1 new nitpick(s) dropped; "
        "1 new suggestion(s) listed as follow-up candidates" in body
    )
    # the follow-up is not counted, nor in the prompt for agents
    assert "🟡 1 suggestion(s)" in body and "[SUGGESTION] f.rs:12: Name the constant" not in body
    # under a policy v12 pipeline (no fix-forward) and without an approval: no such note
    assert "**Approved:**" not in publish.render(_model(_verified([sugg]), fix_forward=False))
    assert "**Approved:**" not in publish.render(_model(_verified([sugg], action="COMMENT")))


def test_the_confirmation_outcome_is_disclosed():
    line = publish._late_line(
        {"since": "abcd1234", "candidates": 3, "confirmed": 1, "unconfirmed": 1, "invalid": 1}
    )
    assert "3 new blocker(s) went to an independent confirmation pass" in line
    assert "1 confirmed, 1 not confirmed (now follow-up candidates), 1 judged invalid" in line
    failed = publish._late_line({"since": "abcd1234", "candidates": 2, "failed": True})
    assert "2 new blocker(s) stand as raised: the independent confirmation pass failed" in failed


# ---- end to end: a later round on unchanged code ----

C, D = "c" * 40, "d" * 40
NIT = {**_sugg("Rename the local"), "severity": "nitpick", "line_start": 10, "line_end": 10}
LATE_SUGG = {**_sugg("Document the invariant"), "line_start": 13, "line_end": 13}
REAL = {**_blocking("Overflow on the fee sum"), "category": "bug"}
FALSE = {**_blocking("Lock order may deadlock"), "line_start": 14, "line_end": 14}
FRESH_SUGG = _sugg("Cover the new branch")  # on code this push changed
FALSE2 = {**FALSE, "title": "Lock order may deadlock (2)"}
PANIC = {
    **_blocking("Unchecked index into the peers list"),
    "category": "bug",
    "body": "peers[i] panics when the list is empty.",
    "line_start": 13,
    "line_end": 13,
}
STATUS = {
    PANIC["title"]: late.SINCE_FIRST,
    FALSE2["title"]: late.SINCE_FIRST,
    NIT["title"]: late.SINCE_FIRST,
    LATE_SUGG["title"]: late.IN_REVIEW,
    REAL["title"]: late.SINCE_FIRST,
    FALSE["title"]: late.SINCE_FIRST,
    FRESH_SUGG["title"]: late.CHANGED,
}


@pytest.fixture
def statuses(monkeypatch):
    monkeypatch.setattr(late.CodeHistory, "status", lambda self, f: STATUS.get(f.title, "A"))
    # no file changed since any earlier round (what a remembered confirmation asks)
    monkeypatch.setattr(late.CodeHistory, "changed_since", lambda self, since, path: False)


def _round(cfg, conn, gh, lanes, sha, final, recon=()):
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": [f for f in final if "finding_hash" in f],
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": list(recon),
    }
    lanes.verifier["preliminary"] = _verifier([])
    lanes.verifier["final"] = _verifier(final)
    rid = _queue_new_head(cfg, conn, gh, sha, lanes)
    assert worker_mod.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False) == (
        RunStatus.DONE
    )
    return rid


def _stages(conn, rid):
    return {
        (r["stage"], r["title"]): (r["severity"], r["code_status"])
        for r in conn.execute(
            "SELECT stage, title, severity, code_status FROM findings WHERE run_id=? "
            "AND (phase='verify2' OR stage='followup')",
            (rid,),
        )
    }


def _first_round(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["preliminary"] = lanes.verifier["final"] = _verifier([])
    _, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE and gh.posted_reviews[-1]["event"] == "APPROVE"


def test_a_later_round_gates_new_findings_on_unchanged_code(cfg, conn, gh, lanes, statuses):
    _first_round(cfg, conn, gh, lanes)
    lanes.confirm = {FALSE["title"]: "non_blocking"}
    rid = _round(cfg, conn, gh, lanes, C, [NIT, LATE_SUGG, REAL, FALSE, FRESH_SUGG])

    (confirm,) = [s for s in lanes.calls if s.role == "confirm"]
    assert confirm.model == "gpt-6-astra" and confirm.effort == "high"  # the final verifier's
    assert REAL["title"] in confirm.prompt and FALSE["title"] in confirm.prompt
    assert NIT["title"] not in confirm.prompt and FRESH_SUGG["title"] not in confirm.prompt
    assert "The change is sound overall" not in confirm.prompt  # no verifier reasoning
    review = gh.posted_reviews[-1]
    assert review["event"] == "REQUEST_CHANGES"
    inline = {c["body"].split("**")[1] for c in review["comments"]}
    assert inline == {f"🔴 Blocking: {REAL['title']}", f"🟡 Suggestion: {FRESH_SUGG['title']}"}
    real = next(c["body"] for c in review["comments"] if REAL["title"] in c["body"])
    assert "confirmed by an independent pass: a reachable failure" in real
    body = review["body"]
    assert NIT["title"] not in body
    followups = body.split("Follow-up candidates (optional) (2)", 1)[1]
    assert LATE_SUGG["title"] in followups and FALSE["title"] in followups
    assert "🔴 1 blocking | 🟡 1 suggestion(s)" in body
    assert "2 new blocker(s) went to an independent confirmation pass" in body
    assert (
        "1 confirmed, 1 not confirmed (now follow-up candidates); 1 new nitpick(s) dropped; "
        "1 new suggestion(s) listed as follow-up candidates, not posted inline" in body
    )
    assert _stages(conn, rid) == {
        ("verified", REAL["title"]): ("blocking", "C"),
        ("verified", FRESH_SUGG["title"]): ("suggestion", "A"),
        ("late-nit", NIT["title"]): ("nitpick", "C"),
        ("late-followup", LATE_SUGG["title"]): ("suggestion", "B"),
        ("late-unconfirmed", FALSE["title"]): ("blocking", "C"),
        ("followup", LATE_SUGG["title"]): ("suggestion", "B"),
        ("followup", FALSE["title"]): ("suggestion", "C"),
    }
    steps = {r["name"]: r for r in conn.execute("SELECT * FROM steps WHERE run_id=?", (rid,))}
    assert steps["confirm"]["status"] == "ok"
    assert json.loads(steps["confirm"]["detail"]) == {
        "candidates": 2,
        "blocking": 1,
        "non_blocking": 1,
        "invalid": 0,
    }
    v2 = json.loads(steps["verify2"]["detail"])
    assert (v2["findings"], v2["blockers"], v2["raw_findings"], v2["raw_blockers"]) == (2, 1, 5, 2)
    assert "fresh_phase2" not in steps  # the previous final review approved: no fresh pass
    (ev,) = conn.execute("SELECT detail FROM events WHERE kind='late.gated' AND run_id=?", (rid,))
    assert json.loads(ev["detail"])["nits"] == 1

    # the next push: the confirmed blocker is carried (it stays blocking, no new confirmation),
    # the follow-up candidate is raised again and is not restated
    hashes = {
        r["title"]: r["hash"]
        for r in conn.execute(
            "SELECT title, hash FROM posted_findings pf "
            "JOIN findings f USING (hash) WHERE f.stage='posted'"
        )
    }
    carried = {**REAL, "finding_hash": hashes[REAL["title"]]}
    recon = [
        {"finding_hash": hashes[REAL["title"]], "status": "STILL_VALID", "reason": "Still."},
        {"finding_hash": hashes[FRESH_SUGG["title"]], "status": "FIXED", "reason": "Done."},
    ]
    calls = len(lanes.calls)
    # the blocker the confirmation did not confirm is raised again on the same untouched file:
    # its earlier verdict stands, no lane asks again
    third = _round(cfg, conn, gh, lanes, D, [carried, LATE_SUGG, FALSE], recon)
    assert not [s for s in lanes.calls[calls:] if s.role == "confirm"]
    review = gh.posted_reviews[-1]
    assert review["event"] == "REQUEST_CHANGES"
    assert "Follow-up candidates" not in review["body"] and LATE_SUGG["title"] not in review["body"]
    assert FALSE["title"] not in review["body"]
    assert "2 follow-up candidate(s) already listed by an earlier review" in review["body"]
    assert "(1 as an earlier confirmation decided, the file unchanged since)" in review["body"]
    assert ("late-repeat", LATE_SUGG["title"]) in _stages(conn, third)
    assert ("late-unconfirmed", FALSE["title"]) in _stages(conn, third)
    assert ("verified", REAL["title"]) in _stages(conn, third)


def test_an_invalid_blocker_is_dropped_and_a_failed_confirmation_keeps_it(
    cfg, conn, gh, lanes, statuses
):
    _first_round(cfg, conn, gh, lanes)
    lanes.confirm = {FALSE["title"]: "invalid"}
    rid = _round(cfg, conn, gh, lanes, C, [FALSE])
    review = gh.posted_reviews[-1]
    assert review["event"] == "APPROVE" and not review["comments"]
    assert FALSE["title"] not in review["body"] and "1 judged invalid (dropped)" in review["body"]
    assert ("late-invalid", FALSE["title"]) in _stages(conn, rid)

    # the lane keeps answering out of contract: the blocker stands as raised
    lanes.confirm = lambda reported: {"decisions": [{"id": "c1", "verdict": "blocking"}]}
    rid = _round(cfg, conn, gh, lanes, D, [FALSE2])
    review = gh.posted_reviews[-1]
    assert review["event"] == "REQUEST_CHANGES"
    assert (
        "1 new blocker(s) stand as raised: the independent confirmation pass failed"
        in (review["body"])
    )
    step = conn.execute("SELECT * FROM steps WHERE run_id=? AND name='confirm'", (rid,)).fetchone()
    assert step["status"] == "failed"
    assert conn.execute(
        "SELECT 1 FROM events WHERE kind='late.confirm_failed' AND run_id=?", (rid,)
    ).fetchone()


def test_a_new_blocker_on_unchanged_code_holds_the_fresh_gate_and_is_confirmed(
    cfg, conn, gh, lanes, statuses
):
    """The previous final review's blocker is fixed, and a new blocker turns up on untouched
    code: the fresh final gate (which could lose it unconfirmed) does not run; it goes to its
    confirmation and, confirmed, blocks."""
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["preliminary"] = _verifier([])
    lanes.verifier["final"] = _verifier([FRESH_SUGG, _blocking()])
    _run(cfg, conn, gh, lanes)
    assert gh.posted_reviews[-1]["event"] == "REQUEST_CHANGES"
    fixed = [
        {"finding_hash": Finding.from_dict(f).hash, "status": "FIXED", "reason": "ok"}
        for f in (FRESH_SUGG, _blocking())
    ]
    rid = _round(cfg, conn, gh, lanes, C, [PANIC], fixed)
    steps = {
        r["name"]: r["status"] for r in conn.execute("SELECT * FROM steps WHERE run_id=?", (rid,))
    }
    assert "fresh_phase2" not in steps and steps["confirm"] == "ok"
    assert gh.posted_reviews[-1]["event"] == "REQUEST_CHANGES"
    assert ("verified", PANIC["title"]) in _stages(conn, rid)


def test_a_degraded_final_review_is_no_basis_for_unchanged_code(cfg, conn, gh, lanes, statuses):
    _first_round(cfg, conn, gh, lanes)
    with tx(conn):
        conn.execute("UPDATE runs SET degraded=1")
    rid = _round(cfg, conn, gh, lanes, C, [NIT, LATE_SUGG])
    steps = {r["name"] for r in conn.execute("SELECT name FROM steps WHERE run_id=?", (rid,))}
    # the next full-strength review is the first one: its second pass, no late gating
    assert "fresh_phase2" in steps
    assert NIT["title"] in gh.posted_reviews[-1]["body"] or any(
        NIT["title"] in c["body"] for c in gh.posted_reviews[-1]["comments"]
    )
    assert not conn.execute("SELECT 1 FROM events WHERE kind='late.gated'").fetchone()


def test_the_confirmation_is_never_repaired_and_unreadable_keeps_the_blocker(
    cfg, conn, gh, lanes, statuses
):
    _first_round(cfg, conn, gh, lanes)
    lanes.confirm = lambda reported: "I think c1 is fine"
    _round(cfg, conn, gh, lanes, C, [REAL])
    assert not [s for s in lanes.calls if s.role == "repair"]
    assert gh.posted_reviews[-1]["event"] == "REQUEST_CHANGES"


def test_the_first_reviews_second_pass_is_best_effort(cfg, conn, gh, lanes):
    from reviewsys.lane import LaneResult

    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["preliminary"] = lanes.verifier["final"] = _verifier([])

    def runner(spec, art, worktree):
        if "fresh final review" in spec.prompt:
            return LaneResult(exit_code=1, stdout="", stderr="boom", duration_s=1)
        return lanes(spec, art, worktree)

    rid, status = _run(cfg, conn, gh, runner)
    assert status == RunStatus.DONE
    steps = {
        r["name"]: r["status"] for r in conn.execute("SELECT * FROM steps WHERE run_id=?", (rid,))
    }
    assert steps["fresh_phase2"] == "failed" and steps["verify2"] == "ok"
    final_verifier = [c for c in lanes.calls if c.role == "verifier"][-1]
    assert "fresh:general" not in final_verifier.prompt
    body = gh.posted_reviews[-1]["body"]
    assert "Second Phase-2 pass" not in body and gh.posted_reviews[-1]["event"] == "APPROVE"
    assert conn.execute("SELECT 1 FROM events WHERE kind='fresh_first.failed'").fetchone()


def test_a_fast_tier_skips_the_second_pass(cfg, conn, gh, lanes):
    lanes.triage = {"tier": "low", "reasoning": "small"}
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["preliminary"] = lanes.verifier["final"] = _verifier([])
    rid, _ = _run(cfg, conn, gh, lanes)
    steps = {r["name"] for r in conn.execute("SELECT name FROM steps WHERE run_id=?", (rid,))}
    assert "phase2" in steps and "fresh_phase2" not in steps


def test_a_verdict_update_lists_the_follow_ups():
    later = Finding.from_dict(_sugg("Name the constant"))
    body = publish.render_verdict_update(
        head_sha=HEAD,
        phase="final",
        verified=_verified([], [later]),
        provenance=publish.Provenance(
            reviewers=[],
            verifier={"model": "m", "agent": "a", "role": "final-verifier"},
            policy_fingerprint="f" * 64,
        ),
        previous_event="CHANGES_REQUESTED",
        new_event="APPROVE",
        lifted_blockers=[],
    )
    assert "<summary>Follow-up candidates (optional) (1)</summary>" in body
    assert "Name the constant" in body
