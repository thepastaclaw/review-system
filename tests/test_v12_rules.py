"""Policy v12, pure rules: the candidate contract, sharding, the verify+match answer, the code
rules (§2.3), the ledger verdict (§3), the thread lane's answer, the reconcile, and the
config's switch combinations."""

from __future__ import annotations

import dataclasses
import json

import pytest

from reviewsys import converse, ledger, prompts, rules
from reviewsys.contract import Finding
from reviewsys.db import kv_get, tx
from reviewsys.ledger import Issue
from reviewsys.models import ReviewError
from tests.test_pipeline_policy import load_with

H_OPEN, H_SUGG, H_WD, H_FIXED, H_RES, H_PEND = (
    "aaaaaaaaaaaa",
    "bbbbbbbbbbbb",
    "cccccccccccc",
    "dddddddddddd",
    "eeeeeeeeeeee",
    "ffffffffffff",
)


def issue(h: str, status: str = "open", severity: str = "blocking", **kw) -> Issue:
    base = dict(
        repo="r/x",
        number=1,
        hash=h,
        file="f.rs",
        line_start=10,
        line_end=12,
        anchor_sha="s1",
        severity=severity,
        category="logic",
        title=f"issue {h[:3]}",
        body="b",
        status=status,
        status_sha="s1",
        status_source="verifier",
        opened_run=1,
        opened_sha="s1",
        last_seen_run=1,
        last_seen_sha="s1",
        closed_sha=None if status in rules.VERDICT_STANDING else "s2",
        closed_reason=None,
        thread_id="T",
        comment_id=77,
        created_at="2026-10-01T00:00:00Z",
        updated_at="2026-10-01T00:00:00Z",
    )
    base.update(kw)
    return Issue(**base)


LEDGER = {
    H_OPEN: issue(H_OPEN),
    H_SUGG: issue(H_SUGG, severity="suggestion"),
    H_WD: issue(H_WD, "withdrawn"),
    H_FIXED: issue(H_FIXED, "fixed"),
    H_RES: issue(H_RES, "resolved"),
    H_PEND: issue(H_PEND, "deferral_pending"),
}


def cands(*sev: str) -> dict[str, Finding]:
    out = {}
    for i, s in enumerate(sev, 1):
        f = Finding(
            file="f.rs",
            title=f"c{i}",
            body="body",
            severity=s,
            line_start=10 + i,
            line_end=10 + i,
            source="phase2:general",
            candidate_id=f"p2-general-{i}",
        )
        out[f.candidate_id] = f
    return out


def answer(*groups: dict) -> rules.VerifyAnswer:
    gs = [
        rules.Group(
            members=g["members"],
            kind=g.get("kind", "new"),
            target=g.get("target", ""),
            verdict=g.get("verdict", "CONFIRMED"),
            evidence=g.get("evidence", "line 11"),
            confirm_by=g.get("confirm_by", ""),
            severity=g.get("severity", "blocking"),
            title=g.get("title", "T"),
            body=g.get("body", "B"),
            suggestion=g.get("suggestion"),
        )
        for g in groups
    ]
    return rules.VerifyAnswer(summary="s", groups=gs, raw={})


def apply(ans, cs, *, changed=None, extra_open=()):
    return rules.apply_rules(
        ans,
        cs,
        LEDGER,
        changed_since_closed=lambda i: changed,
        source_of=lambda ms: ",".join(m.candidate_id or "" for m in ms),
        thread_url=lambda i: f"https://x/{i.hash}",
        extra_open=extra_open,
    )


# ---- finder candidates ----


# ---- shards ----


def _many(files: dict[str, int]) -> list[Finding]:
    out = []
    for f, n in files.items():
        out += [
            Finding(file=f, title=f"{f}{i}", body="", candidate_id=f"{f}-{i}") for i in range(n)
        ]
    return out


def test_twelve_candidates_are_one_batch():
    cs = _many({"a": 12})
    assert rules.shards(cs) == [cs]


def test_more_are_split_by_file_into_at_most_three_batches():
    cs = _many({"a": 9, "b": 6, "c": 5, "d": 1})
    batches = rules.shards(cs)
    assert len(batches) == 2 and sorted(map(len, batches)) == [10, 11]
    for b in batches:  # a file's candidates are never split
        files = {c.file for c in b}
        assert all(sum(c.file == f for c in b) == sum(c.file == f for c in cs) for f in files)
    assert len(rules.shards(_many({f"f{i}": 3 for i in range(30)}))) == 3


# ---- the verify+match answer ----


def _raw_group(members, **kw):
    return {
        "members": members,
        "match": "new",
        "verdict": "CONFIRMED",
        "evidence": "line",
        "severity": "blocking",
        "title": "t",
        "body": "b",
        **kw,
    }


def test_every_candidate_in_exactly_one_group_or_every_problem_named():
    raw = {
        "summary": "s",
        "groups": [
            _raw_group(["p2-general-1", "p2-general-9"]),
            _raw_group(["p2-general-1"], match="open:zzzzzzzzzzzz", verdict="MAYBE", evidence=""),
            _raw_group(["p2-general-2"], match="coderabbit:5", severity="huge", title=""),
        ],
    }
    with pytest.raises(ReviewError) as exc:
        rules.parse_verify(
            raw,
            candidate_ids=["p2-general-1", "p2-general-2", "p2-general-3"],
            issues=LEDGER,
            coderabbit_ids=[7],
        )
    msg = exc.value.message
    for part in (
        "'p2-general-9' is not a candidate id",
        "p2-general-1 is already in group 1",
        "names no issue listed",
        "verdict 'MAYBE'",
        "needs `evidence`",
        "names no CodeRabbit comment",
        "severity 'huge'",
        "no title",
        "candidates in no group: ['p2-general-3']",
    ):
        assert part in msg


def test_matches_expand_prefixes_and_follow_the_ledger_status():
    raw = {
        "summary": "s",
        "groups": [
            _raw_group(["a"], match=f"open:{H_WD[:8]}"),  # withdrawn: settled
            _raw_group(["b"], match=f"closed:{H_FIXED}"),
            _raw_group(["c"], match=f"settled:{H_RES}"),  # hand-resolved still counts: open
            _raw_group(["d"], match="coderabbit:7"),
        ],
    }
    out = rules.parse_verify(raw, candidate_ids=list("abcd"), issues=LEDGER, coderabbit_ids=[7])
    assert [g.match for g in out.groups] == [
        f"settled:{H_WD}",
        f"fixed:{H_FIXED}",
        f"open:{H_RES}",
        "coderabbit:7",
    ]
    assert any("-> 'cccccccccccc'" in n for n in out.normalized)


def test_a_coderabbit_retrigger_is_refused():
    with pytest.raises(ReviewError, match="retrigger"):
        rules.parse_verify(
            {"groups": [], "summary": "@coderabbitai review"}, candidate_ids=[], issues={}
        )


# ---- the code rules ----


def test_blocking_requires_confirmed():
    cs = cands("blocking", "blocking")
    out = apply(
        answer(
            {"members": ["p2-general-1"], "title": "real"},
            {
                "members": ["p2-general-2"],
                "verdict": "PLAUSIBLE",
                "confirm_by": "a loom test",
                "title": "maybe",
            },
        ),
        cs,
    )
    real, maybe = out.new
    assert real.severity == "blocking" and maybe.severity == "suggestion"
    assert "What would confirm it: a loom test" in maybe.body
    assert real.candidate_id == "p2-general-1" and real.source == "p2-general-1"


def test_refuted_is_dropped_and_a_refuted_coderabbit_restatement_disagrees():
    out = apply(
        answer(
            {"members": ["p2-general-1"], "verdict": "REFUTED", "evidence": "guard at L9"},
            {
                "members": ["p2-general-2"],
                "kind": "coderabbit",
                "target": "7",
                "verdict": "REFUTED",
                "evidence": "no",
            },
            {"members": ["p2-general-3"], "kind": "coderabbit", "target": "8", "title": "cr"},
        ),
        cands("blocking", "suggestion", "suggestion"),
    )
    assert [d["reason"] for d in out.dropped] == ["REFUTED: guard at L9", "REFUTED: no"]
    assert out.reactions == [
        {"comment_id": 7, "action": "disagree", "reply": "no"},
        {"comment_id": 8, "action": "agree"},
    ]
    assert [f.title for f in out.new] == ["cr"]  # kept as ours, as v11 did


def test_an_open_match_corroborates_and_never_opens_a_thread():
    out = apply(
        answer(
            {"members": ["p2-general-1"], "kind": "open", "target": H_OPEN},
            {"members": ["p2-general-2"], "kind": "open", "target": H_SUGG, "severity": "blocking"},
            {"members": ["p2-general-3"], "kind": "open", "target": H_SUGG, "verdict": "PLAUSIBLE"},
        ),
        cands("blocking", "blocking", "blocking"),
    )
    assert out.new == []
    assert out.corroborated == {H_OPEN: ["p2-general-1"], H_SUGG: ["p2-general-2", "p2-general-3"]}
    # a CONFIRMED blocker raises an open suggestion in place; a PLAUSIBLE one never would
    assert list(out.upgrades) == [H_SUGG]


@pytest.mark.parametrize(("changed", "kept"), [(True, True), (False, False), (None, False)])
def test_a_settled_issue_revives_only_when_its_code_changed(changed, kept):
    out = apply(
        answer({"members": ["p2-general-1"], "kind": "settled", "target": H_WD, "title": "again"}),
        cands("blocking"),
        changed=changed,
    )
    assert bool(out.new) is kept
    if kept:
        assert out.new[0].extra["revives"] == H_WD and "withdrawn at `s2`" in out.new[0].body
    else:
        assert out.dropped[0]["reason"].startswith("SETTLED: matches withdrawn issue")


def test_a_refound_fixed_issue_is_a_regression_under_its_own_identity():
    title = LEDGER[H_FIXED].title
    f = Finding(file="f.rs", title=title, body="")
    fixed = dataclasses.replace(LEDGER[H_FIXED], hash=f.hash)
    out = rules.apply_rules(
        answer({"members": ["p2-general-1"], "kind": "fixed", "target": f.hash, "title": title}),
        cands("blocking"),
        {f.hash: fixed},
        changed_since_closed=lambda i: None,
        source_of=lambda ms: "",
        thread_url=lambda i: "https://x/old",
    )
    (reg,) = out.new
    assert reg.title == f"{title} (regression)" and reg.hash != f.hash
    assert "regression of an earlier finding that was fixed: https://x/old" in reg.body


def test_a_new_title_that_is_a_ledger_issue_is_that_issue():
    title = "issue aaa"
    h = Finding(file="f.rs", title=title, body="", category="logic").hash
    led = {h: dataclasses.replace(LEDGER[H_WD], hash=h, title=title)}
    cs = cands("blocking")
    cs["p2-general-1"].category = "logic"
    out = rules.apply_rules(
        answer({"members": ["p2-general-1"], "title": title}),
        cs,
        led,
        changed_since_closed=lambda i: False,
        source_of=lambda ms: "",
    )
    assert out.new == [] and out.dropped[0]["reason"].startswith("SETTLED")


def test_same_title_twice_and_this_rounds_own_issues_are_not_posted_twice():
    out = apply(
        answer(
            {"members": ["p2-general-1"], "title": "same"},
            {"members": ["p2-general-2"], "title": "same"},
            {"members": ["p2-general-3"], "kind": "open", "target": "123456789abc"},
            {"members": ["p2-general-4"], "title": "raised before"},  # "new", same title
        ),
        cands("blocking", "blocking", "suggestion", "suggestion"),
        extra_open={"123456789abc", Finding(file="f.rs", title="raised before", body="").hash},
    )
    assert len(out.new) == 1
    assert [d["reason"].split(":")[0] for d in out.dropped] == ["DUPLICATE"] * 3


# ---- the ledger verdict ----


@pytest.mark.parametrize(
    ("blockers", "suggestions", "full", "confirmed", "verdict"),
    [
        (1, 0, True, True, "REQUEST_CHANGES"),
        (1, 3, False, False, "REQUEST_CHANGES"),
        (0, 1, True, True, "COMMENT"),  # no APPROVE while a suggestion stands
        (0, 0, True, True, "APPROVE"),
        (0, 0, False, True, "COMMENT"),  # Phase 1 only, degraded, a dropped shard
        (0, 0, True, False, "COMMENT"),  # the confirmation pass was due and did not run
    ],
)
def test_verdict_table(blockers, suggestions, full, confirmed, verdict):
    got = rules.verdict(
        blockers=blockers, suggestions=suggestions, full_strength=full, confirmed=confirmed
    )
    assert got == verdict


def test_standing_counts_hand_resolved_and_pending_deferrals():
    assert {i.hash for i in rules.standing(LEDGER)} == {H_OPEN, H_SUGG, H_RES, H_PEND}


def test_until_reconciled_only_recent_or_threaded_issues_count():
    led = {
        H_OPEN: issue(H_OPEN, last_seen_sha="old"),
        H_SUGG: issue(H_SUGG, last_seen_sha="prev"),
        H_RES: issue(H_RES, last_seen_sha="old"),
    }
    got = rules.standing(led, reconciled=False, last_reviewed_sha="prev", open_threads={H_RES})
    assert {i.hash for i in got} == {H_SUGG, H_RES}


def test_projection_applies_pending_statuses_and_upgrades_without_touching_the_rows():
    got = rules.project(LEDGER, {H_OPEN: "withdrawn"}, [H_SUGG])
    assert got[H_OPEN].status == "withdrawn" and got[H_SUGG].severity == "blocking"
    assert LEDGER[H_OPEN].status == "open" and LEDGER[H_SUGG].severity == "suggestion"


# ---- what the lanes are told ----


def test_settled_list_is_withdrawn_and_deferred_newest_ten():
    led = {
        f"{i:012x}": issue(
            f"{i:012x}", "withdrawn" if i % 2 else "deferred", updated_at=f"2026-10-{i + 1:02d}"
        )
        for i in range(14)
    }
    led["0" * 11 + "f"] = issue("0" * 11 + "f", "outdated")
    got = rules.settled_list(led, {f"{13:012x}": "We keep it as is."})
    assert len(got) == 10 and got[0]["reason"] == "We keep it as is."
    assert all(r["status"] in {"withdrawn", "deferred"} for r in got)


def test_ledger_summary_buckets():
    raised = Finding(file="g.rs", title="new one", body="")
    s = rules.ledger_summary(
        LEDGER,
        thread_status={H_OPEN: "FIXED"},
        last_message={H_OPEN: "knst: done"},
        raised_now=[raised],
    )
    assert {r["hash"] for r in s["open"]} == {H_OPEN, H_SUGG, H_RES, H_PEND, raised.hash}
    assert {r["hash"] for r in s["settled"]} == {H_WD} and {r["hash"] for r in s["fixed"]} == {
        H_FIXED
    }
    row = next(r for r in s["open"] if r["hash"] == H_OPEN)
    assert row["thread_lane"] == "FIXED" and row["last_message"] == "knst: done"


# ---- the thread lane's answer ----


def test_thread_answers_cover_the_work_set_exactly_once():
    with pytest.raises(ReviewError) as exc:
        converse.parse_threads(
            {"threads": [{"finding_hash": "a", "status": "FIXED", "reply": "x"}] * 2},
            hashes=["a", "b"],
            fix_pending=set(),
        )
    assert (
        "answered more than once" in exc.value.message and "no entry for ['b']" in exc.value.message
    )


def test_thread_answer_rules():
    out = converse.parse_threads(
        {
            "threads": [
                {"finding_hash": "a", "status": "FIXED", "reply": "the patch does it"},
                {"finding_hash": "b", "status": "fixed", "reply": "done"},
                {"finding_hash": "c", "status": "Still Valid", "reply": "NO_REPLY"},
                {"finding_hash": "d", "status": "NO_REPLY", "reply": ""},
            ]
        },
        hashes=["a", "b", "c", "d"],
        fix_pending={"a"},
    )
    assert {h: o.status for h, o in out.outcomes.items()} == {
        "a": "FIX_PENDING",  # nothing under it changed
        "b": "FIXED",
        "c": "STILL_VALID",
        "d": "STILL_VALID",
    }
    assert out.outcomes["c"].reply == "" and out.outcomes["d"].reply == ""
    with pytest.raises(ReviewError, match="WITHDRAWN needs a reply"):
        converse.parse_threads(
            {"threads": [{"finding_hash": "a", "status": "WITHDRAWN", "reply": "NO_REPLY"}]},
            hashes=["a"],
            fix_pending=set(),
        )


def test_a_deferral_needs_a_maintainer_on_the_thread_and_a_visible_tracking_issue():
    def out(h, cited=""):
        return converse.ConversationOutput(
            {h: converse.ThreadOutcome(h, "INTENTIONALLY_DEFERRED", "ok", tracking_url=cited)}
        )

    def status(o, transcript, **kw):
        t = {"a": {"transcript": transcript, "severity": "blocking"}}
        return converse.accept_deferrals(o, t, repo="r/x", number=1, can_file=False, **kw)

    maint = [{"is_bot": True}, {"association": "MEMBER", "body": "defer, tracked in #12"}]
    untracked = [{"is_bot": True}, {"association": "MEMBER", "body": "a follow-up later"}]
    outsider = [{"association": "MEMBER"}, {"is_bot": True}, {"association": "CONTRIBUTOR"}]
    tracked = status(out("a"), maint).outcomes["a"]
    assert (tracked.status, tracked.tracking_url) == (
        "INTENTIONALLY_DEFERRED",
        "https://github.com/r/x/issues/12",
    )
    # cited by the lane (its reply) and visible on the PR (here: the description)
    cited = status(out("a", "r/x#40"), untracked, pr_text="Follow-up: r/x#40").outcomes["a"]
    assert cited.status == "INTENTIONALLY_DEFERRED"
    # a maintainer's call, but nothing tracks it: the issue stands and the reply asks for a
    # link (the thread lane cannot have one filed)
    held = status(out("a", "r/x#41"), untracked).outcomes["a"]
    assert held.status == converse.DEFERRAL_NEEDS_TRACKING
    assert "Link one here" in held.reply and "file it" not in held.reply
    assert status(out("a"), outsider).outcomes["a"].status == "DEFERRAL_PENDING"
    assert status(out("a"), []).outcomes["a"].status == "DEFERRAL_PENDING"
    # already deferred in the ledger: a restatement is not vetted again
    assert status(out("a"), untracked, settled={"a"}).outcomes["a"].status == (
        "INTENTIONALLY_DEFERRED"
    )


def test_a_thread_lane_reply_cites_its_tracking_reference():
    def cite(reply, said="", visible=""):
        return converse.cited_reference(
            reply, said=said, visible=said + "\n" + visible, repo="r/x", number=1
        )

    reply = "This matches upstream bitcoin/bitcoin#31000; the follow-up is tracked in #7012."
    # the one the maintainer named beats an upstream PR the description links
    assert (
        cite(reply, said="I opened #7012", visible="Backport of bitcoin/bitcoin#31000")
        == "https://github.com/r/x/issues/7012"
    )
    assert cite(reply, said="see #7012") == "https://github.com/r/x/issues/7012"
    # only the description links one: that one
    assert cite(reply, visible="Tracked in #7012.") == "https://github.com/r/x/issues/7012"
    # none visible: the first named, which accept_deferrals reports as not found
    assert cite(reply) == "https://github.com/bitcoin/bitcoin/issues/31000"
    # the line the reply marks as tracking wins over an upstream PR a maintainer named
    both = "This mirrors bitcoin/bitcoin#31000.\nTracked in #7012."
    assert (
        cite(both, said="Defer, mirrors bitcoin/bitcoin#31000", visible="Follow-up: #7012")
        == "https://github.com/r/x/issues/7012"
    )
    assert cite("Deferred to #3; see #1.") == "https://github.com/r/x/issues/3"
    assert cite("This PR (#1) only.") == ""


# ---- ledger: reconcile, severity, seen, outcomes without a thread ----


def _seed(conn, h, *, run=1, sha="s1", severity="blocking"):
    f = Finding(file="f.rs", title=f"t-{h}", body="b", severity=severity, line_start=5, line_end=6)
    ev = ledger._posted(f, run_id=run, sha=sha, source="backfill", h=h)
    with tx(conn):
        ledger.append(conn, "r/x", 1, [ev])


def test_reconcile_outdates_what_cannot_stand_and_leaves_the_rest_to_the_thread_lane(conn):
    for h in ("a" * 12, "b" * 12, "c" * 12, "d" * 12):
        _seed(conn, h)
    with tx(conn):
        ledger.transition(
            conn, "r/x", 1, "d" * 12, "withdrawn", run_id=None, sha=None, source="manual"
        )
        pending = ledger.reconcile(
            conn,
            "r/x",
            1,
            pr_open=True,
            open_threads={"a" * 12},
            moved={"a" * 12: True, "b" * 12: True, "c" * 12: None},
            run_id=9,
            sha="head",
        )
    led = ledger.load(conn, "r/x", 1)
    assert pending == ["a" * 12, "c" * 12]  # a threaded one stays; git could not tell for c
    assert led["b" * 12].status == "outdated" and led["b" * 12].status_source == "backfill"
    assert led["d" * 12].status == "withdrawn"
    rec = json.loads(kv_get(conn, "ledger.reconciled:r/x#1"))
    assert rec["at"] and {k: rec[k] for k in ("run", "sha", "pending")} == {
        "run": 9,
        "sha": "head",
        "pending": pending,
    }
    with tx(conn):
        ledger.settle_pending(conn, "r/x", 1, ["a" * 12])
    after = ledger.reconciled(conn, "r/x", 1)
    assert after["pending"] == ["c" * 12]
    assert after["pending_at_reconcile"] == pending  # what the reconcile left, kept


def test_reconcile_on_a_closed_pr_outdates_every_standing_issue(conn):
    _seed(conn, "a" * 12)
    with tx(conn):
        assert (
            ledger.reconcile(conn, "r/x", 1, pr_open=False, open_threads={"a" * 12}, moved={}) == []
        )
    assert ledger.load(conn, "r/x", 1)["a" * 12].status == "outdated"


def test_severity_is_raised_in_place_and_seen_moves_last_seen(conn):
    _seed(conn, "a" * 12, severity="suggestion")
    with tx(conn):
        ledger.raise_severity(
            conn, "r/x", 1, "a" * 12, "blocking", run_id=2, sha="s2", source="verifier"
        )
        ledger.record_seen(
            conn, "r/x", 1, run_id=2, sha="s2", hashes=["a" * 12, "z" * 12], source="verifier"
        )
    i = ledger.load(conn, "r/x", 1)["a" * 12]
    assert (i.severity, i.last_seen_sha, i.anchor_sha) == ("blocking", "s2", "s1")


def test_an_outcome_on_an_issue_without_a_thread_still_moves_it(conn):
    _seed(conn, "a" * 12)
    with tx(conn):
        ledger.record_answers(
            conn,
            "r/x",
            1,
            run_id=3,
            sha="s3",
            answered=[{"finding_hash": "a" * 12, "status": "FIXED", "action": "no_thread"}],
            source="thread",
        )
    assert ledger.load(conn, "r/x", 1)["a" * 12].status == "fixed"


# ---- config: switches that only work together ----


@pytest.mark.parametrize(
    ("block", "error"),
    [
        (
            {"verifier": {"contract": "v12"}, "threads": {"lane": True}},
            'needs finders.anchoring "unanchored"',
        ),
        ({"finders": {"anchoring": "unanchored"}, "verifier": {"contract": "v12"}}, "go together"),
        ({"threads": {"lane": True}}, "go together"),
    ],
)
def test_half_switched_pipelines_fail_the_load(cfg, skills_dir, block, error):
    with pytest.raises(ValueError, match=error):
        load_with(cfg, skills_dir, {"version": "v12", **block})


# ---- prompts ----


def test_correction_turns_re_tell_the_ids_and_follow_the_output_mode():
    text = prompts.correction_prompt(
        errors=["candidates in no group: ['p2-general-2']"],
        kind="verify",
        expected_phase="final",
        head_sha="h",
        prior=[],
        coderabbit_ids=[],
        turn=1,
        turns=2,
        stamp_echo=True,
        ids=["p2-general-1", "p2-general-2"],
        structured=True,
    )
    assert "keys summary, groups[]" in text and "p2-general-1, p2-general-2" in text
    assert "StructuredOutput tool" in text and "raw JSON object" not in text
    thread = prompts.correction_prompt(
        errors=["x"],
        kind="thread",
        expected_phase="",
        head_sha="h",
        prior=[],
        coderabbit_ids=[],
        turn=1,
        turns=2,
        ids=["aaa"],
    )
    assert "`finding_hash` values, copied exactly, and no others: aaa" in thread
    assert thread.endswith("no prose before or after it.")


def test_repair_prompt_for_v12_kinds_names_their_ids():
    text = prompts.repair_prompt(
        "{", kind="thread", expected_phase="", head_sha="h", prior_hashes=["aaa"], stamp_echo=True
    )
    assert 'every `finding_hash` in threads is one of "aaa"' in text and "review_phase" not in text


@pytest.mark.parametrize("state", ["open", "closed"])
def test_reconcile_all_before_the_flip(conn, gh, state):
    """`reviewsys ledger reconcile`: every PR with standing issues, once; a closed PR's are
    outdated, an open PR's are reconciled against its live head (no mirror: nothing is
    outdated for a code change)."""
    for h in ("a" * 12, "b" * 12):
        _seed(conn, h)
    gh.pr["state"] = state
    out = ledger.reconcile_all(conn, gh, "thepastaclaw", mirror_for=lambda r: None)
    led = ledger.load(conn, "r/x", 1)
    if state == "closed":
        assert out["closed"] == 1 and {i.status for i in led.values()} == {"outdated"}
    else:
        assert out["open"] == 1 and out["pending"] == 2
        assert {i.status for i in led.values()} == {"open"}
    again = ledger.reconcile_all(conn, gh, "thepastaclaw", mirror_for=lambda r: None)
    assert again["prs"] == 0 and again["skipped"] == (state == "open")


def test_anchor_moved():
    i = issue(H_OPEN, anchor_sha="s1", last_seen_sha="s2")
    asked = []

    def changed(a, b, file, ls, le):
        asked.append((a, ls))
        return {("s1", 10): True, ("s2", None): False}.get((a, ls))

    assert ledger.anchor_moved(i, "head", changed) is False  # nothing since it was last seen
    assert asked == [("s1", 10), ("s2", None)]
    assert ledger.anchor_moved(i, "s2", changed) is False  # judged at this very head
    assert ledger.anchor_moved(dataclasses.replace(i, last_seen_sha="s1"), "head", changed) is True
    assert (
        ledger.anchor_moved(
            dataclasses.replace(i, anchor_sha=None, opened_sha=None), "head", changed
        )
        is False
    )


def test_the_comment_budget_keeps_every_blocker_and_the_best_suggestions():
    def f(title, sev, verdict="CONFIRMED", members=1):
        return Finding(
            file="f.rs",
            title=title,
            body="",
            severity=sev,
            extra={"verdict": verdict, "members": ["x"] * members},
        )

    new = [
        f("n1", "nitpick"),
        f("s-plausible", "suggestion", "PLAUSIBLE"),
        f("b1", "blocking"),
        f("s-two-lanes", "suggestion", members=2),
        f("b2", "blocking"),
        f("s1", "suggestion"),
    ]
    posted, over = rules.within_budget(new, 4)
    assert [x.title for x in posted] == ["b1", "s-two-lanes", "b2", "s1"]
    assert [x.title for x in over] == ["n1", "s-plausible"]
    assert rules.within_budget(new, 1)[0] == [new[2], new[4]]  # blockers are never over budget
    assert rules.within_budget(new, 0) == (new, [])


def test_the_thread_prompt_carries_ci_and_bounded_discussion(cfg, skills_dir):
    (skills_dir / "prompts" / "v12" / "thread.md").write_text(
        "{static_review}\n{issues}\n## Discussion\n{discussion}\n## CI\n{ci}\n"
    )
    text = prompts.thread_prompt(
        cfg,
        repo="dashpay/platform",
        number=7,
        head_sha="a" * 40,
        meta={"title": "t"},
        issues=[{"finding_hash": "abc"}],
        linked_commits="",
        evidence={"issue_comments": [{"body": "x" * 20000}], "ci": {"state": "failure"}},
        standing_verdict=None,
        same_sha=False,
        structured=False,
    )
    discussion = text.split("## Discussion\n", 1)[1].split("## CI", 1)[0]
    assert discussion.startswith("```json\n") and discussion.rstrip().endswith("```")
    assert '"state": "failure"' in text and "the `ci` snapshot in the CI section" in text
    assert "`abc`" in text.split("## Output: thread outcomes", 1)[1]


def test_outdated_where_nothing_changed_is_a_pending_fix_too():
    out = converse.parse_threads(
        {
            "threads": [
                {
                    "finding_hash": "a",
                    "status": "OUTDATED",
                    "reply": "that path is gone in the patch",
                },
                {"finding_hash": "b", "status": "OUTDATED", "reply": "the code moved away"},
            ]
        },
        hashes=["a", "b"],
        fix_pending={"a"},
    )
    assert {h: o.status for h, o in out.outcomes.items()} == {"a": "FIX_PENDING", "b": "OUTDATED"}


def test_thread_shards_split_by_count_and_size_into_at_most_three():
    assert rules.thread_shards({}) == []
    small = {f"h{i}": 100 for i in range(12)}
    assert rules.thread_shards(small) == [list(small)]
    many = {f"h{i}": 100 for i in range(14)}
    batches = rules.thread_shards(many)
    assert len(batches) == 2 and sorted(map(len, batches)) == [7, 7]
    assert sorted(h for b in batches for h in b) == sorted(many)
    big = {"a": 50_000, "b": 50_000, "c": 10}  # few issues, but too much to read at once
    assert len(rules.thread_shards(big)) == 2
    assert len(rules.thread_shards({f"h{i}": 100 for i in range(100)})) == 3


def test_a_one_click_suggestion_only_where_every_report_points():
    cs = cands("blocking", "blocking", "blocking")  # lines 11, 12, 13
    same = dict(cs)
    same["p2-general-2"].line_start = same["p2-general-2"].line_end = 11
    out = apply(
        answer(
            {"members": ["p2-general-1", "p2-general-2"], "title": "one", "suggestion": "fix()"},
            {"members": ["p2-general-3"], "title": "two", "suggestion": "other()"},
        ),
        same,
    )
    assert [f.suggestion for f in out.new] == ["fix()", "other()"]
    out = apply(
        answer(
            {"members": ["p2-general-1", "p2-general-3"], "title": "one", "suggestion": "fix()"}
        ),
        cands("blocking", "blocking", "blocking"),
    )
    (f,) = out.new
    assert f.suggestion is None and "**Suggested change**" in f.body and "fix()" in f.body


def test_a_retried_answer_records_what_the_thread_shows(gh):
    """A retry of a round already answered a thread (same head, same reply) with another
    outcome than this attempt decided: nothing new is posted, and the outcome recorded is
    the one on the thread, so the ledger never contradicts it."""
    from reviewsys import publish

    marker = publish.THREAD_ANSWER_MARKER.format(sha="h" * 40, reply=901, finding="abc")
    gh.inline = [
        {
            "id": 1,
            "body": publish.render_thread_answer(
                marker=marker, status="WITHDRAWN", head_sha="h" * 40, reply="no"
            ),
        },
    ]
    thread = {
        "comment_id": 800,
        "thread_id": "T",
        "latest_reply_id": 901,
        "awaiting_answer": True,
        "body": "root",
    }
    out = {"abc": converse.ThreadOutcome("abc", "STILL_VALID", "it still holds")}
    (item,) = publish.answer_thread_lane(
        gh, "r/x", 1, "h" * 40, outcomes=out, threads={"abc": thread}, same_sha=False
    )
    assert (item["action"], item["status"], item["decided"]) == (
        "already_answered",
        "WITHDRAWN",
        "STILL_VALID",
    )
    assert gh.replies == []
    for status in (
        "WITHDRAWN",
        "FIX_PENDING",
        "INTENTIONALLY_DEFERRED",
        "DEFERRAL_PENDING",
        "STILL_VALID",
    ):
        body = publish.render_conversation_answer(marker=marker, status=status, reply="prose")
        assert publish.shown_status(body) == status
    for status in ("FIXED", "OUTDATED", "STILL_VALID", "WITHDRAWN"):
        body = publish.render_thread_answer(
            marker=marker, status=status, head_sha="h" * 40, reply="x"
        )
        assert publish.shown_status(body) == status


def test_thread_answers_may_name_a_hash_by_a_unique_prefix():
    out = converse.parse_threads(
        {"threads": [{"finding_hash": "abcdef12", "status": "STILL_VALID", "reply": "NO_REPLY"}]},
        hashes=["abcdef123456", "0123456789ab"][:1],
        fix_pending=set(),
    )
    assert list(out.outcomes) == ["abcdef123456"]
    with pytest.raises(ReviewError, match="not one of the listed"):
        converse.parse_threads(
            {"threads": [{"finding_hash": "abc", "status": "STILL_VALID", "reply": "x"}]},
            hashes=["abcdef123456"],
            fix_pending=set(),
        )


def test_a_refuted_group_needs_no_final_wording():
    out = rules.parse_verify(
        {
            "summary": "s",
            "groups": [
                {"members": ["a"], "match": "new", "verdict": "REFUTED", "evidence": "guard at L3"}
            ],
        },
        candidate_ids=["a"],
        issues={},
    )
    (g,) = out.groups
    assert g.verdict == "REFUTED" and g.severity == "nitpick"


def test_cut_json_says_so_inside_its_fence():
    from reviewsys.prompts import _json_block

    text = _json_block(["x" * 100], limit=20)
    assert (
        text.startswith("```json\n")
        and text.endswith("\n```")
        and "more characters not shown" in text
    )
    assert _json_block([1]) == "```json\n[\n 1\n]\n```"


@pytest.mark.parametrize(
    "status, cited",
    [
        ("INTENTIONALLY_DEFERRED", ""),
        ("INTENTIONALLY_DEFERRED", "https://github.com/r/x/issues/12"),
        ("STILL_VALID", ""),
    ],
)
def test_a_restated_deferral_keeps_the_roots_line_unless_it_names_tracking(gh, status, cited):
    """An issue the ledger already holds deferred, restated without a tracking link (the
    author asking "still deferred?"), keeps the status line and link its root shows; one that
    names its tracking refreshes the line, and one the lane re-opens loses it."""
    from reviewsys import publish

    root = "<!-- thepastaclaw-review v1 finding=abc -->\n" + publish.with_thread_status(
        "**T**\n\nbody",
        status="INTENTIONALLY_DEFERRED",
        head_sha="c" * 40,
        tracking_url="https://github.com/r/x/issues/9",
    )
    thread = {
        "comment_id": 800,
        "thread_id": "T",
        "latest_reply_id": 901,
        "awaiting_answer": True,
        "body": root,
    }
    out = {"abc": converse.ThreadOutcome("abc", status, "Yes, still deferred.", tracking_url=cited)}
    (item,) = publish.answer_thread_lane(
        gh,
        "r/x",
        1,
        "h" * 40,
        outcomes=out,
        threads={"abc": thread},
        same_sha=True,
        settled={"abc"},
    )
    assert item["action"] == "replied"
    if status == "STILL_VALID":  # the lane re-opened it: the root no longer says deferred
        ((_, edited),) = gh.comment_edits
        assert "Deferred" not in edited
    elif cited:
        ((_, edited),) = gh.comment_edits
        assert "tracked in https://github.com/r/x/issues/12" in edited
    else:
        assert gh.comment_edits == [] and "marked" not in item
