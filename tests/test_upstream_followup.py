"""dashpay/dash#8158: a backport batch drew twelve review rounds. A maintainer's "that upstream
fix comes in its own batch" was first held for a tracking issue and then argued as a dispute,
the thread never heard the review's answers, and a pass told "this is the first review round"
raised the same settled findings again."""

from __future__ import annotations

import dataclasses
import json

import pytest

from reviewsys import converse, late, prompts, stepback, worker
from reviewsys.contract import Finding
from reviewsys.db import kv_get
from tests.test_settle_disputes import (
    ENGAGED,
    C,
    D,
    _bot,
    _deferral_round_one,
    _lanes,
    _review,
    _thread,
)
from tests.test_worker_e2e import (  # fake_git: autouse, fakes the mirror and worktree
    HEAD,
    KNST,
    _blocking,
    _gql_comment,
    _no_filing,
    _seed_prior,
    fake_git,  # noqa: F401
)

FOLLOW_UP = "Missing follow-up: bitcoin#28051 keeps the shutdown latch alive"
UPSTREAM = "https://github.com/bitcoin/bitcoin/issues/28051"
LATER = {**KNST, "body": "Shipped in v26 as is; #28051 will be backported in its turn."}
MEMBER = {"is_bot": False, "association": "MEMBER", "author": "PastaPastaPasta"}
UPSTREAMS = ("bitcoin/bitcoin", "dashpay/dash")


@pytest.fixture
def cfg(cfg):
    """The reviewed repository backports from Bitcoin Core."""
    repos = tuple(
        dataclasses.replace(r, upstream_repos=("bitcoin/bitcoin",))
        if r.repo == "dashpay/platform"
        else r
        for r in cfg.repos
    )
    return dataclasses.replace(cfg, repos=repos)


def _defer_upstream(lanes, h, finding, **row):
    """The verifier defers `h` to the upstream change, as the prompt asks."""
    _lanes(lanes, h, finding, "INTENTIONALLY_DEFERRED", reason="Deferred to its own backport.")
    row = {"tracking": "bitcoin/bitcoin#28051", "upstream_followup": True, **row}
    for lane in (lanes.reviewer["default"], lanes.verifier["default"]):
        lane["prior_finding_reconciliation"][0].update(row)


# ---- the upstream change is the tracking ----


def _followup(cited, said, **kw):
    kw = {"repo": "dashpay/dash", "number": 8158, "upstreams": UPSTREAMS, **kw}
    return converse.upstream_followup(cited, {"body": said}, **kw)


def test_the_upstream_change_must_be_in_a_repository_backported_from_and_on_the_thread():
    said = "Missing follow-up: bitcoin#28051\nwill be backported in its turn"
    assert _followup("bitcoin/bitcoin#28051", said) == UPSTREAM
    url = "https://github.com/bitcoin/bitcoin/pull/28051"
    assert _followup(url, said) == url
    # a release-branch backport of the repository's own change
    assert _followup("dashpay/dash#7050", "needs #7050 too").endswith("dashpay/dash/issues/7050")
    for cited, text, kw in (
        ("tokio-rs/tokio#28051", said, {}),  # not a repository this one backports from
        ("bitcoin/bitcoin#28051", said, {"upstreams": ()}),  # a repository that backports nothing
        ("dashpay/dash#8158", "see #8158", {}),  # this pull request itself
        ("bitcoin/bitcoin#28051", "see #280511", {}),  # another number
        ("bitcoin/bitcoin#28051", "comes later", {}),  # nobody named it
        ("#28051", said, {}),  # no repository: not an upstream reference
        (None, said, {}),
    ):
        assert _followup(cited, text, **kw) == "", cited


def test_the_conversation_lane_cites_the_upstream_change():
    row = {
        "finding_hash": "aaa",
        "status": "INTENTIONALLY_DEFERRED",
        "reply": "ok",
        "tracking": {"upstream": " bitcoin/bitcoin#28051 "},
    }
    out = converse.parse({"threads": [row]}, expected={"aaa"})
    assert out.outcomes["aaa"].upstream == "bitcoin/bitcoin#28051"
    assert out.outcomes["aaa"].tracking_url == ""


def _vet(replies, *, severity="blocking"):
    outcome = converse.ThreadOutcome(
        "aaa", "INTENTIONALLY_DEFERRED", "Understood.", upstream="bitcoin/bitcoin#28051"
    )
    thread = {"severity": severity, "body": f"**{FOLLOW_UP}**", "transcript": replies}
    out = converse.ConversationOutput(outcomes={"aaa": outcome})
    vetted = converse.accept_deferrals(
        out, {"aaa": thread}, repo="dashpay/dash", number=8158, author="x", upstreams=UPSTREAMS
    )
    return vetted.outcomes["aaa"]


def test_a_maintainer_defers_a_missing_follow_up_without_a_tracking_issue():
    got = _vet([{**MEMBER, "body": "will be backported in its turn"}])
    assert got.status == "INTENTIONALLY_DEFERRED" and got.tracking_url == UPSTREAM
    assert not got.auto_file


def test_only_a_maintainer_defers_a_follow_up_without_a_tracking_issue():
    author = {"is_bot": False, "association": "CONTRIBUTOR", "author": "x", "body": "later"}
    assert _vet([author]).status == converse.DEFERRAL_PENDING
    # a suggestion is the author's to defer, but the usual way: with tracking
    assert _vet([author], severity="suggestion").status == converse.DEFERRAL_NEEDS_TRACKING


def test_the_conversation_prompt_asks_for_the_upstream_change_not_an_issue():
    text = converse.prompt(
        repo="dashpay/dash",
        number=8158,
        head_sha=HEAD,
        meta={},
        project_skill="",
        review_skill="",
        threads={},
        fetched_commits=[],
        unfetched_commits=[],
        standing_verdict=None,
        evidence={},
    )
    assert '`"tracking": {"upstream": "<owner/repo#N of the upstream change>"}`' in text
    assert "the upstream change itself is not such a link" not in text


def test_a_full_review_defers_a_follow_up_to_its_upstream_change(cfg, conn, gh, lanes):
    blocker = _blocking(FOLLOW_UP)
    h = _seed_prior(conn, blocker)
    _thread(gh, h, blocker["title"], [LATER])
    _defer_upstream(lanes, h, blocker)
    _review(cfg, conn, gh, lanes, HEAD)
    review = gh.posted_reviews[-1]
    assert review["event"] != "REQUEST_CHANGES"
    assert "Awaiting tracking link" not in review["body"]
    (answer,) = gh.replies
    assert "**Deferred**" in answer["body"] and "tracking issue" not in answer["body"]
    assert kv_get(conn, f"defer.asked:dashpay/platform#1:{h}") is None
    row = conn.execute("SELECT body FROM findings WHERE stage='conceded' AND hash=?", (h,))
    assert row.fetchone()["body"] == f"INTENTIONALLY_DEFERRED\n{UPSTREAM}"
    assert not gh.created_issues and not any(c[:1] == ["issue"] for c in gh.calls)

    # the next head is shown what the deferral rests on
    gh.threads[0]["comments"]["nodes"].append(_gql_comment(_bot(answer["body"])))
    _defer_upstream(lanes, h, blocker)
    _review(cfg, conn, gh, lanes, C)
    prompt = [c.prompt for c in lanes.calls if c.role == "verifier"][-1]
    assert f'"tracking": "{UPSTREAM}"' in prompt


def test_an_upstream_deferral_still_needs_a_maintainer_and_the_flag(cfg, conn, gh, lanes):
    cfg = _no_filing(cfg)
    blocker = _blocking(FOLLOW_UP)
    h = _seed_prior(conn, blocker)
    _thread(gh, h, blocker["title"], [{**LATER, "author": "x", "association": "CONTRIBUTOR"}])
    _defer_upstream(lanes, h, blocker)
    _review(cfg, conn, gh, lanes, HEAD)
    assert "**Needs a tracking issue**" in gh.replies[-1]["body"]

    gh.threads[0]["comments"]["nodes"][1] = _gql_comment(LATER)
    _defer_upstream(lanes, h, blocker, upstream_followup=False)
    _review(cfg, conn, gh, lanes, C)
    assert gh.posted_reviews[-1]["event"] == "REQUEST_CHANGES"

    # nor does a change in a repository this one does not backport from
    _defer_upstream(lanes, h, blocker, tracking="tokio-rs/tokio#28051")
    _review(cfg, conn, gh, lanes, D)
    assert gh.posted_reviews[-1]["event"] == "REQUEST_CHANGES"


def test_a_maintainer_deferral_ends_a_standing_dispute(cfg, conn, gh, lanes):
    """A blocker left DISPUTED withholds the approval; the maintainer's deferral lifts it."""
    blocker = _blocking(FOLLOW_UP)
    h = _seed_prior(conn, blocker)
    _thread(gh, h, blocker["title"], [{**KNST, "body": "Not going to fix here."}])
    _lanes(lanes, h, blocker, "STILL_VALID", reason=ENGAGED, disputed=True)
    _review(cfg, conn, gh, lanes, HEAD)
    gh.threads[0]["comments"]["nodes"].append(_gql_comment(_bot(gh.replies[0]["body"])))
    _lanes(lanes, h, blocker, "STILL_VALID", reason="Still applies.")
    _review(cfg, conn, gh, lanes, C)
    assert "### Disputed — left to maintainers (1)" in gh.posted_reviews[-1]["body"]

    later = {**LATER, "id": 902, "created_at": "z"}
    gh.threads[0]["comments"]["nodes"] += [
        _gql_comment(_bot(gh.replies[-1]["body"], id=951, at="y2")),
        _gql_comment(later),
    ]
    _defer_upstream(lanes, h, blocker)
    _review(cfg, conn, gh, lanes, D)
    assert "Disputed — left to maintainers" not in gh.posted_reviews[-1]["body"]
    row = conn.execute(
        "SELECT body FROM findings WHERE stage='conceded' AND hash=? ORDER BY id DESC", (h,)
    ).fetchone()
    assert row["body"] == f"INTENTIONALLY_DEFERRED\n{UPSTREAM}"


# ---- what the thread hears ----


def test_a_reply_first_read_as_a_deferral_is_answered_once_as_a_dispute(cfg, conn, gh, lanes):
    """The ask for a tracking issue stood for six rounds while the review, reading the same
    reply as a dispute, wrote answers nobody saw."""
    cfg = _no_filing(cfg)
    blocker = _blocking()
    h = _deferral_round_one(
        cfg, conn, gh, lanes, blocker, label="🔴 Blocking", reply={**KNST, "body": "later"}
    )
    _lanes(lanes, h, blocker, "STILL_VALID", reason=ENGAGED, disputed=True)
    _review(cfg, conn, gh, lanes, C)
    answer = gh.replies[-1]["body"]
    assert f"**Still applies** (re-reviewed at `cccccccc`): {ENGAGED}" in answer
    assert "Awaiting tracking link" not in gh.posted_reviews[-1]["body"]
    assert json.loads(kv_get(conn, f"dispute.answered:dashpay/platform#1:{h}"))["sha"] == C

    gh.threads[0]["comments"]["nodes"].append(_gql_comment(_bot(answer, id=951, at="y2")))
    _lanes(lanes, h, blocker, "STILL_VALID", reason="Still applies.")
    _review(cfg, conn, gh, lanes, D)
    assert gh.posted_reviews[-1]["event"] == "COMMENT"
    assert converse.DISPUTE_REPLY in gh.replies[-1]["body"]


def test_a_finding_whose_code_changed_says_what_still_stands(cfg, conn, gh, lanes, monkeypatch):
    """An author applied the suggested guard; the next reviews said only "still open"."""
    blocker = _blocking()
    h = _seed_prior(conn, blocker)
    _thread(gh, h, blocker["title"], [])
    changed = [True]
    monkeypatch.setattr(worker, "_lines_changed", lambda *a: changed[0])
    remains = "The new guard covers the tip, but `GetSummary()` still runs before it."
    _lanes(lanes, h, blocker, "STILL_VALID", reason=remains)
    _review(cfg, conn, gh, lanes, HEAD)
    (answer,) = gh.replies
    assert f"**Still applies** (re-reviewed at `aaaaaaaa`): {remains}" in answer["body"]

    # a head that leaves the code alone repeats nothing
    gh.threads[0]["comments"]["nodes"].append(_gql_comment(_bot(answer["body"])))
    changed[0] = False
    _lanes(lanes, h, blocker, "STILL_VALID", reason="Still applies.")
    _review(cfg, conn, gh, lanes, C)
    assert len(gh.replies) == 1


def test_saying_what_still_stands_is_no_answer_to_a_dispute(cfg, conn, gh, lanes, monkeypatch):
    """A reply the code change prompted must not count as the one answer a dispute gets: the
    blocker would be left to the maintainers on the next head with nobody having disputed."""
    blocker = _blocking()
    h = _seed_prior(conn, blocker)
    _thread(gh, h, blocker["title"], [{**KNST, "body": "I'll think about a guard."}])
    _lanes(lanes, h, blocker, "STILL_VALID", reason="Thanks.")
    _review(cfg, conn, gh, lanes, HEAD)
    gh.threads[0]["comments"]["nodes"].append(_gql_comment(_bot(gh.replies[0]["body"])))
    monkeypatch.setattr(worker, "_lines_changed", lambda *a: True)
    _lanes(lanes, h, blocker, "STILL_VALID", reason="The guard misses the tip.", disputed=True)
    _review(cfg, conn, gh, lanes, C)
    assert "The guard misses the tip." in gh.replies[-1]["body"]
    assert kv_get(conn, f"dispute.answered:dashpay/platform#1:{h}") is None


# ---- the fresh pass, the confirmation lane and the step back ----


def _finding(**kw):
    return Finding.from_dict({**_blocking(), **kw})


def test_a_carried_finding_is_not_raised_to_blocking_on_unchanged_code():
    posted = {"aaa": "suggestion", "bbb": "suggestion", "ccc": "blocking"}
    raised = _finding(finding_hash="aaa")
    raised.extra["code_status"] = late.IN_REVIEW
    moved = _finding(finding_hash="bbb")
    moved.extra["code_status"] = late.CHANGED
    same = _finding(finding_hash="ccc")
    same.extra["code_status"] = late.SINCE_FIRST
    new = _finding()
    new.extra["code_status"] = late.SINCE_FIRST
    assert late.hold_severity([raised, moved, same, new], posted) == [raised]
    assert raised.severity == "suggestion"
    assert [f.severity for f in (moved, same, new)] == ["blocking"] * 3


PRIOR = [
    {
        "finding_hash": "0123456789ab",
        "original_title": "Open finding",
        "file": "src/init.cpp",
        "severity": "blocking",
    },
    {
        "finding_hash": "ba9876543210",
        "original_title": FOLLOW_UP,
        "file": "src/shutdown.cpp",
        "severity": "blocking",
        "prior_status": "INTENTIONALLY_DEFERRED",
        "tracking": UPSTREAM,
    },
]


def test_the_fresh_pass_is_told_it_is_not_the_first_and_what_is_settled(cfg):
    kw = {
        "repo": "dashpay/platform",
        "number": 7,
        "head_sha": HEAD,
        "meta": {"title": "t", "body": "b"},
        "coverage_from": "b" * 40,
        "evidence": {},
        "phase": "final",
        "role": "general",
    }
    fresh = prompts.reviewer_prompt(cfg, **kw, prior=PRIOR, prior_sha=C, fresh=True)
    assert "first PastaClaw automated review round" not in fresh
    assert f"This pull request was reviewed before, last at `{C}`" in fresh
    assert "### Settled on this pull request" in fresh
    assert FOLLOW_UP in fresh and UPSTREAM in fresh and "Open finding" not in fresh
    # the second pass of a first review really is a first round
    first = prompts.reviewer_prompt(cfg, **kw, prior=[], prior_sha=None, fresh=True)
    assert "first PastaClaw automated review round" in first
    assert "Settled on this pull request" not in first


def test_the_confirmation_lane_judges_by_the_projects_rules(cfg):
    kw = {
        "repo": "dashpay/platform",
        "number": 1,
        "head_sha": HEAD,
        "meta": {"title": "t", "body": "b"},
        "coverage_from": "b" * 40,
        "reported": [{"id": "c1", "title": "x"}],
    }
    _, review_skill = prompts.skill_texts(cfg, "dashpay/platform")
    p = prompts.confirmation_prompt(cfg, **kw, settled=prompts.settled_matters(PRIOR))
    assert review_skill.strip() and review_skill.strip() in p
    assert "## Settled on this pull request" in p and FOLLOW_UP in p
    assert "Open finding" not in p
    assert "Settled on this pull request" not in prompts.confirmation_prompt(cfg, **kw)


def test_the_step_back_reads_where_each_finding_stands():
    def found(h):
        return {"hash": h, "severity": "blocking", "file": "f", "title": h, "body": ""}

    earlier = [
        stepback.Round("a", (found("wrong"), found("stands"))),
        stepback.Round("b", (found("later"), found("argued"), found("fixed"))),
    ]
    earlier = stepback.without(earlier, {"wrong"})
    assert [r.blockers for r in earlier] == [1, 3]
    settled = {"later": "INTENTIONALLY_DEFERRED", "argued": "DISPUTED"}
    items = stepback.raised(earlier, [found("now")], {"stands"}, settled)
    assert [(r.title, r.round, r.state) for r in items] == [
        ("stands", 1, "open"),
        ("later", 2, "deferred"),
        ("argued", 2, "disputed"),
        ("fixed", 2, "closed"),
        ("now", 3, "new"),
    ]


def test_a_repository_names_what_it_backports_from(tmp_path, skills_dir):
    from reviewsys import config as cfg_mod

    path = skills_dir / "config.json"
    raw = json.loads(path.read_text())
    raw["repos"][0]["upstream_repos"] = ["bitcoin/bitcoin", "dashpay/dash"]
    path.write_text(json.dumps(raw))
    toml = tmp_path / "config.toml"
    toml.write_text(
        cfg_mod.DEFAULT_TOML.replace('skills = "~/Projects/skills"', f'skills = "{skills_dir}"')
    )
    repos = cfg_mod.load(toml).repos
    assert repos[0].upstream_repos == ("bitcoin/bitcoin", "dashpay/dash")
    assert repos[1].upstream_repos == ()
    raw["repos"][0]["upstream_repos"] = ["bitcoin"]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="upstream_repos entry 'bitcoin': need owner/name"):
        cfg_mod.load(toml)
