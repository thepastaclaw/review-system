"""Later rounds on a pull request that is rebased often (dashpay/dash#8158, a batch of Bitcoin
Core backports): the same two lines of `src/init.cpp`, byte-identical to upstream, drew three
different blockers over a week. After a rebase the round read the base branch's changes as the
pull request's; the confirmation lane judged a finding on lines eleven full reviews had read
without knowing it; a missing upstream follow-up turned up at round twelve; and a blocker the
lane did not confirm came back reworded on the same lines and was confirmed."""

from __future__ import annotations

import pytest

from reviewsys import late, prompts, worker
from reviewsys.contract import Finding
from tests.test_late import (  # noqa: F401
    STATUS,
    _f,
    _first_round,
    _lines,
    _round,
    _stages,
    _with,
    statuses,
)
from tests.test_ledger import _commit, _git
from tests.test_settle_disputes import _thread
from tests.test_worker_e2e import (  # fake_git: autouse, fakes the mirror and worktree
    KNST,
    _blocking,
    _gql_comment,
    _run,
    _verifier,
    fake_git,  # noqa: F401
)

C, D, E = "c" * 40, "d" * 40, "e" * 40


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    return r


INIT = "src/init.cpp"


def _base(repo, n=40):
    """The base branch `develop` (mirrored as `origin/develop`) with a file of `n` lines."""
    (repo / "src").mkdir(exist_ok=True)
    sha = _commit(repo, {"base.cpp": "\n".join(_lines(n, "b")) + "\n"})
    _git(repo, "branch", "-M", "develop")
    _git(repo, "update-ref", "refs/remotes/origin/develop", sha)
    return sha


def _pr(repo, base, commits):
    """The pull request's commits on top of `base`: each a dict of files."""
    _git(repo, "checkout", "-q", "-B", "pr", base)
    (repo / "src").mkdir(exist_ok=True)
    return [_commit(repo, files) for files in commits]


def _advance(repo, files):
    """A newer base branch tip."""
    _git(repo, "checkout", "-q", "develop")
    sha = _commit(repo, files)
    _git(repo, "update-ref", "refs/remotes/origin/develop", sha)
    return sha


# ---- 1: the round's scope after a rebase ----


def test_a_rebase_is_detected_with_both_merge_bases(repo):
    base = _base(repo)
    pr = _pr(repo, base, [{"a.cpp": "one\n"}, {"b.cpp": "two\n"}])
    since = pr[-1]
    # a linear push: today's round
    head = _commit(repo, {"c.cpp": "three\n"})
    assert late.rewrite_since(repo, since, head, "develop", base) is None
    # a rebase onto a newer base: the PR's own commits, before and after
    newer = _advance(repo, {"base.cpp": "rewritten by the base branch\n"})
    rebased = _pr(repo, newer, [{"a.cpp": "one\n"}, {"b.cpp": "two\n"}])[-1]
    assert late.rewrite_since(repo, since, rebased, "develop", None) == late.Rewrite(base, newer)
    # history rewritten on the same base (a squash, an amended commit)
    squashed = _pr(repo, base, [{"a.cpp": "one\n", "b.cpp": "two!\n"}])[-1]
    assert late.rewrite_since(repo, since, squashed, "develop", base) == late.Rewrite(base, base)
    # the base branch ref is missing: rebased, but the merge bases cannot be computed
    assert late.rewrite_since(repo, since, rebased, "gone", None) == late.Rewrite(None, None)
    # the old head is not in the repository: git cannot tell, today's round
    assert late.rewrite_since(repo, "0" * 40, rebased, "develop", newer) is None


AFTER_REBASE = (
    f"- This is a later review round: the pull request was reviewed in full at `{C}`. Do not "
    f"review it again from the start. The branch was rebased, or its history rewritten, since "
    f"then: `{C}` is not an ancestor of `{D}`. Compare the pull request's own commits before "
    f"and after with `git range-diff {'1' * 40}..{C} {'2' * 40}..{D}`; a commit it shows as "
    f"unchanged (`=`) was already reviewed. `git diff {'2' * 40}..{D}` is the pull request as "
    f"it stands. `git diff {C}..{D}` would mostly show the base branch's changes, which are "
    "not this pull request's and are not under review. Review what the pull request's own "
    "commits do differently since then and check the prior findings against it. The complete "
    "range is context for that, not the subject.\n"
)


def test_after_a_rebase_every_lane_is_given_the_pull_requests_own_change(cfg):
    rewrite = late.Rewrite("1" * 40, "2" * 40)
    assert prompts.incremental_block(C, D, rewrite).startswith(AFTER_REBASE)
    verifier = prompts.verifier_incremental_block(C, D, rewrite)
    assert "this round reviews what the pull request's own commits do differently" in verifier
    assert f"git range-diff {'1' * 40}..{C} {'2' * 40}..{D}" in verifier
    assert f"`git diff {C}..{D}` would mostly show the base branch's changes" in verifier
    confirm = prompts.confirmation_prompt(
        cfg,
        repo="dashpay/dash",
        number=8158,
        head_sha=D,
        meta={"title": "t", "body": "b"},
        coverage_from="2" * 40,
        reported=[],
        since_sha=C,
        rewrite=rewrite,
    )
    assert f"git range-diff {'1' * 40}..{C} {'2' * 40}..{D}" in confirm
    assert f"Its change since then is `git diff {C}..{D}`" not in confirm
    # a linear push keeps today's wording
    linear = prompts.incremental_block(C, D)
    assert f"Review what changed since then (`git diff {C}..{D}`; when the branch" in linear
    assert "is not an ancestor" not in linear
    # merge bases unknown: today's commands, and a warning
    unknown = prompts.incremental_block(C, D, late.Rewrite(None, None))
    assert f"`git diff {C}..{D}`" in unknown and "could not be computed" in unknown
    assert "would mostly show" not in unknown


@pytest.mark.usefixtures("statuses")
def test_the_8158_rebase_reads_the_pull_requests_commits_not_the_base_branch(
    cfg,
    conn,
    gh,
    lanes,
    monkeypatch,
    repo,
):
    """One rebase of #8158 showed 374 files in `git diff since..head` for an 11-commit pull
    request. The lanes are given the range-diff of its own commits instead."""
    base = _base(repo)
    since = _pr(repo, base, [{INIT: "\n".join(_lines(40, "init")) + "\n"}, {"b.cpp": "x\n"}])[-1]
    newer = _advance(repo, {f"base{i}.cpp": f"base change {i}\n" for i in range(12)})
    head = _pr(repo, newer, [{INIT: "\n".join(_lines(40, "init")) + "\n"}, {"b.cpp": "x\n"}])[-1]
    whole = _git(repo, "diff", "--name-only", since, head).split()
    assert len(whole) == 12 and INIT not in whole  # all the base branch's
    rewrite = late.rewrite_since(repo, since, head, "develop", newer)
    assert rewrite == late.Rewrite(base, newer)
    commits = late.Commits(repo, head, since, newer, rewrite)
    assert len(commits.same() or ()) == 2  # both commits unchanged by the rebase

    monkeypatch.setattr(late, "rewrite_since", lambda *a: rewrite)
    _first_round(cfg, conn, gh, lanes)
    _round(cfg, conn, gh, lanes, C, [])
    later = [c.prompt for c in lanes.calls if c.prompt and "later review round" in c.prompt]
    assert {c.role for c in lanes.calls if c.prompt in later} >= {"general", "verifier"}
    assert all(f"git range-diff {base}..{'a' * 40} {newer}..{C}" in p for p in later)


def test_a_rebase_leaves_the_pull_requests_untouched_lines_unchanged(repo):
    """Hunks are read on the head side: a base-branch edit far from a line the author did not
    touch leaves it unchanged; one within the margin marks it changed."""
    (repo / "src").mkdir()
    base = _commit(repo, {INIT: "\n".join(_lines(80, "init")) + "\n"})
    _git(repo, "branch", "-M", "develop")
    pr_lines = _lines(80, "init")
    pr_lines[40] = "the backported call, byte-identical to upstream"
    since = _pr(repo, base, [{INIT: "\n".join(pr_lines) + "\n"}])[-1]
    f = _f(INIT, 41, 41)

    def rebase_onto(changed_line):
        _git(repo, "checkout", "-q", "develop")
        lines = _lines(80, "init")
        lines[changed_line] = "edited on the base branch"
        newer = _commit(repo, {INIT: "\n".join(lines) + "\n"})
        rebased = list(lines)
        rebased[40] = pr_lines[40]
        return _pr(repo, newer, [{INIT: "\n".join(rebased) + "\n"}])[-1]

    far = rebase_onto(5)
    assert late.CodeHistory(repo, far, since).status(f) == late.SINCE_FIRST
    assert worker._unchanged_since(_later(repo, far, since), since, f)
    near = rebase_onto(42)
    assert late.CodeHistory(repo, near, since).status(f) == late.CHANGED
    assert not worker._unchanged_since(_later(repo, near, since), since, f)


def _later(worktree, head, prev):
    from types import SimpleNamespace

    return SimpleNamespace(
        cfg=SimpleNamespace(policy=SimpleNamespace(pipeline=None)),
        is_audit=False,
        sha=head,
        prev_final_sha=prev,
        first_reviewed_sha=None,
        worktree=worktree,
        code_history=None,
    )


# ---- 2: the confirmation lane knows the round and the code status ----

NULL_TIP = {
    **_blocking("A null tip is dereferenced when block data is checked"),
    "file": INIT,
    "line_start": 2934,
    "line_end": 2936,
    "category": "bug",
}


@pytest.mark.usefixtures("statuses")
def test_the_confirmation_lane_is_told_the_code_status_and_the_already_reviewed_rule(
    cfg,
    conn,
    gh,
    lanes,
    monkeypatch,
):
    """#8158 head 5823797d: a blocker on lines unchanged since the first review, confirmed in
    isolation. The lane now sees the status, how many full reviews read the lines, and the
    rule."""
    monkeypatch.setitem(STATUS, NULL_TIP["title"], late.SINCE_FIRST)
    _first_round(cfg, conn, gh, lanes)
    lanes.confirm = {NULL_TIP["title"]: "non_blocking"}
    _round(cfg, conn, gh, lanes, C, [NULL_TIP])
    (confirm,) = [c.prompt for c in lanes.calls if c.role == "confirm"]
    assert '"code_status": "unchanged since the first review"' in confirm
    assert '"full_reviews_of_these_lines": 1' in confirm
    assert f"last reviewed in full at `{'a' * 40}`" in confirm
    assert "(a) it names the change in this pull request since that review" in confirm
    assert "loses funds, breaks consensus or exposes key material" in confirm
    assert '"already reviewed, nothing in the change affects it"' in confirm
    assert gh.posted_reviews[-1]["event"] == "APPROVE"


def test_the_brief_says_the_status_in_words():
    f = _with(_f(INIT, 3, 4), late.IN_REVIEW)
    f.extra["full_reviews"] = 4
    (brief,) = late.candidate_briefs([f])
    assert brief["code_status"] == "unchanged since the last reviewed head"
    assert brief["full_reviews_of_these_lines"] == 4
    assert "code_status" not in late.candidate_briefs([_f(INIT, 3)])[0]


# ---- 3: a missing upstream follow-up is enumerated once ----

FOLLOWUP = {
    **_blocking("Missing follow-up: bitcoin#29770 checks for undo data before starting indexes"),
    "file": INIT,
    "line_start": 2929,
    "line_end": 2935,
    "category": "bug",
}


@pytest.mark.usefixtures("statuses")
def test_a_late_missing_follow_up_in_a_reviewed_commit_is_listed_once_unasked(
    cfg,
    conn,
    gh,
    lanes,
    monkeypatch,
):
    """#8158 run 12925: "Missing follow-up: bitcoin#29770" at round twelve, on lines unchanged
    since the first review, in a commit the rebase left as it was; confirmed by the lane while
    the maintainer had said twice that follow-ups get backported in their turn."""
    monkeypatch.setitem(STATUS, FOLLOWUP["title"], late.SINCE_FIRST)
    monkeypatch.setattr(late.Commits, "reviewed", lambda self, f: True)
    _first_round(cfg, conn, gh, lanes)
    rid = _round(cfg, conn, gh, lanes, C, [FOLLOWUP])
    assert not [c for c in lanes.calls if c.role == "confirm"]
    review = gh.posted_reviews[-1]
    assert review["event"] == "APPROVE" and not review["comments"]
    followups = review["body"].split("Follow-up candidates (optional) (1)", 1)[1]
    assert "bitcoin#29770" in followups
    assert "missing upstream follow-up(s) of commits already reviewed listed" in review["body"]
    assert _stages(conn, rid)[("late-followup", FOLLOWUP["title"])][0] == "suggestion"

    # raised again under another wording at the next push: not listed again
    again = {**FOLLOWUP, "title": "Missing follow-up: bitcoin#29770 (undo data)"}
    monkeypatch.setitem(STATUS, again["title"], late.SINCE_FIRST)
    monkeypatch.setattr(late, "listed_spots", lambda repo, head, rows: _same_places(rows))
    _round(cfg, conn, gh, lanes, D, [again])
    assert "Follow-up candidates" not in gh.posted_reviews[-1]["body"]
    assert not [c for c in lanes.calls if c.role == "confirm"]


def _same_places(rows):
    return [(f, s, e or s) for f, s, e, _ in rows if f and s]


@pytest.mark.usefixtures("statuses")
def test_a_missing_follow_up_in_a_new_or_changed_commit_still_blocks_until_confirmed(
    cfg,
    conn,
    gh,
    lanes,
    monkeypatch,
):
    monkeypatch.setitem(STATUS, FOLLOWUP["title"], late.SINCE_FIRST)
    monkeypatch.setattr(late.Commits, "reviewed", lambda self, f: False)
    _first_round(cfg, conn, gh, lanes)
    _round(cfg, conn, gh, lanes, C, [FOLLOWUP])
    assert [c for c in lanes.calls if c.role == "confirm"]
    assert gh.posted_reviews[-1]["event"] == "REQUEST_CHANGES"


def test_a_missing_follow_up_in_the_first_full_review_blocks():
    """First round: no gate. In a later round, on changed code: the confirmation lane."""
    f = _with(Finding.from_dict(FOLLOWUP), late.CHANGED)
    split = late.split([f], posted=lambda f: None, listed=set(), reviewed=lambda f: True)
    assert split.candidates == [f] and f.severity == "blocking"


def test_what_marks_a_missing_follow_up():
    assert late.is_missing_followup(Finding.from_dict(FOLLOWUP))
    marked = Finding.from_dict({**_blocking("Undo data is not checked"), "upstream_followup": True})
    assert late.is_missing_followup(marked)
    assert not late.is_missing_followup(Finding.from_dict(_blocking("A follow-up is missing")))


def test_which_commits_the_last_full_review_read(repo):
    base = _base(repo)
    lines = _lines(20, "init")
    first = _pr(repo, base, [{INIT: "\n".join(lines) + "\n"}])
    since = first[-1]
    head = _commit(repo, {INIT: "\n".join([*lines, "added after the review"]) + "\n"})
    commits = late.Commits(repo, head, since, base)
    assert commits.reviewed(_f(INIT, 3, 5))  # an ancestor of the reviewed head
    assert not commits.reviewed(_f(INIT, 21))  # the commit after it
    assert commits.reviewed(_f("base.cpp", 2))  # the base branch's lines
    # rebased: an unchanged commit is reviewed, an amended one is not
    newer = _advance(repo, {"base.cpp": "\n".join(_lines(40, "b")) + "\nbase tail\n"})
    amended = list(lines)
    amended[10] = "amended in the rebase"
    rebased = _pr(repo, newer, [{INIT: "\n".join(lines) + "\n"}, {"x.cpp": "x\n"}])[-1]
    rewrite = late.rewrite_since(repo, since, rebased, "develop", newer)
    assert late.Commits(repo, rebased, since, newer, rewrite).reviewed(_f(INIT, 3, 5))
    changed = _pr(repo, newer, [{INIT: "\n".join(amended) + "\n"}])[-1]
    rewrite = late.rewrite_since(repo, since, changed, "develop", newer)
    assert not late.Commits(repo, changed, since, newer, rewrite).reviewed(_f(INIT, 3, 5))
    # no merge bases after a rewrite: git cannot tell
    assert not late.Commits(repo, changed, since, newer, late.Rewrite(None, None)).reviewed(
        _f(INIT, 3, 5)
    )


# ---- 4: a remembered verdict is matched by place ----

STOPATHEIGHT = {
    **_blocking("Preserve -stopatheight during background snapshot replay"),
    "file": "src/validation.cpp",
    "line_start": 3946,
    "line_end": 3958,
    "category": "bug",
}
REWORDED = {
    **STOPATHEIGHT,
    "title": "Background replay ignores -stopatheight",
    "line_start": 3944,
}


@pytest.fixture
def places(monkeypatch, request):
    request.getfixturevalue("statuses")
    for f in (STOPATHEIGHT, REWORDED):
        monkeypatch.setitem(STATUS, f["title"], late.IN_REVIEW)
    monkeypatch.setattr(late, "listed_spots", lambda repo, head, rows: _same_places(rows))


def test_a_reworded_blocker_on_the_same_lines_keeps_its_earlier_verdict(
    cfg, conn, gh, lanes, places
):
    """#8158: 635d64ec1b53, not confirmed in run 12925, came back as 64cc089d0e88 on the same
    `validation.cpp:3944-3958` lines and was confirmed in run 12981."""
    _first_round(cfg, conn, gh, lanes)
    lanes.confirm = {STOPATHEIGHT["title"]: "non_blocking"}
    _round(cfg, conn, gh, lanes, C, [STOPATHEIGHT])
    lanes.confirm = {}
    _round(cfg, conn, gh, lanes, D, [REWORDED])
    assert len([c for c in lanes.calls if c.role == "confirm"]) == 1
    review = gh.posted_reviews[-1]
    assert review["event"] == "APPROVE"
    assert "(1 as an earlier confirmation decided" in review["body"]


def test_a_reply_since_the_verdict_has_the_reworded_blocker_asked_again(
    cfg, conn, gh, lanes, places, monkeypatch
):
    """A carried suggestion raised as a blocker, not confirmed; a human then replies on its
    thread. A reworded blocker on the same lines is asked about again."""
    sugg = {**STOPATHEIGHT, "severity": "suggestion", "title": "Snapshot replay stop height"}
    monkeypatch.setitem(STATUS, sugg["title"], late.CHANGED)
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["preliminary"] = _verifier([])
    lanes.verifier["final"] = _verifier([sugg])
    _run(cfg, conn, gh, lanes)
    hs = Finding.from_dict(sugg).hash
    still = [{"finding_hash": hs, "status": "STILL_VALID", "reason": "still"}]
    lanes.confirm = {sugg["title"]: "non_blocking"}
    _round(cfg, conn, gh, lanes, C, [{**sugg, "severity": "blocking", "finding_hash": hs}], still)
    assert len([c for c in lanes.calls if c.role == "confirm"]) == 1
    _thread(gh, hs, sugg["title"], [])
    gh.threads[0]["isResolved"] = True
    gh.threads[0]["comments"]["nodes"].append(_gql_comment({**KNST, "body": "Repro: ..."}))
    lanes.confirm = {}
    _round(cfg, conn, gh, lanes, D, [{**sugg, "finding_hash": hs}, REWORDED], still)
    assert len([c for c in lanes.calls if c.role == "confirm"]) == 2


def test_a_blocker_elsewhere_in_the_file_is_asked(cfg, conn, gh, lanes, places, monkeypatch):
    elsewhere = {**REWORDED, "title": "Another defect", "line_start": 100, "line_end": 104}
    monkeypatch.setitem(STATUS, elsewhere["title"], late.IN_REVIEW)
    _first_round(cfg, conn, gh, lanes)
    lanes.confirm = {STOPATHEIGHT["title"]: "non_blocking"}
    _round(cfg, conn, gh, lanes, C, [STOPATHEIGHT])
    _round(cfg, conn, gh, lanes, D, [elsewhere])
    assert len([c for c in lanes.calls if c.role == "confirm"]) == 2
    assert _verifier  # the e2e helpers above build these rounds


def test_the_step_back_lane_reads_the_pull_requests_growth_after_a_rebase(cfg):
    common = {
        "repo": "dashpay/dash",
        "number": 8158,
        "head_sha": D,
        "meta": {"title": "t", "body": "b"},
        "coverage_from": "2" * 40,
        "first_sha": C,
        "round_now": 12,
        "raised": [],
    }
    rebased = prompts.stepback_prompt(cfg, **common, first_rewrite=late.Rewrite("1" * 40, "2" * 40))
    assert f"`git range-diff {'1' * 40}..{C} {'2' * 40}..{D}`" in rebased
    assert f"git diff --stat {C}..{D}" not in rebased
    assert f"git diff --stat {C}..{D}" in prompts.stepback_prompt(cfg, **common)
