"""Policy v12 inside the worker's steps: verifier shards and their failure semantics, the
one-shot ledger reconcile, the thread lane's work set and failure fallback, the deferral guard
on a new push, and structured output."""

from __future__ import annotations

import json

import pytest

from reviewsys import ledger, prompts, worker
from reviewsys.contract import Finding
from reviewsys.db import tx
from reviewsys.models import RunStatus, Trigger
from tests.test_worker_e2e import fake_git  # noqa: F401  (autouse: fake mirror/worktree)
from tests.v12_support import (
    REPO,
    A,
    B,
    backfilled,
    cand,
    issues,
    publish_threads,
    refute_gate,
    reply,
    run_round,
    shard_setup,
    try_round,
    v12_cfg,
)

pytestmark = pytest.mark.usefixtures("wide_diff")

# ---- verifier shards ----


def test_more_than_twelve_candidates_split_and_a_failed_shard_without_blockers_is_dropped(
    cfg, skills_dir, conn, gh, lanes, git
):
    v12 = v12_cfg(cfg, skills_dir)
    shard_setup(lanes, blocking_in_failing=False)
    rid = run_round(v12, conn, gh, lanes, A)
    shards = sorted(
        {tuple(i) for k, label, i in lanes.v12_calls if k == "verify" and label == "verify2"}
    )
    assert len(shards) == 2 and {len(s) for s in shards} == {7, 6}
    roles = {
        r["role"]
        for r in conn.execute("SELECT role FROM lanes WHERE run_id=? AND phase='verify2'", (rid,))
    }
    assert roles == {"verifier-1of2", "verifier-2of2"}
    ev = conn.execute("SELECT detail FROM events WHERE kind='verify.shard_dropped'").fetchone()
    assert "7 non-blocking candidate(s) dropped" in ev["detail"]
    review = gh.posted_reviews[-1]
    # only nitpicks survived, but a dropped shard means the review cannot approve
    assert review["event"] == "COMMENT"


def test_a_failed_shard_holding_a_blocker_fails_the_run(cfg, skills_dir, conn, gh, lanes, git):
    v12 = v12_cfg(cfg, skills_dir)
    shard_setup(lanes, blocking_in_failing=True)
    rid, status = try_round(v12, conn, gh, lanes, A)
    assert status == RunStatus.FAILED
    row = conn.execute("SELECT reason, fail_kind FROM runs WHERE id=?", (rid,)).fetchone()
    assert (
        row["fail_kind"] == "infra"
        and "blocking candidates unchecked (p2-general-1)" in row["reason"]
    )
    assert not gh.posted_reviews


def test_no_candidates_no_verifier_lane(cfg, skills_dir, conn, gh, lanes, git):
    v12 = v12_cfg(cfg, skills_dir)
    lanes.candidates["default"] = []
    run_round(v12, conn, gh, lanes, A)
    assert not [c for c in lanes.v12_calls if c[0] == "verify"]
    review = gh.posted_reviews[-1]
    assert review["event"] == "APPROVE"  # nothing stands
    assert "The reviewers raised nothing to verify" in review["body"]


# ---- the one-shot reconcile ----


def test_the_first_v12_round_reconciles_the_backfilled_ledger(
    cfg, skills_dir, conn, gh, lanes, git
):
    v12 = v12_cfg(cfg, skills_dir)
    gone = backfilled(conn, "Vanished blocker", line=5)
    kept = backfilled(conn, "Unchanged blocker", line=40)
    threaded = backfilled(conn, "Threaded blocker", line=50)
    git.change("f.rs", 5, at=A)
    gh.threads = [
        {
            "id": "PRRT_1",
            "isResolved": False,
            "isOutdated": False,
            "path": "f.rs",
            "line": 50,
            "comments": {
                "nodes": [
                    {
                        "databaseId": 700,
                        "author": {"login": "thepastaclaw"},
                        "body": f"<!-- thepastaclaw-review v1 finding={threaded} dedupe=x -->\n**🔴 Blocking: Threaded blocker**\n\nold",
                        "createdAt": "2026-09-01T00:00:00Z",
                        "authorAssociation": "NONE",
                    }
                ]
            },
        }
    ]
    lanes.candidates["default"] = []
    lanes.thread_out = {
        kept: {"status": "OUTDATED", "reply": "That code moved away."},
        threaded: {"status": "STILL_VALID", "reply": "NO_REPLY"},
    }
    run_round(v12, conn, gh, lanes, A)
    led = issues(conn)
    assert led[gone].status == "outdated" and led[gone].status_source == "backfill"
    (work,) = [i for k, _, i in lanes.v12_calls if k == "thread"]
    assert sorted(work) == sorted([kept, threaded])  # left to the thread lane, not dropped
    assert led[kept].status == "outdated" and led[kept].status_source == "thread"
    assert led[threaded].status == "open" and led[threaded].last_seen_sha == A
    assert ledger.reconciled(conn, REPO, 1)["pending"] == []
    assert gh.posted_reviews[-1]["event"] == "REQUEST_CHANGES"  # the threaded one still stands
    # never again: a second round does not reconcile, and nothing pending is re-judged
    lanes.v12_calls.clear()
    run_round(v12, conn, gh, lanes, B)
    assert not [c for c in lanes.v12_calls if c[0] == "thread"]


def test_unknown_git_puts_open_issues_in_the_work_set(cfg, skills_dir, conn, gh, lanes, git):
    v12 = v12_cfg(cfg, skills_dir)
    h = backfilled(conn, "Some blocker", line=5)
    with tx(conn):
        ledger.reconcile(conn, REPO, 1, pr_open=True, open_threads=set(), moved={})
        ledger.settle_pending(conn, REPO, 1, [h])
    git.unknown = True
    lanes.candidates["default"] = []
    run_round(v12, conn, gh, lanes, A)
    assert [i for k, _, i in lanes.v12_calls if k == "thread"] == [[h]]


# ---- the thread lane ----


def test_a_failed_thread_lane_leaves_every_issue_standing(cfg, skills_dir, conn, gh, lanes, git):
    v12 = v12_cfg(cfg, skills_dir)
    lanes.verify12 = refute_gate
    lanes.candidates["default"] = []
    lanes.candidates["general"] = [cand("Fee estimate omits drainage")]
    run_round(v12, conn, gh, lanes, A)
    ids = publish_threads(gh)
    (fee,) = issues(conn)
    reply(gh, ids[fee], "Not an issue.", rid=901)
    lanes.thread_out[fee] = {"status": "WITHDRAWN", "reply": "NO_REPLY"}  # never valid
    lanes.candidates["general"] = []
    before = len(gh.posted_reviews)
    rid = run_round(v12, conn, gh, lanes, B)
    assert conn.execute(
        "SELECT 1 FROM events WHERE kind='threads.failed' AND run_id=?", (rid,)
    ).fetchone()
    assert issues(conn)[fee].status == "open"
    assert gh.posted_reviews[before]["event"] == "REQUEST_CHANGES"


def test_a_deferral_by_an_outside_contributor_still_blocks_on_a_new_push(
    cfg, skills_dir, conn, gh, lanes, git
):
    v12 = v12_cfg(cfg, skills_dir)
    lanes.verify12 = refute_gate
    lanes.candidates["default"] = []
    lanes.candidates["general"] = [cand("Fee estimate omits drainage")]
    run_round(v12, conn, gh, lanes, A)
    ids = publish_threads(gh)
    (fee,) = issues(conn)
    reply(
        gh,
        ids[fee],
        "I'd rather do it in a follow-up.",
        rid=901,
        who="drive-by",
        assoc="CONTRIBUTOR",
    )
    lanes.thread_out[fee] = {"status": "INTENTIONALLY_DEFERRED", "reply": "OK, a follow-up then."}
    lanes.candidates["general"] = []
    before = len(gh.posted_reviews)
    run_round(v12, conn, gh, lanes, B)
    assert issues(conn)[fee].status == "deferral_pending"
    (answer,) = [r for r in gh.replies if f"finding={fee}" in r["body"]]
    assert "needs a maintainer's call" in answer["body"]
    assert gh.posted_reviews[before]["event"] == "REQUEST_CHANGES"
    assert "a deferral was proposed" in gh.posted_reviews[before]["body"]


@pytest.mark.parametrize("tracked", [True, False])
def test_a_maintainer_deferral_needs_a_tracking_issue_on_a_new_push(
    cfg, skills_dir, conn, gh, lanes, git, tracked
):
    """dashpay/platform#5258 under v12: a maintainer deferral lifts the issue only once a
    tracking issue visible on the PR exists; otherwise the issue keeps blocking and the reply
    asks for a link."""
    v12 = v12_cfg(cfg, skills_dir)
    lanes.verify12 = refute_gate
    lanes.candidates["default"] = []
    lanes.candidates["general"] = [cand("Fee estimate omits drainage")]
    run_round(v12, conn, gh, lanes, A)
    ids = publish_threads(gh)
    (fee,) = issues(conn)
    said = "Deferring; tracked in #4321." if tracked else "Let's do this in a follow-up."
    reply(gh, ids[fee], said, rid=901)
    lanes.thread_out[fee] = {
        "status": "INTENTIONALLY_DEFERRED",
        "reply": "Agreed, #4321 covers it." if tracked else "Agreed, a follow-up then.",
    }
    lanes.candidates["general"] = []
    before = len(gh.posted_reviews)
    run_round(v12, conn, gh, lanes, B)
    (answer,) = [r for r in gh.replies if f"finding={fee}" in r["body"]]
    if tracked:
        assert issues(conn)[fee].status == "deferred"
        assert gh.posted_reviews[before]["event"] != "REQUEST_CHANGES"
        row = conn.execute("SELECT body FROM findings WHERE stage='conceded'").fetchone()
        assert (
            row["body"] == "INTENTIONALLY_DEFERRED\nhttps://github.com/dashpay/platform/issues/4321"
        )
    else:
        assert issues(conn)[fee].status == "open"
        assert "needs a tracking issue" in answer["body"] and "file it" not in answer["body"]
        assert gh.posted_reviews[before]["event"] == "REQUEST_CHANGES"
        assert not conn.execute("SELECT 1 FROM findings WHERE stage='conceded'").fetchone()


def test_fixed_on_the_reviewed_commit_is_a_pending_fix(cfg, skills_dir, conn, gh, lanes, git):
    v12 = v12_cfg(cfg, skills_dir)
    lanes.verify12 = refute_gate
    lanes.candidates["default"] = []
    lanes.candidates["general"] = [cand("Fee estimate omits drainage")]
    run_round(v12, conn, gh, lanes, A)
    ids = publish_threads(gh)
    (fee,) = issues(conn)
    reply(
        gh, ids[fee], "Here is a patch: https://github.com/dashpay/platform/commit/abcdef1", rid=901
    )
    lanes.thread_out[fee] = {"status": "FIXED", "reply": "That patch resolves it."}
    run_round(v12, conn, gh, lanes, A, Trigger.REVIEW_REPLY)
    assert issues(conn)[fee].status == "open"
    (answer,) = [r for r in gh.replies if f"finding={fee}" in r["body"]]
    assert "the push gets a full review" in answer["body"]
    detail = json.loads(
        conn.execute("SELECT detail FROM events WHERE kind='thread.answered'").fetchone()[0]
    )
    assert detail["status"] == "FIX_PENDING"


# ---- lane hygiene wiring ----


def test_structured_output_reaches_every_v12_lane(cfg, skills_dir, conn, gh, lanes, git):
    v12 = v12_cfg(cfg, skills_dir, structured_output=True)
    lanes.verify12 = refute_gate
    lanes.candidates["default"] = []
    lanes.candidates["general"] = [cand("Fee estimate omits drainage")]
    run_round(v12, conn, gh, lanes, A)
    ids = publish_threads(gh)
    (fee,) = issues(conn)
    reply(gh, ids[fee], "?", rid=901)
    run_round(v12, conn, gh, lanes, B)
    kinds = {
        "finder": "## Output: candidates",
        "verify": "## Output: verified groups",
        "thread": "## Output: thread outcomes",
    }
    for kind, marker in kinds.items():
        specs = [s for s in lanes.calls if marker in s.prompt]
        assert specs, kind
        for s in specs:
            assert s.json_schema is not None, kind
            assert (
                prompts.STRUCTURED_CONTRACT in s.prompt
                and prompts.RAW_JSON_CONTRACT not in s.prompt
            )
    thread = next(s for s in lanes.calls if kinds["thread"] in s.prompt)
    enum = thread.json_schema["properties"]["threads"]["items"]["properties"]["finding_hash"][
        "enum"
    ]
    assert enum == [fee]
    prep = next(s for s in lanes.calls if s.role == "prep")
    assert prep.json_schema is None


def test_v12_lanes_without_structured_output_answer_in_text(cfg, skills_dir, conn, gh, lanes, git):
    v12 = v12_cfg(cfg, skills_dir)
    lanes.candidates["default"] = [cand("x", severity="nitpick")]
    run_round(v12, conn, gh, lanes, A)
    v12_specs = [s for s in lanes.calls if "## Output: " in s.prompt]
    assert v12_specs and all(s.json_schema is None for s in v12_specs)
    assert all(prompts.RAW_JSON_CONTRACT in s.prompt for s in v12_specs)


def test_findings_over_the_comment_budget_are_not_posted(cfg, skills_dir, conn, gh, lanes, git):
    v12 = v12_cfg(cfg, skills_dir)
    lanes.verify12 = refute_gate
    lanes.candidates["default"] = []
    lanes.candidates["general"] = [
        cand(f"s{i}", severity="suggestion", line=i) for i in range(1, v12.comment_budget + 3)
    ]
    run_round(v12, conn, gh, lanes, A)
    review = gh.posted_reviews[-1]
    assert len(review["comments"]) == v12.comment_budget
    assert "2 more non-blocking finding(s) over the comment budget" in review["body"]
    assert len(issues(conn)) == v12.comment_budget


def test_outdated_on_the_reviewed_commit_lifts_nothing(cfg, skills_dir, conn, gh, lanes, git):
    """dashpay/dash#7778 by another word: a same-sha thread pass calling the only blocker
    OUTDATED must not close it, let alone approve."""
    v12 = v12_cfg(cfg, skills_dir)
    lanes.verify12 = refute_gate
    lanes.candidates["default"] = []
    lanes.candidates["general"] = [cand("Fee estimate omits drainage")]
    run_round(v12, conn, gh, lanes, A)
    ids = publish_threads(gh)
    (fee,) = issues(conn)
    reply(gh, ids[fee], "That code goes away in my next commit.", rid=901)
    lanes.thread_out[fee] = {"status": "OUTDATED", "reply": "Then this goes with it."}
    before = len(gh.posted_reviews)
    run_round(v12, conn, gh, lanes, A, Trigger.REVIEW_REPLY)
    assert issues(conn)[fee].status == "open"
    assert len(gh.posted_reviews) == before  # nothing lifted: no follow-up review
    (answer,) = [r for r in gh.replies if f"finding={fee}" in r["body"]]
    assert "the push gets a full review" in answer["body"]


def _git_repo(path):
    """A real repository: commit(lines) -> sha of f.rs holding `lines`."""
    import subprocess

    def git(*args):
        return subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q")

    def commit(lines):
        (path / "f.rs").write_text("\n".join(lines) + "\n")
        git("add", "f.rs")
        git("commit", "-q", "-m", "c")
        return git("rev-parse", "HEAD")

    return commit


def test_a_settled_issue_is_revived_by_its_own_lines_after_a_shift(cfg, conn, tmp_path):
    """The ledger's lines are numbered in the commit the issue was anchored on. 40 lines land
    above it before it is withdrawn; afterwards, a change to the line now sitting where it
    used to be does not revive it, and a change to its code (40 lines lower) does."""
    from tests.test_v12_rules import issue

    repo = tmp_path / "repo"
    repo.mkdir()
    commit = _git_repo(repo)
    base = [f"line {n}" for n in range(1, 101)]
    a = commit(base)
    shifted = [f"new {n}" for n in range(1, 41)] + base
    b = commit(shifted)  # withdrawn here; its code is at lines 51-52 now
    elsewhere = list(shifted)
    elsewhere[10] = "changed: unrelated, at the old line number"
    c1 = commit(elsewhere)
    commit(shifted)  # back to b's content, then change the issue's own code
    own = list(shifted)
    own[50] = "changed: the issue's own line"
    c2 = commit(own)
    w = issue(
        "a" * 12,
        "withdrawn",
        line_start=11,
        line_end=12,
        anchor_sha=a,
        opened_sha=a,
        last_seen_sha=a,
        closed_sha=b,
    )
    ctx = worker.RunContext(
        cfg=cfg,
        main_conn=conn,
        gh=None,
        run_id=1,
        head_id=1,
        repo=REPO,
        number=1,
        sha=c1,
        token="t",
        run_dir=tmp_path,
        worktree=repo,
    )
    assert worker._closed_changed(ctx, w) is False
    ctx.sha = c2
    assert worker._closed_changed(ctx, w) is True
    # what the anchor-blind check (closing commit, anchor lines) would have said instead
    from reviewsys import ledger as ledger_mod

    assert ledger_mod.lines_changed(repo, b, c1, "f.rs", 11, 12) is True
    assert ledger_mod.lines_changed(repo, b, c2, "f.rs", 11, 12) is False


def test_a_reconcile_leftover_does_not_count_until_judged(cfg, skills_dir, conn, gh, lanes, git):
    """The reconcile leaves an unchanged backfilled blocker with no thread to the thread lane;
    when that lane fails, the leftover must not request changes on its own (it may be one of
    the blockers that vanished without an outcome), though nothing can approve either."""
    v12 = v12_cfg(cfg, skills_dir)
    h = backfilled(conn, "Old blocker", line=40)
    lanes.candidates["default"] = []
    lanes.thread_out[h] = {"status": "WITHDRAWN", "reply": "NO_REPLY"}  # never valid
    rid = run_round(v12, conn, gh, lanes, A)
    assert conn.execute(
        "SELECT 1 FROM events WHERE kind='threads.failed' AND run_id=?", (rid,)
    ).fetchone()
    assert issues(conn)[h].status == "open"
    assert ledger.reconciled(conn, REPO, 1)["pending"] == [h]  # still to judge
    assert gh.posted_reviews[-1]["event"] == "COMMENT"


def _judged_backfill(conn, n):
    """`n` backfilled blockers the reconcile already handed over and the lane already judged
    once (nothing pending): from here on only replies and code changes bring them back."""
    hs = [backfilled(conn, f"Old blocker {i}", line=10 * (i + 1)) for i in range(n)]
    with tx(conn):
        ledger.reconcile(conn, REPO, 1, pr_open=True, open_threads=set(), moved={})
        ledger.settle_pending(conn, REPO, 1, hs)
    return hs


def test_a_failed_thread_shard_leaves_only_its_own_issues(cfg, skills_dir, conn, gh, lanes, git):
    v12 = v12_cfg(cfg, skills_dir)
    hs = _judged_backfill(conn, 14)
    git.unknown = True  # every anchored issue is judged: git cannot tell
    bad = hs[0]
    lanes.thread_out = {h: {"status": "FIXED", "reply": "Gone."} for h in hs}
    lanes.thread_out[bad] = {"status": "WITHDRAWN", "reply": "NO_REPLY"}  # never valid
    lanes.candidates["default"] = []
    rid = run_round(v12, conn, gh, lanes, A)
    shards = [i for k, _, i in lanes.v12_calls if k == "thread"]
    assert sorted(len(s) for s in {tuple(s) for s in shards}) == [7, 7]
    failed = next(s for s in shards if bad in s)
    led = issues(conn)
    assert {h for h in hs if led[h].status == "open"} == set(failed)
    assert {h for h in hs if led[h].status == "fixed"} == set(hs) - set(failed)
    roles = {
        r[0]
        for r in conn.execute("SELECT role FROM lanes WHERE run_id=? AND phase='threads'", (rid,))
    }
    assert roles == {"thread-1of2", "thread-2of2"}
    ev = conn.execute(
        "SELECT detail FROM events WHERE kind='threads.failed' AND run_id=?", (rid,)
    ).fetchone()
    assert "7 issue(s) left as they were" in ev["detail"]
    assert gh.posted_reviews[-1]["event"] == "REQUEST_CHANGES"  # the unread blockers stand


def test_an_issue_git_cannot_place_waits_for_a_reply(cfg, skills_dir, conn, gh, lanes, git):
    v12 = v12_cfg(cfg, skills_dir)
    f = Finding(file="", title="Design concern", body="b", severity="suggestion")
    with tx(conn):
        ledger.append(
            conn, REPO, 1, [ledger._posted(f, run_id=None, sha="0" * 40, source="backfill")]
        )
    anchorless = f.hash
    git.unknown = True
    lanes.candidates["default"] = []
    run_round(v12, conn, gh, lanes, A)
    run_round(v12, conn, gh, lanes, B)
    judged = [i for k, _, i in lanes.v12_calls if k == "thread"]
    assert judged == [[anchorless]]  # the first round's reconcile handed it over, once


def test_an_outdated_issue_is_revived_only_by_its_rewritten_lines(cfg, conn, tmp_path):
    """Its lines were rewritten before it closed (OUTDATED at b): a later edit elsewhere in the
    file revives nothing; an edit to the rewritten lines, where they sit at b, does; and when
    git cannot place them, nothing is revived."""
    from tests.test_v12_rules import issue

    repo = tmp_path / "repo"
    repo.mkdir()
    commit = _git_repo(repo)
    base = [f"line {n}" for n in range(1, 101)]
    a = commit(base)
    rewritten = [*base[:10], "new 1", "new 2", "new 3", *base[12:]]  # 11-12 become 11-13
    b = commit(rewritten)
    elsewhere = list(rewritten)
    elsewhere[80] = "changed: far from it"
    c1 = commit(elsewhere)
    commit(rewritten)
    own = list(rewritten)
    own[12] = "changed: its rewritten code"  # line 13 at b
    c2 = commit(own)
    o = issue(
        "a" * 12,
        "outdated",
        line_start=11,
        line_end=12,
        anchor_sha=a,
        opened_sha=a,
        last_seen_sha=a,
        closed_sha=b,
    )
    ctx = worker.RunContext(
        cfg=cfg,
        main_conn=conn,
        gh=None,
        run_id=1,
        head_id=1,
        repo=REPO,
        number=1,
        sha=c1,
        token="t",
        run_dir=tmp_path,
        worktree=repo,
    )
    assert worker._closed_changed(ctx, o) is False
    # what anchor_moved(since=closed) said: the anchor changed (before closing) + the file
    # changed since closing, so any later edit in the file revived it
    from reviewsys import ledger as ledger_mod

    def changed(x, y, file, ls, le):
        return ledger_mod.lines_changed(repo, x, y, file, ls, le)

    assert ledger_mod.anchor_moved(o, c1, changed, since=b) is True
    ctx.sha = c2
    assert worker._closed_changed(ctx, o) is True
    ctx.sha = "f" * 40  # a commit the mirror does not have
    assert worker._closed_changed(ctx, o) is None


def test_lines_are_relocated_through_a_diff():
    from reviewsys import ledger as ledger_mod

    hunks = [(5, 0, 6, 3), (20, 2, 23, 1), (40, 1, 41, 0)]  # +3 after 5; 20-21 -> 23; 40 gone
    m = ledger_mod._map_line
    assert [m(hunks, n, end=False) for n in (3, 5, 6, 19)] == [3, 5, 9, 22]
    assert (m(hunks, 20, end=False), m(hunks, 21, end=True)) == (23, 23)  # rewritten
    assert m(hunks, 30, end=False) == 32  # +3 -1
    assert m(hunks, 40, end=False) == 41 and m(hunks, 50, end=False) == 51
