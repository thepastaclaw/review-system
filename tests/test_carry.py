"""A push that leaves the PR's own diff as it was carries the earlier review (worker._carry)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from reviewsys import audit, ci, labels, status, worker
from reviewsys.db import kv_get, kv_set, now, tx
from reviewsys.ingest import carried_from, enqueue_head
from reviewsys.models import RunStatus, Trigger
from reviewsys.scheduler import schedule
from reviewsys.steps import worktree as wt

REPO = "dashpay/platform"
A, B, C = "a" * 40, "c" * 40, "d" * 40  # "b" * 40 is the fake merge base
PAST = "2020-01-01T00:00:00Z"  # queued long enough ago that the debounce has passed


@pytest.fixture
def prints(monkeypatch, tmp_path):
    """Local directories instead of git; the PR's diff per head is whatever the test says."""
    by_sha: dict[str, str | None] = {}
    monkeypatch.setattr(wt, "ensure_mirror", lambda mirrors, repo: tmp_path / "mirror")
    monkeypatch.setattr(wt, "fetch_head", lambda mirror, number, sha: None)

    def create(mirror, wts, name, sha):
        p = wts / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(wt, "create_worktree", create)
    monkeypatch.setattr(wt, "remove_worktree", lambda mirror, path: None)
    monkeypatch.setattr(wt, "merge_base", lambda worktree, base, sha: "b" * 40)
    monkeypatch.setattr(wt, "diff_print", lambda worktree, base, sha: by_sha.get(sha))
    return by_sha


@pytest.fixture
def reviewed(lanes):
    """Lanes that publish a final review with no findings, round after round."""
    verifier = {
        "summary": "Sound.",
        "review_action": "COMMENT",
        "findings": [],
        "dropped_findings": [],
        "out_of_scope_findings": [],
        "coderabbit_reactions": [],
        "prerequisite_adjudications": [],
        "adjudication_complete": True,
    }
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["preliminary"] = verifier
    lanes.verifier["final"] = verifier
    return lanes


def _push(conn, cfg, gh, sha, trigger=Trigger.NEW_PUSH):
    """The PR's head moves to `sha` (GitHub, ingest's `prs` row) and a head is queued for it."""
    gh.pr["head"]["sha"] = sha
    with tx(conn):
        conn.execute(
            "INSERT INTO prs (repo, number, head_sha, state, updated_at) VALUES (?,1,?,'open',?) "
            "ON CONFLICT(repo, number) DO UPDATE SET head_sha=excluded.head_sha",
            (REPO, sha, PAST),
        )
        return enqueue_head(conn, cfg, REPO, 1, sha, trigger, ts=PAST)


def _run(cfg, conn, gh, lanes, sha):
    lanes.head = sha
    (rid,) = schedule(conn, cfg, spawn=False)
    return rid, worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False)


def _review(cfg, conn, gh, lanes, sha, trigger=Trigger.MENTION):
    _push(conn, cfg, gh, sha, trigger)
    return _run(cfg, conn, gh, lanes, sha)


def _steps(conn, rid):
    return {r["name"] for r in conn.execute("SELECT name FROM steps WHERE run_id=?", (rid,))}


def _reviewed_itself(conn, rid):
    """The run reviewed its commit: lanes ran and it recorded a review of that commit."""
    sha = conn.execute(
        "SELECT h.sha FROM runs r JOIN heads h ON h.id=r.head_id WHERE r.id=?", (rid,)
    ).fetchone()["sha"]
    own = conn.execute("SELECT 1 FROM reviews WHERE run_id=? AND sha=?", (rid, sha)).fetchone()
    steps = _steps(conn, rid)
    return "carry" not in steps and "phase2" in steps and own is not None


def test_push_that_leaves_the_diff_alone_carries_the_review(cfg, conn, gh, reviewed, prints):
    prints.update({A: "p1", B: "p1"})
    _review(cfg, conn, gh, reviewed, A)
    assert len(gh.posted_reviews) == 1
    lanes_before = len(reviewed.calls)

    rid, st = _review(cfg, conn, gh, reviewed, B, Trigger.NEW_PUSH)

    assert st == RunStatus.DONE
    assert len(reviewed.calls) == lanes_before  # no lane ran
    assert len(gh.posted_reviews) == 1  # and nothing new was posted as a review
    assert carried_from(conn, REPO, 1, B) == A
    assert labels.verdict_for(conn, REPO, 1, B) == "COMMENT"
    assert labels.wanted(conn, REPO, 1) == labels.VERDICT_LABELS["COMMENT"]
    gate = gh.gate_bodies[-1]
    assert f"No change to this PR's diff since commit {A[:8]}" in gate
    assert f"(commit {B[:8]})" in gate and "commented" in gate and "@thepastaclaw review" in gate
    assert _steps(conn, rid) == {"worktree", "carry"}
    ev = conn.execute("SELECT detail FROM events WHERE kind='head.carried'").fetchone()
    assert ev and A[:8] in ev["detail"]


def test_carried_commit_passes_the_review_on_to_the_next_one(cfg, conn, gh, reviewed, prints):
    prints.update({A: "p1", B: "p1", C: "p1"})
    _review(cfg, conn, gh, reviewed, A)
    _review(cfg, conn, gh, reviewed, B, Trigger.NEW_PUSH)
    _review(cfg, conn, gh, reviewed, C, Trigger.NEW_PUSH)
    assert carried_from(conn, REPO, 1, C) == A  # the commit that was reviewed, not B
    assert len(gh.posted_reviews) == 1


def test_carried_runs_are_no_sample_of_review_time(cfg, conn, gh, reviewed, prints):
    prints.update({A: "p1", B: "p1", C: "p1"})
    rid, _ = _review(cfg, conn, gh, reviewed, A)
    _review(cfg, conn, gh, reviewed, B, Trigger.NEW_PUSH)
    _review(cfg, conn, gh, reviewed, C, Trigger.NEW_PUSH)
    with tx(conn):
        conn.execute(
            "UPDATE runs SET started_at='2026-01-01T00:00:00Z', finished_at='2026-01-01T01:00:00Z'"
            " WHERE id=?",
            (rid,),
        )
    assert status.median_run_minutes(conn) == 60


def test_changed_diff_gets_a_full_review(cfg, conn, gh, reviewed, prints):
    prints.update({A: "p1", B: "p2"})
    _review(cfg, conn, gh, reviewed, A)
    rid, _ = _review(cfg, conn, gh, reviewed, B, Trigger.NEW_PUSH)
    assert _reviewed_itself(conn, rid)
    assert carried_from(conn, REPO, 1, B) is None


@pytest.mark.parametrize("unknown", [A, B])
def test_unknown_diff_never_counts_as_unchanged(cfg, conn, gh, reviewed, prints, unknown):
    prints.update({A: "p1", B: "p1"})
    prints[unknown] = None  # git could not say
    _review(cfg, conn, gh, reviewed, A)
    rid, _ = _review(cfg, conn, gh, reviewed, B, Trigger.NEW_PUSH)
    assert _reviewed_itself(conn, rid)


def test_asking_for_a_review_is_never_carried(cfg, conn, gh, reviewed, prints):
    prints.update({A: "p1", B: "p1"})
    _review(cfg, conn, gh, reviewed, A)
    rid, _ = _review(cfg, conn, gh, reviewed, B, Trigger.MENTION)
    assert _reviewed_itself(conn, rid)
    assert carried_from(conn, REPO, 1, B) is None


def test_a_superseded_predecessor_breaks_the_chain(cfg, conn, gh, reviewed, prints):
    """B was replaced by C before any run recorded B's diff: what changed in between is not
    known, so C starts over even though its diff equals A's."""
    prints.update({A: "p1", C: "p1"})
    _review(cfg, conn, gh, reviewed, A)
    _push(conn, cfg, gh, B)
    rid, _ = _review(cfg, conn, gh, reviewed, C, Trigger.NEW_PUSH)
    assert _reviewed_itself(conn, rid)
    assert carried_from(conn, REPO, 1, C) is None


def test_asking_on_a_carried_commit_gets_it_reviewed(cfg, conn, gh, reviewed, prints):
    prints.update({A: "p1", B: "p1"})
    _review(cfg, conn, gh, reviewed, A)
    _review(cfg, conn, gh, reviewed, B, Trigger.NEW_PUSH)
    with tx(conn):
        # a mention on a commit that was reviewed itself still changes nothing
        assert enqueue_head(conn, cfg, REPO, 1, A, Trigger.MENTION) == "noop"
        assert enqueue_head(conn, cfg, REPO, 1, B, Trigger.REVIEW_REQUESTED) == "requeued"
    rid, _ = _run(cfg, conn, gh, reviewed, B)
    assert _reviewed_itself(conn, rid)
    assert carried_from(conn, REPO, 1, B) is None  # its own review stands now
    with tx(conn):
        assert enqueue_head(conn, cfg, REPO, 1, B, Trigger.MENTION) == "noop"


def test_carry_restores_the_verdict_label(cfg, conn, gh, reviewed, prints):
    """The push cleared the label (no verdict on the new commit yet); the carry writes no
    `reviews` row, and the next label pass must still put it back."""
    prints.update({A: "p1", B: "p1"})
    _review(cfg, conn, gh, reviewed, A)
    _push(conn, cfg, gh, B)
    with tx(conn):
        conn.execute("UPDATE reviews SET posted_at=?", (PAST,))  # older than the cursor
        kv_set(conn, "labels.reconciled_at", now())
    gh.labels = []
    _run(cfg, conn, gh, reviewed, B)
    labels.reconcile(conn, gh, repos={})
    assert gh.labels == [labels.VERDICT_LABELS["COMMENT"]]


def test_asking_while_the_run_checks_out_is_not_carried(
    cfg, conn, gh, reviewed, prints, monkeypatch
):
    prints.update({A: "p1", B: "p1"})
    _review(cfg, conn, gh, reviewed, A)

    def mentioned_meanwhile(worktree, base, sha):
        with tx(conn):
            assert enqueue_head(conn, cfg, REPO, 1, sha, Trigger.MENTION) == "promoted"
        return prints[sha]

    monkeypatch.setattr(wt, "diff_print", mentioned_meanwhile)
    rid, _ = _review(cfg, conn, gh, reviewed, B, Trigger.NEW_PUSH)
    assert _reviewed_itself(conn, rid)


def test_asking_as_the_carry_finishes_queues_a_review(cfg, conn, gh, reviewed, prints, monkeypatch):
    """A mention landing after the carry committed, while its run still writes the gate comment:
    the run ends done, and the review the person asked for follows it."""
    prints.update({A: "p1", B: "p1"})
    _review(cfg, conn, gh, reviewed, A)
    gate = worker._gate_comment

    def mentioned_meanwhile(ctx, status_, **kw):
        if status_ == "carried":
            with tx(conn):
                assert enqueue_head(conn, cfg, REPO, 1, B, Trigger.MENTION) == "promoted"
        gate(ctx, status_, **kw)

    monkeypatch.setattr(worker, "_gate_comment", mentioned_meanwhile)
    rid, st = _review(cfg, conn, gh, reviewed, B, Trigger.NEW_PUSH)
    assert st == RunStatus.DONE and "carry" in _steps(conn, rid)
    head = conn.execute("SELECT status, trigger FROM heads WHERE sha=?", (B,)).fetchone()
    assert (head["status"], head["trigger"]) == ("queued", "manual")
    monkeypatch.setattr(worker, "_gate_comment", gate)
    rid, _ = _run(cfg, conn, gh, reviewed, B)
    assert _reviewed_itself(conn, rid)


def test_a_head_superseded_during_checkout_carries_nothing(
    cfg, conn, gh, reviewed, prints, monkeypatch
):
    prints.update({A: "p1", B: "p1"})
    _review(cfg, conn, gh, reviewed, A)

    def pushed_meanwhile(worktree, base, sha):
        with tx(conn):  # what the scheduler's supersede sweep does when the next push lands
            conn.execute("UPDATE heads SET status='superseded' WHERE sha=?", (sha,))
        return prints[sha]

    monkeypatch.setattr(wt, "diff_print", pushed_meanwhile)
    gates = len(gh.gate_bodies)
    _, st = _review(cfg, conn, gh, reviewed, B, Trigger.NEW_PUSH)
    assert st == RunStatus.CANCELLED
    assert carried_from(conn, REPO, 1, B) is None
    assert not any("No change to this PR" in g for g in gh.gate_bodies[gates:])


@pytest.mark.parametrize("weakened", ["preliminary", "degraded", "ci_updated"])
def test_a_weakened_review_is_not_carried(cfg, conn, gh, reviewed, prints, weakened):
    """Only a final review on the primary models that no CI update downgraded is carried: a
    preliminary one still owes its Phase 2, and the push after a downgrade or a degraded run
    is the one that may win the full verdict back."""
    prints.update({A: "p1", B: "p1"})
    rid_a, _ = _review(cfg, conn, gh, reviewed, A)
    with tx(conn):
        if weakened == "preliminary":
            conn.execute("UPDATE reviews SET phase='preliminary' WHERE sha=?", (A,))
        elif weakened == "degraded":
            conn.execute("UPDATE runs SET degraded=1 WHERE id=?", (rid_a,))
        else:
            kv_set(conn, ci.updated_key(REPO, 1, A), "5000")
    rid, _ = _review(cfg, conn, gh, reviewed, B, Trigger.NEW_PUSH)
    assert carried_from(conn, REPO, 1, B) is None and "carry" not in _steps(conn, rid)


def _carried_approval(conn):
    """A approved and reviewed; B carried over from it and the PR's live head."""
    with tx(conn):
        conn.execute(
            "INSERT INTO prs (repo, number, head_sha, state, updated_at) VALUES (?,1,?,'open',?)",
            (REPO, B, PAST),
        )
        for sha, src in ((A, None), (B, A)):
            conn.execute(
                "INSERT INTO heads (repo, number, sha, trigger, status, queued_at, eligible_at, "
                "finished_at, diff_print, carried_from) VALUES (?,1,?,'new_push','done',?,?,?,'p1',?)",
                (REPO, sha, PAST, PAST, PAST, src),
            )
        conn.execute(
            "INSERT INTO reviews (repo, number, sha, phase, github_review_id, event, posted_at) "
            "VALUES (?,1,?,'final',5000,'APPROVED',?)",
            (REPO, A, PAST),
        )


def test_merged_carried_head_needs_no_audit(cfg, conn):
    _carried_approval(conn)
    pr = audit.MergedPr(
        repo=REPO,
        number=1,
        title="feat: x",
        author="someone",
        merged_by="someone",
        merged_at=PAST,
        head_sha=B,
        base_ref="v4.2-dev",
        merge_commit="e" * 40,
        size=10,
    )
    with tx(conn):
        assert audit.enqueue(conn, cfg, pr, source="sweep", rank=0) == "clean"


def test_red_ci_on_a_carried_approval_queues_a_review_of_the_commit(cfg, conn):
    _carried_approval(conn)
    assert labels.verdict_for(conn, REPO, 1, B) == "APPROVE"
    w = {"repo": REPO, "number": 1, "sha": B, "run_id": None, "state": ci.WATCHING}
    key, raw = ci.watch_key(REPO, 1, B), json.dumps(w)
    with tx(conn):
        conn.execute("INSERT INTO kv (key, value) VALUES (?,?)", (key, raw))
    assert ci._enqueue(conn, cfg, key, raw, w, ["build"]) == ("enqueued", True)
    head = conn.execute("SELECT status, trigger, priority FROM heads WHERE sha=?", (B,)).fetchone()
    # not the light CI check (it judges a review of this commit, and there is none)
    assert (head["status"], head["trigger"], head["priority"]) == ("queued", "manual", 1)
    # and the watch is over: the review reads CI itself (and watches anew if it approves), so
    # a review that fails for good is never re-opened pass after pass
    assert kv_get(conn, key) is None
    ev = conn.execute("SELECT detail FROM events WHERE kind='ci.failed'").fetchone()
    assert "review of this commit queued" in ev["detail"]


def test_red_ci_never_reopens_a_carried_commit_whose_review_failed(cfg, conn):
    _carried_approval(conn)
    with tx(conn):  # someone asked, and the review of B failed for good
        conn.execute("UPDATE heads SET status='failed' WHERE sha=?", (B,))
    w = {"repo": REPO, "number": 1, "sha": B, "run_id": None, "state": ci.WATCHING}
    key, raw = ci.watch_key(REPO, 1, B), json.dumps(w)
    with tx(conn):
        conn.execute("INSERT INTO kv (key, value) VALUES (?,?)", (key, raw))
    assert ci._enqueue(conn, cfg, key, raw, w, ["build"]) == ("dropped", True)
    assert conn.execute("SELECT status FROM heads WHERE sha=?", (B,)).fetchone()[0] == "failed"
    assert kv_get(conn, key) is None


# ---- the fingerprint itself, on a real repository ----


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"},
    ).stdout.strip()


def _commit(repo: Path, files: dict[str, bytes], msg: str) -> str:
    for name, data in files.items():
        (repo / name).write_bytes(data)
        _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def _print(repo: Path, sha: str) -> str | None:
    return wt.diff_print(repo, _git(repo, "merge-base", "main", sha), sha)


def test_diff_print_sees_through_empty_commits_merges_and_rebases(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, {"f.txt": b"one\n", "g.txt": b"g\n"}, "base")
    _git(repo, "checkout", "-q", "-b", "pr")
    head = _commit(repo, {"f.txt": b"one\ntwo\n", "bin.dat": b"\xff\xfe\x00latin\xe9"}, "change")
    reviewed = _print(repo, head)
    assert reviewed is not None  # a diff that is not UTF-8 still has a print

    _git(repo, "commit", "-q", "--allow-empty", "-m", "retrigger CI")
    assert _print(repo, _git(repo, "rev-parse", "HEAD")) == reviewed

    _git(repo, "checkout", "-q", "main")
    _commit(repo, {"g.txt": b"g\nmore\n"}, "base moves on")
    _git(repo, "checkout", "-q", "pr")
    _git(repo, "merge", "-q", "--no-edit", "main")
    assert _print(repo, _git(repo, "rev-parse", "HEAD")) == reviewed  # merge of the base

    _git(repo, "checkout", "-q", "-b", "rebased", head)
    _git(repo, "rebase", "-q", "main")
    rebased = _git(repo, "rev-parse", "HEAD")
    assert rebased != head and _print(repo, rebased) == reviewed  # rebase onto it

    changed = _commit(repo, {"f.txt": b"one\nthree\n"}, "fixup")
    assert _print(repo, changed) != reviewed
    assert wt.diff_print(repo, "0" * 40, changed) is None  # git cannot say
