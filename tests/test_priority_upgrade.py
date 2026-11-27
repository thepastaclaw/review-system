"""Upgrading a running normal review to priority (Pasta, 2026-11-27)."""

from reviewsys import github, progress
from reviewsys.db import kv_get, tx
from reviewsys.ingest import enqueue_head
from reviewsys.models import Trigger
from reviewsys.queue_status import update_queue_comments
from reviewsys.scheduler import schedule

REPO = "dashpay/platform"
SHA = "c" * 40


def _running(conn, cfg):
    with tx(conn):
        enqueue_head(conn, cfg, REPO, 7, SHA, Trigger.NORMAL_REQUEST)
        conn.execute(
            "INSERT INTO prs (repo, number, head_sha, state, updated_at) VALUES (?,?,?,?,?)",
            (REPO, 7, SHA, "open", "2026-11-27T00:00:00Z"),
        )
    (rid,) = schedule(conn, cfg, spawn=False)
    return rid


def _gate(gh, body):
    gh.issue_comments[:] = [{"id": 5, "user": {"login": "thepastaclaw"}, "body": body}]


def _head(conn):
    return conn.execute("SELECT priority, status, trigger FROM heads WHERE number=7").fetchone()


def test_the_running_body_offers_the_upgrade_only_on_a_normal_review():
    normal = github.gate_body("in_progress", SHA, run_id=3, priority=False)
    assert github.RUNNING_PRIORITY_BOX in normal
    assert normal.endswith(github.RUNNING_PRIORITY_BOX)
    assert github.gate_body("in_progress", SHA, run_id=3, priority=True).endswith(" · priority")
    assert "Request" not in github.gate_body("in_progress", SHA, run_id=3)


def test_a_ticked_box_upgrades_the_running_review_in_place(conn, cfg, gh):
    rid = _running(conn, cfg)
    body = github.gate_body("in_progress", SHA, run_id=rid, priority=False)
    _gate(gh, body.replace("- [ ] **Request priority", "- [x] **Request priority"))
    stats = update_queue_comments(conn, cfg, gh)
    assert stats["promoted"] == 1
    head = _head(conn)
    assert (head["priority"], head["status"], head["trigger"]) == (1, "running", "normal_request")
    run = conn.execute("SELECT status, path FROM runs WHERE id=?", (rid,)).fetchone()
    # the same run, not restarted; its timing stays on the path it started on
    assert run["path"] == progress.path_of("live", 0)
    assert gh.gate_bodies[-1] == github.gate_body("in_progress", SHA, run_id=rid, priority=True)
    ev = conn.execute("SELECT detail FROM events WHERE kind='head.priority_requested'").fetchone()
    assert "mid-review" in ev[0]
    # no second review is asked for after this one
    assert kv_get(conn, f"rereview.after_run:{rid}") is None
    assert conn.execute("SELECT COUNT(*) FROM heads WHERE number=7").fetchone()[0] == 1


def test_an_unticked_or_stale_body_changes_nothing(conn, cfg, gh):
    rid = _running(conn, cfg)
    _gate(gh, github.gate_body("in_progress", SHA, run_id=rid, priority=False))
    assert update_queue_comments(conn, cfg, gh)["promoted"] == 0
    stale = github.gate_body("in_progress", "d" * 40, run_id=rid, priority=False)
    _gate(gh, stale.replace("- [ ]", "- [x]"))  # another commit's run
    assert update_queue_comments(conn, cfg, gh)["promoted"] == 0
    assert _head(conn)["priority"] == 0


def test_a_running_drafts_ticked_box_asks_for_no_second_review(conn, cfg, gh):
    """The deferred loop honours a draft's boxes by asking for a review; on the running
    body's box that would queue a re-review after this one."""
    rid = _running(conn, cfg)
    with tx(conn):
        conn.execute("UPDATE prs SET is_draft=1 WHERE number=7")
    body = github.gate_body("in_progress", SHA, run_id=rid, priority=False)
    _gate(gh, body.replace("- [ ]", "- [x]"))
    update_queue_comments(conn, cfg, gh)
    assert _head(conn)["priority"] == 1
    assert kv_get(conn, f"rereview.after_run:{rid}") is None


def test_a_push_replaces_the_old_commits_running_body(conn, cfg, gh):
    """A push during a running review leaves its "in progress" body; the next pass replaces
    it with the new head's own (a draft's boxes), and that body's box asks for nothing."""
    from reviewsys.queue_status import DEFERRED_MARKER

    rid = _running(conn, cfg)
    new = "e" * 40
    with tx(conn):
        conn.execute("UPDATE prs SET is_draft=1, head_sha=? WHERE number=7", (new,))
    old = github.gate_body("in_progress", SHA, run_id=rid, priority=False)
    _gate(gh, old.replace("- [ ]", "- [x]"))
    update_queue_comments(conn, cfg, gh)
    assert gh.gate_bodies and DEFERRED_MARKER in gh.gate_bodies[-1]
    assert f"Commit {new[:8]}" in gh.gate_bodies[-1]
    # the old commit's run may take the tick (its own body); the new commit gets no review
    assert conn.execute("SELECT COUNT(*) FROM heads WHERE sha=?", (new,)).fetchone()[0] == 0


def test_a_priority_run_still_showing_the_box_loses_it(conn, cfg, gh):
    rid = _running(conn, cfg)
    with tx(conn):
        conn.execute("UPDATE heads SET priority=1 WHERE number=7")
    _gate(gh, github.gate_body("in_progress", SHA, run_id=rid, priority=False))
    stats = update_queue_comments(conn, cfg, gh)
    assert stats["promoted"] == 0
    assert gh.gate_bodies[-1] == github.gate_body("in_progress", SHA, run_id=rid, priority=True)
