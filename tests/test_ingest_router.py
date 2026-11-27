import dataclasses
from datetime import timedelta

from reviewsys.db import kv_get, kv_set, now, parse_ts, tx
from reviewsys.ingest import enqueue_head, ingest_notifications, ingest_prs
from reviewsys.models import Trigger
from reviewsys.queue_status import (
    PRIORITY_BOX,
    SKIP_LABEL_NOTE,
    queued_order,
    update_queue_comments,
)
from reviewsys.router import route_inbox


def pr(number: int, sha: str, **kw):
    return {
        "number": number,
        "headRefOid": sha,
        "isDraft": False,
        "title": f"PR {number}",
        "updatedAt": "2026-09-01T00:00:00Z",
        "author": {"login": "dev"},
        **kw,
    }


def test_ingest_creates_and_supersedes_heads(cfg, conn, gh):
    gh.open_prs["dashpay/platform"] = [
        pr(1, "a" * 40),
        pr(2, "b" * 40, isDraft=True),
        pr(3, "c" * 40, author={"login": "dependabot[bot]"}),
    ]
    stats = ingest_prs(conn, cfg, gh)["dashpay/platform"]
    assert stats["created"] == 1 and stats["ignored"] == 1
    heads = conn.execute("SELECT * FROM heads").fetchall()
    assert len(heads) == 1 and heads[0]["trigger"] == "new_pr" and heads[0]["status"] == "queued"
    assert parse_ts(heads[0]["eligible_at"]) - parse_ts(heads[0]["queued_at"]) == timedelta(
        minutes=30
    )
    # push a new commit -> old head superseded, new one queued as new_push
    gh.open_prs["dashpay/platform"] = [pr(1, "d" * 40)]
    stats = ingest_prs(conn, cfg, gh)["dashpay/platform"]
    assert stats["created"] == 1
    rows = {r["sha"][:1]: r for r in conn.execute("SELECT * FROM heads")}
    assert (
        rows["a"]["status"] == "superseded"
        and rows["d"]["status"] == "queued"
        and rows["d"]["trigger"] == "new_push"
    )
    # PR closes -> head closed
    gh.open_prs["dashpay/platform"] = []
    stats = ingest_prs(conn, cfg, gh)["dashpay/platform"]
    assert stats["closed"] == 1
    assert (
        conn.execute("SELECT status FROM heads WHERE sha=?", ("d" * 40,)).fetchone()["status"]
        == "closed"
    )


def test_ingest_disabled_repo_not_polled(cfg, conn, gh):
    gh.open_prs["dashpay/grovedb"] = [pr(9, "e" * 40)]
    ingest_prs(conn, cfg, gh)
    assert conn.execute("SELECT COUNT(*) FROM heads").fetchone()[0] == 0


def test_ingest_error_streak_and_isolation(cfg, conn, gh):
    gh.fail_next = ["error connecting to api.github.com"]
    res = ingest_prs(conn, cfg, gh)
    assert res == {} and kv_get(conn, "ingest.error_streak") == "1"
    assert conn.execute("SELECT COUNT(*) FROM events WHERE kind='ingest.error'").fetchone()[0] == 1


def test_priority_promotes_existing_queued_head(cfg, conn, gh):
    with tx(conn):
        assert enqueue_head(conn, cfg, "dashpay/platform", 1, "a" * 40, Trigger.NEW_PR) == "created"
        assert enqueue_head(conn, cfg, "dashpay/platform", 1, "a" * 40, Trigger.NEW_PR) == "noop"
        assert (
            enqueue_head(conn, cfg, "dashpay/platform", 1, "a" * 40, Trigger.REVIEW_REQUESTED)
            == "promoted"
        )
    row = conn.execute("SELECT priority, eligible_at, queued_at FROM heads").fetchone()
    assert (row["priority"] == 1 and row["eligible_at"] <= row["queued_at"]) or row["priority"] == 1


def test_notifications_to_inbox_and_routing(cfg, conn, gh, notifier):
    gh.notifications = [
        {
            "id": "1",
            "reason": "review_requested",
            "updated_at": "2026-09-01T10:00:00Z",
            "subject": {
                "type": "PullRequest",
                "url": "https://api.github.com/repos/dashpay/platform/pulls/7",
                "latest_comment_url": None,
            },
            "repository": {"full_name": "dashpay/platform"},
        },
        {
            "id": "2",
            "reason": "mention",
            "updated_at": "2026-09-01T10:01:00Z",
            "subject": {
                "type": "PullRequest",
                "url": "https://api.github.com/repos/dashpay/platform/pulls/8",
                "latest_comment_url": "https://api.github.com/repos/dashpay/platform/issues/comments/555",
            },
            "repository": {"full_name": "dashpay/platform"},
        },
        {
            "id": "3",
            "reason": "comment",
            "updated_at": "2026-09-01T10:02:00Z",
            "subject": {
                "type": "Issue",
                "url": "https://api.github.com/repos/dashpay/platform/issues/9",
            },
            "repository": {"full_name": "dashpay/platform"},
        },
    ]
    gh.routes["repos/dashpay/platform/issues/comments/555"] = {
        "body": "@thepastaclaw please review this",
        "user": {"login": "QuantumExplorer"},
        "author_association": "MEMBER",
    }
    added = ingest_notifications(conn, cfg, gh)
    assert added == 2  # issue notification filtered
    # cursor is bootstrapped to "now - 10m" on first run and only ever moves forward
    assert kv_get(conn, "notify.cursor") >= "2026-09-01T10:01:00Z"
    assert ingest_notifications(conn, cfg, gh) == 0  # idempotent
    gh.pr = {**gh.pr, "head": {"sha": "f" * 40}}
    stats = route_inbox(conn, cfg, gh, notifier)
    assert stats["review_requested"] == 1 and stats["mention"] == 1
    heads = conn.execute("SELECT number, trigger, priority FROM heads ORDER BY number").fetchall()
    assert [(h["number"], h["trigger"], h["priority"]) for h in heads] == [
        (7, "review_requested", 1),
        (8, "mention", 1),
    ]
    assert conn.execute("SELECT COUNT(*) FROM inbox WHERE handled_at IS NULL").fetchone()[0] == 0


def test_own_pr_comment_batches_to_agent(cfg, conn, gh, notifier):
    gh.pr = {**gh.pr, "user": {"login": "thepastaclaw"}}
    gh.notifications = [
        {
            "id": "10",
            "reason": "comment",
            "updated_at": "2026-09-01T10:00:00Z",
            "subject": {
                "type": "PullRequest",
                "url": "https://api.github.com/repos/dashpay/platform/pulls/5",
                "latest_comment_url": "https://api.github.com/repos/dashpay/platform/pulls/comments/1",
            },
            "repository": {"full_name": "dashpay/platform"},
        },
        {
            "id": "11",
            "reason": "comment",
            "updated_at": "2026-09-01T10:00:30Z",
            "subject": {
                "type": "PullRequest",
                "url": "https://api.github.com/repos/dashpay/platform/pulls/5",
                "latest_comment_url": "https://api.github.com/repos/dashpay/platform/pulls/comments/2",
            },
            "repository": {"full_name": "dashpay/platform"},
        },
        {
            "id": "12",
            "reason": "comment",
            "updated_at": "2026-09-01T10:01:00Z",
            "subject": {
                "type": "PullRequest",
                "url": "https://api.github.com/repos/dashpay/platform/pulls/5",
                "latest_comment_url": "https://api.github.com/repos/dashpay/platform/pulls/comments/3",
            },
            "repository": {"full_name": "dashpay/platform"},
        },
    ]
    gh.routes["repos/dashpay/platform/pulls/comments/1"] = {
        "body": "fix this",
        "user": {"login": "shumkov"},
        "author_association": "MEMBER",
    }
    gh.routes["repos/dashpay/platform/pulls/comments/2"] = {
        "body": "and this",
        "user": {"login": "coderabbitai[bot]"},
        "author_association": "NONE",
    }
    gh.routes["repos/dashpay/platform/pulls/comments/3"] = {
        "body": "drive-by",
        "user": {"login": "randomuser"},
        "author_association": "NONE",
    }
    ingest_notifications(conn, cfg, gh)
    stats = route_inbox(conn, cfg, gh, notifier)
    assert stats["own_pr_comment"] == 2 and stats["ignored"] == 1
    # batch is older than 5 minutes (occurred_at is 2026-09-01) -> delivered as one wake
    assert stats["batches_sent"] == 1
    assert len([s for s in notifier.sent if s[0] == "wake"]) == 1
    assert "2 new comment(s) from coderabbitai[bot], shumkov" in notifier.sent[-1][1]
    assert (
        conn.execute("SELECT COUNT(*) FROM heads").fetchone()[0] == 0
    )  # own PRs are never reviewed by us


def test_shadow_mode_leaves_own_pr_batches_unhandled(cfg, conn, gh):
    from reviewsys.db import tx
    from reviewsys.notify import Notifier
    from reviewsys.router import flush_batches

    with tx(conn):
        conn.execute(
            "INSERT INTO inbox (source_id, kind, repo, number, actor, body, occurred_at, seen_at, action) VALUES ('s1','comment','dashpay/platform',5,'shumkov','fix','2026-09-01T10:00:00Z','2026-09-01T10:00:00Z','own_pr_comment')"
        )
    shadow = Notifier(cfg, runner=lambda argv: None, wake_enabled=False)  # type: ignore[arg-type,return-value]
    assert flush_batches(conn, cfg, shadow) == 0
    assert conn.execute("SELECT handled_at FROM inbox").fetchone()[0] is None
    live = Notifier(
        cfg, runner=lambda argv: __import__("subprocess").CompletedProcess(list(argv), 0, "", "")
    )
    assert flush_batches(conn, cfg, live) == 1
    assert conn.execute("SELECT action FROM inbox").fetchone()[0] == "own_pr_comment_delivered"


def test_failed_wake_keeps_batch_and_retries_later(cfg, conn, gh):
    import subprocess

    from reviewsys.db import kv_get, tx
    from reviewsys.notify import Notifier
    from reviewsys.router import flush_batches

    with tx(conn):
        conn.execute(
            "INSERT INTO inbox (source_id, kind, repo, number, actor, body, occurred_at, seen_at, action) VALUES ('s1','comment','dashpay/platform',5,'shumkov','fix','2026-09-01T10:00:00Z','2026-09-01T10:00:00Z','own_pr_comment')"
        )
    broken = Notifier(
        cfg, runner=lambda argv: subprocess.CompletedProcess(list(argv), 1, "", "down")
    )
    assert flush_batches(conn, cfg, broken) == 0
    assert conn.execute("SELECT handled_at FROM inbox").fetchone()[0] is None
    assert kv_get(conn, "router.batch_retry_after:dashpay/platform#5") is not None
    assert conn.execute("SELECT kind FROM events ORDER BY id DESC LIMIT 1").fetchone()[0] == (
        "router.batch_failed"
    )
    # still inside the retry window: not re-attempted
    assert flush_batches(conn, cfg, broken) == 0
    with tx(conn):
        conn.execute("DELETE FROM kv WHERE key LIKE 'router.batch_retry_after:%'")
    live = Notifier(cfg, runner=lambda argv: subprocess.CompletedProcess(list(argv), 0, "", ""))
    assert flush_batches(conn, cfg, live) == 1
    assert conn.execute("SELECT action FROM inbox").fetchone()[0] == "own_pr_comment_delivered"


def _queue(conn, cfg, n):
    with tx(conn):
        for i in range(n):
            enqueue_head(conn, cfg, "dashpay/platform", 300 + i, f"{i:040x}", Trigger.NEW_PR)


def test_queue_comments_are_static_when_the_order_changes(cfg, conn, gh):
    _queue(conn, cfg, 3)
    with tx(conn):
        conn.execute("UPDATE heads SET eligible_at=queued_at")
    stats = update_queue_comments(conn, cfg, gh)
    assert stats == {"written": 3, "promoted": 0, "deferred": 0, "yielded": 0}
    first = gh.gate_bodies[0]
    assert "thepastaclaw-gate" in first
    assert "Queued for automated review (commit 00000000)" in first
    assert "[queue and estimated start](" in first
    assert PRIORITY_BOX in first
    assert "in line" not in first and "estimated review time" not in first
    written = len(gh.gate_bodies)
    with tx(conn):
        conn.execute("UPDATE heads SET queued_at='z' WHERE number=302")
    assert update_queue_comments(conn, cfg, gh)["written"] == 0
    assert len(gh.gate_bodies) == written


def test_queue_body_does_not_depend_on_run_cap(cfg, conn, gh):
    cfg = dataclasses.replace(cfg, max_runs=0)
    _queue(conn, cfg, 5)
    with tx(conn):
        conn.execute("UPDATE heads SET eligible_at=queued_at")
    update_queue_comments(conn, cfg, gh)
    assert all("in line" not in b and "estimated review time" not in b for b in gh.gate_bodies)


def test_ticked_checkbox_promotes_head_to_front(cfg, conn, gh):
    _queue(conn, cfg, 3)
    with tx(conn):
        conn.execute("UPDATE heads SET eligible_at=queued_at")
    update_queue_comments(conn, cfg, gh)
    # serve a ticked box only for PR 302 (comment id 302), untouched bodies for the others
    for n, body in zip((300, 301, 302), gh.gate_bodies, strict=True):
        b = body.replace("- [ ] **Request", "- [x] **Request") if n == 302 else body
        gh.routes[f"repos/dashpay/platform/issues/comments/{n}"] = {
            "id": n,
            "node_id": f"IC_{n}",
            "user": {"login": "thepastaclaw"},
            "body": b,
        }
    with tx(conn):
        for n in (300, 301, 302):
            conn.execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
                (f"queue.comment_id:dashpay/platform#{n}", str(n)),
            )
    gh.routes["graphql:userContentEdits"] = {
        "node": {
            "userContentEdits": {
                "nodes": [
                    {"editedAt": "2026-09-01T00:00:00Z", "editor": {"login": "thepastaclaw"}},
                    {"editedAt": "2026-09-02T00:00:00Z", "editor": {"login": "QuantumExplorer"}},
                ]
            }
        }
    }
    stats = update_queue_comments(conn, cfg, gh)
    assert stats["promoted"] == 1
    promoted = conn.execute(
        "SELECT number, trigger, priority FROM heads WHERE priority=1"
    ).fetchall()
    assert [(p["number"], p["trigger"]) for p in promoted] == [(302, "priority_request")]
    ev = conn.execute("SELECT detail FROM events WHERE kind='head.priority_requested'").fetchone()[
        "detail"
    ]
    assert "QuantumExplorer" in ev
    assert queued_order(conn)[0]["number"] == 302
    latest_302 = [
        b for b in gh.gate_bodies if "⚡ Priority review — starts as soon as a slot frees" in b
    ]
    assert latest_302 and "- [ ]" not in latest_302[-1]
    # second pass: box already consumed, nothing promoted again
    assert update_queue_comments(conn, cfg, gh)["promoted"] == 0


def test_requeued_head_replaces_worker_written_gate_comment(cfg, conn, gh):
    _queue(conn, cfg, 1)
    with tx(conn):
        conn.execute("UPDATE heads SET eligible_at=queued_at")
    update_queue_comments(conn, cfg, gh)
    # the worker overwrote the comment ("failed"), then the head was requeued for retry
    gh.routes["repos/dashpay/platform/issues/comments/77"] = {
        "id": 77,
        "user": {"login": "thepastaclaw"},
        "body": "<!-- thepastaclaw-gate v1 -->\n⚠️ Automated review could not complete",
    }
    stats = update_queue_comments(conn, cfg, gh)
    assert stats["written"] == 1 and "Queued for automated review" in gh.gate_bodies[-1]


def _debounced(conn, cfg, gh, number=400, *, attempts=0):
    """An open PR whose new head waits out the push debounce, with an existing gate comment."""
    sha = f"{number:040x}"
    with tx(conn):
        conn.execute(
            "INSERT INTO prs (repo, number, head_sha, state, updated_at) "
            "VALUES ('dashpay/platform', ?, ?, 'open', 't')",
            (number, sha),
        )
        enqueue_head(conn, cfg, "dashpay/platform", number, sha, Trigger.NEW_PUSH)
        conn.execute("UPDATE heads SET attempts=? WHERE number=?", (attempts, number))
        kv_set(conn, f"queue.comment_id:dashpay/platform#{number}", str(number))
    ep = f"repos/dashpay/platform/issues/comments/{number}"
    gh.routes[ep] = {
        "id": number,
        "node_id": f"IC_{number}",
        "user": {"login": "thepastaclaw"},
        "body": "<!-- thepastaclaw-gate v1 -->\n✅ Final review complete (commit 11111111)",
    }
    return ep


def test_debounced_head_gets_one_stable_body(cfg, conn, gh):
    """The deferred pass and the queue pass both rendered a debounced head, one after the
    other, so its comment flipped between the two bodies on every pass."""
    from reviewsys.queue_status import DEFERRED_MARKER

    ep = _debounced(conn, cfg, gh)
    update_queue_comments(conn, cfg, gh)
    assert len(gh.gate_bodies) == 1 and DEFERRED_MARKER in gh.gate_bodies[0]
    assert "30-minute push debounce" in gh.routes[ep]["body"]
    for _ in range(3):
        stats = update_queue_comments(conn, cfg, gh)
        assert stats["written"] == 0, "an unchanged body is never rewritten"
    assert len(gh.gate_bodies) == 1
    assert "Queued for automated review" not in gh.routes[ep]["body"]


def test_debounced_head_boxes_still_work(cfg, conn, gh):
    gh.routes["graphql:userContentEdits"] = {
        "node": {"userContentEdits": {"nodes": [{"editedAt": "t", "editor": {"login": "knst"}}]}}
    }
    ep = _debounced(conn, cfg, gh, 401)
    update_queue_comments(conn, cfg, gh)
    gh.routes[ep]["body"] = gh.routes[ep]["body"].replace(
        "- [ ] **Request priority", "- [x] **Request priority"
    )
    assert update_queue_comments(conn, cfg, gh)["promoted"] == 1
    head = conn.execute("SELECT priority, trigger FROM heads WHERE number=401").fetchone()
    assert tuple(head) == (1, "priority_request")
    assert (
        gh.routes[ep]["body"]
        .splitlines()[1]
        .startswith("⚡ Priority review — starts as soon as a slot frees")
    )
    # the normal box: an explicit request, eligible at once but behind the priority head
    # (it used to be queued as a priority review: dashpay/dash-platform-website#249)
    ep = _debounced(conn, cfg, gh, 402)
    update_queue_comments(conn, cfg, gh)
    gh.routes[ep]["body"] = gh.routes[ep]["body"].replace(
        "- [ ] **Request normal", "- [x] **Request normal"
    )
    update_queue_comments(conn, cfg, gh)
    head = conn.execute(
        "SELECT priority, trigger, eligible_at <= ? FROM heads WHERE number=402", (now(),)
    ).fetchone()
    assert tuple(head) == (0, "normal_request", 1)
    update_queue_comments(conn, cfg, gh)
    assert (
        gh.routes[ep]["body"].splitlines()[1].startswith("🕓 Queued for automated review (commit")
    )
    assert "- [ ] **Request priority review**" in gh.routes[ep]["body"]


def test_normal_request_reopens_a_reviewed_commit_at_the_back_of_the_line(cfg, conn):
    sha = "4" * 40
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 403, sha, Trigger.NEW_PR)
        conn.execute("UPDATE heads SET status='failed' WHERE number=403")
        assert enqueue_head(conn, cfg, "dashpay/platform", 403, sha, Trigger.NORMAL_REQUEST) == (
            "requeued"
        )
    row = conn.execute("SELECT status, priority, trigger FROM heads WHERE number=403").fetchone()
    assert tuple(row) == ("queued", 0, "normal_request")
    # the priority box still moves it to the front
    with tx(conn):
        assert enqueue_head(conn, cfg, "dashpay/platform", 403, sha, Trigger.PRIORITY_REQUEST) == (
            "promoted"
        )
    assert conn.execute("SELECT priority FROM heads WHERE number=403").fetchone()[0] == 1


def test_review_requested_on_a_draft_runs_to_the_end(cfg, conn, gh):
    """dashpay/platform#5285 (2026-10-05): ticking a box on a draft queued a head, and the next
    ingest pass closed it as `pr_draft`, cancelling the run minutes after it started, then
    put the draft body back with the box unticked."""
    from reviewsys.queue_status import DEFERRED_MARKER, deferred_body

    sha = "5" * 40
    gh.open_prs["dashpay/platform"] = [pr(5285, sha, isDraft=True)]
    ingest_prs(conn, cfg, gh)
    with tx(conn):
        kv_set(conn, "queue.comment_id:dashpay/platform#5285", "5285")
    ep = "repos/dashpay/platform/issues/comments/5285"
    gh.routes[ep] = {
        "id": 5285,
        "user": {"login": "thepastaclaw"},
        "body": "<!-- thepastaclaw-gate v1 -->",
    }
    update_queue_comments(conn, cfg, gh)
    assert DEFERRED_MARKER in gh.routes[ep]["body"]
    gh.routes[ep]["body"] = gh.routes[ep]["body"].replace(
        "- [ ] **Request normal", "- [x] **Request normal"
    )
    update_queue_comments(conn, cfg, gh)
    head = conn.execute(
        "SELECT id, trigger, status, priority FROM heads WHERE number=5285"
    ).fetchone()
    assert (head["trigger"], head["status"], head["priority"]) == ("normal_request", "queued", 0)
    update_queue_comments(conn, cfg, gh)
    assert "Queued for automated review" in gh.routes[ep]["body"]

    # still a draft: the requested head survives ingest while queued and while running
    ingest_prs(conn, cfg, gh)
    assert (
        conn.execute("SELECT status FROM heads WHERE id=?", (head["id"],)).fetchone()[0] == "queued"
    )
    with tx(conn):
        conn.execute("UPDATE heads SET status='running' WHERE id=?", (head["id"],))
    ingest_prs(conn, cfg, gh)
    assert (
        conn.execute("SELECT status FROM heads WHERE id=?", (head["id"],)).fetchone()[0]
        == "running"
    )
    # the worker's progress body is left alone, and so is its final body
    gh.routes[ep]["body"] = "<!-- thepastaclaw-gate v1 -->\n🔍 Review in progress (commit 55555555)"
    assert update_queue_comments(conn, cfg, gh)["written"] == 0
    with tx(conn):
        conn.execute("UPDATE heads SET status='done' WHERE id=?", (head["id"],))
    gh.routes[ep]["body"] = (
        "<!-- thepastaclaw-gate v1 -->\n✅ Final review complete (commit 55555555)"
    )
    ingest_prs(conn, cfg, gh)
    assert update_queue_comments(conn, cfg, gh)["written"] == 0
    assert DEFERRED_MARKER not in gh.routes[ep]["body"]
    # a failed review gets the draft body back, so its boxes can retry it
    with tx(conn):
        conn.execute("UPDATE heads SET status='failed' WHERE id=?", (head["id"],))
    update_queue_comments(conn, cfg, gh)
    assert DEFERRED_MARKER in gh.routes[ep]["body"]
    # a box ticked on a draft body written before the fix still requests a review
    gh.routes[ep]["body"] = deferred_body(sha, draft=True, debounce_minutes=30).replace(
        "- [ ] **Request normal", "- [x] **Request normal"
    )
    update_queue_comments(conn, cfg, gh)
    assert tuple(
        conn.execute("SELECT status, priority FROM heads WHERE id=?", (head["id"],)).fetchone()
    ) == ("queued", 0)

    # a push to the draft: the requested head is for an old commit and the new one waits again
    gh.open_prs["dashpay/platform"] = [pr(5285, "6" * 40, isDraft=True)]
    ingest_prs(conn, cfg, gh)
    assert (
        conn.execute("SELECT status FROM heads WHERE id=?", (head["id"],)).fetchone()[0] == "closed"
    )
    update_queue_comments(conn, cfg, gh)
    assert DEFERRED_MARKER in gh.routes[ep]["body"]


def _draft_with_deferred_body(conn, cfg, gh, number, sha, **kw):
    """An open draft whose gate comment is the deferred ("not started: a draft") body."""
    from reviewsys.queue_status import DEFERRED_MARKER

    gh.open_prs["dashpay/platform"] = [pr(number, sha, isDraft=True, **kw)]
    ingest_prs(conn, cfg, gh)
    ep = _gate(conn, gh, number)
    gh.routes[ep]["node_id"] = f"IC_{number}"
    gh.routes["graphql:userContentEdits"] = {
        "node": {"userContentEdits": {"nodes": [{"editedAt": "t", "editor": {"login": "lklimek"}}]}}
    }
    update_queue_comments(conn, cfg, gh)
    assert DEFERRED_MARKER in gh.routes[ep]["body"]
    assert "because this PR is a draft" in gh.routes[ep]["body"]
    return ep


def _tick(gh, ep, box="priority"):
    gh.routes[ep]["body"] = gh.routes[ep]["body"].replace(
        f"- [ ] **Request {box}", f"- [x] **Request {box}"
    )


def test_priority_tick_on_a_draft_starts_a_review_at_once(cfg, conn, gh):
    """dashpay/platform#6349 (2026-11-26): lklimek ticked the priority box on a draft for early
    feedback. Pasta: "The manual tick on a PR always overrides". The draft gets a priority
    review now, ingest leaves it alone while the PR stays a draft, and marking the PR ready
    later neither duplicates nor delays it."""
    sha = "6349" * 10
    ep = _draft_with_deferred_body(conn, cfg, gh, 6349, sha)
    assert conn.execute("SELECT COUNT(*) FROM heads WHERE number=6349").fetchone()[0] == 0
    # unticked, a draft stays unreviewed, pass after pass
    for _ in range(2):
        ingest_prs(conn, cfg, gh)
        update_queue_comments(conn, cfg, gh)
    assert conn.execute("SELECT COUNT(*) FROM heads WHERE number=6349").fetchone()[0] == 0

    _tick(gh, ep)
    update_queue_comments(conn, cfg, gh)
    head = conn.execute("SELECT * FROM heads WHERE number=6349").fetchone()
    assert (head["status"], head["trigger"], head["priority"]) == (
        "queued",
        "priority_request",
        1,
    )
    assert parse_ts(head["eligible_at"]) <= parse_ts(now()), "no push debounce for a tick"
    assert queued_order(conn)[0]["id"] == head["id"]
    update_queue_comments(conn, cfg, gh)
    assert gh.routes[ep]["body"].splitlines()[1].startswith("⚡ Priority review")

    ingest_prs(conn, cfg, gh)  # still a draft
    assert conn.execute("SELECT status FROM heads WHERE id=?", (head["id"],)).fetchone()[0] == (
        "queued"
    )
    with tx(conn):
        conn.execute("UPDATE heads SET status='running' WHERE id=?", (head["id"],))
    ingest_prs(conn, cfg, gh)
    gh.open_prs["dashpay/platform"] = [pr(6349, sha)]  # marked ready mid-review
    stats = ingest_prs(conn, cfg, gh)["dashpay/platform"]
    assert stats["created"] == stats["requeued"] == stats["promoted"] == 0
    assert _heads(conn, 6349) == [(sha, "running")]


def test_tick_on_a_draft_whose_automatic_head_was_closed(cfg, conn, gh):
    """A PR turned draft closes its automatic head (`pr_draft`); a tick re-opens that head."""
    sha = "7" * 40
    gh.open_prs["dashpay/platform"] = [pr(24, sha)]
    ingest_prs(conn, cfg, gh)
    ep = _draft_with_deferred_body(conn, cfg, gh, 24, sha)
    assert _heads(conn, 24) == [(sha, "closed")]
    _tick(gh, ep)
    update_queue_comments(conn, cfg, gh)
    ingest_prs(conn, cfg, gh)
    row = conn.execute("SELECT status, trigger, priority FROM heads WHERE number=24").fetchone()
    assert tuple(row) == ("queued", "priority_request", 1)


def test_tick_on_a_labelled_draft_after_a_failed_review_retries_it(cfg, conn, gh):
    """A draft's failed review gets the draft body back so its boxes can retry it; once the PR
    also carried a skip label, neither pass read that body, so a tick there did nothing."""
    sha = "8" * 40
    ep = _draft_with_deferred_body(conn, cfg, gh, 25, sha)
    _tick(gh, ep, "normal")
    update_queue_comments(conn, cfg, gh)
    with tx(conn):
        conn.execute("UPDATE heads SET status='failed' WHERE number=25")
    gh.routes[ep]["body"] = "<!-- thepastaclaw-gate v1 -->\n⚠️ Automated review could not complete"
    update_queue_comments(conn, cfg, gh)
    assert "- [ ] **Request priority" in gh.routes[ep]["body"]
    gh.open_prs["dashpay/platform"] = [pr(25, sha, isDraft=True, labels=SKIP)]
    ingest_prs(conn, cfg, gh)
    _tick(gh, ep)
    update_queue_comments(conn, cfg, gh)
    row = conn.execute("SELECT status, trigger, priority FROM heads WHERE number=25").fetchone()
    assert tuple(row) == ("queued", "priority_request", 1)
    ingest_prs(conn, cfg, gh)
    assert conn.execute("SELECT status FROM heads WHERE number=25").fetchone()[0] == "queued"


def test_labelled_draft_after_a_failed_review_without_a_tick_is_noted(cfg, conn, gh):
    sha = "9" * 40
    ep = _draft_with_deferred_body(conn, cfg, gh, 26, sha)
    _tick(gh, ep, "normal")
    update_queue_comments(conn, cfg, gh)
    with tx(conn):
        conn.execute("UPDATE heads SET status='failed' WHERE number=26")
    gh.routes[ep]["body"] = "<!-- thepastaclaw-gate v1 -->\n⚠️ Automated review could not complete"
    update_queue_comments(conn, cfg, gh)
    gh.open_prs["dashpay/platform"] = [pr(26, sha, isDraft=True, labels=SKIP)]
    ingest_prs(conn, cfg, gh)
    update_queue_comments(conn, cfg, gh)
    assert SKIP_LABEL_NOTE in gh.routes[ep]["body"] and "Request" not in gh.routes[ep]["body"]
    assert _heads(conn, 26) == [(sha, "failed")]


def test_labelled_draft_keeps_the_body_of_its_failed_review(cfg, conn, gh):
    """A review asked for on a labelled draft failed: its comment says why, not "not reviewed"."""
    sha = "a" * 40
    ep = _draft_with_deferred_body(conn, cfg, gh, 27, sha)
    _tick(gh, ep)
    update_queue_comments(conn, cfg, gh)
    gh.open_prs["dashpay/platform"] = [pr(27, sha, isDraft=True, labels=SKIP)]
    failed = (
        "<!-- thepastaclaw-gate v1 -->\n⚠️ Automated review could not complete (commit aaaaaaaa)"
    )
    with tx(conn):
        conn.execute("UPDATE heads SET status='failed' WHERE number=27")
    gh.routes[ep]["body"] = failed
    for _ in range(2):
        ingest_prs(conn, cfg, gh)
        update_queue_comments(conn, cfg, gh)
    assert gh.routes[ep]["body"] == failed
    assert kv_get(conn, "queue.skip_noted:dashpay/platform#27") == sha


def test_automatic_head_on_a_pr_turned_draft_is_closed(cfg, conn, gh):
    gh.open_prs["dashpay/platform"] = [pr(7, "7" * 40)]
    ingest_prs(conn, cfg, gh)
    gh.open_prs["dashpay/platform"] = [pr(7, "7" * 40, isDraft=True)]
    assert ingest_prs(conn, cfg, gh)["dashpay/platform"]["draft"] == 1
    row = conn.execute("SELECT status, reason FROM heads WHERE number=7").fetchone()
    assert tuple(row) == ("closed", "pr_draft")


def _labels(*names):
    return {"nodes": [{"name": n} for n in names]}


SKIP = _labels("tag:skip-review")


def _gate(conn, gh, n):
    with tx(conn):
        kv_set(conn, f"queue.comment_id:dashpay/platform#{n}", str(n))
    ep = f"repos/dashpay/platform/issues/comments/{n}"
    gh.routes[ep] = {
        "id": n,
        "user": {"login": "thepastaclaw"},
        "body": "<!-- thepastaclaw-gate v1 -->\n🕓 queued",
    }
    return ep


def _heads(conn, n):
    return [
        tuple(r)
        for r in conn.execute("SELECT sha, status FROM heads WHERE number=? ORDER BY id", (n,))
    ]


def test_skip_label_closes_automatic_heads_until_removed(cfg, conn, gh):
    """dashpay/base-sdk marks PRs `tag:skip-review` that must not be reviewed agentically."""
    sha = "8" * 40
    # labelled from the start: no head at all
    gh.open_prs["dashpay/platform"] = [pr(8, sha, labels=_labels("topic:P2P", "tag:skip-review"))]
    ingest_prs(conn, cfg, gh)
    assert conn.execute("SELECT COUNT(*) FROM heads WHERE number=8").fetchone()[0] == 0
    # unlabelled: queued as a new PR
    gh.open_prs["dashpay/platform"] = [pr(8, sha)]
    assert ingest_prs(conn, cfg, gh)["dashpay/platform"]["created"] == 1
    # labelled while waiting (case-insensitive): closed, and its gate comment says why
    ep = _gate(conn, gh, 8)
    gh.open_prs["dashpay/platform"] = [pr(8, sha, labels=_labels("Tag:Skip-Review"))]
    assert ingest_prs(conn, cfg, gh)["dashpay/platform"]["skipped"] == 1
    row = conn.execute("SELECT status, reason FROM heads WHERE number=8").fetchone()
    assert tuple(row) == ("closed", "pr_skip_label")
    assert update_queue_comments(conn, cfg, gh)["written"] == 1
    assert SKIP_LABEL_NOTE in gh.routes[ep]["body"]
    assert "Request" not in gh.routes[ep]["body"]  # no boxes: the label asks for quiet
    # checked once per commit: later passes do not read the comment again
    calls = len(gh.calls)
    assert update_queue_comments(conn, cfg, gh)["written"] == 0
    assert not any(ep in " ".join(c) for c in gh.calls[calls:])
    # still labelled: nothing re-opens it
    assert ingest_prs(conn, cfg, gh)["dashpay/platform"]["requeued"] == 0
    assert conn.execute("SELECT status FROM heads WHERE number=8").fetchone()[0] == "closed"
    # label removed: the same commit is queued again, behind the push debounce
    gh.open_prs["dashpay/platform"] = [pr(8, sha)]
    assert ingest_prs(conn, cfg, gh)["dashpay/platform"]["requeued"] == 1
    row = conn.execute("SELECT * FROM heads WHERE number=8").fetchone()
    assert (row["status"], row["priority"], row["reason"]) == ("queued", 0, None)
    assert parse_ts(row["eligible_at"]) - parse_ts(row["queued_at"]) == timedelta(minutes=30)


def test_explicit_request_on_a_skip_labelled_pr_still_runs(cfg, conn, gh):
    sha = "9" * 40
    gh.open_prs["dashpay/platform"] = [pr(9, sha)]
    ingest_prs(conn, cfg, gh)
    gh.open_prs["dashpay/platform"] = [pr(9, sha, labels=SKIP)]
    ingest_prs(conn, cfg, gh)
    with tx(conn):
        res = enqueue_head(conn, cfg, "dashpay/platform", 9, sha, Trigger.MENTION)
    assert res == "requeued"
    row = conn.execute("SELECT status, priority, trigger FROM heads WHERE number=9").fetchone()
    assert tuple(row) == ("queued", 1, "mention")
    ingest_prs(conn, cfg, gh)  # the label stays; the request survives
    assert conn.execute("SELECT status FROM heads WHERE number=9").fetchone()[0] == "queued"


def test_labelled_draft_gets_no_comment(cfg, conn, gh):
    gh.open_prs["dashpay/platform"] = [pr(11, "a" * 40, isDraft=True, labels=SKIP)]
    ingest_prs(conn, cfg, gh)
    update_queue_comments(conn, cfg, gh)
    assert gh.issue_comments == [] and gh.gate_bodies == []


def test_relabel_on_the_same_commit_notes_again(cfg, conn, gh):
    sha = "b" * 40
    gh.open_prs["dashpay/platform"] = [pr(12, sha)]
    ingest_prs(conn, cfg, gh)
    ep = _gate(conn, gh, 12)
    for labels, noted in ((SKIP, True), (_labels(), False), (SKIP, True)):
        gh.open_prs["dashpay/platform"] = [pr(12, sha, labels=labels)]
        ingest_prs(conn, cfg, gh)
        update_queue_comments(conn, cfg, gh)
        assert (SKIP_LABEL_NOTE in gh.routes[ep]["body"]) is noted


def test_label_and_push_in_one_poll_notes_the_comment(cfg, conn, gh):
    gh.open_prs["dashpay/platform"] = [pr(13, "c" * 40)]
    ingest_prs(conn, cfg, gh)
    ep = _gate(conn, gh, 13)
    gh.open_prs["dashpay/platform"] = [pr(13, "d" * 40, labels=SKIP)]
    ingest_prs(conn, cfg, gh)
    assert _heads(conn, 13) == [("c" * 40, "superseded")]
    update_queue_comments(conn, cfg, gh)
    assert SKIP_LABEL_NOTE in gh.routes[ep]["body"] and "dddddddd" in gh.routes[ep]["body"]


def test_skip_label_ends_a_ci_check_and_refuses_new_ones(cfg, conn, gh):
    from reviewsys import ci

    sha = "e" * 40
    gh.open_prs["dashpay/platform"] = [pr(14, sha)]
    ingest_prs(conn, cfg, gh)
    with tx(conn):
        conn.execute("UPDATE heads SET status='done' WHERE number=14")
        assert enqueue_head(conn, cfg, "dashpay/platform", 14, sha, Trigger.CI_FAILED) == "requeued"
    gh.open_prs["dashpay/platform"] = [pr(14, sha, labels=SKIP)]
    ingest_prs(conn, cfg, gh)
    # back to the review it re-opened, not re-reviewed when the label goes
    assert _heads(conn, 14) == [(sha, "done")]
    assert ci.load(conn, "dashpay/platform", 14, sha) is None
    with tx(conn):
        assert enqueue_head(conn, cfg, "dashpay/platform", 14, sha, Trigger.CI_FAILED) == "noop"
    gh.open_prs["dashpay/platform"] = [pr(14, sha)]
    assert ingest_prs(conn, cfg, gh)["dashpay/platform"]["requeued"] == 0


def test_skip_label_ends_only_automatic_heads_of_other_commits(cfg, conn, gh):
    """A requested review of another commit runs on (the worker retires it if the commit is
    really stale); an automatic one is superseded as a push would."""
    gh.open_prs["dashpay/platform"] = [pr(15, "f" * 40)]
    ingest_prs(conn, cfg, gh)
    with tx(conn):  # as the router leaves it beside the automatic head (no supersede)
        conn.execute(
            "INSERT INTO heads (repo, number, sha, trigger, priority, status, queued_at, eligible_at) "
            "VALUES ('dashpay/platform', 15, ?, 'mention', 1, 'queued', ?, ?)",
            ("0" * 40, now(), now()),
        )
    gh.open_prs["dashpay/platform"] = [pr(15, "1" * 40, labels=SKIP)]
    ingest_prs(conn, cfg, gh)
    assert _heads(conn, 15) == [("f" * 40, "superseded"), ("0" * 40, "queued")]


def test_reopened_head_supersedes_the_other_commit(cfg, conn, gh):
    a, b = "2" * 40, "3" * 40
    gh.open_prs["dashpay/platform"] = [pr(16, a)]
    ingest_prs(conn, cfg, gh)
    gh.open_prs["dashpay/platform"] = [pr(16, a, labels=SKIP)]
    ingest_prs(conn, cfg, gh)
    gh.open_prs["dashpay/platform"] = [pr(16, b)]
    ingest_prs(conn, cfg, gh)
    gh.open_prs["dashpay/platform"] = [pr(16, a)]  # force-pushed back
    ingest_prs(conn, cfg, gh)
    assert _heads(conn, 16) == [(a, "queued"), (b, "superseded")]


def test_no_note_while_a_run_winds_down(cfg, conn, gh):
    sha = "4" * 40
    gh.open_prs["dashpay/platform"] = [pr(17, sha)]
    ingest_prs(conn, cfg, gh)
    ep = _gate(conn, gh, 17)
    with tx(conn):
        conn.execute("UPDATE heads SET status='running' WHERE number=17")
        head = conn.execute("SELECT id FROM heads WHERE number=17").fetchone()["id"]
        conn.execute(
            "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at) "
            "VALUES (?, 1, 'running', 't', ?, ?)",
            (head, now(), now()),
        )
    gh.open_prs["dashpay/platform"] = [pr(17, sha, labels=SKIP)]
    ingest_prs(conn, cfg, gh)
    assert _heads(conn, 17) == [(sha, "closed")]
    update_queue_comments(conn, cfg, gh)
    assert SKIP_LABEL_NOTE not in gh.routes[ep]["body"]
    with tx(conn):
        conn.execute("UPDATE runs SET status='cancelled' WHERE head_id=?", (head,))
    update_queue_comments(conn, cfg, gh)
    assert SKIP_LABEL_NOTE in gh.routes[ep]["body"]


def test_reply_on_a_labelled_pr_reviews_no_new_commit(cfg, conn, gh):
    """ "Fixed in abc" under a finding of an older commit is no request to review this one."""
    a, b = "5" * 40, "6" * 40
    gh.open_prs["dashpay/platform"] = [pr(18, a)]
    ingest_prs(conn, cfg, gh)
    with tx(conn):
        conn.execute("UPDATE heads SET status='done' WHERE number=18")
    gh.open_prs["dashpay/platform"] = [pr(18, b, labels=SKIP)]
    ingest_prs(conn, cfg, gh)
    with tx(conn):
        assert enqueue_head(conn, cfg, "dashpay/platform", 18, b, Trigger.REVIEW_REPLY) == "noop"
        # nor does one on a commit whose automatic head closed for another reason
        conn.execute(
            "UPDATE heads SET status='closed', reason='pr_draft' WHERE number=18 AND sha=?", (a,)
        )
        assert enqueue_head(conn, cfg, "dashpay/platform", 18, a, Trigger.REVIEW_REPLY) == "noop"
        conn.execute("UPDATE heads SET status='done', reason=NULL WHERE number=18 AND sha=?", (a,))
        # a reviewed commit still gets its conversation
        assert enqueue_head(conn, cfg, "dashpay/platform", 18, a, Trigger.REVIEW_REPLY) == (
            "requeued"
        )


def test_box_ticked_before_the_label_is_honoured_not_erased(cfg, conn, gh):
    sha = "7" * 40
    gh.open_prs["dashpay/platform"] = [pr(19, sha)]
    ingest_prs(conn, cfg, gh)
    ep = _gate(conn, gh, 19)
    gh.routes[ep]["body"] += "\n- [x] **Request normal review** — click when ready."
    gh.open_prs["dashpay/platform"] = [pr(19, sha, labels=SKIP)]
    ingest_prs(conn, cfg, gh)
    update_queue_comments(conn, cfg, gh)
    row = conn.execute("SELECT status, trigger FROM heads WHERE number=19").fetchone()
    assert tuple(row) == ("queued", "normal_request")
    assert SKIP_LABEL_NOTE not in gh.routes[ep]["body"]


def test_ticked_box_on_a_commit_a_force_push_brought_back_is_reviewed(cfg, conn, gh):
    """A head left superseded by a force-push back to this commit was once a no-op for the
    boxes: the tick was dropped and the label's note written over it."""
    sha = "c" * 40
    gh.open_prs["dashpay/platform"] = [pr(22, sha, labels=SKIP)]
    ingest_prs(conn, cfg, gh)
    with tx(conn):
        conn.execute(
            "INSERT INTO heads (repo, number, sha, trigger, priority, status, queued_at, "
            "eligible_at, superseded_at, finished_at, reason) VALUES ('dashpay/platform', 22, ?, "
            "'new_pr', 0, 'superseded', ?, ?, ?, ?, 'new head dddddddd')",
            (sha, now(), now(), now(), now()),
        )
    ep = _gate(conn, gh, 22)
    gh.routes[ep]["body"] += "\n- [x] **Request normal review** — click when ready."
    update_queue_comments(conn, cfg, gh)
    row = conn.execute("SELECT * FROM heads WHERE number=22").fetchone()
    assert (row["status"], row["trigger"], row["superseded_at"], row["reason"]) == (
        "queued",
        "normal_request",
        None,
        None,
    )
    assert SKIP_LABEL_NOTE not in gh.routes[ep]["body"]
    assert kv_get(conn, "queue.skip_noted:dashpay/platform#22") is None
    ingest_prs(conn, cfg, gh)  # the label stays; the request survives
    assert conn.execute("SELECT status FROM heads WHERE number=22").fetchone()[0] == "queued"


def test_only_a_box_reopens_a_superseded_head(cfg, conn):
    sha = "d" * 40
    with tx(conn):
        conn.execute(
            "INSERT INTO heads (repo, number, sha, trigger, priority, status, queued_at, eligible_at) "
            "VALUES ('dashpay/platform', 23, ?, 'new_pr', 0, 'superseded', ?, ?)",
            (sha, now(), now()),
        )
        for trigger in (Trigger.NEW_PUSH, Trigger.MENTION, Trigger.REVIEW_REPLY):
            assert enqueue_head(conn, cfg, "dashpay/platform", 23, sha, trigger) == "noop"
        assert enqueue_head(conn, cfg, "dashpay/platform", 23, sha, Trigger.PRIORITY_REQUEST) == (
            "requeued"
        )
    row = conn.execute("SELECT status, priority, trigger FROM heads WHERE number=23").fetchone()
    assert tuple(row) == ("queued", 1, "priority_request")


def test_lagging_poll_leaves_a_newer_requested_review_alone(cfg, conn, gh):
    """The router saw commit B over REST; the next GraphQL poll still reports A."""
    a, b = "8" * 40, "9" * 40
    gh.open_prs["dashpay/platform"] = [pr(20, a, labels=SKIP)]
    ingest_prs(conn, cfg, gh)
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 20, b, Trigger.MENTION)
    ingest_prs(conn, cfg, gh)
    assert _heads(conn, 20) == [(b, "queued")]


def test_no_ci_watch_on_a_labelled_pr(cfg, conn, gh):
    from reviewsys import ci

    gh.open_prs["dashpay/platform"] = [pr(21, "a" * 40, labels=SKIP)]
    ingest_prs(conn, cfg, gh)
    res = ci.register(
        conn, cfg, gh, repo="dashpay/platform", number=21, sha="a" * 40, run_id=1, known=[]
    )
    assert res == "skip label"


def test_skip_labels_config(tmp_path, skills_dir):
    from reviewsys import config as cfg_mod

    base = cfg_mod.DEFAULT_TOML.replace('skills = "~/Projects/skills"', f'skills = "{skills_dir}"')
    p = tmp_path / "c.toml"
    p.write_text(base)
    assert cfg_mod.load(p).skip_labels == ("tag:skip-review",)
    # a box config.toml predating the knob gets the default; a bare string is one label
    p.write_text(base.replace('skip_labels = ["tag:skip-review"]\n', ""))
    assert cfg_mod.load(p).skip_labels == ("tag:skip-review",)
    p.write_text(base.replace('skip_labels = ["tag:skip-review"]', 'skip_labels = "No-Bot"'))
    assert cfg_mod.load(p).skip_labels == ("no-bot",)
    p.write_text(base.replace('skip_labels = ["tag:skip-review"]', "skip_labels = []"))
    assert cfg_mod.load(p).skip_labels == ()


def test_head_in_retry_backoff_keeps_the_queue_body(cfg, conn, gh):
    """A retry backoff keeps the stable queue body while the head waits."""
    ep = _debounced(conn, cfg, gh, 403, attempts=1)
    update_queue_comments(conn, cfg, gh)
    update_queue_comments(conn, cfg, gh)
    assert len(gh.gate_bodies) == 1
    assert "Queued for automated review" in gh.routes[ep]["body"]


def test_queue_comment_writes_are_capped_per_pass(cfg, conn, gh, monkeypatch):
    from reviewsys import queue_status

    monkeypatch.setattr(queue_status, "MAX_WRITES_PER_PASS", 2)
    _queue(conn, cfg, 5)
    stats = queue_status.update_queue_comments(conn, cfg, gh)
    assert stats["written"] == 2 and stats["deferred"] == 3


def _reply_comment(
    cid, parent, *, login="knst", created="2026-09-08T20:17:36Z", repo="dashpay/platform", number=7
):
    return {
        "id": cid,
        "in_reply_to_id": parent,
        "user": {"login": login},
        "author_association": "COLLABORATOR",
        "body": "I disagree: the pool has no priorities.",
        "created_at": created,
        "updated_at": created,
        "url": f"https://api.github.com/repos/{repo}/pulls/comments/{cid}",
        "pull_request_url": f"https://api.github.com/repos/{repo}/pulls/{number}",
    }


def test_review_reply_under_bot_finding_queues_priority_head(cfg, conn, gh, notifier):
    from reviewsys.ingest import ingest_review_replies

    # we have posted on this repo before, so it is polled for replies
    with tx(conn):
        conn.execute(
            "INSERT INTO posted_findings (repo, number, hash, sha, review_id, posted_at) VALUES (?,?,?,?,?,?)",
            ("dashpay/platform", 7, "abc", "b" * 40, 1, "2026-09-07T00:00:00Z"),
        )
    reply = _reply_comment(2, 1)
    bot_reply = _reply_comment(4, 1, login="thepastaclaw")
    with tx(conn):
        kv_set(conn, "replies.cursor", "2026-09-08T00:00:00Z")
    gh.inline = [reply, bot_reply]
    gh.routes["repos/dashpay/platform/pulls/comments/1"] = {
        "user": {"login": "thepastaclaw"},
        "body": "<!-- thepastaclaw-review v1 finding=abc dedupe=x -->\n**🟡 Suggestion: T**\n\nbody",
    }
    gh.routes["repos/dashpay/platform/pulls/comments/2"] = reply
    assert ingest_review_replies(conn, cfg, gh) == 1  # bot reply skipped
    assert ingest_review_replies(conn, cfg, gh) == 0  # idempotent by comment id
    per_repo = kv_get(conn, "replies.cursor:dashpay/platform")
    assert per_repo and per_repo > "2026-09-08T00:00:00Z"  # cursor advanced, per repo
    row = conn.execute("SELECT kind, actor, number FROM inbox").fetchone()
    assert (row["kind"], row["actor"], row["number"]) == ("review_reply", "knst", 7)
    # no push since the review: the live head is the reviewed commit, the discussion moved
    _reviewed(conn, cfg, 7, "f" * 40)
    gh.pr = {**gh.pr, "head": {"sha": "f" * 40}}
    stats = route_inbox(conn, cfg, gh, notifier)
    assert stats["review_reply"] == 1
    head = conn.execute(
        "SELECT number, sha, status, trigger, priority, eligible_at, queued_at FROM heads"
    ).fetchone()
    # re-opened at the front of the line; with a final review standing, the worker answers
    # the threads in conversation mode (worker.run, `_standing_review`)
    assert tuple(head)[:5] == (7, "f" * 40, "queued", "review_reply", 1)
    assert head["eligible_at"] <= head["queued_at"]  # no debounce
    assert (
        conn.execute("SELECT COUNT(*) FROM events WHERE kind='head.review_reply'").fetchone()[0]
        == 1
    )


def _reviewed(conn, cfg, number, sha):
    """`sha` was reviewed: its head is done and a final review of it stands."""
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", number, sha, Trigger.NEW_PUSH)
        conn.execute("UPDATE heads SET status='done', finished_at='x' WHERE sha=?", (sha,))
        conn.execute(
            "INSERT INTO reviews (repo, number, sha, phase, github_review_id, event, posted_at) "
            "VALUES ('dashpay/platform',?,?,'final',1,'COMMENTED','2026-09-08T00:00:00Z')",
            (number, sha),
        )


def _reply_row(conn, cid, *, occurred, seen=None, number=7):
    with tx(conn):
        conn.execute(
            "INSERT INTO inbox (source_id, kind, repo, number, actor, body, occurred_at, seen_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                f"reply:{cid}",
                "review_reply",
                "dashpay/platform",
                number,
                "knst",
                f"https://api.github.com/repos/dashpay/platform/pulls/comments/{cid}",
                occurred,
                seen or occurred,
            ),
        )


def _route_reply_at(conn, cfg, gh, ts, source_id="reply:2"):
    from reviewsys.router import _mention_re, _route_row

    row = conn.execute("SELECT * FROM inbox WHERE source_id=?", (source_id,)).fetchone()
    return _route_row(conn, cfg, gh, row, meta=gh.pr, mention_re=_mention_re("thepastaclaw"), ts=ts)


def _finding_routes(gh, cid=2):
    gh.routes[f"repos/dashpay/platform/pulls/comments/{cid}"] = _reply_comment(cid, 1)
    gh.routes["repos/dashpay/platform/pulls/comments/1"] = {
        "user": {"login": "thepastaclaw"},
        "body": "<!-- thepastaclaw-review v1 finding=abc dedupe=x -->\n**T**",
    }


def test_review_reply_on_a_draft_is_read(cfg, conn, gh, notifier):
    """dashpay/dash#7638 (2026-11-27): the author answered a blocking finding on a draft and
    the reply was dropped, so the finding stood unanswered. An answer is an ask, and an ask
    runs on a draft like a ticked box; ingest leaves the head alone while the PR stays a draft."""
    from reviewsys.ingest import ingest_prs

    reply = _reply_comment(2, 1)
    gh.routes["repos/dashpay/platform/pulls/comments/1"] = {
        "user": {"login": "thepastaclaw"},
        "body": "<!-- thepastaclaw-review v1 finding=abc dedupe=x -->\n**🔴 Blocking: T**\n\nbody",
    }
    gh.routes["repos/dashpay/platform/pulls/comments/2"] = reply
    _reviewed(conn, cfg, 7, "f" * 40)
    gh.pr = {**gh.pr, "head": {"sha": "f" * 40}, "draft": True}
    with tx(conn):
        conn.execute(
            "INSERT INTO inbox (source_id, kind, repo, number, actor, body, occurred_at, seen_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (
                "reply:2",
                "review_reply",
                "dashpay/platform",
                7,
                "knst",
                "https://api.github.com/repos/dashpay/platform/pulls/comments/2",
                now(),
                now(),
            ),
        )
    assert route_inbox(conn, cfg, gh, notifier)["review_reply"] == 1
    gh.open_prs["dashpay/platform"] = [
        {
            "number": 7,
            "headRefOid": "f" * 40,
            "isDraft": True,
            "title": "PR 7",
            "updatedAt": "2026-11-27T00:00:00Z",
            "author": {"login": "lklimek"},
            "labels": {"nodes": []},
        }
    ]
    ingest_prs(conn, cfg, gh)  # still a draft: the asked-for head is not closed
    head = conn.execute("SELECT status, trigger FROM heads WHERE number=7 AND sha=?", ("f" * 40,))
    assert tuple(head.fetchone()) == ("queued", "review_reply")


def test_reply_after_a_push_leaves_that_push_to_its_own_review(cfg, conn, gh, notifier):
    """ "Fixed in <sha>" right after a push: the push is reviewed like any other (no priority,
    the debounce intact, still carryable) and its review reads and answers the reply; the row
    is done at once."""
    old, new = "e" * 40, "f" * 40
    _reviewed(conn, cfg, 7, old)
    _finding_routes(gh)
    gh.pr = {**gh.pr, "head": {"sha": new}}
    _reply_row(conn, 2, occurred="2026-09-08T20:17:36Z")

    # ingest has not seen the push yet: the reply queues it as ingest would
    assert _route_reply_at(conn, cfg, gh, "2026-09-08T20:18:00Z") == "review_reply"
    head = conn.execute(
        "SELECT status, trigger, priority, queued_at, eligible_at FROM heads WHERE sha=?", (new,)
    ).fetchone()
    assert tuple(head)[:3] == ("queued", "new_push", 0)
    assert parse_ts(head["eligible_at"]) - parse_ts(head["queued_at"]) == timedelta(minutes=30)
    row = conn.execute("SELECT handled_at, action FROM inbox").fetchone()
    assert row["handled_at"] and row["action"] == "review_reply_queued"
    ev = conn.execute("SELECT detail FROM events WHERE kind='head.review_reply'").fetchone()
    assert "on comment 1 -> queued" in ev["detail"]
    # ingest's own pass over the same push changes nothing
    with tx(conn):
        assert enqueue_head(conn, cfg, "dashpay/platform", 7, new, Trigger.NEW_PUSH) == "noop"

    # a second reply while the push waits in its debounce leaves it as it is
    _reply_row(conn, 3, occurred="2026-09-08T20:19:00Z")
    _finding_routes(gh, 3)
    assert _route_reply_at(conn, cfg, gh, "2026-09-08T20:20:00Z", "reply:3") == "review_reply"
    assert conn.execute(
        "SELECT trigger, priority, eligible_at FROM heads WHERE sha=?", (new,)
    ).fetchone()[:] == ("new_push", 0, head["eligible_at"])


def test_first_push_reply_queues_a_new_pr_head(cfg, conn, gh, notifier):
    """No head was ever queued for the PR (a finding of the legacy pipeline): its first head
    is a new PR, as ingest would make it."""
    _finding_routes(gh)
    gh.pr = {**gh.pr, "head": {"sha": "f" * 40}}
    _reply_row(conn, 2, occurred="2026-09-08T20:17:36Z")
    assert _route_reply_at(conn, cfg, gh, "2026-09-08T20:18:00Z") == "review_reply"
    assert conn.execute("SELECT trigger, priority FROM heads").fetchone()[:] == ("new_pr", 0)


def test_reply_on_a_repo_reviewed_only_on_request_still_asks(cfg, conn, gh, notifier):
    """A repo the bot does not watch (an ad hoc review) reviews no push on its own: a reply
    there is a request, at priority, as before."""
    repo = "dashpay/quorum-list-server"
    assert repo not in cfg.enabled_repos
    reply = _reply_comment(2, 1, repo=repo)
    gh.routes[f"repos/{repo}/pulls/comments/2"] = reply
    gh.routes[f"repos/{repo}/pulls/comments/1"] = {
        "user": {"login": "thepastaclaw"},
        "body": "<!-- thepastaclaw-review v1 finding=abc dedupe=x -->\n**T**",
    }
    gh.pr = {**gh.pr, "head": {"sha": "f" * 40}}
    with tx(conn):
        conn.execute(
            "INSERT INTO inbox (source_id, kind, repo, number, actor, body, occurred_at, seen_at) "
            "VALUES ('reply:2','review_reply',?,7,'knst',?,?,?)",
            (repo, reply["url"], reply["created_at"], reply["created_at"]),
        )
    assert route_inbox(conn, cfg, gh, notifier)["review_reply"] == 1
    head = conn.execute("SELECT trigger, priority FROM heads").fetchone()
    assert tuple(head) == ("review_reply", 1)


def test_reply_while_the_push_review_runs_waits_for_it(cfg, conn, gh, notifier):
    """The push's run may have read the threads before the reply: the row waits for the run
    (bounded, as for any running head) and then re-opens the commit like any reviewed one,
    whose conversation answers whatever the run did not."""
    old, new = "e" * 40, "f" * 40
    _reviewed(conn, cfg, 7, old)
    _finding_routes(gh)
    gh.pr = {**gh.pr, "head": {"sha": new}}
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 7, new, Trigger.NEW_PUSH)
        conn.execute("UPDATE heads SET status='running' WHERE sha=?", (new,))
    _reply_row(conn, 2, occurred="2026-09-08T20:17:36Z")
    assert _route_reply_at(conn, cfg, gh, "2026-09-08T20:18:00Z") == "review_reply_deferred"
    head = conn.execute("SELECT trigger, priority FROM heads WHERE sha=?", (new,)).fetchone()
    assert tuple(head) == ("new_push", 0)  # never promoted: its carry stays possible
    assert conn.execute("SELECT handled_at FROM inbox").fetchone()[0] is None
    _reviewed(conn, cfg, 7, new)
    assert _route_reply_at(conn, cfg, gh, "2026-09-08T20:40:00Z") == "review_reply"
    head = conn.execute("SELECT status, trigger, priority FROM heads WHERE sha=?", (new,))
    assert tuple(head.fetchone()) == ("queued", "review_reply", 1)


def test_review_reply_from_another_bot_never_queues_a_run(cfg, conn, gh, notifier):
    """CodeRabbit is trusted for mentions, but its reply under one of our findings is not a
    human waiting on an answer: ignored at the router, no head, no worktree."""
    reply = {**_reply_comment(2, 1), "user": {"login": "coderabbitai[bot]"}}
    with tx(conn):
        conn.execute(
            "INSERT INTO inbox (source_id, kind, repo, number, actor, body, occurred_at, seen_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                "reply:2",
                "review_reply",
                "dashpay/platform",
                7,
                "coderabbitai[bot]",
                reply["url"],
                reply["created_at"],
                reply["created_at"],
            ),
        )
    gh.routes["repos/dashpay/platform/pulls/comments/2"] = reply
    gh.routes["repos/dashpay/platform/pulls/comments/1"] = {
        "user": {"login": "thepastaclaw"},
        "body": "<!-- thepastaclaw-review v1 finding=abc -->",
    }
    stats = route_inbox(conn, cfg, gh, notifier)
    assert stats["review_reply"] == 0 and stats["ignored"] == 1
    assert conn.execute("SELECT COUNT(*) FROM heads").fetchone()[0] == 0


def test_review_reply_under_someone_elses_thread_is_ignored(cfg, conn, gh, notifier):
    reply = _reply_comment(2, 1)
    with tx(conn):
        conn.execute(
            "INSERT INTO inbox (source_id, kind, repo, number, actor, body, occurred_at, seen_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                "reply:2",
                "review_reply",
                "dashpay/platform",
                7,
                "knst",
                reply["url"],
                reply["created_at"],
                reply["created_at"],
            ),
        )
    gh.routes["repos/dashpay/platform/pulls/comments/2"] = reply
    gh.routes["repos/dashpay/platform/pulls/comments/1"] = {
        "user": {"login": "coderabbitai[bot]"},
        "body": "x",
    }
    stats = route_inbox(conn, cfg, gh, notifier)
    assert stats["review_reply"] == 0 and stats["ignored"] == 1
    assert conn.execute("SELECT COUNT(*) FROM heads").fetchone()[0] == 0


def test_mention_on_unlisted_repo_from_trusted_user_queues_adhoc_review(cfg, conn, gh, notifier):
    gh.notifications = [
        {
            "id": "9",
            "reason": "mention",
            "updated_at": "2026-09-09T01:00:00Z",
            "subject": {
                "type": "PullRequest",
                "url": "https://api.github.com/repos/dashpay/quorum-list-server/pulls/14",
                "latest_comment_url": "https://api.github.com/repos/dashpay/quorum-list-server/issues/comments/1",
            },
            "repository": {"full_name": "dashpay/quorum-list-server"},
        },
        {
            "id": "10",
            "reason": "mention",
            "updated_at": "2026-09-09T01:00:01Z",
            "subject": {
                "type": "PullRequest",
                "url": "https://api.github.com/repos/someone/else/pulls/3",
                "latest_comment_url": "https://api.github.com/repos/someone/else/issues/comments/2",
            },
            "repository": {"full_name": "someone/else"},
        },
    ]
    gh.routes["repos/dashpay/quorum-list-server/issues/comments/1"] = {
        "body": "@thepastaclaw review please",
        "user": {"login": "lklimek"},
        "author_association": "NONE",
    }
    gh.routes["repos/someone/else/issues/comments/2"] = {
        "body": "@thepastaclaw review",
        "user": {"login": "stranger"},
        "author_association": "NONE",
    }
    ingest_notifications(conn, cfg, gh)
    stats = route_inbox(conn, cfg, gh, notifier)
    assert stats["mention"] == 1 and stats["ignored"] == 1
    head = conn.execute("SELECT repo, number, trigger FROM heads").fetchone()
    assert (head["repo"], head["number"], head["trigger"]) == (
        "dashpay/quorum-list-server",
        14,
        "mention",
    )


def test_review_reply_reopens_a_finished_head_and_waits_for_a_running_one(cfg, conn, gh, notifier):
    sha = "f" * 40
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 7, sha, Trigger.NEW_PR)
        conn.execute("UPDATE heads SET status='done', finished_at='x', attempts=2")
    with tx(conn):
        assert (
            enqueue_head(conn, cfg, "dashpay/platform", 7, sha, Trigger.REVIEW_REPLY) == "requeued"
        )
    head = conn.execute(
        "SELECT status, priority, trigger, attempts, finished_at FROM heads"
    ).fetchone()
    assert tuple(head) == ("queued", 1, "review_reply", 0, None)
    assert conn.execute("SELECT COUNT(*) FROM events WHERE kind='head.requeued'").fetchone()[0] == 1
    # a reply while that head is running is deferred: the inbox row stays unhandled
    with tx(conn):
        conn.execute("UPDATE heads SET status='running'")
        conn.execute(
            "INSERT INTO inbox (source_id, kind, repo, number, actor, body, occurred_at, seen_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                "reply:2",
                "review_reply",
                "dashpay/platform",
                7,
                "knst",
                "https://api.github.com/repos/dashpay/platform/pulls/comments/2",
                now(),
                now(),
            ),
        )
    gh.routes["repos/dashpay/platform/pulls/comments/2"] = _reply_comment(2, 1)
    gh.routes["repos/dashpay/platform/pulls/comments/1"] = {
        "user": {"login": "thepastaclaw"},
        "body": "<!-- thepastaclaw-review v1 finding=abc dedupe=x -->\n**T**",
    }
    gh.pr = {**gh.pr, "head": {"sha": sha}}
    stats = route_inbox(conn, cfg, gh, notifier)
    assert stats["review_reply_deferred"] == 1
    assert conn.execute("SELECT handled_at FROM inbox").fetchone()[0] is None
    with tx(conn):
        conn.execute("UPDATE heads SET status='done'")
    stats = route_inbox(conn, cfg, gh, notifier)
    assert stats["review_reply"] == 1
    assert conn.execute("SELECT status, trigger FROM heads").fetchone()[:] == (
        "queued",
        "review_reply",
    )


def test_reply_fetch_failure_is_retried_not_dropped(cfg, conn, gh, notifier):
    with tx(conn):
        conn.execute(
            "INSERT INTO inbox (source_id, kind, repo, number, actor, body, occurred_at, seen_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                "reply:2",
                "review_reply",
                "dashpay/platform",
                7,
                "knst",
                "https://api.github.com/repos/dashpay/platform/pulls/comments/2",
                "t",
                "2026-09-08T20:18:00Z",
            ),
        )
    gh.routes["repos/dashpay/platform/pulls/comments/2"] = _reply_comment(2, 1)
    gh.pr = {**gh.pr, "head": {"sha": "f" * 40}}
    _reviewed(conn, cfg, 7, "f" * 40)
    # the thread-root lookup fails transiently: row stays unhandled
    gh.fail_next = ["502 bad gateway"]
    gh.routes["repos/dashpay/platform/pulls/comments/1"] = {
        "user": {"login": "thepastaclaw"},
        "body": "<!-- thepastaclaw-review v1 finding=abc -->",
    }
    import reviewsys.router as router_mod
    from reviewsys.router import _mention_re, _route_row

    row = conn.execute("SELECT * FROM inbox").fetchone()
    out = _route_row(
        conn,
        cfg,
        gh,
        row,
        meta=gh.pr,
        mention_re=_mention_re("thepastaclaw"),
        ts="2026-09-08T20:20:00Z",
    )
    assert out == "deferred"
    assert conn.execute("SELECT handled_at FROM inbox").fetchone()[0] is None
    # next tick succeeds
    out = _route_row(
        conn,
        cfg,
        gh,
        row,
        meta=gh.pr,
        mention_re=_mention_re("thepastaclaw"),
        ts="2026-09-08T20:22:00Z",
    )
    assert out == "review_reply"
    # a row that has waited longer than DEFER_MAX is given up on, with a distinct action
    with tx(conn):
        conn.execute(
            "INSERT INTO inbox (source_id, kind, repo, number, actor, body, occurred_at, seen_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                "reply:9",
                "review_reply",
                "dashpay/platform",
                7,
                "knst",
                "https://api.github.com/repos/dashpay/platform/pulls/comments/9",
                "t",
                "2026-09-08T00:00:00Z",
            ),
        )
    gh.fail_next = ["502 bad gateway"]
    row = conn.execute("SELECT * FROM inbox WHERE source_id='reply:9'").fetchone()
    out = _route_row(
        conn,
        cfg,
        gh,
        row,
        meta=gh.pr,
        mention_re=_mention_re("thepastaclaw"),
        ts="2026-09-08T20:22:00Z",
    )
    assert out == "ignored"
    assert (
        conn.execute("SELECT action FROM inbox WHERE source_id='reply:9'").fetchone()[0]
        == "ignored_unfetchable"
    )
    assert router_mod.DEFER_MAX.total_seconds() == 6 * 3600


def test_mention_on_disabled_repo_stays_blocked_even_for_trusted_user(cfg, conn, gh, notifier):
    gh.notifications = [
        {
            "id": "11",
            "reason": "mention",
            "updated_at": "2026-09-09T01:00:00Z",
            "subject": {
                "type": "PullRequest",
                "url": "https://api.github.com/repos/dashpay/grovedb/pulls/5",
                "latest_comment_url": "https://api.github.com/repos/dashpay/grovedb/issues/comments/3",
            },
            "repository": {"full_name": "dashpay/grovedb"},
        }
    ]
    gh.routes["repos/dashpay/grovedb/issues/comments/3"] = {
        "body": "@thepastaclaw review",
        "user": {"login": "QuantumExplorer"},
        "author_association": "MEMBER",
    }
    ingest_notifications(conn, cfg, gh)
    stats = route_inbox(conn, cfg, gh, notifier)
    assert stats["mention"] == 0 and stats["ignored"] == 1
    assert conn.execute("SELECT COUNT(*) FROM heads").fetchone()[0] == 0


def test_priority_author_prs_are_priority_without_debounce(cfg, conn, gh):
    gh.open_prs["dashpay/platform"] = [
        pr(60, "a" * 40, author={"login": "quantumexplorer"}),
        pr(61, "b" * 40),
    ]
    ingest_prs(conn, cfg, gh)
    rows = {r["number"]: r for r in conn.execute("SELECT * FROM heads")}
    assert rows[60]["priority"] == 1 and rows[60]["eligible_at"] == rows[60]["queued_at"]
    assert rows[61]["priority"] == 0 and rows[61]["eligible_at"] > rows[61]["queued_at"]
    # every later push is priority too
    gh.open_prs["dashpay/platform"] = [pr(60, "c" * 40, author={"login": "QuantumExplorer"})]
    ingest_prs(conn, cfg, gh)
    head = conn.execute("SELECT * FROM heads WHERE sha=?", ("c" * 40,)).fetchone()
    assert head["trigger"] == "new_push" and head["priority"] == 1
    assert head["eligible_at"] == head["queued_at"]


def test_priority_author_promotes_head_queued_before_the_knob(cfg, conn, gh):
    gh.open_prs["dashpay/platform"] = [pr(62, "a" * 40, author={"login": "QuantumExplorer"})]
    ingest_prs(conn, dataclasses.replace(cfg, priority_authors=()), gh)
    assert conn.execute("SELECT priority FROM heads").fetchone()["priority"] == 0
    stats = ingest_prs(conn, cfg, gh)["dashpay/platform"]
    assert stats["promoted"] == 1
    head = conn.execute("SELECT * FROM heads").fetchone()
    assert head["priority"] == 1 and parse_ts(head["eligible_at"]) <= parse_ts(now())
    # a running head is left alone
    with tx(conn):
        conn.execute("UPDATE heads SET status='running', priority=0")
    assert ingest_prs(conn, cfg, gh)["dashpay/platform"]["noop"] == 1
    assert conn.execute("SELECT priority FROM heads").fetchone()["priority"] == 0


FINDING_TITLE = "Accept CRLF in the embedded reference vectors"
DISPUTE = (
    f"> 🔴 Blocking: {FINDING_TITLE}\r\n> src/test/bls_reference_tests.cpp:46\r\n\r\n"
    "Not changing this: native MSVC builds are unsupported."
)


def _posted_summary_finding(conn, number=7, sha="f" * 40):
    """A finding published in the review body only (no inline thread to reply under)."""
    with tx(conn):
        hid = conn.execute(
            "SELECT id FROM heads WHERE repo='dashpay/platform' AND number=? AND sha=?",
            (number, sha),
        ).fetchone()[0]
        rid = conn.execute(
            "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at) "
            "VALUES (?, 1, 'done', 't', '2026-09-08T00:00:00Z', '2026-09-08T00:00:00Z')",
            (hid,),
        ).lastrowid
        conn.execute(
            "INSERT INTO findings (run_id, phase, stage, hash, severity, title, body) "
            "VALUES (?, 'final', 'posted', 'crlf', 'blocking', ?, 'b')",
            (rid, FINDING_TITLE),
        )
        conn.execute(
            "INSERT INTO posted_findings (repo, number, hash, sha, review_id, posted_at) "
            "VALUES ('dashpay/platform', ?, 'crlf', ?, 1, '2026-09-08T00:00:00Z')",
            (number, sha),
        )


def _issue_comment_row(conn, gh, source_id, cid, *, kind="comment", body=DISPUTE, **comment):
    url = f"https://api.github.com/repos/dashpay/platform/issues/comments/{cid}"
    gh.routes[f"repos/dashpay/platform/issues/comments/{cid}"] = {
        "id": cid,
        "body": body,
        "user": {"login": "knst"},
        "author_association": "MEMBER",
        **comment,
    }
    with tx(conn):
        conn.execute(
            "INSERT INTO inbox (source_id, kind, repo, number, actor, body, occurred_at, seen_at) "
            "VALUES (?,?,'dashpay/platform',7,NULL,?,'2026-09-08T12:00:00Z','2026-09-08T12:00:00Z')",
            (source_id, kind, url),
        )


def _action(conn, source_id):
    return conn.execute("SELECT action FROM inbox WHERE source_id=?", (source_id,)).fetchone()[0]


def test_top_level_dispute_of_a_summary_finding_reopens_the_reviewed_head(cfg, conn, gh, notifier):
    """A maintainer quotes a finding the review published only in its body and disputes it in
    a top-level comment: the reviewed head re-opens for a full same-sha re-review (the
    conversation lane answers threads only), once per comment."""
    sha = "f" * 40
    _reviewed(conn, cfg, 7, sha)
    _posted_summary_finding(conn)
    gh.pr = {**gh.pr, "head": {"sha": sha}}
    _issue_comment_row(conn, gh, "n:1", 100)
    assert _route_reply_at(conn, cfg, gh, "2026-09-08T12:01:00Z", "n:1") == "review_reply"
    assert _action(conn, "n:1") == "discussion_reply"
    head = conn.execute("SELECT status, trigger, priority FROM heads WHERE sha=?", (sha,))
    assert tuple(head.fetchone()) == ("queued", "manual", 1)
    ev = conn.execute("SELECT detail FROM events WHERE kind='head.review_reply'").fetchone()
    assert "knst answered a finding in PR comment 100" in ev["detail"]

    # GitHub notifies again for the same comment (an edit added an @mention): routed once
    _issue_comment_row(conn, gh, "n:2", 100, kind="mention")
    assert _route_reply_at(conn, cfg, gh, "2026-09-08T12:02:00Z", "n:2") == "ignored"


def test_a_hand_typed_quote_of_a_title_with_markup_still_matches(cfg, conn, gh, notifier):
    sha = "f" * 40
    _reviewed(conn, cfg, 7, sha)
    _posted_summary_finding(conn)
    with tx(conn):
        conn.execute("UPDATE findings SET title='Accept `CRLF` in the  reference vectors'")
    gh.pr = {**gh.pr, "head": {"sha": sha}}
    _issue_comment_row(conn, gh, "n:1", 100, body="> accept CRLF in the reference vectors\n\nno")
    assert _route_reply_at(conn, cfg, gh, "2026-09-08T12:01:00Z", "n:1") == "review_reply"


def test_top_level_dispute_after_a_push_leaves_that_push_to_its_own_review(cfg, conn, gh, notifier):
    old, new = "e" * 40, "f" * 40
    _reviewed(conn, cfg, 7, old)
    _posted_summary_finding(conn, sha=old)
    gh.pr = {**gh.pr, "head": {"sha": new}}
    _issue_comment_row(conn, gh, "n:1", 100)
    assert _route_reply_at(conn, cfg, gh, "2026-09-08T12:01:00Z", "n:1") == "review_reply"
    assert _action(conn, "n:1") == "discussion_reply_queued"
    head = conn.execute("SELECT trigger, priority FROM heads WHERE sha=?", (new,)).fetchone()
    assert tuple(head) == ("new_push", 0)


def test_top_level_comments_that_answer_no_finding_are_ignored(cfg, conn, gh, notifier):
    sha = "f" * 40
    _reviewed(conn, cfg, 7, sha)
    gh.pr = {**gh.pr, "head": {"sha": sha}}
    # no finding posted on the PR yet: a quote answers nothing
    _issue_comment_row(conn, gh, "n:0", 99)
    assert _route_reply_at(conn, cfg, gh, "2026-09-08T12:01:00Z", "n:0") == "ignored"
    _posted_summary_finding(conn)
    cases = {
        "n:1": dict(body="utACK, LGTM"),
        "n:6": dict(body="thanks @thepastaclaw"),
        "n:2": dict(body="> some other quote\n\nagreed"),
        "n:3": dict(author_association="NONE", user={"login": "stranger"}),
        "n:4": dict(user={"login": "thepastaclaw"}, author_association="COLLABORATOR"),
        "n:5": dict(user={"login": "coderabbitai[bot]"}, author_association="NONE"),
    }
    for i, (source_id, kw) in enumerate(cases.items()):
        _issue_comment_row(conn, gh, source_id, 101 + i, **kw)
        assert _route_reply_at(conn, cfg, gh, "2026-09-08T12:01:00Z", source_id) == "ignored"
    assert conn.execute("SELECT status FROM heads WHERE sha=?", (sha,)).fetchone()[0] == "done"


def _ask(conn, source, kind, number, body=None):
    with tx(conn):
        conn.execute(
            "INSERT INTO inbox (source_id, kind, repo, number, actor, body, occurred_at, seen_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (source, kind, "dashpay/platform", number, None, body, now(), now()),
        )


def _requeues(conn, number):
    return conn.execute(
        "SELECT COUNT(*) FROM events WHERE kind='head.requeued' AND number=?", (number,)
    ).fetchone()[0]


def _finish(conn, number):
    with tx(conn):
        conn.execute("UPDATE heads SET status='done' WHERE number=?", (number,))


def test_a_pending_review_request_rereviews_a_reviewed_commit(cfg, conn, gh, notifier):
    """dashpay/dash#8192 (2026-11-19): a maintainer requested a review from the bot on a commit
    it had reviewed a day earlier, and nothing happened."""
    sha = "8" * 40
    _reviewed(conn, cfg, 8192, sha)
    gh.pr = {**gh.pr, "head": {"sha": sha}, "requested_reviewers": [{"login": "ThePastaClaw"}]}
    _ask(conn, "n1", "review_requested", 8192)
    assert route_inbox(conn, cfg, gh, notifier)["review_requested"] == 1
    head = conn.execute("SELECT status, trigger, priority FROM heads WHERE number=8192").fetchone()
    assert tuple(head) == ("queued", "review_requested", 1)
    assert _requeues(conn, 8192) == 1


def test_a_repeated_request_notification_rereviews_nothing(cfg, conn, gh, notifier):
    """GitHub keeps the `review_requested` reason on every later update of the pull request.
    Once the bot's review cleared the request, those notifications ask for nothing."""
    sha = "8" * 40
    _reviewed(conn, cfg, 8135, sha)
    gh.pr = {**gh.pr, "head": {"sha": sha}, "requested_reviewers": [{"login": "UdjinM6"}]}
    for i in range(3):
        _ask(conn, f"n{i}", "review_requested", 8135)
    assert route_inbox(conn, cfg, gh, notifier)["review_requested"] == 3
    assert conn.execute("SELECT status FROM heads WHERE number=8135").fetchone()[0] == "done"
    assert _requeues(conn, 8135) == 0


def test_requested_rereviews_of_one_commit_are_bounded(cfg, conn, gh, notifier):
    from reviewsys import router

    sha = "8" * 40
    _reviewed(conn, cfg, 77, sha)
    gh.pr = {**gh.pr, "head": {"sha": sha}, "requested_reviewers": [{"login": "thepastaclaw"}]}
    for i in range(router.MAX_ASKED_REREVIEWS + 2):
        _ask(conn, f"n{i}", "review_requested", 77)
        route_inbox(conn, cfg, gh, notifier)
        _finish(conn, 77)
    assert _requeues(conn, 77) == router.MAX_ASKED_REREVIEWS


def test_a_trusted_mention_rereviews_a_reviewed_commit_once(cfg, conn, gh, notifier):
    sha = "9" * 40
    _reviewed(conn, cfg, 90, sha)
    gh.pr = {**gh.pr, "head": {"sha": sha}}
    url = "https://api.github.com/repos/dashpay/platform/issues/comments/901"
    gh.routes["repos/dashpay/platform/issues/comments/901"] = {
        "id": 901,
        "body": "@thepastaclaw review again please",
        "user": {"login": "knst"},
        "author_association": "MEMBER",
    }
    _ask(conn, "m1", "mention", 90, url)
    assert route_inbox(conn, cfg, gh, notifier)["mention"] == 1
    assert _requeues(conn, 90) == 1
    _finish(conn, 90)
    _ask(conn, "m2", "mention", 90, url)
    route_inbox(conn, cfg, gh, notifier)
    assert _requeues(conn, 90) == 1, "the same comment, notified again, asks once"


def test_an_untrusted_mention_does_not_rereview(cfg, conn, gh, notifier):
    sha = "9" * 40
    _reviewed(conn, cfg, 91, sha)
    gh.pr = {**gh.pr, "head": {"sha": sha}}
    gh.routes["repos/dashpay/platform/issues/comments/911"] = {
        "id": 911,
        "body": "@thepastaclaw review",
        "user": {"login": "passerby"},
        "author_association": "NONE",
    }
    _ask(
        conn,
        "m1",
        "mention",
        91,
        "https://api.github.com/repos/dashpay/platform/issues/comments/911",
    )
    route_inbox(conn, cfg, gh, notifier)
    assert conn.execute("SELECT status FROM heads WHERE number=91").fetchone()[0] == "done"
    assert _requeues(conn, 91) == 0


def _mention(
    conn, cfg, gh, notifier, number, cid, body, login, association="MEMBER", kind="issues"
):
    sha = "9" * 40
    _reviewed(conn, cfg, number, sha)
    gh.pr = {**gh.pr, "head": {"sha": sha}}
    path = f"repos/dashpay/platform/{kind}/comments/{cid}"
    gh.routes[path] = {
        "id": cid,
        "body": body,
        "user": {"login": login},
        "author_association": association,
    }
    _ask(conn, f"m{cid}", "mention", number, f"https://api.github.com/{path}")
    route_inbox(conn, cfg, gh, notifier)
    return _requeues(conn, number)


def test_a_bot_mentioning_the_reviewer_does_not_rereview(cfg, conn, gh, notifier):
    """The bot's own gate comment and CodeRabbit both write "@thepastaclaw review"."""
    assert _mention(conn, cfg, gh, notifier, 92, 921, "@thepastaclaw review", "coderabbitai") == 0
    assert (
        _mention(
            conn, cfg, gh, notifier, 93, 931, "@thepastaclaw review", "thepastaclaw", "COLLABORATOR"
        )
        == 0
    )
    assert _mention(conn, cfg, gh, notifier, 94, 941, "@thepastaclaw review", "helper[bot]") == 0


def test_a_mention_that_only_talks_about_a_review_does_not_rereview(cfg, conn, gh, notifier):
    said = "@thepastaclaw thanks for the review"
    assert _mention(conn, cfg, gh, notifier, 95, 951, said, "knst") == 0
    wrong = "@thepastaclaw this review finding is wrong"
    assert _mention(conn, cfg, gh, notifier, 96, 961, wrong, "knst") == 0
    asked = "@thepastaclaw, could you please re-review?"
    assert _mention(conn, cfg, gh, notifier, 97, 971, asked, "knst") == 1


def test_a_mention_in_a_finding_thread_does_not_rereview(cfg, conn, gh, notifier):
    """A reply under a finding is answered as a conversation, not by a full re-review."""
    said = "@thepastaclaw review this again, the guard is there"
    assert _mention(conn, cfg, gh, notifier, 98, 981, said, "knst", kind="pulls") == 0


def test_a_pending_request_takes_over_a_queued_ci_check(cfg, conn, gh, notifier):
    sha = "8" * 40
    _reviewed(conn, cfg, 99, sha)
    with tx(conn):
        conn.execute(
            "UPDATE heads SET status='queued', trigger=? WHERE number=99",
            (Trigger.CI_FAILED.value,),
        )
    gh.pr = {**gh.pr, "head": {"sha": sha}, "requested_reviewers": [{"login": "thepastaclaw"}]}
    _ask(conn, "n1", "review_requested", 99)
    route_inbox(conn, cfg, gh, notifier)
    head = conn.execute("SELECT status, trigger FROM heads WHERE number=99").fetchone()
    assert tuple(head) == ("queued", "review_requested")


def test_a_failed_rereview_is_not_restarted_by_later_activity(cfg, conn, gh, notifier):
    """A re-review that gave up leaves the commit's earlier review standing. The repeated
    `review_requested` notifications of later comments must not start it again."""
    sha = "8" * 40
    _reviewed(conn, cfg, 78, sha)
    gh.pr = {**gh.pr, "head": {"sha": sha}, "requested_reviewers": [{"login": "thepastaclaw"}]}
    _ask(conn, "n0", "review_requested", 78)
    route_inbox(conn, cfg, gh, notifier)
    assert _requeues(conn, 78) == 1
    with tx(conn):
        conn.execute("UPDATE heads SET status='failed' WHERE number=78")
    gh.pr = {**gh.pr, "requested_reviewers": []}
    for i in range(1, 4):
        _ask(conn, f"n{i}", "review_requested", 78)
    route_inbox(conn, cfg, gh, notifier)
    assert conn.execute("SELECT status FROM heads WHERE number=78").fetchone()[0] == "failed"
    assert _requeues(conn, 78) == 1


def test_a_failed_first_review_is_still_restarted_by_someone_asking(cfg, conn, gh, notifier):
    sha = "7" * 40
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 79, sha, Trigger.NEW_PUSH)
        conn.execute("UPDATE heads SET status='failed', finished_at='x' WHERE sha=?", (sha,))
    gh.pr = {**gh.pr, "head": {"sha": sha}, "requested_reviewers": []}
    _ask(conn, "n0", "review_requested", 79)
    route_inbox(conn, cfg, gh, notifier)
    assert conn.execute("SELECT status FROM heads WHERE number=79").fetchone()[0] == "queued"


def test_a_quoted_or_code_command_is_not_the_commenter_asking(cfg, conn, gh, notifier):
    quoted = "> @thepastaclaw review\n\nThanks, @thepastaclaw, the existing review is enough."
    assert _mention(conn, cfg, gh, notifier, 61, 611, quoted, "knst") == 0
    code = "To ask again, write `@thepastaclaw review`. Not needed here, @thepastaclaw."
    assert _mention(conn, cfg, gh, notifier, 62, 621, code, "knst") == 0
    fenced = "```\n@thepastaclaw review\n```\n@thepastaclaw fyi"
    assert _mention(conn, cfg, gh, notifier, 63, 631, fenced, "knst") == 0
    lazy = "> @thepastaclaw\nreview it again\n\nNo need, @thepastaclaw, it is fine."
    assert _mention(conn, cfg, gh, notifier, 66, 661, lazy, "knst") == 0
    carried = "> as I said\n@thepastaclaw review\n\nSee above, @thepastaclaw."
    assert _mention(conn, cfg, gh, notifier, 67, 671, carried, "knst") == 0
    own = "> the guard is there\n\n@thepastaclaw please re-review"
    assert _mention(conn, cfg, gh, notifier, 64, 641, own, "knst") == 1


def test_mentions_are_bounded_per_commit_and_leave_requests_their_own(cfg, conn, gh, notifier):
    from reviewsys import router

    sha = "9" * 40
    _reviewed(conn, cfg, 65, sha)
    gh.pr = {**gh.pr, "head": {"sha": sha}}
    for i in range(router.MAX_ASKED_REREVIEWS + 2):
        path = f"repos/dashpay/platform/issues/comments/{650 + i}"
        gh.routes[path] = {
            "id": 650 + i,
            "body": "@thepastaclaw review",
            "user": {"login": "knst"},
            "author_association": "CONTRIBUTOR",
        }
        _ask(conn, f"m{i}", "mention", 65, f"https://api.github.com/{path}")
        route_inbox(conn, cfg, gh, notifier)
        _finish(conn, 65)
    assert _requeues(conn, 65) == router.MAX_ASKED_REREVIEWS
    gh.pr = {**gh.pr, "requested_reviewers": [{"login": "thepastaclaw"}]}
    _ask(conn, "n1", "review_requested", 65)
    route_inbox(conn, cfg, gh, notifier)
    assert _requeues(conn, 65) == router.MAX_ASKED_REREVIEWS + 1, "a request has its own"


def test_promoting_an_already_queued_review_spends_nothing(cfg, conn, gh, notifier):
    sha = "8" * 40
    _reviewed(conn, cfg, 68, sha)
    with tx(conn):
        conn.execute(
            "UPDATE heads SET status='queued', priority=0, trigger=? WHERE number=68",
            (Trigger.NORMAL_REQUEST.value,),
        )
    gh.pr = {**gh.pr, "head": {"sha": sha}, "requested_reviewers": [{"login": "thepastaclaw"}]}
    _ask(conn, "n1", "review_requested", 68)
    route_inbox(conn, cfg, gh, notifier)
    assert conn.execute("SELECT priority FROM heads WHERE number=68").fetchone()[0] == 1
    assert (
        kv_get(conn, f"router.asked_rereviews:review_requested:dashpay/platform#68:{sha}") is None
    )


QEMU_TITLE = "The QEMU backport leaves the s390x profile without the target container"
BARE_DISPUTE = (
    "> 1 finding(s) not shown inline (the lines are not part of this PR's diff)\r\n"
    f"> 🔴 Blocking: {QEMU_TITLE}\r\nImpact: Selecting BUILD_TARGET=s390x ...\r\n"
    "ci/test/00_setup_env_s390x.sh:9-14\r\n\r\n@thepastaclaw \r\n\r\n"
    "The only usages of QEMU_USER_CMD+x are removed, there's nothing else to do complete "
    "backport. It's done in full\r\nthere's nothing to fix."
)


def _api_comment(cid, body, login, *, number=8159, assoc="MEMBER", created="2026-11-26T19:49:31Z"):
    return {
        "id": cid,
        "url": f"https://api.github.com/repos/dashpay/platform/issues/comments/{cid}",
        "html_url": f"https://github.com/dashpay/platform/pull/{number}#issuecomment-{cid}",
        "issue_url": f"https://api.github.com/repos/dashpay/platform/issues/{number}",
        "body": body,
        "user": {"login": login},
        "author_association": assoc,
        "created_at": created,
    }


def _reviewed_with_summary_finding(conn, cfg, gh, number, sha, title):
    gh.open_prs["dashpay/platform"] = [pr(number, sha, author={"login": "knst"})]
    ingest_prs(conn, cfg, gh)
    _reviewed(conn, cfg, number, sha)
    with tx(conn):
        conn.execute("UPDATE heads SET status='done' WHERE number=? AND sha=?", (number, sha))
    _posted_summary_finding(conn, number=number, sha=sha)
    with tx(conn):
        conn.execute("UPDATE findings SET title=? WHERE hash='crlf'", (title,))
    gh.pr = {**gh.pr, "head": {"sha": sha}}


def test_a_quoted_dispute_hidden_by_the_bots_next_comment_reaches_review(cfg, conn, gh, notifier):
    """dashpay/dash#8159 (2026-11-26): knst quoted a finding published only in the review body,
    added a bare @thepastaclaw and disputed it. 35 s later the job runner's acknowledgement (as
    thepastaclaw) became the PR's latest comment, the only one the notification names, so the
    router read the bot's own comment, ignored it, and never saw the dispute."""
    from reviewsys.ingest import ingest_pr_comments

    sha = "8" * 40
    _reviewed_with_summary_finding(conn, cfg, gh, 8159, sha, QEMU_TITLE)
    dispute = _api_comment(7201999720, BARE_DISPUTE, "knst")
    ack = _api_comment(
        7202009637,
        "@knst on it — I've queued this as job `cj-20261126-195005-ade5`",
        "thepastaclaw",
        assoc="COLLABORATOR",
        created="2026-11-26T19:50:06Z",
    )
    for c in (dispute, ack):
        gh.routes[c["url"].removeprefix("https://api.github.com/")] = c
    gh.notifications = [
        {
            "id": "27233384687",
            "reason": "mention",
            "updated_at": "2026-11-26T19:50:30Z",
            "subject": {
                "type": "PullRequest",
                "url": "https://api.github.com/repos/dashpay/platform/pulls/8159",
                "latest_comment_url": ack["url"],
            },
            "repository": {"full_name": "dashpay/platform"},
        }
    ]
    gh.issue_comments = [dispute, ack]
    with tx(conn):
        kv_set(conn, "comments.cursor:dashpay/platform", "2026-11-26T19:45:00Z")
    assert ingest_notifications(conn, cfg, gh) == 1
    assert ingest_pr_comments(conn, cfg, gh) == 1  # the bot's own acknowledgement is skipped
    assert ingest_pr_comments(conn, cfg, gh) == 0  # idempotent by comment id
    assert kv_get(conn, "comments.cursor:dashpay/platform") > "2026-11-26T19:45:00Z"
    row = conn.execute("SELECT kind FROM inbox WHERE source_id='comment:7201999720'").fetchone()
    assert row["kind"] == "mention"

    route_inbox(conn, cfg, gh, notifier)
    assert _action(conn, "27233384687:2026-11-26T19:50:30Z") == "ignored"  # the bot's comment
    assert _action(conn, "comment:7201999720") == "discussion_reply"
    head = conn.execute("SELECT status, trigger, priority FROM heads WHERE sha=?", (sha,))
    assert tuple(head.fetchone()) == ("queued", "manual", 1)  # a full same-sha re-review
    ev = conn.execute("SELECT detail FROM events WHERE kind='head.review_reply'").fetchone()
    assert "knst answered a finding in PR comment 7201999720" in ev["detail"]

    # the same comment named by a later notification is routed once
    with tx(conn):
        conn.execute("UPDATE heads SET status='done' WHERE sha=?", (sha,))
    _issue_comment_row(conn, gh, "n:again", 7201999720, kind="mention", body=BARE_DISPUTE)
    with tx(conn):
        conn.execute("UPDATE inbox SET number=8159 WHERE source_id='n:again'")
    route_inbox(conn, cfg, gh, notifier)
    assert conn.execute("SELECT status FROM heads WHERE sha=?", (sha,)).fetchone()[0] == "done"


def test_a_quoted_dispute_that_says_review_is_a_dispute_not_a_vague_mention(
    cfg, conn, gh, notifier
):
    """ "@bot this review is wrong" matches the mention rule but asks for no re-review of its
    own, so a quoted finding beside it was dropped as a vague mention of a reviewed commit."""
    sha = "f" * 40
    _reviewed(conn, cfg, 7, sha)
    _posted_summary_finding(conn)
    gh.pr = {**gh.pr, "head": {"sha": sha}}
    body = (
        f"> 🔴 Blocking: {FINDING_TITLE}\n\n@thepastaclaw this review is wrong: MSVC is unsupported"
    )
    _issue_comment_row(conn, gh, "n:1", 100, kind="mention", body=body)
    assert _route_reply_at(conn, cfg, gh, "2026-09-08T12:01:00Z", "n:1") == "review_reply"
    assert _action(conn, "n:1") == "discussion_reply"
    head = conn.execute("SELECT status, trigger FROM heads WHERE sha=?", (sha,)).fetchone()
    assert tuple(head) == ("queued", "manual")
    # a repeat of it is not a fresh explicit ask either: one re-review per comment
    _finish(conn, 7)
    _issue_comment_row(conn, gh, "n:2", 100, kind="mention", body=body)
    _route_reply_at(conn, cfg, gh, "2026-09-08T12:02:00Z", "n:2")
    assert conn.execute("SELECT status FROM heads WHERE sha=?", (sha,)).fetchone()[0] == "done"


def test_pr_comment_poll_keeps_only_comments_reviewsys_can_act_on(cfg, conn, gh):
    from reviewsys.ingest import ingest_pr_comments

    gh.open_prs["dashpay/platform"] = [
        pr(1, "a" * 40),
        pr(2, "b" * 40, author={"login": "thepastaclaw"}),
    ]
    ingest_prs(conn, cfg, gh)
    with tx(conn):
        kv_set(conn, "comments.cursor:dashpay/platform", "2026-11-26T19:00:00Z")
    issue = _api_comment(10, "@thepastaclaw hi", "knst", number=1)
    issue["html_url"] = "https://github.com/dashpay/platform/issues/1#issuecomment-10"
    edited = _api_comment(17, "> quote\n\n@thepastaclaw", "knst", number=1)
    edited["created_at"] = "2026-10-01T00:00:00Z"  # an old comment edited since
    gh.issue_comments = [
        _api_comment(11, "plain words, no quote", "knst", number=1),
        _api_comment(12, "@ThePastaClaw review", "knst", number=1),
        _api_comment(13, "mine", "thepastaclaw", number=1),
        _api_comment(14, "> q\n\non the bot's own PR", "knst", number=2),
        _api_comment(15, "> q\n\nunknown or closed PR", "knst", number=3),
        _api_comment(16, "> q\n\nwalkthrough", "coderabbitai[bot]", number=1),
        _api_comment(18, "> a quote\n\nanswered", "knst", number=1),
        edited,
        issue,
    ]
    assert ingest_pr_comments(conn, cfg, gh) == 2
    rows = conn.execute("SELECT source_id, kind, number FROM inbox ORDER BY source_id")
    assert [tuple(r) for r in rows] == [("comment:12", "mention", 1), ("comment:18", "comment", 1)]


def test_pr_comment_poll_seeds_each_repo_from_now(cfg, conn, gh):
    """A repo enabled after the first deploy must not replay everything since then."""
    from reviewsys.ingest import ingest_pr_comments

    with tx(conn):
        kv_set(conn, "comments.cursor", "2026-01-01T00:00:00Z")
    gh.issue_comments = []
    ingest_pr_comments(conn, cfg, gh)
    since = [c for c in gh.calls if any("issues/comments?" in a for a in c)]
    assert since and "since=2026-01-01" not in " ".join(since[0])


def test_an_explicit_review_ask_beside_a_quote_stays_a_review_ask(cfg, conn, gh, notifier):
    """ "@bot please review" in so many words is a review ask even when it quotes a finding:
    on a push nobody reviewed yet it is a priority review now, not the push's debounced one."""
    old, new = "e" * 40, "f" * 40
    _reviewed(conn, cfg, 7, old)
    _posted_summary_finding(conn, sha=old)
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 7, new, Trigger.NEW_PUSH)
    gh.pr = {**gh.pr, "head": {"sha": new}}
    body = f"> 🔴 Blocking: {FINDING_TITLE}\n\nfixed, @thepastaclaw please review again"
    _issue_comment_row(conn, gh, "n:1", 100, kind="mention", body=body)
    assert _route_reply_at(conn, cfg, gh, "2026-09-08T12:01:00Z", "n:1") == "mention"
    head = conn.execute("SELECT trigger, priority FROM heads WHERE sha=?", (new,)).fetchone()
    assert tuple(head) == ("mention", 1)


def test_a_dispute_already_routed_is_no_fresh_review_ask(cfg, conn, gh, notifier):
    """The same comment arriving as a plain comment row (routed as a dispute) and then as a
    mention row re-reviews the commit once."""
    sha = "f" * 40
    _reviewed(conn, cfg, 7, sha)
    _posted_summary_finding(conn)
    gh.pr = {**gh.pr, "head": {"sha": sha}}
    body = f"> 🔴 Blocking: {FINDING_TITLE}\n\nwrong. @thepastaclaw please review again"
    _issue_comment_row(conn, gh, "n:1", 100, kind="comment", body=body)
    assert _route_reply_at(conn, cfg, gh, "2026-09-08T12:01:00Z", "n:1") == "review_reply"
    assert _requeues(conn, 7) == 1
    _finish(conn, 7)
    _issue_comment_row(conn, gh, "n:2", 100, kind="mention", body=body)
    assert _route_reply_at(conn, cfg, gh, "2026-09-08T12:02:00Z", "n:2") == "mention"
    assert _requeues(conn, 7) == 1
