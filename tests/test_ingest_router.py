from datetime import timedelta

from reviewsys.db import kv_get, kv_set, now, parse_ts, tx
from reviewsys.ingest import enqueue_head, ingest_notifications, ingest_prs
from reviewsys.models import Trigger
from reviewsys.queue_status import PRIORITY_BOX, queued_order, update_queue_comments
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


def test_queue_comments_posted_with_position_eta_and_checkbox(cfg, conn, gh):
    _queue(conn, cfg, 3)
    with tx(conn):
        conn.execute("UPDATE heads SET eligible_at=queued_at")  # past debounce
    stats = update_queue_comments(conn, cfg, gh)
    assert stats == {"written": 3, "promoted": 0, "deferred": 0}
    assert len(gh.gate_bodies) == 3
    first = gh.gate_bodies[0]
    assert "thepastaclaw-gate" in first and "1st in line" in first
    # idle system (no active runs): the first two heads fit the two slots -> start now
    assert "estimated start in ~5 min" in first
    assert "3rd in line" in gh.gate_bodies[2] and "estimated start in ~2.0 h" in gh.gate_bodies[2]
    assert PRIORITY_BOX in first
    assert "of 3" not in first  # total omitted so bodies stay stable
    assert (
        conn.execute("SELECT COUNT(*) FROM kv WHERE key LIKE 'queue.comment_id:%'").fetchone()[0]
        == 3
    )
    # unchanged queue -> re-read (by cached id) but no writes
    gh.issue_comments = [{"id": 77, "user": {"login": "thepastaclaw"}, "body": gh.gate_bodies[-1]}]
    gh.routes["repos/dashpay/platform/issues/comments/77"] = gh.issue_comments[0]
    # PRs 300/301 differ from the shared body, so only those two are rewritten
    assert update_queue_comments(conn, cfg, gh)["written"] == 2


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
    latest_302 = [b for b in gh.gate_bodies if "⚡ Priority review — 1st in line" in b]
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
    gh.pr = {**gh.pr, "head": {"sha": "f" * 40}}
    stats = route_inbox(conn, cfg, gh, notifier)
    assert stats["review_reply"] == 1
    head = conn.execute(
        "SELECT number, sha, trigger, priority, eligible_at, queued_at FROM heads"
    ).fetchone()
    assert (head["number"], head["sha"], head["trigger"], head["priority"]) == (
        7,
        "f" * 40,
        "review_reply",
        1,
    )
    assert head["eligible_at"] <= head["queued_at"]  # no debounce
    assert (
        conn.execute("SELECT COUNT(*) FROM events WHERE kind='head.review_reply'").fetchone()[0]
        == 1
    )


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
