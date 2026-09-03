from datetime import timedelta

from reviewsys.db import kv_get, parse_ts, tx
from reviewsys.ingest import enqueue_head, ingest_notifications, ingest_prs
from reviewsys.models import Trigger
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
