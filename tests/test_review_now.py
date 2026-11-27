"""The `pastaclaw:review-now` label: review the current head now, then take the label off."""

import pytest

from reviewsys import labels
from reviewsys.daemon import Daemon
from reviewsys.db import kv_get, now, parse_ts, tx
from reviewsys.ingest import REVIEW_NOW_LABEL, ingest_prs, review_now_key
from reviewsys.models import FailKind, ReviewError

REPO = "dashpay/platform"
A, B = "a" * 40, "b" * 40


def pr(number, sha, *names, draft=False):
    return {
        "number": number,
        "headRefOid": sha,
        "isDraft": draft,
        "title": f"PR {number}",
        "updatedAt": "2026-11-27T00:00:00Z",
        "author": {"login": "lklimek"},
        "labels": {"nodes": [{"name": n} for n in names]},
    }


def _head(conn, sha):
    return conn.execute("SELECT * FROM heads WHERE sha=?", (sha,)).fetchone()


def _now_eligible(row):
    return parse_ts(row["eligible_at"]) <= parse_ts(now())


def test_the_label_skips_the_push_debounce(cfg, conn, gh):
    """Without the label a new PR waits 30 minutes; with it, its head is asked for at once."""
    gh.open_prs[REPO] = [pr(1, A)]
    ingest_prs(conn, cfg, gh)
    assert not _now_eligible(_head(conn, A))
    gh.open_prs[REPO] = [pr(1, A, "Pastaclaw:Review-Now")]  # case-insensitive, as GitHub
    ingest_prs(conn, cfg, gh)
    head = _head(conn, A)
    assert (head["status"], head["trigger"], head["priority"]) == ("queued", "normal_request", 0)
    assert _now_eligible(head)
    assert kv_get(conn, review_now_key(REPO, 1)) == A


def test_one_label_is_one_review_of_the_head(cfg, conn, gh):
    """The label stays on while its review runs and until it is taken off: the finished head
    is never asked for again by the same label."""
    gh.open_prs[REPO] = [pr(2, A, REVIEW_NOW_LABEL)]
    ingest_prs(conn, cfg, gh)
    with tx(conn):
        conn.execute("UPDATE heads SET status='done' WHERE sha=?", (A,))
    gh.label_calls.clear()
    for _ in range(3):
        ingest_prs(conn, cfg, gh)
    assert _head(conn, A)["status"] == "done"
    assert (
        conn.execute("SELECT COUNT(*) FROM events WHERE kind='head.review_now'").fetchone()[0] == 1
    )


def test_the_label_overrides_draft_and_skip_label(cfg, conn, gh):
    """An explicit request, like the manual box (Pasta, 2026-11-26: "The manual tick on a PR
    always overrides")."""
    gh.open_prs[REPO] = [
        pr(3, A, REVIEW_NOW_LABEL, draft=True),
        pr(4, B, REVIEW_NOW_LABEL, "tag:skip-review"),
    ]
    ingest_prs(conn, cfg, gh)
    ingest_prs(conn, cfg, gh)  # the draft and label rules leave the requested heads alone
    assert _head(conn, A)["status"] == "queued" and _head(conn, B)["status"] == "queued"


def test_the_label_comes_off_once_the_review_is_done(cfg, conn, gh):
    gh.open_prs[REPO] = [pr(5, A, REVIEW_NOW_LABEL)]
    gh.labels = [REVIEW_NOW_LABEL, "pastaclaw:commented"]
    ingest_prs(conn, cfg, gh)
    assert labels.release_review_now(conn, gh) == 0  # queued: it stays
    with tx(conn):
        conn.execute("UPDATE heads SET status='running' WHERE sha=?", (A,))
    assert labels.release_review_now(conn, gh) == 0
    with tx(conn):
        conn.execute("UPDATE heads SET status='done' WHERE sha=?", (A,))
    assert labels.release_review_now(conn, gh) == 1
    assert gh.label_calls == [("DELETE", REVIEW_NOW_LABEL)]
    assert gh.labels == ["pastaclaw:commented"]
    assert kv_get(conn, review_now_key(REPO, 5)) is None
    # the next push waits out the debounce again
    gh.open_prs[REPO] = [pr(5, B)]
    ingest_prs(conn, cfg, gh)
    assert not _now_eligible(_head(conn, B))


def test_a_failed_review_releases_the_label_too(cfg, conn, gh):
    gh.open_prs[REPO] = [pr(6, A, REVIEW_NOW_LABEL)]
    gh.labels = [REVIEW_NOW_LABEL]
    ingest_prs(conn, cfg, gh)
    with tx(conn):
        conn.execute("UPDATE heads SET status='failed' WHERE sha=?", (A,))
    assert labels.release_review_now(conn, gh) == 1 and gh.labels == []


def test_a_push_under_the_label_is_reviewed_now_too(cfg, conn, gh):
    """The label asks for the head it is seen on: a push before its review ends is the new
    head, asked for at once, and the label comes off after that one."""
    gh.open_prs[REPO] = [pr(7, A, REVIEW_NOW_LABEL)]
    gh.labels = [REVIEW_NOW_LABEL]
    ingest_prs(conn, cfg, gh)
    gh.open_prs[REPO] = [pr(7, B, REVIEW_NOW_LABEL)]
    ingest_prs(conn, cfg, gh)
    assert _head(conn, A)["status"] == "superseded"
    head = _head(conn, B)
    assert head["trigger"] == "normal_request" and _now_eligible(head)
    assert labels.release_review_now(conn, gh) == 0 and gh.labels == [REVIEW_NOW_LABEL]


def test_a_label_that_cannot_be_removed_asks_nothing_more(cfg, conn, gh, monkeypatch):
    gh.open_prs[REPO] = [pr(8, A, REVIEW_NOW_LABEL)]
    ingest_prs(conn, cfg, gh)
    with tx(conn):
        conn.execute("UPDATE heads SET status='done' WHERE sha=?", (A,))

    def refuse(*a, **kw):
        raise ReviewError(FailKind.CONTRACT, "gh: Must have triage access to Issue. (HTTP 403)")

    monkeypatch.setattr(gh, "api", refuse)
    assert labels.release_review_now(conn, gh) == 0
    assert labels.release_review_now(conn, gh) == 0
    assert kv_get(conn, review_now_key(REPO, 8)) == A
    monkeypatch.undo()
    ingest_prs(conn, cfg, gh)
    assert _head(conn, A)["status"] == "done"
    # nor does it turn the debounce off for later pushes, and it is recorded once
    gh.open_prs[REPO] = [pr(8, B, REVIEW_NOW_LABEL)]
    ingest_prs(conn, cfg, gh)
    assert not _now_eligible(_head(conn, B))
    kinds = [r[0] for r in conn.execute("SELECT kind FROM events WHERE kind LIKE 'label.review%'")]
    assert kinds == ["label.review_now_unremovable"]


def test_verdict_label_sync_leaves_the_review_now_label_alone(gh):
    gh.labels = [REVIEW_NOW_LABEL, "pastaclaw:changes-requested"]
    assert labels.sync(gh, REPO, 9, "pastaclaw:approved")
    assert ("DELETE", REVIEW_NOW_LABEL) not in gh.label_calls
    assert sorted(gh.labels) == ["pastaclaw:approved", REVIEW_NOW_LABEL]


class _LabelApi:
    """gh for the repo label endpoints: `exists`, and what creating it answers."""

    def __init__(self, *, exists, create_error=None, kind=FailKind.CONTRACT):
        self.exists, self.create_error, self.created = exists, create_error, []
        self.kind = kind

    def api(self, ep, method="GET", body=None, **kw):
        if method == "GET":
            if not self.exists:
                raise ReviewError(FailKind.CONTRACT, "gh: Not Found (HTTP 404)")
            return {"name": REVIEW_NOW_LABEL}
        if self.create_error:
            raise ReviewError(self.kind, self.create_error)
        self.created.append((ep, body))
        return body


def test_the_label_is_created_where_the_bot_may(conn):
    gh = _LabelApi(exists=True)
    assert labels.ensure_review_now_label(conn, gh, REPO) and gh.created == []
    gh = _LabelApi(exists=False)
    assert labels.ensure_review_now_label(conn, gh, REPO)
    [(ep, body)] = gh.created
    assert ep == f"repos/{REPO}/labels" and body["name"] == REVIEW_NOW_LABEL
    assert "30-min" in body["description"]
    assert len(body["description"]) <= 100  # GitHub's limit: longer is a 422
    gh = _LabelApi(
        exists=False, create_error="gh: Must have admin rights to Repository. (HTTP 403)"
    )
    assert not labels.ensure_review_now_label(conn, gh, REPO)
    kinds = [r[0] for r in conn.execute("SELECT kind FROM events WHERE kind LIKE 'label.review%'")]
    assert kinds == ["label.review_now_created", "label.review_now_missing"]
    # a transient failure is no answer: raised, so the daemon tries again next pass
    gh = _LabelApi(exists=False, create_error="gh: HTTP 502", kind=FailKind.INFRA)
    with pytest.raises(ReviewError):
        labels.ensure_review_now_label(conn, gh, REPO)


def test_a_label_a_person_took_off_is_released(cfg, conn, gh):
    """GitHub answers 404 for a label no longer on the PR: released, not retried forever."""
    gh.open_prs[REPO] = [pr(10, A, REVIEW_NOW_LABEL)]
    gh.labels = [REVIEW_NOW_LABEL]
    ingest_prs(conn, cfg, gh)
    gh.labels = []  # taken off by hand mid-review
    gh.open_prs[REPO] = [pr(10, A)]
    with tx(conn):
        conn.execute("UPDATE heads SET status='done' WHERE sha=?", (A,))
    assert labels.release_review_now(conn, gh) == 1
    assert kv_get(conn, review_now_key(REPO, 10)) is None


def test_added_again_after_release_asks_again(cfg, conn, gh):
    """Like ticking the box again: another review of the same commit."""
    gh.open_prs[REPO] = [pr(11, A, REVIEW_NOW_LABEL)]
    gh.labels = [REVIEW_NOW_LABEL]
    ingest_prs(conn, cfg, gh)
    with tx(conn):
        conn.execute("UPDATE heads SET status='done' WHERE sha=?", (A,))
    labels.release_review_now(conn, gh)
    gh.labels = [REVIEW_NOW_LABEL]
    ingest_prs(conn, cfg, gh)
    assert _head(conn, A)["status"] == "queued"


def test_a_push_after_the_review_ended_waits_out_the_debounce(cfg, conn, gh):
    """The review ended but the label is not off yet (release runs after ingest, each pass):
    a push in that window is not asked for under the old label."""
    gh.open_prs[REPO] = [pr(12, A, REVIEW_NOW_LABEL)]
    gh.labels = [REVIEW_NOW_LABEL]
    ingest_prs(conn, cfg, gh)
    with tx(conn):
        conn.execute("UPDATE heads SET status='done' WHERE sha=?", (A,))
    gh.open_prs[REPO] = [pr(12, B, REVIEW_NOW_LABEL)]
    ingest_prs(conn, cfg, gh)
    assert not _now_eligible(_head(conn, B))
    assert labels.release_review_now(conn, gh) == 1 and gh.labels == []


def test_the_label_on_a_running_automatic_head_makes_it_requested(cfg, conn, gh):
    """Its run must review the commit, not carry an earlier review over it (worker._carry
    re-reads the trigger)."""
    gh.open_prs[REPO] = [pr(13, A)]
    ingest_prs(conn, cfg, gh)
    with tx(conn):
        conn.execute("UPDATE heads SET status='running' WHERE sha=?", (A,))
    gh.open_prs[REPO] = [pr(13, A, REVIEW_NOW_LABEL)]
    ingest_prs(conn, cfg, gh)
    head = _head(conn, A)
    assert (head["status"], head["trigger"]) == ("running", "normal_request")


def test_a_no_spawn_daemon_writes_no_labels(cfg, conn, gh, notifier):
    gh.open_prs[REPO] = [pr(14, A, REVIEW_NOW_LABEL)]
    gh.labels = [REVIEW_NOW_LABEL]
    calls = len(gh.calls)
    Daemon(cfg, conn, gh=gh, notifier=notifier, spawn=False).t_ingest()
    assert gh.label_calls == []
    assert not any(f"repos/{REPO}/labels" in " ".join(c) for c in gh.calls[calls:])
    assert kv_get(conn, review_now_key(REPO, 14)) == A  # still asked for: ingest is DB only
