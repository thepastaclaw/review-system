"""A clean re-review and the bot's own stale CHANGES_REQUESTED review (dashpay/dash#8198)."""

import dataclasses
from types import SimpleNamespace

from reviewsys import publish, worker

REPO = "dashpay/dash"
REVIEWS = f"repos/{REPO}/pulls/8198/reviews?per_page=100"
NEW_URL = "https://github.com/dashpay/dash/pull/8198#pullrequestreview-6085252688"


def _review(rid, state, at, login="thepastaclaw", sha="860ede23", body="b"):
    return {
        "id": rid,
        "node_id": f"PRR_{rid}",
        "user": {"login": login},
        "state": state,
        "submitted_at": at,
        "commit_id": sha + "0" * 32,
        "html_url": f"https://github.com/dashpay/dash/pull/8198#pullrequestreview-{rid}",
        "body": body,
    }


STALE = _review(5966020750, "CHANGES_REQUESTED", "2026-11-18T22:12:59Z")
POSTED = _review(6085252688, "COMMENTED", "2026-11-27T08:16:02Z", sha="5971656a")


def _ctx(conn, cfg, gh, *, disputed=(), degraded=False):
    prior = {h: SimpleNamespace(title=f"Finding {h}") for h in disputed}
    return SimpleNamespace(
        conn=conn,
        cfg=cfg,
        gh=gh,
        repo=REPO,
        number=8198,
        sha="5971656a" + "0" * 32,
        run_id=None,
        dry_run=False,
        is_degraded=degraded,
        disputed_blockers=len(disputed),
        disputed_blocker_hashes=list(disputed),
        prior_findings=prior,
        open_threads={h: {"comment_id": 4621575724} for h in disputed},
    )


def _result():
    return publish.PublishResult(
        posted=True,
        event="COMMENT",
        transport_event="COMMENT",
        body="b",
        review_id=POSTED["id"],
        review_url=NEW_URL,
    )


def _settle(ctx, blockers=0):
    worker._settle_stale_request(ctx, SimpleNamespace(blocker_count=blockers), _result())


def _events(conn):
    return [
        tuple(r)
        for r in conn.execute("SELECT kind, detail FROM events WHERE kind LIKE 'review.stale%'")
    ]


def test_stale_change_requests_are_the_ones_github_still_counts():
    cr1 = _review(1, "CHANGES_REQUESTED", "2026-11-01T00:00:00Z")
    ok = _review(2, "APPROVED", "2026-11-02T00:00:00Z")
    cr2 = _review(3, "CHANGES_REQUESTED", "2026-11-03T00:00:00Z")
    note = _review(4, "COMMENTED", "2026-11-04T00:00:00Z")
    human = _review(5, "CHANGES_REQUESTED", "2026-11-05T00:00:00Z", login="knst")
    gone = _review(6, "DISMISSED", "2026-11-06T00:00:00Z")
    stale = publish.stale_change_requests
    assert stale([note, cr2, ok, cr1, human], "ThePastaClaw") == [cr2]  # any order in
    assert stale([cr1, ok], "thepastaclaw") == []
    assert stale([cr1, cr2, gone], "thepastaclaw") == []
    assert stale([cr1, note], "thepastaclaw") == [cr1]


def test_a_clean_rereview_dismisses_the_stale_request(conn, cfg, gh):
    """No blocker open and none disputed: the earlier CHANGES_REQUESTED is dismissed, its
    message linking the clean re-review."""
    gh.routes[REVIEWS] = [STALE, POSTED]
    _settle(_ctx(conn, cfg, gh))
    assert len(gh.dismissals) == 1
    node, message = gh.dismissals[0]
    assert node == "PRR_5966020750"
    assert NEW_URL in message and "5971656a" in message and "no blocking finding" in message
    assert gh.review_edits == []
    assert _events(conn) == [("review.stale_request_dismissed", "5966020750")]


def test_only_disputed_blockers_left_says_a_maintainer_can_dismiss(conn, cfg, gh):
    """The #8198 case: nothing blocks but a disputed blocker. No dismissal; the re-review says
    the earlier review still shows, that a maintainer can dismiss it, and links the dispute."""
    gh.routes[REVIEWS] = [STALE, POSTED]
    ctx = _ctx(conn, cfg, gh, disputed=("aaaaaaaaaaaa",))
    _settle(ctx)
    assert gh.dismissals == []
    assert len(gh.review_edits) == 1
    rid, body = gh.review_edits[0]
    assert rid == POSTED["id"] and body.endswith("\nb")
    assert publish.STALE_REQUEST_NOTE_MARKER in body
    assert "GitHub still shows changes requested" in body
    assert "a maintainer can dismiss" in body
    assert STALE["html_url"] in body
    assert (
        "[Finding aaaaaaaaaaaa](https://github.com/dashpay/dash/pull/8198#discussion_r4621575724)"
        in body
    )
    assert _events(conn) == [("review.stale_request_noted", "1 left to maintainers")]

    # a retry of the round, whose review already carries the note, says it once
    gh.routes[REVIEWS] = [STALE, {**POSTED, "body": body}]
    _settle(ctx)
    assert len(gh.review_edits) == 1


def test_disputed_blockers_dismiss_too_when_configured(conn, cfg, gh):
    """`dismiss_stale_when_disputed` (Pasta's pending call) turns the note into a dismissal."""
    gh.routes[REVIEWS] = [STALE, POSTED]
    cfg = dataclasses.replace(cfg, dismiss_stale_when_disputed=True)
    _settle(_ctx(conn, cfg, gh, disputed=("aaaaaaaaaaaa",)))
    assert len(gh.dismissals) == 1 and "1 disputed blocker(s)" in gh.dismissals[0][1]
    assert gh.review_edits == []


def test_a_review_that_still_blocks_leaves_the_request_alone(conn, cfg, gh):
    gh.routes[REVIEWS] = [STALE, POSTED]
    _settle(_ctx(conn, cfg, gh), blockers=1)
    assert gh.dismissals == [] and gh.review_edits == [] and _events(conn) == []


def test_nothing_to_do_without_a_standing_request_or_on_a_degraded_review(conn, cfg, gh):
    approved = _review(6000000000, "APPROVED", "2026-11-20T00:00:00Z")
    gh.routes[REVIEWS] = [STALE, approved, POSTED]
    _settle(_ctx(conn, cfg, gh))
    gh.routes[REVIEWS] = [STALE, POSTED]
    _settle(_ctx(conn, cfg, gh, degraded=True))  # stand-ins never overrule a full review
    assert gh.dismissals == [] and gh.review_edits == [] and _events(conn) == []


def test_a_refused_dismissal_is_recorded_not_raised(conn, cfg, gh):
    gh.routes[REVIEWS] = [STALE, POSTED]
    gh.dismiss_error = "gh: Resource not accessible by integration"
    _settle(_ctx(conn, cfg, gh))
    [(kind, detail)] = _events(conn)
    assert kind == "review.stale_request_failed" and "not accessible" in detail


def test_policy_v12_skips_and_says_so(conn, cfg, gh):
    gh.routes[REVIEWS] = [STALE, POSTED]
    cfg = SimpleNamespace(
        bot_login=cfg.bot_login,
        dismiss_stale_when_disputed=False,
        policy=SimpleNamespace(verifier_v12=True),
    )
    _settle(_ctx(conn, cfg, gh))
    assert gh.dismissals == [] and gh.review_edits == []
    assert _events(conn)[0][0] == "review.stale_request_skipped"
