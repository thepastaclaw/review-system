"""A deferral the bot itself accepts as reasonable does not keep blocking on the paperwork:
it finds or files the tracking issue and lifts the finding (dashpay/dash-evo-tool#1210,
#1161). Where it cannot file, it asks once, and a maintainer's answer without a link defers
it untracked."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from reviewsys import converse, publish, worker
from reviewsys.db import event, kv_get, kv_set, now, tx
from reviewsys.ingest import enqueue_head
from reviewsys.models import FailKind, ReviewError, RunStatus, Trigger
from reviewsys.scheduler import schedule
from tests.test_settle_disputes import _bot, _lanes, _review
from tests.test_worker_e2e import (  # fake_git: autouse, fakes the mirror and worktree
    HEAD,
    KNST,
    OUTSIDER,
    _blocking,
    _converse,
    _finding_thread,
    _full_review_deferral,
    _gql_comment,
    _issue_calls,
    _no_filing,
    _run_repo,
    _seed_prior,
    _standing_final,
    fake_git,  # noqa: F401
)

C = "c" * 40
REPO = "dashpay/platform"
FILED = "https://github.com/dashpay/platform/issues/9000"
IMPACT = "A refresh racing the roster check can drop a contact update."
AGENT = {**KNST, "author": "pr-author-agent", "body": "Out of scope for this PR; declining."}
CONCEDED = (
    "I checked v1.0-dev: refresh already used the same writer, and the new roster check "
    "doesn't widen the race. I'm comfortable deferring it. Please link a tracking issue."
)


def _root(h, title, *, impact=IMPACT):
    lines = [f"<!-- thepastaclaw-review v1 finding={h} dedupe=x -->", f"**🔴 Blocking: {title}**"]
    if impact:
        lines.append(f"**Impact:** {impact}")
    return "\n".join([*lines, "", "Two writers race on the contact store."])


def _thread_for(h, title, root_id, replies):
    t = _finding_thread(h, title, root_id, replies)
    t["comments"]["nodes"][0]["body"] = _root(h, title)
    return t


def _deferred(h, reply="Agreed, this predates the PR; deferring it."):
    return {"finding_hash": h, "status": "INTENTIONALLY_DEFERRED", "reply": reply}


def _events(conn, kind):
    return [r["detail"] for r in conn.execute("SELECT detail FROM events WHERE kind=?", (kind,))]


def _conceded(conn, h):
    row = conn.execute(
        "SELECT body FROM findings WHERE stage='conceded' AND hash=? ORDER BY id DESC", (h,)
    ).fetchone()
    return row["body"] if row else None


def _standing_with_lead(gh, *titles):
    fence = "```"
    lead = "\n".join(
        f"- **{t}** — `f.rs:11-12` ([thread](https://x/{i}))\n  Impact: {IMPACT}"
        for i, t in enumerate(titles)
    )
    items = "\n".join(f"- [BLOCKING] f.rs:11-12: {t}\n  Two writers race." for t in titles)
    gh.posted_reviews.append(
        {
            "id": 4242,
            "event": "REQUEST_CHANGES",
            "body": "<!-- thepastaclaw-review v1 -->\n"
            f"<!-- thepastaclaw-review-phase v1 phase=final sha={HEAD} policy=x -->\n"
            "## Re-review — Final validation — Phase 1 + Phase 2\n\n"
            f"{publish.LEAD_MARKER}\n**Changes requested: {len(titles)} blockers**\n{lead}\n\n"
            "Two races remain.\n\n"
            f"🔴 {len(titles)} blocking\n\n"
            "<details>\n<summary>🤖 Prompt for all review comments with AI agents</summary>\n\n"
            f"{fence}\nIn `f.rs`:\n{items}\n{fence}\n</details>",
        }
    )


def test_a_conceded_decline_is_filed_and_lifted_end_to_end(cfg, conn, gh, lanes):
    """dash-evo-tool#1210: the PR author's agent account (a COLLABORATOR) declined a blocker
    as out of scope; the lane agreed and asked for a link, and the blocker stood for three
    more rounds as "Awaiting tracking link"."""
    race, other = _blocking("Refresh races the roster check"), _blocking("Roster check skips l2")
    h = _seed_prior(conn, race, body="Two writers race on the contact store.")
    h2 = _seed_prior(conn, other, body="l2 is never checked.")
    gh.threads = [
        _thread_for(h, race["title"], 900, [AGENT]),
        _thread_for(h2, other["title"], 910, []),
    ]
    _standing_with_lead(gh, race["title"], other["title"])
    lane = {"finding_hash": h, "status": "DEFERRAL_NEEDS_TRACKING", "reply": CONCEDED}
    _converse(cfg, conn, gh, lanes, [lane])

    (issue,) = gh.created_issues
    assert issue["repo"] == REPO and issue["title"] == race["title"]
    body = issue["body"]
    assert body.startswith(
        "PastaClaw flagged this while reviewing https://github.com/dashpay/platform/pull/1, "
        "where it was deferred to a follow-up instead of being fixed in that pull request."
    )
    assert f"**Impact:** {IMPACT}" in body
    assert (
        f"**Where:** `f.rs:11-12` at `bbbbbbbbbbbb`: https://github.com/{REPO}/blob/"
        f"{'b' * 40}/f.rs#L11-L12"
    ) in body
    assert "> Two writers race on the contact store." in body
    assert f"- Review thread: https://github.com/{REPO}/pull/1#discussion_r900" in body
    assert "@" not in body and "at the request of" not in body

    reply = gh.replies[-1]["body"]
    assert converse.DEFERRAL_REPLY in reply and "Please link" not in reply
    assert reply.endswith(
        f"_Filed {FILED} for the follow-up. Deferred on a maintainer's call: this finding no "
        "longer blocks this pull request._"
    )
    root = gh.threads[0]["comments"]["nodes"][0]["body"]
    assert f"not blocking; tracked in {FILED})" in root
    assert _conceded(conn, h) == f"INTENTIONALLY_DEFERRED\n{FILED}"

    review = gh.posted_reviews[0]["body"]
    assert (
        "🔴 ~~2~~ 1 blocking" in review and "**Updated after discussion:** 1 finding(s)" in review
    )
    assert f"(tracked in {FILED})" in review
    assert "**Changes requested: ~~2~~ 1 blocker**" in review
    assert f"- ~~**{race['title']}** — `f.rs:11-12`" in review
    assert f"- **{other['title']}** — `f.rs:11-12`" in review
    assert len(gh.posted_reviews) == 1  # the other blocker still stands
    assert _events(conn, "defer.issue_auto_filed") == [f"{h}: {FILED}"]
    assert not kv_get(conn, f"defer.asked:{REPO}#1:{h}")

    # a retry (the worker died before the concession was recorded) files nothing again and
    # does not even search
    with tx(conn):
        conn.execute("DELETE FROM findings WHERE stage='conceded'")
    _converse(cfg, conn, gh, lanes, [lane], first=False)
    assert len(gh.created_issues) == 1 and len(_issue_calls(gh, "list")) == 1
    assert _conceded(conn, h) == f"INTENTIONALLY_DEFERRED\n{FILED}"


def test_an_open_issue_that_already_matches_is_reused(cfg, conn, gh, lanes):
    race = _blocking("Refresh races the roster check")
    h = _seed_prior(conn, race)
    gh.threads = [_thread_for(h, race["title"], 900, [AGENT])]
    _standing_final(gh)
    existing = f"https://github.com/{REPO}/issues/42"
    gh.open_issues = [
        {"number": 42, "title": "Refresh races with the roster check", "url": existing}
    ]
    _converse(cfg, conn, gh, lanes, [_deferred(h)])
    assert not gh.created_issues
    (search,) = _issue_calls(gh, "list")
    assert search[search.index("--state") + 1] == "all"
    assert f"_Tracked in {existing}. Deferred" in gh.replies[-1]["body"]
    assert _events(conn, "defer.issue_auto_linked") == [f"{h}: {existing}"]
    assert _conceded(conn, h) == f"INTENTIONALLY_DEFERRED\n{existing}"


def test_filing_unasked_needs_a_near_match_to_reuse_an_open_issue(cfg, conn, gh, lanes):
    """Nobody checks the match the bot makes by itself: a merely similar open issue (enough
    for the maintainer-asked path) is not taken as this finding's tracking."""
    race = _blocking("Refresh races the roster check")
    h = _seed_prior(conn, race)
    gh.threads = [_thread_for(h, race["title"], 900, [AGENT])]
    _standing_final(gh)
    similar = "Refresh races the roster"
    score = worker.title_similarity(race["title"], similar)
    assert worker.TRACKING_ISSUE_MATCH <= score < worker.TRACKING_ISSUE_MATCH_STRICT
    gh.open_issues = [{"number": 42, "title": similar, "url": f"https://x/{REPO}/42"}]
    _converse(cfg, conn, gh, lanes, [_deferred(h)])
    assert len(gh.created_issues) == 1 and FILED in gh.replies[-1]["body"]


def test_a_closed_issue_is_reused_only_on_a_near_match(cfg, conn, gh, lanes):
    """A maintainer who closed the follow-up as won't-fix does not get it filed again; a
    closed issue that only resembles it is no tracking."""
    race = _blocking("Refresh races the roster check")
    h = _seed_prior(conn, race)
    gh.threads = [_thread_for(h, race["title"], 900, [AGENT])]
    _standing_final(gh)
    closed = f"https://github.com/{REPO}/issues/41"
    gh.closed_issues = [
        {
            "number": 41,
            "title": "Refresh races the roster check",
            "url": closed,
            "stateReason": "NOT_PLANNED",
        }
    ]
    _converse(cfg, conn, gh, lanes, [_deferred(h)])
    assert not gh.created_issues
    assert f"_Tracked in {closed}." in gh.replies[-1]["body"]
    assert _events(conn, "defer.issue_auto_linked") == [f"{h}: {closed} (closed)"]


def test_an_issue_closed_as_completed_is_no_tracking(cfg, conn, gh, lanes):
    """A match closed as completed is a dead link for a defect still in the code."""
    race = _blocking("Refresh races the roster check")
    h = _seed_prior(conn, race)
    gh.threads = [_thread_for(h, race["title"], 900, [AGENT])]
    _standing_final(gh)
    gh.closed_issues = [
        {
            "number": 41,
            "title": race["title"],
            "url": f"https://github.com/{REPO}/issues/41",
            "stateReason": "COMPLETED",
        }
    ]
    _converse(cfg, conn, gh, lanes, [_deferred(h)])
    assert len(gh.created_issues) == 1 and FILED in gh.replies[-1]["body"]


def test_a_closed_issue_that_only_resembles_it_is_not_reused(cfg, conn, gh, lanes):
    race = _blocking("Refresh races the roster check")
    h = _seed_prior(conn, race)
    gh.threads = [_thread_for(h, race["title"], 900, [AGENT])]
    _standing_final(gh)
    gh.closed_issues = [
        {
            "number": 41,
            "title": "Refresh races the roster",
            "url": f"https://x/{REPO}/41",
            "stateReason": "NOT_PLANNED",
        }
    ]
    assert worker.title_similarity(race["title"], "Refresh races the roster") < (
        worker.TRACKING_ISSUE_MATCH_STRICT
    )
    _converse(cfg, conn, gh, lanes, [_deferred(h)])
    assert len(gh.created_issues) == 1 and FILED in gh.replies[-1]["body"]


def test_where_the_bot_cannot_file_it_asks_once_then_a_maintainer_defers_it_untracked(
    cfg, conn, gh, lanes
):
    """dash-evo-tool#1210: "Rejected as out of scope, with no tracking." had no way out."""
    cfg = _no_filing(cfg)
    race = _blocking("Refresh races the roster check")
    h = _seed_prior(conn, race)
    gh.threads = [_thread_for(h, race["title"], 900, [AGENT])]
    _standing_final(gh)
    _converse(cfg, conn, gh, lanes, [_deferred(h)])
    ask = gh.replies[-1]["body"]
    assert "Before I mark this deferred it needs a tracking issue" in ask
    assert "a maintainer can say so here" in ask and "I'll file it" not in ask
    assert not _issue_calls(gh, "list") and not _issue_calls(gh, "create")
    assert kv_get(conn, f"defer.asked:{REPO}#1:{h}")
    assert len(gh.posted_reviews) == 1 and _conceded(conn, h) is None

    rejected = {
        **KNST,
        "id": 902,
        "body": "Rejected as out of scope, with no tracking.",
        "created_at": "z",
    }
    gh.threads[0]["comments"]["nodes"] += [_gql_comment(_bot(ask)), _gql_comment(rejected)]
    _converse(cfg, conn, gh, lanes, [_deferred(h, "Still a fair deferral.")], first=False)
    reply = gh.replies[-1]["body"]
    assert converse.UNTRACKED_BLOCKER_REPLY in reply and "Before I mark" not in reply
    assert reply.endswith(
        "_Deferred on a maintainer's call: this finding no longer blocks this pull request._"
    )
    assert _conceded(conn, h) == "INTENTIONALLY_DEFERRED"
    assert len(gh.posted_reviews) == 2  # the only blocker lifted: the verdict moves
    assert not gh.created_issues
    assert len(gh.replies) == 2  # asked once, never again
    assert _events(conn, "defer.accepted_untracked")


def test_an_outside_contributors_decline_still_waits_for_a_maintainer(cfg, conn, gh, lanes):
    race = _blocking("Refresh races the roster check")
    h = _seed_prior(conn, race)
    gh.threads = [_thread_for(h, race["title"], 900, [{**OUTSIDER, "body": "out of scope"}])]
    _standing_final(gh)
    _converse(cfg, conn, gh, lanes, [_deferred(h)])
    assert "needs a maintainer's call" in gh.replies[-1]["body"]
    assert not _issue_calls(gh, "list") and not gh.created_issues
    assert len(gh.posted_reviews) == 1 and _conceded(conn, h) is None


def test_a_failed_create_keeps_the_blocker_and_asks_then_a_maintainer_can_defer_it(
    cfg, conn, gh, lanes
):
    race = _blocking("Refresh races the roster check")
    h = _seed_prior(conn, race)
    gh.threads = [_thread_for(h, race["title"], 900, [AGENT])]
    _standing_final(gh)
    gh.issue_create_error = "HTTP 502: Bad Gateway"
    _converse(cfg, conn, gh, lanes, [_deferred(h)])
    ask = gh.replies[-1]["body"]
    assert "could not confirm it was filed" in ask and ask.endswith(
        "until then the finding stands._"
    )
    assert "⏸️ **Needs a tracking issue**" in gh.threads[0]["comments"]["nodes"][0]["body"]
    assert len(gh.posted_reviews) == 1 and _conceded(conn, h) is None
    (fallback,) = _events(conn, "defer.issue_auto_fallback")
    assert fallback.startswith(f"{h}: create in {REPO}: ")
    assert gh.gate_bodies[-1].splitlines()[1].startswith("⛔ Final review complete — 1 blocking")

    # the create may have gone through: never retried, and no untracked lift on a failure a
    # later run may get past, whatever the maintainer says next
    gh.issue_create_error = ""
    again = {**KNST, "id": 902, "body": "We won't track this one.", "created_at": "z"}
    gh.threads[0]["comments"]["nodes"] += [_gql_comment(_bot(ask)), _gql_comment(again)]
    _converse(cfg, conn, gh, lanes, [_deferred(h)], first=False)
    assert len(_issue_calls(gh, "create")) == 1 and not gh.created_issues
    assert "could not confirm it was filed" in gh.replies[-1]["body"]
    assert _conceded(conn, h) is None and len(gh.posted_reviews) == 1

    # an hour on, the claim is abandoned: the search finds the issue the lost create made
    lost = f"https://github.com/{REPO}/issues/77"
    gh.open_issues = [{"number": 77, "title": race["title"], "url": lost}]
    with tx(conn):
        conn.execute(
            "UPDATE kv SET value='pending 2026-01-01T00:00:00Z' WHERE value LIKE 'pending%'"
        )
    third = {**KNST, "id": 903, "body": "Still deferred.", "created_at": "zz"}
    gh.threads[0]["comments"]["nodes"] += [
        _gql_comment(_bot("ok", id=952, at="z2")),
        _gql_comment(third),
    ]
    _converse(cfg, conn, gh, lanes, [_deferred(h)], first=False)
    assert len(_issue_calls(gh, "create")) == 1 and not gh.created_issues
    assert f"_Tracked in {lost}." in gh.replies[-1]["body"]
    assert _conceded(conn, h) == f"INTENTIONALLY_DEFERRED\n{lost}"
    (expired,) = _events(conn, "defer.issue_pending_expired")
    assert expired.startswith(f"{h}: pending 2026-01-01")


def _asked_and_answered(conn, gh, h, title):
    """A thread that had the one ask, answered by a maintainer declining to link an issue."""
    ask = f"<!-- thepastaclaw-thread-answer v1 sha={HEAD} reply=901 finding={h} -->\nLink one?"
    no = {**KNST, "id": 902, "body": "No tracking issue.", "created_at": "z"}
    gh.threads = [_thread_for(h, title, 900, [AGENT, _bot(ask), no])]
    _standing_final(gh)
    with tx(conn):
        kv_set(conn, f"defer.asked:{REPO}#1:{h}", json.dumps({"reply": "901", "sha": HEAD}))


@pytest.mark.parametrize(
    "error",
    [
        "HTTP 401: Bad credentials",
        "HTTP 403: You have triggered an abuse detection mechanism",
        "context deadline exceeded",
    ],
)
def test_a_create_failure_that_is_no_refusal_keeps_the_finding_held(cfg, conn, gh, lanes, error):
    """Bad credentials, abuse detection or a deadline (the create may have gone through) say
    nothing about whether the bot may file here: the claim stays pending, the finding held."""
    race = _blocking("Refresh races the roster check")
    h = _seed_prior(conn, race)
    _asked_and_answered(conn, gh, h, race["title"])
    gh.issue_create_error = error
    _converse(cfg, conn, gh, lanes, [_deferred(h)])
    assert _conceded(conn, h) is None and len(gh.posted_reviews) == 1
    assert "could not confirm it was filed" in gh.replies[-1]["body"]
    assert kv_get(conn, f"defer.issue:{REPO}#1:{h}").startswith(worker.TRACKING_ISSUE_PENDING)
    assert not _events(conn, "defer.accepted_untracked")


@pytest.mark.parametrize(
    "error",
    [
        "HTTP 410: Issues are disabled for this repo",
        "HTTP 403: Resource not accessible by integration",
    ],
)
def test_a_refused_create_defers_untracked_only_after_the_one_ask(cfg, conn, gh, lanes, error):
    race = _blocking("Refresh races the roster check")
    h = _seed_prior(conn, race)
    gh.threads = [_thread_for(h, race["title"], 900, [AGENT])]
    _standing_final(gh)
    gh.issue_create_error = error
    _converse(cfg, conn, gh, lanes, [_deferred(h)])
    assert "I can't file issues in `dashpay/platform`" in gh.replies[-1]["body"]
    assert _conceded(conn, h) is None and kv_get(conn, f"defer.issue:{REPO}#1:{h}") is None
    ask = gh.replies[-1]["body"]
    again = {**KNST, "id": 902, "body": "No tracking issue.", "created_at": "z"}
    gh.threads[0]["comments"]["nodes"] += [_gql_comment(_bot(ask)), _gql_comment(again)]
    _converse(cfg, conn, gh, lanes, [_deferred(h)], first=False)
    assert converse.UNTRACKED_BLOCKER_REPLY in gh.replies[-1]["body"]
    assert _conceded(conn, h) == "INTENTIONALLY_DEFERRED" and len(gh.posted_reviews) == 2


@pytest.mark.parametrize("failure", ["search_failed", "capped", "capped_daily"])
def test_a_failure_a_later_run_may_get_past_never_lifts_untracked(
    cfg, conn, gh, lanes, monkeypatch, failure
):
    """The thread had its one ask and a maintainer answered it, but the bot can still file
    here: it keeps the finding held and asks, so a later run files the issue."""
    race = _blocking("Refresh races the roster check")
    h = _seed_prior(conn, race)
    _asked_and_answered(conn, gh, h, race["title"])
    with tx(conn):
        if failure == "capped_daily":
            for _ in range(worker.AUTO_FILE_DAILY_LIMIT):
                event(conn, "defer.issue_auto_filed", repo=REPO, number=2, detail="x")
    if failure == "search_failed":

        def down(*a, **k):
            raise ReviewError(FailKind.INFRA, "HTTP 502")

        monkeypatch.setattr(worker.github, "open_issues_matching", down)
    elif failure == "capped":
        monkeypatch.setattr(worker, "AUTO_FILE_LIMIT", 0)
    _converse(cfg, conn, gh, lanes, [_deferred(h)])
    assert not gh.created_issues and _conceded(conn, h) is None
    assert len(gh.posted_reviews) == 1
    reply = gh.replies[-1]["body"]
    assert (
        "Link one, or say so and I'll file it" in reply
        and converse.UNTRACKED_BLOCKER_REPLY not in reply
    )
    assert not _events(conn, "defer.accepted_untracked")
    assert _events(conn, "defer.issue_auto_fallback")


def test_one_pass_files_at_most_the_cap_and_asks_for_the_rest(cfg, conn, gh, lanes):
    titles = [f"Race number {w} in the roster" for w in ("one", "two", "three", "four")]
    hashes = [_seed_prior(conn, _blocking(t)) for t in titles]
    gh.threads = [
        _thread_for(h, t, 900 + 10 * i, [{**AGENT, "id": 901 + 10 * i}])
        for i, (h, t) in enumerate(zip(hashes, titles, strict=True))
    ]
    _standing_final(gh)
    _converse(cfg, conn, gh, lanes, [_deferred(h) for h in hashes])
    assert len(gh.created_issues) == worker.AUTO_FILE_LIMIT == 3
    asked = [r["body"] for r in gh.replies if f"finding={hashes[3]}" in r["body"]]
    assert "Link one, or say so and I'll file it in `dashpay/platform`." in asked[0]
    assert [d.split(": ", 1)[1] for d in _events(conn, "defer.issue_auto_fallback")] == [
        "3 filed by this run already"
    ]
    assert _conceded(conn, hashes[3]) is None
    assert all(_conceded(conn, h) for h in hashes[:3])


def _dry(cfg, conn, gh, lanes, trigger):
    with tx(conn):
        enqueue_head(conn, cfg, REPO, 1, HEAD, trigger)
    (rid,) = schedule(conn, cfg, spawn=False)
    status = worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False, dry_run=True)
    assert status == RunStatus.DONE


def test_a_dry_run_never_searches_or_files(cfg, conn, gh, lanes):
    race = _blocking("Refresh races the roster check")
    h = _seed_prior(conn, race)
    gh.threads = [_thread_for(h, race["title"], 900, [AGENT])]
    _standing_final(gh)
    lanes.conversation = {"threads": [_deferred(h)]}
    with tx(conn):
        conn.execute("UPDATE heads SET sha=?, status='done'", (HEAD,))
    _dry(cfg, conn, gh, lanes, Trigger.REVIEW_REPLY)
    assert not _issue_calls(gh, "list") and not _issue_calls(gh, "create")
    assert not gh.replies


def test_a_full_review_dry_run_holds_instead_of_filing(cfg, conn, gh, lanes):
    _full_review_deferral(gh, conn, lanes, "out of scope here, a follow-up later")
    _dry(cfg, conn, gh, lanes, Trigger.REVIEW_REPLY)
    assert not _issue_calls(gh, "list") and not _issue_calls(gh, "create")
    assert _events(conn, "defer.needs_tracking")


def _ctx(conn, cfg, gh):
    return SimpleNamespace(
        conn=conn, cfg=cfg, gh=gh, repo=REPO, number=1, run_id=None, dry_run=False, auto_filed=0
    )


def test_the_same_finding_is_never_filed_twice_under_another_hash(conn, cfg, gh):
    """A re-hashed carry of the same finding (on the same thread) finds the issue already
    filed for it, and a create another run has claimed is never started again."""
    ctx = _ctx(conn, cfg, gh)
    thread = {"path": "f.rs", "title": "Refresh races the roster check", "comment_id": 900}
    req = converse.FileRequest(REPO, thread["title"], "body")
    url, filed, _ = worker._tracking_issue(ctx, "aaaaaaaaaaaa", req, thread, auto=True)
    assert (url, filed) == (FILED, True)
    retitled = {**thread, "title": "Refresh still races the roster check"}
    again = worker._tracking_issue(ctx, "bbbbbbbbbbbb", req, retitled, auto=True)
    assert again == (FILED, False, "") and len(gh.created_issues) == 1
    assert kv_get(conn, f"defer.issue:{REPO}#1:bbbbbbbbbbbb") == FILED
    other = {"path": "g.rs", "title": "Another race", "comment_id": 910}
    with tx(conn):
        kv_set(conn, f"defer.issue_thread:{REPO}#1:910", f"pending {now()}")
    claimed = worker._tracking_issue(
        ctx, "cccccccccccc", converse.FileRequest(REPO, "Another race", "b"), other, auto=True
    )
    assert claimed == ("", False, "pending") and len(gh.created_issues) == 1


def test_two_findings_sharing_a_file_and_title_never_share_an_issue(conn, cfg, gh):
    """Two "Unchecked unwrap" findings in one file are two follow-ups, even though a search
    finds the first one's issue for the second."""
    ctx = _ctx(conn, cfg, gh)
    first = {"path": "f.rs", "title": "Unchecked unwrap", "comment_id": 900}
    second = {**first, "comment_id": 910}
    req = converse.FileRequest(REPO, "Unchecked unwrap", "body")
    url, _, _ = worker._tracking_issue(ctx, "aaaaaaaaaaaa", req, first, auto=True)
    gh.open_issues = [{"number": 9000, "title": "Unchecked unwrap", "url": url}]
    url2, filed, _ = worker._tracking_issue(ctx, "bbbbbbbbbbbb", req, second, auto=True)
    assert filed and url2 != url and len(gh.created_issues) == 2


def test_an_issue_title_is_one_line_and_pages_nobody():
    assert (
        publish.issue_title("Ping @knst\nabout <!-- x -->")
        == "Ping @\u200bknst about &lt;!-- x -->"
    )


def test_a_full_review_files_the_issue_for_a_maintainers_deferral(cfg, conn, gh, lanes):
    """The verifier deferring a prior blocker on a maintainer's reply is the same deferral:
    filed and lifted, not carried as "Awaiting tracking link"."""
    bh, blocker = _full_review_deferral(gh, conn, lanes, "out of scope here, a follow-up later")
    _rid, status = _run_repo(cfg, conn, gh, lanes, REPO, 1, Trigger.REVIEW_REPLY)
    assert status == RunStatus.DONE
    (issue,) = gh.created_issues
    assert issue["title"] == blocker["title"] and "**Where:** `f.rs:11-12`" in issue["body"]
    review = gh.posted_reviews[-1]
    assert review["event"] != "REQUEST_CHANGES" and "phase=final" in review["body"]
    assert "Awaiting tracking link" not in review["body"]
    (answer,) = gh.replies
    assert (
        f"**Deferred** (re-reviewed at `aaaaaaaa`): This moves to a follow-up: I filed {FILED} "
        "to track it, so it no longer blocks this pull request."
    ) in answer["body"]
    root = gh.threads[0]["comments"]["nodes"][0]["body"]
    assert f"not blocking; tracked in {FILED})" in root
    assert _conceded(conn, bh) == f"INTENTIONALLY_DEFERRED\n{FILED}"
    assert _events(conn, "defer.issue_auto_filed") == [f"{bh}: {FILED}"]
    assert not _events(conn, "defer.needs_tracking")

    # the next push (an incremental round) shows it settled and keeps it deferred: nothing
    # is filed again, nothing is held
    gh.threads[0]["comments"]["nodes"].append(_gql_comment(_bot(answer["body"])))
    _review(cfg, conn, gh, lanes, C)
    assert len(gh.created_issues) == 1 and gh.posted_reviews[-1]["event"] != "REQUEST_CHANGES"
    prompt = [c.prompt for c in lanes.calls if c.role == "verifier"][-1]
    assert '"prior_status": "INTENTIONALLY_DEFERRED"' in prompt and FILED in prompt
    assert _conceded(conn, bh) == f"INTENTIONALLY_DEFERRED\n{FILED}"
    assert not _events(conn, "defer.needs_tracking")


def test_a_failed_upstream_claim_is_held_not_filed(cfg, conn, gh, lanes):
    """The backport path is unchanged: a deferral to an upstream change in a repository this
    one does not backport from is held as before, and no issue is filed for it."""
    _full_review_deferral(gh, conn, lanes, "a later backport brings the fix")
    for out in (lanes.reviewer["default"], lanes.verifier["default"]):
        out["prior_finding_reconciliation"][0].update(
            tracking="tokio-rs/tokio#28051", upstream_followup=True
        )
    _rid, status = _run_repo(cfg, conn, gh, lanes, REPO, 1, Trigger.REVIEW_REPLY)
    assert status == RunStatus.DONE
    assert gh.posted_reviews[-1]["event"] == "REQUEST_CHANGES"
    assert not _issue_calls(gh, "list") and not gh.created_issues

    # asked once and answered by a maintainer: still held, as before (never lifted untracked)
    ask = gh.replies[-1]["body"]
    no = {
        **KNST,
        "id": 902,
        "body": "No issue, it comes with the next backport.",
        "created_at": "z",
    }
    gh.threads[0]["comments"]["nodes"] += [_gql_comment(_bot(ask)), _gql_comment(no)]
    _review(cfg, conn, gh, lanes, C)
    assert gh.posted_reviews[-1]["event"] == "REQUEST_CHANGES"
    assert not gh.created_issues and not _events(conn, "defer.accepted_untracked")


def _stuck(conn, gh, h, title):
    """A thread left "Needs a tracking issue (not yet deferred)" before this change: a
    maintainer deferred it, the bot asked for a link, and nobody linked one."""
    ask = (
        f"<!-- thepastaclaw-thread-answer v1 sha={HEAD} reply=901 finding={h} -->\n"
        f"**Needs a tracking issue** (re-reviewed at `aaaaaaaa`): {converse.UNTRACKED_DEFERRAL_REASON}"
    )
    t = _thread_for(h, title, 900, [{**KNST, "body": "out of scope, later"}, _bot(ask)])
    root = t["comments"]["nodes"][0]
    root["body"] = publish.with_thread_status(
        root["body"], status=converse.DEFERRAL_NEEDS_TRACKING, head_sha=HEAD
    )
    gh.threads = [t]
    with tx(conn):
        kv_set(
            conn,
            f"defer.asked:{REPO}#1:{h}",
            json.dumps({"reply": "901", "standing": True, "sha": HEAD}),
        )


def test_a_thread_stuck_on_the_old_ask_is_unstuck_by_the_next_round(cfg, conn, gh, lanes):
    """The verifier, told to reconcile a reasonable untracked deferral INTENTIONALLY_DEFERRED,
    does so on the next push: the bot files the issue and the finding lifts."""
    blocker = _blocking("Refresh races the roster check")
    h = _seed_prior(conn, blocker)
    _stuck(conn, gh, h, blocker["title"])
    _lanes(lanes, h, blocker, "INTENTIONALLY_DEFERRED", reason="Still fine to defer.")
    _review(cfg, conn, gh, lanes, C)
    assert len(gh.created_issues) == 1
    review = gh.posted_reviews[-1]
    assert review["event"] != "REQUEST_CHANGES" and "Awaiting tracking link" not in review["body"]
    assert (
        f"**Deferred** (re-reviewed at `cccccccc`): This moves to a follow-up: I filed {FILED}"
        in (gh.replies[-1]["body"])
    )
    root = gh.threads[0]["comments"]["nodes"][0]["body"]
    assert "Needs a tracking issue" not in root and f"tracked in {FILED}" in root


@pytest.mark.parametrize("changed", [False, True, None])
def test_a_verifier_that_keeps_the_finding_is_never_overridden(
    cfg, conn, gh, lanes, monkeypatch, changed
):
    """STILL_VALID means the verifier did not judge it deferrable, whatever the conversation
    lane conceded earlier and whether or not the cited lines moved: nothing is filed, and
    the finding stands."""
    blocker = _blocking("Refresh races the roster check")
    h = _seed_prior(conn, blocker)
    _stuck(conn, gh, h, blocker["title"])
    monkeypatch.setattr(worker, "_lines_changed", lambda *a: changed)
    _lanes(lanes, h, blocker, "STILL_VALID", reason="The new writer widens the race.")
    _review(cfg, conn, gh, lanes, C)
    assert not gh.created_issues and not _issue_calls(gh, "list")
    assert gh.posted_reviews[-1]["event"] == "REQUEST_CHANGES"


def test_a_thread_stuck_on_the_old_ask_is_unstuck_by_a_reply(cfg, conn, gh, lanes):
    """dash-evo-tool#1161: "no tracking issue" left two blockers standing. The lane, told
    that a maintainer declining to link one is the deferral still standing, keeps it."""
    blocker = _blocking("Refresh races the roster check")
    h = _seed_prior(conn, blocker)
    _stuck(conn, gh, h, blocker["title"])
    no = {**KNST, "id": 902, "body": "no tracking issue", "created_at": "z"}
    gh.threads[0]["comments"]["nodes"].append(_gql_comment(no))
    _standing_final(gh)
    _converse(cfg, conn, gh, lanes, [_deferred(h, "Still reasonable to defer.")])
    assert len(gh.created_issues) == 1
    reply = gh.replies[-1]["body"]
    assert "Still reasonable to defer." in reply and f"_Filed {FILED} for the follow-up." in reply
    assert _conceded(conn, h) == f"INTENTIONALLY_DEFERRED\n{FILED}"
    assert len(gh.posted_reviews) == 2
    prompt = lanes.calls[-1].prompt
    assert '"tracking_asked": true' in prompt
    assert "only if you still judge it reasonable" in prompt


@pytest.mark.parametrize(
    "said", ["Looking again, this PR introduced it, please fix here", "@bob thoughts?", "thanks"]
)
def test_a_maintainers_other_reply_after_the_ask_is_no_deferral(cfg, conn, gh, lanes, said):
    blocker = _blocking("Refresh races the roster check")
    h = _seed_prior(conn, blocker)
    _stuck(conn, gh, h, blocker["title"])
    gh.threads[0]["comments"]["nodes"].append(
        _gql_comment({**KNST, "id": 902, "body": said, "created_at": "z"})
    )
    _standing_final(gh)
    _converse(cfg, conn, gh, lanes, [{"finding_hash": h, "status": "NO_REPLY", "reply": ""}])
    assert not gh.created_issues and not _issue_calls(gh, "list")
    assert _conceded(conn, h) is None and len(gh.posted_reviews) == 1


def test_an_ask_from_before_asks_were_recorded_counts_as_the_one_ask(cfg, conn, gh, lanes):
    """No `defer.asked` memo, but the thread shows our ask: a maintainer's answer to it is
    handled like any other (here: where the bot cannot file, deferred untracked)."""
    cfg = _no_filing(cfg)
    blocker = _blocking("Refresh races the roster check")
    h = _seed_prior(conn, blocker)
    _stuck(conn, gh, h, blocker["title"])
    with tx(conn):
        conn.execute("DELETE FROM kv WHERE key LIKE 'defer.asked:%'")
    no = {
        **KNST,
        "id": 902,
        "body": "Rejected as out of scope, with no tracking.",
        "created_at": "z",
    }
    gh.threads[0]["comments"]["nodes"].append(_gql_comment(no))
    _standing_final(gh)
    _converse(cfg, conn, gh, lanes, [_deferred(h)])
    assert converse.UNTRACKED_BLOCKER_REPLY in gh.replies[-1]["body"]
    assert _conceded(conn, h) == "INTENTIONALLY_DEFERRED"
