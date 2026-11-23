"""Route inbox rows to actions.

- `review_requested` on a watched repo  -> priority head at the live PR head
- `mention` whose comment says `@<bot> review` -> priority head (any repo the bot can
  read: repos without a skills entry get an ad hoc review)
- either of those on a commit already reviewed -> a fresh review of it, when the ask is
  explicit and new: the bot is still among the requested reviewers (`_review_awaited`), or a
  trusted human's mention comment is routed for the first time
- `review_reply` from a trusted human under one of the bot's finding threads: on a reviewed
  live head, the head re-opens at priority so the reply is answered on its thread (a
  conversation when a final review stands); after a push nothing has reviewed yet ("fixed in
  ..."), the push is reviewed like any other (debounced, in line, carried when its diff did
  not change) and that review reads and answers the reply
- a top-level PR comment from a trusted human quoting one of the bot's posted findings (one
  published only in the review body has no thread to reply under; the conversation lane
  answers threads only): routed as a `review_reply` is, except that a reviewed head re-opens
  as a full same-sha re-review, which reads the comment in the discussion
- comments/reviews on the bot's own PRs from trusted humans or CodeRabbit ->
  batched per (repo, pr) and delivered to the OpenClaw agent as one wake event

Replaces gh_notify_processor.py and review-bot/bot.py.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import timedelta
from typing import Any

from .config import Config
from .db import event, fmt_ts, kv_get, kv_set, now, now_dt, parse_ts, tx
from .dedupe import FINDING_MARKER_RE
from .gh import Gh
from .ingest import AUTOMATIC_TRIGGERS, enqueue_head, push_trigger
from .models import HeadStatus, Trigger
from .notify import Notifier

log = logging.getLogger(__name__)

API_ROOT = "https://api.github.com/"
BATCH_DELAY = timedelta(minutes=5)
BATCH_RETRY = timedelta(minutes=15)
# re-reviews of one already-reviewed commit that review requests, and mentions, may each start
MAX_ASKED_REREVIEWS = 3
DEFER_MAX = timedelta(hours=6)  # how long an inbox row may wait on a retryable condition
TRUSTED_ASSOCIATIONS = frozenset({"MEMBER", "COLLABORATOR", "CONTRIBUTOR", "OWNER"})
ALLOWED_BOTS = frozenset({"coderabbitai", "coderabbitai[bot]"})
_MARKUP_RE = re.compile(r"[`*_]")


def _mention_re(bot: str) -> re.Pattern[str]:
    return re.compile(rf"@{re.escape(bot)}\b[^\n]{{0,40}}\breview\b", re.IGNORECASE)


def _rereview_re(bot: str) -> re.Pattern[str]:
    """A mention that asks for a review in so many words ("@bot review", "@bot please
    re-review", "@bot can you review again"), unlike one that only talks about a review
    ("@bot thanks for the review"), which `_mention_re` also matches."""
    return re.compile(
        rf"@{re.escape(bot)}[\s,:]+(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?(?:re-?)?review\b",
        re.IGNORECASE,
    )


def _fetch(gh: Gh, endpoint: str, what: str) -> dict[str, Any] | None:
    """One JSON object from the API, or None when the call fails or returns something else.

    Routing is best effort: a failed lookup leaves the inbox row for the next tick.
    """
    try:
        data = gh.api(endpoint)
    except Exception as exc:
        log.warning("%s fetch %s failed: %s", what, endpoint, exc)
        return None
    return data if isinstance(data, dict) else None


def _pr_meta(gh: Gh, repo: str, number: int) -> dict[str, Any] | None:
    return _fetch(gh, f"repos/{repo}/pulls/{number}", "pr meta")


def _comment(gh: Gh, url: str) -> dict[str, Any] | None:
    if not url.startswith(API_ROOT):
        return None
    return _fetch(gh, url.removeprefix(API_ROOT), "comment")


def route_inbox(
    conn: sqlite3.Connection, cfg: Config, gh: Gh, notifier: Notifier
) -> dict[str, int]:
    stats = {
        "review_requested": 0,
        "mention": 0,
        "review_reply": 0,
        "review_reply_deferred": 0,
        "deferred": 0,
        "own_pr_comment": 0,
        "ignored": 0,
        "batches_sent": 0,
    }
    rows = conn.execute(
        "SELECT * FROM inbox WHERE handled_at IS NULL ORDER BY occurred_at LIMIT 200"
    ).fetchall()
    mention_re = _mention_re(cfg.bot_login)
    ts = now()
    meta_cache: dict[tuple[str, int], dict[str, Any] | None] = {}
    for row in rows:
        repo, number = row["repo"], row["number"]
        if number is None or not repo:
            _handled(conn, row["id"], "ignored", ts)
            stats["ignored"] += 1
            continue
        if (repo, number) not in meta_cache:
            meta_cache[(repo, number)] = _pr_meta(gh, repo, number)
        meta = meta_cache[(repo, number)]
        if meta is None:
            continue  # leave unhandled; retry next tick
        stats[_route_row(conn, cfg, gh, row, meta=meta, mention_re=mention_re, ts=ts)] += 1

    stats["batches_sent"] = flush_batches(conn, cfg, notifier)
    return stats


def _route_row(
    conn: sqlite3.Connection,
    cfg: Config,
    gh: Gh,
    row: sqlite3.Row,
    *,
    meta: dict[str, Any],
    mention_re: re.Pattern[str],
    ts: str,
) -> str:
    """Handle one inbox row. Returns the stats key describing what was done."""
    repo, number, kind = row["repo"], row["number"], row["kind"]
    watched = repo in cfg.enabled_repos
    own_pr = str((meta.get("user") or {}).get("login") or "").lower() == cfg.bot_login.lower()
    head = str((meta.get("head") or {}).get("sha") or "")
    is_open = meta.get("state") == "open" and not meta.get("draft")

    if kind == "review_requested" and watched and is_open and head and not own_pr:
        with tx(conn):
            _ask(
                conn,
                cfg,
                repo,
                number,
                head,
                Trigger.REVIEW_REQUESTED,
                ts,
                rereview=_review_awaited(cfg, meta),
            )
            _handled(conn, row["id"], "review_requested", ts)
        return "review_requested"

    comment = _comment(gh, str(row["body"] or "")) if row["body"] else None
    retry = kind in {"mention", "review_reply"} or (
        kind == "comment" and watched and is_open and "/issues/comments/" in str(row["body"])
    )
    if row["body"] and comment is None and retry:
        if _too_old(row, ts):
            with tx(conn):
                _handled(conn, row["id"], "ignored_unfetchable", ts)
            return "ignored"
        return "deferred"  # transient fetch failure: retry next tick, never drop a human's words
    body = str((comment or {}).get("body") or "")
    actor = str(((comment or {}).get("user") or {}).get("login") or "")
    # a repo with no skills entry is reviewed ad hoc when a trusted human asks; a repo that
    # is listed but disabled stays off no matter who asks
    adhoc_ok = cfg.repo(repo) is None and _trusted(cfg, comment or {}, actor)

    if (
        kind == "mention"
        and is_open
        and head
        and mention_re.search(body)
        and not own_pr
        and (watched or adhoc_ok)
    ):
        with tx(conn):
            first = _asks_again(conn, cfg, row, comment or {}, actor, body)
            _ask(conn, cfg, repo, number, head, Trigger.MENTION, ts, rereview=first)
            _handled(conn, row["id"], "mention", ts)
        return "mention"

    if kind == "review_reply" and comment and is_open and head and not own_pr:
        # another bot (CodeRabbit, a GitHub App) is trusted for mentions but is never someone
        # waiting on an answer from us; queuing a run for it would only ever find nothing to say
        if not _trusted(cfg, comment, actor) or actor.lower().endswith("[bot]"):
            with tx(conn):
                _handled(conn, row["id"], "ignored", ts)
            return "ignored"
        is_ours = _replies_to_bot_finding(gh, comment, cfg.bot_login)
        if is_ours is None:
            if _too_old(row, ts):
                with tx(conn):
                    _handled(conn, row["id"], "ignored_unfetchable", ts)
                return "ignored"
            return "deferred"
        if not is_ours:
            with tx(conn):
                _handled(conn, row["id"], "ignored", ts)
            return "ignored"
        return _queue_reply(
            conn,
            cfg,
            row,
            head=head,
            watched=watched,
            ts=ts,
            trigger=Trigger.REVIEW_REPLY,
            action="review_reply",
            detail=f"{actor} replied on comment {comment.get('in_reply_to_id')}",
        )

    if (
        kind != "review_reply"
        and comment
        and watched
        and is_open
        and head
        and not own_pr
        and "/issues/comments/" in str(row["body"])
        and actor.lower() not in (cfg.bot_login.lower(), *ALLOWED_BOTS)
        and not actor.lower().endswith("[bot]")
        and _trusted(cfg, comment, actor)
        and _quotes_bot_finding(conn, repo, number, body)
        and not _already_routed(conn, row, "discussion_reply%")
    ):
        return _queue_reply(
            conn,
            cfg,
            row,
            head=head,
            watched=True,
            ts=ts,
            trigger=Trigger.MANUAL,
            action="discussion_reply",
            detail=f"{actor} answered a finding in PR comment {comment.get('id')}",
        )

    if (
        own_pr
        and comment
        and actor
        and actor.lower() != cfg.bot_login.lower()
        and _trusted(cfg, comment, actor)
    ):
        with tx(conn):
            conn.execute(
                "UPDATE inbox SET actor=?, body=?, action='own_pr_comment' WHERE id=?",
                (actor, body[:4000], row["id"]),
            )
            # handled_at stays NULL until the batch is flushed
        return "own_pr_comment"

    with tx(conn):
        _handled(conn, row["id"], "ignored", ts)
    return "ignored"


def _queue_reply(
    conn: sqlite3.Connection,
    cfg: Config,
    row: sqlite3.Row,
    *,
    head: str,
    watched: bool,
    ts: str,
    trigger: Trigger,
    action: str,
    detail: str,
) -> str:
    """Re-open the live head so a human's answer to a finding is read: a push nothing has
    reviewed yet keeps its own review (`_reply_after_push`), a reviewed head re-opens at
    priority with `trigger`. Returns the stats key."""
    repo, number = row["repo"], row["number"]
    with tx(conn):
        # only a watched repo reviews its pushes on its own; elsewhere a reply still asks
        res = _reply_after_push(conn, cfg, repo, number, head, ts) if watched else None
        if res is None:
            res = enqueue_head(conn, cfg, repo, number, head, trigger, ts=ts)
        if res == "running" or (res == "noop" and _head_running(conn, repo, number, head)):
            # the same head is mid-review and its context may predate this reply: leave the
            # row unhandled so it re-opens the head once that run finishes (bounded)
            if _too_old(row, ts):
                _handled(conn, row["id"], f"{action}_expired", ts)
                return "ignored"
            return "review_reply_deferred"
        _handled(conn, row["id"], f"{action}_queued" if res == "queued" else action, ts)
        event(
            conn,
            "head.review_reply",
            repo=repo,
            number=number,
            detail=f"{detail} -> {res} {head[:8]}",
        )
    return "review_reply"


def _plain(text: str) -> str:
    return " ".join(_MARKUP_RE.sub("", text).split()).lower()


def _quotes_bot_finding(conn: sqlite3.Connection, repo: str, number: int, body: str) -> bool:
    """The comment quotes ("> ...") the title of one of the bot's posted findings on this PR."""
    quoted = _plain(" ".join(ln.lstrip()[1:] for ln in body.splitlines() if ln.lstrip()[:1] == ">"))
    if not quoted:
        return False
    rows = conn.execute(
        "SELECT DISTINCT f.title FROM posted_findings pf JOIN findings f "
        "ON f.hash=pf.hash AND f.stage='posted' WHERE pf.repo=? AND pf.number=?",
        (repo, number),
    )
    return any(t and t in quoted for t in (_plain(str(r[0] or "")) for r in rows))


def _already_routed(conn: sqlite3.Connection, row: sqlite3.Row, action: str) -> bool:
    """GitHub re-notifies for the same comment (an edit, a later mention): route it once.
    `action`: a LIKE pattern for the inbox action that routing it leaves."""
    return (
        conn.execute(
            "SELECT 1 FROM inbox WHERE body=? AND id<>? AND action LIKE ? "
            "AND action NOT LIKE '%_expired' LIMIT 1",
            (row["body"], row["id"], action),
        ).fetchone()
        is not None
    )


def _own_words(body: str) -> str:
    """A comment without what it quotes or shows as code: a quoted "> @bot review" or a
    command in backticks is not the commenter asking. A quote runs to the next blank line,
    as Markdown reads it: the lines after a ">" line continue it without a marker."""
    kept: list[str] = []
    fenced = quoted = False
    for line in body.splitlines():
        text = line.strip()
        if text.startswith(("```", "~~~")):
            fenced = not fenced
            quoted = False
            continue
        if fenced:
            continue
        quoted = text.startswith(">") or (quoted and bool(text))
        if not quoted:
            kept.append(re.sub(r"`[^`\n]*`", " ", line))
    return "\n".join(kept)


def _asks_again(
    conn: sqlite3.Connection,
    cfg: Config,
    row: sqlite3.Row,
    comment: dict[str, Any],
    actor: str,
    body: str,
) -> bool:
    """A mention that re-reviews a commit already reviewed: a trusted human's top-level
    comment asking for a review outright, routed for the first time. A bot's comment never
    does (the bot's own gate comment and CodeRabbit both write "@bot review"), nor a reply
    under a finding, which is answered as a conversation."""
    login = actor.lower()
    return (
        "/issues/comments/" in str(row["body"])
        and login not in (cfg.bot_login.lower(), *ALLOWED_BOTS)
        and not login.endswith("[bot]")
        and _trusted(cfg, comment, actor)
        and bool(_rereview_re(cfg.bot_login).search(_own_words(body)))
        and not _already_routed(conn, row, "mention")
    )


def _review_awaited(cfg: Config, meta: dict[str, Any]) -> bool:
    """Someone asked the bot for a review that it has not given yet: it is among the pull
    request's requested reviewers, which GitHub clears when the bot next submits a review.
    Every later update of the pull request repeats the `review_requested` notification, so
    the notification alone says nothing."""
    return any(
        str((r or {}).get("login") or "").lower() == cfg.bot_login.lower()
        for r in meta.get("requested_reviewers") or []
    )


def _ask(
    conn: sqlite3.Connection,
    cfg: Config,
    repo: str,
    number: int,
    head: str,
    trigger: Trigger,
    ts: str,
    *,
    rereview: bool,
) -> None:
    """Queue `head` for someone asking. A fresh explicit ask (`rereview`) also re-reviews a
    commit that has its own review, at most MAX_ASKED_REREVIEWS times per commit and route
    (requests and mentions are counted apart, so comments cannot use up what a review
    request may start): a re-review that ends without submitting a review leaves the
    request standing, and each new comment is a new ask. Only an ask that starts a review
    is counted: re-opening the finished commit, or turning its queued CI check into one.
    Caller holds the transaction."""
    key = f"router.asked_rereviews:{trigger.value}:{repo}#{number}:{head}"
    done = int(kv_get(conn, key) or 0)
    allowed = rereview and done < MAX_ASKED_REREVIEWS
    before = conn.execute(
        "SELECT h.trigger, EXISTS(SELECT 1 FROM reviews r WHERE r.repo=h.repo "
        "AND r.number=h.number AND r.sha=h.sha) AS reviewed FROM heads h "
        "WHERE h.repo=? AND h.number=? AND h.sha=?",
        (repo, number, head),
    ).fetchone()
    res = enqueue_head(conn, cfg, repo, number, head, trigger, ts=ts, rereview=allowed)
    started = res == "requeued" or (
        res == "promoted" and before is not None and before["trigger"] == Trigger.CI_FAILED.value
    )
    if allowed and started and before is not None and before["reviewed"]:
        kv_set(conn, key, str(done + 1))


def _reply_after_push(
    conn: sqlite3.Connection, cfg: Config, repo: str, number: int, head: str, ts: str
) -> str | None:
    """The reply landed on a push nothing has reviewed yet (typically "fixed in <sha>" right
    after it): no request to review that push sooner or harder. It keeps its automatic head
    (queued here as ingest would queue it when ingest has not seen it yet), whose review reads
    the threads and answers the reply; `worker._carry` does not carry over a reply no run has
    read. "queued", or "running" when that head's run may have read the threads before the
    reply; None when the live head is not such a push."""
    h = conn.execute(
        "SELECT status, trigger, queue FROM heads WHERE repo=? AND number=? AND sha=?",
        (repo, number, head),
    ).fetchone()
    if h is None:
        res = enqueue_head(conn, cfg, repo, number, head, push_trigger(conn, repo, number), ts=ts)
        return "queued" if res == "created" else res  # noop: a skip label holds pushes back
    if h["queue"] != "live" or h["trigger"] not in AUTOMATIC_TRIGGERS:
        return None
    if h["status"] == HeadStatus.QUEUED:
        return "queued"
    return "running" if h["status"] == HeadStatus.RUNNING else None


def _replies_to_bot_finding(gh: Gh, comment: dict[str, Any], bot_login: str) -> bool | None:
    """True when the comment's thread root is one of the bot's finding comments, False when it
    is not, None when the root could not be fetched (caller retries next tick)."""
    parent_id = comment.get("in_reply_to_id")
    pr_url = str(comment.get("pull_request_url") or "")
    if not parent_id or not pr_url.startswith(API_ROOT):
        return False
    repo_path = pr_url.removeprefix(f"{API_ROOT}repos/").split("/pulls/", 1)[0]
    root = _fetch(gh, f"repos/{repo_path}/pulls/comments/{parent_id}", "thread root")
    if root is None:
        return None
    login = str((root.get("user") or {}).get("login") or "")
    return login.lower() == bot_login.lower() and bool(
        FINDING_MARKER_RE.search(str(root.get("body") or ""))
    )


def _head_running(conn: sqlite3.Connection, repo: str, number: int, sha: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM heads WHERE repo=? AND number=? AND sha=? AND status='running'",
            (repo, number, sha),
        ).fetchone()
        is not None
    )


def _too_old(row: sqlite3.Row, ts: str) -> bool:
    """Deferred rows get a bounded life so a stuck lookup can never wedge the inbox."""
    return parse_ts(ts) - parse_ts(str(row["seen_at"])) > DEFER_MAX


def _trusted(cfg: Config, comment: dict[str, Any], actor: str) -> bool:
    return (
        str(comment.get("author_association") or "") in TRUSTED_ASSOCIATIONS
        or actor in cfg.trusted_reviewers
        or actor.lower() in ALLOWED_BOTS
    )


def _handled(conn: sqlite3.Connection, inbox_id: int, action: str, ts: str) -> None:
    conn.execute("UPDATE inbox SET handled_at=?, action=? WHERE id=?", (ts, action, inbox_id))


def flush_batches(conn: sqlite3.Connection, cfg: Config, notifier: Notifier) -> int:
    """Deliver own-PR comment batches whose newest comment is older than BATCH_DELAY."""
    pending = conn.execute(
        "SELECT repo, number, MAX(occurred_at) AS newest, COUNT(*) AS n FROM inbox WHERE handled_at IS NULL AND action='own_pr_comment' GROUP BY repo, number"
    ).fetchall()
    sent = 0
    cutoff = now_dt() - BATCH_DELAY
    for b in pending:
        if parse_ts(b["newest"]) > cutoff:
            continue
        items = conn.execute(
            "SELECT id, actor, body FROM inbox WHERE handled_at IS NULL AND action='own_pr_comment' AND repo=? AND number=? ORDER BY occurred_at",
            (b["repo"], b["number"]),
        ).fetchall()
        actors = sorted({str(i["actor"]) for i in items})
        text = (
            f"🔧 Review feedback on your PR {b['repo']}#{b['number']}: {len(items)} new comment(s) from {', '.join(actors)}. "
            f"Read the PR threads, address valid feedback on the PR branch, and reply on GitHub. "
            f"Preview of latest: {str(items[-1]['body'] or '')[:300]!r}"
        )
        if not notifier.wake_enabled:
            # shadow mode: observe only; leave the batch unhandled so a live daemon delivers it
            continue
        retry_key = f"router.batch_retry_after:{b['repo']}#{b['number']}"
        retry_after = kv_get(conn, retry_key)
        if retry_after and parse_ts(retry_after) > now_dt():
            continue
        ok = notifier.wake_agent(text)
        with tx(conn):
            if ok:
                ts = now()
                for i in items:
                    _handled(conn, i["id"], "own_pr_comment_delivered", ts)
                conn.execute("DELETE FROM kv WHERE key=?", (retry_key,))
            else:
                # leave the rows unhandled and try again later; never drop review feedback
                kv_set(conn, retry_key, fmt_ts(now_dt() + BATCH_RETRY))
            event(
                conn,
                "router.batch" if ok else "router.batch_failed",
                repo=b["repo"],
                number=b["number"],
                detail=f"{len(items)} comments delivered={ok}",
            )
        sent += int(ok)
    return sent
