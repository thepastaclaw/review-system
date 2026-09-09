"""Route inbox rows to actions.

- `review_requested` on a watched repo  -> priority head at the live PR head
- `mention` whose comment says `@<bot> review` -> priority head (any repo the bot can
  read: repos without a skills entry get an ad hoc review)
- `review_reply` from a trusted human under one of the bot's finding threads -> priority
  head at the live PR head, so the reply is answered on its thread within one review cycle
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
from .ingest import enqueue_head
from .models import Trigger
from .notify import Notifier

log = logging.getLogger(__name__)

API_ROOT = "https://api.github.com/"
BATCH_DELAY = timedelta(minutes=5)
BATCH_RETRY = timedelta(minutes=15)
DEFER_MAX = timedelta(hours=6)  # how long an inbox row may wait on a retryable condition
TRUSTED_ASSOCIATIONS = frozenset({"MEMBER", "COLLABORATOR", "CONTRIBUTOR", "OWNER"})
ALLOWED_BOTS = frozenset({"coderabbitai", "coderabbitai[bot]"})


def _mention_re(bot: str) -> re.Pattern[str]:
    return re.compile(rf"@{re.escape(bot)}\b[^\n]{{0,40}}\breview\b", re.IGNORECASE)


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
            enqueue_head(conn, cfg, repo, number, head, Trigger.REVIEW_REQUESTED, ts=ts)
            _handled(conn, row["id"], "review_requested", ts)
        return "review_requested"

    comment = _comment(gh, str(row["body"] or "")) if row["body"] else None
    if row["body"] and comment is None and kind in {"mention", "review_reply"}:
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
            enqueue_head(conn, cfg, repo, number, head, Trigger.MENTION, ts=ts)
            _handled(conn, row["id"], "mention", ts)
        return "mention"

    if kind == "review_reply" and comment and is_open and head and not own_pr:
        if not _trusted(cfg, comment, actor):
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
        with tx(conn):
            res = enqueue_head(conn, cfg, repo, number, head, Trigger.REVIEW_REPLY, ts=ts)
            if res == "noop" and _head_running(conn, repo, number, head):
                # the same head is mid-review and its context predates this reply: leave the
                # row unhandled so it re-opens the head once that run finishes (bounded)
                if _too_old(row, ts):
                    _handled(conn, row["id"], "review_reply_expired", ts)
                    return "ignored"
                return "review_reply_deferred"
            _handled(conn, row["id"], "review_reply", ts)
            event(
                conn,
                "head.review_reply",
                repo=repo,
                number=number,
                detail=f"{actor} replied on comment {comment.get('in_reply_to_id')} -> {res} {head[:8]}",
            )
        return "review_reply"

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
