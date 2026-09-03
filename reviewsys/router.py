"""Route inbox rows to actions.

- `review_requested` on a watched repo  -> priority head at the live PR head
- `mention` whose comment says `@<bot> review` -> priority head
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
from .db import event, now, now_dt, parse_ts, tx
from .gh import Gh
from .ingest import enqueue_head
from .models import Trigger
from .notify import Notifier

log = logging.getLogger(__name__)

BATCH_DELAY = timedelta(minutes=5)
TRUSTED_ASSOCIATIONS = frozenset({"MEMBER", "COLLABORATOR", "CONTRIBUTOR", "OWNER"})
ALLOWED_BOTS = frozenset({"coderabbitai", "coderabbitai[bot]"})


def _mention_re(bot: str) -> re.Pattern[str]:
    return re.compile(rf"@{re.escape(bot)}\b[^\n]{{0,40}}\breview\b", re.IGNORECASE)


def _pr_meta(gh: Gh, repo: str, number: int) -> dict[str, Any] | None:
    try:
        data = gh.api(f"repos/{repo}/pulls/{number}")
    except Exception as exc:
        log.warning("pr meta %s#%s failed: %s", repo, number, exc)
        return None
    return data if isinstance(data, dict) else None


def _comment(gh: Gh, url: str) -> dict[str, Any] | None:
    if not url.startswith("https://api.github.com/"):
        return None
    try:
        data = gh.api(url.removeprefix("https://api.github.com/"))
    except Exception as exc:
        log.warning("comment fetch %s failed: %s", url, exc)
        return None
    return data if isinstance(data, dict) else None


def route_inbox(
    conn: sqlite3.Connection, cfg: Config, gh: Gh, notifier: Notifier
) -> dict[str, int]:
    stats = {
        "review_requested": 0,
        "mention": 0,
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
    body = str((comment or {}).get("body") or "")
    actor = str(((comment or {}).get("user") or {}).get("login") or "")

    if (
        kind == "mention"
        and watched
        and is_open
        and head
        and mention_re.search(body)
        and not own_pr
    ):
        with tx(conn):
            enqueue_head(conn, cfg, repo, number, head, Trigger.MENTION, ts=ts)
            _handled(conn, row["id"], "mention", ts)
        return "mention"

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
        ok = notifier.wake_agent(text)
        ts = now()
        with tx(conn):
            for i in items:
                _handled(
                    conn,
                    i["id"],
                    "own_pr_comment_delivered" if ok else "own_pr_comment_delivery_failed",
                    ts,
                )
            event(
                conn,
                "router.batch",
                repo=b["repo"],
                number=b["number"],
                detail=f"{len(items)} comments delivered={ok}",
            )
        sent += 1
    return sent
