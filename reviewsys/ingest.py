"""Ingest: poll open PRs into `prs`/`heads`, poll notifications into `inbox`."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from .config import Config
from .db import event, fmt_ts, kv_get, kv_set, now, now_dt, parse_ts, tx
from .gh import Gh
from .models import HeadStatus, Trigger

log = logging.getLogger(__name__)

OPEN_PRS_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(first: 100, states: OPEN, after: $cursor, orderBy: {field: UPDATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes { number headRefOid isDraft title updatedAt author { login } }
    }
  }
}
"""

IGNORED_AUTHORS = frozenset(
    {"dependabot", "dependabot[bot]", "renovate", "renovate[bot]", "github-actions[bot]"}
)


@dataclass(slots=True)
class OpenPr:
    number: int
    head_sha: str
    is_draft: bool
    title: str
    author: str
    updated_at: str


def fetch_open_prs(gh: Gh, repo: str) -> list[OpenPr]:
    owner, name = repo.split("/", 1)
    out: list[OpenPr] = []
    cursor: str | None = None
    for _ in range(20):
        variables: dict[str, Any] = {"owner": owner, "name": name}
        if cursor:
            variables["cursor"] = cursor
        data = gh.graphql(OPEN_PRS_QUERY, variables)
        conn = (data or {}).get("repository", {}).get("pullRequests") or {}
        for n in conn.get("nodes") or []:
            out.append(
                OpenPr(
                    number=int(n["number"]),
                    head_sha=str(n["headRefOid"]),
                    is_draft=bool(n.get("isDraft")),
                    title=str(n.get("title") or ""),
                    author=str((n.get("author") or {}).get("login") or ""),
                    updated_at=str(n.get("updatedAt") or ""),
                )
            )
        if not conn.get("pageInfo", {}).get("hasNextPage"):
            break
        cursor = conn["pageInfo"]["endCursor"]
    return out


def enqueue_head(
    conn: sqlite3.Connection,
    cfg: Config,
    repo: str,
    number: int,
    sha: str,
    trigger: Trigger,
    *,
    ts: str | None = None,
) -> str:
    """Create a queued head for (repo, number, sha), superseding any older active head.

    Returns one of: created, promoted (existing head got priority), requeued (a finished
    head re-opened by a review reply), noop. Caller holds a transaction.
    """
    ts = ts or now()
    existing = conn.execute(
        "SELECT id, status, priority FROM heads WHERE repo=? AND number=? AND sha=?",
        (repo, number, sha),
    ).fetchone()
    priority = 1 if trigger.priority else 0
    if existing:
        if (
            existing["status"] in (HeadStatus.QUEUED, HeadStatus.RUNNING)
            and priority
            and not existing["priority"]
        ):
            conn.execute(
                "UPDATE heads SET priority=1, trigger=?, eligible_at=? WHERE id=?",
                (trigger.value, ts, existing["id"]),
            )
            return "promoted"
        if trigger in (Trigger.REVIEW_REPLY, Trigger.MANUAL) and existing["status"] in (
            HeadStatus.DONE,
            HeadStatus.FAILED,
        ):
            # same commit, already reviewed: a human answered a finding (or an operator asked),
            # so review it again with the threads in context and answer on them
            conn.execute(
                "UPDATE heads SET status='queued', priority=1, trigger=?, queued_at=?, eligible_at=?, finished_at=NULL, reason=NULL, attempts=0 WHERE id=?",
                (trigger.value, ts, ts, existing["id"]),
            )
            event(
                conn,
                "head.requeued",
                repo=repo,
                number=number,
                detail=f"{sha[:8]} re-opened by {trigger.value}",
            )
            return "requeued"
        return "noop"
    # supersede older active heads for this PR
    for row in conn.execute(
        "SELECT id, status FROM heads WHERE repo=? AND number=? AND status IN ('queued','running')",
        (repo, number),
    ).fetchall():
        conn.execute(
            "UPDATE heads SET status='superseded', superseded_at=?, finished_at=?, reason=? WHERE id=?",
            (ts, ts, f"new head {sha[:8]}", row["id"]),
        )
        event(
            conn,
            "head.superseded",
            repo=repo,
            number=number,
            detail=f"head {row['id']} superseded by {sha[:8]}",
        )
    eligible = ts if priority else fmt_ts(parse_ts(ts) + timedelta(minutes=cfg.debounce_minutes))
    conn.execute(
        "INSERT INTO heads (repo, number, sha, trigger, priority, status, queued_at, eligible_at) VALUES (?,?,?,?,?,?,?,?)",
        (repo, number, sha, trigger.value, priority, HeadStatus.QUEUED.value, ts, eligible),
    )
    event(
        conn,
        "head.queued",
        repo=repo,
        number=number,
        detail=f"{sha[:8]} trigger={trigger.value} priority={priority}",
    )
    return "created"


def close_pr_heads(
    conn: sqlite3.Connection, repo: str, number: int, reason: str, *, ts: str | None = None
) -> int:
    ts = ts or now()
    rows = conn.execute(
        "SELECT id FROM heads WHERE repo=? AND number=? AND status IN ('queued','running')",
        (repo, number),
    ).fetchall()
    for row in rows:
        conn.execute(
            "UPDATE heads SET status='closed', finished_at=?, reason=? WHERE id=?",
            (ts, reason, row["id"]),
        )
        event(conn, "head.closed", repo=repo, number=number, detail=reason)
    return len(rows)


def ingest_repo(
    conn: sqlite3.Connection, cfg: Config, gh: Gh, repo: str, prs: list[OpenPr]
) -> dict[str, int]:
    """Reconcile one repo's open PR set into the DB. Pure DB work; PRs already fetched."""
    stats = {"created": 0, "promoted": 0, "noop": 0, "closed": 0, "draft": 0, "ignored": 0}
    ts = now()
    open_numbers = {p.number for p in prs}
    with tx(conn):
        for p in prs:
            conn.execute(
                "INSERT INTO prs (repo, number, head_sha, title, author, is_draft, state, updated_at) VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(repo, number) DO UPDATE SET head_sha=excluded.head_sha, title=excluded.title, author=excluded.author, is_draft=excluded.is_draft, state='open', updated_at=excluded.updated_at",
                (
                    repo,
                    p.number,
                    p.head_sha,
                    p.title,
                    p.author,
                    int(p.is_draft),
                    "open",
                    p.updated_at or ts,
                ),
            )
            if p.author.lower() in IGNORED_AUTHORS:
                stats["ignored"] += 1
                continue
            if p.is_draft:
                if close_pr_heads(conn, repo, p.number, "pr_draft", ts=ts):
                    stats["draft"] += 1
                continue
            prior = conn.execute(
                "SELECT 1 FROM heads WHERE repo=? AND number=? LIMIT 1", (repo, p.number)
            ).fetchone()
            trigger = Trigger.NEW_PUSH if prior else Trigger.NEW_PR
            stats[enqueue_head(conn, cfg, repo, p.number, p.head_sha, trigger, ts=ts)] += 1
        # PRs we knew as open that are no longer in the open set
        known = conn.execute(
            "SELECT number FROM prs WHERE repo=? AND state='open'", (repo,)
        ).fetchall()
        for row in known:
            if row["number"] not in open_numbers:
                conn.execute(
                    "UPDATE prs SET state='closed', updated_at=? WHERE repo=? AND number=?",
                    (ts, repo, row["number"]),
                )
                stats["closed"] += close_pr_heads(conn, repo, row["number"], "pr_closed", ts=ts)
    return stats


def ingest_prs(conn: sqlite3.Connection, cfg: Config, gh: Gh) -> dict[str, dict[str, int]]:
    """Poll every enabled repo. Network happens outside any transaction."""
    results: dict[str, dict[str, int]] = {}
    errors = 0
    for repo in cfg.enabled_repos:
        try:
            prs = fetch_open_prs(gh, repo)
        except Exception as exc:
            errors += 1
            log.warning("ingest %s failed: %s", repo, exc)
            with tx(conn):
                event(conn, "ingest.error", repo=repo, detail=str(exc))
            continue
        results[repo] = ingest_repo(conn, cfg, gh, repo, prs)
    with tx(conn):
        streak = int(kv_get(conn, "ingest.error_streak", "0") or 0)
        kv_set(conn, "ingest.error_streak", str(streak + 1 if errors and not results else 0))
        kv_set(conn, "ingest.last_at", now())
    return results


# ---- notifications -> inbox ----

INBOX_INSERT = (
    "INSERT OR IGNORE INTO inbox (source_id, kind, repo, number, actor, body, occurred_at, seen_at)"
    " VALUES (?,?,?,?,?,?,?,?)"
)


def _poll_cursor(conn: sqlite3.Connection, key: str) -> str:
    """Stored poll cursor for `key`, seeded on first run to just-now rather than replaying
    weeks of history through the router."""
    since = kv_get(conn, key)
    if since is None:
        since = fmt_ts(now_dt() - timedelta(minutes=10))
        with tx(conn):
            kv_set(conn, key, since)
    return since


def ingest_notifications(conn: sqlite3.Connection, cfg: Config, gh: Gh) -> int:
    """Pull GitHub notifications since the stored cursor into `inbox`. Returns rows added."""
    since = _poll_cursor(conn, "notify.cursor")
    endpoint = f"/notifications?all=true&per_page=100&since={since}"
    rows = gh.api(endpoint, paginate=True) or []
    added = 0
    newest = since
    with tx(conn):
        for n in rows:
            if not isinstance(n, dict):
                continue
            subj = n.get("subject") or {}
            if subj.get("type") != "PullRequest":
                continue
            repo = str((n.get("repository") or {}).get("full_name") or "")
            url = str(subj.get("url") or "")
            number = int(url.rsplit("/", 1)[-1]) if url.rsplit("/", 1)[-1].isdigit() else None
            updated = str(n.get("updated_at") or now())
            source_id = f"{n.get('id')}:{updated}"
            cur = conn.execute(
                INBOX_INSERT,
                (
                    source_id,
                    str(n.get("reason") or ""),
                    repo,
                    number,
                    None,
                    str(subj.get("latest_comment_url") or ""),
                    updated,
                    now(),
                ),
            )
            added += cur.rowcount
            if newest is None or updated > newest:
                newest = updated
        if newest:
            kv_set(conn, "notify.cursor", newest)
        kv_set(conn, "notify.last_at", now())
    return added


# ---- review-comment replies -> inbox ----

REPLY_OVERLAP = timedelta(minutes=2)


def ingest_review_replies(conn: sqlite3.Connection, cfg: Config, gh: Gh) -> int:
    """Pull inline review comments updated since the cursor for every repo we have posted on,
    and put replies (comments with `in_reply_to_id`) from non-bot users into `inbox` as kind
    `review_reply`. Notifications do not carry the comment URL for these, so this is the only
    reliable way to notice that someone answered one of our findings. Returns rows added."""
    repos = [
        r["repo"]
        for r in conn.execute("SELECT DISTINCT repo FROM posted_findings ORDER BY repo").fetchall()
    ]
    added = 0
    for repo in repos:
        # one cursor per repo so a repo that stops answering only stalls itself; the cursor
        # trails the poll start by REPLY_OVERLAP so a comment that becomes listable late is
        # still seen (dedupe is by comment id, so re-listing is free)
        key = f"replies.cursor:{repo}"
        since = kv_get(conn, key) or _poll_cursor(conn, "replies.cursor")
        poll_start = now_dt()
        try:
            rows = (
                gh.api(
                    f"repos/{repo}/pulls/comments?sort=updated&direction=asc&since={since}&per_page=100",
                    paginate=True,
                )
                or []
            )
        except Exception as exc:
            log.warning("reply ingest %s failed: %s", repo, exc)
            continue
        with tx(conn):
            for c in rows:
                if not isinstance(c, dict):
                    continue
                actor = str((c.get("user") or {}).get("login") or "")
                if not c.get("in_reply_to_id") or actor.lower() == cfg.bot_login.lower():
                    continue
                pr_url = str(c.get("pull_request_url") or "")
                tail = pr_url.rsplit("/", 1)[-1]
                if not tail.isdigit():
                    continue
                cur = conn.execute(
                    INBOX_INSERT,
                    (
                        f"reply:{c.get('id')}",
                        "review_reply",
                        repo,
                        int(tail),
                        actor,
                        str(c.get("url") or ""),
                        str(c.get("created_at") or now()),
                        now(),
                    ),
                )
                added += cur.rowcount
            kv_set(conn, key, fmt_ts(poll_start - REPLY_OVERLAP))
    with tx(conn):
        kv_set(conn, "replies.last_at", now())
    return added
