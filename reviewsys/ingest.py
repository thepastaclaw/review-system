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
      nodes {
        number headRefOid isDraft title updatedAt author { login }
        labels(first: 50) { nodes { name } }
      }
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
    labels: tuple[str, ...] = ()


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
                    labels=label_names(n),
                )
            )
        if not conn.get("pageInfo", {}).get("hasNextPage"):
            break
        cursor = conn["pageInfo"]["endCursor"]
    return out


def label_names(node: dict[str, Any]) -> tuple[str, ...]:
    """Label names of a GraphQL PR node (`labels { nodes { name } }`)."""
    return tuple(
        str(n["name"]) for n in (node.get("labels") or {}).get("nodes") or [] if n and n.get("name")
    )


def skip_label(cfg: Config, labels: tuple[str, ...]) -> str | None:
    """The first of the PR's labels that turns automatic reviews off (`skip_labels`), or None."""
    return next((name for name in labels if name.lower() in cfg.skip_labels), None)


# a head ingest closed because the PR carries a skip label; re-opened when the label goes
SKIP_LABEL_REASON = "pr_skip_label"
# what a skip label refuses: the heads nobody asked for (a new PR or push, a CI check)
SKIPPABLE_TRIGGERS = frozenset(
    {Trigger.NEW_PR.value, Trigger.NEW_PUSH.value, Trigger.CI_FAILED.value}
)


def skip_noted_key(repo: str, number: int) -> str:
    """kv: the commit whose gate comment last got the skip-label note (queue_status.py)."""
    return f"queue.skip_noted:{repo}#{number}"


def _has_skip_label(conn: sqlite3.Connection, repo: str, number: int) -> bool:
    row = conn.execute(
        "SELECT skip_label FROM prs WHERE repo=? AND number=?", (repo, number)
    ).fetchone()
    return bool(row and row["skip_label"])


# what re-opens an already-reviewed commit: a reply under a finding, an operator, the
# priority box (a mention or review request on a reviewed commit changes nothing)
REOPENING = frozenset({Trigger.REVIEW_REPLY, Trigger.MANUAL, Trigger.PRIORITY_REQUEST})


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

    `Trigger.CI_FAILED` (a check failed after the head's APPROVE, ci.py) only ever re-opens
    a finished head for the light CI-check run: it never creates a head or takes one over.
    A person asking for anything on a head whose CI check is waiting gets their run instead
    (the CI check is re-queued by its watch if still needed), or, while it runs, right after.
    """
    ts = ts or now()
    if trigger.value in SKIPPABLE_TRIGGERS and _has_skip_label(conn, repo, number):
        return "noop"
    existing = conn.execute(
        "SELECT id, status, priority, queue, trigger, reason FROM heads WHERE repo=? AND number=? AND sha=?",
        (repo, number, sha),
    ).fetchone()
    if existing and existing["queue"] == "audit":
        return "noop"  # the PR merged: its head belongs to the audit queue now
    if trigger == Trigger.CI_FAILED:
        # a reviewed head only, at the front of the line like any same-sha follow-up
        if existing is None or existing["status"] != HeadStatus.DONE:
            return "noop"
        return _requeue(conn, existing["id"], repo, number, sha, trigger, ts)
    priority = 1 if trigger.priority else 0
    if (
        existing
        and existing["status"] == HeadStatus.CLOSED
        and existing["reason"] == SKIP_LABEL_REASON
    ):
        # closed only because of a skip label: the label is gone (ingest) or someone asked
        return _reopen_skipped(conn, cfg, existing["id"], repo, number, sha, trigger, ts)
    if existing and existing["trigger"] == Trigger.CI_FAILED.value and trigger in REOPENING:
        taken = _take_over_ci_check(conn, existing, trigger, ts)
        if taken:
            return taken
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
        if trigger in REOPENING and existing["status"] in (
            HeadStatus.DONE,
            HeadStatus.FAILED,
            HeadStatus.CLOSED,
        ):
            # same commit, already reviewed: a human answered a finding (or an operator asked),
            # so review it again with the threads in context and answer on them
            return _requeue(conn, existing["id"], repo, number, sha, trigger, ts)
        return "noop"
    supersede_other_heads(conn, repo, number, sha, ts)
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


def supersede_other_heads(
    conn: sqlite3.Connection, repo: str, number: int, sha: str, ts: str
) -> None:
    """End the PR's queued and running live heads for commits other than `sha`."""
    for row in conn.execute(
        "SELECT id FROM heads WHERE repo=? AND number=? AND sha<>? AND status IN ('queued','running') AND queue='live'",
        (repo, number, sha),
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


def _requeue(
    conn: sqlite3.Connection,
    head_id: int,
    repo: str,
    number: int,
    sha: str,
    trigger: Trigger,
    ts: str,
) -> str:
    """Re-open a finished head at the front of the line."""
    conn.execute(
        "UPDATE heads SET status='queued', priority=1, trigger=?, queued_at=?, eligible_at=?, finished_at=NULL, reason=NULL, attempts=0 WHERE id=?",
        (trigger.value, ts, ts, head_id),
    )
    event(
        conn,
        "head.requeued",
        repo=repo,
        number=number,
        detail=f"{sha[:8]} re-opened by {trigger.value}",
    )
    return "requeued"


def _reopen_skipped(
    conn: sqlite3.Connection,
    cfg: Config,
    head_id: int,
    repo: str,
    number: int,
    sha: str,
    trigger: Trigger,
    ts: str,
) -> str:
    """Re-open a head closed for a skip label as if it had just been queued: an automatic
    trigger waits out the debounce, a request goes to the front."""
    priority = 1 if trigger.priority else 0
    eligible = ts if priority else fmt_ts(parse_ts(ts) + timedelta(minutes=cfg.debounce_minutes))
    supersede_other_heads(conn, repo, number, sha, ts)
    conn.execute(
        "UPDATE heads SET status='queued', priority=?, trigger=?, queued_at=?, eligible_at=?, finished_at=NULL, reason=NULL, attempts=0 WHERE id=?",
        (priority, trigger.value, ts, eligible, head_id),
    )
    event(
        conn,
        "head.requeued",
        repo=repo,
        number=number,
        detail=f"{sha[:8]} skip label gone or overridden, trigger={trigger.value}",
    )
    return "requeued"


def _take_over_ci_check(
    conn: sqlite3.Connection, existing: sqlite3.Row, trigger: Trigger, ts: str
) -> str | None:
    """Someone re-opened (`REOPENING`) a head whose CI-check run is queued or running: the CI
    check must not swallow the request. A queued one becomes their run (the watch queues the
    CI check again after it, if still needed). A running one hands them the head: a retry of
    it is their run, and once it ends a re-review is queued (`scheduler.finish_run`); a reply
    is the exception, which the router holds until the head finishes and then queues as a
    conversation."""
    if existing["status"] == HeadStatus.QUEUED:
        conn.execute(
            "UPDATE heads SET trigger=?, eligible_at=? WHERE id=?",
            (trigger.value, ts, existing["id"]),
        )
        return "promoted"
    if existing["status"] == HeadStatus.RUNNING and trigger != Trigger.REVIEW_REPLY:
        run = conn.execute(
            "SELECT id FROM runs WHERE head_id=? AND status IN ('spawned','running') ORDER BY id DESC LIMIT 1",
            (existing["id"],),
        ).fetchone()
        if run:
            conn.execute("UPDATE heads SET trigger=? WHERE id=?", (trigger.value, existing["id"]))
            kv_set(conn, f"rereview.after_run:{run['id']}", "1")
            return "promoted"
    return None


# heads the ingest pass creates on its own; every other trigger is someone asking for a review
AUTOMATIC_TRIGGERS = frozenset({Trigger.NEW_PR.value, Trigger.NEW_PUSH.value})


def close_pr_heads(
    conn: sqlite3.Connection,
    repo: str,
    number: int,
    reason: str,
    *,
    ts: str | None = None,
    keep_requested_sha: str | None = None,
) -> int:
    """Close the PR's queued and running heads. A head for `keep_requested_sha` that someone
    asked for (not a new PR/push head) stays open."""
    ts = ts or now()
    rows = conn.execute(
        "SELECT id, sha, trigger FROM heads WHERE repo=? AND number=? AND status IN ('queued','running') AND queue='live'",
        (repo, number),
    ).fetchall()
    rows = [
        r
        for r in rows
        if not (r["sha"] == keep_requested_sha and r["trigger"] not in AUTOMATIC_TRIGGERS)
    ]
    for row in rows:
        conn.execute(
            "UPDATE heads SET status='closed', finished_at=?, reason=? WHERE id=?",
            (ts, reason, row["id"]),
        )
        event(conn, "head.closed", repo=repo, number=number, detail=reason)
    return len(rows)


def skip_pr_heads(conn: sqlite3.Connection, repo: str, number: int, sha: str, *, ts: str) -> int:
    """The PR carries a skip label: end the reviews nobody asked for. Heads of older commits
    are superseded (as a push would; a requested review there may still publish), a new PR/push
    head of `sha` closes (re-opened when the label goes), and a queued or running CI check goes
    back to the review it re-opened, its watch ended. A review someone asked for on `sha` runs
    on. Returns how many heads ended."""
    from . import ci  # ci imports this module

    supersede_other_heads(conn, repo, number, sha, ts)
    n = 0
    for row in conn.execute(
        "SELECT id, trigger FROM heads WHERE repo=? AND number=? AND sha=? AND status IN ('queued','running') AND queue='live'",
        (repo, number, sha),
    ).fetchall():
        if row["trigger"] == Trigger.CI_FAILED.value:
            conn.execute(
                "UPDATE heads SET status='done', finished_at=?, reason=? WHERE id=?",
                (ts, SKIP_LABEL_REASON, row["id"]),
            )
            ci.release(conn, repo, number, sha, judged=[], pending=False)
        elif row["trigger"] in AUTOMATIC_TRIGGERS:
            conn.execute(
                "UPDATE heads SET status='closed', finished_at=?, reason=? WHERE id=?",
                (ts, SKIP_LABEL_REASON, row["id"]),
            )
        else:
            continue
        n += 1
    return n


def ingest_repo(
    conn: sqlite3.Connection, cfg: Config, gh: Gh, repo: str, prs: list[OpenPr]
) -> dict[str, int]:
    """Reconcile one repo's open PR set into the DB. Pure DB work; PRs already fetched."""
    stats = {
        "created": 0,
        "promoted": 0,
        "requeued": 0,
        "noop": 0,
        "closed": 0,
        "draft": 0,
        "skipped": 0,
        "ignored": 0,
    }
    ts = now()
    open_numbers = {p.number for p in prs}
    with tx(conn):
        for p in prs:
            label = skip_label(cfg, p.labels)
            if not label and _has_skip_label(conn, repo, p.number):
                # label removed: a later one gets its note again (queue_status.py)
                conn.execute("DELETE FROM kv WHERE key=?", (skip_noted_key(repo, p.number),))
            conn.execute(
                "INSERT INTO prs (repo, number, head_sha, title, author, is_draft, skip_label, state, updated_at) VALUES (?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(repo, number) DO UPDATE SET head_sha=excluded.head_sha, title=excluded.title, author=excluded.author, is_draft=excluded.is_draft, skip_label=excluded.skip_label, state='open', updated_at=excluded.updated_at",
                (
                    repo,
                    p.number,
                    p.head_sha,
                    p.title,
                    p.author,
                    int(p.is_draft),
                    int(bool(label)),
                    "open",
                    p.updated_at or ts,
                ),
            )
            if p.author.lower() in IGNORED_AUTHORS:
                stats["ignored"] += 1
                continue
            if label:
                # checked before the draft rule: a labelled draft gets no draft comment either
                if skip_pr_heads(conn, repo, p.number, p.head_sha, ts=ts):
                    stats["skipped"] += 1
                    event(conn, "head.skip_label", repo=repo, number=p.number, detail=label)
                continue
            if p.is_draft:
                # drafts are not reviewed on their own, but a review requested on the draft's
                # current commit (its gate comment's boxes) runs to the end: closing it here
                # cancelled it minutes after it started (dashpay/platform#5285, 2026-10-05)
                if close_pr_heads(
                    conn, repo, p.number, "pr_draft", ts=ts, keep_requested_sha=p.head_sha
                ):
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
