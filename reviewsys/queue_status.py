"""Queue-position gate comments: every queued PR gets an immediate comment with its position,
an ETA, and a checkbox that requests a priority review.

Runs as a daemon task. Each pass reads the PR's gate comment back (one GET per queued head;
the comment id is cached in `kv` so no search is needed after the first write), honours a
ticked priority box, and rewrites only when the rendered body differs from what GitHub has.
Comparing against the live body, not a cache, means a worker-written "in progress"/"failed"
body on a requeued head is always replaced.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from typing import Any

from . import github
from .config import Config
from .db import event, kv_get, kv_set, now, now_dt, parse_ts, tx
from .gh import Gh
from .models import Trigger
from .status import median_run_minutes

log = logging.getLogger(__name__)

PRIORITY_BOX = "- [ ] **Request priority review** — tick this box and the review moves to the front of the queue."
PRIORITY_CHECKED_RE = re.compile(
    r"^\s*-\s*\[[xX]\]\s*\*\*Request priority review\*\*", re.MULTILINE
)
DEFAULT_RUN_MINUTES = 120
MAX_WRITES_PER_PASS = 25  # stay well under GitHub's content-creation secondary limit


def _fmt_minutes(m: float) -> str:
    if m < 55:
        return f"~{max(5, int(round(m / 5) * 5))} min"
    h = m / 60
    return f"~{h:.1f} h" if h < 3 else f"~{h:.0f} h"


def queue_body(
    sha: str, *, position: int, eta_minutes: float, run_minutes: float, priority: bool
) -> str:
    """Rendered queue comment. Deliberately omits the queue total so a PR's body only changes
    when its own position or ETA changes."""
    if priority:
        head = f"⚡ Priority review — {_ordinal(position)} in line, starts as soon as a slot frees"
    else:
        head = f"🕓 Queued for automated review — {_ordinal(position)} in line, estimated start in {_fmt_minutes(eta_minutes)}"
    lines = [
        github.GATE_MARKER,
        f"{head} (commit {sha[:8]})",
        f"_Estimated review time once started: {_fmt_minutes(run_minutes)} (two-phase automated review; median of recent runs)._",
    ]
    if not priority:
        lines += ["", PRIORITY_BOX]
    return "\n".join(lines)


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def queued_order(conn: sqlite3.Connection, *, ts: str | None = None) -> list[sqlite3.Row]:
    """Queued heads in the order the scheduler will pick them: eligible priority first, then
    eligible normal, then heads still in debounce/backoff (by when they become eligible)."""
    ts = ts or now()
    return conn.execute(
        "SELECT id, repo, number, sha, priority, eligible_at FROM heads WHERE status='queued' "
        "ORDER BY (eligible_at > ?) ASC, priority DESC, CASE WHEN eligible_at > ? THEN eligible_at ELSE queued_at END ASC",
        (ts, ts),
    ).fetchall()


def _comment_key(repo: str, number: int) -> str:
    return f"queue.comment_id:{repo}#{number}"


def priority_requested(comment_body: str) -> bool:
    return bool(PRIORITY_CHECKED_RE.search(comment_body))


def _promote(conn: sqlite3.Connection, head: sqlite3.Row, actor: str) -> bool:
    with tx(conn):
        n = conn.execute(
            "UPDATE heads SET priority=1, trigger=?, eligible_at=? WHERE id=? AND status='queued' AND priority=0",
            (Trigger.PRIORITY_REQUEST.value, now(), head["id"]),
        ).rowcount
        if n:
            event(
                conn,
                "head.priority_requested",
                repo=head["repo"],
                number=head["number"],
                detail=f"{head['sha'][:8]} checkbox ticked by {actor or 'unknown'}",
            )
    return bool(n)


def _read_comment(
    conn: sqlite3.Connection, gh: Gh, cfg: Config, repo: str, number: int
) -> dict[str, Any] | None:
    """The PR's gate comment, by cached id when known, else by marker search."""
    key = _comment_key(repo, number)
    cid = kv_get(conn, key)
    if cid:
        try:
            c = gh.api(f"repos/{repo}/issues/comments/{cid}")
            if isinstance(c, dict) and github.GATE_MARKER in str(c.get("body") or ""):
                return c
        except Exception as exc:
            log.info("cached gate comment %s#%s id=%s unusable: %s", repo, number, cid, exc)
    c = github.find_gate_comment(gh, repo, number, cfg.bot_login)
    with tx(conn):
        if c:
            kv_set(conn, key, str(c["id"]))
        elif cid:
            conn.execute("DELETE FROM kv WHERE key=?", (key,))
    return c


def _write(
    conn: sqlite3.Connection,
    gh: Gh,
    repo: str,
    number: int,
    existing: dict[str, Any] | None,
    body: str,
) -> None:
    if existing:
        gh.api(
            f"repos/{repo}/issues/comments/{existing['id']}", method="PATCH", body={"body": body}
        )
        return
    created = gh.api(f"repos/{repo}/issues/{number}/comments", method="POST", body={"body": body})
    if isinstance(created, dict) and created.get("id"):
        with tx(conn):
            kv_set(conn, _comment_key(repo, number), str(created["id"]))


def update_queue_comments(conn: sqlite3.Connection, cfg: Config, gh: Gh) -> dict[str, int]:
    """Post or refresh the queue comment on every queued PR and honour ticked priority boxes."""
    stats = {"written": 0, "promoted": 0, "deferred": 0}
    ts = now()
    run_min = median_run_minutes(conn) or DEFAULT_RUN_MINUTES
    slots = max(cfg.max_concurrent, 1)
    rows = queued_order(conn, ts=ts)
    comments: dict[int, dict[str, Any] | None] = {}
    for h in rows:
        try:
            c = _read_comment(conn, gh, cfg, h["repo"], h["number"])
        except Exception as exc:
            log.warning("gate comment read %s#%s failed: %s", h["repo"], h["number"], exc)
            continue
        comments[h["id"]] = c
        ticked = c is not None and priority_requested(str(c.get("body") or ""))
        if ticked and not h["priority"] and _promote(conn, h, _editor(gh, c or {})):
            stats["promoted"] += 1
    if stats["promoted"]:
        rows = queued_order(conn, ts=ts)
    active = conn.execute(
        "SELECT COUNT(*) AS n FROM runs WHERE status IN ('spawned','running')"
    ).fetchone()["n"]
    for pos, h in enumerate(rows, 1):
        if h["id"] not in comments:
            continue  # read failed this pass; try again next time
        # runs ahead of this head in its slot, plus the remainder of whatever occupies the slot
        # now (half a run) when all slots are busy, plus any debounce/backoff still to elapse
        ahead = ((pos - 1) // slots) * run_min
        busy = run_min / 2 if active >= slots else 0.0
        wait = max(0.0, (parse_ts(h["eligible_at"]) - now_dt()).total_seconds() / 60)
        body = queue_body(
            h["sha"],
            position=pos,
            eta_minutes=max(ahead + busy, wait),
            run_minutes=run_min,
            priority=bool(h["priority"]),
        )
        existing = comments[h["id"]]
        if existing and str(existing.get("body") or "") == body:
            continue
        if stats["written"] >= MAX_WRITES_PER_PASS:
            stats["deferred"] += 1
            continue
        try:
            _write(conn, gh, h["repo"], h["number"], existing, body)
        except Exception as exc:
            log.warning("queue comment %s#%s failed: %s", h["repo"], h["number"], exc)
            continue
        stats["written"] += 1
    return stats


def _editor(gh: Gh, comment: dict[str, Any]) -> str:
    """Login of the most recent editor of the comment (whoever ticked the box), or ''."""
    try:
        data = gh.graphql(
            "query($id: ID!) { node(id: $id) { ... on IssueComment { userContentEdits(first: 20) { nodes { editedAt editor { login } } } } } }",
            {"id": str(comment.get("node_id") or "")},
        )
        nodes = ((data or {}).get("node") or {}).get("userContentEdits", {}).get("nodes") or []
        latest = max(
            (n for n in nodes if isinstance(n, dict)),
            key=lambda n: str(n.get("editedAt") or ""),
            default=None,
        )
        return str(((latest or {}).get("editor") or {}).get("login") or "")
    except Exception as exc:
        log.info("editor lookup failed: %s", exc)
        return ""
