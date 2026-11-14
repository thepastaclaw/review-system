"""Stable queue gate comments: every queued PR gets an immediate comment with a queue link
and a checkbox that requests a priority review.

Runs as a daemon task. Each pass reads the PR's gate comment back (one GET per queued head;
the comment id is cached in `kv` so no search is needed after the first write), honours a
ticked priority box, and rewrites only when the rendered body differs from what GitHub has.
Comparing against the live body, not a cache, means a worker-written "in progress"/"failed"
body on a requeued head is always replaced.

Each PR gets one body per pass, from one of two renderers: `deferred_body` for a draft and for
a new head waiting out the push debounce (not started), `queue_body` for every other queued head.
Both used to render a debounced head, one after the other in the same pass, so its comment
flipped between the two every pass (dashpay/platform#5237, 2026-10-01).

The body holds nothing that moves with the queue (position, ETA, degraded mode): every edit
wakes whatever watches the PR, so the live queue stays on the dashboard.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from typing import Any

from . import github
from .config import Config
from .db import event, kv_get, kv_set, now, tx
from .gh import Gh
from .models import HeadStatus, Trigger

log = logging.getLogger(__name__)

PRIORITY_BOX = (
    "- [ ] **Request priority review** — click to move this review to the front of the queue."
)
NORMAL_BOX = "- [ ] **Request normal review** — click when the PR is ready for review."
DEFERRED_MARKER = "<!-- thepastaclaw-review-deferred v1 -->"
PRIORITY_CHECKED_RE = re.compile(
    r"^\s*-\s*\[[xX]\]\s*\*\*Request priority review\*\*", re.MULTILINE
)
MAX_WRITES_PER_PASS = 25  # stay well under GitHub's content-creation secondary limit


def queue_body(
    sha: str,
    *,
    priority: bool,
) -> str:
    """Rendered queue comment; its text stays stable while the queue moves."""
    if priority:
        head = "⚡ Priority review — starts as soon as a slot frees"
    else:
        head = "🕓 Queued for automated review"
    lines = [
        github.GATE_MARKER,
        f"{head} (commit {sha[:8]}) · [queue and estimated start]({github.DASHBOARD_URL})",
    ]
    if not priority:
        lines += ["", PRIORITY_BOX]
    return "\n".join(lines)


def queued_order(conn: sqlite3.Connection, *, ts: str | None = None) -> list[sqlite3.Row]:
    """Queued heads in the order the scheduler will pick them: eligible priority first, then
    eligible normal, then heads still in debounce/backoff (by when they become eligible).
    A reviewed head re-opened for its CI check (ci.py) is not in line for a review: its PR
    keeps the gate comment the review left."""
    ts = ts or now()
    return conn.execute(
        "SELECT id, repo, number, sha, priority, eligible_at FROM heads WHERE status='queued' AND queue='live' "
        "AND trigger<>? ORDER BY (eligible_at > ?) ASC, priority DESC, CASE WHEN eligible_at > ? THEN eligible_at ELSE queued_at END ASC",
        (Trigger.CI_FAILED.value, ts, ts),
    ).fetchall()


def _comment_key(repo: str, number: int) -> str:
    return f"queue.comment_id:{repo}#{number}"


def priority_requested(comment_body: str) -> bool:
    return bool(PRIORITY_CHECKED_RE.search(comment_body))


def normal_requested(comment_body: str) -> bool:
    return bool(
        re.search(r"^\s*-\s*\[[xX]\]\s*\*\*Request normal review\*\*", comment_body, re.MULTILINE)
    )


def deferred_body(sha: str, *, draft: bool, debounce_minutes: int) -> str:
    reason = (
        "this PR is a draft"
        if draft
        else f"the new head is waiting for the {debounce_minutes}-minute push debounce"
    )
    return "\n".join(
        [
            github.GATE_MARKER,
            DEFERRED_MARKER,
            f"🕓 Review not started yet because {reason}.",
            "",
            NORMAL_BOX,
            PRIORITY_BOX,
            "",
            f"_Commit {sha[:8]}. Normal review starts when eligible; priority review starts as soon as a slot is available._",
        ]
    )


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

    # Drafts and heads in push debounce need an actionable explanation too. A checked box is
    # an explicit request and is converted into a normal or priority queue entry. This pass is
    # the only writer for the heads it selects; the queue pass below leaves them alone. A head
    # waiting out a retry backoff (attempts > 0) or a priority head is not in debounce: it
    # keeps the queue body. A draft whose commit is being or has
    # been reviewed keeps the worker's body, but a box ticked on an older draft body still
    # counts; after a failed review the draft body's boxes are the only way to retry.
    deferred = conn.execute(
        "SELECT p.repo,p.number,p.head_sha,p.is_draft,h.id,h.priority,h.status FROM prs p "
        "LEFT JOIN heads h ON h.repo=p.repo AND h.number=p.number AND h.sha=p.head_sha "
        "WHERE p.state='open' AND p.skip_label=0 AND ((p.is_draft=1 AND (h.status IS NULL OR h.status!='queued')) "
        "OR (h.status='queued' AND h.eligible_at>? AND h.priority=0 AND h.attempts=0))",
        (ts,),
    ).fetchall()
    in_debounce = {p["id"] for p in deferred if p["id"] is not None}
    for p in deferred:
        try:
            c = _read_comment(conn, gh, cfg, p["repo"], p["number"])
            body = str(c.get("body") or "") if c else ""
            if normal_requested(body) or priority_requested(body):
                trigger = (
                    Trigger.PRIORITY_REQUEST if priority_requested(body) else Trigger.NORMAL_REQUEST
                )
                from .ingest import enqueue_head

                with tx(conn):
                    enqueue_head(conn, cfg, p["repo"], p["number"], p["head_sha"], trigger, ts=ts)
                continue
            if p["status"] in (HeadStatus.RUNNING, HeadStatus.DONE):
                continue
            rendered = deferred_body(
                p["head_sha"], draft=bool(p["is_draft"]), debounce_minutes=cfg.debounce_minutes
            )
            if c and github.GATE_MARKER in body and body == rendered:
                continue
            if stats["written"] < MAX_WRITES_PER_PASS:
                _write(conn, gh, p["repo"], p["number"], c, rendered)
                stats["written"] += 1
            else:
                stats["deferred"] += 1
        except Exception as exc:
            log.warning("deferred comment %s#%s failed: %s", p["repo"], p["number"], exc)
    _note_skip_labels(conn, cfg, gh, stats)
    for h in rows:
        if h["id"] not in comments or h["id"] in in_debounce:
            continue  # read failed this pass (try again next time), or the deferred body's
        body = queue_body(
            h["sha"],
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


SKIP_LABEL_NOTE = "a label on the PR turns automatic review off"


def _note_skip_labels(conn: sqlite3.Connection, cfg: Config, gh: Gh, stats: dict[str, int]) -> None:
    """A labelled PR's gate comment, left by a review that ended or never started ("queued",
    "starts in 30 min"), says the PR is not reviewed instead. Never creates one: the label asks
    for quiet. Left alone while a run is in flight (a requested review, or one still winding
    down after ingest closed its head) and when a requested review finished on the commit (its
    body stands). Each (PR, commit) is checked once, so a labelled PR costs no read per pass."""
    from .ingest import skip_noted_key

    rows = conn.execute(
        "SELECT p.repo,p.number,p.head_sha FROM prs p WHERE p.state='open' AND p.skip_label=1 "
        "AND NOT EXISTS (SELECT 1 FROM heads h WHERE h.repo=p.repo AND h.number=p.number "
        "AND h.queue='live' AND (h.status IN ('queued','running') "
        "OR (h.sha=p.head_sha AND h.status IN ('done','failed')))) "
        "AND NOT EXISTS (SELECT 1 FROM runs r JOIN heads h ON h.id=r.head_id "
        "WHERE h.repo=p.repo AND h.number=p.number AND r.status IN ('spawned','running'))"
    ).fetchall()
    for p in rows:
        key = skip_noted_key(p["repo"], p["number"])
        if kv_get(conn, key) == p["head_sha"]:
            continue
        if stats["written"] >= MAX_WRITES_PER_PASS:
            stats["deferred"] += 1
            continue
        try:
            c = _read_comment(conn, gh, cfg, p["repo"], p["number"])
            body = str(c.get("body") or "") if c else ""
            if normal_requested(body) or priority_requested(body):
                # a box ticked before the label landed is a request: honour it, never erase it
                from .ingest import enqueue_head

                trigger = (
                    Trigger.PRIORITY_REQUEST if priority_requested(body) else Trigger.NORMAL_REQUEST
                )
                with tx(conn):
                    res = enqueue_head(conn, cfg, p["repo"], p["number"], p["head_sha"], trigger)
                if res != "noop":
                    continue
                # nothing to re-open (the box predates a force-push): note it, once
            rendered = github.gate_body(
                "closed",
                p["head_sha"],
                reason=f"{SKIP_LABEL_NOTE}; `@{cfg.bot_login} review` still asks for one",
            )
            if c and body != rendered:
                _write(conn, gh, p["repo"], p["number"], c, rendered)
                stats["written"] += 1
            with tx(conn):
                kv_set(conn, key, p["head_sha"])
        except Exception as exc:
            log.warning("skip-label comment %s#%s failed: %s", p["repo"], p["number"], exc)


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
