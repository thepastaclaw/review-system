"""Verdict labels: mirror the bot's standing review verdict onto a PR label.

GitHub only counts reviews from accounts with write access toward the merge decision, so
on repos where the bot has triage its APPROVE / REQUEST_CHANGES is invisible to
`review:approved` searches and to the merge box. A label is the only per-reviewer filter
triage can set, so a repo that defines the `pastaclaw:*` labels gets exactly one of them
mirroring the bot's latest verdict on the live head, and none while no verdict stands.

The label is a *reconciliation* of database state, not an event hook: the wanted label
for a PR is derived from its live commit (newest head row, cross-checked against GitHub's
head sha) and the latest row in `reviews` for that sha, and `reconcile()` compares it with what GitHub shows and fixes the
difference. Running that on a daemon tick covers every path that can change either side
(a review posted, a same-sha follow-up, a push seen by ingest or by the router, a revert
to an already-reviewed commit, a run that published after being superseded, a run that
died between posting and labelling, a bot-authored PR whose canonical verdict moved while
GitHub's transport state could not)
without each of them having to remember to touch the label.

Repos that have not created the labels are opted out: their add fails (404 on write
repos, "not permitted to create labels" on triage repos), the failure is recorded as
`label.sync_failed` and the repo is skipped until the daemon restarts.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import timedelta
from typing import Any
from urllib.parse import quote

from .db import event, fmt_ts, kv_get, kv_set, now_dt, parse_ts, tx
from .gh import Gh
from .models import FailKind, ReviewError
from .publish import canonical_event

log = logging.getLogger(__name__)

PREFIX = "pastaclaw:"
VERDICT_LABELS = {
    "APPROVE": PREFIX + "approved",
    "REQUEST_CHANGES": PREFIX + "changes-requested",
    "COMMENT": PREFIX + "commented",
}
# how far the cursor trails the start of a pass: a `reviews` row can commit with a
# `posted_at` older than the commit (its writer takes `now()` before waiting for the write
# lock), so re-check a short window every pass; sync is idempotent
CURSOR_OVERLAP = timedelta(seconds=60)
# after a transient GitHub failure, wait this long before the next pass
RETRY_BACKOFF = timedelta(minutes=5)


def wanted(conn: sqlite3.Connection, repo: str, number: int) -> str | None:
    """The label this PR should carry right now, or None: the bot's latest review of the PR's
    live head, if any. The live head is the newest head row (ingest and the router both create
    one for a new commit) and it must agree with `prs.head_sha` (GitHub's view, refreshed every
    poll): when they differ the commit moved without a review being possible yet, or with no
    new head row at all (revert to an already-reviewed sha, push while draft), and no verdict
    stands. A PR with no `prs` row (ad hoc review of an unlisted repo) has only the head."""
    pr = conn.execute(
        "SELECT state, head_sha FROM prs WHERE repo=? AND number=?", (repo, number)
    ).fetchone()
    if pr and pr["state"] != "open":
        return None
    head = conn.execute(
        "SELECT sha FROM heads WHERE repo=? AND number=? ORDER BY id DESC LIMIT 1", (repo, number)
    ).fetchone()
    if not head or (pr and pr["head_sha"] != head["sha"]):
        return None
    row = conn.execute(
        "SELECT event FROM reviews WHERE repo=? AND number=? AND sha=? ORDER BY posted_at DESC, id DESC LIMIT 1",
        (repo, number, head["sha"]),
    ).fetchone()
    if not row:
        return None
    return VERDICT_LABELS.get(canonical_event(str(row["event"])))


def _dirty_prs(conn: sqlite3.Connection, since: str) -> list[tuple[str, int]]:
    """PRs whose wanted label may have changed since `since`: a review row written, a head
    created or re-queued, or GitHub activity on the PR (`prs.updated_at` is GitHub's updatedAt:
    a push, a reopen, a close, a label change). With no cursor yet (first pass after deploy)
    every open PR is dirty once, which backfills labels for standing verdicts; closed history
    is never swept."""
    if not since:
        rows = conn.execute("SELECT repo, number FROM prs WHERE state='open'").fetchall()
    else:
        rows = conn.execute(
            "SELECT repo, number FROM reviews WHERE posted_at >= ?"
            " UNION SELECT repo, number FROM heads WHERE queued_at >= ?"
            " UNION SELECT repo, number FROM prs WHERE updated_at >= ?",
            (since, since, since),
        ).fetchall()
    return [(str(r["repo"]), int(r["number"])) for r in rows]


def reconcile(
    conn: sqlite3.Connection,
    gh: Gh,
    *,
    disabled: set[str],
    since_key: str = "labels.reconciled_at",
) -> int:
    """Daemon task: bring the verdict label of every PR touched since the last pass in line
    with `wanted()`. Returns the number of PRs whose labels changed. A repo whose label add
    fails is added to `disabled` (it has not created the labels) and skipped from then on;
    a transient GitHub failure holds the cursor and backs off before the pass is retried."""
    start = now_dt()
    retry_at = kv_get(conn, since_key + ".retry_at")
    if retry_at and parse_ts(retry_at) > start:
        return 0
    since = kv_get(conn, since_key, "") or ""
    changed = 0
    deferred = False
    for repo, number in _dirty_prs(conn, since):
        if repo in disabled:
            continue
        want = wanted(conn, repo, number)
        try:
            if sync(gh, repo, number, want):
                changed += 1
                with tx(conn):
                    event(
                        conn, "label.synced", repo=repo, number=number, detail=want or "(cleared)"
                    )
        except ReviewError as exc:
            if exc.kind is FailKind.INFRA:
                log.warning("verdict label sync for %s#%s deferred: %s", repo, number, exc)
                deferred = True
                continue
            disabled.add(repo)
            log.warning("verdict labels disabled for %s until restart: %s", repo, exc)
            with tx(conn):
                event(conn, "label.sync_failed", repo=repo, number=number, detail=str(exc)[:300])
    with tx(conn):
        if deferred:
            kv_set(conn, since_key + ".retry_at", fmt_ts(start + RETRY_BACKOFF))
        else:
            kv_set(conn, since_key, fmt_ts(start - CURSOR_OVERLAP))
    return changed


def sync(gh: Gh, repo: str, number: int, want: str | None) -> bool:
    """Make the PR carry exactly `want` among the `pastaclaw:*` labels (None: none of them).
    Returns True when a label was added or removed. Raises ReviewError on API failure."""
    current = _current(gh, repo, number)
    if want in current and len(current) == 1:
        return False
    for name in sorted(current - ({want} if want else set())):
        gh.api(f"repos/{repo}/issues/{number}/labels/{quote(name, safe='')}", method="DELETE")
    if want and want not in current:
        gh.api(f"repos/{repo}/issues/{number}/labels", method="POST", body={"labels": [want]})
    return bool(current or want)


def _current(gh: Gh, repo: str, number: int) -> set[str]:
    data: Any = gh.api(f"repos/{repo}/issues/{number}/labels?per_page=100", paginate=True)
    if not isinstance(data, list):
        return set()
    return {
        str(lbl.get("name"))
        for lbl in data
        if isinstance(lbl, dict) and str(lbl.get("name") or "").startswith(PREFIX)
    }
