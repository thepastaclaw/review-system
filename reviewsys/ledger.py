"""The issue ledger (schema v9): every issue the bot raised on a PR, its identity, and where it
stands. One `issues` row per (repo, number, hash), folded from its `issue_events` history.

Identity is the `finding_hash` of the title the issue was first posted under: byte-compatible
with the markers on GitHub (`contract.finding_hash`), assigned once, never recomputed.

Only code writes it, never a model:
- `record_published`: what a publication put on GitHub (new or re-posted issues), and the
  earlier issues the same verifier set kept (`seen`, the issue is carried);
- `record_answers`: the outcome posted on a finding thread (the full path's thread answers,
  the conversation lane's replies, and so its concessions);
- `sync_threads`: what the PR's threads say (a legacy finding this database never posted,
  a thread a human resolved by hand);
- `transition`: an explicit status change (v12 code rules, an operator);
- `raise_severity`: an open issue the v12 verifier now confirmed as blocking (raised in place);
- `reconcile`: once per PR before its first policy v12 round, the backfilled standing issues
  that can no longer stand (v12 design §3).

Every write appends events under a `key` naming the fact (`posted:<run>:<hash>`, ...) and
re-folds the issues they touch (`_fold`), so the history decides the row, whatever order the
facts were recorded in. That is what lets `backfill` replay the history of PRs reviewed before
the ledger existed, before or after the live write-through began, and run any number of times:
it derives the same keys for the same facts, so it only fills gaps.

Under policy v11 the ledger is write-only: nothing reads it to decide anything, and a failed
write never fails a run (`safe_write`). Under policy v12 (verifier contract "v12") it is what
the verdict, the thread lane's work set and the verifier's matching read."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sqlite3
import subprocess
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import github
from .contract import SEVERITIES, Finding
from .db import event, kv_get, kv_set, now, tx
from .dedupe import FINDING_MARKER_RE
from .publish import shown_status

THREAD_ANSWER_PREFIX = "<!-- thepastaclaw-thread-answer v1"

log = logging.getLogger(__name__)

OPEN = "open"
STATUSES = ("open", "fixed", "outdated", "withdrawn", "deferred", "deferral_pending", "resolved")
# still holds up the verdict (a deferral nobody with the authority made is still pending)
STANDING = frozenset({OPEN, "deferral_pending"})
# what a policy v12 verdict counts: a thread a human resolved by hand still counts (only an
# explicit outcome lifts a finding)
COUNTING = STANDING | {"resolved"}
CLOSED = frozenset(STATUSES) - STANDING
# closed by a call the code has to respect: such an issue only comes back when the code it
# points at changed since (`lines_changed`); a fixed one that comes back is a regression
SETTLED = frozenset({"withdrawn", "deferred", "outdated", "resolved"})
SOURCES = ("thread", "verifier", "conversation", "backfill", "manual")

# a thread outcome (full-path answer or conversation reply) -> the status it leaves
OUTCOME_STATUS = {
    "STILL_VALID": OPEN,
    "FIX_PENDING": OPEN,
    "FIXED": "fixed",
    "OUTDATED": "outdated",
    "WITHDRAWN": "withdrawn",
    "INTENTIONALLY_DEFERRED": "deferred",
    "DEFERRAL_PENDING": "deferral_pending",
    # settled like a deferral: left to the maintainers, back only if its code changes
    "DISPUTED": "deferred",
}
# an answer whose outcome is on the thread: a reply that failed or was scrubbed changes nothing.
# `already_answered` means a bot answer with that status is there already: the status is the
# one this run's final verifier set gives (`publish._answer_outcome`), which the earlier answer
# on the thread said too, though not necessarily on this head
DELIVERED_ACTIONS = frozenset({"replied", "already_answered", "status_cleared", "no_thread"})
# `no_thread` (policy v12 thread lane): the issue has no open thread to answer on (a human
# resolved it, or it never had an inline comment), so the outcome stands without a reply
KINDS = frozenset({"thread", "posted", "seen", "outcome", "status", "severity"})

_BODY_MAX = 8000


@dataclass(slots=True)
class Issue:
    repo: str
    number: int
    hash: str
    file: str
    line_start: int | None
    line_end: int | None
    anchor_sha: str | None  # the commit `line_start..line_end` refer to
    severity: str
    category: str
    title: str
    body: str
    status: str
    status_sha: str | None
    status_source: str
    opened_run: int | None  # None: a finding this database never posted (legacy thread)
    opened_sha: str | None
    last_seen_run: int | None
    last_seen_sha: str | None
    closed_sha: str | None
    closed_reason: str | None
    thread_id: str | None
    comment_id: int | None
    created_at: str
    updated_at: str

    @property
    def standing(self) -> bool:
        return self.status in STANDING

    @property
    def settled(self) -> bool:
        return self.status in SETTLED

    @property
    def blocking(self) -> bool:
        return self.severity == "blocking"

    def as_finding(self) -> Finding:
        """The issue as a finding carried into a round (not re-posted)."""
        return Finding(
            file=self.file,
            title=self.title,
            body=self.body,
            severity=self.severity,
            category=self.category,
            line_start=self.line_start,
            line_end=self.line_end,
            source="carried",
            prior_hash=self.hash,
        )


_COLUMNS = tuple(f.name for f in dataclasses.fields(Issue))
_UPSERT = (
    f"INSERT OR REPLACE INTO issues ({', '.join(_COLUMNS)}) "
    f"VALUES ({', '.join('?' * len(_COLUMNS))})"
)


@dataclass(frozen=True, slots=True)
class Event:
    """One fact about one issue, before it is appended. `ts` defaults to now."""

    hash: str
    key: str
    kind: str
    source: str
    run_id: int | None
    sha: str | None
    detail: dict[str, Any] = field(default_factory=dict)
    ts: str | None = None


# ---- reading ----


def _issue(row: sqlite3.Row) -> Issue:
    return Issue(**{c: row[c] for c in _COLUMNS})


def load(conn: sqlite3.Connection, repo: str, number: int) -> dict[str, Issue]:
    """Every issue on the PR by hash, oldest first."""
    rows = conn.execute(
        "SELECT * FROM issues WHERE repo=? AND number=? ORDER BY created_at, rowid", (repo, number)
    ).fetchall()
    return {str(r["hash"]): _issue(r) for r in rows}


def get(conn: sqlite3.Connection, repo: str, number: int, h: str) -> Issue | None:
    row = conn.execute(
        "SELECT * FROM issues WHERE repo=? AND number=? AND hash=?", (repo, number, h)
    ).fetchone()
    return _issue(row) if row else None


def history(
    conn: sqlite3.Connection, repo: str, number: int, h: str | None = None
) -> list[dict[str, Any]]:
    """The PR's events (one issue's with `h`) in the order they fold."""
    rows = conn.execute(
        "SELECT * FROM issue_events WHERE repo=? AND number=? AND (? IS NULL OR hash=?) "
        "ORDER BY ts, id",
        (repo, number, h, h),
    )
    return [{**dict(r), "detail": json.loads(r["detail"] or "{}")} for r in rows]


def known(conn: sqlite3.Connection, repo: str, number: int) -> set[str]:
    """The hashes of the PR's issues."""
    return {
        str(r[0])
        for r in conn.execute("SELECT hash FROM issues WHERE repo=? AND number=?", (repo, number))
    }


# ---- folding ----


def _set_status(cur: Issue, status: str, ev: sqlite3.Row, *, source: str, reason: str) -> None:
    cur.status, cur.status_sha, cur.status_source = status, ev["sha"], source
    if status in CLOSED:
        cur.closed_sha, cur.closed_reason = ev["sha"], reason[:1000] or status
    else:
        cur.closed_sha = cur.closed_reason = None


def _new(ev: sqlite3.Row, d: dict[str, Any], *, status: str, opened: bool) -> Issue:
    sha, run = ev["sha"], ev["run_id"]
    return Issue(
        repo=ev["repo"],
        number=ev["number"],
        hash=ev["hash"],
        file=str(d.get("file") or ""),
        line_start=d.get("line_start"),
        line_end=d.get("line_end"),
        anchor_sha=sha if opened else None,
        severity=str(d.get("severity") or "nitpick"),
        category=str(d.get("category") or "general"),
        title=str(d.get("title") or ""),
        body=str(d.get("body") or ""),
        status=status,
        status_sha=sha,
        status_source=ev["source"],
        opened_run=run if opened else None,
        opened_sha=sha if opened else None,
        last_seen_run=run if opened else None,
        last_seen_sha=sha if opened else None,
        closed_sha=None,
        closed_reason=None,
        thread_id=d.get("thread_id"),
        comment_id=d.get("comment_id"),
        created_at=ev["ts"],
        updated_at=ev["ts"],
    )


def _apply(cur: Issue | None, ev: sqlite3.Row) -> Issue | None:
    """The issue after one event. Events about an issue nobody posted or saw are kept in the
    history but make no row."""
    d: dict[str, Any] = json.loads(ev["detail"] or "{}")
    kind, source = ev["kind"], ev["source"]
    if kind == "posted":
        if cur is None:
            return _new(ev, d, status=OPEN, opened=True)
        # posted again (a re-raise, or the same finding on a new comment): the new comment's
        # anchor and text are the issue's now, and it stands again whatever it was before
        for k in ("file", "line_start", "line_end", "severity", "category", "title", "body"):
            if k in d:
                setattr(cur, k, d[k])
        cur.anchor_sha, cur.last_seen_run, cur.last_seen_sha = ev["sha"], ev["run_id"], ev["sha"]
        if cur.opened_sha is None:
            cur.opened_run, cur.opened_sha = ev["run_id"], ev["sha"]
        if cur.status != OPEN:
            _set_status(cur, OPEN, ev, source=source, reason="")
    elif cur is None:
        # an open bot finding thread this database never posted (a legacy finding)
        if kind == "thread" and d.get("title") and not d.get("resolved"):
            d = {**d, "file": d.get("path"), "line_start": d.get("line"), "line_end": d.get("line")}
            return _new(ev, d, status=OPEN, opened=False)
        return None
    elif kind == "seen":
        cur.last_seen_run, cur.last_seen_sha = ev["run_id"], ev["sha"]
    elif kind == "thread":
        cur.thread_id = d.get("thread_id") or cur.thread_id
        cur.comment_id = d.get("comment_id") or cur.comment_id
        if d.get("resolved") and cur.standing:
            # a human resolved the thread without the bot conceding: still counts toward the
            # verdict under policy, but it is no longer an open discussion
            _set_status(cur, "resolved", ev, source="manual", reason="thread resolved on GitHub")
        elif d.get("resolved") is False and cur.status == "resolved":
            _set_status(cur, OPEN, ev, source="manual", reason="")
    elif kind == "outcome":
        cur.comment_id = cur.comment_id or d.get("comment_id")
        status = OUTCOME_STATUS.get(str(d.get("outcome") or ""))
        if status and d.get("delivered"):
            _set_status(cur, status, ev, source=source, reason=str(d.get("outcome")))
    elif kind == "status":
        status = str(d.get("status") or "")
        if status in STATUSES:
            _set_status(cur, status, ev, source=source, reason=str(d.get("reason") or ""))
    elif kind == "severity":
        if d.get("severity") in SEVERITIES:
            cur.severity = str(d["severity"])
    cur.updated_at = ev["ts"]
    return cur


def _refold(conn: sqlite3.Connection, repo: str, number: int, hashes: Iterable[str]) -> None:
    """Recompute the rows of `hashes` from their events, and each event's from/to status."""
    for h in sorted(set(hashes)):
        # in time order; within a second (timestamps are whole seconds), in the order recorded.
        # A backfilled fact that shares its second with a live one folds after it, though it may
        # have happened first: both are within the same second of the same run, which in
        # practice only pairs a run's own events (its publication, then its answers)
        rows = conn.execute(
            "SELECT * FROM issue_events WHERE repo=? AND number=? AND hash=? ORDER BY ts, id",
            (repo, number, h),
        ).fetchall()
        cur: Issue | None = None
        moves: list[tuple[str | None, str | None, int]] = []
        for r in rows:
            before = cur.status if cur else None
            cur = _apply(cur, r)
            after = cur.status if cur else None
            if (r["from_status"], r["to_status"]) != (before, after):
                moves.append((before, after, int(r["id"])))
        conn.executemany("UPDATE issue_events SET from_status=?, to_status=? WHERE id=?", moves)
        if cur is None:
            conn.execute(
                "DELETE FROM issues WHERE repo=? AND number=? AND hash=?", (repo, number, h)
            )
            continue
        conn.execute(_UPSERT, dataclasses.astuple(cur))


# ---- writing ----


def append(conn: sqlite3.Connection, repo: str, number: int, events: Iterable[Event]) -> int:
    """Record the events not already recorded (by key) and re-fold the issues they touch.
    Returns how many were new. Caller holds the write transaction."""
    ts = now()
    touched: set[str] = set()
    added = 0
    for e in events:
        if e.source not in SOURCES:
            raise ValueError(f"ledger: unknown source {e.source!r}")
        if e.kind not in KINDS:
            raise ValueError(f"ledger: unknown event kind {e.kind!r}")
        res = conn.execute(
            "INSERT INTO issue_events (ts, repo, number, hash, key, kind, run_id, sha, source, detail) "
            "VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(repo, number, key) DO NOTHING",
            (
                e.ts or ts,
                repo,
                number,
                e.hash,
                e.key,
                e.kind,
                e.run_id,
                e.sha,
                e.source,
                json.dumps(e.detail, sort_keys=True),
            ),
        )
        if res.rowcount:
            touched.add(e.hash)
            added += 1
    _refold(conn, repo, number, touched)
    return added


def _posted(
    f: Finding,
    *,
    run_id: int | None,
    sha: str,
    source: str,
    ts: str | None = None,
    h: str | None = None,
) -> Event:
    """`h`: the hash on record when it is not the title's (a legacy row)."""
    h = h or f.hash
    return Event(
        hash=h,
        key=f"posted:{run_id}:{h}",
        kind="posted",
        source=source,
        run_id=run_id,
        sha=sha,
        detail={
            "file": f.file,
            "line_start": f.line_start,
            "line_end": f.line_end,
            "severity": f.severity,
            "category": f.category,
            "title": f.title,
            "body": f.body[:_BODY_MAX],
        },
        ts=ts,
    )


def _seen(h: str, *, run_id: int | None, sha: str, source: str, ts: str | None = None) -> Event:
    return Event(
        hash=h, key=f"seen:{run_id}:{h}", kind="seen", source=source, run_id=run_id, sha=sha, ts=ts
    )


def record_published(
    conn: sqlite3.Connection,
    repo: str,
    number: int,
    *,
    run_id: int,
    sha: str,
    posted: Iterable[Finding],
    kept: Iterable[Finding] = (),
    deduped_comment_ids: Iterable[int | str] = (),
    source: str = "verifier",
) -> None:
    """A publication: `posted` went on GitHub as new comments (opening, or re-opening, their
    issues); `kept` is the rest of the verifier's set, which carries the earlier issues it
    names (by echoed `finding_hash`, else its own hash); `deduped_comment_ids` are earlier
    finding comments a kept finding was deduped onto. Caller holds the transaction."""
    posted = list(posted)
    events = [_posted(f, run_id=run_id, sha=sha, source=source) for f in posted]
    new = {f.hash for f in posted}
    comments = {
        str(r["hash"]): r["comment_id"]
        for r in conn.execute(
            "SELECT hash, comment_id FROM issues WHERE repo=? AND number=?", (repo, number)
        )
    }
    by_comment = {str(c): h for h, c in comments.items() if c}
    carried = {f.prior_hash or f.hash for f in kept}
    carried |= {by_comment[str(c)] for c in deduped_comment_ids if str(c) in by_comment}
    carried -= new
    events += [
        _seen(h, run_id=run_id, sha=sha, source=source) for h in sorted(carried) if h in comments
    ]
    append(conn, repo, number, events)


def _answer(
    item: dict[str, Any],
    *,
    run_id: int | None,
    sha: str | None,
    source: str,
    conceded: bool,
    ts: str | None = None,
    origin: str | None = None,
    key: str | None = None,
) -> Event | None:
    h, outcome = str(item.get("finding_hash") or ""), str(item.get("status") or "")
    if not h or outcome == "DUPLICATE_THREAD":
        return None
    action = str(item.get("action") or "")
    detail: dict[str, Any] = {
        "outcome": outcome,
        "action": action,
        "delivered": action in DELIVERED_ACTIONS,
        "comment_id": item.get("comment_id"),
    }
    if conceded:
        detail["conceded"] = True
    if origin:
        detail["origin"] = origin
    return Event(
        hash=h,
        key=key or f"answer:{run_id}:{h}:{outcome}",
        kind="outcome",
        source=source,
        run_id=run_id,
        sha=sha,
        detail=detail,
        ts=ts,
    )


def record_answers(
    conn: sqlite3.Connection,
    repo: str,
    number: int,
    *,
    run_id: int,
    sha: str,
    answered: Iterable[dict[str, Any]],
    source: str,
    conceded: Iterable[str] = (),
) -> None:
    """The outcomes a run posted on finding threads (the `thread.answered` items): each moves
    its issue to the status it states, once it is on the thread. `conceded` are the hashes the
    conversation lifted (they also get `conceded` rows). Outcomes on threads the ledger does not
    know are left out (`sync_threads` adds open legacy threads first). Caller holds the
    transaction."""
    lifted, have = set(conceded), known(conn, repo, number)
    events = [
        _answer(a, run_id=run_id, sha=sha, source=source, conceded=a.get("finding_hash") in lifted)
        for a in answered
    ]
    append(conn, repo, number, [e for e in events if e is not None and e.hash in have])


def _thread_facts(
    threads: list[dict[str, Any]], bot_login: str
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """(open, resolved): the PR's bot finding threads by hash. A hash with any unresolved
    thread is open; `resolved` only has hashes all of whose threads a human resolved. A thread
    whose latest bot answer retired the finding (withdrawn, fixed, no longer applies) was
    resolved by the bot itself, never by hand: if its issue stands again later (revived on
    changed code), that is no reason to count it as resolved."""
    open_ = github.finding_threads(threads, bot_login)
    resolved: dict[str, dict[str, Any]] = {}
    for t in threads:
        cs = t.get("comments") or []
        if not t.get("is_resolved") or not cs:
            continue
        if not github._is_bot(cs[0], bot_login) or _bot_retired(cs[1:], bot_login):
            continue
        m = FINDING_MARKER_RE.search(str(cs[0].get("body") or ""))
        if m and m.group(1) not in open_:
            resolved[m.group(1)] = {"thread_id": t.get("thread_id"), "comment_id": cs[0].get("id")}
    return open_, resolved


_RETIRING = frozenset({"WITHDRAWN", "FIXED", "OUTDATED"})


def _bot_retired(replies: list[dict[str, Any]], bot_login: str) -> bool:
    """The latest bot answer on a finding thread retired the finding (and resolved the
    thread with it)."""
    answers = [
        str(c.get("body") or "")
        for c in replies
        if github._is_bot(c, bot_login) and THREAD_ANSWER_PREFIX in str(c.get("body") or "")
    ]
    return bool(answers) and shown_status(answers[-1]) in _RETIRING


def sync_threads(
    conn: sqlite3.Connection,
    repo: str,
    number: int,
    *,
    run_id: int | None,
    sha: str | None,
    threads: list[dict[str, Any]],
    bot_login: str,
    source: str = "thread",
) -> int:
    """Bring the ledger in line with the PR's review threads (`github.review_threads`): an open
    bot finding thread it does not know becomes an open issue; thread ids are filled in; a
    standing issue whose every thread a human resolved is `resolved`, and one reopened on
    GitHub stands again. Only differences are recorded. Returns how many. Caller holds the
    transaction."""
    open_, resolved = _thread_facts(threads, bot_login)
    issues = load(conn, repo, number)
    ts = now()
    events: list[Event] = []

    def ev(h: str, detail: dict[str, Any]) -> None:
        key = f"thread:{run_id}:{h}" if run_id is not None else f"thread:{ts}:{h}"
        events.append(
            Event(
                hash=h, key=key, kind="thread", source=source, run_id=run_id, sha=sha, detail=detail
            )
        )

    for h, t in open_.items():
        ids = {"thread_id": t.get("thread_id"), "comment_id": t.get("comment_id")}
        cur = issues.get(h)
        if cur is None:
            if t.get("title"):
                keys = ("path", "line", "severity", "title")
                ev(
                    h,
                    {
                        **ids,
                        **{k: t.get(k) for k in keys},
                        "body": str(t.get("body") or "")[:_BODY_MAX],
                        "resolved": False,
                    },
                )
        elif cur.status == "resolved" or (cur.thread_id, cur.comment_id) != tuple(ids.values()):
            ev(h, {**ids, "resolved": False})
    for h, ids in resolved.items():
        cur = issues.get(h)
        if cur is not None and cur.standing:
            ev(h, {**ids, "resolved": True})
    return append(conn, repo, number, events)


def transition(
    conn: sqlite3.Connection,
    repo: str,
    number: int,
    h: str,
    status: str,
    *,
    run_id: int | None,
    sha: str | None,
    source: str,
    reason: str = "",
    key: str | None = None,
) -> Issue | None:
    """Set an issue's status explicitly. `key` names the fact (default: one per run, hash and
    status, so a run recording it twice is a no-op; without a run, one per call). Returns the
    issue after it, or None when the ledger does not know the hash. Caller holds the
    transaction."""
    if status not in STATUSES:
        raise ValueError(f"ledger: unknown status {status!r}")
    if get(conn, repo, number, h) is None:
        return None
    detail = {"status": status, "reason": reason[:1000]}
    once = run_id if run_id is not None else f"{now()}:{uuid.uuid4().hex[:8]}"
    ev = Event(
        hash=h,
        key=key or f"status:{once}:{h}:{status}",
        kind="status",
        source=source,
        run_id=run_id,
        sha=sha,
        detail=detail,
    )
    append(conn, repo, number, [ev])
    return get(conn, repo, number, h)


def record_seen(
    conn: sqlite3.Connection,
    repo: str,
    number: int,
    *,
    run_id: int,
    sha: str,
    hashes: Iterable[str],
    source: str,
) -> int:
    """Issues this round affirmed at `sha` (policy v12: a candidate restated them, or the
    thread lane judged them still valid): their `last_seen` moves to it. Unknown hashes are
    skipped. Caller holds the transaction."""
    have = known(conn, repo, number)
    return append(
        conn,
        repo,
        number,
        [_seen(h, run_id=run_id, sha=sha, source=source) for h in sorted(set(hashes)) if h in have],
    )


def raise_severity(
    conn: sqlite3.Connection,
    repo: str,
    number: int,
    h: str,
    severity: str,
    *,
    run_id: int,
    sha: str,
    source: str,
    reason: str = "",
) -> None:
    """An open issue's severity, raised in place (policy v12: the verifier CONFIRMED a
    candidate restating a non-blocking issue as blocking). Caller holds the transaction."""
    if severity not in SEVERITIES:
        raise ValueError(f"ledger: unknown severity {severity!r}")
    append(
        conn,
        repo,
        number,
        [
            Event(
                hash=h,
                key=f"severity:{run_id}:{h}:{severity}",
                kind="severity",
                source=source,
                run_id=run_id,
                sha=sha,
                detail={"severity": severity, "reason": reason[:1000]},
            )
        ],
    )


def attach_folded(
    conn: sqlite3.Connection,
    repo: str,
    number: int,
    *,
    run_id: int,
    sha: str,
    finding: Finding,
    comment_hash: str,
    comment_id: int | str | None,
    source: str = "verifier",
    reopens: Callable[[Issue], bool] = lambda i: True,
) -> None:
    """A new issue publishing folded onto an existing finding comment (dedupe): it is that
    comment's issue now, so it can be judged and withdrawn on that thread. A standing issue is
    affirmed (`seen`), a closed one re-opened when `reopens` allows it (a settled one only on
    changed code), and one the ledger does not know yet is recorded under the comment's hash
    and id. Caller holds the transaction."""
    cid = int(comment_id) if comment_id not in (None, "") else None
    h = comment_hash
    if not h and cid is not None:
        row = conn.execute(
            "SELECT hash FROM issues WHERE repo=? AND number=? AND comment_id=?",
            (repo, number, cid),
        ).fetchone()
        h = str(row["hash"]) if row else ""
    if not h:
        return
    cur = get(conn, repo, number, h)
    if cur is None:
        ev = _posted(finding, run_id=run_id, sha=sha, source=source, h=h)
        ev.detail["comment_id"] = cid
        append(conn, repo, number, [ev])
    elif cur.status in COUNTING:
        append(conn, repo, number, [_seen(h, run_id=run_id, sha=sha, source=source)])
    elif reopens(cur):
        transition(
            conn,
            repo,
            number,
            h,
            OPEN,
            run_id=run_id,
            sha=sha,
            source=source,
            reason="raised again; published onto its earlier thread",
        )


def _reconciled_key(repo: str, number: int) -> str:
    return f"ledger.reconciled:{repo}#{number}"


def moved_anchors(
    issues: Mapping[str, Issue],
    head: str,
    changed: Callable[[str, str, str, int | None, int | None], bool | None],
    *,
    open_threads: Iterable[str],
) -> dict[str, bool | None]:
    """`anchor_moved` at `head` for every issue the reconcile may outdate: counting, and with
    no open thread. Asked before the reconcile's write transaction, never inside it."""
    threads = set(open_threads)
    return {
        h: anchor_moved(i, head, changed)
        for h, i in issues.items()
        if i.status in COUNTING and h not in threads
    }


def reconcile(
    conn: sqlite3.Connection,
    repo: str,
    number: int,
    *,
    pr_open: bool,
    open_threads: Iterable[str],
    moved: Mapping[str, bool | None],
    run_id: int | None = None,
    sha: str | None = None,
) -> list[str]:
    """Once per PR, before its first policy v12 round: the backfilled history leaves most
    blockers `open` that vanished without an outcome (v11 reconciled them inside model lanes
    and recorded nothing when a lane dropped one). On a closed PR every standing issue is
    `outdated`. On an open PR, an issue with no open thread whose anchored code changed since
    it was last seen (`moved`, from `moved_anchors`) is `outdated`; everything else stays and
    is returned, for the first round's thread lane to judge. Records the list in kv
    (`reconciled`). Caller holds the transaction (after `sync_threads`, so thread facts are
    current)."""
    threads = set(open_threads)
    pending: list[str] = []
    for h, i in load(conn, repo, number).items():
        if i.status not in COUNTING:
            continue
        if not pr_open:
            reason = "the PR closed before policy v12 reconciled its ledger"
        elif h not in threads and moved.get(h) is True:
            reason = "reconciled for policy v12: no open thread, and the code at it changed"
        else:
            pending.append(h)
            continue
        transition(
            conn,
            repo,
            number,
            h,
            "outdated",
            run_id=run_id,
            sha=sha,
            source="backfill",
            reason=reason,
            key=f"reconcile:{h}",
        )
    kv_set(
        conn,
        _reconciled_key(repo, number),
        # `at` and the first `pending` (`pending_at_reconcile`; `pending` shrinks as the thread
        # lane judges) let a replay tell what the record said at an earlier moment
        json.dumps(
            {
                "run": run_id,
                "sha": sha,
                "pending": pending,
                "at": now(),
                "pending_at_reconcile": pending,
            }
        ),
    )
    return pending


def reconcile_all(
    conn: sqlite3.Connection,
    gh: Any,
    bot_login: str,
    *,
    mirror_for: Callable[[str], Path | None],
    repo: str | None = None,
    number: int | None = None,
) -> dict[str, int]:
    """The one-shot reconcile (`reconcile`) of every PR with standing issues that has not had
    it (or one repo / PR), to run before policy v12 is switched on so its first rounds start
    from a clean ledger: a closed PR's standing issues are outdated; an open PR's threads are
    synced and its issues reconciled against its live head (git in the repo's mirror,
    `mirror_for`; without one, nothing is outdated for a code change). A PR whose first v12
    round already reconciled it is left alone. Needs the network."""
    keep = ",".join(f"'{s}'" for s in sorted(COUNTING))
    prs = conn.execute(
        f"SELECT DISTINCT repo, number FROM issues WHERE status IN ({keep}) "
        "AND (?1 IS NULL OR repo=?1) AND (?2 IS NULL OR number=?2) ORDER BY 1, 2",
        (repo, number),
    ).fetchall()
    out = dict.fromkeys(("prs", "closed", "open", "outdated", "pending", "skipped"), 0)
    for pr in prs:
        r, n = str(pr["repo"]), int(pr["number"])
        if reconciled(conn, r, n) is not None:
            out["skipped"] += 1
            continue
        out["prs"] += 1
        before = {h for h, i in load(conn, r, n).items() if i.status in COUNTING}
        meta = github.pr_meta(gh, r, n)
        if meta.state != "open" or meta.merged:
            with tx(conn):
                reconcile(conn, r, n, pr_open=False, open_threads=(), moved={})
            out["closed"] += 1
            out["outdated"] += len(before)
            continue
        threads = github.review_threads(gh, r, n)
        open_ = set(github.finding_threads(threads, bot_login))
        path = mirror_for(r)
        head = meta.head_sha

        def changed(
            a: str, b: str, file: str, ls: int | None, le: int | None, path: Path | None = path
        ) -> bool | None:
            return lines_changed(path, a, b, file, ls, le) if path is not None and file else None

        moved = moved_anchors(load(conn, r, n), head, changed, open_threads=open_)
        with tx(conn):
            sync_threads(
                conn,
                r,
                n,
                run_id=None,
                sha=None,
                threads=threads,
                bot_login=bot_login,
                source="backfill",
            )
            pending = reconcile(
                conn,
                r,
                n,
                pr_open=True,
                open_threads=open_,
                moved=moved,
                sha=head,
            )
        out["open"] += 1
        out["pending"] += len(pending)
        out["outdated"] += len(before) - len(pending)
    return out


def record_reconciled(
    conn: sqlite3.Connection, repo: str, number: int, rec: dict[str, Any]
) -> None:
    """Write a PR's reconcile record as given (a replay copying the source's, as of its
    cutoff). Caller holds the transaction."""
    kv_set(conn, _reconciled_key(repo, number), json.dumps(rec))


def reconciled(conn: sqlite3.Connection, repo: str, number: int) -> dict[str, Any] | None:
    """The PR's reconcile record ({run, sha, pending, at, pending_at_reconcile}; records
    written before `at` have only the first three), or None before it ran."""
    raw = kv_get(conn, _reconciled_key(repo, number))
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return {"pending": []}
    return value if isinstance(value, dict) else {"pending": []}


def settle_pending(conn: sqlite3.Connection, repo: str, number: int, judged: Iterable[str]) -> None:
    """Hashes the thread lane has now judged leave the reconcile record's pending list. Caller
    holds the transaction."""
    rec = reconciled(conn, repo, number)
    if rec is None:
        return
    done = set(judged)
    rec["pending"] = [h for h in rec.get("pending") or [] if h not in done]
    kv_set(conn, _reconciled_key(repo, number), json.dumps(rec))


def safe_write(
    conn: sqlite3.Connection,
    what: str,
    fn: Callable[[sqlite3.Connection], object],
    *,
    repo: str,
    number: int,
    run_id: int | None,
) -> bool:
    """Run one ledger write in its own transaction, never raising: a ledger bug must never fail
    a review. A failure is logged and recorded as a `ledger.write_failed` event (the backfill
    fills the gap it leaves). Must be called outside any transaction: inside one it writes
    nothing (it cannot commit the caller's work, or roll back its own without the caller's)."""
    if conn.in_transaction:
        log.warning(
            "ledger write %s for %s#%s skipped: called inside a transaction", what, repo, number
        )
        return False
    try:
        with tx(conn):
            fn(conn)
        return True
    except Exception as exc:
        log.warning("ledger write %s failed for %s#%s: %s", what, repo, number, exc, exc_info=True)
        try:
            with tx(conn):
                event(
                    conn,
                    "ledger.write_failed",
                    repo=repo,
                    number=number,
                    run_id=run_id,
                    detail=f"{what}: {type(exc).__name__}: {exc}",
                )
        except Exception:
            log.warning("could not record ledger.write_failed", exc_info=True)
        return False


# ---- git: did the code an issue points at change? ----


def _git(repo_path: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    """Run git in `repo_path`, never fetching: a partial clone would otherwise go to the network
    for a missing object. None when git could not run."""
    try:
        return subprocess.run(
            ["git", "-C", str(repo_path), *args],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            env={**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0"},
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def git_diff_u0(repo_path: Path, from_sha: str, to_sha: str, file: str) -> str | None:
    """`git diff -U0 from to -- file`, or None when git cannot produce it (a commit missing)."""
    res = _git(
        repo_path, "diff", "-U0", "--no-color", "--no-ext-diff", from_sha, to_sha, "--", file
    )
    return res.stdout if res is not None and res.returncode == 0 else None


def git_diff(
    repo_path: Path, from_sha: str, to_sha: str, file: str, limit: int = 8000
) -> str | None:
    """`git diff from to -- file` with context, for a prompt (cut at `limit` characters), or
    None when git cannot produce it."""
    res = _git(repo_path, "diff", "--no-color", "--no-ext-diff", from_sha, to_sha, "--", file)
    if res is None or res.returncode != 0:
        return None
    out = res.stdout
    return out if len(out) <= limit else out[:limit] + f"\n[... {len(out) - limit} more characters]"


def git_has(repo_path: Path, sha: str, file: str | None = None) -> bool:
    """Whether the commit `sha` (with `file`: that path at it) is in `repo_path`."""
    obj = f"{sha}:{file}" if file else f"{sha}^{{commit}}"
    res = _git(repo_path, "cat-file", "-e", obj)
    return res is not None and res.returncode == 0


def file_absent_on_old_side(diff_text: str) -> bool:
    """The diff creates the file: it did not exist on the `from` side."""
    return any(line == "--- /dev/null" for line in diff_text.splitlines())


def anchor_touched(
    diff_text: str, line_start: int | None, line_end: int | None, margin: int = 0
) -> bool:
    """Whether a `-U0` diff touches lines `line_start..line_end` (± `margin`) of its old side.
    No anchor: any change to the file counts."""
    if not diff_text.strip():
        return False
    if not line_start:
        return True
    lo = line_start - margin
    hi = max(line_start, line_end or line_start) + margin
    for line in diff_text.splitlines():
        if not line.startswith("@@"):
            continue
        # @@ -a,b +c,d @@ : old lines a..a+b-1 replaced (b == 0: an insertion after line a)
        try:
            old = line.split()[1]
            start, _, count = old[1:].partition(",")
            s, n = int(start), int(count) if count else 1
        except (IndexError, ValueError):
            return True
        if n == 0:
            if lo <= s + 1 and s <= hi:
                return True
        elif s <= hi and s + n - 1 >= lo:
            return True
    return False


def diff_touches(
    diff_text: str, line_start: int | None, line_end: int | None, margin: int = 3
) -> bool:
    """Whether a `-U0` diff changed the anchored code of its old side: the file did not exist
    there, or the diff touches the anchor (± `margin`)."""
    return file_absent_on_old_side(diff_text) or anchor_touched(
        diff_text, line_start, line_end, margin
    )


def anchor_moved(
    issue: Issue,
    head: str,
    changed: Callable[[str, str, str, int | None, int | None], bool | None],
    *,
    since: str | None = None,
) -> bool | None:
    """Whether the code an issue points at changed since a commit (policy v12): its anchored
    lines (± 3, read in the numbering of the commit it was anchored on, the only one they are
    valid in) changed by `head`, and its file changed since `since` (default: the commit it
    was last seen standing at; the revival rule passes the one it was closed at). The thread
    lane's work set, the reconcile and the revival of settled issues all ask this. `changed` is
    `lines_changed` in some repository (`(from, to, file, ls, le)`; a None anchor asks whether
    the file changed at all). None when git cannot tell."""
    since = since or issue.last_seen_sha
    if since == head:
        return False
    anchor = issue.anchor_sha or issue.opened_sha
    if anchor == head:
        return False
    touched = (
        changed(anchor, head, issue.file, issue.line_start, issue.line_end) if anchor else None
    )
    if touched is False:
        return False
    if since and since != anchor and changed(since, head, issue.file, None, None) is False:
        return False
    return touched


def _hunks(diff_text: str) -> list[tuple[int, int, int, int]] | None:
    """The (old start, old count, new start, new count) of every `-U0` hunk; None when one
    cannot be read."""
    out: list[tuple[int, int, int, int]] = []
    for line in diff_text.splitlines():
        if not line.startswith("@@"):
            continue
        try:
            old, new = line.split()[1:3]
            a, _, b = old[1:].partition(",")
            c, _, d = new[1:].partition(",")
            out.append((int(a), int(b) if b else 1, int(c), int(d) if d else 1))
        except (IndexError, ValueError):
            return None
    return out


def _map_line(hunks: list[tuple[int, int, int, int]], line: int, *, end: bool) -> int:
    """`line` of the old side of a diff on its new side: shifted by what the hunks before it
    inserted and deleted; a line a hunk replaced is that hunk's new lines (their first for a
    range's start, their last for its end)."""
    offset = 0
    for a, b, c, d in hunks:
        if b == 0:  # an insertion after old line `a`
            if a < line:
                offset += d
                continue
            break
        if a + b - 1 < line:
            offset += d - b
            continue
        if a <= line:
            return max(c, 1) if not end or d == 0 else c + d - 1
        break
    return line + offset


def map_lines(
    repo_path: Path,
    from_sha: str,
    to_sha: str,
    file: str,
    line_start: int | None,
    line_end: int | None,
) -> tuple[int | None, int | None] | None:
    """`line_start..line_end` of `file` as numbered at `from_sha`, re-located at `to_sha` by
    the diff between them (lines a hunk rewrote become that hunk's new lines). No anchor: no
    lines either. None when git cannot tell, or the file is gone at `to_sha`."""
    if from_sha == to_sha or not line_start:
        return line_start, line_end
    diff = git_diff_u0(repo_path, from_sha, to_sha, file)
    if diff is None or "+++ /dev/null" in diff.splitlines():
        return None
    if not diff.strip():
        return (line_start, line_end) if git_has(repo_path, to_sha, file) else None
    hunks = _hunks(diff)
    if hunks is None:
        return None
    return (
        _map_line(hunks, line_start, end=False),
        _map_line(hunks, line_end or line_start, end=True),
    )


def lines_changed(
    repo_path: Path,
    from_sha: str,
    to_sha: str,
    file: str,
    line_start: int | None,
    line_end: int | None,
    margin: int = 3,
) -> bool | None:
    """Whether the code at `file:line_start..line_end` (± `margin`), as it reads at `from_sha`,
    changed by `to_sha`. The anchor is in `from_sha`'s line numbers: pass the commit an issue
    was raised (or closed) on first; for an anchor on the later commit, swap the two (the diff
    is symmetric in what it touches). A file missing on the `from` side counts as changed.
    None when git cannot tell (a commit missing from `repo_path`, or a path that exists on
    neither side: the anchor is wrong, not unchanged)."""
    if from_sha == to_sha:
        return False
    diff = git_diff_u0(repo_path, from_sha, to_sha, file)
    if diff is None or (not diff.strip() and not git_has(repo_path, from_sha, file)):
        return None
    return diff_touches(diff, line_start, line_end, margin)


# ---- backfill: the history of PRs reviewed before the write-through ----


# the PR's review runs' finding rows (a post-merge audit is not part of its review history)
_PR_FINDINGS = (
    "FROM findings f JOIN runs r ON r.id=f.run_id JOIN heads h ON h.id=r.head_id "
    "WHERE h.repo=? AND h.number=? AND h.queue='live'"
)


def _pr_history(
    conn: sqlite3.Connection, repo: str, number: int, counts: dict[str, int]
) -> list[Event]:
    """Every fact on record about the PR's issues, as events keyed like the write-through's:
    publications (`findings` stage `posted`), carried findings (a later done run's verified
    set), thread answers (`thread.answered` events) and concessions (`conceded` rows) that no
    answer event records. Audit runs are not part of a PR's review history.

    What it cannot recover: the `finding_hash` a verifier echoed for a finding it retitled (the
    rows keep only the new title's hash), so a retitled carried finding is not `seen` and its
    issue reads as vanished where only the backfill knows it (fate coverage undercounts
    `carried` on PRs reviewed before the write-through)."""
    runs: dict[int, sqlite3.Row] = {
        int(r["id"]): r
        for r in conn.execute(
            "SELECT r.id, r.status, r.started_at, r.finished_at, h.sha FROM runs r "
            "JOIN heads h ON h.id=r.head_id WHERE h.repo=? AND h.number=? AND h.queue='live'",
            (repo, number),
        )
    }
    published: dict[tuple[int, str], str] = {}
    for r in conn.execute(
        "SELECT run_id, phase, min(posted_at) AS ts FROM reviews "
        "WHERE repo=? AND number=? AND run_id IS NOT NULL GROUP BY run_id, phase",
        (repo, number),
    ):
        published[(int(r["run_id"]), str(r["phase"]))] = str(r["ts"])

    def run_ts(run_id: int, phase: str | None = None) -> str:
        """When the run published (its `phase` review), else when it ended or started."""
        if phase and (run_id, phase) in published:
            return published[(run_id, phase)]
        times = [t for (rid, _), t in published.items() if rid == run_id]
        r = runs[run_id]
        return min(times) if times else str(r["finished_at"] or r["started_at"])

    # a run that published on a commit an earlier run already published on is a same-sha
    # re-review: the findings it records as posted that the PR already had were carried, not
    # posted again (`worker._backfill_posted` records its whole verifier set as posted)
    first_on_sha: dict[str, tuple[str, int]] = {}
    for (rid, _), t in published.items():
        if rid in runs:
            sha = str(runs[rid]["sha"])
            first_on_sha[sha] = min(first_on_sha.get(sha, (t, rid)), (t, rid))

    def rereview(run_id: int) -> bool:
        first = first_on_sha.get(str(runs[run_id]["sha"]))
        return first is not None and first[1] != run_id

    events: list[Event] = []
    first_post: dict[str, str] = {}
    posted_in: set[tuple[int, str]] = set()
    for r in conn.execute(
        "SELECT f.run_id, f.phase, f.hash, f.file, f.line_start, f.line_end, f.severity, "
        f"f.category, f.title, f.body {_PR_FINDINGS} AND f.stage='posted' ORDER BY f.id",
        (repo, number),
    ):
        rid, h = int(r["run_id"]), str(r["hash"])
        f = Finding(
            file=str(r["file"] or ""),
            title=str(r["title"]),
            body=str(r["body"] or ""),
            severity=str(r["severity"]),
            category=str(r["category"] or "general"),
            line_start=r["line_start"],
            line_end=r["line_end"],
        )
        if f.hash != h:
            # the stored hash is the identity GitHub knows; never re-derive it
            counts["posted_hash_mismatch"] += 1
        ts, sha = run_ts(rid, str(r["phase"])), str(runs[rid]["sha"])
        posted_in.add((rid, h))
        if h in first_post and first_post[h] < ts and rereview(rid):
            events.append(_seen(h, run_id=rid, sha=sha, source="backfill", ts=ts))
            counts["seen"] += 1
            continue
        events.append(_posted(f, run_id=rid, sha=sha, source="backfill", ts=ts, h=h))
        first_post[h] = min(ts, first_post.get(h, ts))
        counts["posted"] += 1
    have = set(first_post) | known(conn, repo, number)
    for r in conn.execute(
        f"SELECT DISTINCT f.run_id, f.hash {_PR_FINDINGS} "
        "AND r.status='done' AND f.stage IN ('verified','verified-fresh')",
        (repo, number),
    ):
        rid, h = int(r["run_id"]), str(r["hash"])
        ts = run_ts(rid)
        if h not in have or (rid, h) in posted_in or ts <= first_post.get(h, ""):
            continue
        events.append(_seen(h, run_id=rid, sha=str(runs[rid]["sha"]), source="backfill", ts=ts))
        counts["seen"] += 1
    conceded: dict[tuple[int, str], str] = {
        (int(r["run_id"]), str(r["hash"])): str(r["sha"])
        for r in conn.execute(
            f"SELECT f.run_id, f.hash, h.sha {_PR_FINDINGS} AND f.stage='conceded'", (repo, number)
        )
    }
    answered: set[tuple[int, str]] = set()
    for r in conn.execute(
        "SELECT e.ts, e.run_id, e.detail, h.sha FROM events e LEFT JOIN runs r ON r.id=e.run_id "
        "LEFT JOIN heads h ON h.id=r.head_id WHERE e.kind='thread.answered' AND e.repo=? "
        "AND e.number=? ORDER BY e.id",
        (repo, number),
    ):
        try:
            item = json.loads(r["detail"] or "{}")
        except ValueError:
            counts["answers_unreadable"] += 1
            continue
        if not isinstance(item, dict):
            counts["answers_unreadable"] += 1
            continue
        h = str(item.get("finding_hash") or "")
        if h not in have:
            counts["answers_unknown_issue"] += 1
            continue
        rid = r["run_id"]
        ans = _answer(
            item,
            run_id=rid,
            sha=r["sha"],
            source="backfill",
            conceded=(rid, h) in conceded,
            ts=str(r["ts"]),
            origin="conversation" if item.get("mode") == "conversation" else "verifier",
        )
        if ans is not None:
            events.append(ans)
            answered.add((rid, h))
            counts["answers"] += 1
    for (rid, h), sha in sorted(conceded.items()):
        if (rid, h) in answered or h not in have or rid not in runs:
            continue
        # a concession whose answer event is missing (the run died between the reply and the
        # bookkeeping, or predates the event): the conversation withdrew it
        item = {"finding_hash": h, "status": "WITHDRAWN", "action": "replied"}
        ans = _answer(
            item,
            run_id=rid,
            sha=sha,
            source="backfill",
            conceded=True,
            ts=run_ts(rid),
            origin="conceded",
            key=f"conceded:{rid}:{h}",
        )
        assert ans is not None
        events.append(ans)
        counts["conceded_only"] += 1
    return events


def backfill(
    conn: sqlite3.Connection, *, repo: str | None = None, number: int | None = None
) -> dict[str, Any]:
    """Replay the recorded history of every PR the bot posted findings on (or one repo / PR)
    into the ledger. Idempotent, and safe beside the live write-through: facts already recorded
    (by either) are skipped by key. Must be called outside a transaction (one per PR)."""
    prs = conn.execute(
        "SELECT repo, number FROM posted_findings WHERE (?1 IS NULL OR repo=?1) "
        "AND (?2 IS NULL OR number=?2) UNION SELECT h.repo, h.number FROM findings f "
        "JOIN runs r ON r.id=f.run_id JOIN heads h ON h.id=r.head_id "
        "WHERE f.stage='posted' AND h.queue='live' AND (?1 IS NULL OR h.repo=?1) "
        "AND (?2 IS NULL OR h.number=?2) ORDER BY 1, 2",
        (repo, number),
    ).fetchall()
    counts: dict[str, int] = dict.fromkeys(
        (
            "prs",
            "posted",
            "seen",
            "answers",
            "conceded_only",
            "answers_unknown_issue",
            "answers_unreadable",
            "posted_hash_mismatch",
            "events_added",
        ),
        0,
    )
    for pr in prs:
        with tx(conn):
            events = _pr_history(conn, pr["repo"], pr["number"], counts)
            counts["events_added"] += append(conn, pr["repo"], pr["number"], events)
        counts["prs"] += 1
    statuses = {
        str(r[0]): int(r[1])
        for r in conn.execute(
            "SELECT status, count(*) FROM issues WHERE (?1 IS NULL OR repo=?1) "
            "AND (?2 IS NULL OR number=?2) GROUP BY status ORDER BY 2 DESC",
            (repo, number),
        )
    }
    return {**counts, "issues": sum(statuses.values()), "by_status": statuses}


def backfill_threads(
    conn: sqlite3.Connection,
    gh: Any,
    bot_login: str,
    *,
    repo: str | None = None,
    number: int | None = None,
) -> dict[str, int]:
    """Sync the PRs that have standing issues with their GitHub threads (`sync_threads`):
    legacy open finding threads, thread ids, hand-resolved threads. Needs the network."""
    standing = ",".join(f"'{s}'" for s in sorted(STANDING))
    prs = conn.execute(
        f"SELECT DISTINCT repo, number FROM issues WHERE status IN ({standing}) "
        "AND (?1 IS NULL OR repo=?1) AND (?2 IS NULL OR number=?2) ORDER BY 1, 2",
        (repo, number),
    ).fetchall()
    out = {"prs": 0, "changes": 0}
    for pr in prs:
        threads = github.review_threads(gh, pr["repo"], pr["number"])
        with tx(conn):
            out["changes"] += sync_threads(
                conn,
                pr["repo"],
                pr["number"],
                run_id=None,
                sha=None,
                threads=threads,
                bot_login=bot_login,
                source="backfill",
            )
        out["prs"] += 1
    return out


def render(issues: dict[str, Issue], events: list[dict[str, Any]] | None = None) -> str:
    """The issue table `reviewsys ledger show` prints (and with `events`, the history)."""

    def at(sha: str | None, run: int | None) -> str:
        return f"{(sha or '-')[:8]}/{run if run is not None else '-'}"

    def loc(i: Issue) -> str:
        if not i.line_start:
            return i.file or "-"
        end = f"-{i.line_end}" if i.line_end and i.line_end != i.line_start else ""
        return f"{i.file}:{i.line_start}{end}"

    rows = [("hash", "status", "severity", "where", "opened", "last seen", "by", "title")]
    rows += [
        (
            i.hash,
            i.status,
            i.severity,
            loc(i)[-40:],
            at(i.opened_sha, i.opened_run),
            at(i.last_seen_sha, i.last_seen_run),
            i.status_source,
            i.title[:70],
        )
        for i in issues.values()
    ]
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]) - 1)]
    out = [
        "  ".join(v.ljust(w) for v, w in zip(r, widths, strict=False)) + "  " + r[-1] for r in rows
    ]
    counts = Counter(i.status for i in issues.values())
    out.append(", ".join(f"{s} {counts[s]}" for s in STATUSES if counts[s]) or "no issues")
    for e in events or []:
        d = e["detail"]
        what = d.get("outcome") or d.get("status") or ""
        out.append(
            f"{e['ts']} {e['hash']} {e['kind']:<7} {what:<22} {e['from_status'] or '-'} -> "
            f"{e['to_status'] or '-'}  {e['source']} run={e['run_id']} {(e['sha'] or '')[:8]}"
        )
    return "\n".join(out) + "\n"
