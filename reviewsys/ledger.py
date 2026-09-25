"""The per-PR issue ledger (schema v4): every issue reviewsys raised on a PR, its identity
(`finding_hash`, byte-compatible with the markers on GitHub), and where it stands.

Only the harness writes it: new issues when a review is published, status changes when the
thread lane's outcome for an issue reached GitHub. A PR reviewed before the ledger existed is
seeded once from `posted_findings` + `findings` + the open threads on GitHub."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from .contract import Finding
from .db import kv_get, kv_set, now, tx

OPEN = "open"
# thread-lane status -> ledger status
CLOSING = {
    "FIXED": "fixed",
    "OUTDATED": "outdated",
    "WITHDRAWN": "withdrawn",
    "INTENTIONALLY_DEFERRED": "deferred",
}
# closed issues that only come back when the code they point at changed since
# (a maintainer's call stands; "resolved" = seeded from a thread closed by hand)
STICKY = {"withdrawn", "deferred", "outdated", "resolved"}


@dataclass(slots=True)
class Issue:
    hash: str
    file: str
    line_start: int | None
    line_end: int | None
    severity: str
    category: str
    title: str
    body: str
    status: str
    note: str
    opened_sha: str
    closed_sha: str | None

    @property
    def is_open(self) -> bool:
        return self.status == OPEN

    def as_finding(self) -> Finding:
        """The issue as a finding carried into this round's verdict (not re-posted)."""
        return Finding(
            file=self.file,
            title=self.title,
            body=self.body,
            severity=self.severity,
            category=self.category or "general",
            line_start=self.line_start,
            line_end=self.line_end,
            source="carried",
            prior_hash=self.hash,
            carried=True,
        )

    def summary(self, last_message: str = "") -> dict[str, Any]:
        d: dict[str, Any] = {
            "hash": self.hash,
            "status": self.status,
            "severity": self.severity,
            "file": self.file,
            "lines": [self.line_start, self.line_end],
            "title": self.title,
        }
        if self.note:
            d["closed_because"] = self.note[:300]
        if last_message:
            d["last_message"] = last_message[:300]
        return d


def load(conn: sqlite3.Connection, repo: str, number: int) -> dict[str, Issue]:
    rows = conn.execute(
        "SELECT * FROM ledger WHERE repo=? AND number=? ORDER BY rowid", (repo, number)
    ).fetchall()
    return {
        str(r["hash"]): Issue(
            hash=str(r["hash"]),
            file=str(r["file"] or ""),
            line_start=r["line_start"],
            line_end=r["line_end"],
            severity=str(r["severity"]),
            category=str(r["category"] or "general"),
            title=str(r["title"]),
            body=str(r["body"] or ""),
            status=str(r["status"]),
            note=str(r["note"] or ""),
            opened_sha=str(r["opened_sha"]),
            closed_sha=r["closed_sha"],
        )
        for r in rows
    }


def sync(
    conn: sqlite3.Connection,
    repo: str,
    number: int,
    open_threads: dict[str, dict[str, Any]],
    *,
    dry_run: bool = False,
) -> dict[str, Issue]:
    """Make the ledger cover everything reviewsys has raised on this PR, then return it.

    Once per PR (kv `ledger.seeded`): seed from what this database posted. Open when its
    GitHub thread is still open; else what the conversation concluded, read from the
    `conceded` rows (a concession is `withdrawn`), else `resolved` (closed by hand or gone).
    Every run: an open bot finding thread the ledger does not know yet (posted by the legacy
    pipeline or a v9 run) is added as open. A dry run writes nothing and returns what the
    ledger would be. Caller must not hold a transaction."""
    ts = now()
    seeded_key = f"ledger.seeded:{repo}#{number}"
    with tx(conn):
        rows: list[tuple[Any, ...]] = []
        have = {
            str(r["hash"])
            for r in conn.execute(
                "SELECT hash FROM ledger WHERE repo=? AND number=?", (repo, number)
            )
        }
        if kv_get(conn, seeded_key) is None:
            conceded = {
                str(r["hash"])
                for r in conn.execute(
                    "SELECT f.hash FROM findings f JOIN runs r ON r.id=f.run_id JOIN heads h ON h.id=r.head_id "
                    "WHERE h.repo=? AND h.number=? AND f.stage='conceded'",
                    (repo, number),
                )
            }
            for r in conn.execute(
                "SELECT pf.hash, pf.sha, f.file, f.line_start, f.line_end, f.severity, f.category, f.title, f.body "
                "FROM posted_findings pf LEFT JOIN findings f ON f.hash=pf.hash AND f.stage='posted' "
                "WHERE pf.repo=? AND pf.number=? ORDER BY pf.posted_at DESC, f.id DESC",
                (repo, number),
            ).fetchall():
                h = str(r["hash"])
                if h in have or not r["title"]:
                    continue
                have.add(h)
                status = OPEN if h in open_threads else "withdrawn" if h in conceded else "resolved"
                rows.append(
                    (
                        h,
                        r["file"] or "",
                        r["line_start"],
                        r["line_end"],
                        r["severity"] or "nitpick",
                        r["category"] or "general",
                        r["title"],
                        r["body"] or "",
                        status,
                        r["sha"],
                        None if status == OPEN else r["sha"],
                    )
                )
            if not dry_run:
                kv_set(conn, seeded_key, ts)
        for h, t in open_threads.items():
            if h in have or not t.get("title"):
                continue
            have.add(h)
            rows.append(
                (
                    h,
                    t.get("path") or "",
                    t.get("line"),
                    t.get("line"),
                    t.get("severity") or "nitpick",
                    "general",
                    t["title"],
                    str(t.get("body") or "")[:8000],
                    OPEN,
                    "",
                    None,
                )
            )
        if not dry_run:
            conn.executemany(
                "INSERT INTO ledger (repo, number, hash, file, line_start, line_end, severity, category, title, body, status, opened_sha, closed_sha, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(repo, number, *row, ts) for row in rows],
            )
    issues = load(conn, repo, number)
    if dry_run:
        for row in rows:
            issues.setdefault(
                row[0],
                Issue(
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    row[6],
                    row[7],
                    row[8],
                    "",
                    row[9],
                    row[10],
                ),
            )
    return issues


def record_published(
    conn: sqlite3.Connection, repo: str, number: int, sha: str, findings: list[Finding]
) -> None:
    """Issues a published review put on GitHub. A hash already in the ledger is reopened (a
    fixed issue that came back, or a sticky one whose code changed). Carried findings are
    already in the ledger and skipped. Caller holds the transaction."""
    ts = now()
    for f in findings:
        if f.carried:
            continue
        conn.execute(
            "INSERT INTO ledger (repo, number, hash, file, line_start, line_end, severity, category, title, body, status, opened_sha, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(repo, number, hash) DO UPDATE SET "
            "status='open', note='', closed_sha=NULL, severity=excluded.severity, body=excluded.body, "
            "line_start=excluded.line_start, line_end=excluded.line_end, updated_at=excluded.updated_at",
            (
                repo,
                number,
                f.hash,
                f.file,
                f.line_start,
                f.line_end,
                f.severity,
                f.category,
                f.title,
                f.body[:8000],
                OPEN,
                sha,
                ts,
            ),
        )


def record_outcome(
    conn: sqlite3.Connection, repo: str, number: int, sha: str, h: str, status: str, note: str
) -> None:
    """A thread-lane outcome that reached GitHub. Closing statuses close the issue at `sha`;
    anything else leaves it open. Caller holds the transaction."""
    new = CLOSING.get(status)
    if new is None:
        return
    conn.execute(
        "UPDATE ledger SET status=?, note=?, closed_sha=?, updated_at=? WHERE repo=? AND number=? AND hash=?",
        (new, note[:1000], sha, now(), repo, number, h),
    )
