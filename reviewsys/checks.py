"""The review as a GitHub check run on each PR's head commit.

One check, `[checks] name` ("PastaClaw review"), per commit the bot is responsible for:

- `queued` while the review waits (push debounce, its turn in line, a retry backoff, a draft);
- `in_progress` while a run reviews it;
- `completed` once a final verdict stands for the commit (its own, or one carried over a push
  that left the diff unchanged): `success` = APPROVE, `neutral` = COMMENT,
  `failure` = REQUEST_CHANGES;
- `completed` / `action_required` when the review gave up (failed after its retries, or ended
  without a verdict): someone has to ask again (`@thepastaclaw review`);
- `completed` / `skipped` when the PR carries a skip label (no automatic review);
- `completed` / `cancelled` for a commit that stopped being the PR's head, or a PR that closed,
  before it was reviewed.

A final verdict wins over later activity on the same commit: a conversation or CI re-check
keeps the check completed and replaces the conclusion if the verdict moves, so a check that
said `success` never goes back to pending without a push. A run that a Phase 1 gate stopped
ends on its preliminary review, which stands the same way once the run is over.

Like the verdict labels this is a reconciliation of database state, not an event hook: every
pass derives the wanted check of every open PR's head (and of every commit whose check is still
open) and writes only what differs from what was last written. Missing an event, a restart or a
failed write is repaired by the next pass. GitHub only accepts check runs from a GitHub App, so
the writes go through the App's installation (ghapp.py); a repository without it is skipped.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .db import event, kv_get, now, now_dt, parse_ts, tx
from .gh import Gh
from .ghapp import AppAuth
from .ingest import AUTOMATIC_TRIGGERS, reviewed_sha
from .models import FailKind, HeadStatus, ReviewError, Trigger
from .publish import canonical_event

log = logging.getLogger(__name__)

# writes per pass: the first pass after enabling backfills every open PR, spread over passes
MAX_WRITES_PER_PASS = 20
STATUS_PAGE = "https://thepastaclaw.github.io/review-system/"

VERDICT_CONCLUSIONS = {
    "APPROVE": ("success", "Approved"),
    "COMMENT": ("neutral", "Reviewed — comments, no changes requested"),
    "REQUEST_CHANGES": ("failure", "Changes requested"),
}
STEP_LABELS = {
    "worktree": "preparing",
    "carry": "preparing",
    "select": "preparing",
    "triage": "preparing",
    "context": "preparing",
    "converse": "answering the discussion",
    "phase1": "Phase 1 review",
    "verify1": "verifying Phase 1 findings",
    "gate": "Phase 1 gate",
    "phase2": "Phase 2 review",
    "verify2": "verifying findings",
    "fresh_phase2": "fresh Phase 2 pass",
    "fresh_verify2": "verifying the fresh pass",
    "ci_check": "checking a CI failure",
    "publish": "publishing",
}


@dataclass(frozen=True, slots=True)
class Want:
    """What the check of one commit should say."""

    status: str  # queued | in_progress | completed
    title: str
    conclusion: str | None = None
    summary: str = ""
    details_url: str | None = None
    started_at: str | None = None
    completed_at: str | None = None

    @property
    def digest(self) -> str:
        """Identity of what GitHub shows; timestamps are left out so they cause no writes."""
        shown = [self.status, self.conclusion, self.title, self.summary, self.details_url]
        return hashlib.sha256(json.dumps(shown).encode()).hexdigest()[:16]


@dataclass(slots=True)
class Stats:
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped_repos: set[str] = field(default_factory=set)
    deferred: int = 0
    failed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "created": self.created,
            "updated": self.updated,
            "unchanged": self.unchanged,
            "skipped_repos": len(self.skipped_repos),
            "deferred": self.deferred,
            "failed": self.failed,
        }


def _pr_url(repo: str, number: int) -> str:
    return f"https://github.com/{repo}/pull/{number}"


def _standing_review(
    conn: sqlite3.Connection, repo: str, number: int, sha: str, head: sqlite3.Row | None
) -> sqlite3.Row | None:
    """The `reviews` row whose verdict stands for `sha` (its own review, or one carried over):
    the newest final one; with none, the newest preliminary one of a finished head (a Phase 1
    gate that held Phase 2 back ends the run on that preliminary review)."""
    target = reviewed_sha(conn, repo, number, sha)
    final = conn.execute(
        "SELECT event, phase, posted_at FROM reviews WHERE repo=? AND number=? AND sha=? "
        "AND phase='final' ORDER BY posted_at DESC, id DESC LIMIT 1",
        (repo, number, target),
    ).fetchone()
    if final is not None or head is None or head["status"] != HeadStatus.DONE:
        return final  # type: ignore[no-any-return]
    return conn.execute(  # type: ignore[no-any-return]
        "SELECT event, phase, posted_at FROM reviews WHERE repo=? AND number=? AND sha=? "
        "ORDER BY posted_at DESC, id DESC LIMIT 1",
        (repo, number, target),
    ).fetchone()


def _review_url(conn: sqlite3.Connection, repo: str, number: int, sha: str) -> str | None:
    """Link to the bot's newest posted review of the standing commit (a verdict recorded
    without a GitHub review, on a bot-authored PR, has none)."""
    row = conn.execute(
        "SELECT github_review_id FROM reviews WHERE repo=? AND number=? AND sha=? "
        "AND github_review_id IS NOT NULL ORDER BY posted_at DESC, id DESC LIMIT 1",
        (repo, number, reviewed_sha(conn, repo, number, sha)),
    ).fetchone()
    if not row:
        return None
    return f"{_pr_url(repo, number)}#pullrequestreview-{row['github_review_id']}"


def _gate_url(conn: sqlite3.Connection, repo: str, number: int) -> str:
    cid = kv_get(conn, f"queue.comment_id:{repo}#{number}")
    return f"{_pr_url(repo, number)}#issuecomment-{cid}" if cid else _pr_url(repo, number)


def _summary(sha: str, *lines: str) -> str:
    body = [f"Commit `{sha[:12]}`.", *[line for line in lines if line]]
    body.append(f"Review queue and history: {STATUS_PAGE}")
    return "\n\n".join(body)


def wanted(conn: sqlite3.Connection, repo: str, number: int, sha: str) -> Want | None:
    """The check `sha` of PR `repo#number` should carry, or None when the bot has nothing to
    say about it yet (no check is created; an existing one is left as it is)."""
    pr = conn.execute(
        "SELECT state, head_sha, is_draft, skip_label FROM prs WHERE repo=? AND number=?",
        (repo, number),
    ).fetchone()
    head = conn.execute(
        "SELECT * FROM heads WHERE repo=? AND number=? AND sha=? AND queue='live'",
        (repo, number, sha),
    ).fetchone()
    if pr is not None:
        live = pr["state"] == "open" and pr["head_sha"] == sha
    else:  # ad hoc review of a repo the bot does not list: the newest head row is the head
        newest = conn.execute(
            "SELECT sha FROM heads WHERE repo=? AND number=? AND queue='live' ORDER BY id DESC LIMIT 1",
            (repo, number),
        ).fetchone()
        live = newest is not None and newest["sha"] == sha
    final = _standing_review(conn, repo, number, sha, head)
    if final is not None:
        conclusion, title = VERDICT_CONCLUSIONS.get(
            canonical_event(str(final["event"])), ("neutral", "Reviewed")
        )
        carried = reviewed_sha(conn, repo, number, sha)
        busy = live and head is not None and head["status"] in ("queued", "running")
        return Want(
            status="completed",
            conclusion=conclusion,
            title=title,
            summary=_summary(
                sha,
                f"The review of `{carried[:12]}` stands: this push left the PR's diff unchanged."
                if carried != sha
                else "",
                "A follow-up pass on this commit is running; this result is replaced if the "
                "verdict changes."
                if busy
                else "",
                "Phase 1 found blocking issues, so Phase 2 did not run on this commit."
                if final["phase"] != "final"
                else "",
            ),
            details_url=_review_url(conn, repo, number, sha) or _pr_url(repo, number),
            completed_at=str(final["posted_at"]),
        )
    if not live:
        closed = pr is not None and pr["state"] != "open"
        return Want(
            status="completed",
            conclusion="cancelled",
            title="Not reviewed: the PR was closed"
            if closed
            else "Not reviewed: a newer commit replaced this one",
            summary=_summary(sha),
            details_url=_pr_url(repo, number),
        )
    if pr is not None and pr["skip_label"]:
        return Want(
            status="completed",
            conclusion="skipped",
            title="Automatic review is off for this PR (skip label)",
            summary=_summary(
                sha, "Remove the label, or comment `@thepastaclaw review`, to get a review."
            ),
            details_url=_pr_url(repo, number),
        )
    draft = pr is not None and bool(pr["is_draft"])
    if head is None or (head["status"] == HeadStatus.CLOSED and draft):
        if not draft:
            return None
        return Want(
            status="queued",
            title="Waiting: the PR is a draft",
            summary=_summary(sha, "The review starts when the PR is marked ready for review."),
            details_url=_gate_url(conn, repo, number),
        )
    status = head["status"]
    if status == HeadStatus.QUEUED:
        return _queued(conn, head)
    if status == HeadStatus.RUNNING:
        run = conn.execute(
            "SELECT phase, started_at FROM runs WHERE head_id=? ORDER BY id DESC LIMIT 1",
            (head["id"],),
        ).fetchone()
        step = STEP_LABELS.get(str(run["phase"] or "")) if run else None
        return Want(
            status="in_progress",
            title=f"Reviewing — {step}" if step else "Reviewing",
            summary=_summary(sha),
            details_url=_gate_url(conn, repo, number),
            started_at=str(run["started_at"]) if run else None,
        )
    if status in (HeadStatus.FAILED, HeadStatus.DONE):
        return Want(
            status="completed",
            conclusion="action_required",
            title="The review could not finish"
            if status == HeadStatus.FAILED
            else "The review ended without a verdict",
            summary=_summary(
                sha,
                "This is a failure of the review service, not a finding about the change. "
                "Comment `@thepastaclaw review` to try again.",
            ),
            details_url=_gate_url(conn, repo, number),
            completed_at=str(head["finished_at"] or "") or None,
        )
    if status == HeadStatus.CLOSED:
        return Want(
            status="completed",
            conclusion="skipped",
            title="Not reviewed",
            summary=_summary(sha, f"Reason: {head['reason']}" if head["reason"] else ""),
            details_url=_pr_url(repo, number),
            completed_at=str(head["finished_at"] or "") or None,
        )
    return None  # superseded, yet GitHub's head again (a revert): ingest decides what follows


def _queued(conn: sqlite3.Connection, head: sqlite3.Row) -> Want:
    repo, number, sha = str(head["repo"]), int(head["number"]), str(head["sha"])
    waiting = parse_ts(str(head["eligible_at"])) > now_dt()
    if waiting and int(head["attempts"]) > 0:
        title = "Queued — retrying after a failed attempt"
    elif waiting and head["trigger"] in AUTOMATIC_TRIGGERS:
        title = "Queued — waiting for pushes to settle"
    elif head["trigger"] == Trigger.CI_FAILED.value:
        title = "Queued — re-checking after a CI failure"
    elif head["priority"]:
        title = "Queued — priority review"
    else:
        title = "Queued for review"
    return Want(
        status="queued",
        title=title,
        summary=_summary(sha, "The PR's queue comment shows its place in line."),
        details_url=_gate_url(conn, repo, number),
    )


def _candidates(conn: sqlite3.Connection) -> list[tuple[str, int, str]]:
    """Commits to reconcile, those whose check moves first: still-open checks, then commits
    being reviewed, then every open PR's head (the backfill after enabling)."""
    rows = conn.execute(
        "SELECT repo, number, sha FROM check_runs WHERE status<>'completed'"
    ).fetchall()
    rows += conn.execute(
        "SELECT repo, number, sha FROM heads WHERE queue='live' AND status IN ('queued','running')"
        " ORDER BY status='running' DESC, id"
    ).fetchall()
    rows += conn.execute(
        "SELECT repo, number, head_sha AS sha FROM prs WHERE state='open' ORDER BY updated_at DESC"
    ).fetchall()
    seen: dict[tuple[str, int, str], None] = {}
    for r in rows:
        seen.setdefault((str(r["repo"]), int(r["number"]), str(r["sha"])), None)
    return list(seen)


def _body(want: Want, *, name: str | None = None, sha: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "status": want.status,
        "output": {"title": want.title, "summary": want.summary},
    }
    if name is not None:
        body["name"] = name
    if sha is not None:
        body["head_sha"] = sha
    if want.details_url:
        body["details_url"] = want.details_url
    if want.started_at:
        body["started_at"] = want.started_at
    if want.status == "completed":
        body["conclusion"] = want.conclusion
        body["completed_at"] = want.completed_at or now()
    return body


class Reconciler:
    """Daemon task state: the App's clients and, per process, what failed for good."""

    def __init__(self, cfg: Config, app: AppAuth) -> None:
        self.cfg = cfg
        self.app = app
        # (repo, number, sha) -> digest GitHub refused (a contract error); not retried until
        # the wanted check changes or the daemon restarts
        self.refused: dict[tuple[str, int, str], str] = {}
        self.auth_error = ""

    def reconcile(self, conn: sqlite3.Connection, *, dry_run: bool = False) -> dict[str, Any]:
        stats = Stats()
        writes = 0
        clients: dict[str, Gh | None] = {}
        excluded = {r.lower() for r in self.cfg.checks.exclude_repos}
        for repo, number, sha in _candidates(conn):
            if repo.lower() in excluded or repo in stats.skipped_repos:
                continue
            want = wanted(conn, repo, number, sha)
            if want is None:
                continue
            row = conn.execute(
                "SELECT check_id, status, digest FROM check_runs WHERE repo=? AND number=? AND sha=?",
                (repo, number, sha),
            ).fetchone()
            if row is None and want.conclusion == "cancelled":
                continue  # never open a check only to say a commit we never touched was skipped
            if row is not None and row["digest"] == want.digest:
                stats.unchanged += 1
                continue
            key = (repo, number, sha)
            if self.refused.get(key) == want.digest:
                continue
            if writes >= MAX_WRITES_PER_PASS:
                stats.deferred += 1
                continue
            if dry_run:
                log.info("check %s#%s@%s -> %s", repo, number, sha[:8], want)
                writes += 1
                continue
            if repo not in clients:
                try:
                    clients[repo] = self.app.gh_for(repo)
                except ReviewError as exc:
                    # the App itself cannot authenticate (key, id, GitHub down): nothing else
                    # in this pass can be written either
                    self._auth_failed(conn, exc)
                    stats.deferred += 1
                    break
            gh = clients[repo]
            if gh is None:
                stats.skipped_repos.add(repo)
                continue
            writes += 1
            try:
                created = self._write(gh, repo, sha, want, row)
            except ReviewError as exc:
                if exc.kind is FailKind.INFRA:
                    log.warning("check for %s#%s deferred: %s", repo, number, exc)
                    stats.deferred += 1
                    continue
                stats.failed += 1
                self.refused[key] = want.digest
                if "HTTP 403" in exc.message or "HTTP 404" in exc.message:
                    # installation removed, or this repo dropped from it: look it up again later
                    self.app.forget(repo)
                    clients.pop(repo, None)
                log.warning("check for %s#%s refused: %s", repo, number, exc)
                with tx(conn):
                    event(conn, "check.failed", repo=repo, number=number, detail=str(exc)[:300])
                continue
            with tx(conn):
                conn.execute(
                    "INSERT INTO check_runs (repo, number, sha, check_id, status, conclusion, digest, updated_at)"
                    " VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(repo, number, sha) DO UPDATE SET"
                    " check_id=excluded.check_id, status=excluded.status,"
                    " conclusion=excluded.conclusion, digest=excluded.digest,"
                    " updated_at=excluded.updated_at",
                    (repo, number, sha, created, want.status, want.conclusion, want.digest, now()),
                )
                event(
                    conn,
                    "check.synced",
                    repo=repo,
                    number=number,
                    detail=f"{sha[:8]} {want.status}"
                    + (f"/{want.conclusion}" if want.conclusion else "")
                    + f": {want.title}",
                )
            if row is None or created != row["check_id"]:
                stats.created += 1
            else:
                stats.updated += 1
        return stats.as_dict()

    def _auth_failed(self, conn: sqlite3.Connection, exc: ReviewError) -> None:
        log.warning("check runs: GitHub App authentication failed: %s", exc)
        if exc.kind is not FailKind.INFRA and str(exc) != self.auth_error:
            self.auth_error = str(exc)  # recorded once per distinct error, not every pass
            with tx(conn):
                event(conn, "check.auth_failed", detail=str(exc)[:300])

    def _write(self, gh: Gh, repo: str, sha: str, want: Want, row: sqlite3.Row | None) -> int:
        """Create or update the commit's check; returns its id. A completed check is never
        reopened in place: a commit going back to pending (a retry after a failure, a skip
        label removed) gets a new check run, which GitHub shows instead of the old one."""
        if row is not None and not (row["status"] == "completed" and want.status != "completed"):
            gh.api(f"repos/{repo}/check-runs/{row['check_id']}", method="PATCH", body=_body(want))
            return int(row["check_id"])
        data = gh.api(
            f"repos/{repo}/check-runs",
            method="POST",
            body=_body(want, name=self.cfg.checks.name, sha=sha),
        )
        if not isinstance(data, dict) or not data.get("id"):
            raise ReviewError(FailKind.CONTRACT, "check run response has no id")
        return int(data["id"])


def plan(conn: sqlite3.Connection, cfg: Config) -> list[dict[str, Any]]:
    """Every commit's wanted check next to what was last written; reads no GitHub."""
    excluded = {r.lower() for r in cfg.checks.exclude_repos}
    out = []
    for repo, number, sha in _candidates(conn):
        if repo.lower() in excluded:
            continue
        want = wanted(conn, repo, number, sha)
        row = conn.execute(
            "SELECT check_id, digest FROM check_runs WHERE repo=? AND number=? AND sha=?",
            (repo, number, sha),
        ).fetchone()
        if want is None and row is None:
            continue
        out.append(
            {
                "pr": f"{repo}#{number}",
                "sha": sha[:12],
                "status": want.status if want else None,
                "conclusion": want.conclusion if want else None,
                "title": want.title if want else None,
                "check_id": row["check_id"] if row else None,
                "in_sync": bool(row and want and row["digest"] == want.digest),
            }
        )
    return out


def app_auth(cfg: Config) -> AppAuth:
    return AppAuth(cfg.checks.app_id, cfg.checks.private_key, gh_bin=cfg.gh_bin)
