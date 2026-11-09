"""The review as a GitHub check run on each PR's head commit.

One check, `[checks] name` ("PastaClaw review"), per commit the bot is responsible for:

- `queued` while the review waits (push debounce, its turn in line, a retry backoff);
- `in_progress` while a run reviews it;
- `completed` once a final verdict stands for the commit (its own, or one carried over a push
  that left the diff unchanged): `success` = APPROVE, `neutral` = COMMENT,
  `failure` = REQUEST_CHANGES;
- `completed` / `neutral` when a maintainer explicitly dismissed the standing bot review:
  an override, not a claim of technical validation;
- `completed` / `action_required` when the review gave up (failed after its retries, or ended
  without a verdict): someone has to ask again (`@thepastaclaw review`);
- `completed` / `skipped` when a skip label keeps automatic review off the commit;
- `completed` / `cancelled` for a commit that stopped being the PR's head, or a PR that closed
  or went back to draft, before it was reviewed.

Drafts and PRs the bot never reviews (bot authors) get no check.

A final verdict wins over later activity on the same commit: a conversation or CI re-check
keeps the check completed and replaces the conclusion if the verdict moves, so a check that
said `success` never goes back to pending without a push. A run that a Phase 1 gate stopped
ends on its preliminary review, which stands the same way once the run is over.

This is reconciliation, not an event hook: every pass derives the wanted check of every open
PR's head (and of every commit whose check is still open) and writes only what differs from
what was last written. Bounded live reads of the exact backing review observe explicit human
dismissals even when the original database verdict hasn't changed. Confirmed dismissal
identity, head, actor, reason and time are persisted separately from that publication record,
so a restart, retry or stale API snapshot cannot reassert it. GitHub only accepts check runs
from a GitHub App, so writes go through the App's installation (ghapp.py); a repository without
it is skipped.

Check runs belong to a commit, not to a PR: one commit open in two PRs of the same repository
(rare: the same branch proposed to two bases) shows whichever PR's check was written last.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from urllib.parse import quote

from .config import Config
from .db import event, kv_get, kv_set, now, now_dt, parse_ts, tx
from .gh import Gh
from .ghapp import AppAuth, AppAuthError
from .ingest import AUTOMATIC_TRIGGERS, reviewed_sha
from .models import FailKind, HeadStatus, ReviewError, Trigger
from .publish import canonical_event

log = logging.getLogger(__name__)

# writes per pass: the first pass after enabling backfills every open PR, spread over passes
MAX_WRITES_PER_PASS = 20
MAX_READS_PER_PASS = 20
# after a transient GitHub failure the pass stops and the next one waits this long
RETRY_BACKOFF = timedelta(minutes=2)
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
RETRY_HINT = (
    "This is a failure of the review service, not a finding about the change. Comment "
    "`@thepastaclaw review` to try again (GitHub's Re-run button does nothing for this check)."
)


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


def _pr_url(repo: str, number: int) -> str:
    return f"https://github.com/{repo}/pull/{number}"


def _gate_url(conn: sqlite3.Connection, repo: str, number: int) -> str:
    cid = kv_get(conn, f"queue.comment_id:{repo}#{number}")
    return f"{_pr_url(repo, number)}#issuecomment-{cid}" if cid else _pr_url(repo, number)


def _summary(sha: str, *lines: str) -> str:
    return "\n\n".join(
        [f"Commit `{sha[:12]}`.", *filter(None, lines), f"Review queue and history: {STATUS_PAGE}"]
    )


def _standing(
    conn: sqlite3.Connection, repo: str, number: int, sha: str
) -> tuple[str, sqlite3.Row | None, int | None]:
    """The exact substantive review backing this check, never a later inline-only COMMENT."""
    target = reviewed_sha(conn, repo, number, sha)
    rows = conn.execute(
        "SELECT event, phase, posted_at, github_review_id FROM reviews WHERE repo=? AND number=?"
        " AND sha=? ORDER BY posted_at DESC, id DESC",
        (repo, number, target),
    ).fetchall()
    standing = next((r for r in rows if r["phase"] == "final"), None)
    head = conn.execute(
        "SELECT status FROM heads WHERE repo=? AND number=? AND sha=? AND queue='live'",
        (repo, number, sha),
    ).fetchone()
    if standing is None and rows and head is not None and head["status"] == HeadStatus.DONE:
        standing = rows[0]
    # A canonical-only update on our own PR has no new GitHub review. Bind it to the
    # preceding review of the SAME phase, not an unrelated preliminary/inline review.
    review_id = next(
        (
            r["github_review_id"]
            for r in rows
            if standing is not None and r["phase"] == standing["phase"] and r["github_review_id"]
        ),
        None,
    )
    return target, standing, review_id


def _dismissal_key(repo: str, number: int, review_id: int) -> str:
    return f"checks.dismissal:{repo}#{number}:{review_id}"


def _dismissal(
    conn: sqlite3.Connection, repo: str, number: int, target: str, review_id: int
) -> dict[str, Any] | None:
    raw = kv_get(conn, _dismissal_key(repo, number, review_id))
    data = json.loads(raw) if raw else None
    return data if data and data.get("sha") == target else None


def _verdict(
    conn: sqlite3.Connection,
    repo: str,
    number: int,
    sha: str,
    head: sqlite3.Row | None,
    live: bool,
) -> Want | None:
    """The completed check of the final review (or a finished Phase 1 gate) standing for sha.
    A confirmed maintainer dismissal overrides the immutable publication record."""
    target, standing, review_id = _standing(conn, repo, number, sha)
    if standing is None:
        return None
    dismissal = _dismissal(conn, repo, number, target, review_id) if review_id else None
    if dismissal:
        return Want(
            status="completed",
            conclusion="neutral",
            title="Review dismissed — maintainer override",
            summary=_summary(
                sha,
                f"@{dismissal['actor']} dismissed review [{review_id}]"
                f"({_pr_url(repo, number)}#pullrequestreview-{review_id}) of `{target}`"
                f" at {dismissal['dismissed_at']}.",
                "Dismissal reason (as supplied to GitHub):\n\n"
                + "\n".join("> " + line for line in dismissal["reason"].splitlines()),
                "The maintainer's decision is authoritative. This check is non-blocking; "
                "it does not claim the findings were fixed or technical validation passed. "
                "Other CI checks are unchanged.",
            ),
            details_url=f"{_pr_url(repo, number)}#pullrequestreview-{review_id}",
            completed_at=dismissal["dismissed_at"],
        )
    verdict = canonical_event(str(standing["event"]))
    conclusion, title = VERDICT_CONCLUSIONS.get(verdict, ("neutral", "Reviewed"))
    busy = live and head is not None and head["status"] in ("queued", "running")
    gated = ""
    if standing["phase"] != "final":
        gated = (
            "Phase 1 found blocking issues, so Phase 2 did not run on this commit."
            if verdict == "REQUEST_CHANGES"
            else "The Phase 1 gate held Phase 2 back on this commit."
        )
    return Want(
        status="completed",
        conclusion=conclusion,
        title=title,
        summary=_summary(
            sha,
            f"The review of `{target[:12]}` stands: this push left the PR's diff unchanged."
            if target != sha
            else "",
            "A follow-up pass on this commit is running; this result is replaced if the "
            "verdict changes."
            if busy
            else "",
            gated,
        ),
        details_url=f"{_pr_url(repo, number)}#pullrequestreview-{review_id}"
        if review_id
        else _pr_url(repo, number),
        completed_at=str(standing["posted_at"]),
    )


def wanted(conn: sqlite3.Connection, repo: str, number: int, sha: str) -> Want | None:
    """The check `sha` of PR `repo#number` should carry, or None when the bot has nothing to
    say about it (no check is created; an open one is closed out by `closed_out`)."""
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
    verdict = _verdict(conn, repo, number, sha, head, live)
    if verdict is not None:
        return verdict
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
    if (
        pr is not None
        and pr["skip_label"]
        and (head is None or head["status"] == HeadStatus.CLOSED)
    ):
        return Want(
            status="completed",
            conclusion="skipped",
            title="Automatic review is off for this PR (skip label)",
            summary=_summary(
                sha, "Remove the label, or comment `@thepastaclaw review`, to get a review."
            ),
            details_url=_pr_url(repo, number),
        )
    if head is None:
        return None  # a draft, a bot author, or a push ingest has not seen yet
    status = head["status"]
    if status == HeadStatus.QUEUED:
        return _queued(conn, repo, number, sha, head)
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
            summary=_summary(sha, RETRY_HINT),
            details_url=_gate_url(conn, repo, number),
            completed_at=head["finished_at"] or None,
        )
    # closed for a draft (re-opened when it is ready) or by an operator; superseded yet
    # GitHub's head again (a revert): ingest decides what follows
    return None


def _queued(conn: sqlite3.Connection, repo: str, number: int, sha: str, head: sqlite3.Row) -> Want:
    waiting = parse_ts(str(head["eligible_at"])) > now_dt()
    if waiting and head["attempts"]:
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


def closed_out(conn: sqlite3.Connection, repo: str, number: int, sha: str) -> Want:
    """What an open check says once the bot has nothing more to say about its commit (the PR
    went back to draft, an operator closed the head, a revert): it is not left pending."""
    pr = conn.execute(
        "SELECT is_draft FROM prs WHERE repo=? AND number=?", (repo, number)
    ).fetchone()
    draft = pr is not None and bool(pr["is_draft"])
    return Want(
        status="completed",
        conclusion="cancelled",
        title="Not reviewed: the PR is a draft" if draft else "Not reviewed",
        summary=_summary(
            sha, "The review starts when the PR is marked ready for review." if draft else ""
        ),
        details_url=_pr_url(repo, number),
    )


def _disabled_repos(conn: sqlite3.Connection, cfg: Config) -> set[str]:
    """Repos with PRs on record that the bot no longer reviews: ingest stops polling them, so
    their rows are frozen (ad hoc repos have no `prs` rows and are not in here)."""
    known = {str(r[0]) for r in conn.execute("SELECT DISTINCT repo FROM prs")}
    return known - set(cfg.enabled_repos)


def _candidates(conn: sqlite3.Connection, cfg: Config) -> list[tuple[str, int, str]]:
    """Commits to reconcile, those whose check moves first: still-open checks, a cancelled
    check whose commit got a review after all (a superseded run that was past verify2
    publishes; the overlap covers a review row stamped just before it committed), commits
    being reviewed, then every open PR's head (the backfill after enabling). Of a repo the bot
    no longer reviews only open checks are visited, to close them; `exclude_repos` are left
    out entirely."""
    disabled = _disabled_repos(conn, cfg)
    rows = conn.execute(
        "SELECT repo, number, sha FROM check_runs WHERE status<>'completed'"
        " UNION ALL SELECT c.repo, c.number, c.sha FROM check_runs c"
        " WHERE c.conclusion='cancelled' AND EXISTS (SELECT 1 FROM reviews r WHERE"
        " r.repo=c.repo AND r.number=c.number AND r.sha=c.sha"
        " AND r.posted_at>=strftime('%Y-%m-%dT%H:%M:%SZ', c.updated_at, '-60 seconds'))"
    ).fetchall()
    fresh = conn.execute(
        "SELECT repo, number, sha FROM heads WHERE queue='live' AND status IN ('queued','running')"
        " ORDER BY status='running' DESC, id"
    ).fetchall()
    fresh += conn.execute(
        "SELECT repo, number, head_sha AS sha FROM prs WHERE state='open' ORDER BY updated_at DESC"
    ).fetchall()
    rows += [r for r in fresh if r["repo"] not in disabled]
    excluded = {r.lower() for r in cfg.checks.exclude_repos}
    return list(
        dict.fromkeys(
            (str(r["repo"]), int(r["number"]), str(r["sha"]))
            for r in rows
            if str(r["repo"]).lower() not in excluded
        )
    )


def _last_written(conn: sqlite3.Connection, repo: str, number: int, sha: str) -> sqlite3.Row | None:
    return conn.execute(  # type: ignore[no-any-return]
        "SELECT check_id, status, conclusion, digest FROM check_runs WHERE repo=? AND number=? AND sha=?",
        (repo, number, sha),
    ).fetchone()


def _target(
    conn: sqlite3.Connection, repo: str, number: int, sha: str, disabled: set[str]
) -> tuple[Want | None, sqlite3.Row | None]:
    """The check to write for a commit (None: nothing to write) and the row last written."""
    want = None if repo in disabled else wanted(conn, repo, number, sha)
    row = _last_written(conn, repo, number, sha)
    if want is None and row is not None and row["status"] != "completed":
        want = closed_out(conn, repo, number, sha)
    if row is None and want is not None and want.conclusion == "cancelled":
        want = None  # never open a check only to say a commit we never touched was skipped
    return want, row


def _body(want: Want) -> dict[str, Any]:
    body: dict[str, Any] = {
        "status": want.status,
        "output": {"title": want.title, "summary": want.summary},
    }
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

    def __init__(self, cfg: Config, app: AppAuth, *, review_gh: Gh | None = None) -> None:
        self.cfg = cfg
        self.app = app
        # Installation tokens deliberately have only checks:write. Review/timeline reads use
        # the daemon's normal client, not broader installation permissions.
        self.review_gh = review_gh or Gh(cfg.gh_bin, default_timeout=30)
        self.checked: dict[tuple[str, int, str], str] = {}
        # (repo, number, sha) -> digest GitHub refused (a contract error); not retried until
        # the wanted check changes or the daemon restarts
        self.refused: dict[tuple[str, int, str], str] = {}
        self.retry_at = now_dt()
        self.reported: set[str] = set()  # errors already recorded as events

    def reconcile(self, conn: sqlite3.Connection) -> dict[str, int]:
        stats = dict.fromkeys(("created", "updated", "unchanged", "deferred", "failed"), 0)
        if now_dt() < self.retry_at:
            return stats
        writes = 0
        reads = 0
        clients: dict[str, Gh | None] = {}
        disabled = _disabled_repos(conn, self.cfg)
        # Rotate bounded live reads, including checks whose database digest hasn't changed.
        for key in sorted(_candidates(conn, self.cfg), key=lambda k: self.checked.get(k, "")):
            repo, number, sha = key
            if clients.get(repo, False) is None:
                continue  # not installed there, this pass
            want = None
            try:
                target, _, review_id = _standing(conn, repo, number, sha)
                if (
                    repo not in disabled
                    and review_id
                    and not _dismissal(conn, repo, number, target, review_id)
                ):
                    if reads >= MAX_READS_PER_PASS:
                        stats["deferred"] += 1
                        continue  # don't write an unchecked, possibly dismissed verdict
                    reads += 1
                    self.checked[key] = now()
                    self._observe_dismissal(conn, repo, number, target, review_id)
                want, row = _target(conn, repo, number, sha, disabled)
                if want is None:
                    continue
                if row is not None and row["digest"] == want.digest:
                    stats["unchanged"] += 1
                    continue
                if self.refused.get(key) == want.digest:
                    continue
                if writes >= MAX_WRITES_PER_PASS:
                    stats["deferred"] += 1
                    continue
                if repo not in clients:
                    clients[repo] = self._client(conn, repo)
                gh = clients[repo]
                if gh is None:
                    continue
                writes += 1
                check_id = self._write(gh, repo, sha, want, row)
            except AppAuthError as exc:
                self._report(conn, "check.auth_failed", str(exc))
                self.retry_at = now_dt() + RETRY_BACKOFF
                stats["deferred"] += 1
                break  # nothing else in this pass can authenticate either
            except ReviewError as exc:
                if exc.kind is FailKind.INFRA:
                    log.warning("check runs: GitHub unavailable, pausing: %s", exc)
                    self.retry_at = now_dt() + RETRY_BACKOFF
                    stats["deferred"] += 1
                    break  # a hung GitHub must not hold the tick loop for a whole pass
                stats["failed"] += 1
                if want is not None:
                    self.refused[key] = want.digest
                if any(f"HTTP {c}" in exc.message for c in (401, 403, 404)):
                    # installation removed, or this repo dropped from it: look it up again
                    self.app.forget(repo)
                    clients.pop(repo, None)
                log.warning("check for %s#%s refused: %s", repo, number, exc)
                with tx(conn):
                    event(conn, "check.failed", repo=repo, number=number, detail=str(exc)[:300])
                continue
            assert want is not None
            self._record(conn, repo, number, sha, want, row, check_id)
            stats["created" if row is None or check_id != row["check_id"] else "updated"] += 1
        stats["skipped_repos"] = sum(c is None for c in clients.values())
        return stats

    def _observe_dismissal(
        self, conn: sqlite3.Connection, repo: str, number: int, target: str, review_id: int
    ) -> None:
        """Only an explicit GitHub-authorized human dismissal of this exact bot review counts.
        Approvals, resolved threads, push-triggered stale-review dismissal, and bot actions
        are not maintainer overrides. GitHub enforces permission to dismiss another user's
        review; the timeline supplies the actor and unmodified reason, absent from /reviews.
        Persist the evidence before publishing, so a retry/restart cannot resurrect it.
        """
        review = self.review_gh.api(f"repos/{repo}/pulls/{number}/reviews/{review_id}", timeout=30)
        if not isinstance(review, dict) or not (
            review.get("id") == review_id
            and review.get("commit_id") == target
            and str((review.get("user") or {}).get("login", "")).lower()
            == self.cfg.bot_login.lower()
            and review.get("state") == "DISMISSED"
        ):
            return
        timeline = self.review_gh.api(
            f"repos/{repo}/issues/{number}/timeline?per_page=100", paginate=True, timeout=30
        )
        for item in timeline or []:
            dismissed = item.get("dismissed_review") or {}
            actor = item.get("actor") or {}
            login = actor.get("login")
            if not (
                item.get("event") == "review_dismissed"
                and dismissed.get("review_id") == review_id
                and not dismissed.get("dismissal_commit_id")
                and not item.get("performed_via_github_app")
                and actor.get("type") == "User"
                and login
                and login.lower() != self.cfg.bot_login.lower()
                and item.get("created_at")
            ):
                continue
            evidence = {
                "sha": target,
                "review_id": review_id,
                "actor": login,
                "reason": str(dismissed.get("dismissal_message") or "(no reason supplied)"),
                "dismissed_at": item["created_at"],
                "timeline_event_id": item.get("id"),
            }
            with tx(conn):
                kv_set(conn, _dismissal_key(repo, number, review_id), json.dumps(evidence))
                event(
                    conn,
                    "check.review_dismissed",
                    repo=repo,
                    number=number,
                    detail=json.dumps(evidence),
                )
            return

    def _client(self, conn: sqlite3.Connection, repo: str) -> Gh | None:
        """The installation client for `repo`; None when the App is not installed there or its
        installation cannot be used (recorded once; looked up again on a later pass)."""
        try:
            return self.app.gh_for(repo)
        except AppAuthError:
            raise
        except ReviewError as exc:
            if exc.kind is FailKind.INFRA:
                raise
            self.app.forget(repo)
            self._report(conn, "check.installation_failed", f"{repo}: {exc}", repo=repo)
            return None

    def _report(self, conn: sqlite3.Connection, kind: str, detail: str, **kw: Any) -> None:
        log.warning("check runs: %s %s", kind, detail)
        if detail not in self.reported:  # once per distinct error, not every pass
            self.reported.add(detail)
            with tx(conn):
                event(conn, kind, detail=detail[:300], **kw)

    def _record(
        self,
        conn: sqlite3.Connection,
        repo: str,
        number: int,
        sha: str,
        want: Want,
        row: sqlite3.Row | None,
        check_id: int,
    ) -> None:
        with tx(conn):
            conn.execute(
                "INSERT INTO check_runs (repo, number, sha, check_id, status, conclusion, digest, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(repo, number, sha) DO UPDATE SET"
                " check_id=excluded.check_id, status=excluded.status,"
                " conclusion=excluded.conclusion, digest=excluded.digest,"
                " updated_at=excluded.updated_at",
                (repo, number, sha, check_id, want.status, want.conclusion, want.digest, now()),
            )
            # a new step title is not worth a row; a new state is
            if row is None or (row["status"], row["conclusion"]) != (want.status, want.conclusion):
                event(
                    conn,
                    "check.synced",
                    repo=repo,
                    number=number,
                    detail=f"{sha[:8]} {want.status}"
                    + (f"/{want.conclusion}" if want.conclusion else "")
                    + f": {want.title}",
                )

    def _write(self, gh: Gh, repo: str, sha: str, want: Want, row: sqlite3.Row | None) -> int:
        """Create or update the commit's check; returns its id. A completed check is never
        reopened in place: a commit going back to pending (a retry after a failure, a skip
        label removed) gets a new check run, which GitHub shows instead of the old one."""
        reopening = row is not None and row["status"] == "completed" and want.status != "completed"
        if row is not None and not reopening:
            try:
                gh.api(
                    f"repos/{repo}/check-runs/{row['check_id']}", method="PATCH", body=_body(want)
                )
                return int(row["check_id"])
            except ReviewError as exc:
                if "HTTP 404" not in exc.message:
                    raise
                row = None  # gone, or written by another App: adopt or create one
        if row is None:
            adopted = self._existing(gh, repo, sha)
            if adopted and (adopted["status"] != "completed" or want.status == "completed"):
                gh.api(f"repos/{repo}/check-runs/{adopted['id']}", method="PATCH", body=_body(want))
                return int(adopted["id"])
        body = {**_body(want), "name": self.cfg.checks.name, "head_sha": sha}
        if want.status == "completed" and not want.started_at:
            body["started_at"] = body["completed_at"]  # a backfilled verdict: not "started now"
        data = gh.api(f"repos/{repo}/check-runs", method="POST", body=body)
        if not isinstance(data, dict) or not data.get("id"):
            raise ReviewError(FailKind.CONTRACT, "check run response has no id")
        return int(data["id"])

    def _existing(self, gh: Gh, repo: str, sha: str) -> dict[str, Any] | None:
        """This App's newest check of our name on `sha`, when the database has none: a create
        whose response was lost (timeout), or a database restored from a backup. Adopting it
        keeps one check per commit instead of an orphan stuck pending beside a new one."""
        app_id = self.app.app().get("id")
        endpoint = (
            f"repos/{repo}/commits/{sha}/check-runs?check_name={quote(self.cfg.checks.name)}"
            f"&filter=latest&app_id={app_id}"
        )
        data = gh.api(endpoint)
        runs = data.get("check_runs") if isinstance(data, dict) else None
        return runs[0] if runs else None


def plan(conn: sqlite3.Connection, cfg: Config) -> list[dict[str, Any]]:
    """Every commit's wanted check next to what was last written; reads no GitHub."""
    out = []
    disabled = _disabled_repos(conn, cfg)
    for repo, number, sha in _candidates(conn, cfg):
        want, row = _target(conn, repo, number, sha, disabled)
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
