"""PR-level GitHub reads/writes used by the worker (evidence, reviews, comments, gate comment)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .db import now
from .dedupe import FINDING_MARKER_RE
from .gh import Gh
from .models import FailKind, ReviewError
from .progress import LABELS as STEP_LABELS

GATE_MARKER = "<!-- thepastaclaw-gate v1 -->"
REVIEW_MARKER = "<!-- thepastaclaw-review v1 -->"
# stand-in models were in use; prefixes gate comments, queue comments and review titles
DEGRADED_BADGE = "⚠️ DEGRADED"
CODERABBIT_USER = "coderabbitai[bot]"

NON_ACTIONABLE_CR = (
    "confirmed as addressed",
    "this is addressed",
    "already addressed",
    "thanks for the update",
    "thanks for addressing",
    "looks good now",
    "no further action needed",
    "resolving this thread",
    "muted this thread",
    "notification settings",
)

THREADS_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) { pullRequest(number: $number) {
    reviewThreads(first: 100, after: $cursor) {
      pageInfo { hasNextPage endCursor }
      nodes { id isResolved isOutdated path line
        comments(first: 50) { nodes { databaseId body author { login } createdAt authorAssociation } } }
    } } } }
"""


@dataclass(slots=True)
class PrMeta:
    title: str
    body: str
    base_ref: str
    head_sha: str
    author: str
    state: str
    is_draft: bool
    url: str
    merged: bool
    base_sha: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "body": self.body,
            "baseRefName": self.base_ref,
            "baseRefOid": self.base_sha,
            "headRefOid": self.head_sha,
            "author": self.author,
            "state": self.state,
            "isDraft": self.is_draft,
            "url": self.url,
        }


def pr_meta(gh: Gh, repo: str, number: int) -> PrMeta:
    d = gh.api(f"repos/{repo}/pulls/{number}")
    if not isinstance(d, dict):
        raise ReviewError(FailKind.CONTRACT, "pulls endpoint returned non-object")
    return PrMeta(
        title=str(d.get("title") or ""),
        body=str(d.get("body") or ""),
        base_ref=str((d.get("base") or {}).get("ref") or ""),
        base_sha=str((d.get("base") or {}).get("sha") or ""),
        head_sha=str((d.get("head") or {}).get("sha") or ""),
        author=str((d.get("user") or {}).get("login") or ""),
        state=str(d.get("state") or ""),
        is_draft=bool(d.get("draft")),
        url=str(d.get("html_url") or ""),
        merged=bool(d.get("merged")),
    )


def review_threads(gh: Gh, repo: str, number: int) -> list[dict[str, Any]]:
    owner, name = repo.split("/", 1)
    out: list[dict[str, Any]] = []
    cursor: str | None = None
    for _ in range(10):
        v: dict[str, Any] = {"owner": owner, "name": name, "number": number}
        if cursor:
            v["cursor"] = cursor
        data = gh.graphql(THREADS_QUERY, v) or {}
        conn = ((data.get("repository") or {}).get("pullRequest") or {}).get("reviewThreads") or {}
        for n in conn.get("nodes") or []:
            comments = [
                {
                    "id": c.get("databaseId"),
                    "author": (c.get("author") or {}).get("login"),
                    "body": c.get("body") or "",
                    "created_at": c.get("createdAt"),
                    "association": c.get("authorAssociation"),
                }
                for c in (n.get("comments") or {}).get("nodes") or []
            ]
            out.append(
                {
                    "thread_id": n.get("id"),
                    "is_resolved": bool(n.get("isResolved")),
                    "is_outdated": bool(n.get("isOutdated")),
                    "path": n.get("path"),
                    "line": n.get("line"),
                    "comments": comments,
                }
            )
        if not conn.get("pageInfo", {}).get("hasNextPage"):
            break
        cursor = conn["pageInfo"]["endCursor"]
    return out


def issue_comments(gh: Gh, repo: str, number: int) -> list[dict[str, Any]]:
    rows = gh.api(f"repos/{repo}/issues/{number}/comments?per_page=100", paginate=True) or []
    return [r for r in rows if isinstance(r, dict)]


def inline_comments(gh: Gh, repo: str, number: int) -> list[dict[str, Any]]:
    rows = gh.api(f"repos/{repo}/pulls/{number}/comments?per_page=100", paginate=True) or []
    return [r for r in rows if isinstance(r, dict)]


def reviews(gh: Gh, repo: str, number: int) -> list[dict[str, Any]]:
    rows = gh.api(f"repos/{repo}/pulls/{number}/reviews?per_page=100", paginate=True) or []
    return [r for r in rows if isinstance(r, dict)]


def pr_diff(gh: Gh, repo: str, number: int) -> str | None:
    """Unified diff, or None when GitHub refuses it as too large (body-only fallback)."""
    try:
        return gh.run("pr", "diff", str(number), "--repo", repo, timeout=300)
    except ReviewError as exc:
        msg = exc.message.lower()
        if "too_large" in msg or "exceeded the maximum number of lines" in msg:
            return None
        raise


def _is_bot(comment: dict[str, Any], bot_login: str) -> bool:
    """True when a review-thread comment was authored by us."""
    return str(comment.get("author") or "").lower() == bot_login.lower()


def _is_other_bot(comment: dict[str, Any]) -> bool:
    """A GitHub App or another review bot: part of the thread, but never someone waiting on
    an answer from us (answering it would start a bot-to-bot loop)."""
    login = str(comment.get("author") or "").lower()
    return login.endswith("[bot]") or login == CODERABBIT_USER.removesuffix("[bot]")


def evidence_bundle(
    gh: Gh, repo: str, number: int, meta: PrMeta, bot_login: str, *, include_coderabbit: bool
) -> dict[str, Any]:
    """PR discussion evidence for prompts: metadata, human issue comments, review threads.
    Bot-authored (our own) review bodies and CodeRabbit content are filtered unless requested."""
    threads = review_threads(gh, repo, number)
    comments = issue_comments(gh, repo, number)

    def keep_author(login: str | None) -> bool:
        login = (login or "").lower()
        if login == bot_login.lower():
            return False
        return include_coderabbit or login != CODERABBIT_USER

    ev_threads = []
    for t in threads:
        cs = [c for c in t["comments"] if keep_author(c.get("author"))]
        if not cs:
            continue
        root = t["comments"][0]
        if _is_bot(root, bot_login):
            # a human replied under one of our findings (keep_author dropped the root): the
            # reply only makes sense with the finding it answers, so keep it in this thread only
            cs = [root, *cs]
        ev_threads.append({**t, "comments": cs})
    ev_comments = [
        {
            "author": (c.get("user") or {}).get("login"),
            "body": str(c.get("body") or "")[:4000],
            "created_at": c.get("created_at"),
        }
        for c in comments
        if keep_author((c.get("user") or {}).get("login"))
        and GATE_MARKER not in str(c.get("body") or "")
    ]
    return {"pr": meta.as_dict(), "issue_comments": ev_comments, "review_threads": ev_threads}


def ci_checks(gh: Gh, repo: str, sha: str) -> dict[str, Any]:
    """The head's CI as reviewers need it: check runs and commit statuses, one line each,
    as of `fetched_at` (the run's later lanes read it hours on). The first 100 of each;
    `truncated` says when GitHub has more. Best effort: lanes can still ask `gh pr checks`
    themselves, so a failed read only notes the error instead of failing the run."""
    fetched_at = now()
    try:
        runs = gh.api(f"repos/{repo}/commits/{sha}/check-runs?per_page=100") or {}
        status = gh.api(f"repos/{repo}/commits/{sha}/status?per_page=100") or {}
    except Exception as exc:  # evidence only, never the run's problem
        return {"error": str(exc)[:300]}
    checks = [
        {
            "name": r.get("name"),
            "status": r.get("status"),
            "conclusion": r.get("conclusion"),
            "url": r.get("html_url") or r.get("details_url"),
        }
        for r in (runs.get("check_runs") or [])
        if isinstance(r, dict)
    ]
    checks += [
        {
            "name": st.get("context"),
            "status": "completed" if st.get("state") != "pending" else "in_progress",
            "conclusion": st.get("state"),
            "url": st.get("target_url"),
        }
        for st in (status.get("statuses") or [])
        if isinstance(st, dict)
    ]
    total = int(runs.get("total_count") or 0) + int(status.get("total_count") or 0)
    return {
        "head_sha": sha,
        "fetched_at": fetched_at,
        "checks": checks,
        "truncated": total > len(checks),
    }


def finding_threads(threads: list[dict[str, Any]], bot_login: str) -> dict[str, dict[str, Any]]:
    """finding_hash -> thread facts for every unresolved bot finding thread.

    `awaiting_answer` is True when the newest human reply is newer than the bot's newest
    answer in that thread (someone is waiting on us). Resolved threads are left out entirely:
    a maintainer who closed the discussion does not want it reopened by a bot comment."""
    out: dict[str, dict[str, Any]] = {}
    for t in threads:
        cs = t.get("comments") or []
        if not cs or t.get("is_resolved"):
            continue
        root = cs[0]
        if not _is_bot(root, bot_login):
            continue
        body = str(root.get("body") or "")
        m = FINDING_MARKER_RE.search(body)
        if not m:
            continue
        # `transcript` is the whole exchange after the root, our own earlier answers included,
        # so a reviewer can see what it already said and never repeat itself; `replies` is the
        # human side only and drives the awaiting/answered bookkeeping
        transcript = [
            {
                "id": c.get("id"),
                "author": c.get("author"),
                "association": c.get("association"),
                "body": str(c.get("body") or "")[:4000],
                "created_at": c.get("created_at"),
                "is_bot": _is_bot(c, bot_login),
            }
            for c in cs[1:]
        ]
        replies = [
            {k: v for k, v in c.items() if k != "is_bot"}
            for c in transcript
            if not c["is_bot"] and not _is_other_bot(c)
        ]
        last_bot = max(
            (str(c.get("created_at") or "") for c in cs[1:] if _is_bot(c, bot_login)),
            default="",
        )
        awaiting = bool(replies) and str(replies[-1].get("created_at") or "") > last_bot
        severity, title = _finding_title(body)
        out[m.group(1)] = {
            "comment_id": root.get("id"),
            "thread_id": t.get("thread_id"),
            "awaiting_answer": awaiting,
            "latest_reply_id": replies[-1].get("id") if replies else None,
            # Keep the bot's own answers so publishing can make open-thread reconciliation
            # idempotent across new heads.  A thread can remain open when the resolve mutation
            # races with a reply; that must not cause the same status note to be posted again.
            "bot_answers": [
                {
                    "id": c.get("id"),
                    "body": str(c.get("body") or "")[:4000],
                    "created_at": c.get("created_at"),
                }
                for c in cs[1:]
                if _is_bot(c, bot_login)
                and "<!-- thepastaclaw-thread-answer v1" in str(c.get("body") or "")
            ],
            "path": t.get("path"),
            "line": t.get("line"),
            "severity": severity,
            "title": title,
            "body": body,
            "replies": replies,
            "transcript": transcript,
        }
    return out


def _finding_title(body: str) -> tuple[str, str]:
    """(severity, title) from a posted finding comment's bold headline."""
    for line in body.splitlines():
        s = line.strip()
        if not (s.startswith("**") and s.endswith("**")):
            continue
        label, _, title = s.strip("*").partition(":")
        low = label.lower()
        if "blocking" in low:
            severity = "blocking"
        elif "suggestion" in low:
            severity = "suggestion"
        else:
            severity = "nitpick"
        return severity, (title or label).strip()
    return "nitpick", ""


def coderabbit_context(threads: list[dict[str, Any]]) -> dict[str, Any]:
    """Active CodeRabbit inline findings with concrete comment ids (port of coderabbit_context.py)."""
    findings = []
    for t in threads:
        if t.get("is_resolved"):
            continue
        root = (t.get("comments") or [None])[0]
        if not root or (root.get("author") or "").lower() != CODERABBIT_USER:
            continue
        body = str(root.get("body") or "")
        if any(p in body.lower() for p in NON_ACTIONABLE_CR):
            continue
        replies = t["comments"][1:]
        findings.append(
            {
                "comment_id": root.get("id"),
                "thread_id": t.get("thread_id"),
                "path": t.get("path"),
                "line": t.get("line"),
                "body": body[:6000],
                "replies": [
                    {"author": r.get("author"), "body": str(r.get("body") or "")[:2000]}
                    for r in replies
                ],
                "is_outdated": t.get("is_outdated"),
            }
        )
    return {"findings": findings, "count": len(findings)}


def existing_review_for_sha(
    gh: Gh, repo: str, number: int, sha: str, phase: str, bot_login: str
) -> dict[str, Any] | None:
    marker = f"phase={phase} sha={sha}"
    for r in reviews(gh, repo, number):
        if (r.get("user") or {}).get("login") == bot_login and marker in str(r.get("body") or ""):
            return r
    return None


def post_review(gh: Gh, repo: str, number: int, payload: dict[str, Any]) -> dict[str, Any]:
    d = gh.api(f"repos/{repo}/pulls/{number}/reviews", method="POST", body=payload, timeout=300)
    if not isinstance(d, dict) or not d.get("id"):
        raise ReviewError(
            FailKind.INFRA, f"review POST returned unexpected payload: {json.dumps(d)[:200]}"
        )
    return d


def find_gate_comment(gh: Gh, repo: str, number: int, bot_login: str) -> dict[str, Any] | None:
    for c in issue_comments(gh, repo, number):
        if (c.get("user") or {}).get("login") == bot_login and GATE_MARKER in str(
            c.get("body") or ""
        ):
            return c
    return None


def upsert_gate_comment(gh: Gh, repo: str, number: int, bot_login: str, body: str) -> int | None:
    """Write the gate comment; returns its id (None if GitHub did not say)."""
    existing = find_gate_comment(gh, repo, number, bot_login)
    if existing:
        gh.api(
            f"repos/{repo}/issues/comments/{existing['id']}", method="PATCH", body={"body": body}
        )
        return int(existing["id"])
    created = gh.api(f"repos/{repo}/issues/{number}/comments", method="POST", body={"body": body})
    return int(created["id"]) if isinstance(created, dict) and created.get("id") else None


DASHBOARD_URL = "https://thepastaclaw.github.io/review-system/"
# setup steps say little a PR author cares about once done: they show only while running (so
# the comment says where the run is from its first minute), the rest are the review's milestones
_SETUP_STEPS = frozenset({"worktree", "select", "context"})
_ESTIMATED_FROM = {
    "tier": "Estimated from recent reviews of this tier",
    "reviews": "Estimated from recent reviews",
    "conversation": "Estimated from recent conversations",
}
_STEP_MARKS = {
    "ok": "✅",
    "running": "⏳",
    "failed": "❌",
    "skipped": "⏭️",
    "cancelled": "⏹️",
    "upcoming": "▫️",
}


def _fmt_span(seconds: int) -> str:
    m = seconds // 60
    return f"{m} min" if m < 60 else f"{m // 60} h {m % 60} min"


def _fmt_left(seconds: int) -> str:
    """Rounded, since it is an estimate: to 5 minutes, or 10 above an hour."""
    m = seconds / 60
    if m < 3:
        return "finishing up"
    if m < 60:
        return f"about {max(5, round(m / 5) * 5)} min left"
    m = round(m / 10) * 10
    return f"about {m // 60} h {m % 60} min left" if m % 60 else f"about {m // 60} h left"


def progress_lines(progress: dict[str, Any]) -> list[str]:
    """The live progress under an in-progress gate line: a bar with the estimated share done
    and time left, the review's steps (done, running, upcoming) and a link to the dashboard."""
    lines = []
    fraction = progress.get("fraction")
    if fraction is not None:
        filled = round(fraction * 20)
        left = progress.get("remaining_seconds")
        bits = [f"`{'█' * filled}{'░' * (20 - filled)}` **{int(fraction * 100)}%**"]
        if progress.get("overdue"):
            bits.append("taking longer than usual")
        elif left is not None:
            bits.append(_fmt_left(int(left)))
        bits.append(f"running for {_fmt_span(int(progress.get('elapsed_seconds', 0)))}")
        lines.append(" · ".join(bits))
    chips = []
    steps = progress.get("steps", [])
    # one prep lane chooses the lanes and rates the tier: its triage chip stands for both
    prep = any(st[0] == "triage" for st in steps)
    for name, status, *note in steps:
        if (name in _SETUP_STEPS and status != "running") or (name == "select" and prep):
            continue
        label = STEP_LABELS.get(name, name)
        label = f"**{label}**" if status == "running" else label
        if note and note[0]:
            label += f" ({note[0]})"
        chips.append(f"{_STEP_MARKS.get(status, '▫️')} {label}")
    if chips:
        lines.append(" → ".join(chips))
    basis = _ESTIMATED_FROM.get(progress.get("basis") or "tier") if fraction is not None else None
    footer = [
        *([basis] if basis else []),
        f"updated {progress.get('updated_at', '')} UTC",
        f"[live progress]({DASHBOARD_URL}#run={progress.get('run_id')})",
    ]
    lines.append(f"<sub>{' · '.join(footer)}</sub>")
    return lines


# the parts of an in-progress body that move by themselves: the bar line (share, time left,
# time running) and the update time; "taking longer than usual" still counts as a change
_UPDATED = r"updated \d\d:\d\d UTC"
_MOVING = re.compile(rf"^`[█░]+`.*$|{_UPDATED}", re.MULTILINE)


def gate_digest(body: str) -> str:
    """`body` without what moves on its own, i.e. what a reader would notice changing: the
    status line, the step chips, the estimate's basis and whether it is overdue. The worker
    edits a live gate comment when this changes, else at most every few minutes."""
    return _MOVING.sub(lambda m: "overdue" if "longer than usual" in m.group(0) else "", body)


def without_update_time(body: str) -> str:
    return re.sub(_UPDATED, "", body)


# why a run reviewed with Phase 2 alone (RunContext.phase1_skipped starts with one of these)
PHASE1_BACKLOG = "skipped for throughput:"
PHASE1_FAILED = "failed on every model; its output was dropped"
PHASE1_REPO_OFF = "disabled for this repository"


def phase2_only_label(phase1_skipped: str) -> str:
    """Short label for a Phase-2-only review, for titles and the gate comment."""
    if phase1_skipped.startswith(PHASE1_BACKLOG):
        return "queue backlog"
    if phase1_skipped == PHASE1_FAILED:
        return "Phase 1 failed"
    if phase1_skipped == PHASE1_REPO_OFF:
        return "no Phase 1 for this repository"
    return "Phase 1 not run"


def gate_body(
    status: str,
    sha: str,
    *,
    phase: str | None = None,
    blocker_count: int | None = None,
    queue_ahead: int | None = None,
    reason: str | None = None,
    tier: str | None = None,
    phase2_skipped: str | None = None,
    phase1_skipped: str | None = None,
    adhoc: bool = False,
    degraded: bool = False,
    points: int | None = None,
    block_above: int | None = None,
    progress: dict[str, Any] | None = None,
) -> str:
    s = sha[:8]
    t = f" · triage: {tier}" if tier else ""
    if phase1_skipped:
        t += f" · Phase 2 only ({phase2_only_label(phase1_skipped)})"
    if adhoc:
        t += " · ad hoc (no repo skill)"
    if degraded:
        t += " · stand-in models (primary models out of quota)"
    # the degraded badge replaces the status icon so the line reads as one warning
    warn = f"{DEGRADED_BADGE} —" if degraded else ""
    if status == "queued":
        q = "next in queue" if not queue_ahead else f"{queue_ahead} ahead in queue"
        return f"{GATE_MARKER}\n{warn or '🕓'} Ready for review — {q} (commit {s})"
    if status == "in_progress":
        line = f"{GATE_MARKER}\n{warn or '🔍'} Review in progress — actively reviewing now (commit {s}){t}"
        return "\n".join([line, *progress_lines(progress)]) if progress else line
    if status == "done":
        if phase == "preliminary":
            # the score is only passed when the points gate deferred Phase 2
            what = (
                "Phase-1 findings over the gate"
                if points is not None and not blocker_count
                else "Blockers found"
            )
            score = (
                f" · gate points: {points} (Phase 2 deferred above {block_above})"
                if points is not None and block_above is not None
                else ""
            )
            return f"{GATE_MARKER}\n{warn or '⛔'} {what} — Phase 2 deferred (commit {s}){t}\n_Canonical validated blockers: {blocker_count or 0}{score}_"
        n = blocker_count or 0
        head = warn or ("⛔" if n else "✅")
        scope = " — Phase 1 only" if phase2_skipped else ""
        return f"{GATE_MARKER}\n{head} Final review complete{scope} — {'no blockers' if not n else f'{n} blocking finding(s)'} (commit {s}){t}"
    if status == "closed":  # the PR closed or merged before its review finished
        return f"{GATE_MARKER}\n⏹️ Not reviewed — {reason or 'the PR is closed'} (commit {s})"
    if status == "failed":
        return f"{GATE_MARKER}\n{warn or '⚠️'} Automated review could not complete (commit {s})\n_Reason: {reason or 'unknown'}_"
    head = f"{warn} " if warn else ""
    return f"{GATE_MARKER}\n{head}{status} (commit {s})"


def post_reply(gh: Gh, repo: str, number: int, comment_id: int, body: str) -> None:
    gh.api(
        f"repos/{repo}/pulls/{number}/comments/{comment_id}/replies",
        method="POST",
        body={"body": body},
    )


def edit_review_comment(gh: Gh, repo: str, comment_id: int, body: str) -> None:
    gh.api(f"repos/{repo}/pulls/comments/{comment_id}", method="PATCH", body={"body": body})


def react(gh: Gh, repo: str, comment_id: int, content: str) -> None:
    gh.api(
        f"repos/{repo}/pulls/comments/{comment_id}/reactions",
        method="POST",
        body={"content": content},
    )


def resolve_thread(gh: Gh, thread_id: str) -> None:
    gh.graphql(
        "mutation($id: ID!) { resolveReviewThread(input: {threadId: $id}) { thread { id isResolved } } }",
        {"id": thread_id},
    )


def thread_for_comment(gh: Gh, node_id: str) -> dict[str, Any] | None:
    if not node_id:
        return None
    q = "query($id: ID!) { node(id: $id) { ... on PullRequestReviewComment { pullRequestReviewThread { id isResolved comments(first: 50) { nodes { body author { login } } } } } } }"
    try:
        data = gh.graphql(q, {"id": node_id}) or {}
    except ReviewError:
        return None
    node = data.get("node") or {}
    t = node.get("pullRequestReviewThread")
    return t if isinstance(t, dict) else None


def viewer_login(gh: Gh) -> str:
    d = gh.api("user")
    return str((d or {}).get("login") or "")
