"""PR-level GitHub reads/writes used by the worker (evidence, reviews, comments, gate comment)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .gh import Gh
from .models import FailKind, ReviewError

GATE_MARKER = "<!-- thepastaclaw-gate v1 -->"
REVIEW_MARKER = "<!-- thepastaclaw-review v1 -->"
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "body": self.body,
            "baseRefName": self.base_ref,
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
        if cs:
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


def upsert_gate_comment(gh: Gh, repo: str, number: int, bot_login: str, body: str) -> None:
    existing = find_gate_comment(gh, repo, number, bot_login)
    if existing:
        gh.api(
            f"repos/{repo}/issues/comments/{existing['id']}", method="PATCH", body={"body": body}
        )
    else:
        gh.api(f"repos/{repo}/issues/{number}/comments", method="POST", body={"body": body})


def gate_body(
    status: str,
    sha: str,
    *,
    phase: str | None = None,
    blocker_count: int | None = None,
    queue_ahead: int | None = None,
    reason: str | None = None,
) -> str:
    s = sha[:8]
    if status == "queued":
        q = "next in queue" if not queue_ahead else f"{queue_ahead} ahead in queue"
        return f"{GATE_MARKER}\n🕓 Ready for review — {q} (commit {s})"
    if status == "in_progress":
        return f"{GATE_MARKER}\n🔍 Review in progress — actively reviewing now (commit {s})"
    if status == "done":
        if phase == "preliminary":
            return f"{GATE_MARKER}\n⛔ Blockers found — Phase 2 deferred (commit {s})\n_Canonical validated blockers: {blocker_count or 0}_"
        n = blocker_count or 0
        head = "⛔" if n else "✅"
        return f"{GATE_MARKER}\n{head} Final review complete — {'no blockers' if not n else f'{n} blocking finding(s)'} (commit {s})"
    if status == "failed":
        return f"{GATE_MARKER}\n⚠️ Automated review could not complete (commit {s})\n_Reason: {reason or 'unknown'}_"
    return f"{GATE_MARKER}\n{status} (commit {s})"


def post_reply(gh: Gh, repo: str, number: int, comment_id: int, body: str) -> None:
    gh.api(
        f"repos/{repo}/pulls/{number}/comments/{comment_id}/replies",
        method="POST",
        body={"body": body},
    )


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
