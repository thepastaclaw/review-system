"""Audit queue: post-merge reviews of PRs merged without a clean review.

A PR is *clean* when the bot's standing verdict on the exact merged head is an approval or a
final review with no blocking findings. Every other merge (never reviewed, reviewed only on an
older commit, merged over blockers, merged while only the preliminary gate had run) gets a
full review of the merged head in the audit queue, followed by a persistence check: each
blocker is re-examined against the base branch as it is *now*, because a later PR may have
fixed it already.

Audit heads are ordinary `heads` rows with `queue='audit'`. The scheduler starts them only
when no live head is waiting and never lets them occupy a live slot (see scheduler.py), and
they are left out of every live-queue count (backlog rule, queue comments, watchdog, labels).

Two sources feed the queue, through one function (`enqueue`):
- `sweep()`, a daemon task, asks GitHub search for PRs merged since its cursor in every
  watched repo. Search (not the ingest close path) because a PR opened and merged between two
  ingest polls never appears in the open set at all.
- `reviewsys audit seed --since DATE` backfills history once, ranked by the operator.

Results: a markdown report per PR (on disk, and committed to the private `audit.report_repo`
when configured) plus an index the daemon regenerates. Merges the sweep caught live also get
a "Post-merge review" comment on the PR and one issue per PR whose blockers are still
present; backfilled audits never post on GitHub.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from .config import Config
from .contract import Finding, parse_json_object
from .db import event, fmt_ts, kv_get, kv_set, now, now_dt, tx
from .gh import Gh
from .models import FailKind, ReviewError

log = logging.getLogger(__name__)

IGNORED_AUTHORS = frozenset(
    {"dependabot", "dependabot[bot]", "renovate", "renovate[bot]", "github-actions[bot]"}
)

# mechanical merges: one cheap Phase-1-only pass; escalated to a full audit on any blocker
LIGHT_TITLE_RE = re.compile(
    r"^(ci: re-pin |chore\(release\)|chore: bump (to v|version)|chore: release )"
    r"|bump version to |bump GroveDB to ",
    re.IGNORECASE,
)
# branch-to-branch syncs carry content that was (or will be) audited through its source PRs;
# a 10k+ line merge is not reviewable as one unit, so these are recorded, not run
SYNC_TITLE_RE = re.compile(
    r"^(chore: )?(merge|sync) v?\d[\w.-]* (dev )?into |^merge/.+ into ", re.IGNORECASE
)
CANONICAL_RC = "Canonical verifier result: `REQUEST_CHANGES`"
BLOCKING_COUNT_RE = re.compile(r"🔴 ([1-9]\d*) blocking")
PHASE_MARKER_RE = re.compile(r"phase=([a-z]+) sha=([0-9a-f]{40})")

MODE_FULL, MODE_LIGHT, MODE_SYNC = "full", "light", "sync"
SOURCE_LIVE, SOURCE_SEED = "live", "seed"
PERSISTENCE_STATUSES = ("STILL_PRESENT", "FIXED", "OBSOLETE", "UNKNOWN")
OPEN_STATUSES = frozenset({"STILL_PRESENT", "UNKNOWN"})

SWEEP_OVERLAP = timedelta(hours=1)
SWEEP_FIRST_LOOKBACK = timedelta(hours=6)
KV_CONCURRENCY = "audit.max_concurrent"  # operator override of `audit.max_concurrent`

MERGED_QUERY = """
query($q: String!, $cursor: String) {
  search(type: ISSUE, query: $q, first: 50, after: $cursor) {
    pageInfo { hasNextPage endCursor }
    nodes { ... on PullRequest {
      number title url mergedAt headRefOid baseRefName additions deletions
      author { login } mergedBy { login } mergeCommit { oid }
      reviews(last: 60) { nodes { author { login } state submittedAt commit { oid } body } }
    } }
  }
}
"""


@dataclass(slots=True)
class MergedPr:
    repo: str
    number: int
    title: str
    author: str
    merged_by: str
    merged_at: str
    head_sha: str
    base_ref: str
    merge_commit: str
    size: int
    reviews: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_node(cls, repo: str, n: dict[str, Any]) -> MergedPr:
        return cls(
            repo=repo,
            number=int(n["number"]),
            title=str(n.get("title") or ""),
            author=str((n.get("author") or {}).get("login") or ""),
            merged_by=str((n.get("mergedBy") or {}).get("login") or ""),
            merged_at=str(n.get("mergedAt") or ""),
            head_sha=str(n.get("headRefOid") or ""),
            base_ref=str(n.get("baseRefName") or ""),
            merge_commit=str((n.get("mergeCommit") or {}).get("oid") or ""),
            size=int(n.get("additions") or 0) + int(n.get("deletions") or 0),
            reviews=[r for r in (n.get("reviews") or {}).get("nodes") or [] if r],
        )


def fetch_merged(gh: Gh, repo: str, since: str) -> list[MergedPr]:
    """Every PR in `repo` merged at or after `since` (ISO timestamp)."""
    out: list[MergedPr] = []
    cursor: str | None = None
    for _ in range(40):  # search caps at 1000 results; 40 pages of 50 is past that
        v: dict[str, Any] = {"q": f"repo:{repo} is:pr is:merged merged:>={since}"}
        if cursor:
            v["cursor"] = cursor
        data = gh.graphql(MERGED_QUERY, v) or {}
        page = data.get("search") or {}
        out += [MergedPr.from_node(repo, n) for n in page.get("nodes") or [] if n]
        if not (page.get("pageInfo") or {}).get("hasNextPage"):
            break
        cursor = page["pageInfo"]["endCursor"]
    return out


# ---- classification ----


def _bot_reviews_at(pr: MergedPr, bot_login: str) -> list[dict[str, Any]]:
    """The bot's reviews of the merged head, oldest first, with the phase from the marker
    (`legacy` for the old pipeline's markerless reviews, which pin the commit instead)."""
    out = []
    for r in pr.reviews:
        if str((r.get("author") or {}).get("login") or "").lower() != bot_login.lower():
            continue
        body = str(r.get("body") or "")
        m = PHASE_MARKER_RE.search(body)
        sha = m.group(2) if m else str((r.get("commit") or {}).get("oid") or "")
        if sha != pr.head_sha:
            continue
        out.append(
            {
                "state": str(r.get("state") or ""),
                "phase": m.group(1) if m else "legacy",
                "at": str(r.get("submittedAt") or ""),
                "body": body,
            }
        )
    return sorted(out, key=lambda r: r["at"])


def db_verdict(
    conn: sqlite3.Connection, repo: str, number: int, sha: str
) -> tuple[str, str] | None:
    """(canonical event, phase) of the newest `reviews` row for this commit: our own record,
    which knows the canonical verdict even where GitHub only shows COMMENT transport."""
    row = conn.execute(
        "SELECT event, phase FROM reviews WHERE repo=? AND number=? AND sha=? "
        "ORDER BY posted_at DESC, id DESC LIMIT 1",
        (repo, number, sha),
    ).fetchone()
    if row is None:
        return None
    ev = {"APPROVED": "APPROVE", "CHANGES_REQUESTED": "REQUEST_CHANGES", "COMMENTED": "COMMENT"}
    return ev.get(str(row["event"]), str(row["event"])), str(row["phase"])


def is_clean(pr: MergedPr, bot_login: str, local: tuple[str, str] | None = None) -> bool:
    """Whether a clean review stands on the merged head: an approval, or a final review with
    no blocking findings. The local record wins when it has one (it knows the canonical
    verdict of bot-authored PRs); otherwise the newest bot review of that commit decides."""
    if local is not None and local[0] in {"APPROVE", "COMMENT", "REQUEST_CHANGES"}:
        return local[1] == "final" and local[0] != "REQUEST_CHANGES"
    rows = _bot_reviews_at(pr, bot_login)
    if not rows:
        return False
    last = rows[-1]
    if CANONICAL_RC in last["body"] or BLOCKING_COUNT_RE.search(last["body"]):
        return False
    if last["state"] == "APPROVED":
        return True
    return (
        last["state"] == "COMMENTED"
        and last["phase"] in {"final", "legacy"}
        and "## Preliminary review" not in last["body"]
    )


def prior_note(pr: MergedPr, bot_login: str) -> str:
    """What the bot had said on the PR by merge time, for the report: `none`, or the newest
    review's state, phase and commit."""
    best: dict[str, Any] | None = None
    for r in pr.reviews:
        if str((r.get("author") or {}).get("login") or "").lower() != bot_login.lower():
            continue
        at = str(r.get("submittedAt") or "")
        if pr.merged_at and at > pr.merged_at:
            continue
        if best is None or at > str(best.get("submittedAt") or ""):
            best = r
    if best is None:
        return "none"
    body = str(best.get("body") or "")
    m = PHASE_MARKER_RE.search(body)
    sha = m.group(2) if m else str((best.get("commit") or {}).get("oid") or "")
    where = "merged head" if sha == pr.head_sha else f"older commit {sha[:8]}"
    return f"{best.get('state')} ({m.group(1) if m else 'legacy'}) on {where}"


def mode_for(pr: MergedPr) -> str:
    if SYNC_TITLE_RE.search(pr.title):
        return MODE_SYNC
    if LIGHT_TITLE_RE.search(pr.title):
        return MODE_LIGHT
    return MODE_FULL


REPO_RISK = (
    "dashpay/platform",
    "dashpay/grovedb",
    "dashpay/tenderdash",
    "dashpay/dash",
    "dashpay/dashwallet-ios",
    "dashpay/dash-wallet",
    "dashpay/dash-evo-tool",
)


def seed_rank(pr: MergedPr, mode: str, first_mergers: tuple[str, ...]) -> int:
    """Backfill order (lower first): the named mergers' PRs, then everyone's; within that, by
    repo risk; mechanical PRs after substantive ones. Live audits use rank 0."""
    risk = REPO_RISK.index(pr.repo) if pr.repo in REPO_RISK else len(REPO_RISK)
    first = pr.merged_by.lower() in {m.lower() for m in first_mergers}
    return 100 + (0 if first else 100) + risk * 10 + (50 if mode == MODE_LIGHT else 0)


# ---- enqueue ----


def enqueue(
    conn: sqlite3.Connection,
    cfg: Config,
    pr: MergedPr,
    *,
    source: str,
    rank: int,
    ts: str | None = None,
) -> str:
    """Queue an audit of `pr`'s merged head unless one exists or a clean review stands.
    Caller holds a transaction. Returns created | clean | exists | skipped | excluded."""
    ts = ts or now()
    if pr.repo in cfg.audit.exclude_repos or pr.author.lower() in IGNORED_AUTHORS:
        return "excluded"
    if conn.execute(
        "SELECT 1 FROM audits WHERE repo=? AND number=?", (pr.repo, pr.number)
    ).fetchone():
        return "exists"
    if is_clean(pr, cfg.bot_login, db_verdict(conn, pr.repo, pr.number, pr.head_sha)):
        return "clean"
    mode = mode_for(pr)
    cols = (
        "repo, number, sha, source, mode, rank, title, author, merged_by, merged_at, "
        "merge_commit, base_ref, coverage, queued_at"
    )
    vals = (
        pr.repo,
        pr.number,
        pr.head_sha,
        source,
        mode,
        rank,
        pr.title,
        pr.author,
        pr.merged_by,
        pr.merged_at,
        pr.merge_commit,
        pr.base_ref,
        prior_note(pr, cfg.bot_login),
        ts,
    )
    if mode == MODE_SYNC:
        conn.execute(
            f"INSERT INTO audits ({cols}, finished_at, verdict) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (*vals, ts, "SKIPPED"),
        )
        event(conn, "audit.skipped", repo=pr.repo, number=pr.number, detail="branch sync merge")
        return "skipped"
    head_id = _audit_head(conn, pr, ts=ts)
    conn.execute(
        f"INSERT INTO audits ({cols}, head_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (*vals, head_id),
    )
    event(
        conn,
        "audit.queued",
        repo=pr.repo,
        number=pr.number,
        detail=f"{pr.head_sha[:8]} source={source} mode={mode} rank={rank}",
    )
    return "created"


def _audit_head(conn: sqlite3.Connection, pr: MergedPr, *, ts: str) -> int:
    """The audit head for the merged commit. `heads` is unique per (repo, number, sha), so a
    live head of that commit (usually closed by ingest when the PR left the open set) becomes
    the audit head; a live run still in flight on the PR is cancelled so the audit can start
    (a live publish would fail on the merged PR anyway)."""
    conn.execute(
        "UPDATE runs SET cancel_requested=1, reason=COALESCE(reason, 'PR merged: audit queued') "
        "WHERE status IN ('spawned','running') AND head_id IN "
        "(SELECT id FROM heads WHERE repo=? AND number=?)",
        (pr.repo, pr.number),
    )
    row = conn.execute(
        "SELECT id FROM heads WHERE repo=? AND number=? AND sha=?",
        (pr.repo, pr.number, pr.head_sha),
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE heads SET queue='audit', status='queued', trigger='audit', priority=0, "
            "attempts=0, queued_at=?, eligible_at=?, finished_at=NULL, reason=NULL WHERE id=?",
            (ts, ts, row["id"]),
        )
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO heads (repo, number, sha, trigger, priority, status, queued_at, eligible_at, queue) "
        "VALUES (?,?,?,'audit',0,'queued',?,?,'audit')",
        (pr.repo, pr.number, pr.head_sha, ts, ts),
    )
    return int(cur.lastrowid or 0)


def sweep(conn: sqlite3.Connection, cfg: Config, gh: Gh) -> dict[str, dict[str, int]]:
    """Daemon task: queue audits for PRs merged since each repo's cursor."""
    if not cfg.audit.enabled:
        return {}
    results: dict[str, dict[str, int]] = {}
    for repo in cfg.enabled_repos:
        if repo in cfg.audit.exclude_repos:
            continue
        key = f"audit.sweep_cursor:{repo}"
        start = now_dt()
        since = kv_get(conn, key) or fmt_ts(start - SWEEP_FIRST_LOOKBACK)
        try:
            prs = fetch_merged(gh, repo, since)
        except ReviewError as exc:
            log.warning("audit sweep %s failed: %s", repo, exc)
            continue
        stats: dict[str, int] = {}
        with tx(conn):
            for pr in prs:
                res = enqueue(conn, cfg, pr, source=SOURCE_LIVE, rank=0)
                stats[res] = stats.get(res, 0) + 1
            kv_set(conn, key, fmt_ts(start - SWEEP_OVERLAP))
        results[repo] = stats
    return results


def seed(
    conn: sqlite3.Connection,
    cfg: Config,
    gh: Gh,
    *,
    since: str,
    repos: tuple[str, ...] = (),
    first_mergers: tuple[str, ...] = (),
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Backfill: rank and queue every unclean merge since `since`. Returns one row per PR."""
    rows: list[dict[str, Any]] = []
    for repo in repos or cfg.enabled_repos:
        for pr in fetch_merged(gh, repo, since):
            mode = mode_for(pr)
            rank = seed_rank(pr, mode, first_mergers)
            if dry_run:
                clean = is_clean(
                    pr, cfg.bot_login, db_verdict(conn, pr.repo, pr.number, pr.head_sha)
                )
                excluded = (
                    pr.repo in cfg.audit.exclude_repos or pr.author.lower() in IGNORED_AUTHORS
                )
                res = "excluded" if excluded else "clean" if clean else f"would-queue:{mode}"
            else:
                with tx(conn):
                    res = enqueue(conn, cfg, pr, source=SOURCE_SEED, rank=rank)
            rows.append(
                {
                    "repo": repo,
                    "number": pr.number,
                    "merged_by": pr.merged_by,
                    "mode": mode,
                    "rank": rank,
                    "result": res,
                    "title": pr.title,
                }
            )
    return rows


def concurrency(conn: sqlite3.Connection, cfg: Config) -> int:
    """Audit runs allowed at once: the operator override when set, else the config."""
    raw = kv_get(conn, KV_CONCURRENCY)
    try:
        return max(0, int(raw)) if raw is not None else cfg.audit.max_concurrent
    except ValueError:
        return cfg.audit.max_concurrent


# ---- persistence check ----

PERSISTENCE_PROMPT = """You are checking whether blocking findings from a code review of a MERGED pull request still apply to the code as it is today.

Repository: {repo}
Pull request: #{number} "{title}" (merged {merged_at})
Reviewed commit (the PR head that was merged): {head_sha}
Current tip of `{tip_ref}`: {tip_sha} (your working directory is checked out here)

The review ran on the merged commit. Since then other PRs may have changed or fixed the same
code. For EACH finding below decide, from the code at the current tip:
- STILL_PRESENT: the defect still exists at the tip (possibly moved or renamed).
- FIXED: the defect no longer exists because the code was corrected; name the commit or PR that fixed it when you can find it (`git log`, `git log -S`, `git blame` work here; `git show {head_sha}:<path>` shows the reviewed version).
- OBSOLETE: the code the finding is about was removed or rewritten so the finding no longer applies, without an actual fix of the defect's concern being needed.
- UNKNOWN: you cannot decide with confidence.

Work read-only. Cite file paths and line numbers at the tip in your evidence.

Findings (JSON):
```json
{findings}
```

Reply with ONLY a raw JSON object, no fences, of this shape:
{{"findings": [{{"finding_hash": "<hash from the input>", "status": "STILL_PRESENT|FIXED|OBSOLETE|UNKNOWN", "evidence": "<2-5 sentences with tip file:line citations>", "fixed_by": "<commit sha or PR number, or null>"}}]}}
Every input finding_hash must appear exactly once.
"""


def persistence_prompt(
    *,
    repo: str,
    number: int,
    title: str,
    merged_at: str,
    head_sha: str,
    tip_ref: str,
    tip_sha: str,
    blockers: list[Finding],
) -> str:
    items = [
        {
            "finding_hash": f.hash,
            "file": f.file,
            "line_start": f.line_start,
            "line_end": f.line_end,
            "title": f.title,
            "body": f.body[:4000],
        }
        for f in blockers
    ]
    return PERSISTENCE_PROMPT.format(
        repo=repo,
        number=number,
        title=title,
        merged_at=merged_at,
        head_sha=head_sha,
        tip_ref=tip_ref,
        tip_sha=tip_sha,
        findings=json.dumps(items, indent=1),
    )


def parse_persistence(raw: dict[str, Any], hashes: list[str]) -> dict[str, dict[str, Any]]:
    """finding_hash -> {status, evidence, fixed_by}; anything missing or malformed is UNKNOWN,
    so a sloppy answer can only ever keep a blocker open, never clear it."""
    out = {h: {"status": "UNKNOWN", "evidence": "not assessed", "fixed_by": None} for h in hashes}
    for row in raw.get("findings") or []:
        if not isinstance(row, dict):
            continue
        h = str(row.get("finding_hash") or "")
        status = str(row.get("status") or "").upper()
        if h in out and status in PERSISTENCE_STATUSES:
            fixed = row.get("fixed_by")
            out[h] = {
                "status": status,
                "evidence": str(row.get("evidence") or "")[:3000],
                "fixed_by": str(fixed)[:80] if fixed not in (None, "", "null") else None,
            }
    return out


def parse_persistence_text(text: str, hashes: list[str]) -> dict[str, dict[str, Any]]:
    try:
        return parse_persistence(parse_json_object(text), hashes)
    except ReviewError:
        return parse_persistence({}, hashes)


# ---- reporting ----


@dataclass(slots=True)
class Outcome:
    """What one audit run concluded, as the report and the live post render it."""

    audit: dict[str, Any]
    verdict: str  # APPROVE | COMMENT | REQUEST_CHANGES
    findings: list[Finding]
    persistence: dict[str, dict[str, Any]]
    tip_ref: str
    tip_sha: str
    review_body: str
    degraded: bool
    run_id: int

    @property
    def blockers(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "blocking"]

    @property
    def open_blockers(self) -> list[Finding]:
        return [
            f
            for f in self.blockers
            if self.persistence.get(f.hash, {}).get("status", "UNKNOWN") in OPEN_STATUSES
        ]


def report_path(repo: str, number: int) -> str:
    return f"reports/{repo}/{number}.md"


def local_report(cfg: Config, repo: str, number: int) -> Path:
    return cfg.work_dir / "audit" / report_path(repo, number)


def _pr_url(repo: str, number: int) -> str:
    return f"https://github.com/{repo}/pull/{number}"


def _loc(f: Finding) -> str:
    return f"{f.file}:{f.line_start}" if f.line_start else (f.file or "(no file)")


def render_report(o: Outcome) -> str:
    a = o.audit
    head = [
        f"# {a['repo']}#{a['number']}: {a['title']}",
        "",
        f"- PR: {_pr_url(a['repo'], a['number'])}",
        f"- Author: `{a['author']}` · merged by `{a['merged_by']}` at {a['merged_at']}",
        f"- Merged head reviewed: `{a['sha']}` (base `{a['base_ref']}`)",
        f"- Bot review standing at merge: {a['coverage']}",
        f"- Audit: {a['mode']} · source {a['source']} · run {o.run_id}"
        + (" · ⚠️ DEGRADED (stand-in models)" if o.degraded else ""),
        f"- Verdict on the merged head: **{o.verdict}** · {len(o.blockers)} blocking · "
        f"{len(o.open_blockers)} still open on `{o.tip_ref}` @ `{o.tip_sha[:8]}`",
        "",
    ]
    parts = head
    if o.blockers:
        parts += ["## Blocking findings", ""]
        for f in o.blockers:
            p = o.persistence.get(f.hash, {"status": "UNKNOWN", "evidence": "", "fixed_by": None})
            fixed = f" · fixed by {p['fixed_by']}" if p.get("fixed_by") else ""
            parts += [
                f"### 🔴 {f.title}",
                f"`{_loc(f)}` · today: **{p['status']}**{fixed}",
                "",
                f.body or "",
                "",
                f"> Tip check: {p.get('evidence') or '—'}",
                "",
            ]
    others = [f for f in o.findings if f.severity != "blocking"]
    if others:
        parts += ["## Other findings", ""]
        for f in others:
            parts += [f"- **{f.severity}: {f.title}** (`{_loc(f)}`)", ""]
    parts += [
        "<details><summary>Full review body</summary>",
        "",
        o.review_body,
        "",
        "</details>",
        "",
    ]
    return "\n".join(parts)


def write_report(cfg: Config, gh: Gh, o: Outcome) -> None:
    """Write the per-PR report on disk. Committing it to the report repo is the index task's
    job (`sync_reports`): one writer, so concurrent audit workers never race the contents API
    on the same branch, and a failed commit is simply retried on the next pass."""
    local = local_report(cfg, o.audit["repo"], o.audit["number"])
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(render_report(o))


def report_repo_private(gh: Gh, repo: str) -> bool:
    """The findings corpus must never land in a public repository by a config typo."""
    try:
        d = gh.api(f"repos/{repo}")
    except ReviewError as exc:
        log.warning("audit report repo %s unreadable: %s", repo, exc)
        return False
    return isinstance(d, dict) and bool(d.get("private"))


def sync_reports(conn: sqlite3.Connection, cfg: Config, gh: Gh) -> int:
    """Commit every finished audit's report whose content the repo does not have yet.
    Tracks the committed digest per report in kv so unchanged reports cost no API call."""
    import hashlib

    pushed = 0
    rows = conn.execute(
        "SELECT repo, number, verdict FROM audits WHERE finished_at IS NOT NULL "
        "AND verdict IS NOT NULL AND verdict!='SKIPPED'"
    ).fetchall()
    for r in rows:
        local = local_report(cfg, r["repo"], r["number"])
        if not local.exists():
            continue
        text = local.read_text()
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]
        key = f"audit.synced:{r['repo']}#{r['number']}"
        if kv_get(conn, key) == digest:
            continue
        try:
            put_file(
                gh,
                cfg.audit.report_repo,
                report_path(r["repo"], r["number"]),
                text,
                branch=cfg.audit.report_branch,
                message=f"audit: {r['repo']}#{r['number']} ({r['verdict']})",
            )
        except ReviewError as exc:
            log.warning("audit report commit %s#%s failed: %s", r["repo"], r["number"], exc)
            continue
        with tx(conn):
            kv_set(conn, key, digest)
        pushed += 1
    return pushed


def put_file(gh: Gh, repo: str, path: str, text: str, *, branch: str, message: str) -> bool:
    """Create or update one file through the contents API. False when unchanged."""
    sha = None
    try:
        cur = gh.api(f"repos/{repo}/contents/{path}?ref={branch}")
        if isinstance(cur, dict):
            sha = cur.get("sha")
            existing = base64.b64decode(str(cur.get("content") or "")).decode("utf-8", "replace")
            if existing == text:
                return False
    except ReviewError as exc:
        if exc.kind is FailKind.INFRA:
            raise
    body: dict[str, Any] = {
        "message": message,
        "content": base64.b64encode(text.encode()).decode(),
        "branch": branch,
    }
    if sha:
        body["sha"] = sha
    gh.api(f"repos/{repo}/contents/{path}", method="PUT", body=body)
    return True


def render_index(conn: sqlite3.Connection) -> str:
    rows = conn.execute(
        "SELECT a.*, h.status AS head_status FROM audits a LEFT JOIN heads h ON h.id=a.head_id "
        "ORDER BY a.rank, a.merged_at"
    ).fetchall()
    done = [r for r in rows if r["finished_at"] and r["verdict"] and r["verdict"] != "SKIPPED"]
    blocked = [r for r in done if (r["blockers"] or 0) > 0]
    still = [r for r in done if (r["still_present"] or 0) > 0]
    by_merger: dict[str, list[int]] = {}
    for r in done:
        s = by_merger.setdefault(r["merged_by"], [0, 0, 0])
        s[0] += 1
        s[1] += int((r["blockers"] or 0) > 0)
        s[2] += int((r["still_present"] or 0) > 0)
    parts = [
        "# reviewsys audit: post-merge reviews",
        "",
        f"_Generated {now()}. Audited: {len(done)} of {len(rows)} queued merges · "
        f"{len(blocked)} would have been blocked · {len(still)} with blockers still present today._",
        "",
        "## By merger",
        "",
        "| Merged by | Audited | Would have been blocked | Blockers still present |",
        "|---|---|---|---|",
        *[
            f"| {m} | {s[0]} | {s[1]} | {s[2]} |"
            for m, s in sorted(by_merger.items(), key=lambda x: -x[1][0])
        ],
        "",
        "## Merges with blockers still present",
        "",
        "| PR | Title | Merged by | Blockers | Still present | Report |",
        "|---|---|---|---|---|---|",
    ]
    for r in still:
        parts.append(_index_row(r))
    parts += [
        "",
        "## All audits",
        "",
        "| PR | Title | Merged by | State | Verdict | Blockers | Still present | Report |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        state = "done" if r["finished_at"] else (r["head_status"] or "?")
        parts.append(
            f"| [{r['repo'].split('/')[1]}#{r['number']}]({_pr_url(r['repo'], r['number'])}) | "
            f"{_cell(r['title'])} | {r['merged_by']} | {state} | {r['verdict'] or ''} | "
            f"{'' if r['blockers'] is None else r['blockers']} | "
            f"{'' if r['still_present'] is None else r['still_present']} | "
            + (
                f"[report]({report_path(r['repo'], r['number'])})"
                if r["verdict"] and r["verdict"] != "SKIPPED"
                else ""
            )
            + " |"
        )
    return "\n".join(parts) + "\n"


def _index_row(r: sqlite3.Row) -> str:
    return (
        f"| [{r['repo'].split('/')[1]}#{r['number']}]({_pr_url(r['repo'], r['number'])}) | "
        f"{_cell(r['title'])} | {r['merged_by']} | {r['blockers']} | {r['still_present']} | "
        f"[report]({report_path(r['repo'], r['number'])}) |"
    )


def _cell(text: str) -> str:
    return str(text or "").replace("|", "\\|")[:90]


def publish_index(conn: sqlite3.Connection, cfg: Config, gh: Gh) -> dict[str, Any]:
    """Daemon task, the report repo's only writer: commit new or changed reports, then the
    index. Refuses to write anywhere that is not a private repository."""
    text = render_index(conn)
    local = cfg.work_dir / "audit" / "README.md"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(text)
    if not cfg.audit.report_repo:
        return {"repo": None}
    if not report_repo_private(gh, cfg.audit.report_repo):
        with tx(conn):
            event(conn, "audit.report_repo_refused", detail="not private or unreadable")
        return {"repo": cfg.audit.report_repo, "refused": "not private"}
    pushed = sync_reports(conn, cfg, gh)
    index = put_file(
        gh,
        cfg.audit.report_repo,
        "README.md",
        text,
        branch=cfg.audit.report_branch,
        message="audit: refresh index",
    )
    return {"reports": pushed, "index": index}


# ---- live post-merge posting ----

POST_MERGE_MARKER = "<!-- thepastaclaw-post-merge-review v1 -->"
# GitHub rejects comment and issue bodies over 65,536 characters; the full review is always
# in the private report, so the public copy is cut well below that
BODY_LIMIT = 60_000


def _cap(text: str, limit: int = BODY_LIMIT) -> str:
    if len(text) <= limit:
        return text
    note = "\n\n_…truncated; the full review is in the audit report._"
    return text[: limit - len(note)] + note


def render_live_comment(o: Outcome) -> str:
    a = o.audit
    lines = [
        POST_MERGE_MARKER,
        f"## Post-merge review (commit {a['sha'][:8]})",
        "",
        f"This PR was merged before a clean automated review of its final commit "
        f"(standing at merge: {a['coverage']}), so the review ran after the merge.",
        "",
        f"**Verdict on the merged commit: {o.verdict}** · {len(o.blockers)} blocking finding(s)",
    ]
    if o.degraded:
        lines += ["", "⚠️ Ran on stand-in models while the primary models were out of quota."]
    if o.blockers:
        lines += ["", f"Checked against `{o.tip_ref}` @ `{o.tip_sha[:8]}`:", ""]
        for f in o.blockers:
            p = o.persistence.get(f.hash, {})
            fixed = f" (fixed by {p['fixed_by']})" if p.get("fixed_by") else ""
            lines.append(f"- **{p.get('status', 'UNKNOWN')}**{fixed} — {f.title} (`{_loc(f)}`)")
    others = [f for f in o.findings if f.severity != "blocking"]
    if others:
        lines += ["", f"<details><summary>{len(others)} non-blocking finding(s)</summary>", ""]
        lines += [f"- {f.severity}: {f.title} (`{_loc(f)}`)" for f in others]
        lines += ["", "</details>"]
    head = "\n".join(lines)
    review = _cap(o.review_body, BODY_LIMIT - len(head) - 200)
    return head + "\n\n<details><summary>Full review</summary>\n\n" + review + "\n\n</details>"


def render_issue(o: Outcome, comment_url: str | None) -> tuple[str, str]:
    a = o.audit
    n = len(o.open_blockers)
    title = f"Post-merge review: {n} blocking finding(s) in #{a['number']} — {a['title']}"[:250]
    unknown = sum(
        1 for f in o.open_blockers if o.persistence.get(f.hash, {}).get("status") != "STILL_PRESENT"
    )
    lines = [
        POST_MERGE_MARKER,
        f"Automated post-merge review of #{a['number']} (merged by @{a['merged_by']} at "
        f"{a['merged_at']}, commit `{a['sha'][:8]}`) found blocking issues that were not shown "
        f"fixed on `{o.tip_ref}` @ `{o.tip_sha[:8]}`"
        + (
            f" ({unknown} could not be checked against the tip and need a look)."
            if unknown
            else "."
        ),
        "",
    ]
    if comment_url:
        lines += [f"Full review: {comment_url}", ""]
    for f in o.open_blockers:
        p = o.persistence.get(f.hash, {})
        lines += [
            f"### {f.title}",
            f"`{_loc(f)}` (at the merged commit) · today: **{p.get('status', 'UNKNOWN')}**",
            "",
            f.body or "",
            "",
            f"> {p.get('evidence') or ''}",
            "",
        ]
    return title, _cap("\n".join(lines))


def _existing_comment(gh: Gh, repo: str, number: int, bot_login: str) -> str | None:
    rows = gh.api(f"repos/{repo}/issues/{number}/comments?per_page=100", paginate=True) or []
    for c in rows:
        if (
            isinstance(c, dict)
            and str((c.get("user") or {}).get("login") or "").lower() == bot_login.lower()
            and POST_MERGE_MARKER in str(c.get("body") or "")
        ):
            return str(c.get("html_url") or "") or None
    return None


def _existing_issue(gh: Gh, repo: str, number: int, bot_login: str) -> str | None:
    rows = (
        gh.api(
            f"repos/{repo}/issues?creator={bot_login}&state=all&per_page=100",
            paginate=True,
        )
        or []
    )
    prefix = f"in #{number} — "
    for i in rows:
        if (
            isinstance(i, dict)
            and not i.get("pull_request")
            and POST_MERGE_MARKER in str(i.get("body") or "")
            and prefix in str(i.get("title") or "")
        ):
            return str(i.get("html_url") or "") or None
    return None


def post_live(
    gh: Gh,
    o: Outcome,
    *,
    bot_login: str,
    persist: Callable[[str, str], None],
) -> tuple[str | None, str | None]:
    """Comment on the merged PR, and open one issue when blockers are still open.

    Idempotent across retries and crashes: each URL is persisted (`persist(column, url)`, its
    own transaction) the moment it exists, and before posting GitHub is searched for the
    marker, so a worker that died between the POST and the write never posts twice."""
    a = o.audit
    comment_url = a.get("comment_url") or _existing_comment(gh, a["repo"], a["number"], bot_login)
    if not comment_url:
        d = gh.api(
            f"repos/{a['repo']}/issues/{a['number']}/comments",
            method="POST",
            body={"body": render_live_comment(o)},
        )
        comment_url = str((d or {}).get("html_url") or "") or None
    if comment_url:
        persist("comment_url", comment_url)
    issue_url = a.get("issue_url")
    if o.open_blockers and not issue_url:
        issue_url = _existing_issue(gh, a["repo"], a["number"], bot_login)
        if not issue_url:
            title, body = render_issue(o, comment_url)
            d = gh.api(
                f"repos/{a['repo']}/issues", method="POST", body={"title": title, "body": body}
            )
            issue_url = str((d or {}).get("html_url") or "") or None
        if issue_url:
            persist("issue_url", issue_url)
    return comment_url, issue_url


def persist_url(conn: sqlite3.Connection, audit_id: int, column: str, url: str) -> None:
    if column not in {"comment_url", "issue_url"}:
        raise ValueError(column)
    with tx(conn):
        conn.execute(f"UPDATE audits SET {column}=? WHERE id=?", (url, audit_id))


# ---- bookkeeping ----


def load(conn: sqlite3.Connection, head_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM audits WHERE head_id=?", (head_id,)).fetchone()
    return dict(row) if row else None


def record(
    conn: sqlite3.Connection, o: Outcome, *, comment_url: str | None, issue_url: str | None
) -> None:
    """Persist the outcome. Caller holds the write transaction."""
    aid = o.audit["id"]
    conn.execute("DELETE FROM audit_findings WHERE audit_id=?", (aid,))
    conn.executemany(
        "INSERT OR REPLACE INTO audit_findings (audit_id, run_id, hash, severity, file, line_start, "
        "line_end, category, title, body, status, evidence, fixed_by) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                aid,
                o.run_id,
                f.hash,
                f.severity,
                f.file,
                f.line_start,
                f.line_end,
                f.category,
                f.title,
                f.body[:8000],
                (o.persistence.get(f.hash) or {}).get("status"),
                (o.persistence.get(f.hash) or {}).get("evidence"),
                (o.persistence.get(f.hash) or {}).get("fixed_by"),
            )
            for f in o.findings
        ],
    )
    conn.execute(
        "UPDATE audits SET finished_at=?, run_id=?, verdict=?, blockers=?, still_present=?, "
        "findings=?, tip_ref=?, tip_sha=?, comment_url=?, issue_url=?, degraded=? WHERE id=?",
        (
            now(),
            o.run_id,
            o.verdict,
            len(o.blockers),
            len(o.open_blockers),
            len(o.findings),
            o.tip_ref,
            o.tip_sha,
            comment_url,
            issue_url,
            int(o.degraded),
            aid,
        ),
    )
    event(
        conn,
        "audit.done",
        repo=o.audit["repo"],
        number=o.audit["number"],
        run_id=o.run_id,
        detail=f"{o.verdict} blockers={len(o.blockers)} open={len(o.open_blockers)}",
    )


def escalate(conn: sqlite3.Connection, audit_id: int, head_id: int) -> None:
    """A light audit found a blocker: run the full pipeline on the same head."""
    with tx(conn):
        conn.execute(
            "UPDATE audits SET mode='full', escalated=1, finished_at=NULL WHERE id=?", (audit_id,)
        )
        conn.execute(
            "UPDATE heads SET status='queued', attempts=0, eligible_at=?, finished_at=NULL, "
            "reason='escalated to a full audit' WHERE id=?",
            (now(), head_id),
        )
        row = conn.execute("SELECT repo, number FROM audits WHERE id=?", (audit_id,)).fetchone()
        event(
            conn,
            "audit.escalated",
            repo=row["repo"] if row else None,
            number=row["number"] if row else None,
            detail=f"light audit found a blocker; full audit queued (head {head_id})",
        )


def summary(conn: sqlite3.Connection, cfg: Config) -> dict[str, Any]:
    """Counts for `status`, the export and the CLI; no PR names (the export is public)."""
    heads = {
        r["status"]: int(r["n"])
        for r in conn.execute(
            "SELECT status, COUNT(*) AS n FROM heads WHERE queue='audit' GROUP BY status"
        )
    }
    done = conn.execute(
        "SELECT COUNT(*) AS n, SUM(blockers>0) AS blocked, SUM(still_present>0) AS still "
        "FROM audits WHERE finished_at IS NOT NULL AND verdict IS NOT NULL AND verdict!='SKIPPED'"
    ).fetchone()
    skipped = conn.execute("SELECT COUNT(*) FROM audits WHERE verdict='SKIPPED'").fetchone()[0]
    return {
        "enabled": cfg.audit.enabled,
        "max_concurrent": concurrency(conn, cfg),
        "heads": heads,
        "audited": int(done["n"] or 0),
        "would_block": int(done["blocked"] or 0),
        "still_present": int(done["still"] or 0),
        "skipped_sync": int(skipped),
    }
