"""Shared setup for the policy v12 tests: a v12 skills config, fake git answers, and a
simulated PR whose review threads follow what the bot posts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reviewsys import config as cfg_mod
from reviewsys import ledger, worker
from reviewsys.contract import Finding
from reviewsys.db import tx
from reviewsys.ingest import enqueue_head
from reviewsys.models import RunStatus, Trigger
from reviewsys.scheduler import schedule
from tests.conftest import group

REPO = "dashpay/platform"
A, B = "a" * 40, "c" * 40
V12 = {
    "version": "v12",
    "lane_env_isolation": True,
    "structured_output": False,
    "finders": {"anchoring": "unanchored"},
    "threads": {"lane": True},
    "verifier": {"contract": "v12"},
    "confirmation": {"roles": ["general", "security-auditor"], "trigger": "no_blockers"},
    "late_nonblocking": "post",
}


def v12_cfg(cfg, skills_dir: Path, *, tiers: dict[str, Any] | None = None, **pipeline: Any):
    """The fixture config under policy v12 (`pipeline` overrides merged in), fingerprinted."""
    raw = json.loads((skills_dir / "config.json").read_text())
    pol = raw["review_model_policy"]
    block = {**V12, **pipeline}
    for k, v in pipeline.items():
        if isinstance(v, dict) and isinstance(V12.get(k), dict):
            block[k] = {**V12[k], **v}
    pol["pipeline"] = block
    if tiers:
        for name, extra in tiers.items():
            pol["triage"]["tiers"][name].update(extra)
    pol["fingerprint"] = cfg_mod.policy_fingerprint(pol)
    (skills_dir / "config.json").write_text(json.dumps(raw))
    return cfg_mod.load(cfg.skills_dir.parent / "config.toml")


class Git:
    """`worker._lines_changed` answered from a linear history: `change(file, line, at=sha)`
    records that commit `sha` changed that line (None: the file somewhere). A question about
    (a, b) sees the changes of the commits after the older and up to the newer one, on the
    anchor's lines ± 3 (any line for a None anchor). A sha outside `history` is one git does
    not have (None); so is every question while `unknown` is set."""

    HISTORY = ("0" * 40, "a" * 40, "c" * 40, "d" * 40, "e" * 40, "f" * 40)

    def __init__(self, history: tuple[str, ...] = HISTORY) -> None:
        self.history = list(history)
        self.changes: set[tuple[str, str, int | None]] = set()
        self.unknown = False
        self.asked: list[tuple[str, str, str, int | None]] = []

    def change(self, file: str, line: int | None, *, at: str) -> None:
        assert at in self.history, at
        self.changes.add((at, file, line))

    def map(self, ctx, a: str, b: str, file: str, ls: int | None, le: int | None):
        """`worker._map_lines`: lines never move in this history (None for an unknown sha)."""
        if self.unknown or a not in self.history or b not in self.history:
            return None
        return ls, le

    def __call__(self, ctx, a: str, b: str, file: str, ls: int | None, le: int | None):
        self.asked.append((a, b, file, ls))
        if self.unknown or a not in self.history or b not in self.history:
            return None
        lo, hi = sorted((self.history.index(a), self.history.index(b)))
        between = set(self.history[lo + 1 : hi + 1])
        for at, f, line in self.changes:
            if at not in between or f != file:
                continue
            if ls is None or line is None or ls - 3 <= line <= (le or ls) + 3:
                return True
        return False


def cand(title: str, *, severity: str = "blocking", file: str = "f.rs", line: int = 11, **kw):
    return {
        "file": file,
        "line_start": line,
        "line_end": line + 1,
        "severity": severity,
        "category": "logic",
        "title": title,
        "failure_scenario": f"when {title.lower()}",
        "body": f"{title} because of line {line}.",
        **kw,
    }


def try_round(
    cfg, conn, gh, lanes, sha: str, trigger=Trigger.MENTION, **main: Any
) -> tuple[int, RunStatus]:
    """Queue `sha` (the PR's live head now) and run its review: (run id, status). `main`:
    more `worker.main` options (`dry_run`, `replay`)."""
    lanes.head = sha
    gh.pr["head"]["sha"] = sha
    with tx(conn):
        enqueue_head(conn, cfg, REPO, 1, sha, trigger)
    (rid,) = schedule(conn, cfg, spawn=False)
    return rid, worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False, **main)


def run_round(cfg, conn, gh, lanes, sha: str, trigger=Trigger.MENTION, **main: Any) -> int:
    """`try_round`, which must end DONE: the run id."""
    rid, status = try_round(cfg, conn, gh, lanes, sha, trigger, **main)
    assert status == RunStatus.DONE, conn.execute(
        "SELECT reason FROM runs WHERE id=?", (rid,)
    ).fetchone()["reason"]
    return rid


def gql(c: dict[str, Any]) -> dict[str, Any]:
    return {
        "databaseId": c["id"],
        "author": {"login": c["author"]},
        "body": c["body"],
        "createdAt": c.get("created_at"),
        "authorAssociation": c.get("association"),
    }


def publish_threads(gh, *, start: int = 800) -> dict[str, int]:
    """Turn every inline comment the bot posted (in `gh.posted_reviews`) into a GitHub review
    thread and inline comment, the way GitHub would: finding hash -> root comment id. Threads
    already there keep their replies."""
    have = {
        t["comments"]["nodes"][0]["databaseId"]: t for t in gh.threads if t["comments"]["nodes"]
    }
    ids: dict[str, int] = {}
    n = start + len(have)
    for review in gh.posted_reviews:
        for c in review.get("comments") or []:
            h = c["body"].split("finding=", 1)[1].split()[0]
            if h in ids or any(
                f"finding={h}" in t["comments"]["nodes"][0]["body"] for t in have.values()
            ):
                continue
            ids[h] = n
            gh.inline.append(
                {
                    "id": n,
                    "node_id": f"N{n}",
                    "user": {"login": "thepastaclaw"},
                    "body": c["body"],
                    "path": c["path"],
                    "line": c["line"],
                    "html_url": f"https://github.com/{REPO}/pull/1#discussion_r{n}",
                }
            )
            gh.threads.append(
                {
                    "id": f"PRRT_{n}",
                    "isResolved": False,
                    "isOutdated": False,
                    "path": c["path"],
                    "line": c["line"],
                    "comments": {
                        "nodes": [
                            gql(
                                {
                                    "id": n,
                                    "author": "thepastaclaw",
                                    "body": c["body"],
                                    "created_at": "2026-10-01T00:00:00Z",
                                    "association": "NONE",
                                }
                            )
                        ]
                    },
                }
            )
            n += 1
    for t in gh.threads:
        root = t["comments"]["nodes"][0]["body"]
        if "finding=" in root:
            ids.setdefault(
                root.split("finding=", 1)[1].split()[0], t["comments"]["nodes"][0]["databaseId"]
            )
    return ids


def reply(
    gh,
    root_id: int,
    body: str,
    *,
    rid: int,
    who: str = "knst",
    assoc: str = "COLLABORATOR",
    at: str = "2026-10-02T00:00:00Z",
) -> None:
    t = next(t for t in gh.threads if t["comments"]["nodes"][0]["databaseId"] == root_id)
    t["comments"]["nodes"].append(
        gql({"id": rid, "author": who, "body": body, "created_at": at, "association": assoc})
    )


def resolve(gh, root_id: int) -> None:
    next(t for t in gh.threads if t["comments"]["nodes"][0]["databaseId"] == root_id)[
        "isResolved"
    ] = True


def resolve_with_answers(gh, root_id: int, *, at: str = "2026-10-03T00:00:00Z") -> None:
    """What GitHub shows once the bot answered and resolved a finding thread: every answer it
    posted on that finding (`gh.replies`) under the root, and the thread resolved."""
    t = next(t for t in gh.threads if t["comments"]["nodes"][0]["databaseId"] == root_id)
    h = t["comments"]["nodes"][0]["body"].split("finding=", 1)[1].split()[0]
    for n, r in enumerate(gh.replies):
        if f"finding={h}" in r["body"]:
            t["comments"]["nodes"].append(
                gql({"id": 5000 + n, "author": "thepastaclaw", "body": r["body"], "created_at": at})
            )
    t["isResolved"] = True


def issues(conn) -> dict[str, ledger.Issue]:
    return ledger.load(conn, REPO, 1)


def by_title(conn) -> dict[str, ledger.Issue]:
    return {i.title: i for i in issues(conn).values()}


# ---- fake verify+match answers (`FakeLanes.verify12`) ----


def refuted(cands: list[dict[str, Any]]) -> dict[str, Any]:
    return {"summary": "-", "groups": [group(c, verdict="REFUTED") for c in cands]}


def refute_gate(label, cands, prompt):
    """The gate refutes Phase 1's claims; every other verifier confirms each as new."""
    if label == "verify1":
        return refuted(cands)
    return {"summary": "ok", "groups": [group(c) for c in cands]}


def confirm_matches(label, cands, prompt):
    """`refute_gate`, except that a candidate whose title says which ledger issue it restates
    (`... [open:<hash>]`) is matched to it, under its plain title."""
    if label == "verify1":
        return refuted(cands)
    groups = []
    for c in cands:
        match = c["title"].rsplit("[", 1)[1].rstrip("]") if "[" in c["title"] else "new"
        groups.append(group(c, match=match, title=c["title"].split(" [")[0]))
    return {"summary": "Verified.", "groups": groups}


def shard_setup(lanes, *, blocking_in_failing: bool) -> None:
    """13 Phase-2 candidates in two files (two verifier shards): the a.rs shard never answers
    validly; `blocking_in_failing` puts a blocker in it."""
    a = [cand(f"a{i}", severity="suggestion", file="a.rs", line=i) for i in range(1, 8)]
    if blocking_in_failing:
        a[0]["severity"] = "blocking"
    b = [cand(f"b{i}", severity="nitpick", file="b.rs", line=i) for i in range(1, 7)]
    lanes.candidates["default"] = []
    lanes.candidates["general"] = a + b

    def verify(label, cands, prompt):
        if any(c["file"] == "a.rs" for c in cands) and label != "verify1":
            return {"summary": "x", "groups": "not a list"}  # never valid, corrections or not
        return refute_gate(label, cands, prompt)

    lanes.verify12 = verify


def backfilled(conn, title: str, *, line: int, sha: str = "0" * 40) -> str:
    """A blocker the ledger backfill recorded on `sha`, with no thread."""
    f = Finding(
        file="f.rs", title=title, body="old", severity="blocking", line_start=line, line_end=line
    )
    with tx(conn):
        ledger.append(conn, REPO, 1, [ledger._posted(f, run_id=None, sha=sha, source="backfill")])
    return f.hash
