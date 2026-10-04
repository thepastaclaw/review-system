"""Shared setup for the policy v12 tests: a v12 skills config, fake git answers, and a
simulated PR whose review threads follow what the bot posts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reviewsys import config as cfg_mod
from reviewsys import ledger, worker
from reviewsys.db import tx
from reviewsys.ingest import enqueue_head
from reviewsys.models import RunStatus, Trigger
from reviewsys.scheduler import schedule

REPO = "dashpay/platform"
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
    """`worker._lines_changed` answered from a table: (file, line) -> changed; anything else
    unchanged. `unknown` makes every answer None (git cannot tell)."""

    def __init__(self) -> None:
        self.changed: set[tuple[str, int | None]] = set()
        self.files: set[str] = set()  # files with any change (a None anchor)
        self.unknown = False
        self.asked: list[tuple[str, str, str, int | None]] = []

    def __call__(self, ctx, a: str, b: str, file: str, ls: int | None, le: int | None):
        self.asked.append((a, b, file, ls))
        if self.unknown:
            return None
        if ls is None:
            return file in self.files or any(f == file for f, _ in self.changed)
        return (file, ls) in self.changed


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


def run_round(cfg, conn, gh, lanes, sha: str, trigger=Trigger.MENTION) -> int:
    lanes.head = sha
    gh.pr["head"]["sha"] = sha
    with tx(conn):
        enqueue_head(conn, cfg, REPO, 1, sha, trigger)
    (rid,) = schedule(conn, cfg, spawn=False)
    status = worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False)
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


def issues(conn) -> dict[str, ledger.Issue]:
    return ledger.load(conn, REPO, 1)


def by_title(conn) -> dict[str, ledger.Issue]:
    return {i.title: i for i in issues(conn).values()}
