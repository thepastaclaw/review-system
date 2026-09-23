"""`reviewsys replay`: re-review a PR at the exact head a past run reviewed, under whichever
policy the skills directory given holds, without touching GitHub or the production database.

Built for the v10 evaluation (plan section 6): run the same heads through the new pipeline and
compare its findings with what the historical run posted and what later happened to them.

Safety: the run uses a scratch SQLite database under `--out`, a GitHub client whose runner
refuses every write (non-GET REST, GraphQL mutations), and the worker's dry-run mode. It shares
the production work directory's bare mirrors (read + fetch only) but gets its own worktrees
and run directories under `--out`.

The replay sees the PR as it was when the source run started: its history is seeded from
the runs before it, and every GitHub comment, review and thread comment created after that
moment is filtered out. One thing cannot be rewound: whether a thread was resolved, which is
read as of now; the report says so.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from . import config as cfg_mod
from . import worker
from .db import connect, now, tx
from .gh import Gh, read_only_runner
from .models import RunStatus, Trigger


def _source_run(prod: sqlite3.Connection, run_id: int) -> sqlite3.Row:
    row = prod.execute(
        "SELECT r.id, r.started_at, h.repo, h.number, h.sha, h.trigger, r.tier FROM runs r JOIN heads h ON h.id=r.head_id WHERE r.id=?",
        (run_id,),
    ).fetchone()
    if not isinstance(row, sqlite3.Row):
        raise SystemExit(f"run {run_id} not found in the production database")
    return row


def _seed(scratch: sqlite3.Connection, prod: sqlite3.Connection, src: sqlite3.Row) -> None:
    """Copy this PR's history from before the source run (findings posted and conceded by
    earlier runs, the heads they were posted at) into the scratch database."""
    repo, number = src["repo"], src["number"]
    earlier = prod.execute(
        "SELECT f.stage, f.phase, f.hash, f.file, f.line_start, f.line_end, f.severity, f.confidence, "
        "f.category, f.title, f.body, h.sha, r.started_at FROM findings f JOIN runs r ON r.id=f.run_id "
        "JOIN heads h ON h.id=r.head_id WHERE h.repo=? AND h.number=? AND f.run_id < ? "
        "AND f.stage IN ('posted','conceded') ORDER BY f.id",
        (repo, number, src["id"]),
    ).fetchall()
    with tx(scratch):
        scratch.execute(
            "INSERT OR REPLACE INTO prs (repo, number, head_sha, state, updated_at) VALUES (?,?,?,?,?)",
            (repo, number, src["sha"], "open", now()),
        )
        if not earlier:
            return
        # one placeholder run holds the history rows (the worker only reads them per PR)
        scratch.execute(
            "INSERT INTO heads (repo, number, sha, trigger, status, queued_at, eligible_at) VALUES (?,?,?,?,?,?,?)",
            (repo, number, "0" * 40, "manual", "done", now(), now()),
        )
        hid = scratch.execute("SELECT last_insert_rowid()").fetchone()[0]
        scratch.execute(
            "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at) VALUES (?,?,?,?,?,?)",
            (hid, 1, "done", "seed", now(), now()),
        )
        rid = scratch.execute("SELECT last_insert_rowid()").fetchone()[0]
        for f in earlier:
            scratch.execute(
                "INSERT INTO findings (run_id, phase, stage, hash, file, line_start, line_end, severity, confidence, category, title, body) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    rid,
                    f["phase"],
                    f["stage"],
                    f["hash"],
                    f["file"],
                    f["line_start"],
                    f["line_end"],
                    f["severity"],
                    f["confidence"],
                    f["category"],
                    f["title"],
                    f["body"],
                ),
            )
            if f["stage"] == "posted":
                scratch.execute(
                    "INSERT INTO posted_findings (repo, number, hash, sha, posted_at) VALUES (?,?,?,?,?) "
                    "ON CONFLICT(repo, number, hash) DO UPDATE SET sha=excluded.sha, posted_at=excluded.posted_at",
                    (repo, number, f["hash"], f["sha"], f["started_at"]),
                )


def _historical(prod: sqlite3.Connection, run_id: int) -> list[dict[str, Any]]:
    return [
        dict(r)
        for r in prod.execute(
            "SELECT hash, severity, file, line_start, title FROM findings WHERE run_id=? AND stage='posted'",
            (run_id,),
        )
    ]


def replay(
    cfg: cfg_mod.Config, *, run_id: int, out: Path, skills: Path | None = None
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    rcfg = cfg_mod.load_skills_config(skills or cfg.skills_dir)
    repos, specialists, policy, _ = rcfg
    work = out / f"run-{run_id}"
    if work.exists():
        shutil.rmtree(work)
    (work / "work").mkdir(parents=True)
    # share the bare mirrors (fetch-only), isolate everything else
    (work / "work" / "mirrors").symlink_to(cfg.mirrors_dir)
    rc = dataclasses.replace(
        cfg,
        db_path=work / "replay.db",
        work_dir=work / "work",
        skills_dir=skills or cfg.skills_dir,
        repos=repos,
        specialists=specialists,
        policy=policy,
        backlog_skip_phase1_above=0,
    )
    prod = sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True)
    prod.row_factory = sqlite3.Row
    src = _source_run(prod, run_id)
    scratch = connect(rc.db_path)
    _seed(scratch, prod, src)
    with tx(scratch):
        scratch.execute(
            "INSERT INTO heads (repo, number, sha, trigger, status, queued_at, eligible_at) VALUES (?,?,?,?,?,?,?)",
            (src["repo"], src["number"], src["sha"], Trigger.MANUAL.value, "running", now(), now()),
        )
        hid = scratch.execute("SELECT last_insert_rowid()").fetchone()[0]
        scratch.execute(
            "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at) VALUES (?,?,?,?,?,?)",
            (hid, 1, "spawned", "replay", now(), now()),
        )
        new_run = scratch.execute("SELECT last_insert_rowid()").fetchone()[0]
    gh = _PinnedHead(cfg.gh_bin, src["sha"], str(src["started_at"]))
    status = worker.main(rc, scratch, new_run, gh=gh, dry_run=True, heartbeat=False)
    found = [
        dict(r)
        for r in scratch.execute(
            "SELECT stage, phase, hash, severity, file, line_start, title FROM findings WHERE run_id=? AND stage IN ('verified','published','posted')",
            (new_run,),
        )
    ]
    lanes = [
        dict(r)
        for r in scratch.execute(
            "SELECT phase, role, model, effort, status, tokens_in, tokens_out FROM lanes WHERE run_id=?",
            (new_run,),
        )
    ]
    report = {
        "source_run": run_id,
        "repo": src["repo"],
        "number": src["number"],
        "sha": src["sha"],
        "policy": policy.name,
        "status": status.value,
        "reason": scratch.execute("SELECT reason FROM runs WHERE id=?", (new_run,)).fetchone()[0],
        "historical_posted": _historical(prod, run_id),
        "replay_findings": found,
        "lanes": lanes,
        "tokens_in": sum(int(lane["tokens_in"] or 0) for lane in lanes),
        "review_body": next(
            (p.read_text() for p in (rc.runs_dir / f"run-{new_run}").glob("review-*.md")), ""
        ),
        "caveat": "thread resolution state is as of now, not as of the replayed head",
    }
    (work / "report.json").write_text(json.dumps(report, indent=1))
    scratch.close()
    prod.close()
    # the worker keeps a failed run's worktree for debugging; a replay's is disposable (a
    # Rust repo's can reach tens of GB) and the lane artifacts are what the report needs
    shutil.rmtree(rc.worktrees_dir, ignore_errors=True)
    if status != RunStatus.DONE:
        report["hint"] = f"see {work}/work/runs for lane artifacts"
    return report


class _PinnedHead(Gh):
    """Read-only gh that shows the PR as it was when the source run started: open at the
    replayed sha, and without any comment, review or thread comment created later."""

    def __init__(self, bin_path: str, sha: str, cutoff: str) -> None:
        super().__init__(bin_path, runner=read_only_runner())
        self.sha = sha
        self.cutoff = cutoff  # ISO-8601 Z, compares as a string with GitHub's timestamps

    def api(self, endpoint: str, **kw: Any) -> Any:
        data = super().api(endpoint, **kw)
        parts = endpoint.split("?")[0].strip("/").split("/")
        if parts[:1] != ["repos"]:
            return data
        if len(parts) == 5 and parts[3] == "pulls" and isinstance(data, dict):
            # repos/{owner}/{repo}/pulls/{number}
            data = {
                **data,
                "state": "open",
                "merged": False,
                "head": {**(data.get("head") or {}), "sha": self.sha},
            }
        elif len(parts) == 6 and parts[5] in ("comments", "reviews") and isinstance(data, list):
            # repos/{owner}/{repo}/{pulls|issues}/{number}/{comments|reviews}
            key = "submitted_at" if parts[5] == "reviews" else "created_at"
            data = [r for r in data if isinstance(r, dict) and str(r.get(key) or "") < self.cutoff]
        return data

    def graphql(
        self, query: str, variables: dict[str, Any] | None = None, *, timeout: int | None = None
    ) -> Any:
        data = super().graphql(query, variables, timeout=timeout)
        if "reviewThreads(" not in query or not isinstance(data, dict):
            return data
        pr = (data.get("repository") or {}).get("pullRequest") or {}
        threads = pr.get("reviewThreads") or {}
        kept = []
        for n in threads.get("nodes") or []:
            cs = [
                c
                for c in (n.get("comments") or {}).get("nodes") or []
                if str(c.get("createdAt") or "") < self.cutoff
            ]
            if cs:
                kept.append({**n, "comments": {"nodes": cs}})
        threads["nodes"] = kept
        return data
