"""`reviewsys replay`: re-review a PR at the exact head a past run reviewed, under whichever
policy the skills directory given holds, without touching GitHub or the production database.

Built for the v10 evaluation (plan section 6): run the same heads through the new pipeline and
compare its findings with what the historical run posted and what later happened to them.

Safety: the run uses a scratch SQLite database under `--out`, a GitHub client whose runner
refuses every write (non-GET REST, GraphQL mutations), and the worker's dry-run mode. It shares
the production work directory's bare mirrors (read + fetch only) but gets its own worktrees
and run directories under `--out`.

The ledger a v10 run matches against is rebuilt from the production database as it stood
before the replayed head: every issue posted on the PR at an earlier head, with its thread
state as of now. That is an approximation (a thread may have been resolved after the replayed
head); the report says so.
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
        "SELECT r.id, h.repo, h.number, h.sha, h.trigger, r.tier FROM runs r JOIN heads h ON h.id=r.head_id WHERE r.id=?",
        (run_id,),
    ).fetchone()
    if not isinstance(row, sqlite3.Row):
        raise SystemExit(f"run {run_id} not found in the production database")
    return row


def _seed(scratch: sqlite3.Connection, prod: sqlite3.Connection, src: sqlite3.Row) -> None:
    """Copy what the worker reads about this PR's history (posted findings from earlier heads,
    reviews, the `prs` row) into the scratch database."""
    repo, number, sha = src["repo"], src["number"], src["sha"]
    with tx(scratch):
        scratch.execute(
            "INSERT OR REPLACE INTO prs (repo, number, head_sha, state, updated_at) VALUES (?,?,?,?,?)",
            (repo, number, sha, "open", now()),
        )
        earlier = prod.execute(
            "SELECT pf.hash, pf.sha, pf.review_id, pf.posted_at FROM posted_findings pf "
            "WHERE pf.repo=? AND pf.number=? AND pf.sha != ?",
            (repo, number, sha),
        ).fetchall()
        for r in earlier:
            scratch.execute(
                "INSERT OR IGNORE INTO posted_findings (repo, number, hash, sha, review_id, posted_at) VALUES (?,?,?,?,?,?)",
                (repo, number, r["hash"], r["sha"], r["review_id"], r["posted_at"]),
            )
        # the findings rows those hashes point at (stage 'posted'), under a placeholder run
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
        for h in {str(r["hash"]) for r in earlier}:
            f = prod.execute(
                "SELECT phase, file, line_start, line_end, severity, confidence, category, title, body "
                "FROM findings WHERE hash=? AND stage IN ('posted','conceded') ORDER BY id DESC LIMIT 1",
                (h,),
            ).fetchone()
            if f is None:
                continue
            scratch.execute(
                "INSERT INTO findings (run_id, phase, stage, hash, file, line_start, line_end, severity, confidence, category, title, body) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    rid,
                    f["phase"],
                    "posted",
                    h,
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
    gh = _PinnedHead(cfg.gh_bin, src["sha"])
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
        "caveat": "ledger thread state is as of now, not as of the replayed head",
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
    """Read-only gh that reports the PR as open at the replayed sha: a replay reviews that
    exact commit even if the PR has moved on or merged since."""

    def __init__(self, bin_path: str, sha: str) -> None:
        super().__init__(bin_path, runner=read_only_runner())
        self.sha = sha

    def api(self, endpoint: str, **kw: Any) -> Any:
        data = super().api(endpoint, **kw)
        parts = endpoint.split("?")[0].strip("/").split("/")
        if (
            len(parts) == 4
            and parts[0] == "repos"
            and parts[2] == "pulls"
            and isinstance(data, dict)
        ):
            data = {
                **data,
                "state": "open",
                "merged": False,
                "head": {**(data.get("head") or {}), "sha": self.sha},
            }
        return data
