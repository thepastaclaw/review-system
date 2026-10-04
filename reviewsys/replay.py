"""`reviewsys replay`: re-review a PR at the exact head a past run reviewed, under whichever
policy a skills checkout holds, without touching GitHub or the production database.

Built for the v12 evaluation (DESIGN.md section 5): run the same heads under two policies and
compare what each finds with labels and with each other.

The replay sees the PR as it was when the source run started (the cutoff):

- History: the scratch database is seeded with this PR's runs that finished before the cutoff
  (their posted, conceded and verified findings, steps and reviews), the `posted_findings`
  rows and `thread.answered` events from before it, and (hook, see `seed_ledger`) the issue
  ledger as of it.
- GitHub: the harness and every lane read the PR through the pinned view in `ghshim` (the
  replayed head and base, CI from the source run's evidence, nothing created after the
  cutoff); every write is refused. Lanes get it through a `gh` shim first on their PATH, and
  the lane sandbox denies executing the installed gh, so an absolute-path call cannot reach
  the live PR either.
- Range: the review covers `merge-base(base the source run saw, head)..head`, not the merge
  base with today's base branch (for a PR merged since, that would be the head itself).

Eval switches: `force_phase2` runs Phase 2 whatever the gate decides (recording what the gate
would have published), `pin_phase1_model` puts one model on every Phase-1 lane in both arms,
and a replay refuses to start while degraded mode is active (its stand-in models would not be
the policy under test); mid-run it never switches to stand-ins, a quota failure fails it. The
harness adds no turn or candidate caps.

Safety: a scratch SQLite database under `--out`, the worker's dry-run mode, and a read-only gh.
The production work directory's bare mirrors are shared (read + fetch only); worktrees and
run directories live under `--out` and the worktrees are removed when the replay ends.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any

from . import config as cfg_mod
from . import degraded, worker
from .db import connect, now, tx
from .degraded import Prober
from .gh import Gh, GhRunner, read_only_runner
from .ghshim import SHIM_CHECK, Pin, install_shim, pinned_runner, stage_real_gh
from .lane import (
    SANDBOX_EXEC,
    LaneRunner,
    exec_deny_profile,
    run_claude_lane,
    sandbox_problem,
    with_env,
)
from .lanepool import pool_of
from .models import RunStatus, StepName, Trigger
from .steps import worktree as wt

EXIT_ABORTED = 2  # `reviewsys replay` exit code when it refused to start (degraded mode)
# findings stages a later run reads from earlier ones (prior findings, standing verdicts)
SEED_STAGES = ("posted", "conceded", "verified", "verified-fresh")
CAVEATS = (
    "thread resolution state, reactions and labels are as of now, not as of the cutoff",
    "converse silence memory (kv) is not seeded: a reply the bot chose not to answer may be "
    "considered again",
)


class ReplayAborted(Exception):
    """The replay must not run: its results would not measure the policy under test."""


# ---- hooks ----


def seed_ledger(
    scratch_db: sqlite3.Connection,
    source_db: sqlite3.Connection,
    repo: str,
    number: int,
    cutoff: str,
) -> None:
    """HOOK (ledger seeding, wired when the ledger lands): copy this PR's issue ledger
    (`issues` and `issue_events`, schema v6) from the source database into the scratch one as
    of `cutoff` (ISO-8601 Z, the source run's start): the issues opened before it, each with
    the status of its last event before it, and those events. A no-op until then."""
    return None


def policy_overrides(rc: cfg_mod.Config, policy_name: str | None) -> cfg_mod.Config:
    """HOOK (policy dispatch): the replay's config for one eval arm, after the arm's skills
    checkout is loaded. Policy behaviour is switched by that checkout's `config.json` (v12:
    `review_model_policy.pipeline` and its sub-fields), which `--skills` already selects, so
    this is the identity; wire arm-only switches that live outside the skills config here,
    keyed by the arm's name (`replay-set --policy NAME=DIR`, `replay --policy-name`)."""
    return rc


# ---- source ----


def open_source(path: Path) -> sqlite3.Connection:
    """The production (or copied) database, read-only."""
    if not path.exists():
        raise SystemExit(f"source database {path} not found")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _source_run(prod: sqlite3.Connection, run_id: int) -> sqlite3.Row:
    row = prod.execute(
        "SELECT r.id, r.started_at, r.status, r.tier, r.degraded, h.repo, h.number, h.sha, "
        "h.trigger, h.queue FROM runs r JOIN heads h ON h.id=r.head_id WHERE r.id=?",
        (run_id,),
    ).fetchone()
    if not isinstance(row, sqlite3.Row):
        raise SystemExit(f"run {run_id} not found in the source database")
    if row["queue"] == "audit":
        raise SystemExit(f"run {run_id} is a post-merge audit; replay covers live reviews only")
    return row


def _source_trigger(prod: sqlite3.Connection, run_id: int) -> Trigger:
    """A conversation run (it answered replies on a reviewed commit) is replayed as one; any
    other run as a manual review. Whether a reply-triggered run conversed depends on whether a
    final review of that commit already stood, which the cutoff reproduces."""
    conversed = prod.execute(
        "SELECT 1 FROM steps WHERE run_id=? AND name=?", (run_id, StepName.CONVERSE.value)
    ).fetchone()
    return Trigger.REVIEW_REPLY if conversed else Trigger.MANUAL


def source_evidence(runs_dir: Path, run_id: int) -> dict[str, Any]:
    """The source run's `evidence.json` (its PR metadata and CI snapshot), {} when gone."""
    try:
        data = json.loads((runs_dir / f"run-{run_id}" / "evidence.json").read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _historical(prod: sqlite3.Connection, run_id: int) -> list[dict[str, Any]]:
    return [
        dict(r)
        for r in prod.execute(
            "SELECT phase, hash, severity, file, line_start, line_end, category, title "
            "FROM findings WHERE run_id=? AND stage='posted' ORDER BY id",
            (run_id,),
        )
    ]


# ---- seeding ----


def _col(row: sqlite3.Row, key: str) -> Any:
    return row[key] if key in row.keys() else None  # noqa: SIM118 (Row `in` tests values)


def seed_history(
    scratch: sqlite3.Connection,
    source: sqlite3.Connection,
    *,
    repo: str,
    number: int,
    sha: str,
    cutoff: str,
    exclude_run: int,
) -> dict[str, int]:
    """Copy what the worker reads about this PR's history, as of `cutoff`, into the scratch
    database: every run of the PR that finished before it (as its head, run, steps and
    `SEED_STAGES` findings rows, so per-commit lookups such as the standing verdict still
    work), its reviews and `thread.answered` events from before it, and `posted_findings`
    as they stood (a row re-posted after the cutoff is rebuilt from the earlier run that
    posted it). Returns row counts per table."""
    runs = source.execute(
        "SELECT r.*, h.sha, h.trigger, h.queue FROM runs r JOIN heads h ON h.id=r.head_id "
        "WHERE h.repo=? AND h.number=? AND r.id<>? AND r.finished_at IS NOT NULL "
        "AND r.finished_at < ? ORDER BY r.id",
        (repo, number, exclude_run, cutoff),
    ).fetchall()
    counts = dict.fromkeys(
        ("runs", "steps", "findings", "reviews", "posted_findings", "thread_answered"), 0
    )
    ids: dict[int, int] = {}
    with tx(scratch):
        scratch.execute(
            "INSERT OR REPLACE INTO prs (repo, number, head_sha, state, updated_at) "
            "VALUES (?,?,?,?,?)",
            (repo, number, sha, "open", cutoff),
        )
        for r in runs:
            hid = _ensure_head(scratch, repo, number, str(r["sha"]), str(r["trigger"]), "done")
            cur = scratch.execute(
                "INSERT INTO runs (head_id, attempt, status, reason, fail_kind, token, "
                "started_at, finished_at, deadline_at, phase, blocker_count, review_id, "
                "review_url, tier, degraded, path) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    hid,
                    r["attempt"],
                    r["status"],
                    r["reason"],
                    r["fail_kind"],
                    "seed",
                    r["started_at"],
                    r["finished_at"],
                    r["deadline_at"],
                    r["phase"],
                    r["blocker_count"],
                    r["review_id"],
                    r["review_url"],
                    _col(r, "tier"),
                    _col(r, "degraded") or 0,
                    _col(r, "path"),
                ),
            )
            ids[int(r["id"])] = int(cur.lastrowid or 0)
        counts["runs"] = len(ids)
        marks = ",".join("?" * len(ids))
        for s in source.execute(
            f"SELECT * FROM steps WHERE run_id IN ({marks})", tuple(ids)
        ).fetchall():
            scratch.execute(
                "INSERT INTO steps (run_id, name, status, started_at, finished_at, detail) "
                "VALUES (?,?,?,?,?,?)",
                (
                    ids[s["run_id"]],
                    s["name"],
                    s["status"],
                    s["started_at"],
                    s["finished_at"],
                    s["detail"],
                ),
            )
            counts["steps"] += 1
        posted: dict[str, tuple[Any, ...]] = {}  # hash -> rebuilt posted_findings row
        finished = {int(r["id"]): r for r in runs}
        stages = ",".join("?" * len(SEED_STAGES))
        for f in source.execute(
            f"SELECT * FROM findings WHERE run_id IN ({marks}) AND stage IN ({stages}) ORDER BY id",
            (*ids, *SEED_STAGES),
        ).fetchall():
            scratch.execute(
                "INSERT INTO findings (run_id, phase, stage, hash, file, line_start, line_end, "
                "severity, confidence, category, title, body) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ids[f["run_id"]],
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
            counts["findings"] += 1
            if f["stage"] == "posted":
                run = finished[int(f["run_id"])]
                posted[str(f["hash"])] = (
                    run["sha"],
                    run["review_id"],
                    None,
                    run["finished_at"],
                )
        for p in source.execute(
            "SELECT * FROM posted_findings WHERE repo=? AND number=? AND posted_at < ?",
            (repo, number, cutoff),
        ).fetchall():
            # the latest posting, and it was before the cutoff: authoritative
            posted[str(p["hash"])] = (p["sha"], p["review_id"], p["comment_id"], p["posted_at"])
        for h, (psha, review_id, comment_id, posted_at) in posted.items():
            scratch.execute(
                "INSERT OR REPLACE INTO posted_findings (repo, number, hash, sha, review_id, "
                "comment_id, posted_at) VALUES (?,?,?,?,?,?,?)",
                (repo, number, h, psha, review_id, comment_id, posted_at),
            )
        counts["posted_findings"] = len(posted)
        for rv in source.execute(
            "SELECT * FROM reviews WHERE repo=? AND number=? AND posted_at < ? ORDER BY id",
            (repo, number, cutoff),
        ).fetchall():
            scratch.execute(
                "INSERT INTO reviews (run_id, repo, number, sha, phase, github_review_id, "
                "event, posted_at, imported) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    ids.get(int(rv["run_id"])) if rv["run_id"] is not None else None,
                    repo,
                    number,
                    rv["sha"],
                    rv["phase"],
                    rv["github_review_id"],
                    rv["event"],
                    rv["posted_at"],
                    rv["imported"],
                ),
            )
            counts["reviews"] += 1
        for e in source.execute(
            "SELECT * FROM events WHERE kind='thread.answered' AND repo=? AND number=? "
            "AND ts < ? ORDER BY id",
            (repo, number, cutoff),
        ).fetchall():
            scratch.execute(
                "INSERT INTO events (ts, kind, repo, number, run_id, detail) VALUES (?,?,?,?,?,?)",
                (
                    e["ts"],
                    e["kind"],
                    repo,
                    number,
                    ids.get(int(e["run_id"])) if e["run_id"] is not None else None,
                    e["detail"],
                ),
            )
            counts["thread_answered"] += 1
    return counts


def _ensure_head(
    conn: sqlite3.Connection, repo: str, number: int, sha: str, trigger: str, status: str
) -> int:
    row = conn.execute(
        "SELECT id FROM heads WHERE repo=? AND number=? AND sha=?", (repo, number, sha)
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE heads SET status=?, trigger=?, queue='live', priority=0 WHERE id=?",
            (status, trigger, row["id"]),
        )
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO heads (repo, number, sha, trigger, status, queued_at, eligible_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (repo, number, sha, trigger, status, now(), now()),
    )
    return int(cur.lastrowid or 0)


# ---- policy and mode ----


def replay_policy(policy: cfg_mod.ModelPolicy, pin_phase1_model: str | None) -> cfg_mod.ModelPolicy:
    """The policy a replay runs: never on degraded-mode stand-ins (a quota failure fails the
    run instead), and with `pin_phase1_model` that one model on every Phase-1 lane, ungated
    by quota, so both arms of an eval run Phase 1 on the same rung."""
    pol = dataclasses.replace(policy, degraded=None)
    if not pin_phase1_model:
        return pol
    rung = next((c for c in policy.phase1_candidates if c.model == pin_phase1_model), None)
    if rung is None:
        cap = max(
            (c.effort for c in policy.phase1_candidates),
            key=cfg_mod.EFFORT_LEVELS.index,
            default=policy.phase1_reviewer.effort,
        )
        rung = cfg_mod.LaneModel(
            agent=policy.phase1_reviewer.agent, model=pin_phase1_model, effort=cap
        )
    return dataclasses.replace(
        pol, phase1_candidates=(dataclasses.replace(rung, quota=None, use_up_to=None),)
    )


def degraded_problem(
    policy: cfg_mod.ModelPolicy, source: sqlite3.Connection, prober: Prober | None
) -> str:
    """Why degraded mode is active right now ("" = it is not): forced on in the source
    database, or the policy's sentinel model answers with a quota error."""
    if degraded.forced(source) == "on":
        return "degraded mode is forced on (reviewsys degraded on)"
    if policy.degraded is None:
        return ""
    exhausted, reason = (prober or degraded.probe)(policy.degraded.sentinel)
    return f"the primary models are out of quota: {reason}" if exhausted else ""


# ---- the pinned view ----


def _git_out(git_dir: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(git_dir), *args],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def resolve_base(
    mirror: Path, *, sha: str, base_ref: str, base_sha: str, cutoff: str
) -> tuple[str, str]:
    """(base sha, merge base) the replay reviews from: the base the source run saw (its
    evidence's `baseRefOid`) or, without it, the base branch's tip as of the cutoff."""
    if base_sha and not wt.ensure_commit(mirror, base_sha):
        base_sha = ""
    if not base_sha and base_ref:
        wt.fetch_branch(mirror, base_ref)
        for ref in (f"refs/remotes/origin/{base_ref}", f"refs/heads/{base_ref}"):
            base_sha = _git_out(
                mirror, "rev-list", "-1", "--first-parent", f"--before={cutoff}", ref
            )
            if base_sha:
                break
    merge_base = _git_out(mirror, "merge-base", base_sha, sha) if base_sha else ""
    if not merge_base:
        raise SystemExit(
            f"cannot determine where {sha[:12]} branched from {base_ref or '?'} "
            f"(base {base_sha[:12] or 'unknown'}); is the mirror at {mirror} complete?"
        )
    return base_sha, merge_base


def stage_gh(real_gh: str, bin_root: Path) -> tuple[str, tuple[str, ...]]:
    """(the gh the pinned view runs, paths lanes may not execute). The installed gh is
    copied under `bin_root` and the original denied to lanes, so only the shim reaches
    GitHub. No sandbox (Linux) or no gh found: the installed one, nothing denied."""
    found = shutil.which(real_gh)
    if not found or not os.path.exists(SANDBOX_EXEC):
        return found or real_gh, ()
    return str(stage_real_gh(found, bin_root)), (found,)


def check_shim(shim: Path, profile: str) -> str:
    """Why the shim cannot run in a lane ("" = it can): run it, under the lane sandbox when
    there is one, as far as the gh copy it executes."""
    argv = [str(shim), "--version"]
    if profile and not sandbox_problem(profile):
        argv = [SANDBOX_EXEC, "-p", profile, *argv]
    for args in ([str(shim), SHIM_CHECK], argv):
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=60, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return str(exc)
        if r.returncode:
            return (r.stderr.strip() or f"exit {r.returncode}")[:300]
    return ""


# ---- the replay ----


def run_dir_name(run_id: int, sample: int = 0) -> str:
    return f"run-{run_id}" + (f"-r{sample}" if sample else "")


def replay(
    cfg: cfg_mod.Config,
    *,
    run_id: int,
    out: Path,
    skills: Path | None = None,
    source_db: Path | None = None,
    source_runs: Path | None = None,
    force_phase2: bool = False,
    pin_phase1_model: str | None = None,
    sample: int = 0,
    policy_name: str | None = None,
    prober: Prober | None = None,
    lane_runner: LaneRunner | None = None,
    gh_runner: GhRunner | None = None,
) -> dict[str, Any]:
    """Replay source run `run_id` into `out/run-<id>[-r<sample>]/` and return its report
    (also written there as report.json). Raises ReplayAborted in degraded mode."""
    skills_dir = skills or cfg.skills_dir
    repos, specialists, policy, _ = cfg_mod.load_skills_config(skills_dir)
    prod = open_source(source_db or cfg.db_path)
    try:
        src = _source_run(prod, run_id)
        problem = degraded_problem(policy, prod, prober)
        if problem:
            raise ReplayAborted(f"not replaying run {run_id}: {problem}")
        return _replay(
            cfg,
            prod,
            src,
            out=out,
            skills_dir=skills_dir,
            loaded=(repos, specialists, policy),
            source_runs=source_runs or cfg.runs_dir,
            force_phase2=force_phase2,
            pin_phase1_model=pin_phase1_model,
            sample=sample,
            policy_name=policy_name,
            lane_runner=lane_runner,
            gh_runner=gh_runner,
        )
    finally:
        prod.close()


def _replay(
    cfg: cfg_mod.Config,
    prod: sqlite3.Connection,
    src: sqlite3.Row,
    *,
    out: Path,
    skills_dir: Path,
    loaded: tuple[
        tuple[cfg_mod.RepoConfig, ...], tuple[cfg_mod.Specialist, ...], cfg_mod.ModelPolicy
    ],
    source_runs: Path,
    force_phase2: bool,
    pin_phase1_model: str | None,
    sample: int,
    policy_name: str | None,
    lane_runner: LaneRunner | None,
    gh_runner: GhRunner | None,
) -> dict[str, Any]:
    run_id, repo, number, sha = (
        int(src["id"]),
        str(src["repo"]),
        int(src["number"]),
        str(src["sha"]),
    )
    cutoff = str(src["started_at"])
    repos, specialists, policy = loaded
    work = out / run_dir_name(run_id, sample)
    if work.exists():
        shutil.rmtree(work)
    # a removed replay leaves its worktree registered in the shared mirror, and the same
    # worktree name is reused on a re-run: forget worktrees whose directories are gone
    for mirror in cfg.mirrors_dir.glob("*.git"):
        subprocess.run(
            ["git", "-C", str(mirror), "worktree", "prune"], capture_output=True, check=False
        )
    (work / "work").mkdir(parents=True)
    cfg.mirrors_dir.mkdir(parents=True, exist_ok=True)
    (work / "work" / "mirrors").symlink_to(cfg.mirrors_dir)  # shared, fetch-only
    real_gh, deny_gh = stage_gh(cfg.gh_bin, out / ".replay-bin")
    rc = dataclasses.replace(
        cfg,
        db_path=work / "replay.db",
        work_dir=work / "work",
        skills_dir=skills_dir,
        repos=repos,
        specialists=specialists,
        policy=replay_policy(policy, pin_phase1_model),
        backlog_skip_phase1_above=0,
        lane_deny_exec=(*cfg.lane_deny_exec, *deny_gh),
    )
    rc = policy_overrides(rc, policy_name)
    scratch = connect(rc.db_path)
    try:
        seeded = seed_history(
            scratch, prod, repo=repo, number=number, sha=sha, cutoff=cutoff, exclude_run=run_id
        )
        seed_ledger(scratch, prod, repo, number, cutoff)
        evidence = source_evidence(source_runs, run_id)
        ev_pr = (evidence.get("evidence") or {}).get("pr") or {}
        ci = (evidence.get("evidence") or {}).get("ci")
        mirror = wt.ensure_mirror(rc.mirrors_dir, repo)
        wt.fetch_head(mirror, number, sha)
        base_ref = str(ev_pr.get("baseRefName") or "")
        if not base_ref:  # no evidence on disk: the live base ref (refs are rarely renamed)
            live = Gh(real_gh, runner=read_only_runner(gh_runner)).api(
                f"repos/{repo}/pulls/{number}"
            )
            base_ref = str(((live or {}).get("base") or {}).get("ref") or "")
        base_sha, merge_base = resolve_base(
            mirror,
            sha=sha,
            base_ref=base_ref,
            base_sha=str(ev_pr.get("baseRefOid") or ""),
            cutoff=cutoff,
        )
        pin = Pin(
            repo=repo,
            number=number,
            sha=sha,
            cutoff=cutoff,
            merge_base=merge_base,
            base_ref=base_ref,
            base_sha=base_sha,
            git_dir=str(mirror),
            real_gh=real_gh,
            # a snapshot that failed to read is no snapshot: CI is then "unavailable"
            ci=ci if isinstance(ci, dict) and isinstance(ci.get("checks"), list) else None,
            pr={k: ev_pr[k] for k in ("title", "body") if isinstance(ev_pr.get(k), str)},
            jq=shutil.which("jq") or "",
            log=str(work / "gh-shim.log"),
        )
        pin.save(work / "pin.json")
        bin_dir = work / "bin"
        shim = install_shim(work / "pin.json", bin_dir)
        problem = check_shim(shim, exec_deny_profile(rc.lane_deny_exec))
        if problem:
            raise SystemExit(f"the replay's gh shim cannot run in a lane: {problem}")
        env_file = bin_dir / "env.sh"
        env_file.write_text(f'export PATH="{bin_dir}:$PATH"\n')
        env = {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
            # sourced by Claude Code before each Bash command: keeps the shim first even if
            # the lane's shell setup rewrites PATH
            "CLAUDE_ENV_FILE": str(env_file),
        }
        trigger = _source_trigger(prod, run_id)
        with tx(scratch):
            hid = _ensure_head(scratch, repo, number, sha, trigger.value, "running")
            cur = scratch.execute(
                "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at) "
                "VALUES (?,?,?,?,?,?)",
                (hid, 1, "spawned", "replay", now(), now()),
            )
            new_run = int(cur.lastrowid or 0)
        gh = Gh(real_gh, runner=read_only_runner(pinned_runner(pin, gh_runner)))
        t0 = time.monotonic()
        status = worker.main(
            rc,
            scratch,
            new_run,
            gh=gh,
            lane_runner=with_env(lane_runner or run_claude_lane, env),
            dry_run=True,
            heartbeat=False,
            force_phase2=force_phase2,
            coverage_base=merge_base,
        )
        wall = time.monotonic() - t0
        report = build_report(
            scratch,
            prod,
            rc,
            new_run,
            src,
            status=status,
            wall_clock_s=wall,
            options={
                "policy_name": policy_name,
                "skills": str(skills_dir),
                "force_phase2": force_phase2,
                "pin_phase1_model": pin_phase1_model,
                "sample": sample,
            },
            pin=pin,
            seeded=seeded,
            trigger=trigger,
        )
        (work / "report.json").write_text(json.dumps(report, indent=1))
    finally:
        scratch.close()
        # the worker keeps a failed run's worktree for debugging; a replay's is disposable (a
        # Rust repo's can reach tens of GB) and the lane artifacts are what the report needs
        shutil.rmtree(rc.worktrees_dir, ignore_errors=True)
        for m in cfg.mirrors_dir.glob("*.git"):
            subprocess.run(
                ["git", "-C", str(m), "worktree", "prune"], capture_output=True, check=False
            )
    return report


# ---- the report ----


def _json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _finding(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    d = dict(row)
    return {
        "hash": d.get("hash") or d.get("finding_hash") or "",
        "severity": d.get("severity"),
        "file": d.get("file") or "",
        "line_start": d.get("line_start"),
        "line_end": d.get("line_end"),
        "category": d.get("category") or "",
        "title": d.get("title") or "",
        **({"stage": d["stage"]} if "stage" in d else {}),
        **({"phase": d["phase"]} if "phase" in d else {}),
        **({"body": d["body"]} if d.get("body") else {}),
    }


def build_report(
    scratch: sqlite3.Connection,
    prod: sqlite3.Connection,
    rc: cfg_mod.Config,
    new_run: int,
    src: sqlite3.Row,
    *,
    status: RunStatus,
    wall_clock_s: float,
    options: dict[str, Any],
    pin: Pin,
    seeded: dict[str, int],
    trigger: Trigger,
) -> dict[str, Any]:
    run_dir = rc.runs_dir / f"run-{new_run}"
    run = scratch.execute("SELECT * FROM runs WHERE id=?", (new_run,)).fetchone()
    steps = {
        str(s["name"]): {
            "status": s["status"],
            "started_at": s["started_at"],
            "finished_at": s["finished_at"],
            "detail": _json_file_text(s["detail"]),
        }
        for s in scratch.execute("SELECT * FROM steps WHERE run_id=?", (new_run,))
    }
    lanes = [
        dict(r)
        for r in scratch.execute(
            "SELECT phase, role, agent, model, effort, attempt, status, exit_code, tokens_in, "
            "tokens_out, started_at, finished_at, reason, artifact_dir FROM lanes "
            "WHERE run_id=? ORDER BY id",
            (new_run,),
        )
    ]
    findings = [
        _finding(r)
        for r in scratch.execute(
            "SELECT stage, phase, hash, severity, file, line_start, line_end, category, title "
            "FROM findings WHERE run_id=? ORDER BY id",
            (new_run,),
        )
    ]
    publications = {
        phase: pub
        for phase in ("preliminary", "final")
        if isinstance(pub := _json_file(run_dir / f"publication-{phase}.json"), dict)
    }
    last = publications.get("final") or publications.get("preliminary")
    if last is not None:
        published = [_finding(f) for f in last.get("kept") or []]
        published_from = f"publication-{last.get('phase')}"
    else:  # e.g. a same-sha verdict follow-up: the last verifier's kept set
        phase = "verify2" if any(f.get("phase") == "verify2" for f in findings) else "verify1"
        published = [
            f
            for f in findings
            if f.get("phase") == phase and f.get("stage") in ("verified", "verified-fresh")
        ]
        published_from = f"{phase} verified" if published else ""
    gate = steps.get(StepName.GATE.value)
    would = _json_file(run_dir / "gate-would-publish.json")
    gpt = [x for x in lanes if pool_of(str(x["model"])) == "gpt"]
    contract = {
        "corrected": sum(x["status"] == "corrected" for x in lanes),
        "repaired": sum(x["status"] == "repaired" for x in lanes),
        "failed_attempts": sum(
            x["status"] == "failed"
            and not any(
                k in str(x["reason"] or "") for k in ("timed out", "lane exit", "API Error")
            )
            for x in lanes
        ),
        "run_failed": run["fail_kind"] == "contract",
    }
    reasons = [str(run["reason"] or ""), *(str(x["reason"] or "") for x in lanes)]
    quota_failure = status != RunStatus.DONE and any(
        degraded.looks_like_quota_failure(r) for r in reasons
    )
    shim_calls: dict[str, int] = {}
    refused: list[list[str]] = []
    for line in Path(pin.log).read_text().splitlines() if Path(pin.log).exists() else []:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        key = f"{e.get('source')}:{e.get('outcome')}"
        shim_calls[key] = shim_calls.get(key, 0) + 1
        if e.get("outcome") == "refused" and e.get("source") == "lane":
            refused.append(e.get("args") or [])
    return {
        "source_run": int(src["id"]),
        "repo": pin.repo,
        "number": pin.number,
        "sha": pin.sha,
        "cutoff": pin.cutoff,
        "source": {
            "status": src["status"],
            "tier": src["tier"],
            "degraded": bool(src["degraded"]),
            "trigger_replayed_as": trigger.value,
        },
        "policy": rc.policy.name,
        "policy_fingerprint": rc.policy.fingerprint,
        "options": options,
        "status": status.value,
        "reason": run["reason"],
        "fail_kind": run["fail_kind"],
        "invalid": "quota failure mid-run" if quota_failure else None,
        "tier": run["tier"],
        "wall_clock_s": round(wall_clock_s, 1),
        "gate": {
            "decision": (gate or {}).get("detail"),
            "forced_phase2": bool(((gate or {}).get("detail") or {}).get("forced_phase2")),
            "would_publish": would,
        },
        "steps": steps,
        "lanes": lanes,
        "gpt": {
            "lanes": len(gpt),
            "tokens_in": sum(int(x["tokens_in"] or 0) for x in gpt),
            "tokens_out": sum(int(x["tokens_out"] or 0) for x in gpt),
        },
        "tokens_in": sum(int(x["tokens_in"] or 0) for x in lanes),
        "tokens_out": sum(int(x["tokens_out"] or 0) for x in lanes),
        "contract": contract,
        "findings": findings,
        "published": published,
        "published_from": published_from,
        "publications": {
            k: {kk: v.get(kk) for kk in ("event", "transport_event", "suppressed")}
            for k, v in publications.items()
        },
        "review_body": (last or {}).get("body") or "",
        "conversation": _json_file(run_dir / "conversation.json"),
        "historical_posted": _historical(prod, int(src["id"])),
        "seeded": seeded,
        "pin": {
            "merge_base": pin.merge_base,
            "base_ref": pin.base_ref,
            "base_sha": pin.base_sha,
            "ci_snapshot": pin.ci is not None,
            "pr_text_pinned": sorted(pin.pr),
        },
        "gh_shim": {"calls": shim_calls, "lane_refused": refused[:50]},
        "run_dir": str(run_dir),
        "caveats": list(CAVEATS),
    }


def _json_file_text(text: str | None) -> Any:
    try:
        return json.loads(text or "null")
    except ValueError:
        return text
