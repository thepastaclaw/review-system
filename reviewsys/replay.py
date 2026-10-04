"""`reviewsys replay`: re-review a PR at the exact head a past run reviewed, under whichever
policy a skills checkout holds, without touching GitHub or the production database.

Built for the v12 evaluation (DESIGN.md section 5): run the same heads under two policies and
compare what each finds with labels and with each other.

The replay sees the PR as it was when the source run started (the cutoff):

- History: the scratch database is seeded with this PR's runs that finished before the cutoff
  (their posted, conceded and verified findings, steps and reviews), the `posted_findings`
  rows, reviews and `thread.answered` events from before it, and (hook, see `seed_ledger`)
  the issue ledger as of it.
- GitHub: the harness and every lane read the PR through the pinned view in `ghshim` (the
  replayed head and base, CI from the source run's evidence, nothing created after the
  cutoff, edits rewound where they can be); every write is refused. Lanes reach it through a
  `gh` shim first on their PATH that asks the replay's broker; the lane sandbox denies
  executing the installed gh, a curl shim refuses GitHub hosts, and Claude Code's web tools
  are taken away. With a read-only token (`--gh-token-env`, default
  `REVIEWSYS_REPLAY_GH_TOKEN`) every live read the view makes uses it, and the lanes' own
  GH_TOKEN/GITHUB_TOKEN are set to it (blank without one).
- Range: the review covers `merge-base(base the source run saw, head)..head`, not the merge
  base with today's base branch (for a PR merged since, that would be the head itself).
- Machine: lanes take slots in the configured work dir's `lane-slots` (the same pools as the
  box config), so concurrent replays share the model caps instead of multiplying them.

Eval switches: `force_phase2` runs Phase 2 whatever the gate decides (recording what the gate
would have published), `pin_phase1_model` puts one model on every Phase-1 lane in both arms,
and a replay refuses to start while degraded mode is active (its stand-in models would not be
the policy under test); mid-run it never switches to stand-ins, and a quota failure marks the
report invalid. The harness adds no turn or candidate caps.

Live-GitHub detection is a heuristic: lanes keep no transcript (`--no-session-persistence`),
so after the run their answers and stderr are searched for the replayed repository's API and
raw-content URLs, and refused gh and curl attempts are listed. A hit marks the report invalid.

Safety: a scratch SQLite database under `--out`, the worker's dry-run mode, and a read-only gh.
The production work directory's bare mirrors are shared (read + fetch only); worktrees and
run directories live under `--out` and the worktrees are removed when the replay ends.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from . import config as cfg_mod
from . import degraded, worker
from .db import connect, now, tx
from .degraded import Prober
from .gh import Gh, GhRunner, _default_runner, env_runner, read_only_runner
from .ghshim import Broker, Pin, install_shim, pinned_runner, real_gh
from .lane import (
    SANDBOX_EXEC,
    LaneRunner,
    exec_deny_profile,
    run_claude_lane,
    sandbox_problem,
    with_overrides,
)
from .lanepool import pool_of
from .models import RunStatus, StepName, Trigger
from .steps import worktree as wt

EXIT_ABORTED = 75  # EX_TEMPFAIL: the replay refused to start or its models ran dry mid-run
TOKEN_ENV = "REVIEWSYS_REPLAY_GH_TOKEN"
LANE_DISALLOWED_TOOLS = ("WebFetch", "WebSearch")
# findings stages a later run reads from earlier ones (prior findings, standing verdicts)
SEED_STAGES = ("posted", "conceded", "verified", "verified-fresh")
# thread answers that put a status line on the finding's root comment (publish.py)
LIFTING = frozenset({"WITHDRAWN", "FIXED", "OUTDATED", "INTENTIONALLY_DEFERRED"})
CAVEATS = (
    "thread resolution and outdatedness, reactions and labels are as of now where the cutoff "
    "cannot rewind them (counted under gh_shim.rewind)",
    "converse silence memory (kv) is not seeded: a reply the bot chose not to answer may be "
    "considered again",
    "a conversation run's replies are not posted, so its verdict follow-up sees nothing lifted",
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


def read_json(path: Path) -> Any:
    """A JSON file's content, None when it is missing or not JSON."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def open_source(path: Path) -> sqlite3.Connection:
    """The production (or copied) database, read-only."""
    if not path.exists():
        raise SystemExit(f"source database {path} not found")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _source_run(prod: sqlite3.Connection, run_id: int) -> sqlite3.Row:
    """The run to replay. An audit-queue run (a review of a PR posted from the audit queue, or
    a conversation there) is replayed as a live review of its head."""
    row = prod.execute(
        "SELECT r.id, r.started_at, r.status, r.tier, r.degraded, h.repo, h.number, h.sha, "
        "h.trigger, h.queue FROM runs r JOIN heads h ON h.id=r.head_id WHERE r.id=?",
        (run_id,),
    ).fetchone()
    if not isinstance(row, sqlite3.Row):
        raise SystemExit(f"run {run_id} not found in the source database")
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
    data = read_json(runs_dir / f"run-{run_id}" / "evidence.json")
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


def _insert(conn: sqlite3.Connection, table: str, row: dict[str, Any], verb: str = "INSERT") -> int:
    cols = ", ".join(row)
    cur = conn.execute(
        f"{verb} INTO {table} ({cols}) VALUES ({','.join('?' * len(row))})", tuple(row.values())
    )
    return int(cur.lastrowid or 0)


def _col(row: sqlite3.Row, key: str) -> Any:
    return row[key] if key in row.keys() else None  # noqa: SIM118 (Row `in` tests values)


_FINDING_COLS = (
    "phase",
    "stage",
    "hash",
    "file",
    "line_start",
    "line_end",
    "severity",
    "confidence",
    "category",
    "title",
    "body",
)


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
    posted it). A finding posted before the cutoff by a run that finished after it gets its
    `posted` findings row under a placeholder run at its commit (without one the worker's
    prior list would drop it). Returns row counts per table."""
    runs = source.execute(
        "SELECT r.*, h.sha, h.trigger FROM runs r JOIN heads h ON h.id=r.head_id "
        "WHERE h.repo=? AND h.number=? AND r.id<>? AND r.finished_at IS NOT NULL "
        "AND r.finished_at < ? ORDER BY r.id",
        (repo, number, exclude_run, cutoff),
    ).fetchall()
    counts: Counter[str] = Counter()
    ids: dict[int, int] = {}
    with tx(scratch):
        _insert(
            scratch,
            "prs",
            {
                "repo": repo,
                "number": number,
                "head_sha": sha,
                "state": "open",
                "updated_at": cutoff,
            },
            "INSERT OR REPLACE",
        )
        for r in runs:
            hid = _ensure_head(scratch, repo, number, str(r["sha"]), str(r["trigger"]))
            ids[int(r["id"])] = _insert(
                scratch,
                "runs",
                {
                    "head_id": hid,
                    "token": "seed",
                    **{
                        k: r[k]
                        for k in (
                            "attempt",
                            "status",
                            "reason",
                            "fail_kind",
                            "started_at",
                            "finished_at",
                            "deadline_at",
                            "phase",
                            "blocker_count",
                            "review_id",
                            "review_url",
                        )
                    },
                    "tier": _col(r, "tier"),
                    "degraded": _col(r, "degraded") or 0,
                    "path": _col(r, "path"),
                },
            )
        counts["runs"] = len(ids)
        marks = ",".join("?" * len(ids))
        for s in source.execute(
            f"SELECT * FROM steps WHERE run_id IN ({marks})", tuple(ids)
        ).fetchall():
            _insert(
                scratch,
                "steps",
                {
                    "run_id": ids[s["run_id"]],
                    **{k: s[k] for k in ("name", "status", "started_at", "finished_at", "detail")},
                },
            )
            counts["steps"] += 1
        posted: dict[str, tuple[Any, ...]] = {}  # hash -> rebuilt posted_findings row
        with_row: set[str] = set()  # posted hashes with a seeded `posted` findings row
        finished = {int(r["id"]): r for r in runs}
        stages = ",".join("?" * len(SEED_STAGES))
        for f in source.execute(
            f"SELECT * FROM findings WHERE run_id IN ({marks}) AND stage IN ({stages}) ORDER BY id",
            (*ids, *SEED_STAGES),
        ).fetchall():
            _insert(
                scratch,
                "findings",
                {"run_id": ids[f["run_id"]], **{k: f[k] for k in _FINDING_COLS}},
            )
            counts["findings"] += 1
            if f["stage"] == "posted":
                run = finished[int(f["run_id"])]
                posted[str(f["hash"])] = (run["sha"], run["review_id"], None, run["finished_at"])
                with_row.add(str(f["hash"]))
        for p in source.execute(
            "SELECT * FROM posted_findings WHERE repo=? AND number=? AND posted_at < ?",
            (repo, number, cutoff),
        ).fetchall():
            # the latest posting, and it was before the cutoff: authoritative
            posted[str(p["hash"])] = (p["sha"], p["review_id"], p["comment_id"], p["posted_at"])
        placeholders: dict[str, int] = {}  # commit -> placeholder run
        for h, (psha, review_id, comment_id, posted_at) in posted.items():
            _insert(
                scratch,
                "posted_findings",
                {
                    "repo": repo,
                    "number": number,
                    "hash": h,
                    "sha": psha,
                    "review_id": review_id,
                    "comment_id": comment_id,
                    "posted_at": posted_at,
                },
                "INSERT OR REPLACE",
            )
            if h in with_row:
                continue
            f = source.execute(
                "SELECT f.* FROM findings f JOIN runs r ON r.id=f.run_id "
                "JOIN heads hd ON hd.id=r.head_id WHERE hd.repo=? AND hd.number=? AND f.hash=? "
                "AND f.stage='posted' AND r.started_at < ? ORDER BY f.id DESC LIMIT 1",
                (repo, number, h, cutoff),
            ).fetchone()
            if f is None:
                continue
            if psha not in placeholders:
                hid = _ensure_head(scratch, repo, number, str(psha), Trigger.MANUAL.value)
                placeholders[psha] = _insert(
                    scratch,
                    "runs",
                    {
                        "head_id": hid,
                        "attempt": 1,
                        "status": "done",
                        "token": "seed-posted",
                        "started_at": posted_at,
                        "finished_at": posted_at,
                        "deadline_at": posted_at,
                    },
                )
            _insert(
                scratch,
                "findings",
                {"run_id": placeholders[psha], **{k: f[k] for k in _FINDING_COLS}},
            )
            counts["posted_rows_recovered"] += 1
        counts["posted_findings"] = len(posted)
        for rv in source.execute(
            "SELECT * FROM reviews WHERE repo=? AND number=? AND posted_at < ? ORDER BY id",
            (repo, number, cutoff),
        ).fetchall():
            _insert(
                scratch,
                "reviews",
                {
                    "run_id": ids.get(int(rv["run_id"])) if rv["run_id"] is not None else None,
                    "repo": repo,
                    "number": number,
                    **{
                        k: rv[k]
                        for k in (
                            "sha",
                            "phase",
                            "github_review_id",
                            "event",
                            "posted_at",
                            "imported",
                        )
                    },
                },
            )
            counts["reviews"] += 1
        for e in _answered(source, repo, number, cutoff):
            _insert(
                scratch,
                "events",
                {
                    "ts": e["ts"],
                    "kind": e["kind"],
                    "repo": repo,
                    "number": number,
                    "run_id": ids.get(int(e["run_id"])) if e["run_id"] is not None else None,
                    "detail": e["detail"],
                },
            )
            counts["thread_answered"] += 1
    return dict(counts)


def _answered(source: sqlite3.Connection, repo: str, number: int, cutoff: str) -> list[sqlite3.Row]:
    return source.execute(
        "SELECT * FROM events WHERE kind='thread.answered' AND repo=? AND number=? "
        "AND ts < ? ORDER BY id",
        (repo, number, cutoff),
    ).fetchall()


def lifted_before(
    source: sqlite3.Connection, repo: str, number: int, cutoff: str
) -> dict[str, str]:
    """finding hash -> the lifting status its thread showed at the cutoff: the last thread
    answer before it that reached the thread (anything but a silent or skipped one)."""
    out: dict[str, str] = {}
    for e in _answered(source, repo, number, cutoff):
        try:
            d = json.loads(e["detail"] or "{}")
        except ValueError:
            continue
        h, status = str(d.get("finding_hash") or ""), str(d.get("status") or "")
        if not h or d.get("action") in ("no_reply", "suppressed_reply", "duplicate_thread_skipped"):
            continue
        if status in LIFTING:
            out[h] = status
        else:
            out.pop(h, None)
    return out


def _ensure_head(conn: sqlite3.Connection, repo: str, number: int, sha: str, trigger: str) -> int:
    row = conn.execute(
        "SELECT id FROM heads WHERE repo=? AND number=? AND sha=?", (repo, number, sha)
    ).fetchone()
    if row:
        return int(row["id"])
    return _insert(
        conn,
        "heads",
        {
            "repo": repo,
            "number": number,
            "sha": sha,
            "trigger": trigger,
            "status": "done",
            "queued_at": now(),
            "eligible_at": now(),
        },
    )


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


def check_shim(shim: Path, profile: str) -> str:
    """Why the shim cannot serve a lane ("" = it can): `gh --version` through it, under the
    lane sandbox when there is one, as far as the broker's real gh."""
    argv = [str(shim), "--version"]
    if profile and not sandbox_problem(profile):
        argv = [SANDBOX_EXEC, "-p", profile, *argv]
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    return "" if r.returncode == 0 else (r.stderr.strip() or f"exit {r.returncode}")[:300]


def _prune_worktrees(mirrors_dir: Path) -> None:
    """Forget worktrees whose directories are gone: a removed replay leaves its worktree
    registered in the shared mirror, and a re-run reuses the same worktree name."""
    for m in mirrors_dir.glob("*.git"):
        subprocess.run(["git", "-C", str(m), "worktree", "prune"], capture_output=True, check=False)


def _link(path: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target)


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
    gh_token: str | None = None,
    prober: Prober | None = None,
    lane_runner: LaneRunner | None = None,
    gh_runner: GhRunner | None = None,
) -> dict[str, Any]:
    """Replay source run `run_id` into `out/run-<id>[-r<sample>]/` and return its report
    (also written there as report.json). Raises ReplayAborted in degraded mode."""
    skills_dir = skills or cfg.skills_dir
    repos, specialists, policy, _ = cfg_mod.load_skills_config(skills_dir)
    arm = dataclasses.replace(
        cfg, skills_dir=skills_dir, repos=repos, specialists=specialists, policy=policy
    )
    if not gh_token:
        print(
            f"WARNING: no read-only GitHub token ({TOKEN_ENV} or --gh-token-env): the replay "
            "reads GitHub with your own gh credentials; writes are refused by code only",
            file=sys.stderr,
        )
    prod = open_source(source_db or cfg.db_path)
    try:
        src = _source_run(prod, run_id)
        problem = degraded_problem(policy, prod, prober)
        if problem:
            raise ReplayAborted(f"not replaying run {run_id}: {problem}")
        return _replay(
            arm,
            prod,
            src,
            out=out,
            source_runs=source_runs or cfg.runs_dir,
            force_phase2=force_phase2,
            pin_phase1_model=pin_phase1_model,
            sample=sample,
            policy_name=policy_name,
            gh_token=gh_token,
            lane_runner=lane_runner,
            gh_runner=gh_runner,
        )
    finally:
        prod.close()


def _replay(
    arm: cfg_mod.Config,
    prod: sqlite3.Connection,
    src: sqlite3.Row,
    *,
    out: Path,
    source_runs: Path,
    force_phase2: bool,
    pin_phase1_model: str | None,
    sample: int,
    policy_name: str | None,
    gh_token: str | None,
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
    work = out / run_dir_name(run_id, sample)
    if work.exists():
        shutil.rmtree(work)
    _prune_worktrees(arm.mirrors_dir)
    (work / "work").mkdir(parents=True)
    _link(work / "work" / "mirrors", arm.mirrors_dir)  # shared, fetch-only
    _link(work / "work" / "lane-slots", arm.work_dir / "lane-slots")  # the machine's pools
    installed_gh = shutil.which(arm.gh_bin) or arm.gh_bin
    # lanes get no gh of their own: the shim's broker is the only way to GitHub
    deny_gh = (
        (installed_gh,) if os.path.exists(SANDBOX_EXEC) and os.path.isabs(installed_gh) else ()
    )
    rc = dataclasses.replace(
        arm,
        db_path=work / "replay.db",
        work_dir=work / "work",
        policy=replay_policy(arm.policy, pin_phase1_model),
        backlog_skip_phase1_above=0,
        lane_deny_exec=(*arm.lane_deny_exec, *deny_gh),
    )
    rc = policy_overrides(rc, policy_name)
    token_env = {"GH_TOKEN": gh_token or "", "GITHUB_TOKEN": gh_token or ""}
    inner = gh_runner or (env_runner(token_env) if gh_token else _default_runner)
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
            live = Gh(installed_gh, runner=read_only_runner(inner)).api(
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
            real_gh=installed_gh,
            # a snapshot that failed to read is no snapshot: CI is then "unavailable"
            ci=ci if isinstance(ci, dict) and isinstance(ci.get("checks"), list) else None,
            pr={k: ev_pr[k] for k in ("title", "body") if isinstance(ev_pr.get(k), str)},
            lifted=lifted_before(prod, repo, number, cutoff),
            jq=shutil.which("jq") or "",
            log=str(work / "gh-shim.log"),
        )
        pin.save(work / "pin.json")
        trigger = _source_trigger(prod, run_id)
        with tx(scratch):
            hid = _ensure_head(scratch, repo, number, sha, trigger.value)
            scratch.execute(
                "UPDATE heads SET status='running', trigger=?, queue='live', priority=0 WHERE id=?",
                (trigger.value, hid),
            )
            new_run = _insert(
                scratch,
                "runs",
                {
                    "head_id": hid,
                    "attempt": 1,
                    "status": "spawned",
                    "token": "replay",
                    "started_at": now(),
                    "deadline_at": now(),
                },
            )
        gh = Gh(installed_gh, runner=read_only_runner(pinned_runner(pin, inner)))
        bin_dir = work / "bin"
        with Broker(pin, real_gh(pin, inner)) as broker:
            shim = install_shim(broker.path, bin_dir, curl_log=work / "curl-refused.log")
            problem = check_shim(shim, exec_deny_profile(rc.lane_deny_exec))
            if problem:
                raise SystemExit(f"the replay's gh shim cannot serve a lane: {problem}")
            env_file = bin_dir / "env.sh"
            env_file.write_text(f'export PATH="{bin_dir}:$PATH"\n')
            env = {
                "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
                # sourced by Claude Code before each Bash command: keeps the shims first even
                # if the lane's shell setup rewrites PATH
                "CLAUDE_ENV_FILE": str(env_file),
                **token_env,
            }
            t0 = time.monotonic()
            status = worker.main(
                rc,
                scratch,
                new_run,
                gh=gh,
                lane_runner=with_overrides(
                    lane_runner or run_claude_lane,
                    env=env,
                    disallowed_tools=LANE_DISALLOWED_TOOLS,
                ),
                dry_run=True,
                heartbeat=False,
                replay=True,
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
                "skills": str(rc.skills_dir),
                "force_phase2": force_phase2,
                "pin_phase1_model": pin_phase1_model,
                "sample": sample,
                "gh_token": "read-only token" if gh_token else "none (own gh credentials)",
            },
            pin=pin,
            seeded=seeded,
            trigger=trigger,
            work=work,
        )
        (work / "report.json").write_text(json.dumps(report, indent=1))
    finally:
        scratch.close()
        # the worker keeps a failed run's worktree for debugging; a replay's is disposable (a
        # Rust repo's can reach tens of GB) and the lane artifacts are what the report needs
        shutil.rmtree(rc.worktrees_dir, ignore_errors=True)
        _prune_worktrees(arm.mirrors_dir)
    return report


# ---- the report ----


def _finding(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    d = dict(row)
    return {
        "hash": d.get("hash") or "",
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


def live_github_hits(lanes: list[dict[str, Any]], repo: str) -> list[str]:
    """Lane answers and stderr that show the replayed repository read from GitHub directly
    (its API or raw-content URLs): `artifact:pattern`, at most 20."""
    pat = re.compile(
        rf"(api\.github\.com/repos/{re.escape(repo)}\b|raw\.githubusercontent\.com/{re.escape(repo)}/)",
        re.IGNORECASE,
    )
    hits: list[str] = []
    for lane in lanes:
        d = Path(str(lane.get("artifact_dir") or ""))
        if not d.is_dir():
            continue
        for f in [*d.glob("stdout.json"), *d.glob("stderr.txt"), *d.glob("correction-*/std*")]:
            try:
                m = pat.search(f.read_text(errors="replace"))
            except OSError:
                continue
            if m:
                hits.append(f"{f}:{m.group(0)}")
    return sorted(set(hits))[:20]


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
    work: Path,
) -> dict[str, Any]:
    run_dir = rc.runs_dir / f"run-{new_run}"
    run = scratch.execute("SELECT * FROM runs WHERE id=?", (new_run,)).fetchone()
    steps = {
        str(s["name"]): {
            "status": s["status"],
            "started_at": s["started_at"],
            "finished_at": s["finished_at"],
            "detail": _json_text(s["detail"]),
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
        if isinstance(pub := read_json(run_dir / f"publication-{phase}.json"), dict)
    }
    last = publications.get("final") or publications.get("preliminary")
    if last is not None:
        published = [_finding(f) for f in last.get("kept") or []]
        published_from = f"publication-{last.get('phase')}"
    else:
        phase = "verify2" if any(f.get("phase") == "verify2" for f in findings) else "verify1"
        published = [
            f
            for f in findings
            if f.get("phase") == phase and f.get("stage") in ("verified", "verified-fresh")
        ]
        published_from = f"{phase} verified" if published else ""
    gate = steps.get(StepName.GATE.value)
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
    calls: Counter[str] = Counter()
    rewind: Counter[str] = Counter()
    refused: list[list[str]] = []
    log = Path(pin.log)
    for line in log.read_text().splitlines() if log.exists() else []:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        calls[f"{e.get('source')}:{e.get('outcome')}"] += 1
        rewind.update(e.get("notes") or {})
        if e.get("outcome") == "refused" and e.get("source") == "lane":
            refused.append(e.get("args") or [])
    curl_log = work / "curl-refused.log"
    curl_refused = curl_log.read_text().splitlines() if curl_log.exists() else []
    hits = live_github_hits(lanes, pin.repo)
    invalid = None
    if hits:
        invalid = "lane reached live GitHub"
    elif status != RunStatus.DONE and any(degraded.looks_like_quota_failure(r) for r in reasons):
        invalid = "quota failure mid-run"
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
            "queue": src["queue"],
            "trigger_replayed_as": trigger.value,
        },
        "policy": rc.policy.name,
        "policy_fingerprint": rc.policy.fingerprint,
        "options": options,
        "status": status.value,
        "reason": run["reason"],
        "fail_kind": run["fail_kind"],
        "invalid": invalid,
        "tier": run["tier"],
        "wall_clock_s": round(wall_clock_s, 1),
        "gate": {
            "decision": (gate or {}).get("detail"),
            "forced_phase2": bool(((gate or {}).get("detail") or {}).get("forced_phase2")),
            "would_publish": read_json(run_dir / "gate-would-publish.json"),
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
            k: {
                kk: v.get(kk) for kk in ("event", "transport_event", "suppressed", "verdict_update")
            }
            for k, v in publications.items()
        },
        "review_body": (last or {}).get("body") or "",
        "conversation": read_json(run_dir / "conversation.json"),
        "historical_posted": _historical(prod, int(src["id"])),
        "seeded": seeded,
        "pin": {
            "merge_base": pin.merge_base,
            "base_ref": pin.base_ref,
            "base_sha": pin.base_sha,
            "ci_snapshot": pin.ci is not None,
            "pr_text_pinned": sorted(pin.pr),
            "lifted_before_cutoff": len(pin.lifted),
        },
        "gh_shim": {"calls": dict(calls), "rewind": dict(rewind), "lane_refused": refused[:50]},
        "live_github": {"lane_output_hits": hits, "curl_refused": curl_refused[:50]},
        "run_dir": str(run_dir),
        "caveats": list(CAVEATS),
    }


def _json_text(text: str | None) -> Any:
    try:
        return json.loads(text or "null")
    except ValueError:
        return text
