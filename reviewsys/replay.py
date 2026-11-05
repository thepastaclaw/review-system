"""`reviewsys replay`: re-review a PR at the exact head a past run reviewed, under whichever
policy a skills checkout holds, without touching GitHub or the production database.

Built for the v12 evaluation (DESIGN.md section 5): run the same heads under two policies and
compare what each finds with labels and with each other.

The replay sees the PR as it was when the source run started (the cutoff):

- History: the scratch database is seeded with this PR's runs that finished before the cutoff
  (their posted, conceded and verified findings, steps and reviews, under their source run
  ids), the `posted_findings` rows, reviews and `thread.answered` events from before it. For
  a policy that reads the issue ledger (v12), also the ledger as of the cutoff
  (`seed_ledger`), reconciled as `ledger reconcile` would have at that moment
  (`reconcile_as_of`: closed from the source database, threads from the pinned view, code
  from the pinned repository). A v11 arm gets no ledger: it never reads one.
- GitHub: the harness and every lane read the PR through the pinned view in `ghshim` (the
  replayed head and base, CI from the source run's evidence, nothing created after the
  cutoff, edits rewound where they can be); every write is refused. Lanes reach it through a
  `gh` shim first on their PATH that asks the replay's broker; the lane sandbox denies
  executing the installed gh, a curl shim refuses GitHub hosts, and Claude Code's web tools
  are taken away. With a read-only token (`--gh-token-env`, default
  `REVIEWSYS_REPLAY_GH_TOKEN`) every live read the view makes uses it. The token stays with
  the broker: lanes get GH_TOKEN, GITHUB_TOKEN and REVIEWSYS_REPLAY_GH_TOKEN blank and an
  empty GH_CONFIG_DIR.
- Range: the review covers `merge-base(base the source run saw, head)..head`, not the merge
  base with today's base branch (for a PR merged since, that would be the head itself).
- Git (replaygit.py): the worktree comes from a per-replay bare repository that borrows the
  production mirror's objects read-only (alternates: nothing is copied) and has no refs but
  the base branch at the base the source run saw and the PR head at the replayed sha, so
  `git log origin/<base>`, `refs/pull/N/head` and `--all` show nothing pushed after the
  cutoff. Whatever the mirror lacks is fetched into that repository; nothing is ever written
  to a production mirror (a repository without one is mirrored under `eval_mirrors`).
- Machine: lanes take slots in the configured work dir's `lane-slots` (the same pools as the
  box config, the gpt pool sized from the configured database's account reading), at the
  `replay` rank: behind every production lane, never the side-lane slot, at most
  `replay_pool_share` of a pool (lanepool.py). Concurrent replays and the production daemon
  share the model caps instead of multiplying them.

Eval switches: `force_phase2` runs Phase 2 whatever the gate decides (recording what the gate
would have published), `pin_phase1_model` puts one model on every Phase-1 lane in both arms,
and a replay refuses to start while degraded mode is active (its stand-in models would not be
the policy under test); mid-run it never switches to stand-ins, and a quota failure marks the
report invalid. The harness adds no turn or candidate caps.

Live-GitHub detection is a heuristic: lanes keep no transcript (`--no-session-persistence`),
so after the run their answers and stderr are searched for the replayed repository's API and
raw-content URLs, and refused gh and curl attempts are listed. A hit marks the report invalid.

Safety: a scratch SQLite database under `--out`, the worker's dry-run mode, and a read-only gh.
The production work directory's bare mirrors are only read; the pinned clone,
worktrees and run directories live under `--out`, and the clone and worktrees are removed when
the replay ends (the run artifacts and report stay).
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
from . import converse, degraded, github, ledger, replaygit, worker
from .db import connect, kv_get, kv_set, now, parse_ts, tx
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

EXIT_ABORTED = 75  # EX_TEMPFAIL: the replay refused to start or its models ran dry mid-run
TOKEN_ENV = "REVIEWSYS_REPLAY_GH_TOKEN"
# never in a lane's environment (blanked there; the CLI drops them from its own once read)
TOKEN_VARS = (TOKEN_ENV, "GH_TOKEN", "GITHUB_TOKEN")
LANE_DISALLOWED_TOOLS = ("WebFetch", "WebSearch")
# findings stages a later run reads from earlier ones (prior findings, standing verdicts)
SEED_STAGES = ("posted", "conceded", "verified", "verified-fresh")
# events the worker reads about an earlier run (besides each PR's `thread.answered`): what
# makes the standing review strong enough to approve over under v12 (`worker._strength_row`)
RUN_EVENT_KINDS = (
    "verify.shard_dropped",
    "threads.conflict",
    "confirmation.skipped",
    "confirmation.failed",
)
# thread answers that put a status line on the finding's root comment (publish.py)
LIFTING = frozenset({"WITHDRAWN", "FIXED", "OUTDATED", "INTENTIONALLY_DEFERRED"})
CAVEATS = (
    "thread resolution and outdatedness, reactions and labels are as of now where the cutoff "
    "cannot rewind them (counted under gh_shim.rewind)",
    "converse silence memory (kv) is not seeded: a reply the bot chose not to answer may be "
    "considered again",
    "a conversation run's replies are not posted: under policy v11 its verdict follow-up sees "
    "nothing lifted; under v12 it decides on what the outcomes would lift and records the "
    "follow-up it would post (conversation.json `would_post`) without posting it",
)


V12_GATE_NOTE = (
    "not recorded under policy v12: its preliminary publication comes from the post-rules "
    "pass of _publish12 (budget, settled re-folds, thread outcomes, ledger verdict), which "
    "cannot be rerun for a publication that never happens"
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
) -> dict[str, Any]:
    """This PR's issue ledger as of `cutoff` (ISO-8601 Z, the source run's start), in the
    scratch database: the source's `issue_events` from before it, folded (when the source has
    the ledger), then `ledger.backfill` over the scratch history `seed_history` left (runs,
    reviews, answers and concessions from before the cutoff). The backfill derives the same
    event key for the same fact as the write-through, and the scratch runs keep their source
    ids, so it only fills what the source lacks; on a source without a ledger (the box before
    its backfill) it is the whole ledger. Must be called outside a transaction."""
    copied = 0
    if _has_table(source_db, "issue_events"):
        events = [
            ledger.Event(
                hash=str(r["hash"]),
                key=str(r["key"]),
                kind=str(r["kind"]),
                source=str(r["source"]),
                run_id=r["run_id"],
                sha=r["sha"],
                detail=json.loads(r["detail"] or "{}"),
                ts=str(r["ts"]),
            )
            for r in source_db.execute(
                "SELECT * FROM issue_events WHERE repo=? AND number=? AND ts < ? ORDER BY ts, id",
                (repo, number, cutoff),
            )
        ]
        with tx(scratch_db):
            copied = ledger.append(scratch_db, repo, number, events)
    filled = ledger.backfill(scratch_db, repo=repo, number=number)
    return {
        "events_copied": copied,
        "events_backfilled": int(filled["events_added"]),
        "issues": int(filled["issues"]),
        "by_status": filled["by_status"],
    }


def reconcile_as_of(
    scratch_db: sqlite3.Connection,
    source_db: sqlite3.Connection,
    *,
    pin: Pin,
    gh: Gh,
    bot_login: str,
    git_dir: Path,
) -> dict[str, Any]:
    """`ledger reconcile` (the one-shot before a PR's first policy v12 round) for this PR, at
    the cutoff, as if v12 had been switched on then: whether the PR was closed comes from the
    source database (`closed_before`), its threads from the pinned view, and whether an
    issue's code moved from git in the pinned repository against the replayed head. No
    network beyond the pinned view's reads. Records the reconcile in the scratch kv, where
    the worker (which skips it in a dry run) reads its pending list.

    Production reconciles a PR once. When the source database recorded that reconcile before
    the cutoff (`recorded_reconcile`), its record as of the cutoff is copied and nothing runs
    (`source: recorded`); only a PR v12 had not reconciled by then is reconciled here
    (`source: replayed`), and the thread-lane work only that adds is counted apart in the
    report (`reconcile_work`)."""
    repo, number, head = pin.repo, pin.number, pin.sha
    rec = recorded_reconcile(source_db, repo, number, pin.cutoff)
    if rec is not None:
        with tx(scratch_db):
            ledger.record_reconciled(
                scratch_db,
                repo,
                number,
                {k: rec[k] for k in ("run", "sha", "pending", "at")},
            )
        return {
            "source": "recorded",
            "at": rec["at"],
            "pending": len(rec["pending"]),
            "pending_exact": rec["exact"],
        }
    issues = ledger.load(scratch_db, repo, number)
    before = {h for h, i in issues.items() if i.status in ledger.COUNTING}
    closed = closed_before(source_db, repo, number, pin.cutoff)
    if closed:
        with tx(scratch_db):
            ledger.reconcile(scratch_db, repo, number, pr_open=False, open_threads=(), moved={})
        return {
            "source": "replayed",
            "pr_open": False,
            "closed": closed,
            "standing": len(before),
            "pending": 0,
        }
    threads = github.review_threads(gh, repo, number)
    open_ = set(github.finding_threads(threads, bot_login))
    # a thread's resolved state is as of now (the view cannot rewind it): what the sync below
    # makes of resolved threads is not exact, and the head is flagged for scoring
    as_of_now = sum(1 for t in threads if t.get("is_resolved"))

    def changed(a: str, b: str, file: str, ls: int | None, le: int | None) -> bool | None:
        return ledger.lines_changed(git_dir, a, b, file, ls, le) if file else None

    moved = ledger.moved_anchors(issues, head, changed, open_threads=open_)
    with tx(scratch_db):
        ledger.sync_threads(
            scratch_db,
            repo,
            number,
            run_id=None,
            sha=None,
            threads=threads,
            bot_login=bot_login,
            source="backfill",
        )
        pending = ledger.reconcile(
            scratch_db,
            repo,
            number,
            pr_open=True,
            open_threads=open_,
            moved=moved,
            sha=head,
        )
    return {
        "source": "replayed",
        "pr_open": True,
        "resolution_as_of_now": as_of_now,
        "open_threads": len(open_),
        "standing": len(before),
        "outdated": sum(1 for h, m in moved.items() if m is True and h not in pending),
        "pending": len(pending),
    }


def recorded_reconcile(
    source: sqlite3.Connection, repo: str, number: int, cutoff: str
) -> dict[str, Any] | None:
    """The PR's v12 reconcile as the source database had it at `cutoff`, or None when it had
    not run by then: {run, sha, at, pending, exact}. When it ran is the record's `at`, else
    (an older record) its run's start, else its earliest `reconcile:` ledger event. The
    pending list at the cutoff is the one the reconcile left (`pending_at_reconcile`) less the
    hashes a thread answer judged between it and the cutoff (`exact`), or, for an older
    record, its pending list now (`exact` False: hashes judged after the cutoff are missing
    from it)."""
    if not _has_table(source, "kv"):
        return None
    rec = ledger.reconciled(source, repo, number)
    if rec is None:
        return None
    at = str(rec.get("at") or "")
    if not at and rec.get("run") is not None:
        row = source.execute("SELECT started_at FROM runs WHERE id=?", (rec["run"],)).fetchone()
        at = str(row["started_at"]) if row else ""
    if not at and _has_table(source, "issue_events"):
        row = source.execute(
            "SELECT MIN(ts) AS ts FROM issue_events WHERE repo=? AND number=? "
            "AND key LIKE 'reconcile:%'",
            (repo, number),
        ).fetchone()
        at = str(row["ts"] or "") if row else ""
    if not at or at >= cutoff:
        return None
    exact = isinstance(rec.get("pending_at_reconcile"), list)
    pending = [str(h) for h in rec.get("pending_at_reconcile" if exact else "pending") or []]
    if exact:
        judged = set()
        for e in _answered(source, repo, number, cutoff):
            if str(e["ts"]) < at:
                continue
            try:
                judged.add(str(json.loads(e["detail"] or "{}").get("finding_hash") or ""))
            except ValueError:
                continue
        pending = [h for h in pending if h not in judged]
    return {"run": rec.get("run"), "sha": rec.get("sha"), "at": at, "pending": pending,
            "exact": exact}  # fmt: skip


def closed_before(source: sqlite3.Connection, repo: str, number: int, cutoff: str) -> str:
    """Why the source database says the PR was closed before `cutoff` ("" = open, or not
    known to be closed): merged before it (its audit row), or closed with no change to its
    `prs` row since."""
    if _has_table(source, "audits"):
        row = source.execute(
            "SELECT merged_at FROM audits WHERE repo=? AND number=?", (repo, number)
        ).fetchone()
        if row is not None and row["merged_at"] and str(row["merged_at"]) < cutoff:
            return f"merged at {row['merged_at']}"
    pr = source.execute(
        "SELECT state, updated_at FROM prs WHERE repo=? AND number=?", (repo, number)
    ).fetchone()
    if pr is not None and pr["state"] != "open" and str(pr["updated_at"]) < cutoff:
        return f"{pr['state']} since {pr['updated_at']}"
    return ""


def reads_ledger(policy: cfg_mod.ModelPolicy) -> bool:
    """Whether a policy reads the issue ledger (the v12 pieces); under v11 it is write-only."""
    return policy.verifier_v12 or policy.unanchored or policy.thread_lane


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


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
) -> dict[str, Any]:
    """Copy what the worker reads about this PR's history, as of `cutoff`, into the scratch
    database: every run of the PR that finished before it (as its head, run, steps and
    `SEED_STAGES` findings rows, so per-commit lookups such as the standing verdict still
    work; under its source id, so a ledger event naming a run means the same run), its
    reviews and `thread.answered` events from before it, and `posted_findings`
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
        # when production started requiring tracking for deferrals: one recorded before it
        # keeps lifting its finding (converse.tracking_required_since; never recorded by a
        # replay, which would otherwise take its own start as the cutoff)
        since = kv_get(source, converse.TRACKING_CUTOFF_KEY) if _has_table(source, "kv") else None
        if since:
            kv_set(scratch, converse.TRACKING_CUTOFF_KEY, since)
        for r in runs:
            hid = _ensure_head(scratch, repo, number, str(r["sha"]), str(r["trigger"]))
            ids[int(r["id"])] = _insert(
                scratch,
                "runs",
                {
                    "id": int(r["id"]),
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
        # the events the worker reads about an earlier run of the PR (the v12 review strength
        # of the standing run: `worker._strength_row`), as of the cutoff
        if ids:
            kinds = ",".join("?" * len(RUN_EVENT_KINDS))
            for e in source.execute(
                f"SELECT * FROM events WHERE kind IN ({kinds}) AND run_id IN ({marks}) "
                "AND ts < ? ORDER BY id",
                (*RUN_EVENT_KINDS, *ids, cutoff),
            ).fetchall():
                _insert(
                    scratch,
                    "events",
                    {
                        "ts": e["ts"],
                        "kind": e["kind"],
                        "repo": repo,
                        "number": number,
                        "run_id": ids[int(e["run_id"])],
                        "detail": e["detail"],
                    },
                )
                counts["run_events"] += 1
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
    answer before it that reached the thread (`ledger.DELIVERED_ACTIONS`, as the ledger reads
    them: a silent, skipped, suppressed or failed answer put nothing there)."""
    out: dict[str, str] = {}
    for e in _answered(source, repo, number, cutoff):
        try:
            d = json.loads(e["detail"] or "{}")
        except ValueError:
            continue
        h, status = str(d.get("finding_hash") or ""), str(d.get("status") or "")
        if not h or d.get("action") not in ledger.DELIVERED_ACTIONS:
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
    run instead), without comparison lanes (a second model's findings feed the compare report,
    never the review: in an eval they are spend for nothing), and with `pin_phase1_model` that
    one model on every Phase-1 lane, ungated by quota, so both arms of an eval run Phase 1 on
    the same rung."""
    pol = dataclasses.replace(policy, degraded=None, comparison=None)
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
    policy: cfg_mod.ModelPolicy,
    source: sqlite3.Connection,
    prober: Prober | None,
    live: sqlite3.Connection | None = None,
) -> str:
    """Why degraded mode is active right now ("" = it is not): forced on in the source
    database or in the live one (the box's, when the source is a snapshot of it), or the
    policy's sentinel model answers with a quota error."""
    if any(degraded.forced(c) == "on" for c in (source, live) if c is not None):
        return "degraded mode is forced on (reviewsys degraded on)"
    if policy.degraded is None:
        return ""
    exhausted, reason = (prober or degraded.probe)(policy.degraded.sentinel)
    return f"the primary models are out of quota: {reason}" if exhausted else ""


# ---- the pinned view ----


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
    eval_mirrors: Path | None = None,
) -> dict[str, Any]:
    """Replay source run `run_id` into `out/run-<id>[-r<sample>]/` and return its report
    (also written there as report.json). Raises ReplayAborted in degraded mode. A repository
    the configured work dir has no mirror of is mirrored under `eval_mirrors` (default
    `out/.mirrors`), never in the production work dir."""
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
    live = None
    if (
        source_db is not None
        and cfg.db_path.exists()
        and cfg.db_path.resolve() != source_db.resolve()
    ):
        live = open_source(cfg.db_path)  # the source is a snapshot: degraded mode is set here
    try:
        src = _source_run(prod, run_id)
        problem = degraded_problem(policy, prod, prober, live)
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
            eval_mirrors=eval_mirrors,
        )
    finally:
        prod.close()
        if live is not None:
            live.close()


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
    eval_mirrors: Path | None = None,
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
    (work / "work").mkdir(parents=True)
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
        # the gpt pool is sized from the configured (production) database's account reading
        lane_capacity_db=arm.db_path,
    )
    rc = policy_overrides(rc, policy_name)
    token_env = {"GH_TOKEN": gh_token or "", "GITHUB_TOKEN": gh_token or ""}
    inner = gh_runner or (env_runner(token_env) if gh_token else _default_runner)
    scratch = connect(rc.db_path)
    try:
        seeded = seed_history(
            scratch, prod, repo=repo, number=number, sha=sha, cutoff=cutoff, exclude_run=run_id
        )
        evidence = source_evidence(source_runs, run_id)
        ev_pr = (evidence.get("evidence") or {}).get("pr") or {}
        ci = (evidence.get("evidence") or {}).get("ci")
        base_ref = str(ev_pr.get("baseRefName") or "")
        if not base_ref:  # no evidence on disk: the live base ref (refs are rarely renamed)
            live = Gh(installed_gh, runner=read_only_runner(inner)).api(
                f"repos/{repo}/pulls/{number}"
            )
            base_ref = str(((live or {}).get("base") or {}).get("ref") or "")
        # the replay's own repository: the production mirror's objects borrowed read-only,
        # anything missing fetched into it, only the pinned refs (replaygit.py)
        pinned, base_sha, merge_base = replaygit.pinned_clone(
            arm.mirrors_dir,
            eval_mirrors or out / ".mirrors",
            rc.mirrors_dir / replaygit.mirror_name(repo),
            repo=repo,
            number=number,
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
            git_dir=str(pinned),
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
        # lanepool pushes the deadline out by every stretch the run only waited for slots:
        # the difference at the end is the run's slot wait
        deadline0 = now()
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
                    "deadline_at": deadline0,
                },
            )
        gh = Gh(installed_gh, runner=read_only_runner(pinned_runner(pin, inner)))
        if reads_ledger(rc.policy):
            seeded["ledger"] = seed_ledger(scratch, prod, repo, number, cutoff)
            seeded["reconcile"] = reconcile_as_of(
                scratch, prod, pin=pin, gh=gh, bot_login=rc.bot_login, git_dir=pinned
            )
        bin_dir = work / "bin"
        with Broker(pin, real_gh(pin, inner)) as broker:
            shim = install_shim(broker.path, bin_dir, curl_log=work / "curl-refused.log")
            problem = check_shim(shim, exec_deny_profile(rc.lane_deny_exec))
            if problem:
                raise SystemExit(f"the replay's gh shim cannot serve a lane: {problem}")
            gh_config = work / "gh-config"
            gh_config.mkdir(exist_ok=True)
            env_file = bin_dir / "env.sh"
            env_file.write_text(f'export PATH="{bin_dir}:$PATH"\n')
            env = {
                "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
                # sourced by Claude Code before each Bash command: keeps the shims first even
                # if the lane's shell setup rewrites PATH
                "CLAUDE_ENV_FILE": str(env_file),
                # the token stays with the broker; no gh in a lane can find credentials
                "GH_CONFIG_DIR": str(gh_config),
                **dict.fromkeys(TOKEN_VARS, ""),
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
                replay_context={
                    k: v
                    for k, v in (evidence.get("evidence") or {}).items()
                    if k in ("sister_repos", "base_drift")
                },
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
            deadline0=deadline0,
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
        # Rust repo's can reach tens of GB) and the lane artifacts are what the report needs.
        # The pinned clone goes with it (with any blobs fetched into it)
        shutil.rmtree(rc.worktrees_dir, ignore_errors=True)
        shutil.rmtree(rc.mirrors_dir, ignore_errors=True)
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


def v12_signals(scratch: sqlite3.Connection, run_id: int, run_dir: Path) -> dict[str, Any]:
    """What a v12 round records for the eval (all zero under v11): concessions a finder
    contradicted (`threads.conflict`), thread-lane shards that failed (`threads.failed`),
    late non-blocking findings (`v12.late_nonblocking`), new issues dropped as settled
    re-folds (`v12.settled_refold`), and the share of verify+match candidates matched to an
    open ledger issue (every verify pass pooled, from its `<phase>.json`)."""
    counts: Counter[str] = Counter()
    late = 0
    for e in scratch.execute(
        "SELECT kind, detail FROM events WHERE run_id=? AND kind IN "
        "('threads.conflict', 'threads.failed', 'v12.late_nonblocking', 'v12.settled_refold')",
        (run_id,),
    ):
        counts[str(e["kind"])] += 1
        if e["kind"] == "v12.late_nonblocking":
            m = re.match(r"(\d+)", str(e["detail"] or ""))
            late += int(m.group(1)) if m else 0
    candidates = matched = 0
    for f in sorted(run_dir.glob("*.json")) if run_dir.is_dir() else []:
        data = read_json(f)
        if not isinstance(data, dict) or "candidates" not in data or "corroborated" not in data:
            continue
        candidates += len(data.get("candidates") or [])
        matched += sum(len(v or []) for v in (data.get("corroborated") or {}).values())
    return {
        "threads_conflict": counts["threads.conflict"],
        "threads_failed": counts["threads.failed"],
        "late_nonblocking": late,
        "settled_refold": counts["v12.settled_refold"],
        "candidates": candidates,
        "candidates_matched_open": matched,
        "match_share": round(matched / candidates, 4) if candidates else None,
    }


def _prior_open_threads(
    scratch: sqlite3.Connection, rc: cfg_mod.Config, run_id: int, run_dir: Path, pin: Pin
) -> list[dict[str, Any]]:
    """The bot's finding threads open when the run started: under v12 the ledger's standing
    issues with a thread (`ledger.json`), else the posted findings seeded as of the cutoff
    less those a thread answer lifted."""
    keys = ("file", "line_start", "line_end", "title")
    led = read_json(run_dir / "ledger.json")
    if isinstance(led, dict) and isinstance(led.get("issues"), dict):
        return [
            {"hash": h, **{k: i.get(k) for k in keys}}
            for h, i in led["issues"].items()
            if i.get("status") in ledger.STANDING and (i.get("comment_id") or i.get("thread_id"))
        ]
    rows = scratch.execute(
        "SELECT pf.hash, f.file, f.line_start, f.line_end, f.title FROM posted_findings pf "
        "JOIN findings f ON f.hash=pf.hash AND f.stage='posted' AND f.run_id<>? "
        "WHERE pf.repo=? AND pf.number=? GROUP BY pf.hash",
        (run_id, pin.repo, pin.number),
    ).fetchall()
    return [
        {"hash": str(r["hash"]), **{k: r[k] for k in keys}}
        for r in rows
        if str(r["hash"]) not in pin.lifted
    ]


def reconcile_work(threads: Any, lanes: list[dict[str, Any]], *, counted: bool) -> dict[str, Any]:
    """The thread lane's work that only the reconcile asked for (issues whose every reason is
    `reconcile`), from `threads.json`: how many, the shards made of nothing else, and each
    shard's lanes and tokens in proportion to its reconcile-only issues (all, and on gpt).
    Zero unless `counted` (the replay ran the reconcile itself)."""
    out: dict[str, Any] = {
        "counted": counted,
        "issues": 0,
        "shards": 0,
        "lanes": 0.0,
        "gpt_lanes": 0.0,
        "tokens_in": 0,
        "tokens_out": 0,
        "gpt_tokens_in": 0,
        "gpt_tokens_out": 0,
    }
    if not counted or not isinstance(threads, dict):
        return out
    work = threads.get("work") or {}
    shards = [list(s) for s in threads.get("shards") or []]
    only = {h for h, why in work.items() if set(why or []) == {"reconcile"}}
    out["issues"] = len(only)
    for n, shard in enumerate(shards):
        if not shard:
            continue
        share = len(only.intersection(shard)) / len(shard)
        if share == 1:
            out["shards"] += 1
        role = "thread" if len(shards) == 1 else f"thread-{n + 1}of{len(shards)}"
        for x in lanes:
            if x.get("role") != role:
                continue
            gpt = pool_of(str(x.get("model") or "")) == "gpt"
            tin = round(int(x.get("tokens_in") or 0) * share)
            tout = round(int(x.get("tokens_out") or 0) * share)
            out["lanes"] += share
            out["tokens_in"] += tin
            out["tokens_out"] += tout
            if gpt:
                out["gpt_lanes"] += share
                out["gpt_tokens_in"] += tin
                out["gpt_tokens_out"] += tout
    out["lanes"], out["gpt_lanes"] = round(out["lanes"], 3), round(out["gpt_lanes"], 3)
    return out


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
    deadline0: str | None = None,
    pin: Pin,
    seeded: dict[str, Any],
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
    signals = v12_signals(scratch, new_run, run_dir)
    degraded_for_scoring = (
        "a thread-lane shard failed (threads.failed): some issues were never judged"
        if signals["threads_failed"]
        else None
    )
    replayed = (seeded.get("reconcile") or {}).get("source") == "replayed"
    rw = reconcile_work(read_json(run_dir / "threads.json"), lanes, counted=replayed)
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
    slot_wait = 0.0
    if deadline0 and run["deadline_at"]:
        slot_wait = max(0.0, (parse_ts(run["deadline_at"]) - parse_ts(deadline0)).total_seconds())
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
        # the time the run only waited for lane slots (some lane waiting, none running), and
        # the rest: wall-clock gates read `active_s`
        "slot_wait_s": slot_wait,
        "active_s": round(max(0.0, wall_clock_s - slot_wait), 1),
        "gate": {
            "decision": (gate or {}).get("detail"),
            "forced_phase2": bool(((gate or {}).get("detail") or {}).get("forced_phase2")),
            "would_publish": read_json(run_dir / "gate-would-publish.json"),
            **({"would_publish_note": V12_GATE_NOTE} if rc.policy.pipeline is not None else {}),
        },
        "steps": steps,
        "lanes": lanes,
        "gpt": {
            "lanes": len(gpt),
            "tokens_in": sum(int(x["tokens_in"] or 0) for x in gpt),
            "tokens_out": sum(int(x["tokens_out"] or 0) for x in gpt),
        },
        # the thread-lane work only the replay's own one-shot reconcile adds (production did
        # it once, at some earlier round): counted apart so cost gates can leave it out
        "reconcile_work": rw,
        "gpt_excl_reconcile": {
            "lanes": round(len(gpt) - rw["gpt_lanes"], 3),
            "tokens_in": sum(int(x["tokens_in"] or 0) for x in gpt) - rw["gpt_tokens_in"],
            "tokens_out": sum(int(x["tokens_out"] or 0) for x in gpt) - rw["gpt_tokens_out"],
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
        # the inline comments (new threads) the last publication would open
        "inline": (last or {}).get("inline") or [],
        # bot finding threads open at the run's start: an inline comment on one of these
        # issues is a duplicate thread (the S5 gate is zero)
        "prior_open_threads": _prior_open_threads(scratch, rc, new_run, run_dir, pin),
        "v12": signals,
        "degraded_for_scoring": degraded_for_scoring,
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
        # threads whose resolved state is as of now, not the cutoff (the view cannot rewind
        # it): the head's thread facts are not exact (replay-score sets such heads apart)
        "thread_resolution_as_of_now": int(rewind.get("thread_resolved_as_of_now", 0)),
        "live_github": {"lane_output_hits": hits, "curl_refused": curl_refused[:50]},
        "run_dir": str(run_dir),
        "caveats": list(CAVEATS),
    }


def _json_text(text: str | None) -> Any:
    try:
        return json.loads(text or "null")
    except ValueError:
        return text
