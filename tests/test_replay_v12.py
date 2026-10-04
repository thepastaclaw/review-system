"""A v12 replay decides what production would: the standing run's strength (its events), the
reconcile production already ran, and what the report exports for the v12 gates."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

from reviewsys import ledger, replay, worker
from reviewsys.db import connect, now, tx
from reviewsys.models import Trigger
from tests.test_v12_confirmation import _withdrawn_blocker
from tests.test_worker_e2e import (
    fake_git,  # noqa: F401  (autouse: fake mirror/worktree)
)
from tests.v12_support import REPO, A, run_round, v12_cfg

pytestmark = pytest.mark.usefixtures("wide_diff")

LATER = "2999-01-01T00:00:00Z"  # after everything the source database holds


def _replay_in_scratch(cfg, conn, gh, lanes, tmp_path: Path, exclude: int) -> dict:
    """What the replay harness does before the worker, by hand: a scratch database seeded
    from `conn` (history, ledger, the reconcile production recorded), then the same reply
    as a replay there. Returns its conversation.json."""
    scfg = dataclasses.replace(cfg, db_path=tmp_path / "scratch.db", work_dir=tmp_path / "sw")
    scratch = connect(scfg.db_path)
    replay.seed_history(
        scratch, conn, repo=REPO, number=1, sha=A, cutoff=LATER, exclude_run=exclude
    )
    replay.seed_ledger(scratch, conn, REPO, 1, LATER)
    rec = replay.recorded_reconcile(conn, REPO, 1, LATER)
    assert rec is not None  # production reconciled at its first v12 round
    with tx(scratch):
        ledger.record_reconciled(
            scratch, REPO, 1, {k: rec[k] for k in ("run", "sha", "pending", "at")}
        )
        hid = replay._ensure_head(scratch, REPO, 1, A, Trigger.REVIEW_REPLY.value)
        scratch.execute(
            "UPDATE heads SET status='running', trigger=?, queue='live' WHERE id=?",
            (Trigger.REVIEW_REPLY.value, hid),
        )
        rid = scratch.execute(
            "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at) "
            "VALUES (?, 1, 'spawned', 'replay', ?, ?)",
            (hid, now(), now()),
        ).lastrowid
    worker.main(
        scfg, scratch, int(rid or 0), gh=gh, lane_runner=lanes, dry_run=True, replay=True,
        heartbeat=False,
    )  # fmt: skip
    return json.loads((scfg.runs_dir / f"run-{rid}" / "conversation.json").read_text())


@pytest.mark.parametrize("kind", [None, "verify.shard_dropped", "threads.conflict"])
def test_a_replayed_reply_decides_the_follow_up_production_would(
    cfg, skills_dir, conn, gh, lanes, git, tmp_path, kind
):
    """M2: the standing review's strength reads its run's events. With a dropped verifier
    shard or a conflict on record the #38 follow-up must not approve, in production and in a
    replay alike: the replay seeds those events."""
    v12 = v12_cfg(cfg, skills_dir, confirmation={"trigger": "off"})
    _withdrawn_blocker(v12, conn, gh, lanes)
    standing = conn.execute("SELECT MAX(id) FROM runs").fetchone()[0]
    if kind:
        with tx(conn):
            conn.execute(
                "INSERT INTO events (ts, kind, repo, number, run_id, detail) VALUES (?,?,?,?,?,?)",
                (now(), kind, REPO, 1, standing, "{}"),
            )
    rid = run_round(v12, conn, gh, lanes, A, Trigger.REVIEW_REPLY, dry_run=True, replay=True)
    production = json.loads((v12.runs_dir / f"run-{rid}" / "conversation.json").read_text())
    replayed = _replay_in_scratch(v12, conn, gh, lanes, tmp_path, exclude=rid)
    assert replayed["would_post"] == production["would_post"]
    assert (production["would_post"]["event"] == "APPROVE") is (kind is None)


def test_the_seeded_run_events_are_the_ones_the_worker_reads():
    """Every event kind `worker._strength_row` reads about an earlier run is seeded."""
    import inspect

    src = inspect.getsource(worker._strength_row)
    read = set(re.findall(r"'([a-z0-9_]+\.[a-z0-9_]+)'", src))
    assert read and read <= set(replay.RUN_EVENT_KINDS)
