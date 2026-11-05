"""The issue ledger (ledger.py): schema v9, identity, transitions, folding, backfill, git."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from reviewsys import db as db_mod
from reviewsys import ledger
from reviewsys.contract import Finding, finding_hash
from reviewsys.db import connect, tx

REPO, PR = "dashpay/platform", 7
BOT = "thepastaclaw"


def _f(title="Fee estimation omits drainage costs", sev="blocking", file="f.rs", line=11):
    return Finding(
        file=file,
        title=title,
        body="Estimation must be >= actual.",
        severity=sev,
        category="logic",
        line_start=line,
        line_end=line + 1,
    )


def _pub(conn, run_id, sha, posted=(), kept=(), **kw):
    with tx(conn):
        ledger.record_published(
            conn, REPO, PR, run_id=run_id, sha=sha, posted=list(posted), kept=list(kept), **kw
        )


def _answer(conn, run_id, sha, h, status, action="replied", source="verifier", conceded=()):
    item = {"finding_hash": h, "status": status, "comment_id": 900, "action": action}
    with tx(conn):
        ledger.record_answers(
            conn,
            REPO,
            PR,
            run_id=run_id,
            sha=sha,
            answered=[item],
            source=source,
            conceded=conceded,
        )


# ---- schema ----


def test_v9_migration_is_additive_on_a_v8_database(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    monkeypatch.setattr(db_mod, "MIGRATIONS", {k: v for k, v in db_mod.MIGRATIONS.items() if k < 9})
    old = connect(path)
    with tx(old):
        hid = old.execute(
            "INSERT INTO heads (repo, number, sha, trigger, queued_at, eligible_at) "
            "VALUES (?,?,?,?,?,?)",
            (REPO, PR, "a" * 40, "new_pr", "2026-09-01", "2026-09-01"),
        ).lastrowid
        rid = old.execute(
            "INSERT INTO runs (head_id, attempt, token, started_at, deadline_at) "
            "VALUES (?,1,'t','2026-09-01','2026-09-01')",
            (hid,),
        ).lastrowid
        old.execute(
            "INSERT INTO findings (run_id, phase, stage, hash, severity, title, body) "
            "VALUES (?,?,?,?,?,?,?)",
            (rid, "final", "posted", "abc", "blocking", "T", "B"),
        )
    assert old.execute("SELECT version FROM schema_version").fetchone()[0] == 8
    old.close()
    monkeypatch.undo()
    new = connect(path)
    assert new.execute("SELECT version FROM schema_version").fetchone()[0] == 9
    assert db_mod.SCHEMA_VERSION == 9
    row = new.execute("SELECT hash, title, candidate_id FROM findings").fetchone()
    assert tuple(row) == ("abc", "T", None)
    assert new.execute("SELECT count(*) FROM issues").fetchone()[0] == 0
    assert new.execute("SELECT count(*) FROM issue_events").fetchone()[0] == 0
    # v11's own insert (no candidate_id) still works, and a second open migrates nothing
    with tx(new):
        new.execute(
            "INSERT INTO findings (run_id, phase, stage, hash, severity, title, body) "
            "VALUES (?,?,?,?,?,?,?)",
            (rid, "final", "posted", "def", "nitpick", "T2", "B"),
        )
    new.close()
    again = connect(path)
    assert again.execute("SELECT count(*) FROM findings").fetchone()[0] == 2
    assert [tuple(r) for r in again.execute("SELECT version FROM schema_version")] == [(9,)]


def test_a_database_the_unmerged_v12_branch_migrated_still_opens(tmp_path, monkeypatch):
    """Before the merge with main, the ledger was migration 6 (main's 6 is an index): a copy
    that code migrated (an eval snapshot) has the ledger and `findings.candidate_id` already,
    and must still reach the current version instead of failing on the column."""
    path = tmp_path / "v12-v6.db"
    old_v6 = db_mod.MIGRATIONS[9] + "CREATE INDEX IF NOT EXISTS events_run ON events (run_id);"
    early = {k: v for k, v in db_mod.MIGRATIONS.items() if k < 6}
    monkeypatch.setattr(db_mod, "MIGRATIONS", {**early, 6: old_v6})
    connect(path).close()
    monkeypatch.undo()
    new = connect(path)
    assert new.execute("SELECT version FROM schema_version").fetchone()[0] == db_mod.SCHEMA_VERSION
    cols = {r[1] for r in new.execute("PRAGMA table_info(findings)")}
    assert "candidate_id" in cols
    assert {r[1] for r in new.execute("PRAGMA table_info(heads)")} >= {"diff_print", "carried_from"}
    assert {r[1] for r in new.execute("PRAGMA table_info(prs)")} >= {"skip_label"}
    new.close()


# ---- identity ----


def test_issue_hash_is_byte_compatible_with_finding_hash(conn):
    f = _f()
    # the marker hash GitHub carries: sha256('{file}:{category}:{title lowercased}')[:12]
    assert f.hash == finding_hash("f.rs", "logic", "Fee estimation omits drainage costs")
    raw = b"f.rs:logic:fee estimation omits drainage costs"
    assert f.hash == hashlib.sha256(raw).hexdigest()[:12] == "d6dc000eee17"
    _pub(conn, 1, "a" * 40, posted=[f])
    (issue,) = ledger.load(conn, REPO, PR).values()
    assert issue.hash == f.hash and issue.title == f.title
    # a verifier that retitles the carried finding (echoing its hash) never renames the issue
    retitled = _f(title="Fee estimate is lower than the real fee")
    retitled.prior_hash = f.hash
    _pub(conn, 2, "b" * 40, kept=[retitled])
    (issue,) = ledger.load(conn, REPO, PR).values()
    assert issue.hash == f.hash and issue.title == f.title
    assert issue.last_seen_run == 2 and issue.last_seen_sha == "b" * 40


# ---- transitions ----


def test_publication_opens_and_carries(conn):
    blocker, sugg = _f(), _f("Add a test", "suggestion", line=40)
    _pub(conn, 1, "a" * 40, posted=[blocker, sugg])
    issues = ledger.load(conn, REPO, PR)
    assert {h: i.status for h, i in issues.items()} == {blocker.hash: "open", sugg.hash: "open"}
    i = issues[blocker.hash]
    assert (i.opened_run, i.opened_sha, i.anchor_sha, i.status_source) == (
        1,
        "a" * 40,
        "a" * 40,
        "verifier",
    )
    assert (i.line_start, i.line_end, i.severity) == (11, 12, "blocking")
    # round 2 posts nothing new for the blocker but keeps it: carried, not re-opened
    _pub(conn, 2, "b" * 40, kept=[blocker])
    i = ledger.get(conn, REPO, PR, blocker.hash)
    assert i is not None and (i.opened_run, i.last_seen_run, i.status) == (1, 2, "open")
    assert ledger.get(conn, REPO, PR, sugg.hash).last_seen_run == 1  # type: ignore[union-attr]


def test_a_deduped_finding_carries_the_issue_its_comment_belongs_to(conn):
    f = _f()
    _pub(conn, 1, "a" * 40, posted=[f])
    with tx(conn):
        ledger.sync_threads(
            conn, REPO, PR, run_id=1, sha="a" * 40, threads=[_thread(f, 900)], bot_login=BOT
        )
    other = _f(title="Something reworded entirely")
    _pub(conn, 2, "b" * 40, kept=[other], deduped_comment_ids=["900"])
    i = ledger.get(conn, REPO, PR, f.hash)
    assert i is not None and i.last_seen_run == 2
    assert ledger.get(conn, REPO, PR, other.hash) is None  # never posted: no issue


@pytest.mark.parametrize(
    ("outcome", "status"),
    [
        ("FIXED", "fixed"),
        ("OUTDATED", "outdated"),
        ("WITHDRAWN", "withdrawn"),
        ("INTENTIONALLY_DEFERRED", "deferred"),
        ("DEFERRAL_PENDING", "deferral_pending"),
        ("STILL_VALID", "open"),
        ("FIX_PENDING", "open"),
    ],
)
def test_thread_outcomes_move_the_issue(conn, outcome, status):
    f = _f()
    _pub(conn, 1, "a" * 40, posted=[f])
    _answer(conn, 2, "b" * 40, f.hash, outcome)
    i = ledger.get(conn, REPO, PR, f.hash)
    assert i is not None and i.status == status
    if status in ledger.CLOSED:
        assert (i.closed_sha, i.status_sha, i.closed_reason) == ("b" * 40, "b" * 40, outcome)
    else:
        assert i.closed_sha is None
    (ev,) = [e for e in ledger.history(conn, REPO, PR, f.hash) if e["kind"] == "outcome"]
    assert (ev["from_status"], ev["to_status"], ev["detail"]["outcome"]) == (
        "open",
        status,
        outcome,
    )


@pytest.mark.parametrize("action", ["reply_failed: boom", "suppressed_reply", "no_reply"])
def test_an_outcome_that_never_reached_the_thread_changes_nothing(conn, action):
    f = _f()
    _pub(conn, 1, "a" * 40, posted=[f])
    _answer(conn, 2, "b" * 40, f.hash, "WITHDRAWN", action=action)
    assert ledger.get(conn, REPO, PR, f.hash).status == "open"  # type: ignore[union-attr]
    assert len(ledger.history(conn, REPO, PR, f.hash)) == 2  # still on record


def test_conversation_concession_then_reraise_then_fix(conn):
    f = _f()
    _pub(conn, 1, "a" * 40, posted=[f])
    _answer(conn, 2, "a" * 40, f.hash, "WITHDRAWN", source="conversation", conceded={f.hash})
    i = ledger.get(conn, REPO, PR, f.hash)
    assert i is not None and (i.status, i.status_source) == ("withdrawn", "conversation")
    ev = ledger.history(conn, REPO, PR, f.hash)[-1]
    assert ev["detail"]["conceded"] is True
    # a later round posts the same finding again: it stands again, anchored on the new head
    moved = _f(line=30)
    _pub(conn, 3, "c" * 40, posted=[moved])
    i = ledger.get(conn, REPO, PR, f.hash)
    assert i is not None and (i.status, i.closed_sha, i.anchor_sha, i.line_start) == (
        "open",
        None,
        "c" * 40,
        30,
    )
    assert (i.opened_run, i.last_seen_run) == (1, 3)
    _answer(conn, 4, "d" * 40, f.hash, "FIXED")
    assert ledger.get(conn, REPO, PR, f.hash).status == "fixed"  # type: ignore[union-attr]
    statuses = [(e["kind"], e["to_status"]) for e in ledger.history(conn, REPO, PR, f.hash)]
    assert statuses == [
        ("posted", "open"),
        ("outcome", "withdrawn"),
        ("posted", "open"),
        ("outcome", "fixed"),
    ]


def test_explicit_transition_is_validated_and_idempotent(conn):
    f = _f()
    _pub(conn, 1, "a" * 40, posted=[f])
    with tx(conn):
        i = ledger.transition(
            conn, REPO, PR, f.hash, "deferred", run_id=5, sha="e" * 40, source="manual", reason="x"
        )
        assert i is not None and (i.status, i.closed_reason, i.status_source) == (
            "deferred",
            "x",
            "manual",
        )
        ledger.transition(
            conn, REPO, PR, f.hash, "deferred", run_id=5, sha="e" * 40, source="manual"
        )
        assert (
            ledger.transition(conn, REPO, PR, "nope", "fixed", run_id=5, sha=None, source="manual")
            is None
        )
        with pytest.raises(ValueError):
            ledger.transition(conn, REPO, PR, f.hash, "gone", run_id=5, sha=None, source="manual")
        with pytest.raises(ValueError):
            ledger.transition(conn, REPO, PR, f.hash, "open", run_id=6, sha=None, source="model")
    assert len(ledger.history(conn, REPO, PR, f.hash)) == 2


def test_manual_transitions_without_a_run_are_each_recorded(conn):
    f = _f()
    _pub(conn, 1, "a" * 40, posted=[f])
    for status in ("withdrawn", "open", "withdrawn"):
        with tx(conn):
            ledger.transition(
                conn, REPO, PR, f.hash, status, run_id=None, sha=None, source="manual"
            )
    kinds = [(e["kind"], e["to_status"]) for e in ledger.history(conn, REPO, PR, f.hash)]
    assert kinds == [
        ("posted", "open"),
        ("status", "withdrawn"),
        ("status", "open"),
        ("status", "withdrawn"),
    ]


def test_unknown_outcomes_are_ignored(conn):
    _answer(conn, 2, "b" * 40, "f" * 12, "WITHDRAWN")
    assert ledger.load(conn, REPO, PR) == {}
    assert conn.execute("SELECT count(*) FROM issue_events").fetchone()[0] == 0


# ---- threads ----


def _thread(f, root_id, *, resolved=False, replies=()):
    return {
        "thread_id": f"PRRT_{root_id}",
        "is_resolved": resolved,
        "is_outdated": False,
        "path": f.file,
        "line": f.line_start,
        "comments": [
            {
                "id": root_id,
                "author": BOT,
                "body": f"<!-- thepastaclaw-review v1 finding={f.hash} dedupe=x -->\n"
                f"**🔴 Blocking: {f.title}**\n\n{f.body}",
                "created_at": "2026-09-07T00:00:00Z",
                "association": "NONE",
            },
            *replies,
        ],
    }


def test_threads_add_legacy_issues_and_track_hand_resolution(conn):
    known, legacy = _f(), _f("Legacy finding nobody recorded", line=50)
    _pub(conn, 1, "a" * 40, posted=[known])

    def sync(run_id, threads):
        with tx(conn):
            return ledger.sync_threads(
                conn, REPO, PR, run_id=run_id, sha="b" * 40, threads=threads, bot_login=BOT
            )

    assert sync(2, [_thread(known, 900), _thread(legacy, 901)]) == 2
    issues = ledger.load(conn, REPO, PR)
    leg = issues[legacy.hash]
    assert (leg.status, leg.opened_run, leg.severity, leg.title, leg.line_start) == (
        "open",
        None,
        "blocking",
        legacy.title,
        50,
    )
    assert issues[known.hash].thread_id == "PRRT_900" and issues[known.hash].comment_id == 900
    assert sync(3, [_thread(known, 900), _thread(legacy, 901)]) == 0  # nothing new
    # a maintainer resolves the thread by hand: still counts under policy, recorded as such
    sync(4, [_thread(known, 900, resolved=True), _thread(legacy, 901)])
    i = ledger.get(conn, REPO, PR, known.hash)
    assert i is not None and (i.status, i.status_source) == ("resolved", "manual")
    # ... and unresolves it
    sync(5, [_thread(known, 900), _thread(legacy, 901)])
    assert ledger.get(conn, REPO, PR, known.hash).status == "open"  # type: ignore[union-attr]
    # a fixed issue whose thread is resolved stays fixed
    _answer(conn, 6, "c" * 40, legacy.hash, "FIXED")
    sync(7, [_thread(legacy, 901, resolved=True)])
    assert ledger.get(conn, REPO, PR, legacy.hash).status == "fixed"  # type: ignore[union-attr]


# ---- safety ----


def test_a_failing_write_is_recorded_never_raised(conn, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("ledger bug")

    monkeypatch.setattr(ledger, "_refold", boom)
    ok = ledger.safe_write(
        conn,
        "publication",
        lambda c: ledger.record_published(c, REPO, PR, run_id=1, sha="a" * 40, posted=[_f()]),
        repo=REPO,
        number=PR,
        run_id=1,
    )
    assert ok is False
    # the half-done write rolled back with its transaction
    assert conn.execute("SELECT count(*) FROM issue_events").fetchone()[0] == 0
    ev = conn.execute("SELECT * FROM events WHERE kind='ledger.write_failed'").fetchone()
    assert ev["repo"] == REPO and ev["run_id"] == 1
    assert ev["detail"] == "publication: RuntimeError: ledger bug"
    assert not conn.in_transaction


def test_safe_write_inside_a_transaction_writes_nothing(conn):
    calls = []
    with tx(conn):
        ok = ledger.safe_write(conn, "x", lambda c: calls.append(c), repo=REPO, number=PR, run_id=1)
        assert conn.in_transaction  # the caller's transaction is left alone
    assert ok is False and calls == []


# ---- backfill ----


def _head(conn, sha, *, queue="live", number=PR):
    return conn.execute(
        "INSERT INTO heads (repo, number, sha, trigger, status, queued_at, eligible_at, queue) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (REPO, number, sha, "new_push", "done", "2026-09-01", "2026-09-01", queue),
    ).lastrowid


def _run(conn, hid, started, status="done"):
    return conn.execute(
        "INSERT INTO runs (head_id, attempt, status, token, started_at, finished_at, deadline_at) "
        "VALUES (?,1,?,'t',?,?,?)",
        (hid, status, started, started, started),
    ).lastrowid


def _rows(conn, rid, stage, findings, phase="final"):
    conn.executemany(
        "INSERT INTO findings (run_id, phase, stage, hash, file, line_start, line_end, severity, "
        "category, title, body) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                rid,
                phase,
                stage,
                f.hash,
                f.file,
                f.line_start,
                f.line_end,
                f.severity,
                f.category,
                f.title,
                f.body,
            )
            for f in findings
        ],
    )


def _review(conn, rid, sha, ts, event="REQUEST_CHANGES", number=PR):
    conn.execute(
        "INSERT INTO reviews (run_id, repo, number, sha, phase, event, posted_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (rid, REPO, number, sha, "final", event, ts),
    )


def _answered(conn, rid, ts, h, status, action="replied", **extra):
    conn.execute(
        "INSERT INTO events (ts, kind, repo, number, run_id, detail) VALUES (?,?,?,?,?,?)",
        (
            ts,
            "thread.answered",
            REPO,
            PR,
            rid,
            json.dumps({"finding_hash": h, "status": status, "action": action, **extra}),
        ),
    )


A, B, C = "a" * 40, "b" * 40, "c" * 40
BLOCK, SUGG, GONE, CONC = (
    _f("Fee estimation omits drainage costs"),
    _f("Add a regression test", "suggestion", line=40),
    _f("Overflow in the fee sum", line=70),
    _f("Lock order inverts on shutdown", file="g.rs", line=5),
)


def _history(conn):
    """Three rounds of one PR plus an audit and another PR's noise, as v11 records them."""
    with tx(conn):
        r1 = _run(conn, _head(conn, A), "2026-09-01T00:00:00Z")
        _rows(conn, r1, "verified", [BLOCK, SUGG, GONE], phase="verify2")
        _rows(conn, r1, "posted", [BLOCK, SUGG, GONE])
        _review(conn, r1, A, "2026-09-01T00:30:00Z")
        for f in (BLOCK, SUGG, GONE):
            conn.execute(
                "INSERT INTO posted_findings (repo, number, hash, sha, posted_at) VALUES (?,?,?,?,?)",
                (REPO, PR, f.hash, A, "2026-09-01T00:30:00Z"),
            )
        # round 2: BLOCK kept (carried), SUGG fixed, CONC newly posted, GONE vanished
        r2 = _run(conn, _head(conn, B), "2026-09-02T00:00:00Z")
        _rows(conn, r2, "verified", [BLOCK, CONC], phase="verify2")
        _rows(conn, r2, "posted", [CONC])
        _review(conn, r2, B, "2026-09-02T00:30:00Z")
        _answered(conn, r2, "2026-09-02T00:31:00Z", SUGG.hash, "FIXED")
        _answered(conn, r2, "2026-09-02T00:31:00Z", BLOCK.hash, "STILL_VALID", "already_answered")
        _answered(conn, r2, "2026-09-02T00:31:00Z", "0" * 12, "FIXED")  # unknown to the DB
        _answered(conn, r2, "2026-09-02T00:31:00Z", GONE.hash, "DUPLICATE_THREAD")
        # same-sha conversation: CONC conceded, but the answer event never got written
        r3 = _run(
            conn,
            conn.execute("SELECT head_id FROM runs WHERE id=?", (r2,)).fetchone()[0],
            "2026-09-02T05:00:00Z",
        )
        _rows(conn, r3, "conceded", [CONC])
        # round 3: BLOCK withdrawn after a reply; a failed run's rows count for nothing
        r4 = _run(conn, _head(conn, C), "2026-09-03T00:00:00Z")
        _rows(conn, r4, "verified", [], phase="verify2")
        _review(conn, r4, C, "2026-09-03T00:30:00Z", "APPROVE")
        _answered(conn, r4, "2026-09-03T00:31:00Z", BLOCK.hash, "WITHDRAWN", mode="conversation")
        r5 = _run(conn, _head(conn, "d" * 40), "2026-09-04T00:00:00Z", status="failed")
        _rows(conn, r5, "verified", [GONE], phase="verify2")
        # a post-merge audit is not part of the PR's review history
        ra = _run(conn, _head(conn, "e" * 40, queue="audit"), "2026-09-05T00:00:00Z")
        _rows(conn, ra, "posted", [_f("Audit-only finding", line=99)])
    return r1, r2, r3, r4


def _table(conn):
    return {
        h: (i.status, i.status_source, i.opened_run, i.last_seen_run, i.closed_sha)
        for h, i in ledger.load(conn, REPO, PR).items()
    }


GOLDEN = {
    BLOCK.hash: ("withdrawn", "backfill", 1, 2, C),
    SUGG.hash: ("fixed", "backfill", 1, 1, B),
    GONE.hash: ("open", "backfill", 1, 1, None),
    CONC.hash: ("withdrawn", "backfill", 2, 2, B),
}


def test_backfill_golden_and_idempotent(conn):
    _history(conn)
    res = ledger.backfill(conn)
    assert _table(conn) == GOLDEN
    assert {k: res[k] for k in ("prs", "posted", "seen", "answers", "conceded_only")} == {
        "prs": 1,
        "posted": 4,
        "seen": 1,
        "answers": 3,
        "conceded_only": 1,
    }
    assert res["answers_unknown_issue"] == 1 and res["issues"] == 4
    assert res["by_status"] == {"withdrawn": 2, "fixed": 1, "open": 1}
    blk = ledger.history(conn, REPO, PR, BLOCK.hash)
    assert [(e["kind"], e["to_status"]) for e in blk] == [
        ("posted", "open"),
        ("seen", "open"),
        ("outcome", "open"),
        ("outcome", "withdrawn"),
    ]
    assert blk[-1]["detail"]["origin"] == "conversation"
    assert blk[0]["ts"] == "2026-09-01T00:30:00Z"  # when it was published
    first = conn.execute("SELECT count(*) FROM issue_events").fetchone()[0]
    again = ledger.backfill(conn)
    assert again["events_added"] == 0 and _table(conn) == GOLDEN
    assert conn.execute("SELECT count(*) FROM issue_events").fetchone()[0] == first
    # scoped runs touch only their PR
    assert ledger.backfill(conn, repo=REPO, number=PR + 1)["prs"] == 0


def test_backfill_after_live_writes_only_fills_gaps(conn, monkeypatch):
    r1, _r2, _r3, r4 = _history(conn)
    # the live write-through recorded round 1 and round 3's answer, as they happened, before
    # the backfill ran
    monkeypatch.setattr(ledger, "now", lambda: "2026-09-01T00:30:00Z")
    _pub(conn, r1, A, posted=[BLOCK, SUGG, GONE])
    monkeypatch.setattr(ledger, "now", lambda: "2026-09-03T00:31:00Z")
    _answer(conn, r4, C, BLOCK.hash, "WITHDRAWN", source="conversation")
    monkeypatch.undo()
    before = conn.execute("SELECT count(*) FROM issue_events").fetchone()[0]
    res = ledger.backfill(conn)
    assert res["events_added"] == res["posted"] + res["seen"] + res["answers"] + res[
        "conceded_only"
    ] - (before)
    table = _table(conn)
    # same fates; the facts the live writes recorded keep their source
    assert {h: v[0] for h, v in table.items()} == {h: v[0] for h, v in GOLDEN.items()}
    assert table[BLOCK.hash][1] == "conversation"
    assert ledger.backfill(conn)["events_added"] == 0


def test_backfill_ignores_audit_concessions_and_carries_same_sha_rereviews(conn):
    f = _f()
    with tx(conn):
        r1 = _run(conn, _head(conn, A), "2026-09-01T00:00:00Z")
        _rows(conn, r1, "posted", [f])
        _review(conn, r1, A, "2026-09-01T00:30:00Z")
        # a same-sha re-review records the whole verifier set as posted again
        r2 = _run(
            conn,
            conn.execute("SELECT head_id FROM runs WHERE id=?", (r1,)).fetchone()[0],
            "2026-09-01T05:00:00Z",
        )
        _rows(conn, r2, "posted", [f])
        _review(conn, r2, A, "2026-09-01T05:30:00Z", "COMMENT")
        # an audit run's conceded row is not a concession on the PR
        ra = _run(conn, _head(conn, B, queue="audit"), "2026-09-02T00:00:00Z")
        _rows(conn, ra, "conceded", [f])
    res = ledger.backfill(conn)
    assert (res["posted"], res["seen"], res["conceded_only"]) == (1, 1, 0)
    i = ledger.get(conn, REPO, PR, f.hash)
    assert i is not None and (i.status, i.opened_run, i.last_seen_run, i.anchor_sha) == (
        "open",
        r1,
        r2,
        A,
    )


def test_backfill_keeps_the_stored_hash(conn):
    with tx(conn):
        r = _run(conn, _head(conn, A), "2026-09-01T00:00:00Z")
        conn.execute(
            "INSERT INTO findings (run_id, phase, stage, hash, file, severity, category, title, body) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (r, "final", "posted", "legacyhash01", "f.rs", "blocking", "logic", "T", "B"),
        )
    res = ledger.backfill(conn)
    assert res["posted_hash_mismatch"] == 1
    assert list(ledger.load(conn, REPO, PR)) == ["legacyhash01"]


def test_render_lists_issues_and_history(conn):
    _history(conn)
    ledger.backfill(conn)
    out = ledger.render(ledger.load(conn, REPO, PR), ledger.history(conn, REPO, PR))
    assert out.splitlines()[0].split()[:4] == ["hash", "status", "severity", "where"]
    assert f"{BLOCK.hash}  withdrawn" in out and "f.rs:11-12" in out
    assert "open 1, fixed 1, withdrawn 2" in out
    assert "outcome WITHDRAWN" in out


# ---- git ----

DIFF = """diff --git a/f.rs b/f.rs
--- a/f.rs
+++ b/f.rs
@@ -40,2 +40,3 @@ fn x() {
-    a();
+    b();
+    c();
"""


@pytest.mark.parametrize(
    ("start", "end", "margin", "changed"),
    [
        (40, 41, 0, True),
        (41, 45, 0, True),
        (10, 20, 0, False),
        (50, 60, 0, False),
        (None, None, 0, True),
        (44, 44, 0, False),
        (44, 44, 3, True),
        (35, 36, 3, False),
        (36, 37, 3, True),
    ],
)
def test_anchor_touched(start, end, margin, changed):
    assert ledger.anchor_touched(DIFF, start, end, margin) is changed


def test_anchor_touched_pure_insertion_and_empty_diff():
    ins = "@@ -9,0 +10,2 @@\n+x\n+y\n"
    assert ledger.anchor_touched(ins, 10, 10) is True
    assert ledger.anchor_touched(ins, 30, 31) is False
    assert ledger.anchor_touched("", 1, 5) is False


def _git(repo: Path, *args: str) -> str:
    # a throwaway fixture repo: hermetic, so no user or injected config (signing) applies
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@t",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@t",
    )
    res = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, env=env, check=True
    )
    return res.stdout.strip()


def _commit(repo: Path, files: dict[str, str]) -> str:
    for name, text in files.items():
        (repo / name).write_text(text)
    _git(repo, "add", *files)
    _git(repo, "commit", "-q", "-m", "c")
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    return r


def test_lines_changed_on_a_real_repo(repo):
    lines = [f"line {i}" for i in range(1, 41)]
    c1 = _commit(repo, {"f.rs": "\n".join(lines) + "\n", "g.rs": "x\n"})
    lines[19] = "line 20 changed"
    c2 = _commit(repo, {"f.rs": "\n".join(lines) + "\n"})
    c3 = _commit(repo, {"h.rs": "new\n"})
    lc = ledger.lines_changed
    assert lc(repo, c1, c2, "f.rs", 20, 20) is True
    assert lc(repo, c1, c2, "f.rs", 17, 17) is True  # within the ±3 margin
    assert lc(repo, c1, c2, "f.rs", 17, 17, margin=0) is False
    assert lc(repo, c1, c2, "f.rs", 5, 8) is False
    assert lc(repo, c1, c2, "g.rs", 1, 1) is False  # file untouched
    assert lc(repo, c1, c3, "h.rs", 1, 1) is True  # did not exist on the old side
    assert lc(repo, c2, c2, "f.rs", 20, 20) is False
    assert lc(repo, c1, "0" * 40, "f.rs", 20, 20) is None  # git cannot tell
    assert lc(repo, c1, c2, "f.rs", None, None) is True  # no anchor: any change
