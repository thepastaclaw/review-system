"""`reviewsys quality` (quality.py): the production metrics, on a real git repo and a ledger."""

from __future__ import annotations

import json

import pytest
from test_ledger import _commit, _git

from reviewsys import cli, ledger, quality
from reviewsys.contract import Finding
from reviewsys.db import connect, tx

REPO, PR = "dashpay/platform", 3


def _f(title, line, sev="blocking", file="f.rs"):
    return Finding(
        file=file,
        title=title,
        body="b",
        severity=sev,
        category="logic",
        line_start=line,
        line_end=line,
    )


@pytest.fixture
def mirror(tmp_path):
    r = tmp_path / "mirror"
    r.mkdir()
    _git(r, "init", "-q")
    return r


def _round(conn, sha, started, *, posted=(), verified=(), answers=(), event="REQUEST_CHANGES"):
    """One reviewed head as v11 records it, then the backfill."""
    with tx(conn):
        hid = conn.execute(
            "INSERT INTO heads (repo, number, sha, trigger, status, queued_at, eligible_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (REPO, PR, sha, "new_push", "done", started, started),
        ).lastrowid
        rid = conn.execute(
            "INSERT INTO runs (head_id, attempt, status, token, started_at, finished_at, "
            "deadline_at) VALUES (?,1,'done','t',?,?,?)",
            (hid, started, started, started),
        ).lastrowid
        rows = [(f, "final", "posted") for f in posted]
        rows += [(f, "verify2", "verified") for f in (*posted, *verified)]
        conn.executemany(
            "INSERT INTO findings (run_id, phase, stage, hash, file, line_start, line_end, "
            "severity, category, title, body) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    rid,
                    ph,
                    st,
                    f.hash,
                    f.file,
                    f.line_start,
                    f.line_end,
                    f.severity,
                    f.category,
                    f.title,
                    f.body,
                )
                for f, ph, st in rows
            ],
        )
        conn.execute(
            "INSERT INTO reviews (run_id, repo, number, sha, phase, event, posted_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (rid, REPO, PR, sha, "final", event, started),
        )
        for h, status in answers:
            conn.execute(
                "INSERT INTO events (ts, kind, repo, number, run_id, detail) VALUES (?,?,?,?,?,?)",
                (
                    started,
                    "thread.answered",
                    REPO,
                    PR,
                    rid,
                    json.dumps({"finding_hash": h, "status": status, "action": "replied"}),
                ),
            )
    return rid


def _lines(n, changed=()):
    return "".join(f"line {i}{' changed' if i in changed else ''}\n" for i in range(1, n + 1))


def test_quality_metrics_on_a_reviewed_pr(conn, mirror):
    s1 = _commit(mirror, {"f.rs": _lines(100), "g.rs": "g\n"})
    s2 = _commit(mirror, {"f.rs": _lines(100, changed={50})})
    s3 = _commit(mirror, {"g.rs": "g2\n"})  # f.rs unchanged since s2
    fixed = _f("Round-one blocker", 50)  # FIXED at s2 (its line changed)
    carried = _f("Carried into round two", 80)  # kept by s2's verifier
    withdrawn = _f("Withdrawn after a reply", 30)  # WITHDRAWN at s2, re-posted at s3
    late = _f("Late find on old code", 10)  # s2, line 10 unchanged since s1: late
    touched = _f("Late find on new code", 50)  # s2, on the line s2 changed: not late
    _round(conn, s1, "2026-09-01T00:00:00Z", posted=[fixed, carried, withdrawn])
    _round(
        conn,
        s2,
        "2026-09-02T00:00:00Z",
        posted=[late, touched],
        verified=[carried],
        answers=[(fixed.hash, "FIXED"), (withdrawn.hash, "WITHDRAWN")],
    )
    # the same title again (so the same hash), a line off, with its code unchanged: a re-raise
    _round(conn, s3, "2026-09-03T00:00:00Z", posted=[_f(withdrawn.title, 31)], event="COMMENT")
    ledger.backfill(conn)
    rep = quality.report(conn, since=None, paths=quality.RepoPaths(None, {REPO: mirror}))
    assert rep["posted_blockers"] == 5  # by first post: the re-raise is the withdrawn issue
    m1 = rep["m1_late_find"]
    assert (m1["round1"], m1["later_round"], m1["unchecked"]) == (3, 2, 0)
    assert (m1["late"], m1["anchor_unchanged"], m1["anchor_changed"]) == (1, 1, 1)
    assert (m1["checkable"], m1["rate_pct"], m1["later_round_rate_pct"]) == (5, 20.0, 50.0)
    assert m1["late_prev_final"] == 1 and m1["late_fixed"] == 0
    assert m1["examples"] == [f"{REPO}#{PR} {late.hash} f.rs:10"]
    # everything raised before the last round (s3) has a later round
    m5 = rep["m5_fate_coverage"]
    assert m5["with_later_round"] == 5 and m5["recorded"] == 3 and m5["rate_pct"] == 60.0
    assert m5["fates"] == {"vanished": 2, "FIXED": 1, "carried": 1, "WITHDRAWN": 1}
    assert rep["m3_pushback"] == {"pushed_back": 1, "with_later_round": 5, "rate_pct": 20.0}
    # late and touched are gone at s3, where f.rs did not change at all
    m6 = rep["m6_silent_drop"]
    assert (m6["silent_drops"], m6["vanished_checked"], m6["unchecked"]) == (2, 2, 0)
    m4 = rep["m4_reraise"]
    assert (m4["settled"], m4["by_title"], m4["by_location"], m4["unchecked"]) == (1, 1, 1, 0)
    assert m4["examples"] == [f"{REPO}#{PR} {withdrawn.hash} f.rs:30"]
    assert rep["notes"] == []
    text = quality.render(rep)
    assert "M1 late-find rate: 1/5 checkable blockers (20.0%)" in text
    assert "M4 re-raise: by title 1/1 settled" in text


def test_quality_without_a_mirror_skips_the_git_metrics_with_a_note(conn):
    _round(conn, "a" * 40, "2026-09-01T00:00:00Z", posted=[_f("One", 1)])
    _round(conn, "b" * 40, "2026-09-02T00:00:00Z", posted=[_f("Two", 2)])
    ledger.backfill(conn)
    rep = quality.report(conn, since=None, paths=quality.RepoPaths(None, {}))
    m1 = rep["m1_late_find"]
    # round 1 is no more checkable than round 2 without the commits: neither counts
    assert (m1["round1_unchecked"], m1["unchecked"], m1["checkable"]) == (1, 1, 0)
    assert m1["rate_pct"] is None and m1["later_round_rate_pct"] is None
    assert rep["notes"] == [f"no mirror for {REPO}: 3 git checks skipped"]
    assert "M1 late-find rate: 0/0 checkable blockers (n/a)" in quality.render(rep)


def test_round_one_alone_is_not_a_rate(conn, mirror):
    s1 = _commit(mirror, {"f.rs": _lines(20)})
    _round(conn, s1, "2026-09-01T00:00:00Z", posted=[_f("One", 1)])
    _round(conn, "0" * 40, "2026-09-02T00:00:00Z", posted=[_f("Two", 2)])  # not in the mirror
    ledger.backfill(conn)
    m1 = quality.report(conn, since=None, paths=quality.RepoPaths(None, {REPO: mirror}))[
        "m1_late_find"
    ]
    assert (m1["round1"], m1["round1_unchecked"], m1["unchecked"], m1["checkable"]) == (1, 0, 1, 1)
    assert m1["rate_pct"] is None


def test_a_path_on_neither_side_is_unchecked_not_unchanged(conn, mirror):
    s1 = _commit(mirror, {"f.rs": _lines(20)})
    s2 = _commit(mirror, {"f.rs": _lines(20, changed={5})})
    _round(conn, s1, "2026-09-01T00:00:00Z", posted=[_f("One", 1)])
    _round(conn, s2, "2026-09-02T00:00:00Z", posted=[_f("Wrong path", 2, file="nope.rs")])
    ledger.backfill(conn)
    m1 = quality.report(conn, since=None, paths=quality.RepoPaths(None, {REPO: mirror}))[
        "m1_late_find"
    ]
    assert (m1["unchecked"], m1["late"], m1["file_unchanged"]) == (1, 0, 0)
    assert ledger.lines_changed(mirror, s1, s2, "nope.rs", 2, 2) is None
    assert ledger.lines_changed(mirror, s1, s2, "f.rs", 18, 18) is False


def test_an_explicit_withdrawal_counts_as_pushback_and_settles(conn, mirror):
    s1 = _commit(mirror, {"f.rs": _lines(20)})
    s2 = _commit(mirror, {"g.rs": "x\n"})
    f = _f("Withdrawn by an operator", 5)
    _round(conn, s1, "2026-09-01T00:00:00Z", posted=[f])
    ledger.backfill(conn)
    with tx(conn):
        ledger.transition(conn, REPO, PR, f.hash, "withdrawn", run_id=None, sha=s1, source="manual")
    _round(conn, s2, "2099-01-01T00:00:00Z", posted=[f])  # posted again, code unchanged
    ledger.backfill(conn)
    rep = quality.report(conn, since=None, paths=quality.RepoPaths(None, {REPO: mirror}))
    assert rep["m3_pushback"]["pushed_back"] == 1
    assert (rep["m4_reraise"]["settled"], rep["m4_reraise"]["by_title"]) == (1, 1)


def test_quality_since_and_contract_classes(conn):
    with tx(conn):
        hid = conn.execute(
            "INSERT INTO heads (repo, number, sha, trigger, queued_at, eligible_at) "
            "VALUES ('o/r', 1, 's', 'manual', 'x', 'x')"
        ).lastrowid
        for started, reason in [
            ("2026-09-01", "reconciliation missing prior hashes: ['x']"),
            ("2026-10-02", "STILL_VALID set ['a'] != carried findings []"),
            ("2026-10-02", "reviewer output review_phase='plan', expected 'final'"),
            ("2026-10-02", "finding 'x' has invalid severity 'medium'"),
        ]:
            conn.execute(
                "INSERT INTO runs (head_id, attempt, status, fail_kind, reason, token, started_at, "
                "deadline_at) VALUES (?,1,'failed','contract',?,'t',?,?)",
                (hid, reason, started, started),
            )
    rep = quality.report(conn, since="2026-10-01", paths=quality.RepoPaths())
    assert rep["m9_contract"] == {
        "contract": 3,
        "failed_runs": 3,
        "runs": 3,
        "reconciliation": 1,
        "echo_field": 1,
        "other": 1,
    }
    assert "the ledger is empty" in rep["notes"][0]


def test_cli_ledger_and_quality(tmp_path, capsys):
    db = tmp_path / "copy.db"
    c = connect(db)
    _round(c, "a" * 40, "2026-09-01T00:00:00Z", posted=[_f("One", 1)])
    c.close()
    assert cli.main(["ledger", "backfill", "--db", str(db)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["issues"] == 1 and out["by_status"] == {"open": 1}
    assert cli.main(["ledger", "show", REPO, str(PR), "--db", str(db), "--history"]) == 0
    shown = capsys.readouterr().out
    assert "One" in shown and "open 1" in shown and "posted" in shown
    assert cli.main(["ledger", "show", REPO, str(PR), "--db", str(db), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["issues"][0]["status"] == "open"
    assert cli.main(["quality", "--db", str(db), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["posted_blockers"] == 1
    assert cli.main(["quality", "--db", str(db), "--repo-path", "bad"]) == 1
    capsys.readouterr()
    missing = tmp_path / "missing.db"
    assert cli.main(["quality", "--db", str(missing)]) == 1
    assert cli.main(["ledger", "show", REPO, "1", "--db", str(missing)]) == 1
    assert not missing.exists() and "no database at" in capsys.readouterr().err
    assert cli.main(["ledger", "backfill", "--db", str(db), "--number", "3"]) == 1
    assert "--number needs --repo" in capsys.readouterr().err
