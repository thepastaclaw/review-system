from __future__ import annotations

import json
from datetime import UTC, datetime

from reviewsys import cli, rounds
from reviewsys.exporter import build_export

AT = datetime(2026, 11, 9, 12, 0, 0, tzinfo=UTC)
REPO = "dashpay/platform"


def _pr(conn, number, *, state="closed", closed="2026-11-05T10:00:00Z", repo=REPO, author="a"):
    conn.execute(
        "INSERT INTO prs (repo, number, head_sha, author, state, updated_at) VALUES (?,?,?,?,?,?)",
        (repo, number, "h", author, state, closed),
    )


def _review(conn, number, sha, *, event="REQUEST_CHANGES", phase="final", imported=0, repo=REPO):
    n = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    conn.execute(
        "INSERT INTO reviews (repo, number, sha, phase, event, posted_at, imported) "
        "VALUES (?,?,?,?,?,?,?)",
        (repo, number, sha, phase, event, f"2026-11-01T00:{n % 60:02d}:00Z", imported),
    )


def _pr_with_rounds(conn, number, count, *, approve_last=True, **kw):
    _pr(conn, number, **kw)
    for i in range(count):
        ok = approve_last and i == count - 1
        _review(
            conn,
            number,
            f"{number}-{i}",
            event="APPROVE" if ok else "REQUEST_CHANGES",
            repo=kw.get("repo", REPO),
        )


def test_round_counts_distinct_final_shas_only(conn):
    _pr(conn, 1)
    _review(conn, 1, "a")
    _review(conn, 1, "a", event="COMMENT")
    _review(conn, 1, "b", phase="preliminary")
    _review(conn, 1, "c", imported=1)
    _review(conn, 1, "d", event="APPROVE")
    _review(conn, 1, "d", event="COMMENT")
    _pr(conn, 2)
    data = rounds.compute(conn, at=AT)
    assert data["overall"]["n"] == 1
    assert data["overall"]["mean"] == 2
    assert data["overall"]["to_approve_mean"] == 2


def test_open_prs_and_unreviewed_prs_are_not_completed(conn):
    _pr_with_rounds(conn, 1, 3, state="open")
    _pr(conn, 2)
    data = rounds.compute(conn, at=AT)
    assert data["overall"]["n"] == 0
    assert data["overall"]["mean"] is None


def test_stats_and_window(conn):
    for i, r in enumerate([1, 2, 3, 4, 10], 1):
        _pr_with_rounds(conn, i, r)
    _pr_with_rounds(conn, 9, 7, closed="2026-09-01T00:00:00Z")
    data = rounds.compute(conn, at=AT)
    o = data["overall"]
    assert (o["n"], o["mean"], o["median"], o["p90"], o["max"]) == (5, 4.0, 3, 10, 10)
    assert o["share_over_5"] == 0.2
    assert not o["above_target"]
    assert rounds.compute(conn, days=90, at=AT)["overall"]["n"] == 6


def test_flags_need_five_prs(conn):
    for i in range(1, 5):
        _pr_with_rounds(conn, i, 7, repo="o/few")
    for i in range(10, 15):
        _pr_with_rounds(conn, i, 5, repo="o/mid")
    for i in range(20, 25):
        _pr_with_rounds(conn, i, 6, repo="o/high")
    by_repo = {r["repo"]: r for r in rounds.compute(conn, at=AT)["repos"]}
    assert not by_repo["o/few"]["above_target"]
    assert by_repo["o/mid"]["above_target"] and not by_repo["o/mid"]["above_ceiling"]
    assert by_repo["o/high"]["above_ceiling"]
    assert [r["repo"] for r in rounds.compute(conn, at=AT)["repos"]][-1] == "o/few"


def test_no_approve_prs_leave_first_approve_out(conn):
    _pr_with_rounds(conn, 1, 3, approve_last=False)
    _pr_with_rounds(conn, 2, 2)
    o = rounds.compute(conn, at=AT)["overall"]
    assert o["approved_n"] == 1 and o["to_approve_mean"] == 2


def test_trend_and_rolling_and_watch(conn):
    _pr_with_rounds(conn, 1, 2, closed="2026-11-05T10:00:00Z")
    _pr_with_rounds(conn, 2, 4, closed="2026-11-03T10:00:00Z")
    _pr_with_rounds(conn, 3, 6, closed="2026-10-28T10:00:00Z")
    _pr_with_rounds(conn, 4, 5, state="open", author="bob")
    _pr_with_rounds(conn, 5, 4, state="open")
    data = rounds.compute(conn, weeks=3, at=AT)
    assert data["trend"]["weeks"] == ["2026-W44", "2026-W45", "2026-W46"]
    assert [(w["n"], w["mean"]) for w in data["trend"]["overall"]] == [(1, 6), (2, 3), (0, None)]
    assert data["rolling"]["7d"]["overall"]["n"] == 2
    assert data["rolling"]["28d"]["overall"]["n"] == 3
    assert data["rolling"]["28d"]["repos"][REPO]["mean"] == 4
    assert data["watch"] == [{"repo": REPO, "number": 4, "author": "bob", "rounds": 5}]


def test_repo_filter(conn):
    _pr_with_rounds(conn, 1, 2)
    _pr_with_rounds(conn, 2, 8, repo="o/other")
    data = rounds.compute(conn, repo="o/other", at=AT)
    assert [r["repo"] for r in data["repos"]] == ["o/other"]
    assert data["overall"]["n"] == 1


def test_render_and_cli(conn, cfg, capsys):
    for i in range(1, 6):
        _pr_with_rounds(conn, i, 6)
    _pr_with_rounds(conn, 7, 5, state="open", author="bob")
    text = rounds.render(rounds.compute(conn, days=3650, at=AT))
    assert text.splitlines()[2].startswith("ALL")
    assert "ABOVE CEILING" in text
    assert "dashpay/platform#7  bob  5 rounds" in text
    assert cli.main(["rounds", "--db", str(cfg.db_path), "--days", "3650", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["overall"]["above_ceiling"] and out["definition"]


def test_export_carries_rounds(cfg, conn):
    _pr_with_rounds(conn, 1, 2)
    assert build_export(conn, cfg)["rounds"]["target"] == 4
