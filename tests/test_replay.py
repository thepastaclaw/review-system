"""`reviewsys replay`: a past run's exact head, its history as of the run's start, a pinned
read-only GitHub, lanes that see the same; degraded mode refuses to start."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeGh, FakeLanes

from reviewsys import degraded, replay, replaygit
from reviewsys import worker as worker_mod
from reviewsys.config import DegradedPolicy, Substitute
from reviewsys.contract import finding_hash
from reviewsys.db import kv_set, tx
from reviewsys.lane import LaneResult, LaneSpec
from reviewsys.models import ReviewError, RunStatus
from reviewsys.steps import worktree as wt

REPO, NUMBER = "dashpay/platform", 1
CUTOFF = "2026-09-20T12:00:00Z"
BEFORE, AFTER = "2026-09-20T10:00:00Z", "2026-09-20T13:00:00Z"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def mirror(cfg, tmp_path) -> dict[str, str]:
    """The repo's bare mirror where the replay looks for it: base on main, the PR's head."""
    src = tmp_path / "src"
    src.mkdir()
    _git(src, "init", "-q", "-b", "main")
    _git(src, "config", "user.email", "t@example.com")
    _git(src, "config", "user.name", "T")
    (src / "f.rs").write_text("".join(f"fn f{i}() {{}}\n" for i in range(20)))
    _git(src, "add", ".")
    _git(src, "commit", "-qm", "base")
    base = _git(src, "rev-parse", "HEAD")
    _git(src, "checkout", "-qb", "pr")
    (src / "f.rs").write_text("".join(f"fn f{i}() {{ {i} }}\n" for i in range(20)))
    _git(src, "commit", "-qam", "change")
    head = _git(src, "rev-parse", "HEAD")
    cfg.mirrors_dir.mkdir(parents=True)
    subprocess.run(
        ["git", "clone", "-q", "--bare", str(src), str(cfg.mirrors_dir / "dashpay__platform.git")],
        check=True,
    )
    return {"base": base, "head": head}


def _source_run(conn, sha: str, *, started: str, finished: str | None, status: str = "done") -> int:
    with tx(conn):
        row = conn.execute(
            "SELECT id FROM heads WHERE repo=? AND number=? AND sha=?", (REPO, NUMBER, sha)
        ).fetchone()
        hid = (
            row["id"]
            if row
            else conn.execute(
                "INSERT INTO heads (repo, number, sha, trigger, status, queued_at, eligible_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (REPO, NUMBER, sha, "new_push", "done", started, started),
            ).lastrowid
        )
        return int(
            conn.execute(
                "INSERT INTO runs (head_id, attempt, status, token, started_at, finished_at, "
                "deadline_at, tier) VALUES (?,?,?,?,?,?,?,?)",
                (hid, 1, status, "t", started, finished, started, "normal"),
            ).lastrowid
            or 0
        )


def _finding(conn, run_id: int, h: str, stage: str, title: str, phase: str = "final") -> None:
    with tx(conn):
        conn.execute(
            "INSERT INTO findings (run_id, phase, stage, hash, file, line_start, line_end, "
            "severity, category, title, body) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, phase, stage, h, "f.rs", 3, 4, "blocking", "logic", title, "b"),
        )


@pytest.fixture
def history(conn, mirror) -> dict[str, Any]:
    """One earlier round (finished before the replayed run), the replayed run, and a later
    round whose findings and events must not leak into the replay."""
    old_sha = "1" * 40
    early = _source_run(conn, old_sha, started="2026-09-19T10:00:00Z", finished=BEFORE)
    _finding(conn, early, "h-early", "posted", "Early blocker")
    _finding(conn, early, "h-early", "verified", "Early blocker", phase="verify2")
    src = _source_run(conn, mirror["head"], started=CUTOFF, finished=AFTER)
    _finding(conn, src, "h-src", "posted", "What the replayed run posted")
    straddle = _source_run(conn, old_sha, started=BEFORE, finished=AFTER)  # still running
    _finding(conn, straddle, "h-straddle", "posted", "Posted after the cutoff")
    later = _source_run(conn, "2" * 40, started=AFTER, finished="2026-09-20T15:00:00Z")
    _finding(conn, later, "h-later", "posted", "Later blocker")
    _finding(conn, later, "h-early", "conceded", "Early blocker")
    with tx(conn):
        conn.executemany(
            "INSERT INTO posted_findings (repo, number, hash, sha, review_id, posted_at) "
            "VALUES (?,?,?,?,?,?)",
            [
                (REPO, NUMBER, "h-early", old_sha, 11, BEFORE),
                (REPO, NUMBER, "h-later", "2" * 40, 13, AFTER),
            ],
        )
        conn.executemany(
            "INSERT INTO reviews (run_id, repo, number, sha, phase, github_review_id, event, "
            "posted_at) VALUES (?,?,?,?,?,?,?,?)",
            [
                (early, REPO, NUMBER, old_sha, "final", 11, "REQUEST_CHANGES", BEFORE),
                (later, REPO, NUMBER, "2" * 40, "final", 13, "APPROVE", AFTER),
            ],
        )
        conn.executemany(
            "INSERT INTO events (ts, kind, repo, number, run_id, detail) VALUES (?,?,?,?,?,?)",
            [
                (BEFORE, "thread.answered", REPO, NUMBER, early, '{"finding_hash": "h-early"}'),
                (AFTER, "thread.answered", REPO, NUMBER, later, '{"finding_hash": "h-later"}'),
                (BEFORE, "run.done", REPO, NUMBER, early, ""),
            ],
        )
    return {"early": early, "src": src, "later": later, "old_sha": old_sha, **mirror}


def _evidence(cfg, run_id: int, base: str, *, ci: dict[str, Any] | None = None) -> None:
    d = cfg.runs_dir / f"run-{run_id}"
    d.mkdir(parents=True)
    ev = {
        "pr": {
            "title": "Title then",
            "body": "Body then",
            "baseRefName": "main",
            "baseRefOid": base,
        },
        "ci": ci
        or {
            "fetched_at": CUTOFF,
            "checks": [
                {"name": "lint", "status": "completed", "conclusion": "success", "url": "u"}
            ],
        },
    }
    (d / "evidence.json").write_text(json.dumps({"evidence": ev}))


# ---- seeding ----


def test_seeding_copies_history_from_before_the_cutoff_only(conn, history, tmp_path):
    from reviewsys.db import connect

    scratch = connect(tmp_path / "scratch.db")
    counts = replay.seed_history(
        scratch,
        conn,
        repo=REPO,
        number=NUMBER,
        sha=history["head"],
        cutoff=CUTOFF,
        exclude_run=history["src"],
    )
    hashes = {r["hash"]: r["stage"] for r in scratch.execute("SELECT hash, stage FROM findings")}
    assert set(hashes) == {"h-early"}  # not the replayed run's, the straddler's or later ones
    assert {r["stage"] for r in scratch.execute("SELECT stage FROM findings")} == {
        "posted",
        "verified",
    }
    assert [
        (r["hash"], r["sha"]) for r in scratch.execute("SELECT hash, sha FROM posted_findings")
    ] == [("h-early", history["old_sha"])]
    assert [r["event"] for r in scratch.execute("SELECT event FROM reviews")] == ["REQUEST_CHANGES"]
    answered = scratch.execute("SELECT detail FROM events WHERE kind='thread.answered'").fetchall()
    assert [json.loads(r["detail"])["finding_hash"] for r in answered] == ["h-early"]
    assert not scratch.execute("SELECT 1 FROM events WHERE kind='run.done'").fetchone()
    assert counts["runs"] == 1 and counts["thread_answered"] == 1
    # the seeded run sits on its own commit, so per-commit lookups (standing verdict) work
    (sha,) = scratch.execute("SELECT h.sha FROM runs r JOIN heads h ON h.id=r.head_id").fetchone()
    assert sha == history["old_sha"]


def test_seeding_keeps_productions_deferral_tracking_cutoff(conn, history, tmp_path):
    """A replay runs dry, so it never records the cutoff itself: without production's, its
    own start would be the cutoff and every earlier deferral would count as grandfathered."""
    from reviewsys import converse
    from reviewsys.db import connect, kv_get, kv_set

    scratch = connect(tmp_path / "scratch.db")
    with tx(conn):
        kv_set(conn, converse.TRACKING_CUTOFF_KEY, "2026-10-01T00:00:00Z")
    replay.seed_history(
        scratch, conn, repo=REPO, number=NUMBER, sha=history["head"], cutoff=CUTOFF, exclude_run=0
    )
    assert kv_get(scratch, converse.TRACKING_CUTOFF_KEY) == "2026-10-01T00:00:00Z"


def test_a_finding_reposted_after_the_cutoff_keeps_its_earlier_posting(conn, history, tmp_path):
    from reviewsys.db import connect

    with tx(conn):  # the later round re-posted h-early: the live row says so
        conn.execute(
            "UPDATE posted_findings SET sha=?, posted_at=? WHERE hash='h-early'", ("2" * 40, AFTER)
        )
    scratch = connect(tmp_path / "scratch.db")
    replay.seed_history(
        scratch, conn, repo=REPO, number=NUMBER, sha=history["head"], cutoff=CUTOFF, exclude_run=0
    )
    row = scratch.execute("SELECT sha, posted_at FROM posted_findings").fetchone()
    assert (row["sha"], row["posted_at"]) == (history["old_sha"], BEFORE)


def test_a_v11_arm_gets_no_ledger(cfg, conn, history, tmp_path, monkeypatch):
    """Under v11 nothing reads the ledger: the arm runs exactly as before it existed."""
    calls: list[tuple[str, int, str]] = []
    monkeypatch.setattr(
        replay,
        "seed_ledger",
        lambda s, p, repo, number, cutoff: calls.append((repo, number, cutoff)) or {},
    )
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [])
    report = _replay(cfg, history, tmp_path, lanes)
    assert report["status"] == "done", report["reason"]
    assert calls == [] and "ledger" not in report["seeded"]


def test_a_v12_arm_gets_the_ledger_as_of_the_cutoff_reconciled(
    cfg, conn, history, tmp_path, skills_dir
):
    from v12_support import v12_cfg

    v12 = v12_cfg(cfg, skills_dir)
    with tx(conn):  # the early finding was withdrawn before the cutoff; the later one is not seen
        conn.execute(
            "INSERT INTO events (ts, kind, repo, number, run_id, detail) VALUES (?,?,?,?,?,?)",
            (
                BEFORE,
                "thread.answered",
                REPO,
                NUMBER,
                history["early"],
                json.dumps({"finding_hash": "h-early", "status": "WITHDRAWN", "action": "replied"}),
            ),
        )
    _evidence(v12, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    report = replay.replay(
        v12,
        run_id=history["src"],
        out=tmp_path / "out",
        gh_runner=FakeGh().runner,
        lane_runner=lanes,
    )
    assert report["status"] == "done", report["reason"]
    led = report["seeded"]["ledger"]
    assert led["events_copied"] == 0 and led["events_backfilled"] >= 1  # no source ledger
    assert led["by_status"] == {"withdrawn": 1}
    assert report["seeded"]["reconcile"]["pr_open"] is True
    scratch = connect_ro(tmp_path / "out" / f"run-{history['src']}" / "replay.db")
    issues = {r["hash"]: r["status"] for r in scratch.execute("SELECT hash, status FROM issues")}
    assert issues == {"h-early": "withdrawn"}  # nothing from after the cutoff
    rec = scratch.execute(
        "SELECT value FROM kv WHERE key=?", (f"ledger.reconciled:{REPO}#{NUMBER}",)
    )
    assert json.loads(rec.fetchone()[0])["sha"] == history["head"]


def connect_ro(path: Path):
    import sqlite3

    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


# ---- the ledger as of the cutoff ----


def _scratch(conn, history, tmp_path):
    from reviewsys.db import connect

    scratch = connect(tmp_path / "scratch.db")
    replay.seed_history(
        scratch,
        conn,
        repo=REPO,
        number=NUMBER,
        sha=history["head"],
        cutoff=CUTOFF,
        exclude_run=history["src"],
    )
    return scratch


def test_seed_ledger_folds_the_source_events_before_the_cutoff(conn, history, tmp_path):
    from reviewsys import ledger

    early = ledger.Event(
        hash="h-early",
        key=f"posted:{history['early']}:h-early",
        kind="posted",
        source="verifier",
        run_id=history["early"],
        sha=history["old_sha"],
        detail={"file": "f.rs", "line_start": 3, "line_end": 4, "severity": "blocking",
                "category": "logic", "title": "Early blocker"},
        ts=BEFORE,
    )  # fmt: skip
    withdrawn_late = ledger.Event(
        hash="h-early",
        key="status:late:h-early",
        kind="status",
        source="manual",
        run_id=history["later"],
        sha="2" * 40,
        detail={"status": "withdrawn"},
        ts=AFTER,
    )
    with tx(conn):
        ledger.append(conn, REPO, NUMBER, [early, withdrawn_late])
    scratch = _scratch(conn, history, tmp_path)
    got = replay.seed_ledger(scratch, conn, REPO, NUMBER, CUTOFF)
    assert got["events_copied"] == 1
    # the backfill derives the same key for the same posting, so it is not doubled; it only
    # adds the thread answer the source ledger lacks
    assert got["events_backfilled"] == 1
    kinds = [r[0] for r in scratch.execute("SELECT kind FROM issue_events ORDER BY id")]
    assert kinds == ["posted", "outcome"]
    issue = ledger.get(scratch, REPO, NUMBER, "h-early")
    assert issue is not None and issue.status == "open" and issue.opened_run == history["early"]


def test_seed_ledger_on_a_source_without_the_ledger_tables(conn, history, tmp_path):
    import shutil
    import sqlite3

    from reviewsys import ledger

    conn.execute("PRAGMA wal_checkpoint(FULL)")
    old = tmp_path / "old.db"
    shutil.copy(tmp_path / "review.db", old)
    c = sqlite3.connect(old)
    c.executescript("DROP TABLE issue_events; DROP TABLE issues;")
    c.close()
    source = replay.open_source(old)
    scratch = _scratch(source, history, tmp_path)
    got = replay.seed_ledger(scratch, source, REPO, NUMBER, CUTOFF)
    assert got["events_copied"] == 0 and got["issues"] == 1
    assert ledger.get(scratch, REPO, NUMBER, "h-early").status == "open"  # type: ignore[union-attr]


def _thread(h: str, title: str) -> dict[str, Any]:
    return {
        "thread_id": f"PRRT_{h}",
        "is_resolved": False,
        "is_outdated": False,
        "path": "f.rs",
        "line": 3,
        "comments": [
            {
                "id": 7,
                "author": "thepastaclaw",
                "body": f"<!-- thepastaclaw-review v1 finding={h} dedupe=x -->\n"
                f"**🔴 Blocking: {title}**\n\nbody",
                "created_at": BEFORE,
                "association": "NONE",
            }
        ],
    }


def _pin(history, git_dir: str) -> replay.Pin:
    return replay.Pin(
        repo=REPO,
        number=NUMBER,
        sha=history["head"],
        cutoff=CUTOFF,
        merge_base=history["base"],
        base_ref="main",
        base_sha=history["base"],
        git_dir=git_dir,
        real_gh="gh",
    )


def _two_issues(conn, history):
    """h-early (anchored where the code then changed) and 0e0e0e (an open thread)."""
    _finding(conn, history["early"], "0e0e0e", "posted", "Open thread blocker")
    with tx(conn):
        conn.execute(  # the early round's commit is the PR's base, before f.rs changed
            "UPDATE heads SET sha=? WHERE id=(SELECT head_id FROM runs WHERE id=?)",
            (history["base"], history["early"]),
        )
        conn.execute("UPDATE posted_findings SET sha=?", (history["base"],))


def test_reconcile_as_of_outdates_moved_issues_and_leaves_open_threads_pending(
    cfg, conn, history, tmp_path, monkeypatch
):
    from reviewsys import ledger

    _two_issues(conn, history)
    scratch = _scratch(conn, history, tmp_path)
    replay.seed_ledger(scratch, conn, REPO, NUMBER, CUTOFF)
    mirror = cfg.mirrors_dir / "dashpay__platform.git"
    monkeypatch.setattr(
        replay.github, "review_threads", lambda gh, r, n: [_thread("0e0e0e", "Open thread blocker")]
    )
    got = replay.reconcile_as_of(
        scratch, conn, pin=_pin(history, str(mirror)), gh=FakeGh(), bot_login="thepastaclaw",
        git_dir=mirror,
    )  # fmt: skip
    assert got == {
        "source": "replayed",
        "pr_open": True,
        "resolution_as_of_now": 0,
        "open_threads": 1,
        "standing": 2,
        "outdated": 1,
        "pending": 1,
    }
    issues = ledger.load(scratch, REPO, NUMBER)
    assert issues["h-early"].status == "outdated" and issues["0e0e0e"].status == "open"
    assert issues["0e0e0e"].thread_id == "PRRT_0e0e0e"  # the thread sync ran
    rec = ledger.reconciled(scratch, REPO, NUMBER)
    assert rec is not None and rec["at"]
    assert {k: rec[k] for k in ("run", "sha", "pending", "pending_at_reconcile")} == {
        "run": None,
        "sha": history["head"],
        "pending": ["0e0e0e"],
        "pending_at_reconcile": ["0e0e0e"],
    }


def test_reconcile_as_of_a_pr_merged_before_the_cutoff(cfg, conn, history, tmp_path, monkeypatch):
    from reviewsys import ledger

    _two_issues(conn, history)
    with tx(conn):
        conn.execute(
            "INSERT INTO audits (repo, number, sha, source, mode, merged_at, queued_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (REPO, NUMBER, history["head"], "sync", "full", BEFORE, BEFORE),
        )
    assert replay.closed_before(conn, REPO, NUMBER, CUTOFF) == f"merged at {BEFORE}"
    assert replay.closed_before(conn, REPO, NUMBER, BEFORE) == ""  # not yet merged then
    scratch = _scratch(conn, history, tmp_path)
    replay.seed_ledger(scratch, conn, REPO, NUMBER, CUTOFF)
    monkeypatch.setattr(replay.github, "review_threads", lambda *a: pytest.fail("no threads read"))
    mirror = cfg.mirrors_dir / "dashpay__platform.git"
    got = replay.reconcile_as_of(
        scratch, conn, pin=_pin(history, str(mirror)), gh=FakeGh(), bot_login="thepastaclaw",
        git_dir=mirror,
    )  # fmt: skip
    assert got["pr_open"] is False
    assert {i.status for i in ledger.load(scratch, REPO, NUMBER).values()} == {"outdated"}


# ---- git pinned to the cutoff ----


def test_lanes_see_only_the_pinned_refs(cfg, conn, history, tmp_path):
    """The mirror moved on after the cutoff (main, the PR ref, a tag): the lanes' worktree
    has none of it, and the production mirror is left as it was."""
    m = cfg.mirrors_dir / "dashpay__platform.git"
    ident = ("-c", "user.name=T", "-c", "user.email=t@example.com")
    later = _git(
        m, *ident, "commit-tree", f"{history['head']}^{{tree}}", "-p", history["head"], "-m", "x"
    )
    _git(m, "update-ref", "refs/heads/main", later)
    _git(m, "update-ref", "refs/pull/1/head", later)
    _git(m, "tag", "v99", later)
    before = _git(m, "for-each-ref")
    seen: dict[str, str] = {}
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [])

    def spy(spec, art, wt_):
        if not seen:
            seen["refs"] = _git(spec.cwd, "for-each-ref", "--format=%(refname) %(objectname)")
            seen["all"] = _git(spec.cwd, "log", "--all", "--format=%H")
            seen["origin_main"] = _git(spec.cwd, "rev-parse", "origin/main")
            seen["remote"] = _git(spec.cwd, "remote", "get-url", "origin")
        return lanes(spec, art, wt_)

    _evidence(cfg, history["src"], history["base"])
    report = _replay(cfg, history, tmp_path, spy)
    assert report["status"] == "done", report["reason"]
    refs = dict(line.split(" ") for line in seen["refs"].splitlines())
    assert refs == {
        "refs/heads/main": history["base"],
        "refs/heads/replay-head": history["head"],
        "refs/pull/1/head": history["head"],
        "refs/remotes/origin/main": history["base"],
    }
    assert later not in seen["all"] and seen["origin_main"] == history["base"]
    pinned = tmp_path / "out" / f"run-{history['src']}" / "work" / "mirrors"
    assert seen["remote"] == str((pinned / "dashpay__platform.git").resolve())
    assert not pinned.exists()  # removed with the worktree when the replay ended
    assert _git(m, "for-each-ref") == before  # the mirror's refs are untouched
    assert len(_git(m, "worktree", "list").splitlines()) == 1  # nothing registered there


def test_a_pinned_clone_of_a_partial_mirror_fetches_into_itself(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _git(src, "init", "-q", "-b", "main")
    _git(src, "config", "user.email", "t@example.com")
    _git(src, "config", "user.name", "T")
    _git(src, "config", "uploadpack.allowFilter", "true")
    (src / "a.txt").write_text("hello\n")
    _git(src, "add", ".")
    _git(src, "commit", "-qm", "one")
    head = _git(src, "rev-parse", "HEAD")
    mirror = tmp_path / "m.git"
    subprocess.run(
        ["git", "clone", "-q", "--bare", "--filter=blob:none", f"file://{src}", str(mirror)],
        check=True,
    )
    packs_before = sorted(p.name for p in (mirror / "objects" / "pack").iterdir())
    prod = tmp_path / "prod-mirrors"
    prod.mkdir()
    mirror = mirror.rename(prod / replaygit.mirror_name("o/r"))
    clone, base, mb = replaygit.pinned_clone(
        prod, None, tmp_path / "pinned.git", repo="o/r", number=1, sha=head,
        base_ref="main", base_sha=head, cutoff=CUTOFF,
    )  # fmt: skip
    assert (base, mb) == (head, head)
    assert _git(clone, "config", "extensions.partialClone") == replaygit.SOURCE
    tree = wt.create_worktree(clone, tmp_path / "wts", "w", head)
    assert (tree / "a.txt").read_text() == "hello\n"  # the blob came in lazily
    assert sorted(p.name for p in (mirror / "objects" / "pack").iterdir()) == packs_before
    assert any(p.suffix == ".promisor" for p in (clone / "objects" / "pack").iterdir())


def _tree_state(d: Path) -> list[tuple[str, int, float]]:
    return sorted(
        (str(p.relative_to(d)), p.stat().st_size, p.stat().st_mtime) for p in d.rglob("*")
    )


@pytest.fixture
def read_only_mirror(cfg):
    """The production mirror, write-protected for the test: a replay may only read it."""
    import os
    import stat

    m = cfg.mirrors_dir / "dashpay__platform.git"
    paths = [m, *m.rglob("*")]
    modes = {p: p.stat().st_mode for p in paths}
    for p in paths:
        os.chmod(p, modes[p] & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    yield m
    for p in reversed(paths):
        os.chmod(p, modes[p])


def test_a_replay_never_writes_the_production_mirror(
    cfg, conn, history, tmp_path, read_only_mirror
):
    """M2: the PR head is not in the mirror (pushed after the daemon last fetched): it is
    fetched into the replay's own clone; the read-only mirror is left exactly as it was."""
    src = tmp_path / "src"
    ident = ("-c", "user.name=T", "-c", "user.email=t@example.com")
    _git(src, "checkout", "-q", "pr")
    (src / "g.rs").write_text("fn g() {}\n")
    _git(src, "add", ".")
    _git(src, *ident, "commit", "-qm", "pushed later")
    new_head = _git(src, "rev-parse", "HEAD")
    with tx(conn):
        conn.execute(
            "UPDATE heads SET sha=? WHERE id=(SELECT head_id FROM runs WHERE id=?)",
            (new_head, history["src"]),
        )
    before = _tree_state(read_only_mirror)
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(new_head)
    _lanes_ok(lanes, [])
    report = _replay(cfg, history, tmp_path, lanes)
    assert report["status"] == "done", report["reason"]
    assert report["sha"] == new_head
    assert _tree_state(read_only_mirror) == before
    assert not (tmp_path / "out" / ".mirrors").exists()  # the production mirror sufficed


def test_a_repository_without_a_production_mirror_is_mirrored_in_the_eval_dir(
    tmp_path, monkeypatch
):
    src = tmp_path / "src"
    src.mkdir()
    _git(src, "init", "-q", "-b", "main")
    (src / "a").write_text("a\n")
    _git(src, "add", ".")
    _git(src, "-c", "user.name=T", "-c", "user.email=t@e", "commit", "-qm", "a")
    head = _git(src, "rev-parse", "HEAD")
    prod, mine = tmp_path / "prod", tmp_path / "eval-mirrors"
    prod.mkdir()

    def ensure(mirrors_dir: Path, repo: str) -> Path:  # what wt.ensure_mirror clones, offline
        m = mirrors_dir / replaygit.mirror_name(repo)
        if not (m / "HEAD").exists():
            subprocess.run(["git", "clone", "-q", "--bare", str(src), str(m)], check=True)
        return m

    monkeypatch.setattr(replaygit.wt, "ensure_mirror", ensure)
    clone, _, _ = replaygit.pinned_clone(
        prod, mine, tmp_path / "c.git", repo="o/r", number=1, sha=head
    )
    assert list(prod.iterdir()) == []  # no new production mirror
    assert (mine / replaygit.mirror_name("o/r") / "HEAD").exists()
    assert replaygit.has_commit(clone, head)


def test_a_snapshot_source_still_sees_degraded_mode_on_the_live_database(
    cfg, conn, history, tmp_path
):
    import shutil

    conn.execute("PRAGMA wal_checkpoint(FULL)")
    snap = tmp_path / "snap.db"
    shutil.copy(tmp_path / "review.db", snap)
    with tx(conn):  # forced on the live database after the snapshot was taken
        kv_set(conn, degraded.KV_FORCE, "on")
    with pytest.raises(replay.ReplayAborted, match="forced on"):
        replay.replay(cfg, run_id=history["src"], out=tmp_path / "out", source_db=snap,
                      lane_runner=FakeLanes(history["head"]), gh_runner=FakeGh().runner)  # fmt: skip


def test_replay_lanes_line_up_at_the_replay_rank_and_size_gpt_from_the_live_db(
    cfg, conn, history, tmp_path, monkeypatch
):
    from reviewsys import lanepool

    seen: list[tuple[str, Any]] = []
    real = lanepool.gated

    def spy(runner, conn_, cfg_, run_stopped, queue, run_id, **kw):
        seen.append((queue, cfg_.lane_capacity_db))
        return real(runner, conn_, cfg_, run_stopped, queue, run_id, **kw)

    monkeypatch.setattr(lanepool, "gated", spy)
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [])
    _replay(cfg, history, tmp_path, lanes)
    assert seen == [("replay", cfg.db_path)]


def test_replay_policy_runs_no_comparison_lanes(cfg):
    import dataclasses

    from reviewsys.config import ComparisonPolicy

    pol = dataclasses.replace(
        cfg.policy,
        comparison=ComparisonPolicy(model="gpt-6-astra", tiers=("critical",), fraction=1),
    )
    assert replay.replay_policy(pol, None).comparison is None


# ---- the replay ----


def _lanes_ok(lanes: FakeLanes, findings: list[dict[str, Any]]) -> None:
    """Lanes that report `findings` and reconcile the one seeded prior finding as FIXED."""
    prior = [
        {
            "finding_hash": finding_hash("f.rs", "logic", "Early blocker"),
            "status": "FIXED",
            "reason": "rewritten",
        }
    ]
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": findings,
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": prior,
    }
    verifier = {
        "prior_finding_reconciliation": prior,
        "summary": "s",
        "review_action": "COMMENT",
        "findings": findings,
        "dropped_findings": [],
        "out_of_scope_findings": [],
        "coderabbit_reactions": [],
        "prerequisite_adjudications": [],
        "adjudication_complete": True,
    }
    lanes.verifier["default"] = verifier


def _blocker(title: str = "Overflow in f3") -> dict[str, Any]:
    return {
        "file": "f.rs",
        "line_start": 3,
        "line_end": 4,
        "severity": "blocking",
        "confidence": 0.9,
        "category": "logic",
        "title": title,
        "body": "x",
    }


def _replay(cfg, history, tmp_path, lanes, **kw) -> dict[str, Any]:
    return replay.replay(
        cfg,
        run_id=history["src"],
        out=tmp_path / "out",
        gh_runner=kw.pop("gh_runner", FakeGh().runner),
        lane_runner=lanes,
        **kw,
    )


class _FakeBroker:
    """The broker binds a Unix socket, which a sandboxed test host may refuse; test_ghshim
    runs the real one. Lanes here are fakes and never call the shim."""

    def __init__(self, pin, real) -> None:
        self.path = "/nonexistent/socket"

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


@pytest.fixture(autouse=True)
def _no_broker(monkeypatch):
    monkeypatch.setattr(replay, "Broker", _FakeBroker)
    monkeypatch.setattr(replay, "check_shim", lambda shim, profile: "")


def test_replay_reviews_the_source_head_with_pinned_lanes_and_posts_nothing(
    cfg, conn, history, tmp_path
):
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [_blocker()])
    gh = FakeGh()
    gh.pr = {**gh.pr, "state": "closed", "merged": True, "title": "Retitled since"}
    report = _replay(cfg, history, tmp_path, lanes, gh_runner=gh.runner, force_phase2=True)
    assert report["status"] == "done", report["reason"]
    assert not gh.posted_reviews and not gh.gate_bodies and not gh.replies
    assert all(c[:2] != ["pr", "diff"] for c in gh.calls)  # the diff came from git
    work = tmp_path / "out" / f"run-{history['src']}"
    pin = json.loads((work / "pin.json").read_text())
    assert (pin["sha"], pin["merge_base"], pin["cutoff"]) == (
        history["head"],
        history["base"],
        CUTOFF,
    )
    # every lane runs with the shim first on PATH
    bin_dir = str(work / "bin")
    assert lanes.calls and all(dict(s.env)["PATH"].startswith(bin_dir) for s in lanes.calls)
    assert all(dict(s.env)["CLAUDE_ENV_FILE"].startswith(bin_dir) for s in lanes.calls)
    # no web tools, and no GitHub token of the developer's own (none was given)
    assert all(set(s.disallowed_tools) == {"WebFetch", "WebSearch"} for s in lanes.calls)
    assert all(dict(s.env)["GH_TOKEN"] == "" == dict(s.env)["GITHUB_TOKEN"] for s in lanes.calls)
    # lane slots are the machine's, shared by every replay
    assert (work / "work" / "lane-slots").resolve() == (cfg.work_dir / "lane-slots").resolve()
    assert report["options"]["gh_token"].startswith("none")
    reviewer = next(s for s in lanes.calls if s.role == "general")
    assert f"`{history['base']}..{history['head']}`" in reviewer.prompt
    assert "Title then" in reviewer.prompt and "Retitled" not in reviewer.prompt
    # Phase 2 ran past the gate; the gate's own publication is kept for scoring
    assert report["gate"]["forced_phase2"] is True
    assert report["gate"]["would_publish"]["blockers"] == 1
    assert {x["phase"] for x in report["lanes"]} >= {"phase1", "verify1", "phase2", "verify2"}
    assert [(f["title"], f["severity"]) for f in report["published"]] == [
        ("Overflow in f3", "blocking")
    ]
    assert report["published_from"] == "publication-final"
    assert "phase=final" in report["review_body"]
    assert report["historical_posted"][0]["hash"] == "h-src"
    assert report["seeded"]["runs"] == 1
    assert report["gpt"]["lanes"] == sum(x["model"].startswith("gpt") for x in report["lanes"])
    assert all(x["started_at"] and x["finished_at"] for x in report["lanes"])
    assert report["pin"]["ci_snapshot"] is True
    assert json.loads((work / "report.json").read_text())["source_run"] == history["src"]
    assert not (work / "work" / "worktrees").exists()  # disposable worktrees removed


def test_replay_never_touches_the_source_database(cfg, conn, history, tmp_path):
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [])
    before = (
        conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0],
    )
    _replay(cfg, history, tmp_path, lanes)
    after = (
        conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0],
    )
    assert before == after


def test_replay_pins_the_phase1_model_and_never_degrades(cfg, conn, history, tmp_path):
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [])
    report = _replay(cfg, history, tmp_path, lanes, pin_phase1_model="gemini-3.5-pro")
    p1 = {x["model"] for x in report["lanes"] if x["phase"] == "phase1"}
    assert p1 == {"gemini-3.5-pro"}
    assert report["options"]["pin_phase1_model"] == "gemini-3.5-pro"


def test_replay_refuses_to_start_in_degraded_mode(cfg, conn, history, tmp_path):
    with tx(conn):
        kv_set(conn, degraded.KV_FORCE, "on")
    lanes = FakeLanes(history["head"])
    with pytest.raises(replay.ReplayAborted, match="forced on"):
        _replay(cfg, history, tmp_path, lanes)
    assert lanes.calls == [] and not (tmp_path / "out").exists()


def test_degraded_probe_aborts_a_policy_with_stand_ins(cfg, conn):
    import dataclasses

    pol = dataclasses.replace(
        cfg.policy,
        degraded=DegradedPolicy(
            sentinel="gpt-6-astra", substitutes={"gpt-6-astra": Substitute("muse-spark")}
        ),
    )
    assert "out of quota" in replay.degraded_problem(pol, conn, lambda m: (True, "429"))
    assert replay.degraded_problem(pol, conn, lambda m: (False, "ok")) == ""
    assert replay.replay_policy(pol, None).degraded is None


def test_a_quota_failure_mid_replay_marks_the_report_invalid(cfg, conn, history, tmp_path):
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [])
    lanes.dead_models = {"gpt-6-astra"}
    lanes.dead_stderr = "API Error: 429 usage limit has been reached"
    report = _replay(cfg, history, tmp_path, lanes)
    assert report["status"] == "failed" and report["invalid"] == "quota failure mid-run"


def test_without_a_ci_snapshot_lanes_are_told_ci_is_unavailable(cfg, conn, history, tmp_path):
    _evidence(cfg, history["src"], history["base"], ci={"error": "rate limited"})
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [])
    report = _replay(cfg, history, tmp_path, lanes)
    assert report["pin"]["ci_snapshot"] is False
    evidence = json.loads((Path(report["run_dir"]) / "evidence.json").read_text())["evidence"]["ci"]
    assert "CI unavailable in this replay" in evidence["error"]


def test_a_conversation_run_is_replayed_as_one(cfg, conn, history, tmp_path, monkeypatch):
    with tx(conn):
        conn.execute(
            "INSERT INTO steps (run_id, name, status, started_at) VALUES (?,?,?,?)",
            (history["src"], "converse", "ok", CUTOFF),
        )
    seen: list[str] = []
    real_main = worker_mod.main

    def spy(cfg_, conn_, run_id, **kw):
        seen.append(
            conn_.execute(
                "SELECT h.trigger FROM runs r JOIN heads h ON h.id=r.head_id WHERE r.id=?",
                (run_id,),
            ).fetchone()[0]
        )
        return real_main(cfg_, conn_, run_id, **kw)

    monkeypatch.setattr(worker_mod, "main", spy)
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [])
    report = _replay(cfg, history, tmp_path, lanes)
    assert seen == ["review_reply"] and report["source"]["trigger_replayed_as"] == "review_reply"


def test_rerunning_a_replay_replaces_its_directory(cfg, conn, history, tmp_path):
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [])
    _replay(cfg, history, tmp_path, lanes)
    stale = tmp_path / "out" / f"run-{history['src']}" / "stale.txt"
    stale.write_text("x")
    report = _replay(cfg, history, tmp_path, lanes, sample=0)
    assert report["status"] == "done" and not stale.exists()


def test_lane_env_reaches_the_spawned_process(tmp_path, monkeypatch):
    """`LaneSpec.env` is set in the real lane's environment (run_claude_lane)."""
    from reviewsys import lane as lane_mod

    script = tmp_path / "claude"
    script.write_text('#!/bin/sh\nprintf \'{"result": "%s"}\' "$REPLAY_PROBE"\n')
    script.chmod(0o755)
    spec = LaneSpec(
        role="general",
        agent="a",
        model="m",
        effort="high",
        prompt="p",
        cwd=tmp_path,
        add_dir=tmp_path,
        timeout_seconds=30,
        claude_bin=str(script),
    )
    run = lane_mod.with_overrides(
        lane_mod.run_claude_lane, env={"REPLAY_PROBE": "pinned"}, disallowed_tools=("WebFetch",)
    )
    res: LaneResult = run(spec, tmp_path / "art", tmp_path)
    assert res.result_text == "pinned"
    argv = json.loads((tmp_path / "art" / "lane-meta.json").read_text())["argv"]
    assert argv[argv.index("--disallowedTools") + 1] == "WebFetch"


def test_status_is_reported_for_failed_replays(cfg, conn, history, tmp_path):
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [])
    lanes.timeout_roles = {"general"}
    report = _replay(cfg, history, tmp_path, lanes)
    assert report["status"] == RunStatus.FAILED.value and report["invalid"] is None


def test_fetch_head_keeps_going_when_the_commit_is_already_there(cfg, mirror):
    """Two arms of one head fetch the PR ref into the shared mirror at once (one loses the
    ref lock), or the PR has moved on: the replayed commit is what counts."""
    m = cfg.mirrors_dir / "dashpay__platform.git"
    wt.fetch_head(m, 1, mirror["head"])  # there is no refs/pull/1/head to fetch
    with pytest.raises(ReviewError, match="fetch"):
        wt.fetch_head(m, 1, "d" * 40)


def test_a_finding_posted_before_the_cutoff_by_a_run_still_going_is_seeded(conn, history, tmp_path):
    """The straddling run posted h-straddle before the cutoff and finished after it: its
    findings row is recovered, so the worker's prior list keeps the finding."""
    from reviewsys.db import connect

    with tx(conn):
        conn.execute(
            "INSERT INTO posted_findings (repo, number, hash, sha, posted_at) VALUES (?,?,?,?,?)",
            (REPO, NUMBER, "h-straddle", history["old_sha"], "2026-09-20T11:30:00Z"),
        )
    scratch = connect(tmp_path / "scratch.db")
    counts = replay.seed_history(
        scratch, conn, repo=REPO, number=NUMBER, sha=history["head"], cutoff=CUTOFF, exclude_run=0
    )
    assert counts["posted_rows_recovered"] == 1
    row = scratch.execute(
        "SELECT pf.hash, f.title FROM posted_findings pf JOIN findings f "
        "ON f.hash=pf.hash AND f.stage='posted' WHERE pf.hash='h-straddle'"
    ).fetchone()
    assert row["title"] == "Posted after the cutoff"


def test_lifted_statuses_as_of_the_cutoff(conn, history):
    with tx(conn):
        for ts, h, status, action in (
            (BEFORE, "h1", "WITHDRAWN", "replied"),
            (BEFORE, "h2", "FIXED", "replied"),
            ("2026-09-20T11:00:00Z", "h2", "STILL_VALID", "replied"),  # reopened before cutoff
            (BEFORE, "h3", "WITHDRAWN", "no_reply"),  # never reached the thread
            (AFTER, "h4", "WITHDRAWN", "replied"),  # after the cutoff
        ):
            conn.execute(
                "INSERT INTO events (ts, kind, repo, number, detail) VALUES (?,?,?,?,?)",
                (
                    ts,
                    "thread.answered",
                    REPO,
                    NUMBER,
                    json.dumps({"finding_hash": h, "status": status, "action": action}),
                ),
            )
    got = replay.lifted_before(conn, REPO, NUMBER, CUTOFF)
    assert got == {"h1": "WITHDRAWN"}


def test_a_same_sha_rereview_records_its_verdict_update_without_posting(
    cfg, conn, history, tmp_path
):
    """With the pinned gh a replay takes the paths a dry run skips: a final review of this
    head already stands, so the verdict follow-up is computed and recorded, never posted."""
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [_blocker()])
    gh = FakeGh()
    gh.posted_reviews = [
        {
            "id": 9,
            "body": f"<!-- thepastaclaw-review v1 phase=final sha={history['head']} -->",
            "event": "COMMENT",
        }
    ]
    report = _replay(cfg, history, tmp_path, lanes, gh_runner=gh.runner)
    assert report["status"] == "done", report["reason"]
    assert len(gh.posted_reviews) == 1  # nothing posted
    assert report["publications"]["final"]["verdict_update"] is True
    assert report["publications"]["final"]["event"] == "REQUEST_CHANGES"
    assert [f["title"] for f in report["published"]] == ["Overflow in f3"]


def test_lane_output_reading_github_directly_marks_the_replay_invalid(cfg, conn, history, tmp_path):
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [])

    def leaky(spec, art, wt_):
        res = lanes(spec, art, wt_)
        if spec.role == "general":
            (art / "stdout.json").write_text(
                "fetched https://api.github.com/repos/dashpay/platform/pulls/1/files"
            )
        return res

    report = _replay(cfg, history, tmp_path, leaky)
    assert report["invalid"] == "lane reached live GitHub"
    assert report["live_github"]["lane_output_hits"]


def test_the_token_stays_with_the_broker_and_never_reaches_a_lane(
    cfg, conn, history, tmp_path, monkeypatch
):
    monkeypatch.setenv("REVIEWSYS_REPLAY_GH_TOKEN", "ro-token")
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [])
    report = _replay(cfg, history, tmp_path, lanes, gh_token="ro-token")
    for s in lanes.calls:
        env = dict(s.env)
        assert env["GH_TOKEN"] == env["GITHUB_TOKEN"] == env["REVIEWSYS_REPLAY_GH_TOKEN"] == ""
        gh_config = Path(env["GH_CONFIG_DIR"])
        assert gh_config.is_dir() and not any(gh_config.iterdir())
    assert report["options"]["gh_token"] == "read-only token"


def test_the_replay_cli_drops_the_token_from_its_own_environment(monkeypatch, tmp_path):
    import os

    from reviewsys import cli

    seen: dict[str, Any] = {}

    def fake_replay(cfg, **kw):
        seen["token"] = kw["gh_token"]
        seen["env"] = {k: os.environ.get(k) for k in replay.TOKEN_VARS}
        raise replay.ReplayAborted("stop here")

    monkeypatch.setattr(replay, "replay", fake_replay)
    monkeypatch.setattr(cli, "_cfg", lambda args: None)
    for var in replay.TOKEN_VARS:
        monkeypatch.setenv(var, "secret")
    rc = cli.main(["replay", "1", "--out", str(tmp_path)])
    assert rc == replay.EXIT_ABORTED and seen["token"] == "secret"
    assert set(seen["env"].values()) == {None}


def test_an_audit_queue_source_run_is_replayed_as_a_live_review(cfg, conn, history, tmp_path):
    with tx(conn):
        conn.execute(
            "UPDATE heads SET queue='audit' WHERE id=(SELECT head_id FROM runs WHERE id=?)",
            (history["src"],),
        )
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [])
    report = _replay(cfg, history, tmp_path, lanes)
    assert report["status"] == "done" and report["source"]["queue"] == "audit"


def test_the_report_has_the_slot_wait_and_the_thread_state_flag(
    cfg, conn, history, tmp_path, monkeypatch
):
    """L8 and L4: the stall credit lanepool adds to the run's deadline is its slot wait."""
    from reviewsys import lanepool

    real = lanepool.gated

    def stalled(runner, conn_, cfg_, run_stopped, queue, run_id, **kw):
        lanepool.extend_deadline(conn_(), run_id, 600)  # as if its first lane waited 10 min
        return real(runner, conn_, cfg_, run_stopped, queue, run_id, **kw)

    monkeypatch.setattr(lanepool, "gated", stalled)
    _evidence(cfg, history["src"], history["base"])
    lanes = FakeLanes(history["head"])
    _lanes_ok(lanes, [])
    report = _replay(cfg, history, tmp_path, lanes)
    assert report["status"] == "done", report["reason"]
    assert report["slot_wait_s"] == 600.0
    assert report["active_s"] == 0.0  # the run took seconds; the credit exceeds them
    assert report["thread_resolution_as_of_now"] == 0
