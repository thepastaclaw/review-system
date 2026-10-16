"""Review context beyond the PR: sister repo checkouts (sister.py) and base drift (drift.py)."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from reviewsys import config as cfg_mod
from reviewsys import drift, gc, sister, worker
from reviewsys.db import kv_get, kv_set, now, tx
from reviewsys.github import PrMeta
from reviewsys.lane import LaneSpec, argv_for
from reviewsys.models import FailKind, ReviewError
from reviewsys.prompts import context_notes, reviewer_prompt
from reviewsys.steps import worktree as wt

SHA1 = "1" * 40
SHA2 = "2" * 40
SHA3 = "3" * 40


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def commit(repo: Path, message: str, **files: str) -> str:
    for name, text in files.items():
        (repo / name).write_text(text)
    git(repo, "add", *files)
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def lock(*packages: tuple[str, str]) -> str:
    return "version = 4\n\n" + "\n".join(
        f'[[package]]\nname = "{name}"\nversion = "0.1.0"\nsource = "{source}"\n'
        for name, source in packages
    )


# ---- Cargo.lock ----


def test_cargo_pins_groups_by_repo_and_commit():
    text = lock(
        ("dashcore", f"git+https://github.com/dashpay/rust-dashcore?rev={SHA1}#{SHA1}"),
        ("key-wallet", f"git+https://github.com/dashpay/rust-dashcore?rev={SHA1}#{SHA1}"),
        ("dash-spv", f"git+https://github.com/dashpay/rust-dashcore?branch=v0.42-dev#{SHA2}"),
        ("orchard", f"git+https://github.com/dashpay/orchard.git?tag=dashified-0.14.1#{SHA3}"),
        ("grovedb", f"git+https://github.com/DashPay/grovedb#{SHA2}"),
        ("rocksdb", f"git+https://github.com/QuantumExplorer/rust-rocksdb.git?rev=x#{SHA1}"),
        ("self", f"git+https://github.com/dashpay/platform?branch=v3.1-dev#{SHA1}"),
        ("short", "git+https://github.com/dashpay/short?rev=9f7e93d#9f7e93d"),
        ("serde", "registry+https://github.com/rust-lang/crates.io-index"),
        ("fork", f"git+https://github.com/dashpay/serde-wasm-bindgen?branch=fix%2Fu8#{SHA3}"),
    )
    pins = sister.cargo_pins(text, owners=("dashpay",), exclude="DashPay/Platform")
    assert set(pins) == {
        "dashpay/rust-dashcore",
        "dashpay/orchard",
        "dashpay/grovedb",  # lowercased: GitHub ignores case, paths and locks must not
        "dashpay/serde-wasm-bindgen",
    }
    core = pins["dashpay/rust-dashcore"]
    assert [p.sha for p in core] == [SHA1, SHA2]  # most packages first
    assert core[0] == sister.Pin(SHA1, f"rev={SHA1}", ("dashcore", "key-wallet"))
    assert core[1] == sister.Pin(SHA2, "branch=v0.42-dev", ("dash-spv",))
    assert pins["dashpay/orchard"] == [sister.Pin(SHA3, "tag=dashified-0.14.1", ("orchard",))]
    assert pins["dashpay/grovedb"][0].ref == ""
    assert pins["dashpay/serde-wasm-bindgen"][0].ref == "branch=fix/u8"
    # any owner on the allowlist, none other
    assert set(sister.cargo_pins(text, owners=("QuantumExplorer",), exclude="")) == {
        "quantumexplorer/rust-rocksdb"
    }


def test_wanted_puts_configured_repos_first_then_by_packages():
    pins = {
        "dashpay/a": [sister.Pin(SHA1, "", ("x",))],
        "dashpay/b": [sister.Pin(SHA1, "", ("x", "y")), sister.Pin(SHA2, "", ("z",))],
        "dashpay/tenderdash": [sister.Pin(SHA3, "", ("t",))],
    }
    order = sister.wanted(
        ("dashpay/Tenderdash", "dashpay/extra", "dashpay/platform"),
        pins,
        exclude="dashpay/platform",
    )
    assert [(r, [p.sha for p in ps]) for r, ps in order] == [
        ("dashpay/tenderdash", [SHA3]),  # configured and pinned: both checkouts
        ("dashpay/extra", []),  # configured only: the default branch
        ("dashpay/b", [SHA1, SHA2]),
        ("dashpay/a", [SHA1]),
    ]


# ---- config ----


def test_config_reads_sister_repos_and_context(tmp_path, skills_dir):
    raw = json.loads((skills_dir / "config.json").read_text())
    raw["repos"][0]["sister_repos"] = ["dashpay/tenderdash"]
    (skills_dir / "config.json").write_text(json.dumps(raw))
    toml = (
        cfg_mod.DEFAULT_TOML.replace('skills = "~/Projects/skills"', f'skills = "{skills_dir}"')
        .replace('sister_owners = ["dashpay"]', 'sister_owners = "QuantumExplorer"')
        .replace("sister_max_repos = 6", "sister_max_repos = -1")
    )
    p = tmp_path / "config.toml"
    p.write_text(toml)
    cfg = cfg_mod.load(p)
    assert cfg.repo("dashpay/platform").sister_repos == ("dashpay/tenderdash",)
    assert cfg.repo("dashpay/grovedb").sister_repos == ()
    assert cfg.context == cfg_mod.ContextConfig(
        sister_owners=("QuantumExplorer",), sister_max_repos=0, sister_budget_minutes=10
    )
    # a config.toml predating the section gets the defaults
    p.write_text(toml.split("[context]")[0])
    assert cfg_mod.load(p).context == cfg_mod.ContextConfig()
    # one repo written as a bare string is one repo; a name without its owner is refused
    raw["repos"][0]["sister_repos"] = "dashpay/tenderdash"
    (skills_dir / "config.json").write_text(json.dumps(raw))
    assert cfg_mod.load(p).repo("dashpay/platform").sister_repos == ("dashpay/tenderdash",)
    raw["repos"][0]["sister_repos"] = ["tenderdash"]
    (skills_dir / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="need owner/name"):
        cfg_mod.load(p)


# ---- sister checkouts (real git, a local "upstream" instead of GitHub) ----


@pytest.fixture
def upstream(tmp_path, cfg, monkeypatch):
    """dashpay/rust-dashcore as a local repo with two commits on main, and its bare mirror
    (as `ensure_mirror` makes it) handed out instead of a GitHub clone."""
    src = tmp_path / "upstream"
    src.mkdir()
    git(src, "init", "-q", "-b", "main")
    c1 = commit(src, "one", **{"lib.rs": "1\n"})
    c2 = commit(src, "two", **{"lib.rs": "2\n"})
    mirror = cfg.mirrors_dir / "dashpay__rust-dashcore.git"
    mirror.parent.mkdir(parents=True)
    git(tmp_path, "clone", "-q", "--bare", str(src), str(mirror))
    git(mirror, "config", "remote.origin.fetch", "+refs/heads/*:refs/heads/*")

    def ensure_mirror(mirrors_dir, repo, *, timeout=1800):
        if repo != "dashpay/rust-dashcore":
            raise ReviewError(FailKind.INFRA, f"clone of {repo} failed: could not resolve host")
        return mirror

    monkeypatch.setattr(wt, "ensure_mirror", ensure_mirror)
    return {"src": src, "c1": c1, "c2": c2}


def _pr_worktree(tmp_path: Path, text: str) -> Path:
    d = tmp_path / "pr"
    d.mkdir(exist_ok=True)
    (d / "Cargo.lock").write_text(text)
    return d


def _gather(cfg, conn, pr: Path, run_id: int = 7, **kw):
    pins = sister.lock_pins(cfg, "dashpay/platform", pr)
    return sister.gather(cfg, conn, repo="dashpay/platform", run_id=run_id, pins=pins, **kw)


def test_gather_checks_out_pinned_and_default_and_shares_them(tmp_path, cfg, conn, upstream):
    c1, c2 = upstream["c1"], upstream["c2"]
    pr = _pr_worktree(
        tmp_path,
        lock(
            ("dashcore", f"git+https://github.com/dashpay/rust-dashcore?rev={c1}#{c1}"),
            ("key-wallet", f"git+https://github.com/DashPay/rust-dashcore?rev={c1}#{c1}"),
            ("grovedb", f"git+https://github.com/dashpay/grovedb?rev={SHA1}#{SHA1}"),
        ),
    )
    entries = _gather(cfg, conn, pr)
    core, grove = entries
    assert core["repo"] == "dashpay/rust-dashcore"  # one repo, however its owner is written
    assert core["packages"] == ["dashcore", "key-wallet"]
    assert "error" not in core
    pinned, default = Path(core["pinned"]["path"]), Path(core["default"]["path"])
    assert core["pinned"]["sha"] == c1 and core["pinned"]["ref"] == f"rev={c1}"
    assert pinned == sister.sister_root(cfg) / f"dashpay__rust-dashcore@{c1[:12]}"
    assert (pinned / "lib.rs").read_text() == "1\n"
    # the default branch is the upstream's HEAD
    assert core["default"] == {"branch": "main", "sha": c2, "path": str(default)}
    assert (default / "lib.rs").read_text() == "2\n"
    # a repo that cannot be had is evidence, not a failure; its pin is still listed
    assert grove["pinned"] == {"sha": SHA1, "ref": f"rev={SHA1}", "path": None}
    assert grove["default"] is None
    assert "could not resolve host" in grove["error"]
    # the run's checkouts are recorded for gc, and handed to the lanes
    assert json.loads(kv_get(conn, "sister.inuse:7")) == [str(pinned), str(default)]
    assert sister.lane_dirs({"sister_repos": entries}) == (pinned, default)
    assert not list(sister.sister_root(cfg).glob(".new-*"))

    # the next run: the pinned checkout is reused (mtime refreshed), the default branch moved
    os.utime(pinned, (1, 1))
    c3 = commit(upstream["src"], "three", **{"lib.rs": "3\n"})
    again = _gather(cfg, conn, pr, run_id=8)
    assert again[0]["pinned"]["path"] == str(pinned)
    assert pinned.stat().st_mtime > time.time() - 60
    assert again[0]["default"]["sha"] == c3
    assert (Path(again[0]["default"]["path"]) / "lib.rs").read_text() == "3\n"

    # a lane wrote into the shared checkout: the next run gets a clean one at the same path
    (pinned / "lib.rs").write_text("scribble")
    (pinned / "notes.txt").write_text("x")
    assert _gather(cfg, conn, pr, run_id=9)[0]["pinned"]["path"] == str(pinned)
    assert (pinned / "lib.rs").read_text() == "1\n" and not (pinned / "notes.txt").exists()

    # gc deleted a checkout (rmtree, as it does): the lanes are not sent to it, and the same
    # path checks out again on next use
    shutil.rmtree(pinned)
    assert sister.lane_dirs({"sister_repos": entries}) == (default,)
    assert _gather(cfg, conn, pr, run_id=10)[0]["pinned"]["path"] == str(pinned)
    assert (pinned / "lib.rs").read_text() == "1\n"


def test_gather_reports_a_bad_pin_and_keeps_the_default(tmp_path, cfg, conn, upstream):
    c1 = upstream["c1"]
    pr = _pr_worktree(
        tmp_path,
        lock(
            ("a", f"git+https://github.com/dashpay/rust-dashcore#{c1}"),
            ("b", f"git+https://github.com/dashpay/rust-dashcore#{'d' * 40}"),
            ("c", f"git+https://github.com/dashpay/rust-dashcore#{'e' * 40}"),
        ),
    )
    (entry,) = _gather(cfg, conn, pr)
    assert entry["pinned"]["packages"] == ["a"] and entry["pinned"]["path"]
    assert entry["also_pinned"] == [{"sha": "d" * 40, "ref": "", "path": None, "packages": ["b"]}]
    assert entry["note"].endswith("listed, not eeeeeeeeeeee (c)")
    assert entry["error"].startswith(f"pinned {'d' * 12}: ")
    assert entry["default"]["branch"] == "main" and entry["default"]["sha"] == upstream["c2"]


def test_gather_caps_repos_per_run_and_reads_configured_ones(tmp_path, cfg, conn, upstream):
    pr = _pr_worktree(
        tmp_path, lock(("g", f"git+https://github.com/dashpay/grovedb?tag=v1#{SHA1}"))
    )
    rc = dataclasses.replace(cfg.repos[0], sister_repos=("DashPay/rust-dashcore",))
    cfg = dataclasses.replace(
        cfg,
        repos=(rc, *cfg.repos[1:]),
        context=dataclasses.replace(cfg.context, sister_max_repos=1),
    )
    core, grove = _gather(cfg, conn, pr)
    # configured: default branch only
    assert core["repo"] == "dashpay/rust-dashcore" and core["pinned"] is None
    assert core["default"]["sha"] == upstream["c2"] and "error" not in core
    assert grove["skipped"] == "not checked out: over the cap of 1 per run"
    assert grove["pinned"] == {"sha": SHA1, "ref": "tag=v1", "path": None}
    # no Cargo.lock, nothing configured: nothing to do; 0 = none at all
    assert sister.gather(cfg, conn, repo="dashpay/dash", run_id=7, pins={}) == []
    off = dataclasses.replace(cfg, context=dataclasses.replace(cfg.context, sister_max_repos=0))
    assert _gather(off, conn, pr) == []


def test_gather_gives_up_on_a_busy_repo_and_stops_when_cancelled(
    tmp_path, cfg, conn, upstream, monkeypatch
):
    c1 = upstream["c1"]
    pr = _pr_worktree(tmp_path, lock(("a", f"git+https://github.com/dashpay/rust-dashcore#{c1}")))
    monkeypatch.setattr(sister, "LOCK_WAIT_SECONDS", 1)
    with sister._repo_lock(cfg.mirrors_dir / ".dashpay__rust-dashcore.lock", wait=0) as held:
        assert held  # another run is cloning it
        (entry,) = _gather(cfg, conn, pr)
    assert entry["error"] == "another run holds the repo"
    assert entry["pinned"]["path"] is None
    assert _gather(cfg, conn, pr, should_stop=lambda: True) == []


def test_lane_gets_the_sister_checkouts_as_extra_dirs(tmp_path):
    spec = LaneSpec(
        role="general",
        agent="a",
        model="m",
        effort="high",
        prompt="p",
        cwd=tmp_path,
        add_dir=tmp_path / "run",
        extra_dirs=(tmp_path / "s1", tmp_path / "s2"),
        timeout_seconds=60,
        claude_bin="claude",
    )
    argv = argv_for(spec)
    i = argv.index("--add-dir")
    assert argv[i + 1 : i + 4] == [
        str(tmp_path / "run"),
        str(tmp_path / "s1"),
        str(tmp_path / "s2"),
    ]
    assert argv[i + 4].startswith("--")


# ---- gc ----


def _run(conn, status: str) -> int:
    with tx(conn):
        hid = conn.execute(
            "INSERT INTO heads (repo, number, sha, trigger, status, queued_at, eligible_at) "
            "VALUES ('dashpay/platform', ?, ?, 'new_pr', 'running', ?, ?)",
            (int(time.time() * 1000) % 100000, os.urandom(20).hex(), now(), now()),
        ).lastrowid
        return int(
            conn.execute(
                "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at) "
                "VALUES (?, 1, ?, 't', ?, ?)",
                (hid, status, now(), now()),
            ).lastrowid
        )


def test_gc_keeps_sister_checkouts_active_runs_use(cfg, conn):
    root = sister.sister_root(cfg)
    used, unused, fresh = root / "dashpay__a@111", root / "dashpay__a@222", root / "x@333"
    half = root / ".new-dashpay__b@444-123"  # a dead worker's half-made checkout
    for p in (used, unused, fresh, half):
        p.mkdir(parents=True)
        (p / "f").write_text("x" * 100)
    old = time.time() - 3 * 86400
    for p in (used, unused, half, root):
        os.utime(p, (old, old))
    active, finished = _run(conn, "running"), _run(conn, "done")
    with tx(conn):
        kv_set(conn, f"sister.inuse:{active}", json.dumps([str(used)]))
        kv_set(conn, f"sister.inuse:{finished}", json.dumps([str(unused)]))
    # a run is checking out dashpay/a right now: gc leaves that repo for next time
    with sister._repo_lock(cfg.mirrors_dir / ".dashpay__a.lock", wait=0) as held:
        assert held
        gc.run(conn, cfg)
    assert unused.exists() and not half.exists()
    stats = gc.run(conn, cfg)
    assert used.exists() and fresh.exists() and not unused.exists()
    assert root.exists()  # never swept as a whole, however old
    assert stats["sister_leases_released"] == 0  # released by the first pass
    assert kv_get(conn, f"sister.inuse:{finished}") is None
    assert kv_get(conn, f"sister.inuse:{active}") is not None
    # over the budget: the LRU sweep takes everything else, never the in-use checkout
    gc.run(conn, dataclasses.replace(cfg, worktree_budget_gb=0))
    assert used.exists() and not fresh.exists()


def test_gc_rechecks_a_checkout_a_run_picked_up_meanwhile(cfg, conn):
    root = sister.sister_root(cfg)
    path = root / "dashpay__a@111"
    path.mkdir(parents=True)
    active = _run(conn, "running")
    # gc listed it as unused; a run recorded and touched it before gc got the lock
    with tx(conn):
        kv_set(conn, f"sister.inuse:{active}", json.dumps([str(path)]))
    assert not sister.remove_unused(cfg, conn, path, older_than=time.time() - 86400)
    with tx(conn):
        kv_set(conn, f"sister.inuse:{active}", "[]")
    assert not sister.remove_unused(cfg, conn, path, older_than=time.time() - 86400)  # fresh
    assert sister.remove_unused(cfg, conn, path) and not path.exists()


# ---- base drift (real git) ----


def test_base_drift_lists_base_commits_conflicts_and_prs(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    base = commit(repo, "start", **{"a.txt": "a\n", "b.txt": "1\n2\n3\n4\n5\n6\n", "c.txt": "c\n"})
    git(repo, "checkout", "-q", "-b", "feature")
    head = commit(repo, "the PR", **{"a.txt": "pr\n", "b.txt": "1\n2\n3\n4\n5\npr\n"})
    git(repo, "checkout", "-q", "main")
    m1 = commit(repo, "fix: same line (#11)", **{"a.txt": "base\n"})
    commit(repo, "chore: elsewhere (#12)", **{"c.txt": "base\n"})
    git(repo, "checkout", "-q", "-b", "side")
    commit(repo, "side work", **{"b.txt": "base\n2\n3\n4\n5\n6\n"})
    git(repo, "checkout", "-q", "main")
    git(repo, "merge", "-q", "--no-ff", "-m", "Merge pull request #13 from x/side", "side")
    merge = git(repo, "rev-parse", "HEAD")
    git(repo, "update-ref", "refs/remotes/origin/main", "main")
    git(repo, "checkout", "-q", "--detach", head)

    out = drift.base_drift(repo, "main", base, head)
    assert out == {
        "base_branch": "main",
        "base_tip": merge,
        "merge_base": base,
        "commits_behind": 4,
        "conflicting_files": ["a.txt"],
        "base_commits_touching_changed_files": [
            {
                "sha": merge,
                "subject": "Merge pull request #13 from x/side",
                "pr": 13,
                "files": ["b.txt"],
            },
            {"sha": m1, "subject": "fix: same line (#11)", "pr": 11, "files": ["a.txt"]},
        ],
    }
    # a PR on top of the current base: nothing behind, nothing to check
    assert drift.base_drift(repo, "main", merge, merge)["commits_behind"] == 0
    # a merge check git refuses (unrelated histories) leaves the rest of the evidence
    git(repo, "checkout", "-q", "--orphan", "other")
    commit(repo, "unrelated", **{"z.txt": "z\n"})
    git(repo, "update-ref", "refs/remotes/origin/other", "other")
    out = drift.base_drift(repo, "other", base, head)
    assert out["commits_behind"] == 1 and out["conflicting_files"] is None
    assert out["error"].startswith("merge check: git merge-tree")


def test_pr_number_from_subjects():
    assert drift.pr_number("Merge pull request #5 from a/b") == 5
    assert drift.pr_number("Merge #7811: fix: thing") == 7811
    assert drift.pr_number("feat(sdk): x (#4392)") == 4392
    assert drift.pr_number("Merge bitcoin/bitcoin#27000: upstream") is None
    assert drift.pr_number("refactor: no number") is None


# ---- the context step ----


def _ctx(cfg, conn, gh, tmp_path, **kw) -> worker.RunContext:
    wt_dir = tmp_path / "wt"
    wt_dir.mkdir(exist_ok=True)
    ctx = worker.RunContext(
        cfg=cfg,
        main_conn=conn,
        gh=gh,
        run_id=3,
        head_id=1,
        repo="dashpay/platform",
        number=9,
        sha="a" * 40,
        token="t",
        run_dir=tmp_path / "run",
        **kw,
    )
    ctx.worktree = wt_dir
    ctx.merge_base = "b" * 40
    ctx.meta = PrMeta("T", "B", "v3.1-dev", "a" * 40, "x", "open", False, "u", False)
    return ctx


def test_context_failures_become_evidence_and_events(tmp_path, cfg, conn, gh, monkeypatch):
    ctx = _ctx(cfg, conn, gh, tmp_path)
    (ctx.worktree / "Cargo.lock").write_text("not [ toml")

    def broken(*a, **k):
        raise ReviewError(FailKind.INFRA, "git rev-list failed: not a git repository")

    monkeypatch.setattr(drift, "base_drift", broken)
    worker._review_context(ctx)  # never raises
    assert "sister_repos" not in ctx.evidence
    assert ctx.evidence["base_drift"]["base_branch"] == "v3.1-dev"
    assert "not a git repository" in ctx.evidence["base_drift"]["error"]
    kinds = [r[0] for r in conn.execute("SELECT kind FROM events WHERE run_id=3 ORDER BY id")]
    assert kinds == ["context.sister_failed", "context.base_drift_failed"]


def test_context_records_sister_repos_and_skips_drift_on_audits(
    tmp_path, cfg, conn, gh, monkeypatch
):
    entries = [{"repo": "dashpay/x", "packages": [], "pinned": None, "default": None, "error": "e"}]
    monkeypatch.setattr(sister, "gather", lambda *a, **k: entries)
    monkeypatch.setattr(drift, "base_drift", lambda *a: pytest.fail("no drift on an audit"))
    ctx = _ctx(cfg, conn, gh, tmp_path, audit={"mode": "full"})
    worker._review_context(ctx)
    assert ctx.evidence == {"sister_repos": entries}
    (detail,) = [r[0] for r in conn.execute("SELECT detail FROM events WHERE run_id=3")]
    assert detail == "dashpay/x: e"


def test_a_broken_cargo_lock_keeps_the_configured_sister_repos(
    tmp_path, cfg, conn, gh, upstream, monkeypatch
):
    rc = dataclasses.replace(cfg.repos[0], sister_repos=("dashpay/rust-dashcore",))
    cfg = dataclasses.replace(cfg, repos=(rc, *cfg.repos[1:]))
    ctx = _ctx(cfg, conn, gh, tmp_path, audit={"mode": "full"})
    (ctx.worktree / "Cargo.lock").write_text("<<<<<<< HEAD\nnot [ toml")
    worker._review_context(ctx)
    (core,) = ctx.evidence["sister_repos"]
    assert core["default"]["sha"] == upstream["c2"] and core["pinned"] is None
    (detail,) = [r[0] for r in conn.execute("SELECT detail FROM events WHERE run_id=3")]
    assert detail.startswith("Cargo.lock: TOMLDecodeError")
    # a run cancelled meanwhile stops here
    ctx.cancel_flag.set()
    with pytest.raises(worker.Cancelled):
        worker._review_context(ctx)


def test_a_conversation_run_gathers_no_review_context(tmp_path, cfg, conn, gh, monkeypatch):
    monkeypatch.setattr(worker, "_review_context", lambda ctx: pytest.fail("not for a reply"))
    ctx = _ctx(cfg, conn, gh, tmp_path)
    ctx.run_dir.mkdir()
    ctx.conversation = True
    worker.step_context(ctx)
    assert "base_drift" not in ctx.evidence and (ctx.run_dir / "evidence.json").exists()


def test_prompts_explain_the_context_only_when_present(cfg):
    assert context_notes({}) == ""
    assert context_notes({"base_drift": {"base_branch": "x", "error": "e"}}) == ""
    assert context_notes({"base_drift": {"commits_behind": 0}}) == ""
    evidence = {
        "sister_repos": [{"repo": "dashpay/rust-dashcore"}],
        "base_drift": {"commits_behind": 3},
    }
    notes = context_notes(evidence)
    assert "gh issue list -R <repo>" in notes and "Never edit them" in notes
    assert "no longer merges cleanly" in notes
    kw = dict(
        repo="dashpay/platform",
        number=9,
        head_sha="a" * 40,
        phase="final",
        role="general",
        meta={"title": "T"},
        coverage_from="b" * 40,
        prior=[],
        prior_sha=None,
    )
    assert notes in reviewer_prompt(cfg, evidence=evidence, **kw)
    assert "Sister Dash repositories" not in reviewer_prompt(cfg, evidence={}, **kw)


def test_ensure_mirror_clones_aside_and_replaces_an_interrupted_one(tmp_path, monkeypatch):
    calls: list[tuple[str, ...]] = []
    fail = []

    def fake_git(*args, cwd=None, timeout=600, ok=(0,)):
        calls.append(args)
        if args[0] == "clone":
            d = Path(args[-1])
            d.mkdir()
            if fail:
                raise ReviewError(FailKind.INFRA, "git clone timed out")
            (d / "HEAD").write_text("ref: refs/heads/main\n")
        return ""

    monkeypatch.setattr(wt, "_git", fake_git)
    fail.append(True)
    with pytest.raises(ReviewError):
        wt.ensure_mirror(tmp_path, "dashpay/x")
    assert not list(tmp_path.iterdir())  # a killed clone leaves nothing to mistake for a mirror
    fail.clear()
    leftover = tmp_path / "dashpay__x.git"  # what a clone killed in place used to leave
    (leftover / "objects").mkdir(parents=True)
    mirror = wt.ensure_mirror(tmp_path, "dashpay/x")
    assert mirror == leftover and (mirror / "HEAD").exists() and not (mirror / "objects").exists()
    assert [p.name for p in tmp_path.iterdir()] == ["dashpay__x.git"]
    calls.clear()
    assert wt.ensure_mirror(tmp_path, "dashpay/x") == mirror and calls == []
