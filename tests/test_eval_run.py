"""The eval end to end on the smoke-test fixture (replayfake.py): replays of both arms, the
judge, the score; and deploy/eval-run.sh's guards and its `--dry` run."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from reviewsys import config as cfg_mod
from reviewsys import judge as judge_mod
from reviewsys import lane as lane_mod
from reviewsys import replay, replayfake, replayscore
from reviewsys.replayset import Job, Policy, load_set, run_set

SCRIPT = Path(__file__).resolve().parent.parent / "deploy" / "eval-run.sh"


class _NoBroker:
    def __init__(self, pin, real) -> None:
        self.path = "/nonexistent/socket"

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


def _arm(fx: dict, name: str, root: Path) -> Policy:
    d = root / f"skills-{name}"
    subprocess.run(["git", "clone", "-q", fx["skills_repo"], str(d)], check=True)
    subprocess.run(["git", "-C", str(d), "checkout", "-q", "--detach", fx[name]], check=True)
    return Policy(name, d)


def test_the_fixture_eval_runs_end_to_end_in_process(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "Broker", _NoBroker)
    monkeypatch.setattr(replay, "check_shim", lambda shim, profile: "")
    fx = replayfake.build_fixture(tmp_path / "fx")
    cfg = cfg_mod.load(Path(fx["config"]))
    arms = [_arm(fx, "v11", tmp_path), _arm(fx, "v12", tmp_path)]
    out = tmp_path / "out"
    reports: dict[str, dict] = {}

    def runner(job: Job, out_: Path) -> int:
        rep = replay.replay(
            cfg,
            run_id=job.item.run_id,
            out=out_ / job.policy.name,
            skills=job.policy.skills,
            source_db=Path(fx["db"]),
            force_phase2=True,
            sample=job.sample,
            policy_name=job.policy.name,
            gh_token="ro",
            lane_runner=replayfake.FakeLanes(),
        )
        reports[f"{job.policy.name}/{job.name}"] = rep
        return 0 if rep["status"] == "done" else 1

    res = run_set(load_set(Path(fx["set"])), arms, out, concurrency=2, runner=runner)
    assert res["ok"] == 6, {k: (r["status"], r["reason"]) for k, r in reports.items()}
    v12 = reports[f"v12/run-{fx['source_runs'][0]}"]
    assert (
        v12["seeded"]["ledger"]["issues"] == 1
        and "ledger" not in reports[f"v11/run-{fx['source_runs'][0]}"]["seeded"]
    )
    first, todo = replayscore.score_dir(Path(fx["set"]), Path(fx["labels"]), out, baseline="v11")
    assert first["heads"]["paired"] == 2 and {t["kind"] for t in todo} == {
        "new_blocker",
        "label_match",
    }
    jres = judge_mod.judge(cfg, out, lane_runner=replayfake.FakeLanes(), min_free_gb=0)
    assert jres["judged"] == len(todo) and not jres["failed"]
    final, todo = replayscore.score_dir(Path(fx["set"]), Path(fx["labels"]), out, baseline="v11")
    assert todo == [] and final["judged"] == jres["judged"]
    for arm in ("v11", "v12"):
        assert final["sets"]["S1"][arm]["recall"] == 1.0
        # the judge matched the f3 finding to the fp label
        assert final["sets"]["S4"][arm]["fp_blockers"] == 1.0
        assert final["arms"][arm]["precision"]["new_blocker_precision"] == 1.0
    assert (out / "spot_check.jsonl").exists()
    assert not [p for p in out.glob("*/run-*/work/*") if p.name in ("worktrees", "mirrors")]
    calls = [json.loads(x) for x in (Path(fx["root"]) / "gh-calls.log").read_text().splitlines()]
    assert calls and all(c[c.index("--method") + 1] == "GET" for c in calls if "--method" in c)


def test_the_fake_gh_refuses_writes(tmp_path):
    state = tmp_path / "gh.json"
    state.write_text(json.dumps({"prs": {"1": {"number": 1}}}))
    post = ["api", "repos/o/r/issues/1/comments", "--method", "POST"]
    assert replayfake.gh_main(post, state) == 1
    mutation = ["api", "graphql", "-f", "query=mutation { x }"]
    assert replayfake.gh_main(mutation, state) == 1
    assert replayfake.gh_main(["api", "repos/o/r/pulls/1", "--method", "GET"], state) == 0


# ---- deploy/eval-run.sh ----


def _script(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env={**os.environ, "REVIEWSYS_EVAL_PYTHON": sys.executable, **env},
        timeout=600,
        check=False,
    )


@pytest.fixture
def prod(tmp_path) -> dict:
    """The fixture as the box would see it: a production dir with its config."""
    fx = replayfake.build_fixture(tmp_path / "fx")
    p = tmp_path / "prod"
    p.mkdir()
    shutil.copy(fx["config"], p / "config.toml")
    return {**fx, "prod": str(p), "eval_root": str(tmp_path / "eval")}


def _real_args(fx: dict) -> list[str]:
    return [
        "--v12", fx["v12"], "--v11", fx["v11"], "--set", fx["set"], "--labels", fx["labels"],
        "--eval-root", fx["eval_root"], "--skills-repo", fx["skills_repo"],
    ]  # fmt: skip


def test_the_script_refuses_without_a_read_only_token(prod):
    env = {"REVIEWSYS_PROD": prod["prod"], "REVIEWSYS_REPLAY_GH_TOKEN": ""}
    r = _script(_real_args(prod), env)
    assert r.returncode == 1 and "--allow-bot-token" in r.stdout + r.stderr
    assert not (Path(prod["eval_root"]) / "source.db").exists()  # nothing was started
    r = _script([*_real_args(prod), "--allow-bot-token", "--only", "preflight"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "WARNING: --allow-bot-token" in r.stdout


def test_the_script_refuses_in_degraded_mode(prod):
    import sqlite3

    c = sqlite3.connect(prod["db"])
    c.execute("INSERT INTO kv (key, value) VALUES ('degraded.force', 'on')")
    c.commit()
    c.close()
    env = {"REVIEWSYS_PROD": prod["prod"], "REVIEWSYS_REPLAY_GH_TOKEN": "ro"}
    r = _script([*_real_args(prod), "--only", "preflight"], env)
    assert r.returncode == 1 and "degraded mode is active" in r.stdout + r.stderr


def test_the_script_snapshots_checks_the_set_and_checks_out_the_arms(prod):
    os.remove(Path(prod["runs"]) / f"run-{prod['source_runs'][1]}" / "evidence.json")
    env = {"REVIEWSYS_PROD": prod["prod"], "REVIEWSYS_REPLAY_GH_TOKEN": "ro"}
    r = _script([*_real_args(prod), "--only", "preflight,snapshot,arms"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    root = Path(prod["eval_root"])
    assert (root / "source.db").exists()
    assert f'"missing_evidence": [\n  {prod["source_runs"][1]}\n ]' in r.stdout
    for arm in ("v11", "v12"):
        head = subprocess.run(
            ["git", "-C", str(root / "skills" / arm), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert head == prod[arm]
    assert list((root / "logs").glob("eval-run-*.log"))


def _can_run_the_broker_and_sandbox() -> str:
    """Why the dry run cannot run here ("" = it can): it binds the replay broker's Unix
    socket and nests the lane sandbox, which another sandbox may refuse."""
    d = tempfile.mkdtemp(prefix="rs-")
    try:
        s = socket.socket(socket.AF_UNIX)
        try:
            s.bind(os.path.join(d, "s"))
        except OSError as exc:
            return f"cannot bind a Unix socket: {exc}"
        finally:
            s.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)
    if os.path.exists(lane_mod.SANDBOX_EXEC):
        r = subprocess.run(
            [lane_mod.SANDBOX_EXEC, "-p", "(version 1) (allow default)", "/usr/bin/true"],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            return f"cannot nest the lane sandbox: {r.stderr.strip()[:100]}"
    return ""


def test_the_script_dry_run_exercises_every_step(tmp_path):
    why = _can_run_the_broker_and_sandbox()
    if why:
        pytest.skip(why)
    r = _script(["--dry", "--eval-root", str(tmp_path / "eval"), "--min-free-gb", "0"], {})
    assert r.returncode == 0 and "DRY RUN OK" in r.stdout, r.stdout[-3000:] + r.stderr[-2000:]
    again = _script(["--dry", "--eval-root", str(tmp_path / "eval"), "--min-free-gb", "0"], {})
    assert again.returncode == 0 and "0 to run, 6 already done" in again.stdout
