"""Policy v12 lane hygiene: isolated lane environments, structured output through
`--json-schema`, and echo fields the harness stamps. All of it is off under a v11 policy,
whose lanes must start exactly as v0.24's did (pinned below, end to end through the real
`run_claude_lane`)."""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

import pytest

from reviewsys import lane as lane_mod
from reviewsys import worker
from reviewsys.db import tx
from reviewsys.ingest import enqueue_head
from reviewsys.lane import LaneResult, LaneSpec
from reviewsys.models import RunStatus, Trigger
from reviewsys.scheduler import schedule
from reviewsys.steps import worktree as wt
from tests.test_pipeline_policy import load_with

HEAD = "a" * 40
V024_ENV = {"GODEBUG": "netdns=go", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
ISOLATION = {"CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1", "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS": "1"}


@pytest.fixture(autouse=True)
def fake_git(monkeypatch, tmp_path):
    monkeypatch.setattr(wt, "ensure_mirror", lambda mirrors, repo: tmp_path / "mirror")
    monkeypatch.setattr(wt, "fetch_head", lambda mirror, number, sha: None)

    def create(mirror, wts, name, sha):
        p = wts / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(wt, "create_worktree", create)
    monkeypatch.setattr(wt, "remove_worktree", lambda mirror, path: None)
    monkeypatch.setattr(wt, "merge_base", lambda worktree, base, sha: "b" * 40)


def v024_argv(spec: LaneSpec) -> list[str]:
    """`lane.argv_for` as v0.24.0 shipped it, copied rather than imported: what a v11 lane's
    command line must stay."""
    sandbox = [lane_mod.SANDBOX_EXEC, "-p", spec.sandbox_profile] if spec.sandbox_profile else []
    if spec.resume and spec.session_id:
        session = ["--resume", spec.session_id]
    elif spec.session_id:
        session = ["--session-id", spec.session_id]
    else:
        session = ["--no-session-persistence"]
    argv = [
        *sandbox,
        "nice",
        "-n",
        "10",
        spec.claude_bin,
        "--bare",
        "--settings",
        '{"permissions": {"disableAutoMode": "disable"}}',
        "--permission-mode",
        "plan",
        "--model",
        spec.model,
        "--effort",
        spec.effort,
        "--add-dir",
        str(spec.add_dir),
        "--output-format",
        "json",
        *session,
        "--print",
    ]
    if spec.max_budget_usd:
        argv += ["--max-budget-usd", f"{spec.max_budget_usd:.2f}"]
    return argv


class ClaudeProcesses:
    """A lane runner that is the real `run_claude_lane`, with only the `claude` process faked:
    it records each lane's argv and environment exactly as Popen gets them, and answers with
    the canned output of `lanes` (conftest.FakeLanes) for that lane's spec."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, lanes) -> None:
        self.lanes = lanes
        self.started: list[tuple[LaneSpec, list[str], dict[str, str]]] = []
        self._lane = threading.local()
        real = subprocess.Popen
        outer = self

        class Proc:
            def __init__(self, argv, *, env, **kw):
                self.real = None
                if not hasattr(outer._lane, "spec"):
                    self.real = real(argv, env=env, **kw)
                    return
                spec, art, worktree = outer._lane.spec, outer._lane.art, outer._lane.worktree
                outer.started.append((spec, list(argv), dict(env)))
                res = outer.lanes(spec, art, worktree)
                self.out, self.err, self.returncode = res.stdout, res.stderr, res.exit_code

            def communicate(self, input=None, timeout=None):
                if self.real is not None:
                    return self.real.communicate(input, timeout=timeout)
                return self.out, self.err

        monkeypatch.setattr(lane_mod.subprocess, "Popen", Proc)

    def __call__(self, spec: LaneSpec, art: Path, worktree: Path) -> LaneResult:
        self._lane.spec, self._lane.art, self._lane.worktree = spec, art, worktree
        try:
            return lane_mod.run_claude_lane(spec, art, worktree)
        finally:
            del self._lane.spec


def _run(cfg, conn, gh, runner):
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.MENTION)
    (rid,) = schedule(conn, cfg, spawn=False)
    return rid, worker.main(cfg, conn, rid, gh=gh, lane_runner=runner, heartbeat=False)


def _verifier():
    return {
        "summary": "ok",
        "review_action": "COMMENT",
        "findings": [],
        "dropped_findings": [],
        "out_of_scope_findings": [],
        "coderabbit_reactions": [],
        "prerequisite_adjudications": [],
        "adjudication_complete": True,
    }


def _review(lanes) -> None:
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier()
    lanes.broken_once = {"general"}  # one correction turn, resumed in the lane's session


# ---- v11: byte-identical lanes ----


def test_v11_lanes_start_exactly_as_v024(cfg, conn, gh, lanes, monkeypatch, _claude_config_dir):
    _review(lanes)
    procs = ClaudeProcesses(monkeypatch, lanes)
    _, status = _run(cfg, conn, gh, procs)
    assert status == RunStatus.DONE
    roles = {s.role for s, _, _ in procs.started}
    assert {"prep", "general", "verifier"} <= roles
    assert any(s.resume for s, _, _ in procs.started), "a correction turn ran"
    for spec, argv, env in procs.started:
        assert argv == v024_argv(spec), spec.role
        assert env == {**os.environ, **V024_ENV}, spec.role
        assert not spec.isolate_env


# ---- lane_env_isolation ----


def test_isolated_runner_marks_every_lane_and_only_when_on(tmp_path):
    seen: list[LaneSpec] = []

    def runner(spec, art, worktree):
        seen.append(spec)
        return LaneResult(exit_code=0, stdout="", stderr="", duration_s=0)

    assert lane_mod.isolated(runner, False) is runner
    spec = LaneSpec(
        role="general",
        agent="a",
        model="m",
        effort="high",
        prompt="p",
        cwd=tmp_path,
        add_dir=tmp_path,
        timeout_seconds=60,
        claude_bin="claude",
    )
    lane_mod.isolated(runner, True)(spec, tmp_path, tmp_path)
    assert seen[0].isolate_env and seen[0] == LaneSpec(**{**_fields(spec), "isolate_env": True})
    assert lane_mod.lane_env(seen[0]) == {**os.environ, **V024_ENV, **ISOLATION}
    assert lane_mod.lane_env(spec) == {**os.environ, **V024_ENV}


def _fields(spec: LaneSpec) -> dict:
    return {f: getattr(spec, f) for f in LaneSpec.__dataclass_fields__}


def test_v12_isolation_reaches_every_lane_and_correction_turn(
    cfg, conn, gh, lanes, monkeypatch, skills_dir, _claude_config_dir
):
    """The correction turn resumes the lane's session under the same CLAUDE_CONFIG_DIR, with
    the same isolated environment, and nothing else about the command line changes."""
    cfg = load_with(cfg, skills_dir, {"version": "v12", "lane_env_isolation": True})
    _review(lanes)
    procs = ClaudeProcesses(monkeypatch, lanes)
    _, status = _run(cfg, conn, gh, procs)
    assert status == RunStatus.DONE
    resumed = [(s, env) for s, _, env in procs.started if s.resume]
    assert resumed, "a correction turn ran"
    for spec, argv, env in procs.started:
        assert spec.isolate_env, spec.role
        assert env == {**os.environ, **V024_ENV, **ISOLATION}, spec.role
        assert env["CLAUDE_CONFIG_DIR"] == str(_claude_config_dir)
        assert argv == v024_argv(spec), "isolation is environment only"
    first = next(s for s, _, _ in procs.started if s.session_id == resumed[0][0].session_id)
    assert first is not resumed[0][0] and not first.resume


def test_v12_without_isolation_keeps_the_v11_environment(
    cfg, conn, gh, lanes, monkeypatch, skills_dir, _claude_config_dir
):
    cfg = load_with(cfg, skills_dir, {"version": "v12"})
    _review(lanes)
    procs = ClaudeProcesses(monkeypatch, lanes)
    _, status = _run(cfg, conn, gh, procs)
    assert status == RunStatus.DONE
    assert all(env == {**os.environ, **V024_ENV} for _, _, env in procs.started)
