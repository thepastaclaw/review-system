"""Review lanes are static: an exec-deny sandbox around every lane, CI results as evidence."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from reviewsys import config as cfg_mod
from reviewsys import github, lane
from reviewsys.lane import LaneSpec, argv_for, exec_deny_profile, sandboxed

darwin_only = pytest.mark.skipif(
    sys.platform != "darwin" or not os.path.exists(lane.SANDBOX_EXEC), reason="needs sandbox-exec"
)


def _spec(tmp_path: Path, **kw) -> LaneSpec:
    return LaneSpec(
        role="general",
        agent="a",
        model="m",
        effort="high",
        prompt="p",
        cwd=tmp_path,
        add_dir=tmp_path,
        timeout_seconds=1,
        claude_bin="claude",
        **kw,
    )


def test_argv_runs_the_lane_under_its_sandbox_profile(tmp_path):
    plain = argv_for(_spec(tmp_path))
    boxed = argv_for(_spec(tmp_path, sandbox_profile="(version 1)"))
    assert plain[0] == "nice"
    assert boxed[:3] == [lane.SANDBOX_EXEC, "-p", "(version 1)"] and boxed[3:] == plain


def test_sandboxed_sets_the_profile_on_every_spec(tmp_path):
    seen: list[str] = []

    def runner(spec, art, wt):
        seen.append(spec.sandbox_profile)
        return lane.LaneResult(exit_code=0, stdout="", stderr="", duration_s=0)

    sandboxed(runner, "P")(_spec(tmp_path), tmp_path, tmp_path)
    assert seen == ["P"]
    assert sandboxed(runner, "") is runner


def test_a_sandbox_that_cannot_start_fails_the_lane_as_infra(tmp_path, monkeypatch):
    probes: list[str] = []

    def problem(profile: str) -> str:
        probes.append(profile)
        return "sandbox_apply: Operation not permitted"

    monkeypatch.setattr(lane, "sandbox_problem", problem)
    run = sandboxed(lambda *a: pytest.fail("lane must not start"), "P")
    for _ in range(2):
        with pytest.raises(lane.ReviewError) as exc:
            run(_spec(tmp_path), tmp_path, tmp_path)
        assert exc.value.kind == lane.FailKind.INFRA and "cannot start" in exc.value.message
    assert probes == ["P"]  # tried once per worker, not per lane


def test_no_profile_without_paths_or_without_sandbox_exec(monkeypatch):
    assert exec_deny_profile(()) == ""
    monkeypatch.setattr(lane, "SANDBOX_EXEC", "/nonexistent/sandbox-exec")
    assert exec_deny_profile(("/usr/bin/make",)) == ""


@darwin_only
def test_profile_lists_given_and_resolved_paths_quoted(tmp_path):
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / 'li"nk'
    link.symlink_to(target)
    p = exec_deny_profile((str(link),))
    assert p.startswith("(version 1) (allow default) (deny process-exec ")
    assert f'(subpath "{os.path.realpath(target)}")' in p
    assert '\\"' in p  # the quote in the link name is escaped


@darwin_only
def test_the_kernel_stops_a_denied_tool_even_from_a_script(tmp_path):
    """The point of an OS sandbox over permission rules: `sh build.sh` running a denied tool
    is stopped too, while ordinary tools (git, the shell, python) keep working."""
    script = tmp_path / "build.sh"
    script.write_text("#!/bin/sh\n/usr/bin/make --version\n")
    profile = exec_deny_profile(("/usr/bin/make",))

    def sh(cmd: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [lane.SANDBOX_EXEC, "-p", profile, "/bin/sh", "-c", cmd],
            capture_output=True,
            text=True,
            check=False,
        )

    if "sandbox_apply" in sh("true").stderr:
        pytest.skip("already inside a sandbox that forbids nesting one")
    blocked = sh(f"/bin/sh {script}")
    assert blocked.returncode != 0 and "not permitted" in blocked.stderr.lower()
    assert "sandbox_apply" not in blocked.stderr
    assert sh("/usr/bin/git --version").returncode == 0


def test_deny_list_defaults_and_overrides(tmp_path, skills_dir):
    toml = cfg_mod.DEFAULT_TOML.replace('skills = "~/Projects/skills"', f'skills = "{skills_dir}"')
    p = tmp_path / "config.toml"
    p.write_text(toml)
    assert cfg_mod.load(p).lane_deny_exec == cfg_mod.DEFAULT_LANE_DENY_EXEC
    assert "~/.rustup" in cfg_mod.DEFAULT_LANE_DENY_EXEC
    p.write_text(toml.replace("[lanes]\n", '[lanes]\ndeny_exec = ["/opt/x"]\n'))
    assert cfg_mod.load(p).lane_deny_exec == ("/opt/x",)
    p.write_text(toml.replace("[lanes]\n", "[lanes]\ndeny_exec = []\n"))
    assert cfg_mod.load(p).lane_deny_exec == ()


def test_ci_checks_merges_check_runs_and_statuses(gh):
    gh.routes["repos/o/r/commits/abc/check-runs?per_page=100"] = {
        "check_runs": [
            {"name": "build", "status": "completed", "conclusion": "failure", "html_url": "u1"}
        ]
    }
    gh.routes["repos/o/r/commits/abc/status?per_page=100"] = {
        "statuses": [{"context": "lint", "state": "pending", "target_url": "u2"}]
    }
    ci = github.ci_checks(gh, "o/r", "abc")
    assert ci["checks"] == [
        {"name": "build", "status": "completed", "conclusion": "failure", "url": "u1"},
        {"name": "lint", "status": "in_progress", "conclusion": "pending", "url": "u2"},
    ]


@pytest.mark.parametrize(
    "bad", ['"~/.rustup"', '[""]', '["~"]', '["/"]', '["relative/path"]', "[1]"]
)
def test_a_bad_deny_list_is_rejected_not_turned_into_deny_everything(tmp_path, skills_dir, bad):
    toml = cfg_mod.DEFAULT_TOML.replace('skills = "~/Projects/skills"', f'skills = "{skills_dir}"')
    p = tmp_path / "config.toml"
    p.write_text(toml.replace("[lanes]\n", f"[lanes]\ndeny_exec = {bad}\n"))
    with pytest.raises(ValueError, match="deny_exec"):
        cfg_mod.load(p)


def test_ci_checks_says_when_github_has_more(gh):
    gh.routes["repos/o/r/commits/abc/check-runs?per_page=100"] = {
        "total_count": 140,
        "check_runs": [{"name": "a", "status": "completed", "conclusion": "success"}],
    }
    gh.routes["repos/o/r/commits/abc/status?per_page=100"] = {"total_count": 0, "statuses": []}
    ci = github.ci_checks(gh, "o/r", "abc")
    assert ci["truncated"] is True and ci["fetched_at"]


def test_ci_checks_never_fails_the_run(gh):
    gh.fail_next = ["HTTP 502: Bad Gateway"]
    assert "error" in github.ci_checks(gh, "o/r", "abc")
