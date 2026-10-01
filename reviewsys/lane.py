"""Run one model lane via the `claude` CLI in a worktree and capture structured output."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contract import parse_json_object
from .models import FailKind, ReviewError

LaneRunner = Callable[["LaneSpec", Path, Path], "LaneResult"]
STOP_POLL_SECONDS = 5  # how often a running lane checks `LaneSpec.should_stop`
PIPE_GRACE_SECONDS = 10  # after claude exits, how long a child may hold its pipes open
LANE_NICE = 10  # claude lanes run at low scheduling priority so CLIProxyAPI is never starved
# Plan mode otherwise enables Claude Code's separate LLM permission classifier.
# Scope the opt-out to Reviewsys; retain plan mode and ordinary permission rules.
CLAUDE_SETTINGS = json.dumps({"permissions": {"disableAutoMode": "disable"}})
SANDBOX_EXEC = "/usr/bin/sandbox-exec"


@dataclass(frozen=True, slots=True)
class LaneSpec:
    role: str
    agent: str
    model: str
    effort: str
    prompt: str
    cwd: Path
    add_dir: Path
    timeout_seconds: int
    claude_bin: str
    max_budget_usd: float | None = None
    # polled while the lane runs: True stops it (the run was cancelled, or a sibling lane of
    # the same phase failed and the phase is being abandoned)
    should_stop: Callable[[], bool] | None = None
    parallel: bool = False  # a reviewer lane running beside its siblings (lanepool lines these up)
    # the lane-slot pool when it is not the model family's (comparison lanes: lanepool.COMPARE_POOL)
    pool: str | None = None
    # macOS sandbox profile the lane runs under ("" = none); see `exec_deny_profile`
    sandbox_profile: str = ""


@dataclass(slots=True)
class LaneResult:
    exit_code: int | None
    stdout: str
    stderr: str
    duration_s: float
    tokens_in: int | None = None
    tokens_out: int | None = None
    timed_out: bool = False
    result_text: str = ""
    cost_usd: float | None = None
    turns: int | None = None
    subtype: str | None = None
    cancelled: bool = False  # stopped by `LaneSpec.should_stop`, not by the model or a timeout
    started: bool = True  # False: stopped while waiting for a lane slot, never ran

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.cancelled

    @property
    def api_error(self) -> str:
        """The upstream error Claude Code reported for a failed call ("API Error: 429 ...").
        It lands in the envelope's `result` with `is_error`, not on stderr. Only set when the
        lane failed and the text is Claude Code's own API-error prefix, so it is never model
        output. "" otherwise."""
        text = self.result_text.strip()
        if self.exit_code != 0 and text.startswith("API Error"):
            return text.splitlines()[0][:500]
        return ""

    @property
    def infra_error(self) -> str:
        """Infrastructure error text for quota detection and disclosure: the API error when
        there is one, else stderr (a launcher crash). Never model output."""
        return "\n".join(x for x in (self.api_error, self.stderr or "") if x)

    @property
    def first_stderr_line(self) -> str:
        """The lane's first meaningful infrastructure error line (an upstream 429, a launcher
        crash), never model output. "" when it said nothing. Claude Code's own diagnostics
        (`[claude-code:...]`, printed for every proxied model) are skipped."""
        if self.api_error:
            return self.api_error
        for line in (self.stderr or "").splitlines():
            s = line.strip()
            if s and not s.startswith("[claude-code:"):
                return s
        return ""


def exec_deny_profile(paths: tuple[str, ...]) -> str:
    """A sandbox profile that allows everything except executing anything under `paths`
    (compilers, build systems, test runners), or "" when there is nothing to deny or no
    `sandbox-exec` (Linux CI). It is enforced by the kernel on every process the lane starts,
    so `./build_ios.sh` running cargo is stopped as surely as cargo itself. Each path is
    listed as given and resolved, so a symlink in the list still matches what exec sees."""
    if not paths or not os.path.exists(SANDBOX_EXEC):
        return ""
    seen: list[str] = []
    for raw in paths:
        p = os.path.expanduser(raw)
        for q in (p, os.path.realpath(p)):
            if q not in seen:
                seen.append(q)
    rules = " ".join(
        '(subpath "{}")'.format(q.replace("\\", "\\\\").replace('"', '\\"')) for q in seen
    )
    return f"(version 1) (allow default) (deny process-exec {rules})"


def sandbox_problem(profile: str) -> str:
    """Why `profile` cannot be applied ("" = it can): a malformed profile, or a worker that is
    itself inside a sandbox that refuses nesting ("sandbox_apply: Operation not permitted")."""
    try:
        r = subprocess.run(
            [SANDBOX_EXEC, "-p", profile, "/usr/bin/true"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    return "" if r.returncode == 0 else (r.stderr.strip() or f"exit {r.returncode}")[:300]


def sandboxed(runner: LaneRunner, profile: str) -> LaneRunner:
    """`runner`, with every lane under `profile` (no change when it is ""). The profile is
    tried once, before the first lane: a sandbox that cannot start would otherwise fail every
    lane with an error that reads like the model's own."""
    if not profile:
        return runner
    checked: list[str] = []

    def run(spec: LaneSpec, artifact_dir: Path, worktree: Path) -> LaneResult:
        if not checked:
            checked.append(sandbox_problem(profile))
        if checked[0]:
            raise ReviewError(FailKind.INFRA, f"lane sandbox cannot start: {checked[0]}")
        return runner(dataclasses.replace(spec, sandbox_profile=profile), artifact_dir, worktree)

    return run


def argv_for(spec: LaneSpec) -> list[str]:
    """The lane's command line, under `nice`: review lanes are batch work and the proxies they
    talk to must win the CPU. (Not `preexec_fn=os.nice`: that runs Python in the forked child,
    which can deadlock now that a worker starts lanes from several threads.)"""
    sandbox = [SANDBOX_EXEC, "-p", spec.sandbox_profile] if spec.sandbox_profile else []
    argv = [
        *sandbox,
        "nice",
        "-n",
        str(LANE_NICE),
        spec.claude_bin,
        "--bare",
        "--settings",
        CLAUDE_SETTINGS,
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
        "--no-session-persistence",
        "--print",
    ]
    if spec.max_budget_usd:
        argv += ["--max-budget-usd", f"{spec.max_budget_usd:.2f}"]
    return argv


def run_claude_lane(spec: LaneSpec, artifact_dir: Path, _unused: Path) -> LaneResult:
    """Spawn `claude` in the worktree with the prompt on stdin. Same process group as the worker
    so a group kill takes the lane down with us."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "prompt.md").write_text(spec.prompt, encoding="utf-8")
    env = {**os.environ, "GODEBUG": "netdns=go", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
    t0 = time.monotonic()
    proc = subprocess.Popen(
        argv_for(spec),
        cwd=str(spec.cwd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    timed_out = cancelled = False
    deadline = t0 + spec.timeout_seconds
    stdin: str | None = spec.prompt
    exited_at: float | None = None
    while True:
        try:
            # a retried communicate() loses no output; the input is only sent the first time
            out, err = proc.communicate(
                stdin, timeout=max(0.0, min(STOP_POLL_SECONDS, deadline - time.monotonic()))
            )
            break
        except subprocess.TimeoutExpired as exc:
            stdin = None
            if proc.poll() is not None:
                # claude is done but a tool it started still holds the pipes: keep what it
                # wrote rather than waiting out the deadline for an EOF that is not coming
                exited_at = exited_at or time.monotonic()
                if time.monotonic() - exited_at >= PIPE_GRACE_SECONDS:
                    out, err = _decoded(exc)
                    _close_pipes(proc)
                    break
                continue
            cancelled = bool(spec.should_stop and spec.should_stop())
            timed_out = not cancelled and time.monotonic() >= deadline
            if cancelled or timed_out:
                out, err = _terminate(proc, exc)
                break
    dur = time.monotonic() - t0
    (artifact_dir / "stdout.json").write_text(out or "", encoding="utf-8")
    (artifact_dir / "stderr.txt").write_text(err or "", encoding="utf-8")
    res = LaneResult(
        exit_code=proc.returncode,
        stdout=out or "",
        stderr=err or "",
        duration_s=dur,
        timed_out=timed_out,
        cancelled=cancelled,
    )
    _extract_result(res)
    (artifact_dir / "lane-meta.json").write_text(
        json.dumps(
            {
                "exit_code": res.exit_code,
                "duration_s": round(dur, 1),
                "timed_out": timed_out,
                "cancelled": cancelled,
                "tokens_in": res.tokens_in,
                "tokens_out": res.tokens_out,
                "cost_usd": res.cost_usd,
                "turns": res.turns,
                "subtype": res.subtype,
                "argv": argv_for(spec),
            },
            indent=1,
        )
    )
    return res


def _decoded(exc: subprocess.TimeoutExpired) -> tuple[str, str]:
    """The output a timed-out communicate() had collected so far (it is cumulative)."""

    def text(b: bytes | str | None) -> str:
        return b.decode("utf-8", "replace") if isinstance(b, bytes) else (b or "")

    return text(exc.output), text(exc.stderr)


def _close_pipes(proc: subprocess.Popen[str]) -> None:
    for pipe in (proc.stdout, proc.stderr):
        if pipe is not None:
            pipe.close()
    proc.wait()


def _terminate(proc: subprocess.Popen[str], last: subprocess.TimeoutExpired) -> tuple[str, str]:
    """SIGTERM, then SIGKILL, and whatever output there is. A tool the lane started (a build,
    a test run) can outlive it and keep the pipes open, so the last wait is bounded too:
    the lane is stopped either way, its partial output is only diagnostics."""
    proc.send_signal(signal.SIGTERM)
    try:
        return proc.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    try:
        return proc.communicate(timeout=15)
    except subprocess.TimeoutExpired as exc:
        _close_pipes(proc)
        (out, err), (last_out, last_err) = _decoded(exc), _decoded(last)
        return out or last_out, err or last_err


def _extract_result(res: LaneResult) -> None:
    """claude --output-format json wraps the model's text in {"result": ..., "usage": {...}}.
    Error envelopes (budget exhausted, max turns) carry usage/cost but no `result`."""
    text = res.stdout.strip()
    if not text:
        return
    try:
        env = json.loads(text)
    except json.JSONDecodeError:
        res.result_text = text
        return
    if not isinstance(env, dict) or ("result" not in env and env.get("type") != "result"):
        res.result_text = text
        return
    res.result_text = str(env.get("result") or "")
    usage = env.get("usage") or {}
    if isinstance(usage, dict):
        res.tokens_in = int(usage.get("input_tokens") or 0) + int(
            usage.get("cache_read_input_tokens") or 0
        )
        res.tokens_out = int(usage.get("output_tokens") or 0)
    cost = env.get("total_cost_usd")
    if isinstance(cost, int | float):
        res.cost_usd = round(float(cost), 4)
    turns = env.get("num_turns")
    if isinstance(turns, int):
        res.turns = turns
    subtype = env.get("subtype")
    if subtype:
        res.subtype = str(subtype)
    if env.get("is_error"):
        res.exit_code = res.exit_code or 1


def lane_output(res: LaneResult) -> dict[str, Any]:
    if res.cancelled:
        raise ReviewError(FailKind.INFRA, "lane stopped before it finished")
    if res.timed_out:
        raise ReviewError(FailKind.INFRA, "lane timed out")
    if res.exit_code != 0:
        if res.subtype and res.subtype.startswith("error_"):
            raise ReviewError(
                FailKind.INFRA,
                f"lane {res.subtype.removeprefix('error_')} after {res.turns or '?'} turns, ${res.cost_usd or '?'}",
            )
        raise ReviewError(
            FailKind.INFRA,
            f"lane exit {res.exit_code}: {(res.first_stderr_line or res.result_text)[:200]}",
        )
    return parse_json_object(res.result_text)


def prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
