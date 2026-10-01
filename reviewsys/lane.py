"""Run one model lane via the `claude` CLI in a worktree and capture structured output."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
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
# A correction turn only re-emits an answer the lane already worked out: bounded well below
# the lane timeout, so two of them cannot triple a lane's wall clock
CORRECTION_TIMEOUT_SECONDS = 30 * 60
# (artifact dir, spec, "running" | None): how `run_with_corrections` tells the slot holder
# (lanepool.gated) that a correction turn started or ended abnormally, for the status page
TurnMark = Callable[[Path, "LaneSpec", str | None], None]


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
    # Correction turns (see `run_with_corrections`). `session_id` (a UUID) makes the lane keep
    # its Claude Code session under that id instead of `--no-session-persistence`; with
    # `resume` the lane continues that session, `prompt` being the next user turn. `check` looks
    # at a finished turn and returns the follow-up prompt when its output must be corrected
    # (None: accept it); at most `corrections` follow-ups run.
    session_id: str = ""
    resume: bool = False
    check: Callable[[LaneResult], str | None] | None = None
    corrections: int = 0


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
    session_id: str | None = None  # the envelope's, when the CLI reported one
    # the correction turns run in this lane's session and slot after it, in order
    followups: list[LaneResult] = field(default_factory=list)

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


def session_args(spec: LaneSpec) -> list[str]:
    """Verified with Claude Code 2.1.286 (2026-10-01): `--bare --print --output-format json
    --session-id <uuid>` saves the session under that id (the envelope's `session_id` echoes
    it), and `--resume <uuid>` with the same flags continues it with the full earlier context
    (prompt cache included), from any cwd, keeping the id. A missing session exits 1 with "No
    conversation found with session ID" on stderr. Every lane without a session id keeps
    `--no-session-persistence`: nothing to clean up."""
    if spec.resume and spec.session_id:
        return ["--resume", spec.session_id]
    if spec.session_id:
        return ["--session-id", spec.session_id]
    return ["--no-session-persistence"]


def new_session_id() -> str:
    return str(uuid.uuid4())


def session_files(session_id: str) -> list[Path]:
    """Where Claude Code keeps a session: `<config dir>/projects/<cwd slug>/<id>.jsonl` (plus a
    `<id>/` directory for tool output it spilled), the config dir being `$CLAUDE_CONFIG_DIR` or
    `~/.claude` as the lane inherits it. Lanes run in per-run worktrees, so the slug is new per
    run; the id is looked up in every slug rather than re-deriving Claude Code's slug rules."""
    try:
        uuid.UUID(session_id)
    except ValueError:
        return []  # never glob with anything but a UUID
    projects = claude_projects_dir()
    if not projects.is_dir():
        return []
    return [*projects.glob(f"*/{session_id}.jsonl"), *projects.glob(f"*/{session_id}")]


def claude_projects_dir() -> Path:
    root = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude").expanduser()
    return root / "projects"


def project_slug(cwd: Path) -> str:
    """Claude Code's project directory name for a cwd: the resolved path with every
    character but ASCII letters and digits turned into `-` (2.1.286: /private/tmp/x/wt ->
    -private-tmp-x-wt)."""
    return re.sub(r"[^A-Za-z0-9]", "-", str(cwd.resolve()))


def forget_session(session_id: str) -> None:
    """Delete a lane's saved session once its corrections are over. A transcript holds every
    tool result of the lane (megabytes for a long review) and nothing reads it again: the
    lane's artifacts keep the prompts and answers. The per-worktree project directory stays
    (the run's other lanes share it); gc removes it once empty, and any leftover."""
    for p in session_files(session_id):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            with contextlib.suppress(OSError):
                p.unlink()


def run_with_corrections(
    runner: LaneRunner,
    spec: LaneSpec,
    artifact_dir: Path,
    worktree: Path,
    mark: TurnMark | None = None,
) -> LaneResult:
    """Run the lane, then up to `spec.corrections` correction turns in its own session while
    `spec.check` asks for one: the model that wrote the answer, with everything it read still
    in context, is told what was wrong and answers again. Each turn is a whole `runner` call
    (same model, effort, launcher, sandbox, stop polling), with its own artifacts in
    `correction-<n>/` and a timeout of at most CORRECTION_TIMEOUT_SECONDS. Called by the slot
    holder (lanepool.gated), so the corrections run in the lane's slot: never released, never
    lined up again. A turn that did not finish (failed, timed out, stopped) ends the
    corrections; the caller reads every turn in `followups` and decides. The saved session is
    deleted at the end however it went."""
    sessions = {spec.session_id} if spec.session_id else set()
    try:
        res = runner(spec, artifact_dir, worktree)
        if spec.session_id and res.session_id:
            sessions.add(res.session_id)
        turn = res
        while spec.check is not None and spec.session_id and len(res.followups) < spec.corrections:
            if not turn.ok:
                break
            prompt = spec.check(turn)
            if prompt is None or (spec.should_stop is not None and spec.should_stop()):
                break
            n = len(res.followups) + 1
            follow = dataclasses.replace(
                spec,
                prompt=prompt,
                session_id=turn.session_id or spec.session_id,
                resume=True,
                check=None,
                corrections=0,
                timeout_seconds=min(spec.timeout_seconds, CORRECTION_TIMEOUT_SECONDS),
            )
            turn_dir = artifact_dir / f"correction-{n}"
            if mark is not None:
                mark(turn_dir, follow, "running")
            try:
                turn = runner(follow, turn_dir, worktree)
            except BaseException:
                if mark is not None:
                    mark(turn_dir, follow, None)
                raise
            res.followups.append(turn)
    finally:
        for sid in sessions:
            forget_session(sid)
    return res


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
        *session_args(spec),
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
                "session_id": res.session_id if spec.session_id else None,
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
    sid = env.get("session_id")
    if isinstance(sid, str) and sid:
        res.session_id = sid
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
