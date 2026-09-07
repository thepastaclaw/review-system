"""Run one model lane via the `claude` CLI in a worktree and capture structured output."""

from __future__ import annotations

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

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def argv_for(spec: LaneSpec) -> list[str]:
    argv = [
        spec.claude_bin,
        "--bare",
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
    timed_out = False
    try:
        out, err = proc.communicate(spec.prompt, timeout=spec.timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.send_signal(signal.SIGTERM)
        try:
            out, err = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
    dur = time.monotonic() - t0
    (artifact_dir / "stdout.json").write_text(out or "", encoding="utf-8")
    (artifact_dir / "stderr.txt").write_text(err or "", encoding="utf-8")
    res = LaneResult(
        exit_code=proc.returncode,
        stdout=out or "",
        stderr=err or "",
        duration_s=dur,
        timed_out=timed_out,
    )
    _extract_result(res)
    (artifact_dir / "lane-meta.json").write_text(
        json.dumps(
            {
                "exit_code": res.exit_code,
                "duration_s": round(dur, 1),
                "timed_out": timed_out,
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
    if res.timed_out:
        raise ReviewError(FailKind.INFRA, "lane timed out")
    if res.exit_code != 0:
        if res.subtype and res.subtype.startswith("error_"):
            raise ReviewError(
                FailKind.INFRA,
                f"lane {res.subtype.removeprefix('error_')} after {res.turns or '?'} turns, ${res.cost_usd or '?'}",
            )
        raise ReviewError(
            FailKind.INFRA, f"lane exit {res.exit_code}: {(res.stderr or res.result_text)[:200]}"
        )
    return parse_json_object(res.result_text)


def prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
