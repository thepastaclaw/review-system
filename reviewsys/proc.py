"""Process helpers: spawn detached, liveness, group kill."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path


def spawn(
    argv: Sequence[str], *, cwd: Path | None, log_path: Path, env: dict[str, str] | None = None
) -> subprocess.Popen[bytes]:
    """Start a child in its own session so its whole process group can be killed."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")  # noqa: SIM115 - handed to Popen, closed by GC
    return subprocess.Popen(
        list(argv),
        cwd=str(cwd) if cwd else None,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, **(env or {})},
    )


def alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # zombie check: on macOS/Linux a zombie still answers kill(0); try waitpid non-blocking
    try:
        wpid, _ = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            return False
    except ChildProcessError:
        pass
    return True


def kill_group(pid: int, *, grace: float = 10.0) -> str:
    """SIGTERM the process group, escalate to SIGKILL after `grace` seconds.

    Returns a short description of what happened.
    """
    if not alive(pid):
        return "already-dead"
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return "already-dead"
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return "already-dead"
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not alive(pid):
            return "terminated"
        time.sleep(0.2)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return "terminated"
    time.sleep(0.5)
    return "killed"


def reap(proc: subprocess.Popen[bytes]) -> int | None:
    """Non-blocking wait; returns exit code if exited."""
    return proc.poll()
