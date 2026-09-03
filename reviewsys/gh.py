"""Thin, typed wrapper over the `gh` CLI.

Every call runs with GODEBUG=netdns=go so a wedged macOS system resolver
cannot stall the pipeline (2026-09-01 incident). Failures are classified as
transient (retryable) or contract (not retryable) so callers can decide.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Sequence
from typing import Any

from .models import FailKind, ReviewError

GhRunner = Callable[[Sequence[str], str | None, int], subprocess.CompletedProcess[str]]

_TRANSIENT_MARKERS = (
    "error connecting to",
    "timeout",
    "timed out",
    "500",
    "502",
    "503",
    "504",
    "connection refused",
    "rate limit",
    "secondary rate limit",
    "connection reset",
    "eof",
    "tls handshake",
    "no such host",
)


def _default_runner(
    argv: Sequence[str], stdin: str | None, timeout: int
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "GODEBUG": "netdns=go", "GH_PROMPT_DISABLED": "1", "NO_COLOR": "1"}
    return subprocess.run(
        list(argv),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )


class Gh:
    def __init__(
        self, bin_path: str = "gh", runner: GhRunner | None = None, *, default_timeout: int = 120
    ) -> None:
        self.bin = bin_path
        self.runner = runner or _default_runner
        self.default_timeout = default_timeout

    def run(self, *args: str, stdin: str | None = None, timeout: int | None = None) -> str:
        argv = [self.bin, *args]
        try:
            proc = self.runner(argv, stdin, timeout or self.default_timeout)
        except subprocess.TimeoutExpired as exc:
            raise ReviewError(FailKind.INFRA, f"gh {' '.join(args[:3])} timed out") from exc
        except OSError as exc:
            raise ReviewError(FailKind.FATAL, f"cannot execute gh: {exc}") from exc
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            kind = (
                FailKind.INFRA
                if any(m in err.lower() for m in _TRANSIENT_MARKERS)
                else FailKind.CONTRACT
            )
            raise ReviewError(kind, f"gh {' '.join(args[:3])} rc={proc.returncode}: {err[:400]}")
        return proc.stdout

    def json(self, *args: str, stdin: str | None = None, timeout: int | None = None) -> Any:
        out = self.run(*args, stdin=stdin, timeout=timeout)
        if not out.strip():
            return None
        try:
            return json.loads(out)
        except json.JSONDecodeError as exc:
            raise ReviewError(
                FailKind.CONTRACT, f"gh {' '.join(args[:3])} returned non-JSON: {out[:200]}"
            ) from exc

    def api(
        self,
        endpoint: str,
        *,
        method: str = "GET",
        body: dict[str, Any] | None = None,
        paginate: bool = False,
        jq: str | None = None,
        timeout: int | None = None,
    ) -> Any:
        args = ["api", endpoint, "--method", method]
        if paginate:
            args += ["--paginate", "--slurp"] if jq is None else ["--paginate"]
        if jq:
            args += ["--jq", jq]
        stdin = None
        if body is not None:
            args += ["--input", "-"]
            stdin = json.dumps(body)
        data = self.json(*args, stdin=stdin, timeout=timeout)
        if (
            paginate
            and jq is None
            and isinstance(data, list)
            and data
            and isinstance(data[0], list)
        ):
            return [row for page in data for row in page]
        return data

    def graphql(
        self, query: str, variables: dict[str, Any] | None = None, *, timeout: int | None = None
    ) -> Any:
        args = ["api", "graphql", "-f", f"query={query}"]
        for k, v in (variables or {}).items():
            flag = "-F" if isinstance(v, int | bool) else "-f"
            args += [flag, f"{k}={json.dumps(v) if isinstance(v, bool) else v}"]
        data = self.json(*args, timeout=timeout)
        if isinstance(data, dict) and data.get("errors"):
            raise ReviewError(
                FailKind.CONTRACT, f"graphql errors: {json.dumps(data['errors'])[:400]}"
            )
        return data.get("data") if isinstance(data, dict) else data
