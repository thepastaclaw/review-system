"""Human-facing alerts (Slack via openclaw) and agent wake events."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable, Sequence

from .config import Config

log = logging.getLogger(__name__)

Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), capture_output=True, text=True, timeout=60, check=False)


class Notifier:
    def __init__(
        self, cfg: Config, runner: Runner | None = None, *, wake_enabled: bool = True
    ) -> None:
        self.cfg = cfg
        self.runner = runner or _run
        self.wake_enabled = wake_enabled
        self.sent: list[tuple[str, str]] = []

    def alert(self, text: str) -> bool:
        """Slack message to the operator. Returns True if delivered."""
        self.sent.append(("alert", text))
        if not self.cfg.slack_target:
            log.warning("alert (no slack target configured): %s", text)
            return False
        argv = [
            self.cfg.openclaw_bin,
            "message",
            "send",
            "--channel",
            "slack",
            "--account",
            self.cfg.slack_account,
            "--target",
            self.cfg.slack_target,
            "-m",
            f":rotating_light: reviewsys: {text}",
        ]
        try:
            proc = self.runner(argv)
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.error("alert delivery failed: %s", exc)
            return False
        if proc.returncode != 0:
            log.error("alert delivery rc=%s: %s", proc.returncode, (proc.stderr or "")[:200])
            return False
        return True

    def wake_agent(self, text: str) -> bool:
        """Wake the OpenClaw main agent with a system event (used for incoming review comments)."""
        self.sent.append(("wake", text))
        if not self.wake_enabled:
            log.info("wake suppressed (shadow mode): %s", text[:120])
            return True
        argv = [
            self.cfg.openclaw_bin,
            "system",
            "event",
            "--text",
            text,
            "--session-key",
            self.cfg.agent_session_key,
            "--mode",
            "now",
            "--timeout",
            "60000",
        ]
        try:
            proc = self.runner(argv)
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.error("wake delivery failed: %s", exc)
            return False
        return proc.returncode == 0
