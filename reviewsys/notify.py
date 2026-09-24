"""Human-facing alerts (Slack via openclaw) and agent wake events."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable, Sequence

from .config import Config

log = logging.getLogger(__name__)

Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), capture_output=True, text=True, timeout=120, check=False)


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
        return self._send(self.cfg.slack_target, f":rotating_light: reviewsys: {text}")

    def page(self, text: str, *, resolved: bool = False) -> bool:
        """An outage a person has to fix: posted to the shared page channel with the on-call
        users @-mentioned, and to the operator DM as well. `resolved=True` is the all-clear:
        same places, no mentions. True if either copy landed."""
        self.sent.append(("resolved" if resolved else "page", text))
        if resolved:
            loud = f":white_check_mark: reviewsys: {text}"
        else:
            loud = f":rotating_light::rotating_light: *reviewsys needs a human* :rotating_light::rotating_light:\n{text}"
        delivered = False
        if self.cfg.page_target:
            mentions = "" if resolved else " ".join(f"<@{u}>" for u in self.cfg.page_mentions)
            delivered = self._send(self.cfg.page_target, f"{mentions} {loud}".strip())
        if self.cfg.slack_target and self.cfg.slack_target != self.cfg.page_target:
            delivered = self._send(self.cfg.slack_target, loud) or delivered
        if not delivered:
            log.error("page not delivered anywhere: %s", text)
        return delivered

    def _send(self, target: str, message: str) -> bool:
        argv = [
            self.cfg.openclaw_bin,
            "message",
            "send",
            "--channel",
            "slack",
            "--account",
            self.cfg.slack_account,
            "--target",
            target,
            "-m",
            message,
        ]
        try:
            proc = self.runner(argv)
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.error("alert delivery to %s failed: %s", target, exc)
            return False
        if proc.returncode != 0:
            log.error(
                "alert delivery to %s rc=%s: %s", target, proc.returncode, (proc.stderr or "")[:200]
            )
            return False
        return True

    def wake_agent(self, text: str) -> bool:
        """Wake the OpenClaw main agent with a system event (used for incoming review comments)."""
        self.sent.append(("wake", text))
        if not self.wake_enabled:
            log.info("wake suppressed (shadow mode): %s", text[:120])
            return False
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
