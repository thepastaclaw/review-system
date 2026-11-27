"""GitHub App webhook receiver: a review-now label or a ticked gate box, the moment it happens.

GitHub posts each delivery to a tunnel (Tailscale Funnel or a Cloudflare tunnel) that forwards
it to this listener on a local port. Nothing is trusted before its X-Hub-Signature-256 matches
the shared secret (constant-time). Only two deliveries do anything: `pull_request.labeled` with
the review-now label and `issue_comment.edited` on the bot's gate comment with a box ticked.
Each makes the same pass a watch hit does (watch.py) due now, the ingest or the queue-comment
pass, which is unchanged and idempotent: the polls and the watch stay the fallback, and a
request all of them see asks for one review. The listener thread touches no database.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import stat
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import WebhookConfig
from .github import GATE_MARKER
from .ingest import REVIEW_NOW_LABEL
from .queue_status import normal_requested, priority_requested
from .watch import INGEST, QUEUE

log = logging.getLogger(__name__)

MAX_BODY = 1 << 20  # GitHub caps payloads at 25 MB; the two we act on are a few KB
SEEN_DELIVERIES = 1000
MIN_SECRET_CHARS = 32
READ_TIMEOUT_S = 10


class HookError(Exception):
    pass


def load_secret(path: Path) -> bytes:
    """The shared secret: the file must be the owner's alone (0600) and long enough."""
    try:
        st = path.stat()
        secret = path.read_text().strip()
    except OSError as exc:
        raise HookError(f"cannot read the webhook secret {path}: {exc}") from exc
    if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise HookError(f"{path} is readable by others; chmod 600 it")
    if len(secret) < MIN_SECRET_CHARS:
        raise HookError(f"{path} holds under {MIN_SECRET_CHARS} characters")
    return secret.encode()


def signed(secret: bytes, body: bytes, header: str | None) -> bool:
    if not header or not header.startswith("sha256="):
        return False
    want = "sha256=" + hmac.new(secret, body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(want.encode(), header.encode())


def classify(event: str, payload: dict[str, Any], bot_login: str, repos: set[str]) -> str | None:
    """The pass a delivery asks for, or None."""
    repo = str((payload.get("repository") or {}).get("full_name") or "").lower()
    if repo not in repos:
        return None
    action = payload.get("action")
    if event == "pull_request" and action == "labeled":
        name = str((payload.get("label") or {}).get("name") or "")
        return INGEST if name.lower() == REVIEW_NOW_LABEL else None
    if event == "issue_comment" and action == "edited":
        comment = payload.get("comment") or {}
        body = str(comment.get("body") or "")
        if (
            (payload.get("issue") or {}).get("pull_request")
            and str((comment.get("user") or {}).get("login") or "").lower() == bot_login.lower()
            and GATE_MARKER in body
            and (priority_requested(body) or normal_requested(body))
        ):
            return QUEUE
    return None


class Receiver:
    def __init__(
        self,
        cfg: WebhookConfig,
        secret: bytes,
        bot_login: str,
        repos: Iterable[str],
        on_hit: Callable[[str, str], None],
    ) -> None:
        self.cfg = cfg
        self.secret = secret
        self.bot_login = bot_login
        self.repos = {r.lower() for r in repos}
        self.on_hit = on_hit
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()
        self.server: ThreadingHTTPServer | None = None

    def handle(self, method: str, path: str, headers: Any, body: bytes) -> int:
        """One request: its HTTP status. `headers` is case-insensitive (http.client)."""
        if path.split("?", 1)[0] != self.cfg.path:
            return 404
        if method != "POST":
            return 405
        if not signed(self.secret, body, headers.get("X-Hub-Signature-256")):
            log.warning("webhook: unsigned or bad signature, %d bytes", len(body))
            return 401
        delivery = str(headers.get("X-GitHub-Delivery") or "")
        with self._lock:
            if delivery and delivery in self._seen:
                return 200  # GitHub's redelivery of one already acted on
            if delivery:
                self._seen[delivery] = None
                while len(self._seen) > SEEN_DELIVERIES:
                    self._seen.popitem(last=False)
        event = str(headers.get("X-GitHub-Event") or "")
        try:
            payload = json.loads(body)
        except ValueError:
            return 400
        if not isinstance(payload, dict):
            return 400
        task = classify(event, payload, self.bot_login, self.repos)
        if task is None:
            return 204
        number = (payload.get("pull_request") or payload.get("issue") or {}).get("number")
        detail = f"{event}.{payload.get('action')} {payload['repository']['full_name']}#{number}"
        log.info("webhook %s (delivery %s) -> %s", detail, delivery, task)
        self.on_hit(task, detail)
        return 202

    def start(self) -> ThreadingHTTPServer:
        host, _, port = self.cfg.listen.rpartition(":")
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            timeout = READ_TIMEOUT_S

            def _serve(self) -> None:
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = -1
                if length < 0 or length > MAX_BODY:
                    status = 413
                else:
                    body = self.rfile.read(length) if length else b""
                    try:
                        status = receiver.handle(self.command, self.path, self.headers, body)
                    except Exception:
                        log.exception("webhook handler failed")
                        status = 500
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()

            do_POST = do_GET = do_PUT = do_DELETE = do_HEAD = _serve

            def log_message(self, format: str, *args: Any) -> None:
                log.debug("webhook %s", format % args)

        self.server = ThreadingHTTPServer((host, int(port)), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, name="webhook", daemon=True).start()
        log.info("webhook receiver on %s%s", self.cfg.listen, self.cfg.path)
        return self.server
