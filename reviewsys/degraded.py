"""Degraded mode: keep reviewing when the primary (OpenAI/Codex) models are out of quota.

`gpt-6-astra` sits on the critical path of every run (triage, gate verifier, Phase-2
reviewers, final verifier, conversation lane), and the selector/repair models share the
same Codex pool. When that pool is exhausted every run dies on a 429 and the queue fills
with silently unreviewed PRs (2026-09-17: 60 failed runs in 8 h). With a
`review_model_policy.degraded` block the pipeline instead:

- probes the policy's `sentinel` model before every run (cached briefly in `kv`);
- while it answers with a quota error, or an operator forced the mode on, runs every lane
  whose model has a stand-in on that stand-in, caps the Phase-1 tier effort so the
  included-quota rungs (Gemini, GLM) stay eligible, and keeps both phases running unless
  the queue is deep enough for the backlog rule, which still sends a run straight to
  Phase 2 on the stand-in (`backlog_skip_phase1`, on by default: the slow Phase-1 rungs
  are what a backlog can least afford);
- switches the rest of a run over when a lane on a primary model dies on a quota error
  mid-run, and remembers that so the next run starts degraded;
- says so everywhere: review title and banner, provenance, gate and queue comments, the
  status snapshot and export, a Slack alert on every transition.

Operator override: `reviewsys degraded on|off|auto` (kv `degraded.force`).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from .config import Config
from .db import fmt_ts, kv_get, kv_set, now_dt, parse_ts, tx

log = logging.getLogger(__name__)

PROXY_URL = os.environ.get("REVIEWSYS_PROXY_URL", "http://127.0.0.1:8317")
CLIENT_KEYS_PATH = Path.home() / ".cli-proxy-api" / "client-keys.json"
PROBE_TIMEOUT_S = 15  # a 1-token request; the daemon's tick loop blocks on it
PROBE_CACHE = timedelta(seconds=120)  # a worker reuses a probe this fresh instead of re-sending
# status views (snapshot, queue comments, export) never probe; they show the last reading as
# long as it is plausibly current. The daemon re-probes every 2 min, so in a real outage this
# is never exceeded; it only lapses when the daemon itself is down.
SNAPSHOT_MAX_AGE = timedelta(minutes=15)
# a lane that died on quota is stronger evidence than a 1-token probe succeeding (the proxy
# may admit a tiny request while every real lane is refused), so hold the mode this long
LANE_HOLD = timedelta(minutes=20)
KV_FORCE = "degraded.force"  # on | off | absent (auto)
KV_PROBE = "degraded.probe"  # last probe: {"at", "quota_exhausted", "reason"}
KV_STATE = "degraded.state"  # last state the daemon alerted on: {"active", "reason", "since"}

# patterns in an upstream/proxy error that mean "no quota", as opposed to a dead proxy.
# Deliberately specific: these are only ever matched against a lane's stderr or an HTTP
# error body, never against model output, and a bare "quota"/"429" would also match a
# reviewer talking about quota handling in the PR under review.
QUOTA_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"cooling down",
        r"usage limit has been reached",
        r"usage_limit",
        r"rate_limit_error",
        r"model_cooldown",
        r"insufficient_quota",
        r"quota[ _]exceeded",
        r"request rejected \(429\)",
        r"api error:? 429\b",
        r"\b(?:http|status|error) 429\b",
    )
)


def looks_like_quota_failure(text: str) -> bool:
    """An infrastructure error text (lane stderr, HTTP error body) that says the upstream is
    out of quota or cooling down. Callers must not pass model output: a review of code that
    mentions rate limits is not an outage."""
    return any(p.search(text) for p in QUOTA_PATTERNS)


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PATH_RE = re.compile(r"(?:/[\w.-]+){2,}")
_URL_RE = re.compile(r"https?://\S+")


def publishable_reason(model: str, text: str) -> str:
    """A reason that is safe in a public review body: the upstream's short message with
    account emails, filesystem paths and URLs removed, capped. Full text stays in events."""
    msg = _URL_RE.sub("<url>", text)
    msg = _EMAIL_RE.sub("<account>", msg)
    msg = _PATH_RE.sub("<path>", msg)
    msg = " ".join(msg.split())
    for lead in ("API Error: ", "[infra] ", "lane exit 1: "):
        msg = msg.replace(lead, "")
    return f"`{model}` unavailable: {msg[:120]}" if msg else f"`{model}` unavailable"


@dataclass(frozen=True, slots=True)
class State:
    active: bool
    reason: str  # short, safe to publish
    source: str  # "forced" | "probe" | "lane" | "off" | "unconfigured"
    since: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        if not self.active:
            return "normal"
        return f"degraded ({self.reason}; {self.source})"


# ---- probe ----


def client_key() -> str | None:
    """The proxy client key the lanes use (`~/.cli-proxy-api/client-keys.json`)."""
    try:
        d = json.loads(CLIENT_KEYS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(d, dict):
        return None
    return d.get("claude-code") or d.get("openclaw") or (next(iter(d.values())) if d else None)


Prober = Callable[[str], tuple[bool, str]]  # a `probe`-shaped callable (tests inject one)


def probe(model: str, *, key: str | None = None, base_url: str = PROXY_URL) -> tuple[bool, str]:
    """(quota_exhausted, reason). One 1-token request through the proxy's Claude-protocol
    endpoint, exactly what a lane sends. Only a quota-shaped failure counts as exhausted: a
    dead proxy, a timeout or a 5xx is not a reason to swap models (nothing would work)."""
    key = key or client_key()
    if not key:
        return False, "no proxy client key to probe with"
    payload = {"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_S):
            return False, f"{model} answered"
    except urllib.error.HTTPError as exc:
        body = exc.read()[:400].decode("utf-8", "replace")
        text = f"HTTP {exc.code}: {body}"
        if exc.code == 429 or looks_like_quota_failure(text):
            return True, _publishable(model, exc.code, body)
        return False, f"{model}: {text[:160]} (not a quota failure)"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, f"{model}: probe failed ({str(exc)[:120]}); not treated as exhausted"


def _publishable(model: str, code: int, body: str) -> str:
    """A reason safe for a public review body: the upstream's message, sanitised."""
    try:
        obj = json.loads(body)
        msg = str((obj.get("error") or {}).get("message") or obj.get("message") or "")
    except (json.JSONDecodeError, AttributeError):
        msg = ""
    return publishable_reason(model, msg.strip() or f"HTTP {code}")


# ---- state ----


def _cached_probe(
    conn: sqlite3.Connection, *, max_age: timedelta | None = PROBE_CACHE
) -> dict[str, Any] | None:
    """The stored probe when it is younger than `max_age`, or while a lane-observed hold is
    in force. `max_age=None` (the daemon's periodic re-probe) only honours the hold."""
    raw = kv_get(conn, KV_PROBE)
    if not raw:
        return None
    try:
        blob = json.loads(raw)
        if not isinstance(blob, dict):
            return None
        at = now_dt()
        hold = blob.get("hold_until")
        if hold and at <= parse_ts(str(hold)):
            return dict(blob)
        if max_age is not None and at - parse_ts(str(blob["at"])) <= max_age:
            return dict(blob)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return None


def record_probe(
    conn: sqlite3.Connection, *, quota_exhausted: bool, reason: str, hold: bool = False
) -> None:
    """Store a probe result. `hold=True` for a lane's quota failure: the mode then stays on
    for `LANE_HOLD` regardless of what the sentinel probe says."""
    at = now_dt()
    blob: dict[str, Any] = {
        "at": fmt_ts(at),
        "quota_exhausted": quota_exhausted,
        "reason": reason,
    }
    if hold:
        blob["hold_until"] = fmt_ts(at + LANE_HOLD)
    with tx(conn):
        kv_set(conn, KV_PROBE, json.dumps(blob))


def force(conn: sqlite3.Connection, mode: str) -> None:
    """Operator override: `on` (always degraded), `off` (never), `auto` (probe)."""
    if mode not in {"on", "off", "auto"}:
        raise ValueError(f"degraded mode {mode!r} must be on, off or auto")
    with tx(conn):
        if mode == "auto":
            conn.execute("DELETE FROM kv WHERE key=?", (KV_FORCE,))
        else:
            kv_set(conn, KV_FORCE, mode)


def forced(conn: sqlite3.Connection) -> str | None:
    return kv_get(conn, KV_FORCE)


def detect(
    conn: sqlite3.Connection,
    cfg: Config,
    *,
    prober: Prober = probe,
    refresh: bool = False,
) -> State:
    """Current mode. Order: no policy -> off; operator override -> as forced; a fresh cached
    probe (or a lane's quota failure, held for `LANE_HOLD`) -> that; else probe the sentinel
    now and cache it. `refresh` re-probes even when the cache is fresh (hold still wins)."""
    pol = cfg.policy.degraded
    if pol is None:
        return State(False, "no degraded policy configured", "unconfigured")
    override = forced(conn)
    since = _since(conn)
    if override == "on":
        return State(True, "forced on by operator", "forced", since)
    if override == "off":
        return State(False, "forced off by operator", "off")
    blob = _cached_probe(conn, max_age=None if refresh else PROBE_CACHE)
    if blob is None:
        exhausted, reason = prober(pol.sentinel)
        record_probe(conn, quota_exhausted=exhausted, reason=reason)
        blob = {"quota_exhausted": exhausted, "reason": reason}
    if blob.get("quota_exhausted"):
        return State(True, str(blob.get("reason") or pol.label), "probe", since)
    return State(False, str(blob.get("reason") or "primary models answering"), "probe")


def _since(conn: sqlite3.Connection) -> str | None:
    raw = kv_get(conn, KV_STATE)
    if not raw:
        return None
    try:
        blob = json.loads(raw)
        return str(blob["since"]) if blob.get("active") and blob.get("since") else None
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def transition(conn: sqlite3.Connection, state: State) -> str | None:
    """Persist `state` as the last-known mode; returns an alert text when the mode flipped
    since the previous call (None when unchanged). Called by the daemon on its timer only;
    a worker that flips mid-run records a hold, which the daemon's next probe picks up, so
    the operator hears about every flip exactly once, within one daemon interval."""
    raw = kv_get(conn, KV_STATE)
    prev: dict[str, Any] = {}
    if raw:
        try:
            prev = json.loads(raw)
        except json.JSONDecodeError:
            prev = {}
    was = bool(prev.get("active"))
    if was == state.active:
        return None
    ts = fmt_ts(now_dt())
    with tx(conn):
        kv_set(
            conn,
            KV_STATE,
            json.dumps({"active": state.active, "reason": state.reason, "since": ts}),
        )
    if state.active:
        return f"entering DEGRADED mode: {state.reason} — reviews run on stand-in models until the primary models answer again"
    return f"leaving degraded mode: {state.reason}; reviews are back on the primary models"


def snapshot(conn: sqlite3.Connection, cfg: Config) -> dict[str, Any]:
    """For `status`, the export, the queue comments and doctor: the current mode from the
    last probe, never sending one (status must not block on the proxy). Tolerates a probe up
    to `SNAPSHOT_MAX_AGE` old so the view cannot flap between two daemon re-probes."""
    pol = cfg.policy.degraded
    if pol is None:
        return {"configured": False, "active": False}
    override = forced(conn)
    blob = _cached_probe(conn, max_age=SNAPSHOT_MAX_AGE)
    if override in {"on", "off"}:
        active = override == "on"
    else:
        active = bool(blob and blob.get("quota_exhausted"))
    return {
        "configured": True,
        "active": active,
        "forced": override,
        "since": _since(conn),
        "sentinel": pol.sentinel,
        "last_probe": blob,
        "substitutes": {k: v.model for k, v in pol.substitutes.items()},
        "phase1_effort_cap": pol.phase1_effort_cap,
    }
