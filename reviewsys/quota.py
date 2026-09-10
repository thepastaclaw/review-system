"""Remaining subscription quota, read through CLIProxyAPI's management API.

Phase 1 runs on a ladder of candidate models (`review_model_policy.phase1.candidates` in the
skills config): the first one whose subscription still has quota is used, so the included
Antigravity (Gemini) and Z.AI (GLM) allowances are spent before the pay-per-token last rung.

The proxy itself has no "remaining quota" endpoint; it does expose `POST /api-call`, which
sends an arbitrary request signed with one of its stored credentials. Each provider's own
quota endpoint is called that way, so the credentials never leave the proxy:

- antigravity: `cloudcode-pa.googleapis.com/v1internal:retrieveUserQuotaSummary` (what the
  Antigravity IDE shows): groups ("Gemini Models", "Claude and GPT models") of 5h + weekly
  buckets with a `remainingFraction`.
- zai: `api.z.ai/api/monitor/usage/quota/limit` (what the Z.AI console shows): a 5h and a
  weekly `CREDIT_LIMIT` with `percentage` used.

A candidate is usable when the *smallest* remaining fraction across its windows is at least
`quota_reserve`. A lookup failure counts as "no quota" for that rung: the run moves down the
ladder rather than gambling on a lane that may die on 429 an hour in, and the last rung is
never gated.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from .config import LaneModel, QuotaSource
from .db import fmt_ts, kv_get, kv_set, now_dt, parse_ts, tx

log = logging.getLogger(__name__)

PROXY_URL = os.environ.get("REVIEWSYS_PROXY_URL", "http://127.0.0.1:8317")
MANAGEMENT_KEY_PATH = Path.home() / ".cli-proxy-api" / "management-key"
TIMEOUT_S = 30  # localhost proxy, but the box is shared with CI builds and stalls under load
RETRIES = 2  # attempts per lookup before the cached reading is used
CACHE_MAX_AGE = timedelta(
    hours=1
)  # a reading this fresh still gates a rung when the live one fails

ANTIGRAVITY_QUOTA_URL = "https://cloudcode-pa.googleapis.com/v1internal:retrieveUserQuotaSummary"
# the endpoint answers 403 "no valid license" to anything but an Antigravity client UA
ANTIGRAVITY_USER_AGENT = "antigravity/cli/1.0.13 (aidev_client; os_type=darwin; arch=arm64)"
ZAI_QUOTA_URL = "https://api.z.ai/api/monitor/usage/quota/limit"
ZAI_HOST = "api.z.ai"


class QuotaError(Exception):
    """The remaining quota could not be determined."""


@dataclass(frozen=True, slots=True)
class Window:
    name: str  # "5h" | "weekly" | ...
    remaining: float  # 0..1
    reset_at: str | None = None

    def describe(self) -> str:
        return f"{self.name} {round(self.remaining * 100)}% left"


@dataclass(frozen=True, slots=True)
class QuotaStatus:
    account: str
    windows: tuple[Window, ...]

    @property
    def remaining(self) -> float:
        return min((w.remaining for w in self.windows), default=0.0)

    def describe(self) -> str:
        return ", ".join(w.describe() for w in self.windows) or "no windows reported"


class Management:
    """Thin client for the proxy's management API (`~/.cli-proxy-api/management-key`)."""

    def __init__(self, base_url: str = PROXY_URL, key_path: Path = MANAGEMENT_KEY_PATH) -> None:
        self.base_url = base_url.rstrip("/")
        self.key_path = key_path
        self._key: str | None = None

    def _headers(self) -> dict[str, str]:
        if self._key is None:
            try:
                self._key = self.key_path.read_text().strip()
            except OSError as exc:
                raise QuotaError(f"management key unreadable: {exc}") from exc
        return {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}

    def get(self, path: str) -> Any:
        return self._call("GET", path, None)

    def post(self, path: str, body: dict[str, Any]) -> Any:
        return self._call("POST", path, json.dumps(body).encode())

    def _call(self, method: str, path: str, data: bytes | None) -> Any:
        req = urllib.request.Request(
            f"{self.base_url}/v0/management/{path.lstrip('/')}",
            data=data,
            method=method,
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                return json.loads(resp.read() or b"null")
        except urllib.error.HTTPError as exc:
            raise QuotaError(f"management {method} {path}: HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise QuotaError(f"management {method} {path}: {exc}") from exc

    def api_call(
        self,
        auth_index: str,
        method: str,
        url: str,
        *,
        body: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, Any]:
        """Send `url` upstream signed with the credential `auth_index`; `$TOKEN$` is the
        proxy's placeholder for that credential's access token / API key."""
        payload: dict[str, Any] = {
            "auth_index": auth_index,
            "method": method,
            "url": url,
            "header": {
                "Authorization": "Bearer $TOKEN$",
                "Content-Type": "application/json",
                **(headers or {}),
            },
        }
        if body is not None:
            payload["data"] = body
        res = self.post("api-call", payload)
        if not isinstance(res, dict):
            raise QuotaError(f"api-call {url}: unexpected response")
        status = int(res.get("status_code") or 0)
        raw: Any = res.get("body") or ""
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as exc:
            raise QuotaError(f"api-call {url}: HTTP {status}, non-JSON body") from exc
        if status != 200:
            raise QuotaError(f"api-call {url}: HTTP {status}: {str(raw)[:160]}")
        return status, parsed


# ---- providers ----


def _dicts(obj: Any, key: str) -> list[dict[str, Any]]:
    """`obj[key]` as a list of dicts; the proxy is Go, so absent slices arrive as `null`, and
    an element of an unexpected type is dropped rather than crashing the read."""
    value = obj.get(key) if isinstance(obj, dict) else None
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def _fraction_left(value: Any) -> float:
    """A remaining fraction; a missing field is 0 (proto3 JSON omits zero-valued fields)."""
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _collect[C](
    provider: str, credentials: list[tuple[str, C]], windows_of: Callable[[C], tuple[Window, ...]]
) -> list[QuotaStatus]:
    """Read every credential; one bad one must not hide a good one. Raises only when none
    could be read, with every per-account error in the message."""
    out: list[QuotaStatus] = []
    errors: list[str] = []
    for account, credential in credentials:
        try:
            out.append(QuotaStatus(account=account, windows=windows_of(credential)))
        except QuotaError as exc:
            errors.append(f"{account}: {exc}")
    if not out:
        raise QuotaError(
            f"{provider}: " + ("; ".join(errors) or "no usable credential in the proxy")
        )
    return out


def _antigravity_windows(
    mgmt: Management, source: QuotaSource, credential: tuple[str, str]
) -> tuple[Window, ...]:
    auth_index, project = credential
    if not auth_index or not project:
        raise QuotaError("credential has no auth_index/project_id")
    _, data = mgmt.api_call(
        auth_index,
        "POST",
        ANTIGRAVITY_QUOTA_URL,
        body=json.dumps({"project": project}),
        headers={"User-Agent": ANTIGRAVITY_USER_AGENT},
    )
    windows = tuple(
        Window(
            name=str(bucket.get("window") or bucket.get("bucketId") or "?"),
            remaining=_fraction_left(bucket.get("remainingFraction")),
            reset_at=str(bucket["resetTime"]) if bucket.get("resetTime") else None,
        )
        for group in _dicts(data, "groups")
        if not source.group or str(group.get("displayName", "")).lower() == source.group.lower()
        for bucket in _dicts(group, "buckets")
    )
    if not windows:
        raise QuotaError(f"no quota group {source.group!r} in response")
    return windows


def _antigravity(mgmt: Management, source: QuotaSource) -> list[QuotaStatus]:
    """Every enabled Antigravity credential in the proxy (`source.account` pins one email)."""
    credentials = [
        (
            str(f.get("email") or f.get("auth_index") or "?"),
            (str(f.get("auth_index") or ""), str(f.get("project_id") or "")),
        )
        for f in _dicts(mgmt.get("auth-files"), "files")
        if f.get("provider") == "antigravity"
        and not f.get("disabled")
        and not f.get("unavailable")
        and (not source.account or str(f.get("email") or "") == source.account)
    ]
    return _collect("antigravity", credentials, lambda c: _antigravity_windows(mgmt, source, c))


def _zai_window_name(unit: int, number: int) -> str:
    """Z.AI reports a window as a count plus a unit code: 3 is hours, 6 is weeks."""
    if unit == 3:
        return f"{number}h"
    if unit == 6:
        return "weekly" if number == 1 else f"{number}w"
    return f"unit{unit}x{number}"


def _zai_windows(mgmt: Management, auth_index: str) -> tuple[Window, ...]:
    _, data = mgmt.api_call(auth_index, "GET", ZAI_QUOTA_URL)
    body = data.get("data") if isinstance(data, dict) else None
    windows: list[Window] = []
    for limit in _dicts(body, "limits"):
        if limit.get("type") != "CREDIT_LIMIT":
            continue
        try:
            used = float(limit.get("percentage") or 0.0)
            unit, number = int(limit.get("unit") or 0), int(limit.get("number") or 0)
        except (TypeError, ValueError):
            continue
        reset = limit.get("nextResetTime")
        windows.append(
            Window(
                name=_zai_window_name(unit, number),
                remaining=max(0.0, 1.0 - used / 100.0),
                reset_at=str(reset) if reset is not None else None,
            )
        )
    if not windows:
        raise QuotaError("no CREDIT_LIMIT windows in response")
    return tuple(windows)


def _zai(mgmt: Management, source: QuotaSource) -> list[QuotaStatus]:
    """Every api.z.ai key in the proxy's `openai-compatibility` providers (`source.account`
    narrows to one provider name)."""
    credentials: list[tuple[str, str]] = []
    for prov in _dicts(mgmt.get("openai-compatibility"), "openai-compatibility"):
        if prov.get("disabled") or ZAI_HOST not in str(prov.get("base-url", "")):
            continue
        if source.account and str(prov.get("name")) != source.account:
            continue
        for entry in _dicts(prov, "api-key-entries"):
            auth_index = str(entry.get("auth-index") or "")
            if auth_index:
                credentials.append((f"{prov.get('name')}/{auth_index[:6]}", auth_index))
    return _collect("zai", credentials, lambda idx: _zai_windows(mgmt, idx))


def read(mgmt: Management, source: QuotaSource) -> QuotaStatus:
    """The account with the most quota left for `source` (the proxy fails over between
    accounts, so the best one is what a lane will effectively get). Anything unexpected in
    the proxy's or the upstream's answer surfaces as QuotaError, never as a bare exception."""
    try:
        if source.provider == "antigravity":
            statuses = _antigravity(mgmt, source)
        elif source.provider == "zai":
            statuses = _zai(mgmt, source)
        else:
            raise QuotaError(f"unknown quota provider {source.provider!r}")
    except QuotaError:
        raise
    except Exception as exc:  # a quota read must never take a run down
        raise QuotaError(f"{source.provider}: unexpected {type(exc).__name__}: {exc}") from exc
    return max(statuses, key=lambda s: s.remaining)


# ---- selection ----


@dataclass(frozen=True, slots=True)
class Skipped:
    model: str
    reason: str  # short and safe to publish
    detail: str = ""  # full error text; events and logs only

    def as_dict(self) -> dict[str, str]:
        """Publishable form: no error text."""
        return {"model": self.model, "reason": self.reason}

    def describe(self) -> str:
        """Log form: the error text included."""
        return f"{self.model} ({self.reason}{': ' + self.detail if self.detail else ''})"


@dataclass(frozen=True, slots=True)
class Choice:
    model: LaneModel
    reason: str  # safe to publish
    skipped: tuple[Skipped, ...] = ()  # every rung passed over, in ladder order

    def as_dict(self) -> dict[str, Any]:
        """Publishable form: no error text, no account names."""
        return {
            "model": self.model.model,
            "reason": self.reason,
            "skipped": [s.as_dict() for s in self.skipped],
        }

    def log_line(self) -> str:
        line = f"{self.model.model}: {self.reason}"
        if self.skipped:
            line += "; skipped " + ", ".join(s.describe() for s in self.skipped)
        return line

    def remaining(self, candidates: tuple[LaneModel, ...]) -> tuple[LaneModel, ...]:
        """The rungs below the chosen one, for a later fallback."""
        for i, candidate in enumerate(candidates):
            if candidate.model == self.model.model:
                return candidates[i + 1 :]
        return ()


QuotaReader = Callable[[QuotaSource], QuotaStatus]


def choose(
    candidates: tuple[LaneModel, ...],
    reserve: float,
    reader: QuotaReader | None = None,
    *,
    skipped: tuple[Skipped, ...] = (),
) -> Choice:
    """First rung with at least `reserve` of every quota window left. Rungs without a quota
    source are always usable, so a ladder whose last rung is pay-per-token always resolves;
    a fully gated ladder falls through to its last rung with the shortfall recorded. A
    lookup that fails for any reason skips the rung; nothing here can raise on bad data."""
    if not candidates:
        raise ValueError("no Phase-1 candidates")
    if len(candidates) == 1 and not skipped:
        return Choice(candidates[0], "only candidate")
    reader = reader or _default_reader()
    passed = list(skipped)
    for lm in candidates:
        if lm.quota is None:
            return Choice(lm, "not quota-gated", tuple(passed))
        try:
            status = reader(lm.quota)
        except Exception as exc:  # see docstring
            log.warning("quota lookup for %s failed: %s", lm.model, exc)
            passed.append(
                Skipped(lm.model, f"{lm.quota.provider} quota lookup failed", str(exc)[:600])
            )
            continue
        if status.remaining >= reserve:
            return Choice(lm, f"{lm.quota.provider} quota: {status.describe()}", tuple(passed))
        passed.append(
            Skipped(
                lm.model,
                f"{lm.quota.provider} below {round(reserve * 100)}% reserve: {status.describe()}",
            )
        )
    last = passed.pop()
    return Choice(
        candidates[-1], f"every rung short of quota; last rung used ({last.reason})", tuple(passed)
    )


def cached_reader(conn: sqlite3.Connection, mgmt: Management | None = None) -> QuotaReader:
    """A reader that retries a failed live lookup and then falls back to the last good reading
    stored in `kv` (`quota.last:<provider>[:<group>]`), so a proxy that is merely slow under
    CI load does not push a review onto the paid rung. Quota moves slowly; an hour-old reading
    is a far better guess than "none"."""
    live = mgmt or Management()

    def reader(source: QuotaSource) -> QuotaStatus:
        key = "quota.last:" + ":".join(
            x for x in (source.provider, source.group, source.account) if x
        )
        last_exc: Exception | None = None
        for _ in range(RETRIES):
            try:
                status = read(live, source)
            except QuotaError as exc:
                last_exc = exc
                continue
            with tx(conn):
                kv_set(conn, key, json.dumps({"at": fmt_ts(now_dt()), "status": asdict(status)}))
            return status
        cached = kv_get(conn, key)
        if cached:
            try:
                blob = json.loads(cached)
                age = now_dt() - parse_ts(str(blob["at"]))
                if age <= CACHE_MAX_AGE:
                    st = blob["status"]
                    windows = tuple(Window(**w) for w in st["windows"])
                    log.warning(
                        "quota lookup for %s failed (%s); using reading from %s ago",
                        source.provider,
                        last_exc,
                        age,
                    )
                    return QuotaStatus(
                        account=f"{st['account']} (cached {age.seconds // 60} min)", windows=windows
                    )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
        raise QuotaError(f"{last_exc}; no reading under {CACHE_MAX_AGE} to fall back on")

    return reader


def _default_reader() -> QuotaReader:
    mgmt = Management()
    return lambda source: read(mgmt, source)
