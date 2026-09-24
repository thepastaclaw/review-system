"""Review slots that scale with the number of usable OpenAI (Codex) accounts.

The primary models are limited per ChatGPT account, not globally: one account starts
refusing work above ~3 concurrent streams or once its weekly window is nearly spent
(see the 2026-09-17 `server_is_overloaded` investigation), while the proxy spreads lanes
over every account it holds. So `max_concurrent` + `priority_overflow` (2 + 1) is the
budget of *one* account, and the effective capacity is that unit times the number of
accounts that can take work now, capped at `account_scale_max`.

The daemon refreshes the reading every few minutes from the proxy's management API
(`auth-files`: the proxy's own record of each credential's cooldowns and the
`X-Codex-*-Used-Percent` quota headers it last saw), so no request reaches OpenAI. The
scheduler only reads the stored reading; a missing, stale or unreadable one falls back to
a scale of 1, never to more capacity than the static config grants.

An account is spent to its last percent unless `account_reserves` names it. A reserved
account (one lent on condition that e.g. 15% is always left) is *disabled in the proxy* once
it reaches its floor,
so no client at all (reviewsys, T3, dashvm) eats into the reserve, and re-enabled once the
window that hit the floor has reset. reviewsys only ever re-enables an account it disabled
itself (kv `slots.parked`, dropped as soon as the account is seen enabled again); one an
operator disabled stays disabled. The reserves are box config, not code: the list names
people's accounts and this repository is public.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from .config import Config
from .db import fmt_ts, kv_get, kv_set, now_dt, parse_ts, tx
from .quota import Management, QuotaError

log = logging.getLogger(__name__)

KV_ACCOUNTS = "slots.accounts"  # {"at", "accounts": [Account...]}
KV_PARKED = "slots.parked"  # {auth file name: {"email", "until"}} disabled by reviewsys
UNKNOWN_RESET_RETRY = timedelta(hours=6)
# three missed daemon refreshes: after that the reading no longer says anything about now
READING_MAX_AGE = timedelta(minutes=20)


@dataclass(frozen=True, slots=True)
class Account:
    name: str
    usable: bool
    reason: str  # why not usable, or its quota headroom
    used_pct: int | None = None  # highest used percent over the reported windows
    reset_at: str | None = None  # when an unusable account is expected back


@dataclass(frozen=True, slots=True)
class Capacity:
    normal: int
    priority: int
    scale: int
    usable: int | None  # None: no current reading
    reason: str

    @property
    def ceiling(self) -> int:
        return self.normal + self.priority

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ceiling": self.ceiling}


def _pct(signals: dict[str, Any], key: str) -> int | None:
    try:
        return int(float(signals[key]))
    except (KeyError, TypeError, ValueError):
        return None


def _future(ts: Any, at_iso: str) -> str | None:
    """`ts` when it is a timestamp later than `at_iso`, else None. One without an offset is
    taken as UTC; one that does not parse is ignored."""
    if not ts:
        return None
    try:
        when = parse_ts(str(ts))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return str(ts) if when > parse_ts(at_iso) else None


def _signals(f: dict[str, Any]) -> dict[str, Any]:
    quota = f.get("quota")
    signals = quota.get("signals") if isinstance(quota, dict) else None
    return signals if isinstance(signals, dict) else {}


def _email(f: dict[str, Any]) -> str:
    return str(f.get("email") or f.get("label") or f.get("auth_index") or "?")


def _used(signals: dict[str, Any], at_iso: str) -> list[tuple[int, str | None]]:
    """(used percent, reset time) for every quota window the proxy last saw. The signals are
    the headers of the account's last request, so a window whose reset time has passed has
    rolled over since: it counts as unused, or a parked account could never come back."""
    out: list[tuple[int, str | None]] = []
    for kind in ("Primary", "Secondary"):
        if kind == "Secondary" and (_pct(signals, "X-Codex-Secondary-Window-Minutes") or 0) <= 0:
            continue
        used = _pct(signals, f"X-Codex-{kind}-Used-Percent")
        if used is None:
            continue
        reset = _epoch(signals.get(f"X-Codex-{kind}-Reset-At"))
        out.append((0, None) if reset and not _future(reset, at_iso) else (used, reset))
    return out


def _epoch(value: Any) -> str | None:
    try:
        return fmt_ts(datetime.fromtimestamp(int(float(value)), UTC))
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def at_floor(f: dict[str, Any], reserve: float, at_iso: str) -> tuple[int, str | None] | None:
    """(used percent, when it resets) of the window that leaves less than `reserve`, or None
    while every window is above it. With no reserve an account is never at its floor here:
    the proxy's own cooldown takes it out when it is truly empty."""
    if reserve <= 0:
        return None
    floor = round((1.0 - reserve) * 100)
    hit = [(u, r) for u, r in _used(_signals(f), at_iso) if u >= floor]
    if not hit:
        return None
    # the account is back only once every window at the floor has reset
    resets = [r for _, r in hit if r]
    return max(u for u, _ in hit), (max(resets) if len(resets) == len(hit) else None)


def classify(f: dict[str, Any], *, model: str, reserve: float, at_iso: str) -> Account:
    """One proxy credential: usable when enabled, not parked by the proxy, not cooling down
    for the credential or for `model`, and every quota window it reported has at least
    `reserve` left (0 for all but the reserved accounts: those are spent to the end). A
    credential the proxy has not seen quota headers for yet is usable."""
    name = _email(f)
    for cd in f.get("cooldowns") or []:
        if not isinstance(cd, dict):
            continue
        if cd.get("scope") == "credential" or cd.get("model_key") == model:
            until = _future(cd.get("retry_at"), at_iso)
            if until:
                return Account(name, False, f"cooling down ({cd.get('reason')})", reset_at=until)
    if f.get("unavailable"):
        return Account(
            name, False, "unavailable", reset_at=_future(f.get("next_retry_after"), at_iso)
        )
    used = [u for u, _ in _used(_signals(f), at_iso)]
    if not used:
        return Account(name, True, "no quota reading yet")
    worst = max(used)
    floor = at_floor(f, reserve, at_iso)
    if floor is not None:
        return Account(
            name,
            False,
            f"{worst}% used, at its {round(reserve * 100)}% reserve",
            used_pct=worst,
            reset_at=floor[1],
        )
    return Account(name, True, f"{worst}% of quota used", used_pct=worst)


def _codex_files(mgmt: Management) -> list[dict[str, Any]]:
    files = mgmt.get("auth-files")
    rows = files.get("files") if isinstance(files, dict) else None
    if not isinstance(rows, list):
        raise QuotaError("auth-files: unexpected response")
    return [f for f in rows if isinstance(f, dict) and f.get("provider") == "codex"]


def read_accounts(
    mgmt: Management, *, model: str, reserves: dict[str, float] | None = None
) -> list[Account]:
    """Every enabled Codex credential in the proxy, classified."""
    at_iso = fmt_ts(now_dt())
    return [
        classify(
            f, model=model, reserve=(reserves or {}).get(_email(f).lower(), 0.0), at_iso=at_iso
        )
        for f in _codex_files(mgmt)
        if not f.get("disabled")
    ]


def _parked(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    raw = kv_get(conn, KV_PARKED)
    try:
        blob = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}
    return blob if isinstance(blob, dict) else {}


def enforce_reserves(
    conn: sqlite3.Connection, cfg: Config, mgmt: Management
) -> list[tuple[str, str, str]]:
    """Disable every reserved account at its floor in the proxy, and re-enable the ones this
    function disabled once their window has reset. Returns (action, email, detail) for each
    change made, so the daemon can tell a human. A PATCH that fails is retried next time."""
    if not cfg.account_reserves:
        return []
    now_iso = fmt_ts(now_dt())
    parked = _parked(conn)
    changes: list[tuple[str, str, str]] = []
    files = _codex_files(mgmt)
    # an auth file deleted from the proxy is not coming back to be re-enabled
    gone = set(parked) - {str(f.get("name") or "") for f in files}
    for name in gone:
        parked.pop(name)
    dirty = bool(gone)
    for f in files:
        name, email = str(f.get("name") or ""), _email(f)
        reserve = cfg.account_reserves.get(email.lower(), 0.0)
        if not name or reserve <= 0:
            continue
        mine = parked.get(name)
        if not f.get("disabled") and mine is not None:
            # someone re-enabled it by hand: it is no longer ours to re-enable later
            parked.pop(name)
            dirty = True
        if f.get("disabled"):
            if mine is None or _future(mine.get("until"), now_iso):
                continue  # someone else disabled it, or its window has not reset yet
            # the window has reset; if the proxy's next reading is still low it is re-parked
            if _set_disabled(mgmt, name, False):
                parked.pop(name, None)
                changes.append(("enabled", email, "quota window reset; reserve restored"))
            continue
        floor = at_floor(f, reserve, now_iso)
        if floor is None:
            continue
        used, until = floor
        # no reset time reported: look again in a few hours rather than flap every tick
        until = until or fmt_ts(now_dt() + UNKNOWN_RESET_RETRY)
        # recorded before the PATCH: a request that fails after the proxy applied it must
        # still leave the account ours to re-enable. If it really failed, the next pass sees
        # the account enabled, forgets the entry and tries again.
        parked[name] = {"email": email, "until": until}
        dirty = True
        if _set_disabled(mgmt, name, True):
            changes.append(
                (
                    "disabled",
                    email,
                    f"{used}% used, keeping its {round(reserve * 100)}% reserve until {until}",
                )
            )
    if changes or dirty:
        with tx(conn):
            kv_set(conn, KV_PARKED, json.dumps(parked))
    return changes


def _set_disabled(mgmt: Management, name: str, disabled: bool) -> bool:
    try:
        mgmt.patch("auth-files/status", {"name": name, "disabled": disabled})
    except Exception as exc:  # retried next pass; must not skip saving the parked state
        log.error("could not %s %s: %s", "disable" if disabled else "enable", name, exc)
        return False
    return True


def refresh(
    conn: sqlite3.Connection, cfg: Config, mgmt: Management | None = None
) -> tuple[Capacity, list[tuple[str, str, str]]]:
    """Daemon task: enforce the per-account reserves, then re-read the accounts and store
    them. A failed read keeps the previous reading, which ages out to the scale-1 fallback
    on its own."""
    mgmt = mgmt or Management()
    model = (
        cfg.policy.degraded.sentinel if cfg.policy.degraded else cfg.policy.phase2_reviewer.model
    )
    changes: list[tuple[str, str, str]] = []
    try:
        changes = enforce_reserves(conn, cfg, mgmt)
        accounts = read_accounts(mgmt, model=model, reserves=cfg.account_reserves)
    except Exception as exc:  # anything odd in the proxy's answer must not stop the task
        log.warning("codex account read failed: %s", exc)
        return capacity(conn, cfg), changes
    # parked accounts are disabled, so read_accounts skips them; a human reading the page
    # must see them as out on purpose, not as something to re-enable
    accounts += [
        Account(
            str(p.get("email") or n), False, "parked to keep its reserve", reset_at=p.get("until")
        )
        for n, p in _parked(conn).items()
    ]
    blob = {"at": fmt_ts(now_dt()), "accounts": [asdict(a) for a in accounts]}
    with tx(conn):
        kv_set(conn, KV_ACCOUNTS, json.dumps(blob))
    return capacity(conn, cfg), changes


def stored_accounts(conn: sqlite3.Connection) -> tuple[str, list[Account]] | None:
    """(read_at, accounts) from the last refresh, however old."""
    raw = kv_get(conn, KV_ACCOUNTS)
    if not raw:
        return None
    try:
        blob = json.loads(raw)
        return str(blob["at"]), [Account(**a) for a in blob["accounts"]]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def capacity(conn: sqlite3.Connection, cfg: Config) -> Capacity:
    """The slots the scheduler may fill now. Never below the static 2 + 1."""
    base_n, base_p = cfg.max_concurrent, cfg.priority_overflow
    if cfg.account_scale_max <= 1:
        return Capacity(base_n, base_p, 1, None, "scaling disabled")
    stored = stored_accounts(conn)
    if stored is None:
        return Capacity(base_n, base_p, 1, None, "no account reading yet")
    at, accounts = stored
    if now_dt() - parse_ts(at) > READING_MAX_AGE:
        return Capacity(base_n, base_p, 1, None, f"account reading from {at} is stale")
    usable = sum(1 for a in accounts if a.usable)
    scale = max(1, min(usable, cfg.account_scale_max))
    reason = f"{usable} of {len(accounts)} OpenAI accounts usable"
    if usable > cfg.account_scale_max:
        reason += f", capped at {cfg.account_scale_max}"
    return Capacity(base_n * scale, base_p * scale, scale, usable, reason)
