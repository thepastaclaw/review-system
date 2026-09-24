"""Review slots scale with the usable OpenAI accounts: 2 + 1 per account, at most 3 accounts."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from typing import Any

from reviewsys import slots
from reviewsys.db import fmt_ts, kv_set, now_dt, tx
from reviewsys.notify import Notifier
from reviewsys.scheduler import schedule
from tests.test_scheduler_reaper import queue

MODEL = "gpt-6-astra"
AT = "2026-09-24T18:00:00Z"


def _file(email: str, **kw: Any) -> dict[str, Any]:
    return {"provider": "codex", "email": email, "disabled": False, "unavailable": False, **kw}


def _quota(primary: int, secondary: int | None = None) -> dict[str, Any]:
    signals = {"X-Codex-Primary-Used-Percent": str(primary)}
    if secondary is not None:
        signals |= {
            "X-Codex-Secondary-Used-Percent": str(secondary),
            "X-Codex-Secondary-Window-Minutes": "300",
        }
    return {"signals": signals}


def _classify(f: dict[str, Any]) -> slots.Account:
    return slots.classify(f, model=MODEL, reserve=0.15, at_iso=AT)


def test_classify_matches_what_the_proxy_reports():
    assert _classify(_file("a", quota=_quota(6))).usable
    assert _classify(_file("fresh", quota={"signals": {}})).usable, "not yet used is usable"
    spent = _classify(_file("b", quota=_quota(90)))
    assert not spent.usable and spent.used_pct == 90, "inside the 15% reserve"
    assert not _classify(_file("c", quota=_quota(10, secondary=97))).usable, "5h window counts"
    cooling = _classify(
        _file(
            "d",
            quota=_quota(0),
            cooldowns=[{"scope": "model", "model_key": MODEL, "retry_at": "2026-09-30T02:38:46Z"}],
        )
    )
    assert not cooling.usable and cooling.reset_at == "2026-09-30T02:38:46Z"
    other_model = _file(
        "e",
        cooldowns=[
            {"scope": "model", "model_key": "gpt-5.6-luna", "retry_at": "2026-09-30T02:38:46Z"}
        ],
    )
    assert _classify(other_model).usable, "a cooldown on another model does not park the account"
    expired = _file("f", cooldowns=[{"scope": "credential", "retry_at": "2026-09-01T00:00:00Z"}])
    assert _classify(expired).usable, "a cooldown that already ended is ignored"
    assert not _classify(_file("g", unavailable=True)).usable


class _Mgmt:
    """The proxy's management API: `auth-files` plus the status PATCH, applied in place."""

    def __init__(self, files: list[dict[str, Any]]) -> None:
        self.files = files
        self.patches: list[dict[str, Any]] = []

    def get(self, path: str) -> Any:
        assert path == "auth-files"
        return {"files": self.files}

    def patch(self, path: str, body: dict[str, Any]) -> Any:
        assert path == "auth-files/status"
        self.patches.append(body)
        for f in self.files:
            if f.get("name") == body["name"]:
                f["disabled"] = body["disabled"]
        return {"status": "ok"}


def _accounts(n_usable: int, n_spent: int = 0) -> list[dict[str, Any]]:
    return [_file(f"ok{i}", quota=_quota(10)) for i in range(n_usable)] + [
        _file(f"spent{i}", unavailable=True) for i in range(n_spent)
    ]


def _refresh(conn, cfg, mgmt) -> slots.Capacity:
    return slots.refresh(conn, cfg, mgmt)[0]


def test_capacity_is_two_plus_one_per_usable_account_capped_at_three(cfg, conn):
    assert slots.capacity(conn, cfg).ceiling == 3, "no reading yet: the static 2 + 1"
    for usable, expect in ((0, (2, 1)), (1, (2, 1)), (2, (4, 2)), (3, (6, 3)), (5, (6, 3))):
        cap = _refresh(conn, cfg, _Mgmt(_accounts(usable, n_spent=2)))
        assert (cap.normal, cap.priority) == expect, usable
    assert "5 of 7 OpenAI accounts usable, capped at 3" in cap.reason
    disabled = _file("off", disabled=True, quota=_quota(0))
    cap = _refresh(conn, cfg, _Mgmt([*_accounts(1), disabled]))
    assert cap.usable == 1, "a credential disabled in the proxy is not an account"


def test_accounts_without_a_reserve_are_spent_to_the_end(cfg, conn):
    _refresh(conn, cfg, _Mgmt([_file("chatgpt2@x", quota=_quota(99))]))
    (acct,) = slots.stored_accounts(conn)[1]  # type: ignore[index]
    assert acct.usable, "no reserve: 99% used is still usable until the proxy cools it down"


def _reserved(used: int, reset_in_s: int = 3600, **kw: Any) -> dict[str, Any]:
    reset = int(now_dt().timestamp()) + reset_in_s
    q = {
        "signals": {
            "X-Codex-Primary-Used-Percent": str(used),
            "X-Codex-Primary-Reset-At": str(reset),
        }
    }
    return _file("reserved@example.com", name="codex-reserved@example.com-pro.json", quota=q, **kw)


RESERVED = {"reserved@example.com": 0.15}


def test_reserved_account_is_disabled_at_its_floor_and_back_after_the_reset(cfg, conn):
    assert cfg.account_reserves == {}, "the reserve list is box config, not public code"
    cfg = replace(cfg, account_reserves=RESERVED)
    mgmt = _Mgmt([_reserved(84), _file("chatgpt3@x", quota=_quota(2))])
    cap, changes = slots.refresh(conn, cfg, mgmt)
    assert changes == [] and cap.usable == 2, "84% used: 16% left, above the reserve"

    mgmt.files[0] = _reserved(85)
    cap, changes = slots.refresh(conn, cfg, mgmt)
    assert mgmt.patches == [{"name": "codex-reserved@example.com-pro.json", "disabled": True}]
    assert [(a, e) for a, e, _ in changes] == [("disabled", "reserved@example.com")]
    assert "keeping its 15% reserve until" in changes[0][2]
    assert cap.usable == 1, "the parked account no longer counts"

    assert slots.refresh(conn, cfg, mgmt)[1] == [], "no re-enable before the window resets"
    assert len(mgmt.patches) == 1

    # the window resets: the proxy still shows the last headers, with a reset time now past
    mgmt.files[0] = _reserved(85, reset_in_s=-60, disabled=True)
    with tx(conn):
        parked = json.loads(
            conn.execute("SELECT value FROM kv WHERE key=?", (slots.KV_PARKED,)).fetchone()[0]
        )
        parked["codex-reserved@example.com-pro.json"]["until"] = fmt_ts(now_dt())
        kv_set(conn, slots.KV_PARKED, json.dumps(parked))
    cap, changes = slots.refresh(conn, cfg, mgmt)
    assert mgmt.patches[-1] == {"name": "codex-reserved@example.com-pro.json", "disabled": False}
    assert [(a, e) for a, e, _ in changes] == [("enabled", "reserved@example.com")]
    assert cap.usable == 2, "a rolled-over window counts as fresh"


def test_an_operator_disabled_reserved_account_is_never_re_enabled(cfg, conn):
    cfg = replace(cfg, account_reserves=RESERVED)
    mgmt = _Mgmt([_reserved(10, reset_in_s=-60, disabled=True)])
    assert slots.refresh(conn, cfg, mgmt)[1] == [] and mgmt.patches == []


def test_a_failed_disable_is_retried_next_time(cfg, conn):
    cfg = replace(cfg, account_reserves=RESERVED)

    class Refusing(_Mgmt):
        def patch(self, path: str, body: dict[str, Any]) -> Any:
            raise slots.QuotaError("HTTP 500")

    assert slots.refresh(conn, cfg, Refusing([_reserved(90)]))[1] == []
    assert slots.stored_accounts(conn)[1][0].usable is False  # type: ignore[index]
    mgmt = _Mgmt([_reserved(90)])  # the PATCH really failed: still enabled, so try again
    assert [a for a, _, _ in slots.refresh(conn, cfg, mgmt)[1]] == ["disabled"]


def test_a_disable_that_lands_but_errors_is_still_ours_to_re_enable(cfg, conn):
    """The proxy applied the PATCH, then the connection dropped before the answer."""
    cfg = replace(cfg, account_reserves=RESERVED)

    class DropsTheAnswer(_Mgmt):
        def patch(self, path: str, body: dict[str, Any]) -> Any:
            super().patch(path, body)
            raise ConnectionResetError("peer reset")

    mgmt = DropsTheAnswer([_reserved(90)])
    slots.refresh(conn, cfg, mgmt)
    assert mgmt.files[0]["disabled"] is True
    assert "codex-reserved@example.com-pro.json" in slots._parked(conn)


def test_a_parked_entry_for_a_deleted_auth_file_is_dropped(cfg, conn):
    cfg = replace(cfg, account_reserves=RESERVED)
    slots.refresh(conn, cfg, _Mgmt([_reserved(90)]))
    assert slots._parked(conn)
    slots.refresh(conn, cfg, _Mgmt([_file("chatgpt3@x", quota=_quota(2))]))
    assert slots._parked(conn) == {}


def test_stale_or_unreadable_reading_falls_back_to_one_unit(cfg, conn):
    _refresh(conn, cfg, _Mgmt(_accounts(3)))
    assert slots.capacity(conn, cfg).scale == 3
    with tx(conn):
        blob = json.loads(
            conn.execute("SELECT value FROM kv WHERE key=?", (slots.KV_ACCOUNTS,)).fetchone()[0]
        )
        blob["at"] = fmt_ts(now_dt() - slots.READING_MAX_AGE * 2)
        kv_set(conn, slots.KV_ACCOUNTS, json.dumps(blob))
    cap = slots.capacity(conn, cfg)
    assert (cap.scale, cap.ceiling) == (1, 3) and "stale" in cap.reason

    class Broken:
        def get(self, path: str) -> Any:
            return {"files": None}

    assert _refresh(conn, cfg, Broken()).scale == 1


def test_scaling_disabled_keeps_static_slots(cfg, conn):
    static = replace(cfg, account_scale_max=1)
    _refresh(conn, static, _Mgmt(_accounts(3)))
    assert slots.capacity(conn, static).ceiling == 3


def test_scheduler_admits_the_scaled_capacity(cfg, conn):
    _refresh(conn, cfg, _Mgmt(_accounts(2)))
    queue(conn, cfg, 8)  # below the backlog threshold: no lending
    assert len(schedule(conn, cfg, spawn=False)) == 4, "2 normal x 2 accounts"
    assert schedule(conn, cfg, spawn=False) == []


def test_scheduler_backlog_may_lend_once_there_are_spare_priority_slots(cfg, conn):
    _refresh(conn, cfg, _Mgmt(_accounts(3)))
    queue(conn, cfg, 12)
    assert len(schedule(conn, cfg, spawn=False)) == 7, "6 normal + 1 lent; 2 priority kept"


def test_page_mentions_on_call_in_the_channel_and_dms_the_operator(cfg):
    sent: list[list[str]] = []

    def runner(argv):
        sent.append(list(argv))
        return subprocess.CompletedProcess(list(argv), 0, "", "")

    n = Notifier(cfg, runner=runner)
    assert n.page("entering DEGRADED mode: x")
    (channel, dm) = sent
    assert channel[channel.index("--target") + 1] == "channel:C0AEQ5D7SJ3"
    body = channel[channel.index("-m") + 1]
    assert body.startswith("<@UCW1VE04T> <@U02CNG35EGG> :rotating_light:")
    assert "entering DEGRADED mode: x" in body
    assert dm[dm.index("--target") + 1] == "user:UCW1VE04T"
    assert "<@" not in dm[dm.index("-m") + 1]
    sent.clear()
    n.page("leaving degraded mode", resolved=True)
    assert all("<@" not in argv[argv.index("-m") + 1] for argv in sent), "all-clear pings nobody"


def test_page_survives_one_target_failing(cfg):
    def runner(argv):
        rc = 1 if "channel:C0AEQ5D7SJ3" in argv else 0
        return subprocess.CompletedProcess(list(argv), rc, "", "boom")

    assert Notifier(cfg, runner=runner).page("x"), "the DM copy still landed"


def test_a_hand_re_enabled_account_is_forgotten_so_a_later_operator_disable_sticks(cfg, conn):
    cfg = replace(cfg, account_reserves=RESERVED)
    mgmt = _Mgmt([_reserved(90)])
    slots.refresh(conn, cfg, mgmt)
    assert "codex-reserved@example.com-pro.json" in slots._parked(conn)
    mgmt.files[0] = _reserved(10)  # someone re-enabled it by hand and it has quota again
    slots.refresh(conn, cfg, mgmt)
    assert slots._parked(conn) == {}
    mgmt.files[0] = _reserved(10, reset_in_s=-60, disabled=True)  # now disabled on purpose
    n = len(mgmt.patches)
    assert slots.refresh(conn, cfg, mgmt)[1] == [] and len(mgmt.patches) == n


def test_parked_account_is_listed_as_out_on_purpose(cfg, conn):
    cfg = replace(cfg, account_reserves=RESERVED)
    slots.refresh(conn, cfg, _Mgmt([_reserved(90)]))
    (acct,) = slots.stored_accounts(conn)[1]  # type: ignore[index]
    assert (acct.name, acct.usable, acct.reason) == (
        "reserved@example.com",
        False,
        "parked to keep its reserve",
    )


def test_odd_proxy_data_never_stops_the_refresh(cfg, conn):
    weird = [
        _file("naive", cooldowns=[{"scope": "credential", "retry_at": "2099-09-30T02:38:46"}]),
        _file("bad-ts", cooldowns=[{"scope": "credential", "retry_at": "soon"}]),
        _file("bad-quota", quota="n/a"),
        _file("bad-signals", quota={"signals": ["x"]}),
    ]
    cap, _ = slots.refresh(conn, cfg, _Mgmt(weird))
    accounts = {a.name: a for a in slots.stored_accounts(conn)[1]}  # type: ignore[index]
    assert not accounts["naive"].usable, "a timestamp without an offset is read as UTC"
    assert accounts["bad-ts"].usable and accounts["bad-quota"].usable
    assert cap.usable == 3


def test_reserve_emails_match_case_insensitively(tmp_path, skills_dir):
    from reviewsys import config as cfg_mod
    from tests.test_config import _write

    p = _write(tmp_path, skills_dir, "max_concurrent = 2")
    p.write_text(
        p.read_text().replace(
            "account_reserves = {}", 'account_reserves = { "Reserved@Example.com" = 0.15 }'
        )
    )
    assert cfg_mod.load(p).account_reserves == RESERVED
