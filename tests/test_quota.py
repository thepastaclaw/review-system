"""Quota-gated Phase-1 model ladder: selection rules and the provider parsers."""

from __future__ import annotations

import json
from typing import Any

import pytest

from reviewsys import quota
from reviewsys.config import LaneModel, QuotaSource

GEMINI = LaneModel(
    "phase1-reviewer", "gemini-3.8-flash-high", "high", QuotaSource("antigravity", "Gemini Models")
)
GLM = LaneModel("phase1-reviewer", "glm-5.3-flash", "max", QuotaSource("zai"))
MUSE = LaneModel("phase1-reviewer", "muse-spark-1.3-contributor", "xhigh")
LADDER = (GEMINI, GLM, MUSE)


def _status(*remaining: float) -> quota.QuotaStatus:
    return quota.QuotaStatus(
        "acct", tuple(quota.Window(f"w{i}", r) for i, r in enumerate(remaining))
    )


def _reader(table: dict[str, Any]):
    def read(source: QuotaSource) -> quota.QuotaStatus:
        v = table[source.provider]
        if isinstance(v, Exception):
            raise v
        return v

    return read


def test_first_rung_with_quota_wins():
    c = quota.choose(LADDER, 0.15, _reader({"antigravity": _status(0.9, 0.5)}))
    assert c.model is GEMINI and c.skipped == ()
    assert "antigravity quota" in c.reason and "50% left" in c.reason
    assert c.remaining(LADDER) == (GLM, MUSE)


def test_smallest_window_gates_and_next_rung_is_tried():
    reader = _reader({"antigravity": _status(0.99, 0.05), "zai": _status(0.4, 0.2)})
    c = quota.choose(LADDER, 0.15, reader)
    assert c.model is GLM
    assert c.skipped == (
        quota.Skipped(GEMINI.model, "antigravity below 15% reserve: w0 99% left, w1 5% left"),
    )


def test_lookup_failure_skips_the_rung():
    reader = _reader({"antigravity": quota.QuotaError("boom acct@x"), "zai": _status(1.0)})
    c = quota.choose(LADDER, 0.15, reader)
    assert c.model is GLM
    # the published reason never carries the error text; the event detail does
    assert c.skipped[0] == quota.Skipped(
        GEMINI.model, "antigravity quota lookup failed", "boom acct@x"
    )
    assert c.as_dict()["skipped"] == [
        {"model": GEMINI.model, "reason": "antigravity quota lookup failed"}
    ]
    assert "boom acct@x" in c.log_line()


def test_any_reader_exception_skips_the_rung():
    reader = _reader(
        {"antigravity": TypeError("'NoneType' object is not iterable"), "zai": _status(1.0)}
    )
    assert quota.choose(LADDER, 0.15, reader).model is GLM


def test_empty_ladder_is_a_config_error():
    with pytest.raises(ValueError):
        quota.choose((), 0.15, _reader({}))


def test_resume_below_a_failed_rung_keeps_earlier_skips():
    prior = (quota.Skipped(GEMINI.model, "lane failed", "429"),)
    c = quota.choose((GLM, MUSE), 0.15, _reader({"zai": _status(0.01)}), skipped=prior)
    assert c.model is MUSE and c.reason == "not quota-gated"
    assert [s.model for s in c.skipped] == [GEMINI.model, GLM.model]


def test_ungated_last_rung_always_resolves():
    reader = _reader({"antigravity": _status(0.0), "zai": quota.QuotaError("down")})
    c = quota.choose(LADDER, 0.15, reader)
    assert c.model is MUSE and c.reason == "not quota-gated"
    assert [s.model for s in c.skipped] == [GEMINI.model, GLM.model]


def test_fully_gated_ladder_falls_through_to_last_rung():
    reader = _reader({"antigravity": _status(0.01), "zai": _status(0.02)})
    c = quota.choose((GEMINI, GLM), 0.15, reader)
    assert c.model is GLM
    assert c.reason.startswith("every rung short of quota; last rung used")
    assert c.skipped == (quota.Skipped(GEMINI.model, "antigravity below 15% reserve: w0 1% left"),)


def test_single_rung_never_consults_quota():
    def reader(_: QuotaSource) -> quota.QuotaStatus:
        raise AssertionError("must not be called")

    assert quota.choose((GLM,), 0.15, reader).reason == "only candidate"


class FakeManagement(quota.Management):
    """Scripted management API: `routes` maps (method, path) -> JSON; `upstream` maps a URL to
    the (status, body) the proxy's api-call would relay."""

    def __init__(self, routes: dict[tuple[str, str], Any], upstream: dict[str, tuple[int, Any]]):
        super().__init__()
        self.routes, self.upstream = routes, upstream
        self.api_calls: list[dict[str, Any]] = []

    def _call(self, method: str, path: str, data: bytes | None) -> Any:
        if path == "api-call" and method == "POST":
            req = json.loads(data or b"{}")
            self.api_calls.append(req)
            status, body = self.upstream[req["url"]]
            return {"status_code": status, "body": json.dumps(body), "header": {}}
        return self.routes[(method, path)]


AG_FILES = {
    "files": [
        {"provider": "codex", "auth_index": "c0", "email": "x@y"},
        {
            "provider": "antigravity",
            "auth_index": "ag-disabled",
            "project_id": "p",
            "disabled": True,
        },
        {"provider": "antigravity", "auth_index": "ag1", "project_id": "proj-1", "email": "a@g"},
    ]
}
AG_SUMMARY = {
    "groups": [
        {
            "displayName": "Gemini Models",
            "buckets": [
                {
                    "bucketId": "gemini-weekly",
                    "window": "weekly",
                    "remainingFraction": 0.7,
                    "resetTime": "2026-09-17T15:14:15Z",
                },
                {"bucketId": "gemini-5h", "window": "5h", "remainingFraction": 0.2},
            ],
        },
        {
            "displayName": "Claude and GPT models",
            "buckets": [{"bucketId": "3p-5h", "window": "5h", "remainingFraction": 1}],
        },
    ]
}


def test_antigravity_reads_only_the_named_group_with_the_stored_credential():
    mgmt = FakeManagement(
        {("GET", "auth-files"): AG_FILES}, {quota.ANTIGRAVITY_QUOTA_URL: (200, AG_SUMMARY)}
    )
    st = quota.read(mgmt, QuotaSource("antigravity", "Gemini Models"))
    assert st.account == "a@g" and st.remaining == pytest.approx(0.2)
    assert [w.name for w in st.windows] == ["weekly", "5h"]
    assert st.windows[0].reset_at == "2026-09-17T15:14:15Z"
    (call,) = mgmt.api_calls
    assert call["auth_index"] == "ag1" and json.loads(call["data"]) == {"project": "proj-1"}
    assert call["header"]["Authorization"] == "Bearer $TOKEN$"
    assert call["header"]["User-Agent"] == quota.ANTIGRAVITY_USER_AGENT


def test_antigravity_missing_group_or_credential_is_an_error():
    mgmt = FakeManagement(
        {("GET", "auth-files"): AG_FILES}, {quota.ANTIGRAVITY_QUOTA_URL: (200, AG_SUMMARY)}
    )
    with pytest.raises(quota.QuotaError, match="no quota group"):
        quota.read(mgmt, QuotaSource("antigravity", "Nope"))
    mgmt = FakeManagement({("GET", "auth-files"): {"files": []}}, {})
    with pytest.raises(quota.QuotaError, match="no usable credential"):
        quota.read(mgmt, QuotaSource("antigravity", "Gemini Models"))


def test_go_null_slices_are_quota_errors_not_crashes():
    for files in ({"files": None}, None, {"files": [None]}):
        mgmt = FakeManagement({("GET", "auth-files"): files}, {})
        with pytest.raises(quota.QuotaError):
            quota.read(mgmt, QuotaSource("antigravity", "Gemini Models"))
    for summary in (
        {"groups": None},
        {"groups": [{"displayName": "Gemini Models", "buckets": None}]},
        None,
    ):
        mgmt = FakeManagement(
            {("GET", "auth-files"): AG_FILES}, {quota.ANTIGRAVITY_QUOTA_URL: (200, summary)}
        )
        with pytest.raises(quota.QuotaError, match="no quota group"):
            quota.read(mgmt, QuotaSource("antigravity", "Gemini Models"))
    for conf in (
        {"openai-compatibility": None},
        {"openai-compatibility": [{"base-url": "https://api.z.ai/x", "api-key-entries": None}]},
    ):
        mgmt = FakeManagement({("GET", "openai-compatibility"): conf}, {})
        with pytest.raises(quota.QuotaError):
            quota.read(mgmt, QuotaSource("zai"))
    for limit in (
        {"data": None},
        {"data": {"limits": None}},
        {"data": {"limits": [{"type": "CREDIT_LIMIT", "unit": "x"}]}},
    ):
        mgmt = FakeManagement(
            {("GET", "openai-compatibility"): ZAI_CONF}, {quota.ZAI_QUOTA_URL: (200, limit)}
        )
        with pytest.raises(quota.QuotaError, match="no CREDIT_LIMIT"):
            quota.read(mgmt, QuotaSource("zai"))


def test_missing_remaining_fraction_means_exhausted():
    # proto3 JSON omits zero-valued fields, so an absent remainingFraction is 0
    summary = {"groups": [{"displayName": "Gemini Models", "buckets": [{"window": "5h"}]}]}
    mgmt = FakeManagement(
        {("GET", "auth-files"): AG_FILES}, {quota.ANTIGRAVITY_QUOTA_URL: (200, summary)}
    )
    assert quota.read(mgmt, QuotaSource("antigravity", "Gemini Models")).remaining == 0.0


def test_account_selector_pins_one_credential():
    files = {
        "files": [
            {"provider": "antigravity", "auth_index": "a", "project_id": "p", "email": "low@g"},
            {"provider": "antigravity", "auth_index": "b", "project_id": "p", "email": "high@g"},
        ]
    }

    class Mgmt(FakeManagement):
        def _call(self, method, path, data):
            if path == "api-call":
                idx = json.loads(data)["auth_index"]
                body = {
                    "groups": [
                        {
                            "displayName": "Gemini Models",
                            "buckets": [
                                {"window": "5h", "remainingFraction": 0.1 if idx == "a" else 0.8}
                            ],
                        }
                    ]
                }
                return {"status_code": 200, "body": json.dumps(body)}
            return files

    st = quota.read(Mgmt({}, {}), QuotaSource("antigravity", "Gemini Models", account="low@g"))
    assert st.account == "low@g" and st.remaining == pytest.approx(0.1)


def test_one_bad_credential_does_not_hide_a_good_one():
    files = {
        "files": [
            {"provider": "antigravity", "auth_index": "bad", "project_id": "p", "email": "bad@g"},
            {"provider": "antigravity", "auth_index": "ok", "project_id": "p", "email": "ok@g"},
        ]
    }

    class Mgmt(FakeManagement):
        def _call(self, method, path, data):
            if path == "api-call":
                idx = json.loads(data)["auth_index"]
                if idx == "bad":
                    return {"status_code": 403, "body": "no license"}
                body = {
                    "groups": [
                        {
                            "displayName": "Gemini Models",
                            "buckets": [{"window": "5h", "remainingFraction": 0.6}],
                        }
                    ]
                }
                return {"status_code": 200, "body": json.dumps(body)}
            return files

    st = quota.read(Mgmt({}, {}), QuotaSource("antigravity", "Gemini Models"))
    assert st.account == "ok@g" and st.remaining == pytest.approx(0.6)


ZAI_CONF = {
    "openai-compatibility": [
        {
            "name": "openrouter",
            "base-url": "https://openrouter.ai/api/v1",
            "api-key-entries": [{"auth-index": "or1"}],
        },
        {
            "name": "zai",
            "base-url": "https://api.z.ai/api/coding/paas/v4",
            "api-key-entries": [{"auth-index": "z1"}],
        },
    ]
}
ZAI_LIMIT = {
    "code": 200,
    "data": {
        "limits": [
            {
                "type": "CREDIT_LIMIT",
                "unit": 3,
                "number": 5,
                "percentage": 1,
                "nextResetTime": 1789064327849,
            },
            {
                "type": "CREDIT_LIMIT",
                "unit": 6,
                "number": 1,
                "percentage": 54,
                "nextResetTime": 1789381606994,
            },
            {"type": "TIME_LIMIT", "unit": 5, "number": 1, "percentage": 99},
        ],
        "level": "pro",
    },
}


def test_zai_reads_credit_windows_only():
    mgmt = FakeManagement(
        {("GET", "openai-compatibility"): ZAI_CONF}, {quota.ZAI_QUOTA_URL: (200, ZAI_LIMIT)}
    )
    st = quota.read(mgmt, QuotaSource("zai"))
    assert st.account == "zai/z1"
    assert [(w.name, round(w.remaining, 2)) for w in st.windows] == [("5h", 0.99), ("weekly", 0.46)]
    assert st.remaining == pytest.approx(0.46)
    assert mgmt.api_calls[0]["auth_index"] == "z1"


def test_upstream_failure_is_a_quota_error():
    mgmt = FakeManagement(
        {("GET", "openai-compatibility"): ZAI_CONF},
        {quota.ZAI_QUOTA_URL: (401, {"msg": "bad key"})},
    )
    with pytest.raises(quota.QuotaError, match="HTTP 401"):
        quota.read(mgmt, QuotaSource("zai"))


def test_best_account_wins_when_several_credentials_exist():
    files = {
        "files": [
            {"provider": "antigravity", "auth_index": "a", "project_id": "p", "email": "low@g"},
            {"provider": "antigravity", "auth_index": "b", "project_id": "p", "email": "high@g"},
        ]
    }

    class Mgmt(FakeManagement):
        def _call(self, method, path, data):
            if path == "api-call":
                idx = json.loads(data)["auth_index"]
                frac = 0.1 if idx == "a" else 0.8
                body = {
                    "groups": [
                        {
                            "displayName": "Gemini Models",
                            "buckets": [{"window": "5h", "remainingFraction": frac}],
                        }
                    ]
                }
                return {"status_code": 200, "body": json.dumps(body)}
            return files

    st = quota.read(Mgmt({}, {}), QuotaSource("antigravity", "Gemini Models"))
    assert st.account == "high@g" and st.remaining == pytest.approx(0.8)
