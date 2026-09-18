"""Degraded mode: reviews keep running on stand-in models while the OpenAI pool is out of
quota, and everything published says so."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from reviewsys import config as cfg_mod
from reviewsys import degraded, worker
from reviewsys.daemon import Daemon
from reviewsys.db import kv_get, tx
from reviewsys.github import gate_body
from reviewsys.ingest import enqueue_head
from reviewsys.models import RunStatus, Trigger
from reviewsys.publish import DEGRADED_BADGE
from reviewsys.queue_status import queue_body
from reviewsys.scheduler import schedule

HEAD = "a" * 40
MUSE = "muse-spark-1.3-contributor"
DEGRADED_BLOCK = {
    "sentinel": "gpt-6-astra",
    "phase1_effort_cap": "high",
    "substitutes": {
        "gpt-6-astra": {"model": MUSE, "effort_cap": "xhigh"},
        "gpt-5.6-sol": {"model": MUSE},
        "gpt-5.6-terra": {"model": "glm-5.3-flash"},
        "gpt-5.6-luna": {"model": "glm-5.3-flash"},
    },
}


def _cfg_with(skills_dir, tmp_path, block=DEGRADED_BLOCK, **policy):
    raw = json.loads((skills_dir / "config.json").read_text())
    raw["review_model_policy"]["degraded"] = block
    raw["review_model_policy"].update(policy)
    (skills_dir / "config.json").write_text(json.dumps(raw))
    return cfg_mod.load(tmp_path / "config.toml")


@pytest.fixture(autouse=True)
def fake_git(monkeypatch, tmp_path):
    from reviewsys.steps import worktree as wt

    monkeypatch.setattr(wt, "ensure_mirror", lambda mirrors, repo: tmp_path / "mirror")
    monkeypatch.setattr(wt, "fetch_head", lambda mirror, number, sha: None)

    def create(mirror, wts, name, sha):
        p = wts / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(wt, "create_worktree", create)
    monkeypatch.setattr(wt, "remove_worktree", lambda mirror, path: None)
    monkeypatch.setattr(wt, "merge_base", lambda worktree, base, sha: "b" * 40)


def _verifier(action="COMMENT"):
    return {
        "summary": "Looks fine.",
        "review_action": action,
        "findings": [],
        "dropped_findings": [],
        "out_of_scope_findings": [],
        "coderabbit_reactions": [],
        "prerequisite_adjudications": [],
        "adjudication_complete": True,
    }


def _run(cfg, conn, gh, lanes, *, prober, number=1):
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", number, HEAD, Trigger.MENTION)
    (rid,) = schedule(conn, cfg, spawn=False)
    status = worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False, prober=prober)
    return rid, status


def _reviewer_calls(lanes):
    return [s for s in lanes.calls if s.role not in ("selector", "triage", "verifier", "repair")]


EXHAUSTED = lambda model: (True, f"`{model}` unavailable: cooling down")  # noqa: E731
FINE = lambda model: (False, f"{model} answered")  # noqa: E731


# ---- config ----


def test_degraded_block_is_parsed_and_validated(cfg, skills_dir, tmp_path):
    assert cfg.policy.degraded is None
    c = _cfg_with(skills_dir, tmp_path)
    d = c.policy.degraded
    assert d is not None and d.sentinel == "gpt-6-astra" and d.phase1_effort_cap == "high"
    assert d.substitutes["gpt-6-astra"].model == MUSE
    assert d.substitutes["gpt-6-astra"].effort_cap == "xhigh"
    assert d.substitutes["gpt-5.6-sol"].effort_cap is None
    # resolve: model swapped, effort capped by the stand-in, origin remembered, idempotent
    lm = d.resolve(cfg_mod.LaneModel(agent="phase2-reviewer", model="gpt-6-astra", effort="max"))
    assert (lm.model, lm.effort, lm.substitute_for) == (MUSE, "xhigh", "gpt-6-astra")
    assert d.resolve(lm) is lm
    untouched = cfg_mod.LaneModel(agent="x", model="glm-5.3-flash", effort="max")
    assert d.resolve(untouched) is untouched
    assert d.phase1_effort("max") == "high" and d.phase1_effort("low") == "low"
    # bad blocks fail closed at load, not at run time
    for bad in (
        {"substitutes": {"gpt-6-astra": MUSE}},  # no sentinel
        {"sentinel": "gpt-6-astra", "substitutes": {}},
        {"sentinel": "gpt-6-astra", "substitutes": {"gpt-5.6-sol": MUSE}},  # sentinel unmapped
        {"sentinel": "gpt-6-astra", "substitutes": {"gpt-6-astra": {"model": ""}}},
        {"sentinel": "a", "substitutes": {"a": "b", "b": "c"}},  # chained stand-in
        {"sentinel": "a", "substitutes": {"a": "b"}, "phase1_effort_cap": "ultra"},
        {"sentinel": "a", "substitutes": {"a": {"model": "b", "effort_cap": "ultra"}}},
    ):
        with pytest.raises(ValueError):
            _cfg_with(skills_dir, tmp_path, block=bad)
    # a bare string is shorthand for {"model": ...}
    c = _cfg_with(skills_dir, tmp_path, block={"sentinel": "a", "substitutes": {"a": "b"}})
    assert c.policy.degraded is not None
    assert c.policy.degraded.substitutes["a"] == cfg_mod.Substitute(model="b")


# ---- probe classification ----


class _Resp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(code, body):
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(body.encode()))


def test_probe_only_treats_quota_errors_as_exhausted(monkeypatch):
    monkeypatch.setattr(degraded, "_client_key", lambda: "k")
    calls = {}

    def urlopen(req, timeout):
        calls["model"] = json.loads(req.data)["model"]
        err = calls.get("raise")
        if err:
            raise err
        return _Resp()

    monkeypatch.setattr(degraded.urllib.request, "urlopen", urlopen)
    assert degraded.probe("gpt-6-astra") == (False, "gpt-6-astra answered")
    assert calls["model"] == "gpt-6-astra"
    calls["raise"] = _http_error(
        429,
        json.dumps(
            {"error": {"message": "All credentials for model gpt-6-astra are cooling down"}}
        ),
    )
    exhausted, reason = degraded.probe("gpt-6-astra")
    assert (
        exhausted
        and reason
        == "`gpt-6-astra` unavailable: All credentials for model gpt-6-astra are cooling down"
    )
    calls["raise"] = _http_error(429, "The usage limit has been reached")
    assert degraded.probe("gpt-6-astra")[0]
    # a broken proxy or upstream is not "out of quota": swapping models would not help
    calls["raise"] = _http_error(502, "bad gateway")
    exhausted, reason = degraded.probe("gpt-6-astra")
    assert not exhausted and "not a quota failure" in reason
    calls["raise"] = urllib.error.URLError("connection refused")
    exhausted, reason = degraded.probe("gpt-6-astra")
    assert not exhausted and "not treated as exhausted" in reason
    monkeypatch.setattr(degraded, "_client_key", lambda: None)
    assert degraded.probe("gpt-6-astra") == (False, "no proxy client key to probe with")


def test_quota_failure_classifier():
    assert degraded.looks_like_quota_failure(
        "[infra] lane exit 1: API Error: Request rejected (429) · All credentials for model gpt-6-astra are cooling down"
    )
    assert degraded.looks_like_quota_failure(
        "API Error: Request rejected (429) · The usage limit has been reached"
    )
    assert not degraded.looks_like_quota_failure("lane timed out")
    assert not degraded.looks_like_quota_failure("model output is not a JSON object")


# ---- detection, cache, override, hold ----


def test_detect_probes_once_then_caches_and_honours_overrides(cfg, conn, skills_dir, tmp_path):
    c = _cfg_with(skills_dir, tmp_path)
    probes = []

    def prober(model):
        probes.append(model)
        return EXHAUSTED(model)

    st = degraded.detect(conn, c, prober=prober)
    assert st.active and st.source == "probe" and probes == ["gpt-6-astra"]
    st = degraded.detect(conn, c, prober=prober)
    assert st.active and probes == ["gpt-6-astra"], "fresh cached probe is reused"
    st = degraded.detect(conn, c, prober=FINE, refresh=True)
    assert not st.active and st.source == "probe"
    degraded.force(conn, "on")
    assert degraded.detect(conn, c, prober=FINE).active
    assert degraded.detect(conn, c, prober=FINE).source == "forced"
    degraded.force(conn, "off")
    assert not degraded.detect(conn, c, prober=EXHAUSTED, refresh=True).active
    degraded.force(conn, "auto")
    assert kv_get(conn, degraded.KV_FORCE) is None
    with pytest.raises(ValueError):
        degraded.force(conn, "maybe")
    # a policy without a degraded block never probes
    assert degraded.detect(conn, cfg, prober=EXHAUSTED).source == "unconfigured"
    assert degraded.snapshot(conn, cfg) == {"configured": False, "active": False}


def test_lane_observed_failure_holds_the_mode_against_a_passing_probe(conn, skills_dir, tmp_path):
    c = _cfg_with(skills_dir, tmp_path)
    degraded.record_probe(conn, quota_exhausted=True, reason="lane died on 429", hold=True)
    st = degraded.detect(conn, c, prober=FINE, refresh=True)
    assert st.active and st.reason == "lane died on 429", "a 1-token probe cannot lift a hold"
    snap = degraded.snapshot(conn, c)
    assert snap["active"] and snap["last_probe"]["hold_until"]


def test_transition_alerts_once_per_flip(conn):
    on = degraded.State(True, "quota", "probe")
    off = degraded.State(False, "ok", "probe")
    assert degraded.transition(conn, off) is None, "starting in normal mode is not news"
    assert "entering DEGRADED" in (degraded.transition(conn, on) or "")
    assert degraded.transition(conn, on) is None
    assert "leaving degraded" in (degraded.transition(conn, off) or "")
    assert degraded.transition(conn, off) is None


# ---- end to end ----


def test_degraded_run_swaps_every_primary_lane_and_discloses_it(
    conn, gh, lanes, skills_dir, tmp_path
):
    c = _cfg_with(skills_dir, tmp_path)
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier("APPROVE")
    rid, status = _run(c, conn, gh, lanes, prober=EXHAUSTED)
    assert status == RunStatus.DONE
    calls = [(s.role, s.model, s.effort) for s in lanes.calls]
    assert calls[0] == ("selector", "glm-5.3-flash", "low")  # terra -> glm
    assert calls[1] == ("triage", MUSE, "low")  # astra -> muse
    # Phase 1 stays on its own model but the tier's `max` is capped to `high`
    assert calls[2:5] == [
        (r, "glm-5.3-flash", "high") for r in ("general", "always-on", "security-auditor")
    ]
    assert calls[5] == ("verifier", MUSE, "high")  # sol gate verifier -> muse
    assert calls[6:9] == [(r, MUSE, "high") for r in ("general", "always-on", "security-auditor")]
    assert calls[9] == ("verifier", MUSE, "high")
    assert {s.model for s in lanes.calls} == {"glm-5.3-flash", MUSE}, "no primary model touched"
    # persisted + evented
    assert conn.execute("SELECT degraded FROM runs WHERE id=?", (rid,)).fetchone()[0] == 1
    kinds = [r["kind"] for r in conn.execute("SELECT kind FROM events WHERE run_id=?", (rid,))]
    assert "degraded.run" in kinds
    # published: title badge, banner, provenance, and never an approval
    review = gh.posted_reviews[0]
    body = review["body"]
    assert review["event"] == "COMMENT", "a degraded review never approves"
    assert f"## {DEGRADED_BADGE} — Final validation — Phase 1 + Phase 2" in body
    assert f"> **{DEGRADED_BADGE} review.**" in body and "cooling down" in body
    assert f"`gpt-6-astra` → `{MUSE}`" in body
    assert "Phase 1 capped at `high` effort" in body
    assert "- **Degraded mode**: `gpt-6-astra` unavailable: cooling down (detected by probe" in body
    assert f"final verifier: `{MUSE}` (standing in for `gpt-6-astra`)" in body
    assert f"`{MUSE}` (standing in for `gpt-6-astra`) — general (completed, effort high)" in body
    assert (
        "- Triage: `normal` by `muse-spark-1.3-contributor` (standing in for `gpt-6-astra`) (effort low)"
        in body
    )
    assert "⚠️ DEGRADED (stand-in models" in gh.gate_bodies[-1]
    assert (
        gh.gate_bodies[-1].startswith("<!--")
        and "✅ ⚠️ DEGRADED — Final review complete" in gh.gate_bodies[-1]
    )


def test_normal_run_is_untouched_when_the_sentinel_answers(conn, gh, lanes, skills_dir, tmp_path):
    c = _cfg_with(skills_dir, tmp_path)
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier("APPROVE")
    rid, status = _run(c, conn, gh, lanes, prober=FINE)
    assert status == RunStatus.DONE
    assert MUSE not in {s.model for s in lanes.calls}
    assert conn.execute("SELECT degraded FROM runs WHERE id=?", (rid,)).fetchone()[0] == 0
    assert gh.posted_reviews[0]["event"] == "APPROVE"
    assert DEGRADED_BADGE not in gh.posted_reviews[0]["body"]
    assert "DEGRADED" not in gh.gate_bodies[-1]


def test_quota_failure_mid_run_switches_the_rest_of_the_run(conn, gh, lanes, skills_dir, tmp_path):
    c = _cfg_with(skills_dir, tmp_path)
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier()
    lanes.dead_models = {"gpt-6-astra"}
    lanes.dead_stderr = (
        "API Error: Request rejected (429) · All credentials for model gpt-6-astra are cooling down"
    )
    rid, status = _run(c, conn, gh, lanes, prober=FINE)  # the probe said fine; the lane knew better
    assert status == RunStatus.DONE
    # triage on astra dies on 429 (twice, its own retry): that flips the mode, triage is re-run
    # on the stand-in and every later lane avoids the dead pool
    triage_models = [s.model for s in lanes.calls if s.role == "triage"]
    assert triage_models == ["gpt-6-astra", "gpt-6-astra", MUSE]
    assert conn.execute("SELECT tier FROM runs WHERE id=?", (rid,)).fetchone()["tier"] == "normal"
    assert [s.model for s in lanes.calls if s.role == "verifier"] == [MUSE, MUSE]
    assert {s.model for s in _reviewer_calls(lanes)} == {"glm-5.3-flash", MUSE}
    kinds = [r["kind"] for r in conn.execute("SELECT kind FROM events WHERE run_id=?", (rid,))]
    assert "degraded.entered_midrun" in kinds and "degraded.run" not in kinds
    assert "triage.degraded" not in kinds, "the re-run triage succeeded"
    assert conn.execute("SELECT degraded FROM runs WHERE id=?", (rid,)).fetchone()[0] == 1
    body = gh.posted_reviews[0]["body"]
    assert DEGRADED_BADGE in body and "(detected by lane" in body
    assert "triage lane failed on quota" in body
    # the failure is remembered with a hold: the next run starts degraded without a probe
    blob = json.loads(kv_get(conn, degraded.KV_PROBE))
    assert blob["quota_exhausted"] and blob["hold_until"]
    lanes.calls.clear()
    gh.pr["number"] = 2
    rid2, status2 = _run(c, conn, gh, lanes, prober=FINE, number=2)
    assert status2 == RunStatus.DONE
    assert "gpt-6-astra" not in {s.model for s in lanes.calls}
    kinds2 = [r["kind"] for r in conn.execute("SELECT kind FROM events WHERE run_id=?", (rid2,))]
    assert "degraded.run" in kinds2


def test_reviewer_lane_quota_failure_switches_mid_run(conn, gh, lanes, skills_dir, tmp_path):
    """The side lanes were fine (say, on a credential that just ran dry); the Phase-1 gate
    verifier is the first lane to hit the 429. It gets its attempts again on the stand-in."""
    c = _cfg_with(skills_dir, tmp_path)
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier()
    lanes.dead_models = {"gpt-5.6-sol"}
    lanes.dead_stderr = "API Error: Request rejected (429) · The usage limit has been reached"
    rid, status = _run(c, conn, gh, lanes, prober=FINE)
    assert status == RunStatus.DONE
    verifiers = [(s.model, s.effort) for s in lanes.calls if s.role == "verifier"]
    assert verifiers == [("gpt-5.6-sol", "high"), (MUSE, "high"), (MUSE, "high")]
    rows = conn.execute(
        "SELECT model, status FROM lanes WHERE run_id=? AND role='verifier' ORDER BY id", (rid,)
    ).fetchall()
    assert [(r["model"], r["status"]) for r in rows] == [
        ("gpt-5.6-sol", "failed"),
        (MUSE, "completed"),
        (MUSE, "completed"),
    ]
    assert all(s.model == MUSE for s in _reviewer_calls(lanes) if s.model != "glm-5.3-flash")
    # Phase 1 had already run at the tier's `max` before the flip; that is disclosed as-is
    assert [s.effort for s in _reviewer_calls(lanes) if s.model == "glm-5.3-flash"] == ["max"] * 3
    body = gh.posted_reviews[0]["body"]
    assert "`gpt-5.6-sol` lane failed on quota" in body


def test_non_quota_failure_does_not_degrade(conn, gh, lanes, skills_dir, tmp_path):
    c = _cfg_with(skills_dir, tmp_path)
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scape_findings": []}
    lanes.verifier["default"] = _verifier()
    lanes.dead_models = {"gpt-5.6-sol"}
    lanes.dead_stderr = "segfault in the launcher"
    rid, status = _run(c, conn, gh, lanes, prober=FINE)
    assert status == RunStatus.FAILED
    assert MUSE not in {s.model for s in lanes.calls}
    assert conn.execute("SELECT degraded FROM runs WHERE id=?", (rid,)).fetchone()[0] == 0
    assert (
        kv_get(conn, degraded.KV_PROBE) is None
        or not json.loads(kv_get(conn, degraded.KV_PROBE))["quota_exhausted"]
    )


def test_without_a_degraded_block_a_quota_failure_still_fails_the_run(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier()
    lanes.dead_models = {"gpt-5.6-sol"}
    rid, status = _run(cfg, conn, gh, lanes, prober=FINE)
    assert status == RunStatus.FAILED
    assert conn.execute("SELECT degraded FROM runs WHERE id=?", (rid,)).fetchone()[0] == 0


def test_degraded_mode_keeps_both_phases_under_a_deep_backlog(
    conn, gh, lanes, skills_dir, tmp_path
):
    c = _cfg_with(skills_dir, tmp_path)
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier()
    with tx(conn):
        for i in range(c.backlog_skip_phase1_above + 5):
            enqueue_head(conn, c, "dashpay/platform", 100 + i, f"{i:040x}", Trigger.NEW_PR)
    rid, status = _run(c, conn, gh, lanes, prober=EXHAUSTED)
    assert status == RunStatus.DONE
    assert (
        conn.execute(
            "SELECT status FROM steps WHERE run_id=? AND name='phase1'", (rid,)
        ).fetchone()["status"]
        == "ok"
    )
    assert {s.model for s in _reviewer_calls(lanes)} == {"glm-5.3-flash", MUSE}
    assert "Phase 1 + Phase 2" in gh.posted_reviews[0]["body"]
    assert "queue backlog" not in gh.posted_reviews[0]["body"]


def test_degraded_preliminary_review_still_requests_changes(conn, gh, lanes, skills_dir, tmp_path):
    c = _cfg_with(skills_dir, tmp_path)
    blocker = {
        "file": "f.rs",
        "line_start": 11,
        "line_end": 12,
        "severity": "blocking",
        "confidence": 0.9,
        "category": "logic",
        "title": "Overflow",
        "body": "x",
    }
    lanes.reviewer["default"] = {
        "summary": "bad",
        "findings": [blocker],
        "out_of_scope_findings": [],
    }
    lanes.verifier["preliminary"] = {**_verifier(), "findings": [blocker]}
    _rid, status = _run(c, conn, gh, lanes, prober=EXHAUSTED)
    assert status == RunStatus.DONE
    review = gh.posted_reviews[0]
    assert review["event"] == "REQUEST_CHANGES"
    assert f"## {DEGRADED_BADGE} — Preliminary review — Phase 1 blocker gate" in review["body"]
    assert "⛔ ⚠️ DEGRADED — Blockers found" in gh.gate_bodies[-1]


# ---- daemon, status, comments ----


def test_daemon_probes_periodically_and_alerts_on_transition(
    cfg, conn, gh, notifier, skills_dir, tmp_path
):
    c = _cfg_with(skills_dir, tmp_path)
    state = {"exhausted": True}
    d = Daemon(
        c, conn, gh=gh, notifier=notifier, spawn=False, prober=lambda m: (state["exhausted"], "r")
    )
    assert any(t.name == "degraded" for t in d.tasks)
    assert d.t_degraded()["active"] is True
    assert [t for k, t in notifier.sent if k == "alert" and "DEGRADED" in t]
    n = len(notifier.sent)
    d.t_degraded()
    assert len(notifier.sent) == n, "no repeat alert while the mode is unchanged"
    state["exhausted"] = False
    assert d.t_degraded()["active"] is False
    assert any("leaving degraded" in t for _, t in notifier.sent[n:])
    kinds = [r["kind"] for r in conn.execute("SELECT kind FROM events")]
    assert kinds.count("degraded.transition") == 2
    # no task at all without a policy block (`cfg` was loaded before the block was added)
    assert cfg.policy.degraded is None
    plain = Daemon(cfg, conn, gh=gh, notifier=notifier, spawn=False)
    assert not any(t.name == "degraded" for t in plain.tasks)


def test_status_snapshot_and_cli_show_the_mode(conn, gh, skills_dir, tmp_path, capsys):
    from reviewsys import cli
    from reviewsys.status import snapshot

    c = _cfg_with(skills_dir, tmp_path)
    assert snapshot(conn, c)["degraded"]["active"] is False
    degraded.force(conn, "on")
    assert snapshot(conn, c)["degraded"] == {
        **snapshot(conn, c)["degraded"],
        "active": True,
        "forced": "on",
        "sentinel": "gpt-6-astra",
        "phase1_effort_cap": "high",
    }
    assert cli.main(["--config", str(tmp_path / "config.toml"), "status"]) == 0
    assert "mode: DEGRADED forced=on" in capsys.readouterr().out
    assert cli.main(["--config", str(tmp_path / "config.toml"), "degraded", "off"]) == 0
    assert "set to off" in capsys.readouterr().out
    assert kv_get(conn, degraded.KV_FORCE) == "off"
    assert cli.main(["--config", str(tmp_path / "config.toml"), "degraded", "auto"]) == 0
    capsys.readouterr()
    assert kv_get(conn, degraded.KV_FORCE) is None
    assert cli.main(["--config", str(tmp_path / "config.toml"), "degraded"]) == 0
    assert '"configured": true' in capsys.readouterr().out


def test_comment_renderers_carry_the_tag():
    assert "DEGRADED" not in gate_body("in_progress", HEAD)
    assert "🔍 ⚠️ DEGRADED — Review in progress" in gate_body("in_progress", HEAD, degraded=True)
    assert "⚠️ ⚠️ DEGRADED — Automated review could not complete" in gate_body(
        "failed", HEAD, degraded=True, reason="x"
    )
    plain = queue_body(HEAD, position=1, eta_minutes=10, run_minutes=30, priority=False)
    tagged = queue_body(
        HEAD, position=1, eta_minutes=10, run_minutes=30, priority=False, degraded=True
    )
    assert "DEGRADED" not in plain and "🕓 ⚠️ DEGRADED — Queued" in tagged
    assert "stand-in models and be marked as degraded" in tagged
