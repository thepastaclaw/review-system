"""Model comparison: on sampled runs every Phase-2 reviewer also runs on a second model, the
verifier weighs both sets under neutral labels, and `reviewsys compare` credits kept findings
per model. A comparison lane never fails, holds up or degrades the run."""

from __future__ import annotations

import dataclasses
import json
import os
import threading
import time

import pytest

from reviewsys import compare, worker
from reviewsys import config as cfg_mod
from reviewsys.db import tx
from reviewsys.ingest import enqueue_head
from reviewsys.lane import LaneResult, LaneSpec
from reviewsys.models import RunStatus, Trigger
from reviewsys.scheduler import schedule
from reviewsys.steps import worktree as wt

HEAD = "a" * 40
PRIMARY, SECOND = "gpt-6.1-sol", "gpt-6-astra"
ROLES = {"general", "always-on", "security-auditor"}
NON_REVIEWER_ROLES = {"selector", "triage", "prep", "verifier", "repair"}


@pytest.fixture(autouse=True)
def fake_git(monkeypatch, tmp_path):
    monkeypatch.setattr(wt, "ensure_mirror", lambda mirrors, repo: tmp_path / "mirror")
    monkeypatch.setattr(wt, "fetch_head", lambda mirror, number, sha: None)

    def create(mirror, wts, name, sha):
        p = wts / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(wt, "create_worktree", create)
    monkeypatch.setattr(wt, "remove_worktree", lambda mirror, path: None)
    monkeypatch.setattr(wt, "merge_base", lambda worktree, base, sha: "b" * 40)


def _finding(title, severity="suggestion", line=12):
    return {
        "file": "f.rs",
        "line_start": line,
        "line_end": line,
        "severity": severity,
        "confidence": 0.9,
        "category": "logic",
        "title": title,
        "body": "details",
    }


def _verifier(findings=()):
    return {
        "summary": "ok",
        "review_action": "COMMENT",
        "findings": list(findings),
        "dropped_findings": [],
        "out_of_scope_findings": [],
        "coderabbit_reactions": [],
        "prerequisite_adjudications": [],
        "adjudication_complete": True,
    }


def _is_phase2_reviewer(spec):
    return spec.role not in NON_REVIEWER_ROLES and "set to `final`" in spec.prompt


def _with_comparison(cfg, *, fraction=1.0, tiers=("critical",)):
    """As in production: everything on the primary, only comparison lanes on the second."""
    pol = cfg.policy
    pol = dataclasses.replace(
        pol,
        triage=dataclasses.replace(pol.triage, model=PRIMARY),
        phase1_verifier=dataclasses.replace(pol.phase1_verifier, model=PRIMARY),
        phase2_reviewer=dataclasses.replace(pol.phase2_reviewer, model=PRIMARY),
        phase2_verifier=dataclasses.replace(pol.phase2_verifier, model=PRIMARY),
        comparison=cfg_mod.ComparisonPolicy(model=SECOND, tiers=tiers, fraction=fraction),
    )
    return dataclasses.replace(cfg, policy=pol)


class ByModel:
    """FakeLanes, but a Phase-2 reviewer's output depends on its model."""

    def __init__(self, inner, per_model):
        self.inner, self.per_model = inner, per_model

    def __call__(self, spec, art, worktree):
        out = self.per_model.get(spec.model)
        if out is None or not _is_phase2_reviewer(spec):
            return self.inner(spec, art, worktree)
        # built here, not through FakeLanes' shared per-role dict: a role's primary and
        # comparison lanes run at the same time
        self.inner.calls.append(spec)
        text = json.dumps({**out, "review_phase": "final", "head_sha": HEAD})
        return LaneResult(0, json.dumps({"result": text}), "", 1, result_text=text)


class Hang:
    """Lanes on `model` run until they are told to stop (as `run_claude_lane` does)."""

    def __init__(self, inner, model, *, fail_primary_general=False):
        self.inner, self.model, self.fail = inner, model, fail_primary_general
        self.stopped = threading.Event()

    def __call__(self, spec, art, worktree):
        if (
            self.fail
            and spec.model == PRIMARY
            and spec.role == "general"
            and _is_phase2_reviewer(spec)
        ):
            time.sleep(0.05)  # let the comparison lanes start first
            return LaneResult(exit_code=1, stdout="", stderr="boom", duration_s=1)
        if spec.model == self.model and _is_phase2_reviewer(spec):
            deadline = time.monotonic() + 10
            while not spec.should_stop() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.stopped.set()
            return LaneResult(exit_code=None, stdout="", stderr="", duration_s=1, cancelled=True)
        return self.inner(spec, art, worktree)


def _run(cfg, conn, gh, lanes, **kw):
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.MENTION)
    (rid,) = schedule(conn, cfg, spawn=False)
    status = worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False, **kw)
    return rid, status


def _base(lanes, tier="critical", final=()):
    lanes.triage = {"tier": tier, "reasoning": "consensus change"}
    lanes.reviewer["default"] = {"summary": "s", "findings": []}
    lanes.verifier["preliminary"] = _verifier()
    lanes.verifier["final"] = _verifier(final)


def _events(conn, rid, kind):
    return [
        r["detail"]
        for r in conn.execute("SELECT detail FROM events WHERE run_id=? AND kind=?", (rid, kind))
    ]


def test_selected_run_doubles_phase2_lanes_and_verifier_sees_both(cfg, conn, gh, lanes):
    cfg = _with_comparison(cfg)
    both = _finding("Both")
    sol_only = _finding("Sol only", "blocking", 11)
    astra_only = _finding("Astra only", line=13)
    _base(lanes, final=[both, sol_only, astra_only])
    runner = ByModel(
        lanes,
        {
            PRIMARY: {"summary": "s", "findings": [both, sol_only]},
            SECOND: {"summary": "s", "findings": [both, astra_only]},
        },
    )
    rid, status = _run(cfg, conn, gh, runner)
    assert status == RunStatus.DONE

    p2 = [(s.role, s.model, s.effort) for s in lanes.calls if _is_phase2_reviewer(s)]
    assert sorted(p2) == sorted((r, m, "xhigh") for r in ROLES for m in (PRIMARY, SECOND))
    # comparison lanes never take a production pool slot
    twins = [s for s in lanes.calls if s.model == SECOND]
    assert {(s.pool, s.parallel) for s in twins} == {(worker.lanepool.COMPARE_POOL, False)}
    # the final verifier is the primary model and got both sets under neutral labels
    (final,) = [s for s in lanes.calls if s.role == "verifier" and "must be `final`" in s.prompt]
    assert final.model == PRIMARY
    assert '"general/a"' in final.prompt and '"general/b"' in final.prompt
    assert "#2" not in final.prompt

    stages = {
        (r["stage"], r["title"])
        for r in conn.execute(
            "SELECT stage, title FROM findings WHERE run_id=? AND phase='phase2'", (rid,)
        )
    }
    # stages carry the role, so the report can pair each role's two lanes
    assert ("compare:general", "Astra only") in stages
    assert ("lane:general", "Sol only") in stages
    assert not any(st.startswith("lane:") and t == "Astra only" for st, t in stages)

    body = gh.posted_reviews[-1]["body"]
    assert f"every Phase-2 reviewer also ran on `{SECOND}`" in body
    # inline footers credit the model that actually raised each finding
    footer = {
        c["body"].split("**")[1]: c["body"].split("source:")[1]
        for c in gh.posted_reviews[-1]["comments"]
    }
    astra = next(v for k, v in footer.items() if "Astra only" in k)
    assert f"`{SECOND}`" in astra and f"`{PRIMARY}`" not in astra
    assert _events(conn, rid, "compare.lanes") == ["phase=phase2 kept=3 dropped=0"]

    (c,) = compare.compare_runs(conn)
    assert (c.primary, c.second, c.tier, c.kept, c.unpaired) == (
        PRIMARY,
        SECOND,
        "critical",
        3,
        [],
    )
    assert c.roles == sorted(ROLES)
    assert {m: (st["kept"], st["only"], st["only_blockers"]) for m, st in c.models.items()} == {
        PRIMARY: (2, 1, 1),
        SECOND: (2, 1, 0),
    }
    text = compare.render([c])
    assert "1 comparison runs" in text and PRIMARY in text and SECOND in text
    assert json.loads(json.dumps(compare.summarize([c])))["models"][SECOND]["only"] == 1

    # a run that went degraded compared stand-ins, not the two models
    with tx(conn):
        conn.execute("UPDATE runs SET degraded=1 WHERE id=?", (rid,))
    assert compare.compare_runs(conn) == []


def test_unselected_tier_runs_primary_only(cfg, conn, gh, lanes):
    cfg = _with_comparison(cfg)
    _base(lanes, tier="normal")
    _, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert SECOND not in {s.model for s in lanes.calls}
    assert compare.compare_runs(conn) == []
    assert "Model comparison" not in gh.posted_reviews[-1]["body"]


def test_failed_comparison_lanes_are_dropped_after_one_attempt(cfg, conn, gh, lanes):
    cfg = _with_comparison(cfg)
    _base(lanes)
    lanes.dead_models = {SECOND}
    lanes.dead_stderr = "API Error: 500 upstream exploded"
    rid, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert sum(s.model == SECOND for s in lanes.calls) == 3  # one attempt each
    assert len(_events(conn, rid, "compare.lane_dropped")) == 3
    assert conn.execute("SELECT degraded FROM runs WHERE id=?", (rid,)).fetchone()[0] == 0
    body = gh.posted_reviews[-1]["body"]
    assert "Model comparison" not in body and SECOND not in body
    assert compare.compare_runs(conn) == []  # nothing was compared


def test_twin_dropped_at_parse_leaves_no_provenance(cfg, conn, gh, lanes):
    """A twin whose lane finished but whose output breaks the contract is out entirely: the
    review must not list it, credit it, or claim a comparison ran."""
    cfg = _with_comparison(cfg)
    _base(lanes, final=[_finding("Retitled by the verifier")])
    bad = {"summary": "s", "findings": [], "head_sha": "b" * 12}  # another commit

    class BadTwins(ByModel):
        def __call__(self, spec, art, worktree):
            # and in every correction turn (`resume`): the twin never fixes it
            if spec.model == SECOND and (spec.resume or _is_phase2_reviewer(spec)):
                self.inner.calls.append(spec)
                text = json.dumps({**bad, "review_phase": "final"})
                return LaneResult(0, json.dumps({"result": text}), "", 1, result_text=text)
            return super().__call__(spec, art, worktree)

    rid, status = _run(cfg, conn, gh, BadTwins(lanes, {}))
    assert status == RunStatus.DONE
    assert len(_events(conn, rid, "compare.lane_dropped")) == 3
    # each twin was asked to correct it in its own session first, twice, in the compare pool
    assert len(_events(conn, rid, "lane.correction_failed")) == 3
    resumed = [s for s in lanes.calls if s.resume]
    assert len(resumed) == 6 and {s.pool for s in resumed} == {"compare"}
    review = gh.posted_reviews[-1]
    assert SECOND not in review["body"] and "Model comparison" not in review["body"]
    assert all(SECOND not in c["body"] for c in review["comments"])
    assert compare.compare_runs(conn) == []


def test_quota_failure_on_comparison_lane_never_flips_degraded(cfg, conn, gh, lanes):
    cfg = _with_comparison(cfg)
    stand_in = cfg_mod.Substitute(model="muse-spark-1.3-contributor", effort_cap="xhigh")
    pol = dataclasses.replace(
        cfg.policy,
        degraded=cfg_mod.DegradedPolicy(
            sentinel=PRIMARY, substitutes={PRIMARY: stand_in, SECOND: stand_in}
        ),
    )
    cfg = dataclasses.replace(cfg, policy=pol)
    _base(lanes)
    lanes.dead_models = {SECOND}
    lanes.dead_stderr = "API Error: 429 All credentials for model gpt-6-astra are cooling down"
    rid, status = _run(cfg, conn, gh, lanes, prober=lambda model: (False, ""))
    assert status == RunStatus.DONE
    assert conn.execute("SELECT degraded FROM runs WHERE id=?", (rid,)).fetchone()[0] == 0
    assert "muse-spark-1.3-contributor" not in {s.model for s in lanes.calls}


def test_slow_twin_is_stopped_after_the_grace_period(cfg, conn, gh, lanes, monkeypatch):
    monkeypatch.setattr(worker, "COMPARE_GRACE_MINUTES", 0.001)
    cfg = _with_comparison(cfg)
    _base(lanes)
    runner = Hang(lanes, SECOND)
    rid, status = _run(cfg, conn, gh, runner)
    assert status == RunStatus.DONE and runner.stopped.is_set()
    dropped = _events(conn, rid, "compare.lane_dropped")
    # every twin is dropped: out of grace, or never started (only 2 compare slots)
    assert len(dropped) == 3
    assert all("still running" in d or "never started" in d for d in dropped)
    assert sum("still running" in d for d in dropped) <= 2
    assert "Model comparison" not in gh.posted_reviews[-1]["body"]


def test_primary_failure_stops_running_twins(cfg, conn, gh, lanes):
    cfg = _with_comparison(cfg)
    _base(lanes)
    runner = Hang(lanes, SECOND, fail_primary_general=True)
    rid, status = _run(cfg, conn, gh, runner)
    assert status == RunStatus.FAILED and runner.stopped.is_set()
    reason = conn.execute("SELECT reason FROM runs WHERE id=?", (rid,)).fetchone()[0]
    assert "phase2/general" in reason
    # stopped because the phase failed, not dropped
    assert _events(conn, rid, "compare.lane_dropped") == []


def test_fresh_final_pass_runs_no_twins(cfg, conn, gh, lanes):
    cfg = _with_comparison(cfg)
    _base(lanes)
    _run(cfg, conn, gh, lanes)
    first = len(lanes.calls)
    with tx(conn):
        assert enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.MANUAL) == "requeued"
    (rid,) = schedule(conn, cfg, spawn=False)
    assert worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False) == RunStatus.DONE
    fresh = [s for s in lanes.calls[first:] if "fresh final review" in s.prompt]
    assert fresh and {s.model for s in fresh if _is_phase2_reviewer(s)} == {PRIMARY}
    second_round = [s for s in lanes.calls[first:] if _is_phase2_reviewer(s)]
    assert sum(s.model == SECOND for s in second_round) == 3  # the first Phase 2 only


def test_blind_labels_hide_which_set_is_the_comparison():
    outputs = {"general": 1, "general#2": 2, "fresh:general": 3}
    seen = set()
    for seed in range(40):
        blind = worker._blind(outputs, seed=seed)
        assert set(blind) == {"general/a", "general/b", "fresh:general"}
        assert worker._blind(outputs, seed=seed) == blind  # stable per run
        seen.add(blind["general/a"])
    assert seen == {1, 2}  # either set can be `a`
    assert worker._blind({"general": 1}, seed=0) == {"general": 1}


def test_selection_is_stable_and_roughly_the_fraction():
    pol = cfg_mod.ComparisonPolicy(model=SECOND, tiers=("critical",), fraction=0.25)

    def pick(n):
        return pol.selects("dashpay/platform", n, f"{n:040x}", "critical")

    picks = [pick(n) for n in range(4000)]
    assert 0.22 < sum(picks) / len(picks) < 0.28
    assert picks == [pick(n) for n in range(4000)]
    assert not pol.selects("dashpay/platform", 1, HEAD, "normal")
    none = cfg_mod.ComparisonPolicy(model=SECOND, tiers=("critical",), fraction=0)
    assert not none.selects("r", 1, HEAD, "critical")


def test_policy_block_is_parsed_and_validated(skills_dir):
    raw = json.loads((skills_dir / "config.json").read_text())

    def load(comparison):
        raw["review_model_policy"]["comparison"] = comparison
        (skills_dir / "config.json").write_text(json.dumps(raw))
        return cfg_mod.load_skills_config(skills_dir)[2].comparison

    assert load({"model": "gpt-5.6-sol", "tiers": ["critical"], "fraction": 0.25}) == (
        cfg_mod.ComparisonPolicy(model="gpt-5.6-sol", tiers=("critical",), fraction=0.25)
    )
    with pytest.raises(ValueError, match="not configured tiers"):
        load({"model": "gpt-5.6-sol", "tiers": ["urgent"], "fraction": 0.25})
    with pytest.raises(ValueError, match="fraction"):
        load({"model": "gpt-5.6-sol", "tiers": [], "fraction": 1.5})
    with pytest.raises(ValueError, match="Phase-2 model"):
        load({"model": "gpt-6-astra", "tiers": ["critical"], "fraction": 0.25})


def test_partial_comparison_counts_paired_roles_only(cfg, conn, gh, lanes):
    """One role's comparison lane is dropped: that role's primary findings must not count as
    findings only the primary model found."""
    cfg = _with_comparison(cfg)
    general_only = _finding("Primary general only", line=11)
    shared = _finding("Shared", line=12)
    _base(lanes, final=[general_only, shared])

    class OneTwinDies(ByModel):
        def __call__(self, spec, art, worktree):
            if spec.model == SECOND and spec.role == "general" and _is_phase2_reviewer(spec):
                return LaneResult(exit_code=1, stdout="", stderr="boom", duration_s=1)
            if _is_phase2_reviewer(spec) and spec.model == PRIMARY and spec.role == "general":
                return ByModel(self.inner, {PRIMARY: {"summary": "s", "findings": [general_only]}})(
                    spec, art, worktree
                )
            return super().__call__(spec, art, worktree)

    both = {"summary": "s", "findings": [shared]}
    _run(cfg, conn, gh, OneTwinDies(lanes, {PRIMARY: both, SECOND: both}))
    (c,) = compare.compare_runs(conn)
    assert c.roles == ["always-on", "security-auditor"] and c.unpaired == ["general"]
    assert c.kept == 1  # `Primary general only` is not comparable
    assert {m: (st["kept"], st["only"]) for m, st in c.models.items()} == {
        PRIMARY: (1, 0),
        SECOND: (1, 0),
    }
    assert "1 left out" in compare.render([c])


def test_fresh_pass_findings_are_kept_apart(cfg, conn, gh, lanes):
    """A re-review's fresh final pass re-runs the primary alone: its lanes and its verifier
    write their own stages, so the report still compares the first Phase 2 only."""
    cfg = _with_comparison(cfg)
    seen = _finding("Seen")
    _base(lanes)  # the first review posts nothing, so the re-review has nothing to reconcile
    lanes.reviewer["default"] = {"summary": "s", "findings": [seen]}

    class FreshGateKeeps:
        def __call__(self, spec, art, worktree):
            if spec.role == "verifier" and "This is the fresh final gate" in spec.prompt:
                text = json.dumps({**_verifier([seen]), "review_phase": "final"})
                return LaneResult(0, json.dumps({"result": text}), "", 1, result_text=text)
            return lanes(spec, art, worktree)

    runner = FreshGateKeeps()
    _run(cfg, conn, gh, runner)
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.MANUAL)
    (rid,) = schedule(conn, cfg, spawn=False)
    assert worker.main(cfg, conn, rid, gh=gh, lane_runner=runner, heartbeat=False) == RunStatus.DONE
    stages = {
        r[0] for r in conn.execute("SELECT DISTINCT stage FROM findings WHERE run_id=?", (rid,))
    }
    assert {"lane:general", "compare:general", "fresh:general", "verified-fresh"} <= stages
    # the first verifier kept nothing; the fresh gate's finding must not be credited
    runs = {c.run_id: c for c in compare.compare_runs(conn)}
    assert runs[rid].kept == 0
    assert runs[rid].models[PRIMARY]["raised"] == runs[rid].models[SECOND]["raised"] == 1


def test_compare_slot_needs_idle_production_headroom(tmp_path):
    from reviewsys import lanepool

    stop = threading.Event()
    admitted = []

    def admit():
        admitted.append(True)
        if len(admitted) >= 3:
            stop.set()
        return False  # production is saturated

    assert lanepool.acquire(tmp_path, "compare", lambda: 2, stop.is_set, admit=admit) is None
    assert len(admitted) >= 3
    fd = lanepool.acquire(tmp_path, "compare", lambda: 2, lambda: False, admit=lambda: True)
    assert fd is not None
    lanepool.release(fd)


def test_twin_that_never_got_a_slot_says_so(cfg, conn, gh, lanes, monkeypatch):
    monkeypatch.setattr(worker, "COMPARE_GRACE_MINUTES", 0.001)
    cfg = _with_comparison(cfg)
    _base(lanes)

    class NoSlot:
        def __call__(self, spec, art, worktree):
            if spec.model == SECOND and _is_phase2_reviewer(spec):
                while not spec.should_stop():
                    time.sleep(0.01)
                return LaneResult(None, "", "", 0, cancelled=True, started=False)
            return lanes(spec, art, worktree)

    rid, status = _run(cfg, conn, gh, NoSlot())
    assert status == RunStatus.DONE
    dropped = _events(conn, rid, "compare.lane_dropped")
    assert len(dropped) == 3 and all("never started" in d for d in dropped)


def test_unpaired_primary_role_is_credited_on_a_shared_finding(cfg, conn, gh, lanes):
    """X is raised by the paired twin `general#2` and by the primary `security-auditor` lane,
    whose twin was dropped: both models found X, not the second one alone."""
    cfg = _with_comparison(cfg)
    x = _finding("X")
    _base(lanes, final=[x])

    class Runner(ByModel):
        def __call__(self, spec, art, worktree):
            if _is_phase2_reviewer(spec):
                if spec.model == SECOND and spec.role == "security-auditor":
                    return LaneResult(exit_code=1, stdout="", stderr="boom", duration_s=1)
                raised = (spec.model == SECOND and spec.role == "general") or (
                    spec.model == PRIMARY and spec.role == "security-auditor"
                )
                out = {"summary": "s", "findings": [x] if raised else []}
                return ByModel(self.inner, {spec.model: out})(spec, art, worktree)
            return super().__call__(spec, art, worktree)

    _run(cfg, conn, gh, Runner(lanes, {}))
    (c,) = compare.compare_runs(conn)
    assert c.unpaired == ["security-auditor"] and c.kept == 1
    assert {m: (st["kept"], st["only"]) for m, st in c.models.items()} == {
        PRIMARY: (1, 0),
        SECOND: (1, 0),
    }


def test_twins_wait_while_production_reviewers_are_in_line(cfg, conn, tmp_path):
    from reviewsys import lanepool

    slot_dir = cfg.work_dir / "lane-slots"
    (slot_dir / "gpt.line").mkdir(parents=True)
    # any production lane in the gpt line holds comparison lanes back, a verifier included
    ticket = slot_dir / "gpt.line" / f"{lanepool.RANK_LIVE_SIDE}-{1:012d}-{0:020d}-{os.getpid()}-1"
    ticket.touch()
    ran = []

    def runner(spec, art, worktree):
        ran.append(spec.role)
        return LaneResult(0, "{}", "", 0)

    gated = lanepool.gated(runner, lambda: conn, cfg, lambda: False, "live", 0)
    stop = threading.Event()
    threading.Timer(0.1, stop.set).start()
    spec = LaneSpec(
        role="general",
        agent="a",
        model=SECOND,
        effort="high",
        prompt="",
        cwd=tmp_path,
        add_dir=tmp_path,
        timeout_seconds=1,
        claude_bin="x",
        should_stop=stop.is_set,
        pool=lanepool.COMPARE_POOL,
    )
    res = gated(spec, tmp_path / "art", tmp_path)
    assert res.cancelled and not res.started and ran == []
    ticket.unlink()
    spec = dataclasses.replace(spec, should_stop=lambda: False)
    assert gated(spec, tmp_path / "art", tmp_path).exit_code == 0 and ran == ["general"]
