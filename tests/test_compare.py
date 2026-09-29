"""Model comparison: on sampled runs every Phase-2 reviewer also runs on a second model, the
verifier weighs both sets, and `reviewsys compare` credits kept findings per model."""

from __future__ import annotations

import dataclasses
import json

import pytest

from reviewsys import compare, worker
from reviewsys import config as cfg_mod
from reviewsys.db import tx
from reviewsys.ingest import enqueue_head
from reviewsys.models import RunStatus, Trigger
from reviewsys.scheduler import schedule
from reviewsys.steps import worktree as wt

HEAD = "a" * 40
PRIMARY, SECOND = "gpt-6.1-sol", "gpt-6-astra"


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


def _verifier(findings):
    return {
        "summary": "ok",
        "review_action": "COMMENT",
        "findings": findings,
        "dropped_findings": [],
        "out_of_scope_findings": [],
        "coderabbit_reactions": [],
        "prerequisite_adjudications": [],
        "adjudication_complete": True,
    }


def _with_comparison(cfg, *, fraction=1.0, tiers=("critical",)):
    pol = dataclasses.replace(
        cfg.policy,
        phase2_reviewer=dataclasses.replace(cfg.policy.phase2_reviewer, model=PRIMARY),
        phase2_verifier=dataclasses.replace(cfg.policy.phase2_verifier, model=PRIMARY),
        # as in production: nothing but the comparison lanes runs on the second model
        phase1_verifier=dataclasses.replace(cfg.policy.phase1_verifier, model=PRIMARY),
        triage=dataclasses.replace(cfg.policy.triage, model=PRIMARY),
        comparison=cfg_mod.ComparisonPolicy(model=SECOND, tiers=tiers, fraction=fraction),
    )
    return dataclasses.replace(cfg, policy=pol)


class ByModel:
    """FakeLanes, but a Phase-2 reviewer's findings depend on its model."""

    def __init__(self, inner, per_model):
        self.inner, self.per_model = inner, per_model

    def __call__(self, spec, art, worktree):
        findings = self.per_model.get(spec.model)
        is_phase2_reviewer = (
            spec.role
            not in {
                "selector",
                "triage",
                "verifier",
                "repair",
            }
            and "set to `final`" in spec.prompt
        )
        if findings is not None and is_phase2_reviewer:
            self.inner.reviewer[spec.role] = {"summary": "s", "findings": findings}
        else:
            self.inner.reviewer.pop(spec.role, None)
        return self.inner(spec, art, worktree)


def _run(cfg, conn, gh, lanes):
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.MENTION)
    (rid,) = schedule(conn, cfg, spawn=False)
    status = worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False)
    return rid, status


def _base(lanes, tier="critical"):
    lanes.triage = {"tier": tier, "reasoning": "consensus change"}
    lanes.reviewer["default"] = {"summary": "s", "findings": []}
    lanes.verifier["preliminary"] = _verifier([])


def test_selected_run_doubles_phase2_lanes_and_verifier_sees_both(cfg, conn, gh, lanes):
    cfg = _with_comparison(cfg)
    _base(lanes)
    both, sol_only, astra_only = (
        _finding("Both"),
        _finding("Sol only", "blocking", 11),
        _finding("Astra only", line=13),
    )
    runner = ByModel(lanes, {PRIMARY: [both, sol_only], SECOND: [both, astra_only]})
    lanes.verifier["final"] = _verifier([both, sol_only, astra_only])
    rid, status = _run(cfg, conn, gh, runner)
    assert status == RunStatus.DONE

    p2 = [
        (s.role, s.model, s.effort)
        for s in lanes.calls
        if "set to `final`" in s.prompt and s.role != "verifier"
    ]
    roles = {"general", "always-on", "security-auditor"}
    assert sorted(p2) == sorted([(r, m, "xhigh") for r in roles for m in (PRIMARY, SECOND)])
    # the final verifier is the primary model and got every lane's findings, keyed apart
    final = [s for s in lanes.calls if s.role == "verifier" and "must be `final`" in s.prompt]
    assert [s.model for s in final] == [PRIMARY]
    assert '"general#2"' in final[0].prompt and '"general"' in final[0].prompt

    # comparison lanes' own findings are recorded under their own stage
    stages = {
        (r["stage"], r["title"])
        for r in conn.execute(
            "SELECT stage, title FROM findings WHERE run_id=? AND phase='phase2'", (rid,)
        )
    }
    assert ("compare", "Astra only") in stages and ("lane", "Sol only") in stages
    assert ("lane", "Astra only") not in stages

    body = gh.posted_reviews[-1]["body"]
    assert f"every Phase-2 reviewer also ran on `{SECOND}`" in body
    # inline footers credit the model that actually raised each finding
    comments = {c["body"].split("**")[1]: c["body"] for c in gh.posted_reviews[-1]["comments"]}
    astra = next(v for k, v in comments.items() if "Astra only" in k)
    assert f"`{SECOND}`" in astra and f"`{PRIMARY}`" not in astra.split("source:")[1]

    runs = compare.compare_runs(conn)
    assert len(runs) == 1
    c = runs[0]
    assert (c.primary, c.second, c.tier, c.kept) == (PRIMARY, SECOND, "critical", 3)
    assert c.kept_by == {PRIMARY: 2, SECOND: 2}
    assert c.only == {PRIMARY: 1, SECOND: 1}
    assert c.blockers_only == {PRIMARY: 1, SECOND: 0}
    text = compare.render(runs)
    assert "1 comparison runs" in text and PRIMARY in text and SECOND in text
    assert json.loads(json.dumps(compare.summarize(runs)))["models"][SECOND]["only"] == 1


def test_unselected_tier_runs_primary_only(cfg, conn, gh, lanes):
    cfg = _with_comparison(cfg)
    _base(lanes, tier="normal")
    lanes.verifier["final"] = _verifier([])
    _, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert SECOND not in {s.model for s in lanes.calls if s.role != "triage"}
    assert compare.compare_runs(conn) == []
    assert "Model comparison" not in gh.posted_reviews[-1]["body"]


def test_failed_comparison_lane_is_dropped_not_fatal(cfg, conn, gh, lanes):
    cfg = _with_comparison(cfg)
    _base(lanes)
    lanes.verifier["final"] = _verifier([])
    lanes.dead_models = {SECOND}
    lanes.dead_stderr = "API Error: 500 upstream exploded"
    rid, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    dropped = conn.execute(
        "SELECT COUNT(*) FROM events WHERE run_id=? AND kind='compare.lane_dropped'", (rid,)
    ).fetchone()[0]
    assert dropped == 3
    assert conn.execute("SELECT degraded FROM runs WHERE id=?", (rid,)).fetchone()[0] == 0
    assert compare.compare_runs(conn)[0].dropped_lanes == 3
    assert "Model comparison" not in gh.posted_reviews[-1]["body"]


def test_quota_failure_on_comparison_lane_never_flips_degraded(cfg, conn, gh, lanes):
    cfg = _with_comparison(cfg)
    subs = {PRIMARY: cfg_mod.Substitute(model="muse-spark-1.3-contributor", effort_cap="xhigh")}
    subs[SECOND] = subs[PRIMARY]
    pol = dataclasses.replace(
        cfg.policy, degraded=cfg_mod.DegradedPolicy(sentinel=PRIMARY, substitutes=subs)
    )
    cfg = dataclasses.replace(cfg, policy=pol)
    _base(lanes)
    lanes.verifier["final"] = _verifier([])
    lanes.dead_models = {SECOND}
    lanes.dead_stderr = "API Error: 429 All credentials for model gpt-6-astra are cooling down"
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.MENTION)
    (rid,) = schedule(conn, cfg, spawn=False)
    status = worker.main(
        cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False, prober=lambda model: (False, "")
    )
    assert status == RunStatus.DONE
    assert conn.execute("SELECT degraded FROM runs WHERE id=?", (rid,)).fetchone()[0] == 0
    assert "muse-spark-1.3-contributor" not in {s.model for s in lanes.calls}


def test_selection_is_stable_and_roughly_the_fraction():
    pol = cfg_mod.ComparisonPolicy(model=SECOND, tiers=("critical",), fraction=0.25)
    picks = [pol.selects("dashpay/platform", n, f"{n:040x}", "critical") for n in range(4000)]
    assert 0.22 < sum(picks) / len(picks) < 0.28
    assert picks == [
        pol.selects("dashpay/platform", n, f"{n:040x}", "critical") for n in range(4000)
    ]
    assert not pol.selects("dashpay/platform", 1, HEAD, "normal")
    assert not cfg_mod.ComparisonPolicy(model=SECOND, tiers=("critical",), fraction=0).selects(
        "r", 1, HEAD, "critical"
    )


def test_policy_block_is_parsed_and_validated(skills_dir, cfg):
    raw = json.loads((skills_dir / "config.json").read_text())
    raw["review_model_policy"]["comparison"] = {
        "model": SECOND,
        "tiers": ["critical"],
        "fraction": 0.25,
    }
    (skills_dir / "config.json").write_text(json.dumps(raw))
    _, _, pol, _ = cfg_mod.load_skills_config(skills_dir)
    assert pol.comparison == cfg_mod.ComparisonPolicy(
        model=SECOND, tiers=("critical",), fraction=0.25
    )
    raw["review_model_policy"]["comparison"]["tiers"] = ["urgent"]
    (skills_dir / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="not configured tiers"):
        cfg_mod.load_skills_config(skills_dir)
    raw["review_model_policy"]["comparison"] = {"model": SECOND, "tiers": [], "fraction": 1.5}
    (skills_dir / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="fraction"):
        cfg_mod.load_skills_config(skills_dir)
