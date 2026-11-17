"""A triage tier may run its Phase-2 reviewers on a model of its own (`phase2_model`)."""

from __future__ import annotations

import dataclasses

import pytest

from reviewsys.models import RunStatus
from tests.test_parallel_lanes import _run
from tests.test_phase1_policy import (  # noqa: F401  (fake_git is an autouse fixture)
    _findings,
    _policy,
    _reviewers,
    _single_stage,
    _verifier,
    fake_git,
)

TIER_MODEL = "gpt-6.2-astra"


def _critical_on_tier_model(raw):
    raw["review_model_policy"]["triage"]["tiers"]["critical"].update(
        phase2="high", phase2_model=TIER_MODEL
    )


def test_tier_model_parses_and_other_tiers_keep_the_policy_model(cfg, skills_dir, tmp_path):
    assert all(t.phase2_model is None for t in cfg.policy.tiers.values())
    pol = _policy(skills_dir, tmp_path, _critical_on_tier_model).policy
    critical = pol.phase2_lane(pol.tier_effort("critical"))
    assert (critical.model, critical.effort, critical.agent) == (
        TIER_MODEL,
        "high",
        "phase2-reviewer",
    )
    normal = pol.phase2_lane(pol.tier_effort("normal"))
    assert (normal.model, normal.effort) == ("gpt-6-astra", "high")
    assert pol.phase2_verifier.model == "gpt-6-astra"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda p: p["triage"]["tiers"]["critical"].update(phase2_model=" "),
            "phase2_model must be a model name",
        ),
        (
            lambda p: p["triage"]["tiers"]["trivial"].update(phase2_model=TIER_MODEL),
            "phase2_model needs a phase2 effort",
        ),
        (
            lambda p: (
                p["triage"]["tiers"]["critical"].update(phase2_model=TIER_MODEL),
                p.update(
                    degraded={"sentinel": "gpt-6-astra", "substitutes": {"gpt-6-astra": "muse-x"}}
                ),
            ),
            "has no degraded substitute",
        ),
        (
            lambda p: (
                p["triage"]["tiers"]["critical"].update(phase2_model=TIER_MODEL),
                p.update(comparison={"model": TIER_MODEL, "tiers": ["critical"], "fraction": 0.5}),
            ),
            "already run Phase 2 on",
        ),
    ],
)
def test_bad_tier_models_are_rejected(skills_dir, tmp_path, cfg, mutate, message):
    with pytest.raises(ValueError, match=message):
        _policy(skills_dir, tmp_path, lambda raw: mutate(raw["review_model_policy"]))


def test_degraded_mode_accepts_a_substituted_tier_model(cfg, skills_dir, tmp_path):
    def mutate(raw):
        _critical_on_tier_model(raw)
        raw["review_model_policy"]["degraded"] = {
            "sentinel": "gpt-6-astra",
            "substitutes": {"gpt-6-astra": "muse-x", TIER_MODEL: "muse-x"},
        }

    assert _policy(skills_dir, tmp_path, mutate).policy.tiers["critical"].phase2_model == TIER_MODEL


def test_critical_phase2_runs_on_the_tier_model_and_the_verifier_keeps_its_own(
    cfg, conn, gh, lanes, skills_dir, tmp_path
):
    def mutate(raw):
        _single_stage(raw)
        _critical_on_tier_model(raw)

    c = dataclasses.replace(_policy(skills_dir, tmp_path, mutate), max_concurrent=4)
    lanes.triage = {"tier": "critical", "reasoning": "consensus"}
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["final"] = _verifier(_findings(suggestion=1))
    _, status = _run(c, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert {(s.model, s.effort) for s in _reviewers(lanes)} == {
        ("glm-5.3-flash", "max"),
        (TIER_MODEL, "high"),
    }
    assert {s.model for s in lanes.calls if s.role == "verifier"} == {"gpt-6-astra"}
    assert f"`{TIER_MODEL}`" in gh.posted_reviews[0]["body"]


def test_normal_tier_still_runs_the_policy_model(cfg, conn, gh, lanes, skills_dir, tmp_path):
    c = _policy(skills_dir, tmp_path, _critical_on_tier_model)
    lanes.triage = {"tier": "normal", "reasoning": "ordinary"}
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    _, status = _run(c, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert TIER_MODEL not in {s.model for s in lanes.calls}
