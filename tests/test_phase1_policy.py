"""Phase-1 policy (v0.21): the Phase-1 specialist allowlist, the points gate, repos without
Phase 1, single-stage tiers, unbounded per-run parallelism and keep-forever run artifacts."""

from __future__ import annotations

import dataclasses
import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from reviewsys import config as cfg_mod
from reviewsys import gc
from reviewsys.contract import parse_verifier_output
from reviewsys.gate import gate_score, phase1_blocks
from reviewsys.lane import LaneResult, LaneSpec
from reviewsys.models import RunStatus
from reviewsys.steps import worktree as wt
from tests.test_parallel_lanes import HEAD, REVIEWER_ROLES, Overlap, _run


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


def _finding(severity: str, title: str, line: int = 12) -> dict:
    return {
        "file": "f.rs",
        "line_start": line,
        "line_end": line,
        "severity": severity,
        "confidence": 0.9,
        "category": "logic",
        "title": title,
        "body": "Explained.",
    }


def _findings(**counts: int) -> list[dict]:
    """`_findings(blocking=1, suggestion=2)`: distinct findings of each severity."""
    return [_finding(sev, f"{sev} {i}", line=10 + i) for sev, n in counts.items() for i in range(n)]


def _verifier(findings, action="COMMENT"):
    return {
        "summary": "ok",
        "review_action": action,
        "findings": findings,
        "dropped_findings": [],
        "out_of_scope_findings": [],
        "coderabbit_reactions": [],
        "prerequisite_adjudications": [],
        "adjudication_complete": True,
    }


def _policy(skills_dir: Path, tmp_path: Path, mutate) -> cfg_mod.Config:
    raw = json.loads((skills_dir / "config.json").read_text())
    mutate(raw)
    (skills_dir / "config.json").write_text(json.dumps(raw))
    return cfg_mod.load(tmp_path / "config.toml")  # written by the cfg fixture


def _gate(raw, block_above=5):
    raw["review_model_policy"]["phase1"]["gate"] = {"block_above": block_above}


def _steps(conn, rid) -> dict[str, str]:
    return {
        r["name"]: r["status"]
        for r in conn.execute("SELECT name, status FROM steps WHERE run_id=?", (rid,))
    }


def _reviewers(lanes, model=None):
    return [
        s for s in lanes.calls if s.role in REVIEWER_ROLES and (model is None or s.model == model)
    ]


# ---- config ----


def test_policy_knobs_parse_and_default_off(cfg, skills_dir, tmp_path):
    assert cfg.policy.phase1_specialists is None and cfg.policy.phase1_gate is None
    assert not any(t.single_stage for t in cfg.policy.tiers.values())
    assert all(r.phase1 for r in cfg.repos)
    assert cfg.phase_parallelism == 0 and cfg.artifact_retention_days == 0

    def mutate(raw):
        p = raw["review_model_policy"]
        p["phase1"]["specialists"] = ["security-auditor"]
        p["phase1"]["gate"] = {"block_above": 5, "weights": {"suggestion": 2}}
        p["triage"]["tiers"]["critical"]["single_stage"] = True
        raw["repos"][0]["phase1"] = False

    c = _policy(skills_dir, tmp_path, mutate)
    assert c.policy.phase1_specialists == ("security-auditor",)
    gate = c.policy.phase1_gate
    assert gate is not None and gate.block_above == 5
    assert gate.weights == {"blocking": 3, "suggestion": 2, "nitpick": 0}
    assert gate.points(["blocking", "suggestion", "nitpick"]) == 5
    assert c.policy.tiers["critical"].single_stage and not c.policy.tiers["normal"].single_stage
    assert not c.repo("dashpay/platform").phase1


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["phase1"].update(specialists=["nope"]), "not configured specialists"),
        (lambda p: p["phase1"].update(gate={"block_above": -1}), "must be >= 0"),
        (
            lambda p: p["phase1"].update(gate={"block_above": 5, "weights": {"praise": 1}}),
            "unknown severity",
        ),
        (
            lambda p: p["triage"]["tiers"]["trivial"].update(single_stage=True),
            "single_stage needs a phase2 effort",
        ),
        (lambda p: p["phase1"].update(gate={}), "needs `block_above`"),
        (lambda p: p["phase1"].update(specialists="rust-quality"), "must be a list"),
        (
            lambda p: p["triage"]["tiers"]["critical"].update(single_stage="false"),
            "must be true or false",
        ),
    ],
)
def test_bad_policy_knobs_are_rejected(skills_dir, tmp_path, cfg, mutate, message):
    with pytest.raises(ValueError, match=message):
        _policy(skills_dir, tmp_path, lambda raw: mutate(raw["review_model_policy"]))


# ---- Phase-1 roster ----


def test_phase1_runs_only_its_allowlisted_specialists(cfg, conn, gh, lanes, skills_dir, tmp_path):
    cfg = _policy(
        skills_dir,
        tmp_path,
        lambda raw: raw["review_model_policy"]["phase1"].update(specialists=["security-auditor"]),
    )
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    _, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    # always-on is an always-run specialist, but Phase 1 runs only the allowlisted ones
    assert {s.role for s in _reviewers(lanes, "glm-5.3-flash")} == {"general", "security-auditor"}
    assert {s.role for s in _reviewers(lanes, "gpt-6-astra")} == REVIEWER_ROLES
    body = gh.posted_reviews[0]["body"]
    assert "Phase 1 + Phase 2" in body


def test_phase1_only_review_keeps_every_specialist(cfg, conn, gh, lanes, skills_dir, tmp_path):
    """No Phase 2 follows a trivial tier, so nothing else would cover the specialists."""
    cfg = _policy(
        skills_dir,
        tmp_path,
        lambda raw: raw["review_model_policy"]["phase1"].update(specialists=[]),
    )
    lanes.triage = {"tier": "trivial", "reasoning": "typo"}
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    _, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert {s.role for s in _reviewers(lanes)} == REVIEWER_ROLES


# ---- points gate ----


@pytest.fixture
def gated(cfg, skills_dir, tmp_path):
    return _policy(skills_dir, tmp_path, _gate)


def test_one_blocker_under_the_budget_does_not_hold_phase2(gated, conn, gh, lanes):
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": _findings(blocking=1),
        "out_of_scope_findings": [],
    }
    lanes.verifier["preliminary"] = _verifier(_findings(blocking=1, suggestion=2))  # 5 points
    lanes.verifier["final"] = _verifier(_findings(blocking=1))
    rid, status = _run(gated, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert _steps(conn, rid)["phase2"] == "ok"
    gate = json.loads(
        conn.execute("SELECT detail FROM steps WHERE run_id=? AND name='gate'", (rid,)).fetchone()[
            "detail"
        ]
    )
    assert gate["admit_phase2"] and gate["points"] == 5 and gate["block_above"] == 5
    assert gate["blockers"] == 1
    (review,) = gh.posted_reviews
    assert "phase=final" in review["body"] and review["event"] == "REQUEST_CHANGES"


def test_findings_over_the_budget_hold_phase2(gated, conn, gh, lanes):
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": _findings(blocking=2),
        "out_of_scope_findings": [],
    }
    lanes.verifier["preliminary"] = _verifier(_findings(blocking=2))  # 6 points
    rid, status = _run(gated, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert "phase2" not in _steps(conn, rid)
    (review,) = gh.posted_reviews
    assert "phase=preliminary" in review["body"] and review["event"] == "REQUEST_CHANGES"
    assert "gate points: 6 (Phase 2 deferred above 5)" in gh.gate_bodies[-1]
    assert "Blockers found" in gh.gate_bodies[-1]


def test_carried_forward_suggestions_never_count_but_carried_blockers_do():
    gate = cfg_mod.GatePolicy(block_above=5)

    def verified(findings):
        return parse_verifier_output(
            {**_verifier(findings), "review_phase": "preliminary"},
            expected_phase="preliminary",
            expected_coderabbit_ids=[],
        )

    carried = [{**f, "finding_hash": f"{i:012x}"} for i, f in enumerate(_findings(suggestion=6))]
    v = verified(carried)
    assert not phase1_blocks(v, gate) and gate_score(v, gate)["points"] == 0
    assert phase1_blocks(verified(carried + _findings(suggestion=6)), gate)
    blockers = [{**f, "finding_hash": f"b{i:011x}"} for i, f in enumerate(_findings(blocking=2))]
    assert phase1_blocks(verified(blockers), gate), "an unfixed blocker still counts"


def test_suggestions_alone_over_the_budget_defer_but_never_approve(gated, conn, gh, lanes):
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": _findings(suggestion=6),
        "out_of_scope_findings": [],
    }
    lanes.verifier["preliminary"] = _verifier(_findings(suggestion=6), action="APPROVE")
    rid, status = _run(gated, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert "phase2" not in _steps(conn, rid)
    (review,) = gh.posted_reviews
    assert "phase=preliminary" in review["body"]
    assert review["event"] == "COMMENT", "Phase 2 has not seen this head: never APPROVE"
    assert "are above the gate budget" in review["body"]
    assert "Validated blockers were found" not in review["body"]
    assert "- Phase 2 reviewers: **not run (deferred by the Phase-1 gate)**" in review["body"]
    assert "Phase-1 findings over the gate" in gh.gate_bodies[-1]
    assert "gate points: 6 (Phase 2 deferred above 5)" in gh.gate_bodies[-1]


def test_suggestions_never_hold_back_a_head_whose_final_review_stands(gated, conn, gh, lanes):
    """A same-sha re-review: Phase 2 already reviewed this exact head, so suggestions from the
    cheap Phase-1 model neither defer it nor retract the standing verdict."""
    gh.posted_reviews.append(
        {"id": 1, "body": f"<!-- thepastaclaw-review-phase v1 phase=final sha={HEAD} policy=x -->"}
    )
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["preliminary"] = _verifier(_findings(suggestion=6))
    lanes.verifier["final"] = _verifier([])
    rid, status = _run(gated, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert _steps(conn, rid)["phase2"] == "ok"


def test_trivial_tier_over_the_budget_still_publishes_phase1_only(gated, conn, gh, lanes):
    """No Phase 2 to defer: the gate does not apply, so it is the usual Phase-1-only final."""
    lanes.triage = {"tier": "trivial", "reasoning": "typo"}
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["preliminary"] = _verifier(_findings(suggestion=6))
    _, status = _run(gated, conn, gh, lanes)
    assert status == RunStatus.DONE
    (review,) = gh.posted_reviews
    assert "## Final review — Phase 1 only (trivial change)" in review["body"]
    assert "gate points" not in gh.gate_bodies[-1]


def test_nitpicks_never_count_toward_the_gate(gated, conn, gh, lanes):
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["preliminary"] = _verifier(_findings(suggestion=5, nitpick=4))
    lanes.verifier["final"] = _verifier([])
    rid, status = _run(gated, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert _steps(conn, rid)["phase2"] == "ok"


def test_trivial_tier_keeps_a_blocker_preliminary_under_the_budget(gated, conn, gh, lanes):
    """No Phase 2 to go to: a verified blocker must still be published as a blocker."""
    lanes.triage = {"tier": "trivial", "reasoning": "typo"}
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["preliminary"] = _verifier(_findings(blocking=1))
    _, status = _run(gated, conn, gh, lanes)
    assert status == RunStatus.DONE
    (review,) = gh.posted_reviews
    assert "phase=preliminary" in review["body"] and review["event"] == "REQUEST_CHANGES"


# ---- repos without Phase 1 ----


def _repo_off(raw):
    for r in raw["repos"]:
        if r["name"] == "platform":
            r["phase1"] = False


def test_repo_without_phase1_goes_straight_to_phase2(cfg, conn, gh, lanes, skills_dir, tmp_path):
    cfg = _policy(skills_dir, tmp_path, _repo_off)
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    rid, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert not _reviewers(lanes, "glm-5.3-flash"), "no Phase-1 lane may run"
    assert {s.role for s in _reviewers(lanes, "gpt-6-astra")} == REVIEWER_ROLES
    steps = _steps(conn, rid)
    assert steps["phase1"] == "skipped" and "verify1" not in steps and "gate" not in steps
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM events WHERE kind='phase1.skipped_repo' AND run_id=?", (rid,)
        ).fetchone()[0]
        == 1
    )
    body = gh.posted_reviews[0]["body"]
    assert "## Final validation — Phase 2 only (no Phase 1 for this repository)" in body
    assert "- Phase 1 reviewers: **not run (disabled for this repository)**" in body
    assert "Phase 2 only (no Phase 1 for this repository)" in gh.gate_bodies[-1]


def test_trivial_change_on_a_repo_without_phase1_reviews_at_low_effort(
    cfg, conn, gh, lanes, skills_dir, tmp_path
):
    cfg = _policy(skills_dir, tmp_path, _repo_off)
    lanes.triage = {"tier": "trivial", "reasoning": "typo"}
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    _, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert {(s.model, s.effort) for s in _reviewers(lanes)} == {("gpt-6-astra", "low")}


# ---- single stage ----


def _single_stage(raw):
    raw["review_model_policy"]["triage"]["tiers"]["critical"]["single_stage"] = True
    _gate(raw)


@pytest.fixture
def single(cfg, skills_dir, tmp_path, lanes):
    lanes.triage = {"tier": "critical", "reasoning": "consensus"}
    c = _policy(skills_dir, tmp_path, _single_stage)
    # gpt slots for the three Phase-2 reviewers at once (the top slot is the verifiers')
    return dataclasses.replace(c, max_concurrent=4)


def test_single_stage_runs_both_phases_side_by_side_without_a_gate(single, conn, gh, lanes):
    lanes.reviewer["default"] = {
        "summary": "ok",
        "findings": _findings(blocking=3),  # far over the gate budget: no gate here
        "out_of_scope_findings": [],
    }
    lanes.verifier["final"] = _verifier(_findings(blocking=1))
    runner = Overlap(lanes, barrier=6)  # 3 Phase-1 + 3 Phase-2 lanes, all at once
    rid, status = _run(single, conn, gh, runner)
    assert status == RunStatus.DONE
    assert runner.peak == 6
    assert {(s.model, s.effort) for s in _reviewers(lanes)} == {
        ("glm-5.3-flash", "max"),
        ("gpt-6-astra", "xhigh"),
    }
    steps = _steps(conn, rid)
    assert steps["phase1"] == "ok" and steps["phase2"] == "ok" and steps["verify2"] == "ok"
    assert "verify1" not in steps and "gate" not in steps
    verifiers = [s for s in lanes.calls if s.role == "verifier"]
    assert len(verifiers) == 1, "only the final verifier runs"
    assert "Phase 1 ran beside Phase 2 on this head" in verifiers[0].prompt
    assert '"general"' in verifiers[0].prompt.split("Codex:", 1)[1], "Phase-1 output reaches it"
    (review,) = gh.posted_reviews
    body = review["body"]
    assert "phase=final" in body and "## Final validation — Phase 1 + Phase 2" in body
    assert "- Single stage: Phase 1 and Phase 2 reviewed this head side by side" in body
    assert body.index("- Phase 1 reviewers: `glm-5.3-flash`") < body.index("- Phase 2 reviewers:")


def test_single_stage_drops_a_failed_phase1_and_still_publishes(single, conn, gh, lanes):
    lanes.dead_models = {"glm-5.3-flash"}  # the one-rung ladder has nowhere to fall
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["final"] = _verifier([])
    seen_while_phase2_ran: list[tuple[str, str]] = []

    def runner(spec: LaneSpec, art: Path, worktree: Path) -> LaneResult:
        if spec.role == "general" and spec.model == "gpt-6-astra":
            # Phase 2 is still going: the Phase-1 failure must already be on record (the
            # step's error for the status page, and the event), not hours later
            own = sqlite3.connect(str(single.db_path))
            try:
                for _ in range(1000):
                    row = own.execute(
                        "SELECT s.detail, (SELECT detail FROM events e WHERE e.run_id=s.run_id "
                        "AND e.kind='phase1.failed_single_stage') FROM steps s "
                        "WHERE s.name='phase1' AND s.status='failed'"
                    ).fetchone()
                    if row and row[1]:
                        seen_while_phase2_ran.append(row)
                        break
                    time.sleep(0.01)
            finally:
                own.close()
        return lanes(spec, art, worktree)

    rid, status = _run(single, conn, gh, runner)
    assert status == RunStatus.DONE
    ((detail, ev),) = seen_while_phase2_ran
    # whichever Phase-1 lane failed first
    assert re.search(r"phase1/\S+ lane failed twice: \[infra\]", json.loads(detail)["error"])
    assert re.search(r"phase1/\S+ lane failed twice", ev)
    assert {s.role for s in _reviewers(lanes, "gpt-6-astra")} == REVIEWER_ROLES
    assert _steps(conn, rid)["phase1"] == "failed"
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM events WHERE kind='phase1.failed_single_stage' AND run_id=?",
            (rid,),
        ).fetchone()[0]
        == 1
    )
    (verifier,) = [s for s in lanes.calls if s.role == "verifier"]
    assert "There is no Phase-1 evidence" in verifier.prompt
    body = gh.posted_reviews[0]["body"]
    assert "- Phase 1 reviewers: **failed (" in body and "- Single stage:" not in body


def test_single_stage_phase2_failure_stops_phase1_and_fails_the_run(single, conn, gh, lanes):
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    stopped = threading.Event()

    def runner(spec: LaneSpec, art: Path, worktree: Path) -> LaneResult:
        if spec.role == "general" and spec.model == "gpt-6-astra":
            return LaneResult(exit_code=None, stdout="", stderr="", duration_s=1, timed_out=True)
        if spec.model == "glm-5.3-flash" and spec.role in REVIEWER_ROLES:
            # a slow Phase-1 lane: it must be told to stop, not waited out
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if spec.should_stop and spec.should_stop():
                    stopped.set()
                    return LaneResult(
                        exit_code=None, stdout="", stderr="", duration_s=1, cancelled=True
                    )
                time.sleep(0.01)
        return lanes(spec, art, worktree)

    rid, status = _run(single, conn, gh, runner)
    assert status == RunStatus.FAILED
    assert stopped.is_set(), "Phase-1 lanes are stopped when Phase 2 fails"
    assert not gh.posted_reviews
    # the failure is charged to Phase 2; the stopped Phase 1 does not stay `running`
    assert _steps(conn, rid)["phase2"] == "failed"
    assert _steps(conn, rid)["phase1"] == "cancelled"
    assert conn.execute("SELECT phase FROM runs WHERE id=?", (rid,)).fetchone()["phase"] == "phase2"


# ---- retention ----


def test_run_artifacts_are_kept_forever_at_retention_zero(cfg, conn):
    old = cfg.runs_dir / "run-1"
    (old / "attempts").mkdir(parents=True)
    t = time.time() - 400 * 86400
    os.utime(old, (t, t))
    gc.run(conn, cfg)
    assert old.exists()
    gc.run(conn, dataclasses.replace(cfg, artifact_retention_days=14))
    assert not old.exists()
