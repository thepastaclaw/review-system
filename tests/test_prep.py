"""The prep lane: specialist selection and effort triage asked in one lane, each half
validated and falling back on its own, with the observable outputs of the two lanes it
replaced (step rows, selector.json / triage.json, events, runs.tier, provenance)."""

from __future__ import annotations

import json

import pytest

from reviewsys import config as cfg_mod
from reviewsys import prep, triage, worker
from reviewsys.db import tx
from reviewsys.ingest import enqueue_head
from reviewsys.lane import LaneResult
from reviewsys.models import RunStatus, Trigger
from reviewsys.scheduler import schedule

HEAD = "a" * 40
MUSE = "muse-spark-1.3-contributor"


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


def _verifier():
    return {
        "summary": "ok",
        "review_action": "COMMENT",
        "findings": [],
        "dropped_findings": [],
        "out_of_scope_findings": [],
        "coderabbit_reactions": [],
        "prerequisite_adjudications": [],
        "adjudication_complete": True,
    }


def _run(cfg, conn, gh, lanes, **kw):
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier()
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.MENTION)
    (rid,) = schedule(conn, cfg, spawn=False)
    status = worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False, **kw)
    assert status == RunStatus.DONE
    return rid


def _artifacts(cfg, rid):
    d = cfg.runs_dir / f"run-{rid}"
    return (
        json.loads((d / "selector.json").read_text()),
        json.loads((d / "triage.json").read_text()),
    )


def _events(conn, rid):
    return [r[0] for r in conn.execute("SELECT kind FROM events WHERE run_id=?", (rid,))]


def _steps(conn, rid):
    return {
        r["name"]: dict(r)
        for r in conn.execute(
            "SELECT name,status,started_at,finished_at,detail FROM steps WHERE run_id=?", (rid,)
        )
    }


def _reload(skills_dir, tmp_path, edit):
    raw = json.loads((skills_dir / "config.json").read_text())
    edit(raw)
    (skills_dir / "config.json").write_text(json.dumps(raw))
    return cfg_mod.load(tmp_path / "config.toml")


def test_one_lane_answers_both_and_keeps_both_step_rows(cfg, conn, gh, lanes):
    lanes.triage = {"tier": "low", "reasoning": "small"}
    lanes.selector = {"selected": ["security-auditor", "nope"], "reasoning": "auth code"}
    rid = _run(cfg, conn, gh, lanes)
    side = [s for s in lanes.calls if s.role in ("prep", "selector", "triage")]
    assert [(s.role, s.model, s.effort) for s in side] == [("prep", "gpt-6-astra", "low")]
    # both questions, the tier guide and rules and the specialist rule verbatim
    p = side[0].prompt
    assert all(g in p for g in triage.TIER_GUIDE.values())
    assert triage.TIER_RULES.strip() in p
    assert "- `security-auditor`: security" in p and "`always-on`" not in p
    assert "## Part 1: effort tier" in p and "## Part 2: specialist reviewers" in p
    sel, tri = _artifacts(cfg, rid)
    assert sel["method"] == tri["method"] == "llm:gpt-6-astra"
    assert sel["selected"] == ["always-on", "security-auditor"], "unknown ids dropped"
    assert sel["reasoning"] == "auth code" and sel["error"] is None
    assert tri == {"tier": "low", "method": "llm:gpt-6-astra", "reasoning": "small", "error": None}
    assert conn.execute("SELECT tier FROM runs WHERE id=?", (rid,)).fetchone()[0] == "low"
    steps = _steps(conn, rid)
    assert steps["select"]["status"] == steps["triage"]["status"] == "ok"
    assert json.loads(steps["select"]["detail"]) == {"selection": sel["selected"]}
    assert json.loads(steps["triage"]["detail"])["tier"] == "low"
    # recorded from the one lane: the triage row lies within the select row
    assert steps["select"]["started_at"] <= steps["triage"]["started_at"]
    assert steps["triage"]["finished_at"] >= steps["select"]["finished_at"]
    assert "select.degraded" not in _events(conn, rid)
    assert "triage.degraded" not in _events(conn, rid)
    assert "- Triage: `low` by `gpt-6-astra` (effort low) — small" in gh.posted_reviews[0]["body"]


def test_bad_tier_falls_back_alone(cfg, conn, gh, lanes):
    lanes.triage = {"tier": "enormous", "reasoning": "?"}
    rid = _run(cfg, conn, gh, lanes)
    assert [s.role for s in lanes.calls].count("prep") == 2, "retried for the missing half"
    sel, tri = _artifacts(cfg, rid)
    assert sel["method"] == "llm:gpt-6-astra" and sel["selected"] == [
        "always-on",
        "security-auditor",
    ]
    assert tri["tier"] == "normal" and tri["method"] == "fallback"
    assert tri["error"] == "unknown tier 'enormous'"
    kinds = _events(conn, rid)
    assert "triage.degraded" in kinds and "select.degraded" not in kinds
    assert "- Triage: `normal` by fallback after triage failure" in gh.posted_reviews[0]["body"]


def test_bad_selection_falls_back_alone(cfg, conn, gh, lanes):
    lanes.triage = {"tier": "critical", "reasoning": "consensus"}
    lanes.selector = "not a list"
    rid = _run(cfg, conn, gh, lanes)
    sel, tri = _artifacts(cfg, rid)
    assert tri["tier"] == "critical" and tri["method"] == "llm:gpt-6-astra"
    assert sel["method"] == "heuristic" and "always-on" in sel["selected"]
    assert "not a list" in sel["error"]
    kinds = _events(conn, rid)
    assert "select.degraded" in kinds and "triage.degraded" not in kinds
    assert conn.execute("SELECT tier FROM runs WHERE id=?", (rid,)).fetchone()[0] == "critical"


def test_both_halves_bad(cfg, conn, gh, lanes):
    lanes.triage, lanes.selector = "garbage", "more garbage"  # no JSON at all
    rid = _run(cfg, conn, gh, lanes)
    assert [s.role for s in lanes.calls].count("prep") == 2
    sel, tri = _artifacts(cfg, rid)
    assert sel["method"] == "heuristic" and tri["method"] == "fallback"
    assert sel["error"] and tri["error"]
    kinds = _events(conn, rid)
    assert "select.degraded" in kinds and "triage.degraded" in kinds


def test_a_good_half_is_kept_across_attempts(cfg, tmp_path):
    """The second attempt only fills the half the first one missed; a later answer for the
    half already settled is ignored."""
    outs = iter(
        [
            {"tier": "low", "tier_reasoning": "small", "selected": "oops"},
            {"tier": "critical", "selected": ["security-auditor"], "selection_reasoning": "r"},
        ]
    )

    def runner(spec, art, wt):
        text = json.dumps(next(outs))
        return LaneResult(exit_code=0, stdout="", stderr="", duration_s=1, result_text=text)

    p = prep.prep(
        cfg,
        repo="dashpay/platform",
        base_ref="develop",
        title="t",
        body="b",
        files=[{"filename": "f.rs", "additions": 1}],
        run_dir=tmp_path,
        worktree=tmp_path,
        runner=runner,
    )
    assert p.triage.tier == "low" and p.triage.reasoning == "small"
    assert (
        p.selection.selected == ["always-on", "security-auditor"] and p.selection.reasoning == "r"
    )
    assert p.attempts == 2 and p.infra_error is None


def test_without_triage_the_selector_runs_alone(cfg, conn, gh, lanes, skills_dir, tmp_path):
    c = _reload(skills_dir, tmp_path, lambda raw: raw["review_model_policy"].pop("triage"))
    rid = _run(c, conn, gh, lanes)
    side = [s for s in lanes.calls if s.role in ("prep", "selector", "triage")]
    assert [(s.role, s.model) for s in side] == [("selector", "gpt-5.6-terra")]
    assert "triage" not in _steps(conn, rid) and "select" in _steps(conn, rid)


def test_without_discretionary_specialists_the_lane_only_rates(
    cfg, conn, gh, lanes, skills_dir, tmp_path
):
    c = _reload(
        skills_dir,
        tmp_path,
        lambda raw: raw.update(specialists=[s for s in raw["specialists"] if s.get("always_run")]),
    )
    lanes.triage = {"tier": "low", "reasoning": "small"}
    rid = _run(c, conn, gh, lanes)
    side = [s for s in lanes.calls if s.role in ("prep", "selector", "triage")]
    assert [s.role for s in side] == ["triage"], "nothing to choose: no selection asked"
    sel, tri = _artifacts(c, rid)
    assert sel["method"] == "config" and sel["selected"] == ["always-on"]
    assert tri["tier"] == "low"
    assert {"select", "triage"} <= set(_steps(conn, rid))


def test_quota_failure_retries_on_the_stand_in_keeping_the_good_half(
    cfg, conn, gh, lanes, skills_dir, tmp_path
):
    """A prep lane dying on a dry pool flips the run degraded and asks the stand-in again;
    an infra failure is what counts, never a half the model got wrong."""
    c = _reload(
        skills_dir,
        tmp_path,
        lambda raw: raw["review_model_policy"].update(
            degraded={
                "sentinel": "gpt-6-astra",
                "substitutes": {"gpt-6-astra": MUSE, "gpt-5.6-sol": MUSE},
            }
        ),
    )
    lanes.dead_models = {"gpt-6-astra"}
    rid = _run(c, conn, gh, lanes, prober=lambda m: (False, "fine"))
    assert [s.model for s in lanes.calls if s.role == "prep"] == ["gpt-6-astra"] * 2 + [MUSE]
    sel, tri = _artifacts(c, rid)
    assert sel["method"] == tri["method"] == f"llm:{MUSE}"
    kinds = _events(conn, rid)
    assert "degraded.entered_midrun" in kinds
    assert "select.degraded" not in kinds and "triage.degraded" not in kinds


def test_a_contract_failure_never_flips_the_run(cfg, conn, gh, lanes, skills_dir, tmp_path):
    c = _reload(
        skills_dir,
        tmp_path,
        lambda raw: raw["review_model_policy"].update(
            degraded={"sentinel": "gpt-6-astra", "substitutes": {"gpt-6-astra": MUSE}}
        ),
    )
    lanes.triage = {"tier": "429 rate limited, cooling down", "reasoning": "x"}
    rid = _run(c, conn, gh, lanes, prober=lambda m: (False, "fine"))
    assert MUSE not in {s.model for s in lanes.calls}
    assert "degraded.entered_midrun" not in _events(conn, rid)
    assert _artifacts(c, rid)[1]["method"] == "fallback"
