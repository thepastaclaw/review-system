"""Policy v12 finders (B1): the candidate contract, harness ids, the finder prompt (no prior
checklist, the whole range, a pointer to what changed, the settled list), and unanchored
finders under a v11 verifier."""

from __future__ import annotations

import pytest

from reviewsys import prompts, rules, worker
from reviewsys.models import FailKind, ReviewError
from tests.test_pipeline_policy import load_with
from tests.test_worker_e2e import (
    _verifier,
    fake_git,  # noqa: F401  (autouse: fake mirror/worktree)
)
from tests.v12_support import REPO, cand, run_round

A = "a" * 40


def test_candidates_get_harness_ids_in_order():
    out = rules.parse_finder_output(
        {
            "summary": "s",
            "candidates": [
                {
                    "file": "a.rs",
                    "line_start": 3,
                    "severity": "P1",
                    "category": "BUG",
                    "title": "x",
                    "failure_scenario": "f",
                    "body": "b",
                },
                {
                    "file": "b.rs",
                    "severity": "nitpick",
                    "category": "weird",
                    "title": "y",
                    "body": "b",
                    "confidence": 0.3,
                },
            ],
        },
        id_prefix="p2-general",
        source="phase2:general",
        expected_phase="final",
        head_sha="h",
    )
    a, b = out.findings
    assert (a.candidate_id, b.candidate_id) == ("p2-general-1", "p2-general-2")
    assert (a.severity, a.category, a.line_end) == ("blocking", "bug", 3)
    assert b.category == "general" and b.confidence is None  # confidence is not read
    assert a.extra[rules.FAILURE_SCENARIO] == "f"
    # the shape a v11 verifier reads
    assert out.raw["findings"][0]["candidate_id"] == "p2-general-1"


def test_candidate_problems_are_reported_together_and_findings_is_an_alias():
    with pytest.raises(ReviewError) as exc:
        rules.parse_finder_output(
            {"candidates": [{"severity": "huge", "title": "x"}, {"severity": "nitpick"}, 3]},
            id_prefix="p1-general",
            source="s",
            expected_phase="preliminary",
            head_sha="h",
        )
    msg = exc.value.message
    assert "candidate 1 has invalid severity" in msg and "candidate 2 has no title" in msg
    assert "candidate 3 is not an object" in msg and exc.value.kind is FailKind.CONTRACT
    out = rules.parse_finder_output(
        {"findings": [{"severity": "nitpick", "title": "x"}]},
        id_prefix="p1-general",
        source="s",
        expected_phase="preliminary",
        head_sha="h",
    )
    assert out.normalized == ["candidates read from `findings`"] and len(out.findings) == 1


def test_candidate_prefixes():
    assert rules.candidate_prefix("phase1", "general") == "p1-general"
    assert rules.candidate_prefix("phase2", "general/a") == "p2-general/a"
    assert rules.candidate_prefix("phase2", "security-auditor", fresh=True) == "cf-security-auditor"


def test_unanchored_finders_alone_load(cfg, skills_dir):
    p = load_with(
        cfg, skills_dir, {"version": "v12", "finders": {"anchoring": "unanchored"}}
    ).policy
    assert p.unanchored and p.pipeline is not None
    assert p.pipeline.verifier.contract == "v11" and not p.pipeline.threads.lane


def test_unanchored_finders_under_a_v11_verifier(cfg, skills_dir, conn, gh, lanes):
    """Piece B1 alone: finders on the candidate contract, the v11 verifier weighing them."""
    from tests.test_pipeline_policy import load_with

    b1 = load_with(cfg, skills_dir, {"version": "v12", "finders": {"anchoring": "unanchored"}})
    lanes.candidates["default"] = [
        cand("Fee estimate omits drainage", severity="suggestion", line=12)
    ]
    lanes.verifier["default"] = _verifier([])
    run_round(b1, conn, gh, lanes, A)
    v11_verify = [s for s in lanes.calls if s.role == "verifier"]
    assert v11_verify and '"candidate_id": "p2-general-1"' in v11_verify[-1].prompt
    assert all(
        "prior_finding_reconciliation" not in s.prompt.split("## Output")[0]
        for s in lanes.calls
        if "## Output: candidates" in s.prompt
    )


def test_comparison_twins_get_blinded_candidate_ids(cfg, conn, tmp_path):
    ctx = worker.RunContext(
        cfg=cfg,
        main_conn=conn,
        gh=None,
        run_id=7,
        head_id=1,
        repo=REPO,
        number=1,
        sha=A,
        token="t",
        run_dir=tmp_path,
    )
    assert worker._blinded_key(ctx, "general#2") == "general#2"
    ctx.compare_model = "m2"
    a, b = worker._blinded_key(ctx, "general"), worker._blinded_key(ctx, "general#2")
    assert {a, b} == {"general/a", "general/b"}
    blind = worker._blind({"general": 1, "general#2": 2}, seed=7)
    assert blind[a] == 1 and blind[b] == 2  # the same labels the verifier sees


def _finder(cfg, **kw):
    base = dict(
        repo=REPO,
        number=1,
        head_sha=A,
        role="general",
        lane_title="general review",
        meta={"title": "Fix it", "body": "Body", "baseRefName": "develop"},
        coverage_from="b" * 40,
        evidence={"pr": {"title": "Fix it"}},
        guidance="",
        last_reviewed_sha=None,
        settled=[],
        structured=False,
    )
    return prompts.finder_prompt(cfg, **{**base, **kw})


def test_the_finder_prompt_carries_no_anchor(cfg):
    first = _finder(cfg)
    head = first.split("## Output: candidates")[0]
    for absent in ("prior_finding_reconciliation", "finding_hash", "confidence", "review_phase"):
        assert absent not in head
    assert f"git diff {'b' * 40}..{A}" in first and "General method for dashpay/platform" in first
    assert "last reviewed" not in first and "Settled in earlier rounds" not in first
    assert first.endswith(prompts.RAW_JSON_CONTRACT)
    later = _finder(
        cfg,
        last_reviewed_sha="c" * 40,
        settled=[
            {
                "file": "f.rs",
                "lines": [3, 4],
                "title": "Old",
                "status": "withdrawn",
                "reason": "No.",
            }
        ],
        structured=True,
    )
    assert f"last reviewed at `{'c' * 40}`" in later and "review the whole range" in later
    assert "### Settled in earlier rounds" in later and '"reason": "No."' in later
    assert later.endswith(prompts.STRUCTURED_CONTRACT) and prompts.RAW_JSON_CONTRACT not in later


def test_v12_lanes_are_told_about_sister_checkouts_and_base_drift(cfg):
    """As v11's reviewer and verifier are (prompts.context_notes), and only when the context
    step found them."""
    evidence = {
        "pr": {"title": "Fix it"},
        "sister_repos": [{"repo": "dashpay/rust-dashcore", "pinned": None, "default": None}],
        "base_drift": {"commits_behind": 3, "conflicting_files": []},
    }
    assert prompts.SISTER_REPOS_NOTE not in _finder(cfg)
    finder = _finder(cfg, evidence=evidence)
    assert prompts.SISTER_REPOS_NOTE in finder and prompts.BASE_DRIFT_NOTE in finder
    verify = prompts.verify_prompt(
        cfg,
        repo=REPO,
        number=1,
        head_sha=A,
        meta={"title": "Fix it", "body": "Body", "baseRefName": "develop"},
        coverage_from="b" * 40,
        candidates=[],
        coderabbit={},
        ledger_summary={},
        evidence=evidence,
        guidance="",
        phase_note="",
        structured=False,
    )
    assert prompts.SISTER_REPOS_NOTE in verify and prompts.BASE_DRIFT_NOTE in verify


def test_the_v12_verifier_reads_the_repos_verifier_skill(cfg, skills_dir):
    """A repo's verifier-only `verifier.md` reaches the v12 verify lane as it does v11's
    verifier (in its `{review_skill}`, as the skills repo's template has it), and never a
    finder."""
    template = skills_dir / "prompts" / "v12" / "verifier.md"
    template.write_text("{review_skill}\n" + template.read_text())
    path = cfg.skills_dir / cfg.repo(REPO).skill_path / "verifier.md"
    path.write_text("VERIFIER-ONLY SKILL")
    kw = dict(
        repo=REPO,
        number=1,
        head_sha=A,
        meta={"title": "Fix it", "body": "Body", "baseRefName": "develop"},
        coverage_from="b" * 40,
        candidates=[],
        coderabbit={},
        ledger_summary={},
        evidence={},
        guidance="",
        phase_note="",
        structured=False,
    )
    assert "VERIFIER-ONLY SKILL" in prompts.verify_prompt(cfg, **kw)
    assert "VERIFIER-ONLY SKILL" not in _finder(cfg)


def test_a_template_without_its_slots_is_fatal(cfg, skills_dir):
    (skills_dir / "prompts" / "v12" / "finder.md").write_text("no slots here")
    with pytest.raises(ReviewError) as exc:
        _finder(cfg)
    assert exc.value.kind is FailKind.FATAL and "{method}" in exc.value.message


def test_a_finder_correction_turn_names_the_keys(cfg):
    text = prompts.correction_prompt(
        errors=["finder output: candidate 1 has no title"],
        kind="finder",
        expected_phase="final",
        head_sha=A,
        prior=[],
        coderabbit_ids=[],
        turn=1,
        turns=2,
        stamp_echo=True,
    )
    assert "keys summary, candidates[]" in text and "review_phase" not in text
