"""Under a v11 policy every lane prompt must stay exactly what v0.24.0 sent. The expected
text under golden/v11-prompts/ was generated with v0.24.0's prompts.py (72ff47d) from the
cases below and is compared as copied text, never re-derived from today's code."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

from reviewsys import prompts

GOLDEN = Path(__file__).parent / "golden" / "v11-prompts"
HEAD = "a" * 40
PRIOR = [
    {
        "finding_hash": "0123456789ab",
        "original_title": "Old finding",
        "file": "src/lib.rs",
        "line_start": 3,
        "line_end": 4,
        "severity": "blocking",
        "category": "bug",
        "prior_head_sha": "c" * 40,
    }
]
EVIDENCE = {"pr": {"title": "t"}, "ci": {"state": "pending"}}


def v11_prompts(p: ModuleType, cfg) -> dict[str, str]:
    """Every prompt kind, with and without prior findings, from prompts module `p`."""
    review = {
        "repo": "dashpay/platform",
        "number": 7,
        "head_sha": HEAD,
        "meta": {"title": "Fix it", "body": "Body", "baseRefName": "v3.1-dev"},
        "coverage_from": "b" * 40,
        "evidence": EVIDENCE,
    }
    verify = {
        "repo": "dashpay/platform",
        "number": 7,
        "head_sha": HEAD,
        "phase1_outputs": {"general": {"summary": "p1"}},
        "phase2_outputs": {"general": {"summary": "p2"}},
        "coderabbit": {"comments": [{"id": 11}]},
        "evidence": EVIDENCE,
    }
    correct = {"errors": ["first", "second"], "turn": 1, "turns": 2}
    return {
        "reviewer-first": p.reviewer_prompt(
            cfg, **review, phase="preliminary", role="general", prior=[], prior_sha=None
        ),
        "reviewer-prior": p.reviewer_prompt(
            cfg, **review, phase="final", role="security-auditor", prior=PRIOR, prior_sha="c" * 40
        ),
        "reviewer-fresh": p.reviewer_prompt(
            cfg,
            **review,
            phase="final",
            role="general",
            prior=PRIOR,
            prior_sha="c" * 40,
            fresh=True,
        ),
        "verifier-prior": p.verifier_prompt(
            cfg,
            **verify,
            phase="preliminary",
            coderabbit_ids=[11],
            prior=PRIOR,
            prior_sha="c" * 40,
        ),
        "verifier-fresh-final": p.verifier_prompt(
            cfg,
            **verify,
            phase="final",
            coderabbit_ids=[],
            prior=[],
            prior_sha=None,
            fresh_final=True,
            single_stage=True,
        ),
        "correction-reviewer": p.correction_prompt(
            **correct,
            kind="reviewer",
            expected_phase="final",
            head_sha=HEAD,
            prior=PRIOR,
            coderabbit_ids=[],
        ),
        "correction-verifier": p.correction_prompt(
            **correct,
            kind="verifier",
            expected_phase="preliminary",
            head_sha=HEAD,
            prior=[],
            coderabbit_ids=[11],
        ),
        "repair-reviewer": p.repair_prompt(
            '{"summary": "x"',
            kind="reviewer",
            expected_phase="final",
            head_sha=HEAD,
            prior_hashes=["0123456789ab"],
        ),
        "repair-verifier": p.repair_prompt(
            '{"summary": "x"',
            kind="verifier",
            expected_phase="final",
            head_sha=HEAD,
            prior_hashes=[],
        ),
        "repair-plain": p.repair_prompt(
            "{", kind="", expected_phase="", head_sha=HEAD, prior_hashes=[]
        ),
    }


@pytest.mark.parametrize(
    "name",
    sorted(path.stem for path in GOLDEN.glob("*.txt")),
)
def test_v11_prompt_is_v024s(cfg, name):
    assert cfg.policy.pipeline is None
    got = v11_prompts(prompts, cfg)[name]
    assert got == (GOLDEN / f"{name}.txt").read_text(encoding="utf-8")


def test_every_case_has_its_golden(cfg):
    assert sorted(v11_prompts(prompts, cfg)) == sorted(p.stem for p in GOLDEN.glob("*.txt"))
