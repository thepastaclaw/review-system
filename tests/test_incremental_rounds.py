"""Later review rounds are incremental. Over 260 client pull requests reviewed since
2026-11-10, 60% of the blockers raised after round one were on lines the previous push had not
touched, and the rate of new blockers did not fall from round 2 to round 15: every round
re-read the whole pull request with every lane. dashpay/platform#6057 took 13 rounds in six
hours."""

from __future__ import annotations

from reviewsys import prompts

HEAD, SINCE, BASE = "a" * 40, "c" * 40, "b" * 40
PRIOR = [
    {
        "finding_hash": "0123456789ab",
        "original_title": "Old finding",
        "file": "src/lib.rs",
        "line_start": 3,
        "line_end": 4,
        "severity": "blocking",
        "category": "bug",
        "prior_head_sha": SINCE,
    }
]
REVIEW = {
    "repo": "dashpay/platform",
    "number": 7,
    "head_sha": HEAD,
    "meta": {"title": "Fix it", "body": "Body", "baseRefName": "v3.1-dev"},
    "coverage_from": BASE,
    "evidence": {},
    "phase": "final",
    "role": "general",
}
VERIFY = {
    "repo": "dashpay/platform",
    "number": 7,
    "head_sha": HEAD,
    "phase": "final",
    "phase1_outputs": {},
    "phase2_outputs": {},
    "coderabbit": {},
    "coderabbit_ids": [],
    "evidence": {},
}


def test_a_later_round_is_asked_for_what_changed_since_the_last_full_review(cfg):
    p = prompts.reviewer_prompt(cfg, **REVIEW, prior=PRIOR, prior_sha=SINCE, since_sha=SINCE)
    assert f"reviewed in full at `{SINCE}`. Do not review it again from the start" in p
    assert f"`git diff {SINCE}..{HEAD}`" in p and f"`git range-diff {SINCE}...{HEAD}`" in p
    assert "Report a new finding only on code that changed since then" in p
    assert "loses funds, breaks consensus or exposes key material" in p
    assert f"Review what changed since `{SINCE}`, within the complete range `{BASE}..{HEAD}`." in p
    assert "Review exactly the complete range" not in p
    assert "Review the full stated range" not in p
    # the prior findings are still reconciled
    assert "Reconcile every prior PastaClaw finding exactly once" in p and "Old finding" in p


def test_a_new_blocker_of_a_later_round_needs_actual_reachability(cfg):
    p = prompts.reviewer_prompt(cfg, **REVIEW, prior=PRIOR, prior_sha=SINCE, since_sha=SINCE)
    assert "A new finding is `blocking` only with actual reachability" in p
    assert "(1) the entry point: something a user of the shipped product" in p
    assert "a sequence of steps nobody would perform" in p
    v = prompts.verifier_prompt(cfg, **VERIFY, prior=PRIOR, prior_sha=SINCE, since_sha=SINCE)
    assert "a reason that starts with `ALREADY_REVIEWED`" in v
    assert "which you check in the code yourself: (1) the entry point" in v
    assert "A carried prior finding is not raised to `blocking` unless code that changed" in v
    # a blocker that is a project rule, not a failure, is judged by the rule
    assert "judged by that rule instead: name the rule" in p and "name the rule" in v


def test_a_first_round_a_fresh_pass_and_a_same_head_review_read_everything(cfg):
    first = prompts.reviewer_prompt(cfg, **REVIEW, prior=[], prior_sha=None)
    fresh = prompts.reviewer_prompt(
        cfg, **REVIEW, prior=PRIOR, prior_sha=SINCE, fresh=True, since_sha=SINCE
    )
    again = prompts.reviewer_prompt(cfg, **REVIEW, prior=PRIOR, prior_sha=SINCE)
    for p in (first, fresh, again):
        assert "later review round" not in p
        assert f"Review exactly the complete range `{BASE}..{HEAD}`." in p
    v = prompts.verifier_prompt(cfg, **VERIFY, prior=PRIOR, prior_sha=SINCE)
    assert "ALREADY_REVIEWED" not in v


def test_a_later_round_with_nothing_left_to_reconcile_is_not_called_a_first_round(cfg):
    p = prompts.reviewer_prompt(cfg, **REVIEW, prior=[], prior_sha=None, since_sha=SINCE)
    assert "first PastaClaw automated review round" not in p
    assert "No finding of an earlier round is left to reconcile" in p
    assert "later review round" in p
    v = prompts.verifier_prompt(cfg, **VERIFY, prior=[], prior_sha=None, since_sha=SINCE)
    assert "first PastaClaw automated review round" not in v
