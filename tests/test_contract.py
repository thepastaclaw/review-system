import pytest

from reviewsys.contract import (
    Finding,
    finding_dedupe_key,
    finding_hash,
    parse_json_object,
    parse_reviewer_output,
    parse_verifier_output,
)
from reviewsys.models import ReviewError


def test_hashes_match_legacy_posted_markers():
    # From dashpay/platform#4581 inline comment: finding=2f4048ab34cd dedupe=8a923bd878dc1450
    f = "packages/rs-drive/src/drive/document/insert/add_indices_for_top_index_level_for_contract_operations/v2/mod.rs"
    title = "Fee estimation omits the TTL drainage costs charged during execution"
    assert (
        finding_hash(f, "logic", title) == "2f4048ab34cd"
        or finding_hash(f, "bug", title) == "2f4048ab34cd"
    )


def test_dedupe_key_is_stable_under_punctuation():
    a = finding_dedupe_key("f.rs", "bug", "Missing bounds check!", "The index can overflow.")
    b = finding_dedupe_key("f.rs", "bug", "missing bounds check", "the index can overflow")
    assert a == b and len(a) == 16


def test_parse_json_object_tolerates_fences_and_prose():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Here you go:\n{"a": {"b": 2}}\nthanks') == {"a": {"b": 2}}
    with pytest.raises(ReviewError):
        parse_json_object("no json here")


def test_reviewer_output_requires_phase_and_head():
    raw = {"summary": "s", "findings": [], "review_phase": "final", "head_sha": "x"}
    with pytest.raises(ReviewError, match="review_phase"):
        parse_reviewer_output(
            raw, expected_phase="preliminary", head_sha="x", source="p1", prior_hashes=set()
        )
    with pytest.raises(ReviewError, match="head_sha"):
        parse_reviewer_output(
            raw, expected_phase="final", head_sha="y", source="p1", prior_hashes=set()
        )


def test_reviewer_output_reconciliation_contract():
    prior = {"abc"}
    ok = {
        "summary": "s",
        "review_phase": "final",
        "head_sha": "h",
        "findings": [
            {"file": "f", "title": "T", "body": "b", "severity": "nitpick", "finding_hash": "abc"}
        ],
        "prior_finding_reconciliation": [{"finding_hash": "abc", "status": "STILL_VALID"}],
    }
    out = parse_reviewer_output(
        ok, expected_phase="final", head_sha="h", source="x", prior_hashes=prior
    )
    assert out.findings[0].prior_hash == "abc"
    bad = {**ok, "prior_finding_reconciliation": [{"finding_hash": "abc", "status": "FIXED"}]}
    with pytest.raises(ReviewError, match="STILL_VALID"):
        parse_reviewer_output(
            bad, expected_phase="final", head_sha="h", source="x", prior_hashes=prior
        )
    missing = {**ok, "prior_finding_reconciliation": []}
    with pytest.raises(ReviewError, match="missing"):
        parse_reviewer_output(
            missing, expected_phase="final", head_sha="h", source="x", prior_hashes=prior
        )


def test_verifier_output_contract():
    base = {
        "summary": "s",
        "review_action": "COMMENT",
        "findings": [],
        "prerequisite_adjudications": [],
        "adjudication_complete": True,
        "review_phase": "final",
        "coderabbit_reactions": [],
    }
    out = parse_verifier_output(base, expected_phase="final", expected_coderabbit_ids=[])
    assert out.blocker_count == 0
    with pytest.raises(ReviewError, match="adjudication_complete"):
        parse_verifier_output(
            {**base, "adjudication_complete": False},
            expected_phase="final",
            expected_coderabbit_ids=[],
        )
    with pytest.raises(ReviewError, match="retrigger"):
        parse_verifier_output(
            {**base, "summary": "please @coderabbitai review"},
            expected_phase="final",
            expected_coderabbit_ids=[],
        )
    with pytest.raises(ReviewError, match="reactions cover"):
        parse_verifier_output(base, expected_phase="final", expected_coderabbit_ids=[42])
    good = {
        **base,
        "coderabbit_reactions": [{"comment_id": 42, "action": "disagree", "reply": "no"}],
    }
    assert (
        parse_verifier_output(
            good, expected_phase="final", expected_coderabbit_ids=[42]
        ).coderabbit_reactions[0]["action"]
        == "disagree"
    )


def test_finding_invalid_severity():
    with pytest.raises(ReviewError, match="severity"):
        Finding.from_dict({"file": "f", "title": "t", "severity": "critical"})
