import json

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


HEAD = "cff60ad9f4" + "1" * 30


def _reviewer(raw, *, phase="preliminary", head=HEAD, prior=frozenset()):
    return parse_reviewer_output(
        raw, expected_phase=phase, head_sha=head, source="p1", prior_hashes=set(prior)
    )


@pytest.mark.parametrize(
    ("given", "note"),
    [
        ({}, "review_phase missing -> 'preliminary'"),  # Gemini, run 2883: no echo fields
        ({"review_phase": "final"}, "review_phase 'final' -> 'preliminary'"),
        ({"review_phase": "planning"}, "review_phase 'planning' -> 'preliminary'"),
        ({"review_phase": ""}, "review_phase missing -> 'preliminary'"),
    ],
)
def test_review_phase_is_an_echo_field_set_to_the_expected_phase(given, note):
    out = _reviewer({"findings": [], "out_of_scope_findings": [], **given})
    assert out.review_phase == out.raw["review_phase"] == "preliminary"
    assert note in out.normalized
    # the verifier is shown the normalized output, with the full head too
    assert out.raw["head_sha"] == HEAD and out.head_sha == HEAD


def test_head_sha_prefix_or_missing_is_expanded_a_foreign_sha_rejected():
    # repaired outputs on the box came back with the sha cut to 8-14 hex chars
    out = _reviewer({"findings": [], "review_phase": "preliminary", "head_sha": "cff60ad9f4"})
    assert out.head_sha == out.raw["head_sha"] == HEAD
    assert out.normalized == [f"head_sha 'cff60ad9f4' -> {HEAD}"]
    out = _reviewer({"findings": [], "review_phase": "preliminary", "head_sha": HEAD.upper()})
    assert out.head_sha == HEAD
    clean = _reviewer({"findings": [], "review_phase": "preliminary", "head_sha": HEAD})
    assert clean.normalized == []
    for foreign in ("a" * 40, "cff60ad9f5", "cff60a", HEAD + "00", "HEAD"):
        # another commit, too short to be sure, longer than the sha, or not a sha at all
        with pytest.raises(ReviewError, match="head_sha"):
            _reviewer({"findings": [], "review_phase": "preliminary", "head_sha": foreign})


def test_verifier_review_phase_is_normalized_too():
    base = {
        "findings": [],
        "prerequisite_adjudications": [],
        "adjudication_complete": True,
        "coderabbit_reactions": [],
    }
    for phase in (None, "preliminary|final", "complete"):
        raw = {**base, "review_phase": phase} if phase else base
        out = parse_verifier_output(raw, expected_phase="final", expected_coderabbit_ids=[])
        assert out.review_phase == out.raw["review_phase"] == "final" and out.normalized


PRIOR = {"ccf9a4a010e4", "a265b2d54f12", "0123456789ab"}


def _recon_raw(rows, findings=()):
    return {
        "findings": list(findings),
        "review_phase": "final",
        "head_sha": HEAD,
        "prior_finding_reconciliation": rows,
    }


def test_reconciliation_hash_prefix_and_overlong_hash_map_to_the_one_prior_hash():
    rows = [
        {"finding_hash": "ccf9a4a010e", "status": "FIXED"},  # 11 chars: a prefix
        {"finding_hash": "a265b2d54f12f", "status": "still valid"},  # 13: prior is its prefix
        {"finding_hash": "0123456789AB", "status": "Withdrawn"},
    ]
    carried = {"file": "f", "title": "T", "body": "b", "finding_hash": "a265b2d54"}
    out = _reviewer(_recon_raw(rows, [carried]), phase="final", prior=PRIOR)
    statuses = {r["finding_hash"]: r["status"] for r in out.prior_reconciliation}
    assert statuses == {
        "ccf9a4a010e4": "FIXED",
        "a265b2d54f12": "STILL_VALID",
        "0123456789ab": "WITHDRAWN",
    }
    assert out.findings[0].prior_hash == "a265b2d54f12"
    assert "reconciliation hash 'ccf9a4a010e' -> 'ccf9a4a010e4'" in out.normalized
    assert "reconciliation status 'still valid' -> 'STILL_VALID'" in out.normalized
    assert "carried finding hash 'a265b2d54' -> 'a265b2d54f12'" in out.normalized


def test_ambiguous_or_short_hash_and_other_vocabularies_are_left_to_the_correction_turn():
    prior = {"abcdef012345", "abcdef019999"}
    rows = [
        {"finding_hash": "abcdef01", "status": "FIXED"},  # a prefix of both
        {"finding_hash": "abcdef019999", "status": "REFUTED"},  # verifier vocabulary
    ]
    with pytest.raises(ReviewError) as exc:
        _reviewer(_recon_raw(rows), phase="final", prior=prior)
    msg = exc.value.message
    # every problem at once, so one correction turn can fix them all
    assert "hash='abcdef01' status='FIXED': not a prior hash" in msg
    assert "status='REFUTED': unknown status" in msg
    assert "reconciliation missing prior hashes" in msg
    with pytest.raises(ReviewError, match="not a prior hash"):
        _reviewer(
            _recon_raw([{"finding_hash": "abcde", "status": "FIXED"}]),  # under 7 chars
            phase="final",
            prior={"abcdef012345"},
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


def test_finding_priority_severity_alias():
    assert Finding.from_dict({"file": "f", "title": "t", "severity": "p2"}).severity == "suggestion"


def test_lane_argv_includes_budget_cap(tmp_path):
    from reviewsys.lane import LaneSpec, argv_for

    spec = LaneSpec(
        role="general",
        agent="a",
        model="m",
        effort="high",
        prompt="p",
        cwd=tmp_path,
        add_dir=tmp_path,
        timeout_seconds=1,
        claude_bin="claude",
        max_budget_usd=50.0,
    )
    argv = argv_for(spec)
    assert argv[-2:] == ["--max-budget-usd", "50.00"] and "--output-format" in argv
    assert "--max-budget-usd" not in argv_for(
        LaneSpec(
            role="r",
            agent="a",
            model="m",
            effort="low",
            prompt="p",
            cwd=tmp_path,
            add_dir=tmp_path,
            timeout_seconds=1,
            claude_bin="claude",
        )
    )


def test_extract_result_error_envelope_is_readable():
    from reviewsys.lane import LaneResult, _extract_result, lane_output

    env = {
        "type": "result",
        "subtype": "error_max_budget_usd",
        "is_error": True,
        "num_turns": 63,
        "total_cost_usd": 4.0086,
        "usage": {"input_tokens": 100, "cache_read_input_tokens": 900, "output_tokens": 50},
    }
    res = LaneResult(exit_code=0, stdout=json.dumps(env), stderr="", duration_s=1.0)
    _extract_result(res)
    assert (res.tokens_in, res.tokens_out, res.turns, res.cost_usd) == (1000, 50, 63, 4.0086)
    assert res.exit_code == 1
    with pytest.raises(ReviewError, match=r"lane max_budget_usd after 63 turns, \$4.0086"):
        lane_output(res)


def test_quota_errors_in_the_envelope_reach_quota_detection():
    """Claude Code reports an upstream error in the result envelope, not on stderr, and prints
    a `[claude-code:unrecognized_model]` diagnostic on stderr for every proxied model. The
    quota check must see the former and skip the latter (2026-09-24: Astra cooled down
    mid-run and lanes failed as `unrecognized_model` instead of flipping to degraded mode)."""
    import json

    from reviewsys.degraded import looks_like_quota_failure
    from reviewsys.lane import LaneResult, _extract_result, lane_output

    env = {
        "type": "result",
        "subtype": "success",
        "is_error": True,
        "result": "API Error: Request rejected (429) · All credentials for model gpt-6-astra are cooling down",
    }
    res = LaneResult(
        exit_code=1,
        stdout=json.dumps(env),
        stderr='[claude-code:unrecognized_model] {"model":"gpt-6-astra","query_source":"sdk"}\n',
        duration_s=1,
    )
    _extract_result(res)
    assert res.first_stderr_line.startswith("API Error: Request rejected (429)")
    assert looks_like_quota_failure(res.infra_error)
    with pytest.raises(ReviewError) as e:
        lane_output(res)
    assert "cooling down" in e.value.message and "unrecognized_model" not in e.value.message
    # a successful lane's result is model output and is never treated as an API error
    ok = LaneResult(
        exit_code=0,
        stdout=json.dumps({"type": "result", "result": "API Error: x"}),
        stderr="",
        duration_s=1,
    )
    _extract_result(ok)
    assert ok.api_error == "" and ok.infra_error == ""
