"""v10 pure rules: candidate/group/verdict parsing, the severity gate, ranking, the
withdrawn-issue revival diff check, lane plumbing for structured output."""

from __future__ import annotations

import json

import pytest

from reviewsys import v10
from reviewsys.lane import LaneResult, LaneSpec, _extract_result, argv_for, lane_output
from reviewsys.models import FailKind, ReviewError


def _cand(
    i: str, sev: str = "suggestion", lane: str = "phase2:scan", scen: str = "x"
) -> v10.Candidate:
    return v10.Candidate(
        id=i,
        lane=lane,
        file="f.rs",
        line_start=10,
        line_end=12,
        severity=sev,
        category="logic",
        title=f"t{i}",
        failure_scenario=scen,
        body="b",
    )


def test_parse_candidates_caps_normalises_and_rejects_untitled():
    raw = {
        "summary": "s",
        "candidates": [
            {
                "file": "a",
                "severity": "P1",
                "category": "weird",
                "title": "one",
                "failure_scenario": "f",
                "body": "b",
                "line_start": 3,
            }
        ]
        + [
            {
                "file": "a",
                "severity": "nitpick",
                "category": "bug",
                "title": f"n{i}",
                "failure_scenario": "f",
                "body": "b",
            }
            for i in range(10)
        ],
    }
    out = v10.parse_candidates(raw, lane="phase2:scan", prefix="x")
    assert len(out) == v10.MAX_CANDIDATES_PER_LANE
    assert out[0].severity == "blocking"  # P1 alias
    assert out[0].category == "general"  # unknown category folded
    assert out[0].line_end == 3  # defaults to line_start
    with pytest.raises(ReviewError):
        v10.parse_candidates({"candidates": [{"severity": "nitpick"}]}, lane="l", prefix="x")


def test_parse_groups_places_every_candidate_once_and_validates_matches():
    cands = [_cand("a"), _cand("b"), _cand("c"), _cand("d")]
    raw = {
        "groups": [
            {"members": ["a", "b"], "representative": "b", "match": "open:h1"},
            {"members": ["b", "c"], "representative": "zzz", "match": "closed:nope"},
            {"members": ["ghost"], "representative": "ghost", "match": "new"},
        ]
    }
    gs = v10.parse_groups(
        raw, cands, open_hashes={"h1"}, closed_hashes={"h2"}, coderabbit_ids=set()
    )
    ids = sorted(m.id for g in gs for m in g.members)
    assert ids == ["a", "b", "c", "d"]  # every candidate exactly once; "b" not double-placed
    assert gs[0].match == "open:h1" and gs[0].representative.id == "b"
    assert gs[1].match == "new"  # unknown closed hash falls back to new
    assert gs[1].members[0].id == "c"
    assert gs[-1].members[0].id == "d" and gs[-1].match == "new"  # left out -> singleton new


def test_severity_gate_requires_confirmed_for_blocking():
    v = v10.Verdict("PLAUSIBLE", "e", "blocking", "run the race test")
    assert v10.gated_severity(v) == "suggestion"
    assert v10.gated_severity(v10.Verdict("CONFIRMED", "e", "blocking")) == "blocking"
    assert v10.gated_severity(v10.Verdict("PLAUSIBLE", "e", "nitpick")) == "nitpick"


def test_parse_verdict_needs_evidence_and_known_verdict():
    assert (
        v10.parse_verdict(
            {"verdict": "confirmed", "evidence": "line 3", "severity": "blocking"}
        ).verdict
        == "CONFIRMED"
    )
    with pytest.raises(ReviewError):
        v10.parse_verdict({"verdict": "MAYBE", "evidence": "x", "severity": "nitpick"})
    with pytest.raises(ReviewError):
        v10.parse_verdict({"verdict": "REFUTED", "evidence": "", "severity": "nitpick"})


def test_rank_groups_severity_then_lane_agreement():
    g1 = v10.Group([_cand("a", "suggestion")], _cand("a", "suggestion"), "new")
    two = [_cand("b", "suggestion", "phase2:scan"), _cand("c", "suggestion", "phase2:trace")]
    g2 = v10.Group(two, two[0], "new")
    g3 = v10.Group([_cand("d", "blocking")], _cand("d", "blocking"), "new")
    assert [g.representative.id for g in v10.rank_groups([g1, g2, g3])] == ["d", "b", "a"]


def test_thread_decisions_contract():
    ok = v10.parse_thread_decisions(
        {
            "issues": [
                {"finding_hash": "h1", "status": "no_reply", "reply": "ignored"},
                {"finding_hash": "h2", "status": "WITHDRAWN", "reply": "you are right"},
            ]
        },
        expected={"h1", "h2"},
    )
    assert {h: (d.status, d.reply) for h, d in ok.items()} == {
        "h1": ("NO_REPLY", ""),
        "h2": ("WITHDRAWN", "you are right"),
    }
    with pytest.raises(ReviewError):
        v10.parse_thread_decisions({"issues": []}, expected={"h1"})
    # retiring a finding without saying why on its thread is not an outcome
    with pytest.raises(ReviewError, match="without a reply"):
        v10.parse_thread_decisions(
            {"issues": [{"finding_hash": "h1", "status": "FIXED", "reply": ""}]}, expected={"h1"}
        )


DIFF = """diff --git a/f.rs b/f.rs
--- a/f.rs
+++ b/f.rs
@@ -40,2 +40,3 @@ fn x() {
-    a();
+    b();
+    c();
"""


@pytest.mark.parametrize(
    ("start", "end", "changed"),
    [(40, 41, True), (41, 45, True), (10, 20, False), (50, 60, False), (None, None, True)],
)
def test_lines_changed(start, end, changed):
    assert v10.lines_changed(DIFF, start, end) is changed


def test_lines_changed_pure_insertion_and_empty_diff():
    ins = "@@ -9,0 +10,2 @@\n+x\n+y\n"
    assert v10.lines_changed(ins, 10, 10) is True
    assert v10.lines_changed(ins, 30, 31) is False
    assert v10.lines_changed("", 1, 5) is False


def _spec(**kw):
    base = dict(
        role="r",
        agent="a",
        model="m",
        effort="high",
        prompt="p",
        cwd=".",
        add_dir=".",
        timeout_seconds=1,
        claude_bin="claude",
    )
    return LaneSpec(**{**base, **kw})


def test_argv_carries_schema_and_turn_cap():
    argv = argv_for(_spec(json_schema={"type": "object"}, max_turns=30))
    assert argv[argv.index("--json-schema") + 1] == '{"type":"object"}'
    assert argv[argv.index("--max-turns") + 1] == "30"
    assert "--json-schema" not in argv_for(_spec())


def test_structured_output_wins_and_schema_retry_exhaustion_is_contract():
    res = LaneResult(
        exit_code=0,
        stdout=json.dumps(
            {"type": "result", "result": "prose", "structured_output": {"verdict": "CONFIRMED"}}
        ),
        stderr="",
        duration_s=1,
    )
    _extract_result(res)
    assert lane_output(res) == {"verdict": "CONFIRMED"}
    bad = LaneResult(
        exit_code=1,
        stdout=json.dumps(
            {
                "type": "result",
                "is_error": True,
                "subtype": "error_max_structured_output_retries",
                "num_turns": 7,
            }
        ),
        stderr="",
        duration_s=1,
    )
    _extract_result(bad)
    with pytest.raises(ReviewError) as e:
        lane_output(bad)
    assert e.value.kind == FailKind.CONTRACT
