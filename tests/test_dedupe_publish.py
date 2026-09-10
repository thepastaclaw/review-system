from reviewsys.contract import Finding, VerifierOutput
from reviewsys.dedupe import (
    ExistingComment,
    collapse_same_root,
    find_duplicate,
    has_same_root_duplicates,
    thread_has_resolution_reply,
)
from reviewsys.publish import Provenance, ReviewModel, comment_body, map_comment, parse_diff, render


def f(title: str, **kw) -> Finding:
    d = {
        "file": "src/a.rs",
        "body": "body",
        "severity": "suggestion",
        "category": "bug",
        "line_start": 10,
        "line_end": 12,
    }
    d.update(kw)
    return Finding(title=title, **d)


def test_collapse_same_root_by_carry_forward_token():
    a = f("Null header (prior-2)", severity="nitpick")
    b = f(
        "Header can be null, see prior-2 discussion", severity="blocking", body="longer body here"
    )
    kept, collapsed = collapse_same_root([a, b])
    assert [k.title for k in kept] == [b.title]  # loudest wins
    assert collapsed[0].action == "deduped_same_batch_root"
    assert not has_same_root_duplicates(kept)


def test_distinct_findings_on_same_line_stay_distinct():
    kept, collapsed = collapse_same_root([f("Missing bounds check"), f("Off-by-one in loop")])
    assert len(kept) == 2 and not collapsed


def test_find_duplicate_by_hash_then_fuzzy():
    x = f("Fee estimation omits drainage costs")
    rec = ExistingComment.from_github(
        {
            "id": 1,
            "node_id": "n",
            "path": "src/a.rs",
            "line": 12,
            "html_url": "u",
            "body": f"<!-- thepastaclaw-review v1 finding={x.hash} dedupe=zz -->\n**🟡 Suggestion: {x.title}**\n\nbody",
        }
    )
    assert find_duplicate(x, [rec])[1] == "finding_hash"
    y = f("Fee estimation omits the drainage costs entirely")
    rec2 = ExistingComment.from_github(
        {
            "id": 2,
            "node_id": "n",
            "path": "src/a.rs",
            "line": 12,
            "body": "**🟡 Suggestion: Fee estimation omits drainage costs**\n\nbody",
        }
    )
    assert find_duplicate(y, [rec2])[1] == "same_line_similar_text"
    z = f("Completely unrelated thing about docs")
    assert find_duplicate(z, [rec, rec2]) == (None, None)


def test_resolution_reply_detection():
    t = {
        "comments": {
            "nodes": [{"author": {"login": "QuantumExplorer"}, "body": "Fixed in 614a123."}]
        }
    }
    assert thread_has_resolution_reply(t, "thepastaclaw")
    neg = {
        "comments": {"nodes": [{"author": {"login": "q"}, "body": "This isn't fixed in code yet."}]}
    }
    assert not thread_has_resolution_reply(neg, "thepastaclaw")
    own = {"comments": {"nodes": [{"author": {"login": "thepastaclaw"}, "body": "Fixed in abc."}]}}
    assert not thread_has_resolution_reply(own, "thepastaclaw")


DIFF = """diff --git a/src/a.rs b/src/a.rs
index 1..2 100644
--- a/src/a.rs
+++ b/src/a.rs
@@ -5,4 +5,6 @@
 a
+b
+c
 d
@@ -40,2 +42,3 @@
 x
+y
diff --git a/old.rs b/new.rs
rename from old.rs
rename to new.rs
--- a/old.rs
+++ b/new.rs
@@ -1 +1,2 @@
+z
"""


def test_parse_diff_and_map_comment():
    parsed = parse_diff(DIFF)
    assert parsed["src/a.rs"][0] == {"new_start": 5, "new_count": 6, "new_end": 10}
    assert parsed["old.rs"] is parsed["new.rs"]
    c = map_comment(f("T", line_start=6, line_end=8), parsed)
    assert c and c["line"] == 8 and c["start_line"] == 6 and c["path"] == "src/a.rs"
    assert map_comment(f("T", line_start=20, line_end=20), parsed) is None
    clamped = map_comment(f("T", line_start=1, line_end=8), parsed)
    assert clamped and clamped["start_line"] == 5


def test_comment_body_markers_are_legacy_compatible():
    x = f("Title here", suggestion="let x = 1;", source="phase1:general")
    body = comment_body(x)
    assert body.startswith(
        f"<!-- thepastaclaw-review v1 finding={x.hash} dedupe={x.dedupe_key} -->"
    )
    assert "**🟡 Suggestion: Title here**" in body and "```suggestion\nlet x = 1;\n```" in body
    assert "<sub>source: phase1:general</sub>" in body


def _model(phase: str, findings: list[Finding]) -> ReviewModel:
    v = VerifierOutput(
        summary="Solid PR.\nSource: bogus line that must be stripped\nOne concern remains.",
        review_action="COMMENT",
        findings=findings,
        dropped=[],
        out_of_scope=[{"title": "Refactor X", "body": "pre-existing"}],
        coderabbit_reactions=[],
        prerequisite_adjudications=[],
        adjudication_complete=True,
        review_phase=phase,
        raw={},
    )
    prov = Provenance(
        reviewers=[
            {
                "model": "glm-5.3-flash",
                "agent": "phase1-reviewer",
                "role": "general",
                "status": "completed",
                "phase": "phase1",
            }
        ],
        verifier={"model": "gpt-5.6-sol", "agent": "sol-verifier", "role": "verifier"},
        policy_fingerprint="3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f",
    )
    return ReviewModel(
        repo="dashpay/platform",
        number=1,
        head_sha="a" * 40,
        phase=phase,
        verified=v,
        provenance=prov,
        kept=findings,
        skipped=[],
        suppressed=[],
        comments=[],
    )


def test_render_preliminary_body_golden():
    m = _model("preliminary", [f("Fee estimation omits drainage costs", severity="blocking")])
    body = render(m)
    assert body.splitlines()[0] == "<!-- thepastaclaw-review v1 -->"
    assert (
        body.splitlines()[1]
        == "<!-- thepastaclaw-review-phase v1 phase=preliminary sha="
        + "a" * 40
        + " policy=3f3f3f3f3f3f3f3f -->"
    )
    assert "## Preliminary review — Phase 1 blocker gate" in body
    assert body.count("Source: ") == 1, "exactly one Source line"
    assert "bogus line" not in body
    assert (
        "Source: reviewer 1: `glm-5.3-flash` (agent: `phase1-reviewer`, role: `general`); final verifier: `gpt-5.6-sol` (agent: `sol-verifier`, role: `verifier`)"
        in body
    )
    assert "🔴 1 blocking" in body
    assert "Phase 2 reviewers: **not run (deferred by blocker gate)**" in body
    assert "🤖 Prompt for all review comments with AI agents" in body
    assert "- [BLOCKING] src/a.rs:10-12: Fee estimation omits drainage costs" in body
    assert "Out-of-scope follow-up suggestions (1)" in body
    for word in ("Codex", "Sonnet", "Opus"):
        assert word not in body


def test_render_final_no_findings_is_approve_when_requested():
    m = _model("final", [])
    m.verified.review_action = "APPROVE"
    assert m.canonical_event == "APPROVE"
    body = render(m)
    assert "## Final validation — Phase 1 + Phase 2" in body
    assert "\U0001f534" not in body and "1 blocking" not in body


def _lane(phase, role, model, agent):
    return {"phase": phase, "role": role, "model": model, "agent": agent, "status": "completed"}


LANES = [
    _lane("phase1", "general", "glm-5.3-flash", "phase1-reviewer"),
    _lane("phase1", "security-auditor", "glm-5.3-flash", "phase1-reviewer"),
    _lane("phase2", "general", "gpt-6-astra", "phase2-reviewer"),
    _lane("phase2", "security-auditor", "gpt-6-astra", "phase2-reviewer"),
]


def _verified(*findings):
    return VerifierOutput(
        summary="s",
        review_action="COMMENT",
        findings=list(findings),
        dropped=[],
        out_of_scope=[],
        coderabbit_reactions=[],
        prerequisite_adjudications=[],
        adjudication_complete=True,
        review_phase="final",
        raw={},
    )


def test_attribute_sources_names_the_lane_that_raised_the_finding():
    from reviewsys.publish import attribute_sources

    a = f("Race in flush", source="['claude']")
    b = f("Missing bounds check", source="['claude', 'codex']")
    c = f("Retitled by verifier", source="['codex']")
    d = f("CodeRabbit agreed", source="['codex', 'coderabbit']")
    e = f("No label at all", source="unknown")
    lanes = {
        "phase1:general": [f("Missing bounds check"), f("CodeRabbit agreed"), f("No label at all")],
        "phase1:security-auditor": [f("Missing bounds check")],
        "phase2:general": [f("Race in flush")],
        "phase2:security-auditor": [f("Missing bounds check")],
    }
    attribute_sources(_verified(a, b, c, d, e), LANES, lanes)
    assert a.source == "`gpt-6-astra` (phase2-reviewer: general)"
    assert b.source == (
        "`glm-5.3-flash` (phase1-reviewer: general, security-auditor); "
        "`gpt-6-astra` (phase2-reviewer: security-auditor)"
    )
    # verifier retitled it: no hash match, so every lane of that phase is named
    assert c.source == "`glm-5.3-flash` (phase1-reviewer: general, security-auditor)"
    assert d.source == "`glm-5.3-flash` (phase1-reviewer: general); coderabbit"
    # no usable label: attributed by hash alone
    assert e.source == "`glm-5.3-flash` (phase1-reviewer: general)"


def test_attribute_sources_ignores_phases_that_did_not_run():
    from reviewsys.publish import attribute_sources

    x = f("Only phase 2 ran", source="['codex']")
    y = f("Nothing known", source="unknown")
    p2 = [r for r in LANES if r["phase"] == "phase2"]
    attribute_sources(_verified(x, y), p2, {"phase2:general": []})
    # `codex` names Phase 1, which never ran (backlog mode): nothing to name, label kept
    assert x.source == "['codex']"
    # nothing at all: every lane that ran
    assert y.source == "`gpt-6-astra` (phase2-reviewer: general, security-auditor)"
    assert (
        "<sub>source: `gpt-6-astra` (phase2-reviewer: general, security-auditor)</sub>"
        in comment_body(y)
    )
