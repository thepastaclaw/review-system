"""The review body's generated lead (what blocks this verdict, from the final findings, never
the verifier's prose) and a finding's `impact` line: parsed tolerantly, rendered under the
title, recovered from a thread's root, and invisible to dedupe."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from reviewsys import audit, ci, dedupe, github, publish, rules, worker
from reviewsys.contract import (
    MAX_IMPACT_CHARS,
    Finding,
    VerifierOutput,
    clean_impact,
    normalize_finding_text,
    parse_verifier_output,
)
from reviewsys.dedupe import ExistingComment, Suppressed, find_duplicate, impact_of

HEAD = "a" * 40
THREAD = "https://github.com/dashpay/platform/pull/1#discussion_r{}"
LOCKED = "A group member with a locked wallet can't see a proposal's amount, recipient or note"
STALE = "Invalidate token confirmations when a cache refresh replaces the signer"
LOCKED_IMPACT = (
    "Group members only: with the wallet locked, the proposal screen shows blanks, so they "
    "vote without seeing what they approve."
)


def finding(title: str, **kw) -> Finding:
    d = {
        "file": "src/ui/tokens/token_action_screen.rs",
        "body": "body",
        "severity": "blocking",
        "category": "bug",
        "line_start": 553,
        "line_end": 574,
    }
    d.update(kw)
    return Finding(title=title, **d)


def verified(findings, *, carried=(), action="REQUEST_CHANGES", summary="Adds group actions."):
    return VerifierOutput(
        summary=summary,
        review_action=action,
        findings=list(findings),
        dropped=[],
        out_of_scope=[],
        coderabbit_reactions=[],
        prerequisite_adjudications=[],
        adjudication_complete=True,
        review_phase="final",
        raw={},
        carried=list(carried),
    )


def model(v, *, kept=(), skipped=(), suppressed=(), phase="final", **kw) -> publish.ReviewModel:
    prov = publish.Provenance(
        reviewers=[],
        verifier={"model": "m", "agent": "a", "role": "final-verifier"},
        policy_fingerprint="f" * 64,
        **kw.pop("prov", {}),
    )
    return publish.ReviewModel(
        repo="dashpay/platform",
        number=1,
        head_sha=HEAD,
        phase=phase,
        verified=v,
        provenance=prov,
        kept=list(kept),
        skipped=list(skipped),
        suppressed=list(suppressed),
        comments=[],
        **kw,
    )


def lead(body: str) -> list[str]:
    lines = body.split("\n")
    start = lines.index(publish.LEAD_MARKER)
    end = lines.index("", start)
    return lines[start + 1 : end]


def counted_blockers(body: str) -> int:
    counts = next(ln for ln in body.split("\n") if publish._COUNTS_LINE_RE.match(ln))
    m = publish._COUNT_SEGMENT_RE.search(counts)
    assert m and m.group(4) == "blocking"
    return int(m.group(3))


def folded(f: Finding, n: int, action="deduped_existing_open_thread") -> Suppressed:
    return Suppressed(f, action, html_url=THREAD.format(n), comment_hash=f.hash)


# ---- the lead, one kind of entry at a time ----


def test_a_new_blocker_posted_inline_leads_with_its_impact():
    new = finding(LOCKED, impact=LOCKED_IMPACT)
    body = publish.render(model(verified([new]), kept=[new]))
    assert lead(body) == [
        "**Changes requested: 1 blocker**",
        f"- **{LOCKED}** — `src/ui/tokens/token_action_screen.rs:553-574` (new)",
        f"  Impact: {LOCKED_IMPACT}",
    ]
    lines = body.split("\n")
    assert lines.index(publish.LEAD_MARKER) < lines.index("Adds group actions.")
    assert counted_blockers(body) == 1


def test_a_new_blocker_not_shown_inline_says_where_it_is():
    new = finding(LOCKED, line_start=900, line_end=900)
    body = publish.render(model(verified([new]), kept=[new], skipped=[new]))
    assert lead(body)[1] == (
        f"- **{LOCKED}** — `src/ui/tokens/token_action_screen.rs:900` "
        "(new; not shown inline, see below)"
    )
    assert "### 1 finding(s) not shown inline" in body


def test_a_carried_blocker_links_its_thread_and_shows_the_impact_recovered_from_it():
    old = finding(STALE, line_start=472, line_end=483, prior_hash="0123456789ab")
    recovered = dataclasses.replace(old, impact="Signers see a stale confirmation.")
    body = publish.render(model(verified([old]), suppressed=[folded(recovered, 7)], rereview=True))
    assert lead(body) == [
        "**Changes requested: 1 blocker**",
        f"- **{STALE}** — `src/ui/tokens/token_action_screen.rs:472-483` "
        f"([thread]({THREAD.format(7)}), carried)",
        "  Impact: Signers see a stale confirmation.",
    ]
    assert "**New on this head:** Adds group actions." in body
    assert "### Still open from earlier rounds (1)" in body


def test_a_v12_carried_blocker_counts_and_is_listed():
    old = finding(STALE, prior_hash="0123456789ab", extra={"thread_url": THREAD.format(3)})
    body = publish.render(model(verified([], carried=[old])))
    assert lead(body)[0] == "**Changes requested: 1 blocker**"
    assert f"([thread]({THREAD.format(3)}), carried)" in lead(body)[1]


def test_a_blocker_awaiting_its_tracking_link_says_what_lifts_it():
    held = finding(STALE, prior_hash="0123456789ab", extra={"awaiting_tracking": True})
    body = publish.render(model(verified([held]), suppressed=[folded(held, 9)]))
    assert lead(body) == [
        "**Changes requested: 1 blocker**",
        f"- **{STALE}** — `src/ui/tokens/token_action_screen.rs:553-574` "
        f"([thread]({THREAD.format(9)}), carried; accepted as a deferral; lifts when a "
        "tracking issue is linked on its thread)",
    ]
    assert "⏸️ **Awaiting tracking link** (1)" in body


def test_entries_are_ordered_new_then_carried_then_awaiting_and_skip_non_blockers():
    new = finding("New one")
    sugg = finding("Just a suggestion", severity="suggestion")
    old = finding("Old one", prior_hash="0123456789ab")
    held = finding("Held one", prior_hash="ba9876543210", extra={"awaiting_tracking": True})
    v = verified([held, old, new, sugg])
    m = model(v, kept=[new, sugg], suppressed=[folded(held, 2), folded(old, 1)])
    assert [(f.title, how) for f, how in publish.lead_entries(m)] == [
        ("New one", "new"),
        ("Old one", "carried"),
        ("Held one", "awaiting"),
    ]
    body = publish.render(m)
    assert lead(body)[0] == "**Changes requested: 3 blockers**" and counted_blockers(body) == 3
    assert "Just a suggestion" not in "\n".join(lead(body))


# ---- count parity, disputes, nothing blocking, the preliminary gate ----


def test_a_blocker_collapsed_under_a_root_id_suggestion_still_leads(monkeypatch):
    """A suggestion carrying a root id and a blocker on the same lines (sharing a carry-forward
    token) collapse into one comment: it is posted as the blocker, keeping the root id, and the
    lead lists every blocker the verdict counts."""
    monkeypatch.setattr(publish.github, "inline_comments", lambda *a: [])
    sugg = finding("Cache the lookup (prior-7)", severity="suggestion", root_id="r-1")
    blocker = finding("Lookup races the refresh, see prior-7", body="a longer body here")
    second = finding("Lookup races the refresh again, see prior-7")
    v = verified([sugg, blocker, second])
    m = publish.build(
        None,
        repo="dashpay/platform",
        number=1,
        head_sha=HEAD,
        phase="final",
        verified=v,
        provenance=model(v).provenance,
        bot_login="thepastaclaw",
        diff_text="",
        dry_run=True,
    )
    (posted,) = m.kept
    assert (posted.title, posted.severity, posted.root_id) == (blocker.title, "blocking", "r-1")
    assert publish.lead_count_mismatch(m) is None
    body = publish.render(m)
    assert m.canonical_event == "REQUEST_CHANGES" and counted_blockers(body) == 2
    assert lead(body) == [
        "**Changes requested: 2 blockers**",
        f"- **{blocker.title}** — `src/ui/tokens/token_action_screen.rs:553-574` "
        "(new; not shown inline, see below)",
        f"- **{second.title}** — `src/ui/tokens/token_action_screen.rs:553-574` "
        "(new; same root as another finding here, posted in its comment)",
    ]


def test_a_blocker_whose_comment_went_to_a_lesser_finding_still_has_an_entry():
    sugg, blocker = finding("Cache it", severity="suggestion"), finding("Races the refresh")
    m = model(
        verified([sugg, blocker]),
        kept=[sugg],
        suppressed=[Suppressed(blocker, "deduped_same_batch_root", detail=sugg.title)],
    )
    assert lead(publish.render(m))[:2] == [
        "**Changes requested: 1 blocker**",
        f"- **{blocker.title}** — `src/ui/tokens/token_action_screen.rs:553-574` "
        "(new; same root as another finding here, posted in its comment)",
    ]


def test_a_list_that_disagrees_with_the_verdict_count_is_not_numbered():
    """A guard: a blocker the verdict counts that no final set holds."""
    shown, lost = finding("Shown"), finding("Lost")
    m = model(verified([shown, lost]), kept=[shown])
    assert publish.lead_count_mismatch(m) == (
        "the lead lists 1 blocking finding(s), the verdict counts 2"
    )
    body = publish.render(m)
    assert lead(body)[0] == publish.LEAD_UNCOUNTED
    assert not any(publish._LEAD_HEADING_RE.match(ln) for ln in body.split("\n"))
    assert counted_blockers(body) == 2


def test_the_publish_step_records_an_unnumbered_lead(cfg, conn, monkeypatch, tmp_path):
    shown, lost = finding("Shown"), finding("Lost")
    m = model(verified([shown, lost]), kept=[shown])
    monkeypatch.setattr(worker, "_build_review", lambda ctx, **kw: m)
    monkeypatch.setattr(
        worker.publish,
        "publish",
        lambda *a, **kw: publish.PublishResult(False, "REQUEST_CHANGES", "COMMENT", "b"),
    )
    monkeypatch.setattr(worker, "_record_publication", lambda *a: None)
    monkeypatch.setattr(worker, "_record_lifted_reconciliations", lambda *a: None)
    ctx = SimpleNamespace(
        conn=conn,
        repo="dashpay/platform",
        number=1,
        run_id=None,
        has_prior_review=False,
        run_dir=tmp_path,
        dry_run=True,
        gh=None,
        cfg=cfg,
        kept_verdict=None,
    )
    worker._publish_review(
        ctx,
        phase="final",
        verified=m.verified,
        verifier_lm=None,
        phase2_skipped=None,
        rereview=True,
    )
    (row,) = conn.execute("SELECT detail FROM events WHERE kind='review.lead_uncounted'")
    assert row["detail"] == "final: the lead lists 1 blocking finding(s), the verdict counts 2"


def test_disputed_blockers_are_counted_out_in_one_line():
    new = finding(LOCKED)
    gone = finding("Argued away", extra={"thread_url": THREAD.format(4)})
    nit = finding("Argued nit", severity="nitpick")
    body = publish.render(model(verified([new]), kept=[new], disputed=[gone, nit]))
    assert lead(body) == [
        "**Changes requested: 1 blocker**",
        f"- **{LOCKED}** — `src/ui/tokens/token_action_screen.rs:553-574` (new)",
        "Not counted: 1 disputed blocking finding(s) left to maintainers.",
    ]
    assert "Argued away" not in "\n".join(lead(body))


def test_blockers_disputed_on_an_earlier_head_explain_a_withheld_approval():
    """The head after a dispute settled: nothing new is disputed, but the approval is still
    withheld on it, so the lead says why the verdict is not an approval."""
    sugg = finding("Cache the lookup", severity="suggestion")
    m = model(verified([sugg], action="COMMENT"), kept=[sugg], disputed_blockers=2)
    assert lead(publish.render(m)) == [
        "**Nothing blocks:** no blocking finding stands on this head.",
        "Not counted: 2 disputed blocking finding(s) left to maintainers.",
    ]


def test_a_comment_with_no_blocker_says_nothing_blocks():
    sugg = finding("Cache the lookup", severity="suggestion")
    body = publish.render(model(verified([sugg], action="COMMENT"), kept=[sugg]))
    assert lead(body) == ["**Nothing blocks:** no blocking finding stands on this head."]
    assert "Changes requested" not in body


def test_an_approval_has_no_lead_and_keeps_its_approved_line():
    sugg = finding("Cache the lookup", severity="suggestion")
    m = model(verified([sugg], action="APPROVE"), kept=[sugg], fix_forward=True)
    body = publish.render(m)
    assert publish.LEAD_MARKER not in body
    assert "**Approved:** nothing here blocks the merge." in body


def test_a_preliminary_gate_review_leads_with_its_blockers():
    new = finding(LOCKED)
    body = publish.render(model(verified([new]), kept=[new], phase="preliminary"))
    assert lead(body)[0] == "**Changes requested: 1 blocker**"
    lines = body.split("\n")
    assert lines.index(publish.LEAD_MARKER) < next(
        i for i, ln in enumerate(lines) if ln.startswith("Validated blockers were found")
    )
    assert counted_blockers(body) == 1


def test_a_preliminary_points_gate_with_no_blocker_says_nothing_blocks():
    sugg = finding("Cache the lookup", severity="suggestion")
    m = model(
        verified([sugg], action="COMMENT"),
        kept=[sugg],
        phase="preliminary",
        prov={"phase2_skipped": "deferred by the points gate"},
    )
    body = publish.render(m)
    assert lead(body) == ["**Nothing blocks:** no blocking finding stands on this head."]
    assert "above the gate budget" in body


def test_a_degraded_review_leads_after_its_banner():
    new = finding(LOCKED)
    m = model(verified([new]), kept=[new], prov={"degraded": {"reason": "quota"}})
    lines = publish.render(m).split("\n")
    banner = next(i for i, ln in enumerate(lines) if ln.startswith("> **"))
    assert banner < lines.index(publish.LEAD_MARKER)


# ---- the real case: prose that names a follow-up while two blockers stand ----


def test_prose_naming_only_a_follow_up_still_leads_with_the_two_blockers():
    """dash-evo-tool: the verifier's summary blamed a follow-up candidate; a new blocker
    outside the diff and one carried from round 1 were the actual blockers."""
    prose = (
        "The proposal flow is mostly right. The remaining problem is the missing debounce "
        "on the refresh button, which should be handled before merge."
    )
    new = finding(LOCKED, line_start=553, line_end=574, impact=LOCKED_IMPACT)
    carried = finding(STALE, line_start=472, line_end=483, prior_hash="0123456789ab")
    followup = finding("Refresh button has no debounce", severity="suggestion")
    v = verified([new, carried], summary=prose)
    v.followups = [followup]
    m = model(v, kept=[new], skipped=[new], suppressed=[folded(carried, 11)], rereview=True)
    body = publish.render(m)
    assert lead(body) == [
        "**Changes requested: 2 blockers**",
        f"- **{LOCKED}** — `src/ui/tokens/token_action_screen.rs:553-574` "
        "(new; not shown inline, see below)",
        f"  Impact: {LOCKED_IMPACT}",
        f"- **{STALE}** — `src/ui/tokens/token_action_screen.rs:472-483` "
        f"([thread]({THREAD.format(11)}), carried)",
    ]
    assert "debounce" not in "\n".join(lead(body))
    assert f"**New on this head:** {prose}" in body
    assert counted_blockers(body) == 2
    assert m.canonical_event == "REQUEST_CHANGES"


# ---- edits and readers of a published body ----


def lifted(f: Finding, status="WITHDRAWN") -> publish.LiftedFinding:
    return publish.LiftedFinding(
        hash=f.hash, severity=f.severity, file=f.file, title=f.title, status=status
    )


def test_a_lifted_blocker_is_struck_in_the_lead_and_the_heading_recounted():
    new = finding(LOCKED, impact=LOCKED_IMPACT)
    old = finding(STALE, line_start=472, line_end=483, prior_hash="0123456789ab")
    body = publish.render(
        model(verified([new, old]), kept=[new], suppressed=[folded(old, 5)], rereview=True)
    )
    edited = publish.with_lifted_findings(body, [lifted(old)])
    assert lead(edited) == [
        "**Changes requested: ~~2~~ 1 blocker**",
        f"- **{LOCKED}** — `src/ui/tokens/token_action_screen.rs:553-574` (new)",
        f"  Impact: {LOCKED_IMPACT}",
        f"- ~~**{STALE}** — `src/ui/tokens/token_action_screen.rs:472-483` "
        f"([thread]({THREAD.format(5)}), carried)~~ — **settled:** ✅ Withdrawn",
    ]
    assert publish.with_lifted_findings(edited, [lifted(old)]) == edited
    assert counted_blockers(edited) == 1
    assert "1 blocking finding(s) still stand" in edited
    both = publish.with_lifted_findings(edited, [lifted(old), lifted(new, "DISPUTED")])
    assert lead(both)[0] == "**Changes requested: ~~2~~ 0 blockers: none remains**"
    assert lead(both)[1].startswith(f"- ~~**{LOCKED}**")
    assert lead(both)[1].endswith("~~ — **settled:** ⚖️ Disputed (left to the maintainers)")
    assert lead(both)[2] == f"  ~~Impact: {LOCKED_IMPACT}~~"
    assert publish.with_lifted_findings(both, [lifted(old), lifted(new, "DISPUTED")]) == both
    assert counted_blockers(both) == 0
    assert not audit.BLOCKING_COUNT_RE.search(both)
    assert publish.with_lifted_findings(both, []) == body


def test_the_ci_lane_reads_the_lead_without_its_marker():
    new = finding(LOCKED)
    body = publish.render(model(verified([new]), kept=[new]))
    text = ci.review_summary(body)
    assert text.startswith("## Final validation")
    assert "**Changes requested: 1 blocker**" in text and "thepastaclaw-review-lead" not in text


def test_a_verdict_update_lists_what_still_blocks():
    still = finding(STALE, extra={"thread_url": THREAD.format(6)}, impact="Signers see stale data.")
    body = publish.render_verdict_update(
        head_sha=HEAD,
        phase="final",
        verified=verified([still]),
        provenance=publish.Provenance(
            reviewers=[],
            verifier={"model": "m", "agent": "a", "role": "final-verifier"},
            policy_fingerprint="f" * 64,
        ),
        previous_event="CHANGES_REQUESTED",
        new_event="REQUEST_CHANGES",
        lifted_blockers=["Gone now"],
    )
    assert (
        "1 blocking finding(s) now stand.\n\n🔴 1 blocking | 🟡 0 suggestion(s) | 💬 0 "
        "nitpick(s)\n\nStill blocking:\n"
        f"- **{STALE}** — `src/ui/tokens/token_action_screen.rs:553-574` "
        f"([thread]({THREAD.format(6)}))\n  Impact: Signers see stale data.\n\n"
        "No longer blocking:\n- Gone now"
    ) in body


# ---- impact: parsed, rendered, recovered ----


def test_impact_is_parsed_and_kept_out_of_identity():
    raw = {"file": "f.rs", "title": "T", "body": "B", "severity": "blocking"}
    with_impact = Finding.from_dict({**raw, "impact": "  Devnet only:\n wipes   race. "})
    plain = Finding.from_dict(raw)
    assert with_impact.impact == "Devnet only: wipes race."
    assert "impact" not in with_impact.extra
    assert with_impact.to_dict()["impact"] == "Devnet only: wipes race."
    assert "impact" not in plain.to_dict() and plain.impact is None
    assert (with_impact.hash, with_impact.dedupe_key) == (plain.hash, plain.dedupe_key)
    assert Finding.from_dict(with_impact.to_dict()).impact == with_impact.impact


def test_a_long_impact_is_cut_at_a_word_boundary():
    words = " ".join(f"word{i}" for i in range(200))
    cut = clean_impact(words)
    assert cut and len(cut) <= MAX_IMPACT_CHARS and cut.endswith("…")
    assert words.startswith(cut[:-1]) and words[len(cut) - 1] == " "
    assert clean_impact("x" * 1000) == "x" * (MAX_IMPACT_CHARS - 1) + "…"
    exact = "y" * MAX_IMPACT_CHARS
    assert clean_impact(exact) == exact
    long = {"file": "f.rs", "title": "T", "body": "B", "impact": words}
    assert Finding.from_dict(long).impact == cut


@pytest.mark.parametrize("impact", [None, "", 42, ["x"], {"a": 1}])
def test_a_missing_or_malformed_impact_is_no_contract_error(impact):
    f = {"file": "f.rs", "title": "T", "body": "B", "severity": "blocking"}
    if impact is not None:
        f["impact"] = impact
    out = parse_verifier_output(
        {
            "summary": "s",
            "review_action": "REQUEST_CHANGES",
            "findings": [f],
            "coderabbit_reactions": [],
            "prerequisite_adjudications": [],
            "adjudication_complete": True,
        },
        expected_phase="final",
        expected_coderabbit_ids=[],
    )
    assert out.findings[0].impact is None and "impact" not in out.findings[0].extra


def test_the_impact_line_sits_under_the_title_everywhere():
    f = finding(LOCKED, impact=LOCKED_IMPACT, source="phase2")
    comment = publish.comment_body(f).split("\n")
    assert comment[1].endswith(f"{LOCKED}**") and comment[2] == f"**Impact:** {LOCKED_IMPACT}"
    block = publish._finding_block(f).split("\n")
    assert block[1] == f"**Impact:** {LOCKED_IMPACT}" and block[2].startswith("`src/")
    prompt = publish._ai_prompt([f], [])
    assert prompt and f"{LOCKED}\n  Impact: {LOCKED_IMPACT}\n  body" in prompt
    assert "Impact" not in publish.comment_body(dataclasses.replace(f, impact=None))


def test_a_posted_comment_reads_back_the_same_with_or_without_its_impact_line():
    f = finding(LOCKED, impact=LOCKED_IMPACT, body="The amount, recipient and note are blank.")
    posted = publish.comment_body(f)
    bare = publish.comment_body(dataclasses.replace(f, impact=None))
    status = publish.with_thread_status(posted, status="DISPUTED", head_sha=HEAD)
    for body in (posted, status):
        assert github._finding_title(body) == ("blocking", LOCKED)
        assert impact_of(body) == LOCKED_IMPACT
    assert impact_of(bare) is None

    def read(body: str) -> ExistingComment:
        return ExistingComment.from_github(
            {"id": 1, "path": f.file, "line": f.line_end, "html_url": "u", "body": body}
        )

    a, b, c = read(posted), read(bare), read(status)
    assert a.impact == c.impact == LOCKED_IMPACT and b.impact is None
    assert a.norm_title == b.norm_title == c.norm_title
    assert a.norm_body == b.norm_body
    assert (a.finding_hash, a.dedupe_key) == (f.hash, f.dedupe_key)


BODY = " ".join(f"detail{i}" for i in range(40))
LONG_IMPACT = " ".join(f"consequence{i}" for i in range(40))


@pytest.mark.parametrize("posted_with_impact", [True, False])
@pytest.mark.parametrize("raised_with_impact", [True, False])
def test_a_reworded_re_raise_still_dedupes_onto_its_thread(posted_with_impact, raised_with_impact):
    """Retitled, so no hash, dedupe-key or title match: only the same-line body comparison
    carries it onto its thread, and an impact line on either side must not tip that."""
    first = finding(LOCKED, body=BODY, impact=LONG_IMPACT if posted_with_impact else None)
    again = finding(
        "Voters approve proposals blind",
        body=BODY,
        impact="Anyone voting with a locked wallet approves blind." if raised_with_impact else None,
    )
    posted = publish.comment_body(first).replace(first.dedupe_key, "0" * 16)
    rec = ExistingComment.from_github(
        {"id": 1, "path": first.file, "line": first.line_end, "html_url": "u", "body": posted}
    )
    assert find_duplicate(again, [rec]) == (rec, "same_line_similar_text")
    if posted_with_impact:  # read with its impact line, the comment would no longer match
        unstripped = normalize_finding_text(posted)
        assert dedupe._jaccard(normalize_finding_text(BODY), unstripped) < 0.5


def test_dedupe_recovers_the_impact_of_a_carried_finding_from_its_thread(monkeypatch):
    posted = finding(STALE, impact="Signers see a stale confirmation.")
    root = {
        "id": 100,
        "node_id": "PRRC_100",
        "path": posted.file,
        "line": posted.line_end,
        "html_url": THREAD.format(100),
        "user": {"login": "thepastaclaw"},
        "body": publish.comment_body(posted),
    }
    monkeypatch.setattr(publish.github, "inline_comments", lambda *a: [root])
    monkeypatch.setattr(
        publish.github, "thread_for_comment", lambda *a: {"isResolved": False, "comments": {}}
    )
    carried = dataclasses.replace(posted, impact=None, prior_hash=posted.hash)
    refreshed = dataclasses.replace(carried, impact="Refreshed by the verifier.")
    kept, sup = publish.dedupe_against_github(
        None, "dashpay/platform", 1, HEAD, [carried], "thepastaclaw", dry_run=True
    )
    assert not kept and sup[0].finding.impact == "Signers see a stale confirmation."
    body = publish.render(model(verified([carried]), suppressed=sup))
    assert "  Impact: Signers see a stale confirmation." in lead(body)
    _, sup = publish.dedupe_against_github(
        None, "dashpay/platform", 1, HEAD, [refreshed], "thepastaclaw", dry_run=True
    )
    assert sup[0].finding.impact == "Refreshed by the verifier."


def test_the_worker_reads_a_prior_finding_s_impact_from_its_thread_root():
    posted = finding(STALE, impact="Signers see a stale confirmation.")
    t = {"body": publish.comment_body(posted)}
    ctx = SimpleNamespace(open_threads={}, thread_index={posted.hash: t})
    assert worker._thread_impact(ctx, posted.hash) == "Signers see a stale confirmation."
    assert worker._thread_impact(ctx, "0" * 12) is None


def test_v12_lanes_carry_impact_from_candidate_to_published_finding():
    out = rules.parse_finder_output(
        {
            "summary": "s",
            "candidates": [
                {
                    "file": "f.rs",
                    "line_start": 3,
                    "severity": "blocking",
                    "category": "bug",
                    "title": "T",
                    "impact": "Who, when, what.",
                    "failure_scenario": "x",
                    "body": "b",
                },
                {
                    "file": "f.rs",
                    "line_start": 9,
                    "severity": "blocking",
                    "category": "bug",
                    "title": "U",
                    "failure_scenario": "x",
                    "body": "b",
                },
            ],
        },
        id_prefix="p2-general",
        source="phase2:general",
        expected_phase="final",
        head_sha=HEAD,
    )
    one, two = out.findings
    assert one.impact == "Who, when, what." and two.impact is None
    assert rules.brief(one)["impact"] == "Who, when, what." and "impact" not in rules.brief(two)
    ans = rules.parse_verify(
        {
            "summary": "s",
            "groups": [
                {
                    "members": ["p2-general-1"],
                    "match": "new",
                    "verdict": "CONFIRMED",
                    "evidence": "line 3",
                    "severity": "blocking",
                    "title": "T",
                    "body": "b",
                },
                {
                    "members": ["p2-general-2"],
                    "match": "new",
                    "verdict": "CONFIRMED",
                    "evidence": "line 9",
                    "severity": "blocking",
                    "title": "U",
                    "impact": "The verifier's own.",
                    "body": "b",
                },
            ],
        },
        candidate_ids=["p2-general-1", "p2-general-2"],
        issues={},
    )
    applied = rules.apply_rules(
        ans,
        {f.candidate_id or "": f for f in out.findings},
        {},
        changed_since_closed=lambda i: None,
        source_of=lambda ms: "phase2",
    )
    assert [f.impact for f in applied.new] == ["Who, when, what.", "The verifier's own."]
