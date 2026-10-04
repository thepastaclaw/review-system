"""The ledger write-through under policy v11: what real runs record in it, and that it changes
nothing a run does (the e2e scenarios pass, assertions and all, with the ledger off and with
every ledger write failing)."""

from __future__ import annotations

import pytest
import test_worker_e2e as e2e

from reviewsys import ledger, worker

fake_git = e2e.fake_git  # the e2e module's autouse fixture, for the scenarios run here

REPO, PR = "dashpay/platform", 1

# end-to-end scenarios covering every write-through hook: a fresh publication, the full
# path's thread answers (replied, unreplied, legacy), same-sha re-reviews, cross-round dedupe,
# and conversation concessions and deferrals
SCENARIOS = [
    e2e.test_two_phase_final_review_posts_once,
    e2e.test_cross_round_dedupe_posts_visible_summary,
    e2e.test_already_published_for_sha_posts_rereview_summary,
    e2e.test_reply_on_prior_finding_reaches_reviewer_and_is_answered,
    e2e.test_reply_on_legacy_finding_without_db_row_is_adjudicated,
    e2e.test_same_sha_rereview_withdraws_blocker_and_updates_verdict,
    e2e.test_conversation_withdrawing_the_only_finding_approves,
    e2e.test_conversation_approves_only_once_no_suggestion_is_left,
    e2e.test_conversation_accepting_a_deferral_lifts_request_changes,
    e2e.test_conversation_concession_is_remembered_across_runs,
]


def _issues(conn):
    return ledger.load(conn, REPO, PR)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda f: f.__name__.removeprefix("test_"))
def test_v11_outputs_are_the_same_with_the_ledger_off(scenario, cfg, conn, gh, lanes, monkeypatch):
    monkeypatch.setattr(worker, "_ledger_write", lambda ctx, what, fn: None)
    scenario(cfg, conn, gh, lanes)
    assert conn.execute("SELECT count(*) FROM issue_events").fetchone()[0] == 0


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda f: f.__name__.removeprefix("test_"))
def test_a_failing_ledger_never_fails_a_run(scenario, cfg, conn, gh, lanes, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("ledger bug")

    monkeypatch.setattr(ledger, "append", boom)
    scenario(cfg, conn, gh, lanes)
    assert conn.execute("SELECT count(*) FROM issues").fetchone()[0] == 0
    failed = conn.execute(
        "SELECT detail FROM events WHERE kind='ledger.write_failed' AND repo=? AND number=?",
        (REPO, PR),
    ).fetchall()
    assert failed and all("RuntimeError: ledger bug" in r["detail"] for r in failed)
    assert conn.execute("SELECT count(*) FROM runs WHERE status<>'done'").fetchone()[0] == 0


def test_a_publication_opens_its_issues(cfg, conn, gh, lanes):
    e2e.test_two_phase_final_review_posts_once(cfg, conn, gh, lanes)
    (rid,) = [r[0] for r in conn.execute("SELECT id FROM runs")]
    (issue,) = _issues(conn).values()
    assert issue.hash == conn.execute("SELECT hash FROM posted_findings").fetchone()[0]
    assert (issue.status, issue.status_source, issue.severity) == ("open", "verifier", "suggestion")
    assert (issue.opened_run, issue.opened_sha, issue.anchor_sha) == (rid, e2e.HEAD, e2e.HEAD)
    assert (issue.file, issue.line_start, issue.line_end) == ("f.rs", 12, 12)
    assert (
        conn.execute("SELECT count(*) FROM events WHERE kind='ledger.write_failed'").fetchone()[0]
        == 0
    )


def test_a_deduped_publication_carries_the_existing_issue(cfg, conn, gh, lanes):
    e2e.test_cross_round_dedupe_posts_visible_summary(cfg, conn, gh, lanes)
    # every finding deduped onto an existing comment: nothing new was posted, so nothing opens
    assert _issues(conn) == {}


def test_a_full_path_answer_closes_the_issue_and_backfill_completes_it(cfg, conn, gh, lanes):
    e2e.test_reply_on_prior_finding_reaches_reviewer_and_is_answered(cfg, conn, gh, lanes)
    (issue,) = _issues(conn).values()
    # posted before the write-through: known from its open thread, then answered
    assert (issue.status, issue.status_source, issue.closed_sha) == (
        "withdrawn",
        "verifier",
        e2e.HEAD,
    )
    assert issue.opened_run is None and issue.thread_id == "PRRT_1" and issue.comment_id == 900
    # the backfill adds the publication it never saw, ahead of the answer in time
    seeded = conn.execute("SELECT run_id FROM findings WHERE stage='posted'").fetchone()[0]
    ledger.backfill(conn)
    issue = ledger.get(conn, REPO, PR, issue.hash)
    assert issue is not None
    assert (issue.status, issue.opened_run, issue.opened_sha) == ("withdrawn", seeded, "b" * 40)
    assert ledger.backfill(conn)["events_added"] == 0


def test_same_sha_rereview_withdrawal_and_still_valid(cfg, conn, gh, lanes):
    e2e.test_same_sha_rereview_withdraws_blocker_and_updates_verdict(cfg, conn, gh, lanes)
    by_sev = {i.severity: i for i in _issues(conn).values()}
    assert by_sev["blocking"].status == "withdrawn"
    assert by_sev["suggestion"].status == "open"
    statuses = [
        (e["kind"], e["detail"].get("outcome"), e["to_status"])
        for e in ledger.history(conn, REPO, PR, by_sev["suggestion"].hash)
        if e["kind"] == "outcome"
    ]
    assert statuses == [("outcome", "STILL_VALID", "open")]
    # nothing new went on GitHub: the carried suggestion is seen again, not posted again
    kinds = [e["kind"] for e in ledger.history(conn, REPO, PR, by_sev["suggestion"].hash)]
    assert "posted" not in kinds and "seen" in kinds


def test_a_conversation_concession_is_a_ledger_event(cfg, conn, gh, lanes):
    e2e.test_conversation_withdrawing_the_only_finding_approves(cfg, conn, gh, lanes)
    issues = _issues(conn)
    withdrawn = [i for i in issues.values() if i.status == "withdrawn"]
    assert len(withdrawn) == 1 and withdrawn[0].severity == "blocking"
    assert withdrawn[0].status_source == "conversation"
    ev = ledger.history(conn, REPO, PR, withdrawn[0].hash)[-1]
    assert ev["detail"]["conceded"] is True and ev["sha"] == e2e.HEAD
    # the conceded row and the event agree, so a backfill adds nothing for it
    assert ledger.backfill(conn)["conceded_only"] == 0


def test_a_deferral_without_a_maintainer_stays_pending(cfg, conn, gh, lanes):
    e2e.test_conversation_deferral_by_an_outside_contributor_keeps_the_blocker(cfg, conn, gh, lanes)
    (issue,) = [i for i in _issues(conn).values() if i.severity == "blocking"]
    assert issue.status == "deferral_pending" and issue.standing


def test_an_accepted_deferral_is_deferred(cfg, conn, gh, lanes):
    e2e.test_conversation_accepting_a_deferral_lifts_request_changes(cfg, conn, gh, lanes)
    (issue,) = [i for i in _issues(conn).values() if i.severity == "blocking"]
    assert (issue.status, issue.status_source) == ("deferred", "conversation")


def test_dry_runs_write_nothing(cfg, conn, gh, lanes):
    e2e.test_dry_run_posts_nothing_but_writes_body(cfg, conn, gh, lanes)
    assert conn.execute("SELECT count(*) FROM issue_events").fetchone()[0] == 0
