"""CI after the verdict: watch registration, the `ci_watch` daemon pass, the CI-check run."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from reviewsys import ci, github, labels, publish, status, worker
from reviewsys.config import DEFAULT_CI_IGNORE, CiConfig
from reviewsys.daemon import Daemon
from reviewsys.db import fmt_ts, kv_get, kv_set, now, now_dt, tx
from reviewsys.ingest import enqueue_head
from reviewsys.models import RunStatus, Trigger
from reviewsys.queue_status import queued_order
from reviewsys.scheduler import schedule
from reviewsys.steps import worktree as wt

HEAD = "a" * 40
REPO = "dashpay/platform"
RUNS_EP = f"repos/{REPO}/commits/{HEAD}/check-runs?per_page=100"
STATUS_EP = f"repos/{REPO}/commits/{HEAD}/status?per_page=100"
LOG_EP = f"repos/{REPO}/actions/jobs/77/logs"
IGNORE = CiConfig().ignore_re


@pytest.fixture(autouse=True)
def fake_git(monkeypatch, tmp_path):
    monkeypatch.setattr(wt, "ensure_mirror", lambda mirrors, repo: tmp_path / "mirror")
    monkeypatch.setattr(wt, "fetch_head", lambda mirror, number, sha: None)

    def create(mirror, wts, name, sha):
        p = wts / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(wt, "create_worktree", create)
    monkeypatch.setattr(wt, "remove_worktree", lambda mirror, path: None)
    monkeypatch.setattr(wt, "merge_base", lambda worktree, base, sha: "b" * 40)


def _check(name, conclusion="success", *, status="completed", job=None):
    url = f"https://github.com/{REPO}/actions/runs/5/job/{job}" if job else f"https://ci/{name}"
    return {
        "name": name,
        "status": status,
        "conclusion": conclusion if status == "completed" else None,
        "html_url": url,
    }


def _ci(gh, *runs, statuses=()):
    gh.routes[RUNS_EP] = {"total_count": len(runs), "check_runs": list(runs)}
    gh.routes[STATUS_EP] = {"total_count": len(statuses), "statuses": list(statuses)}


def _approved(conn, cfg, gh, *, event="APPROVE", head_status="done"):
    """An open PR whose head has the bot's final review standing with `event`."""
    with tx(conn):
        conn.execute(
            "INSERT INTO prs (repo, number, head_sha, state, updated_at) VALUES (?,?,?,?,?)",
            (REPO, 1, HEAD, "open", now()),
        )
        enqueue_head(conn, cfg, REPO, 1, HEAD, Trigger.MENTION)
        conn.execute("UPDATE heads SET status=?, finished_at=?", (head_status, now()))
        rid = conn.execute(
            "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at) "
            "VALUES (1, 1, 'done', 't', ?, ?)",
            (now(), now()),
        ).lastrowid
        conn.execute(
            "INSERT INTO reviews (run_id, repo, number, sha, phase, event, posted_at) VALUES (?,?,?,?,?,?,?)",
            (rid, REPO, 1, HEAD, "final", event, now()),
        )
    gh.posted_reviews.append(
        {
            "id": 4242,
            "event": event,
            "body": f"{github.REVIEW_MARKER}\n<!-- thepastaclaw-review-phase v1 phase=final "
            f"sha={HEAD} policy=x -->\n## Final validation\n\nThe fee path looks right.\n\n"
            "<details>provenance</details>",
        }
    )
    return rid


def _register(conn, cfg, gh, known=()):
    return ci.register(conn, cfg, gh, repo=REPO, number=1, sha=HEAD, run_id=1, known=list(known))


def _watch(conn):
    return ci.load(conn, REPO, 1, HEAD)


def _events(conn, prefix="ci."):
    return [
        (r["kind"], r["detail"])
        for r in conn.execute(
            "SELECT kind, detail FROM events WHERE kind LIKE ? ORDER BY id", (prefix + "%",)
        )
    ]


def _ci_run(cfg, conn, gh, lanes):
    (rid,) = schedule(conn, cfg, spawn=False)
    status = worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False)
    return rid, status


# ---- classification ----


@pytest.mark.parametrize(
    "name",
    [
        "PR title",
        "Validate PR title",
        "check-title",
        "action-semantic-pull-request",
        "codecov/patch",
        "policy / reconcile",
        "PR Hygiene",
        "CodeRabbit",
        "CodeRabbit or PastaClaw review",
        "assign-milestone",
        "conflict-check",
        "predict_conflicts",
        "check_merge",
        "license/cla",
        "ready-for-review-trigger",
        "dependency-review",
    ],
)
def test_meta_checks_are_ignored_by_the_default_regex(name):
    assert IGNORE.search(name), name


@pytest.mark.parametrize(
    "name",
    [
        "Rust workspace tests / Tests",
        "linux64-build / Build source",
        "ClangFormat",
        "Lint / Run linters",
        "macOS / wallet",
        "Swift SDK build / Swift SDK build + tests (warnings as errors)",
        "netlify/docs/deploy-preview",
        "reviewdog / clippy",
    ],
)
def test_code_checks_count(name):
    assert not IGNORE.search(name), name


def test_classify_splits_pending_and_failed_and_skips_meta_checks():
    snap = {
        "checks": [
            {"name": "Tests", "status": "completed", "conclusion": "failure"},
            {"name": "Tests", "status": "completed", "conclusion": "failure"},  # one per name
            {"name": "Build", "status": "in_progress", "conclusion": None},
            {"name": "Lint", "status": "completed", "conclusion": "timed_out"},
            {"name": "Docs", "status": "completed", "conclusion": "cancelled"},
            {"name": "Fuzz", "status": "completed", "conclusion": "skipped"},
            {"name": "PR Hygiene", "status": "in_progress", "conclusion": "pending"},
            {"name": "PR title", "status": "completed", "conclusion": "failure"},
            {"name": "ci/legacy", "status": "completed", "conclusion": "error"},  # a status
        ]
    }
    state = ci.classify(snap, IGNORE)
    assert [c["name"] for c in state.pending] == ["Build"]
    assert [c["name"] for c in state.failing] == ["Tests", "Lint", "ci/legacy"]
    assert ci.classify({"error": "boom"}, IGNORE).error == "boom"
    assert ci.failing_names(None, IGNORE) == []


def test_ignore_regex_is_configurable_and_validated(cfg, tmp_path, skills_dir):
    from reviewsys import config as cfg_mod

    text = (tmp_path / "config.toml").read_text()
    p = tmp_path / "c2.toml"
    p.write_text(text.replace("# ignore_checks = ", "ignore_checks = "))
    loaded = cfg_mod.load(p)
    assert loaded.ci.ignore_checks != DEFAULT_CI_IGNORE
    assert loaded.ci.ignore_re.search("Validate PR TITLE")
    assert not loaded.ci.ignore_re.search("PR Hygiene")  # not in the custom list
    p.write_text(text.replace("[ci]\n", '[ci]\nignore_checks = "("\n'))
    with pytest.raises(ValueError, match="ignore_checks"):
        cfg_mod.load(p)


def test_log_tail_strips_timestamps_and_colour_and_keeps_the_end():
    text = "\n".join(f"2026-10-16T10:00:00.1234567Z \x1b[31mline {i}\x1b[0m" for i in range(500))
    tail = ci.log_tail(text)
    assert tail.splitlines()[0] == "line 300" and tail.splitlines()[-1] == "line 499"
    assert "\x1b" not in tail and "2026-10-16T" not in tail


# ---- registration ----


def test_register_watches_an_approve_while_checks_run(cfg, conn, gh):
    _approved(conn, cfg, gh)
    _ci(gh, _check("Tests", status="in_progress"), _check("PR title", "failure"))
    assert _register(conn, cfg, gh) == "watching"
    w = _watch(conn)
    assert w["state"] == "watching" and w["event"] == "APPROVE" and w["known"] == []
    assert _events(conn)[-1][0] == "ci.watch"
    assert ci.summary(conn) == {"watching": 1, "queued": 0}
    assert status.snapshot(conn, cfg)["ci_watches"] == {"watching": 1, "queued": 0}


def test_register_skips_complete_ci_and_known_failures(cfg, conn, gh):
    _approved(conn, cfg, gh)
    _ci(gh, _check("Tests"), _check("Lint", "failure"))
    # the verifier saw Lint fail and approved anyway (a flake): nothing new to watch
    assert _register(conn, cfg, gh, known=["Lint"]) == "complete"
    assert _watch(conn) is None
    # Lint failed after the run read CI for its context: the verdict never weighed it
    assert _register(conn, cfg, gh) == "watching"
    assert "failed since the review: ['Lint']" in _events(conn)[-1][1]


def test_register_ignores_anything_but_an_approve(cfg, conn, gh):
    _approved(conn, cfg, gh, event="COMMENT")
    _ci(gh, _check("Tests", status="queued"))
    assert _register(conn, cfg, gh) == "not approved"
    assert _watch(conn) is None


# ---- the daemon pass ----


def test_pending_checks_keep_the_watch(cfg, conn, gh):
    _approved(conn, cfg, gh)
    _ci(gh, _check("Tests", status="in_progress"))
    _register(conn, cfg, gh)
    assert ci.watch(conn, cfg, gh)["watching"] == 1
    assert _watch(conn)["checked_at"]


def test_green_ci_drops_the_watch(cfg, conn, gh):
    _approved(conn, cfg, gh)
    _ci(gh, _check("Tests", status="in_progress"))
    _register(conn, cfg, gh)
    # a meta check failing or never finishing does not hold the watch open
    _ci(
        gh,
        _check("Tests"),
        _check("PR title", "failure"),
        statuses=[{"context": "PR Hygiene", "state": "pending"}],
    )
    assert ci.watch(conn, cfg, gh)["green"] == 1
    assert _watch(conn) is None
    assert _events(conn)[-1] == ("ci.watch_green", f"{HEAD[:8]}: CI complete, no failure")


def test_a_failure_the_verdict_weighed_is_not_judged_again(cfg, conn, gh):
    _approved(conn, cfg, gh)
    _ci(gh, _check("Lint", "failure"), _check("Tests", status="in_progress"))
    _register(conn, cfg, gh, known=["Lint"])
    _ci(gh, _check("Lint", "failure"), _check("Tests"))
    assert ci.watch(conn, cfg, gh)["green"] == 1
    assert "besides 1 the verdict weighed" in _events(conn)[-1][1]


def test_a_failure_requeues_the_reviewed_head_for_its_ci_check(cfg, conn, gh):
    _approved(conn, cfg, gh)
    _ci(gh, _check("Tests", status="in_progress"))
    _register(conn, cfg, gh)
    # one check failed while another still runs: no need to wait for the rest
    _ci(gh, _check("Tests", "failure"), _check("Build", status="in_progress"))
    assert ci.watch(conn, cfg, gh)["enqueued"] == 1
    head = conn.execute("SELECT * FROM heads").fetchone()
    assert (head["status"], head["trigger"], head["priority"]) == ("queued", "ci_failed", 1)
    w = _watch(conn)
    assert w["state"] == "queued" and w["failing"] == ["Tests"]
    assert _events(conn)[-1][0] == "ci.failed"
    # its CI check is not in line for a review: no queue comment for it
    assert queued_order(conn) == []
    # while it waits, the next pass leaves it alone (no GitHub read)
    calls = len(gh.calls)
    assert ci.watch(conn, cfg, gh)["queued"] == 1
    assert len(gh.calls) == calls


@pytest.mark.parametrize(
    "change, kind, why",
    [
        ("UPDATE prs SET head_sha='c'", "ci.watch_dropped", "the head moved to c"),
        ("UPDATE prs SET state='closed'", "ci.watch_dropped", "the PR closed"),
        (
            "UPDATE reviews SET event='REQUEST_CHANGES'",
            "ci.watch_dropped",
            "the standing verdict is REQUEST_CHANGES",
        ),
    ],
)
def test_watch_drops_when_there_is_nothing_left_to_decide(cfg, conn, gh, change, kind, why):
    _approved(conn, cfg, gh)
    _ci(gh, _check("Tests", status="in_progress"))
    _register(conn, cfg, gh)
    with tx(conn):
        conn.execute(change)
    calls = len(gh.calls)
    assert ci.watch(conn, cfg, gh)["dropped"] == 1
    assert len(gh.calls) == calls  # decided from the database alone
    assert _watch(conn) is None
    assert _events(conn)[-1][0] == kind and why in _events(conn)[-1][1]


def test_watch_expires(cfg, conn, gh):
    _approved(conn, cfg, gh)
    _ci(gh, _check("Tests", status="in_progress"))
    _register(conn, cfg, gh)
    w = _watch(conn)
    w["posted_at"] = fmt_ts(now_dt() - timedelta(hours=cfg.ci.watch_hours, minutes=1))
    with tx(conn):
        kv_set(conn, ci.watch_key(REPO, 1, HEAD), json.dumps(w))
    assert ci.watch(conn, cfg, gh)["dropped"] == 1
    assert _events(conn)[-1][0] == "ci.watch_expired"


def test_a_ci_check_that_failed_for_good_drops_the_watch(cfg, conn, gh):
    _approved(conn, cfg, gh)
    _ci(gh, _check("Tests", status="in_progress"))
    _register(conn, cfg, gh)
    _ci(gh, _check("Tests", "failure"))
    ci.watch(conn, cfg, gh)
    with tx(conn):  # its retries ran out
        conn.execute("UPDATE heads SET status='failed'")
    assert ci.watch(conn, cfg, gh)["dropped"] == 1
    assert "not a finished review" in _events(conn)[-1][1]


def test_a_ci_check_taken_over_by_a_reply_is_queued_again_after_it(cfg, conn, gh):
    _approved(conn, cfg, gh)
    _ci(gh, _check("Tests", status="in_progress"))
    _register(conn, cfg, gh)
    _ci(gh, _check("Tests", "failure"))
    ci.watch(conn, cfg, gh)
    with tx(conn):
        assert enqueue_head(conn, cfg, REPO, 1, HEAD, Trigger.REVIEW_REPLY) == "promoted"
        # the conversation ran and left the APPROVE standing
        conn.execute("UPDATE heads SET status='done'")
    assert ci.watch(conn, cfg, gh)["enqueued"] == 1
    assert conn.execute("SELECT trigger FROM heads").fetchone()[0] == "ci_failed"


def test_daemon_runs_ci_watch_only_when_live_and_enabled(cfg, conn, gh, notifier):
    import dataclasses

    names = {t.name: t.interval_s for t in Daemon(cfg, conn, gh=gh, notifier=notifier).tasks}
    assert names["ci_watch"] == 180
    shadow = Daemon(cfg, conn, gh=gh, notifier=notifier, spawn=False)
    assert "ci_watch" not in {t.name for t in shadow.tasks}
    off = dataclasses.replace(cfg, ci=CiConfig(enabled=False))
    assert "ci_watch" not in {t.name for t in Daemon(off, conn, gh=gh, notifier=notifier).tasks}


# ---- who gets the head ----


def test_ci_trigger_never_creates_or_takes_over_a_head(cfg, conn, gh):
    with tx(conn):
        assert enqueue_head(conn, cfg, REPO, 1, HEAD, Trigger.CI_FAILED) == "noop"
        assert conn.execute("SELECT COUNT(*) FROM heads").fetchone()[0] == 0
        enqueue_head(conn, cfg, REPO, 1, HEAD, Trigger.NEW_PR)
        assert enqueue_head(conn, cfg, REPO, 1, HEAD, Trigger.CI_FAILED) == "noop"  # in line
        conn.execute("UPDATE heads SET status='failed'")
        assert enqueue_head(conn, cfg, REPO, 1, HEAD, Trigger.CI_FAILED) == "noop"


def test_a_person_asking_takes_a_queued_ci_check_over(cfg, conn, gh):
    _approved(conn, cfg, gh)
    with tx(conn):
        assert enqueue_head(conn, cfg, REPO, 1, HEAD, Trigger.CI_FAILED) == "requeued"
        # ingest's poll of the same commit is nobody asking
        assert enqueue_head(conn, cfg, REPO, 1, HEAD, Trigger.NEW_PUSH) == "noop"
        # nor is a mention: on a reviewed commit it changes nothing, CI check or not
        assert enqueue_head(conn, cfg, REPO, 1, HEAD, Trigger.MENTION) == "noop"
        assert conn.execute("SELECT trigger FROM heads").fetchone()[0] == "ci_failed"
        assert enqueue_head(conn, cfg, REPO, 1, HEAD, Trigger.PRIORITY_REQUEST) == "promoted"
        assert conn.execute("SELECT trigger FROM heads").fetchone()[0] == "priority_request"


def test_an_enqueue_during_a_running_ci_check_queues_a_review_after_it(cfg, conn, gh):
    _approved(conn, cfg, gh)
    with tx(conn):
        enqueue_head(conn, cfg, REPO, 1, HEAD, Trigger.CI_FAILED)
    (rid,) = schedule(conn, cfg, spawn=False)
    with tx(conn):
        # a reply is the router's to hold until the run ends (it asks again then)
        assert enqueue_head(conn, cfg, REPO, 1, HEAD, Trigger.REVIEW_REPLY) == "noop"
        assert kv_get(conn, f"rereview.after_run:{rid}") is None
        assert enqueue_head(conn, cfg, REPO, 1, HEAD, Trigger.MANUAL) == "promoted"
    assert kv_get(conn, f"rereview.after_run:{rid}") == "1"
    # the head is theirs now: should the CI check fail, its retry is their review
    assert conn.execute("SELECT trigger FROM heads").fetchone()[0] == "manual"


# ---- the CI-check run ----


def _red(conn, cfg, gh, *, pending=False):
    """An APPROVE whose watch saw `Tests` fail (its job log readable); CI check queued."""
    _approved(conn, cfg, gh)
    _ci(gh, _check("Tests", status="in_progress"))
    _register(conn, cfg, gh)
    runs = [_check("Tests", "failure", job=77)]
    if pending:
        runs.append(_check("Build", status="in_progress"))
    _ci(gh, *runs)
    gh.routes[LOG_EP] = "2026-10-16T10:00:00.1Z error[E0308]: mismatched types in f.rs:12\n"
    assert ci.watch(conn, cfg, gh)["enqueued"] == 1


def test_related_failure_moves_the_approve_to_comment_once(cfg, conn, gh, lanes):
    _red(conn, cfg, gh)
    gates = len(gh.gate_bodies)
    rid, status = _ci_run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert [c.role for c in lanes.calls] == ["ci_check"]  # no prep, reviewer or verifier
    spec = lanes.calls[0]
    assert (spec.model, spec.effort) == ("gpt-6-astra", "low")  # the triage lane's model
    assert "error[E0308]: mismatched types in f.rs:12" in spec.prompt  # the job log tail
    assert "- f.rs (+0 -0)" in spec.prompt and "The fee path looks right." in spec.prompt
    assert "Tests: failure" in spec.prompt
    assert len(gh.gate_bodies) == gates  # the review's gate comment is left alone
    upd = gh.posted_reviews[-1]
    assert upd["event"] == "COMMENT" and upd["comments"] == [] and upd["commit_id"] == HEAD
    body = upd["body"]
    assert f"<!-- thepastaclaw-review-update v1 phase=final sha={HEAD} -->" in body
    assert publish.CI_UPDATE_MARKER.format(sha=HEAD) in body
    assert f"## CI failed after approval — commit {HEAD[:8]}" in body
    assert f"[`Tests`](https://github.com/{REPO}/actions/runs/5/job/77) (failure)" in body
    assert "Why they look related: the failing test covers f.rs" in body
    reviews = github.reviews(gh, REPO, 1)
    assert publish.standing_verdict(reviews, HEAD, "final", cfg.bot_login) == "COMMENTED"
    assert labels.latest_verdict(conn, REPO, 1, HEAD) == "COMMENT"
    assert labels.wanted(conn, REPO, 1) == "pastaclaw:commented"
    assert kv_get(conn, ci.updated_key(REPO, 1, HEAD))
    assert _watch(conn) is None
    kinds = [k for k, _ in _events(conn, "")]
    assert "review.verdict_updated" in kinds and "ci.related" in kinds
    run = conn.execute("SELECT review_id, path FROM runs WHERE id=?", (rid,)).fetchone()
    assert run["review_id"] == upd["id"] and run["path"] == "priority"
    step = conn.execute("SELECT status, detail FROM steps WHERE run_id=?", (rid,)).fetchone()
    assert step["status"] == "ok" and json.loads(step["detail"])["posted"] is True
    assert conn.execute("SELECT status FROM heads").fetchone()["status"] == "done"
    # never twice per head: another failure later, or the same CI check queued again
    posted = len(gh.posted_reviews)
    _ci(gh, _check("Tests", "failure", job=77), _check("Docs", "failure"))
    _register(conn, cfg, gh)  # nothing approves any more: no watch
    with tx(conn):
        assert enqueue_head(conn, cfg, REPO, 1, HEAD, Trigger.CI_FAILED) == "requeued"
    _ci_run(cfg, conn, gh, lanes)
    assert len(gh.posted_reviews) == posted
    assert "already posted" in _events(conn)[-1][1]


def test_unrelated_failure_posts_nothing(cfg, conn, gh, lanes):
    _red(conn, cfg, gh)
    lanes.ci_check = {"related": "no", "reason": "the runner lost its network"}
    posted = len(gh.posted_reviews)
    assert _ci_run(cfg, conn, gh, lanes)[1] == RunStatus.DONE
    assert len(gh.posted_reviews) == posted
    assert labels.latest_verdict(conn, REPO, 1, HEAD) == "APPROVE"
    assert ("ci.unrelated", f"{HEAD[:8]}: Tests: the runner lost its network") in _events(conn)
    assert _watch(conn) is None  # nothing left running


def test_unrelated_failure_with_checks_still_running_watches_on(cfg, conn, gh, lanes):
    _red(conn, cfg, gh, pending=True)
    lanes.ci_check = {"related": "no", "reason": "flaky network"}
    _ci_run(cfg, conn, gh, lanes)
    w = _watch(conn)
    assert w["state"] == "watching" and w["known"] == ["Tests"]
    # Tests stays red (already judged), Build then fails too: a new CI check for Build only
    _ci(gh, _check("Tests", "failure", job=77), _check("Build", "failure"))
    assert ci.watch(conn, cfg, gh)["enqueued"] == 1
    lanes.ci_check = {"related": "yes", "reason": "Build compiles f.rs"}
    _ci_run(cfg, conn, gh, lanes)
    body = gh.posted_reviews[-1]["body"]
    assert "`Build`" in body and "`Tests`" not in body


def test_a_lane_that_cannot_judge_counts_as_unclear(cfg, conn, gh, lanes):
    _red(conn, cfg, gh)
    lanes.ci_check = "not json at all"
    _ci_run(cfg, conn, gh, lanes)
    assert [c.role for c in lanes.calls] == ["ci_check", "ci_check"]
    body = gh.posted_reviews[-1]["body"]
    assert "may relate to this change" in body
    assert "could not judge the failure" in body
    kinds = [k for k, _ in _events(conn)]
    assert "ci.check_degraded" in kinds and kinds[-1] == "ci.related"


def test_unclear_answer_also_moves_the_verdict(cfg, conn, gh, lanes):
    _red(conn, cfg, gh)
    lanes.ci_check = {"related": "unclear", "reason": "the log is cut off @someone"}
    _ci_run(cfg, conn, gh, lanes)
    body = gh.posted_reviews[-1]["body"]
    assert "Why it is unclear: the log is cut off @​someone" in body


@pytest.mark.parametrize("event", ["REQUEST_CHANGES", "COMMENT"])
def test_never_touches_a_verdict_that_is_not_an_approve(cfg, conn, gh, lanes, event):
    _red(conn, cfg, gh)
    # since the watch fired, a re-review on the same commit moved the verdict
    gh.posted_reviews.append(
        {
            "id": 4300,
            "event": event,
            "body": f"<!-- thepastaclaw-review-update v1 phase=final sha={HEAD} -->",
        }
    )
    posted = len(gh.posted_reviews)
    _ci_run(cfg, conn, gh, lanes)
    assert len(gh.posted_reviews) == posted
    assert lanes.calls == []  # decided before any lane
    assert _events(conn)[-1][0] == "ci.verdict_moved"


def test_a_re_run_that_passed_ends_the_check_quietly(cfg, conn, gh, lanes):
    _red(conn, cfg, gh)
    _ci(gh, _check("Tests"))
    posted = len(gh.posted_reviews)
    _ci_run(cfg, conn, gh, lanes)
    assert len(gh.posted_reviews) == posted and lanes.calls == []
    assert _events(conn)[-1][0] == "ci.recovered"


def test_an_update_posted_before_a_crash_is_recorded_not_reposted(cfg, conn, gh, lanes):
    _red(conn, cfg, gh)
    gh.posted_reviews.append(
        {
            "id": 4400,
            "event": "COMMENT",
            "body": publish.CI_UPDATE_MARKER.format(sha=HEAD),
        }
    )
    posted = len(gh.posted_reviews)
    _ci_run(cfg, conn, gh, lanes)
    assert len(gh.posted_reviews) == posted
    assert labels.latest_verdict(conn, REPO, 1, HEAD) == "COMMENT"
    assert json.loads(kv_get(conn, ci.updated_key(REPO, 1, HEAD)) or "{}")["review_id"] == 4400


def test_a_moved_head_ends_the_ci_check_without_touching_the_gate(cfg, conn, gh, lanes):
    _red(conn, cfg, gh)
    gh.pr = {**gh.pr, "head": {"sha": "c" * 40}}
    gates = len(gh.gate_bodies)
    _rid, status = _ci_run(cfg, conn, gh, lanes)
    assert status == RunStatus.CANCELLED
    assert conn.execute("SELECT status FROM heads").fetchone()["status"] == "superseded"
    assert len(gh.gate_bodies) == gates and lanes.calls == []


# ---- end to end ----


def _verifier(action):
    return {
        "summary": "Sound.",
        "review_action": action,
        "findings": [],
        "dropped_findings": [],
        "out_of_scope_findings": [],
        "coderabbit_reactions": [],
        "prerequisite_adjudications": [],
        "adjudication_complete": True,
    }


def test_approve_while_ci_runs_then_ci_fails(cfg, conn, gh, lanes):
    """The review approves while Tests still runs and Lint has already failed (the verifier
    saw that and approved anyway); Tests then fails: the CI check moves the verdict."""
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["preliminary"] = _verifier("COMMENT")
    lanes.verifier["final"] = _verifier("APPROVE")
    _ci(gh, _check("Tests", status="in_progress"), _check("Lint", "failure"))
    with tx(conn):
        conn.execute(
            "INSERT INTO prs (repo, number, head_sha, state, updated_at) VALUES (?,?,?,?,?)",
            (REPO, 1, HEAD, "open", now()),
        )
        enqueue_head(conn, cfg, REPO, 1, HEAD, Trigger.MENTION)
    (rid,) = schedule(conn, cfg, spawn=False)
    assert worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False) == RunStatus.DONE
    assert gh.posted_reviews[-1]["event"] == "APPROVE"
    w = _watch(conn)
    assert w["known"] == ["Lint"] and w["run_id"] == rid
    lanes.calls.clear()
    _ci(gh, _check("Tests", "failure"), _check("Lint", "failure"))
    assert ci.watch(conn, cfg, gh)["enqueued"] == 1
    _ci_run(cfg, conn, gh, lanes)
    assert [c.role for c in lanes.calls] == ["ci_check"]
    assert "Lint" not in lanes.calls[0].prompt.split("Failing checks:")[1].split("##")[0]
    assert gh.posted_reviews[-1]["event"] == "COMMENT"
    assert labels.wanted(conn, REPO, 1) == "pastaclaw:commented"
    out = ci.summary(conn)
    assert out == {"watching": 0, "queued": 0}


def test_a_dry_run_ci_check_judges_but_posts_nothing(cfg, conn, gh, lanes):
    _red(conn, cfg, gh)
    (rid,) = schedule(conn, cfg, spawn=False)
    posted = len(gh.posted_reviews)
    assert (
        worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False, dry_run=True)
        == RunStatus.DONE
    )
    assert [c.role for c in lanes.calls] == ["ci_check"]
    assert len(gh.posted_reviews) == posted
    assert labels.latest_verdict(conn, REPO, 1, HEAD) == "APPROVE"
    assert kv_get(conn, ci.updated_key(REPO, 1, HEAD)) is None


def test_check_runs_past_the_first_page_count(gh, cfg):
    """A big matrix has more than 100 check runs: one still running on page two keeps the
    watch open, and one failing there is seen."""
    gh.routes[RUNS_EP] = [
        {"total_count": 102, "check_runs": [_check(f"job {i}") for i in range(100)]},
        {
            "total_count": 102,
            "check_runs": [_check("late", status="queued"), _check("x", "failure")],
        },
    ]
    gh.routes[STATUS_EP] = {"total_count": 0, "statuses": []}
    state = ci.read(gh, cfg, REPO, HEAD)
    assert [c["name"] for c in state.pending] == ["late"]
    assert [c["name"] for c in state.failing] == ["x"]


# ---- races, retries, cancels and the paths around them ----


@pytest.mark.parametrize("still_running", [True, False])
def test_a_watch_settled_during_the_pass_is_not_written_back(
    cfg, conn, gh, monkeypatch, still_running
):
    """The daemon reads CI for seconds; a CI-check run that settles the watch meanwhile (re-arms
    it with what it judged, or ends it) wins: the stale copy is never written back."""
    _approved(conn, cfg, gh)
    _ci(gh, _check("Tests", status="in_progress"))
    _register(conn, cfg, gh)
    _ci(gh, _check("Tests", "failure"))
    real = ci.read

    def read(*args):
        state = real(*args)
        with tx(conn):
            ci.release(conn, REPO, 1, HEAD, judged=["Tests"], pending=still_running)
        return state

    monkeypatch.setattr(ci, "read", read)
    assert ci.watch(conn, cfg, gh)["enqueued"] == 0
    assert conn.execute("SELECT status FROM heads").fetchone()[0] == "done"
    w = _watch(conn)
    assert (w["known"] if w else None) == (["Tests"] if still_running else None)


def test_a_conversation_never_re_approves_over_a_ci_update(cfg, conn, gh, lanes, tmp_path):
    """Approved, then CI failed and its CI update moved the verdict to COMMENT. A reply that
    withdraws the review's only suggestion must not approve again (only a review that sees the
    failure may), and the CI update is never taken for the commit's publication."""
    from tests.test_worker_e2e import (
        KNST,
        WITHDRAW,
        _converse,
        _finding_thread,
        _full_strength_prior,
        _seed_prior,
    )
    from tests.test_worker_e2e import _sugg as e2e_sugg

    s = e2e_sugg()
    sh = _seed_prior(conn, s)
    posting_run = _full_strength_prior(conn)
    gh.threads = [_finding_thread(sh, s["title"], 800, [KNST], label="🟡 Suggestion")]
    final = f"<!-- thepastaclaw-review-phase v1 phase=final sha={HEAD} policy=x -->"
    update = f"<!-- thepastaclaw-review-update v1 phase=final sha={HEAD} -->\n"
    gh.posted_reviews += [
        {"id": 4242, "event": "APPROVE", "body": final},
        {
            "id": 4243,
            "event": "COMMENT",
            "body": update + publish.CI_UPDATE_MARKER.format(sha=HEAD),
        },
    ]
    with tx(conn):
        conn.execute("UPDATE heads SET sha=?, status='done'", (HEAD,))
        hid = conn.execute("SELECT head_id FROM runs WHERE id=?", (posting_run,)).fetchone()[0]
        ci_run = conn.execute(
            "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at) "
            "VALUES (?, 1, 'done', 't', ?, ?)",
            (hid, now(), now()),
        ).lastrowid
        conn.execute(
            "INSERT INTO steps (run_id, name, status, started_at) VALUES (?, 'ci_check', 'ok', 'x')",
            (ci_run,),
        )
        for rid, event in ((posting_run, "APPROVE"), (ci_run, "COMMENT")):
            conn.execute(
                "INSERT INTO reviews (run_id, repo, number, sha, phase, event, posted_at) VALUES (?,?,?,?,?,?,?)",
                (rid, REPO, 1, HEAD, "final", event, now()),
            )
        kv_set(conn, ci.updated_key(REPO, 1, HEAD), json.dumps({"review_id": 4243}))
    ctx = worker.RunContext(
        cfg=cfg,
        main_conn=conn,
        gh=gh,
        run_id=0,
        head_id=hid,
        repo=REPO,
        number=1,
        sha=HEAD,
        token="t",
        run_dir=tmp_path,
    )
    standing = worker._standing_findings(ctx, "final")
    assert standing.run_id == posting_run and standing.verified
    _converse(cfg, conn, gh, lanes, [{"finding_hash": sh, **WITHDRAW}], first=False)
    assert [r["event"] for r in gh.posted_reviews] == ["APPROVE", "COMMENT"]
    assert labels.latest_verdict(conn, REPO, 1, HEAD) == "COMMENT"


def test_a_ci_check_that_keeps_failing_leaves_the_review_done_without_an_alert(cfg, conn, gh):
    from reviewsys.models import FailKind
    from reviewsys.scheduler import finish_run

    _red(conn, cfg, gh)
    (rid,) = schedule(conn, cfg, spawn=False)
    finish_run(conn, cfg, rid, RunStatus.FAILED, reason="gh down", fail_kind=FailKind.INFRA)
    assert conn.execute("SELECT status FROM heads").fetchone()[0] == "queued"  # one retry
    with tx(conn):
        conn.execute("UPDATE heads SET eligible_at=?", (now(),))
    (rid2,) = schedule(conn, cfg, spawn=False)
    finish_run(conn, cfg, rid2, RunStatus.FAILED, reason="gh down", fail_kind=FailKind.INFRA)
    head = conn.execute("SELECT status, trigger FROM heads").fetchone()
    assert head["status"] == "done"  # never `failed`: that pages as a failed review
    assert _events(conn)[-1][0] == "ci.check_failed"
    assert _watch(conn) is None
    assert labels.latest_verdict(conn, REPO, 1, HEAD) == "APPROVE"


def test_a_cancel_during_the_lane_posts_nothing(cfg, conn, gh, lanes, monkeypatch):
    _red(conn, cfg, gh)
    real = worker._ci_judge

    def judge(ctx, text, names):
        ctx.cancel_flag.set()  # what the heartbeat does on `reviewsys cancel`
        return real(ctx, text, names)

    monkeypatch.setattr(worker, "_ci_judge", judge)
    posted = len(gh.posted_reviews)
    assert _ci_run(cfg, conn, gh, lanes)[1] == RunStatus.CANCELLED
    assert len(gh.posted_reviews) == posted


def test_judge_stops_without_a_second_attempt_when_the_lane_is_stopped(cfg, tmp_path):
    from reviewsys.lane import LaneResult

    calls = []

    def runner(spec, art, wt_):
        calls.append(spec)
        return LaneResult(exit_code=None, stdout="", stderr="", duration_s=0, cancelled=True)

    j = ci.judge(
        cfg, lane=cfg.policy.ci_check_lane, text="x", names=["T"], run_dir=tmp_path, runner=runner
    )
    assert (j.related, j.judged, j.error, len(calls)) == ("unclear", False, "lane stopped", 1)


def test_a_bot_authored_pr_is_judged_on_its_recorded_approve(cfg, conn, gh, lanes):
    """GitHub records every review on the bot's own PR as COMMENTED; the canonical APPROVE is
    in the database, and that is what the CI update moves."""
    gh.pr = {**gh.pr, "user": {"login": cfg.bot_login}}
    _red(conn, cfg, gh)
    gh.posted_reviews[0]["event"] = "COMMENT"
    _ci_run(cfg, conn, gh, lanes)
    assert gh.posted_reviews[-1]["body"].startswith("<!-- thepastaclaw-review-update v1")
    assert labels.latest_verdict(conn, REPO, 1, HEAD) == "COMMENT"


def test_a_conversation_approving_a_bot_authored_pr_watches_its_ci(cfg, conn, gh, lanes):
    """On the bot's own PR the conversation's APPROVE is only recorded (COMMENT transport
    equals the standing COMMENTED, nothing is posted); its CI is watched all the same."""
    from tests.test_worker_e2e import (
        KNST,
        WITHDRAW,
        _blocking,
        _converse,
        _finding_thread,
        _full_strength_prior,
        _seed_prior,
    )

    blocker = _blocking()
    bh = _seed_prior(conn, blocker, body="blocker body")
    _full_strength_prior(conn)
    gh.pr = {**gh.pr, "user": {"login": cfg.bot_login}}
    gh.threads = [_finding_thread(bh, blocker["title"], 800, [KNST])]
    gh.posted_reviews.append(
        {
            "id": 4242,
            "event": "COMMENT",
            "body": f"<!-- thepastaclaw-review-phase v1 phase=final sha={HEAD} policy=x -->",
        }
    )
    _ci(gh, _check("Tests", status="in_progress"))
    _converse(cfg, conn, gh, lanes, [{"finding_hash": bh, **WITHDRAW}])
    assert len(gh.posted_reviews) == 1  # recorded, not posted
    assert labels.latest_verdict(conn, REPO, 1, HEAD) == "APPROVE"
    w = _watch(conn)
    assert w is not None and w["known"] == []
