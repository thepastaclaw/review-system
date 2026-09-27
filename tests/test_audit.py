"""Audit queue: classification, enqueue, idle-capacity scheduling, and the worker's audit path."""

from __future__ import annotations

import json

import pytest

from reviewsys import audit, worker
from reviewsys.db import tx
from reviewsys.ingest import close_pr_heads, enqueue_head, ingest_repo
from reviewsys.models import RunStatus, Trigger
from reviewsys.scheduler import apply_supersedes, live_queued_count, schedule
from reviewsys.steps import worktree as wt

HEAD = "a" * 40
MERGE = "c" * 40
TIP = "d" * 40


def _pr(
    number=7,
    *,
    reviews=(),
    title="feat(drive): something",
    merged_by="QuantumExplorer",
    repo="dashpay/platform",
    author="QuantumExplorer",
    head=HEAD,
):
    return audit.MergedPr(
        repo=repo,
        number=number,
        title=title,
        author=author,
        merged_by=merged_by,
        merged_at="2026-09-20T12:00:00Z",
        head_sha=head,
        base_ref="v4.2-dev",
        merge_commit=MERGE,
        size=120,
        reviews=list(reviews),
    )


def _bot(state, *, phase="final", sha=HEAD, body_extra="", at="2026-09-20T11:00:00Z"):
    return {
        "author": {"login": "thepastaclaw"},
        "state": state,
        "submittedAt": at,
        "commit": {"oid": sha},
        "body": f"<!-- thepastaclaw-review-phase v1 phase={phase} sha={sha} -->\n## x\n{body_extra}",
    }


# ---- classification ----


def test_clean_only_for_approval_or_blocker_free_final_on_the_merged_head():
    bot = "thepastaclaw"
    assert audit.is_clean(_pr(reviews=[_bot("APPROVED")]), bot)
    assert audit.is_clean(_pr(reviews=[_bot("COMMENTED")]), bot)
    assert not audit.is_clean(_pr(reviews=[]), bot)
    assert not audit.is_clean(_pr(reviews=[_bot("CHANGES_REQUESTED")]), bot)
    assert not audit.is_clean(_pr(reviews=[_bot("COMMENTED", phase="preliminary")]), bot)
    # approved, but an older commit
    assert not audit.is_clean(_pr(reviews=[_bot("APPROVED", sha="b" * 40)]), bot)
    # the newest review of the head wins: an approval later withdrawn by a follow-up
    assert not audit.is_clean(
        _pr(reviews=[_bot("APPROVED"), _bot("CHANGES_REQUESTED", at="2026-09-20T11:30:00Z")]), bot
    )
    # a bot-authored PR only ever gets COMMENT transport; the canonical verdict is in the body
    assert not audit.is_clean(_pr(reviews=[_bot("COMMENTED", body_extra=audit.CANONICAL_RC)]), bot)
    assert not audit.is_clean(_pr(reviews=[_bot("COMMENTED", body_extra="🔴 2 blocking")]), bot)
    # our own record wins over the GitHub transport state
    assert not audit.is_clean(_pr(reviews=[_bot("COMMENTED")]), bot, ("REQUEST_CHANGES", "final"))
    assert audit.is_clean(_pr(reviews=[]), bot, ("APPROVE", "final"))


def test_modes_and_ranks():
    assert audit.mode_for(_pr(title="ci: re-pin PR Hygiene")) == audit.MODE_LIGHT
    assert (
        audit.mode_for(_pr(title="chore(release): update changelog and bump version to 4.2.0"))
        == audit.MODE_LIGHT
    )
    assert audit.mode_for(_pr(title="Merge/v4.2 dev into 4.3 dev")) == audit.MODE_SYNC
    assert audit.mode_for(_pr(title="chore: merge v1.7-dev into v1.8-dev")) == audit.MODE_SYNC
    assert audit.mode_for(_pr(title="fix: merge v4.2-dev into v4.3-dev")) == audit.MODE_SYNC
    assert audit.mode_for(_pr(title="backport: Merge bitcoin#29904, 29967")) == audit.MODE_FULL
    assert (
        audit.mode_for(_pr(title="refactor(ui): merge identity screens into the identity module"))
        == audit.MODE_FULL
    )
    assert audit.mode_for(_pr(title="feat(drive): x")) == audit.MODE_FULL
    first = ("QuantumExplorer",)
    qe_platform = audit.seed_rank(_pr(), audit.MODE_FULL, first)
    qe_grovedb = audit.seed_rank(_pr(repo="dashpay/grovedb"), audit.MODE_FULL, first)
    other_platform = audit.seed_rank(_pr(merged_by="shumkov"), audit.MODE_FULL, first)
    qe_light = audit.seed_rank(_pr(), audit.MODE_LIGHT, first)
    assert qe_platform < qe_grovedb < other_platform
    assert qe_platform < qe_light < other_platform


# ---- enqueue ----


def test_enqueue_creates_audit_head_and_is_idempotent(cfg, conn):
    with tx(conn):
        assert audit.enqueue(conn, cfg, _pr(), source="seed", rank=110) == "created"
        assert audit.enqueue(conn, cfg, _pr(), source="seed", rank=110) == "exists"
        assert (
            audit.enqueue(conn, cfg, _pr(8, reviews=[_bot("APPROVED")]), source="seed", rank=1)
            == "clean"
        )
        assert (
            audit.enqueue(conn, cfg, _pr(9, author="dependabot[bot]"), source="seed", rank=1)
            == "excluded"
        )
        assert (
            audit.enqueue(
                conn, cfg, _pr(10, title="Merge/v4.2 dev into 4.3 dev"), source="seed", rank=1
            )
            == "skipped"
        )
    head = conn.execute("SELECT * FROM heads WHERE number=7").fetchone()
    assert (head["queue"], head["status"], head["trigger"]) == ("audit", "queued", "audit")
    row = conn.execute("SELECT * FROM audits WHERE number=7").fetchone()
    assert row["head_id"] == head["id"] and row["coverage"] == "none"
    assert conn.execute("SELECT verdict FROM audits WHERE number=10").fetchone()[0] == "SKIPPED"
    assert conn.execute("SELECT COUNT(*) FROM heads WHERE number=10").fetchone()[0] == 0


def test_enqueue_adopts_the_live_head_ingest_closed_and_cancels_its_run(cfg, conn):
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 7, HEAD, Trigger.MENTION)
    (rid,) = schedule(conn, cfg, spawn=False)
    with tx(conn):
        close_pr_heads(conn, "dashpay/platform", 7, "pr_closed")
        assert audit.enqueue(conn, cfg, _pr(), source="live", rank=0) == "created"
    heads = conn.execute("SELECT queue, status FROM heads WHERE number=7").fetchall()
    assert [tuple(h) for h in heads] == [("audit", "queued")]
    assert conn.execute("SELECT cancel_requested FROM runs WHERE id=?", (rid,)).fetchone()[0] == 1
    # ingest must leave the audit head alone even if the PR shows up open again
    with tx(conn):
        assert enqueue_head(conn, cfg, "dashpay/platform", 7, HEAD, Trigger.NEW_PUSH) == "noop"
        close_pr_heads(conn, "dashpay/platform", 7, "pr_closed")
    assert conn.execute("SELECT status FROM heads WHERE number=7").fetchone()[0] == "queued"


def test_ingest_closing_a_pr_leaves_an_audit_run_running(cfg, conn, gh):
    with tx(conn):
        audit.enqueue(conn, cfg, _pr(), source="seed", rank=100)
        conn.execute(
            "INSERT INTO prs (repo, number, head_sha, state, updated_at) VALUES (?,?,?,?,?)",
            ("dashpay/platform", 7, HEAD, "open", "2026-01-01T00:00:00Z"),
        )
    (rid,) = schedule(conn, cfg, spawn=False)
    ingest_repo(conn, cfg, gh, "dashpay/platform", [])
    assert apply_supersedes(conn, cfg) == 0
    assert conn.execute("SELECT cancel_requested FROM runs WHERE id=?", (rid,)).fetchone()[0] == 0


# ---- scheduling ----


def _live(conn, cfg, n, start=100):
    with tx(conn):
        for i in range(n):
            enqueue_head(conn, cfg, "dashpay/platform", start + i, f"{i + 1:040x}", Trigger.NEW_PR)
        conn.execute("UPDATE heads SET eligible_at=queued_at WHERE queue='live'")


def _audits(conn, cfg, n, start=500):
    with tx(conn):
        for i in range(n):
            audit.enqueue(
                conn, cfg, _pr(start + i, head=f"{i + 0xA0:040x}"), source="seed", rank=100 + i
            )


def _running(conn, queue):
    return conn.execute(
        "SELECT COUNT(*) FROM runs r JOIN heads h ON h.id=r.head_id "
        "WHERE r.status='spawned' AND h.queue=?",
        (queue,),
    ).fetchone()[0]


def _concurrency(conn, n):
    with tx(conn):
        conn.execute(
            "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (audit.KV_CONCURRENCY, str(n))
        )


def test_audits_fill_only_the_capacity_live_work_leaves_under_the_ceiling(cfg, conn):
    _audits(conn, cfg, 6)
    _concurrency(conn, 4)
    # ceiling = max_concurrent 2 + overflow 1 = 3 streams per account: the override of 4 is
    # capped there, never added on top
    assert len(schedule(conn, cfg, spawn=False)) == 3
    assert _running(conn, "audit") == 3
    # live work arriving now still gets its own slots: audits never take a live slot
    _live(conn, cfg, 3)
    schedule(conn, cfg, spawn=False)
    assert _running(conn, "live") == 2 and _running(conn, "audit") == 3
    # while live work runs, finished audits are not replaced past the ceiling
    with tx(conn):
        conn.execute(
            "UPDATE runs SET status='done' WHERE id IN (SELECT r.id FROM runs r JOIN heads h "
            "ON h.id=r.head_id WHERE h.queue='audit')"
        )
    schedule(conn, cfg, spawn=False)
    assert _running(conn, "live") == 2 and _running(conn, "audit") == 0  # 1 live still waits


def test_no_audit_starts_while_live_work_waits_for_a_slot(cfg, conn):
    _live(conn, cfg, 3)  # 2 start, 1 waits
    _audits(conn, cfg, 2)
    schedule(conn, cfg, spawn=False)
    assert _running(conn, "live") == 2 and _running(conn, "audit") == 0
    # the waiting live head starts once a slot frees; only then do audits start
    with tx(conn):
        conn.execute(
            "UPDATE runs SET status='done' WHERE id=(SELECT MIN(r.id) FROM runs r JOIN heads h "
            "ON h.id=r.head_id WHERE h.queue='live')"
        )
    schedule(conn, cfg, spawn=False)
    assert _running(conn, "live") == 2 and _running(conn, "audit") == 1


def test_audit_backlog_never_triggers_the_live_backlog_rules(cfg, conn):
    _audits(conn, cfg, 30)
    assert live_queued_count(conn) == 0
    _live(conn, cfg, 3)
    assert live_queued_count(conn) == 3


def test_audits_start_in_rank_order(cfg, conn):
    with tx(conn):
        audit.enqueue(
            conn, cfg, _pr(1, head="1" * 40, merged_by="shumkov"), source="seed", rank=300
        )
        audit.enqueue(conn, cfg, _pr(2, head="2" * 40), source="seed", rank=110)
        audit.enqueue(conn, cfg, _pr(3, head="3" * 40), source="live", rank=0)
    first = schedule(conn, cfg, spawn=False)  # config: one at a time
    assert len(first) == 1
    num = conn.execute(
        "SELECT h.number FROM runs r JOIN heads h ON h.id=r.head_id WHERE r.id=?", first
    ).fetchone()[0]
    assert num == 3  # a live merge goes ahead of the backfill


# ---- persistence parsing ----


def test_persistence_parse_defaults_to_unknown_and_never_fixed():
    hashes = ["h1", "h2", "h3"]
    out = audit.parse_persistence(
        {
            "findings": [
                {"finding_hash": "h1", "status": "fixed", "evidence": "gone", "fixed_by": "#4801"},
                {"finding_hash": "h2", "status": "made-up"},
                {"finding_hash": "zz", "status": "FIXED"},
            ]
        },
        hashes,
    )
    assert out["h1"] == {"status": "FIXED", "evidence": "gone", "fixed_by": "#4801"}
    assert out["h2"]["status"] == "UNKNOWN" and out["h3"]["status"] == "UNKNOWN"
    assert "zz" not in out
    assert all(
        v["status"] == "UNKNOWN" for v in audit.parse_persistence_text("oops", hashes).values()
    )


# ---- worker: the audit path end to end ----


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
    monkeypatch.setattr(wt, "merge_base", lambda worktree, base, sha: "e" * 40)
    monkeypatch.setattr(wt, "pre_merge_base", lambda worktree, base, merge, sha: "b" * 40)
    monkeypatch.setattr(wt, "fetch_branch", lambda repo_dir, branch: TIP)
    monkeypatch.setattr(wt, "ensure_commit", lambda repo_dir, sha: True)


def _blocking(title="Fee estimation omits drainage costs"):
    return {
        "file": "f.rs",
        "line_start": 11,
        "line_end": 12,
        "severity": "blocking",
        "confidence": 0.95,
        "category": "logic",
        "title": title,
        "body": "Estimation must be >= actual.",
    }


def _verifier(findings):
    return {
        "summary": "Audit summary.",
        "review_action": "COMMENT",
        "findings": findings,
        "dropped_findings": [],
        "out_of_scope_findings": [],
        "coderabbit_reactions": [],
        "prerequisite_adjudications": [],
        "adjudication_complete": True,
    }


class PersistenceLanes:
    """FakeLanes plus a canned persistence answer (role `persistence`)."""

    def __init__(self, lanes, answer):
        self.inner, self.answer, self.specs = lanes, answer, []

    def __call__(self, spec, art, worktree):
        self.specs.append(spec)
        if spec.role == "persistence":
            from reviewsys.lane import LaneResult

            art.mkdir(parents=True, exist_ok=True)
            text = json.dumps(self.answer(spec))
            return LaneResult(
                exit_code=0,
                stdout=json.dumps({"result": text}),
                stderr="",
                duration_s=1,
                result_text=text,
            )
        return self.inner(spec, art, worktree)


def _merged(gh):
    gh.pr = {**gh.pr, "state": "closed", "merged": True, "base": {"ref": "v4.2-dev"}}


@pytest.fixture
def post_live_cfg(cfg):
    import dataclasses

    return dataclasses.replace(cfg, audit=dataclasses.replace(cfg.audit, post_live=True))


def _audit_run(cfg, conn, gh, lanes, *, source="seed", title="feat(drive): x"):
    with tx(conn):
        assert audit.enqueue(conn, cfg, _pr(1, title=title), source=source, rank=100) == "created"
    (rid,) = schedule(conn, cfg, spawn=False)
    status = worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False)
    return rid, status


def test_seeded_audit_runs_both_phases_checks_persistence_and_posts_nothing(cfg, conn, gh, lanes):
    _merged(gh)
    lanes.reviewer["default"] = {
        "summary": "s",
        "findings": [_blocking()],
        "out_of_scope_findings": [],
    }
    lanes.verifier["preliminary"] = _verifier([_blocking()])  # phase 1 found a blocker ...
    lanes.verifier["final"] = _verifier([_blocking()])
    runner = PersistenceLanes(
        lanes,
        lambda spec: {
            "findings": [
                {
                    "finding_hash": h,
                    "status": "STILL_PRESENT",
                    "evidence": "f.rs:40 unchanged",
                    "fixed_by": None,
                }
                for h in [
                    json.loads(spec.prompt.split("```json\n", 1)[1].split("\n```", 1)[0])[0][
                        "finding_hash"
                    ]
                ]
            ]
        },
    )
    rid, status = _audit_run(cfg, conn, gh, runner)
    assert status == RunStatus.DONE
    steps = {
        r["name"]: r["status"]
        for r in conn.execute("SELECT name, status FROM steps WHERE run_id=?", (rid,))
    }
    # ... and Phase 2 still ran: an audit needs the complete finding set
    assert steps["phase2"] == "ok" and steps["verify2"] == "ok" and steps["persistence"] == "ok"
    assert gh.posted_reviews == [] and gh.gate_bodies == [] and gh.issue_comments == []
    # the persistence lane ran in a worktree at the base tip, not the merged head
    (p,) = [s for s in runner.specs if s.role == "persistence"]
    assert str(p.cwd).endswith("-tip") and TIP in p.prompt and HEAD in p.prompt
    row = conn.execute("SELECT * FROM audits WHERE number=1").fetchone()
    assert (row["verdict"], row["blockers"], row["still_present"], row["tip_sha"]) == (
        "REQUEST_CHANGES",
        1,
        1,
        TIP,
    )
    f = conn.execute("SELECT status, evidence FROM audit_findings").fetchone()
    assert tuple(f) == ("STILL_PRESENT", "f.rs:40 unchanged")
    report = audit.local_report(cfg, "dashpay/platform", 1).read_text()
    assert "today: **STILL_PRESENT**" in report and "QuantumExplorer" in report
    assert conn.execute("SELECT status FROM heads WHERE number=1").fetchone()[0] == "done"
    # the verdict label mirrors only the live verdict: nothing recorded for the merged head
    assert conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 0


def test_audit_reviews_the_diff_as_merged(cfg, conn, gh, lanes):
    _merged(gh)
    lanes.reviewer["default"] = {"summary": "s", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    _audit_run(cfg, conn, gh, lanes)
    general = [s for s in lanes.calls if s.role == "general"]
    assert general and all(f"{'b' * 40}..{HEAD}" in s.prompt for s in general)


def test_live_merge_audit_comments_and_opens_one_issue_for_open_blockers(
    post_live_cfg, conn, gh, lanes
):
    cfg = post_live_cfg
    _merged(gh)
    lanes.reviewer["default"] = {
        "summary": "s",
        "findings": [_blocking(), _blocking("Second")],
        "out_of_scope_findings": [],
    }
    lanes.verifier["default"] = _verifier([_blocking(), _blocking("Second")])
    issues = []

    def answer(spec):
        items = json.loads(spec.prompt.split("```json\n", 1)[1].split("\n```", 1)[0])
        return {
            "findings": [
                {
                    "finding_hash": items[0]["finding_hash"],
                    "status": "FIXED",
                    "evidence": "x",
                    "fixed_by": "#4801",
                },
                {
                    "finding_hash": items[1]["finding_hash"],
                    "status": "STILL_PRESENT",
                    "evidence": "y",
                },
            ]
        }

    orig = gh._dispatch

    def dispatch(args, stdin):
        if args[1].startswith("repos/dashpay/platform/issues?creator="):
            return [[{"title": "unrelated", "body": "x", "html_url": "u"}]]
        if args[:2] == ["api", "repos/dashpay/platform/issues"]:
            issues.append(json.loads(stdin))
            return {"html_url": "https://github.com/dashpay/platform/issues/9999"}
        if args[:2] == ["api", "repos/dashpay/platform/issues/1/comments"] and "POST" in args:
            gh.issue_comments.append(json.loads(stdin))
            return {"html_url": "https://github.com/dashpay/platform/pull/1#issuecomment-1"}
        return orig(args, stdin)

    gh._dispatch = dispatch
    _, status = _audit_run(cfg, conn, gh, PersistenceLanes(lanes, answer), source="live")
    assert status == RunStatus.DONE
    assert gh.posted_reviews == []
    (comment,) = gh.issue_comments
    assert (
        comment["body"].startswith(audit.POST_MERGE_MARKER)
        and "Post-merge review" in comment["body"]
    )
    assert "**FIXED** (fixed by #4801)" in comment["body"]
    (issue,) = issues
    assert issue["title"].startswith("Post-merge review: 1 blocking finding(s) in #1")
    assert "Second" in issue["body"] and "Fee estimation" not in issue["body"]
    row = conn.execute("SELECT comment_url, issue_url, still_present FROM audits").fetchone()
    assert row["issue_url"].endswith("/issues/9999") and row["still_present"] == 1


def test_live_merge_audit_without_blockers_comments_but_opens_no_issue(
    post_live_cfg, conn, gh, lanes
):
    cfg = post_live_cfg
    _merged(gh)
    lanes.reviewer["default"] = {"summary": "s", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    posted = []
    orig = gh._dispatch

    def dispatch(args, stdin):
        if args[1].startswith("repos/dashpay/platform/issues") and "POST" in args:
            posted.append(args[1])
            return {"html_url": "https://x"}
        return orig(args, stdin)

    gh._dispatch = dispatch
    _audit_run(cfg, conn, gh, lanes, source="live")
    assert posted == ["repos/dashpay/platform/issues/1/comments"]


def test_light_audit_escalates_to_full_on_a_blocker(cfg, conn, gh, lanes):
    _merged(gh)
    lanes.reviewer["default"] = {
        "summary": "s",
        "findings": [_blocking()],
        "out_of_scope_findings": [],
    }
    lanes.verifier["default"] = _verifier([_blocking()])
    rid, status = _audit_run(cfg, conn, gh, lanes, title="ci: re-pin PR Hygiene")
    assert status == RunStatus.DONE
    assert not any(s.role == "triage" for s in lanes.calls)
    steps = {r["name"] for r in conn.execute("SELECT name FROM steps WHERE run_id=?", (rid,))}
    assert "phase2" not in steps and "persistence" not in steps
    row = conn.execute("SELECT mode, escalated, finished_at FROM audits").fetchone()
    assert tuple(row) == ("full", 1, None)
    assert conn.execute("SELECT status FROM heads").fetchone()[0] == "queued"


def test_audit_of_an_unmerged_pr_fails_without_retry(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {"summary": "s", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    _, status = _audit_run(cfg, conn, gh, lanes)  # gh.pr is open, not merged
    assert status == RunStatus.FAILED
    assert conn.execute("SELECT status FROM heads").fetchone()[0] == "failed"


def test_index_counts_by_merger(cfg, conn, gh, lanes):
    _merged(gh)
    lanes.reviewer["default"] = {
        "summary": "s",
        "findings": [_blocking()],
        "out_of_scope_findings": [],
    }
    lanes.verifier["default"] = _verifier([_blocking()])
    _audit_run(cfg, conn, gh, PersistenceLanes(lanes, lambda spec: {"findings": []}))
    text = audit.render_index(conn)
    assert "| QuantumExplorer | 1 | 1 | 1 |" in text  # UNKNOWN counts as still present
    assert audit.summary(conn, cfg)["would_block"] == 1


# ---- review fixes: crash-safe posting, public export, empty diff, private report repo ----


def test_live_post_is_not_repeated_after_a_crash_between_post_and_record(cfg, gh):
    o = audit.Outcome(
        audit={
            "id": 1,
            "repo": "dashpay/platform",
            "number": 1,
            "sha": HEAD,
            "title": "t",
            "merged_by": "QuantumExplorer",
            "merged_at": "x",
            "coverage": "none",
            "comment_url": None,
            "issue_url": None,
        },
        verdict="COMMENT",
        findings=[],
        persistence={},
        tip_ref="v4.2-dev",
        tip_sha=TIP,
        review_body="body",
        degraded=False,
        run_id=1,
    )
    orig = gh._dispatch
    posts = []

    def dispatch(args, stdin):
        if "POST" in args:
            posts.append(args[1])
        if args[1].startswith("repos/dashpay/platform/issues/1/comments") and "GET" in args:
            # the comment a previous attempt posted before the worker died
            return [
                [
                    {
                        "user": {"login": "thepastaclaw"},
                        "body": audit.POST_MERGE_MARKER,
                        "html_url": "https://c",
                    }
                ]
            ]
        return orig(args, stdin)

    gh._dispatch = dispatch
    saved = {}
    url, issue = audit.post_live(gh, o, bot_login="thepastaclaw", persist=saved.__setitem__)
    assert (url, issue, posts) == ("https://c", None, [])
    assert saved == {"comment_url": "https://c"}


def test_public_export_never_carries_audit_events(cfg, conn):
    from reviewsys.exporter import build_export

    with tx(conn):
        audit.enqueue(conn, cfg, _pr(), source="seed", rank=100)
        conn.execute(
            "INSERT INTO events (ts, kind, repo, number, detail) VALUES "
            "('2026-09-26T00:00:00Z','audit.done','dashpay/platform',7,'REQUEST_CHANGES blockers=3')"
        )
        conn.execute(
            "INSERT INTO events (ts, kind, detail) VALUES ('2026-09-26T00:00:01Z','run.done','ok')"
        )
    kinds = {e["kind"] for e in build_export(conn, cfg)["history"]["recent_events"]}
    assert "run.done" in kinds and not any(k.startswith("audit.") for k in kinds)


def test_audit_refuses_an_empty_range(cfg, conn, gh, lanes, monkeypatch):
    _merged(gh)
    monkeypatch.setattr(wt, "pre_merge_base", lambda worktree, base, merge, sha: None)
    monkeypatch.setattr(wt, "merge_base", lambda worktree, base, sha: HEAD)  # true merge
    lanes.reviewer["default"] = {"summary": "s", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier([])
    _, status = _audit_run(cfg, conn, gh, lanes)
    assert status == RunStatus.FAILED
    assert not any(s.role == "general" for s in lanes.calls)
    assert conn.execute("SELECT status FROM heads").fetchone()[0] == "failed"


def test_reports_are_never_written_to_a_public_repo(cfg, conn, gh):
    import dataclasses

    cfg = dataclasses.replace(cfg, audit=dataclasses.replace(cfg.audit, report_repo="o/public"))
    orig = gh._dispatch
    writes = []

    def dispatch(args, stdin):
        if args[1] == "repos/o/public":
            return {"private": False}
        if "PUT" in args:
            writes.append(args[1])
        return orig(args, stdin)

    gh._dispatch = dispatch
    assert audit.publish_index(conn, cfg, gh) == {"repo": "o/public", "refused": "not private"}
    assert writes == []
