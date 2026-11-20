"""Check runs (checks.py) and GitHub App auth (ghapp.py)."""

from __future__ import annotations

import base64
import json
import subprocess
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from reviewsys import checks
from reviewsys import config as cfg_mod
from reviewsys.daemon import Daemon
from reviewsys.db import kv_set, now_dt, parse_ts, tx
from reviewsys.gh import Gh
from reviewsys.ghapp import AppAuth, AppAuthError

REPO = "dashpay/platform"
SHA = "a" * 40
OLD = "b" * 40
T0 = "2026-11-01T00:00:00Z"
FUTURE = "2999-01-01T00:00:00Z"


class ScriptedGh(Gh):
    """gh whose `api` calls are answered by `respond(method, endpoint, body)`."""

    def __init__(self, respond: Any) -> None:
        super().__init__("gh", runner=self._run)
        self.respond = respond
        self.calls: list[tuple[str, str, Any, list[str]]] = []

    def _run(
        self, argv: Sequence[str], stdin: str | None, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        args = list(argv)[1:]
        method = args[args.index("--method") + 1] if "--method" in args else "GET"
        body = json.loads(stdin) if stdin else None
        self.calls.append((method, args[1], body, args))
        out = self.respond(method, args[1], body)
        if isinstance(out, Exception):
            return subprocess.CompletedProcess(args, 1, "", str(out))
        return subprocess.CompletedProcess(args, 0, json.dumps(out), "")


class FakeApp:
    """Stands in for AppAuth: one installation client for every repo, or none (gh=None)."""

    def __init__(self, gh: Gh | None) -> None:
        self.gh = gh
        self.forgotten: list[str] = []
        self.lookups = 0

    def gh_for(self, repo: str) -> Gh | None:
        self.lookups += 1
        return self.gh

    def forget(self, repo: str) -> None:
        self.forgotten.append(repo)

    def app(self) -> dict[str, object]:
        return {"id": 1, "slug": "pastaclaw-review"}


class GitHubChecks:
    """A repository's check runs as the fake GitHub keeps them."""

    def __init__(self) -> None:
        self.runs: dict[int, dict[str, Any]] = {}
        self.fail: Exception | None = None
        self.listed: list[str] = []

    def __call__(self, method: str, ep: str, body: Any) -> Any:
        if self.fail is not None:
            return self.fail
        if method == "GET" and "/commits/" in ep:
            self.listed.append(ep)
            sha = ep.split("/commits/")[1].split("/")[0]
            mine = [
                {"id": cid, **run}
                for cid, run in sorted(self.runs.items(), reverse=True)
                if run.get("head_sha") == sha
            ]
            return {"check_runs": mine[:1]}
        if (
            method == "PATCH"
            and "/check-runs/" in ep
            and int(ep.rsplit("/", 1)[1]) not in self.runs
        ):
            return RuntimeError("gh: Not Found (HTTP 404)")
        if method == "POST" and ep.endswith("/check-runs"):
            cid = 100 + len(self.runs)
            self.runs[cid] = dict(body)
            return {"id": cid}
        if method == "PATCH" and "/check-runs/" in ep:
            cid = int(ep.rsplit("/", 1)[1])
            self.runs[cid].update(body)
            return {"id": cid}
        raise AssertionError(f"unrouted {method} {ep}")


def _pr(conn, sha=SHA, *, state="open", draft=0, skip=0, number=1):
    with tx(conn):
        conn.execute(
            "INSERT OR REPLACE INTO prs (repo, number, head_sha, is_draft, state, updated_at, skip_label)"
            " VALUES (?,?,?,?,?,?,?)",
            (REPO, number, sha, draft, state, T0, skip),
        )


def _head(
    conn,
    sha=SHA,
    status="queued",
    *,
    trigger="new_push",
    eligible=T0,
    priority=0,
    attempts=0,
    reason=None,
    carried=None,
    number=1,
) -> int:
    with tx(conn):
        cur = conn.execute(
            "INSERT INTO heads (repo, number, sha, trigger, priority, status, queued_at, eligible_at,"
            " attempts, reason, carried_from, finished_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                REPO,
                number,
                sha,
                trigger,
                priority,
                status,
                T0,
                eligible,
                attempts,
                reason,
                carried,
                T0 if status in ("done", "failed", "closed", "superseded") else None,
            ),
        )
    return int(cur.lastrowid)


def _run(conn, head_id: int, phase: str) -> None:
    with tx(conn):
        conn.execute(
            "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at, phase)"
            " VALUES (?,1,'running','t',?,?,?)",
            (head_id, T0, FUTURE, phase),
        )


def _review(conn, event: str, *, sha=SHA, phase="final", review_id=555, at=T0, number=1) -> None:
    with tx(conn):
        conn.execute(
            "INSERT INTO reviews (run_id, repo, number, sha, phase, github_review_id, event, posted_at)"
            " VALUES (NULL,?,?,?,?,?,?,?)",
            (REPO, number, sha, phase, review_id, event, at),
        )


def _want(conn, sha=SHA, number=1) -> checks.Want | None:
    return checks.wanted(conn, REPO, number, sha)


# ---- wanted(): the state machine ----


def test_queued_states(conn):
    _pr(conn)
    h = _head(conn, eligible=FUTURE)
    w = _want(conn)
    assert (w.status, w.title) == ("queued", "Queued — waiting for pushes to settle")
    with tx(conn):
        conn.execute("UPDATE heads SET eligible_at=? WHERE id=?", (T0, h))
    assert _want(conn).title == "Queued for review"
    with tx(conn):
        conn.execute("UPDATE heads SET priority=1, trigger='mention' WHERE id=?", (h,))
    assert _want(conn).title == "Queued — priority review"
    with tx(conn):
        conn.execute("UPDATE heads SET attempts=1, eligible_at=? WHERE id=?", (FUTURE, h))
    assert _want(conn).title == "Queued — retrying after a failed attempt"


def test_details_url_points_at_the_queue_comment(conn):
    _pr(conn)
    _head(conn)
    with tx(conn):
        kv_set(conn, f"queue.comment_id:{REPO}#1", "4242")
    assert _want(conn).details_url == f"https://github.com/{REPO}/pull/1#issuecomment-4242"


def test_running_names_the_step(conn):
    _pr(conn)
    _run(conn, _head(conn, status="running"), "phase2")
    w = _want(conn)
    assert (w.status, w.title, w.started_at) == ("in_progress", "Reviewing — Phase 2 review", T0)


@pytest.mark.parametrize(
    ("event", "conclusion"),
    [
        ("APPROVE", "success"),
        ("COMMENT", "neutral"),
        ("REQUEST_CHANGES", "failure"),
        ("CHANGES_REQUESTED", "failure"),
    ],
)
def test_final_verdict_maps_to_conclusion(conn, event, conclusion):
    _pr(conn)
    _head(conn, status="done")
    _review(conn, event)
    w = _want(conn)
    assert (w.status, w.conclusion) == ("completed", conclusion)
    assert w.details_url == f"https://github.com/{REPO}/pull/1#pullrequestreview-555"


def test_final_verdict_stands_while_a_follow_up_runs(conn):
    _pr(conn)
    _run(conn, _head(conn, status="running", trigger="review_reply"), "converse")
    _review(conn, "APPROVE")
    w = _want(conn)
    assert (w.status, w.conclusion) == ("completed", "success")
    assert "follow-up pass" in w.summary


def test_latest_verdict_wins(conn):
    _pr(conn)
    _head(conn, status="done")
    _review(conn, "REQUEST_CHANGES", at=T0)
    _review(conn, "APPROVE", at="2026-11-02T00:00:00Z", review_id=556)
    assert _want(conn).conclusion == "success"


def test_phase1_gated_run_ends_on_its_preliminary_review(conn):
    _pr(conn)
    h = _head(conn, status="running")
    _run(conn, h, "verify1")
    _review(conn, "REQUEST_CHANGES", phase="preliminary")
    assert _want(conn).status == "in_progress", "mid-run, Phase 2 may still follow"
    with tx(conn):
        conn.execute("UPDATE heads SET status='done', finished_at=? WHERE id=?", (T0, h))
    w = _want(conn)
    assert (w.status, w.conclusion) == ("completed", "failure")
    assert "Phase 2 did not run" in w.summary


def test_carried_review_stands_for_the_new_commit(conn):
    _pr(conn)
    _head(conn, sha=OLD, status="superseded")
    _review(conn, "APPROVE", sha=OLD)
    _head(conn, status="done", carried=OLD)
    w = _want(conn)
    assert w.conclusion == "success" and OLD[:12] in w.summary


def test_service_failure_is_not_a_finding(conn):
    _pr(conn)
    _head(conn, status="failed", reason="lane exit 1")
    w = _want(conn)
    assert (w.status, w.conclusion) == ("completed", "action_required")
    assert "@thepastaclaw review" in w.summary


def test_skip_label(conn):
    from reviewsys.ingest import SKIP_LABEL_REASON

    _pr(conn, skip=1)
    h = _head(conn, status="closed", reason=SKIP_LABEL_REASON)
    assert _want(conn).conclusion == "skipped"
    # someone asked for a review anyway: the check follows that review, not the label
    with tx(conn):
        conn.execute("UPDATE heads SET status='running', trigger='mention' WHERE id=?", (h,))
    _run(conn, h, "phase1")
    assert _want(conn).status == "in_progress"
    with tx(conn):
        conn.execute("UPDATE heads SET status='failed' WHERE id=?", (h,))
    assert _want(conn).conclusion == "action_required", "a failed requested review is not green"


def test_drafts_get_no_check(conn):
    _pr(conn, draft=1)
    assert _want(conn) is None
    _head(conn, status="closed", reason="pr_draft")
    assert _want(conn) is None
    assert checks.closed_out(conn, REPO, 1, SHA).title == "Not reviewed: the PR is a draft"


def test_no_head_yet_says_nothing(conn):
    _pr(conn)
    assert _want(conn) is None


def test_old_commit_and_closed_pr_are_cancelled(conn):
    _pr(conn, sha=SHA)
    _head(conn, sha=OLD, status="superseded")
    assert _want(conn, OLD).conclusion == "cancelled"
    _review(conn, "COMMENT", sha=OLD)
    assert _want(conn, OLD).conclusion == "neutral", "a reviewed commit keeps its verdict"
    _pr(conn, state="closed")
    _head(conn, status="closed", reason="pr closed")
    assert _want(conn).title == "Not reviewed: the PR was closed"


# ---- the reconciler ----


@pytest.fixture
def github():
    return GitHubChecks()


class GitHubReviews:
    def __init__(self):
        self.review = {
            "id": 555,
            "user": {"login": "thepastaclaw"},
            "commit_id": SHA,
            "state": "CHANGES_REQUESTED",
        }
        self.timeline = []
        self.calls = []
        self.fail = None

    def __call__(self, method, ep, body):
        assert method == "GET", "dismissal reconciliation must not post reviews or enqueue work"
        self.calls.append(ep)
        if self.fail:
            return self.fail
        if "/reviews/" in ep:
            return self.review
        assert "/timeline?per_page=100" in ep
        return self.timeline

    def dismiss(self, *, sha=SHA, review_id=555):
        self.review.update(state="DISMISSED", commit_id=sha, id=review_id)
        self.timeline = [
            {
                "id": 1234,
                "event": "review_dismissed",
                "created_at": "2026-11-09T11:30:13Z",
                "actor": {"login": "lklimek", "type": "User"},
                "performed_via_github_app": None,
                "dismissed_review": {
                    "review_id": review_id,
                    "state": "changes_requested",
                    "dismissal_message": "This edge case is not reachable in pracice.",
                },
            }
        ]


@pytest.fixture
def review_api():
    return GitHubReviews()


@pytest.fixture
def rec(cfg, github, review_api):
    return checks.Reconciler(cfg, FakeApp(ScriptedGh(github)), review_gh=ScriptedGh(review_api))  # type: ignore[arg-type]


def test_create_then_update_in_place(conn, rec, github):
    _pr(conn)
    h = _head(conn)
    assert rec.reconcile(conn)["created"] == 1
    ((cid, run),) = github.runs.items()
    assert run["name"] == "PastaClaw review" and run["head_sha"] == SHA
    assert run["status"] == "queued" and "conclusion" not in run
    assert rec.reconcile(conn)["unchanged"] == 1, "nothing changed, nothing written"
    with tx(conn):
        conn.execute("UPDATE heads SET status='running' WHERE id=?", (h,))
    _run(conn, h, "phase1")
    assert rec.reconcile(conn)["updated"] == 1
    assert github.runs[cid]["status"] == "in_progress"
    with tx(conn):
        conn.execute("UPDATE heads SET status='done' WHERE id=?", (h,))
    _review(conn, "APPROVE")
    rec.reconcile(conn)
    assert len(github.runs) == 1
    assert github.runs[cid]["conclusion"] == "success" and github.runs[cid]["completed_at"] == T0
    row = conn.execute("SELECT status, conclusion FROM check_runs").fetchone()
    assert tuple(row) == ("completed", "success")


def test_reopening_a_completed_check_creates_a_new_run(conn, rec, github):
    _pr(conn)
    h = _head(conn, status="failed")
    rec.reconcile(conn)
    assert [r["conclusion"] for r in github.runs.values()] == ["action_required"]
    with tx(conn):
        conn.execute(
            "UPDATE heads SET status='queued', trigger='mention', priority=1 WHERE id=?", (h,)
        )
    rec.reconcile(conn)
    assert len(github.runs) == 2 and github.runs[101]["status"] == "queued"
    assert conn.execute("SELECT check_id FROM check_runs").fetchone()[0] == 101


def test_superseded_open_check_is_closed_out(conn, rec, github):
    _pr(conn, sha=OLD)
    _head(conn, sha=OLD)
    rec.reconcile(conn)
    _pr(conn, sha=SHA)
    with tx(conn):
        conn.execute("UPDATE heads SET status='superseded' WHERE sha=?", (OLD,))
    _head(conn, sha=SHA)
    rec.reconcile(conn)
    assert github.runs[100]["conclusion"] == "cancelled"
    assert github.runs[101]["status"] == "queued"


def test_untouched_old_commit_gets_no_check(conn, rec, github):
    _pr(conn, sha=SHA)
    _head(conn, sha=OLD, status="superseded")
    rec.reconcile(conn)
    assert github.runs == {}, "a commit replaced before it had a check never gets one"


def test_repo_without_the_app_is_skipped(conn, cfg, github):
    app = FakeApp(None)
    rec = checks.Reconciler(cfg, app)  # type: ignore[arg-type]
    _pr(conn)
    _head(conn)
    _pr(conn, number=2)
    _head(conn, number=2)
    assert rec.reconcile(conn)["skipped_repos"] == 1
    assert github.runs == {} and app.lookups == 1, "looked up once per pass, not per PR"


def test_transient_failure_is_retried_next_pass(conn, rec, github):
    _pr(conn)
    _head(conn)
    _pr(conn, number=2)
    _head(conn, number=2)
    github.fail = RuntimeError("HTTP 502 bad gateway")
    stats = rec.reconcile(conn)
    assert stats["deferred"] == 1, "the pass stops at the first transient failure"
    assert conn.execute("SELECT COUNT(*) FROM check_runs").fetchone()[0] == 0
    github.fail = None
    assert rec.reconcile(conn)["created"] == 0, "backs off before the next try"
    rec.retry_at = now_dt()
    assert rec.reconcile(conn)["created"] == 2


def test_refused_write_is_recorded_once(conn, rec, github):
    _pr(conn)
    _head(conn)
    github.fail = RuntimeError("gh: Resource not accessible by integration (HTTP 403)")
    assert rec.reconcile(conn)["failed"] == 1
    assert rec.reconcile(conn)["failed"] == 0, "the same refused check is not retried"
    events = conn.execute("SELECT COUNT(*) FROM events WHERE kind='check.failed'").fetchone()[0]
    assert events == 1 and rec.app.forgotten == [REPO]


def test_app_auth_failure_stops_the_pass(conn, cfg):

    class Broken(FakeApp):
        def gh_for(self, repo):
            raise AppAuthError("cannot sign the App JWT")

    rec = checks.Reconciler(cfg, Broken(None))  # type: ignore[arg-type]
    _pr(conn)
    _head(conn)
    rec.reconcile(conn)
    rec.reconcile(conn)
    assert (
        conn.execute("SELECT COUNT(*) FROM events WHERE kind='check.auth_failed'").fetchone()[0]
        == 1
    )


def test_writes_per_pass_are_capped(conn, rec, github, monkeypatch):
    monkeypatch.setattr(checks, "MAX_WRITES_PER_PASS", 2)
    for n in range(1, 4):
        _pr(conn, str(n) * 40, number=n)
        _head(conn, str(n) * 40, number=n)
    stats = rec.reconcile(conn)
    assert (stats["created"], stats["deferred"], len(github.runs)) == (2, 1, 2)
    rec.reconcile(conn)
    assert len(github.runs) == 3


def test_excluded_repo(conn, cfg, github):
    cfg = cfg_mod.Config(
        **{
            **{f: getattr(cfg, f) for f in cfg.__dataclass_fields__},
            "checks": cfg_mod.ChecksConfig(exclude_repos=(REPO,)),
        }
    )
    rec = checks.Reconciler(cfg, FakeApp(ScriptedGh(github)))  # type: ignore[arg-type]
    _pr(conn)
    _head(conn)
    rec.reconcile(conn)
    assert github.runs == {}


# ---- daemon and config ----


def _with_checks(cfg, **kw):
    fields = {f: getattr(cfg, f) for f in cfg.__dataclass_fields__}
    return cfg_mod.Config(**{**fields, "checks": cfg_mod.ChecksConfig(**kw)})


def test_daemon_runs_the_task_only_when_enabled(cfg, conn):
    names = lambda d: {t.name for t in d.tasks}  # noqa: E731
    assert "checks" not in names(Daemon(cfg, conn, gh=Gh(), spawn=True))
    on = _with_checks(cfg, enabled=True, app_id="123")
    assert "checks" in names(Daemon(on, conn, gh=Gh(), spawn=True))
    assert "checks" not in names(Daemon(on, conn, gh=Gh(), spawn=False)), "never from shadow"


def test_config_parses_and_requires_app_id(tmp_path):
    assert cfg_mod._checks({}).enabled is False
    c = cfg_mod._checks({"enabled": True, "app_id": 42, "private_key": "~/k.pem"})
    assert c.app_id == "42" and c.private_key == Path("~/k.pem").expanduser()
    with pytest.raises(ValueError, match="app_id"):
        cfg_mod._checks({"enabled": True})


# ---- GitHub App auth ----


def _app(respond, *, clock):
    app_gh = ScriptedGh(respond)
    tokens: list[str] = []

    def client(token: str) -> Gh:
        tokens.append(token)
        return Gh()

    auth = AppAuth(
        "123",
        Path("/k.pem"),
        app_gh=app_gh,
        token_client=client,
        signer=lambda data, key: b"sig",
        clock=clock,
    )
    return auth, app_gh, tokens


def test_jwt_claims_and_bearer_header():
    now = [1_800_000_000.0]
    auth, app_gh, _ = _app(lambda m, ep, b: {"id": 1, "slug": "x"}, clock=lambda: now[0])
    token = auth.jwt()
    header, payload, sig = token.split(".")
    claims = json.loads(base64.urlsafe_b64decode(payload + "=="))
    assert claims == {"iat": 1_800_000_000 - 60, "exp": 1_800_000_000 + 540, "iss": "123"}
    assert json.loads(base64.urlsafe_b64decode(header + "=="))["alg"] == "RS256"
    assert base64.urlsafe_b64decode(sig + "==") == b"sig"
    assert auth.jwt() == token, "cached"
    now[0] += 500
    assert auth.jwt() != token, "renewed before it expires"
    auth.app()
    assert f"Authorization: Bearer {auth.jwt()}" in app_gh.calls[-1][3]


def test_installation_lookup_and_token_cache():
    now = [1_800_000_000.0]
    installed = {"dashpay/platform": 7}

    def respond(method, ep, body):
        if ep.endswith("/installation"):
            repo = ep.removeprefix("repos/").removesuffix("/installation")
            return (
                {"id": installed[repo]}
                if repo in installed
                else RuntimeError("gh: Not Found (HTTP 404)")
            )
        if ep == "app/installations/7/access_tokens":
            assert method == "POST" and body == {"permissions": {"checks": "write"}}
            return {"token": f"ghs_{int(now[0])}", "expires_at": "2030-01-01T00:00:00Z"}
        raise AssertionError(ep)

    auth, app_gh, tokens = _app(respond, clock=lambda: now[0])
    assert auth.gh_for("dashpay/platform") is not None
    assert auth.gh_for("dashpay/platform") is not None
    assert tokens == ["ghs_1800000000"] * 2
    assert sum(1 for c in app_gh.calls if "access_tokens" in c[1]) == 1, "token reused"
    assert auth.gh_for("dashpay/dash") is None
    assert auth.gh_for("dashpay/dash") is None
    lookups = [c for c in app_gh.calls if c[1] == "repos/dashpay/dash/installation"]
    assert len(lookups) == 1, "a missing installation is cached"
    installed["dashpay/dash"] = 7
    now[0] += 3601
    assert auth.gh_for("dashpay/dash") is not None, "looked up again after an hour"


def test_installation_lookup_transient_failure_raises():
    from reviewsys.models import FailKind, ReviewError

    auth, _, _ = _app(lambda m, ep, b: RuntimeError("HTTP 502 timeout"), clock=lambda: 0.0)
    with pytest.raises(ReviewError) as exc:
        auth.installation(REPO)
    assert exc.value.kind is FailKind.INFRA


def test_openssl_signature_verifies(tmp_path):
    import shutil

    from reviewsys.ghapp import openssl_sign

    if not shutil.which("openssl"):
        pytest.skip("no openssl")
    key, pub = tmp_path / "k.pem", tmp_path / "k.pub"
    subprocess.run(["openssl", "genrsa", "-out", str(key), "2048"], check=True, capture_output=True)
    subprocess.run(
        ["openssl", "rsa", "-in", str(key), "-pubout", "-out", str(pub)],
        check=True,
        capture_output=True,
    )
    sig = tmp_path / "sig"
    sig.write_bytes(openssl_sign(b"header.payload", key))
    data = tmp_path / "data"
    data.write_bytes(b"header.payload")
    ok = subprocess.run(
        ["openssl", "dgst", "-sha256", "-verify", str(pub), "-signature", str(sig), str(data)],
        capture_output=True,
    )
    assert ok.returncode == 0, ok.stdout + ok.stderr


# ---- review findings: transitions through the real ingest / CI code ----


def test_a_mention_retries_a_failed_review(conn, cfg):
    from reviewsys.ingest import enqueue_head
    from reviewsys.models import Trigger

    _pr(conn)
    h = _head(conn, status="failed", reason="lane exit 1")
    assert _want(conn).conclusion == "action_required"
    with tx(conn):
        assert enqueue_head(conn, cfg, REPO, 1, SHA, Trigger.MENTION) == "requeued"
    assert conn.execute("SELECT status FROM heads WHERE id=?", (h,)).fetchone()[0] == "queued"
    assert _want(conn).status == "queued"


def test_a_mention_retries_a_review_that_ended_without_a_verdict(conn, cfg):
    from reviewsys.ingest import enqueue_head
    from reviewsys.models import Trigger

    _pr(conn)
    _head(conn, status="done")
    with tx(conn):
        assert enqueue_head(conn, cfg, REPO, 1, SHA, Trigger.MENTION) == "requeued"
    _review(conn, "APPROVE")
    with tx(conn):
        conn.execute("UPDATE heads SET status='done'")
        assert enqueue_head(conn, cfg, REPO, 1, SHA, Trigger.MENTION) == "noop", (
            "a reviewed commit is not reviewed again on a mention"
        )


def test_preliminary_comment_does_not_claim_blockers(conn):
    _pr(conn)
    _head(conn, status="done")
    _review(conn, "COMMENT", phase="preliminary")
    w = _want(conn)
    assert w.conclusion == "neutral" and "blocking" not in w.summary


def test_converted_to_draft_closes_the_open_check(conn, rec, github):
    _pr(conn)
    h = _head(conn)
    rec.reconcile(conn)
    _pr(conn, draft=1)
    with tx(conn):
        conn.execute("UPDATE heads SET status='closed', reason='pr_draft' WHERE id=?", (h,))
    rec.reconcile(conn)
    run = github.runs[100]
    assert (run["status"], run["conclusion"]) == ("completed", "cancelled")
    assert run["output"]["title"] == "Not reviewed: the PR is a draft"


def test_revert_to_a_superseded_commit_does_not_leave_it_running(conn, rec, github):
    _pr(conn)
    h = _head(conn, status="running")
    _run(conn, h, "phase1")
    rec.reconcile(conn)
    with tx(conn):
        conn.execute("UPDATE heads SET status='superseded' WHERE id=?", (h,))
    rec.reconcile(conn)  # GitHub's head is this commit again, but nothing will review it
    assert github.runs[100]["status"] == "completed"


def test_late_review_of_a_cancelled_commit_lands(conn, rec, github):
    _pr(conn, sha=OLD)
    _head(conn, sha=OLD, status="running")
    rec.reconcile(conn)
    _pr(conn, sha=SHA)
    with tx(conn):
        conn.execute("UPDATE heads SET status='superseded' WHERE sha=?", (OLD,))
    rec.reconcile(conn)
    assert github.runs[100]["conclusion"] == "cancelled"
    _review(conn, "APPROVE", sha=OLD, at="2999-01-01T00:00:00Z")  # past verify2, published
    rec.reconcile(conn)
    assert github.runs[100]["conclusion"] == "success"


def test_patch_of_a_vanished_check_creates_a_new_one(conn, rec, github):
    _pr(conn)
    _head(conn)
    rec.reconcile(conn)
    github.runs.clear()  # e.g. a new App: the stored id is not ours to write
    with tx(conn):
        conn.execute("UPDATE heads SET status='running'")
    rec.reconcile(conn)
    assert list(github.runs) == [100] and github.runs[100]["status"] == "in_progress"
    assert conn.execute("SELECT check_id FROM check_runs").fetchone()[0] == 100


def test_lost_create_response_is_adopted_not_duplicated(conn, rec, github):
    _pr(conn)
    _head(conn)
    github.runs[100] = {"name": "PastaClaw review", "head_sha": SHA, "status": "queued"}
    rec.reconcile(conn)  # no row: an earlier POST timed out after GitHub created the run
    assert list(github.runs) == [100]
    assert "app_id=1" in github.listed[0] and "check_name=PastaClaw%20review" in github.listed[0]
    assert conn.execute("SELECT check_id FROM check_runs").fetchone()[0] == 100


def test_backfilled_verdict_starts_when_it_completed(conn, rec, github):
    _pr(conn)
    _head(conn, status="done")
    _review(conn, "APPROVE")
    rec.reconcile(conn)
    assert github.runs[100]["started_at"] == github.runs[100]["completed_at"] == T0


def test_one_broken_installation_does_not_stop_other_repos(conn, cfg, github):
    from reviewsys.models import FailKind, ReviewError

    class OneBroken(FakeApp):
        def gh_for(self, repo):
            if repo == "dashpay/dash":
                raise ReviewError(FailKind.CONTRACT, "gh: Not Found (HTTP 404)")
            return self.gh

    app = OneBroken(ScriptedGh(github))
    rec = checks.Reconciler(cfg, app)  # type: ignore[arg-type]
    with tx(conn):
        conn.execute(
            "INSERT INTO heads (repo, number, sha, trigger, status, queued_at, eligible_at)"
            " VALUES ('dashpay/dash', 9, ?, 'mention', 'queued', ?, ?)",
            (OLD, T0, T0),
        )
    _pr(conn)
    _head(conn)
    assert rec.reconcile(conn)["created"] == 1
    assert app.forgotten == ["dashpay/dash"]


def test_disabled_repos_are_not_backfilled(conn, cfg, rec, github):
    with tx(conn):
        conn.execute(
            "INSERT INTO prs (repo, number, head_sha, state, updated_at) VALUES"
            " ('dashpay/grovedb', 3, ?, 'open', ?)",
            (SHA, T0),
        )
        conn.execute(
            "INSERT INTO heads (repo, number, sha, trigger, status, queued_at, eligible_at)"
            " VALUES ('dashpay/grovedb', 3, ?, 'new_pr', 'done', ?, ?)",
            (SHA, T0, T0),
        )
    assert ("dashpay/grovedb", 3, SHA) not in checks._candidates(conn, cfg)


def test_step_changes_write_no_events(conn, rec, github):
    _pr(conn)
    h = _head(conn, status="running")
    _run(conn, h, "phase1")
    rec.reconcile(conn)
    with tx(conn):
        conn.execute("UPDATE runs SET phase='phase2'")
    rec.reconcile(conn)
    assert github.runs[100]["output"]["title"] == "Reviewing — Phase 2 review"
    kinds = [r[0] for r in conn.execute("SELECT detail FROM events WHERE kind='check.synced'")]
    assert len(kinds) == 1


def test_ci_watch_ignores_the_review_check_by_name(cfg, monkeypatch):
    from reviewsys import ci, github

    cfg = _with_checks(cfg, name="Bot verdict")  # matches no ignore pattern on its own
    monkeypatch.setattr(
        github,
        "ci_checks",
        lambda gh, repo, sha: {
            "checks": [
                {"name": "Bot verdict", "status": "in_progress"},
                {"name": "build", "status": "completed", "conclusion": "success"},
            ]
        },
    )
    assert ci.read(Gh(), cfg, REPO, SHA).pending == []


def test_plan_reads_no_github(conn, cfg):
    _pr(conn)
    _head(conn)
    (row,) = checks.plan(conn, cfg)
    assert row["status"] == "queued" and row["check_id"] is None and not row["in_sync"]


def test_ready_again_on_the_same_commit_requeues_the_review(conn, cfg):
    from reviewsys.ingest import DRAFT_REASON, enqueue_head
    from reviewsys.models import Trigger

    _pr(conn)
    h = _head(conn, status="closed", reason=DRAFT_REASON)  # it went back to draft
    with tx(conn):
        assert enqueue_head(conn, cfg, REPO, 1, SHA, Trigger.NEW_PUSH) == "requeued"
    assert conn.execute("SELECT status FROM heads WHERE id=?", (h,)).fetchone()[0] == "queued"
    assert _want(conn).status == "queued"


def test_labelled_commit_without_a_head_is_skipped(conn):
    _pr(conn, skip=1)
    assert _want(conn).conclusion == "skipped"


def test_disabled_repo_open_check_is_closed_and_nothing_new_created(conn, cfg, rec, github):
    with tx(conn):
        conn.execute(
            "INSERT INTO prs (repo, number, head_sha, state, updated_at) VALUES"
            " ('dashpay/grovedb', 3, ?, 'open', ?), ('dashpay/grovedb', 4, ?, 'open', ?)",
            (SHA, T0, OLD, T0),
        )
        conn.execute(
            "INSERT INTO heads (repo, number, sha, trigger, status, queued_at, eligible_at)"
            " VALUES ('dashpay/grovedb', 3, ?, 'new_pr', 'queued', ?, ?),"
            " ('dashpay/grovedb', 4, ?, 'new_pr', 'queued', ?, ?)",
            (SHA, T0, T0, OLD, T0, T0),
        )
        conn.execute(
            "INSERT INTO check_runs VALUES ('dashpay/grovedb', 3, ?, 50, 'queued', NULL, 'x', ?)",
            (SHA, T0),
        )
    github.runs[50] = {"head_sha": SHA, "status": "queued"}
    rec.reconcile(conn)
    assert github.runs[50]["conclusion"] == "cancelled"
    assert list(github.runs) == [50], "no check for the other frozen PR"


def test_late_review_stamped_before_the_cancel_still_lands(conn, rec, github):
    _pr(conn, sha=OLD)
    _head(conn, sha=OLD, status="running")
    rec.reconcile(conn)
    _pr(conn, sha=SHA)
    with tx(conn):
        conn.execute("UPDATE heads SET status='superseded' WHERE sha=?", (OLD,))
    rec.reconcile(conn)
    written = conn.execute("SELECT updated_at FROM check_runs WHERE sha=?", (OLD,)).fetchone()[0]
    stamped = (parse_ts(written) - timedelta(seconds=30)).isoformat().replace("+00:00", "Z")
    _review(conn, "APPROVE", sha=OLD, at=stamped)
    rec.reconcile(conn)
    assert github.runs[100]["conclusion"] == "success"


# ---- explicit maintainer override, independent of automated findings ----


def test_maintainer_dismissal_repairs_failure_without_new_review(conn, rec, github, review_api):
    _pr(conn)
    _head(conn, status="done")
    _review(conn, "REQUEST_CHANGES")
    rec.reconcile(conn)
    assert github.runs[100]["conclusion"] == "failure"
    # The database verdict and its digest have not changed. A later inline-only bot review
    # and a human approval must not hide dismissal of the substantive review linked here.
    review_api.dismiss()
    review_api.timeline += [
        {"event": "reviewed", "state": "approved", "actor": {"login": "lklimek"}},
        {"event": "reviewed", "state": "commented", "actor": {"login": "thepastaclaw"}},
    ]
    assert rec.reconcile(conn)["updated"] == 1
    run = github.runs[100]
    assert (run["status"], run["conclusion"]) == ("completed", "neutral")
    assert len(github.runs) == 1
    assert run["output"]["title"] == "Review dismissed — maintainer override"
    summary = run["output"]["summary"]
    assert "@lklimek" in summary
    assert "This edge case is not reachable in pracice." in summary
    assert SHA in summary and "does not claim" in summary
    assert "technical validation passed" in summary
    assert run["completed_at"] == "2026-11-09T11:30:13Z"
    assert run["details_url"].endswith("#pullrequestreview-555")
    assert conn.execute("SELECT event FROM reviews").fetchone()[0] == "REQUEST_CHANGES"
    assert conn.execute("SELECT status FROM heads").fetchone()[0] == "done"
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    evidence = json.loads(
        conn.execute("SELECT detail FROM events WHERE kind='check.review_dismissed'").fetchone()[0]
    )
    assert evidence["timeline_event_id"] == 1234 and evidence["actor"] == "lklimek"


def test_dismissal_survives_restart_and_stale_github_reads(conn, cfg, rec, github, review_api):
    _pr(conn)
    _head(conn, status="done")
    _review(conn, "REQUEST_CHANGES")
    review_api.dismiss()
    rec.reconcile(conn)
    review_api.review["state"] = "CHANGES_REQUESTED"  # stale API snapshot
    review_api.calls.clear()
    restarted = checks.Reconciler(cfg, rec.app, review_gh=ScriptedGh(review_api))
    assert restarted.reconcile(conn)["unchanged"] == 1
    assert review_api.calls == []
    assert github.runs[100]["conclusion"] == "neutral"
    assert (
        conn.execute("SELECT COUNT(*) FROM events WHERE kind='check.review_dismissed'").fetchone()[
            0
        ]
        == 1
    )


def test_dismissal_retries_failed_check_write_from_persisted_evidence(
    conn, rec, github, review_api
):
    _pr(conn)
    _head(conn, status="done")
    _review(conn, "REQUEST_CHANGES")
    rec.reconcile(conn)
    review_api.dismiss()
    github.fail = RuntimeError("HTTP 502 bad gateway")
    assert rec.reconcile(conn)["deferred"] == 1
    assert _want(conn).conclusion == "neutral", "evidence committed before publication"
    github.fail = None
    review_api.fail = RuntimeError("HTTP 502 no more reads")
    rec.retry_at = now_dt()
    assert rec.reconcile(conn)["updated"] == 1
    assert github.runs[100]["conclusion"] == "neutral"


@pytest.mark.parametrize("mismatch", ["review_id", "sha", "author"])
def test_dismissal_must_match_exact_review_and_head(conn, rec, github, review_api, mismatch):
    _pr(conn)
    _head(conn, status="done")
    _review(conn, "REQUEST_CHANGES")
    review_api.dismiss()
    if mismatch == "review_id":
        review_api.review["id"] = 556
    elif mismatch == "sha":
        review_api.review["commit_id"] = OLD
    else:
        review_api.review["user"]["login"] = "someone-else"
    rec.reconcile(conn)
    assert github.runs[100]["conclusion"] == "failure"
    assert (
        conn.execute("SELECT COUNT(*) FROM events WHERE kind='check.review_dismissed'").fetchone()[
            0
        ]
        == 0
    )


@pytest.mark.parametrize(
    "unrelated",
    [
        "human_approval",
        "resolved_thread",
        "different_review",
        "bot_dismissal",
        "self_dismissal",
        "automatic_stale",
        "app_dismissal",
        "no_actor",
        "no_event",
    ],
)
def test_only_explicit_authorized_human_dismissal_overrides(
    conn, rec, github, review_api, unrelated
):
    _pr(conn)
    _head(conn, status="done")
    _review(conn, "REQUEST_CHANGES")
    review_api.dismiss()
    item = review_api.timeline[0]
    if unrelated == "human_approval":
        review_api.review["state"] = "CHANGES_REQUESTED"
        item["event"] = "reviewed"
        item["state"] = "approved"
    elif unrelated == "resolved_thread":
        item["event"] = "review_thread_resolved"
    elif unrelated == "different_review":
        item["dismissed_review"]["review_id"] = 999
    elif unrelated == "bot_dismissal":
        item["actor"] = {"login": "a-bot[bot]", "type": "Bot"}
    elif unrelated == "self_dismissal":
        item["actor"]["login"] = "thepastaclaw"
    elif unrelated == "automatic_stale":
        item["dismissed_review"]["dismissal_commit_id"] = OLD
    elif unrelated == "app_dismissal":
        item["performed_via_github_app"] = {"id": 1}
    elif unrelated == "no_actor":
        item["actor"] = None
    else:
        review_api.timeline = []
    rec.reconcile(conn)
    assert github.runs[100]["conclusion"] == "failure"


def test_dismissal_is_not_inherited_by_a_changed_head_or_other_checks(
    conn, rec, github, review_api
):
    _pr(conn, sha=OLD)
    _head(conn, sha=OLD, status="done")
    _review(conn, "REQUEST_CHANGES", sha=OLD)
    review_api.dismiss(sha=OLD)
    github.runs[42] = {
        "name": "build",
        "head_sha": OLD,
        "status": "completed",
        "conclusion": "failure",
    }
    # Existing recorded review check; don't adopt the fake build check.
    github.runs[100] = {
        "name": "PastaClaw review",
        "head_sha": OLD,
        "status": "completed",
        "conclusion": "failure",
    }
    with tx(conn):
        conn.execute(
            "INSERT INTO check_runs VALUES (?,?,?,100,'completed','failure','old',?)",
            (REPO, 1, OLD, T0),
        )
    rec.reconcile(conn)
    assert github.runs[100]["conclusion"] == "neutral"
    assert github.runs[42]["conclusion"] == "failure"
    _pr(conn)
    _head(conn)
    rec.reconcile(conn)
    assert _want(conn).status == "queued"
    assert _want(conn, OLD).conclusion == "neutral"


def test_dismissal_of_carried_review_follows_unchanged_diff(conn, rec, github, review_api):
    _pr(conn)
    _head(conn, sha=OLD, status="superseded")
    _review(conn, "REQUEST_CHANGES", sha=OLD)
    _head(conn, status="done", carried=OLD)
    review_api.dismiss(sha=OLD)
    rec.reconcile(conn)
    assert github.runs[100]["head_sha"] == SHA
    assert github.runs[100]["conclusion"] == "neutral"
    assert OLD in github.runs[100]["output"]["summary"]


def test_phase1_gate_dismissal_is_nonblocking(conn, rec, github, review_api):
    _pr(conn)
    _head(conn, status="done")
    _review(conn, "REQUEST_CHANGES", phase="preliminary")
    review_api.dismiss()
    rec.reconcile(conn)
    assert github.runs[100]["conclusion"] == "neutral"
    assert "technical validation passed" in github.runs[100]["output"]["summary"]


def test_canonical_only_verdict_update_keeps_exact_review_binding(conn, rec, github, review_api):
    _pr(conn)
    _head(conn, status="done")
    _review(conn, "COMMENT", review_id=555)
    _review(conn, "REQUEST_CHANGES", review_id=None, at="2026-11-02T00:00:00Z")
    review_api.dismiss()
    rec.reconcile(conn)
    assert github.runs[100]["conclusion"] == "neutral"
    assert github.runs[100]["details_url"].endswith("#pullrequestreview-555")


def test_read_failure_never_reasserts_cached_failure(conn, rec, github, review_api):
    _pr(conn)
    _head(conn, status="done")
    _review(conn, "REQUEST_CHANGES")
    rec.reconcile(conn)
    previous = dict(github.runs[100])
    review_api.fail = RuntimeError("HTTP 502 timeout")
    assert rec.reconcile(conn)["deferred"] == 1
    assert github.runs[100] == previous


def test_live_dismissal_reads_are_bounded_and_rotate(conn, rec, review_api, monkeypatch):
    monkeypatch.setattr(checks, "MAX_READS_PER_PASS", 1)
    for n in (1, 2):
        _pr(conn, number=n)
        _head(conn, status="done", number=n)
        _review(conn, "REQUEST_CHANGES", number=n)
    assert rec.reconcile(conn)["deferred"] == 1
    assert len(review_api.calls) == 1 and "/pulls/1/" in review_api.calls[0]
    rec.reconcile(conn)
    assert len(review_api.calls) == 2 and "/pulls/2/" in review_api.calls[1]


def test_dismissal_timeline_is_paginated(conn, rec, github, review_api):
    _pr(conn)
    _head(conn, status="done")
    _review(conn, "REQUEST_CHANGES")
    review_api.dismiss()
    # gh --paginate --slurp returns pages; the matching event is not on the first one.
    review_api.timeline = [[{"event": "commented"}] * 100, review_api.timeline]
    rec.reconcile(conn)
    assert github.runs[100]["conclusion"] == "neutral"
    call = rec.review_gh.calls[-1][3]
    assert "--paginate" in call and "--slurp" in call


def test_dismissal_without_reason_does_not_invent_one(conn, rec, github, review_api):
    _pr(conn)
    _head(conn, status="done")
    _review(conn, "REQUEST_CHANGES")
    review_api.dismiss()
    review_api.timeline[0]["dismissed_review"]["dismissal_message"] = ""
    rec.reconcile(conn)
    assert "(no reason supplied)" in github.runs[100]["output"]["summary"]


def test_follow_up_activity_does_not_reset_dismissal(conn, rec, github, review_api):
    _pr(conn)
    _head(conn, status="done")
    _review(conn, "REQUEST_CHANGES")
    review_api.dismiss()
    rec.reconcile(conn)
    with tx(conn):
        conn.execute("UPDATE heads SET status='running', trigger='review_reply'")
    assert rec.reconcile(conn)["unchanged"] == 1
    assert github.runs[100]["conclusion"] == "neutral"


def test_new_substantive_review_is_not_dismissed_by_old_override(conn, rec, github, review_api):
    _pr(conn)
    _head(conn, status="done")
    _review(conn, "REQUEST_CHANGES")
    review_api.dismiss()
    rec.reconcile(conn)
    _review(conn, "COMMENT", review_id=556, at="2026-11-10T00:00:00Z")
    review_api.review.update(id=556, state="COMMENTED")
    rec.reconcile(conn)
    run = github.runs[100]
    assert run["conclusion"] == "neutral"
    assert "override" not in run["output"]["title"]
    assert run["details_url"].endswith("#pullrequestreview-556")


def test_review_binding_does_not_fall_back_to_another_phase(conn):
    _pr(conn)
    _head(conn, status="done")
    _review(conn, "REQUEST_CHANGES", phase="preliminary", review_id=555)
    _review(conn, "COMMENT", review_id=None, at="2026-11-02T00:00:00Z")
    assert _want(conn).details_url == f"https://github.com/{REPO}/pull/1"
