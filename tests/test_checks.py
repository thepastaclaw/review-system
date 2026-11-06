"""Check runs (checks.py) and GitHub App auth (ghapp.py)."""

from __future__ import annotations

import base64
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from reviewsys import checks
from reviewsys import config as cfg_mod
from reviewsys.daemon import Daemon
from reviewsys.db import kv_set, tx
from reviewsys.gh import Gh
from reviewsys.ghapp import AppAuth

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
    """Stands in for AppAuth: one installation client per repo, or none."""

    def __init__(self, gh: Gh | None, installed: bool = True) -> None:
        self.gh = gh
        self.installed = installed
        self.forgotten: list[str] = []
        self.lookups = 0

    def gh_for(self, repo: str) -> Gh | None:
        self.lookups += 1
        return self.gh if self.installed else None

    def forget(self, repo: str) -> None:
        self.forgotten.append(repo)


class GitHubChecks:
    """A repository's check runs as the fake GitHub keeps them."""

    def __init__(self) -> None:
        self.runs: dict[int, dict[str, Any]] = {}
        self.fail: Exception | None = None

    def __call__(self, method: str, ep: str, body: Any) -> Any:
        if self.fail is not None:
            return self.fail
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


def test_skip_label_and_draft(conn):
    _pr(conn, skip=1)
    _head(conn, status="closed", reason="skip label")
    assert _want(conn).conclusion == "skipped"
    _pr(conn, draft=1)
    with tx(conn):
        conn.execute("UPDATE heads SET reason='pr_draft'")
    w = _want(conn)
    assert (w.status, w.title) == ("queued", "Waiting: the PR is a draft")


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


@pytest.fixture
def rec(cfg, github):
    return checks.Reconciler(cfg, FakeApp(ScriptedGh(github)))  # type: ignore[arg-type]


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
    app = FakeApp(ScriptedGh(github), installed=False)
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
    github.fail = RuntimeError("HTTP 502 bad gateway")
    assert rec.reconcile(conn)["deferred"] == 1
    assert conn.execute("SELECT COUNT(*) FROM check_runs").fetchone()[0] == 0
    github.fail = None
    assert rec.reconcile(conn)["created"] == 1


def test_refused_write_is_recorded_once(conn, rec, github):
    _pr(conn)
    _head(conn)
    github.fail = RuntimeError("gh: Resource not accessible by integration (HTTP 403)")
    assert rec.reconcile(conn)["failed"] == 1
    assert rec.reconcile(conn)["failed"] == 0, "the same refused check is not retried"
    events = conn.execute("SELECT COUNT(*) FROM events WHERE kind='check.failed'").fetchone()[0]
    assert events == 1 and rec.app.forgotten == [REPO]


def test_app_auth_failure_stops_the_pass(conn, cfg):
    from reviewsys.models import FailKind, ReviewError

    class Broken(FakeApp):
        def gh_for(self, repo):
            raise ReviewError(FailKind.FATAL, "cannot sign the App JWT")

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
        _pr(conn, number=n)
        _head(conn, number=n)
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
