"""The webhook receiver: only signed review-now and gate-box deliveries make a pass due."""

import dataclasses
import hashlib
import hmac
import json
import urllib.error
import urllib.request
from email.message import Message

import pytest

from reviewsys import hook, watch
from reviewsys.config import WebhookConfig
from reviewsys.daemon import Daemon
from reviewsys.github import GATE_MARKER
from reviewsys.ingest import REVIEW_NOW_LABEL

SECRET = b"s" * 64
REPO = "dashpay/dash"
TICKED = f"{GATE_MARKER}\n- [x] **Request priority review**\n"


def sign(body: bytes, secret: bytes = SECRET) -> str:
    return "sha256=" + hmac.new(secret, body, hashlib.sha256).hexdigest()


def labeled(name=REVIEW_NOW_LABEL, repo=REPO):
    return {
        "action": "labeled",
        "label": {"name": name},
        "pull_request": {"number": 8159},
        "repository": {"full_name": repo},
    }


def edited(body=TICKED, login="thepastaclaw", action="edited", pr=True):
    return {
        "action": action,
        "comment": {"body": body, "user": {"login": login}},
        "issue": {"number": 8159, **({"pull_request": {"url": "u"}} if pr else {})},
        "repository": {"full_name": REPO},
    }


def headers(event, body, delivery="d1", sig=None):
    h = Message()
    h["X-GitHub-Event"] = event
    h["X-GitHub-Delivery"] = delivery
    if sig is not False:
        h["X-Hub-Signature-256"] = sig or sign(body)
    return h


def receiver(hits):
    return hook.Receiver(
        WebhookConfig(enabled=True, listen="127.0.0.1:0"),
        SECRET,
        "thepastaclaw",
        [REPO],
        lambda task, detail: hits.append((task, detail)),
    )


def test_classify_acts_on_two_deliveries_only():
    repos = {REPO.lower()}
    c = hook.classify
    assert c("pull_request", labeled("Pastaclaw:Review-Now"), "thepastaclaw", repos) == watch.INGEST
    assert c("pull_request", labeled("bug"), "thepastaclaw", repos) is None
    assert c("pull_request", labeled(repo="other/repo"), "thepastaclaw", repos) is None
    assert c("pull_request", {**labeled(), "action": "unlabeled"}, "thepastaclaw", repos) is None
    assert c("issue_comment", edited(), "ThePastaClaw", repos) == watch.QUEUE
    assert (
        c("issue_comment", edited(body=TICKED.replace("[x]", "[ ]")), "thepastaclaw", repos) is None
    )
    assert c("issue_comment", edited(login="knst"), "thepastaclaw", repos) is None
    assert c("issue_comment", edited(action="created"), "thepastaclaw", repos) is None
    assert c("issue_comment", edited(pr=False), "thepastaclaw", repos) is None
    assert c("issues", labeled(), "thepastaclaw", repos) is None


def test_nothing_is_trusted_unsigned():
    hits = []
    r = receiver(hits)
    body = json.dumps(labeled()).encode()
    assert r.handle("POST", "/other", headers("pull_request", body), body) == 404
    assert r.handle("GET", "/github/webhook", headers("pull_request", body), body) == 405
    assert (
        r.handle("POST", "/github/webhook", headers("pull_request", body, sig=False), body) == 401
    )
    bad = sign(body, b"x" * 64)
    assert r.handle("POST", "/github/webhook", headers("pull_request", body, sig=bad), body) == 401
    tampered = body.replace(b"8159", b"8160")
    assert r.handle("POST", "/github/webhook", headers("pull_request", body), tampered) == 401
    assert hits == []
    assert hook.signed(SECRET, body, sign(body)) and not hook.signed(SECRET, body, "sha1=x")


def test_a_signed_label_is_a_hit_once_per_delivery():
    hits = []
    r = receiver(hits)
    body = json.dumps(labeled()).encode()
    assert r.handle("POST", "/github/webhook", headers("pull_request", body), body) == 202
    assert hits == [(watch.INGEST, "pull_request.labeled dashpay/dash#8159")]
    assert r.handle("POST", "/github/webhook", headers("pull_request", body), body) == 200
    assert len(hits) == 1  # GitHub's redelivery
    other = json.dumps(labeled("bug")).encode()
    assert r.handle("POST", "/github/webhook", headers("pull_request", other, "d2"), other) == 204
    ping = b'{"zen": "x"}'
    assert r.handle("POST", "/github/webhook", headers("ping", ping, "d3"), ping) == 204
    junk = b"not json"
    assert r.handle("POST", "/github/webhook", headers("push", junk, "d4"), junk) == 400
    gate = json.dumps(edited()).encode()
    assert r.handle("POST", "/github/webhook", headers("issue_comment", gate, "d5"), gate) == 202
    assert hits[-1][0] == watch.QUEUE


def test_the_secret_file_is_the_owners_alone(tmp_path):
    f = tmp_path / "secret"
    f.write_text("a" * 64 + "\n")
    f.chmod(0o644)
    with pytest.raises(hook.HookError, match="chmod 600"):
        hook.load_secret(f)
    f.chmod(0o600)
    assert hook.load_secret(f) == b"a" * 64
    f.write_text("short")
    with pytest.raises(hook.HookError, match="under 32"):
        hook.load_secret(f)
    with pytest.raises(hook.HookError, match="cannot read"):
        hook.load_secret(tmp_path / "missing")


def _post(port, body, sig, length=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/github/webhook",
        data=body,
        method="POST",
        headers={
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "live-1",
            "X-Hub-Signature-256": sig,
            **({"Content-Length": str(length)} if length is not None else {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as res:
            return res.status
    except urllib.error.HTTPError as exc:
        return exc.code


def test_the_daemon_runs_a_delivered_label_as_a_watch_hit(cfg, conn, gh, notifier, tmp_path):
    secret = tmp_path / "secret"
    secret.write_bytes(SECRET)
    secret.chmod(0o600)
    cfg = dataclasses.replace(
        cfg,
        watch_interval_seconds=0,
        webhook=WebhookConfig(enabled=True, listen="127.0.0.1:0", secret_file=secret),
    )
    cfg = dataclasses.replace(cfg, extra={**cfg.extra, "additional_repos": [REPO]})
    d = Daemon(cfg, conn, gh=gh, notifier=notifier)
    r = d.start_webhook()
    assert r is not None and r.server is not None
    port = r.server.server_address[1]
    try:
        body = json.dumps(labeled()).encode()
        assert _post(port, body, sign(body)) == 202
        assert _post(port, body, "sha256=00") == 401
        assert _post(port, b"x" * 10, sign(b"x" * 10), length=hook.MAX_BODY + 1) == 413
    finally:
        r.server.shutdown()
    assert d.wake.is_set() and d._due == {watch.INGEST}
    for task in d.tasks:
        task.fn = lambda: None
    d.run_watched()
    row = conn.execute("SELECT detail FROM events WHERE kind='watch.hit'").fetchone()
    assert row[0] == "ingest via webhook"


def test_a_receiver_that_cannot_start_leaves_the_polls(cfg, conn, gh, notifier, tmp_path):
    cfg = dataclasses.replace(
        cfg, webhook=WebhookConfig(enabled=True, secret_file=tmp_path / "missing")
    )
    assert Daemon(cfg, conn, gh=gh, notifier=notifier).start_webhook() is None
    assert conn.execute("SELECT COUNT(*) FROM events WHERE kind='webhook.failed'").fetchone()[0]
    assert Daemon(cfg, conn, gh=gh, notifier=notifier, spawn=False).start_webhook() is None
