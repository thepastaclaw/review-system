"""The replay's pinned gh: what a lane (and the harness) sees of the PR is the replayed head as
of the source run's start, and nothing it runs can write to GitHub."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from reviewsys.gh import Gh, read_only_runner
from reviewsys.ghshim import Broker, Pin, install_shim, pinned_runner, real_gh, serve
from reviewsys.models import ReviewError

CUTOFF = "2026-09-20T12:00:00Z"
OLD, NEW = "2026-09-20T11:00:00Z", "2026-09-20T13:00:00Z"
BOT = "thepastaclaw"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def history(tmp_path: Path) -> dict[str, Any]:
    """main: base commit; PR: two commits (the replayed head), then a later push."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "T")
    (repo / "a.txt").write_text("".join(f"line {i}\n" for i in range(30)))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-qb", "pr")
    (repo / "a.txt").write_text(
        "".join(f"line {i}{' changed' if i == 5 else ''}\n" for i in range(30))
    )
    _git(repo, "commit", "-qam", "fix: change line 5\n\nwith a body")
    (repo / "b.txt").write_text("new file\n")
    _git(repo, "add", "b.txt")
    _git(repo, "commit", "-qm", "feat: add b")
    head = _git(repo, "rev-parse", "HEAD")
    (repo / "c.txt").write_text("pushed after the cutoff\n")
    _git(repo, "add", "c.txt")
    _git(repo, "commit", "-qm", "later push")
    return {"repo": repo, "base": base, "head": head, "later": _git(repo, "rev-parse", "HEAD")}


def _pin(history: dict[str, Any], tmp_path: Path, **kw: Any) -> Pin:
    defaults: dict[str, Any] = {
        "repo": "o/r",
        "number": 7,
        "sha": history["head"],
        "cutoff": CUTOFF,
        "merge_base": history["base"],
        "base_ref": "main",
        "base_sha": history["base"],
        "git_dir": str(history["repo"]),
        "real_gh": "gh",
        "ci": {
            "head_sha": history["head"],
            "fetched_at": OLD,
            "checks": [
                {"name": "lint", "status": "completed", "conclusion": "success", "url": "u1"},
                {"name": "test", "status": "completed", "conclusion": "failure", "url": "u2"},
            ],
            "truncated": False,
        },
        "pr": {"title": "Old title", "body": "Old body", "baseRefOid": history["base"]},
        "jq": shutil.which("jq") or "",
        "log": str(tmp_path / "gh-shim.log"),
    }
    return Pin(**{**defaults, **kw})


class Real:
    """The real gh, scripted: (args prefix) -> JSON; records every call."""

    def __init__(self, routes: dict[str, Any] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.routes = routes or {}

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        key = " ".join(args[:2]) if args[0] != "api" else args[1].split("?")[0].lstrip("/")
        if args[:2] == ["api", "graphql"]:
            key = "graphql"
        if key not in self.routes:
            return subprocess.CompletedProcess(args, 1, "", f"unrouted {args}")
        out = self.routes[key]
        if "--paginate" in args and "--slurp" in args:
            out = [out]
        return subprocess.CompletedProcess(
            args, 0, out if isinstance(out, str) else json.dumps(out), ""
        )


def _serve(pin: Pin, real: Real, *args: str) -> subprocess.CompletedProcess[str]:
    return serve(pin, list(args), real=real)


def test_pr_diff_is_the_replayed_range_not_the_live_pr(history, tmp_path):
    pin, real = _pin(history, tmp_path), Real()
    for args in (("pr", "diff", "7", "--repo", "o/r"), ("pr", "diff")):
        res = _serve(pin, real, *args)
        assert res.returncode == 0
        assert "+line 5 changed" in res.stdout and "b.txt" in res.stdout
        assert "c.txt" not in res.stdout  # pushed after the replayed head
    names = _serve(pin, real, "pr", "diff", "7", "--name-only").stdout.split()
    assert names == ["a.txt", "b.txt"]
    assert real.calls == []  # never asked GitHub


def test_pr_checks_come_from_the_source_runs_snapshot(history, tmp_path):
    pin, real = _pin(history, tmp_path), Real()
    res = _serve(pin, real, "pr", "checks", "7", "--repo", "o/r")
    assert res.returncode == 1  # a failing check, as gh exits
    assert res.stdout.splitlines() == ["lint\tpass\t\tu1\t", "test\tfail\t\tu2\t"]
    assert "source run recorded" in res.stderr
    js = _serve(pin, real, "pr", "checks", "--json", "name,bucket")
    assert json.loads(js.stdout) == [
        {"name": "lint", "bucket": "pass"},
        {"name": "test", "bucket": "fail"},
    ]
    # the harness's own CI read is served from the snapshot too
    runs = json.loads(_serve(pin, real, "api", f"repos/o/r/commits/{pin.sha}/check-runs").stdout)
    assert [c["name"] for c in runs["check_runs"]] == ["lint", "test"]
    assert real.calls == []


def test_pr_checks_without_a_snapshot_say_unavailable(history, tmp_path):
    pin, real = _pin(history, tmp_path, ci={"error": "boom"}), Real()
    res = _serve(pin, real, "pr", "checks")
    assert res.returncode == 1 and "unavailable" in res.stderr
    res = _serve(pin, real, "api", f"repos/o/r/commits/{pin.sha}/status")
    assert res.returncode == 1 and "unavailable" in res.stderr
    assert real.calls == []
    # CI of another PR is never served live either
    assert "not pinned" in _serve(pin, real, "pr", "checks", "8").stderr


def test_comments_reviews_and_threads_after_the_cutoff_are_hidden(history, tmp_path):
    pin = _pin(history, tmp_path)
    real = Real(
        {
            "repos/o/r/issues/7/comments": [
                {"id": 1, "user": {"login": "alice"}, "created_at": OLD},
                {"id": 2, "user": {"login": BOT}, "created_at": NEW},  # the bot's own, later
                {"id": 3, "user": {"login": "bob"}, "created_at": CUTOFF},  # at the cutoff
            ],
            "repos/o/r/pulls/7/reviews": [
                {"id": 4, "submitted_at": OLD},
                {"id": 5, "submitted_at": NEW},
            ],
            "graphql": {
                "data": {
                    "repository": {
                        "pullRequest": {
                            "headRefOid": history["later"],
                            "state": "MERGED",
                            "reviewThreads": {
                                "nodes": [
                                    {
                                        "id": "T1",
                                        "comments": {
                                            "nodes": [{"createdAt": OLD}, {"createdAt": NEW}]
                                        },
                                    },
                                    {"id": "T2", "comments": {"nodes": [{"createdAt": NEW}]}},
                                ]
                            },
                        }
                    }
                }
            },
        }
    )
    got = json.loads(_serve(pin, real, "api", "repos/o/r/issues/7/comments").stdout)
    assert [c["id"] for c in got] == [1]
    got = json.loads(
        _serve(pin, real, "api", "/repos/o/r/pulls/7/reviews?per_page=100", "--paginate").stdout
    )
    assert [r["id"] for r in got] == [4]
    q = 'query { repository(owner: "o", name: "r") { pullRequest(number: 7) { headRefOid } } }'
    data = json.loads(_serve(pin, real, "api", "graphql", "-f", f"query={q}").stdout)
    pr = data["data"]["repository"]["pullRequest"]
    assert (pr["headRefOid"], pr["state"]) == (pin.sha, "OPEN")
    assert [(n["id"], len(n["comments"]["nodes"])) for n in pr["reviewThreads"]["nodes"]] == [
        ("T1", 1)
    ]
    # gh pr view --json comments: same cutoff; `--jq` applies to the filtered view
    real.routes["pr view"] = {
        "comments": [{"body": "early", "createdAt": OLD}, {"body": "late", "createdAt": NEW}],
        "author": {"login": "alice"},
    }
    got = json.loads(_serve(pin, real, "pr", "view", "7", "--json", "comments,author").stdout)
    assert [c["body"] for c in got["comments"]] == ["early"]
    if pin.jq:
        res = _serve(pin, real, "pr", "view", "--json", "comments", "--jq", ".comments[].body")
        assert res.stdout == "early\n"


def test_the_pr_object_is_open_at_the_replayed_head(history, tmp_path):
    pin = _pin(history, tmp_path)
    real = Real(
        {
            "repos/o/r/pulls/7": {
                "number": 7,
                "title": "New title",
                "body": "edited after the cutoff",
                "state": "closed",
                "merged": True,
                "merged_at": NEW,
                "head": {"sha": history["later"], "ref": "pr"},
                "base": {"sha": "f" * 40, "ref": "main"},
                "user": {"login": "alice"},
            }
        }
    )
    pr = json.loads(_serve(pin, real, "api", "repos/o/r/pulls/7").stdout)
    assert pr["head"] == {"sha": pin.sha, "ref": "pr"}
    assert pr["base"]["sha"] == history["base"]
    assert (pr["state"], pr["merged"], pr["merged_at"]) == ("open", False, None)
    assert (pr["title"], pr["body"]) == ("Old title", "Old body")
    view = json.loads(
        _serve(
            pin, real, "pr", "view", "7", "--json", "headRefOid,state,title,commits,files,additions"
        ).stdout
    )
    assert (view["headRefOid"], view["state"], view["title"]) == (pin.sha, "OPEN", "Old title")
    assert [c["messageHeadline"] for c in view["commits"]] == ["fix: change line 5", "feat: add b"]
    assert [(f["path"], f["changeType"]) for f in view["files"]] == [
        ("a.txt", "MODIFIED"),
        ("b.txt", "ADDED"),
    ]
    assert view["additions"] == 2
    assert all(c[:2] != ["pr", "view"] for c in real.calls)  # every field computed: no live read
    real.routes["pr view"] = {"number": 7, "author": {"login": "alice"}, "labels": []}
    text = _serve(pin, real, "pr", "view").stdout
    assert "title:\tOld title" in text and "state:\tOPEN" in text


def test_files_and_commits_come_from_git_through_the_harness_client(history, tmp_path):
    pin, real = _pin(history, tmp_path), Real()
    gh = Gh(
        "gh",
        runner=read_only_runner(pinned_runner(pin, lambda argv, stdin, t: real(list(argv)[1:]))),
    )
    files = gh.api("repos/o/r/pulls/7/files?per_page=100", paginate=True)
    assert [(f["filename"], f["status"], f["additions"], f["deletions"]) for f in files] == [
        ("a.txt", "modified", 1, 1),
        ("b.txt", "added", 1, 0),
    ]
    assert files[1]["patch"].startswith("@@") and "+new file" in files[1]["patch"]
    commits = gh.api("repos/o/r/pulls/7/commits", paginate=True)
    assert [c["sha"] for c in commits][-1] == pin.sha
    assert commits[0]["commit"]["message"] == "fix: change line 5\n\nwith a body"
    assert "+line 5 changed" in gh.run("pr", "diff", "7", "--repo", "o/r")
    assert real.calls == []
    log = [json.loads(x) for x in Path(pin.log).read_text().splitlines()]
    assert {e["source"] for e in log} == {"harness"} and {e["outcome"] for e in log} == {"served"}


@pytest.mark.parametrize(
    "args",
    [
        ("pr", "comment", "7", "--body", "x"),
        ("pr", "review", "7", "--approve"),
        ("pr", "merge", "7"),
        ("pr", "checkout", "7"),
        ("issue", "create", "--title", "x"),
        ("api", "repos/o/r/issues/7/comments", "-X", "POST", "-f", "body=x"),
        ("api", "repos/o/r/issues/7/comments", "-f", "body=x"),  # fields imply POST
        ("api", "--method=PATCH", "repos/o/r/pulls/comments/1"),
        ("api", "repos/o/r/pulls/7/reviews", "--input", "payload.json"),
        (
            "api",
            "graphql",
            "-f",
            'query=mutation { resolveReviewThread(input: {threadId: "t"}) { x } }',
        ),
        ("api", "graphql", "-f", "query=query Q { x } mutation M { y }"),
    ],
)
def test_every_write_is_refused(history, tmp_path, args):
    pin, real = _pin(history, tmp_path), Real()
    res = _serve(pin, real, *args)
    assert res.returncode == 1
    assert "read-only replay of o/r#7" in res.stderr or "refused" in res.stderr
    assert real.calls == []


@pytest.mark.parametrize(
    "args",
    [
        ("search", "prs", "fix"),
        ("pr", "list"),
        ("issue", "list"),
        ("run", "list"),
        ("run", "view"),  # no run id: would pick the latest run
        ("release", "view"),
        ("pr", "diff", "feature-branch"),
        ("status",),
    ],
)
def test_unknown_or_unpinnable_commands_are_refused_never_passed_live(history, tmp_path, args):
    pin, real = _pin(history, tmp_path), Real()
    res = _serve(pin, real, *args)
    assert res.returncode == 1 and res.stderr.startswith("gh (reviewsys replay):")
    assert real.calls == []
    assert json.loads(Path(pin.log).read_text().splitlines()[-1])["outcome"] == "refused"


def test_other_prs_of_the_repo_opened_after_the_cutoff_are_refused(history, tmp_path):
    pin = _pin(history, tmp_path)
    real = Real({"repos/o/r/issues/9": NEW, "repos/o/r/issues/3": OLD, "pr diff": "old diff"})
    assert "did not exist yet" in _serve(pin, real, "pr", "view", "9").stderr
    assert _serve(pin, real, "pr", "diff", "3").stdout == "old diff"


def test_reads_of_other_repositories_pass_through(history, tmp_path):
    pin = _pin(history, tmp_path)
    real = Real({"repos/bitcoin/bitcoin/pulls/27000": "abc"})
    res = _serve(
        pin, real, "api", "/repos/bitcoin/bitcoin/pulls/27000", "--jq", ".merge_commit_sha"
    )
    assert res.stdout == "abc"
    assert real.calls == [
        ["api", "/repos/bitcoin/bitcoin/pulls/27000", "--jq", ".merge_commit_sha"]
    ]


def test_the_installed_shim_asks_the_broker(history, tmp_path):
    """The lane's gh is a client of the replay's broker: it runs no gh of its own."""
    pin = _pin(history, tmp_path)
    real = Real({"--version": "gh version 9\n"})
    try:
        broker = Broker(
            pin,
            lambda args: (
                real(args)
                if args != ["--version"]
                else subprocess.CompletedProcess(args, 0, "gh version 9\n", "")
            ),
        ).__enter__()
    except PermissionError:
        pytest.skip("this sandbox refuses binding a Unix socket")
    try:
        shim = install_shim(broker.path, tmp_path / "bin", curl_log=tmp_path / "curl.log")
        ver = subprocess.run([str(shim), "--version"], capture_output=True, text=True)
        assert ver.returncode == 0 and ver.stdout == "gh version 9\n"
        diff = subprocess.run([str(shim), "pr", "diff", "7"], capture_output=True, text=True)
        assert diff.returncode == 0 and "+new file" in diff.stdout
        post = subprocess.run(
            [str(shim), "pr", "comment", "7", "--body", "hi"], capture_output=True, text=True
        )
        assert post.returncode == 1 and "read-only replay" in post.stderr
        q = subprocess.run(
            [str(shim), "api", "graphql", "-F", "query=@-"],
            input="mutation { x }",
            capture_output=True,
            text=True,
        )
        assert q.returncode == 1 and "mutation" in q.stderr
        curl = tmp_path / "bin" / "curl"
        if curl.exists():
            got = subprocess.run(
                [str(curl), "-s", "https://api.github.com/repos/o/r/pulls/7"],
                capture_output=True,
                text=True,
            )
            assert got.returncode == 22 and "refused" in got.stderr
            assert "api.github.com" in (tmp_path / "curl.log").read_text()
    finally:
        broker.__exit__(None, None, None)
    sources = {json.loads(x)["source"] for x in Path(pin.log).read_text().splitlines()}
    assert sources == {"lane"}


@pytest.mark.parametrize(
    "args",
    [
        ("pr", "comment", "5", "--body", "--help"),  # gh would post a comment saying --help
        ("pr", "comment", "5", "-b", "-h"),
        ("api", "-X", "POST", "repos/o/r/issues/7/comments", "-f", "body=--help", "--help"),
    ],
)
def test_help_is_served_for_the_command_words_only(history, tmp_path, args):
    pin, real = _pin(history, tmp_path), Real()
    real.routes = {}
    _serve(pin, real, *args)
    assert len(real.calls) == 1
    call = real.calls[0]
    assert call[-1] == "--help" and all(a.isalpha() for a in call[:-1])
    assert "5" not in call and "POST" not in call


def test_graphql_needs_exactly_one_checked_query(history, tmp_path):
    pin = _pin(history, tmp_path)
    real = Real({"graphql": {"data": {}}})
    two = _serve(
        pin, real, "api", "graphql", "-f", "query={viewer{login}}", "-f", "query=mutation{x}"
    )
    assert two.returncode == 1 and "exactly one" in two.stderr
    assert _serve(pin, real, "api", "graphql", "-f", "query={x}", "--input", "q.json").returncode
    assert real.calls == []
    qfile = tmp_path / "q.graphql"
    qfile.write_text("{ viewer { login } }")
    assert _serve(pin, real, "api", "graphql", "-F", f"query=@{qfile}").returncode == 0
    # the query that was checked is the one sent, as text: gh never re-reads the file
    assert real.calls[-1][:4] == ["api", "graphql", "-f", "query={ viewer { login } }"]
    res = serve(pin, ["api", "graphql", "-F", "query=@-"], real=real, stdin="{ a }")
    assert res.returncode == 0 and real.calls[-1][3] == "query={ a }"


def test_auth_status_never_shows_the_token(history, tmp_path):
    pin, real = _pin(history, tmp_path), Real()
    _serve(pin, real, "auth", "status", "--show-token")
    _serve(pin, real, "auth", "status", "-t")
    assert real.calls == [["auth", "status"], ["auth", "status"]]


def test_bodies_and_threads_are_rewound(history, tmp_path):
    root = "<!-- thepastaclaw-review v1 finding={h} -->\n<!-- thepastaclaw-thread-status v1 -->\n> ✅ **Withdrawn** at `abcdef12`; see the replies below.\n\n**🔴 Blocking: {t}**"
    pin = _pin(history, tmp_path, lifted={"aaaaaaaaaaaa": "INTENTIONALLY_DEFERRED"})
    real = Real(
        {
            "repos/o/r/pulls/7/comments": [
                {
                    "id": 1,
                    "created_at": OLD,
                    "updated_at": NEW,
                    "body": root.format(h="aaaaaaaaaaaa", t="Kept"),
                },
                {
                    "id": 2,
                    "created_at": OLD,
                    "body": root.format(h="bbbbbbbbbbbb", t="Lifted later"),
                },
            ],
            "repos/o/r/issues/7/comments": [
                {
                    "id": 3,
                    "created_at": OLD,
                    "body": "<!-- thepastaclaw-gate v1 -->\n✅ Final review complete (commit later)",
                }
            ],
            "graphql": {
                "data": {
                    "repository": {
                        "pullRequest": {
                            "reviewThreads": {
                                "nodes": [
                                    {
                                        "id": "T1",
                                        "isResolved": True,
                                        "comments": {
                                            "nodes": [{"createdAt": OLD}, {"createdAt": NEW}]
                                        },
                                    },
                                    {
                                        "id": "T2",
                                        "isResolved": True,
                                        "isOutdated": True,
                                        "comments": {"nodes": [{"createdAt": OLD}]},
                                    },
                                ]
                            }
                        }
                    }
                }
            },
        }
    )
    got = json.loads(_serve(pin, real, "api", "repos/o/r/pulls/7/comments").stdout)
    assert "⏭️ **Deferred**" in got[0]["body"] and "`abcdef12`" in got[0]["body"]
    assert "thread-status" not in got[1]["body"] and "Lifted later" in got[1]["body"]
    gate = json.loads(_serve(pin, real, "api", "repos/o/r/issues/7/comments").stdout)
    assert "Final review complete" not in gate[0]["body"]
    assert gate[0]["body"].startswith("<!-- thepastaclaw-gate v1 -->")
    q = 'query { repository(owner: "o", name: "r") { pullRequest(number: 7) { id } } }'
    data = json.loads(_serve(pin, real, "api", "graphql", "-f", f"query={q}").stdout)
    nodes = data["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"]
    assert [(n["id"], n["isResolved"]) for n in nodes] == [("T1", False), ("T2", True)]
    notes: dict[str, int] = {}
    for line in Path(pin.log).read_text().splitlines():
        for k, v in (json.loads(line).get("notes") or {}).items():
            notes[k] = notes.get(k, 0) + v
    assert notes == {
        "edited_after_cutoff": 1,
        "thread_status_stripped": 1,
        "gate_comment_replaced": 1,
        "dropped_after_cutoff": 1,
        "thread_reopened_by_cutoff": 1,
        "thread_resolved_as_of_now": 1,
        "thread_outdated_as_of_now": 1,
    }


def test_later_commits_and_counts_are_pinned_to_the_replayed_range(history, tmp_path):
    pin = _pin(history, tmp_path)
    real = Real(
        {
            "repos/o/r/pulls/7": {
                "number": 7,
                "head": {"sha": history["later"]},
                "commits": 3,
                "additions": 9,
                "deletions": 9,
                "changed_files": 3,
            },
            "repos/o/r/issues/7/timeline": [
                {"event": "committed", "sha": pin.sha},
                {"event": "committed", "sha": history["later"]},
                {"event": "commented", "created_at": OLD},
            ],
            "graphql": {
                "data": {
                    "repository": {
                        "pullRequest": {
                            "additions": 9,
                            "commits": {
                                "totalCount": 3,
                                "nodes": [
                                    {"commit": {"oid": pin.sha}},
                                    {"commit": {"oid": history["later"]}},
                                ],
                            },
                        }
                    }
                }
            },
        }
    )
    pr = json.loads(_serve(pin, real, "api", "repos/o/r/pulls/7").stdout)
    assert (pr["commits"], pr["additions"], pr["deletions"], pr["changed_files"]) == (2, 2, 1, 2)
    tl = json.loads(_serve(pin, real, "api", "repos/o/r/issues/7/timeline").stdout)
    assert [e.get("sha") for e in tl] == [pin.sha, None]
    q = 'query { repository(owner: "o", name: "r") { pullRequest(number: 7) { additions } } }'
    gq = json.loads(_serve(pin, real, "api", "graphql", "-f", f"query={q}").stdout)
    p = gq["data"]["repository"]["pullRequest"]
    assert p["additions"] == 2 and p["commits"]["totalCount"] == 2
    assert [n["commit"]["oid"] for n in p["commits"]["nodes"]] == [pin.sha]


def test_combined_status_follows_the_snapshot(history, tmp_path):
    pin, real = _pin(history, tmp_path), Real()
    st = json.loads(_serve(pin, real, "api", f"repos/o/r/commits/{pin.sha}/status").stdout)
    assert st["state"] == "failure"  # the snapshot has a failing check


def test_read_only_runner_refuses_any_unchecked_query():
    calls: list[list[str]] = []

    def inner(argv, stdin, timeout):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "{}", "")

    run = read_only_runner(inner)
    for argv in (
        ["gh", "api", "graphql", "-f", "query={a}", "-f", "query=mutation{b}"],
        ["gh", "api", "graphql", "-F", "query=@q.graphql"],
        ["gh", "api", "graphql", "-f", "query={a}", "--input", "-"],
    ):
        assert run(argv, None, 1).returncode == 1
    assert calls == []


def test_read_only_runner_refuses_mutations_anywhere_in_the_document():
    calls: list[list[str]] = []

    def inner(argv, stdin, timeout):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "{}", "")

    gh = Gh("gh", runner=read_only_runner(inner))
    with pytest.raises(ReviewError, match="read-only gh"):
        gh.graphql("query A { x }\nmutation B { resolveReviewThread(input: {}) { y } }")
    assert calls == []


def test_the_real_gh_is_read_as_text_whatever_the_runner_returns():
    """The default gh runner hands back bytes (a diff may hold non-UTF-8 blob bytes); the
    pinned view reads text, a bad byte as U+FFFD as `github.get_diff` does."""
    from types import SimpleNamespace

    def runner(argv, stdin, timeout):
        return subprocess.CompletedProcess(argv, 0, b"+caf\xc3\xa9 \xa1\n", b"")

    res = real_gh(SimpleNamespace(real_gh="gh"), runner)(["pr", "diff", "1"])  # type: ignore[arg-type]
    assert res.args == ["gh", "pr", "diff", "1"]
    assert (res.stdout, res.stderr) == ("+café �\n", "")
