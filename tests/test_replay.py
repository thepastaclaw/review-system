"""`reviewsys replay` must be unable to write to GitHub, whatever the pipeline tries."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence

import pytest

from reviewsys.gh import Gh, read_only_runner
from reviewsys.models import ReviewError


def _inner(calls: list[list[str]]):
    def run(argv: Sequence[str], stdin: str | None, timeout: int):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "{}", "")

    return run


@pytest.mark.parametrize(
    "call",
    [
        lambda gh: gh.api("repos/o/r/pulls/1/reviews", method="POST", body={"event": "COMMENT"}),
        lambda gh: gh.api("repos/o/r/issues/comments/7", method="PATCH", body={"body": "x"}),
        lambda gh: gh.api("repos/o/r/issues/1/labels/x", method="DELETE"),
        lambda gh: gh.graphql('mutation { resolveReviewThread(input: {threadId: "t"}) { x } }'),
        lambda gh: gh.run("pr", "comment", "1", "--body", "x"),
    ],
)
def test_read_only_runner_refuses_writes(call):
    calls: list[list[str]] = []
    gh = Gh("gh", runner=read_only_runner(_inner(calls)))
    with pytest.raises(ReviewError, match="read-only gh"):
        call(gh)
    assert calls == []


def test_read_only_runner_passes_reads():
    calls: list[list[str]] = []
    gh = Gh("gh", runner=read_only_runner(_inner(calls)))
    gh.api("repos/o/r/pulls/1")
    gh.graphql("query { viewer { login } }")
    gh.run("pr", "diff", "1")
    assert len(calls) == 3


def test_pinned_head_reports_the_replayed_sha_open():
    import json

    from reviewsys import github
    from reviewsys.replay import _PinnedHead

    pr = {"state": "closed", "merged": True, "head": {"sha": "b" * 40}, "base": {"ref": "main"}}

    def run(argv, stdin, timeout):
        return subprocess.CompletedProcess(argv, 0, json.dumps(pr), "")

    gh = _PinnedHead("gh", "a" * 40, "9999")
    gh.runner = read_only_runner(run)
    meta = github.pr_meta(gh, "o/r", 7)
    assert (meta.head_sha, meta.state, meta.merged) == ("a" * 40, "open", False)


def test_pinned_head_hides_everything_created_after_the_source_run_started():
    import json

    from reviewsys.replay import _PinnedHead

    cutoff = "2026-09-20T12:00:00Z"
    old, new = "2026-09-20T11:00:00Z", "2026-09-20T13:00:00Z"
    responses = {
        "reviews": [{"id": 1, "submitted_at": old}, {"id": 2, "submitted_at": new}],
        "comments": [{"id": 3, "created_at": old}, {"id": 4, "created_at": new}],
        "graphql": {
            "data": {
                "repository": {
                    "pullRequest": {
                        "reviewThreads": {
                            "pageInfo": {"hasNextPage": False},
                            "nodes": [
                                {
                                    "id": "T1",
                                    "comments": {"nodes": [{"createdAt": old}, {"createdAt": new}]},
                                },
                                {"id": "T2", "comments": {"nodes": [{"createdAt": new}]}},
                            ],
                        }
                    }
                }
            }
        },
    }

    def run(argv, stdin, timeout):
        args = list(argv)[1:]
        key = "graphql" if args[1] == "graphql" else args[1].split("?")[0].rsplit("/", 1)[1]
        return subprocess.CompletedProcess(argv, 0, json.dumps(responses[key]), "")

    gh = _PinnedHead("gh", "a" * 40, cutoff)
    gh.runner = read_only_runner(run)
    assert [r["id"] for r in gh.api("repos/o/r/pulls/7/reviews?per_page=100")] == [1]
    assert [r["id"] for r in gh.api("repos/o/r/issues/7/comments?per_page=100")] == [3]
    q = "query { repository { pullRequest { reviewThreads(first: 100) { nodes { id } } } } }"
    nodes = gh.graphql(q)["repository"]["pullRequest"]["reviewThreads"]["nodes"]
    assert [(n["id"], len(n["comments"]["nodes"])) for n in nodes] == [("T1", 1)]
