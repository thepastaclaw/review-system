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
