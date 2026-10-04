"""Policy v12 `late_nonblocking` (B5, owner decision D2): later-round non-blocking issues on
lines unchanged since the last final review are posted (`post`) or collapsed into a note that
does not hold APPROVE (`note`); their count is recorded either way."""

from __future__ import annotations

import pytest

from reviewsys import worker
from tests.test_pipeline_policy import load_with
from tests.test_v12_worker import refute_gate
from tests.test_worker_e2e import fake_git  # noqa: F401  (autouse: fake mirror/worktree)
from tests.v12_support import Git, by_title, cand, run_round, v12_cfg

A, B = "a" * 40, "c" * 40


@pytest.fixture(autouse=True)
def wide_diff(gh):
    gh.diff = "diff --git a/f.rs b/f.rs\n+++ b/f.rs\n@@ -1,3 +1,60 @@\n+x\n"


@pytest.fixture
def git(monkeypatch) -> Git:
    g = Git()
    monkeypatch.setattr(worker, "_lines_changed", g)
    return g


@pytest.mark.parametrize("mode", ["post", "note"])
def test_late_nonblocking(cfg, skills_dir, conn, gh, lanes, git, mode):
    v12 = v12_cfg(cfg, skills_dir, late_nonblocking=mode)
    lanes.verify12 = refute_gate
    lanes.candidates["default"] = []
    run_round(v12, conn, gh, lanes, A)  # a clean final review
    assert gh.posted_reviews[-1]["event"] == "APPROVE"
    # round 2: a suggestion on a line nobody touched since that final review
    lanes.candidates["general"] = [cand("Name the constant", severity="suggestion", line=30)]
    before = len(gh.posted_reviews)
    run_round(v12, conn, gh, lanes, B)
    review = gh.posted_reviews[before]
    ev = conn.execute("SELECT detail FROM events WHERE kind='v12.late_nonblocking'").fetchone()
    assert f"1 late non-blocking finding(s) since aaaaaaaa ({mode})" == ev["detail"]
    if mode == "post":
        assert len(review["comments"]) == 1 and review["event"] == "COMMENT"
        assert by_title(conn)["Name the constant"].status == "open"
    else:
        assert review["comments"] == [] and review["event"] == "APPROVE"
        assert "Also noticed (1)" in review["body"] and "Name the constant" in review["body"]
        assert "Name the constant" not in by_title(conn)


def test_note_needs_the_v12_verifier(cfg, skills_dir):
    with pytest.raises(ValueError, match=r'"note" needs verifier\.contract "v12"'):
        load_with(cfg, skills_dir, {"version": "v12", "late_nonblocking": "note"})
