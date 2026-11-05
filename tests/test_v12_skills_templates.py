"""The skills repo's real policy v12 templates render through the harness: every slot they use
is one the harness fills, and nothing is left unfilled. The copies under
golden/skills-v12-prompts/ are skills-v12's `prompts/v12/` (b4073bd); with
REVIEWSYS_V12_SKILLS pointing at a skills checkout, its templates are rendered too and the copies
must still match them, so a template change on either side shows up here."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import pytest

from reviewsys import prompts

COPIES = Path(__file__).parent / "golden" / "skills-v12-prompts"
LIVE = os.environ.get("REVIEWSYS_V12_SKILLS")
SOURCES = [COPIES] + ([Path(LIVE) / "prompts" / "v12"] if LIVE else [])
LANES = ("finder", "verifier", "thread")
METHODS = sorted(p.stem for p in COPIES.glob("*.md") if p.stem not in LANES)
# a `{slot}` the harness fills that a template could leave behind, rendered or not
LEFTOVER = re.compile(r"\{[a-z_]+\}")
META = {"title": "Fix it", "body": "Body", "baseRefName": "develop", "author": "someone"}


@pytest.fixture(params=SOURCES, ids=lambda p: "copies" if p == COPIES else "live")
def real(request, cfg):
    d = cfg.skills_dir / "prompts" / "v12"
    shutil.rmtree(d)
    shutil.copytree(request.param, d)
    return cfg


def _finder(cfg, role: str, structured: bool) -> str:
    return prompts.finder_prompt(
        cfg,
        repo="dashpay/platform",
        number=7,
        head_sha="a" * 40,
        role=role,
        lane_title=role,
        meta=META,
        coverage_from="b" * 40,
        evidence={"pr": {}},
        guidance="",
        last_reviewed_sha="c" * 40,
        settled=[
            {"file": "f", "lines": [1, 2], "title": "t", "status": "withdrawn", "reason": "r"}
        ],
        structured=structured,
    )


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("role", ["general", *METHODS])
def test_every_finder_method_renders(real, role, structured):
    text = _finder(real, role, structured)
    assert not LEFTOVER.search(text), LEFTOVER.findall(text)
    assert "## Output: candidates" in text and "dashpay/platform" in text


@pytest.mark.parametrize("structured", [False, True])
def test_the_verifier_and_thread_templates_render(real, structured):
    verify = prompts.verify_prompt(
        real,
        repo="dashpay/platform",
        number=7,
        head_sha="a" * 40,
        meta=META,
        coverage_from="b" * 40,
        candidates=[{"id": "p2-general-1"}],
        coderabbit={"findings": [{"comment_id": 5}]},
        ledger_summary={"open": [], "settled": [], "fixed": []},
        evidence={},
        guidance="",
        phase_note="NOTE",
        structured=structured,
    )
    for same_sha in (False, True):
        thread = prompts.thread_prompt(
            real,
            repo="dashpay/platform",
            number=7,
            head_sha="a" * 40,
            meta=META,
            issues=[{"finding_hash": "abcdef123456"}],
            linked_commits="",
            evidence={"issue_comments": [], "ci": {"state": "success"}},
            standing_verdict=None,
            same_sha=same_sha,
            structured=structured,
        )
        assert not LEFTOVER.search(thread), LEFTOVER.findall(thread)
        assert '"state": "success"' in thread and "abcdef123456" in thread
    assert not LEFTOVER.search(verify), LEFTOVER.findall(verify)
    assert "NOTE" in verify and "p2-general-1" in verify


@pytest.mark.skipif(not LIVE, reason="REVIEWSYS_V12_SKILLS is not set")
def test_the_copies_match_the_skills_checkout():
    live = Path(LIVE) / "prompts" / "v12"  # type: ignore[arg-type]
    assert sorted(p.name for p in live.glob("*.md")) == sorted(p.name for p in COPIES.glob("*.md"))
    for p in COPIES.glob("*.md"):
        assert p.read_text() == (live / p.name).read_text(), p.name
