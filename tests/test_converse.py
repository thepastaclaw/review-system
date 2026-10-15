"""Conversation lane: parsing, linked-commit extraction, rendering, once-per-reply posting."""

from __future__ import annotations

import pytest

from reviewsys import converse, publish
from reviewsys.models import ReviewError

HEAD = "a" * 40


def test_parse_accepts_one_row_per_thread_and_normalises_status():
    out = converse.parse(
        {
            "threads": [
                {"finding_hash": "aaa", "status": "still_valid", "reply": "  new point  "},
                {"finding_hash": "bbb", "status": "NO_REPLY", "reply": "ignored text"},
            ]
        },
        expected={"aaa", "bbb"},
    )
    assert out.outcomes["aaa"].status == "STILL_VALID"
    assert out.outcomes["aaa"].reply == "new point"
    assert out.outcomes["bbb"].status == "NO_REPLY" and out.outcomes["bbb"].reply == ""


def test_fixed_is_never_a_conversation_outcome():
    """A conversation runs on the reviewed commit: a promised change is FIX_PENDING, which lifts
    nothing (dashpay/dash#7778 lifted a blocker on a change that was never pushed)."""
    out = converse.parse(
        {"threads": [{"finding_hash": "aaa", "status": "FIXED", "reply": "that does it"}]},
        expected={"aaa"},
    )
    assert out.outcomes["aaa"].status == "FIX_PENDING"
    assert "FIX_PENDING" not in converse.LIFTING_STATUSES
    assert "FIXED" not in converse.CONVERSATION_STATUSES


@pytest.mark.parametrize(
    "rows, msg",
    [
        ([{"finding_hash": "zzz", "status": "FIXED", "reply": "x"}], "unexpected hash"),
        ([{"finding_hash": "aaa", "status": "OUTDATED", "reply": "x"}], "bad status"),
        ([{"finding_hash": "aaa", "status": "FIXED", "reply": ""}], "without a reply"),
        ([], "missing threads"),
        (
            [
                {"finding_hash": "aaa", "status": "FIXED", "reply": "x"},
                {"finding_hash": "aaa", "status": "FIXED", "reply": "y"},
            ],
            "duplicate conversation row",
        ),
    ],
)
def test_parse_rejects_malformed_rows(rows, msg):
    with pytest.raises(ReviewError, match=msg):
        converse.parse({"threads": rows}, expected={"aaa"})


def test_linked_commits_come_from_human_replies_only_and_dedupe():
    sha = "7c19184c849c382d6c72d31baeced0bd4631c616"
    threads = {
        "aaa": {
            "transcript": [
                {"is_bot": False, "body": f"see https://github.com/UdjinM6/dash/commit/{sha}"},
                {"is_bot": True, "body": f"https://github.com/x/y/commit/{'b' * 40}"},
                {
                    "is_bot": False,
                    "body": f"again https://github.com/dashpay/dash/pull/7675/commits/{sha[:10]}",
                },
                {
                    "is_bot": False,
                    "body": "https://github.com/dashpay/dash/pull/7675/commits/ec683243c7",
                },
            ]
        }
    }
    got = converse.linked_commits(threads)
    assert [c["sha"] for c in got] == [sha, "ec683243c7"]
    assert got[0]["url"] == "https://github.com/UdjinM6/dash.git"
    assert got[1]["owner"] == "dashpay"


def test_prompt_carries_exchange_commits_and_rules():
    threads = {
        "aaa": {
            "severity": "suggestion",
            "title": "T",
            "path": "f.py",
            "line": 3,
            "body": "<!-- thepastaclaw-review v1 finding=aaa -->\n**🟡 Suggestion: T**\n\nbody",
            "awaiting_answer": True,
            "transcript": [
                {"id": 1, "author": "knst", "body": "wrong", "is_bot": False},
                {"id": 2, "author": "thepastaclaw", "body": "still", "is_bot": True},
            ],
        }
    }
    text = converse.prompt(
        repo="dashpay/dash",
        number=1,
        head_sha=HEAD,
        meta={"title": "t", "author": "knst", "body": "d"},
        project_skill="P",
        review_skill="R",
        threads=threads,
        fetched_commits=[{"owner": "u", "repo": "dash", "sha": "c" * 40, "url": "x"}],
        unfetched_commits=[
            {"owner": "u", "repo": "priv", "sha": "d" * 40, "url": "y", "reason": "gone"}
        ],
        standing_verdict="CHANGES_REQUESTED",
        evidence={"issue_comments": [{"author": "z", "body": "hi"}]},
    )
    assert "Your previous attempt was rejected" not in text
    again = converse.prompt(
        repo="dashpay/dash",
        number=1,
        head_sha=HEAD,
        meta={},
        project_skill="",
        review_skill="",
        threads=threads,
        fetched_commits=[],
        unfetched_commits=[],
        standing_verdict=None,
        evidence={},
        previous_error="missing threads: ['aaa']",
    )
    assert "Your previous attempt was rejected" in again and "missing threads" in again
    assert '"author": "you (PastaClaw)"' in text and '"body": "still"' in text
    assert "<!--" not in text.split("finding_body")[1].split("\n")[0]
    assert f"git show {'c' * 40}" in text and "could NOT be fetched (gone)" in text
    assert "Never repeat a point you already made" in text
    assert "NO_REPLY" in text and "Emit exactly one raw JSON object" in text


def test_transcript_markers_are_stripped_and_scrub_rejects_echoes():
    threads = {
        "aaa": {
            "transcript": [
                {
                    "author": "thepastaclaw",
                    "body": "<!-- thepastaclaw-thread-answer v1 sha=x reply=1 finding=aaa -->\n**Still applies**: y",
                    "is_bot": True,
                }
            ]
        }
    }
    text = converse.prompt(
        repo="r",
        number=1,
        head_sha=HEAD,
        meta={},
        project_skill="",
        review_skill="",
        threads=threads,
        fetched_commits=[],
        unfetched_commits=[],
        standing_verdict=None,
        evidence={},
    )
    assert "thread-answer v1" not in text and "**Still applies**: y" in text
    assert publish.scrub_reply("<!-- thepastaclaw-thread-answer v1 --> hi", 100) == ""
    cut = publish.scrub_reply("a @b " + "```\ncode " + "x" * 200, 20)
    assert cut.startswith("a @\u200bb ```\ncode x") and cut.endswith("\n```")
    assert len(cut.replace("\u200b", "")) == 24  # 20 kept + the closing fence


class _Gh:
    def __init__(self, inline):
        self.inline = inline
        self.replies = []
        self.resolved = []


def test_answer_conversation_posts_once_resolves_and_defuses(monkeypatch):
    gh = _Gh([])
    monkeypatch.setattr(publish.github, "inline_comments", lambda gh_, r, n: gh.inline)
    monkeypatch.setattr(
        publish.github, "post_reply", lambda gh_, r, n, cid, body: gh.replies.append((cid, body))
    )
    monkeypatch.setattr(publish.github, "resolve_thread", lambda gh_, tid: gh.resolved.append(tid))
    threads = {
        "aaa": {"comment_id": 900, "thread_id": "T1", "latest_reply_id": 901},
        "bbb": {"comment_id": 910, "thread_id": "T2", "latest_reply_id": 911},
        "ccc": {"comment_id": 920, "thread_id": "T3", "latest_reply_id": 921},
        "ddd": {"comment_id": 930, "thread_id": "T4", "latest_reply_id": 931},
    }
    outcomes = {
        "aaa": converse.ThreadOutcome("aaa", "WITHDRAWN", "done, thanks @knst"),
        "bbb": converse.ThreadOutcome("bbb", "NO_REPLY", ""),
        "ccc": converse.ThreadOutcome("ccc", "STILL_VALID", "please @coderabbitai review"),
        "ddd": converse.ThreadOutcome("ddd", "INTENTIONALLY_DEFERRED", "understood"),
    }
    out = publish.answer_conversation(
        gh, "dashpay/dash", 1, HEAD, threads=threads, outcomes=outcomes
    )
    actions = {o["finding_hash"]: o["action"] for o in out}
    assert actions == {
        "aaa": "replied",
        "bbb": "no_reply",
        "ccc": "suppressed_reply",
        "ddd": "replied",
    }
    assert gh.resolved == ["T1"]
    (cid, body), (cid2, body2) = gh.replies
    assert cid == 900 and body.startswith(
        f"<!-- thepastaclaw-thread-answer v1 sha={HEAD} reply=901 finding=aaa -->\n"
    )
    assert "done, thanks @​knst" in body and body.endswith("_Withdrawing this finding._")
    assert cid2 == 930 and body2.endswith("no longer blocks this pull request._")
    # second pass with the same replies: everything already answered, nothing reposted
    gh.inline = [{"body": b} for _, b in gh.replies]
    out = publish.answer_conversation(
        gh, "dashpay/dash", 1, HEAD, threads=threads, outcomes=outcomes
    )
    assert {o["finding_hash"]: o["action"] for o in out} == {
        "aaa": "already_answered",
        "bbb": "no_reply",
        "ccc": "suppressed_reply",
        "ddd": "already_answered",
    }
    assert len(gh.replies) == 2 and gh.resolved == ["T1"]


def test_a_deferral_counts_only_when_a_maintainer_spoke_since_our_last_answer():
    """A maintainer who agreed the blocker was real before our last answer does not make a
    later contributor's "follow-up" a maintainer call; the PR author as MEMBER does count."""
    member = {"is_bot": False, "association": "MEMBER"}
    contributor = {"is_bot": False, "association": "CONTRIBUTOR"}
    bot = {"is_bot": True, "association": None}
    threads = {
        "old": {"transcript": [member, bot, contributor]},
        "new": {"transcript": [contributor, bot, member]},
        "none": {"transcript": [contributor]},
    }
    out = converse.ConversationOutput(
        outcomes={
            h: converse.ThreadOutcome(h, "INTENTIONALLY_DEFERRED", "ok")
            for h in ("old", "new", "none")
        }
        | {"kept": converse.ThreadOutcome("kept", "STILL_VALID", "still")}
    )
    threads["kept"] = {"transcript": [contributor]}
    got = {h: o.status for h, o in converse.accept_deferrals(out, threads).outcomes.items()}
    assert got == {
        "old": converse.DEFERRAL_PENDING,
        "new": "INTENTIONALLY_DEFERRED",
        "none": converse.DEFERRAL_PENDING,
        "kept": "STILL_VALID",
    }


ROOT = (
    "<!-- thepastaclaw-review v1 finding=aaa dedupe=x -->\n"
    "**🔴 Blocking: Sign with the payload's quorum**\n\nbody text"
)


def test_thread_status_line_sits_under_the_marker_and_keeps_the_headline_parseable():
    marked = publish.with_thread_status(ROOT, status="WITHDRAWN", head_sha=HEAD)
    lines = marked.splitlines()
    assert lines[0].startswith("<!-- thepastaclaw-review v1 finding=aaa")
    assert lines[1] == publish.THREAD_STATUS_MARKER
    assert lines[2] == f"> ✅ **Withdrawn** at `{HEAD[:8]}`; see the replies below."
    assert publish.github._finding_title(marked) == (
        "blocking",
        "Sign with the payload's quorum",
    )
    # replaced, never stacked; cleared back to the original byte for byte
    again = publish.with_thread_status(marked, status="INTENTIONALLY_DEFERRED", head_sha="b" * 40)
    assert again.count(publish.THREAD_STATUS_MARKER) == 1 and "**Deferred**" in again
    assert publish.with_thread_status(again, status=None, head_sha=HEAD) == ROOT
    assert publish.with_thread_status(ROOT, status="STILL_VALID", head_sha=HEAD) == ROOT
    # dedupe matches on the finding's own text, never on the status line
    plain, tagged = (publish.ExistingComment.from_github({"body": b}) for b in (ROOT, marked))
    assert tagged.norm_body == plain.norm_body and tagged.finding_hash == "aaa"


def test_answer_conversation_marks_the_root_only_when_the_thread_stays_open(monkeypatch):
    gh = _Gh([])
    edits: list[tuple[int, str]] = []
    monkeypatch.setattr(publish.github, "inline_comments", lambda gh_, r, n: gh.inline)
    monkeypatch.setattr(publish.github, "post_reply", lambda gh_, r, n, cid, body: None)
    monkeypatch.setattr(
        publish.github, "edit_review_comment", lambda gh_, r, cid, body: edits.append((cid, body))
    )

    def refuse(gh_, tid):
        raise ReviewError(publish.FailKind.CONTRACT, "no permission to ResolveReviewThread")

    monkeypatch.setattr(publish.github, "resolve_thread", refuse)
    stale = publish.with_thread_status(ROOT, status="WITHDRAWN", head_sha="b" * 40)
    threads = {
        "aaa": {"comment_id": 900, "thread_id": "T1", "latest_reply_id": 901, "body": ROOT},
        "bbb": {"comment_id": 910, "thread_id": "T2", "latest_reply_id": 911, "body": ROOT},
        "ccc": {"comment_id": 920, "thread_id": "T3", "latest_reply_id": 921, "body": stale},
        "ddd": {"comment_id": 930, "thread_id": "T4", "latest_reply_id": 931, "body": "foreign"},
    }
    outcomes = {
        "aaa": converse.ThreadOutcome("aaa", "WITHDRAWN", "you are right"),
        "bbb": converse.ThreadOutcome("bbb", "FIX_PENDING", "that change would do it"),
        "ccc": converse.ThreadOutcome("ccc", "STILL_VALID", "it is back on this head"),
        "ddd": converse.ThreadOutcome("ddd", "INTENTIONALLY_DEFERRED", "ok"),
    }
    out = {
        o["finding_hash"]: o
        for o in publish.answer_conversation(
            gh, "dashpay/dash", 1, HEAD, threads=threads, outcomes=outcomes
        )
    }
    assert str(out["aaa"]["resolved"]).startswith("failed") and out["aaa"]["marked"] is True
    assert "marked" not in out["bbb"]  # a pending fix lifts nothing: the root is left alone
    assert out["ccc"]["marked"] is True  # a stale line from an earlier head is cleared
    assert str(out["ddd"]["marked"]).startswith("failed")  # never edit a root we did not write
    assert [cid for cid, _ in edits] == [900, 920]
    assert "**Withdrawn**" in edits[0][1] and edits[1][1] == ROOT


STANDING = f"""<!-- thepastaclaw-review v1 -->
<!-- thepastaclaw-review-phase v1 phase=final sha={HEAD} policy=x -->
## Re-review — Final validation — Phase 1 + Phase 2

Summary.

🔴 2 blocking | 💬 1 nitpick(s)

<details>
<summary>🤖 Prompt for all review comments with AI agents</summary>

```
In `t.py`:
- [BLOCKING] t.py:2355-2358: Synchronize the second rotating session
  body one
- [BLOCKING] t.py:2239: Synchronize secondary types
  body two

In `<commit:fb6f0d5207>`:
- [NITPICK] <commit:fb6f0d5207>:1: Reword the polling commit
```
</details>"""


def _lifted(title, *, file="t.py", severity="blocking", status="INTENTIONALLY_DEFERRED"):
    return publish.LiftedFinding(
        hash=title[:6], severity=severity, file=file, title=title, status=status
    )


def test_lifted_findings_rewrite_the_standing_review_idempotently_and_reversibly():
    deferred = _lifted("Synchronize the second rotating session")
    nit = _lifted(
        "Reword the polling commit",
        file="<commit:fb6f0d5207>",
        severity="nitpick",
        status="WITHDRAWN",
    )
    once = publish.with_lifted_findings(STANDING, [deferred])
    lines = once.split("\n")
    assert lines[3] == "" and lines[4] == publish.REVIEW_STATUS_MARKER
    assert lines[5] == (
        "> **Updated after discussion:** 1 finding(s) of this review were settled in the "
        "inline threads; 1 blocking finding(s) still stand."
    )
    assert lines[6] == (
        "> - ⏭️ Deferred (maintainer decision): 🔴 Synchronize the second rotating session"
    )
    assert lines[7] == "" and lines[8] == "Summary."
    assert "🔴 ~~2~~ 1 blocking | 💬 1 nitpick(s)" in once
    assert (
        "- [BLOCKING → DEFERRED by a maintainer; not blocking, do not fix in this PR] "
        "t.py:2355-2358: Synchronize the second rotating session" in once
    )
    assert "- [BLOCKING] t.py:2239: Synchronize secondary types" in once
    # re-applying is a no-op; a wider set re-derives from the original numbers
    assert publish.with_lifted_findings(once, [deferred]) == once
    both = publish.with_lifted_findings(once, [deferred, nit])
    assert "🔴 ~~2~~ 1 blocking | 💬 ~~1~~ 0 nitpick(s)" in both
    assert "- [NITPICK → WITHDRAWN in discussion; do not fix] <commit:fb6f0d5207>:1:" in both
    assert both.count(publish.REVIEW_STATUS_MARKER) == 1 and "2 finding(s)" in both
    # nothing lifted any more (a later full review voided the concessions): back to the original
    assert publish.with_lifted_findings(both, []) == STANDING
    assert publish.with_lifted_findings(STANDING, []) == STANDING


def test_a_lifted_finding_retags_only_its_own_prompt_entry():
    """Same title in another file, a title that ends in the lifted one, and a finding the
    review never listed (a legacy thread): none of them may be retagged or counted."""
    fence = "```"
    body = (
        "## Final validation\n\n🔴 3 blocking\n\n<details>\n<summary>p</summary>\n\n"
        f"{fence}\nIn `a.cpp`:\n- [BLOCKING] a.cpp:1: Missing lock\n\n"
        "In `b.cpp`:\n- [BLOCKING] b.cpp:9: Missing lock\n"
        f"- [BLOCKING] b.cpp:20: Race: Missing lock\n{fence}\n</details>"
    )
    out = publish.with_lifted_findings(body, [_lifted("Missing lock", file="a.cpp")])
    assert "🔴 ~~3~~ 2 blocking" in out and "2 blocking finding(s) still stand" in out
    assert out.count("→ DEFERRED") == 1 and "] a.cpp:1: Missing lock" in out
    assert "- [BLOCKING] b.cpp:9: Missing lock" in out
    assert "- [BLOCKING] b.cpp:20: Race: Missing lock" in out
    legacy = _lifted("Some legacy thread", file="c.cpp")
    assert publish.with_lifted_findings(body, [legacy]) == body


def test_a_multiline_title_keeps_the_status_block_idempotent():
    odd = _lifted("Synchronize the second rotating session")
    odd = publish.LiftedFinding(
        hash=odd.hash, severity="blocking", file="t.py", title=odd.title, status="WITHDRAWN"
    )
    once = publish.with_lifted_findings(STANDING, [odd])
    weird = publish.LiftedFinding(
        hash="w", severity="blocking", file="t.py", title="a\n<!-- x -->", status=""
    )
    block = publish.with_lifted_findings(once, [odd, weird])
    assert publish.with_lifted_findings(block, [odd, weird]) == block
    assert publish.with_lifted_findings(block, []) == STANDING
    assert publish._block_text("a\n<!-- x -->") == "a &lt;!-- x -->"


def test_prompt_tells_the_lane_what_still_blocks_and_what_it_already_settled():
    threads = {
        "aaa": {
            "severity": "blocking",
            "title": "Deferred one",
            "awaiting_answer": True,
            "transcript": [{"id": 1, "author": "knst", "body": "IT IS STILL HERE"}],
        }
    }
    text = converse.prompt(
        repo="dashpay/dash",
        number=1,
        head_sha=HEAD,
        meta={},
        project_skill="",
        review_skill="",
        threads=threads,
        fetched_commits=[],
        unfetched_commits=[],
        standing_verdict="CHANGES_REQUESTED",
        evidence={},
        still_blocking=["The other blocker"],
        settled={"aaa": "INTENTIONALLY_DEFERRED"},
    )
    standing = text.split("## Where your review stands")[1].split("\n## ")[0]
    assert "- The other blocker" in standing and "do not settle it again" in standing
    assert '"already_settled_as": "INTENTIONALLY_DEFERRED"' in text
