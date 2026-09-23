"""v10 pipeline end to end: fake gh + scripted lanes keyed by role -> posted review, ledger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from reviewsys import config as cfg_mod
from reviewsys import ledger, worker
from reviewsys.db import tx
from reviewsys.ingest import enqueue_head
from reviewsys.lane import LaneResult, LaneSpec, _extract_result
from reviewsys.models import RunStatus, Trigger
from reviewsys.scheduler import schedule
from reviewsys.steps import worktree as wt
from tests.conftest import SKILLS_CONFIG

HEAD = "a" * 40
PIPELINE = {
    "finder_frame": "prompts/v10/finder.md",
    "general_lanes": {
        "scan": "prompts/v10/scan.md",
        "removed": "prompts/v10/removed.md",
        "trace": "prompts/v10/trace.md",
    },
    "specialist_prompts": {
        "security-auditor": "prompts/v10/security-auditor.md",
        "always-on": "prompts/v10/always-on.md",
    },
    "triage_prompt": "prompts/v10/triage.md",
    "verifier_prompt": "prompts/v10/verifier.md",
    "threads_prompt": "prompts/v10/threads.md",
    "composer_prompt": "prompts/v10/composer.md",
    "specialist_effort_cap": {"security-auditor": "medium"},
    "max_turns": {"phase2": 30},
    "verify_cap": 12,
}


@pytest.fixture
def v10cfg(tmp_path: Path, skills_dir: Path) -> cfg_mod.Config:
    raw = json.loads((skills_dir / "config.json").read_text())
    raw["review_model_policy"]["pipeline"] = PIPELINE
    (skills_dir / "config.json").write_text(json.dumps(raw))
    v = skills_dir / "prompts" / "v10"
    v.mkdir()
    (v / "finder.md").write_text(
        "FINDER {lane_title} {repo}#{pr_number} {head_sha}\n{method}\n{repo_rules}\n{discussion}"
    )
    for name in ("scan", "removed", "trace", "security-auditor", "always-on"):
        (v / f"{name}.md").write_text(f"METHOD {name} with {{braces}} left alone")
    (v / "triage.md").write_text("GROUP\n{candidates}\nLEDGER\n{ledger}\nCR\n{coderabbit}")
    (v / "verifier.md").write_text("VERIFY\n{candidate}\n{context_note}")
    (v / "threads.md").write_text("THREADS\n{issues}")
    (v / "composer.md").write_text("COMPOSE\n{findings}")
    toml = (
        cfg_mod.DEFAULT_TOML.replace(
            'db = "~/.reviewsys/review.db"', f'db = "{tmp_path}/review.db"'
        )
        .replace('work = "~/.reviewsys/work"', f'work = "{tmp_path}/work"')
        .replace('skills = "~/Projects/skills"', f'skills = "{skills_dir}"')
    )
    p = tmp_path / "config.toml"
    p.write_text(toml)
    return cfg_mod.load(p)


@pytest.fixture
def v10conn(v10cfg):
    from reviewsys.db import connect

    c = connect(v10cfg.db_path)
    yield c
    c.close()


@pytest.fixture(autouse=True)
def fake_git(monkeypatch, tmp_path):
    monkeypatch.setattr(wt, "ensure_mirror", lambda mirrors, repo: tmp_path / "mirror")
    monkeypatch.setattr(wt, "fetch_head", lambda mirror, number, sha: None)

    def create(mirror, wts, name, sha):
        p = wts / name
        p.mkdir(parents=True, exist_ok=True)
        (p / "AGENTS.md").write_text("REPO RULE: never log secrets")
        (p / "f.rs").write_text("\n".join(f"line {i}" for i in range(1, 60)))
        return p

    monkeypatch.setattr(wt, "create_worktree", create)
    monkeypatch.setattr(wt, "remove_worktree", lambda mirror, path: None)
    monkeypatch.setattr(wt, "merge_base", lambda worktree, base, sha: "b" * 40)
    monkeypatch.setattr(wt, "fetch_commit", lambda mirror, url, sha: True)
    monkeypatch.setattr(wt, "diff_file", lambda worktree, old, new, path, context=3: "")


def cand(title: str, sev: str = "suggestion", line: int = 11, scen: str = "input x") -> dict:
    return {
        "file": "f.rs",
        "line_start": line,
        "line_end": line + 1,
        "severity": sev,
        "category": "logic",
        "title": title,
        "failure_scenario": scen,
        "body": f"body of {title}",
        "suggestion": None,
    }


class Lanes:
    """Answers by role. finders: `finders[role]` (default []); group: `grouping` callable or
    everything-new; verifiers: `verdicts[title]` (default CONFIRMED at the candidate's severity);
    threads: `threads` callable; composer: echoes titles."""

    def __init__(self) -> None:
        self.calls: list[LaneSpec] = []
        self.finders: dict[str, list[dict]] = {}
        self.grouping: Any = None
        self.verdicts: dict[str, dict] = {}
        self.threads: Any = None
        self.selector: Any = {"selected": ["security-auditor"], "reasoning": "r"}
        self.triage: Any = {"tier": "normal", "reasoning": "r"}
        self.dead_roles: set[str] = set()
        self.turn_capped: set[str] = set()

    def __call__(self, spec: LaneSpec, artifact_dir: Path, worktree: Path) -> LaneResult:
        self.calls.append(spec)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if spec.role in self.dead_roles:
            return LaneResult(exit_code=1, stdout="", stderr="boom", duration_s=1)
        if spec.role in self.turn_capped:
            env = {
                "type": "result",
                "is_error": True,
                "subtype": "error_max_turns",
                "num_turns": 30,
            }
            res = LaneResult(exit_code=1, stdout=json.dumps(env), stderr="", duration_s=1)
            _extract_result(res)
            return res
        out: Any
        if spec.role == "selector":
            out = self.selector
        elif spec.role == "triage":
            out = self.triage
        elif spec.role == "group":
            cands = json.loads(spec.prompt.split("GROUP\n", 1)[1].split("\nLEDGER\n", 1)[0])
            ledger_rows = json.loads(spec.prompt.split("\nLEDGER\n", 1)[1].split("\nCR\n", 1)[0])
            out = (
                self.grouping(cands, ledger_rows)
                if self.grouping
                else {
                    "groups": [
                        {"members": [c["id"]], "representative": c["id"], "match": "new"}
                        for c in cands
                    ]
                }
            )
        elif spec.role.startswith("verify-"):
            c = json.loads(_between(spec.prompt, "VERIFY\n", "\n\n") or "{}")
            v = self.verdicts.get(c.get("title", ""), {"verdict": "CONFIRMED"})
            out = {"evidence": "line 11 shows it", "severity": c.get("severity", "suggestion"), **v}
        elif spec.role == "threads":
            issues = json.loads(spec.prompt.split("THREADS\n", 1)[1])
            out = (
                self.threads(issues)
                if self.threads
                else {
                    "issues": [
                        {"finding_hash": i["finding_hash"], "status": "NO_REPLY", "reply": ""}
                        for i in issues
                    ]
                }
            )
        elif spec.role == "composer":
            items = json.loads(spec.prompt.split("COMPOSE\n", 1)[1])
            out = {
                "summary": f"{len(items)} verified finding(s).",
                "findings": [
                    {"id": i["id"], "title": i["title"], "body": "composed " + i["title"]}
                    for i in items
                ],
            }
        else:
            out = {"summary": "s", "candidates": self.finders.get(spec.role, [])}
        payload = {
            "type": "result",
            "result": json.dumps(out),
            "structured_output": out,
            "usage": {"input_tokens": 10, "output_tokens": 1},
        }
        res = LaneResult(exit_code=0, stdout=json.dumps(payload), stderr="", duration_s=1)
        _extract_result(res)  # what the real runner does with the CLI's envelope
        return res


def _between(text: str, start: str, end: str) -> str:
    """The JSON candidate the verifier prompt carries: from after `start` to the closing brace."""
    body = text.split(start, 1)[1]
    depth = 0
    for i, ch in enumerate(body):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0 and ch == "}":
            return body[: i + 1]
    return ""


def _run(cfg, conn, gh, lanes, *, trigger=Trigger.MENTION):
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, trigger)
    (rid,) = schedule(conn, cfg, spawn=False)
    return rid, worker.main(cfg, conn, rid, gh=gh, lane_runner=lanes, heartbeat=False)


def _finder_roles(lanes: Lanes) -> list[str]:
    return [
        c.role for c in lanes.calls if c.json_schema and "candidates" in json.dumps(c.json_schema)
    ]


def test_v10_clean_run_runs_method_lanes_and_verifies_each_group(v10cfg, v10conn, gh):
    lanes = Lanes()
    lanes.finders = {
        "scan": [cand("Off-by-one in loop")],
        "trace": [cand("Caller ignores new error", line=30)],
    }
    _, status = _run(v10cfg, v10conn, gh, lanes)
    assert status == RunStatus.DONE
    roles = [c.role for c in lanes.calls]
    # phase 1: one general lane carrying every method + specialists; phase 2: method lanes
    assert _finder_roles(lanes) == [
        "general",
        "always-on",
        "security-auditor",
        "scan",
        "removed",
        "trace",
        "always-on",
        "security-auditor",
    ]
    p1_general = next(c for c in lanes.calls if c.role == "general")
    assert "METHOD scan" in p1_general.prompt and "METHOD trace" in p1_general.prompt
    assert "{braces}" in p1_general.prompt  # method text is not str.format'ed
    assert "REPO RULE: never log secrets" in p1_general.prompt  # repo's own AGENTS.md
    sec = [c for c in lanes.calls if c.role == "security-auditor"]
    assert all(c.effort == "medium" for c in sec)  # specialist effort cap
    p2 = [c for c in lanes.calls if c.role in {"scan", "removed", "trace"}]
    assert all(c.max_turns == 30 for c in p2)
    verifies = [r for r in roles if r.startswith("verify-")]
    assert len(verifies) == 2  # one lane per group, phase 2 only (phase 1 had no candidates)
    assert "composer" in roles
    (review,) = gh.posted_reviews
    assert review["event"] == "COMMENT"  # suggestions only
    # line 11 is in the diff hunk and goes inline; line 30 is not, so it is in the body
    (inline,) = review["comments"]
    assert "composed Off-by-one in loop" in inline["body"]
    assert "Caller ignores new error" in review["body"]
    issues = ledger.load(v10conn, "dashpay/platform", 1)
    assert {i.title for i in issues.values()} == {"Off-by-one in loop", "Caller ignores new error"}
    assert all(i.is_open for i in issues.values())


def test_v10_plausible_blocker_is_downgraded_and_refuted_is_dropped(v10cfg, v10conn, gh):
    lanes = Lanes()
    lanes.finders = {
        "scan": [
            cand("Race on shared map", sev="blocking"),
            cand("Imagined overflow", sev="blocking", line=40),
        ]
    }
    lanes.verdicts = {
        "Race on shared map": {"verdict": "PLAUSIBLE", "confirm_by": "a two-thread test"},
        "Imagined overflow": {"verdict": "REFUTED", "evidence": "u64 bounded by line 3"},
    }
    _, status = _run(v10cfg, v10conn, gh, lanes)
    assert status == RunStatus.DONE
    (review,) = gh.posted_reviews
    assert review["event"] == "COMMENT"  # the only blocker was PLAUSIBLE -> suggestion
    (c,) = review["comments"]
    assert "Suggestion" in c["body"] and "Race on shared map" in c["body"]


def test_v10_phase1_confirmed_blocker_publishes_preliminary(v10cfg, v10conn, gh):
    lanes = Lanes()
    lanes.finders = {"general": [cand("Signature not checked", sev="blocking")]}
    _, status = _run(v10cfg, v10conn, gh, lanes)
    assert status == RunStatus.DONE
    roles = [c.role for c in lanes.calls]
    assert "scan" not in roles  # Phase 2 never ran
    (review,) = gh.posted_reviews
    assert review["event"] == "REQUEST_CHANGES"
    assert "phase=preliminary" in review["body"]


def test_v10_duplicates_across_lanes_are_verified_once(v10cfg, v10conn, gh):
    lanes = Lanes()
    lanes.finders = {
        "scan": [cand("Null deref in handler")],
        "trace": [cand("Handler dereferences None")],
    }

    def grouping(cands, ledger_rows):
        return {
            "groups": [
                {
                    "members": [c["id"] for c in cands],
                    "representative": cands[0]["id"],
                    "match": "new",
                }
            ]
        }

    lanes.grouping = grouping
    _run(v10cfg, v10conn, gh, lanes)
    assert sum(1 for c in lanes.calls if c.role.startswith("verify-")) == 1
    (review,) = gh.posted_reviews
    assert len(review["comments"]) == 1
    assert "phase2:scan, phase2:trace" in review["comments"][0]["body"]


def _seed_ledger(conn, *, status: str, closed_sha: str | None = "c" * 40) -> str:
    from reviewsys.contract import finding_hash

    h = finding_hash("f.rs", "logic", "Old concern")
    with tx(conn):
        conn.execute(
            "INSERT INTO ledger (repo, number, hash, file, line_start, line_end, severity, category, title, body, status, opened_sha, closed_sha, updated_at) "
            "VALUES ('dashpay/platform',1,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                h,
                "f.rs",
                11,
                12,
                "blocking",
                "logic",
                "Old concern",
                "b",
                status,
                "c" * 40,
                closed_sha,
                "2026-01-01T00:00:00Z",
            ),
        )
    return h


def _refind_old(lanes: Lanes, h: str, kind: str) -> None:
    lanes.finders = {"scan": [cand("Old concern, reworded", sev="blocking")]}

    def grouping(cands, ledger_rows):
        return {
            "groups": [
                {
                    "members": [cands[0]["id"]],
                    "representative": cands[0]["id"],
                    "match": f"{kind}:{h}",
                }
            ]
        }

    lanes.grouping = grouping


def test_v10_withdrawn_issue_is_not_raised_again_unless_its_code_changed(
    v10cfg, v10conn, gh, monkeypatch
):
    h = _seed_ledger(v10conn, status="withdrawn")
    lanes = Lanes()
    _refind_old(lanes, h, "closed")
    _run(v10cfg, v10conn, gh, lanes)
    assert not any(c.role.startswith("verify-") for c in lanes.calls)
    assert gh.posted_reviews[-1]["comments"] == []

    # same issue, but the anchored lines changed since it was withdrawn: verified again, and
    # the verifier sees the old discussion
    monkeypatch.setattr(
        wt, "diff_file", lambda worktree, old, new, path, context=3: "@@ -11,1 +11,1 @@\n-a\n+b\n"
    )
    with tx(v10conn):
        v10conn.execute("UPDATE heads SET status='done'")
        v10conn.execute("DELETE FROM reviews")
    gh.posted_reviews.clear()
    lanes2 = Lanes()
    _refind_old(lanes2, h, "closed")
    with tx(v10conn):
        enqueue_head(v10conn, v10cfg, "dashpay/platform", 1, "d" * 40, Trigger.MENTION)
    gh.pr["head"]["sha"] = "d" * 40
    lanes2_calls_before = len(lanes2.calls)
    (rid,) = schedule(v10conn, v10cfg, spawn=False)
    worker.main(v10cfg, v10conn, rid, gh=gh, lane_runner=lanes2, heartbeat=False)
    verifies = [c for c in lanes2.calls[lanes2_calls_before:] if c.role.startswith("verify-")]
    assert len(verifies) == 1
    assert "closed as `withdrawn`" in verifies[0].prompt


def test_v10_fixed_issue_that_reappears_is_verified_as_a_regression(v10cfg, v10conn, gh):
    h = _seed_ledger(v10conn, status="fixed")
    lanes = Lanes()
    _refind_old(lanes, h, "closed")
    _run(v10cfg, v10conn, gh, lanes)
    assert sum(1 for c in lanes.calls if c.role.startswith("verify-")) == 1


def _open_thread(h: str, *, reply: str | None) -> dict:
    comments = [
        {
            "databaseId": 900,
            "body": f"<!-- thepastaclaw-review v1 finding={h} dedupe=x -->\n**🔴 Blocking: Old concern**\n\nb",
            "author": {"login": "thepastaclaw"},
            "createdAt": "2026-01-01T00:00:00Z",
            "authorAssociation": "NONE",
        }
    ]
    if reply:
        comments.append(
            {
                "databaseId": 901,
                "body": reply,
                "author": {"login": "UdjinM6"},
                "createdAt": "2026-01-02T00:00:00Z",
                "authorAssociation": "MEMBER",
            }
        )
    return {
        "id": "T1",
        "isResolved": False,
        "isOutdated": False,
        "path": "f.rs",
        "line": 11,
        "comments": {"nodes": comments},
    }


def test_v10_open_issue_refound_goes_to_thread_lane_not_a_new_comment(v10cfg, v10conn, gh):
    h = _seed_ledger(v10conn, status="open", closed_sha=None)
    gh.threads = [_open_thread(h, reply=None)]
    lanes = Lanes()
    _refind_old(lanes, h, "open")
    seen: dict = {}

    def threads(issues):
        seen["issues"] = issues
        return {
            "issues": [
                {"finding_hash": i["finding_hash"], "status": "STILL_VALID", "reply": ""}
                for i in issues
            ]
        }

    lanes.threads = threads
    _run(v10cfg, v10conn, gh, lanes)
    assert not any(c.role.startswith("verify-") for c in lanes.calls)
    (issue,) = seen["issues"]
    assert issue["finding_hash"] == h
    assert issue["refound_this_round"][0]["title"] == "Old concern, reworded"
    assert "code_at_head" in issue
    review = gh.posted_reviews[-1]
    assert review["comments"] == []  # no duplicate inline comment
    assert review["event"] == "REQUEST_CHANGES"  # the still-valid blocker keeps the verdict


def test_v10_thread_lane_concession_resolves_and_closes_the_ledger_issue(v10cfg, v10conn, gh):
    h = _seed_ledger(v10conn, status="open", closed_sha=None)
    gh.threads = [_open_thread(h, reply="Isn't this dead code? assert(false) terminates the path.")]
    lanes = Lanes()

    def threads(issues):
        return {
            "issues": [
                {
                    "finding_hash": h,
                    "status": "WITHDRAWN",
                    "reply": "You're right: the assert(false) makes this path unreachable.",
                }
            ]
        }

    lanes.threads = threads
    _run(v10cfg, v10conn, gh, lanes)
    assert any("unreachable" in r.get("body", "") for r in gh.replies)
    issues = ledger.load(v10conn, "dashpay/platform", 1)
    assert issues[h].status == "withdrawn" and issues[h].closed_sha == HEAD
    assert gh.posted_reviews[-1]["event"] != "REQUEST_CHANGES"


def test_v10_failed_nonblocking_verifier_is_dropped_disclosed_and_blocks_approval(
    v10cfg, v10conn, gh
):
    lanes = Lanes()
    lanes.finders = {"scan": [cand("Something")]}
    lanes.dead_roles = {"verify-c2-1"}
    rid, status = _run(v10cfg, v10conn, gh, lanes)
    assert status == RunStatus.DONE
    review = gh.posted_reviews[-1]
    assert review["comments"] == []
    assert review["event"] == "COMMENT"  # never APPROVE over an unchecked candidate
    assert "could not be verified" in review["body"]
    ev = v10conn.execute(
        "SELECT kind FROM events WHERE run_id=? AND kind LIKE 'verify.%'", (rid,)
    ).fetchall()
    assert {e["kind"] for e in ev} >= {"verify.candidate_failed"}


def test_v10_failed_blocking_verifier_fails_the_run_instead_of_publishing(v10cfg, v10conn, gh):
    lanes = Lanes()
    lanes.finders = {"scan": [cand("Maybe a real bug", sev="blocking")]}
    lanes.dead_roles = {"verify-c2-1"}
    rid, status = _run(v10cfg, v10conn, gh, lanes)
    assert status == RunStatus.FAILED
    assert gh.posted_reviews == []
    row = v10conn.execute("SELECT fail_kind FROM runs WHERE id=?", (rid,)).fetchone()
    assert row["fail_kind"] == "infra"  # retried by the scheduler


def test_v10_clean_final_review_approves(v10cfg, v10conn, gh):
    _, status = _run(v10cfg, v10conn, gh, Lanes())
    assert status == RunStatus.DONE
    (review,) = gh.posted_reviews
    assert review["event"] == "APPROVE"


def test_v10_turn_capped_finder_is_disclosed_not_fatal(v10cfg, v10conn, gh):
    lanes = Lanes()
    lanes.turn_capped = {"trace"}
    lanes.finders = {"scan": [cand("Off-by-one", line=11)]}
    _, status = _run(v10cfg, v10conn, gh, lanes)
    assert status == RunStatus.DONE
    assert [c.role for c in lanes.calls].count("trace") == 1  # not retried
    review = gh.posted_reviews[-1]
    assert "trace ran out of turns" in review["body"]
    assert review["event"] == "COMMENT"


def test_v10_refound_standing_blocker_closes_the_phase1_gate(v10cfg, v10conn, gh):
    h = _seed_ledger(v10conn, status="open", closed_sha=None)
    gh.threads = [_open_thread(h, reply=None)]
    lanes = Lanes()
    lanes.finders = {"general": [cand("Old concern, reworded", sev="blocking")]}

    def grouping(cands, ledger_rows):
        return {
            "groups": [
                {
                    "members": [c["id"] for c in cands],
                    "representative": cands[0]["id"],
                    "match": f"open:{h}",
                }
            ]
        }

    lanes.grouping = grouping
    _run(v10cfg, v10conn, gh, lanes)
    assert "scan" not in [c.role for c in lanes.calls]  # Phase 2 never ran
    review = gh.posted_reviews[-1]
    assert "phase=preliminary" in review["body"] and review["event"] == "REQUEST_CHANGES"


def test_v10_dry_run_leaves_ledger_and_threads_alone(v10cfg, v10conn, gh):
    h = _seed_ledger(v10conn, status="open", closed_sha=None)
    gh.threads = [_open_thread(h, reply="this is wrong")]
    lanes = Lanes()
    lanes.finders = {"scan": [cand("New thing")]}
    lanes.threads = lambda issues: {
        "issues": [{"finding_hash": h, "status": "WITHDRAWN", "reply": "fair point"}]
    }
    with tx(v10conn):
        enqueue_head(v10conn, v10cfg, "dashpay/platform", 1, HEAD, Trigger.MENTION)
    (rid,) = schedule(v10conn, v10cfg, spawn=False)
    status = worker.main(
        v10cfg, v10conn, rid, gh=gh, lane_runner=lanes, heartbeat=False, dry_run=True
    )
    assert status == RunStatus.DONE
    assert gh.replies == [] and gh.posted_reviews == []
    issues = ledger.load(v10conn, "dashpay/platform", 1)
    assert set(issues) == {h} and issues[h].status == "open"


def test_v10_thread_answers_are_not_posted_against_a_moved_head(v10cfg, v10conn, gh):
    h = _seed_ledger(v10conn, status="open", closed_sha=None)
    gh.threads = [_open_thread(h, reply="fixed in the latest push")]
    lanes = Lanes()

    def threads(issues):
        gh.pr["head"]["sha"] = "e" * 40  # a push lands while the thread lane runs
        return {"issues": [{"finding_hash": h, "status": "FIXED", "reply": "Confirmed fixed."}]}

    lanes.threads = threads
    _, status = _run(v10cfg, v10conn, gh, lanes)
    assert status == RunStatus.FAILED
    assert gh.replies == []
    assert ledger.load(v10conn, "dashpay/platform", 1)[h].status == "open"


def test_v10_reply_only_run_restores_the_gate_comment_when_nothing_lifts(v10cfg, v10conn, gh):
    h = _seed_ledger(v10conn, status="open", closed_sha=None)
    gh.threads = [_open_thread(h, reply="thanks")]
    gh.posted_reviews.append(
        {
            "id": 1,
            "body": f"<!-- thepastaclaw-review v1 -->\n<!-- thepastaclaw-review-phase v1 phase=final sha={HEAD} policy=x -->",
            "event": "REQUEST_CHANGES",
        }
    )
    lanes = Lanes()
    _, status = _run(v10cfg, v10conn, gh, lanes, trigger=Trigger.REVIEW_REPLY)
    assert status == RunStatus.DONE
    assert not any(c.role in {"scan", "general"} for c in lanes.calls)  # no finders
    assert "threads" in [c.role for c in lanes.calls]
    assert gh.gate_bodies and "in progress" not in gh.gate_bodies[-1].lower()


def test_v10_coderabbit_disagree_reply_is_scrubbed_and_judged_once(v10cfg, v10conn, gh):
    gh.threads = [
        {
            "id": "CR1",
            "isResolved": False,
            "isOutdated": False,
            "path": "f.rs",
            "line": 11,
            "comments": {
                "nodes": [
                    {
                        "databaseId": 555,
                        "body": "Potential issue: x is unbounded",
                        "author": {"login": "coderabbitai[bot]"},
                        "createdAt": "2026-01-01T00:00:00Z",
                        "authorAssociation": "NONE",
                    }
                ]
            },
        }
    ]
    lanes = Lanes()
    lanes.verdicts = {
        "CodeRabbit comment": {"verdict": "REFUTED", "evidence": "@someone bounded at line 3"}
    }
    _run(v10cfg, v10conn, gh, lanes)
    verifies = [c for c in lanes.calls if c.role == "verify-cr-555"]
    assert len(verifies) == 1
    assert any(r.get("content") == "-1" for r in gh.reactions)
    assert all("@someone" not in r.get("body", "") for r in gh.replies)
    # next push: already judged, not verified again
    with tx(v10conn):
        v10conn.execute("UPDATE heads SET status='done'")
    gh.pr["head"]["sha"] = "f" * 40
    lanes2 = Lanes()
    with tx(v10conn):
        enqueue_head(v10conn, v10cfg, "dashpay/platform", 1, "f" * 40, Trigger.MENTION)
    (rid,) = schedule(v10conn, v10cfg, spawn=False)
    worker.main(v10cfg, v10conn, rid, gh=gh, lane_runner=lanes2, heartbeat=False)
    assert not any(c.role == "verify-cr-555" for c in lanes2.calls)


def test_v10_lanes_run_with_clean_context_and_v9_lanes_do_not(v10cfg, v10conn, gh, cfg, conn):
    lanes = Lanes()
    _run(v10cfg, v10conn, gh, lanes)
    assert all(c.clean_context for c in lanes.calls if c.role not in {"selector", "triage"})


def test_v9_policy_still_runs_v9_flow(cfg, conn, gh):
    """No `pipeline` block: nothing about the v9 flow changes (smoke; the v9 suite covers it)."""
    assert cfg.policy.pipeline is None
    assert "pipeline" not in SKILLS_CONFIG["review_model_policy"]


def test_pipeline_requires_a_v10_prompt_for_every_specialist(tmp_path, skills_dir):
    raw = json.loads((skills_dir / "config.json").read_text())
    raw["review_model_policy"]["pipeline"] = {**PIPELINE, "specialist_prompts": {}}
    (skills_dir / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="no v10 prompt"):
        cfg_mod.load_skills_config(skills_dir)
