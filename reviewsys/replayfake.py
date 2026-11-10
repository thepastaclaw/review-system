"""Smoke-test doubles for the replay eval (`deploy/eval-run.sh --dry`): canned lanes, a canned
read-only `gh`, and a fixture (a repository, its mirror, a source database, two skills arms,
an eval set and labels) to run every step of the eval on, end to end, without a model or
GitHub. Never used by production: only `--fake-lanes` (replay, replay-set, replay-judge) and
`reviewsys replay-fixture` reach it.

- `FakeLanes`: answers every lane kind of policy v11 and v12 by role and by the v12 prompts'
  output sections, with the same findings on every head (`FINDINGS`), and the judge lane
  (`yes` to both questions).
- `python -m reviewsys.replayfake gh STATE ARGS`: a gh that serves the fixture PRs' reads and
  refuses every write. The fixture's `bin/gh` runs it.
- `build_fixture(root)`: everything the eval script needs, under `root`.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import config as cfg_mod
from .contract import finding_hash
from .db import connect, tx
from .lane import LaneResult, LaneSpec

REPO, BOT = "dashpay/platform", "thepastaclaw"
# what every fake finder / reviewer reports, on every head
FINDINGS = (
    {
        "file": "f.rs",
        "line_start": 3,
        "line_end": 4,
        "severity": "blocking",
        "confidence": 0.9,
        "category": "logic",
        "title": "Overflow in f3",
        "body": "f3 adds without a bound.",
    },
    {
        "file": "f.rs",
        "line_start": 15,
        "line_end": 15,
        "severity": "blocking",
        "confidence": 0.8,
        "category": "logic",
        "title": "Missing bounds check in f7",
        "body": "f7 indexes past the end.",
    },
)
PRIOR_RE = re.compile(
    r"### Prior findings requiring cumulative adjudication\s*```json\n(.*?)\n```", re.S
)


# ---- lanes ----


def _head(cwd: Path) -> str:
    r = subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return r.stdout.strip()


def _prior(prompt: str) -> list[dict[str, Any]]:
    m = PRIOR_RE.search(prompt)
    try:
        rows = json.loads(m.group(1)) if m else []
    except ValueError:
        rows = []
    return [
        {"finding_hash": r["finding_hash"], "status": "FIXED", "reason": "fake: rewritten"}
        for r in rows
        if isinstance(r, dict) and r.get("finding_hash")
    ]


def _phase(prompt: str, *markers: str) -> str:
    return "preliminary" if any(m in prompt for m in markers) else "final"


class FakeLanes:
    """A LaneRunner with canned answers. Every lane's prompt and answer go to its artifact
    dir, as a real lane's would."""

    def __init__(self) -> None:
        self._sessions: dict[str, str] = {}

    def answer(self, spec: LaneSpec) -> Any:
        p = spec.prompt
        if spec.resume and spec.session_id in self._sessions:
            p = self._sessions[spec.session_id]
        elif spec.session_id:
            self._sessions[spec.session_id] = p
        if spec.role == "judge":
            return {
                "real": "yes",
                "evidence": "f.rs:3 `fn f3() { 3 }` (fake)",
                "matches_label": "yes" if "## Question 2" in p else None,
                "reason": "fake judge",
            }
        if "## Output: candidates" in p:
            cands = [
                {**{k: v for k, v in f.items() if k != "confidence"}, "failure_scenario": "x"}
                for f in FINDINGS
            ]
            return {"summary": "ok", "candidates": cands}
        if "## Output: verified groups" in p:
            m = re.search(r"```json candidates\n(.*?)\n```", p, re.S)
            cands = json.loads(m.group(1)) if m else []
            return {
                "summary": "Verified.",
                "groups": [
                    {
                        "members": [c["id"]],
                        "match": "new",
                        "verdict": "CONFIRMED",
                        "evidence": f"{c['file']}:{c.get('line_start')} shows it",
                        "severity": c["severity"],
                        "title": c["title"],
                        "body": c.get("body") or "body",
                    }
                    for c in cands
                ],
            }
        if "## Output: thread outcomes" in p:
            m = re.search(r"```json issues\n(.*?)\n```", p, re.S)
            hashes = [i["finding_hash"] for i in json.loads(m.group(1))] if m else []
            return {
                "threads": [
                    {"finding_hash": h, "status": "STILL_VALID", "reply": "NO_REPLY"}
                    for h in hashes
                ]
            }
        if spec.role == "prep":
            return {
                "tier": "normal",
                "tier_reasoning": "fake",
                "selected": [],
                "selection_reasoning": "fake",
            }
        if spec.role == "selector":
            return {"selected": [], "reasoning": "fake"}
        if spec.role == "triage":
            return {"tier": "normal", "reasoning": "fake"}
        if spec.role in ("conversation", "thread"):
            return {"threads": []}
        if spec.role == "confirm":
            m = re.search(r"```json reported\n(.*?)\n```", p, re.S)
            reported = json.loads(m.group(1)) if m else []
            return {
                "decisions": [
                    {"id": r["id"], "verdict": "blocking", "failure": "fake", "reason": "fake"}
                    for r in reported
                ]
            }
        if spec.role == "stepback":
            return {
                "diagnosis": "converging",
                "themes": [],
                "recommendation": {"kind": "continue_normal", "text": ""},
                "decision_made": "yes",
            }
        prior = _prior(p)
        if spec.role == "verifier":
            return {
                "prior_finding_reconciliation": prior,
                "summary": "s",
                "review_action": "COMMENT",
                "findings": list(FINDINGS),
                "dropped_findings": [],
                "out_of_scope_findings": [],
                "coderabbit_reactions": [],
                "prerequisite_adjudications": [],
                "adjudication_complete": True,
                "review_phase": _phase(
                    p, "must be `preliminary`", '`review_phase` is exactly "preliminary"'
                ),
            }
        return {
            "summary": "ok",
            "findings": list(FINDINGS),
            "out_of_scope_findings": [],
            "prior_finding_reconciliation": prior,
            "review_phase": _phase(
                p, "set to `preliminary`", '`review_phase` is exactly "preliminary"'
            ),
            "head_sha": _head(spec.cwd),
        }

    def __call__(self, spec: LaneSpec, artifact_dir: Path, worktree: Path) -> LaneResult:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "prompt.md").write_text(spec.prompt)
        text = json.dumps(self.answer(spec))
        stdout = json.dumps({"result": text, "usage": {"input_tokens": 100, "output_tokens": 10}})
        (artifact_dir / "stdout.json").write_text(stdout)
        return LaneResult(
            exit_code=0,
            stdout=stdout,
            stderr="",
            duration_s=0.01,
            tokens_in=100,
            tokens_out=10,
            result_text=text,
        )


# ---- gh ----


def gh_main(argv: list[str], state_path: Path) -> int:
    """A read-only gh over the fixture's PRs (`state_path`: {"prs": {number: pull}}); every
    call is logged in `gh-calls.log` beside it."""
    state = json.loads(state_path.read_text())
    with (state_path.parent / "gh-calls.log").open("a") as fh:
        fh.write(json.dumps(argv) + "\n")
    if argv[:1] == ["--version"]:
        print("gh version 0.0.0-fake (reviewsys replayfake)")
        return 0
    method = argv[argv.index("--method") + 1] if "--method" in argv else "GET"
    if argv[:1] != ["api"] or len(argv) < 2 or (method != "GET" and argv[1] != "graphql"):
        print(f"fake gh: refused {argv[:3]}", file=sys.stderr)
        return 1
    ep = argv[1]
    out: Any
    if ep == "graphql":
        q = next((a for a in argv if a.startswith("query=")), "")
        if "mutation" in q:
            print("fake gh: refused a mutation", file=sys.stderr)
            return 1
        page = {"pageInfo": {"hasNextPage": False}, "nodes": []}
        if "reviewThreads(" in q:
            out = {"data": {"repository": {"pullRequest": {"reviewThreads": page}}}}
        elif "pullRequests(" in q:
            out = {"data": {"repository": {"pullRequests": page}}}
        else:
            out = {"data": {}}
    else:
        m = re.match(r"repos/[^/]+/[^/]+/(pulls|issues)/(\d+)(/.*)?$", ep.split("?")[0])
        pr = (state.get("prs") or {}).get(m.group(2)) if m else None
        if ep == "user":
            out = {"login": BOT}
        elif "/check-runs" in ep:
            out = {"check_runs": []}
        elif "/status" in ep:
            out = {"state": "success", "statuses": []}
        elif m and m.group(3) == "/files":
            out = (pr or {}).get("files") or []
        elif m and m.group(3):
            out = []  # comments, reviews, labels, events, commits: none in the fixture
        elif m and pr is not None:
            out = pr
        elif "/labels/" in ep:
            out = {"name": ep.rsplit("/", 1)[1]}
        else:
            print(f"fake gh: no route for {ep}", file=sys.stderr)
            return 1
    if "--jq" in argv:
        print("fake gh: --jq is not supported", file=sys.stderr)
        return 1
    print(json.dumps(out))
    return 0


# ---- the fixture ----

SKILLS_CONFIG: dict[str, Any] = {
    "version": 2,
    "repos": [
        {"owner": "dashpay", "name": "platform", "skill_path": "skills/platform/", "enabled": True}
    ],
    "settings": {
        "max_concurrent_reviews": 2,
        "priority_review_overflow_slots": 1,
        "agent_timeout_minutes": 30,
        "comment_budget": 10,
        "selector_model": "gpt-6.1-sol",
        "debounce_minutes": 30,
    },
    "review_model_policy": {
        "name": "replay-fixture-v11",
        "version": 11,
        "fingerprint": "",
        "phase2_enabled": True,
        "phase1": {
            "reviewer": {"agent": "phase1-reviewer", "model": "glm-5.3-flash", "reasoning": "max"},
            "verifier": {"agent": "sol-verifier", "model": "gpt-6.1-sol", "reasoning": "high"},
        },
        "phase2": {
            "reviewer": {"agent": "phase2-reviewer", "model": "gpt-6.1-sol", "reasoning": "high"},
            "verifier": {"agent": "sol-verifier", "model": "gpt-6.1-sol", "reasoning": "high"},
        },
        "triage": {
            "agent": "triage",
            "model": "gpt-6.1-sol",
            "reasoning": "low",
            "fallback_tier": "normal",
            "tiers": {
                "trivial": {"phase1": "high", "phase2": None},
                "low": {"phase1": "high", "phase2": "medium"},
                "normal": {"phase1": "max", "phase2": "high"},
                "critical": {"phase1": "max", "phase2": "xhigh"},
            },
        },
    },
    "specialists": [],
}
PIPELINE_V12 = {
    "version": "v12",
    "lane_env_isolation": True,
    "structured_output": False,
    "finders": {"anchoring": "unanchored"},
    "threads": {"lane": True},
    "verifier": {"contract": "v12"},
    "confirmation": {"roles": ["general"], "trigger": "no_blockers"},
    "late_nonblocking": "post",
}
TEMPLATES = {
    "prompts/review-agent.md": "# Code Review Agent for {repo}\n{project_skill}\n{review_skill}\n"
    "PR #{pr_number}: {pr_title}\n{pr_description}\nbase={base_branch} head={head_sha}\n"
    "{incremental_context}\n{incremental_instructions}\n",
    "prompts/verifier-agent.md": "# Verifier\n{review_skill}\nPR #{pr_number} in {repo} head "
    "{head_sha}\nClaude:\n{claude_findings}\nCodex:\n{codex_findings}\nCR:\n"
    "{coderabbit_findings}\nbudget {comment_budget}\n",
    "skills/platform/project.md": "PROJECT SKILL",
    "skills/platform/review.md": "REVIEW SKILL",
    "prompts/v12/finder.md": "# Finder: {lane_title}\n{repo} #{pr_number} {pr_title}\n"
    "{pr_description}\nbase={base_branch} head={head_sha} range={coverage_from}\n"
    "{project_skill}\n{review_skill}\n## Method\n{method}\n## Context\n{context}\n",
    "prompts/v12/general.md": "General method for {repo}: scan, removed code, callers.",
    "prompts/v12/verifier.md": "# Verify+match {repo} #{pr_number} head {head_sha}\n"
    "{phase_note}\n## Candidates\n{candidates}\n## CodeRabbit\n{coderabbit}\n## Ledger\n"
    "{ledger}\n{context}\n",
    "prompts/v12/thread.md": "# Threads {repo} #{pr_number} at {head_sha}\n{mode_note}\n"
    "verdict {standing_verdict}\n{issues}\n{linked_commits}\n{discussion}\n",
}


def _git(cwd: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.com",
        "GIT_COMMITTER_NAME": "fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.com",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    r = subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, env=env, check=False
    )
    if r.returncode != 0:
        raise SystemExit(f"fixture: git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def write_skills(d: Path, *, v12: bool) -> None:
    """A fixture skills checkout: one enabled repo, no specialists, policy v11 or v12."""
    raw = json.loads(json.dumps(SKILLS_CONFIG))
    pol = raw["review_model_policy"]
    if v12:
        pol["name"], pol["version"], pol["pipeline"] = "replay-fixture-v12", 12, PIPELINE_V12
    pol["fingerprint"] = cfg_mod.policy_fingerprint(pol)
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(raw, indent=1))
    for rel, text in TEMPLATES.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text)


def _source_run(
    conn: sqlite3.Connection, number: int, sha: str, started: str, finished: str
) -> int:
    hid = conn.execute(
        "INSERT INTO heads (repo, number, sha, trigger, status, queued_at, eligible_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (REPO, number, sha, "new_push", "done", started, started),
    ).lastrowid
    rid = conn.execute(
        "INSERT INTO runs (head_id, attempt, status, token, started_at, finished_at, "
        "deadline_at, tier, phase) VALUES (?,?,?,?,?,?,?,?,?)",
        (hid, 1, "done", "t", started, finished, started, "normal", "publish"),
    ).lastrowid
    return int(rid or 0)


def _label(head: dict[str, Any], **kw: Any) -> dict[str, Any]:
    keys = ("repo", "number", "sha")
    return {**{k: head[k] for k in keys}, "head_id": head["id"], "severity": "blocking", **kw}


def build_fixture(root: Path) -> dict[str, Any]:
    """Build the smoke-test world under `root` and return where everything is: PR 1 with an
    earlier reviewed round (a posted blocker the replay must reconcile) and PR 2 with none,
    each with a source run whose evidence is on disk; a mirror whose `main` moved on after the
    cutoffs; a skills repository with a v11 and a v12 commit; a set (PR 1 repeated) and
    labels (one the fake lanes find, one only the judge can match)."""
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    src = root / "src"
    src.mkdir()
    _git(src, "init", "-q", "-b", "main")
    (src / "f.rs").write_text("".join(f"fn f{i}() {{}}\n" for i in range(20)))
    _git(src, "add", ".")
    _git(src, "commit", "-qm", "base")
    base = _git(src, "rev-parse", "HEAD")
    heads: dict[int, list[str]] = {}
    for number, rounds in ((1, 2), (2, 1)):
        _git(src, "checkout", "-q", "-b", f"pr{number}", base)
        for k in range(rounds):
            (src / "f.rs").write_text(
                "".join(f"fn f{i}() {{ {i * (number + k)} }}\n" for i in range(20))
            )
            _git(src, "commit", "-qam", f"pr{number} round {k + 1}")
            heads.setdefault(number, []).append(_git(src, "rev-parse", "HEAD"))
    _git(src, "checkout", "-q", "main")
    _git(src, "commit", "-q", "--allow-empty", "-m", "main moves on after the cutoff")
    work = root / "work"
    mirror = work / "mirrors" / f"{REPO.replace('/', '__')}.git"
    mirror.parent.mkdir(parents=True)
    _git(root, "clone", "-q", "--bare", str(src), str(mirror))
    _git(mirror, "config", "remote.origin.fetch", "+refs/heads/*:refs/heads/*")
    for number, shas in heads.items():
        _git(mirror, "update-ref", f"refs/pull/{number}/head", shas[-1])
    db = root / "review.db"
    conn = connect(db)
    early_t, cut1, cut2 = "2026-09-20T10:00:00Z", "2026-09-20T12:00:00Z", "2026-09-21T12:00:00Z"
    prior_title = "Early blocker in f10"
    prior_hash = finding_hash("f.rs", "logic", prior_title)
    with tx(conn):
        early = _source_run(conn, 1, heads[1][0], "2026-09-20T09:00:00Z", early_t)
        conn.execute(
            "INSERT INTO findings (run_id, phase, stage, hash, file, line_start, line_end, "
            "severity, category, title, body) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (early, "final", "posted", prior_hash, "f.rs", 10, 11, "blocking", "logic",
             prior_title, "b"),
        )  # fmt: skip
        conn.execute(
            "INSERT INTO posted_findings (repo, number, hash, sha, review_id, posted_at) "
            "VALUES (?,?,?,?,?,?)",
            (REPO, 1, prior_hash, heads[1][0], 11, early_t),
        )
        conn.execute(
            "INSERT INTO reviews (run_id, repo, number, sha, phase, github_review_id, event, "
            "posted_at) VALUES (?,?,?,?,?,?,?,?)",
            (early, REPO, 1, heads[1][0], "final", 11, "REQUEST_CHANGES", early_t),
        )
        run1 = _source_run(conn, 1, heads[1][1], cut1, "2026-09-20T13:00:00Z")
        run2 = _source_run(conn, 2, heads[2][0], cut2, "2026-09-21T13:00:00Z")
        for number in heads:
            conn.execute(
                "INSERT INTO prs (repo, number, head_sha, state, updated_at) VALUES (?,?,?,?,?)",
                (REPO, number, heads[number][-1], "open", cut1),
            )
    conn.close()
    for rid in (run1, run2):
        d = work / "runs" / f"run-{rid}"
        d.mkdir(parents=True)
        ev = {
            "pr": {"title": f"Fixture PR (run {rid})", "body": "Body then",
                   "baseRefName": "main", "baseRefOid": base},
            "ci": {"fetched_at": cut1, "checks": []},
        }  # fmt: skip
        (d / "evidence.json").write_text(json.dumps({"evidence": ev}))
    skills = root / "skills"
    write_skills(skills, v12=False)
    _git(skills, "init", "-q", "-b", "main")
    _git(skills, "add", ".")
    _git(skills, "commit", "-qm", "policy v11 (fixture)")
    v11 = _git(skills, "rev-parse", "HEAD")
    write_skills(skills, v12=True)
    _git(skills, "add", ".")
    _git(skills, "commit", "-qm", "policy v12 (fixture)")
    v12 = _git(skills, "rev-parse", "HEAD")
    prs = {
        str(n): {
            "number": n,
            "title": f"Fixture PR {n}",
            "body": "Body now",
            "state": "open",
            "draft": False,
            "merged": False,
            "html_url": f"https://github.com/{REPO}/pull/{n}",
            "user": {"login": "someone"},
            "base": {"ref": "main", "sha": base},
            "head": {"sha": shas[-1], "ref": f"pr{n}"},
            "files": [{"filename": "f.rs"}],
        }
        for n, shas in heads.items()
    }
    (root / "gh.json").write_text(json.dumps({"prs": prs}))
    bin_dir = root / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" -m reviewsys.replayfake gh "{root / "gh.json"}" "$@"\n'
    )
    gh.chmod(0o755)
    toml = (
        cfg_mod.DEFAULT_TOML.replace('db = "~/.reviewsys/review.db"', f'db = "{db}"')
        .replace('work = "~/.reviewsys/work"', f'work = "{work}"')
        .replace('skills = "~/Projects/skills"', f'skills = "{skills}"')
        .replace('claude = "~/.openclaw/bin/claude"', 'claude = "/nonexistent/claude"')
        .replace('gh = "gh"', f'gh = "{gh}"')
    )
    config = root / "config.toml"
    config.write_text(toml)
    h1 = {"id": f"platform-1-{heads[1][1][:8]}", "repo": REPO, "number": 1, "sha": heads[1][1]}
    h2 = {"id": f"platform-2-{heads[2][0][:8]}", "repo": REPO, "number": 2, "sha": heads[2][0]}
    setfile = root / "set.json"
    setfile.write_text(
        json.dumps(
            {
                "heads": [
                    {**h1, "source_run_id": run1, "set": "S1", "cutoff": cut1},
                    {**h2, "source_run_id": run2, "set": "S4", "cutoff": cut2},
                ],
                "repeats": [h1["id"]],
            },
            indent=1,
        )
    )
    labels = [
        _label(h1, file="f.rs", line_start=3, line_end=4, title="Overflow in f3", label="tp",
               set="S1", source="late_unchanged_fixed"),
        # near the canned f3 finding with no title overlap: only the judge can match it
        _label(h2, file="f.rs", line_start=4, line_end=5, title="Arithmetic wraps on large input",
               label="fp", set="S4", source="withdrawn"),
    ]  # fmt: skip
    labelfile = root / "labels.jsonl"
    labelfile.write_text("".join(json.dumps(x) + "\n" for x in labels))
    return {
        "root": str(root),
        "config": str(config),
        "db": str(db),
        "runs": str(work / "runs"),
        "skills_repo": str(skills),
        "v11": v11,
        "v12": v12,
        "set": str(setfile),
        "labels": str(labelfile),
        "source_runs": [run1, run2],
    }


def main(argv: list[str]) -> int:
    if argv[:1] == ["gh"] and len(argv) >= 2:
        return gh_main(argv[2:], Path(argv[1]))
    if argv[:1] == ["fixture"] and len(argv) == 2:
        print(json.dumps(build_fixture(Path(argv[1]))))
        return 0
    print("usage: python -m reviewsys.replayfake (gh STATE ARGS... | fixture DIR)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
