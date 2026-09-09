"""Shared fixtures: temp config + skills dir, in-memory-ish SQLite, fake gh and fake claude lanes."""

from __future__ import annotations

import json
import subprocess
import urllib.parse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from reviewsys import config as cfg_mod
from reviewsys import db as db_mod
from reviewsys.gh import Gh
from reviewsys.lane import LaneResult, LaneSpec
from reviewsys.notify import Notifier

SKILLS_CONFIG = {
    "version": 2,
    "repos": [
        {"owner": "dashpay", "name": "platform", "skill_path": "skills/platform/", "enabled": True},
        {
            "owner": "dashpay",
            "name": "grovedb",
            "skill_path": "skills/grovedb/",
            "enabled": False,
            "disabled_reason": "ban",
        },
    ],
    "settings": {
        "max_concurrent_reviews": 2,
        "priority_review_overflow_slots": 1,
        "agent_timeout_minutes": 180,
        "comment_budget": 10,
        "selector_model": "gpt-5.6-terra",
        "debounce_minutes": 30,
    },
    "review_model_policy": {
        "name": "glm-tiered-astra-final-v3",
        "version": 3,
        "fingerprint": "3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f",
        "phase2_enabled": True,
        "phase1": {
            "reviewer": {"agent": "phase1-reviewer", "model": "glm-5.3-flash", "reasoning": "max"},
            "verifier": {"agent": "sol-verifier", "model": "gpt-5.6-sol", "reasoning": "high"},
        },
        "phase2": {
            "reviewer": {"agent": "phase2-reviewer", "model": "gpt-6-astra", "reasoning": "high"},
            "verifier": {"agent": "astra-verifier", "model": "gpt-6-astra", "reasoning": "high"},
        },
        "triage": {
            "agent": "triage",
            "model": "gpt-6-astra",
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
    "specialists": [
        {
            "id": "security-auditor",
            "prompt": "prompts/security-auditor.md",
            "repos": ["dashpay/platform"],
            "description": "security",
        },
        {
            "id": "always-on",
            "prompt": "prompts/always-on.md",
            "repos": ["dashpay/platform"],
            "description": "always",
            "always_run": True,
        },
    ],
}

REVIEW_AGENT_MD = """# Code Review Agent for {repo}
{project_skill}
{review_skill}
PR #{pr_number}: {pr_title}
{pr_description}
base={base_branch} head={head_sha}
{incremental_context}
{incremental_instructions}
"""
VERIFIER_MD = """# Verifier
{review_skill}
PR #{pr_number} in {repo} head {head_sha}
Claude:
{claude_findings}
Codex:
{codex_findings}
CR:
{coderabbit_findings}
budget {comment_budget}
"""


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    d = tmp_path / "skills"
    (d / "prompts").mkdir(parents=True)
    (d / "skills" / "platform").mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(SKILLS_CONFIG))
    (d / "prompts" / "review-agent.md").write_text(REVIEW_AGENT_MD)
    (d / "prompts" / "verifier-agent.md").write_text(VERIFIER_MD)
    (d / "prompts" / "security-auditor.md").write_text(
        REVIEW_AGENT_MD.replace("Code Review Agent", "Security Auditor")
    )
    (d / "prompts" / "always-on.md").write_text(
        REVIEW_AGENT_MD.replace("Code Review Agent", "Always On")
    )
    (d / "skills" / "platform" / "project.md").write_text("PROJECT SKILL")
    (d / "skills" / "platform" / "review.md").write_text("REVIEW SKILL")
    return d


@pytest.fixture
def cfg(tmp_path: Path, skills_dir: Path) -> cfg_mod.Config:
    toml = (
        cfg_mod.DEFAULT_TOML.replace(
            'db = "~/.reviewsys/review.db"', f'db = "{tmp_path}/review.db"'
        )
        .replace('work = "~/.reviewsys/work"', f'work = "{tmp_path}/work"')
        .replace('skills = "~/Projects/skills"', f'skills = "{skills_dir}"')
        .replace('claude = "~/.openclaw/bin/claude"', '"claude" = "/nonexistent/claude"')
    )
    p = tmp_path / "config.toml"
    p.write_text(toml)
    return cfg_mod.load(p)


@pytest.fixture
def conn(cfg: cfg_mod.Config):
    c = db_mod.connect(cfg.db_path)
    yield c
    c.close()


class _GhFailure(Exception):
    """Raised inside FakeGh dispatch to simulate a non-zero gh exit."""


class FakeGh(Gh):
    """Scripted gh: `routes` maps a (method, endpoint-prefix) or graphql opname to a response or callable."""

    def __init__(self) -> None:
        super().__init__("gh", runner=self._runner)
        self.calls: list[list[str]] = []
        self.routes: dict[str, Any] = {}
        self.posted_reviews: list[dict[str, Any]] = []
        self.gate_bodies: list[str] = []
        self.replies: list[dict[str, Any]] = []
        self.reactions: list[dict[str, Any]] = []
        self.open_prs: dict[str, list[dict[str, Any]]] = {}
        self.notifications: list[dict[str, Any]] = []
        self.inline: list[dict[str, Any]] = []
        self.issue_comments: list[dict[str, Any]] = []
        self.threads: list[dict[str, Any]] = []
        self.labels: list[str] = []  # current PR labels
        self.label_calls: list[tuple[str, str]] = []  # (method, label) mutations
        self.labels_defined = True  # False: repo has no pastaclaw:* labels -> the add 404s
        self.pr: dict[str, Any] = {
            "title": "T",
            "body": "B",
            "base": {"ref": "develop"},
            "head": {"sha": "a" * 40},
            "user": {"login": "someone"},
            "state": "open",
            "draft": False,
            "html_url": "https://x",
            "merged": False,
        }
        self.diff = "diff --git a/f.rs b/f.rs\n+++ b/f.rs\n@@ -1,3 +10,5 @@\n+x\n"
        self.files = [{"filename": "f.rs"}]
        self.fail_next: list[str] = []

    def _runner(
        self, argv: Sequence[str], stdin: str | None, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        args = list(argv)[1:]
        self.calls.append(args)
        if self.fail_next:
            msg = self.fail_next.pop(0)
            return subprocess.CompletedProcess(args, 1, "", msg)
        try:
            out = self._dispatch(args, stdin)
        except _GhFailure as exc:
            return subprocess.CompletedProcess(args, 1, "", str(exc))
        return subprocess.CompletedProcess(
            args, 0, out if isinstance(out, str) else json.dumps(out), ""
        )

    def _labels(self, ep: str, method: str, body: dict[str, Any] | None) -> Any:
        if method == "GET":
            return [{"name": n} for n in self.labels]
        if method == "DELETE":
            name = urllib.parse.unquote(ep.rsplit("/labels/", 1)[1])
            self.label_calls.append(("DELETE", name))
            self.labels = [n for n in self.labels if n != name]
            return {}
        if method == "POST":
            for name in (body or {})["labels"]:  # GitHub auto-creates missing labels here
                self.label_calls.append(("POST", name))
                if name not in self.labels:
                    self.labels.append(name)
            return [{"name": n} for n in self.labels]
        raise AssertionError(f"unrouted label call {method} {ep}")

    def _dispatch(self, args: list[str], stdin: str | None) -> Any:
        if args[:2] == ["pr", "diff"]:
            return self.diff
        if args[:2] == ["api", "graphql"]:
            q = next(a for a in args if a.startswith("query="))
            if "pullRequests(" in q:
                name = next(a for a in args if a.startswith("name=")).split("=", 1)[1]
                owner = next(a for a in args if a.startswith("owner=")).split("=", 1)[1]
                return {
                    "data": {
                        "repository": {
                            "pullRequests": {
                                "pageInfo": {"hasNextPage": False},
                                "nodes": self.open_prs.get(f"{owner}/{name}", []),
                            }
                        }
                    }
                }
            if "reviewThreads(" in q:
                return {
                    "data": {
                        "repository": {
                            "pullRequest": {
                                "reviewThreads": {
                                    "pageInfo": {"hasNextPage": False},
                                    "nodes": self.threads,
                                }
                            }
                        }
                    }
                }
            if "resolveReviewThread" in q:
                return {
                    "data": {"resolveReviewThread": {"thread": {"id": "t", "isResolved": True}}}
                }
            if "userContentEdits" in q and "graphql:userContentEdits" in self.routes:
                return {"data": self.routes["graphql:userContentEdits"]}
            if "node(id:" in q:
                return {
                    "data": {
                        "node": {
                            "pullRequestReviewThread": {
                                "id": "t",
                                "isResolved": False,
                                "comments": {"nodes": []},
                            }
                        }
                    }
                }
            return {"data": {}}
        if args[0] == "api":
            ep = args[1]
            method = args[args.index("--method") + 1] if "--method" in args else "GET"
            body = json.loads(stdin) if stdin else None
            if "/issues/comments/" in ep and method == "PATCH":
                new_body = (body or {})["body"]
                self.gate_bodies.append(new_body)
                route = self.routes.get(ep)
                if isinstance(route, dict):
                    self.routes[ep] = {**route, "body": new_body}
                return {"id": 77}
            if ep in self.routes:
                return self.routes[ep]
            if "/issues/" in ep and "/labels" in ep:
                return self._labels(ep, method, body)
            if "/labels/" in ep and method == "GET":  # repo label definition lookup
                if not self.labels_defined:
                    raise _GhFailure("HTTP 404: Not Found")
                return {"name": urllib.parse.unquote(ep.rsplit("/labels/", 1)[1])}
            if ep.startswith("/notifications"):
                return self.notifications
            if ep == "user":
                return {"login": "thepastaclaw"}
            if ep == "rate_limit":
                return {"resources": {"core": {"remaining": 5000}}}
            if "/pulls/" in ep and ep.endswith("/reviews") and method == "POST":
                rid = 5000 + len(self.posted_reviews)
                self.posted_reviews.append({"id": rid, **(body or {})})
                return {
                    "id": rid,
                    "html_url": f"https://github.com/r/pull/1#pullrequestreview-{rid}",
                }
            if "/pulls/" in ep and "/reviews" in ep:
                states = {"REQUEST_CHANGES": "CHANGES_REQUESTED", "APPROVE": "APPROVED"}
                return [
                    {
                        "id": r["id"],
                        "user": {"login": "thepastaclaw"},
                        "body": r.get("body", ""),
                        "state": r.get("state") or states.get(r.get("event", ""), "COMMENTED"),
                    }
                    for r in self.posted_reviews
                ]
            if "/pulls/" in ep and "/comments/" in ep and ep.endswith("/replies"):
                self.replies.append(body or {})
                return {"id": 1}
            if "/pulls/comments/" in ep and ep.endswith("/reactions"):
                self.reactions.append(body or {})
                return {"id": 1}
            if "/pulls/" in ep and "/comments" in ep:
                return self.inline
            if "/pulls/" in ep and "/files" in ep:
                return self.files
            if "/issues/" in ep and ep.endswith("/comments") and method == "POST":
                self.gate_bodies.append((body or {})["body"])
                self.issue_comments.append(
                    {"id": 77, "user": {"login": "thepastaclaw"}, "body": (body or {})["body"]}
                )
                return {"id": 77}
            if "/issues/" in ep and "/comments" in ep:
                return self.issue_comments
            if "/pulls/" in ep:
                return self.pr
            if ep.startswith("https://api.github.com/") or (
                ep.startswith("repos/") and "/comments/" in ep
            ):
                return self.routes.get(ep, {"body": "", "user": {"login": "x"}})
        raise AssertionError(f"unrouted gh call: {args}")


@pytest.fixture
def gh() -> FakeGh:
    return FakeGh()


class FakeLanes:
    """Canned model outputs keyed by role for reviewer lanes and by phase for verifiers."""

    def __init__(self, head_sha: str) -> None:
        self.head = head_sha
        self.calls: list[LaneSpec] = []
        self.reviewer: dict[str, Any] = {}
        self.verifier: dict[str, Any] = {}
        self.selector: Any = {"selected": ["security-auditor"], "reasoning": "rust crypto"}
        self.triage: Any = {"tier": "normal", "reasoning": "ordinary change"}
        self.broken_once: set[str] = set()
        self.timeout_roles: set[str] = set()

    def __call__(self, spec: LaneSpec, artifact_dir: Path, worktree: Path) -> LaneResult:
        self.calls.append(spec)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if spec.role in self.timeout_roles:
            return LaneResult(exit_code=None, stdout="", stderr="", duration_s=1, timed_out=True)
        if spec.role == "selector":
            out = self.selector
        elif spec.role == "triage":
            out = self.triage
        elif spec.role == "repair":
            # the broken payload is the original JSON with its closing brace removed
            out = json.loads(spec.prompt.split("Do not add fences.\n\n", 1)[1] + "}")
        elif spec.role == "verifier":
            phase = "preliminary" if "must be `preliminary`" in spec.prompt else "final"
            out = self.verifier.get(phase) or self.verifier["default"]
            out = {**out, "review_phase": phase}
        else:
            phase = "preliminary" if "set to `preliminary`" in spec.prompt else "final"
            base = self.reviewer.get(spec.role) or self.reviewer["default"]
            out = {**base, "review_phase": phase, "head_sha": self.head}
        text = json.dumps(out)
        if spec.role in self.broken_once:
            self.broken_once.discard(spec.role)
            text = text[:-1]  # truncated JSON: unrecoverable by the tolerant parser
        return LaneResult(
            exit_code=0,
            stdout=json.dumps(
                {"result": text, "usage": {"input_tokens": 100, "output_tokens": 10}}
            ),
            stderr="",
            duration_s=1,
            tokens_in=100,
            tokens_out=10,
            result_text=text,
        )


@pytest.fixture
def lanes() -> FakeLanes:
    return FakeLanes("a" * 40)


class FakeNotifier(Notifier):
    def __init__(self, cfg: cfg_mod.Config) -> None:
        super().__init__(
            cfg, runner=lambda argv: subprocess.CompletedProcess(list(argv), 0, "", "")
        )


@pytest.fixture
def notifier(cfg: cfg_mod.Config) -> FakeNotifier:
    return FakeNotifier(cfg)
