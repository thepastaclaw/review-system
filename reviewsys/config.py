"""Configuration: a TOML file for runtime knobs plus the skills repo's config.json.

The skills repo (thepastaclaw/skills) remains the source of truth for repos,
specialists, model policy and prompts. reviewsys reads it read-only.
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(
    os.environ.get("REVIEWSYS_CONFIG", "~/.reviewsys/config.toml")
).expanduser()


@dataclass(frozen=True, slots=True)
class LaneModel:
    agent: str
    model: str
    effort: str = "high"


EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True, slots=True)
class TierEffort:
    """Reviewer effort for one triage tier. `phase2=None` means Phase 2 is skipped."""

    phase1: str
    phase2: str | None


@dataclass(frozen=True, slots=True)
class ModelPolicy:
    name: str
    fingerprint: str
    phase1_reviewer: LaneModel
    phase1_verifier: LaneModel
    phase2_reviewer: LaneModel
    phase2_verifier: LaneModel
    repair_model: str
    selector_model: str
    phase2_enabled: bool = True
    # complexity triage: None = not configured, every PR is `fallback_tier`
    triage: LaneModel | None = None
    tiers: dict[str, TierEffort] = field(default_factory=dict)
    fallback_tier: str = "normal"

    def tier_effort(self, tier: str) -> TierEffort:
        default = TierEffort(phase1=self.phase1_reviewer.effort, phase2=self.phase2_reviewer.effort)
        return self.tiers.get(tier) or self.tiers.get(self.fallback_tier) or default


@dataclass(frozen=True, slots=True)
class RepoConfig:
    repo: str
    skill_path: str
    enabled: bool
    disabled_reason: str | None = None


@dataclass(frozen=True, slots=True)
class Specialist:
    id: str
    prompt: str
    repos: tuple[str, ...]
    description: str
    always_run: bool = False
    trigger_hint: str | None = None


@dataclass(frozen=True, slots=True)
class Config:
    # paths
    db_path: Path
    skills_dir: Path
    work_dir: Path  # worktrees, mirrors, run artifacts
    claude_bin: str
    gh_bin: str
    openclaw_bin: str
    # scheduling
    max_concurrent: int
    priority_overflow: int
    debounce_minutes: int
    max_attempts: int
    retry_backoff_minutes: tuple[int, ...]
    lane_timeout_minutes: int
    lane_budget_usd: float
    run_timeout_minutes: int
    heartbeat_seconds: int
    heartbeat_stale_minutes: int
    ingest_interval_seconds: int
    notify_interval_seconds: int
    tick_seconds: int
    watchdog_minutes: int
    comment_budget: int
    # identity / alerting
    bot_login: str
    slack_target: str | None
    slack_account: str
    agent_session_key: str
    trusted_reviewers: tuple[str, ...]
    # retention
    artifact_retention_days: int
    worktree_budget_gb: int
    # from skills config
    repos: tuple[RepoConfig, ...]
    specialists: tuple[Specialist, ...]
    policy: ModelPolicy
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def enabled_repos(self) -> tuple[str, ...]:
        return tuple(r.repo for r in self.repos if r.enabled)

    def repo(self, repo: str) -> RepoConfig | None:
        for r in self.repos:
            if r.repo == repo:
                return r
        return None

    def specialists_for(self, repo: str) -> tuple[Specialist, ...]:
        return tuple(s for s in self.specialists if repo in s.repos)

    @property
    def mirrors_dir(self) -> Path:
        return self.work_dir / "mirrors"

    @property
    def worktrees_dir(self) -> Path:
        return self.work_dir / "worktrees"

    @property
    def runs_dir(self) -> Path:
        return self.work_dir / "runs"

    @property
    def logs_dir(self) -> Path:
        return self.work_dir / "logs"


DEFAULT_TOML = """\
# reviewsys configuration
[paths]
db = "~/.reviewsys/review.db"
skills = "~/Projects/skills"
work = "~/.reviewsys/work"
claude = "~/.openclaw/bin/claude"
gh = "gh"
openclaw = "openclaw"

[scheduling]
max_concurrent = 2
priority_overflow = 1
debounce_minutes = 30
max_attempts = 3
retry_backoff_minutes = [5, 15, 45]
lane_timeout_minutes = 180
# runaway guard only (claude's list-price estimate, not real spend); a normal lane is $3-10
lane_budget_usd = 50.0
run_timeout_minutes = 360
heartbeat_seconds = 15
heartbeat_stale_minutes = 5
ingest_interval_seconds = 180
notify_interval_seconds = 120
tick_seconds = 20
watchdog_minutes = 30
comment_budget = 10

[identity]
bot_login = "thepastaclaw"
slack_target = "user:UCW1VE04T"
slack_account = "default"
agent_session_key = "agent:main:main"
trusted_reviewers = ["PastaPastaPasta", "QuantumExplorer", "shumkov", "lklimek", "dustinface", "thephez", "knst", "pauldelucia", "UdjinM6"]

[retention]
artifact_days = 14
worktree_budget_gb = 60
"""


def _lane(section: dict[str, Any], key: str) -> LaneModel:
    node = section[key]
    return LaneModel(
        agent=str(node["agent"]),
        model=str(node["model"]),
        effort=str(node.get("reasoning", "high")),
    )


def load_skills_config(
    skills_dir: Path,
) -> tuple[tuple[RepoConfig, ...], tuple[Specialist, ...], ModelPolicy, dict[str, Any]]:
    raw = json.loads((skills_dir / "config.json").read_text())
    repos = tuple(
        RepoConfig(
            repo=f"{r['owner']}/{r['name']}",
            skill_path=str(r["skill_path"]),
            enabled=bool(r.get("enabled", False)),
            disabled_reason=r.get("disabled_reason"),
        )
        for r in raw["repos"]
    )
    specialists = tuple(
        Specialist(
            id=str(s["id"]),
            prompt=str(s["prompt"]),
            repos=tuple(s.get("repos", [])),
            description=str(s.get("description", "")),
            always_run=bool(s.get("always_run", False)),
            trigger_hint=s.get("trigger_hint"),
        )
        for s in raw.get("specialists", [])
    )
    pol = raw["review_model_policy"]
    settings = raw.get("settings", {})
    p1 = _lane(pol["phase1"], "reviewer")
    p2 = _lane(pol["phase2"], "reviewer")
    triage_node = pol.get("triage") or {}
    policy = ModelPolicy(
        name=str(pol["name"]),
        fingerprint=str(pol["fingerprint"]),
        phase1_reviewer=p1,
        phase1_verifier=_lane(pol["phase1"], "verifier"),
        phase2_reviewer=p2,
        phase2_verifier=_lane(pol["phase2"], "verifier"),
        repair_model=str(settings.get("repair_model", "gpt-5.6-luna")),
        selector_model=str(settings.get("selector_model", "gpt-5.6-terra")),
        phase2_enabled=bool(pol.get("phase2_enabled", True)),
        triage=_lane(pol, "triage") if triage_node else None,
        tiers=_tiers(triage_node, p1.effort, p2.effort),
        fallback_tier=str(triage_node.get("fallback_tier", "normal")).lower(),
    )
    if policy.fallback_tier not in policy.tiers:
        raise ValueError(f"triage.fallback_tier {policy.fallback_tier!r} is not a configured tier")
    return repos, specialists, policy, settings


def _tiers(node: dict[str, Any], p1_default: str, p2_default: str) -> dict[str, TierEffort]:
    """Tier -> effort table from `review_model_policy.triage.tiers`.
    Without a triage block there is a single `normal` tier at the policy's default efforts."""
    if not node:
        return {"normal": TierEffort(phase1=p1_default, phase2=p2_default)}
    out: dict[str, TierEffort] = {}
    for tier, spec in node["tiers"].items():
        p1 = str(spec.get("phase1", p1_default))
        raw2 = spec.get("phase2", p2_default)
        p2 = None if raw2 is None else str(raw2)
        for level in (p1, p2):
            if level is not None and level not in EFFORT_LEVELS:
                raise ValueError(f"tier {tier!r}: effort {level!r} not in {EFFORT_LEVELS}")
        out[str(tier).lower()] = TierEffort(phase1=p1, phase2=p2)
    return out


def load(path: Path | None = None, *, skills_override: Path | None = None) -> Config:
    path = path or DEFAULT_CONFIG_PATH
    text = path.read_text() if path.exists() else DEFAULT_TOML
    t = tomllib.loads(text)
    p, s, i, r = t["paths"], t["scheduling"], t["identity"], t["retention"]
    skills_dir = (skills_override or Path(p["skills"])).expanduser()
    repos, specialists, policy, settings = load_skills_config(skills_dir)
    return Config(
        db_path=Path(p["db"]).expanduser(),
        skills_dir=skills_dir,
        work_dir=Path(p["work"]).expanduser(),
        claude_bin=str(Path(p["claude"]).expanduser()),
        gh_bin=str(p["gh"]),
        openclaw_bin=str(p["openclaw"]),
        max_concurrent=int(s.get("max_concurrent", settings.get("max_concurrent_reviews", 2))),
        priority_overflow=int(
            s.get("priority_overflow", settings.get("priority_review_overflow_slots", 1))
        ),
        debounce_minutes=int(s.get("debounce_minutes", settings.get("debounce_minutes", 30))),
        max_attempts=int(s["max_attempts"]),
        retry_backoff_minutes=tuple(int(x) for x in s["retry_backoff_minutes"]),
        lane_timeout_minutes=int(
            s.get("lane_timeout_minutes", settings.get("agent_timeout_minutes", 180))
        ),
        lane_budget_usd=float(s.get("lane_budget_usd", 50.0)),
        run_timeout_minutes=int(s["run_timeout_minutes"]),
        heartbeat_seconds=int(s["heartbeat_seconds"]),
        heartbeat_stale_minutes=int(s["heartbeat_stale_minutes"]),
        ingest_interval_seconds=int(s["ingest_interval_seconds"]),
        notify_interval_seconds=int(s["notify_interval_seconds"]),
        tick_seconds=int(s["tick_seconds"]),
        watchdog_minutes=int(s["watchdog_minutes"]),
        comment_budget=int(s.get("comment_budget", settings.get("comment_budget", 10))),
        bot_login=str(i["bot_login"]),
        slack_target=i.get("slack_target"),
        slack_account=str(i.get("slack_account", "default")),
        agent_session_key=str(i.get("agent_session_key", "agent:main:main")),
        trusted_reviewers=tuple(i.get("trusted_reviewers", [])),
        artifact_retention_days=int(r["artifact_days"]),
        worktree_budget_gb=int(r["worktree_budget_gb"]),
        repos=repos,
        specialists=specialists,
        policy=policy,
        extra=settings,
    )
