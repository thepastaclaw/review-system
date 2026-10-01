"""Configuration: a TOML file for runtime knobs plus the skills repo's config.json.

The skills repo (thepastaclaw/skills) remains the source of truth for repos,
specialists, model policy and prompts. reviewsys reads it read-only.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(
    os.environ.get("REVIEWSYS_CONFIG", "~/.reviewsys/config.toml")
).expanduser()


QUOTA_PROVIDERS = ("antigravity", "zai")


@dataclass(frozen=True, slots=True)
class QuotaSource:
    """Which subscription's remaining quota gates a candidate model (read by `quota.py`)."""

    provider: str  # one of QUOTA_PROVIDERS
    group: str | None = None  # antigravity quota group, e.g. "Gemini Models"
    account: str | None = None  # antigravity: credential email; zai: proxy provider name


@dataclass(frozen=True, slots=True)
class LaneModel:
    agent: str
    model: str
    effort: str = "high"  # for a Phase-1 ladder rung: the cap the tier effort is clamped to
    quota: QuotaSource | None = None  # None: never gated (pay-per-token)
    # for a Phase-1 ladder rung: only eligible when the effort it would run at (the tier's
    # effort clamped to `effort`) is at most this. Differs from `effort` (which clamps): a rung
    # above its ceiling is passed over entirely, for models that are fine at moderate effort
    # but far too slow at the top of the scale. Never allowed on the last rung, which must
    # always be able to run Phase 1.
    use_up_to: str | None = None
    # set on a lane that runs on a degraded-mode stand-in: the primary model it replaces
    substitute_for: str | None = None


EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


def min_effort(a: str, b: str) -> str:
    """The lower of two effort levels."""
    return min(a, b, key=EFFORT_LEVELS.index)


@dataclass(frozen=True, slots=True)
class TierEffort:
    """Reviewer effort for one triage tier. `phase2=None` means Phase 2 is skipped.

    `single_stage`: no blocker gate between the phases. The Phase-1 reviewers (on their
    ladder model) run beside the Phase-2 reviewers and the final verifier weighs both, for
    changes whose authors are expected to have reviewed them closely already, where the
    gate only adds latency. Needs a Phase 2 (`phase2` set)."""

    phase1: str
    phase2: str | None
    single_stage: bool = False


# severity -> points; nitpicks never count toward the gate
DEFAULT_GATE_WEIGHTS = {"blocking": 3, "suggestion": 1, "nitpick": 0}


@dataclass(frozen=True, slots=True)
class GatePolicy:
    """When Phase 1 holds Phase 2 back: the verified Phase-1 findings are scored by severity
    and Phase 2 is deferred only when the total is above `block_above`. Without a gate
    policy, any verified blocker defers Phase 2 (the pre-v0.21 rule)."""

    block_above: int
    weights: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_GATE_WEIGHTS))

    def points(self, severities: list[str]) -> int:
        return sum(self.weights.get(s, 0) for s in severities)


@dataclass(frozen=True, slots=True)
class Substitute:
    """A stand-in model for one primary model while that primary is unavailable."""

    model: str
    effort_cap: str | None = None  # the stand-in's top effort (e.g. no `max` on the tier)


@dataclass(frozen=True, slots=True)
class DegradedPolicy:
    """What the pipeline does when the primary (OpenAI) models are out of quota or down.

    `sentinel` is probed before every run; while it answers 429 (or an operator forces the
    mode on) every lane whose model is in `substitutes` runs on its stand-in, Phase 1 is
    capped at `phase1_effort_cap` so the included-quota rungs stay eligible, and everything
    published says so. A lane that hits a quota failure on a primary model mid-run switches
    the rest of the run the same way. The backlog rule is unchanged by the mode
    (`backlog_skip_phase1`): a deep queue still goes straight to Phase 2, because the slow
    Phase-1 rungs are exactly what a backlog cannot afford. Set it false to trade throughput
    for the cross-model check."""

    sentinel: str
    substitutes: dict[str, Substitute]
    phase1_effort_cap: str | None = None
    backlog_skip_phase1: bool = True  # the backlog rule still applies: a deep queue skips Phase 1
    label: str = "degraded"

    def resolve(self, lm: LaneModel) -> LaneModel:
        """`lm` on its stand-in (effort capped), or unchanged when it has none."""
        sub = self.substitutes.get(lm.model)
        if sub is None or lm.substitute_for is not None:
            return lm
        effort = min_effort(lm.effort, sub.effort_cap) if sub.effort_cap else lm.effort
        return dataclasses.replace(lm, model=sub.model, effort=effort, substitute_for=lm.model)

    def phase1_effort(self, effort: str) -> str:
        return min_effort(effort, self.phase1_effort_cap) if self.phase1_effort_cap else effort


@dataclass(frozen=True, slots=True)
class ComparisonPolicy:
    """Run a second model beside the primary one on a sample of runs, to compare them.

    On a selected run every Phase-2 reviewer lane (general and each specialist; not the fresh
    final pass) also runs on `model`. The verifier sees both sets under neutral labels, so the
    findings it keeps show what each model contributes (`reviewsys compare`). A comparison
    lane never fails or holds up the run and never flips it into degraded mode (see
    `worker._reviewer_lanes`). Selection is a stable hash of the head, so a retried run makes
    the same choice. Never in degraded mode (both would run on the same stand-in) or on
    audits."""

    model: str
    tiers: tuple[str, ...]
    fraction: float  # of the runs in `tiers`, 0..1

    def selects(self, repo: str, number: int, sha: str, tier: str) -> bool:
        if tier not in self.tiers:
            return False
        digest = hashlib.sha256(f"{repo}#{number}@{sha}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**64 < self.fraction


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
    # ordered Phase-1 ladder: the first candidate with quota left runs; `phase1_reviewer`
    # stays the declared default (its effort seeds the tier table). Never empty.
    phase1_candidates: tuple[LaneModel, ...] = ()
    quota_reserve: float = 0.15  # fraction of every quota window a candidate must have left
    # complexity triage: None = not configured, every PR is `fallback_tier`
    triage: LaneModel | None = None
    tiers: dict[str, TierEffort] = field(default_factory=dict)
    fallback_tier: str = "normal"
    # answers human replies on an already-reviewed commit (see converse.py); None = the
    # Phase-2 reviewer model at high effort under the agent name `conversation`
    conversation: LaneModel | None = None
    # stand-ins while the primary models are unavailable; None = no degraded mode, a run
    # whose primary model is down fails as before
    degraded: DegradedPolicy | None = None
    # a second model run beside the primary on a sample of runs; None = never
    comparison: ComparisonPolicy | None = None
    # specialists Phase 1 runs beside `general` (when the selector picked them); None = every
    # selected specialist. Phase 2 always runs every selected specialist.
    phase1_specialists: tuple[str, ...] | None = None
    # how many verified Phase-1 findings defer Phase 2; None = any blocker does
    phase1_gate: GatePolicy | None = None

    @property
    def has_phase1_ladder(self) -> bool:
        """More than one rung, so which one ran is worth recording and disclosing."""
        return len(self.phase1_candidates) > 1

    @property
    def conversation_lane(self) -> LaneModel:
        return self.conversation or LaneModel(
            agent="conversation", model=self.phase2_reviewer.model, effort="high"
        )

    def tier_effort(self, tier: str) -> TierEffort:
        default = TierEffort(phase1=self.phase1_reviewer.effort, phase2=self.phase2_reviewer.effort)
        return self.tiers.get(tier) or self.tiers.get(self.fallback_tier) or default


@dataclass(frozen=True, slots=True)
class RepoConfig:
    repo: str
    skill_path: str
    enabled: bool
    disabled_reason: str | None = None
    # False: reviews of this repo skip Phase 1 and go straight to Phase 2 (measured to add
    # next to nothing there); disclosed in the review like the backlog rule
    phase1: bool = True


@dataclass(frozen=True, slots=True)
class Specialist:
    id: str
    prompt: str
    repos: tuple[str, ...]
    description: str
    always_run: bool = False
    trigger_hint: str | None = None


@dataclass(frozen=True, slots=True)
class AuditConfig:
    """The audit queue: post-merge reviews of PRs merged without a clean review (audit.py).

    Audit runs only ever use capacity live review leaves idle: none starts while a live head
    is eligible, and they never count against the live slots. `max_concurrent` is the steady
    state (one at a time); raise it on the box to drain a backfill."""

    enabled: bool = True
    max_concurrent: int = 1
    # private repository the per-PR audit reports are committed to (never a PR comment for
    # backfilled audits); "" keeps reports on disk under work/audit only
    report_repo: str = ""
    report_branch: str = "main"
    # post a comment + open one issue per PR with blockers for merges the sweep caught live
    post_live: bool = False
    # repos never audited (also: never auto-enrolled on merge)
    exclude_repos: tuple[str, ...] = ()


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
    max_concurrent: int  # per usable OpenAI account (see slots.py)
    priority_overflow: int  # per usable OpenAI account
    account_scale_max: int  # most accounts the slots scale over; 1 = static slots
    # OpenAI account (email) -> fraction of its quota nobody may spend; unlisted = 0
    account_reserves: dict[str, float]
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
    queue_comment_interval_seconds: int
    tick_seconds: int
    watchdog_minutes: int
    comment_budget: int
    backlog_skip_phase1_above: int  # queued heads above this -> runs skip Phase 1 (0 disables)
    # reviewer lanes one phase of one run runs at once (1 = sequential, the pre-v0.19 flow;
    # 0 = every lane of the phase at once). Model pools (lanepool.py) still bound the machine.
    phase_parallelism: int
    # machine-wide lane slots per non-gpt model family (see lanepool.py); unlisted = ungated
    lane_pools: dict[str, int]
    # paths a lane may not execute anything under: reviews are static (see lane.exec_deny_profile)
    lane_deny_exec: tuple[str, ...]
    # identity / alerting
    bot_login: str
    slack_target: str | None
    slack_account: str
    # loud, shared alert for outages someone must fix (degraded mode): a channel plus the
    # Slack user ids to @-mention; None = page the `slack_target` DM only
    page_target: str | None
    page_mentions: tuple[str, ...]
    agent_session_key: str
    trusted_reviewers: tuple[str, ...]
    # retention; 0 = run artifacts (lane outputs) are never deleted
    artifact_retention_days: int
    worktree_budget_gb: int
    # from skills config
    repos: tuple[RepoConfig, ...]
    specialists: tuple[Specialist, ...]
    policy: ModelPolicy
    extra: dict[str, Any] = field(default_factory=dict)
    audit: AuditConfig = field(default_factory=AuditConfig)

    @property
    def enabled_repos(self) -> tuple[str, ...]:
        configured = tuple(r.repo for r in self.repos if r.enabled)
        # Small automation-owned forks (for example backportsys' publication fork) can be
        # reviewed without a skills entry. Keep these in runtime TOML so adding one does not
        # require mutating the shared skills repository.
        extra = tuple(str(x) for x in self.extra.get("additional_repos", []) if str(x))
        return tuple(dict.fromkeys((*configured, *extra)))

    def repo(self, repo: str) -> RepoConfig | None:
        for r in self.repos:
            if r.repo == repo:
                return r
        return None

    def specialists_for(self, repo: str) -> tuple[Specialist, ...]:
        if self.repo(repo) is None:
            # ad hoc review of a repo without a skills entry: offer every discretionary
            # specialist to the selector; always-run ones are repo-specific by design
            return tuple(s for s in self.specialists if not s.always_run)
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


# #claw and pasta + latte: the people who can add or re-enable an OpenAI account. Code
# defaults as well as TOML ones, so a box config.toml predating them still pages.
DEFAULT_PAGE_TARGET = "channel:C0AEQ5D7SJ3"
DEFAULT_PAGE_MENTIONS = ("UCW1VE04T", "U02CNG35EGG")

# box config.toml files predating parallel lanes get the same caps as a fresh one
DEFAULT_LANE_POOLS = {"muse": 8, "glm": 6, "gemini": 6}

# Toolchains a review lane may not run (2026-10-01: lanes building dashd, the platform
# workspace and the iOS FFI side by side filled 60 GB of swap and stalled every review).
# Reviews read the code and the PR's CI results instead. git is /usr/bin/git -> xcrun ->
# Xcode's usr/bin, so neither xcrun nor Developer/usr/bin as a whole can be listed.
DEFAULT_LANE_DENY_EXEC = (
    "~/.rustup",
    "~/.cargo/bin",
    "/usr/bin/make",
    "/usr/bin/gnumake",
    "/usr/bin/xcodebuild",
    "/usr/bin/clang",
    "/usr/bin/clang++",
    "/usr/bin/cc",
    "/usr/bin/c++",
    "/usr/bin/gcc",
    "/usr/bin/g++",
    "/usr/bin/ld",
    "/usr/bin/swift",
    "/usr/bin/swiftc",
    "/usr/bin/java",
    "/Applications/Xcode.app/Contents/Developer/Toolchains",
    "/Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild",
    "/Applications/Xcode.app/Contents/Developer/usr/bin/make",
    "/Library/Developer/CommandLineTools/usr/bin/make",
    "/Library/Developer/CommandLineTools/usr/bin/clang",
    "/Library/Developer/CommandLineTools/usr/bin/clang++",
    "/Library/Developer/CommandLineTools/usr/bin/ld",
    "/Library/Java/JavaVirtualMachines",
    "/opt/homebrew/Cellar/cmake",
    "/opt/homebrew/Cellar/ninja",
    "/opt/homebrew/Cellar/go",
    "/opt/homebrew/Cellar/gcc",
    "/opt/homebrew/Cellar/llvm",
    "/opt/homebrew/Cellar/openjdk",
    "/opt/homebrew/Cellar/gradle",
    "/opt/homebrew/Cellar/sccache",
    "/opt/homebrew/lib/node_modules/npm",
    "~/.npm-global/lib/node_modules/npm",
    "~/.npm-global/lib/node_modules/yarn",
    "~/.npm-global/lib/node_modules/pnpm",
    "~/.local/bin/sccache",
)

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
# Additional automation-owned forks to poll and review (without a skills entry).
additional_repos = []
# slots per usable OpenAI account; the effective capacity is this times the number of
# accounts with quota left (at most account_scale_max), never less than one unit
max_concurrent = 2
priority_overflow = 1
account_scale_max = 3
# fraction of an account's quota to leave untouched, keyed by account email; at the floor
# the account is disabled in the proxy (for every client) until its window resets.
# Unlisted accounts are spent to the end. Set on the box only: this repository is public.
account_reserves = {}
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
# refresh queue-position comments and honour the priority checkbox this often
queue_comment_interval_seconds = 180
tick_seconds = 20
watchdog_minutes = 30
comment_budget = 10
# when more heads than this are queued, new runs skip the Phase-1 reviewers and go
# straight to Phase 2; 0 disables. Disclosed in the review and the gate comment.
backlog_skip_phase1_above = 10
# the general reviewer and the specialists of one phase run side by side, at most this
# many at once per run (1 = one after another, 0 = all of them); the model pools below
# still bound the machine
phase_parallelism = 0
# machine-wide cap on lanes in flight per model family, across every run. gpt lanes are
# capped by the review slot ceiling instead (the per-account stream budget); a family
# not listed here is not capped.
lane_pools = { muse = 8, glm = 6, gemini = 6 }

[lanes]
# lanes may not execute anything under these paths (compilers, build systems, test
# runners): reviews are static and read CI results. Unset = the built-in list
# (config.DEFAULT_LANE_DENY_EXEC); [] = no sandbox.
# deny_exec = ["~/.rustup", "~/.cargo/bin", "/usr/bin/make", "/usr/bin/xcodebuild"]

[identity]
bot_login = "thepastaclaw"
slack_target = "user:UCW1VE04T"
slack_account = "default"
# degraded mode pages this channel (#claw) and @-mentions these users (pasta, latte)
page_target = "channel:C0AEQ5D7SJ3"
page_mentions = ["UCW1VE04T", "U02CNG35EGG"]
agent_session_key = "agent:main:main"
trusted_reviewers = ["PastaPastaPasta", "QuantumExplorer", "shumkov", "lklimek", "dustinface", "thephez", "knst", "pauldelucia", "UdjinM6"]

[retention]
# run artifacts (every lane's prompt, output and meta); 0 = keep forever. They are small
# (~1 GB per two weeks) and the only per-lane record older reviews can be analysed from.
artifact_days = 0
worktree_budget_gb = 60

[audit]
# post-merge reviews of PRs merged without a clean review; they only use slots live review
# leaves idle. Raise max_concurrent to drain a backfill.
enabled = true
max_concurrent = 1
# private repo the per-PR audit reports are committed to ("" = on disk only)
report_repo = ""
report_branch = "main"
# for merges seen live: post a "Post-merge review" comment and one issue per PR with blockers
post_live = false
exclude_repos = []
"""


def _lane(section: dict[str, Any], key: str) -> LaneModel:
    node = section[key]
    return LaneModel(
        agent=str(node["agent"]),
        model=str(node["model"]),
        effort=str(node.get("reasoning", "high")),
    )


def _candidate(node: dict[str, Any], base: LaneModel) -> LaneModel:
    model = str(node.get("model") or "")
    if not model:
        raise ValueError("phase1 candidate without a model")
    q = node.get("quota")
    quota = None
    if q:
        provider = str(q.get("provider") or "")
        if provider not in QUOTA_PROVIDERS:
            raise ValueError(
                f"phase1 candidate {model!r}: quota provider {provider!r} not in {QUOTA_PROVIDERS}"
            )
        quota = QuotaSource(provider=provider, group=q.get("group"), account=q.get("account"))
    effort = str(node.get("reasoning", base.effort))
    if effort not in EFFORT_LEVELS:
        raise ValueError(f"phase1 candidate {model!r}: effort {effort!r} not in {EFFORT_LEVELS}")
    use_up_to = node.get("use_up_to")
    if use_up_to is not None and str(use_up_to) not in EFFORT_LEVELS:
        raise ValueError(
            f"phase1 candidate {model!r}: use_up_to {use_up_to!r} not in {EFFORT_LEVELS}"
        )
    return LaneModel(
        agent=str(node.get("agent", base.agent)),
        model=model,
        effort=effort,
        quota=quota,
        use_up_to=str(use_up_to) if use_up_to is not None else None,
    )


def _candidates(node: dict[str, Any], base: LaneModel) -> tuple[LaneModel, ...]:
    """The Phase-1 ladder; without a `candidates` list it is the declared reviewer alone."""
    out = tuple(_candidate(c, base) for c in node.get("candidates") or [])
    if not out:
        return (base,)
    for c in out[:-1]:
        if c.quota is None:
            raise ValueError(
                f"phase1 candidate {c.model!r} has no quota source but is not the last rung"
            )
    if out[-1].use_up_to is not None:
        raise ValueError(
            f"phase1 candidate {out[-1].model!r} is the last rung and cannot have use_up_to: "
            "Phase 1 must always be able to run"
        )
    return out


def _degraded(node: dict[str, Any] | None) -> DegradedPolicy | None:
    if not node:
        return None
    sentinel = str(node.get("sentinel") or "")
    if not sentinel:
        raise ValueError("degraded policy needs a `sentinel` model to probe")
    subs: dict[str, Substitute] = {}
    for primary, spec in (node.get("substitutes") or {}).items():
        model = str((spec or {}).get("model") or "") if isinstance(spec, dict) else str(spec or "")
        if not model:
            raise ValueError(f"degraded substitute for {primary!r} has no model")
        cap = spec.get("effort_cap") if isinstance(spec, dict) else None
        if cap is not None and str(cap) not in EFFORT_LEVELS:
            raise ValueError(
                f"degraded substitute for {primary!r}: effort_cap {cap!r} not in {EFFORT_LEVELS}"
            )
        subs[str(primary)] = Substitute(
            model=model, effort_cap=str(cap) if cap is not None else None
        )
    if not subs:
        raise ValueError("degraded policy needs at least one substitute")
    for primary, sub in subs.items():
        if sub.model in subs:
            raise ValueError(
                f"degraded substitute {sub.model!r} (for {primary!r}) is itself substituted; stand-ins must be terminal"
            )
    if sentinel not in subs:
        raise ValueError(f"degraded sentinel {sentinel!r} has no substitute")
    cap1 = node.get("phase1_effort_cap")
    if cap1 is not None and str(cap1) not in EFFORT_LEVELS:
        raise ValueError(f"degraded phase1_effort_cap {cap1!r} not in {EFFORT_LEVELS}")
    return DegradedPolicy(
        sentinel=sentinel,
        substitutes=subs,
        phase1_effort_cap=str(cap1) if cap1 is not None else None,
        backlog_skip_phase1=bool(node.get("backlog_skip_phase1", True)),
        label=str(node.get("label") or "degraded"),
    )


def _comparison(
    node: dict[str, Any] | None, tiers: dict[str, TierEffort], primary: str
) -> ComparisonPolicy | None:
    if not node:
        return None
    model = str(node.get("model") or "")
    if not model:
        raise ValueError("comparison policy needs a `model`")
    if model == primary:
        raise ValueError(
            f"comparison model {model!r} is the Phase-2 model it would compare against"
        )
    fraction = float(node.get("fraction", 0))
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"comparison.fraction {fraction!r} must be in [0, 1]")
    names = tuple(str(t).lower() for t in node.get("tiers") or ())
    unknown = [t for t in names if t not in tiers]
    if unknown:
        raise ValueError(f"comparison.tiers {unknown!r} are not configured tiers")
    return ComparisonPolicy(model=model, tiers=names, fraction=fraction)


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
            phase1=bool(r.get("phase1", True)),
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
    p1_node = pol["phase1"]
    p1 = _lane(p1_node, "reviewer")
    reserve = float(p1_node.get("quota_reserve", 0.15))
    if not 0.0 <= reserve < 1.0:
        raise ValueError(f"phase1.quota_reserve {reserve!r} must be a fraction in [0, 1)")
    p2 = _lane(pol["phase2"], "reviewer")
    triage_node = pol.get("triage") or {}
    tiers = _tiers(triage_node, p1.effort, p2.effort)
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
        phase1_candidates=_candidates(p1_node, p1),
        quota_reserve=reserve,
        triage=_lane(pol, "triage") if triage_node else None,
        tiers=tiers,
        fallback_tier=str(triage_node.get("fallback_tier", "normal")).lower(),
        conversation=_lane(pol, "conversation") if pol.get("conversation") else None,
        degraded=_degraded(pol.get("degraded")),
        comparison=_comparison(pol.get("comparison"), tiers, p2.model),
        phase1_specialists=_phase1_specialists(p1_node, specialists),
        phase1_gate=_gate(p1_node.get("gate")),
    )
    if policy.fallback_tier not in policy.tiers:
        raise ValueError(f"triage.fallback_tier {policy.fallback_tier!r} is not a configured tier")
    return repos, specialists, policy, settings


def _phase1_specialists(
    node: dict[str, Any], specialists: tuple[Specialist, ...]
) -> tuple[str, ...] | None:
    raw = node.get("specialists")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("phase1.specialists must be a list of specialist ids")
    ids = tuple(str(x) for x in raw)
    unknown = sorted(set(ids) - {s.id for s in specialists})
    if unknown:
        raise ValueError(f"phase1.specialists {unknown!r} are not configured specialists")
    return ids


def _gate(node: dict[str, Any] | None) -> GatePolicy | None:
    if node is None:
        return None
    if not isinstance(node, dict) or "block_above" not in node:
        raise ValueError("phase1.gate needs `block_above`")
    block_above = int(node["block_above"])
    if block_above < 0:
        raise ValueError(f"phase1.gate.block_above {block_above!r} must be >= 0")
    weights = dict(DEFAULT_GATE_WEIGHTS)
    for sev, pts in (node.get("weights") or {}).items():
        if sev not in DEFAULT_GATE_WEIGHTS:
            raise ValueError(f"phase1.gate.weights: unknown severity {sev!r}")
        if int(pts) < 0:
            raise ValueError(f"phase1.gate.weights[{sev!r}] must be >= 0")
        weights[sev] = int(pts)
    return GatePolicy(block_above=block_above, weights=weights)


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
        single = spec.get("single_stage", False)
        if not isinstance(single, bool):
            raise ValueError(f"tier {tier!r}: single_stage must be true or false")
        if single and p2 is None:
            raise ValueError(f"tier {tier!r}: single_stage needs a phase2 effort")
        out[str(tier).lower()] = TierEffort(phase1=p1, phase2=p2, single_stage=single)
    return out


def _mentions(value: Any) -> tuple[str, ...]:
    """Slack user ids; a single id written as a string is one mention, not one per letter."""
    return (str(value),) if isinstance(value, str) else tuple(str(u) for u in value or ())


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
        # clamped, not rejected: a bad edit must not put the daemon into a restart loop
        max_concurrent=max(
            1, int(s.get("max_concurrent", settings.get("max_concurrent_reviews", 2)))
        ),
        priority_overflow=max(
            0, int(s.get("priority_overflow", settings.get("priority_review_overflow_slots", 1)))
        ),
        account_scale_max=max(1, int(s.get("account_scale_max", 3))),
        account_reserves={
            str(k).lower(): min(1.0, max(0.0, float(v)))
            for k, v in (s.get("account_reserves") or {}).items()
        },
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
        queue_comment_interval_seconds=int(s.get("queue_comment_interval_seconds", 180)),
        tick_seconds=int(s["tick_seconds"]),
        watchdog_minutes=int(s["watchdog_minutes"]),
        comment_budget=int(s.get("comment_budget", settings.get("comment_budget", 10))),
        backlog_skip_phase1_above=int(s.get("backlog_skip_phase1_above", 10)),
        phase_parallelism=max(0, int(s.get("phase_parallelism", 0))),
        lane_pools={
            str(k): max(1, int(v))
            for k, v in (s.get("lane_pools", DEFAULT_LANE_POOLS) or {}).items()
            if str(k) != "gpt"  # gpt is always the review slot ceiling
        },
        lane_deny_exec=tuple(
            str(x) for x in (t.get("lanes") or {}).get("deny_exec", DEFAULT_LANE_DENY_EXEC)
        ),
        bot_login=str(i["bot_login"]),
        slack_target=i.get("slack_target"),
        slack_account=str(i.get("slack_account", "default")),
        page_target=i.get("page_target", DEFAULT_PAGE_TARGET) or None,
        page_mentions=_mentions(i.get("page_mentions", DEFAULT_PAGE_MENTIONS)),
        agent_session_key=str(i.get("agent_session_key", "agent:main:main")),
        trusted_reviewers=tuple(i.get("trusted_reviewers", [])),
        artifact_retention_days=max(0, int(r["artifact_days"])),
        worktree_budget_gb=int(r["worktree_budget_gb"]),
        repos=repos,
        specialists=specialists,
        policy=policy,
        extra={**settings, "additional_repos": s.get("additional_repos", [])},
        audit=_audit(t.get("audit") or {}),
    )


def _audit(a: dict[str, Any]) -> AuditConfig:
    return AuditConfig(
        enabled=bool(a.get("enabled", True)),
        max_concurrent=max(0, int(a.get("max_concurrent", 1))),
        report_repo=str(a.get("report_repo") or ""),
        report_branch=str(a.get("report_branch") or "main"),
        post_live=bool(a.get("post_live", False)),
        exclude_repos=tuple(str(x) for x in a.get("exclude_repos") or ()),
    )
