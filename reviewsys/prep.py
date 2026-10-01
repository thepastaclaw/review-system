"""The prep lane: one cheap lane answers both questions a review asks before it starts, which
effort tier the PR is (triage.py) and which discretionary specialists review it (select.py).

The two used to be two lanes in a row (selector on its own model, then triage), each with a
pool slot and a cold start of its own, although triage only reads the file list and neither
reads the other's answer: measured since v0.23, selection 43 s on average (126 max), triage
14 s (115 max). One lane on the triage model asks both in two separated sections of one
prompt, with triage's tier guide and decision rules and the selector's specialist list and
rule word for word, and replies with one JSON object.

The halves are validated, retried and fall back independently: a bad or missing tier falls
back to `fallback_tier` (as triage.py), a bad or missing selection to the trigger heuristics
(as select.py), and one broken half never discards the other. Each half comes back as the
same `Triage` / `Selection` the single-purpose lanes return, so nothing downstream can tell
which lane produced it except the artifacts under `prep/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config, LaneModel, Specialist
from .contract import parse_json_object
from .lane import LaneSpec, run_claude_lane
from .select import PICK_RULE, Selection, heuristic_selection, llm_selection, specialist_listing
from .select import split as split_specialists
from .triage import TIER_RULES, Triage, parse_tier, pr_block, tier_listing

TIMEOUT_SECONDS = 180  # triage's; the selector had 120


@dataclass(slots=True)
class Prep:
    selection: Selection
    triage: Triage
    # the last lane infrastructure failure (exit status + first stderr line, never model
    # output): what the worker checks for a dry primary pool
    infra_error: str | None = None
    attempts: int = 0  # lane attempts made, so a stand-in retry numbers its artifacts on


def _prompt(
    repo: str,
    base_ref: str,
    title: str,
    body: str,
    files: list[dict[str, Any]],
    tiers: list[str],
    specialists: list[Specialist],
) -> str:
    return (
        f"You prepare an automated review of a pull request in {repo} (base branch `{base_ref}`). "
        "Answer two independent questions about it in one reply: part 1 rates the change so the "
        "review can decide how much reasoning effort to spend on it, part 2 picks its specialist "
        "reviewers. Neither answer depends on the other.\n\n"
        + pr_block(title, body, files)
        + "## Part 1: effort tier\n\n"
        "Rate the complexity and criticality of this pull request.\n\n"
        + tier_listing(tiers)
        + TIER_RULES.rstrip()
        + "\n\n## Part 2: specialist reviewers\n\n"
        "Select specialist code reviewers for this pull request.\n\n"
        + specialist_listing(specialists)
        + PICK_RULE.rstrip()
        + "\n\n"
        'Reply with exactly one JSON object: {"tier": "<one of '
        + ", ".join(tiers)
        + '>", "tier_reasoning": "one sentence", "selected": ["id", ...], '
        '"selection_reasoning": "one sentence"}. No prose, no fences.'
    )


def _picked(obj: dict[str, Any], discretionary: list[Specialist]) -> list[str]:
    """The discretionary ids `obj` selects (unknown ids are dropped, as the selector does);
    ValueError when `selected` is missing or not a list, so the half falls back."""
    sel = obj.get("selected")
    if not isinstance(sel, list):
        raise ValueError(f"`selected` is {type(sel).__name__}, not a list")
    ids = {s.id for s in discretionary}
    return [str(x) for x in sel if str(x) in ids]


def prep(
    cfg: Config,
    *,
    repo: str,
    base_ref: str,
    title: str,
    body: str,
    files: list[dict[str, Any]],
    run_dir: Path,
    worktree: Path,
    runner: Any = run_claude_lane,
    lane: LaneModel | None = None,
    prior: Prep | None = None,
) -> Prep:
    """Both answers from one lane, two attempts. `lane` overrides the policy's triage lane
    (degraded-mode stand-in). `prior`: an earlier result whose good halves are kept, so a
    retry on a stand-in only asks again for the half that fell back.

    The caller makes sure there is a choice to make: triage is configured and the repo has
    discretionary specialists (without them `select.select` answers from config alone)."""
    pol = cfg.policy
    assert pol.triage is not None
    lane = lane or pol.triage
    available = cfg.specialists_for(repo)
    _, discretionary = split_specialists(available)
    tiers = list(pol.tiers)
    names = [str(f.get("filename")) for f in files]
    spec = LaneSpec(
        role="prep",
        agent=lane.agent,
        model=lane.model,
        effort=lane.effort,
        prompt=_prompt(repo, base_ref, title, body, files, tiers, discretionary),
        cwd=worktree,
        add_dir=run_dir,
        timeout_seconds=TIMEOUT_SECONDS,
        claude_bin=cfg.claude_bin,
    )
    tri = prior.triage if prior and not prior.triage.error else None
    sel = prior.selection if prior and not prior.selection.error else None
    tier_error = sel_error = infra = None
    made = prior.attempts if prior else 0
    for _ in range(2):
        if tri and sel:
            break
        made += 1
        try:
            res = runner(spec, run_dir / "prep" / f"attempt-{made}", worktree)
            infra = None  # only the latest attempt says whether the pool answers
            if not res.ok:
                said = res.first_stderr_line  # infrastructure error, never model output
                infra = f"exit {res.exit_code} timed_out={res.timed_out}" + (
                    f": {said[:200]}" if said else ""
                )
                tier_error, sel_error = infra, f"{spec.model}: {infra}"
                continue
            obj = parse_json_object(res.result_text)
        except Exception as exc:
            tier_error, sel_error = str(exc)[:400], f"{spec.model}: {exc}"
            continue
        if tri is None:
            try:
                tri = Triage(
                    tier=parse_tier(obj, pol.tiers),
                    method=f"llm:{spec.model}",
                    reasoning=str(obj.get("tier_reasoning") or "")[:400],
                )
            except ValueError as bad:
                tier_error = str(bad)
        if sel is None:
            try:
                sel = llm_selection(
                    available,
                    _picked(obj, discretionary),
                    spec.model,
                    str(obj.get("selection_reasoning") or ""),
                )
            except ValueError as bad:
                sel_error = f"{spec.model}: {bad}"
    return Prep(
        selection=sel or heuristic_selection(available, title, body, names, sel_error),
        triage=tri or Triage(tier=pol.fallback_tier, method="fallback", error=tier_error),
        infra_error=infra,
        attempts=made,
    )
