"""Complexity/criticality triage: one cheap lane rates the PR so reviewer effort can scale.

Like `select.py`, failure never silently degrades: the result carries `method` and
`error`, the worker records an event, and the review body discloses the tier.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .contract import parse_json_object
from .lane import LaneResult, LaneSpec, run_claude_lane

TIER_GUIDE = {
    "trivial": "typo, comment, docs, rename, formatting or config-only change with no behaviour change; a second review round adds nothing",
    "low": "small, straightforward, well-contained change whose correctness is easy to confirm",
    "normal": "the default: an ordinary change to logic, tests, tooling, UI or build, including large or cross-cutting ones, whenever it does not meet the bar for critical",
    "critical": "BOTH large or intricate AND the diff itself changes a critical surface: consensus rules, funds movement or coin selection, cryptography, signatures or key handling, peer-facing network deserialization, or storage migrations. Size alone never qualifies; a small fix on a critical surface, a dependency bump, logging, metrics, tests, CI or build changes do not qualify. Name the file or function that meets the bar",
}


@dataclass(slots=True)
class Triage:
    tier: str
    method: str
    reasoning: str = ""
    error: str | None = None


def _prompt(
    repo: str, base_ref: str, title: str, body: str, files: list[dict[str, Any]], tiers: list[str]
) -> str:
    listing = "\n".join(f"- `{t}`: {TIER_GUIDE.get(t, '')}" for t in tiers)
    changed = "\n".join(
        f"- {f.get('filename')} (+{f.get('additions', 0)} -{f.get('deletions', 0)})"
        for f in files[:200]
    )
    adds = sum(int(f.get("additions") or 0) for f in files)
    dels = sum(int(f.get("deletions") or 0) for f in files)
    return (
        f"You rate the complexity and criticality of a pull request in {repo} (base branch `{base_ref}`) "
        "so an automated review can decide how much reasoning effort to spend on it.\n\n"
        f"PR title: {title}\n\nPR description:\n{body[:4000]}\n\n"
        f"Changed files ({len(files)}, +{adds} -{dels}):\n{changed}\n\n"
        f"Tiers:\n{listing}\n\n"
        "Judge by what the diff itself changes, not by how it is described or by how sensitive the surrounding subsystem is. "
        "`normal` is the default; move up only when the change clearly meets the bar for `critical`, and move down when the "
        "change is small or contained. When unsure between two tiers pick the lower one. "
        f'Reply with exactly one JSON object: {{"tier": "<one of {", ".join(tiers)}>", "reasoning": "one sentence"}}. No prose, no fences.'
    )


def triage(
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
) -> Triage:
    pol = cfg.policy
    assert pol.triage is not None
    tiers = list(pol.tiers)
    spec = LaneSpec(
        role="triage",
        agent=pol.triage.agent,
        model=pol.triage.model,
        effort=pol.triage.effort,
        prompt=_prompt(repo, base_ref, title, body, files, tiers),
        cwd=worktree,
        add_dir=run_dir,
        timeout_seconds=180,
        claude_bin=cfg.claude_bin,
    )
    error: str | None = None
    for attempt in (1, 2):
        try:
            res: LaneResult = runner(spec, run_dir / "triage" / f"attempt-{attempt}", worktree)
            if not res.ok:
                error = f"exit {res.exit_code} timed_out={res.timed_out}"
                continue
            obj = parse_json_object(res.result_text)
            tier = str(obj.get("tier") or "").strip().lower()
            if tier not in pol.tiers:
                error = f"unknown tier {tier!r}"
                continue
            return Triage(
                tier=tier,
                method=f"llm:{spec.model}",
                reasoning=str(obj.get("reasoning") or "")[:400],
            )
        except Exception as exc:
            error = str(exc)[:400]
    return Triage(tier=pol.fallback_tier, method="fallback", error=error)


def write_triage(run_dir: Path, t: Triage) -> None:
    (run_dir / "triage.json").write_text(json.dumps(dataclasses.asdict(t), indent=1))
