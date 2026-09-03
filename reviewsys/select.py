"""Specialist selection: always-run specialists plus an LLM pick over discretionary ones.

Failure never silently degrades: the result carries `method` and `error` so the
worker records it as an event and the review body can say so.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config, Specialist
from .contract import parse_json_object
from .lane import LaneResult, LaneSpec, run_claude_lane


@dataclass(slots=True)
class Selection:
    selected: list[str]
    method: str
    reasoning: str = ""
    error: str | None = None
    considered: list[str] = field(default_factory=list)


def heuristic(specialists: list[Specialist], title: str, body: str, files: list[str]) -> list[str]:
    text = f"{title}\n{body}".lower()
    out = []
    for s in specialists:
        hint = s.trigger_hint
        if hint and re.search(hint, text, re.IGNORECASE):
            out.append(s.id)
            continue
        if (
            (s.id == "rust-quality" and any(f.endswith(".rs") for f in files))
            or (
                s.id == "ffi-engineer"
                and any(re.search(r"ffi|wasm|\.swift$|bindings", f, re.I) for f in files)
            )
            or (
                s.id == "security-auditor"
                and re.search(r"crypto|signature|auth|proof|consensus|unsafe|verify", text)
            )
        ):
            out.append(s.id)
    return out


def _prompt(
    repo: str, title: str, body: str, files: list[str], specialists: list[Specialist]
) -> str:
    listing = "\n".join(f"- `{s.id}`: {s.description}" for s in specialists)
    return (
        f"You select specialist code reviewers for a pull request in {repo}.\n\n"
        f"PR title: {title}\n\nPR description:\n{body[:4000]}\n\nChanged files ({len(files)}):\n"
        + "\n".join(f"- {f}" for f in files[:200])
        + "\n\n"
        f"Available specialists:\n{listing}\n\n"
        "Pick every specialist whose expertise is clearly relevant to these changes; pick none if none apply. "
        'Reply with exactly one JSON object: {"selected": ["id", ...], "reasoning": "one sentence"}. No prose, no fences.'
    )


def select(
    cfg: Config,
    *,
    repo: str,
    title: str,
    body: str,
    files: list[str],
    run_dir: Path,
    worktree: Path,
    runner: Any = run_claude_lane,
) -> Selection:
    available = list(cfg.specialists_for(repo))
    always = [s.id for s in available if s.always_run]
    discretionary = [s for s in available if not s.always_run]
    if not discretionary:
        return Selection(selected=always, method="config", considered=[s.id for s in available])
    spec = LaneSpec(
        role="selector",
        agent="selector",
        model=cfg.policy.selector_model,
        effort="low",
        prompt=_prompt(repo, title, body, files, discretionary),
        cwd=worktree,
        add_dir=run_dir,
        timeout_seconds=120,
        claude_bin=cfg.claude_bin,
    )
    error: str | None = None
    for attempt, model in enumerate(
        (cfg.policy.selector_model, cfg.policy.phase2_reviewer.model), 1
    ):
        spec = dataclasses.replace(spec, model=model)
        try:
            res: LaneResult = runner(spec, run_dir / "selector" / f"attempt-{attempt}", worktree)
            if not res.ok:
                error = f"{model}: exit {res.exit_code} timed_out={res.timed_out}"
                continue
            obj = parse_json_object(res.result_text)
            picked = [
                str(x) for x in obj.get("selected") or [] if str(x) in {s.id for s in discretionary}
            ]
            return Selection(
                selected=sorted(set(always) | set(picked)),
                method=f"llm:{model}",
                reasoning=str(obj.get("reasoning") or ""),
                considered=[s.id for s in available],
            )
        except Exception as exc:
            error = f"{model}: {exc}"
    fallback = heuristic(discretionary, title, body, files)
    return Selection(
        selected=sorted(set(always) | set(fallback)),
        method="heuristic",
        error=error,
        considered=[s.id for s in available],
    )


def write_selection(run_dir: Path, sel: Selection) -> None:
    (run_dir / "selector.json").write_text(json.dumps(dataclasses.asdict(sel), indent=1))
