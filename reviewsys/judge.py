"""`reviewsys replay-judge`: the blinded judge lane of a replay eval (v12 DESIGN.md §5.1).

`replay-score` leaves what its matcher cannot settle in `to_judge.jsonl` (replayscore.py):
`new_blocker` items (a published blocker matching no label: is it a real defect at this
head?) and `label_match` items (a finding near a label it did not match: is it the same
defect?). This runs one lane per item:

- Blinded: the prompt names no arm, policy or label class, and the items run in an order
  shuffled with `--seed`.
- On gpt (default `gpt-6.1-sol`) at `xhigh`, with read access to a worktree at the replayed
  head, checked out from a pinned clone (replay.pinned_clone: only the base and head refs, no
  later commit, removed after the head's items). No GitHub: the installed gh is denied by the
  lane sandbox, the `gh` on PATH answers nothing and a curl shim refuses GitHub hosts;
  WebFetch/WebSearch are off and the lane has no token.
- Machine: the lanes take slots in the configured work dir's lane pools at the `replay`
  rank (lanepool.py), like the replays, the gpt pool sized from the configured database.
- Resumable: each answer is appended to `judged.jsonl` (the item plus `verdict` and `judge`)
  as soon as it is in; an item whose id is there is not asked again. A lane that fails
  leaves its item unanswered for the next run.
- Disk: like `replay-set`, a head's worktree is only checked out while the free space is at
  least `--min-free-gb`; when it is short and nothing is running, the judge stops.

Run as a whole: score (to refresh `to_judge.jsonl`), judge what is pending, score again (the
verdicts give precision and settle label matches; see replayscore.py).
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import shutil
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from . import config as cfg_mod
from .db import connect, now
from .ghshim import install_shim
from .lane import (
    SANDBOX_EXEC,
    LaneRunner,
    LaneSpec,
    exec_deny_profile,
    isolated,
    lane_output,
    run_claude_lane,
    sandboxed,
    with_overrides,
)
from .lanepool import gated
from .models import ReviewError
from .replay import LANE_DISALLOWED_TOOLS, pinned_clone, pinned_refs
from .replayscore import JUDGED, LABEL_MATCH, NEW_BLOCKER, TO_JUDGE, render, score_dir
from .replayset import DEFAULT_MIN_FREE_GB, EXIT_DISK, GB, free_bytes
from .steps import worktree as wt

DEFAULT_MODEL = "gpt-6.1-sol"
DEFAULT_EFFORT = "xhigh"
ROLE = "judge"
NO_GH_SOCKET = "/nonexistent/reviewsys-judge-has-no-gh"
CHOICE = {"type": "string", "enum": ["yes", "no", "unsure"]}
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "real": CHOICE,
        "evidence": {"type": "string"},
        "matches_label": {"anyOf": [CHOICE, {"type": "null"}]},
        "reason": {"type": "string"},
    },
    "required": ["real", "evidence", "matches_label", "reason"],
    "additionalProperties": False,
}


def prompt(item: dict[str, Any]) -> str:
    """The judge's question about one item; nothing in it says which arm raised it."""
    lines = f"{item.get('line_start')}-{item.get('line_end')}"
    base = item.get("merge_base")
    scope = (
        f"The pull request's own changes are `git diff {base}..{item['sha']}`."
        if base
        else "The pull request's base is not known here; judge the code as it is."
    )
    text = f"""# Judge one code-review finding

Repository {item["repo"]}, pull request #{item["number"]}, checked out read-only in the working
directory at commit `{item["sha"]}`. {scope}

You have no GitHub and no network: read the code with your tools (read files, search, `git
log`, `git show`, `git diff`). Do not build, run tests, or change any file.

## The finding

- File: `{item.get("file")}`, lines {lines}
- Severity claimed: {item.get("severity") or "blocking"}
- Category: {item.get("category") or "-"}
- Title: {item.get("title")}

{item.get("body") or "(no body)"}

## Question 1: is it real?

Is this a real defect in the code at this commit, as the finding describes it (not a style
preference, not something the code already handles elsewhere, not a misreading)? Answer
`yes`, `no` or `unsure`, and quote the code that decides it: `file:line` and the line text.
"""
    if item.get("kind") == LABEL_MATCH:
        lab = item.get("label") or {}
        text += f"""
## Question 2: same defect?

Another review of this commit reported this issue:

- File: `{lab.get("file")}`, lines {lab.get("line_start")}-{lab.get("line_end")}
- Title: {lab.get("title")}

Does the finding above describe the same underlying defect (the same broken behaviour at the
same place, however differently worded)? Answer `yes`, `no` or `unsure`.
"""
    else:
        text += "\nThere is no second question: answer `matches_label` with null.\n"
    text += """
## Output

Only a JSON object:
`{"real": "yes|no|unsure", "evidence": "<quoted code with file:line>",
"matches_label": "yes|no|unsure" or null, "reason": "<one or two sentences>"}`
"""
    return text


def parse_verdict(raw: dict[str, Any], kind: str) -> dict[str, Any]:
    """A lane's answer as a verdict; raises ValueError when it is not one."""

    def choice(v: Any) -> str:
        s = str(v or "").strip().lower()
        if s not in ("yes", "no", "unsure"):
            raise ValueError(f"not yes/no/unsure: {v!r}")
        return s

    out = {
        "real": choice(raw.get("real")),
        "evidence": str(raw.get("evidence") or "")[:4000],
        "matches_label": None,
        "reason": str(raw.get("reason") or "")[:2000],
    }
    if kind == LABEL_MATCH:
        out["matches_label"] = choice(raw.get("matches_label"))
    return out


def judged_ids(out: Path) -> set[str]:
    path = out / JUDGED
    ids: set[str] = set()
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            it = json.loads(line)
        except ValueError:
            continue
        if isinstance(it, dict) and it.get("verdict") and it.get("id"):
            ids.add(str(it["id"]))
    return ids


def pending_items(out: Path, seed: int) -> list[dict[str, Any]]:
    """to_judge.jsonl's items not yet in judged.jsonl, in a seeded shuffled order."""
    path = out / TO_JUDGE
    items = (
        [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    )
    done = judged_ids(out)
    todo = [it for it in items if str(it.get("id")) not in done]
    random.Random(seed).shuffle(todo)
    return todo


def _head_of(it: dict[str, Any]) -> tuple[str, int, str]:
    return (str(it["repo"]), int(it["number"]), str(it["sha"]))


def _thread_conn(path: Path) -> Callable[[], sqlite3.Connection]:
    local = threading.local()

    def get() -> sqlite3.Connection:
        c = getattr(local, "conn", None)
        if c is None:
            c = local.conn = connect(path)
        return c

    return get


def judge(
    cfg: cfg_mod.Config,
    out: Path,
    *,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    concurrency: int = 3,
    seed: int = 0,
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
    lane_runner: LaneRunner | None = None,
    disk_free: Callable[[Path], int] | None = None,
    echo: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Ask every pending item of `out`'s to_judge.jsonl (see the module docstring). Returns
    counts; `stopped` says why it ended early ("" when it did not)."""
    todo = pending_items(out, seed)
    by_head: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for it in todo:
        by_head[_head_of(it)].append(it)
    heads = list(by_head)  # in the order of their first item: the shuffled order
    counts: dict[str, Any] = {"pending": len(todo), "heads": len(heads), "judged": 0, "failed": 0}
    echo(f"replay-judge: {len(todo)} items on {len(heads)} heads, model {model} at {effort}")
    if not todo:
        return {**counts, "stopped": ""}
    root = out / "judge"
    work = root / "work"
    (work / "runs").mkdir(parents=True, exist_ok=True)
    slots = work / "lane-slots"  # the machine's pools, shared with production and the replays
    if not slots.is_symlink():
        (cfg.work_dir / "lane-slots").mkdir(parents=True, exist_ok=True)
        slots.symlink_to((cfg.work_dir / "lane-slots").resolve())
    installed_gh = shutil.which(cfg.gh_bin) or cfg.gh_bin
    deny_gh = (
        (installed_gh,) if os.path.exists(SANDBOX_EXEC) and os.path.isabs(installed_gh) else ()
    )
    jcfg = dataclasses.replace(
        cfg,
        work_dir=work,
        db_path=root / "judge.db",
        lane_capacity_db=cfg.db_path,
        lane_deny_exec=(*cfg.lane_deny_exec, *deny_gh),
    )
    bin_dir = root / "bin"
    install_shim(NO_GH_SOCKET, bin_dir, curl_log=root / "curl-refused.log")
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "GH_TOKEN": "",
        "GITHUB_TOKEN": "",
    }
    runner = gated(
        sandboxed(
            isolated(
                with_overrides(
                    lane_runner or run_claude_lane,
                    env=env,
                    disallowed_tools=LANE_DISALLOWED_TOOLS,
                ),
                True,
            ),
            exec_deny_profile(jcfg.lane_deny_exec),
        ),
        _thread_conn(jcfg.db_path),
        jcfg,
        run_stopped=lambda: False,
        queue="replay",
        run_id=0,
    )
    write_lock = threading.Lock()
    floor = int(min_free_gb * GB)

    def record(it: dict[str, Any], verdict: dict[str, Any], lane_dir: Path) -> None:
        line = {
            **it,
            "verdict": verdict,
            "judge": {"model": model, "effort": effort, "at": now(), "lane": str(lane_dir)},
        }
        with write_lock, (out / JUDGED).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
            fh.flush()

    def one_head(head: tuple[str, int, str]) -> tuple[int, int]:
        repo, number, sha = head
        items = by_head[head]
        first = items[0]
        name = f"{repo.replace('/', '__')}-{number}-{sha[:12]}"
        mirror = wt.ensure_mirror(cfg.mirrors_dir, repo)
        wt.fetch_head(mirror, number, sha)
        base_ref, base_sha = str(first.get("base_ref") or ""), str(first.get("base_sha") or "")
        if base_sha and not wt.ensure_commit(mirror, base_sha):
            base_ref = base_sha = ""
        pinned = pinned_clone(
            mirror,
            work / "mirrors" / name,
            pinned_refs(number=number, sha=sha, base_ref=base_ref, base_sha=base_sha),
            head_branch=base_ref if base_ref and base_sha else "replay-head",
        )
        ok = failed = 0
        tree = None
        try:
            tree = wt.create_worktree(pinned, work / "worktrees", name, sha)
            for it in items:
                lane_dir = work / "runs" / str(it["id"])
                spec = LaneSpec(
                    role=ROLE,
                    agent="replay-judge",
                    model=model,
                    effort=effort,
                    prompt=prompt(it),
                    cwd=tree,
                    add_dir=tree,
                    timeout_seconds=cfg.lane_timeout_minutes * 60,
                    claude_bin=cfg.claude_bin,
                    max_budget_usd=cfg.lane_budget_usd,
                    json_schema=SCHEMA,
                )
                try:
                    res = runner(spec, lane_dir, tree)
                    verdict = parse_verdict(lane_output(res), str(it.get("kind") or NEW_BLOCKER))
                except (ReviewError, ValueError) as exc:
                    failed += 1
                    echo(f"replay-judge: {it['id']}: no verdict ({exc})")
                    continue
                record(it, verdict, lane_dir)
                ok += 1
        finally:
            if tree is not None:
                wt.remove_worktree(pinned, tree)
            shutil.rmtree(pinned, ignore_errors=True)
            shutil.rmtree(work / "worktrees" / name, ignore_errors=True)
        return ok, failed

    stopped = ""
    queue = list(reversed(heads))
    running: dict[Future[tuple[int, int]], tuple[str, int, str]] = {}
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        while queue or running:
            while queue and len(running) < max(1, concurrency) and not stopped:
                free = (disk_free or free_bytes)(out)
                if free < floor:
                    if not running:
                        stopped = f"disk: {free / GB:.1f} GB free < {min_free_gb:g} GB floor"
                        echo(f"replay-judge: stopping: {stopped}")
                    break
                head = queue.pop()
                running[pool.submit(one_head, head)] = head
            if not running:
                break
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                head = running.pop(fut)
                try:
                    ok, failed = fut.result()
                except Exception as exc:  # one head's checkout failing is that head's failure
                    ok, failed = 0, len(by_head[head])
                    echo(f"replay-judge: {head[0]}#{head[1]} {head[2][:12]}: {exc}")
                counts["judged"] += ok
                counts["failed"] += failed
    counts["not_started"] = sum(len(by_head[h]) for h in queue)
    return {**counts, "stopped": stopped}


def main_judge(args: Any, cfg: cfg_mod.Config, lane_runner: LaneRunner | None = None) -> int:
    out = Path(args.out_dir).expanduser()
    setfile, labels = Path(args.set).expanduser(), Path(args.labels).expanduser()
    score_dir(setfile, labels, out, baseline=args.baseline, seed=args.seed, limit=args.limit)
    t0 = time.monotonic()
    res = judge(
        cfg,
        out,
        model=args.model,
        effort=args.effort,
        concurrency=args.concurrency,
        seed=args.seed,
        min_free_gb=args.min_free_gb,
        lane_runner=lane_runner,
    )
    res["wall_clock_s"] = round(time.monotonic() - t0, 1)
    result, _ = score_dir(
        setfile, labels, out, baseline=args.baseline, seed=args.seed, limit=args.limit
    )
    print(json.dumps(res))
    print(render(result), end="")
    if res["stopped"]:
        print(f"replay-judge: stopped early: {res['stopped']}", file=sys.stderr)
        return EXIT_DISK
    return 1 if res["failed"] else 0
