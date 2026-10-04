"""`reviewsys replay-set`: replay a set of past runs under several policies (the eval arms).

The set file is JSON (a list, or `{"heads": [...]}`) or JSON Lines; each entry names a source
run and, optionally, the eval set it belongs to and how many samples to take:

    {"run_id": 495, "set": "S4", "samples": 2, "note": "dash#7683 known FP"}

A bare integer is a run id. Every (head, policy, sample) becomes one `reviewsys replay` in its
own process, into `OUT/<policy>/run-<id>[-r<k>]/` (sample 0 has no suffix), with its output in
`run-<id>[-r<k>].log` beside it. A replay whose report.json is on disk is skipped, so an
interrupted set resumes where it stopped (`--retry-failed` also re-runs failed and invalid
ones). At most `--concurrency` replays run at once; for each head the arms run in turn, the
first arm alternating from head to head, so neither arm always goes first into a window.

The set stops launching replays when one aborts because degraded mode came on (stand-in
models would not measure either policy); the replays already running finish.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .replay import EXIT_ABORTED, run_dir_name

MANIFEST = "replay-set.json"


@dataclass(frozen=True, slots=True)
class SetItem:
    run_id: int
    set: str = ""
    samples: int = 1
    note: str = ""


@dataclass(frozen=True, slots=True)
class Policy:
    name: str
    skills: Path


@dataclass(frozen=True, slots=True)
class Job:
    item: SetItem
    policy: Policy
    sample: int

    @property
    def name(self) -> str:
        return run_dir_name(self.item.run_id, self.sample)

    def dir(self, out: Path) -> Path:
        return out / self.policy.name / self.name

    def log(self, out: Path) -> Path:
        return out / self.policy.name / f"{self.name}.log"


def load_records(path: Path, key: str) -> list[Any]:
    """A JSON list, a JSON object holding the list under `key`, or JSON Lines."""
    text = path.read_text()
    try:
        raw: Any = json.loads(text)
    except ValueError:
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(raw, dict):
        raw = raw.get(key, [raw])  # else a one-line JSON Lines file
    if not isinstance(raw, list):
        raise SystemExit(f"{path}: expected a list (or {{{key!r}: [...]}}, or JSON Lines)")
    return raw


def load_set(path: Path) -> list[SetItem]:
    raw = load_records(path, "heads")
    items: list[SetItem] = []
    seen: set[int] = set()
    for i, e in enumerate(raw):
        if isinstance(e, int):
            e = {"run_id": e}
        if not isinstance(e, dict) or not isinstance(e.get("run_id"), int):
            raise SystemExit(f"{path}: entry {i} has no integer run_id: {e!r}")
        if e["run_id"] in seen:
            raise SystemExit(f"{path}: run {e['run_id']} is listed twice")
        seen.add(e["run_id"])
        items.append(
            SetItem(
                run_id=int(e["run_id"]),
                set=str(e.get("set") or ""),
                samples=max(1, int(e.get("samples") or 1)),
                note=str(e.get("note") or ""),
            )
        )
    return items


def parse_policy(spec: str) -> Policy:
    name, sep, skills = spec.partition("=")
    if not sep or not name or not skills:
        raise SystemExit(f"--policy {spec!r}: expected NAME=SKILLS_DIR")
    if "/" in name or name.startswith("."):
        raise SystemExit(f"--policy {spec!r}: the name is a directory under --out")
    path = Path(skills).expanduser()
    if not (path / "config.json").exists():
        raise SystemExit(f"--policy {spec!r}: {path} has no config.json")
    return Policy(name, path.resolve())


def plan(items: list[SetItem], policies: list[Policy]) -> list[Job]:
    """Every (head, sample, policy), the arms of each head in turn, the first arm rotating."""
    jobs: list[Job] = []
    turn = 0
    for item in items:
        for sample in range(item.samples):
            k = turn % len(policies)
            for p in policies[k:] + policies[:k]:
                jobs.append(Job(item, p, sample))
            turn += 1
    return jobs


def report_state(job: Job, out: Path) -> str:
    """'missing', 'done', or what a finished replay's report says otherwise
    ('failed', 'cancelled', 'invalid')."""
    try:
        rep = json.loads((job.dir(out) / "report.json").read_text())
    except (OSError, ValueError):
        return "missing"
    if rep.get("invalid"):
        return "invalid"
    return str(rep.get("status") or "missing")


JobRunner = Callable[[Job, Path], int]  # (job, out) -> exit code of `reviewsys replay`


def subprocess_runner(
    config: str | None, *, force_phase2: bool, pin_phase1_model: str | None, source_db: str | None
) -> JobRunner:
    """Run each replay as `python -m reviewsys.cli replay ...`, output to the job's log."""

    def run(job: Job, out: Path) -> int:
        argv = [sys.executable, "-m", "reviewsys.cli"]
        if config:
            argv += ["--config", config]
        argv += [
            "replay",
            str(job.item.run_id),
            "--out",
            str(out / job.policy.name),
            "--skills",
            str(job.policy.skills),
            "--policy-name",
            job.policy.name,
            "--sample",
            str(job.sample),
        ]
        if force_phase2:
            argv.append("--force-phase2")
        if pin_phase1_model:
            argv += ["--pin-phase1-model", pin_phase1_model]
        if source_db:
            argv += ["--source-db", source_db]
        log = job.log(out)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(argv)}\n")
            fh.flush()
            return subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, check=False).returncode

    return run


def run_set(
    items: list[SetItem],
    policies: list[Policy],
    out: Path,
    *,
    concurrency: int,
    runner: JobRunner,
    retry_failed: bool = False,
    options: dict[str, Any] | None = None,
    echo: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run every job not already done; returns counts. Writes the set's manifest first."""
    if not policies:
        raise SystemExit("replay-set needs at least one --policy NAME=SKILLS_DIR")
    if len({p.name for p in policies}) != len(policies):
        raise SystemExit("replay-set: two --policy arguments share a name")
    out.mkdir(parents=True, exist_ok=True)
    (out / MANIFEST).write_text(
        json.dumps(
            {
                "policies": [{"name": p.name, "skills": str(p.skills)} for p in policies],
                "heads": [asdict(i) for i in items],
                "options": options or {},
            },
            indent=1,
        )
    )
    redo = {"failed", "cancelled", "invalid"} if retry_failed else set()
    todo: list[Job] = []
    counts = {"planned": 0, "skipped": 0, "ran": 0, "ok": 0, "failed": 0, "not_started": 0}
    for job in plan(items, policies):
        counts["planned"] += 1
        state = report_state(job, out)
        if state != "missing" and state not in redo:
            counts["skipped"] += 1
            continue
        todo.append(job)
    echo(
        f"replay-set: {len(todo)} to run, {counts['skipped']} already done, "
        f"{len(policies)} policies, concurrency {concurrency}"
    )
    aborted = ""
    pending = list(reversed(todo))
    running: dict[Future[int], Job] = {}
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        while pending or running:
            while pending and not aborted and len(running) < max(1, concurrency):
                job = pending.pop()
                running[pool.submit(runner, job, out)] = job
            if not running:
                break
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                job = running.pop(fut)
                try:
                    rc = fut.result()
                except Exception as exc:  # a crashed launcher is that job's failure only
                    rc = 1
                    echo(f"{job.policy.name}/{job.name}: launcher error {exc}")
                counts["ran"] += 1
                state = report_state(job, out)
                counts["ok" if rc == 0 and state == "done" else "failed"] += 1
                echo(f"{job.policy.name}/{job.name}: exit {rc}, report {state}")
                if rc == EXIT_ABORTED and state == "missing" and not aborted:
                    aborted = f"{job.policy.name}/{job.name} aborted (degraded mode?)"
                    echo(f"replay-set: stopping: {aborted}; see {job.log(out)}")
    counts["not_started"] = len(pending)
    return {**counts, "aborted": aborted}


def main_set(args: Any, config: str | None) -> int:
    items = load_set(Path(args.setfile).expanduser())
    policies = [parse_policy(p) for p in args.policy]
    out = Path(args.out).expanduser()
    runner = subprocess_runner(
        config,
        force_phase2=args.force_phase2,
        pin_phase1_model=args.pin_phase1_model,
        source_db=args.source_db,
    )
    res = run_set(
        items,
        policies,
        out,
        concurrency=args.concurrency,
        runner=runner,
        retry_failed=args.retry_failed,
        options={
            "force_phase2": args.force_phase2,
            "pin_phase1_model": args.pin_phase1_model,
            "source_db": args.source_db,
        },
    )
    print(json.dumps(res))
    return EXIT_ABORTED if res["aborted"] else 0 if not res["failed"] else 1
