"""`reviewsys replay-set`: replay a set of past runs under several policies (the eval arms).

The set file is JSON or JSON Lines: a list of heads, or an object with `heads` (and optionally
`repeats`, a list of head ids to take two samples of). A head names its source run (`run_id`,
or `source_run_id` as eval/set.json has it) and optionally its `id`, the eval `set` it belongs
to, `samples` and a `note`; a bare integer is a run id:

    {"heads": [{"id": "dash-7683-5a06c55a", "source_run_id": 495, "set": "S4"}],
     "repeats": ["dash-7683-5a06c55a"]}

Every (head, policy, sample) becomes one `reviewsys replay` in its own process, into
`OUT/<policy>/run-<id>[-r<k>]/` (sample 0 has no suffix), with its output in
`run-<id>[-r<k>].log` beside it. A replay whose report.json is on disk is skipped, so an
interrupted set resumes where it stopped (`--retry-failed` also re-runs failed and invalid
ones, their old report removed first). At most `--concurrency` replays run at once; for each
head the arms run in turn, the first arm alternating from head to head, so neither arm always
goes first into a window. The replays share the machine's lane pools (replay.py), so the
concurrency only bounds how many runs are open at once, not the model streams.

The set stops launching replays when one exits with `replay.EXIT_ABORTED` (degraded mode came
on, or its models ran dry mid-run: stand-ins would not measure either policy); the replays
already running finish.

Before every launch (`Gate`, shared with the judge):
- Production (`production_guard`, the configured database read-only, fresh each time): with
  degraded mode forced or active, or the latest OpenAI account reading missing, stale or
  under `--min-usable-accounts` (2) usable accounts, the set stops launching (the running
  replays finish; exit 75, resumable). With the live queue at half the box's
  `backlog_skip_phase1_above` it holds launches, and when nothing is running waits up to
  `--busy-wait-minutes` (60) for the queue to drain before it stops the same way.
- Disk: a replay's worktree is a full checkout (a few GB on the big repositories) until it
  ends. Free space where `--out` lives, less `--disk-per-job-gb` (6) for every replay
  running, must stay above `--min-free-gb` (35); below it the set holds launches until
  running replays end (each frees its worktree and pinned clone), and with none left waits up
  to `--disk-wait-minutes` (30) before it stops (exit EXIT_DISK); a re-run resumes.

`--limit N` runs only the first N heads, taken round-robin over the eval sets, for a smoke
test.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import config as cfg_mod
from . import degraded, slots
from .db import kv_get, now_dt, parse_ts
from .replay import EXIT_ABORTED, TOKEN_ENV, read_json, run_dir_name
from .scheduler import live_queued_count

MANIFEST = "replay-set.json"
EXIT_DISK = 74  # EX_IOERR: the disk stayed below the floor; nothing more was launched
DEFAULT_MIN_FREE_GB = 35.0
DEFAULT_JOB_GB = 6.0  # a running replay's (or judged head's) worktree and clone, at most
DEFAULT_MIN_ACCOUNTS = 2
WAIT_POLL_SECONDS = 60.0
GB = 1024**3


def free_bytes(path: Path) -> int:
    """Free space on the filesystem holding `path` (its nearest existing parent)."""
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


@dataclass(frozen=True, slots=True)
class SetItem:
    run_id: int
    set: str = ""
    samples: int = 1
    note: str = ""
    head_id: str = ""  # eval/set.json's `id`: labels name their head by it


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


def load_records(path: Path, key: str) -> tuple[list[Any], dict[str, Any]]:
    """(records, the enclosing object) of a JSON list, a JSON object holding the list under
    `key`, or JSON Lines."""
    text = path.read_text()
    try:
        raw: Any = json.loads(text)
    except ValueError:
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    top: dict[str, Any] = {}
    if isinstance(raw, dict):
        top, raw = (raw, raw[key]) if key in raw else ({}, [raw])  # a one-line JSON Lines file
    if not isinstance(raw, list):
        raise SystemExit(f"{path}: expected a list (or {{{key!r}: [...]}}, or JSON Lines)")
    return raw, top


def load_set(path: Path) -> list[SetItem]:
    raw, top = load_records(path, "heads")
    repeats = {str(x) for x in top.get("repeats") or []}
    items: list[SetItem] = []
    seen: set[int] = set()
    for i, e in enumerate(raw):
        if isinstance(e, int):
            e = {"run_id": e}
        rid = e.get("run_id", e.get("source_run_id")) if isinstance(e, dict) else None
        if not isinstance(rid, int):
            raise SystemExit(f"{path}: entry {i} has no integer run_id: {e!r}")
        if rid in seen:
            raise SystemExit(f"{path}: run {rid} is listed twice")
        seen.add(rid)
        head_id = str(e.get("id") or "")
        samples = max(1, int(e.get("samples") or 1))
        items.append(
            SetItem(
                run_id=rid,
                set=str(e.get("set") or ""),
                samples=max(samples, 2) if head_id and head_id in repeats else samples,
                note=str(e.get("note") or e.get("why") or ""),
                head_id=head_id,
            )
        )
    return items


def production_guard(db: Path, *, min_accounts: int, backlog_limit: int) -> tuple[str, bool]:
    """(why launching must wait, whether to stop rather than wait); ("", False) = launch.
    Read from the production database's latest readings, read-only (see the module
    docstring); a database that cannot be read right now means wait."""
    if not db.exists():
        return f"the production database {db} is not there", True
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
    except sqlite3.Error as exc:
        return f"cannot open the production database: {exc}", False
    conn.row_factory = sqlite3.Row
    try:
        if degraded.forced(conn) == "on":
            return "degraded mode is forced on", True
        try:
            active = bool(json.loads(kv_get(conn, degraded.KV_STATE) or "{}").get("active"))
        except (ValueError, AttributeError):
            active = False
        if active:
            return "degraded mode is active", True
        if min_accounts > 0:
            stored = slots.stored_accounts(conn)
            if stored is None:
                return "no OpenAI account reading in the production database", True
            at, accounts = stored
            if now_dt() - parse_ts(at) > slots.READING_MAX_AGE:
                return f"the OpenAI account reading is stale ({at}); is the daemon running?", True
            usable = sum(1 for a in accounts if a.usable)
            if usable < min_accounts:
                return f"{usable} usable OpenAI account(s), fewer than {min_accounts}", True
        if backlog_limit > 0:
            queued = live_queued_count(conn)
            if queued >= max(1, backlog_limit // 2):
                return (
                    f"{queued} live heads queued (the box skips Phase 1 above {backlog_limit})",
                    False,
                )
    except sqlite3.Error as exc:
        return f"cannot read the production database: {exc}", False
    finally:
        conn.close()
    return "", False


def production_settings(config: str | None) -> tuple[Path, int]:
    """(database, backlog_skip_phase1_above) of a config.toml, without loading its skills."""
    path = Path(config).expanduser() if config else cfg_mod.DEFAULT_CONFIG_PATH
    t = tomllib.loads(path.read_text() if path.exists() else cfg_mod.DEFAULT_TOML)
    sched = t.get("scheduling") or {}
    return Path(t["paths"]["db"]).expanduser(), int(sched.get("backlog_skip_phase1_above", 10))


@dataclass
class Gate:
    """Whether one more replay (or judged head) may start now: production's state, then disk
    (see the module docstring). Shared by replay-set and replay-judge."""

    out: Path
    min_free_gb: float = 0.0
    job_gb: float = DEFAULT_JOB_GB
    disk_wait_minutes: float = 0.0
    busy_wait_minutes: float = 0.0
    guard: Callable[[], tuple[str, bool]] | None = None
    disk_free: Callable[[Path], int] | None = None
    sleep: Callable[[float], None] = field(default=time.sleep)

    def check(self, running: int) -> tuple[str, str]:
        """("", "") = go; else (why, kind): kind `stop` (degraded, accounts), `busy`
        (production's queue, an unreadable database) or `disk`."""
        if self.guard is not None:
            why, stop = self.guard()
            if why:
                return why, "stop" if stop else "busy"
        if self.min_free_gb > 0:
            free = (self.disk_free or free_bytes)(self.out)
            if free - running * self.job_gb * GB < self.min_free_gb * GB:
                return (
                    f"{free / GB:.1f} GB free with {running} running (~{self.job_gb:g} GB each), "
                    f"floor {self.min_free_gb:g} GB",
                    "disk",
                )
        return "", ""

    def wait_idle(self, why: str, kind: str) -> tuple[str, str]:
        """Nothing is running and launching is held: poll up to the kind's wait; ("", "") once
        clear, else what still holds it."""
        limit = (self.disk_wait_minutes if kind == "disk" else self.busy_wait_minutes) * 60
        waited = 0.0
        while kind and kind != "stop" and waited < limit:
            self.sleep(WAIT_POLL_SECONDS)
            waited += WAIT_POLL_SECONDS
            why, kind = self.check(0)
        return why, kind


def gate_from_args(args: Any, out: Path, db: Path, backlog_limit: int) -> Gate:
    guard = None
    if not args.no_production_guard:

        def guard() -> tuple[str, bool]:
            return production_guard(
                db, min_accounts=args.min_usable_accounts, backlog_limit=backlog_limit
            )

    return Gate(
        out,
        min_free_gb=args.min_free_gb,
        job_gb=args.disk_per_job_gb,
        disk_wait_minutes=args.disk_wait_minutes,
        busy_wait_minutes=args.busy_wait_minutes,
        guard=guard,
    )


def stopped_exit(aborted: str) -> int:
    return EXIT_DISK if aborted.startswith("disk:") else EXIT_ABORTED


def limit_items(items: list[SetItem], n: int | None) -> list[SetItem]:
    """The first `n` heads taken round-robin over the eval sets in order of appearance (one
    of each set first), in set-file order within a set; all of them without `n`."""
    if not n or n >= len(items):
        return items
    by_set: dict[str, list[SetItem]] = {}
    for i in items:
        by_set.setdefault(i.set, []).append(i)
    picked: list[SetItem] = []
    queues = list(by_set.values())
    while len(picked) < n:
        for q in queues:
            if q and len(picked) < n:
                picked.append(q.pop(0))
    keep = {i.run_id for i in picked}
    return [i for i in items if i.run_id in keep]


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
            jobs += [Job(item, p, sample) for p in policies[k:] + policies[:k]]
            turn += 1
    return jobs


def load_report(job: Job, out: Path) -> dict[str, Any] | None:
    rep = read_json(job.dir(out) / "report.json")
    return rep if isinstance(rep, dict) else None


def report_state(rep: dict[str, Any] | None) -> str:
    """'missing', 'done', or what a finished replay's report says otherwise
    ('failed', 'cancelled', 'invalid')."""
    if rep is None:
        return "missing"
    if rep.get("invalid"):
        return "invalid"
    return str(rep.get("status") or "missing")


JobRunner = Callable[[Job, Path], int]  # (job, out) -> exit code of `reviewsys replay`


def subprocess_runner(
    config: str | None,
    *,
    force_phase2: bool,
    pin_phase1_model: str | None,
    source_db: str | None,
    source_runs: str | None = None,
    gh_token_env: str = TOKEN_ENV,
    fake_lanes: bool = False,
    eval_mirrors: str | None = None,
    pool_slots: list[str] | None = None,
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
            "--gh-token-env",
            gh_token_env,
        ]
        if force_phase2:
            argv.append("--force-phase2")
        if pin_phase1_model:
            argv += ["--pin-phase1-model", pin_phase1_model]
        if source_db:
            argv += ["--source-db", source_db]
        if source_runs:
            argv += ["--source-runs", source_runs]
        if fake_lanes:
            argv.append("--fake-lanes")
        # one place for the mirrors the box has none of, shared by the arms
        argv += ["--eval-mirrors", eval_mirrors or str(out / ".mirrors")]
        for spec in pool_slots or []:
            argv += ["--replay-pool-slots", spec]
        log = job.log(out)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(argv)}\n")
            fh.flush()
            return subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, check=False).returncode

    return run


def _write_manifest(
    out: Path, items: list[SetItem], policies: list[Policy], options: dict[str, Any]
) -> None:
    """The set's manifest, merged with an earlier one: running a subset of the arms (or of the
    heads) adds to what the directory holds, it never forgets the arms already there."""
    old = read_json(out / MANIFEST)
    old = old if isinstance(old, dict) else {}
    arms = {str(p["name"]): str(p["skills"]) for p in old.get("policies") or []}
    arms.update({p.name: str(p.skills) for p in policies})
    heads = {int(h["run_id"]): h for h in old.get("heads") or []}
    heads.update({i.run_id: asdict(i) for i in items})
    (out / MANIFEST).write_text(
        json.dumps(
            {
                "policies": [{"name": n, "skills": s} for n, s in arms.items()],
                "heads": list(heads.values()),
                "options": {**(old.get("options") or {}), **options},
            },
            indent=1,
        )
    )


def gated_pool[T, R](
    todo: list[T],
    run: Callable[[T], R],
    *,
    concurrency: int,
    gate: Gate,
    echo: Callable[[str], None],
    label: str,
    finished: Callable[[T, R | Exception], str],
) -> tuple[str, list[T]]:
    """Run `todo` (in order) at most `concurrency` at once, each launch only when `gate`
    says so (held while it says wait, stopped when it says stop or a wait runs out).
    `finished` books a result (or the exception) and may return a reason to stop launching.
    Returns (why it stopped, "" when it did not; the items never started)."""
    pending = list(reversed(todo))
    running: dict[Future[R], T] = {}
    aborted = ""
    held = ("", "")
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        while pending or running:
            while pending and not aborted and len(running) < max(1, concurrency):
                why, kind = gate.check(len(running))
                if kind == "stop":
                    aborted = f"stopped: {why}"
                    echo(f"{label}: stopping launches: {why}")
                    break
                if kind:
                    if held[0] != why:
                        echo(f"{label}: holding launches: {why}")
                    held = (why, kind)
                    break
                if held[1]:
                    echo(f"{label}: launching again")
                    held = ("", "")
                item = pending.pop()
                running[pool.submit(run, item)] = item
            if not running:
                if pending and not aborted and held[1]:
                    why, kind = gate.wait_idle(*held)
                    if not kind:
                        continue
                    aborted = f"{'disk' if kind == 'disk' else 'stopped'}: {why}"
                    echo(f"{label}: stopping: {aborted}")
                break
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                item = running.pop(fut)
                try:
                    res: R | Exception = fut.result()
                except Exception as exc:  # a crashed launch is that item's failure only
                    res = exc
                reason = finished(item, res)
                if reason and not aborted:
                    aborted = reason
    return aborted, pending


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
    gate: Gate | None = None,
) -> dict[str, Any]:
    """Run every job not already done; returns counts. Writes the set's manifest first. A
    job is launched only when `gate` says so (see the module docstring)."""
    if not policies:
        raise SystemExit("replay-set needs at least one --policy NAME=SKILLS_DIR")
    if len({p.name for p in policies}) != len(policies):
        raise SystemExit("replay-set: two --policy arguments share a name")
    out.mkdir(parents=True, exist_ok=True)
    _write_manifest(out, items, policies, options or {})
    slots_n = max(1, concurrency)
    redo = {"failed", "cancelled", "invalid"} if retry_failed else set()
    jobs = plan(items, policies)
    todo: list[Job] = []
    for job in jobs:
        state = report_state(load_report(job, out))
        if state == "missing" or state in redo:
            todo.append(job)
    counts = {
        "planned": len(jobs),
        "skipped": len(jobs) - len(todo),
        "ran": 0,
        "ok": 0,
        "failed": 0,
        "not_started": 0,
    }
    echo(
        f"replay-set: {len(todo)} to run, {counts['skipped']} already done, "
        f"{len(policies)} policies, concurrency {slots_n}"
    )

    def start(job: Job) -> int:
        # a stale report would read as this attempt's (and hide an abort)
        (job.dir(out) / "report.json").unlink(missing_ok=True)
        return runner(job, out)

    def finished(job: Job, res: int | Exception) -> str:
        if isinstance(res, Exception):
            echo(f"{job.policy.name}/{job.name}: launcher error {res}")
            rc = 1
        else:
            rc = res
        counts["ran"] += 1
        state = report_state(load_report(job, out))
        counts["ok" if rc == 0 and state == "done" else "failed"] += 1
        echo(f"{job.policy.name}/{job.name}: exit {rc}, report {state}")
        if rc != EXIT_ABORTED:
            return ""
        why = f"stopped: {job.policy.name}/{job.name} aborted (degraded mode or dry models)"
        echo(f"replay-set: {why}; see {job.log(out)}")
        return why

    aborted, pending = gated_pool(
        todo,
        start,
        concurrency=slots_n,
        gate=gate or Gate(out),
        echo=echo,
        label="replay-set",
        finished=finished,
    )
    counts["not_started"] = len(pending)
    return {**counts, "aborted": aborted}


def main_set(args: Any) -> int:
    items = limit_items(load_set(Path(args.setfile).expanduser()), args.limit)
    policies = [parse_policy(p) for p in args.policy]
    out = Path(args.out).expanduser()
    runner = subprocess_runner(
        args.config,
        force_phase2=args.force_phase2,
        pin_phase1_model=args.pin_phase1_model,
        source_db=args.source_db,
        source_runs=args.source_runs,
        gh_token_env=args.gh_token_env,
        fake_lanes=args.fake_lanes,
        eval_mirrors=args.eval_mirrors,
        pool_slots=args.replay_pool_slots,
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
            **({"limit": args.limit} if args.limit else {}),
            **({"fake_lanes": True} if args.fake_lanes else {}),
        },
        gate=gate_from_args(args, out, *production_settings(args.config)),
    )
    print(json.dumps(res))
    if res["aborted"]:
        return stopped_exit(str(res["aborted"]))
    return 0 if not res["failed"] else 1
