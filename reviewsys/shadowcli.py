"""`reviewsys shadow {run,tick,status,report,score}` (shadow.py).

Settings come from the command line, else from config.toml's `[shadow]` table (`root`,
`skills`, `sample_rate`, `max_age_hours`, `concurrency`, `max_heads_per_day`, `guard_cmd`,
`stop_on_guard`), else the defaults in `shadow.Options`. The production database is only
ever opened read-only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import tomllib
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from . import shadow
from .replay import TOKEN_ENV
from .replayset import DEFAULT_JOB_GB, DEFAULT_MIN_FREE_GB

TABLE_KEYS = (
    "root",
    "skills",
    "sample_rate",
    "max_age_hours",
    "concurrency",
    "max_heads_per_day",
    "guard_cmd",
    "stop_on_guard",
)


def _table(config: str | None) -> dict[str, Any]:
    """config.toml's `[shadow]` table ({} without one)."""
    from .config import DEFAULT_CONFIG_PATH

    path = Path(config).expanduser() if config else DEFAULT_CONFIG_PATH
    try:
        t = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    table = t.get("shadow") or {}
    return {k: table[k] for k in TABLE_KEYS if k in table} if isinstance(table, dict) else {}


def _pick(args: argparse.Namespace, table: dict[str, Any], key: str, default: Any) -> Any:
    value = getattr(args, key, None)
    if value is not None:
        return value
    return table.get(key, default)


def _guard(cmd: Any) -> str:
    """The guard command; empty or `none` is no guard."""
    cmd = str(cmd or "").strip()
    return "" if cmd.lower() == "none" else cmd


def root_of(args: argparse.Namespace, table: dict[str, Any] | None = None) -> Path:
    """--root, else REVIEWSYS_SHADOW_ROOT (what the status export and doctor read), else
    `[shadow] root`, else ~/.reviewsys-shadow."""
    if args.root:
        return Path(args.root).expanduser()
    if os.environ.get(shadow.ROOT_ENV):
        return Path(os.environ[shadow.ROOT_ENV]).expanduser()
    table = _table(args.config) if table is None else table
    return Path(table.get("root") or shadow.DEFAULT_ROOT).expanduser()


def options(args: argparse.Namespace) -> shadow.Options:
    from . import config as cfg_mod

    cfg = cfg_mod.load(Path(args.config) if args.config else None)
    table = _table(args.config)
    skills = _pick(args, table, "skills", None)
    if not skills:
        raise SystemExit("shadow: --skills (or [shadow] skills) names the v12 skills checkout")
    skills_path = Path(skills).expanduser()
    if not (skills_path / "config.json").exists():
        raise SystemExit(f"shadow: {skills_path} has no config.json")
    return shadow.Options(
        root=root_of(args, table),
        skills=skills_path.resolve(),
        prod_db=cfg.db_path,
        config=args.config,
        sample_rate=float(_pick(args, table, "sample_rate", 1.0)),
        max_age_hours=float(_pick(args, table, "max_age_hours", 24.0)),
        concurrency=int(_pick(args, table, "concurrency", 2)),
        max_heads_per_day=int(_pick(args, table, "max_heads_per_day", 40)),
        guard_cmd=_guard(_pick(args, table, "guard_cmd", "")),
        stop_on_guard=bool(_pick(args, table, "stop_on_guard", False)),
        min_usable_accounts=args.min_usable_accounts,
        backlog_limit=cfg.backlog_skip_phase1_above,
        min_free_gb=args.min_free_gb,
        disk_per_job_gb=args.disk_per_job_gb,
        replay_timeout_hours=args.replay_timeout_hours,
        gh_token_env=args.gh_token_env,
        run_timeout_minutes=cfg.run_timeout_minutes,
        fake_lanes=args.fake_lanes,
        tick_seconds=args.tick_seconds,
        score_every_seconds=args.score_every_minutes * 60,
        keep_artifacts_days=args.keep_artifacts_days,
    )


def _log_to(root: Path) -> None:
    """The runner logs to a rotated file only (stderr goes to an unrotated one on the box)."""
    (root / "logs").mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        root / "logs" / "shadow.log", maxBytes=10 * 1024 * 1024, backupCount=3
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().handlers[:] = [handler]


def cmd_run(args: argparse.Namespace) -> int:
    opts = options(args)
    runner = shadow.Runner(opts, shadow.connect(opts.db_path))
    runner.lock()
    _log_to(opts.root)
    stop = threading.Event()

    def _stop(signum: int, _frame: object) -> None:
        shadow.log.info("shadow: signal %s, stopping", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    wm = runner.set_watermark(args.since)
    shadow.log.info(
        "shadow: runner pid %s, root %s, skills %s, watermark %s, guard %s",
        os.getpid(), opts.root, opts.skills, wm, opts.guard_cmd or "none",
    )  # fmt: skip
    try:
        runner.run_forever(stop)
    finally:
        runner.unlock()
    return 0


def cmd_tick(args: argparse.Namespace) -> int:
    """One pass: discover, launch (unless --no-launch), and see the launched replays to the
    end. Holds the runner's lock: refused while `shadow run` is up."""
    opts = options(args)
    runner = shadow.Runner(opts, shadow.connect(opts.db_path))
    runner.lock(pidfile=False)
    try:
        runner.set_watermark(args.since)
        runner.recover()
        res: dict[str, Any] = {"discovered": runner.discover()}
        if not args.no_launch:
            res["launched"] = runner.launch()
            res["reaped"] = runner.wait_all()
        print(json.dumps(res, indent=1))
    except KeyboardInterrupt:
        runner.stop_all()
        raise
    finally:
        runner.unlock()
    return 0


def _shadow_db(args: argparse.Namespace) -> Path | None:
    db = root_of(args) / shadow.DB_NAME
    if not db.exists():
        print(f"no shadow database at {db}", file=sys.stderr)
        return None
    return db


def cmd_status(args: argparse.Namespace) -> int:
    db = _shadow_db(args)
    if db is None:
        return 1
    conn = shadow.connect_ro(db)
    try:
        state = shadow.runner_state(conn, db.parent)
        rows = shadow.status_rows(conn, args.limit)
    finally:
        conn.close()
    if args.json:
        print(json.dumps({"runner": state, "recent": rows}, indent=1))
        return 0
    print(
        f"runner: {'running pid ' + str(state['pid']) if state['running'] else 'not running'}; "
        f"state {state['state'] or '—'} {state['reason'] or ''} {state['detail'] or ''}".rstrip()
    )
    for r in rows:
        verdicts = (
            f" {r['prod_verdict'] or '—'} -> {r['shadow_verdict'] or '—'} "
            f"m{r['matched']}/p{r['prod_only']}/s{r['shadow_only']}"
            if r["status"] == "done"
            else ""
        )
        print(
            f"  run {r['source_run']} {r['repo']}#{r['number']} {str(r['sha'])[:8]} "
            f"{r['status']}{' (' + r['reason'] + ')' if r['reason'] else ''}{verdicts}"
        )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    db = _shadow_db(args)
    if db is None:
        return 1
    conn = shadow.connect_ro(db)
    try:
        s = shadow.summary(conn, since=args.since or "")
        state = shadow.runner_state(conn, db.parent)
    finally:
        conn.close()
    if args.json:
        print(json.dumps({"runner": state, **s}, indent=1))
    else:
        print(shadow.render(s, state), end="")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from . import config as cfg_mod

    db = _shadow_db(args)
    if db is None:
        return 1
    cfg = cfg_mod.load(Path(args.config) if args.config else None)
    conn = shadow.connect(db)
    try:
        print(json.dumps(shadow.score(conn, cfg.db_path, days=args.days)))
    finally:
        conn.close()
    return 0


def add_parser(sub: Any) -> None:
    s = sub.add_parser(
        "shadow",
        help="live shadow: replay finished production reviews under policy v12, compare",
    )
    ssub = s.add_subparsers(dest="action", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--root",
            help="shadow root (default $REVIEWSYS_SHADOW_ROOT, [shadow] root, ~/.reviewsys-shadow)",
        )

    def launching(p: argparse.ArgumentParser) -> None:
        common(p)
        p.add_argument("--skills", help="the v12 skills checkout")
        p.add_argument("--sample-rate", type=float, help="share of heads shadowed (1.0)")
        p.add_argument(
            "--max-age-hours", type=float, help="never shadow runs older than this (24; 0: any)"
        )
        p.add_argument("--concurrency", type=int, help="replays at once (2)")
        p.add_argument(
            "--max-heads-per-day", type=int, help="launches per UTC day, hard cap (40; 0: none)"
        )
        p.add_argument(
            "--since",
            help="shadow runs finished at or after this ISO time (default: from the first "
            "start); older than --max-age-hours still needs --max-age-hours 0",
        )
        p.add_argument(
            "--guard-cmd",
            help="shell command run before every launch; exit 0 allows it ([shadow] guard_cmd)",
        )
        p.add_argument(
            "--stop-on-guard",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="terminate running replays after two failed guard checks in a row (off)",
        )
        p.add_argument("--min-usable-accounts", type=int, default=2)
        p.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
        p.add_argument("--disk-per-job-gb", type=float, default=DEFAULT_JOB_GB)
        p.add_argument("--replay-timeout-hours", type=float, default=8.0)
        p.add_argument("--tick-seconds", type=float, default=60.0)
        p.add_argument("--score-every-minutes", type=float, default=60.0)
        p.add_argument(
            "--keep-artifacts-days",
            type=float,
            default=14.0,
            help="drop a finished replay's lane artifacts after this (report.json stays)",
        )
        p.add_argument("--gh-token-env", default=TOKEN_ENV)
        p.add_argument("--fake-lanes", action="store_true", help="smoke tests only")

    p = ssub.add_parser("run", help="the long-running loop (launchd)")
    launching(p)
    p.set_defaults(fn=cmd_run)
    p = ssub.add_parser("tick", help="one pass: discover, launch, wait for the launched replays")
    launching(p)
    p.add_argument("--no-launch", action="store_true", help="only discover")
    p.set_defaults(fn=cmd_tick)
    p = ssub.add_parser("status", help="runner state and the latest heads")
    common(p)
    p.add_argument("--json", action="store_true")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(fn=cmd_status)
    p = ssub.add_parser("report", help="v12 vs production: agreement, outcomes, cost, latency")
    common(p)
    p.add_argument("--json", action="store_true")
    p.add_argument("--since", help="only heads discovered at or after this ISO time")
    p.set_defaults(fn=cmd_report)
    p = ssub.add_parser("score", help="read what happened to shadowed PRs since (outcomes)")
    common(p)
    p.add_argument("--days", type=float, default=30.0, help="heads discovered in this window")
    p.set_defaults(fn=cmd_score)
