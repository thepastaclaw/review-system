"""reviewsys command line."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sqlite3
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from . import audit as audit_mod
from . import compare as compare_mod
from . import config as cfg_mod
from . import db as db_mod
from . import degraded as degraded_mod
from . import doctor as doctor_mod
from . import exporter as exporter_mod
from . import ledger as ledger_mod
from . import quality as quality_mod
from . import slots as slots_mod
from . import status as status_mod
from . import worker as worker_mod
from .daemon import Daemon, acquire_singleton_lock
from .gh import Gh
from .ingest import enqueue_head
from .models import Trigger
from .notify import Notifier
from .scheduler import request_cancel


def _cfg(args: argparse.Namespace) -> cfg_mod.Config:
    cfg = cfg_mod.load(Path(args.config) if args.config else None)
    if cfg.lane_claude_config_dir is not None:
        # lanes inherit it, and lane.claude_projects_dir reads it: one place for both
        os.environ["CLAUDE_CONFIG_DIR"] = str(cfg.lane_claude_config_dir)
    return cfg


def cmd_daemon(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    conn = db_mod.connect(cfg.db_path)
    Daemon(
        cfg, conn, spawn=not args.no_spawn, notifier=Notifier(cfg, wake_enabled=not args.no_wake)
    ).run_forever()
    return 0


def cmd_tick(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    conn = db_mod.connect(cfg.db_path)
    if not args.no_spawn:
        # A scheduling tick alongside the live daemon would be a second scheduler with its own
        # in-memory run accounting, so the two together could start a head twice or go past
        # `max_runs`.
        # `--no-spawn` builds a Daemon without the schedule task and is safe to run any time.
        acquire_singleton_lock(cfg.db_path.with_suffix(".daemon.lock"))
    d = Daemon(
        cfg, conn, spawn=not args.no_spawn, notifier=Notifier(cfg, wake_enabled=not args.no_wake)
    )
    if args.no_spawn:
        # without the daemon lock, never race the live daemon over the proxy's accounts
        d.tasks = [t for t in d.tasks if t.name != "accounts"]
    res = d.tick(force=True)
    print(json.dumps(res, default=str, indent=1))
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    conn = db_mod.connect(cfg.db_path)
    status = worker_mod.main(cfg, conn, args.run_id, dry_run=args.dry_run)
    print(status.value)
    return 0 if status.value == "done" else 1


def cmd_status(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    conn = db_mod.connect(cfg.db_path)
    snap = status_mod.snapshot(conn, cfg)
    if args.json:
        print(json.dumps(snap, default=str, indent=1))
    else:
        h = snap["heads"]
        print(
            f"heads: queued={h.get('queued', 0)} running={h.get('running', 0)} done={h.get('done', 0)} failed={h.get('failed', 0)} superseded={h.get('superseded', 0)} closed={h.get('closed', 0)}"
        )
        for a in snap["active"]:
            print(
                f"  run {a['id']} {a['repo']}#{a['number']} {a['sha'][:8]} phase={a['phase']} pid={a['pid']} hb={a['heartbeat_at']}"
            )
        print(status_mod.slots_summary(cfg, slots_mod.capacity(conn, cfg)))
        w = snap["watchdog"]
        print(
            f"watchdog: stuck={w['stuck']} eligible={w['eligible']} ingest_stale={w['ingest_stale']} median_run_min={snap['median_run_minutes']}"
        )
        d = snap["degraded"]
        if d.get("configured"):
            mode = "DEGRADED" if d["active"] else "normal"
            probe = d.get("last_probe") or {}
            print(
                f"mode: {mode} forced={d.get('forced') or 'auto'} sentinel={d['sentinel']} "
                f"last_probe={probe.get('at') or 'never'} ({probe.get('reason') or '-'})"
            )
    return 0


def cmd_degraded(args: argparse.Namespace) -> int:
    """Show or force the degraded mode: `on` / `off` / `auto` (probe the sentinel)."""
    cfg = _cfg(args)
    conn = db_mod.connect(cfg.db_path)
    if cfg.policy.degraded is None:
        print("no `degraded` block in the skills policy; nothing to switch", file=sys.stderr)
        return 1
    if args.mode:
        degraded_mod.force(conn, args.mode)
        state = degraded_mod.detect(conn, cfg, refresh=True)
        with db_mod.tx(conn):
            db_mod.event(
                conn, "degraded.forced", detail=f"{args.mode} by operator -> {state.describe()}"
            )
        print(f"degraded mode set to {args.mode}: now {state.describe()}")
        return 0
    state = degraded_mod.detect(conn, cfg, refresh=bool(args.probe))
    print(json.dumps({**state.as_dict(), **degraded_mod.snapshot(conn, cfg)}, indent=1))
    return 0


def cmd_enqueue(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    conn = db_mod.connect(cfg.db_path)
    gh = Gh(cfg.gh_bin)
    d = gh.api(f"repos/{args.repo}/pulls/{args.number}")
    sha = str((d or {}).get("head", {}).get("sha") or "")
    if not sha:
        print("could not resolve head sha", file=sys.stderr)
        return 1
    with db_mod.tx(conn):
        res = enqueue_head(conn, cfg, args.repo, args.number, sha, Trigger.MANUAL)
    print(f"{res} {args.repo}#{args.number}@{sha[:8]}")
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    conn = db_mod.connect(cfg.db_path)
    active = conn.execute(
        "SELECT 1 FROM runs WHERE id=? AND status IN ('spawned','running')", (args.run_id,)
    ).fetchone()
    if active is None:
        print(f"run {args.run_id} is not active", file=sys.stderr)
        return 1
    request_cancel(conn, args.run_id, args.reason)
    print(f"cancel requested for run {args.run_id}")
    return 0


def cmd_retry(args: argparse.Namespace) -> int:
    """Re-queue the most recent failed head of a PR with a fresh attempt counter."""
    cfg = _cfg(args)
    conn = db_mod.connect(cfg.db_path)
    with db_mod.tx(conn):
        live = conn.execute(
            "SELECT 1 FROM heads WHERE repo=? AND number=? AND status IN ('queued','running')",
            (args.repo, args.number),
        ).fetchone()
        if live:
            print(f"{args.repo}#{args.number} already has a queued/running head", file=sys.stderr)
            return 1
        head = conn.execute(
            "SELECT id, sha FROM heads WHERE repo=? AND number=? AND status='failed' ORDER BY id DESC LIMIT 1",
            (args.repo, args.number),
        ).fetchone()
        if head is None:
            print(f"no failed head for {args.repo}#{args.number}", file=sys.stderr)
            return 1
        pr = conn.execute(
            "SELECT head_sha FROM prs WHERE repo=? AND number=?", (args.repo, args.number)
        ).fetchone()
        if pr and pr["head_sha"] != head["sha"]:
            print(
                f"failed head {head['sha'][:8]} is not the PR's current head {pr['head_sha'][:8]}; "
                "use `enqueue` instead",
                file=sys.stderr,
            )
            return 1
        conn.execute(
            "UPDATE heads SET status='queued', attempts=0, eligible_at=?, finished_at=NULL, reason=NULL WHERE id=?",
            (db_mod.now(), head["id"]),
        )
        db_mod.event(conn, "head.retried", repo=args.repo, number=args.number, detail="operator")
    print(f"requeued {args.repo}#{args.number}@{head['sha'][:8]}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    ok = doctor_mod.run(cfg, probe_models=not args.no_models)
    return 0 if ok else 1


def cmd_import_legacy(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    conn = db_mod.connect(cfg.db_path)
    from .legacy_import import import_queue

    n = import_queue(conn, Path(args.queue_json))
    print(f"imported {n} done rows")
    return 0


def cmd_gc(args: argparse.Namespace) -> int:
    from . import gc as gc_mod

    cfg = _cfg(args)
    conn = db_mod.connect(cfg.db_path)
    print(json.dumps(gc_mod.run(conn, cfg)))
    return 0


def cmd_export_status(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    conn = db_mod.connect(cfg.db_path)
    path = exporter_mod.write_export(conn, cfg, Path(args.output))
    print(path)
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """The audit queue: backfill, progress, concurrency, per-PR reports, retries."""
    cfg = _cfg(args)
    conn = db_mod.connect(cfg.db_path)
    if args.action == "seed":
        if not args.since:
            print("audit seed needs --since YYYY-MM-DD", file=sys.stderr)
            return 1
        rows = audit_mod.seed(
            conn,
            cfg,
            Gh(cfg.gh_bin),
            since=args.since,
            repos=tuple(args.repo or ()),
            first_mergers=tuple(args.first or ()),
            dry_run=args.dry_run,
        )
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["result"]] = counts.get(r["result"], 0) + 1
        if args.json:
            print(json.dumps(rows, indent=1))
        print(json.dumps({"total": len(rows), **counts}))
        return 0
    if args.action == "status":
        print(json.dumps(audit_mod.summary(conn, cfg), indent=1))
        return 0
    if args.action == "concurrency":
        if args.value is None:
            print(audit_mod.concurrency(conn, cfg))
            return 0
        with db_mod.tx(conn):
            if args.value < 0:
                conn.execute("DELETE FROM kv WHERE key=?", (audit_mod.KV_CONCURRENCY,))
            else:
                db_mod.kv_set(conn, audit_mod.KV_CONCURRENCY, str(args.value))
        print(audit_mod.concurrency(conn, cfg))
        return 0
    if args.action == "report":
        if not args.target:
            print(audit_mod.render_index(conn), end="")
            return 0
        repo, _, number = args.target.partition("#")
        path = audit_mod.local_report(cfg, repo, int(number))
        if not path.exists():
            print(f"no report for {args.target} yet", file=sys.stderr)
            return 1
        print(path.read_text(), end="")
        return 0
    if args.action == "retry":
        with db_mod.tx(conn):
            n = conn.execute(
                "UPDATE heads SET status='queued', attempts=0, eligible_at=?, finished_at=NULL, "
                "reason=NULL WHERE queue='audit' AND status='failed'",
                (db_mod.now(),),
            ).rowcount
            db_mod.event(conn, "audit.retried", detail=f"{n} failed audit heads re-queued")
        print(f"re-queued {n} failed audit heads")
        return 0
    return 1


def cmd_compare(args: argparse.Namespace) -> int:
    """How the comparison model did against the primary on sampled runs (compare.py)."""
    cfg = _cfg(args)
    conn = db_mod.connect(cfg.db_path)
    runs = compare_mod.compare_runs(conn, since=args.since)
    if args.json:
        print(json.dumps(compare_mod.summarize(runs), indent=1))
    else:
        print(compare_mod.render(runs), end="")
    return 0


def _conn(args: argparse.Namespace) -> sqlite3.Connection | None:
    """`--db` when given (a copy of the database, say), else the configured one. None (said on
    stderr) for a `--db` that does not exist: never create an empty database by mistake."""
    if not args.db:
        return db_mod.connect(_cfg(args).db_path)
    path = Path(args.db).expanduser()
    if not path.is_file():
        print(f"no database at {path}", file=sys.stderr)
        return None
    return db_mod.connect(path)


def cmd_ledger(args: argparse.Namespace) -> int:
    """The issue ledger: replay history into it, reconcile it for policy v12, or show a PR's
    issues."""
    if args.action in {"backfill", "reconcile"} and args.number is not None and not args.repo:
        print("--number needs --repo", file=sys.stderr)
        return 1
    conn = _conn(args)
    if conn is None:
        return 1
    if args.action == "backfill":
        res: dict[str, object] = ledger_mod.backfill(conn, repo=args.repo, number=args.number)
        if args.github:
            cfg = _cfg(args)
            res["threads"] = ledger_mod.backfill_threads(
                conn, Gh(cfg.gh_bin), cfg.bot_login, repo=args.repo, number=args.number
            )
        print(json.dumps(res, indent=1))
        return 0
    if args.action == "reconcile":
        cfg = _cfg(args)

        def mirror_for(repo: str) -> Path | None:
            p = cfg.mirrors_dir / f"{repo.replace('/', '__')}.git"
            return p if (p / "HEAD").exists() else None

        out = ledger_mod.reconcile_all(
            conn,
            Gh(cfg.gh_bin),
            cfg.bot_login,
            mirror_for=mirror_for,
            repo=args.repo,
            number=args.number,
        )
        print(json.dumps(out, indent=1))
        return 0
    issues = ledger_mod.load(conn, args.repo, args.number)
    events = ledger_mod.history(conn, args.repo, args.number) if args.history else None
    if args.json:
        rows = [dataclasses.asdict(i) for i in issues.values()]
        print(json.dumps({"issues": rows, "events": events}, indent=1))
    else:
        print(ledger_mod.render(issues, events), end="")
    return 0


def cmd_quality(args: argparse.Namespace) -> int:
    """Production review-quality metrics from the ledger and the repo mirrors (quality.py)."""
    conn = _conn(args)
    if conn is None:
        return 1
    mirrors = Path(args.mirrors).expanduser() if args.mirrors else None
    if mirrors is None and not args.db:
        mirrors = _cfg(args).mirrors_dir
    overrides: dict[str, Path] = {}
    for spec in args.repo_path or []:
        repo, sep, path = spec.partition("=")
        if not sep or "/" not in repo:
            print(f"--repo-path wants OWNER/NAME=PATH, got {spec!r}", file=sys.stderr)
            return 1
        overrides[repo] = Path(path).expanduser()
    rep = quality_mod.report(
        conn, since=args.since, paths=quality_mod.RepoPaths(mirrors, overrides)
    )
    if args.json:
        print(json.dumps(rep, indent=1))
    else:
        print(quality_mod.render(rep), end="")
    return 0


def cmd_init_config(args: argparse.Namespace) -> int:
    path = Path(args.config) if args.config else cfg_mod.DEFAULT_CONFIG_PATH
    if path.exists() and not args.force:
        print(f"{path} exists (use --force)", file=sys.stderr)
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cfg_mod.DEFAULT_TOML)
    print(f"wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reviewsys")
    p.add_argument("--config", help="path to config.toml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("daemon", help="run the tick loop forever")
    s.add_argument(
        "--no-spawn",
        action="store_true",
        help="shadow mode: ingest and route only, never start reviews",
    )
    s.add_argument(
        "--no-wake", action="store_true", help="never wake the OpenClaw agent (shadow mode)"
    )
    s.set_defaults(fn=cmd_daemon)
    s = sub.add_parser("tick", help="run one tick of every task and exit")
    s.add_argument("--no-spawn", action="store_true")
    s.add_argument("--no-wake", action="store_true")
    s.set_defaults(fn=cmd_tick)
    s = sub.add_parser("worker", help="execute one run")
    s.add_argument("--run-id", type=int, required=True)
    s.add_argument("--dry-run", action="store_true", help="do everything except post to GitHub")
    s.set_defaults(fn=cmd_worker)
    s = sub.add_parser("status")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_status)
    s = sub.add_parser("enqueue", help="queue a PR for priority review")
    s.add_argument("repo")
    s.add_argument("number", type=int)
    s.set_defaults(fn=cmd_enqueue)
    s = sub.add_parser("cancel", help="request cancellation of an active run")
    s.add_argument("--run-id", type=int, required=True)
    s.add_argument("--reason", default="operator")
    s.set_defaults(fn=cmd_cancel)
    s = sub.add_parser("retry", help="re-queue the failed head of a PR")
    s.add_argument("repo")
    s.add_argument("number", type=int)
    s.set_defaults(fn=cmd_retry)
    s = sub.add_parser("degraded", help="show or force the degraded (stand-in models) mode")
    s.add_argument("mode", nargs="?", choices=["on", "off", "auto"])
    s.add_argument("--probe", action="store_true", help="re-probe the sentinel model now")
    s.set_defaults(fn=cmd_degraded)
    s = sub.add_parser("doctor", help="check gh, claude, proxy, models")
    s.add_argument("--no-models", action="store_true")
    s.set_defaults(fn=cmd_doctor)
    s = sub.add_parser("import-legacy", help="import done reviews from the old queue.json")
    s.add_argument("queue_json")
    s.set_defaults(fn=cmd_import_legacy)
    s = sub.add_parser("gc")
    s.set_defaults(fn=cmd_gc)
    s = sub.add_parser("export-status", help="write a sanitized public observability snapshot")
    s.add_argument("--output", required=True, help="directory receiving status.json")
    s.set_defaults(fn=cmd_export_status)
    s = sub.add_parser("compare", help="primary vs comparison model on sampled runs")
    s.add_argument("--since", help="only runs started at or after this ISO timestamp")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_compare)
    s = sub.add_parser("audit", help="post-merge audit queue: seed, status, concurrency, report")
    s.add_argument("action", choices=["seed", "status", "concurrency", "report", "retry"])
    s.add_argument("target", nargs="?", help="report: owner/repo#N (omit for the index)")
    s.add_argument("--since", help="seed: merged on or after this date (YYYY-MM-DD)")
    s.add_argument("--repo", action="append", help="seed: only this repo (repeatable)")
    s.add_argument(
        "--first", action="append", help="seed: audit PRs merged by this login first (repeatable)"
    )
    s.add_argument("--dry-run", action="store_true", help="seed: classify only, queue nothing")
    s.add_argument("--json", action="store_true", help="seed: print every row")
    s.add_argument(
        "--value", type=int, help="concurrency: audit runs at once (-1 = back to the config value)"
    )
    s.set_defaults(fn=cmd_audit)
    s = sub.add_parser("ledger", help="the issue ledger: backfill history, show a PR's issues")
    lsub = s.add_subparsers(dest="action", required=True)
    b = lsub.add_parser("backfill", help="replay recorded history into the ledger (idempotent)")
    b.add_argument("--repo", help="only this repo (owner/name)")
    b.add_argument("--number", type=int, help="only this PR (with --repo)")
    b.add_argument(
        "--github", action="store_true", help="also sync open/resolved threads from GitHub"
    )
    b.add_argument("--db", help="database file (default: the configured one)")
    b.set_defaults(fn=cmd_ledger)
    rc = lsub.add_parser(
        "reconcile",
        help="policy v12: outdate standing issues that can no longer stand (needs GitHub)",
    )
    rc.add_argument("--repo", help="only this repo (owner/name)")
    rc.add_argument("--number", type=int, help="only this PR (with --repo)")
    rc.add_argument("--db", help="database file (default: the configured one)")
    rc.set_defaults(fn=cmd_ledger)
    v = lsub.add_parser("show", help="a PR's issues")
    v.add_argument("repo")
    v.add_argument("number", type=int)
    v.add_argument("--history", action="store_true", help="also every event, in fold order")
    v.add_argument("--json", action="store_true")
    v.add_argument("--db", help="database file (default: the configured one)")
    v.set_defaults(fn=cmd_ledger)
    s = sub.add_parser("quality", help="review-quality metrics: late finds, fates, re-raises")
    s.add_argument("--since", help="only findings posted (runs started) at or after this date")
    s.add_argument("--json", action="store_true")
    s.add_argument("--db", help="database file (default: the configured one)")
    s.add_argument(
        "--mirrors",
        help="directory of bare mirrors (default: the configured one, or none with --db)",
    )
    s.add_argument(
        "--repo-path",
        action="append",
        help="OWNER/NAME=PATH: a clone to diff in for that repo (repeatable)",
    )
    s.set_defaults(fn=cmd_quality)
    s = sub.add_parser("init-config")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_init_config)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_file = os.environ.get("REVIEWSYS_LOG_FILE")
    handlers: list[logging.Handler] | None = None
    if log_file and args.cmd == "daemon":
        handlers = [RotatingFileHandler(log_file, maxBytes=20 * 1024 * 1024, backupCount=5)]
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
    fn = args.fn
    return int(fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
