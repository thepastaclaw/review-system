"""reviewsys command line."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from . import config as cfg_mod
from . import db as db_mod
from . import doctor as doctor_mod
from . import status as status_mod
from . import worker as worker_mod
from .daemon import Daemon
from .gh import Gh
from .ingest import enqueue_head
from .models import Trigger
from .notify import Notifier
from .scheduler import request_cancel


def _cfg(args: argparse.Namespace) -> cfg_mod.Config:
    return cfg_mod.load(Path(args.config) if args.config else None)


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
    res = Daemon(
        cfg, conn, spawn=not args.no_spawn, notifier=Notifier(cfg, wake_enabled=not args.no_wake)
    ).tick(force=True)
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
        w = snap["watchdog"]
        print(
            f"watchdog: stuck={w['stuck']} eligible={w['eligible']} ingest_stale={w['ingest_stale']} median_run_min={snap['median_run_minutes']}"
        )
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
    s = sub.add_parser("doctor", help="check gh, claude, proxy, models")
    s.add_argument("--no-models", action="store_true")
    s.set_defaults(fn=cmd_doctor)
    s = sub.add_parser("import-legacy", help="import done reviews from the old queue.json")
    s.add_argument("queue_json")
    s.set_defaults(fn=cmd_import_legacy)
    s = sub.add_parser("gc")
    s.set_defaults(fn=cmd_gc)
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
