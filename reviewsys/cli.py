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


def _path(value: str | None) -> Path | None:
    return Path(value).expanduser() if value else None


def _replay_cfg(args: argparse.Namespace) -> cfg_mod.Config:
    """The configured setup with `--replay-pool-slots POOL=N` on top (the eval's own cap on
    a pool, without editing the box's config.toml)."""
    cfg = _cfg(args)
    extra: dict[str, int] = {}
    for spec in args.replay_pool_slots or []:
        pool, sep, n = spec.partition("=")
        if not sep or not pool or not n.isdigit():
            raise SystemExit(f"--replay-pool-slots {spec!r}: expected POOL=N")
        extra[pool] = int(n)
    if not extra:
        return cfg
    return dataclasses.replace(cfg, replay_pool_slots={**cfg.replay_pool_slots, **extra})


def cmd_replay(args: argparse.Namespace) -> int:
    """Re-review past runs' exact heads under a skills checkout (replay.py); posts nothing."""
    from .replay import EXIT_ABORTED, TOKEN_VARS, ReplayAborted, replay

    cfg = _replay_cfg(args)
    token = os.environ.get(args.gh_token_env) or None
    # the token is the broker's alone: nothing this process starts inherits it
    for var in (*TOKEN_VARS, args.gh_token_env):
        os.environ.pop(var, None)
    out = Path(args.out).expanduser()
    lanes = None
    if args.fake_lanes:  # smoke tests only (deploy/eval-run.sh --dry): no model is called
        from .replayfake import FakeLanes

        lanes = FakeLanes()
    for rid in args.run_id:
        try:
            report = replay(
                cfg,
                run_id=rid,
                out=out,
                skills=_path(args.skills),
                source_db=_path(args.source_db),
                source_runs=_path(args.source_runs),
                force_phase2=args.force_phase2,
                pin_phase1_model=args.pin_phase1_model,
                sample=args.sample,
                policy_name=args.policy_name,
                gh_token=token,
                lane_runner=lanes,
                eval_mirrors=_path(args.eval_mirrors),
            )
        except ReplayAborted as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_ABORTED
        published = report["published"]
        print(
            json.dumps(
                {k: report[k] for k in ("source_run", "repo", "number", "status", "reason")}
                | {
                    "published": len(published),
                    "blockers": sum(f["severity"] == "blocking" for f in published),
                    "historical": len(report["historical_posted"]),
                    "forced_phase2": report["gate"]["forced_phase2"],
                    "gpt_lanes": report["gpt"]["lanes"],
                    "wall_clock_s": report["wall_clock_s"],
                }
            )
        )
        if report["invalid"]:
            print(f"run {rid}: replay invalid ({report['invalid']})", file=sys.stderr)
            # dry models stop a replay set; any other invalid replay is that replay's failure
            return EXIT_ABORTED if report["invalid"].startswith("quota") else 1
    return 0


def cmd_replay_set(args: argparse.Namespace) -> int:
    """Replay a set of past runs under several policies (replayset.py)."""
    from .replayset import main_set

    return main_set(args)


def cmd_replay_score(args: argparse.Namespace) -> int:
    """Score a replay set's reports against labels, arm against arm (replayscore.py)."""
    from .replayscore import main_score

    return main_score(args)


def cmd_replay_judge(args: argparse.Namespace) -> int:
    """The blinded judge lane over a replay set's unsettled findings (judge.py)."""
    from .judge import main_judge
    from .replay import TOKEN_VARS

    for var in TOKEN_VARS:  # the judge reads no GitHub: no token anywhere near its lanes
        os.environ.pop(var, None)
    lanes = None
    if args.fake_lanes:
        from .replayfake import FakeLanes

        lanes = FakeLanes()
    return main_judge(args, _replay_cfg(args), lane_runner=lanes)


def cmd_replay_fixture(args: argparse.Namespace) -> int:
    """Build the smoke-test world for `deploy/eval-run.sh --dry` (replayfake.py)."""
    from .replayfake import build_fixture

    print(json.dumps(build_fixture(Path(args.dir).expanduser())))
    return 0


def cmd_replay_preflight(args: argparse.Namespace) -> int:
    """Before an eval (and before its judge): production healthy (replayset.production_guard:
    not degraded, enough usable OpenAI accounts, a short queue) and the set's source runs and
    evidence on hand. Opens the databases read-only (never migrates the production one)."""
    from .replay import EXIT_ABORTED, open_source
    from .replayset import load_set, production_guard

    cfg = _cfg(args)
    problems: list[str] = []
    why, _ = production_guard(
        cfg.db_path,
        min_accounts=args.min_usable_accounts,
        backlog_limit=cfg.backlog_skip_phase1_above,
    )
    if why:
        problems.append(why)
    out: dict[str, object] = {"production": problems[:]}
    if args.set:
        source = open_source(_path(args.source_db) or cfg.db_path)
        runs_dir = _path(args.source_runs) or cfg.runs_dir
        missing_runs, missing_evidence = [], []
        try:
            for item in load_set(Path(args.set).expanduser()):
                row = source.execute("SELECT 1 FROM runs WHERE id=?", (item.run_id,)).fetchone()
                if row is None:
                    missing_runs.append(item.run_id)
                elif not (runs_dir / f"run-{item.run_id}" / "evidence.json").exists():
                    missing_evidence.append(item.run_id)
        finally:
            source.close()
        out |= {"missing_runs": missing_runs, "missing_evidence": missing_evidence}
        if missing_runs:
            problems.append(f"{len(missing_runs)} source runs are not in the source database")
    print(json.dumps(out | {"ok": not problems}, indent=1))
    if problems:
        print("preflight: " + "; ".join(problems), file=sys.stderr)
        return EXIT_ABORTED if out["production"] else 1
    return 0


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
        cw = snap["ci_watches"]
        print(f"ci watches: watching={cw.get('watching', 0)} ci_check_queued={cw.get('queued', 0)}")
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


def _pool_slots_arg(s: argparse.ArgumentParser) -> None:
    s.add_argument(
        "--replay-pool-slots",
        action="append",
        metavar="POOL=N",
        help="replay lanes hold at most N slots of POOL (over [scheduling] replay_pool_slots)",
    )


def _gate_args(s: argparse.ArgumentParser) -> None:
    """What replay-set and replay-judge check before each launch (replayset.Gate)."""
    s.add_argument(
        "--min-free-gb",
        type=float,
        default=35.0,
        help="free disk to keep under OUT, after --disk-per-job-gb per running job (35)",
    )
    s.add_argument(
        "--disk-per-job-gb",
        type=float,
        default=6.0,
        help="disk a running job may still take, counted before launching another (6)",
    )
    s.add_argument(
        "--disk-wait-minutes",
        type=float,
        default=30.0,
        help="with nothing running and the disk short, wait this long, then stop (30)",
    )
    s.add_argument(
        "--min-usable-accounts",
        type=int,
        default=2,
        help="stop launching below this many usable OpenAI accounts (2; 0: not checked)",
    )
    s.add_argument(
        "--busy-wait-minutes",
        type=float,
        default=60.0,
        help="with production's queue long and nothing running, wait this long, then stop (60)",
    )
    s.add_argument(
        "--no-production-guard",
        action="store_true",
        help="do not read the configured database before launches (tests only)",
    )


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
    s = sub.add_parser(
        "replay",
        help="re-review past runs' exact heads under a skills checkout; read-only, posts nothing",
    )
    s.add_argument("run_id", type=int, nargs="+")
    s.add_argument("--out", required=True, help="directory for scratch DBs, worktrees, reports")
    s.add_argument("--skills", help="skills checkout holding the policy to replay under")
    s.add_argument("--source-db", help="database the runs come from (default: the configured one)")
    s.add_argument(
        "--source-runs",
        help="run artifact dir holding the runs' evidence.json (default: work/runs)",
    )
    s.add_argument(
        "--force-phase2",
        action="store_true",
        help="record the gate's decision, then run Phase 2 anyway",
    )
    s.add_argument("--pin-phase1-model", help="run every Phase-1 lane on this model")
    s.add_argument("--sample", type=int, default=0, help="repeat sample number (dir run-<id>-r<k>)")
    s.add_argument("--policy-name", help="eval arm name (replay.policy_overrides)")
    s.add_argument(
        "--gh-token-env",
        default="REVIEWSYS_REPLAY_GH_TOKEN",
        help="env var holding a read-only GitHub token for every read the replay makes",
    )
    s.add_argument(
        "--fake-lanes",
        action="store_true",
        help="smoke tests only: canned lane answers, no model is called (replayfake.py)",
    )
    s.add_argument(
        "--eval-mirrors",
        help="where to mirror a repository the work dir has no mirror of (default OUT/.mirrors)",
    )
    _pool_slots_arg(s)
    s.set_defaults(fn=cmd_replay)
    s = sub.add_parser(
        "replay-set", help="replay a set of past runs under several policies (eval arms)"
    )
    s.add_argument("setfile", help="JSON/JSONL list of {run_id, set, samples}")
    s.add_argument(
        "--policy",
        action="append",
        required=True,
        help="NAME=SKILLS_DIR, one per arm (repeatable); NAME is a directory under --out",
    )
    s.add_argument("--out", required=True, help="reports go to OUT/<policy>/run-<id>[-r<k>]/")
    s.add_argument("--concurrency", type=int, default=2, help="replays at once (default 2)")
    s.add_argument("--force-phase2", action="store_true")
    s.add_argument("--pin-phase1-model")
    s.add_argument("--source-db")
    s.add_argument("--source-runs", help="run artifact dir holding the runs' evidence.json")
    s.add_argument(
        "--gh-token-env",
        default="REVIEWSYS_REPLAY_GH_TOKEN",
        help="env var holding a read-only GitHub token (passed to every replay)",
    )
    s.add_argument(
        "--retry-failed", action="store_true", help="also re-run failed and invalid replays"
    )
    s.add_argument(
        "--limit", type=int, help="only the first N heads, round-robin over the sets (smoke test)"
    )
    _gate_args(s)
    s.add_argument("--eval-mirrors", help="as replay --eval-mirrors (default OUT/.mirrors)")
    _pool_slots_arg(s)
    s.add_argument("--fake-lanes", action="store_true", help="smoke tests only (replay --help)")
    s.set_defaults(fn=cmd_replay_set)
    s = sub.add_parser("replay-score", help="score a replay set against labels, arm against arm")
    s.add_argument("setfile")
    s.add_argument("labels", help="JSON/JSONL labels (see replayscore.py)")
    s.add_argument("--out", required=True, help="the replay-set's --out; score.json goes there")
    s.add_argument("--baseline", help="arm the others are compared with (default: the first)")
    s.add_argument("--seed", type=int, default=0, help="shuffle seed for to_judge.jsonl")
    s.add_argument("--limit", type=int, help="score only replay-set --limit N's heads")
    s.set_defaults(fn=cmd_replay_score)
    s = sub.add_parser(
        "replay-judge",
        help="blinded judge lanes over a replay set's new blockers and unsettled label matches",
    )
    s.add_argument("out_dir", help="the replay-set's --out; judged.jsonl goes there")
    s.add_argument("--set", required=True, help="the set file (as replay-score's SETFILE)")
    s.add_argument("--labels", required=True, help="the labels (as replay-score's LABELS)")
    s.add_argument("--model", default="gpt-6.1-sol", help="judge model (default gpt-6.1-sol)")
    s.add_argument("--effort", default="xhigh", help="judge effort (default xhigh)")
    s.add_argument("--concurrency", type=int, default=3, help="heads judged at once (default 3)")
    s.add_argument("--baseline", help="as replay-score")
    s.add_argument("--seed", type=int, default=0, help="item order and to_judge shuffle seed")
    s.add_argument("--limit", type=int, help="as replay-set --limit")
    _gate_args(s)
    s.add_argument("--work-dir", help="worktrees and lane artifacts (default OUT.judge-work)")
    s.add_argument("--eval-mirrors", help="as replay-set (default OUT/.mirrors)")
    _pool_slots_arg(s)
    s.add_argument("--fake-lanes", action="store_true", help="smoke tests only (replay --help)")
    s.set_defaults(fn=cmd_replay_judge)
    s = sub.add_parser(
        "replay-preflight", help="before an eval: production healthy, source runs on hand"
    )
    s.add_argument(
        "--min-usable-accounts", type=int, default=2, help="as replay-set (0: not checked)"
    )
    s.add_argument("--set", help="check the set's source runs and their evidence.json")
    s.add_argument("--source-db", help="database the runs come from (default: the configured)")
    s.add_argument("--source-runs", help="run artifact dir (default: work/runs)")
    s.set_defaults(fn=cmd_replay_preflight)
    s = sub.add_parser("replay-fixture", help="smoke-test world for deploy/eval-run.sh --dry")
    s.add_argument("dir")
    s.set_defaults(fn=cmd_replay_fixture)
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
    from .shadowcli import add_parser as add_shadow_parser

    add_shadow_parser(sub)
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
