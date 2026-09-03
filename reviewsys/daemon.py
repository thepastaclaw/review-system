"""Single-threaded tick loop. Each tick runs whichever tasks are due."""

from __future__ import annotations

import logging
import signal
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass

from . import gc as gc_mod
from .config import Config
from .db import event, kv_get, kv_set, now, now_dt, parse_ts, tx
from .gh import Gh
from .ingest import ingest_notifications, ingest_prs
from .notify import Notifier
from .reaper import reap
from .router import route_inbox
from .scheduler import apply_supersedes, schedule
from .status import watchdog

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Task:
    name: str
    interval_s: int
    fn: Callable[[], object]
    last: float = 0.0


class Daemon:
    def __init__(
        self,
        cfg: Config,
        conn: sqlite3.Connection,
        *,
        gh: Gh | None = None,
        notifier: Notifier | None = None,
        spawn: bool = True,
    ) -> None:
        self.cfg, self.conn = cfg, conn
        self.gh = gh or Gh(cfg.gh_bin)
        self.notifier = notifier or Notifier(cfg)
        self.spawn = spawn
        self.stop = False
        self.tasks = [
            Task("ingest", cfg.ingest_interval_seconds, self.t_ingest),
            Task("notify", cfg.notify_interval_seconds, self.t_notify),
            Task("route", 30, self.t_route),
        ]
        if spawn:
            # shadow mode (spawn=False) observes ingest/routing only: no runs are created,
            # so nothing needs reaping, superseding or scheduling
            self.tasks += [
                Task("supersede", 0, self.t_supersede),
                Task("reap", 0, self.t_reap),
                Task("schedule", 0, self.t_schedule),
            ]
        self.tasks += [Task("watchdog", 300, self.t_watchdog), Task("gc", 3600, self.t_gc)]

    # ---- tasks ----
    def t_ingest(self) -> object:
        return ingest_prs(self.conn, self.cfg, self.gh)

    def t_notify(self) -> object:
        return ingest_notifications(self.conn, self.cfg, self.gh)

    def t_route(self) -> object:
        return route_inbox(self.conn, self.cfg, self.gh, self.notifier)

    def t_supersede(self) -> object:
        return apply_supersedes(self.conn, self.cfg)

    def t_reap(self) -> object:
        actions = reap(self.conn, self.cfg)
        for rid, action in actions:
            log.warning("reaper: run %s %s", rid, action)
        return actions

    def t_schedule(self) -> object:
        started = schedule(self.conn, self.cfg, spawn=self.spawn)
        # alert on heads that just exhausted retries
        rows = self.conn.execute(
            "SELECT id, repo, number, sha, reason FROM heads WHERE status='failed' AND finished_at > COALESCE((SELECT value FROM kv WHERE key='alert.failed_after'), '')"
        ).fetchall()
        for r in rows:
            self.notifier.alert(
                f"review of {r['repo']}#{r['number']} ({r['sha'][:8]}) failed after all retries: {(r['reason'] or '')[:200]}"
            )
        if rows:
            with tx(self.conn):
                kv_set(self.conn, "alert.failed_after", now())
        return started

    def t_watchdog(self) -> object:
        w = watchdog(self.conn, self.cfg)
        streak = int(kv_get(self.conn, "ingest.error_streak", "0") or 0)
        key = "alert.watchdog_at"
        last = kv_get(self.conn, key)
        recently = last is not None and (now_dt() - parse_ts(last)).total_seconds() < 3600
        stuck = w["stuck"] and self.spawn  # a shadow daemon never schedules, so "stuck" is expected
        if (stuck or w["ingest_stale"] or streak >= 3) and not recently:
            self.notifier.alert(
                f"watchdog: stuck={w['stuck']} eligible={w['eligible']} active={w['active']} ingest_stale={w['ingest_stale']} ingest_error_streak={streak}"
            )
            with tx(self.conn):
                kv_set(self.conn, key, now())
        return w

    def t_gc(self) -> object:
        return gc_mod.run(self.conn, self.cfg)

    # ---- loop ----
    def tick(self, *, force: bool = False) -> dict[str, object]:
        results: dict[str, object] = {}
        t = time.monotonic()
        for task in self.tasks:
            if not force and task.last and t - task.last < task.interval_s:
                continue
            try:
                results[task.name] = task.fn()
            except Exception as exc:
                log.exception("task %s failed", task.name)
                results[task.name] = f"error: {exc}"
                with tx(self.conn):
                    event(self.conn, f"daemon.task_error.{task.name}", detail=str(exc))
            task.last = time.monotonic()
        return results

    def run_forever(self) -> None:
        def _stop(*_: object) -> None:
            self.stop = True

        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
        with tx(self.conn):
            event(self.conn, "daemon.start")
        self.notifier.alert(
            "daemon started" + ("" if self.spawn else " in shadow mode (no reviews will run)")
        )
        try:
            while not self.stop:
                self.tick()
                time.sleep(self.cfg.tick_seconds)
        finally:
            with tx(self.conn):
                event(self.conn, "daemon.stop")
