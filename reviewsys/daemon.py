"""Single-threaded tick loop. Each tick runs whichever tasks are due.

The one other thread is the watch (watch.py), which touches no database: it only says which
passes to run now and wakes the loop."""

from __future__ import annotations

import fcntl
import logging
import os
import signal
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from . import audit, checks, ci, converse, degraded, hook, labels, slots, watch
from . import gc as gc_mod
from .config import Config
from .db import event, kv_get, kv_set, now, now_dt, parse_ts, tx
from .gh import Gh
from .ingest import ingest_notifications, ingest_pr_comments, ingest_prs, ingest_review_replies
from .models import ReviewError
from .notify import Notifier
from .queue_status import update_queue_comments
from .reaper import reap
from .router import route_inbox
from .scheduler import apply_supersedes, schedule
from .status import watchdog

log = logging.getLogger(__name__)

# at most one @-mention page per this interval: while degraded it re-pages this often until
# someone fixes it, and a probe flapping in and out of the mode cannot page more often
DEGRADED_REPAGE_S = 3600
KV_PAGED_AT = "degraded.paged_at"  # last delivered @-mention page
KV_PAGE_OPEN = "degraded.page_open"  # "1" while #claw has a page with no all-clear yet
AUDIT_SWEEP_S = 600  # look for merges this often (GitHub search, one query per repo)
AUDIT_INDEX_S = 900  # regenerate the audit index at most this often
SLOW_TASK_S = 10  # a task that holds the tick loop longer is logged (and delays watch hits)


def acquire_singleton_lock(path: Path) -> IO[str]:
    """Hold an exclusive flock for the daemon's lifetime; a second daemon exits immediately.

    Guards against launchd starting a twin of a manually started daemon (or vice versa).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = open(path, "w")  # noqa: SIM115 - kept open for the process lifetime
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fd.close()
        log.error("another reviewsys daemon already holds %s; exiting", path)
        raise SystemExit(3) from None
    fd.write(str(os.getpid()))
    fd.flush()
    return fd


@dataclass(slots=True)
class Task:
    name: str
    interval_s: int
    fn: Callable[[], object]
    last: float = 0.0
    rerun: bool = False  # due again at once (a pass that gave way to a hit)


class Daemon:
    def __init__(
        self,
        cfg: Config,
        conn: sqlite3.Connection,
        *,
        gh: Gh | None = None,
        notifier: Notifier | None = None,
        spawn: bool = True,
        prober: degraded.Prober | None = None,
    ) -> None:
        self.cfg, self.conn = cfg, conn
        self.gh = gh or Gh(cfg.gh_bin)
        self.notifier = notifier or Notifier(cfg)
        self.spawn = spawn
        self.prober = prober or degraded.probe
        self.labels_repos: dict[str, bool] = {}  # repo -> has created the pastaclaw:* labels
        self.review_now_checked: set[str] = set()  # repos whose review-now label was checked
        if spawn:  # the first scheduling start with the tracking rule fixes when it took effect
            converse.tracking_required_since(conn, cfg.deferral.tracking_required_since)
        self.stop = False
        self.wake = threading.Event()
        self._due: set[str] = set()
        self._due_from: set[str] = set()  # "watch" / "webhook": who asked, for the event
        self._queue_yielded = False  # the last gate-comment pass gave way to a hit
        self._due_lock = threading.Lock()
        self.watcher = (
            watch.Watcher(Gh(cfg.gh_bin), cfg.bot_login, cfg.enabled_repos)
            if spawn and cfg.watch_interval_seconds
            else None
        )
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
                Task("queue_comments", cfg.queue_comment_interval_seconds, self.t_queue_comments),
                Task("labels", 60, self.t_labels),
            ]
            if cfg.checks.enabled:
                self.checks = checks.Reconciler(cfg, checks.app_auth(cfg), review_gh=self.gh)
                self.tasks.append(Task("checks", cfg.checks.interval_seconds, self.t_checks))
            if cfg.ci.enabled:
                self.tasks.append(Task("ci_watch", cfg.ci.interval_seconds, self.t_ci_watch))
            if cfg.audit.enabled:
                self.tasks += [
                    Task("audit_sweep", AUDIT_SWEEP_S, self.t_audit_sweep),
                    Task("audit_index", AUDIT_INDEX_S, self.t_audit_index),
                ]
        # the reserves protect accounts other clients share, so they are enforced even in
        # shadow mode; slot scaling only matters to a daemon that schedules
        if cfg.account_reserves or (spawn and cfg.account_scale_max > 1):
            self.tasks.append(Task("accounts", 120, self.t_accounts))
        if cfg.policy.degraded is not None:
            self.tasks.append(Task("degraded", 120, self.t_degraded))
        self.tasks += [Task("watchdog", 300, self.t_watchdog), Task("gc", 3600, self.t_gc)]

    # ---- tasks ----
    def t_ingest(self) -> object:
        stats: dict[str, object] = dict(ingest_prs(self.conn, self.cfg, self.gh))
        if not self.spawn:  # a no-spawn or shadow daemon writes nothing to GitHub
            return stats
        for repo in self.cfg.enabled_repos:
            if repo not in self.review_now_checked:
                try:
                    labels.ensure_review_now_label(self.conn, self.gh, repo)
                    self.review_now_checked.add(repo)
                except Exception as exc:  # transient: tried again next pass
                    log.warning("review-now label check on %s failed: %s", repo, exc)
        released = labels.release_review_now(self.conn, self.gh)
        if released:
            stats["review_now_released"] = released
        return stats

    def t_notify(self) -> object:
        added = ingest_notifications(self.conn, self.cfg, self.gh)
        return {
            "notifications": added,
            "replies": ingest_review_replies(self.conn, self.cfg, self.gh),
            "comments": ingest_pr_comments(self.conn, self.cfg, self.gh),
        }

    def t_route(self) -> object:
        return route_inbox(self.conn, self.cfg, self.gh, self.notifier)

    def t_queue_comments(self) -> object:
        # a pass resumed after giving way runs to the end: hits less than a pass apart would
        # otherwise keep its later loops (ticked draft boxes, the comment writes) from running
        resumed, self._queue_yielded = self._queue_yielded, False
        stats = update_queue_comments(
            self.conn,
            self.cfg,
            self.gh,
            should_yield=(lambda: False) if resumed else self._hit_waiting,
        )
        if stats["yielded"]:
            self._queue_yielded = True
            for task in self.tasks:
                if task.name == "queue_comments":
                    task.rerun = True  # once the hit ran, not a full interval later
        return stats

    def _hit_waiting(self) -> bool:
        return bool(self._due)

    def t_labels(self) -> object:
        return labels.reconcile(self.conn, self.gh, repos=self.labels_repos)

    def t_checks(self) -> object:
        return self.checks.reconcile(self.conn)

    def t_ci_watch(self) -> object:
        return ci.watch(self.conn, self.cfg, self.gh)

    def t_audit_sweep(self) -> object:
        return audit.sweep(self.conn, self.cfg, self.gh)

    def t_audit_index(self) -> object:
        last = kv_get(self.conn, "audit.index_at") or ""
        newest = self.conn.execute("SELECT MAX(finished_at) FROM audits").fetchone()[0] or ""
        pending = kv_get(self.conn, "audit.index_pending") == "1"
        if newest and newest <= last and not pending:
            return "unchanged"
        try:
            res = audit.publish_index(self.conn, self.cfg, self.gh)
            ok = not res.get("refused")
        except ReviewError as exc:
            log.warning("audit index publish failed: %s", exc)
            res, ok = {"error": str(exc)}, False
        with tx(self.conn):
            kv_set(self.conn, "audit.index_at", now())
            kv_set(self.conn, "audit.index_pending", "0" if ok else "1")
        return res

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
            "SELECT id, repo, number, sha, reason FROM heads WHERE status='failed' AND queue='live' AND finished_at > COALESCE((SELECT value FROM kv WHERE key='alert.failed_after'), '')"
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

    def t_accounts(self) -> object:
        cap, changes = slots.refresh(self.conn, self.cfg)
        for action, email, detail in changes:
            # events are exported to the public status page: no account names there
            with tx(self.conn):
                event(self.conn, f"account.{action}", detail="reserved account: " + detail)
            self.notifier.alert(f"OpenAI account {email} {action} in the proxy: {detail}")
        return cap.as_dict()

    def t_degraded(self) -> object:
        """Re-probe the sentinel so the mode flips back on its own once quota returns.

        Only a person can add or re-enable an OpenAI account, so an outage the probe found
        pages #claw with `page_mentions` mentioned, and re-pages every `DEGRADED_REPAGE_S`
        until it clears. The same interval bounds a flapping probe: a flip back into the mode
        within it goes to the operator DM only. The all-clear goes to #claw only when #claw
        has an open page and the probe (not an operator override) says the models answer. A
        page that fails to deliver is retried on the next pass, not an hour later."""
        state = degraded.detect(self.conn, self.cfg, prober=self.prober, refresh=True)
        alert = degraded.transition(self.conn, state)
        if alert:
            with tx(self.conn):
                event(self.conn, "degraded.transition", detail=state.describe())
        page_open = kv_get(self.conn, KV_PAGE_OPEN) == "1"
        if not state.active:
            if alert and page_open and state.source == "probe":
                self.notifier.page(alert, resolved=True)
            elif alert:
                self.notifier.alert(alert)
            if page_open:
                with tx(self.conn):
                    self.conn.execute("DELETE FROM kv WHERE key=?", (KV_PAGE_OPEN,))
            return state.as_dict()
        if state.source == "forced":
            if alert:
                self.notifier.alert(alert)  # the operator who forced it already knows
            return state.as_dict()
        last = kv_get(self.conn, KV_PAGED_AT)
        due = last is None or (now_dt() - parse_ts(last)).total_seconds() >= DEGRADED_REPAGE_S
        if due:
            mode = "PAUSED" if state.pause else "in DEGRADED mode"
            text = (
                alert
                if alert or not page_open
                else f"STILL {mode} since {state.since or '?'}: {state.reason}"
            ) or f"{mode}: {state.reason}"
            if self.notifier.page(f"{text}\n{self._accounts_summary()}"):
                with tx(self.conn):
                    kv_set(self.conn, KV_PAGED_AT, now())
                    kv_set(self.conn, KV_PAGE_OPEN, "1")
        elif alert:
            self.notifier.alert(alert)  # flapping back in within the hour: no new @-mentions
        return state.as_dict()

    def _accounts_summary(self) -> str:
        """What a human needs to act on: which OpenAI accounts are out and until when."""
        stored = slots.stored_accounts(self.conn)
        if stored is None:
            return "OpenAI accounts: no reading from the proxy yet."
        at, accounts = stored
        lines = [f"OpenAI accounts (as of {at}):"]
        for a in accounts:
            back = f", back {a.reset_at}" if a.reset_at else ""
            lines.append(f"• {'ok  ' if a.usable else 'OUT '} {a.name}: {a.reason}{back}")
        lines.append("Fix: add a fresh account to the proxy or re-enable one with quota left.")
        return "\n".join(lines)

    # ---- loop ----
    def watch_round(self) -> set[str]:
        """One watch read (the watch thread's body): passes it finds due run before the tick's
        next task, and it wakes an idle loop."""
        assert self.watcher is not None
        try:
            due = self.watcher.poll()
        except Exception:
            log.exception("watch round failed")
            return set()
        if due:
            with self._due_lock:
                self._due |= due
                self._due_from.add("watch")
            self.wake.set()
        return due

    def webhook_hit(self, task: str, detail: str) -> None:
        """A signed delivery asked for `task` (the receiver's thread): as a watch hit."""
        with self._due_lock:
            self._due.add(task)
            self._due_from.add(f"webhook {detail}")
        self.wake.set()

    def start_webhook(self) -> hook.Receiver | None:
        """The receiver, when configured; one that cannot start leaves the polls and watch."""
        if not (self.spawn and self.cfg.webhook.enabled):
            return None
        try:
            receiver = hook.Receiver(
                self.cfg.webhook,
                hook.load_secret(self.cfg.webhook.secret_file),
                self.cfg.bot_login,
                self.cfg.enabled_repos,
                self.webhook_hit,
            )
            receiver.start()
        except (hook.HookError, OSError, OverflowError) as exc:
            log.error("webhook receiver not started: %s", exc)
            with tx(self.conn):
                event(self.conn, "webhook.failed", detail=str(exc))
            return None
        return receiver

    def run_watched(self, results: dict[str, object] | None = None) -> set[str]:
        """Run the passes the watch asked for now, then the scheduler (main thread). The tick
        calls it before each task, so a hit waits for the task in progress, not the tick."""
        with self._due_lock:
            due, self._due = self._due, set()
            sources, self._due_from = self._due_from, set()
        if not due:
            return due
        with tx(self.conn):
            detail = f"{','.join(sorted(due))} via {','.join(sorted(sources))}"
            event(self.conn, "watch.hit", detail=detail)
        results = results if results is not None else {}
        for name in (*(t.name for t in self.tasks if t.name in due), "schedule"):
            for task in self.tasks:
                if task.name == name and not self.stop:
                    self._run(task, results)
        return due

    def _watch_forever(self) -> None:
        while not self.stop:
            self.watch_round()
            time.sleep(self.cfg.watch_interval_seconds)

    def tick(self, *, force: bool = False) -> dict[str, object]:
        results: dict[str, object] = {}
        t = time.monotonic()
        for task in self.tasks:
            if self.stop:
                break
            if self._due:
                self.run_watched(results)
            # an every-tick task (interval 0) a hit ran early runs again in its place, so the
            # tick keeps supersede and reap before schedule
            if not force and task.interval_s and task.last and t - task.last < task.interval_s:
                continue
            self._run(task, results)
        return results

    def _run(self, task: Task, results: dict[str, object]) -> None:
        started = time.monotonic()
        try:
            results[task.name] = task.fn()
        except Exception as exc:
            log.exception("task %s failed", task.name)
            results[task.name] = f"error: {exc}"
            with tx(self.conn):
                event(self.conn, f"daemon.task_error.{task.name}", detail=str(exc))
        ended = task.last = time.monotonic()
        if task.rerun:
            task.rerun, task.last = False, 0.0
        if ended - started > SLOW_TASK_S:
            log.info("task %s took %.0f s", task.name, ended - started)

    def run_forever(self) -> None:
        def _stop(*_: object) -> None:
            self.stop = True  # the wait below ends within tick_seconds, as the sleep did

        self._lock = acquire_singleton_lock(self.cfg.db_path.with_suffix(".daemon.lock"))
        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
        with tx(self.conn):
            event(self.conn, "daemon.start")
        self.notifier.alert(
            "daemon started" + ("" if self.spawn else " in shadow mode (no reviews will run)")
        )
        if self.watcher is not None:
            threading.Thread(target=self._watch_forever, name="watch", daemon=True).start()
        self.start_webhook()
        try:
            while not self.stop:
                self.wake.clear()  # before the tick: a hit during it ends the wait
                self.tick()
                self.wake.wait(self.cfg.tick_seconds)
        finally:
            with tx(self.conn):
                event(self.conn, "daemon.stop")
