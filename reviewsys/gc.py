"""Garbage collection: stale worktrees, old run artifacts, old inbox/events."""

from __future__ import annotations

import shutil
import sqlite3
import time
from datetime import timedelta
from pathlib import Path

from .config import Config
from .db import event, now, parse_ts, tx


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def run(conn: sqlite3.Connection, cfg: Config) -> dict[str, int]:
    stats = {"worktrees_removed": 0, "runs_removed": 0, "events_pruned": 0, "inbox_pruned": 0}
    active_wts = {
        r["worktree"]
        for r in conn.execute(
            "SELECT worktree FROM runs WHERE status IN ('spawned','running') AND worktree IS NOT NULL"
        )
    }
    cutoff = parse_ts(now()) - timedelta(days=cfg.artifact_retention_days)
    # worktrees not tied to an active run and older than 24h
    if cfg.worktrees_dir.exists():
        for wt in sorted(cfg.worktrees_dir.iterdir(), key=lambda p: p.stat().st_mtime):
            if str(wt) in active_wts:
                continue
            age_h = (time.time() - wt.stat().st_mtime) / 3600
            if age_h > 24:
                shutil.rmtree(wt, ignore_errors=True)
                stats["worktrees_removed"] += 1
        # budget enforcement (LRU)
        budget = cfg.worktree_budget_gb * 1024**3
        entries = [p for p in cfg.worktrees_dir.iterdir() if str(p) not in active_wts]
        entries.sort(key=lambda p: p.stat().st_mtime)
        total = _dir_size(cfg.worktrees_dir)
        while total > budget and entries:
            victim = entries.pop(0)
            total -= _dir_size(victim)
            shutil.rmtree(victim, ignore_errors=True)
            stats["worktrees_removed"] += 1
    if cfg.runs_dir.exists():
        for rd in cfg.runs_dir.iterdir():
            try:
                if parse_ts(now()) - timedelta(seconds=time.time() - rd.stat().st_mtime) < cutoff:
                    shutil.rmtree(rd, ignore_errors=True)
                    stats["runs_removed"] += 1
            except OSError:
                pass
    with tx(conn):
        c = conn.execute(
            "DELETE FROM events WHERE ts < ?",
            ((parse_ts(now()) - timedelta(days=90)).isoformat().replace("+00:00", "Z"),),
        )
        stats["events_pruned"] = c.rowcount
        c = conn.execute(
            "DELETE FROM inbox WHERE handled_at IS NOT NULL AND seen_at < ?",
            ((parse_ts(now()) - timedelta(days=30)).isoformat().replace("+00:00", "Z"),),
        )
        stats["inbox_pruned"] = c.rowcount
        event(conn, "gc.run", detail=str(stats))
    return stats
