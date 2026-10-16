"""Sister repositories: the other Dash repos a change builds against or is driven by, checked
out read-only beside the PR worktree for the lanes (the `sister_repos` evidence).

Which repos: the git dependencies the PR's root Cargo.lock pins from an allowlisted GitHub owner
(`[context] sister_owners`), each at the pinned commit and at its default branch's tip, plus
the repo's `sister_repos` in the skills config.json at the default branch only. Why
(2026-10-16 lane audit): lanes accepted "key-wallet can't tell us" without reading
rust-dashcore, and broke Tenderdash re-proposals with no Tenderdash checkout to read.

Checkouts are shared across runs, one per (repo, commit), under `<worktrees>/sister/`. All
work on one repo (its mirror, its checkouts) happens under that repo's flock, gc's removal of
its checkouts included. A run records every checkout it uses (kv `sister.inuse:<run id>`)
under that lock before it touches it, and gc, under the same lock, leaves the paths of runs
still in flight alone (`remove_unused`). Everything here is best effort: a repo that cannot be
had carries an `error`, and the run goes on. Repo names are lowercased throughout (GitHub
ignores case; paths, locks and records must not differ by it).
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import shutil
import sqlite3
import time
import tomllib
import urllib.parse
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .db import kv_get, kv_set, tx
from .models import FailKind, ReviewError
from .steps import worktree as wt

SISTER_DIR = "sister"  # under the worktrees dir
INUSE_KEY = "sister.inuse:"  # + run id -> JSON list of the checkout paths the run uses
GIT_OP_TIMEOUT_SECONDS = 300  # a fetch or checkout; a first clone may use the whole budget
LOCK_WAIT_SECONDS = 120  # for a repo another run is working on, before going on without it
MAX_PINS = 2  # distinct pinned commits of one repo checked out per run
_NEW = ".new-"  # prefix of a checkout being made (moved into place once complete)

# `git+https://github.com/<owner>/<repo>(.git)?[?<query>]#<full sha>`
_SOURCE = re.compile(
    r"^git\+https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?"
    r"(?:\?([^#]*))?#([0-9a-f]{40})$"
)


@dataclass(frozen=True, slots=True)
class Pin:
    """One commit of a sister repo the Cargo.lock pins, and the packages taken from it."""

    sha: str
    ref: str  # the source's query as written (`tag=v1.5.1`, `branch=v0.42-dev`), "" for none
    packages: tuple[str, ...]


def cargo_pins(lock_text: str, *, owners: tuple[str, ...], exclude: str) -> dict[str, list[Pin]]:
    """`owner/repo` (lowercased) -> its distinct pinned commits, most packages first, for the
    git dependencies of a Cargo.lock whose owner is in `owners`, never `exclude` (the reviewed
    repo itself)."""
    allowed = {o.lower() for o in owners}
    found: dict[tuple[str, str], tuple[str, set[str]]] = {}  # (repo, sha) -> (ref, packages)
    for pkg in tomllib.loads(lock_text).get("package") or []:
        m = _SOURCE.match(str(pkg.get("source") or "")) if isinstance(pkg, dict) else None
        if not m:
            continue
        owner, name, query, sha = m.groups()
        repo = f"{owner}/{name}".lower()
        if owner.lower() in allowed and repo != exclude.lower():
            ref = urllib.parse.unquote(query or "")
            found.setdefault((repo, sha), (ref, set()))[1].add(str(pkg.get("name") or ""))
    pins: dict[str, list[Pin]] = {}
    for (repo, sha), (ref, pkgs) in found.items():
        pins.setdefault(repo, []).append(Pin(sha, ref, tuple(sorted(pkgs))))
    for repo_pins in pins.values():
        repo_pins.sort(key=lambda p: -len(p.packages))  # stable: ties keep Cargo.lock order
    return pins


def lock_pins(cfg: Config, repo: str, worktree: Path) -> dict[str, list[Pin]]:
    """The sister repos the PR's root Cargo.lock pins ({} without one). Raises when it cannot
    be read or parsed; a symlink is not followed out of the worktree."""
    lock = worktree / "Cargo.lock"
    if lock.is_symlink() or not lock.is_file():
        return {}
    return cargo_pins(lock.read_text(), owners=cfg.context.sister_owners, exclude=repo)


def wanted(
    configured: tuple[str, ...], pins: dict[str, list[Pin]], *, exclude: str
) -> list[tuple[str, list[Pin]]]:
    """Every sister repo in checkout order: the configured ones first (as listed; someone
    asked for them), then the pinned ones by packages pinned, most first."""
    out: list[tuple[str, list[Pin]]] = []
    seen = {exclude.lower()}
    for repo in (r.lower() for r in configured):
        if repo not in seen:
            seen.add(repo)
            out.append((repo, pins.get(repo, [])))
    rest = sorted(
        ((r, p) for r, p in pins.items() if r not in seen),
        key=lambda v: (-sum(len(p.packages) for p in v[1]), v[0]),
    )
    return out + rest


def gather(
    cfg: Config,
    conn: sqlite3.Connection,
    *,
    repo: str,
    run_id: int,
    pins: dict[str, list[Pin]],
    should_stop: Callable[[], bool] = lambda: False,
) -> list[dict[str, Any]]:
    """The `sister_repos` evidence for a run of `repo` (`pins`: see `lock_pins`), checking
    out what is missing. Each entry: `repo`, `packages` (pinned from it), `pinned` (`sha`,
    `ref`, `path`; null when the Cargo.lock does not pin it), `default` (`branch`, `sha`,
    `path`), and when something could not be had `error` (or `skipped`, over the per-run cap).
    A `path` is null when that checkout is missing. A repo pinned at several commits also has
    `also_pinned` (the second; each slot then lists its `packages`) and, past MAX_PINS, a
    `note`. Never raises; stops early (with what it has) once `should_stop()`."""
    cc = cfg.context
    if cc.sister_max_repos <= 0:
        return []
    rc = cfg.repo(repo)
    deadline = time.monotonic() + cc.sister_budget_minutes * 60
    entries: list[dict[str, Any]] = []
    for i, (name, repo_pins) in enumerate(
        wanted(rc.sister_repos if rc else (), pins, exclude=repo)
    ):
        if should_stop():
            break
        entry: dict[str, Any] = {
            "repo": name,
            "packages": sorted({p for pin in repo_pins for p in pin.packages}),
            "pinned": None,
            "default": None,
        }
        entries.append(entry)
        several = len(repo_pins) > 1
        slots = [_pin_entry(p, several=several) for p in repo_pins[:MAX_PINS]]
        if slots:
            entry["pinned"] = slots[0]
        if slots[1:]:
            entry["also_pinned"] = slots[1:]
        if len(repo_pins) > MAX_PINS:
            entry["note"] = (
                f"Cargo.lock pins {len(repo_pins)} distinct commits of this repo; only the "
                f"{MAX_PINS} with the most packages are listed, not "
                + ", ".join(f"{p.sha[:12]} ({', '.join(p.packages)})" for p in repo_pins[MAX_PINS:])
            )
        if i >= cc.sister_max_repos:
            entry["skipped"] = f"not checked out: over the cap of {cc.sister_max_repos} per run"
            continue
        try:
            _check_out(cfg, conn, run_id, entry, deadline, should_stop)
        except Exception as exc:  # evidence only, never the run's problem
            entry["error"] = _why(exc)
    return entries


def _check_out(
    cfg: Config,
    conn: sqlite3.Connection,
    run_id: int,
    entry: dict[str, Any],
    deadline: float,
    should_stop: Callable[[], bool],
) -> None:
    """Check out `entry`'s repo at each pin (filling its slot's `path`) and at the default
    branch's tip. A failure of one checkout is recorded in `entry["error"]` and the others
    still tried; one that leaves none (the mirror, the lock) raises."""
    name = entry["repo"]
    errors: list[str] = []
    wait = min(deadline, time.monotonic() + LOCK_WAIT_SECONDS)
    with _repo_lock(_lock_path(cfg, name), wait=wait, should_stop=should_stop) as held:
        if not held:
            raise ReviewError(FailKind.INFRA, "another run holds the repo")
        mirror = wt.ensure_mirror(cfg.mirrors_dir, name, timeout=_left(deadline, cap=None))
        for slot in _pin_slots(entry):
            try:
                if not wt.ensure_commit(mirror, slot["sha"], timeout=_left(deadline)):
                    raise ReviewError(FailKind.CONTRACT, "commit not found upstream")
                path = _checkout(cfg, conn, run_id, mirror, name, slot["sha"], deadline)
                slot["path"] = str(path)
            except ReviewError as exc:
                errors.append(f"pinned {slot['sha'][:12]}: {exc.message}")
        try:
            branch = (
                wt.remote_default_branch(mirror, timeout=_left(deadline))
                or wt.mirror_default_branch(mirror)
                or ""
            )
            tip = wt.fetch_branch(mirror, branch, timeout=_left(deadline)) if branch else None
            if not tip:
                raise ReviewError(FailKind.CONTRACT, f"cannot fetch default branch {branch!r}")
            path = _checkout(cfg, conn, run_id, mirror, name, tip, deadline)
            entry["default"] = {"branch": branch, "sha": tip, "path": str(path)}
        except ReviewError as exc:
            errors.append(f"default branch: {exc.message}")
    if errors:
        entry["error"] = "; ".join(errors)


def _pin_entry(pin: Pin, *, several: bool) -> dict[str, Any]:
    slot: dict[str, Any] = {"sha": pin.sha, "ref": pin.ref, "path": None}
    if several:  # the repo-level `packages` no longer says which commit each one comes from
        slot["packages"] = list(pin.packages)
    return slot


def _pin_slots(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in (entry.get("pinned"), *(entry.get("also_pinned") or [])) if s]


def sister_root(cfg: Config) -> Path:
    return cfg.worktrees_dir / SISTER_DIR


def _lock_path(cfg: Config, repo: str) -> Path:
    return cfg.mirrors_dir / f".{repo.replace('/', '__')}.lock"


def _checkout(
    cfg: Config,
    conn: sqlite3.Connection,
    run_id: int,
    mirror: Path,
    repo: str,
    sha: str,
    deadline: float,
) -> Path:
    """The shared detached checkout of `repo` at `sha` (the caller holds the repo's lock).
    Recorded as in use by `run_id` first. Reused when it is still that commit, unmodified (a
    lane's stray write must not reach later runs); else made afresh, under a name of its own
    and moved into place once complete, so a checkout killed half way is never reused."""
    root = sister_root(cfg)
    path = root / f"{repo.replace('/', '__')}@{sha[:12]}"
    _record_use(conn, run_id, path)
    if path.is_dir() and wt.head_of(path) == sha and wt.is_clean(path):
        os.utime(path)  # gc's age and LRU order go by mtime
        return path
    if path.exists():
        wt.remove_worktree(mirror, path)
    wt.prune_worktrees(mirror)  # forget checkouts gc deleted, or this path cannot be reused
    tmp = wt.create_worktree(
        mirror, root, f"{_NEW}{path.name}-{os.getpid()}", sha, timeout=_left(deadline)
    )
    try:
        wt.move_worktree(mirror, tmp, path)
    except ReviewError:
        wt.remove_worktree(mirror, tmp)
        raise
    return path


def _record_use(conn: sqlite3.Connection, run_id: int, path: Path) -> None:
    key = f"{INUSE_KEY}{run_id}"
    with tx(conn):
        paths = json.loads(kv_get(conn, key) or "[]")
        if str(path) not in paths:
            kv_set(conn, key, json.dumps([*paths, str(path)]))


def paths_in_use(conn: sqlite3.Connection) -> set[str]:
    """The sister checkouts runs still in flight use."""
    rows = conn.execute(
        "SELECT kv.value FROM kv JOIN runs r ON kv.key = ? || r.id "
        "WHERE r.status IN ('spawned','running')",
        (INUSE_KEY,),
    )
    return {str(p) for (value,) in rows for p in json.loads(value)}


def release_finished(conn: sqlite3.Connection) -> int:
    """Drop the in-use records of runs that ended (inside the caller's transaction)."""
    return conn.execute(
        "DELETE FROM kv WHERE key LIKE ? AND NOT EXISTS (SELECT 1 FROM runs r "
        "WHERE kv.key = ? || r.id AND r.status IN ('spawned','running'))",
        (f"{INUSE_KEY}%", INUSE_KEY),
    ).rowcount


def remove_unused(
    cfg: Config, conn: sqlite3.Connection, path: Path, *, older_than: float | None = None
) -> bool:
    """gc: remove the shared checkout (or half-made one) at `path` unless a run uses it. Under
    its repo's lock, never waited for (a run is working on that repo: next time), the in-use
    records and, for the age sweep, the mtime (`older_than`) are read again, so a run that
    picked the checkout up since gc listed it keeps it. True when it was removed."""
    name = path.name.removeprefix(_NEW).rsplit("@", 1)[0]
    with _repo_lock(cfg.mirrors_dir / f".{name}.lock", wait=0) as held:
        if not held or str(path) in paths_in_use(conn):
            return False
        with contextlib.suppress(OSError):
            if older_than is not None and path.stat().st_mtime >= older_than:
                return False
        shutil.rmtree(path, ignore_errors=True)
        return True


def lane_dirs(evidence: dict[str, Any]) -> tuple[Path, ...]:
    """The sister checkouts in `evidence` still on disk, for the lanes' `--add-dir`."""
    paths = (
        Path(slot["path"])
        for entry in evidence.get("sister_repos") or []
        for slot in (*_pin_slots(entry), entry.get("default"))
        if slot and slot.get("path")
    )
    return tuple(p for p in dict.fromkeys(paths) if p.is_dir())


def _left(deadline: float, *, cap: int | None = GIT_OP_TIMEOUT_SECONDS) -> int:
    """Seconds a git operation may take: what is left of the run's budget, at most `cap`."""
    left = int(deadline - time.monotonic())
    if left <= 0:
        raise ReviewError(FailKind.INFRA, "sister checkout budget spent")
    return min(left, cap) if cap else left


@contextlib.contextmanager
def _repo_lock(
    path: Path, *, wait: float, should_stop: Callable[[], bool] = lambda: False
) -> Iterator[bool]:
    """One process at a time per sister repo (its mirror and checkouts): two runs would
    otherwise clone the same mirror or add the same checkout at once, and gc would remove a
    checkout a run is picking up. Yields whether it was had by `wait` (a time.monotonic())."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:  # closing it releases the lock
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= wait or should_stop():
                    yield False
                    return
                time.sleep(1)
        yield True


def _why(exc: BaseException) -> str:
    return exc.message if isinstance(exc, ReviewError) else f"{type(exc).__name__}: {exc}"[:300]
