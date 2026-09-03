"""Per-repo bare mirror + per-run detached worktree at the exact head SHA."""

from __future__ import annotations

import contextlib
import shutil
import subprocess
from pathlib import Path

from ..models import FailKind, ReviewError


def _git(*args: str, cwd: Path | None = None, timeout: int = 600) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env={
                "GIT_TERMINAL_PROMPT": "0",
                "GODEBUG": "netdns=go",
                "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
                "HOME": str(Path.home()),
            },
        )
    except subprocess.TimeoutExpired as exc:
        raise ReviewError(FailKind.INFRA, f"git {args[0]} timed out") from exc
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip()
        kind = (
            FailKind.INFRA
            if any(
                k in err.lower()
                for k in (
                    "could not resolve",
                    "connection",
                    "timed out",
                    "early eof",
                    "remote end hung up",
                )
            )
            else FailKind.CONTRACT
        )
        raise ReviewError(kind, f"git {' '.join(args[:2])} failed: {err[:300]}")
    return proc.stdout


def ensure_mirror(mirrors_dir: Path, repo: str) -> Path:
    mirror = mirrors_dir / f"{repo.replace('/', '__')}.git"
    if not (mirror / "HEAD").exists():
        mirrors_dir.mkdir(parents=True, exist_ok=True)
        _git(
            "clone",
            "--bare",
            "--filter=blob:none",
            f"https://github.com/{repo}.git",
            str(mirror),
            timeout=1800,
        )
        _git("config", "remote.origin.fetch", "+refs/heads/*:refs/heads/*", cwd=mirror)
    return mirror


def fetch_head(mirror: Path, number: int, sha: str) -> None:
    """Fetch the PR head ref and make sure `sha` is present."""
    _git(
        "fetch",
        "--no-tags",
        "origin",
        f"+refs/pull/{number}/head:refs/pull/{number}/head",
        cwd=mirror,
        timeout=1800,
    )
    try:
        _git("cat-file", "-e", f"{sha}^{{commit}}", cwd=mirror)
    except ReviewError:
        _git("fetch", "--no-tags", "origin", sha, cwd=mirror, timeout=1800)
        _git("cat-file", "-e", f"{sha}^{{commit}}", cwd=mirror)


def create_worktree(mirror: Path, worktrees_dir: Path, name: str, sha: str) -> Path:
    path = worktrees_dir / name
    if path.exists():
        remove_worktree(mirror, path)
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    _git("worktree", "add", "--detach", str(path), sha, cwd=mirror, timeout=1800)
    return path


def remove_worktree(mirror: Path, path: Path) -> None:
    try:
        _git("worktree", "remove", "--force", str(path), cwd=mirror)
    except ReviewError:
        shutil.rmtree(path, ignore_errors=True)
        with contextlib.suppress(ReviewError):
            _git("worktree", "prune", cwd=mirror)


def merge_base(worktree: Path, base_branch: str, sha: str) -> str | None:
    try:
        _git(
            "fetch",
            "--no-tags",
            "origin",
            f"+refs/heads/{base_branch}:refs/remotes/origin/{base_branch}",
            cwd=worktree,
            timeout=900,
        )
        return _git("merge-base", f"origin/{base_branch}", sha, cwd=worktree).strip() or None
    except ReviewError:
        return None
