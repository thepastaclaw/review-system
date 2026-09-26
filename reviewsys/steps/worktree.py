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


def fetch_commit(mirror: Path, repo_url: str, sha: str) -> bool:
    """Make `sha` available in the mirror, fetching it from `repo_url` (a fork, usually) when the
    mirror does not already have it. False when the commit cannot be obtained; never raises for a
    missing or private fork so a bad link in a comment cannot fail a run."""
    try:
        _git("cat-file", "-e", f"{sha}^{{commit}}", cwd=mirror)
        return True
    except ReviewError:
        pass
    try:
        _git("fetch", "--no-tags", repo_url, sha, cwd=mirror, timeout=300)
        _git("cat-file", "-e", f"{sha}^{{commit}}", cwd=mirror)
        return True
    except ReviewError:
        return False


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


def fetch_branch(repo_dir: Path, branch: str) -> str | None:
    """Fetch `branch` into refs/remotes/origin/<branch>; its tip sha, or None when it is gone."""
    try:
        _git(
            "fetch",
            "--no-tags",
            "origin",
            f"+refs/heads/{branch}:refs/remotes/origin/{branch}",
            cwd=repo_dir,
            timeout=900,
        )
        return _git("rev-parse", f"origin/{branch}", cwd=repo_dir).strip() or None
    except ReviewError:
        return None


def ensure_commit(repo_dir: Path, sha: str) -> bool:
    """Make `sha` present (fetching it by id when needed); False when it cannot be had."""
    try:
        _git("cat-file", "-e", f"{sha}^{{commit}}", cwd=repo_dir)
        return True
    except ReviewError:
        pass
    try:
        _git("fetch", "--no-tags", "origin", sha, cwd=repo_dir, timeout=900)
        _git("cat-file", "-e", f"{sha}^{{commit}}", cwd=repo_dir)
        return True
    except ReviewError:
        return False


def is_ancestor(repo_dir: Path, ancestor: str, descendant: str) -> bool:
    try:
        _git("merge-base", "--is-ancestor", ancestor, descendant, cwd=repo_dir)
        return True
    except ReviewError:
        return False


def pre_merge_base(worktree: Path, base_branch: str, merge_commit: str, sha: str) -> str | None:
    """Where the PR's own diff starts, as GitHub showed it at merge time: the merge base of the
    head with the base branch just before the merge (the merge commit's first parent). Right
    for merge, squash and rebase merges alike; the merge base with *today's* base would be the
    head itself after a true merge and so an empty diff. None when it cannot be determined."""
    if base_branch:
        fetch_branch(worktree, base_branch)
    if not ensure_commit(worktree, merge_commit):
        return None
    try:
        return _git("merge-base", f"{merge_commit}^1", sha, cwd=worktree).strip() or None
    except ReviewError:
        return None
