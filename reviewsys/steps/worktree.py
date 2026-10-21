"""Per-repo bare mirror + per-run detached worktree at the exact head SHA."""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import subprocess
import threading
from pathlib import Path

from ..models import FailKind, ReviewError


def _env() -> dict[str, str]:
    return {
        "GIT_TERMINAL_PROMPT": "0",
        "GODEBUG": "netdns=go",
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME": str(Path.home()),
    }


def _git(
    *args: str, cwd: Path | None = None, timeout: int = 600, ok: tuple[int, ...] = (0,)
) -> str:
    """git's stdout; a ReviewError on a timeout or an exit code not in `ok`."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise ReviewError(FailKind.INFRA, f"git {args[0]} timed out") from exc
    if proc.returncode not in ok:
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


def ensure_mirror(mirrors_dir: Path, repo: str, *, timeout: int = 1800) -> Path:
    """The repo's bare mirror, cloned on first use. The clone lands under a name of its own and
    is renamed into place once complete, so a clone killed half way (a timeout) is never taken
    for a mirror by the next caller, and two first clones at once cannot mix."""
    mirror = mirrors_dir / f"{repo.replace('/', '__')}.git"
    if not (mirror / "HEAD").exists():
        mirrors_dir.mkdir(parents=True, exist_ok=True)
        partial = mirror.with_name(f"{mirror.name}.partial-{os.getpid()}")
        shutil.rmtree(partial, ignore_errors=True)
        try:
            _git(
                "clone",
                "--bare",
                "--filter=blob:none",
                f"https://github.com/{repo}.git",
                str(partial),
                timeout=timeout,
            )
            _git("config", "remote.origin.fetch", "+refs/heads/*:refs/heads/*", cwd=partial)
            try:
                partial.rename(mirror)
            except OSError:
                if not (mirror / "HEAD").exists():  # else another process finished first
                    shutil.rmtree(mirror, ignore_errors=True)  # an interrupted clone's leftovers
                    try:
                        partial.rename(mirror)
                    except OSError as exc:
                        raise ReviewError(FailKind.INFRA, f"cannot place mirror: {exc}") from exc
        finally:
            shutil.rmtree(partial, ignore_errors=True)
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


def create_worktree(
    mirror: Path, worktrees_dir: Path, name: str, sha: str, *, timeout: int = 1800
) -> Path:
    path = worktrees_dir / name
    if path.exists():
        remove_worktree(mirror, path)
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    _git("worktree", "add", "--detach", str(path), sha, cwd=mirror, timeout=timeout)
    return path


def move_worktree(mirror: Path, src: Path, dst: Path) -> None:
    _git("worktree", "move", str(src), str(dst), cwd=mirror)


def prune_worktrees(mirror: Path) -> None:
    """Forget worktrees whose directory is gone (gc deletes them with rmtree), so their
    path can be checked out again."""
    _git("worktree", "prune", cwd=mirror)


def head_of(path: Path) -> str | None:
    """The commit checked out at `path`, or None when it is no usable worktree."""
    try:
        return _git("rev-parse", "HEAD", cwd=path, timeout=60).strip() or None
    except ReviewError:
        return None


def is_clean(path: Path) -> bool:
    """No tracked file changed and nothing untracked in the worktree at `path`."""
    try:
        return not _git("status", "--porcelain", cwd=path, timeout=120).strip()
    except ReviewError:
        return False


def remote_default_branch(mirror: Path, *, timeout: int = 60) -> str | None:
    """The upstream's default branch as it is now (its HEAD), or None when it cannot be asked."""
    try:
        out = _git("ls-remote", "--symref", "origin", "HEAD", cwd=mirror, timeout=timeout)
    except ReviewError:
        return None
    for line in out.splitlines():  # `ref: refs/heads/<branch>\tHEAD`
        ref, _, name = line.removeprefix("ref: ").partition("\t")
        if line.startswith("ref: ") and name == "HEAD" and ref.startswith("refs/heads/"):
            return ref.removeprefix("refs/heads/")
    return None


def mirror_default_branch(mirror: Path) -> str | None:
    """The default branch the mirror was cloned with (its HEAD), or None."""
    try:
        return _git("symbolic-ref", "--short", "HEAD", cwd=mirror, timeout=60).strip() or None
    except ReviewError:
        return None


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


def fetch_branch(repo_dir: Path, branch: str, *, timeout: int = 900) -> str | None:
    """Fetch `branch` into refs/remotes/origin/<branch>; its tip sha, or None when it is gone."""
    try:
        _git(
            "fetch",
            "--no-tags",
            "origin",
            f"+refs/heads/{branch}:refs/remotes/origin/{branch}",
            cwd=repo_dir,
            timeout=timeout,
        )
        return _git("rev-parse", f"origin/{branch}", cwd=repo_dir).strip() or None
    except ReviewError:
        return None


def ensure_commit(repo_dir: Path, sha: str, *, timeout: int = 900) -> bool:
    """Make `sha` present (fetching it by id when needed); False when it cannot be had."""
    try:
        # in a blob-less mirror this may itself fetch the missing commit: same bound
        _git("cat-file", "-e", f"{sha}^{{commit}}", cwd=repo_dir, timeout=timeout)
        return True
    except ReviewError:
        pass
    try:
        _git("fetch", "--no-tags", "origin", sha, cwd=repo_dir, timeout=timeout)
        _git("cat-file", "-e", f"{sha}^{{commit}}", cwd=repo_dir)
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


# ---- base drift (drift.py): what landed on the base branch since the PR branched ----


def rev_parse(repo_dir: Path, ref: str) -> str:
    return _git("rev-parse", "--verify", f"{ref}^{{commit}}", cwd=repo_dir, timeout=60).strip()


def rev_count(repo_dir: Path, rev_range: str) -> int:
    return int(_git("rev-list", "--count", rev_range, cwd=repo_dir, timeout=120).strip() or 0)


def changed_paths(repo_dir: Path, base: str, sha: str) -> list[str]:
    """Every path the range `base..sha` touches; a rename counts as both of its names."""
    out = _git("diff", "--no-renames", "--name-only", base, sha, cwd=repo_dir, timeout=300)
    return [p for p in out.splitlines() if p]


def diff_print(repo_dir: Path, base: str, sha: str) -> str | None:
    """sha256 of the PR's own diff, `base..sha`: equal for two commits exactly when every
    changed file holds the same content and the same patch against its merge base, whatever
    the base or the history underneath (an empty commit, a merge of the base, a rebase onto
    it). `--full-index` pins binary files by blob id; no user or system git config shapes the
    output. Raw bytes: a diff need not be UTF-8. None when git cannot say."""
    digest = hashlib.sha256()
    argv = ["git", "diff", "--no-color", "--no-ext-diff", "--no-textconv", "--no-renames"]
    argv += ["--full-index", base, sha]
    env = {**_env(), "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    with subprocess.Popen(
        argv, cwd=str(repo_dir), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env
    ) as proc:
        out = proc.stdout
        assert out is not None
        timer = threading.Timer(300, proc.kill)  # a killed git exits non-zero: no print
        timer.start()
        try:
            for chunk in iter(lambda: out.read(1 << 20), b""):  # streamed: a diff can be huge
                digest.update(chunk)
            code = proc.wait()
        finally:
            timer.cancel()
    return digest.hexdigest() if code == 0 else None


def first_parent_log(
    repo_dir: Path, rev_range: str, *, max_count: int
) -> list[tuple[str, str, set[str]]]:
    """(sha, subject, paths) of the commits in `rev_range` along the first-parent line, newest
    first: one entry per landed change, a merge commit carrying everything it brought in (its
    diff to the first parent), so a merged PR shows as its merge with the PR number."""
    out = _git(
        "log",
        "--first-parent",
        "--diff-merges=first-parent",
        "--no-renames",
        "--name-only",
        f"--max-count={max_count}",
        "--format=%x00%H%x09%s",
        rev_range,
        cwd=repo_dir,
        timeout=300,
    )
    commits = []
    for chunk in out.split("\0"):
        head, _, rest = chunk.partition("\n")
        sha, sep, subject = head.partition("\t")
        if sep:
            commits.append((sha, subject, {p for p in rest.splitlines() if p}))
    return commits


def merge_conflicts(repo_dir: Path, base_ref: str, sha: str) -> list[str]:
    """The paths that conflict when `sha` is merged onto `base_ref` (git >= 2.38; nothing is
    written to any branch or the worktree). [] when it merges cleanly."""
    out = _git(
        "merge-tree",
        "--write-tree",
        "--name-only",
        "--no-messages",
        base_ref,
        sha,
        cwd=repo_dir,
        timeout=300,
        ok=(0, 1),  # 1: conflicts, listed after the tree id
    )
    return [p for p in out.splitlines()[1:] if p]
