"""Git for `reviewsys replay` and `replay-judge`: a per-replay bare clone pinned to the cutoff,
and never a write to a production mirror.

The production daemon fetches into its bare mirrors (`work/mirrors/<owner>__<name>.git`) all
day; a second writer there races it for ref locks (v0.24's `fetch_head` raises when it loses
one) and a replay of a repository it has no mirror for would leave a new permanent one. So:

- A replay's repository (`open_clone`) borrows a mirror's objects through
  `objects/info/alternates` (read-only; nothing is copied) and gets what it lacks by fetching
  into itself (`fetch_commit`, `branch_tip`), from the mirror's own remote URL, through a
  remote with no ref anyone can fetch (`SOURCE`). A partial mirror's missing blobs come the
  same way, lazily. Deleting the clone frees all of it.
- The mirror borrowed is the production one when it exists (`source_of`), else one under the
  eval's own directory (`eval_mirrors`, created once, under a lock), never a new production
  one.
- Its only refs (`set_refs`, `pinned_refs`): the PR head at the replayed sha and the base
  branch (also as `origin/<base>`) at the base the source run saw. Its `origin` is itself,
  so a lane's fetch from origin changes nothing.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

from .steps import worktree as wt

SOURCE = "replay-source"
_NOWHERE = "+refs/replay-pinned/none:refs/replay-pinned/none"


def mirror_name(repo: str) -> str:
    return f"{repo.replace('/', '__')}.git"


def _run(git_dir: Path, *args: str, lazy: bool = True) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    if not lazy:
        env["GIT_NO_LAZY_FETCH"] = "1"
    return subprocess.run(
        ["git", "-C", str(git_dir), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
        env=env,
    )


def _out(git_dir: Path, *args: str) -> str:
    r = _run(git_dir, *args)
    return r.stdout.strip() if r.returncode == 0 else ""


def _ok(git_dir: Path, *args: str) -> None:
    r = _run(git_dir, *args)
    if r.returncode != 0:
        raise SystemExit(f"git {' '.join(args[:2])} in {git_dir}: {r.stderr.strip()[:300]}")


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def source_of(prod_mirrors: Path, eval_mirrors: Path | None, repo: str) -> tuple[Path | None, str]:
    """(the mirror to borrow objects from, the URL to fetch what it lacks from): the
    production mirror, read-only, when there is one; else the eval's own (cloned once, under
    a lock); else none (the clone fetches everything itself)."""
    url = f"https://github.com/{repo}.git"
    prod = prod_mirrors / mirror_name(repo)
    if (prod / "HEAD").exists():
        return prod, _out(prod, "config", "--get", "remote.origin.url") or url
    if eval_mirrors is None:
        return None, url
    with _locked(eval_mirrors / f"{mirror_name(repo)}.lock"):
        mine = wt.ensure_mirror(eval_mirrors, repo)
    return mine, _out(mine, "config", "--get", "remote.origin.url") or url


def _partial_filter(mirror: Path) -> str:
    """The mirror's partial-clone filter ("" when it is a full clone)."""
    remote = _out(mirror, "config", "--get", "extensions.partialclone")
    if not remote and _out(mirror, "config", "--get", "remote.origin.promisor") == "true":
        remote = "origin"
    if not remote:
        return ""
    return _out(mirror, "config", "--get", f"remote.{remote}.partialclonefilter") or "blob:none"


def open_clone(mirror: Path | None, dest: Path, url: str) -> Path:
    """An empty bare repository at `dest` borrowing `mirror`'s objects, fetching from `url`
    through `SOURCE` (a partial clone of it when the mirror is one, or when there is no
    mirror). No refs yet: see `set_refs`."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _ok(dest.parent, "init", "-q", "--bare", str(dest))
    if mirror is not None:
        (dest / "objects" / "info" / "alternates").write_text(f"{(mirror / 'objects').resolve()}\n")
    settings = {
        "remote.origin.url": str(dest.resolve()),
        "remote.origin.fetch": "+refs/heads/*:refs/heads/*",
        "gc.auto": "0",
        f"remote.{SOURCE}.url": url,
        # a plain `git fetch replay-source` asks for a ref no repository has
        f"remote.{SOURCE}.fetch": _NOWHERE,
    }
    flt = _partial_filter(mirror) if mirror is not None else "blob:none"
    if flt:
        settings |= {
            "core.repositoryformatversion": "1",
            "extensions.partialClone": SOURCE,
            f"remote.{SOURCE}.promisor": "true",
            f"remote.{SOURCE}.partialclonefilter": flt,
        }
    for key, value in settings.items():
        _ok(dest, "config", key, value)
    return dest


def has_commit(clone: Path, sha: str) -> bool:
    return (
        bool(sha) and _run(clone, "cat-file", "-e", f"{sha}^{{commit}}", lazy=False).returncode == 0
    )


def fetch_commit(clone: Path, sha: str) -> bool:
    """Make `sha` present in the clone (fetched into it when the mirror lacks it)."""
    if has_commit(clone, sha):
        return True
    _run(clone, "fetch", "-q", "--no-tags", "--no-write-fetch-head", SOURCE, sha)
    return has_commit(clone, sha)


def branch_tip(clone: Path, branch: str, mirror: Path | None) -> str:
    """The base branch's tip now, fetched into the clone (no ref is written); else the
    mirror's own idea of it (read, never fetched)."""
    r = _run(clone, "fetch", "-q", "--no-tags", SOURCE, f"refs/heads/{branch}")
    tip = _out(clone, "rev-parse", "FETCH_HEAD") if r.returncode == 0 else ""
    (clone / "FETCH_HEAD").unlink(missing_ok=True)
    if not tip and mirror is not None:
        for ref in (f"refs/remotes/origin/{branch}", f"refs/heads/{branch}"):
            tip = _out(mirror, "rev-parse", "--verify", "-q", ref)
            if tip:
                break
    return tip


def resolve_base(
    clone: Path, mirror: Path | None, *, sha: str, base_ref: str, base_sha: str, cutoff: str
) -> tuple[str, str]:
    """(base sha, merge base) the replay reviews from: the base the source run saw (its
    evidence's `baseRefOid`) or, without it, the base branch's tip as of the cutoff."""
    if base_sha and not fetch_commit(clone, base_sha):
        base_sha = ""
    if not base_sha and base_ref:
        tip = branch_tip(clone, base_ref, mirror)
        if tip:
            base_sha = _out(clone, "rev-list", "-1", "--first-parent", f"--before={cutoff}", tip)
    merge_base = _out(clone, "merge-base", base_sha, sha) if base_sha else ""
    if not merge_base:
        raise SystemExit(
            f"cannot determine where {sha[:12]} branched from {base_ref or '?'} "
            f"(base {base_sha[:12] or 'unknown'}) in {clone}"
        )
    return base_sha, merge_base


def pinned_refs(*, number: int, sha: str, base_ref: str, base_sha: str) -> dict[str, str]:
    """The only refs a replay's repository has: the PR head at the replayed sha, and the base
    branch (also as `origin/<base>`) at the base the source run saw."""
    refs = {f"refs/pull/{number}/head": sha, "refs/heads/replay-head": sha}
    if base_ref and base_sha:
        refs |= {f"refs/heads/{base_ref}": base_sha, f"refs/remotes/origin/{base_ref}": base_sha}
    return refs


def set_refs(clone: Path, refs: dict[str, str], *, head_branch: str) -> None:
    for name, sha in refs.items():
        _ok(clone, "update-ref", name, sha)
    _ok(clone, "symbolic-ref", "HEAD", f"refs/heads/{head_branch}")


def pinned_clone(
    prod_mirrors: Path,
    eval_mirrors: Path | None,
    dest: Path,
    *,
    repo: str,
    number: int,
    sha: str,
    base_ref: str = "",
    base_sha: str = "",
    cutoff: str = "",
) -> tuple[Path, str, str]:
    """The whole of it: (clone, base sha, merge base), with the refs set. Without a base
    (none given or none found) the clone has only the head refs and the merge base is ""."""
    mirror, url = source_of(prod_mirrors, eval_mirrors, repo)
    clone = open_clone(mirror, dest, url)
    if not fetch_commit(clone, sha):
        raise SystemExit(f"cannot get {repo}@{sha[:12]} from {url}")
    merge_base = ""
    if base_ref or base_sha:
        try:
            base_sha, merge_base = resolve_base(
                clone, mirror, sha=sha, base_ref=base_ref, base_sha=base_sha, cutoff=cutoff
            )
        except SystemExit:
            if cutoff:  # a replay needs its range; the judge can do without
                raise
            base_sha = ""
    refs = pinned_refs(number=number, sha=sha, base_ref=base_ref, base_sha=base_sha)
    set_refs(clone, refs, head_branch=base_ref if base_ref and base_sha else "replay-head")
    return clone, base_sha, merge_base
