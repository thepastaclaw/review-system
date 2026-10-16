"""Base drift: how the PR's base branch moved since the PR branched off it (the `base_drift`
evidence). Why (2026-10-16 lane audit): an equivalent mechanism, or a conflicting change, had
already landed on the base while lanes reviewed only the PR's own range.

Local git only, in the run's worktree, against the `origin/<base>` that `step_worktree`
fetched to find the merge base; the merge check may fetch base-side blobs (the mirror is a
blob-less partial clone)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .models import ReviewError
from .steps import worktree as wt

MAX_COMMITS = 40  # base commits listed
SCAN_COMMITS = 2000  # first-parent base commits looked at, newest first
MAX_FILES = 20  # overlapping paths listed per commit

# `Merge pull request #N from ...` (GitHub merge), `Merge #N: ...` (dash's merge script),
# `subject (#N)` (squash merge). Not `Merge bitcoin/bitcoin#N`: an upstream number.
_PR_PATTERNS = (
    re.compile(r"^Merge pull request #(\d+)\b"),
    re.compile(r"^Merge #(\d+)\b"),
    re.compile(r"\(#(\d+)\)\s*$"),
)


def pr_number(subject: str) -> int | None:
    for pattern in _PR_PATTERNS:
        if m := pattern.search(subject):
            return int(m.group(1))
    return None


def base_drift(worktree: Path, base_branch: str, merge_base: str, head: str) -> dict[str, Any]:
    """`base_branch`, `base_tip`, `merge_base`, `commits_behind` (merge base..base tip),
    `conflicting_files` (null when the merge check failed: see `error`) and
    `base_commits_touching_changed_files` (`sha`, `subject`, `pr`, `files`: the base commits
    since the merge base that touch a path the PR changes, newest first; `truncated` when
    there were more). Raises ReviewError when the base cannot be read at all."""
    base_ref = f"refs/remotes/origin/{base_branch}"  # in full: a tag could shadow the short name
    behind = wt.rev_count(worktree, f"{merge_base}..{base_ref}")
    out: dict[str, Any] = {
        "base_branch": base_branch,
        "base_tip": wt.rev_parse(worktree, base_ref),
        "merge_base": merge_base,
        "commits_behind": behind,
        "conflicting_files": [],
        "base_commits_touching_changed_files": [],
    }
    if not behind:
        return out  # the PR is on top of the current base: nothing landed, nothing conflicts
    changed = set(wt.changed_paths(worktree, merge_base, head))
    log = wt.first_parent_log(worktree, f"{merge_base}..{base_ref}", max_count=SCAN_COMMITS + 1)
    scanned_all = len(log) <= SCAN_COMMITS
    hits = [(sha, subj, o) for sha, subj, paths in log[:SCAN_COMMITS] if (o := paths & changed)]
    out["base_commits_touching_changed_files"] = [
        {"sha": sha, "subject": subject, "pr": pr_number(subject), "files": sorted(o)[:MAX_FILES]}
        for sha, subject, o in hits[:MAX_COMMITS]
    ]
    if len(hits) > MAX_COMMITS or not scanned_all:
        out["truncated"] = True
    try:
        out["conflicting_files"] = wt.merge_conflicts(worktree, base_ref, head)
    except ReviewError as exc:
        out["conflicting_files"] = None
        out["error"] = f"merge check: {exc.message}"
    return out
