"""Notice a review-now label or a ticked gate-comment box seconds after it is set.

The ingest pass (every `ingest_interval_seconds`) and the queue-comment pass (every
`queue_comment_interval_seconds`) find both anyway, minutes later. This watch only tells the
daemon to run those passes now: it reads each repo's issue events and recently updated issue
comments with `If-None-Match`, and GitHub answers an unchanged listing with a 304 that does
not count against the rate limit (measured 2026-11-27: `x-ratelimit-used` unchanged over 15
of them). A changed listing costs one request; it is a hit only when it carries a new
`labeled` event for the review-now label on a PR, or a bot gate comment whose box is ticked.
What the passes then do is unchanged, so a hit the poll also saw cannot start a second review.

A push channel (a GitHub App webhook) would replace this; see the README.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .gh import Gh
from .github import GATE_MARKER
from .ingest import REVIEW_NOW_LABEL
from .queue_status import normal_requested, priority_requested

log = logging.getLogger(__name__)

PER_PAGE = 30
INGEST = "ingest"  # a review-now label was added: the ingest pass enqueues it
QUEUE = "queue_comments"  # a gate box was ticked: the queue-comment pass honours it
# a changed listing costs a request (about 75 an hour over the enabled repos, 2026-11-27):
# at most this share of the hourly limit goes to the watch, and it reads nothing while less
# than RESERVE_SHARE of the limit is left, which the reviews need
SPEND_SHARE = 0.05
RESERVE_SHARE = 0.1
HOUR_S = 3600
# a read takes ~0.6 s through `gh` on Brody: two per repo one after another would take ~20 s
READERS = 6


@dataclass(slots=True)
class Response:
    status: int
    headers: dict[str, str]
    body: Any


def parse(raw: str) -> Response:
    """`gh api -i` output: the status line, headers, a blank line, then the JSON body."""
    head, _, body = raw.replace("\r\n", "\n").partition("\n\n")
    lines = head.split("\n")
    status = int(lines[0].split()[1]) if lines and len(lines[0].split()) > 1 else 0
    headers = {}
    for line in lines[1:]:
        k, sep, v = line.partition(":")
        if sep:
            headers[k.strip().lower()] = v.strip()
    return Response(status, headers, json.loads(body) if body.strip() else None)


@dataclass(slots=True)
class _Listing:
    etag: str = ""
    mark: Any = None  # newest event id / comment updated_at seen; None until the first read


@dataclass
class Watcher:
    gh: Gh
    bot_login: str
    repos: Iterable[str]
    clock: Callable[[], float] = time.monotonic
    listings: dict[str, _Listing] = field(default_factory=dict)
    spent: list[float] = field(default_factory=list)  # when each charged read was made
    limit: int = 5000
    remaining: int | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _get(self, endpoint: str) -> Response | None:
        """One conditional read; None when it is paused, unchanged or failed."""
        if self.paused():
            return None
        listing = self.listings.setdefault(endpoint, _Listing())
        args = ["api", "-i", endpoint]
        if listing.etag:
            args += ["-H", f"If-None-Match: {listing.etag}"]
        try:
            proc = self.gh.runner([self.gh.bin, *args], None, 30)
        except Exception as exc:
            log.warning("watch %s: %s", endpoint, exc)
            return None
        out = proc.stdout.decode() if isinstance(proc.stdout, bytes) else proc.stdout
        try:
            res = parse(out or "")
        except ValueError as exc:
            log.warning("watch %s: unreadable reply: %s", endpoint, exc)
            return None
        if "x-ratelimit-limit" in res.headers:
            self.limit = int(res.headers["x-ratelimit-limit"])
            self.remaining = int(res.headers.get("x-ratelimit-remaining", self.limit))
        if res.status == 200:
            with self._lock:
                self.spent.append(self.clock())
            listing.etag = res.headers.get("etag", "")
            return res
        if res.status != 304:
            log.warning("watch %s: HTTP %s", endpoint, res.status)
        return None

    def paused(self) -> bool:
        now = self.clock()
        with self._lock:
            self.spent = [t for t in self.spent if now - t < HOUR_S]
            if len(self.spent) >= self.limit * SPEND_SHARE:
                return True
        return self.remaining is not None and self.remaining < self.limit * RESERVE_SHARE

    def poll(self) -> set[str]:
        """Read every repo once; the passes the daemon should run now."""
        reads = [(INGEST, self._labelled, r) for r in self.repos]
        reads += [(QUEUE, self._ticked, r) for r in self.repos]
        with ThreadPoolExecutor(READERS) as pool:
            hits = pool.map(lambda read: read[1](read[2]), reads)
            return {task for (task, _, _), hit in zip(reads, hits, strict=True) if hit}

    def _labelled(self, repo: str) -> bool:
        endpoint = f"repos/{repo}/issues/events?per_page={PER_PAGE}"
        res = self._get(endpoint)
        if res is None or not isinstance(res.body, list):
            return False
        listing = self.listings[endpoint]
        ids = [int(e.get("id") or 0) for e in res.body]
        last, listing.mark = listing.mark, max([listing.mark or 0, *ids])
        if last is None:
            return False  # the first read sets where new events start
        return any(
            int(e.get("id") or 0) > last
            and e.get("event") == "labeled"
            and str((e.get("label") or {}).get("name", "")).lower() == REVIEW_NOW_LABEL
            and (e.get("issue") or {}).get("pull_request")
            for e in res.body
        )

    def _ticked(self, repo: str) -> bool:
        endpoint = f"repos/{repo}/issues/comments?sort=updated&direction=desc&per_page={PER_PAGE}"
        res = self._get(endpoint)
        if res is None or not isinstance(res.body, list):
            return False
        listing = self.listings[endpoint]
        stamps = [str(c.get("updated_at") or "") for c in res.body]
        last, listing.mark = listing.mark, max([listing.mark or "", *stamps])
        if last is None:
            return False
        bot = self.bot_login.lower()
        for c in res.body:
            body = str(c.get("body") or "")
            if (
                str(c.get("updated_at") or "") > last
                and str((c.get("user") or {}).get("login", "")).lower() == bot
                and GATE_MARKER in body
                and (priority_requested(body) or normal_requested(body))
            ):
                return True
        return False
