"""Import the *done* set from the legacy reviews/queue.json so the new system
does not re-review PRs that already have a posted review at their current head."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .db import now, tx


def import_queue(conn: sqlite3.Connection, queue_json: Path) -> int:
    data = json.loads(queue_json.read_text())
    items = data.get("items") or []
    n = 0
    ts = now()
    with tx(conn):
        for it in items:
            if it.get("status") != "done" or not it.get("sha"):
                continue
            repo, number, sha = str(it["repo"]), int(it["pr"]), str(it["sha"])
            phase = str((it.get("result") or {}).get("phase") or it.get("gating_status") or "final")
            review_id = (it.get("result") or {}).get("review_id") or it.get("review_id")
            exists = conn.execute(
                "SELECT 1 FROM reviews WHERE repo=? AND number=? AND sha=? LIMIT 1",
                (repo, number, sha),
            ).fetchone()
            if exists:
                continue
            conn.execute(
                "INSERT INTO reviews (run_id, repo, number, sha, phase, github_review_id, event, posted_at, imported) VALUES (NULL,?,?,?,?,?,?,?,1)",
                (
                    repo,
                    number,
                    sha,
                    "preliminary" if phase == "preliminary" else "final",
                    int(review_id) if isinstance(review_id, int) else None,
                    "IMPORTED",
                    str(it.get("completed_at") or ts),
                ),
            )
            # mark the head as done so ingest treats a later push as new_push, not new_pr
            conn.execute(
                "INSERT OR IGNORE INTO heads (repo, number, sha, trigger, priority, status, queued_at, eligible_at, finished_at, reason) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    repo,
                    number,
                    sha,
                    "new_pr",
                    0,
                    "done",
                    str(it.get("queued_at") or ts),
                    str(it.get("queued_at") or ts),
                    str(it.get("completed_at") or ts),
                    "imported from legacy queue",
                ),
            )
            n += 1
    return n
