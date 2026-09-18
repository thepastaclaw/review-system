"""The public status export must show the queue in the order the scheduler drains it."""

from __future__ import annotations

from datetime import timedelta

from reviewsys import degraded
from reviewsys.db import fmt_ts, now_dt, tx
from reviewsys.exporter import build_export
from reviewsys.ingest import enqueue_head
from reviewsys.models import Trigger
from reviewsys.queue_status import queued_order
from reviewsys.scheduler import eligible_heads


def _queue(conn, cfg, number, *, queued_ago_min, eligible_in_min, trigger=Trigger.NEW_PR):
    at = now_dt()
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", number, f"{number:040x}", trigger)
        conn.execute(
            "UPDATE heads SET queued_at=?, eligible_at=? WHERE repo='dashpay/platform' AND number=?",
            (
                fmt_ts(at - timedelta(minutes=queued_ago_min)),
                fmt_ts(at + timedelta(minutes=eligible_in_min)),
                number,
            ),
        )


def test_export_queue_order_matches_the_scheduler(cfg, conn):
    # oldest head, long eligible; it must be first even though its eligible_at is not the
    # smallest (the dashboard used to sort by eligible_at and showed it 10th)
    _queue(conn, cfg, 10, queued_ago_min=600, eligible_in_min=-5)
    # queued later but its backoff ended earlier
    _queue(conn, cfg, 11, queued_ago_min=300, eligible_in_min=-30)
    # priority head queued last of all: goes first
    _queue(conn, cfg, 12, queued_ago_min=1, eligible_in_min=-1, trigger=Trigger.MENTION)
    # still in debounce: after every eligible head, whatever its queued_at
    _queue(conn, cfg, 13, queued_ago_min=900, eligible_in_min=20)
    export = build_export(conn, cfg)
    numbers = [q["number"] for q in export["live"]["queued"]]
    assert numbers == [12, 10, 11, 13]
    assert [q["position"] for q in export["live"]["queued"]] == [1, 2, 3, 4]
    assert [q["number"] for q in queued_order(conn)] == numbers, "same order as the queue comments"
    assert [h["number"] for h in eligible_heads(conn, ts=fmt_ts(now_dt()))] == [12, 10, 11], (
        "same order as the scheduler picks"
    )
    assert export["live"]["queued"][-1]["reason"] == "waiting for debounce/backoff"
    assert export["live"]["queued"][0]["priority"] is True
    assert export["live"]["degraded"] == {"configured": False, "active": False}


def test_export_carries_degraded_state(cfg, conn, skills_dir, tmp_path):
    import json

    from reviewsys import config as cfg_mod

    raw = json.loads((skills_dir / "config.json").read_text())
    raw["review_model_policy"]["degraded"] = {
        "sentinel": "gpt-6-astra",
        "substitutes": {"gpt-6-astra": "muse-spark-1.3-contributor"},
    }
    (skills_dir / "config.json").write_text(json.dumps(raw))
    c = cfg_mod.load(tmp_path / "config.toml")
    degraded.force(conn, "on")
    d = build_export(conn, c)["live"]["degraded"]
    assert d["active"] and d["forced"] == "on"
    assert d["substitutes"] == {"gpt-6-astra": "muse-spark-1.3-contributor"}
