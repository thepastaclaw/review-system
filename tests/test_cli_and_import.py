import json

from reviewsys import cli
from reviewsys.db import connect
from reviewsys.legacy_import import import_queue


def test_import_legacy_done_rows(cfg, conn, tmp_path):
    q = tmp_path / "queue.json"
    q.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "repo": "dashpay/platform",
                        "pr": 4581,
                        "sha": "6323f4db" + "0" * 32,
                        "status": "done",
                        "queued_at": "2026-09-01T00:00:00Z",
                        "completed_at": "2026-09-01T01:00:00Z",
                        "result": {"phase": "preliminary", "review_id": 5083420338},
                    },
                    {"repo": "dashpay/platform", "pr": 1, "sha": "x" * 40, "status": "pending"},
                    {"repo": "dashpay/platform", "pr": 2, "sha": "y" * 40, "status": "superseded"},
                ]
            }
        )
    )
    assert import_queue(conn, q) == 1
    assert import_queue(conn, q) == 0  # idempotent
    r = conn.execute("SELECT * FROM reviews").fetchone()
    assert (
        r["imported"] == 1 and r["github_review_id"] == 5083420338 and r["phase"] == "preliminary"
    )
    assert conn.execute("SELECT status FROM heads WHERE number=4581").fetchone()["status"] == "done"


def test_cli_status_and_init_config(cfg, tmp_path, capsys):
    cfg_path = tmp_path / "config.toml"
    assert cli.main(["--config", str(cfg_path), "status"]) == 0
    assert "heads:" in capsys.readouterr().out
    assert cli.main(["--config", str(tmp_path / "new.toml"), "init-config"]) == 0
    assert (tmp_path / "new.toml").exists()


def test_db_migration_idempotent(tmp_path):
    p = tmp_path / "x.db"
    c1 = connect(p)
    c1.close()
    c2 = connect(p)
    assert c2.execute("SELECT version FROM schema_version").fetchone()[0] == 1
    assert c2.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
