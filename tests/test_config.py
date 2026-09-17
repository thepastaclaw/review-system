"""Config loading: the slot knobs that bound live concurrency."""

from __future__ import annotations

from pathlib import Path

from reviewsys import config as cfg_mod


def _write(tmp_path: Path, skills_dir: Path, scheduling: str) -> Path:
    toml = cfg_mod.DEFAULT_TOML.replace("max_concurrent = 2", scheduling, 1).replace(
        'skills = "~/Projects/skills"', f'skills = "{skills_dir}"'
    )
    p = tmp_path / "config.toml"
    p.write_text(toml)
    return p


def test_default_toml_is_two_plus_one(tmp_path, skills_dir):
    c = cfg_mod.load(_write(tmp_path, skills_dir, "max_concurrent = 2"))
    assert (c.max_concurrent, c.priority_overflow) == (2, 1)
    assert c.max_concurrent + c.priority_overflow == 3


def test_slot_values_are_clamped_not_rejected(tmp_path, skills_dir):
    p = _write(tmp_path, skills_dir, "max_concurrent = 0")
    assert cfg_mod.load(p).max_concurrent == 1  # 0 would stall the pipeline outright
    p.write_text(p.read_text().replace("priority_overflow = 1", "priority_overflow = -3", 1))
    assert cfg_mod.load(p).priority_overflow == 0
