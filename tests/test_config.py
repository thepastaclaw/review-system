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


def test_scaling_and_paging_defaults(tmp_path, skills_dir):
    c = cfg_mod.load(_write(tmp_path, skills_dir, "max_concurrent = 2"))
    assert c.account_scale_max == 3
    assert c.account_reserves == {}
    assert c.page_target == "channel:C0AEQ5D7SJ3"
    assert c.page_mentions == ("UCW1VE04T", "U02CNG35EGG")


def test_a_box_config_without_the_new_keys_still_scales_and_pages(tmp_path, skills_dir):
    """The live config.toml predates these keys; the code defaults must apply."""
    p = _write(tmp_path, skills_dir, "max_concurrent = 2")
    text = "\n".join(
        line
        for line in p.read_text().splitlines()
        if not line.startswith(("account_scale_max", "page_target", "page_mentions"))
    )
    p.write_text(text)
    c = cfg_mod.load(p)
    assert (c.account_scale_max, c.page_target) == (3, "channel:C0AEQ5D7SJ3")
    assert c.page_mentions == ("UCW1VE04T", "U02CNG35EGG")


def test_a_single_mention_string_is_one_mention(tmp_path, skills_dir):
    p = _write(tmp_path, skills_dir, "max_concurrent = 2")
    p.write_text(
        p.read_text().replace(
            'page_mentions = ["UCW1VE04T", "U02CNG35EGG"]', 'page_mentions = "UCW1VE04T"'
        )
    )
    assert cfg_mod.load(p).page_mentions == ("UCW1VE04T",)
