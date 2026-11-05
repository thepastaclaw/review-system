"""Config loading: the slot knobs that bound live concurrency."""

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_max_runs_defaults_to_a_safety_cap_and_is_clamped(tmp_path, skills_dir):
    p = _write(tmp_path, skills_dir, "max_concurrent = 2")
    assert cfg_mod.load(p).max_runs == 30
    p.write_text(p.read_text().replace("max_runs = 30", "max_runs = -4", 1))
    assert cfg_mod.load(p).max_runs == 0, "negative reads as no cap, never a crash loop"
    # the box config.toml predates the key: the code default applies
    p.write_text(p.read_text().replace("max_runs = -4\n", "", 1))
    assert cfg_mod.load(p).max_runs == 30


def test_event_retention_defaults_to_forever_and_parses(tmp_path, skills_dir):
    p = _write(tmp_path, skills_dir, "max_concurrent = 2")
    assert cfg_mod.load(p).event_retention_days == 0
    p.write_text(p.read_text().replace("event_days = 0", "event_days = 45", 1))
    assert cfg_mod.load(p).event_retention_days == 45
    p.write_text(p.read_text().replace("event_days = 45", "event_days = -5", 1))
    assert cfg_mod.load(p).event_retention_days == 0
    # a box config written before the key existed keeps events forever too
    lines = [ln for ln in p.read_text().splitlines() if not ln.startswith("event_days")]
    p.write_text("\n".join(lines))
    assert cfg_mod.load(p).event_retention_days == 0


def test_deferral_filing_allowlist(tmp_path, skills_dir):
    p = _write(tmp_path, skills_dir, "max_concurrent = 2")
    d = cfg_mod.load(p).deferral
    assert d.may_file("dashpay/platform") and d.may_file("DashPay/rust-dashcore")
    assert not d.may_file("bitcoin/bitcoin")
    # a box config.toml predating the section files in dashpay repos, as a fresh one does
    head, _, _ = p.read_text().partition("[deferral]")
    p.write_text(head)
    assert cfg_mod.load(p).deferral.may_file("dashpay/platform")
    p.write_text(head + '[deferral]\nfile_owners = []\nfile_repos = ["bitcoin/bitcoin"]\n')
    d = cfg_mod.load(p).deferral
    assert d.may_file("bitcoin/bitcoin") and not d.may_file("dashpay/platform")
    # a bare string is one owner, not one per letter
    p.write_text(head + '[deferral]\nfile_owners = "dashpay"\n')
    assert cfg_mod.load(p).deferral.file_owners == ("dashpay",)


def test_tracking_cutoff_is_normalized_and_validated(tmp_path, skills_dir):
    p = _write(tmp_path, skills_dir, "max_concurrent = 2")
    assert cfg_mod.load(p).deferral.tracking_required_since == ""  # unset: recorded at start
    base = p.read_text()
    p.write_text(
        base.replace(
            "file_repos = []",
            'file_repos = []\ntracking_required_since = "2026-10-17T02:00:00+02:00"',
        )
    )
    assert cfg_mod.load(p).deferral.tracking_required_since == "2026-10-17T00:00:00Z"
    p.write_text(
        base.replace("file_repos = []", 'file_repos = []\ntracking_required_since = "soon"')
    )
    with pytest.raises(ValueError, match="tracking_required_since"):
        cfg_mod.load(p)
