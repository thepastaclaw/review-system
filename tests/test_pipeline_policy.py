"""`review_model_policy.pipeline` (policy v12): parsed strictly, every field defaulting to v11
behavior, absent from every v11 config, and covered by the policy fingerprint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reviewsys import config as cfg_mod

FULL = {
    "version": "v12",
    "lane_env_isolation": True,
    "structured_output": True,
    "finders": {"anchoring": "unanchored"},
    "threads": {"lane": True},
    "verifier": {"contract": "v12"},
    "confirmation": {"roles": ["general", "security-auditor"], "trigger": "no_blockers"},
    "late_nonblocking": "note",
}


def write_policy(skills_dir, pipeline, *, refingerprint: bool = True) -> None:
    """The fixture skills config with `pipeline` set (None: removed), its fingerprint
    recomputed the way the skills repo does unless told otherwise."""
    raw = json.loads((skills_dir / "config.json").read_text())
    pol = raw["review_model_policy"]
    pol.pop("pipeline", None)
    if pipeline is not None:
        pol["pipeline"] = pipeline
    if refingerprint:
        pol["fingerprint"] = cfg_mod.policy_fingerprint(pol)
    (skills_dir / "config.json").write_text(json.dumps(raw))


def load_with(cfg, skills_dir, pipeline, **kw) -> cfg_mod.Config:
    write_policy(skills_dir, pipeline, **kw)
    return cfg_mod.load(cfg.skills_dir.parent / "config.toml")


def test_a_v11_policy_has_no_pipeline(cfg):
    assert cfg.policy.pipeline is None


def test_every_field_parses(cfg, skills_dir):
    p = load_with(cfg, skills_dir, FULL).policy.pipeline
    assert p == cfg_mod.PipelinePolicy(
        version="v12",
        lane_env_isolation=True,
        structured_output=True,
        finders=cfg_mod.FinderPolicy(anchoring="unanchored"),
        threads=cfg_mod.ThreadsPolicy(lane=True),
        verifier=cfg_mod.VerifierPolicy(contract="v12"),
        confirmation=cfg_mod.ConfirmationPolicy(
            roles=("general", "security-auditor"), trigger="no_blockers"
        ),
        late_nonblocking="note",
    )


@pytest.mark.parametrize("block", [{"version": "v12"}, "v12"])
def test_a_bare_version_defaults_every_switch_to_v11(cfg, skills_dir, block):
    p = load_with(cfg, skills_dir, block).policy.pipeline
    assert p == cfg_mod.PipelinePolicy(version="v12")
    assert not p.lane_env_isolation and not p.structured_output
    assert p.finders.anchoring == "anchored" and not p.threads.lane
    assert p.verifier.contract == "v11" and p.late_nonblocking == "post"
    assert p.confirmation == cfg_mod.ConfirmationPolicy(roles=None, trigger="rereview")


@pytest.mark.parametrize(
    ("block", "error"),
    [
        ({"version": "v13"}, "pipeline.version"),
        ({"lane_env_isolation": True}, "pipeline.version"),
        ({"version": "v12", "structured_ouput": True}, "unknown keys ['structured_ouput']"),
        ({"version": "v12", "lane_env_isolation": "yes"}, "must be true or false"),
        ({"version": "v12", "finders": {"anchoring": "loose"}}, "pipeline.finders.anchoring"),
        ({"version": "v12", "finders": {"anchor": "unanchored"}}, "pipeline.finders: unknown"),
        ({"version": "v12", "finders": "unanchored"}, "pipeline.finders must be an object"),
        ({"version": "v12", "threads": {"lane": 1}}, "pipeline.threads.lane must be"),
        ({"version": "v12", "verifier": {"contract": "v10"}}, "pipeline.verifier.contract"),
        ({"version": "v12", "confirmation": {"trigger": "always"}}, "confirmation.trigger"),
        ({"version": "v12", "confirmation": {"roles": "general"}}, "must be a list"),
        ({"version": "v12", "confirmation": {"roles": ["fuzzer"]}}, "['fuzzer']"),
        ({"version": "v12", "late_nonblocking": "drop"}, "pipeline.late_nonblocking"),
        (["v12"], "must be an object"),
    ],
)
def test_a_bad_block_fails_the_load(cfg, skills_dir, block, error):
    with pytest.raises(ValueError) as exc:
        load_with(cfg, skills_dir, block)
    assert error in str(exc.value)


def test_fingerprint_is_the_skills_repo_hash_of_the_whole_policy(cfg, skills_dir):
    raw = json.loads((skills_dir / "config.json").read_text())["review_model_policy"]
    base = cfg_mod.policy_fingerprint(raw)
    assert cfg_mod.policy_fingerprint({**raw, "fingerprint": "anything"}) == base
    v12 = cfg_mod.policy_fingerprint({**raw, "pipeline": {"version": "v12"}})
    flipped = cfg_mod.policy_fingerprint(
        {**raw, "pipeline": {"version": "v12", "structured_output": True}}
    )
    assert len({base, v12, flipped}) == 3, "every pipeline switch changes the fingerprint"


def test_a_pipeline_under_a_stale_fingerprint_fails_the_load(cfg, skills_dir):
    """A v12 block added without recomputing the fingerprint would publish v12 reviews under
    the v11 policy's name."""
    with pytest.raises(ValueError, match="does not cover the pipeline block"):
        load_with(cfg, skills_dir, {"version": "v12"}, refingerprint=False)
    # v11 configs keep loading whatever their fingerprint (the fixture's is a placeholder)
    assert load_with(cfg, skills_dir, None, refingerprint=False).policy.pipeline is None


V11_FINGERPRINT = "7503c1767e352a859858bb3f2e183781edd0450c0451a57e9d510ac21846a6a0"


def test_fingerprint_reproduces_the_live_v11_policy():
    """`review_model_policy` of skills c697406 (policy v11.1, live beside reviewsys v0.28.0;
    unchanged since 6157b96), copied as is: the hash must match the one the skills repo
    stored. Its text has non-ASCII ("—"), which only the
    default `ensure_ascii=True` encoding reproduces."""
    pol = json.loads(
        (Path(__file__).parent / "golden" / "skills-v11-policy.json").read_text(encoding="utf-8")
    )
    assert pol["version"] == 11 and pol["fingerprint"] == V11_FINGERPRINT
    assert cfg_mod.policy_fingerprint(pol) == V11_FINGERPRINT
    body = {k: v for k, v in pol.items() if k != "fingerprint"}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert not raw.isascii(), "the pin exercises the non-ASCII encoding"
