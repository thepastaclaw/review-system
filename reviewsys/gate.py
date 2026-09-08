"""Blocker gate: pure decision from verifier output plus the triage tier."""

from __future__ import annotations

from .contract import VerifierOutput


def admit_phase2(
    verified: VerifierOutput, *, phase2_enabled: bool, tier_allows: bool = True
) -> bool:
    """Phase 2 runs only when Phase 1's verifier confirmed zero blocking findings and the
    triage tier calls for a second round (trivial changes stop after Phase 1)."""
    return phase2_enabled and tier_allows and verified.blocker_count == 0
