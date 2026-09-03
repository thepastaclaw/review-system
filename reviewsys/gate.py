"""Blocker gate: pure decision from verifier output."""

from __future__ import annotations

from .contract import VerifierOutput


def admit_phase2(verified: VerifierOutput, *, phase2_enabled: bool) -> bool:
    """Phase 2 runs only when Phase 1's verifier confirmed zero blocking findings."""
    return phase2_enabled and verified.blocker_count == 0
