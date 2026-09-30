"""Blocker gate: pure decision from verifier output plus the triage tier."""

from __future__ import annotations

from .config import GatePolicy
from .contract import VerifierOutput


def gate_points(verified: VerifierOutput, gate: GatePolicy | None) -> int | None:
    """The verified Phase-1 findings scored by the gate policy; None without one."""
    return gate.points([f.severity for f in verified.findings]) if gate else None


def phase1_blocks(verified: VerifierOutput, gate: GatePolicy | None) -> bool:
    """Whether Phase 1's verified findings hold Phase 2 back: above the gate's point budget,
    or (without a gate policy) any verified blocker."""
    if gate is None:
        return verified.blocker_count > 0
    return gate.points([f.severity for f in verified.findings]) > gate.block_above


def admit_phase2(
    verified: VerifierOutput,
    *,
    phase2_enabled: bool,
    tier_allows: bool = True,
    gate: GatePolicy | None = None,
) -> bool:
    """Phase 2 runs when the triage tier calls for a second round (trivial changes stop after
    Phase 1) and Phase 1's verified findings do not block it (see `phase1_blocks`)."""
    return phase2_enabled and tier_allows and not phase1_blocks(verified, gate)
