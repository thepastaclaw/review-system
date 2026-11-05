"""Blocker gate: pure decision from verifier output plus the triage tier."""

from __future__ import annotations

from .config import GatePolicy
from .contract import VerifierOutput


def _scored(verified: VerifierOutput) -> list[str]:
    """Severities the gate scores: every finding new to this head, plus carried-forward
    (STILL_VALID) blockers. Carried suggestions are left out: they were posted by an earlier
    round, often by Phase 2, and would otherwise hold every later head at Phase 1 until the
    author acted on them. Policy v12 carries the ledger's standing issues beside the verified
    set (`carried`, every one with its hash): its blockers count the same way."""
    return [
        f.severity
        for f in (*verified.findings, *verified.carried)
        if not f.prior_hash or f.severity == "blocking"
    ]


def phase1_blocks(verified: VerifierOutput, gate: GatePolicy | None) -> bool:
    """Whether Phase 1's verified findings hold Phase 2 back: above the gate's point budget,
    or (without a gate policy) any verified blocker."""
    if gate is None:
        return verified.blocker_count > 0
    return gate.points(_scored(verified)) > gate.block_above


def gate_score(verified: VerifierOutput, gate: GatePolicy | None) -> dict[str, int | None]:
    """`points` and `block_above` for the gate step detail and the gate comment; both None
    without a gate policy."""
    if gate is None:
        return {"points": None, "block_above": None}
    return {"points": gate.points(_scored(verified)), "block_above": gate.block_above}
