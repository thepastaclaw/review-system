"""Enums and small value types shared across the system."""

from __future__ import annotations

from enum import StrEnum


class HeadStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SUPERSEDED = "superseded"
    CLOSED = "closed"

    @property
    def terminal(self) -> bool:
        return self in {
            HeadStatus.DONE,
            HeadStatus.FAILED,
            HeadStatus.SUPERSEDED,
            HeadStatus.CLOSED,
        }


class RunStatus(StrEnum):
    SPAWNED = "spawned"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

    @property
    def terminal(self) -> bool:
        return self not in {RunStatus.SPAWNED, RunStatus.RUNNING}


class FailKind(StrEnum):
    """Classification of a run failure; decides retry policy."""

    INFRA = "infra"  # transient: gh outage, proxy 5xx, lane crash -> retry with backoff
    CONTRACT = "contract"  # model output violated contract after repair -> retry once
    FATAL = "fatal"  # PR closed, head gone, config error -> no retry


class Trigger(StrEnum):
    NEW_PR = "new_pr"
    NEW_PUSH = "new_push"
    MENTION = "mention"
    REVIEW_REQUESTED = "review_requested"
    PRIORITY_REQUEST = "priority_request"  # checkbox ticked on the queue comment
    REVIEW_REPLY = "review_reply"  # a human replied under one of the bot's finding threads
    MANUAL = "manual"
    AUDIT = "audit"  # post-merge review of a PR merged without a clean review (audit queue)
    # a check failed after the head's APPROVE: the light CI-check run, never a review (ci.py)
    CI_FAILED = "ci_failed"

    @property
    def priority(self) -> bool:
        return self in {
            Trigger.MENTION,
            Trigger.REVIEW_REQUESTED,
            Trigger.PRIORITY_REQUEST,
            Trigger.REVIEW_REPLY,
            Trigger.MANUAL,
        }


class Phase(StrEnum):
    PRELIMINARY = "preliminary"
    FINAL = "final"


class StepName(StrEnum):
    WORKTREE = "worktree"
    CARRY = "carry"  # the push left the PR's diff as reviewed: that review stands, no lanes
    SELECT = "select"
    TRIAGE = "triage"
    CONTEXT = "context"
    CONVERSE = "converse"  # reply-driven re-review of an already-reviewed commit
    PHASE1 = "phase1"
    VERIFY1 = "verify1"
    GATE = "gate"
    PHASE2 = "phase2"
    VERIFY2 = "verify2"
    FRESH_PHASE2 = "fresh_phase2"
    FRESH_VERIFY2 = "fresh_verify2"
    PERSISTENCE = "persistence"  # audit: is each blocker still present on the base tip?
    CI_CHECK = "ci_check"  # does a check that failed after an APPROVE relate to the change?
    PUBLISH = "publish"


class ReviewError(Exception):
    """A classified failure raised by worker steps."""

    def __init__(self, kind: FailKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message

    def __str__(self) -> str:
        return f"[{self.kind}] {self.message}"
