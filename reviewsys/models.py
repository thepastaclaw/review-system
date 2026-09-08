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
    MANUAL = "manual"

    @property
    def priority(self) -> bool:
        return self in {
            Trigger.MENTION,
            Trigger.REVIEW_REQUESTED,
            Trigger.PRIORITY_REQUEST,
            Trigger.MANUAL,
        }


class Phase(StrEnum):
    PRELIMINARY = "preliminary"
    FINAL = "final"


class StepName(StrEnum):
    WORKTREE = "worktree"
    SELECT = "select"
    TRIAGE = "triage"
    CONTEXT = "context"
    PHASE1 = "phase1"
    VERIFY1 = "verify1"
    GATE = "gate"
    PHASE2 = "phase2"
    VERIFY2 = "verify2"
    PUBLISH = "publish"


class ReviewError(Exception):
    """A classified failure raised by worker steps."""

    def __init__(self, kind: FailKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message

    def __str__(self) -> str:
        return f"[{self.kind}] {self.message}"
