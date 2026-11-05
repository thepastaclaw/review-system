"""JSON Schemas for policy v12 lane answers, one per lane kind (v12 design §2.1, §2.2, §2.4).

A lane gets its schema as `--json-schema` only while `pipeline.structured_output` is on
(`worker._run_checked_lane(schema=...)`); otherwise it answers in text as before. Either way
its answer goes through the same parser, which checks every value itself.

Rules every schema here follows:
- Every constrained value is listed in its `description` as well as in `enum` / `pattern`.
  Some upstreams (Gemini) strip `enum` from tool schemas, so the prose is what the model
  reliably sees.
- Optional values are omitted, never null. Nullable union types do not survive every
  upstream's schema dialect.
- No echo fields (`review_phase`, `head_sha`): the harness stamps them.
- No `maxItems` caps: caps cost recall (§2.1). (`thread_schema` pins an exact count; that
  is the contract, not a cap.)
- No `additionalProperties: false`: a model that adds an unasked field would burn its
  structured-output retries on it, and the parser ignores extra fields.

The v12 finder, verify+match and thread lanes pass them (`worker._reviewer_lanes`,
`_verify_shard`, `_run_thread_lane`); with a schema their prompts end in
`prompts.STRUCTURED_CONTRACT` instead of the raw-JSON rule, and their correction turns ask for
the StructuredOutput tool too.
"""

from __future__ import annotations

import copy
from typing import Any

from .contract import CATEGORIES, SEVERITIES

# the verifier's verdict per candidate group (§2.2); blocking requires CONFIRMED (§2.3)
VERDICTS = ("CONFIRMED", "PLAUSIBLE", "REFUTED")
# the thread lane's status per open issue (§2.4). FIXED on the reviewed commit itself is
# turned into FIX_PENDING by the harness, as the conversation lane does today
THREAD_STATUSES = ("STILL_VALID", "FIXED", "OUTDATED", "WITHDRAWN", "INTENTIONALLY_DEFERRED")
NO_REPLY = "NO_REPLY"  # a thread-lane `reply` that posts nothing
# a verifier group's ledger match: a new issue, an issue of the ledger by its 12-hex
# finding_hash, or a CodeRabbit inline comment by its id
MATCH_PATTERN = r"^(new|(open|settled|fixed):[0-9a-f]{12}|coderabbit:[0-9]+)$"


def _text(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def _one_of(values: tuple[str, ...], description: str) -> dict[str, Any]:
    return {
        "type": "string",
        "enum": list(values),
        "description": f"{description} Exactly one of: {' | '.join(values)}.",
    }


SUGGESTION = (
    "Replacement code for the selected lines, committable as is; omit it when there is no "
    "exact fix. Never prose."
)


def _line(description: str) -> dict[str, Any]:
    return {"type": "integer", "minimum": 1, "description": description}


FINDER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": _text("Two or three sentences on what the change does and how it reads."),
        "candidates": {
            "type": "array",
            "description": "Every problem found, as a candidate for the verifier; [] when none.",
            "items": {
                "type": "object",
                "properties": {
                    "file": _text("Repository-relative path of the file the problem is in."),
                    "line_start": _line("First line of the problem at the reviewed head."),
                    "line_end": _line("Last line of the problem; omit for a single line."),
                    "severity": _one_of(SEVERITIES, "How much the problem matters."),
                    "category": _one_of(CATEGORIES, "The kind of problem."),
                    "title": _text("One line naming the problem."),
                    "failure_scenario": _text(
                        "The concrete input, state or sequence that makes it go wrong, "
                        "and what then happens."
                    ),
                    "body": _text("The explanation a maintainer reads, with evidence."),
                    "suggestion": _text(SUGGESTION),
                },
                "required": ["file", "severity", "category", "title", "failure_scenario", "body"],
            },
        },
    },
    "required": ["summary", "candidates"],
}

VERIFIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": _text("The review summary a maintainer reads first."),
        "groups": {
            "type": "array",
            "description": (
                "One entry per distinct problem: candidates that describe the same problem "
                "form one group. Every candidate id appears in exactly one group."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "members": {
                        "type": "array",
                        "minItems": 1,
                        "items": _text("A candidate id exactly as given, e.g. p2-general-3."),
                        "description": "The ids of the candidates this group covers.",
                    },
                    "match": {
                        "type": "string",
                        "pattern": MATCH_PATTERN,
                        "description": (
                            "What the problem is in the issue ledger. Exactly one of: new (not "
                            "in the ledger) | open:<hash> (an open issue) | settled:<hash> (a "
                            "withdrawn, deferred or outdated issue) | fixed:<hash> (a fixed "
                            "issue found again) | coderabbit:<id> (a CodeRabbit inline "
                            "comment). <hash> is the 12-hex finding_hash exactly as listed; "
                            "<id> is the comment id."
                        ),
                    },
                    "verdict": _one_of(
                        VERDICTS,
                        "CONFIRMED: the code shows the problem happens. PLAUSIBLE: it could, "
                        "but something needed to confirm it is missing. REFUTED: the code shows "
                        "it does not.",
                    ),
                    "evidence": _text(
                        "The quoted line(s) and the input or state that triggers the problem. "
                        "For REFUTED, quote the line or guard that contradicts the claim."
                    ),
                    "confirm_by": _text(
                        "For PLAUSIBLE: what would confirm it (a test, a trace, a fact to check)."
                    ),
                    "severity": _one_of(SEVERITIES, "The final severity; omit for REFUTED."),
                    "title": _text(
                        "The final one-line title; omit for REFUTED, except a REFUTED group "
                        "whose evidence starts with `Out of scope:`, which needs one."
                    ),
                    "body": _text(
                        "The final explanation a maintainer reads, with evidence; omit for REFUTED."
                    ),
                    "suggestion": _text(SUGGESTION),
                },
                # severity, title and body too, except for a REFUTED group (the parser checks)
                "required": ["members", "match", "verdict", "evidence"],
            },
        },
    },
    "required": ["summary", "groups"],
}

THREAD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "threads": {
            "type": "array",
            "description": "Exactly one entry per listed finding_hash.",
            "items": {
                "type": "object",
                "properties": {
                    "finding_hash": _text("The issue's finding_hash exactly as listed."),
                    "status": _one_of(THREAD_STATUSES, "Where the issue stands at this head."),
                    "reply": _text(
                        "Prose for the thread, addressed to the people in it; or exactly "
                        f"{NO_REPLY} when there is nothing useful to add."
                    ),
                    "reasoning": _text("One or two private sentences on why; never posted."),
                },
                "required": ["finding_hash", "status", "reply"],
            },
        }
    },
    "required": ["threads"],
}


def verifier_schema(candidate_ids: list[str]) -> dict[str, Any]:
    """VERIFIER_SCHEMA with `members` held to this run's candidate ids."""
    schema = copy.deepcopy(VERIFIER_SCHEMA)
    members = schema["properties"]["groups"]["items"]["properties"]["members"]
    members["items"] = _one_of(tuple(candidate_ids), members["items"]["description"])
    return schema


def thread_schema(hashes: list[str]) -> dict[str, Any]:
    """THREAD_SCHEMA held to this run's work set: as many entries as hashes, each naming one
    of them. That bounds the count and rejects an invented hash, but JSON Schema cannot say
    "each hash exactly once" for objects (a duplicate plus an omission still validates), so
    the parser must also call `check_thread_coverage`."""
    schema = copy.deepcopy(THREAD_SCHEMA)
    threads = schema["properties"]["threads"]
    threads["minItems"] = threads["maxItems"] = len(hashes)
    row = threads["items"]["properties"]
    row["finding_hash"] = _one_of(tuple(hashes), row["finding_hash"]["description"])
    return schema


def check_thread_coverage(answer: dict[str, Any], hashes: list[str]) -> list[str]:
    """Every way a thread-lane answer misses the work set: a hash outside it, one answered
    twice, one never answered. [] when each hash is answered exactly once. All problems, not
    just the first, so one correction turn can fix them together."""
    rows = answer.get("threads")
    if not isinstance(rows, list):
        return ["threads must be a list"]
    expected = set(hashes)
    seen: set[str] = set()
    problems: list[str] = []
    for row in rows:
        h = str(row.get("finding_hash") or "") if isinstance(row, dict) else ""
        if h not in expected:
            problems.append(f"threads: {h!r} is not one of the listed finding_hash values")
        elif h in seen:
            problems.append(f"threads: {h} answered more than once")
        seen.add(h)
    missing = [h for h in hashes if h not in seen]
    if missing:
        problems.append(f"threads: no entry for {missing}")
    return problems
