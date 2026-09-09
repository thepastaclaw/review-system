"""Finding and verifier output contracts, plus byte-compatible identity hashes.

`finding_hash` and `finding_dedupe_key` MUST stay byte-identical to the legacy
implementation: they are embedded in HTML markers on already-posted GitHub
comments and drive cross-round deduplication.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .models import FailKind, ReviewError

SEVERITIES = ("blocking", "suggestion", "nitpick")
CATEGORIES = (
    "bug",
    "security",
    "logic",
    "performance",
    "style",
    "naming",
    "docs",
    "test-coverage",
    "architecture",
    "general",
    "backport-prereq",
)
REVALIDATION_STATUSES = ("STILL_VALID", "FIXED", "OUTDATED", "INTENTIONALLY_DEFERRED", "WITHDRAWN")

_NORMALIZE_RE = re.compile(r"[^\w\s]|_")
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.S)


def finding_hash(file: str, category: str, title: str) -> str:
    """SHA-256 of '{file}:{category}:{title}', truncated to 12 hex chars. Legacy-compatible."""
    raw = f"{file}:{category}:{title.lower().strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def normalize_finding_text(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(_NORMALIZE_RE.sub(" ", str(text).lower()).split())


def finding_dedupe_key(file: str, category: str, title: str, body: str) -> str:
    """Stable hash over normalized file/category/title/body, 16 hex chars. Legacy-compatible."""
    norm = "|".join(
        [
            (file or "").lower().strip(),
            (category or "general").lower().strip(),
            normalize_finding_text(title),
            normalize_finding_text(body),
        ]
    )
    return hashlib.sha256(norm.encode()).hexdigest()[:16]


@dataclass(slots=True)
class Finding:
    file: str
    title: str
    body: str
    severity: str = "nitpick"
    category: str = "general"
    line_start: int | None = None
    line_end: int | None = None
    confidence: float | None = None
    suggestion: str | None = None
    source: str = "unknown"
    prior_hash: str | None = None  # finding_hash carried from a prior round (STILL_VALID)
    root_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def hash(self) -> str:
        return finding_hash(self.file, self.category, self.title)

    @property
    def dedupe_key(self) -> str:
        return finding_dedupe_key(self.file, self.category, self.title, self.body)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "file": self.file,
            "title": self.title,
            "body": self.body,
            "severity": self.severity,
            "category": self.category,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "confidence": self.confidence,
            "suggestion": self.suggestion,
            "source": self.source,
        }
        if self.prior_hash:
            d["finding_hash"] = self.prior_hash
        if self.root_id:
            d["root_id"] = self.root_id
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, source: str | None = None) -> Finding:
        if not isinstance(raw, dict):
            raise ReviewError(FailKind.CONTRACT, f"finding is not an object: {type(raw).__name__}")
        title = str(raw.get("title") or "").strip()
        file = str(raw.get("file") or "").strip()
        if not title:
            raise ReviewError(FailKind.CONTRACT, "finding has no title")
        sev = str(raw.get("severity") or "nitpick").strip().lower()
        if sev not in SEVERITIES:
            raise ReviewError(FailKind.CONTRACT, f"finding {title!r} has invalid severity {sev!r}")
        cat = str(raw.get("category") or "general").strip().lower() or "general"
        ls, le = raw.get("line_start"), raw.get("line_end")
        ls = int(ls) if isinstance(ls, int | float) and not isinstance(ls, bool) else None
        le = int(le) if isinstance(le, int | float) and not isinstance(le, bool) else ls
        conf = raw.get("confidence")
        conf = float(conf) if isinstance(conf, int | float) and not isinstance(conf, bool) else None
        sugg = raw.get("suggestion")
        known = {
            "file",
            "title",
            "body",
            "severity",
            "category",
            "line_start",
            "line_end",
            "confidence",
            "suggestion",
            "source",
            "finding_hash",
            "root_id",
        }
        return cls(
            file=file,
            title=title,
            body=str(raw.get("body") or "").strip(),
            severity=sev,
            category=cat,
            line_start=ls,
            line_end=le,
            confidence=conf,
            suggestion=str(sugg) if isinstance(sugg, str) and sugg.strip() else None,
            source=source or str(raw.get("source") or "unknown"),
            prior_hash=str(raw["finding_hash"]) if raw.get("finding_hash") else None,
            root_id=str(raw["root_id"]) if raw.get("root_id") else None,
            extra={k: v for k, v in raw.items() if k not in known},
        )


@dataclass(slots=True)
class ReviewerOutput:
    summary: str
    findings: list[Finding]
    out_of_scope: list[dict[str, Any]]
    review_phase: str
    head_sha: str
    prior_reconciliation: list[dict[str, Any]]
    raw: dict[str, Any]


@dataclass(slots=True)
class VerifierOutput:
    summary: str
    review_action: str
    findings: list[Finding]
    dropped: list[dict[str, Any]]
    out_of_scope: list[dict[str, Any]]
    coderabbit_reactions: list[dict[str, Any]]
    prerequisite_adjudications: list[dict[str, Any]]
    adjudication_complete: bool
    review_phase: str
    raw: dict[str, Any]

    @property
    def blocker_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "blocking")


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the model's stdout as one JSON object, tolerating fences and prose around it."""
    text = text.strip()
    if not text:
        raise ReviewError(FailKind.CONTRACT, "empty model output")
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = _FENCE_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    raise ReviewError(FailKind.CONTRACT, f"model output is not a JSON object: {text[:160]!r}")


def parse_reviewer_output(
    raw: dict[str, Any], *, expected_phase: str, head_sha: str, source: str, prior_hashes: set[str]
) -> ReviewerOutput:
    findings_raw = raw.get("findings")
    if not isinstance(findings_raw, list):
        raise ReviewError(FailKind.CONTRACT, "reviewer output: findings must be a list")
    findings = [Finding.from_dict(f, source=source) for f in findings_raw]
    phase = str(raw.get("review_phase") or "")
    if phase != expected_phase:
        raise ReviewError(
            FailKind.CONTRACT,
            f"reviewer output review_phase={phase!r}, expected {expected_phase!r}",
        )
    out_head = str(raw.get("head_sha") or "")
    if out_head != head_sha:
        raise ReviewError(
            FailKind.CONTRACT,
            f"reviewer output head_sha={out_head[:12]!r}, expected {head_sha[:12]!r}",
        )
    recon = raw.get("prior_finding_reconciliation") or []
    if not isinstance(recon, list):
        raise ReviewError(FailKind.CONTRACT, "prior_finding_reconciliation must be a list")
    if prior_hashes:
        seen: dict[str, str] = {}
        for row in recon:
            if not isinstance(row, dict):
                raise ReviewError(FailKind.CONTRACT, "reconciliation row is not an object")
            h, st = str(row.get("finding_hash") or ""), str(row.get("status") or "")
            if h not in prior_hashes or st not in REVALIDATION_STATUSES or h in seen:
                raise ReviewError(
                    FailKind.CONTRACT, f"bad reconciliation row hash={h!r} status={st!r}"
                )
            seen[h] = st
        missing = prior_hashes - set(seen)
        if missing:
            raise ReviewError(
                FailKind.CONTRACT, f"reconciliation missing prior hashes: {sorted(missing)}"
            )
        carried = {f.prior_hash for f in findings if f.prior_hash}
        still_valid = {h for h, st in seen.items() if st == "STILL_VALID"}
        if carried != still_valid:
            raise ReviewError(
                FailKind.CONTRACT,
                f"STILL_VALID set {sorted(still_valid)} != carried findings {sorted(carried)}",
            )
    oos = raw.get("out_of_scope_findings") or []
    return ReviewerOutput(
        summary=str(raw.get("summary") or "").strip(),
        findings=findings,
        out_of_scope=[o for o in oos if isinstance(o, dict)],
        review_phase=phase,
        head_sha=out_head,
        prior_reconciliation=[r for r in recon if isinstance(r, dict)],
        raw=raw,
    )


def parse_verifier_output(
    raw: dict[str, Any], *, expected_phase: str, expected_coderabbit_ids: list[int]
) -> VerifierOutput:
    if raw.get("adjudication_complete") is not True:
        raise ReviewError(FailKind.CONTRACT, "verifier: adjudication_complete must be true")
    adj = raw.get("prerequisite_adjudications")
    if not isinstance(adj, list):
        raise ReviewError(FailKind.CONTRACT, "verifier: prerequisite_adjudications must be a list")
    phase = str(raw.get("review_phase") or "")
    if phase != expected_phase:
        raise ReviewError(
            FailKind.CONTRACT, f"verifier review_phase={phase!r}, expected {expected_phase!r}"
        )
    if "@coderabbitai review" in json.dumps(raw).lower():
        raise ReviewError(
            FailKind.CONTRACT, "verifier output contains a forbidden CodeRabbit retrigger"
        )
    findings_raw = raw.get("findings")
    if not isinstance(findings_raw, list):
        raise ReviewError(FailKind.CONTRACT, "verifier: findings must be a list")
    findings = [Finding.from_dict(f) for f in findings_raw]
    reactions = raw.get("coderabbit_reactions")
    if not isinstance(reactions, list):
        raise ReviewError(FailKind.CONTRACT, "verifier: coderabbit_reactions must be a list")
    by_id: dict[int, dict[str, Any]] = {}
    for r in reactions:
        if not isinstance(r, dict) or r.get("comment_id") is None:
            raise ReviewError(FailKind.CONTRACT, "coderabbit reaction without comment_id")
        cid = int(r["comment_id"])
        if cid in by_id:
            raise ReviewError(FailKind.CONTRACT, f"duplicate coderabbit reaction {cid}")
        if r.get("action") not in {"agree", "disagree", "extend"}:
            raise ReviewError(FailKind.CONTRACT, f"invalid coderabbit action {r.get('action')!r}")
        if r["action"] in {"disagree", "extend"} and not str(r.get("reply") or "").strip():
            raise ReviewError(
                FailKind.CONTRACT, f"coderabbit {r['action']} on {cid} requires a reply"
            )
        by_id[cid] = r
    if set(by_id) != set(expected_coderabbit_ids):
        raise ReviewError(
            FailKind.CONTRACT,
            f"coderabbit reactions cover {sorted(by_id)}, expected {sorted(expected_coderabbit_ids)}",
        )
    action = str(raw.get("review_action") or "COMMENT").upper()
    return VerifierOutput(
        summary=str(raw.get("summary") or "").strip(),
        review_action=action,
        findings=findings,
        dropped=[d for d in (raw.get("dropped_findings") or []) if isinstance(d, dict)],
        out_of_scope=[o for o in (raw.get("out_of_scope_findings") or []) if isinstance(o, dict)],
        coderabbit_reactions=list(by_id.values()),
        prerequisite_adjudications=[a for a in adj if isinstance(a, dict)],
        adjudication_complete=True,
        review_phase=phase,
        raw=raw,
    )
