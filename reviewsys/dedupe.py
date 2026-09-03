"""Within-batch same-root collapse and cross-round duplicate matching.

Ported from the legacy review_dedupe_guard + review_poster matching logic.
Pure functions; GitHub records are passed in as plain dicts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .contract import Finding, finding_dedupe_key, finding_hash, normalize_finding_text

_CARRY_FORWARD_RE = re.compile(r"prior[-\s]?(\d+(?:\s*/\s*\d+)*)", re.IGNORECASE)
_NAMED_CARRY_FORWARD_RE = re.compile(r"prior-([a-z][a-z0-9]*(?:-[a-z0-9]+)+)", re.IGNORECASE)
_SEVERITY_RANK = {"blocking": 3, "suggestion": 2, "nitpick": 1}
_ROOT_SLUG_STRIP_RE = re.compile(r"[^A-Za-z0-9._:-]+")
FINDING_MARKER_RE = re.compile(
    r"thepastaclaw-review v1\s+finding=([a-f0-9]+)(?:\s+dedupe=([A-Za-z0-9]+))?(?:\s+root=([A-Za-z0-9._:-]+))?"
)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "the",
        "of",
        "in",
        "on",
        "for",
        "to",
        "is",
        "are",
        "be",
        "was",
        "were",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "by",
        "may",
        "can",
        "could",
        "should",
        "would",
        "after",
        "before",
        "with",
        "without",
        "or",
        "but",
        "if",
        "when",
        "while",
        "during",
    }
)
_SIMILAR_TEXT_THRESHOLD = 0.5
_MIN_CONTENT_TOKENS_FOR_TITLE_MATCH = 4
_RESOLUTION_PHRASES_RE = re.compile(
    r"\b(addressed\s+in|already\s+addressed|fixed\s+in|resolved\s+in|out\s+of\s+scope|out-of-scope|won['’]?t\s+fix|wontfix|not\s+applicable|won['’]?t\s+address)\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(not|no|never|wonder|maybe|might|could|should|whether|if|n['’]t|isn['’]?t|hasn['’]?t|haven['’]?t|don['’]?t|doesn['’]?t|didn['’]?t|wouldn['’]?t|couldn['’]?t|shouldn['’]?t)\b",
    re.IGNORECASE,
)


def normalize_root_id(value: str | None) -> str | None:
    if value is None:
        return None
    slug = _ROOT_SLUG_STRIP_RE.sub("-", str(value).strip()).strip("-")
    return slug or None


def carry_forward_tokens(*texts: str | None) -> set[str]:
    tokens: set[str] = set()
    for text in texts:
        if not text:
            continue
        for m in _CARRY_FORWARD_RE.finditer(text):
            for num in re.findall(r"\d+", m.group(1)):
                tokens.add(f"prior-{num}")
        for m in _NAMED_CARRY_FORWARD_RE.finditer(text):
            tokens.add(f"prior-{m.group(1).lower()}")
    return tokens


def _span(f: Finding) -> tuple[int | None, int | None]:
    s, e = f.line_start, f.line_end
    if e is None:
        e = s
    if s is None:
        s = e
    if s is None:
        return (None, None)
    return (min(s, e), max(s, e)) if e is not None else (s, s)


def _spans_overlap(a: tuple[int | None, int | None], b: tuple[int | None, int | None]) -> bool:
    if a[0] is None or b[0] is None:
        return a[0] is None and b[0] is None
    return a[0] <= (b[1] or b[0]) and b[0] <= (a[1] or a[0])


def _same_root(a: Finding, b: Finding) -> bool:
    if a.file != b.file or not _spans_overlap(_span(a), _span(b)):
        return False
    ta, tb = carry_forward_tokens(a.title, a.body), carry_forward_tokens(b.title, b.body)
    if ta and tb and ta & tb:
        return True
    if a.title and b.title and a.hash == b.hash:
        return True
    return bool((a.title or a.body) and (b.title or b.body) and a.dedupe_key == b.dedupe_key)


def _clusters(findings: list[Finding]) -> list[list[int]]:
    n = len(findings)
    parent = list(range(n))
    root_id = [normalize_root_id(f.root_id) for f in findings]
    cluster_id: dict[int, str | None] = dict(enumerate(root_id))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        ida, idb = cluster_id[ra], cluster_id[rb]
        if ida is not None and idb is not None and ida != idb:
            return
        keep, drop = (ra, rb) if ra < rb else (rb, ra)
        parent[drop] = keep
        cluster_id[keep] = ida or idb

    by_id: dict[str, list[int]] = {}
    for i, rid in enumerate(root_id):
        if rid is not None:
            by_id.setdefault(rid, []).append(i)
    for members in by_id.values():
        for j in members[1:]:
            union(members[0], j)
    for i in range(n):
        for j in range(i + 1, n):
            if _same_root(findings[i], findings[j]):
                union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [groups[r] for r in sorted(groups, key=lambda r: min(groups[r]))]


def _representative(members: list[int], findings: list[Finding]) -> int:
    def key(i: int) -> tuple[int, int, int, str, int]:
        f = findings[i]
        return (
            0 if f.root_id else 1,
            -_SEVERITY_RANK.get(f.severity, 0),
            -len(f.body),
            f.dedupe_key,
            i,
        )

    return min(members, key=key)


@dataclass(slots=True)
class Suppressed:
    finding: Finding
    action: str
    detail: str = ""
    comment_id: str | None = None
    html_url: str | None = None


def collapse_same_root(findings: list[Finding]) -> tuple[list[Finding], list[Suppressed]]:
    if not findings:
        return [], []
    kept: list[Finding] = []
    collapsed: list[Suppressed] = []
    for members in _clusters(findings):
        rep = _representative(members, findings)
        kept.append(findings[rep])
        for i in members:
            if i != rep:
                collapsed.append(
                    Suppressed(findings[i], "deduped_same_batch_root", detail=findings[rep].title)
                )
    return kept, collapsed


def has_same_root_duplicates(findings: list[Finding]) -> bool:
    return any(len(m) > 1 for m in _clusters(findings))


# ---- cross-round matching against existing GitHub inline comments ----


@dataclass(slots=True)
class ExistingComment:
    id: str
    node_id: str
    finding_hash: str
    dedupe_key: str
    root_id: str | None
    cf_tokens: set[str]
    path: str
    line: int | None
    norm_title: str
    norm_body: str
    html_url: str

    @classmethod
    def from_github(cls, c: dict[str, Any]) -> ExistingComment:
        body = str(c.get("body") or "")
        m = FINDING_MARKER_RE.search(body)
        title = ""
        for line in body.splitlines():
            s = line.strip()
            if s.startswith("**") and s.endswith("**"):
                title = s.strip("*")
                if ":" in title:
                    title = title.split(":", 1)[1]
                title = title.strip()
                break
        return cls(
            id=str(c.get("id", "")),
            node_id=str(c.get("node_id") or ""),
            finding_hash=m.group(1) if m else "",
            dedupe_key=(m.group(2) if m else "") or "",
            root_id=normalize_root_id(m.group(3)) if m and m.group(3) else None,
            cf_tokens=carry_forward_tokens(title, body),
            path=str(c.get("path") or ""),
            line=c.get("line") or c.get("original_line"),
            norm_title=normalize_finding_text(title),
            norm_body=normalize_finding_text(body),
            html_url=str(c.get("html_url") or ""),
        )


def _content_tokens(text: str) -> set[str]:
    return {t for t in text.split() if t and t not in _STOPWORDS}


def _jaccard(a: str, b: str) -> float:
    sa, sb = _content_tokens(a), _content_tokens(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def find_duplicate(
    f: Finding, records: list[ExistingComment]
) -> tuple[ExistingComment | None, str | None]:
    fhash, dkey = f.hash, f.dedupe_key
    nt, nb = normalize_finding_text(f.title), normalize_finding_text(f.body)
    rid = normalize_root_id(f.root_id)
    cf = carry_forward_tokens(f.title, f.body)
    ls, le = f.line_start or 0, f.line_end or f.line_start or 0
    if rid:
        for r in records:
            if r.root_id == rid:
                return r, "root_id"
    for r in records:
        if r.finding_hash and r.finding_hash == fhash:
            return r, "finding_hash"
    for r in records:
        if r.dedupe_key and r.dedupe_key == dkey:
            return r, "dedupe_key"
    if cf:
        for r in records:
            if r.path == f.file and r.line and r.line in (ls, le) and cf & r.cf_tokens:
                return r, "carry_forward_root"
    for r in records:
        if r.path != f.file:
            continue
        if r.line and r.line in (ls, le):
            if _jaccard(nt, r.norm_title) >= _SIMILAR_TEXT_THRESHOLD:
                return r, "same_line_similar_text"
            if nb and _jaccard(nb, r.norm_body) >= _SIMILAR_TEXT_THRESHOLD:
                return r, "same_line_similar_text"
    if len(_content_tokens(nt)) >= _MIN_CONTENT_TOKENS_FOR_TITLE_MATCH:
        for r in records:
            if (
                r.path == f.file
                and len(_content_tokens(r.norm_title)) >= _MIN_CONTENT_TOKENS_FOR_TITLE_MATCH
                and _jaccard(nt, r.norm_title) >= _SIMILAR_TEXT_THRESHOLD
            ):
                return r, "similar_title"
    return None, None


def thread_has_resolution_reply(thread: dict[str, Any] | None, bot_login: str) -> bool:
    if not thread:
        return False
    for node in (thread.get("comments") or {}).get("nodes") or []:
        if (node.get("author") or {}).get("login", "") == bot_login:
            continue
        for sentence in re.split(r"[.!?\n]+", node.get("body") or ""):
            if _RESOLUTION_PHRASES_RE.search(sentence) and not _NEGATION_RE.search(sentence):
                return True
    return False


__all__ = [
    "FINDING_MARKER_RE",
    "ExistingComment",
    "Suppressed",
    "carry_forward_tokens",
    "collapse_same_root",
    "find_duplicate",
    "finding_dedupe_key",
    "finding_hash",
    "has_same_root_duplicates",
    "normalize_root_id",
    "thread_has_resolution_reply",
]
