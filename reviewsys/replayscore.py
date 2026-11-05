"""`reviewsys replay-score`: score a replay set's reports against labels, arm against arm.

Labels are JSON (a list, or `{"labels": [...]}`) or JSON Lines, as eval/labels.jsonl has them:

    {"repo": "dashpay/dash", "number": 7683, "sha": "<replayed head>", "head_id": "...",
     "file": "src/x.cpp", "line_start": 247, "line_end": 249, "title": "...",
     "label": "tp|fp|deferred|none", "set": "S1".."S8", "source": "..."}

A label belongs to the head of the set whose `id` is its `head_id` (else to the replay of its
repo, number and sha). Optional fields read: `finding_hash` (or `hash`) and `category` (also
match through `compare._Finding.matches`), `round_kind` (S5: `settling_round` or `post_settle`,
scored as subsets `S5:<kind>` too), `anchor_changed` (true when the code at the anchor changed
since the issue was settled; re-raises count on unchanged anchors only, the default), and
`thread_outcome` (S8: the outcome the thread lane should reach). `label: none` rows only mark a
head as checks-only (S6, S7).

Matching, per replay sample, one to one (a finding answers at most one label and the other way
round; best title overlap first): same file within 10 lines with title-token Jaccard at least
0.2, or `compare._Finding.matches` (same hash; or same file and category with overlapping
lines, and then some title overlap). What a replay *published* is scored (`report.published`).

Metrics per arm and set, on paired heads only (every arm has a usable sample of the head),
every per-head quantity being the mean over that head's samples: recall of `tp` labels found
as a blocker or suggestion (and blocker-only), label-weighted; known-FP blockers (`fp` and
`deferred` labels raised as blockers); re-raises of settled issues on unchanged anchors (any
severity); the share of heads with a blocker; new blockers (matching no label); S8 thread
outcomes reached (from either shape of conversation.json) and, per S8 head, the #38 verdict
follow-up the run decided on (`would_post`: its event, `none`, or `not_recorded` for an arm
that records none, as v11). `S1+S2` pools the two recall sets, as the gate does. Per arm:
duplicates per run (same file within 3 lines, different hash), gpt lanes and tokens per run,
wall clock p50, contract failures, ratios to the baseline.

Wall clock: replays share the box's lane pools behind production, so `active_p50_s` (wall
clock less the run's slot wait, `report.slot_wait_s`) is what a wall-clock gate should read;
both are reported. Thread state: a head where the pinned view could not rewind whether a
thread was resolved (`report.thread_resolution_as_of_now`) is listed in
`heads.thread_resolution_as_of_now`, and S5 and S8 are scored twice: every head (`S5`) and
only the exact ones (`S5:exact_threads`).

Noise: on heads with repeat samples, how far an arm's recall moves between its own two samples
(`recall_delta_pp`, the floor being the largest over arms), and a bootstrap over heads of the
paired recall difference to the baseline (95% interval, in pp).

The judge (judge.py, `reviewsys replay-judge`): what the matcher cannot settle goes to
`to_judge.jsonl`, blinded (no arm; the arm key is in `to_judge.key.json`) and shuffled, with
ids derived from the content (stable from one scoring to the next):
- `new_blocker`: a published blocker matching no label, clustered across arms and samples per
  head (one item per cluster of matching findings): is it a real defect at this head?
- `label_match`: a published finding near an unmatched label (same file within 10 lines) whose
  title does not match it: does it describe the label's defect?
The judge's answers (`judged.jsonl`, the items with a `verdict`) are read back by content, not
id: a judged `label_match` with `matches_label: yes` makes that finding answer that label, and
a judged `new_blocker` verdict applies to every new blocker on the head it matches. Per arm and
set, `new_blocker_precision` is real / (real + not real) over the judged new blockers (unsure
and not yet judged ones are counted apart), and `blocker_precision` adds the blockers that
matched a `tp` label (real) or an `fp` / `deferred` one (not). Only items no verdict covers yet
are written to `to_judge.jsonl`; with verdicts, `spot_check.jsonl` holds 20 judged new
blockers (blinded) for the owner's check.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import re
import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .compare import _Finding
from .replayset import (
    MANIFEST,
    Job,
    Policy,
    SetItem,
    limit_items,
    load_records,
    load_report,
    load_set,
)

LINE_SLACK = 10
JACCARD_MIN = 0.2
DUPLICATE_SLACK = 3
LABELS = ("tp", "fp", "deferred", "none")
COUNTED = ("blocking", "suggestion")  # recall counts a label found at either severity
POOLED = {"S1": "S1+S2", "S2": "S1+S2"}
DEFERRALS = frozenset({"INTENTIONALLY_DEFERRED", "DEFERRAL_PENDING"})
BOOTSTRAP = 2000
SPOT_CHECK = 20
TO_JUDGE, JUDGED = "to_judge.jsonl", "judged.jsonl"
NEW_BLOCKER, LABEL_MATCH = "new_blocker", "label_match"
ANSWERS = ("yes", "no", "unsure")
THREAD_SETS = frozenset({"S5", "S8"})  # scored again on heads with exact thread state
EXACT = "exact_threads"
_STOP = frozenset(
    "a an and are as at be but by can for from has in is it its not of on or so the this to "  # noqa: SIM905
    "was when where which with without".split()
)


@dataclass(frozen=True, slots=True)
class Label:
    repo: str
    number: int
    sha: str
    file: str
    line_start: int | None
    line_end: int | None
    title: str
    label: str
    set: str
    source: str = ""
    hash: str = ""
    category: str = ""
    head_id: str = ""
    round_kind: str = ""
    anchor_changed: bool = False
    thread_outcome: str = ""

    @property
    def head(self) -> tuple[str, int, str]:
        return (self.repo.lower(), self.number, self.sha.lower())

    @property
    def buckets(self) -> list[str]:
        """The sets this label is scored in."""
        out = [self.set]
        if self.round_kind:
            out.append(f"{self.set}:{self.round_kind}")
        if self.set in POOLED:
            out.append(POOLED[self.set])
        return out


def load_labels(path: Path) -> list[Label]:
    out = []
    for i, e in enumerate(load_records(path, "labels")[0]):
        try:
            lab = Label(
                repo=str(e["repo"]),
                number=int(e["number"]),
                sha=str(e["sha"]),
                file=str(e.get("file") or ""),
                line_start=_int(e.get("line_start")),
                line_end=_int(e.get("line_end")),
                title=str(e.get("title") or ""),
                label=str(e["label"]).lower(),
                set=str(e.get("set") or ""),
                source=str(e.get("source") or ""),
                hash=str(e.get("finding_hash") or e.get("hash") or ""),
                category=str(e.get("category") or ""),
                head_id=str(e.get("head_id") or ""),
                round_kind=str(e.get("round_kind") or ""),
                anchor_changed=bool(e.get("anchor_changed", False)),
                thread_outcome=str(e.get("thread_outcome") or ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"{path}: label {i} is malformed ({exc}): {e!r}") from exc
        if lab.label not in LABELS:
            raise SystemExit(f"{path}: label {i}: `label` must be one of {LABELS}")
        out.append(lab)
    return out


def _int(v: Any) -> int | None:
    try:
        return None if v is None or v == "" else int(v)
    except (TypeError, ValueError):
        return None


# ---- matching ----


def title_tokens(title: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_]+", title.lower()) if len(t) > 1 and t not in _STOP}


def jaccard(a: str, b: str) -> float:
    x, y = title_tokens(a), title_tokens(b)
    return len(x & y) / len(x | y) if x and y else 0.0


def _span(lo: Any, hi: Any) -> tuple[int, int] | None:
    lo, hi = _int(lo), _int(hi)
    if lo is None:
        return None
    return lo, hi if hi is not None and hi >= lo else lo


def near(a: dict[str, Any], b: dict[str, Any], slack: int) -> bool:
    """Same file, line ranges within `slack` lines of each other (or both without lines)."""
    if not a.get("file") or a.get("file") != b.get("file"):
        return False
    sa = _span(a.get("line_start"), a.get("line_end"))
    sb = _span(b.get("line_start"), b.get("line_end"))
    if sa is None or sb is None:
        return sa is None and sb is None
    return sa[0] - slack <= sb[1] and sb[0] - slack <= sa[1]


def _as_compare(d: dict[str, Any]) -> _Finding:
    span = _span(d.get("line_start"), d.get("line_end"))
    return _Finding(
        hash=str(d.get("hash") or ""),
        file=str(d.get("file") or ""),
        category=str(d.get("category") or ""),
        lo=span[0] if span else None,
        hi=span[1] if span else None,
        severity=str(d.get("severity") or ""),
    )


def matches(finding: dict[str, Any], label: dict[str, Any]) -> bool:
    j = jaccard(str(finding.get("title") or ""), str(label.get("title") or ""))
    if near(finding, label, LINE_SLACK) and j >= JACCARD_MIN:
        return True
    a, b = _as_compare(finding), _as_compare(label)
    if a.hash and a.hash == b.hash:
        return True
    # else compare's same file + category + overlapping lines, and some title overlap; never
    # on an empty category, and never on two empty hashes (`_Finding.matches` takes those as
    # equal)
    return bool(a.category) and j > 0 and dataclasses.replace(a, hash="\0").matches(b)


def assign(
    findings: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    also: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """One-to-one: (label index -> the finding answering it, findings answering none). The
    pairs with the best title overlap (an equal hash first) are taken first. `also` adds pairs
    the matcher misses (the judge said they are the same defect)."""
    pairs = []
    for fi, f in enumerate(findings):
        for li, lab in enumerate(labels):
            if matches(f, lab) or (also is not None and also(f, lab)):
                same = bool(f.get("hash")) and f.get("hash") == lab.get("hash")
                pairs.append((-(same + jaccard(str(f.get("title")), str(lab["title"]))), fi, li))
    taken: set[int] = set()
    got: dict[int, dict[str, Any]] = {}
    for _, fi, li in sorted(pairs):
        if fi not in taken and li not in got:
            taken.add(fi)
            got[li] = findings[fi]
    return got, [f for i, f in enumerate(findings) if i not in taken]


# ---- reports ----


@dataclass(slots=True)
class Sample:
    policy: str
    item: SetItem
    sample: int
    report: dict[str, Any]

    @property
    def usable(self) -> bool:
        """Done, valid, and not degraded for scoring (a v12 thread-lane shard failed, so some
        issues were never judged: `report.degraded_for_scoring`)."""
        r = self.report
        return (
            r.get("status") == "done" and not r.get("invalid") and not r.get("degraded_for_scoring")
        )

    @property
    def head(self) -> tuple[str, int, str]:
        r = self.report
        return (str(r["repo"]).lower(), int(r["number"]), str(r["sha"]).lower())

    @property
    def published(self) -> list[dict[str, Any]]:
        return [f for f in self.report.get("published") or [] if isinstance(f, dict)]


def load_policies(out: Path) -> list[str]:
    manifest = json.loads((out / MANIFEST).read_text()) if (out / MANIFEST).exists() else None
    if isinstance(manifest, dict) and manifest.get("policies"):
        return [str(p["name"]) for p in manifest["policies"]]
    return sorted(p.name for p in out.iterdir() if p.is_dir() and any(p.glob("run-*")))


def load_samples(items: list[SetItem], policies: list[str], out: Path) -> list[Sample]:
    found = []
    for item in items:
        for name in policies:
            for k in range(item.samples):
                rep = load_report(Job(item, Policy(name, Path()), k), out)
                if rep is not None:
                    found.append(Sample(name, item, k, rep))
    return found


# ---- judge verdicts ----

HeadKey = tuple[str, int, str, str]  # (repo, number, sha, head key)


def _label_key(lab: dict[str, Any]) -> list[Any]:
    return [
        lab.get("hash") or "",
        lab.get("title") or "",
        lab.get("file") or "",
        lab.get("line_start"),
    ]


# ---- blinding: what only one arm writes around a finding ----

# phrases only one policy puts in a finding's title or body (rules.apply_rules under v12:
# revivals, regressions, unconfirmed blockers, merged groups' suggestions; links to the bot's
# earlier threads): never shown to the judge. tests/test_judge.py checks them against rules.py
ARM_PHRASES = (
    "This was raised before",
    "looks like a regression",
    "(regression)",
    "Not confirmed as blocking",
    "offered as a one-click suggestion",
    "#discussion_r",
)
_ARM_LINES = re.compile(
    r"\n*(?:_This was raised before and[^\n]*|_This looks like a regression of[^\n]*"
    r"|\*\*Not confirmed as blocking\.\*\*[^\n]*)"
)
_SUGGESTION_HEAD = re.compile(
    r"\*\*Suggested change\*\* \(the reports point at different lines, so it is not offered as "
    r"a one-click suggestion\):"
)
_THREAD_LINK = re.compile(r"\s*\(?https://github\.com/[^\s)]+#discussion_r\d+\)?")
_REGRESSION = re.compile(r"\s*\(regression\)\s*$")


def neutral_title(title: str) -> str:
    return _REGRESSION.sub("", title or "")


def neutral_body(body: str) -> str:
    """The defect claim without the bot-history framing one arm adds (the sentences about an
    earlier, withdrawn or fixed finding and links to its thread, the unconfirmed-blocker note,
    the merged-group suggestion header)."""
    text = _ARM_LINES.sub("", body or "")
    text = _SUGGESTION_HEAD.sub("**Suggested change**:", text)
    return _THREAD_LINK.sub("", text).strip()


def _brief(f: dict[str, Any]) -> dict[str, Any]:
    """A finding's identity for the judge's bookkeeping (absent text fields read as ""), its
    title as the judge sees it (`neutral_title`)."""
    return {
        "file": f.get("file") or "",
        "line_start": _int(f.get("line_start")),
        "line_end": _int(f.get("line_end")),
        "title": neutral_title(str(f.get("title") or "")),
        "category": f.get("category") or "",
    }


def judge_id(
    kind: str, head: HeadKey, finding: dict[str, Any], label: dict[str, Any] | None
) -> str:
    """A judge item's id, from what it asks about (stable across scorings)."""
    blob = json.dumps(
        [kind, list(head), _brief(finding), _label_key(label) if label else None],
        sort_keys=True,
        default=str,
    )
    return f"{'m' if kind == LABEL_MATCH else 'j'}-{hashlib.sha256(blob.encode()).hexdigest()[:12]}"


def _answer(v: Any) -> str:
    s = str(v or "").strip().lower()
    return s if s in ANSWERS else ""


@dataclass(slots=True)
class Verdicts:
    """The judge's answers (judged.jsonl), looked up by content."""

    real: dict[HeadKey, list[tuple[dict[str, Any], str]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    same: dict[HeadKey, list[tuple[list[Any], dict[str, Any], str]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    count: int = 0

    @classmethod
    def load(cls, items: list[dict[str, Any]]) -> Verdicts:
        v = cls()
        for it in items:
            verdict = it.get("verdict")
            if not isinstance(verdict, dict):
                continue
            head = (
                str(it["repo"]).lower(),
                int(it["number"]),
                str(it["sha"]).lower(),
                str(it.get("head_id") or ""),
            )
            v.count += 1
            if it.get("kind") == LABEL_MATCH and isinstance(it.get("label"), dict):
                v.same[head].append(
                    (_label_key(it["label"]), it, _answer(verdict.get("matches_label")))
                )
            if it.get("kind", NEW_BLOCKER) == NEW_BLOCKER:
                v.real[head].append((it, _answer(verdict.get("real"))))
        return v

    def real_of(self, head: HeadKey, f: dict[str, Any]) -> str | None:
        """The judge's `real` answer for a new blocker (None: not judged): the judged item's
        cluster has this very finding, or the finding matches the one judged."""
        brief = _brief(f)
        for item, ans in self.real.get(head, []):
            if brief in (item.get("members") or [_brief(item)]) or matches(f, item):
                return ans
        return None

    def same_of(self, head: HeadKey, f: dict[str, Any], lab: dict[str, Any]) -> str | None:
        """The judge's `matches_label` answer for this finding and label (None: not asked)."""
        key = _label_key(lab)
        for k, item, ans in self.same.get(head, []):
            if k == key and _brief(item) == _brief(f):
                return ans
        return None


def load_judged(out: Path) -> list[dict[str, Any]]:
    path = out / JUDGED
    if not path.exists():
        return []
    items = []
    for line in path.read_text().splitlines():
        try:
            it = json.loads(line)
        except ValueError:
            continue  # a line cut by a crash mid-write; the judge re-asks it
        if isinstance(it, dict):
            items.append(it)
    return items


# ---- scoring ----


@dataclass(slots=True)
class _Acc:
    """Running totals of one arm over one set; every per-head number is a mean over samples."""

    heads: set[str] = field(default_factory=set)
    blocker_heads: list[float] = field(default_factory=list)  # share of samples with a blocker
    tp: list[float] = field(default_factory=list)  # per tp label: share of samples finding it
    tp_blocker: list[float] = field(default_factory=list)
    fp_blockers: float = 0.0
    reraised: float = 0.0
    new_blockers: float = 0.0
    duplicate_threads: float = 0.0  # new threads on issues with an open thread (S5 gate: 0)
    outcomes: list[float] = field(default_factory=list)  # per S8 label: share reaching it
    # per conversation (S8) head: the #38 follow-up its samples decide on (`would_post`)
    would_post: dict[str, float] = field(default_factory=dict)
    # blockers by what the judge (new ones) or a label (matched ones) says of them
    judged: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys((*ANSWERS, "pending"), 0.0)
    )
    labelled_blockers: dict[str, float] = field(default_factory=lambda: {"tp": 0.0, "not": 0.0})

    def render(self) -> dict[str, Any]:
        j, lb = self.judged, self.labelled_blockers
        return {
            "heads": len(self.heads),
            "tp_labels": len(self.tp),
            "recall": _mean(self.tp),
            "blocker_recall": _mean(self.tp_blocker),
            "fp_blockers": round(self.fp_blockers, 3),
            "reraised_unchanged": round(self.reraised, 3),
            "heads_with_blocker_rate": _mean(self.blocker_heads),
            "new_blockers": round(self.new_blockers, 3),
            "duplicate_threads": round(self.duplicate_threads, 3),
            "new_blockers_judged": {k: round(v, 3) for k, v in j.items()},
            "new_blocker_precision": _ratio(j["yes"], j["yes"] + j["no"]),
            "blocker_precision": _ratio(
                j["yes"] + lb["tp"], j["yes"] + j["no"] + lb["tp"] + lb["not"]
            ),
            "thread_outcomes_reached": _mean(self.outcomes),
            "thread_outcome_labels": len(self.outcomes),
            **(
                {"would_post": {k: round(v, 3) for k, v in sorted(self.would_post.items())}}
                if self.would_post
                else {}
            ),
        }


def _ratio(a: float, b: float) -> float | None:
    return round(a / b, 4) if b else None


@dataclass(slots=True)
class _Noise:
    """One arm against itself on heads with repeat samples (samples 0 and 1)."""

    heads: int = 0
    labels: int = 0  # tp labels on those heads
    found: list[int] = field(default_factory=lambda: [0, 0])  # by sample 0 and sample 1
    flips: int = 0  # labels found in some samples and not others

    def render(self) -> dict[str, Any]:
        delta = abs(self.found[0] - self.found[1]) / self.labels * 100 if self.labels else None
        return {
            "repeat_heads": self.heads,
            "tp_labels": self.labels,
            "recall_delta_pp": None if delta is None else round(delta, 1),
            "label_flips": self.flips,
        }


def _mean(xs: list[float], digits: int = 4) -> float | None:
    return round(statistics.fmean(xs), digits) if xs else None


def _frac(xs: list[bool]) -> float:
    return sum(xs) / len(xs)


def duplicate_threads(sample: Sample) -> int:
    """New threads (inline comments) the run would open on an issue that already had an open
    bot thread when it started (`report.prior_open_threads`): same hash, or the matcher's
    same-defect rule."""
    prior = [p for p in sample.report.get("prior_open_threads") or [] if isinstance(p, dict)]
    n = 0
    for c in sample.report.get("inline") or []:
        if isinstance(c, dict) and any(
            (c.get("hash") and c.get("hash") == p.get("hash")) or matches(c, p) for p in prior
        ):
            n += 1
    return n


def duplicates(published: list[dict[str, Any]]) -> int:
    """Pairs of published findings on the same file within 3 lines and different hashes."""
    n = 0
    for i, a in enumerate(published):
        for b in published[i + 1 :]:
            if a.get("hash") != b.get("hash") and near(a, b, DUPLICATE_SLACK):
                n += 1
    return n


def conversation_outcomes(conv: Any) -> dict[str, Any]:
    """A replay's `conversation` (its conversation.json) as outcomes by hash, either shape:
    v11's conversation lane writes them flat, v12's thread lane under `outcomes` (beside
    `work`, `shards` and, in a replay, `would_post`)."""
    if not isinstance(conv, dict):
        return {}
    inner = conv.get("outcomes")
    return inner if isinstance(inner, dict) else conv


def would_post(sample: Sample) -> str:
    """The #38 verdict follow-up a replayed conversation decided on (conversation.json
    `would_post`, v12): its event, `none` when it would post none, `not_recorded` when the
    run records no decision (v11: a replay's conversation lane delivers nothing, so its
    follow-up is never decided)."""
    conv = sample.report.get("conversation")
    if not isinstance(conv, dict) or "would_post" not in conv:
        return "not_recorded"
    wp = conv["would_post"]
    return str(wp.get("event") or "none") if isinstance(wp, dict) else "none"


def _outcome_reached(sample: Sample, lab: Label) -> bool:
    got = conversation_outcomes(sample.report.get("conversation")).get(lab.hash)
    status = str((got or {}).get("status") or "") if isinstance(got, dict) else ""
    want = lab.thread_outcome
    return status == want or (status in DEFERRALS and want in DEFERRALS)


def _head_key(item: SetItem) -> str:
    return item.head_id or f"run-{item.run_id}"


def score(
    items: list[SetItem],
    labels: list[Label],
    samples: list[Sample],
    policies: list[str],
    *,
    baseline: str | None = None,
    seed: int = 0,
    judged: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """(score, to-judge items not yet judged, blinding key). `judged` are the judge's
    answers (judged.jsonl)."""
    verdicts = Verdicts.load(judged or [])
    baseline = baseline or (policies[0] if policies else None)
    by_id: dict[str, list[Label]] = defaultdict(list)
    by_sha: dict[tuple[str, int, str], list[Label]] = defaultdict(list)
    ids = {i.head_id for i in items if i.head_id}
    for lab in labels:
        if lab.head_id and lab.head_id in ids:
            by_id[lab.head_id].append(lab)
        else:
            by_sha[lab.head].append(lab)
    usable: dict[tuple[int, str], list[Sample]] = defaultdict(list)
    for s in samples:
        if s.usable:
            usable[(s.item.run_id, s.policy)].append(s)
    paired = [i for i in items if policies and all((i.run_id, p) in usable for p in policies)]
    per_set: dict[str, dict[str, _Acc]] = defaultdict(lambda: defaultdict(_Acc))
    # bucket -> arm -> head -> (tp found, tp labels), for the paired bootstrap
    contrib: dict[str, dict[str, dict[str, tuple[float, int]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    noise = {p: _Noise() for p in policies}
    prec = {p: _Acc() for p in policies}  # every paired head: the arm's blocker precision
    judge_pool: dict[HeadKey, list[tuple[str, int, dict[str, Any]]]] = defaultdict(list)
    match_pool: dict[tuple[HeadKey, str], tuple[dict[str, Any], dict[str, Any], set[Any]]] = {}
    bases: dict[HeadKey, dict[str, Any]] = {}
    labelled = 0
    as_of_now: set[int] = set()  # heads whose thread resolution the view could not rewind
    for item in paired:
        key = _head_key(item)
        if any(
            int(s.report.get("thread_resolution_as_of_now") or 0) > 0
            for p in policies
            for s in usable[(item.run_id, p)]
        ):
            as_of_now.add(item.run_id)
        exact = item.run_id not in as_of_now

        def also(st: str, exact: bool = exact) -> list[str]:
            """S5 / S8 are reported twice: every head, and only the heads whose threads read
            as they stood at the cutoff (`<set>:exact_threads`)."""
            return [f"{st}:{EXACT}"] if exact and st in THREAD_SETS else []

        for p in policies:
            runs = sorted(usable[(item.run_id, p)], key=lambda s: s.sample)
            head: HeadKey = (*runs[0].head, key)
            pin = runs[0].report.get("pin") or {}
            bases.setdefault(
                head, {k: pin.get(k) for k in ("merge_base", "base_ref", "base_sha") if pin.get(k)}
            )
            head_labels = by_id.get(item.head_id, []) if item.head_id else []
            head_labels = head_labels + by_sha.get(runs[0].head, [])
            if p == policies[0]:
                labelled += len(head_labels)
            scored = [lab for lab in head_labels if lab.label != "none" and lab.file]
            dicts = [dataclasses.asdict(lab) for lab in scored]

            def same(f: dict[str, Any], lab: dict[str, Any], head: HeadKey = head) -> bool:
                return verdicts.same_of(head, f, lab) == "yes"

            assigned = [assign(r.published, dicts, same) for r in runs]
            for st in (item.set, *also(item.set)):
                acc = per_set[st][p]
                acc.heads.add(key)
                acc.blocker_heads.append(
                    _frac([any(f.get("severity") == "blocking" for f in r.published) for r in runs])
                )
                acc.new_blockers += statistics.fmean(
                    sum(f.get("severity") == "blocking" for f in rest) for _, rest in assigned
                )
                acc.duplicate_threads += statistics.fmean(duplicate_threads(r) for r in runs)
                _count_precision(acc, verdicts, head, scored, assigned)
                if item.set == "S8":
                    for r in runs:
                        ev = would_post(r)
                        acc.would_post[ev] = acc.would_post.get(ev, 0.0) + 1 / len(runs)
            _count_precision(prec[p], verdicts, head, scored, assigned)
            for r, (got, rest) in zip(runs, assigned, strict=True):
                for f in rest:
                    if f.get("severity") == "blocking" and verdicts.real_of(head, f) is None:
                        judge_pool[head].append((p, r.sample, f))
                # a finding near a label it did not match: is it the same defect?
                for li, ld in enumerate(dicts):
                    if li in got:
                        continue
                    for f in rest:
                        if near(f, ld, LINE_SLACK) and verdicts.same_of(head, f, ld) is None:
                            jid = judge_id(LABEL_MATCH, head, f, ld)
                            entry = match_pool.setdefault((head, jid), (f, ld, set()))
                            entry[2].add((p, r.sample))
            for li, lab in enumerate(scored):
                hits = [got.get(li) for got, _ in assigned]
                for b in (*lab.buckets, *also(lab.set)):
                    a = per_set[b][p]
                    a.heads.add(key)
                    if lab.label == "tp":
                        found = _frac(
                            [h is not None and h.get("severity") in COUNTED for h in hits]
                        )
                        a.tp.append(found)
                        a.tp_blocker.append(
                            _frac([h is not None and h.get("severity") == "blocking" for h in hits])
                        )
                        f_sum, n = contrib[b][p].get(key, (0.0, 0))
                        contrib[b][p][key] = (f_sum + found, n + 1)
                    else:
                        a.fp_blockers += _frac(
                            [h is not None and h.get("severity") == "blocking" for h in hits]
                        )
                        if not lab.anchor_changed:
                            a.reraised += _frac([h is not None for h in hits])
            for lab in head_labels:
                if lab.thread_outcome and lab.hash:
                    for b in (*lab.buckets, *also(lab.set)):
                        per_set[b][p].outcomes.append(
                            _frac([_outcome_reached(r, lab) for r in runs])
                        )
            if len(runs) > 1:
                tps = [li for li, lab in enumerate(scored) if lab.label == "tp"]
                if tps:
                    nz = noise[p]
                    nz.heads += 1
                    nz.labels += len(tps)
                    for k in (0, 1):
                        got = assigned[k][0]
                        nz.found[k] += sum(
                            got.get(li) is not None and got[li].get("severity") in COUNTED
                            for li in tps
                        )
                    nz.flips += sum(
                        len({(got.get(li) or {}).get("severity") in COUNTED for got, _ in assigned})
                        > 1
                        for li in tps
                    )
    rng = random.Random(seed)
    sets_out: dict[str, Any] = {}
    for st in sorted(per_set, key=lambda s: (s.split(":")[0], s)):
        row: dict[str, Any] = {p: per_set[st][p].render() for p in policies if p in per_set[st]}
        if baseline and baseline in row:
            row["delta_vs_" + baseline] = {
                p: _deltas(row[p], row[baseline])
                | {"recall_ci95_pp": _bootstrap(contrib[st], p, baseline, rng)}
                for p in policies
                if p != baseline and p in row
            }
        sets_out[st or "(unset)"] = row
    noise_out = {p: nz.render() for p, nz in noise.items()}
    deltas = [v["recall_delta_pp"] for v in noise_out.values() if v["recall_delta_pp"] is not None]
    to_judge, key_out = _blind(judge_pool, match_pool, bases, seed=seed)
    arms = {p: _arm([s for s in samples if s.policy == p], paired) for p in policies}
    for p in policies:
        rendered = prec[p].render()
        arms[p]["precision"] = {
            k: rendered[k]
            for k in ("new_blockers_judged", "new_blocker_precision", "blocker_precision")
        }
    result = {
        "policies": policies,
        "baseline": baseline,
        "heads": {
            "in_set": len(items),
            "paired": len(paired),
            "unpaired": sorted(i.run_id for i in items if i not in paired),
            "thread_resolution_as_of_now": sorted(as_of_now),
        },
        "labels": {"total": len(labels), "on_paired_heads": labelled},
        "arms": arms,
        "ratios_vs_baseline": _ratios(arms, baseline),
        "sets": sets_out,
        "noise": {"per_arm": noise_out, "floor_pp": max(deltas) if deltas else None},
        "to_judge": len(to_judge),
        "judged": verdicts.count,
        "matcher": {
            "line_slack": LINE_SLACK,
            "title_jaccard_min": JACCARD_MIN,
            "also": "compare._Finding.matches (with some title overlap)",
            "one_to_one": True,
        },
    }
    return result, to_judge, key_out


def _deltas(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    def pp(x: Any, y: Any) -> float | None:
        ok = isinstance(x, int | float) and isinstance(y, int | float)
        return round((x - y) * 100, 1) if ok else None

    return {
        **{
            k: pp(a.get(k), b.get(k))
            for k in ("recall", "blocker_recall", "heads_with_blocker_rate")
        },
        **{k: round(a[k] - b[k], 3) for k in ("fp_blockers", "reraised_unchanged", "new_blockers")},
    }


def _bootstrap(
    contrib: dict[str, dict[str, tuple[float, int]]], arm: str, base: str, rng: random.Random
) -> list[float] | None:
    """95% interval (pp) of the arm's label-weighted recall minus the baseline's, resampling
    heads with replacement."""
    heads = sorted(set(contrib.get(arm, {})) & set(contrib.get(base, {})))
    if len(heads) < 2:
        return None
    diffs = []
    for _ in range(BOOTSTRAP):
        pick = [rng.choice(heads) for _ in heads]
        n = sum(contrib[base][h][1] for h in pick)
        d = sum(contrib[arm][h][0] - contrib[base][h][0] for h in pick)
        diffs.append(d / n * 100)
    diffs.sort()
    return [round(diffs[int(0.025 * BOOTSTRAP)], 1), round(diffs[int(0.975 * BOOTSTRAP) - 1], 1)]


def _arm(samples: list[Sample], paired: list[SetItem]) -> dict[str, Any]:
    ids = {i.run_id for i in paired}
    every = [s for s in samples if s.item.run_id in ids]
    runs = [s for s in every if s.usable]

    def gpt(s: Sample, *keys: str, block: str = "gpt") -> float:
        got = s.report.get(block) or s.report.get("gpt") or {}
        return sum(float(got.get(k) or 0) for k in keys)

    def sig(s: Sample, key: str) -> float:
        return float((s.report.get("v12") or {}).get(key) or 0)

    candidates = sum(sig(s, "candidates") for s in runs)
    matched = sum(sig(s, "candidates_matched_open") for s in runs)

    walls = [float(s.report.get("wall_clock_s") or 0) for s in runs]
    # wall clock less the time the run only waited for lane slots (shared with production)
    actives = [float(s.report.get("active_s", s.report.get("wall_clock_s")) or 0) for s in runs]
    waits = [float(s.report.get("slot_wait_s") or 0) for s in runs]
    contract = [s.report.get("contract") or {} for s in every]
    return {
        "samples": len(every),
        "usable": len(runs),
        "failed": sum(s.report.get("status") == "failed" for s in every),
        "invalid": sum(bool(s.report.get("invalid")) for s in every),
        # every sample of the arm, paired head or not (such a sample unpairs its head)
        "degraded_for_scoring": sum(bool(s.report.get("degraded_for_scoring")) for s in samples),
        "gpt_lanes_per_run": _mean([gpt(s, "lanes") for s in runs], 1),
        "gpt_tokens_per_run": _mean([gpt(s, "tokens_in", "tokens_out") for s in runs], 1),
        # without the thread-lane work only the replay's own one-shot reconcile added
        # (production reconciles a PR once): what the 1.15x cost gate reads
        "gpt_lanes_per_run_excl_reconcile": _mean(
            [gpt(s, "lanes", block="gpt_excl_reconcile") for s in runs], 1
        ),
        "gpt_tokens_per_run_excl_reconcile": _mean(
            [gpt(s, "tokens_in", "tokens_out", block="gpt_excl_reconcile") for s in runs], 1
        ),
        "reconcile_issues_per_run": _mean(
            [float((s.report.get("reconcile_work") or {}).get("issues") or 0) for s in runs], 2
        ),
        "tokens_per_run": _mean(
            [
                float(s.report.get("tokens_in") or 0) + float(s.report.get("tokens_out") or 0)
                for s in runs
            ],
            1,
        ),
        "wall_clock_p50_s": round(statistics.median(walls), 1) if walls else None,
        "active_p50_s": round(statistics.median(actives), 1) if actives else None,
        "slot_wait_p50_s": round(statistics.median(waits), 1) if waits else None,
        "contract_failures": {
            "runs_failed": sum(bool(c.get("run_failed")) for c in contract),
            "lane_attempts_failed": sum(int(c.get("failed_attempts") or 0) for c in contract),
            "runs_with_corrections": sum(
                bool(c.get("corrected") or c.get("repaired")) for c in contract
            ),
        },
        "duplicates_per_run": _mean([float(duplicates(s.published)) for s in runs], 3),
        # v12 signals (zero under v11)
        "conflicts_per_run": _mean([sig(s, "threads_conflict") for s in runs], 3),
        "late_nonblocking_per_run": _mean([sig(s, "late_nonblocking") for s in runs], 3),
        "settled_refolds_per_run": _mean([sig(s, "settled_refold") for s in runs], 3),
        "match_share": round(matched / candidates, 4) if candidates else None,
        "duplicate_threads_per_run": _mean([float(duplicate_threads(s)) for s in runs], 3),
        "forced_phase2_runs": sum(
            bool((s.report.get("gate") or {}).get("forced_phase2")) for s in runs
        ),
    }


def _ratios(arms: dict[str, dict[str, Any]], baseline: str | None) -> dict[str, Any]:
    if not baseline or baseline not in arms:
        return {}
    base = arms[baseline]
    return {
        p: {
            k: round(a[k] / base[k], 3) if a.get(k) is not None and base.get(k) else None
            for k in (
                "gpt_lanes_per_run",
                "gpt_tokens_per_run",
                "gpt_lanes_per_run_excl_reconcile",
                "gpt_tokens_per_run_excl_reconcile",
                "wall_clock_p50_s",
                "active_p50_s",
            )
        }
        for p, a in arms.items()
        if p != baseline
    }


def _count_precision(
    acc: _Acc,
    verdicts: Verdicts,
    head: HeadKey,
    scored: list[Label],
    assigned: list[tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]],
) -> None:
    """Blockers by what is known of them, a mean over the head's samples: matched to a `tp`
    label (real) or another label (not real), else the judge's answer (or pending)."""
    n = len(assigned)
    for got, rest in assigned:
        for li, f in got.items():
            if f.get("severity") == "blocking":
                acc.labelled_blockers["tp" if scored[li].label == "tp" else "not"] += 1 / n
        for f in rest:
            if f.get("severity") == "blocking":
                ans = verdicts.real_of(head, f)
                acc.judged["pending" if ans is None else ans or "unsure"] += 1 / n


def _head_fields(head: HeadKey, base: dict[str, Any]) -> dict[str, Any]:
    return {"repo": head[0], "number": head[1], "sha": head[2], "head_id": head[3], **base}


def _finding_fields(f: dict[str, Any]) -> dict[str, Any]:
    """A finding as the judge gets it: the defect claim, blinded (`neutral_body`)."""
    return {
        "file": f.get("file"),
        "line_start": f.get("line_start"),
        "line_end": f.get("line_end"),
        "title": neutral_title(str(f.get("title") or "")),
        "body": neutral_body(str(f.get("body") or "")),
        "category": f.get("category") or "",
        "severity": f.get("severity") or "",
    }


def _blind(
    pool: dict[HeadKey, list[tuple[str, int, dict[str, Any]]]],
    match_pool: dict[tuple[HeadKey, str], tuple[dict[str, Any], dict[str, Any], set[Any]]],
    bases: dict[HeadKey, dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The judge's work, shuffled, with content-derived ids and no arm (the key maps ids back
    to arms): one `new_blocker` item per cluster of matching new blockers on a head (across
    arms and samples), and one `label_match` item per (finding, label) pair to settle."""
    items: list[dict[str, Any]] = []
    key: dict[str, Any] = {}
    for head, entries in pool.items():
        groups: list[list[tuple[str, int, dict[str, Any]]]] = []
        for e in entries:
            g = next((g for g in groups if any(matches(e[2], o[2]) for o in g)), None)
            if g is None:
                groups.append([e])
            else:
                g.append(e)
        for group in groups:
            f = group[0][2]
            jid = judge_id(NEW_BLOCKER, head, f, None)
            members = {json.dumps(_brief(x), sort_keys=True) for _, _, x in group}
            items.append(
                {
                    "id": jid,
                    "kind": NEW_BLOCKER,
                    **_head_fields(head, bases.get(head, {})),
                    **_finding_fields(f),
                    "members": [json.loads(m) for m in sorted(members)],
                }
            )
            key[jid] = {
                "raised_by": sorted({(p, s) for p, s, _ in group}),
                "titles": sorted({str(x.get("title")) for _, _, x in group}),
            }
    for (head, jid), (f, lab, raised) in match_pool.items():
        items.append(
            {
                "id": jid,
                "kind": LABEL_MATCH,
                **_head_fields(head, bases.get(head, {})),
                **_finding_fields(f),
                "label": {
                    k: lab.get(k) for k in ("hash", "title", "file", "line_start", "line_end")
                },
            }
        )
        key[jid] = {"raised_by": sorted(raised), "titles": [str(f.get("title"))]}
    random.Random(seed).shuffle(items)
    return items, key


def render(result: dict[str, Any]) -> str:
    pol = result["policies"]
    lines = [
        f"{result['heads']['paired']}/{result['heads']['in_set']} heads paired across "
        f"{', '.join(pol)}; baseline {result['baseline']}; noise floor "
        f"{result['noise']['floor_pp']} pp; {result['to_judge']} items to judge, "
        f"{result.get('judged', 0)} judged",
        "",
        f"{'arm':<16} {'usable':>6} {'gpt lanes':>9} {'gpt tok':>12} {'wall p50':>9} "
        f"{'active':>7} {'contract':>8} {'dups':>6} {'new prec':>8} {'blk prec':>8}",
    ]
    for p, a in result["arms"].items():
        pr = a.get("precision") or {}
        lines.append(
            f"{p:<16} {a['usable']:>6} {a['gpt_lanes_per_run'] or 0:>9} "
            f"{a['gpt_tokens_per_run'] or 0:>12} {a['wall_clock_p50_s'] or 0:>9} "
            f"{a.get('active_p50_s') or 0:>7} {a['contract_failures']['runs_failed']:>8} {a['duplicates_per_run'] or 0:>6} "
            f"{_pct(pr.get('new_blocker_precision')):>8} {_pct(pr.get('blocker_precision')):>8}"
        )
    lines += [
        "",
        f"{'set':<20} {'arm':<16} {'recall':>7} {'blk rec':>7} {'FP blk':>6} {'reraise':>7} "
        f"{'blk head':>8} {'new':>6}",
    ]
    for st, row in result["sets"].items():
        for p in pol:
            m = row.get(p)
            if not m:
                continue
            lines.append(
                f"{st:<20} {p:<16} {_pct(m['recall']):>7} {_pct(m['blocker_recall']):>7} "
                f"{m['fp_blockers']:>6} {m['reraised_unchanged']:>7} "
                f"{_pct(m['heads_with_blocker_rate']):>8} {m['new_blockers']:>6}"
            )
    return "\n".join(lines) + "\n"


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.0f}%"


def score_dir(
    setfile: Path,
    labels_file: Path,
    out: Path,
    *,
    baseline: str | None = None,
    seed: int = 0,
    limit: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score a replay set's directory and write score.json, to_judge.jsonl (what no verdict
    covers yet), to_judge.key.json (merged with the earlier key) and, once there are verdicts,
    spot_check.jsonl. Returns (score, to-judge items)."""
    items = limit_items(load_set(setfile), limit)
    labels = load_labels(labels_file)
    policies = load_policies(out)
    if baseline and baseline not in policies:
        raise SystemExit(f"--baseline {baseline}: not one of {policies}")
    samples = load_samples(items, policies, out)
    judged = load_judged(out)
    result, to_judge, key = score(
        items, labels, samples, policies, baseline=baseline, seed=seed, judged=judged
    )
    (out / "score.json").write_text(json.dumps(result, indent=1))
    (out / TO_JUDGE).write_text("".join(json.dumps(x) + "\n" for x in to_judge))
    old_key = read_json_file(out / "to_judge.key.json")
    (out / "to_judge.key.json").write_text(json.dumps({**old_key, **key}, indent=1))
    real = [j for j in judged if j.get("kind", NEW_BLOCKER) == NEW_BLOCKER and j.get("verdict")]
    if real:
        pick = random.Random(seed).sample(real, min(SPOT_CHECK, len(real)))
        (out / "spot_check.jsonl").write_text("".join(json.dumps(x) + "\n" for x in pick))
    return result, to_judge


def read_json_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def main_score(args: Any) -> int:
    result, _ = score_dir(
        Path(args.setfile).expanduser(),
        Path(args.labels).expanduser(),
        Path(args.out).expanduser(),
        baseline=args.baseline,
        seed=args.seed,
        limit=args.limit,
    )
    print(render(result), end="")
    return 0
