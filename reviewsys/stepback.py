"""Stepping back from a PR that keeps drawing new blockers (policy v11).

Every push is re-reviewed, and on some PRs every round finds one or two new blockers: the
author patches them, the patch draws the next ones, and the PR never converges. Measured on
09-08..11-10: a tenth of the PRs produce a third of all blockers that way, and on the worst of
them most later blockers trace back to a handful of causes that could be named by round 5 (a
guarantee the layer below cannot give, state with no single owner, a contract nobody wrote
down). A maintainer at that point does not file blocker number 17; they say what the findings
share and, when that does not help, ask for one decision before another round.

So, on a later round that again has new blockers (`due`), one lane reads everything the PR has
drawn so far and groups it by underlying cause (`parse_analysis`). What follows has two steps,
because neither a rule nor a model can tell at that point which PRs will converge on their own
(about a quarter of those the rule fires on draw at most one more blocker, and the lane names
a cause on most of them all the same):

1. The first time it names causes (`Analysis.stands`), the review carries them as a note
   beside its findings, with the decision it suggests (`render`). Nothing else changes.
2. When a later round draws new blockers from named causes again, although the PR was told
   (`holds`), the review asks for that decision and automatic re-review of the PR is paused:
   later pushes keep this review's verdict (`worker._hold`) until someone asks for a review.

The findings themselves are posted as always, with their severities: this never re-rates
one. What stands on a PR is one kv record (`Record`)."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from .contract import normalize_finding_text
from .db import kv_get, kv_set
from .models import FailKind, ReviewError

CONVERGING = "converging"
DIAGNOSES = (
    "foundational_gap",
    "patch_cascade",
    "unstated_contract",
    "scope_too_large",
    CONVERGING,
)
CONTINUE = "continue_normal"
# the decision a step-back asks for, and how the review names it
RECOMMENDATIONS = {
    "land_prerequisite_first": "land the prerequisite first",
    "narrow_contract": "narrow what this PR promises",
    "redesign_component": "redesign the component",
    "split_pr": "split the PR",
    CONTINUE: "",
}
DECISIONS = ("yes", "partly", "no")
MAX_THEMES = 4
MIN_THEME_FINDINGS = 2
# what the lane is shown of each finding's body, and what the review shows of its answer
BLOCKER_BODY_CHARS = 700
OTHER_BODY_CHARS = 300
NAME_CHARS = 140
CAUSE_CHARS = 700
TEXT_CHARS = 1000
# a finding id of the prompt (`p12`, `n3`) in text the PR's readers get: they never see the ids
_ID_RE = re.compile(r"\b[pn]\d{1,3}\b")
_ISSUE_RE = re.compile(r"^(?:https://github\.com/[\w.-]+/[\w.-]+/(?:issues|pull)/\d+|#\d+)$")


@dataclass(frozen=True, slots=True)
class Raised:
    """One finding the PR has drawn, as the lane is shown it. `id` is `p<n>` for one an
    earlier review posted and `n<n>` for a new one of this round."""

    id: str
    round: int
    severity: str
    file: str
    title: str
    body: str
    state: str  # open | closed (an earlier finding's thread) | new

    @property
    def new(self) -> bool:
        return self.state == "new"

    def brief(self) -> dict[str, Any]:
        limit = BLOCKER_BODY_CHARS if self.severity == "blocking" else OTHER_BODY_CHARS
        body = self.body if len(self.body) <= limit else self.body[:limit].rstrip() + " [...]"
        return {
            "id": self.id,
            "round": self.round,
            "severity": self.severity,
            "state": self.state,
            "file": self.file,
            "title": self.title,
            "body": body,
        }


@dataclass(frozen=True, slots=True)
class Round:
    """One earlier review round of the PR: its head and the findings it was first to post."""

    sha: str
    findings: tuple[dict[str, Any], ...] = ()

    @property
    def blockers(self) -> int:
        return sum(1 for f in self.findings if f["severity"] == "blocking")


def rounds(conn: sqlite3.Connection, repo: str, number: int, *, before: str) -> list[Round]:
    """The PR's review rounds other than `before`'s (rounds.py's definition: each head commit
    with a review of its own), oldest first, each with the findings it posted that no earlier
    round had (by hash)."""
    shas = [
        str(r["sha"])
        for r in conn.execute(
            "SELECT sha, MIN(posted_at) first_at FROM reviews WHERE repo=? AND number=? "
            "AND phase IN ('preliminary','final') AND imported=0 AND sha<>? "
            "GROUP BY sha ORDER BY first_at, MIN(id)",
            (repo, number, before),
        )
    ]
    first: dict[str, sqlite3.Row] = {}
    # a same-sha re-review records the findings it kept as posted again (`_backfill_posted`);
    # one of them under another hash (a changed category) is still the finding it was
    titles: set[tuple[str, str]] = set()
    for row in conn.execute(
        "SELECT f.hash, f.severity, f.file, f.title, f.body, h.sha FROM findings f "
        "JOIN runs r ON r.id=f.run_id JOIN heads h ON h.id=r.head_id "
        "WHERE h.repo=? AND h.number=? AND h.queue='live' AND f.stage='posted' ORDER BY f.id",
        (repo, number),
    ):
        title = (str(row["file"] or "").lower(), normalize_finding_text(row["title"]))
        if str(row["hash"]) in first or title in titles:
            continue
        titles.add(title)
        first[str(row["hash"])] = row
    by_sha: dict[str, list[dict[str, Any]]] = {sha: [] for sha in shas}
    for h, row in first.items():
        if row["sha"] in by_sha:
            by_sha[str(row["sha"])].append(
                {
                    "hash": h,
                    "severity": str(row["severity"]),
                    "file": str(row["file"] or ""),
                    "title": str(row["title"]),
                    "body": str(row["body"] or ""),
                }
            )
    return [Round(sha, tuple(by_sha[sha])) for sha in shas]


def due(
    earlier: list[Round],
    new_blockers: int,
    *,
    min_round: int,
    min_blockers: int,
    min_recent: int,
    stood: bool,
) -> str | None:
    """Why this round steps back, or None. A round without a new blocker never does: there
    is nothing new to explain. Otherwise it does once causes were already named on the PR
    (`stood`), or when this is at least round `min_round`, the round before it also drew a
    new blocker, the two together drew at least `min_recent`, and the PR has drawn at least
    `min_blockers` in all."""
    if not new_blockers:
        return None
    if stood:
        return "causes were already named on this PR"
    round_now = len(earlier) + 1
    total = sum(r.blockers for r in earlier) + new_blockers
    if round_now < min_round or total < min_blockers or not earlier or not earlier[-1].blockers:
        return None
    recent = earlier[-1].blockers + new_blockers
    if recent < min_recent:
        return None
    return f"round {round_now}, {total} blockers so far, {recent} new in the last two rounds"


def raised(earlier: list[Round], new: list[dict[str, Any]], open_hashes: set[str]) -> list[Raised]:
    """Everything the PR has drawn, for the lane: the earlier rounds' findings (`p<n>`, open
    while their thread is unresolved) and this round's new ones (`n<n>`)."""
    out: list[Raised] = []
    for i, r in enumerate(earlier, 1):
        for f in r.findings:
            state = "open" if f["hash"] in open_hashes else "closed"
            out.append(
                Raised(
                    f"p{len(out) + 1}", i, f["severity"], f["file"], f["title"], f["body"], state
                )
            )
    now = len(earlier) + 1
    out += [
        Raised(f"n{i}", now, f["severity"], f["file"], f["title"], f["body"], "new")
        for i, f in enumerate(new, 1)
    ]
    return out


@dataclass(frozen=True, slots=True)
class Theme:
    name: str
    root_cause: str
    finding_ids: tuple[str, ...]
    tracked_issue: str = ""


@dataclass(frozen=True, slots=True)
class Analysis:
    """The lane's answer. `stands`: it named causes (and the decision that would settle
    them)."""

    diagnosis: str
    themes: tuple[Theme, ...] = ()
    kind: str = CONTINUE
    text: str = ""
    decision_made: str = ""  # about the previous step-back's request: one of DECISIONS, or ""

    @property
    def stands(self) -> bool:
        return self.diagnosis != CONVERGING


def _word(value: Any) -> str:
    return re.sub(r"[\s\-]+", "_", str(value or "").strip().lower())


def _text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def parse_analysis(raw: dict[str, Any], ids: list[str], *, previous: bool) -> Analysis:
    """The step-back lane's answer against its contract: a known diagnosis; `converging` goes
    with `continue_normal` and nothing else does; any other diagnosis names 1 to MAX_THEMES
    themes, each with a cause and at least MIN_THEME_FINDINGS of the supplied finding `ids`
    (a cause behind a single finding is that finding, not a pattern), and says what to decide.
    With a `previous` step-back, whether its decision was made. Breaches are CONTRACT errors
    (correctable)."""
    problems: list[str] = []
    diagnosis = _word(raw.get("diagnosis"))
    if diagnosis not in DIAGNOSES:
        problems.append(f"diagnosis {diagnosis!r} is not one of {', '.join(DIAGNOSES)}")
    rec = raw.get("recommendation")
    rec = rec if isinstance(rec, dict) else {}
    kind = _word(rec.get("kind"))
    text = _text(rec.get("text"), TEXT_CHARS)
    if kind not in RECOMMENDATIONS:
        problems.append(f"recommendation.kind {kind!r} is not one of {', '.join(RECOMMENDATIONS)}")
    elif diagnosis in DIAGNOSES and (diagnosis == CONVERGING) != (kind == CONTINUE):
        problems.append(f"`{CONVERGING}` goes with `{CONTINUE}`, and no other diagnosis does")
    converging = diagnosis == CONVERGING
    themes: list[Theme] = []
    rows = raw.get("themes")
    if not converging:
        if not text:
            problems.append("recommendation.text must say what to decide")
        if cited := _ID_RE.findall(text):
            problems.append(
                f"recommendation.text refers to findings by id ({', '.join(dict.fromkeys(cited))}): "
                "its readers never see the ids"
            )
        if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_THEMES:
            problems.append(f"themes must list 1 to {MAX_THEMES} underlying causes")
            rows = []
        known = set(ids)
        for i, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                problems.append(f"theme {i} is not an object")
                continue
            name = _text(row.get("name"), NAME_CHARS)
            cause = _text(row.get("root_cause"), CAUSE_CHARS)
            listed = row.get("finding_ids")
            listed = [str(x).strip() for x in listed] if isinstance(listed, list) else []
            unknown = sorted({x for x in listed if x not in known})
            own = tuple(dict.fromkeys(x for x in listed if x in known))
            if not name or not cause:
                problems.append(f"theme {i} needs a `name` and a `root_cause`")
            if cited := _ID_RE.findall(f"{name} {cause}"):
                problems.append(
                    f"theme {i} refers to findings by id ({', '.join(dict.fromkeys(cited))}): "
                    "its readers never see the ids, so say what the finding is about"
                )
            if unknown:
                problems.append(f"theme {i} names unknown finding ids {', '.join(unknown)}")
            elif len(own) < MIN_THEME_FINDINGS:
                problems.append(
                    f"theme {i} needs at least {MIN_THEME_FINDINGS} finding ids: a cause "
                    "behind one finding is not a pattern"
                )
            issue = str(row.get("tracked_issue") or "").strip()
            themes.append(Theme(name, cause, own, issue if _ISSUE_RE.match(issue) else ""))
    decision = _word(raw.get("decision_made")) if previous else ""
    if previous and decision not in DECISIONS:
        problems.append(f"decision_made {decision!r} is not one of {', '.join(DECISIONS)}")
    if problems:
        raise ReviewError(FailKind.CONTRACT, "stepback: " + "; ".join(problems))
    if converging:
        return Analysis(CONVERGING, decision_made=decision)
    return Analysis(diagnosis, tuple(themes), kind, text, decision)


def holds(analysis: Analysis, new_blocker_ids: list[str], *, told: bool) -> bool:
    """This round pauses automatic re-review: the PR was `told` its causes by an earlier
    review, the decision that review suggested has not been made, and the causes named now
    cover at least half of this round's new blockers. The first naming never pauses, and
    causes that explain the PR's past but not what it drew just now do not either."""
    if not told or not analysis.stands or analysis.decision_made == "yes" or not new_blocker_ids:
        return False
    themed = {i for t in analysis.themes for i in t.finding_ids}
    return 2 * sum(1 for i in new_blocker_ids if i in themed) >= len(new_blocker_ids)


@dataclass(slots=True)
class Record:
    """What stands on the PR (kv `key`): the review that last named causes and what it
    suggested. `hold`: automatic re-review is paused on it; `held`: the commits that kept a
    paused review's verdict instead of a review of their own; `overruled`: a maintainer
    dismissed a review that paused, so later ones on this PR name causes and never pause."""

    sha: str
    run_id: int
    round: int
    diagnosis: str
    kind: str
    text: str
    themes: list[dict[str, str]] = field(default_factory=list)
    hold: bool = False
    held: list[str] = field(default_factory=list)
    overruled: bool = False
    at: str = ""

    def brief(self) -> dict[str, Any]:
        """What the next step-back lane is shown of this one."""
        return {
            "posted_at_commit": self.sha,
            "round": self.round,
            "diagnosis": self.diagnosis,
            "themes": self.themes,
            "recommendation": {"kind": self.kind, "text": self.text},
        }


def key(repo: str, number: int) -> str:
    return f"stepback:{repo}#{number}"


def load(conn: sqlite3.Connection, repo: str, number: int) -> Record | None:
    """The PR's standing step-back, or None. A record this code cannot read (written by a
    later version, or damaged) is no record; fields it does not know are ignored."""
    raw = kv_get(conn, key(repo, number))
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("sha"), str):
        return None
    known = {f.name for f in fields(Record)}
    try:
        rec = Record(**{k: v for k, v in data.items() if k in known})
    except TypeError:
        return None
    held, themes = data.get("held"), data.get("themes")
    rec.held = [str(s) for s in held] if isinstance(held, list) else []
    rec.themes = [t for t in themes if isinstance(t, dict)] if isinstance(themes, list) else []
    rec.hold = data.get("hold") is True
    rec.overruled = data.get("overruled") is True
    return rec


def save(conn: sqlite3.Connection, repo: str, number: int, rec: Record) -> None:
    """Caller holds a transaction."""
    kv_set(conn, key(repo, number), json.dumps(asdict(rec)))


def record_of(
    analysis: Analysis,
    *,
    sha: str,
    run_id: int,
    round: int,
    hold: bool,
    at: str,
    previous: Record | None,
) -> Record:
    """The record of a review that named causes; what `previous` knew of held commits and of
    a maintainer's overruling stays."""
    return Record(
        sha=sha,
        run_id=run_id,
        round=round,
        diagnosis=analysis.diagnosis,
        kind=analysis.kind,
        text=analysis.text,
        themes=[
            {"name": t.name, "root_cause": t.root_cause, "tracked_issue": t.tracked_issue}
            for t in analysis.themes
        ],
        hold=hold,
        held=list(previous.held) if previous else [],
        overruled=previous.overruled if previous else False,
        at=at,
    )


def release(
    conn: sqlite3.Connection, repo: str, number: int, *, overruled: bool = False
) -> Record | None:
    """End the PR's pause, if one stands: the record it stood on, or None. Caller holds a
    transaction."""
    rec = load(conn, repo, number)
    if rec is None or not rec.hold:
        return None
    rec.hold = False
    rec.overruled = rec.overruled or overruled
    save(conn, repo, number, rec)
    return rec


def asked_on_paused(conn: sqlite3.Connection, repo: str, number: int, sha: str) -> Record | None:
    """Someone asked for a review on the very commit whose review paused the PR. That commit
    is reviewed already, so the ask starts no run; it is the author saying the decision is
    made, and it ends the pause: the next push is reviewed. The record, or None when `sha` is
    not that commit. Caller holds a transaction."""
    rec = load(conn, repo, number)
    if rec is None or not rec.hold or rec.sha != sha:
        return None
    return release(conn, repo, number)


def is_held(conn: sqlite3.Connection, repo: str, number: int, sha: str) -> bool:
    """`sha` kept the standing step-back's verdict instead of a review of its own."""
    rec = load(conn, repo, number)
    return rec is not None and sha in rec.held


@dataclass(slots=True)
class Outcome:
    """A round's step-back, as the review shows it. `hold`: it pauses automatic re-review;
    `may_hold`: a later one on this PR could (shown as a notice with the first naming)."""

    analysis: Analysis
    raised: dict[str, Raised]
    round: int
    blockers: int  # blocking findings the PR has drawn, this round's new ones included
    hold: bool
    previous: Record | None = None
    may_hold: bool = True


# the headings of the section's two forms and the sentence that says re-review is paused: a
# retry that finds its review already posted reads back from them what it posted
# (worker._backfill_step_back)
NOTE_HEADING = "### A pattern across this PR's review rounds"
HEADING = "### Step back: this PR keeps drawing new blockers"
PAUSED_NOTE = "Automatic re-review is paused until then"


def held_note(bot_login: str) -> str:
    return (
        f"{PAUSED_NOTE}: later pushes keep this review's verdict and are not reviewed on "
        f"their own. Comment `@{bot_login} review` once the decision is made: that resumes "
        "automatic re-review, and reviews the current commit if it has not been reviewed."
    )


def _defuse(text: str) -> str:
    """Model-written text for a review body: no live @-mentions and no HTML (a marker comment
    the bot would read back, a `<details>` left open over the rest of the review)."""
    return text.replace("@", "@\u200b").replace("<", "&lt;")


def _causes(n: int, *, the: bool = False) -> str:
    if the:
        return "this cause" if n == 1 else "these causes"
    return "one underlying cause" if n == 1 else f"{n} underlying causes"


def render(o: Outcome, bot_login: str) -> list[str]:
    """The review's section: a note the first time causes are named, the step-back proper
    (`o.hold`) when the PR drew new blockers from them again. Nothing when the rounds are
    converging."""
    a = o.analysis
    if not a.stands:
        return []
    n = len(a.themes)
    themed = {i for t in a.themes for i in t.finding_ids}
    drawn = (
        f"This is review round {o.round}, and the PR has drawn {o.blockers} blocking findings "
        "so far."
    )
    if o.hold:
        new = sum(1 for i in themed if i in o.raised and o.raised[i].new)
        told = f"The review of `{o.previous.sha[:8]}`" if o.previous else "An earlier review"
        lines = [
            HEADING,
            "",
            f"{drawn} {told} named {_causes(n, the=True)} and the decision that would settle "
            f"it; this round found {new} more finding(s) from the same place. Fixing them "
            "one at a time is not converging, so this review asks for that decision before "
            "another round.",
        ]
    else:
        under = sum(1 for i in themed if i in o.raised and o.raised[i].severity == "blocking")
        share = f"{under} of them share" if under else "Several of its findings share"
        lines = [
            NOTE_HEADING,
            "",
            f"{drawn} {share} {_causes(n)}. They are listed with the other findings as usual; "
            "fixing the cause is likely to be cheaper than fixing each case.",
        ]
    for i, t in enumerate(a.themes, 1):
        own = [o.raised[x] for x in t.finding_ids if x in o.raised]
        fresh = [f for f in own if f.new]
        earlier = len(own) - len(fresh)
        count = ", ".join(
            part
            for part in (
                f"{len(fresh)} new in this round" if fresh else "",
                f"{earlier} earlier" if earlier else "",
            )
            if part
        )
        lines += ["", f"**{i}. {_defuse(t.name)}** ({count} finding(s))", "", _defuse(t.root_cause)]
        if t.tracked_issue:
            lines += ["", f"Already tracked: {t.tracked_issue}"]
        if fresh:
            lines += ["", "In this round: " + "; ".join(f"_{_defuse(f.title)}_" for f in fresh)]
    asked = "Requested decision" if o.hold else "Suggested decision"
    lines += ["", f"**{asked} ({RECOMMENDATIONS[a.kind]}):** {_defuse(a.text)}"]
    if o.hold:
        lines += ["", f"_{held_note(bot_login)}_"]
    elif o.may_hold:
        lines += [
            "",
            f"_If a later review again finds new blockers from {_causes(n, the=True)}, it will "
            "ask for that decision and pause automatic re-review until it is made._",
        ]
    return lines


def agent_note(o: Outcome) -> str:
    """What the review's prompt for AI agents says first: an agent told only to fix the
    listed findings patches the reported cases, which is what keeps the rounds going."""
    a = o.analysis
    if o.hold:
        return (
            'Before fixing anything, read the review\'s "Step back" section. This PR keeps '
            "drawing new blocking findings from the same underlying causes, and patching the "
            "findings below one at a time is not converging. The review asks for this "
            f"decision first ({RECOMMENDATIONS[a.kind]}): {a.text} Raise that decision with "
            "the PR's author or maintainer instead of only patching; if it is theirs to make, "
            "stop and ask them."
        )
    names = "; ".join(t.name for t in a.themes)
    return (
        'Several findings on this PR share an underlying cause (the review\'s "pattern" '
        f"section): {names}. When you fix the findings below, fix the cause, not only the "
        f"reported case. The review suggests ({RECOMMENDATIONS[a.kind]}): {a.text} If that "
        "decision is not yours to make, say so to the PR's author or maintainer."
    )
