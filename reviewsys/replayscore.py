"""`reviewsys replay-score`: score a replay set's reports against labels, arm against arm.

Labels are JSON (a list) or JSON Lines, one per known issue at a replayed head:

    {"repo": "dashpay/dash", "number": 7683, "sha": "<replayed head>", "file": "src/x.cpp",
     "line_start": 247, "line_end": 249, "title": "...", "label": "tp|fp|deferred",
     "set": "S1".."S8", "source": "how it was labelled"}

Optional: `hash` and `category` (also match through `compare._Finding.matches`), and
`anchor_changed` (true when the code at the anchor changed since the issue was settled; S5
counts re-raises only on unchanged anchors, the default).

A replayed finding matches a label at the same file within 10 lines with title-token Jaccard
at least 0.2, or when `compare._Finding.matches` says so (same hash, or same file and
category with overlapping lines). What a replay *published* is scored (`report.published`).

Metrics per arm and set, on paired heads only (every arm has a usable sample of the head):
recall (blocker or suggestion; and blocker-only) of `tp` labels, known-FP blockers (`fp` and
`deferred` labels raised as blockers), re-raises of settled issues on unchanged anchors (any
severity), runs with a blocker, new blockers (matching no label); per arm: duplicates per run
(same file within 3 lines, different hash), gpt lanes and tokens per run, wall clock p50,
contract failures. Repeat samples give the noise floor: how often one arm disagrees with itself
on whether a `tp` label was found. New blockers go to `to_judge.jsonl` for the judge lane,
clustered across arms per head and blinded (the arm key is in `to_judge.key.json`).
"""

from __future__ import annotations

import dataclasses
import json
import random
import re
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .compare import _Finding
from .replayset import MANIFEST, Job, Policy, SetItem, load_records, load_set, report_state

LINE_SLACK = 10
JACCARD_MIN = 0.2
DUPLICATE_SLACK = 3
LABELS = ("tp", "fp", "deferred")
COUNTED = ("blocking", "suggestion")  # recall counts a label found at either severity
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
    anchor_changed: bool = False

    @property
    def head(self) -> tuple[str, int, str]:
        return (self.repo.lower(), self.number, self.sha.lower())


def load_labels(path: Path) -> list[Label]:
    out = []
    for i, e in enumerate(load_records(path, "labels")):
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
                hash=str(e.get("hash") or ""),
                category=str(e.get("category") or ""),
                anchor_changed=bool(e.get("anchor_changed", False)),
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
    sa, sb = (
        _span(a.get("line_start"), a.get("line_end")),
        _span(b.get("line_start"), b.get("line_end")),
    )
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
    if (
        near(finding, label, LINE_SLACK)
        and jaccard(finding["title"], label["title"]) >= JACCARD_MIN
    ):
        return True
    a, b = _as_compare(finding), _as_compare(label)
    if a.hash and a.hash == b.hash:
        return True
    # else compare's same file + category + overlapping lines; never on an empty category, and
    # never on two empty hashes (`_Finding.matches` would take those as equal)
    return bool(a.category) and dataclasses.replace(a, hash="\0").matches(b)


# ---- reports ----


@dataclass(slots=True)
class Sample:
    policy: str
    item: SetItem
    sample: int
    report: dict[str, Any]

    @property
    def usable(self) -> bool:
        return self.report.get("status") == "done" and not self.report.get("invalid")

    @property
    def head(self) -> tuple[str, int, str]:
        r = self.report
        return (str(r["repo"]).lower(), int(r["number"]), str(r["sha"]).lower())

    @property
    def published(self) -> list[dict[str, Any]]:
        return [f for f in self.report.get("published") or [] if isinstance(f, dict)]


def load_policies(out: Path) -> list[str]:
    try:
        manifest = json.loads((out / MANIFEST).read_text())
        return [str(p["name"]) for p in manifest["policies"]]
    except (OSError, ValueError, KeyError, TypeError):
        return sorted(p.name for p in out.iterdir() if p.is_dir() and any(p.glob("run-*")))


def load_samples(items: list[SetItem], policies: list[str], out: Path) -> list[Sample]:
    found = []
    for item in items:
        for name in policies:
            for k in range(item.samples):
                job = Job(item, Policy(name, Path()), k)
                if report_state(job, out) == "missing":
                    continue
                rep = json.loads((job.dir(out) / "report.json").read_text())
                found.append(Sample(name, item, k, rep))
    return found


# ---- scoring ----


@dataclass(slots=True)
class _Acc:
    """Running totals of one arm over one set."""

    heads: set[tuple[str, int, str]] = field(default_factory=set)
    runs: int = 0
    tp: list[float] = field(default_factory=list)  # per tp label: share of samples finding it
    tp_blocker: list[float] = field(default_factory=list)
    fp_blockers: int = 0
    reraised: int = 0
    runs_with_blocker: int = 0
    new_blockers: int = 0

    def render(self) -> dict[str, Any]:
        return {
            "heads": len(self.heads),
            "runs": self.runs,
            "tp_labels": len(self.tp),
            "recall": _mean(self.tp),
            "blocker_recall": _mean(self.tp_blocker),
            "fp_blockers": self.fp_blockers,
            "reraised_unchanged": self.reraised,
            "runs_with_blocker_rate": self.runs_with_blocker / self.runs if self.runs else None,
            "new_blockers": self.new_blockers,
        }


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def _found(sample: Sample, label: dict[str, Any], severities: Iterable[str]) -> bool:
    sev = set(severities)
    return any(f.get("severity") in sev and matches(f, label) for f in sample.published)


def duplicates(published: list[dict[str, Any]]) -> int:
    """Pairs of published findings on the same file within 3 lines and different hashes."""
    n = 0
    for i, a in enumerate(published):
        for b in published[i + 1 :]:
            if a.get("hash") != b.get("hash") and near(a, b, DUPLICATE_SLACK):
                n += 1
    return n


def score(
    items: list[SetItem],
    labels: list[Label],
    samples: list[Sample],
    policies: list[str],
    *,
    baseline: str | None = None,
    seed: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """(score, to-judge items, blinding key)."""
    baseline = baseline or (policies[0] if policies else None)
    by_head: dict[tuple[str, int, str], list[Label]] = {}
    for lab in labels:
        by_head.setdefault(lab.head, []).append(lab)
    usable: dict[tuple[int, str], list[Sample]] = {}
    for s in samples:
        if s.usable:
            usable.setdefault((s.item.run_id, s.policy), []).append(s)
    paired = [i for i in items if policies and all((i.run_id, p) in usable for p in policies)]
    per_set: dict[str, dict[str, _Acc]] = {}
    paired_heads: set[tuple[str, int, str]] = set()
    noise: dict[str, dict[str, int]] = {p: {"labels": 0, "disagree": 0} for p in policies}
    judge_pool: dict[tuple[str, int, str], list[tuple[str, int, dict[str, Any]]]] = {}
    for item in paired:
        for p in policies:
            runs = sorted(usable[(item.run_id, p)], key=lambda s: s.sample)
            head = runs[0].head
            paired_heads.add(head)
            head_labels = [_label_dict(x) for x in by_head.get(head, [])]
            sets = {item.set} | {ld["set"] for ld in head_labels}
            for st in sets:
                acc = per_set.setdefault(st, {}).setdefault(p, _Acc())
                acc.heads.add(head)
                if st == item.set:
                    acc.runs += len(runs)
                    acc.runs_with_blocker += sum(
                        any(f.get("severity") == "blocking" for f in r.published) for r in runs
                    )
            for ld in head_labels:
                acc = per_set.setdefault(ld["set"], {}).setdefault(p, _Acc())
                if ld["label"] == "tp":
                    hits = [_found(r, ld, COUNTED) for r in runs]
                    acc.tp.append(sum(hits) / len(hits))
                    acc.tp_blocker.append(
                        sum(_found(r, ld, ("blocking",)) for r in runs) / len(runs)
                    )
                    if len(runs) > 1:
                        noise[p]["labels"] += 1
                        noise[p]["disagree"] += len(set(hits)) > 1
                else:
                    acc.fp_blockers += sum(_found(r, ld, ("blocking",)) for r in runs)
                    if not ld["anchor_changed"]:
                        acc.reraised += sum(
                            _found(r, ld, ("blocking", "suggestion", "nitpick")) for r in runs
                        )
            for r in runs:
                for f in r.published:
                    if f.get("severity") != "blocking":
                        continue
                    if any(matches(f, ld) for ld in head_labels):
                        continue
                    per_set.setdefault(item.set, {}).setdefault(p, _Acc()).new_blockers += 1
                    judge_pool.setdefault(head, []).append((p, r.sample, f))
    arms = {p: _arm(p, [s for s in samples if s.policy == p], paired) for p in policies}
    sets_out: dict[str, Any] = {}
    for st in sorted(per_set):
        row: dict[str, Any] = {p: per_set[st][p].render() for p in policies if p in per_set[st]}
        if baseline and baseline in row:
            base = row[baseline]
            row["delta_vs_" + baseline] = {
                p: {
                    k: _delta(row[p].get(k), base.get(k))
                    for k in ("recall", "blocker_recall", "runs_with_blocker_rate")
                }
                | {k: row[p][k] - base[k] for k in ("fp_blockers", "reraised_unchanged")}
                for p in policies
                if p != baseline and p in row
            }
        sets_out[st or "(unset)"] = row
    noise_out = {
        p: {
            **n,
            "flip_rate": round(n["disagree"] / n["labels"], 4) if n["labels"] else None,
        }
        for p, n in noise.items()
    }
    pooled = [n for n in noise.values() if n["labels"]]
    floor = (
        round(100 * sum(n["disagree"] for n in pooled) / sum(n["labels"] for n in pooled), 1)
        if pooled
        else None
    )
    to_judge, key = _blind(judge_pool, seed=seed)
    result = {
        "policies": policies,
        "baseline": baseline,
        "heads": {
            "in_set": len(items),
            "paired": len(paired),
            "unpaired": sorted(i.run_id for i in items if i not in paired),
        },
        "labels": {
            "total": len(labels),
            "on_paired_heads": sum(len(by_head.get(h, [])) for h in paired_heads),
        },
        "arms": arms,
        "ratios_vs_baseline": _ratios(arms, baseline),
        "sets": sets_out,
        "noise": {"per_arm": noise_out, "floor_pp": floor},
        "to_judge": len(to_judge),
        "matcher": {
            "line_slack": LINE_SLACK,
            "title_jaccard_min": JACCARD_MIN,
            "also": "compare._Finding.matches",
        },
    }
    return result, to_judge, key


def _label_dict(lab: Label) -> dict[str, Any]:
    return {
        "file": lab.file,
        "line_start": lab.line_start,
        "line_end": lab.line_end,
        "title": lab.title,
        "hash": lab.hash,
        "category": lab.category,
        "label": lab.label,
        "set": lab.set,
        "anchor_changed": lab.anchor_changed,
    }


def _delta(a: Any, b: Any) -> float | None:
    return (
        round((a - b) * 100, 1)
        if isinstance(a, int | float) and isinstance(b, int | float)
        else None
    )


def _arm(policy: str, samples: list[Sample], paired: list[SetItem]) -> dict[str, Any]:
    ids = {i.run_id for i in paired}
    runs = [s for s in samples if s.usable and s.item.run_id in ids]
    every = [s for s in samples if s.item.run_id in ids]

    def per_run(key: str) -> float | None:
        vals = [float((s.report.get("gpt") or {}).get(key) or 0) for s in runs]
        return round(statistics.fmean(vals), 1) if vals else None

    walls = [float(s.report.get("wall_clock_s") or 0) for s in runs]
    contract = [s.report.get("contract") or {} for s in every]
    return {
        "samples": len(every),
        "usable": len(runs),
        "failed": sum(s.report.get("status") == "failed" for s in every),
        "invalid": sum(bool(s.report.get("invalid")) for s in every),
        "gpt_lanes_per_run": per_run("lanes"),
        "gpt_tokens_per_run": round(
            statistics.fmean(
                float((s.report.get("gpt") or {}).get("tokens_in") or 0)
                + float((s.report.get("gpt") or {}).get("tokens_out") or 0)
                for s in runs
            ),
            1,
        )
        if runs
        else None,
        "tokens_per_run": round(
            statistics.fmean(
                float(s.report.get("tokens_in") or 0) + float(s.report.get("tokens_out") or 0)
                for s in runs
            ),
            1,
        )
        if runs
        else None,
        "wall_clock_p50_s": round(statistics.median(walls), 1) if walls else None,
        "contract_failures": {
            "runs_failed": sum(bool(c.get("run_failed")) for c in contract),
            "lane_attempts_failed": sum(int(c.get("failed_attempts") or 0) for c in contract),
            "runs_with_corrections": sum(
                bool(c.get("corrected") or c.get("repaired")) for c in contract
            ),
        },
        "duplicates_per_run": round(statistics.fmean(duplicates(s.published) for s in runs), 3)
        if runs
        else None,
        "forced_phase2_runs": sum(
            bool((s.report.get("gate") or {}).get("forced_phase2")) for s in runs
        ),
    }


def _ratios(arms: dict[str, dict[str, Any]], baseline: str | None) -> dict[str, Any]:
    if not baseline or baseline not in arms:
        return {}
    base = arms[baseline]
    out = {}
    for p, a in arms.items():
        if p == baseline:
            continue
        out[p] = {
            k: round(a[k] / base[k], 3) if a.get(k) is not None and base.get(k) else None
            for k in ("gpt_lanes_per_run", "gpt_tokens_per_run", "wall_clock_p50_s")
        }
    return out


def _blind(
    pool: dict[tuple[str, int, str], list[tuple[str, int, dict[str, Any]]]], *, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One judge item per cluster of matching new blockers on a head (across arms and
    samples), shuffled, with opaque ids; the key maps ids back to arms."""
    items: list[dict[str, Any]] = []
    key: dict[str, Any] = {}
    clusters: list[tuple[tuple[str, int, str], list[tuple[str, int, dict[str, Any]]]]] = []
    for head, entries in pool.items():
        groups: list[list[tuple[str, int, dict[str, Any]]]] = []
        for e in entries:
            g = next((g for g in groups if any(matches(e[2], o[2]) for o in g)), None)
            if g is None:
                groups.append([e])
            else:
                g.append(e)
        clusters += [(head, g) for g in groups]
    rng = random.Random(seed)
    rng.shuffle(clusters)
    for n, (head, group) in enumerate(clusters, 1):
        jid = f"j{n:04d}"
        f = group[0][2]
        items.append(
            {
                "id": jid,
                "repo": head[0],
                "number": head[1],
                "sha": head[2],
                "file": f.get("file"),
                "line_start": f.get("line_start"),
                "line_end": f.get("line_end"),
                "title": f.get("title"),
                "body": f.get("body") or "",
                "category": f.get("category") or "",
            }
        )
        key[jid] = {
            "raised_by": sorted({(p, s) for p, s, _ in group}),
            "titles": sorted({str(x.get("title")) for _, _, x in group}),
        }
    return items, key


def render(result: dict[str, Any]) -> str:
    pol = result["policies"]
    lines = [
        f"{result['heads']['paired']}/{result['heads']['in_set']} heads paired across "
        f"{', '.join(pol)}; baseline {result['baseline']}; noise floor "
        f"{result['noise']['floor_pp']} pp; {result['to_judge']} new-blocker clusters to judge",
        "",
        f"{'arm':<16} {'usable':>6} {'gpt lanes':>9} {'gpt tok':>12} {'wall p50':>9} "
        f"{'contract':>8} {'dups':>6}",
    ]
    for p, a in result["arms"].items():
        lines.append(
            f"{p:<16} {a['usable']:>6} {a['gpt_lanes_per_run'] or 0:>9} "
            f"{a['gpt_tokens_per_run'] or 0:>12} {a['wall_clock_p50_s'] or 0:>9} "
            f"{a['contract_failures']['runs_failed']:>8} {a['duplicates_per_run'] or 0:>6}"
        )
    lines += [
        "",
        f"{'set':<8} {'arm':<16} {'recall':>7} {'blk rec':>7} {'FP blk':>6} {'reraise':>7} {'blk run':>7} {'new':>4}",
    ]
    for st, row in result["sets"].items():
        for p in pol:
            m = row.get(p)
            if not m:
                continue
            lines.append(
                f"{st:<8} {p:<16} {_pct(m['recall']):>7} {_pct(m['blocker_recall']):>7} "
                f"{m['fp_blockers']:>6} {m['reraised_unchanged']:>7} "
                f"{_pct(m['runs_with_blocker_rate']):>7} {m['new_blockers']:>4}"
            )
    return "\n".join(lines) + "\n"


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.0f}%"


def main_score(args: Any) -> int:
    items = load_set(Path(args.setfile).expanduser())
    labels = load_labels(Path(args.labels).expanduser())
    out = Path(args.out).expanduser()
    policies = load_policies(out)
    if args.baseline and args.baseline not in policies:
        raise SystemExit(f"--baseline {args.baseline}: not one of {policies}")
    samples = load_samples(items, policies, out)
    result, to_judge, key = score(
        items, labels, samples, policies, baseline=args.baseline, seed=args.seed
    )
    (out / "score.json").write_text(json.dumps(result, indent=1))
    (out / "to_judge.jsonl").write_text("".join(json.dumps(x) + "\n" for x in to_judge))
    (out / "to_judge.key.json").write_text(json.dumps(key, indent=1))
    print(render(result), end="")
    return 0
