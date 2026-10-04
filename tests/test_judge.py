"""The judge lane (`reviewsys replay-judge`) and what `replay-score` makes of its verdicts:
label matches the matcher could not settle, new-blocker precision, resumability, blinding."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from test_replayset import _f, _label, _sample

from reviewsys import judge as judge_mod
from reviewsys import replayscore
from reviewsys.lane import LaneResult, LaneSpec
from reviewsys.replayscore import LABEL_MATCH, NEW_BLOCKER, score
from reviewsys.replayset import GB, Gate, SetItem

ARMS = ["v11", "v12"]


def _items_and_samples() -> tuple[list[SetItem], list[Any], list[Any]]:
    """Head 1 (S4): a label, a finding near it with another title (to settle), and a new
    blocker elsewhere; raised by both arms."""
    items = [SetItem(1, "S4")]
    labels = [_label(1, "Arithmetic wraps on large input", "fp", "S4", line=10)]
    found = [_f("Overflow in f3", 12), _f("Missing bounds check in f7", 60)]
    samples = [_sample(p, 1, found, set_="S4") for p in ARMS]
    return items, labels, samples


def test_unsettled_pairs_and_new_blockers_go_to_the_judge_blinded_with_stable_ids():
    items, labels, samples = _items_and_samples()
    _, todo, key = score(items, labels, samples, ARMS, seed=1)
    kinds = sorted(t["kind"] for t in todo)
    assert kinds == [LABEL_MATCH, NEW_BLOCKER, NEW_BLOCKER]
    match = next(t for t in todo if t["kind"] == LABEL_MATCH)
    assert match["title"] == "Overflow in f3"
    assert match["label"]["title"] == "Arithmetic wraps on large input"
    assert "label" not in match["label"] or match["label"].get("label") is None  # no tp/fp
    text = json.dumps(todo)
    assert "v11" not in text and "v12" not in text
    assert key[match["id"]]["raised_by"] == [("v11", 0), ("v12", 0)]
    _, again, _ = score(items, labels, samples, ARMS, seed=7)  # another order, same ids
    assert sorted(t["id"] for t in again) == sorted(t["id"] for t in todo)


def _verdict(item: dict[str, Any], real: str, same: str | None = None) -> dict[str, Any]:
    return {**item, "verdict": {"real": real, "evidence": "e", "matches_label": same, "reason": ""}}


def test_verdicts_settle_matches_and_give_precision():
    items, labels, samples = _items_and_samples()
    _, todo, _ = score(items, labels, samples, ARMS)
    judged = []
    for t in todo:
        if t["kind"] == LABEL_MATCH:
            judged.append(_verdict(t, "yes", "yes"))
        elif t["title"] == "Overflow in f3":
            judged.append(_verdict(t, "yes"))
        else:
            judged.append(_verdict(t, "no"))
    result, todo2, _ = score(items, labels, samples, ARMS, judged=judged)
    assert todo2 == [] and result["judged"] == 3
    s4 = result["sets"]["S4"]["v12"]
    # the judge matched the f3 finding to the fp label: a known-FP blocker, no longer new
    assert s4["fp_blockers"] == 1.0 and s4["new_blockers"] == 1.0
    assert s4["new_blockers_judged"] == {"yes": 0.0, "no": 1.0, "unsure": 0.0, "pending": 0.0}
    assert s4["new_blocker_precision"] == 0.0 and s4["blocker_precision"] == 0.0
    assert result["arms"]["v11"]["precision"]["new_blocker_precision"] == 0.0


def test_a_verdict_covers_the_whole_cluster_and_unjudged_ones_stay_pending():
    items = [SetItem(1, "S7")]
    a = _f("Unchecked index into peers vector", 40)
    b = _f("Unchecked peers vector index", 42, hash="other")
    samples = [_sample("v11", 1, [a], set_="S7"), _sample("v12", 1, [b], set_="S7")]
    _, todo, _ = score(items, [], samples, ARMS)
    assert len(todo) == 1 and len(todo[0]["members"]) == 2
    result, todo2, _ = score(items, [], samples, ARMS, judged=[_verdict(todo[0], "yes")])
    assert todo2 == []
    for arm in ARMS:
        assert result["arms"][arm]["precision"]["new_blocker_precision"] == 1.0
    result, _, _ = score(items, [], samples, ARMS, judged=[])
    assert result["arms"]["v12"]["precision"]["new_blockers_judged"]["pending"] == 1.0
    assert result["arms"]["v12"]["precision"]["new_blocker_precision"] is None


def test_the_prompt_is_blind_and_asks_the_second_question_only_for_a_label_match():
    items, labels, samples = _items_and_samples()
    _, todo, _ = score(items, labels, samples, ARMS)
    for t in todo:
        p = judge_mod.prompt(t)
        assert "v11" not in p and "v12" not in p and "fp" not in p.split()
        assert t["title"] in p and "no GitHub" in p
        assert ("## Question 2" in p) == (t["kind"] == LABEL_MATCH)


def test_parse_verdict():
    assert judge_mod.parse_verdict({"real": "Yes", "evidence": "x"}, NEW_BLOCKER)["real"] == "yes"
    with pytest.raises(ValueError):
        judge_mod.parse_verdict({"real": "maybe"}, NEW_BLOCKER)
    with pytest.raises(ValueError):
        judge_mod.parse_verdict({"real": "no", "matches_label": None}, LABEL_MATCH)
    got = judge_mod.parse_verdict({"real": "no", "matches_label": "unsure"}, LABEL_MATCH)
    assert got["matches_label"] == "unsure"


# ---- the lanes ----


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def judged_out(cfg, tmp_path) -> tuple[Path, str]:
    """A replay-set dir with two to-judge items on one head of a real mirror."""
    src = tmp_path / "src"
    src.mkdir()
    _git(src, "init", "-q", "-b", "main")
    _git(src, "config", "user.email", "t@example.com")
    _git(src, "config", "user.name", "T")
    (src / "f.rs").write_text("fn f() {}\n")
    _git(src, "add", ".")
    _git(src, "commit", "-qm", "base")
    base = _git(src, "rev-parse", "HEAD")
    (src / "f.rs").write_text("fn f() { 1 }\n")
    _git(src, "commit", "-qam", "head")
    head = _git(src, "rev-parse", "HEAD")
    cfg.mirrors_dir.mkdir(parents=True)
    subprocess.run(
        ["git", "clone", "-q", "--bare", str(src), str(cfg.mirrors_dir / "o__r.git")], check=True
    )
    out = tmp_path / "out"
    out.mkdir()
    common = {"repo": "o/r", "number": 1, "sha": head, "head_id": "r-1", "merge_base": base,
              "base_ref": "main", "base_sha": base, "file": "f.rs", "line_start": 1,
              "line_end": 1, "body": "b", "category": "logic", "severity": "blocking"}  # fmt: skip
    todo = [
        {"id": "j-1", "kind": NEW_BLOCKER, "title": "Bad f", **common},
        {"id": "m-1", "kind": LABEL_MATCH, "title": "Bad f again", **common,
         "label": {"title": "f is wrong", "file": "f.rs", "line_start": 1, "line_end": 1}},
    ]  # fmt: skip
    (out / replayscore.TO_JUDGE).write_text("".join(json.dumps(t) + "\n" for t in todo))
    return out, head


class JudgeLanes:
    def __init__(self, fail: set[str] = frozenset()) -> None:  # type: ignore[assignment]
        self.calls: list[LaneSpec] = []
        self.fail = fail
        self.seen: dict[str, str] = {}

    def __call__(self, spec: LaneSpec, art: Path, wt: Path) -> LaneResult:
        self.calls.append(spec)
        self.seen["head"] = _git(spec.cwd, "rev-parse", "HEAD")
        self.seen["refs"] = _git(spec.cwd, "for-each-ref", "--format=%(refname)")
        if any(t in spec.prompt for t in self.fail):
            return LaneResult(exit_code=1, stdout="", stderr="API Error: 500", duration_s=0)
        ans = {"real": "no", "evidence": "f.rs:1", "reason": "r"}
        ans["matches_label"] = "yes" if "## Question 2" in spec.prompt else None  # type: ignore[assignment]
        text = json.dumps(ans)
        return LaneResult(exit_code=0, stdout=text, stderr="", duration_s=0, result_text=text)


def test_judge_runs_blinded_pinned_lanes_and_resumes(cfg, judged_out):
    out, head = judged_out
    lanes = JudgeLanes()
    res = judge_mod.judge(cfg, out, lane_runner=lanes, echo=lambda s: None)
    assert res == {"pending": 2, "heads": 1, "judged": 2, "failed": 0, "not_started": 0,
                   "stopped": ""}  # fmt: skip
    spec = lanes.calls[0]
    assert (spec.model, spec.effort, spec.role) == ("gpt-6.1-sol", "xhigh", "judge")
    assert set(spec.disallowed_tools) == {"WebFetch", "WebSearch"}
    env = dict(spec.env)
    work = out.parent / "out.judge-work"  # beside the scored dir, never in it
    assert env["PATH"].startswith(str(work / "bin"))
    assert env["GH_TOKEN"] == env["GITHUB_TOKEN"] == env["REVIEWSYS_REPLAY_GH_TOKEN"] == ""
    assert Path(env["GH_CONFIG_DIR"]).is_dir() and not any(Path(env["GH_CONFIG_DIR"]).iterdir())
    assert not (out / "judge").exists() and not list(out.glob("*judge-work*"))
    assert spec.json_schema == judge_mod.SCHEMA and spec.isolate_env
    assert lanes.seen["head"] == head
    assert sorted(lanes.seen["refs"].split()) == [
        "refs/heads/main",
        "refs/heads/replay-head",
        "refs/pull/1/head",
        "refs/remotes/origin/main",
    ]
    lines = [json.loads(x) for x in (out / replayscore.JUDGED).read_text().splitlines()]
    verdicts = {x["id"]: x["verdict"] for x in lines}
    assert verdicts["j-1"]["real"] == "no" and verdicts["j-1"]["matches_label"] is None
    assert verdicts["m-1"]["matches_label"] == "yes"
    assert not list((work / "work" / "worktrees").iterdir())
    assert not list((work / "work" / "mirrors").iterdir())
    again = JudgeLanes()
    res = judge_mod.judge(cfg, out, lane_runner=again, echo=lambda s: None)
    assert res["pending"] == 0 and again.calls == []


def test_a_failed_lane_leaves_its_item_for_the_next_run(cfg, judged_out):
    out, _ = judged_out
    res = judge_mod.judge(
        cfg, out, lane_runner=JudgeLanes(fail={"Bad f again"}), echo=lambda s: None
    )
    assert res["judged"] == 1 and res["failed"] == 1
    assert [t["id"] for t in judge_mod.pending_items(out, 0)] == ["m-1"]


def test_judge_stops_on_a_short_disk(cfg, judged_out):
    out, _ = judged_out
    lanes = JudgeLanes()
    gate = Gate(out, min_free_gb=25, disk_free=lambda p: GB)
    res = judge_mod.judge(cfg, out, lane_runner=lanes, gate=gate, echo=lambda s: None)
    assert res["stopped"].startswith("disk:") and res["not_started"] == 2 and lanes.calls == []


def test_judge_stops_when_production_degrades(cfg, judged_out):
    out, _ = judged_out
    lanes = JudgeLanes()
    gate = Gate(out, guard=lambda: ("degraded mode is active", True))
    res = judge_mod.judge(cfg, out, lane_runner=lanes, gate=gate, echo=lambda s: None)
    assert res["stopped"] == "stopped: degraded mode is active" and lanes.calls == []


def test_judge_never_writes_the_production_mirror(cfg, judged_out):
    import os
    import stat

    out, _ = judged_out
    m = cfg.mirrors_dir / "o__r.git"
    paths = [m, *m.rglob("*")]
    modes = {p: p.stat().st_mode for p in paths}
    before = sorted((str(p), p.stat().st_mtime) for p in paths)
    for p in paths:
        os.chmod(p, modes[p] & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    try:
        res = judge_mod.judge(cfg, out, lane_runner=JudgeLanes(), echo=lambda s: None)
    finally:
        for p in reversed(paths):
            os.chmod(p, modes[p])
    assert res["judged"] == 2
    assert sorted((str(p), p.stat().st_mtime) for p in [m, *m.rglob("*")]) == before
