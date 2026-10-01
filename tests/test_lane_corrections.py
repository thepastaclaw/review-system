"""A lane whose answer breaks the output contract is corrected in its own session: the
deterministic normalization first, then follow-up turns, then (only when the session cannot
be resumed) the context-free repair lane. Fixtures follow the shapes seen on the box
2026-09-26..10-01."""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import threading

import pytest

from reviewsys import config as cfg_mod
from reviewsys import lane as lane_mod
from reviewsys import lanepool, worker
from reviewsys.contract import finding_hash
from reviewsys.db import tx
from reviewsys.ingest import enqueue_head
from reviewsys.lane import LaneResult, LaneSpec
from reviewsys.models import RunStatus, Trigger
from reviewsys.scheduler import schedule
from reviewsys.steps import worktree as wt

HEAD = "a" * 40


@pytest.fixture(autouse=True)
def fake_git(monkeypatch, tmp_path):
    monkeypatch.setattr(wt, "ensure_mirror", lambda mirrors, repo: tmp_path / "mirror")
    monkeypatch.setattr(wt, "fetch_head", lambda mirror, number, sha: None)

    def create(mirror, wts, name, sha):
        p = wts / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(wt, "create_worktree", create)
    monkeypatch.setattr(wt, "remove_worktree", lambda mirror, path: None)
    monkeypatch.setattr(wt, "merge_base", lambda worktree, base, sha: "b" * 40)


def _verifier():
    return {
        "summary": "ok",
        "review_action": "COMMENT",
        "findings": [],
        "dropped_findings": [],
        "out_of_scope_findings": [],
        "coderabbit_reactions": [],
        "prerequisite_adjudications": [],
        "adjudication_complete": True,
    }


def _run(cfg, conn, gh, runner):
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.MENTION)
    (rid,) = schedule(conn, cfg, spawn=False)
    return rid, worker.main(cfg, conn, rid, gh=gh, lane_runner=runner, heartbeat=False)


def _answer(text: str, *, session: str | None = None) -> LaneResult:
    env = {"result": text, "usage": {"input_tokens": 100, "output_tokens": 10}}
    if session:
        env["session_id"] = session
    return LaneResult(
        exit_code=0,
        stdout=json.dumps(env),
        stderr="",
        duration_s=1,
        tokens_in=100,
        tokens_out=10,
        result_text=text,
        session_id=session,
    )


def _events(conn, rid, kind):
    return [
        r[0]
        for r in conn.execute(
            "SELECT detail FROM events WHERE run_id=? AND kind=? ORDER BY id", (rid, kind)
        )
    ]


def _lane(conn, rid, phase, role):
    return conn.execute(
        "SELECT * FROM lanes WHERE run_id=? AND phase=? AND role=? ORDER BY id DESC",
        (rid, phase, role),
    ).fetchone()


# ---- lane.py: argv, session id, the correction loop ----


def _spec(tmp_path, **kw):
    base = {
        "role": "general",
        "agent": "a",
        "model": "glm-5.3-flash",
        "effort": "high",
        "prompt": "p",
        "cwd": tmp_path,
        "add_dir": tmp_path,
        "timeout_seconds": 7200,
        "claude_bin": "claude",
    }
    return LaneSpec(**{**base, **kw})


def test_session_flags_replace_no_persistence_only_for_correctable_lanes(tmp_path):
    sid = lane_mod.new_session_id()
    plain = lane_mod.argv_for(_spec(tmp_path))
    assert "--no-session-persistence" in plain and "--session-id" not in plain
    kept = lane_mod.argv_for(_spec(tmp_path, session_id=sid))
    assert kept[kept.index("--session-id") + 1] == sid and "--no-session-persistence" not in kept
    resumed = lane_mod.argv_for(_spec(tmp_path, session_id=sid, resume=True))
    assert resumed[resumed.index("--resume") + 1] == sid and "--session-id" not in resumed
    assert resumed[-1] == "--print", "the prompt still goes on stdin"


def test_extract_result_reads_the_session_id():
    res = LaneResult(
        exit_code=0,
        stdout=json.dumps({"type": "result", "result": "{}", "session_id": "s-1"}),
        stderr="",
        duration_s=1,
    )
    lane_mod._extract_result(res)
    assert res.session_id == "s-1"


def test_corrections_resume_the_same_session_and_delete_it(tmp_path, _claude_config_dir):
    sid = lane_mod.new_session_id()
    project = _claude_config_dir / "projects" / lane_mod.project_slug(tmp_path)
    project.mkdir(parents=True)
    calls: list[LaneSpec] = []
    marks: list[tuple[str, str | None]] = []

    def runner(spec, art, worktree):
        calls.append(spec)
        (project / f"{sid}.jsonl").write_text("{}\n")  # what claude writes for the session
        return _answer("bad" if len(calls) < 3 else '{"ok": 1}', session=sid)

    asked: list[str] = []

    def check(turn):
        if turn.result_text.startswith("{"):
            return None
        asked.append(turn.result_text)
        return f"fix #{len(asked)}"

    spec = _spec(tmp_path, session_id=sid, check=check, corrections=2)
    res = lane_mod.run_with_corrections(
        runner, spec, tmp_path / "art", tmp_path, mark=lambda d, s, st: marks.append((d.name, st))
    )
    assert [t.result_text for t in res.followups] == ["bad", '{"ok": 1}']
    assert [c.prompt for c in calls] == ["p", "fix #1", "fix #2"]
    assert all(c.resume and c.session_id == sid and c.check is None for c in calls[1:])
    # a correction only re-emits an answer: bounded well below the lane's own timeout
    assert {c.timeout_seconds for c in calls[1:]} == {lane_mod.CORRECTION_TIMEOUT_SECONDS}
    assert marks == [("correction-1", "running"), ("correction-2", "running")]
    assert project.is_dir() and not any(project.iterdir()), "the transcript is gone"


def test_no_correction_after_a_turn_that_did_not_finish(tmp_path):
    calls: list[LaneSpec] = []

    def runner(spec, art, worktree):
        calls.append(spec)
        return LaneResult(exit_code=None, stdout="", stderr="", duration_s=1, timed_out=True)

    spec = _spec(
        tmp_path, session_id=lane_mod.new_session_id(), check=lambda t: "again", corrections=2
    )
    res = lane_mod.run_with_corrections(runner, spec, tmp_path / "art", tmp_path)
    assert len(calls) == 1 and res.followups == []


def test_corrections_run_inside_the_lane_slot(cfg, conn, tmp_path):
    """The correction turn neither releases the slot nor lines up again: a lane that lined up
    for the pool's only slot while the first one ran starts after its corrections, never
    between them."""
    cfg = dataclasses.replace(cfg, lane_pools={"muse": 1})
    slot_dir = cfg.work_dir / "lane-slots"
    order: list[str] = []
    threads: list[threading.Thread] = []

    def inner(spec, art, worktree):
        order.append(f"{spec.role}{'+' if spec.resume else ''}")
        if spec.role == "first" and not spec.resume:
            # the first lane holds the slot: line the second one up behind it, and wait
            # until it is in the pool's line before answering
            threads.append(threading.Thread(target=other))
            threads[0].start()
            for _ in range(500):
                if lanepool.waiting_line(slot_dir, "muse"):
                    break
                threading.Event().wait(0.01)
            assert lanepool.waiting_line(slot_dir, "muse"), "the second lane never lined up"
        return _answer("{}" if spec.resume or spec.role == "second" else "not json")

    def other():
        own = sqlite3.connect(str(cfg.db_path))  # a lane thread has its own connection
        try:
            g = lanepool.gated(inner, lambda: own, cfg, lambda: False, "live", 0)
            spec = _spec(tmp_path, role="second", model="muse-spark-1.3-contributor")
            g(spec, tmp_path / "b", tmp_path)
        finally:
            own.close()

    gated = lanepool.gated(inner, lambda: conn, cfg, lambda: False, "live", 0)
    spec = _spec(
        tmp_path,
        role="first",
        model="muse-spark-1.3-contributor",
        session_id=lane_mod.new_session_id(),
        check=lambda t: None if t.result_text == "{}" else "again",
        corrections=1,
    )
    res = gated(spec, tmp_path / "a", tmp_path)
    threads[0].join(10)
    assert len(res.followups) == 1
    assert order == ["first", "first+", "second"]
    assert (tmp_path / "a" / "correction-1" / lanepool.LANE_STATE).exists()


# ---- worker: the paths a broken answer takes ----


def test_broken_json_is_corrected_in_the_lanes_own_session(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier()
    lanes.broken_once = {"general"}
    rid, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    assert not any(s.role == "repair" for s in lanes.calls), "the session was resumable"
    first, again = [s for s in lanes.calls if s.role == "general" and s.model == "glm-5.3-flash"]
    assert first.session_id and again.resume and again.session_id == first.session_id
    assert again.model == first.model and again.effort == first.effort
    assert "model output is not a JSON object" in again.prompt
    assert '`head_sha` is exactly "' + HEAD + '"' in again.prompt
    row = _lane(conn, rid, "phase1", "general")
    assert row["status"] == "corrected" and row["attempt"] == 1
    assert (row["tokens_in"], row["tokens_out"]) == (200, 20), "both turns are counted"
    (ev,) = _events(conn, rid, "lane.corrected")
    assert ev.startswith("phase1/general glm-5.3-flash turns=1: first error: model output")


def test_a_contract_error_is_corrected_with_the_prior_hashes_and_rules(cfg, conn, gh, lanes):
    """A re-review whose reconciliation is wrong in a way normalization cannot fix (a prior
    hash left out) is told every prior hash, the statuses and the carry rule."""
    prior = finding_hash("f.rs", "general", "Old finding")
    with tx(conn):  # a finding posted on this PR by an earlier review
        enqueue_head(conn, cfg, "dashpay/platform", 1, "c" * 40, Trigger.NEW_PR)
        old = conn.execute("SELECT id FROM heads WHERE sha=?", ("c" * 40,)).fetchone()[0]
        conn.execute("UPDATE heads SET status='done' WHERE id=?", (old,))
        run = conn.execute(
            "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at) "
            "VALUES (?,1,'done','t','2026-01-01T00:00:00Z','2026-01-01T06:00:00Z')",
            (old,),
        ).lastrowid
        conn.execute(
            "INSERT INTO findings (run_id, phase, stage, hash, file, severity, title, body) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (run, "final", "posted", prior, "f.rs", "suggestion", "Old finding", "b"),
        )
        conn.execute(
            "INSERT INTO posted_findings (repo, number, hash, sha, posted_at) "
            "VALUES ('dashpay/platform', 1, ?, ?, '2026-01-01T00:00:00Z')",
            (prior, "c" * 40),
        )
    good = {
        "summary": "ok",
        "findings": [],
        "out_of_scope_findings": [],
        "prior_finding_reconciliation": [{"finding_hash": prior, "status": "FIXED"}],
    }
    lanes.reviewer["default"] = good
    lanes.verifier["default"] = _verifier()
    seen: list[LaneSpec] = []

    def runner(spec, art, worktree):
        if spec.role == "general" and spec.model == "glm-5.3-flash" and not spec.resume:
            seen.append(spec)
            text = json.dumps({**good, "prior_finding_reconciliation": [], "head_sha": HEAD})
            return _answer(text)
        if spec.resume:
            seen.append(spec)
        return lanes(spec, art, worktree)

    rid, status = _run(cfg, conn, gh, runner)
    assert status == RunStatus.DONE
    _first, again = seen
    assert again.resume and f"reconciliation missing prior hashes: ['{prior}']" in again.prompt
    assert f'"finding_hash": "{prior}"' in again.prompt and "STILL_VALID" in again.prompt
    assert "whose `title` equals its `original_title`" in again.prompt
    assert _lane(conn, rid, "phase1", "general")["status"] == "corrected"


def test_correction_exhausted_fails_the_attempt_as_before(cfg, conn, gh, lanes):
    """Still invalid after every correction turn (and the repair lane): the attempt fails, the
    lane gets its retry (corrections again), then Phase 1 falls through as for any other
    Phase-1 failure."""
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier()

    def runner(spec, art, worktree):
        if (spec.role == "general" and spec.model == "glm-5.3-flash") or spec.role == "repair":
            lanes.calls.append(spec)
            return _answer("I reviewed it and everything looks fine.")
        return lanes(spec, art, worktree)

    rid, status = _run(cfg, conn, gh, runner)
    assert status == RunStatus.DONE
    assert "Phase 2 only (Phase 1 failed)" in gh.posted_reviews[0]["body"]
    general = [s for s in lanes.calls if s.role == "general" and s.model == "glm-5.3-flash"]
    assert [s.resume for s in general] == [False, True, True] * 2, "2 attempts x 2 corrections"
    # still no JSON object after the corrections: the repair lane is the last resort, per attempt
    assert sum(s.role == "repair" for s in lanes.calls) == 2
    failed = _events(conn, rid, "lane.correction_failed")
    assert len(failed) == 2 and all("turns=2 (still invalid)" in d for d in failed)
    rows = conn.execute(
        "SELECT status, reason FROM lanes WHERE run_id=? AND phase='phase1' AND role='general'",
        (rid,),
    ).fetchall()
    assert [r["status"] for r in rows] == ["failed", "failed"]
    assert all("(after 2 correction turns)" in r["reason"] for r in rows)
    step = json.loads(
        conn.execute(
            "SELECT detail FROM steps WHERE run_id=? AND name='phase1'", (rid,)
        ).fetchone()[0]
    )
    assert "phase1/general lane failed twice" in step["error"]


def test_resume_unavailable_falls_back_to_repair_with_normalization(cfg, conn, gh, lanes):
    """The session cannot be resumed (the follow-up exits 1, as `--resume` does for a missing
    session): the repair lane gets the answer, told the echo fields and prior hashes, and its
    output is normalized like a lane's (a repaired output that guessed `review_phase` and cut
    the sha used to fail the contract 44% of the time)."""
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier()
    # a reviewer answer with a slip in each echo field, cut off before its closing brace
    broken = json.dumps(
        {
            "summary": "ok",
            "findings": [],
            "out_of_scope_findings": [],
            "review_phase": "final",
            "head_sha": HEAD[:10],
        }
    )[:-1]
    repair_prompts: list[str] = []

    def runner(spec, art, worktree):
        if spec.role == "general" and spec.model == "glm-5.3-flash":
            if spec.resume:
                return LaneResult(
                    exit_code=1,
                    stdout="",
                    stderr=f"No conversation found with session ID: {spec.session_id}",
                    duration_s=1,
                )
            return _answer(broken)
        if spec.role == "repair":
            repair_prompts.append(spec.prompt)
        return lanes(spec, art, worktree)

    rid, status = _run(cfg, conn, gh, runner)
    assert status == RunStatus.DONE
    (prompt,) = repair_prompts
    assert '`review_phase` is exactly "preliminary"' in prompt and HEAD in prompt
    row = _lane(conn, rid, "phase1", "general")
    assert row["status"] == "repaired" and "normalized: review_phase 'final'" in row["reason"]
    (failed,) = _events(conn, rid, "lane.correction_failed")
    assert "(resume failed: No conversation found" in failed
    assert "rescued by the repair lane" in failed
    assert not _events(conn, rid, "lane.corrected")
    (norm,) = _events(conn, rid, "lane.output_normalized")
    assert norm.startswith("phase1/general glm-5.3-flash: review_phase 'final' -> 'preliminary'")
    assert f"head_sha '{HEAD[:10]}' -> {HEAD}" in norm


def test_gemini_output_without_echo_fields_is_accepted_and_noted(cfg, conn, gh, lanes):
    """Run 2883: a Gemini lane returned findings and out_of_scope_findings only, which failed
    the whole Phase 1 after 28 minutes of good work."""
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier()

    def runner(spec, art, worktree):
        if spec.role == "general" and spec.model == "glm-5.3-flash":
            lanes.calls.append(spec)
            return _answer(json.dumps({"findings": [], "out_of_scope_findings": []}))
        return lanes(spec, art, worktree)

    rid, status = _run(cfg, conn, gh, runner)
    assert status == RunStatus.DONE
    assert not any(s.resume for s in lanes.calls), "nothing for a correction turn to do"
    assert _lane(conn, rid, "phase1", "general")["status"] == "completed"
    (norm,) = _events(conn, rid, "lane.output_normalized")
    assert "review_phase missing -> 'preliminary'" in norm and "head_sha missing" in norm
    verifier = next(s for s in lanes.calls if s.role == "verifier")
    assert f'"head_sha": "{HEAD}"' in verifier.prompt, "the verifier sees the normalized output"


def test_verifier_contract_error_is_corrected_too(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier()
    lanes.verifier["preliminary"] = {**_verifier(), "adjudication_complete": "yes"}

    def runner(spec, art, worktree):
        if spec.role == "verifier" and spec.resume:
            lanes.verifier["preliminary"] = _verifier()
        return lanes(spec, art, worktree)

    rid, status = _run(cfg, conn, gh, runner)
    assert status == RunStatus.DONE
    again = next(s for s in lanes.calls if s.role == "verifier" and s.resume)
    assert "adjudication_complete must be true" in again.prompt
    assert "`coderabbit_reactions` has exactly one" in again.prompt
    assert _lane(conn, rid, "verify1", "verifier")["status"] == "corrected"


def test_correction_turns_off_keeps_no_session(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier()
    _rid, status = _run(dataclasses.replace(cfg, lane_correction_turns=0), conn, gh, lanes)
    assert status == RunStatus.DONE
    assert all(not s.session_id and s.check is None for s in lanes.calls)


def test_config_knob_is_clamped(tmp_path, skills_dir):
    from reviewsys import config as cfg_mod

    for value, want in (("7", 5), ("-1", 0), ("1", 1)):
        p = tmp_path / f"c{value}.toml"
        text = cfg_mod.DEFAULT_TOML.replace(
            'skills = "~/Projects/skills"', f'skills = "{skills_dir}"'
        ).replace("correction_turns = 2", f"correction_turns = {value}")
        p.write_text(text)
        assert cfg_mod.load(p).lane_correction_turns == want
    no_knob = tmp_path / "old.toml"
    no_knob.write_text(
        cfg_mod.DEFAULT_TOML.replace(
            'skills = "~/Projects/skills"', f'skills = "{skills_dir}"'
        ).replace("correction_turns = 2", "")
    )
    assert cfg_mod.load(no_knob).lane_correction_turns == cfg_mod.DEFAULT_CORRECTION_TURNS


def test_gc_sweeps_sessions_a_dead_worker_left(cfg, conn, _claude_config_dir):
    import os
    import time

    from reviewsys import gc

    wts = {
        name: cfg.worktrees_dir / f"dashpay-platform-1-{n}"
        for n, name in enumerate(("done", "empty", "running"), 7)
    }
    with tx(conn):
        hid = conn.execute(
            "INSERT INTO heads (repo, number, sha, trigger, status, queued_at, eligible_at) "
            "VALUES ('dashpay/platform', 1, ?, 'new_pr', 'done', '2026-01-01', '2026-01-01')",
            ("d" * 40,),
        ).lastrowid
        for name, path in wts.items():
            conn.execute(
                "INSERT INTO runs (head_id, attempt, status, token, started_at, deadline_at, "
                "worktree) VALUES (?,1,?,?,'2026-01-01','2026-01-02',?)",
                (hid, "running" if name == "running" else "done", name, str(path)),
            )
    projects = _claude_config_dir / "projects"
    dirs = {name: projects / lane_mod.project_slug(p) for name, p in wts.items()}
    # a sibling directory whose slug only shares the prefix, and somebody else's project
    alike = projects / (lane_mod.project_slug(cfg.worktrees_dir) + "-old-dashpay-platform-1-7")
    other = projects / "-Users-someone-else"
    old = time.time() - 2 * 86400
    for d in (dirs["done"], dirs["running"], alike, other):
        d.mkdir(parents=True)
        (d / "old.jsonl").write_text("{}")
        os.utime(d / "old.jsonl", (old, old))
    (dirs["done"] / "fresh.jsonl").write_text("{}")
    dirs["empty"].mkdir(parents=True)
    assert gc.run(conn, cfg)["lane_sessions_removed"] == 1
    assert not (dirs["done"] / "old.jsonl").exists() and (dirs["done"] / "fresh.jsonl").exists()
    assert not dirs["empty"].exists(), "a finished run's emptied project dir goes"
    assert (dirs["running"] / "old.jsonl").exists(), "a run in flight keeps its sessions"
    for d in (alike, other):
        assert (d / "old.jsonl").exists(), "never anything outside this box's run worktrees"


def test_project_slug_matches_claude_code(tmp_path):
    # observed with Claude Code 2.1.286: cwd /private/tmp/claude-501/ccexp/wt
    d = tmp_path / "a.b_c"
    d.mkdir()
    assert lane_mod.project_slug(d) == str(d.resolve()).replace("/", "-").replace(".", "-").replace(
        "_", "-"
    )


def test_live_lanes_show_a_correction_turn(tmp_path):
    from reviewsys.db import fmt_ts, now_dt
    from reviewsys.exporter import _live_lanes

    turn = tmp_path / "attempts" / "phase2-general-aa" / "correction-1"
    turn.mkdir(parents=True)
    (turn.parent / "lane-meta.json").write_text("{}")  # the first turn ended
    (turn / lanepool.LANE_STATE).write_text(
        json.dumps(
            {"state": "running", "role": "general", "since": fmt_ts(now_dt()), "pool": "gpt"}
        )
    )
    (live,) = _live_lanes(str(tmp_path), tmp_path / "slots", now_dt())
    assert (live["phase"], live["role"]) == ("phase2", "general (correction 1)")


def test_repair_prompt_keeps_the_raw_answer_last():
    from reviewsys.prompts import repair_prompt

    p = repair_prompt(
        '{"a": 1', kind="verifier", expected_phase="final", head_sha=HEAD, prior_hashes=["x" * 12]
    )
    assert p.endswith('Do not add fences.\n\n{"a": 1')
    assert "coderabbit_reactions[]" in p and '`review_phase` is exactly "final"' in p
    assert HEAD not in p, "a verifier has no head_sha field"
    assert '"xxxxxxxxxxxx"' in p
    generic = repair_prompt("x", kind="", expected_phase="", head_sha=HEAD, prior_hashes=[])
    assert "schema" not in generic and "review_phase" not in generic


def test_a_lane_told_to_stop_between_turns_is_stopped_not_failed(cfg, conn, gh, lanes):
    """A sibling fails for good just as this lane's (invalid) answer comes back: the lane is
    recorded as stopped, never a failed attempt, a correction, a retry or a ladder fall."""
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier()
    calls: list[LaneSpec] = []
    running = threading.Event()

    def runner(spec, art, worktree):
        if spec.model == "glm-5.3-flash" and spec.role == "security-auditor":
            calls.append(spec)
            running.set()
            for _ in range(1000):  # answers only once the phase has been abandoned
                if spec.should_stop and spec.should_stop():
                    break
                threading.Event().wait(0.01)
            return _answer("not json")
        if spec.model == "glm-5.3-flash" and spec.role == "general":
            assert running.wait(10)  # fails only once its sibling is in flight
            return LaneResult(exit_code=1, stdout="", stderr="launcher crashed", duration_s=1)
        return lanes(spec, art, worktree)

    rid, status = _run(cfg, conn, gh, runner)
    assert status == RunStatus.DONE, "Phase 1 fell through to Phase 2"
    assert len(calls) == 1 and not calls[0].resume, "no correction for a stopped lane"
    rows = conn.execute(
        "SELECT status, reason FROM lanes WHERE run_id=? AND phase='phase1' "
        "AND role='security-auditor'",
        (rid,),
    ).fetchall()
    assert [r["status"] for r in rows] == ["cancelled"]
    assert rows[0]["reason"].startswith("stopped: phase1/general failed (lane exit 1")
    assert not _events(conn, rid, "phase1.model_fallback")
    assert not any(s.role == "repair" for s in lanes.calls)


def test_a_timed_out_correction_turn_leaves_the_answer_before_it(cfg, conn, gh, lanes):
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier()

    def runner(spec, art, worktree):
        if spec.role == "general" and spec.model == "glm-5.3-flash":
            if spec.resume:
                return LaneResult(
                    exit_code=None, stdout="", stderr="", duration_s=1, timed_out=True
                )
            return _answer("Findings: none.")
        if spec.role == "repair":
            return _answer(json.dumps({"summary": "ok", "findings": []}))
        return lanes(spec, art, worktree)

    rid, status = _run(cfg, conn, gh, runner)
    assert status == RunStatus.DONE
    assert _lane(conn, rid, "phase1", "general")["status"] == "repaired"
    (failed,) = _events(conn, rid, "lane.correction_failed")
    assert "(correction turn timed out)" in failed


def test_a_quota_failure_on_a_correction_turn_flips_the_run_degraded(cfg, conn, gh, lanes):
    """The correction turn of a primary-model verifier dies on a dry pool: the run goes to
    the stand-in as it would for a first turn, instead of retrying into the same 429."""
    stand_in = cfg_mod.Substitute(model="muse-spark-1.3-contributor", effort_cap="xhigh")
    pol = dataclasses.replace(
        cfg.policy,
        degraded=cfg_mod.DegradedPolicy(
            sentinel="gpt-6-astra", substitutes={"gpt-6-astra": stand_in}
        ),
    )
    cfg = dataclasses.replace(cfg, policy=pol, backlog_skip_phase1_above=0)
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier()

    def runner(spec, art, worktree):
        if spec.role == "verifier" and spec.model == "gpt-6-astra":
            lanes.calls.append(spec)
            if spec.resume:
                return LaneResult(
                    exit_code=1,
                    stdout="",
                    stderr="API Error: 429 All credentials for model gpt-6-astra are cooling down",
                    duration_s=1,
                )
            return _answer(json.dumps({**_verifier(), "adjudication_complete": False}))
        return lanes(spec, art, worktree)

    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.MENTION)
    (rid,) = schedule(conn, cfg, spawn=False)
    status = worker.main(
        cfg, conn, rid, gh=gh, lane_runner=runner, heartbeat=False, prober=lambda m: (False, "")
    )
    assert status == RunStatus.DONE
    assert conn.execute("SELECT degraded FROM runs WHERE id=?", (rid,)).fetchone()[0] == 1
    assert any(
        s.role == "verifier" and s.model == "muse-spark-1.3-contributor" for s in lanes.calls
    )


def test_claude_config_dir_knob_is_exported_by_the_cli(tmp_path, skills_dir, monkeypatch):
    """The box launcher forces CLAUDE_CONFIG_DIR=~/.claude-proxy; `[lanes] claude_config_dir`
    makes every reviewsys process (and so its lanes and the session cleanup) use that dir."""
    import argparse

    from reviewsys import cli

    target = tmp_path / "proxy-config"
    base = cfg_mod.DEFAULT_TOML.replace('skills = "~/Projects/skills"', f'skills = "{skills_dir}"')
    unset = tmp_path / "unset.toml"
    unset.write_text(base)
    assert cfg_mod.load(unset).lane_claude_config_dir is None
    p = tmp_path / "config.toml"
    p.write_text(
        base.replace('# claude_config_dir = "~/.claude-proxy"', f'claude_config_dir = "{target}"')
    )
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    cfg = cli._cfg(argparse.Namespace(config=str(p)))
    assert cfg.lane_claude_config_dir == target
    assert lane_mod.claude_projects_dir() == target / "projects"
