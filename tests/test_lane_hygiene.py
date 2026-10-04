"""Policy v12 lane hygiene: isolated lane environments, structured output through
`--json-schema`, and echo fields the harness stamps. All of it is off under a v11 policy,
whose lanes must start exactly as v0.24's did (pinned below, end to end through the real
`run_claude_lane`)."""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import threading
from pathlib import Path

import pytest

from reviewsys import lane as lane_mod
from reviewsys import schemas, worker
from reviewsys.config import LaneModel
from reviewsys.contract import CATEGORIES, SEVERITIES
from reviewsys.db import tx
from reviewsys.ingest import enqueue_head
from reviewsys.lane import LaneResult, LaneSpec
from reviewsys.models import FailKind, ReviewError, RunStatus, Trigger
from reviewsys.scheduler import schedule
from reviewsys.steps import worktree as wt
from tests.test_pipeline_policy import load_with

HEAD = "a" * 40
V024_ENV = {"GODEBUG": "netdns=go", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
ISOLATION = {"CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1", "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS": "1"}


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


def v024_argv(spec: LaneSpec) -> list[str]:
    """`lane.argv_for` as v0.24.0 shipped it, copied rather than imported: what a v11 lane's
    command line must stay."""
    sandbox = [lane_mod.SANDBOX_EXEC, "-p", spec.sandbox_profile] if spec.sandbox_profile else []
    if spec.resume and spec.session_id:
        session = ["--resume", spec.session_id]
    elif spec.session_id:
        session = ["--session-id", spec.session_id]
    else:
        session = ["--no-session-persistence"]
    argv = [
        *sandbox,
        "nice",
        "-n",
        "10",
        spec.claude_bin,
        "--bare",
        "--settings",
        '{"permissions": {"disableAutoMode": "disable"}}',
        "--permission-mode",
        "plan",
        "--model",
        spec.model,
        "--effort",
        spec.effort,
        "--add-dir",
        str(spec.add_dir),
        "--output-format",
        "json",
        *session,
        "--print",
    ]
    if spec.max_budget_usd:
        argv += ["--max-budget-usd", f"{spec.max_budget_usd:.2f}"]
    return argv


class ClaudeProcesses:
    """A lane runner that is the real `run_claude_lane`, with only the `claude` process faked:
    it records each lane's argv and environment exactly as Popen gets them, and answers with
    the canned output of `lanes` (conftest.FakeLanes) for that lane's spec."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, lanes) -> None:
        self.lanes = lanes
        self.started: list[tuple[LaneSpec, list[str], dict[str, str]]] = []
        self._lane = threading.local()
        real = subprocess.Popen
        outer = self

        class Proc:
            def __init__(self, argv, *, env, **kw):
                self.real = None
                if not hasattr(outer._lane, "spec"):
                    self.real = real(argv, env=env, **kw)
                    return
                spec, art, worktree = outer._lane.spec, outer._lane.art, outer._lane.worktree
                outer.started.append((spec, list(argv), dict(env)))
                res = outer.lanes(spec, art, worktree)
                self.out, self.err, self.returncode = res.stdout, res.stderr, res.exit_code

            def communicate(self, input=None, timeout=None):
                if self.real is not None:
                    return self.real.communicate(input, timeout=timeout)
                return self.out, self.err

        monkeypatch.setattr(lane_mod.subprocess, "Popen", Proc)

    def __call__(self, spec: LaneSpec, art: Path, worktree: Path) -> LaneResult:
        self._lane.spec, self._lane.art, self._lane.worktree = spec, art, worktree
        try:
            return lane_mod.run_claude_lane(spec, art, worktree)
        finally:
            del self._lane.spec


def _run(cfg, conn, gh, runner):
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.MENTION)
    (rid,) = schedule(conn, cfg, spawn=False)
    return rid, worker.main(cfg, conn, rid, gh=gh, lane_runner=runner, heartbeat=False)


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


def _review(lanes) -> None:
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier()
    lanes.broken_once = {"general"}  # one correction turn, resumed in the lane's session


# ---- v11: byte-identical lanes ----


def test_v11_lanes_start_exactly_as_v024(cfg, conn, gh, lanes, monkeypatch, _claude_config_dir):
    _review(lanes)
    procs = ClaudeProcesses(monkeypatch, lanes)
    _, status = _run(cfg, conn, gh, procs)
    assert status == RunStatus.DONE
    roles = {s.role for s, _, _ in procs.started}
    assert {"prep", "general", "verifier"} <= roles
    assert any(s.resume for s, _, _ in procs.started), "a correction turn ran"
    for spec, argv, env in procs.started:
        assert argv == v024_argv(spec), spec.role
        assert env == {**os.environ, **V024_ENV}, spec.role
        assert not spec.isolate_env


# ---- lane_env_isolation ----


def test_isolated_runner_marks_every_lane_and_only_when_on(tmp_path):
    seen: list[LaneSpec] = []

    def runner(spec, art, worktree):
        seen.append(spec)
        return LaneResult(exit_code=0, stdout="", stderr="", duration_s=0)

    assert lane_mod.isolated(runner, False) is runner
    spec = LaneSpec(
        role="general",
        agent="a",
        model="m",
        effort="high",
        prompt="p",
        cwd=tmp_path,
        add_dir=tmp_path,
        timeout_seconds=60,
        claude_bin="claude",
    )
    lane_mod.isolated(runner, True)(spec, tmp_path, tmp_path)
    assert seen[0].isolate_env and seen[0] == LaneSpec(**{**_fields(spec), "isolate_env": True})
    assert lane_mod.lane_env(seen[0]) == {**os.environ, **V024_ENV, **ISOLATION}
    assert lane_mod.lane_env(spec) == {**os.environ, **V024_ENV}


def _fields(spec: LaneSpec) -> dict:
    return {f: getattr(spec, f) for f in LaneSpec.__dataclass_fields__}


def test_v12_isolation_reaches_every_lane_and_correction_turn(
    cfg, conn, gh, lanes, monkeypatch, skills_dir, _claude_config_dir
):
    """The correction turn resumes the lane's session under the same CLAUDE_CONFIG_DIR, with
    the same isolated environment, and nothing else about the command line changes."""
    cfg = load_with(cfg, skills_dir, {"version": "v12", "lane_env_isolation": True})
    _review(lanes)
    procs = ClaudeProcesses(monkeypatch, lanes)
    _, status = _run(cfg, conn, gh, procs)
    assert status == RunStatus.DONE
    resumed = [(s, env) for s, _, env in procs.started if s.resume]
    assert resumed, "a correction turn ran"
    for spec, argv, env in procs.started:
        assert spec.isolate_env, spec.role
        assert env == {**os.environ, **V024_ENV, **ISOLATION}, spec.role
        assert env["CLAUDE_CONFIG_DIR"] == str(_claude_config_dir)
        assert argv == v024_argv(spec), "isolation is environment only"
    first = next(s for s, _, _ in procs.started if s.session_id == resumed[0][0].session_id)
    assert first is not resumed[0][0] and not first.resume


def test_v12_without_isolation_keeps_the_v11_environment(
    cfg, conn, gh, lanes, monkeypatch, skills_dir, _claude_config_dir
):
    cfg = load_with(cfg, skills_dir, {"version": "v12"})
    _review(lanes)
    procs = ClaudeProcesses(monkeypatch, lanes)
    _, status = _run(cfg, conn, gh, procs)
    assert status == RunStatus.DONE
    assert all(env == {**os.environ, **V024_ENV} for _, _, env in procs.started)


# ---- structured_output ----

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


def _spec(tmp_path, **kw) -> LaneSpec:
    base = {
        "role": "verifier",
        "agent": "a",
        "model": "gpt-6.1-sol",
        "effort": "high",
        "prompt": "p",
        "cwd": tmp_path,
        "add_dir": tmp_path,
        "timeout_seconds": 60,
        "claude_bin": "claude",
    }
    return LaneSpec(**{**base, **kw})


def _envelope(structured=None, *, result="Done.", subtype="success", session=None) -> LaneResult:
    """A finished lane as `run_claude_lane` reads Claude Code's JSON envelope."""
    env = {
        "type": "result",
        "subtype": subtype,
        "is_error": subtype != "success",
        "result": result,
        "num_turns": 4,
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }
    if structured is not None:
        env["structured_output"] = structured
    if session:
        env["session_id"] = session
    res = LaneResult(exit_code=0, stdout=json.dumps(env), stderr="", duration_s=1)
    lane_mod._extract_result(res)
    return res


def test_json_schema_goes_on_the_command_line_only_when_given(tmp_path):
    assert "--json-schema" not in lane_mod.argv_for(_spec(tmp_path))
    spec = _spec(tmp_path, json_schema=SCHEMA, session_id=lane_mod.new_session_id())
    argv = lane_mod.argv_for(spec)
    assert argv[argv.index("--json-schema") + 1] == json.dumps(SCHEMA, separators=(",", ":"))
    assert argv[: argv.index("--json-schema")] == v024_argv(spec), "appended, nothing else"
    resumed = lane_mod.argv_for(dataclasses.replace(spec, resume=True))
    assert "--resume" in resumed and "--json-schema" in resumed, "a correction turn keeps it"


def test_structured_output_is_read_before_the_text():
    res = _envelope({"ok": True}, result='Here it is: {"ok": false}')
    assert res.structured == {"ok": True}
    assert lane_mod.lane_output(res) == {"ok": True}
    plain = _envelope(None, result='{"ok": false}')
    assert plain.structured is None and lane_mod.lane_output(plain) == {"ok": False}


def test_exhausted_structured_retries_are_a_contract_failure():
    res = _envelope(None, result="", subtype="error_max_structured_output_retries")
    assert not res.ok
    with pytest.raises(ReviewError) as exc:
        lane_mod.lane_output(res)
    assert exc.value.kind is FailKind.CONTRACT
    assert "4 turns" in exc.value.message and "exit" not in exc.value.message
    budget = _envelope(None, result="", subtype="error_max_budget_usd")
    with pytest.raises(ReviewError) as other:
        lane_mod.lane_output(budget)
    assert other.value.kind is FailKind.INFRA, "other error envelopes stay infrastructure"


def _checked_ctx(cfg, conn, tmp_path, runner) -> worker.RunContext:
    with tx(conn):
        enqueue_head(conn, cfg, "dashpay/platform", 1, HEAD, Trigger.MENTION)
    (rid,) = schedule(conn, cfg, spawn=False)
    head_id = conn.execute("SELECT head_id FROM runs WHERE id=?", (rid,)).fetchone()[0]
    return worker.RunContext(
        cfg=cfg,
        main_conn=conn,
        gh=None,
        run_id=rid,
        head_id=head_id,
        repo="dashpay/platform",
        number=1,
        sha=HEAD,
        token="t",
        run_dir=tmp_path / "run",
        worktree=tmp_path,
        # what lanepool.gated does for every lane: the correction turns run in its slot
        lane_runner=lambda spec, art, wt: lane_mod.run_with_corrections(runner, spec, art, wt),
    )


def _verifier_check() -> worker.OutputCheck:
    def validate(raw):
        if raw.get("ok") is not True:
            raise ReviewError(FailKind.CONTRACT, "ok must be true")
        return raw

    return worker.OutputCheck(validate=validate, kind="verifier", expected_phase="final")


def _checked(ctx, **kw):
    return worker._run_checked_lane(
        ctx,
        phase="verify2",
        role="verifier",
        lm=LaneModel(agent="sol-verifier", model="gpt-6.1-sol"),
        prompt="verify",
        is_verifier=True,
        check=_verifier_check(),
        schema=SCHEMA,
        **kw,
    )


@pytest.mark.parametrize(
    ("pipeline", "sent"),
    [
        (None, False),
        ({"version": "v12"}, False),
        ({"version": "v12", "structured_output": True}, True),
    ],
)
def test_a_lane_answers_through_its_schema_only_when_the_policy_says_so(
    cfg, conn, skills_dir, tmp_path, _claude_config_dir, pipeline, sent
):
    cfg = load_with(cfg, skills_dir, pipeline, refingerprint=pipeline is not None)
    calls: list[LaneSpec] = []

    def runner(spec, art, worktree):
        calls.append(spec)
        if spec.json_schema is not None:
            return _envelope({"ok": True}, result="Structured output provided.")
        return _envelope(None, result='{"ok": true}')

    out = _checked(_checked_ctx(cfg, conn, tmp_path, runner))
    assert out == {"ok": True}
    assert [c.json_schema for c in calls] == [SCHEMA if sent else None]


def test_correction_turns_stay_the_fallback_for_a_schema_valid_answer(
    cfg, conn, skills_dir, tmp_path, _claude_config_dir
):
    """The schema checks shape; the contract checks meaning. A schema-valid answer the
    contract rejects is corrected in the lane's own session, through the schema again."""
    cfg = load_with(cfg, skills_dir, {"version": "v12", "structured_output": True})
    calls: list[LaneSpec] = []

    def runner(spec, art, worktree):
        calls.append(spec)
        return _envelope({"ok": spec.resume}, session=spec.session_id)

    ctx = _checked_ctx(cfg, conn, tmp_path, runner)
    assert _checked(ctx) == {"ok": True}
    first, again = calls
    assert again.resume and again.session_id == first.session_id
    assert again.json_schema == first.json_schema == SCHEMA
    assert "ok must be true" in again.prompt
    row = conn.execute("SELECT status FROM lanes WHERE run_id=?", (ctx.run_id,)).fetchone()
    assert row["status"] == "corrected"


def test_exhausted_structured_retries_fail_the_lane_as_a_contract_failure(
    cfg, conn, skills_dir, tmp_path, _claude_config_dir
):
    cfg = load_with(cfg, skills_dir, {"version": "v12", "structured_output": True})
    calls: list[LaneSpec] = []

    def runner(spec, art, worktree):
        calls.append(spec)
        return _envelope(None, result="", subtype="error_max_structured_output_retries")

    ctx = _checked_ctx(cfg, conn, tmp_path, runner)
    with pytest.raises(worker.LaneFailed) as exc:
        _checked(ctx)
    assert exc.value.kind is FailKind.CONTRACT
    assert len(calls) == 2 and not any(c.resume for c in calls), "retried whole, not resumed"
    assert not any(c.role == "repair" for c in calls)
    reasons = [
        r["reason"] for r in conn.execute("SELECT reason FROM lanes WHERE run_id=?", (ctx.run_id,))
    ]
    assert len(reasons) == 2 and all("structured output retries" in r for r in reasons)


# ---- the schemas ----


def _walk(node, path=""):
    """Every (path, property schema) in a schema, recursively."""
    for name, prop in (node.get("properties") or {}).items():
        yield f"{path}.{name}", prop
        yield from _walk(prop, f"{path}.{name}")
    if isinstance(node.get("items"), dict):
        yield f"{path}[]", node["items"]
        yield from _walk(node["items"], f"{path}[]")


ALL_SCHEMAS = {
    "finder": schemas.FINDER_SCHEMA,
    "verifier": schemas.VERIFIER_SCHEMA,
    "thread": schemas.THREAD_SCHEMA,
    "verifier(ids)": schemas.verifier_schema(["p1-general-1", "p2-general-3"]),
    "thread(hashes)": schemas.thread_schema(["0123456789ab", "ba9876543210"]),
}


@pytest.mark.parametrize("name", ALL_SCHEMAS)
def test_every_allowed_value_is_stated_in_prose_too(name):
    """Gemini strips `enum` from tool schemas: the description must carry the values."""
    schema = ALL_SCHEMAS[name]
    json.dumps(schema)  # plain data
    constrained = [(p, s) for p, s in _walk(schema) if "enum" in s]
    assert constrained
    for path, prop in constrained:
        for value in prop["enum"]:
            assert value in prop["description"], f"{name}{path}: {value!r} not in prose"
    for path, prop in _walk(schema):
        assert "pattern" not in prop or "Exactly one of" in prop["description"], path


@pytest.mark.parametrize("name", ALL_SCHEMAS)
def test_no_echo_fields_no_caps_no_nulls(name):
    schema = ALL_SCHEMAS[name]
    names = {p.rsplit(".", 1)[-1] for p, _ in _walk(schema)}
    assert not names & {"review_phase", "head_sha", "confidence"}
    for path, prop in _walk(schema):
        assert "null" not in json.dumps(prop.get("type")), path
        if not name.startswith("thread("):
            assert "maxItems" not in prop, path


def test_schema_values_follow_the_contract():
    finder = schemas.FINDER_SCHEMA["properties"]["candidates"]["items"]["properties"]
    assert finder["severity"]["enum"] == list(SEVERITIES)
    assert finder["category"]["enum"] == list(CATEGORIES)
    group = schemas.VERIFIER_SCHEMA["properties"]["groups"]["items"]["properties"]
    assert group["verdict"]["enum"] == ["CONFIRMED", "PLAUSIBLE", "REFUTED"]
    row = schemas.THREAD_SCHEMA["properties"]["threads"]["items"]["properties"]
    assert row["status"]["enum"] == [
        "STILL_VALID",
        "FIXED",
        "OUTDATED",
        "WITHDRAWN",
        "INTENTIONALLY_DEFERRED",
    ]
    assert "NO_REPLY" in row["reply"]["description"]


@pytest.mark.parametrize(
    ("match", "ok"),
    [
        ("new", True),
        ("open:0123456789ab", True),
        ("settled:0123456789ab", True),
        ("fixed:0123456789ab", True),
        ("coderabbit:2412", True),
        ("open:0123", False),
        ("open:0123456789AB", False),
        ("closed:0123456789ab", False),
        ("NEW", False),
    ],
)
def test_match_pattern(match, ok):
    assert bool(re.match(schemas.MATCH_PATTERN, match)) is ok


def test_narrowed_schemas_pin_the_ids_and_leave_the_base_alone():
    ids = ["p1-general-1", "p2-general-3"]
    v = schemas.verifier_schema(ids)
    assert v["properties"]["groups"]["items"]["properties"]["members"]["items"]["enum"] == ids
    hashes = ["0123456789ab", "ba9876543210"]
    t = schemas.thread_schema(hashes)["properties"]["threads"]
    assert t["minItems"] == t["maxItems"] == 2
    assert t["items"]["properties"]["finding_hash"]["enum"] == hashes
    assert "enum" not in json.dumps(
        schemas.THREAD_SCHEMA["properties"]["threads"]["items"]["properties"]["finding_hash"]
    )
    assert "minItems" not in schemas.THREAD_SCHEMA["properties"]["threads"]
