"""Policy v12 lane hygiene: isolated lane environments (with the repo's own agent guidance
moved into the prompt), structured output through `--json-schema`, and echo fields the
harness stamps. All of it is off under a v11 policy,
whose lanes must start exactly as production's (main at v0.28.0) do (pinned below, end to end
through the real `run_claude_lane`)."""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import threading
import uuid
from pathlib import Path

import pytest

from reviewsys import lane as lane_mod
from reviewsys import prompts, schemas, worker
from reviewsys.config import LaneModel
from reviewsys.contract import (
    CATEGORIES,
    SEVERITIES,
    parse_reviewer_output,
    parse_verifier_output,
)
from reviewsys.db import tx
from reviewsys.ingest import enqueue_head
from reviewsys.lane import LaneResult, LaneSpec
from reviewsys.models import FailKind, ReviewError, RunStatus, Trigger
from reviewsys.prompts import repair_prompt
from reviewsys.scheduler import schedule
from reviewsys.steps import worktree as wt
from tests.test_lane_corrections import (  # fake_git: autouse, fakes the mirror and worktree
    HEAD,
    _events,
    _run,
    _spec,
    _verifier,
    fake_git,  # noqa: F401
)
from tests.test_pipeline_policy import load_with

V028_ENV = {"GODEBUG": "netdns=go", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
ISOLATION = {"CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1", "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS": "1"}


def v028_argv(spec: LaneSpec, argv: list[str]) -> list[str]:
    """`lane.argv_for` as v0.28.0 shipped it, copied rather than imported: what a v11 lane's
    command line must stay. `run_claude_lane` gives a lane without a session id one of its
    own (a fresh UUID, read back from `argv`), so its transcript is kept."""
    sandbox = [lane_mod.SANDBOX_EXEC, "-p", spec.sandbox_profile] if spec.sandbox_profile else []
    if spec.resume and spec.session_id:
        session = ["--resume", spec.session_id]
    elif spec.session_id:
        session = ["--session-id", spec.session_id]
    else:
        own = argv[argv.index("--session-id") + 1] if "--session-id" in argv else ""
        uuid.UUID(own)
        session = ["--session-id", own]
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
        *(str(d) for d in spec.extra_dirs),
        "--output-format",
        "json",
        *session,
        "--print",
    ]
    if spec.max_budget_usd:
        argv += ["--max-budget-usd", f"{spec.max_budget_usd:.2f}"]
    return argv


def v028_env(spec: LaneSpec, env: dict[str, str], extra: dict[str, str] | None = None) -> dict:
    """The environment v0.28.0 starts a lane in (plus `extra`): the worker's own, the base
    lane variables, and on a lane's opening turn the directory its first request is written
    to (`lane.FirstRequest`, a fresh temporary one, read back from `env`)."""
    want = {**os.environ, **V028_ENV, **(extra or {})}
    if not spec.resume:
        bodies = env.get("OTEL_LOG_RAW_API_BODIES", "")
        assert re.fullmatch(r"file:.*/reviewsys-bodies-[^/]+", bodies), bodies
        want["OTEL_LOG_RAW_API_BODIES"] = bodies
    return want


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

            # any other process (`worktree.diff_print` streams git's output) is the real one
            def __getattr__(self, name):
                return getattr(self.real, name)

            def __enter__(self):
                self.real.__enter__()
                return self

            def __exit__(self, *exc):
                return self.real.__exit__(*exc)

        monkeypatch.setattr(lane_mod.subprocess, "Popen", Proc)

    def __call__(self, spec: LaneSpec, art: Path, worktree: Path) -> LaneResult:
        self._lane.spec, self._lane.art, self._lane.worktree = spec, art, worktree
        try:
            return lane_mod.run_claude_lane(spec, art, worktree)
        finally:
            del self._lane.spec


def _review(lanes) -> None:
    lanes.reviewer["default"] = {"summary": "ok", "findings": [], "out_of_scope_findings": []}
    lanes.verifier["default"] = _verifier()
    lanes.broken_once = {"general"}  # one correction turn, resumed in the lane's session


# ---- v11: byte-identical lanes ----


def test_v11_lanes_start_exactly_as_v028(cfg, conn, gh, lanes, monkeypatch, _claude_config_dir):
    _review(lanes)
    procs = ClaudeProcesses(monkeypatch, lanes)
    _, status = _run(cfg, conn, gh, procs)
    assert status == RunStatus.DONE
    roles = {s.role for s, _, _ in procs.started}
    assert {"prep", "general", "verifier"} <= roles
    assert any(s.resume for s, _, _ in procs.started), "a correction turn ran"
    for spec, argv, env in procs.started:
        assert argv == v028_argv(spec, argv), spec.role
        assert env == v028_env(spec, env), spec.role
        assert not spec.isolate_env


# ---- lane_env_isolation ----


def test_isolated_runner_marks_every_lane_and_only_when_on(tmp_path):
    seen: list[LaneSpec] = []

    def runner(spec, art, worktree):
        seen.append(spec)
        return LaneResult(exit_code=0, stdout="", stderr="", duration_s=0)

    assert lane_mod.isolated(runner, False) is runner
    spec = _spec(tmp_path)
    lane_mod.isolated(runner, True)(spec, tmp_path, tmp_path)
    assert seen == [dataclasses.replace(spec, isolate_env=True)]
    assert lane_mod.lane_env(seen[0]) == {**os.environ, **V028_ENV, **ISOLATION}
    assert lane_mod.lane_env(spec) == {**os.environ, **V028_ENV}


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
        assert env == v028_env(spec, env, ISOLATION), spec.role
        assert env["CLAUDE_CONFIG_DIR"] == str(_claude_config_dir)
        assert argv == v028_argv(spec, argv), "isolation is environment only"
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
    assert all(env == v028_env(spec, env) for spec, _, env in procs.started)


# ---- structured_output ----

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


OFF_SCHEMA = "error_max_structured_output_retries"


def _envelope(
    structured=None, *, result="Done.", subtype="success", session=None, errors=None
) -> LaneResult:
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
    if errors:
        env["errors"] = errors
    res = LaneResult(exit_code=0, stdout=json.dumps(env), stderr="", duration_s=1)
    lane_mod._extract_result(res)
    return res


def test_json_schema_goes_on_the_command_line_only_when_given(tmp_path):
    assert "--json-schema" not in lane_mod.argv_for(_spec(tmp_path))
    spec = _spec(tmp_path, json_schema=SCHEMA, session_id=lane_mod.new_session_id())
    argv = lane_mod.argv_for(spec)
    assert argv[argv.index("--json-schema") + 1] == json.dumps(SCHEMA, separators=(",", ":"))
    assert argv[: argv.index("--json-schema")] == v028_argv(spec, argv), "appended, nothing else"
    resumed = lane_mod.argv_for(dataclasses.replace(spec, resume=True))
    assert "--resume" in resumed and "--json-schema" in resumed, "a correction turn keeps it"


def test_structured_output_is_read_before_the_text():
    res = _envelope({"ok": True}, result='Here it is: {"ok": false}')
    assert res.structured == {"ok": True}
    assert lane_mod.lane_output(res) == {"ok": True}
    plain = _envelope(None, result='{"ok": false}')
    assert plain.structured is None and lane_mod.lane_output(plain) == {"ok": False}


def test_exhausted_structured_retries_are_a_contract_failure():
    res = _envelope(None, result="", subtype=OFF_SCHEMA)
    assert not res.ok and res.answered, "a broken answer, not a broken lane"
    with pytest.raises(lane_mod.SchemaRetriesExhausted) as exc:
        lane_mod.lane_output(res)
    assert exc.value.kind is FailKind.CONTRACT
    assert exc.value.message == "no schema-valid answer: after 4 turns"
    said = (
        "Failed to provide valid structured output after 5 attempts \u2014 last "
        "StructuredOutput error: /candidates/0/severity must be one of the allowed values"
    )
    with pytest.raises(lane_mod.SchemaRetriesExhausted) as quoted:
        lane_mod.lane_output(_envelope(None, result="", subtype=OFF_SCHEMA, errors=[said]))
    assert quoted.value.message == f"no schema-valid answer: {said}"
    assert not dataclasses.replace(res, timed_out=True).answered
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


def _off_schema(n: int, *, text: str = "", session=None) -> LaneResult:
    # "exit" in Claude Code's quoted error must not make the failure look like infrastructure
    said = f"Failed to provide valid structured output after 5 attempts: /exit_code ({n})"
    return _envelope(None, result=text, subtype=OFF_SCHEMA, session=session, errors=[said])


def test_an_off_schema_answer_is_corrected_in_the_lanes_own_session(
    cfg, conn, skills_dir, tmp_path, _claude_config_dir
):
    cfg = load_with(cfg, skills_dir, {"version": "v12", "structured_output": True})
    calls: list[LaneSpec] = []

    def runner(spec, art, worktree):
        calls.append(spec)
        if spec.resume:
            return _envelope({"ok": True}, session=spec.session_id)
        return _off_schema(1, session=spec.session_id)

    ctx = _checked_ctx(cfg, conn, tmp_path, runner)
    assert _checked(ctx) == {"ok": True}
    first, again = calls
    assert again.resume and again.session_id == first.session_id and again.json_schema == SCHEMA
    assert "/exit_code (1)" in again.prompt, "told Claude Code's schema error"
    row = conn.execute("SELECT status FROM lanes WHERE run_id=?", (ctx.run_id,)).fetchone()
    assert row["status"] == "corrected"


def test_an_answer_never_on_schema_fails_as_a_contract_failure(
    cfg, conn, skills_dir, tmp_path, _claude_config_dir
):
    """Every turn of both attempts runs out of structured retries with nothing in its text:
    no repair lane (nothing to repair), CONTRACT, and the error reported is the last turn's,
    not a "resume failed"."""
    cfg = load_with(cfg, skills_dir, {"version": "v12", "structured_output": True})
    calls: list[LaneSpec] = []

    def runner(spec, art, worktree):
        calls.append(spec)
        return _off_schema(len(calls), session=spec.session_id)

    ctx = _checked_ctx(cfg, conn, tmp_path, runner)
    with pytest.raises(worker.LaneFailed) as exc:
        _checked(ctx)
    assert exc.value.kind is FailKind.CONTRACT
    assert [c.resume for c in calls] == [False, True, True] * 2, "two corrections per attempt"
    assert not any(c.role == "repair" for c in calls)
    failed = _events(conn, ctx.run_id, "lane.correction_failed")
    assert len(failed) == 2 and all("(still invalid)" in e for e in failed)
    assert "/exit_code (3)" in failed[0] and "/exit_code (6)" in failed[1]
    assert not any("resume failed" in e for e in failed)
    reasons = [r[0] for r in conn.execute("SELECT reason FROM lanes WHERE run_id=?", (ctx.run_id,))]
    assert len(reasons) == 2 and all("no schema-valid answer" in r for r in reasons)


def test_an_off_schema_answer_with_text_goes_to_the_repair_lane(
    cfg, conn, skills_dir, tmp_path, _claude_config_dir
):
    cfg = load_with(cfg, skills_dir, {"version": "v12", "structured_output": True})
    calls: list[LaneSpec] = []

    def runner(spec, art, worktree):
        calls.append(spec)
        if spec.role == "repair":
            return _envelope(None, result='{"ok": true}')
        return _off_schema(len(calls), text='Here: {"ok": true', session=spec.session_id)

    ctx = _checked_ctx(cfg, conn, tmp_path, runner)
    assert _checked(ctx) == {"ok": True}
    repair = calls[-1]
    assert repair.role == "repair" and repair.json_schema is None
    assert repair.prompt.endswith('Here: {"ok": true')
    row = conn.execute("SELECT status FROM lanes WHERE run_id=?", (ctx.run_id,)).fetchone()
    assert row["status"] == "repaired"


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


def test_thread_coverage_needs_every_hash_exactly_once():
    """What the schema cannot say: a duplicate plus an omission has the right count and only
    listed hashes, so it validates, and only this check catches it."""
    hashes = ["0123456789ab", "ba9876543210"]

    def rows(*hs):
        return {"threads": [{"finding_hash": h, "status": "FIXED", "reply": "x"} for h in hs]}

    assert schemas.check_thread_coverage(rows(*hashes), hashes) == []
    assert schemas.check_thread_coverage(rows(hashes[0], hashes[0]), hashes) == [
        "threads: 0123456789ab answered more than once",
        "threads: no entry for ['ba9876543210']",
    ]
    assert schemas.check_thread_coverage(rows(hashes[0], "ffffffffffff"), hashes) == [
        "threads: 'ffffffffffff' is not one of the listed finding_hash values",
        "threads: no entry for ['ba9876543210']",
    ]
    assert schemas.check_thread_coverage({"threads": [None]}, []) == [
        "threads: '' is not one of the listed finding_hash values"
    ]
    assert schemas.check_thread_coverage({}, hashes) == ["threads must be a list"]


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


# ---- harness-stamped echo fields ----

ECHO_ASK = ("Include `review_phase`", "`review_phase` must be", "`review_phase` is exactly")


class WithoutEcho:
    """`lanes`, with every reviewer and verifier answer stripped of the echo fields: what a
    v12 lane, never asked for them, returns."""

    def __init__(self, lanes) -> None:
        self.lanes = lanes

    def __call__(self, spec, art, worktree):
        res = self.lanes(spec, art, worktree)
        try:
            out = json.loads(res.result_text)
        except json.JSONDecodeError:
            return res
        if spec.role in ("prep", "repair") or not isinstance(out, dict):
            return res
        out.pop("review_phase", None)
        out.pop("head_sha", None)
        return dataclasses.replace(res, result_text=json.dumps(out))


def test_stamping_fills_missing_echo_fields_without_a_note():
    raw = {"summary": "s", "findings": []}
    out = parse_reviewer_output(
        raw, expected_phase="final", head_sha=HEAD, source="x", prior_hashes=set(), stamp_echo=True
    )
    assert (out.review_phase, out.head_sha, out.normalized) == ("final", HEAD, [])
    assert out.raw["review_phase"] == "final" and out.raw["head_sha"] == HEAD
    # a phase the lane volunteered anyway is overwritten, unnoted: nobody asked for it
    out = parse_reviewer_output(
        {**raw, "review_phase": "complete"},
        expected_phase="preliminary",
        head_sha=HEAD,
        source="x",
        prior_hashes=set(),
        stamp_echo=True,
    )
    assert out.raw["review_phase"] == "preliminary" and out.normalized == []
    v = parse_verifier_output(
        {
            "findings": [],
            "coderabbit_reactions": [],
            "prerequisite_adjudications": [],
            "adjudication_complete": True,
        },
        expected_phase="final",
        expected_coderabbit_ids=[],
        stamp_echo=True,
    )
    assert v.raw["review_phase"] == "final" and v.normalized == [] and "head_sha" not in v.raw


def test_a_volunteered_head_sha_is_still_checked():
    """It may say the lane reviewed other code: still a contract error when it names another
    commit, still expanded (noted) when it abbreviates the head."""
    raw = {"summary": "s", "findings": [], "head_sha": HEAD[:10]}
    out = parse_reviewer_output(
        raw, expected_phase="final", head_sha=HEAD, source="x", prior_hashes=set(), stamp_echo=True
    )
    assert out.raw["head_sha"] == HEAD and out.normalized == [f"head_sha {HEAD[:10]!r} -> {HEAD}"]
    with pytest.raises(ReviewError) as exc:
        parse_reviewer_output(
            {**raw, "head_sha": "f" * 40},
            expected_phase="final",
            head_sha=HEAD,
            source="x",
            prior_hashes=set(),
            stamp_echo=True,
        )
    assert exc.value.kind is FailKind.CONTRACT


@pytest.mark.parametrize("placeholder", ["HEAD", "<sha>", "<head_sha>", "abc", "", "n/a"])
def test_a_volunteered_placeholder_head_sha_is_overwritten(placeholder):
    """A lane never asked for `head_sha` that writes one anyway is not claiming a commit
    unless it wrote one: only a sha-shaped value is held to the head."""
    out = parse_reviewer_output(
        {"summary": "s", "findings": [], "head_sha": placeholder},
        expected_phase="final",
        head_sha=HEAD,
        source="x",
        prior_hashes=set(),
        stamp_echo=True,
    )
    assert out.raw["head_sha"] == HEAD and out.normalized == []


def test_v11_still_rejects_a_placeholder_head_sha():
    with pytest.raises(ReviewError):
        parse_reviewer_output(
            {"summary": "s", "findings": [], "head_sha": "HEAD"},
            expected_phase="final",
            head_sha=HEAD,
            source="x",
            prior_hashes=set(),
        )


def test_v11_still_notes_missing_echo_fields():
    out = parse_reviewer_output(
        {"summary": "s", "findings": []},
        expected_phase="final",
        head_sha=HEAD,
        source="x",
        prior_hashes=set(),
    )
    assert out.normalized == [
        "review_phase missing -> 'final'",
        "head_sha missing -> assigned head",
    ]


def test_v12_lanes_are_never_asked_for_echo_fields_and_need_not_return_them(
    cfg, conn, gh, lanes, skills_dir
):
    cfg = load_with(cfg, skills_dir, {"version": "v12"})
    _review(lanes)  # the general lane's first answer is broken: one correction turn
    rid, status = _run(cfg, conn, gh, WithoutEcho(lanes))
    assert status == RunStatus.DONE
    asked = [s for s in lanes.calls if s.role not in ("prep",)]
    assert {"general", "verifier"} <= {s.role for s in asked}
    assert any(s.resume for s in asked), "a correction turn ran"
    for spec in asked:
        assert not any(a in spec.prompt for a in ECHO_ASK), spec.role
        assert "`head_sha` is exactly" not in spec.prompt
        assert "review_phase, head_sha" not in spec.prompt
    assert _events(conn, rid, "lane.output_normalized") == []
    assert _events(conn, rid, "lane.corrected"), "the broken answer was still corrected"


def test_v11_lanes_are_asked_for_echo_fields_and_noted_without_them(cfg, conn, gh, lanes):
    _review(lanes)
    rid, status = _run(cfg, conn, gh, WithoutEcho(lanes))
    assert status == RunStatus.DONE
    general = [s for s in lanes.calls if s.role == "general"]
    assert "Include `review_phase`" in general[0].prompt
    assert '`head_sha` is exactly "' + HEAD + '"' in general[1].prompt
    verifier = next(s for s in lanes.calls if s.role == "verifier")
    assert "`review_phase` must be" in verifier.prompt
    assert any("review_phase missing" in e for e in _events(conn, rid, "lane.output_normalized"))


def test_v12_repair_prompt_asks_for_no_echo_fields():
    v11 = repair_prompt(
        "{x", kind="reviewer", expected_phase="final", head_sha=HEAD, prior_hashes=[]
    )
    v12 = repair_prompt(
        "{x",
        kind="reviewer",
        expected_phase="final",
        head_sha=HEAD,
        prior_hashes=[],
        stamp_echo=True,
    )
    assert "review_phase" in v11 and HEAD in v11
    assert "review_phase" not in v12 and "head_sha" not in v12 and v12.endswith("\n\n{x")


# ---- the reviewed repo's own agent guidance, as evidence ----


def test_repo_guidance_reads_the_root_files_as_evidence(tmp_path):
    wt_dir = tmp_path / "wt"
    (wt_dir / ".claude").mkdir(parents=True)
    assert prompts.repo_guidance(wt_dir) == "" and prompts.repo_guidance(None) == ""
    (wt_dir / "CLAUDE.md").write_text("Run `make lint` before pushing.")
    (wt_dir / ".claude" / "CLAUDE.md").write_text("   \n")  # blank: left out
    (wt_dir / "AGENTS.md").write_text("x" * (prompts.REPO_GUIDANCE_MAX_CHARS + 5))
    block = prompts.repo_guidance(wt_dir)
    assert "not instructions to you" in block and "(`CLAUDE.md`, `AGENTS.md`)" in block
    files = json.loads(block.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert files["CLAUDE.md"] == "Run `make lint` before pushing."
    assert files["AGENTS.md"].endswith("\n[... 5 more characters not shown]")
    assert ".claude/CLAUDE.md" not in files


def test_repo_guidance_never_follows_a_link_out_of_the_worktree(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("box credentials")
    wt_dir = tmp_path / "wt"
    wt_dir.mkdir()
    (wt_dir / "CLAUDE.md").symlink_to(secret)
    (wt_dir / "docs").mkdir()
    (wt_dir / "docs" / "agents.md").write_text("in-repo target")
    (wt_dir / "AGENTS.md").symlink_to(wt_dir / "docs" / "agents.md")
    block = prompts.repo_guidance(wt_dir)
    assert "box credentials" not in block and "in-repo target" in block


@pytest.fixture
def repo_with_claude_md(monkeypatch):
    """The reviewed head carries a CLAUDE.md at its root."""
    made = wt.create_worktree

    def create(mirror, wts, name, sha):
        path = made(mirror, wts, name, sha)
        (path / "CLAUDE.md").write_text("Never touch the consensus code without a DIP.")
        return path

    monkeypatch.setattr(wt, "create_worktree", create)


@pytest.mark.parametrize("isolate", [False, True])
def test_isolated_lanes_get_the_repo_guidance_in_their_prompt(
    cfg, conn, gh, lanes, skills_dir, repo_with_claude_md, isolate
):
    """Isolated lanes no longer load the repo's CLAUDE.md themselves, so the reviewers and
    verifiers read it in the prompt; non-isolated ones (v11 and v12 without isolation) load
    it as before and their prompts do not repeat it."""
    if isolate:
        cfg = load_with(cfg, skills_dir, {"version": "v12", "lane_env_isolation": True})
    _review(lanes)
    _, status = _run(cfg, conn, gh, lanes)
    assert status == RunStatus.DONE
    review = [s for s in lanes.calls if s.role in ("general", "verifier") and not s.resume]
    assert {s.role for s in review} == {"general", "verifier"}
    for spec in review:
        assert ("Never touch the consensus code" in spec.prompt) is isolate, spec.role
