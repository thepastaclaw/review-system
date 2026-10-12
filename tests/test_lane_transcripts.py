"""Every lane keeps its Claude Code session (`transcript.jsonl.gz`) and its first request
(`first-request.json.gz`) in its artifact dir, so a run's lanes can be read back without the
proxy logs. File shapes are Claude Code 2.1.280's on the box (2026-10-12)."""

from __future__ import annotations

import gzip
import json
import os
import time

from reviewsys import lane as lane_mod
from reviewsys.lane import LaneResult, LaneSpec


def _spec(tmp_path, **kw):
    base = {
        "role": "general",
        "agent": "a",
        "model": "m",
        "effort": "high",
        "prompt": "p",
        "cwd": tmp_path,
        "add_dir": tmp_path,
        "timeout_seconds": 60,
        "claude_bin": "claude",
    }
    return LaneSpec(**{**base, **kw})


def _session(config_dir, sid, lines):
    project = config_dir / "projects" / "-wt"
    project.mkdir(parents=True, exist_ok=True)
    (project / f"{sid}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    (project / sid / "tool-results").mkdir(parents=True)
    (project / sid / "tool-results" / "b1.txt").write_text("spilled")
    return project


def _transcript(art):
    with gzip.open(art / lane_mod.TRANSCRIPT, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_transcript_is_kept_without_reasoning_signatures(tmp_path, _claude_config_dir):
    sid = lane_mod.new_session_id()
    thinking = {"type": "thinking", "thinking": "", "signature": "x" * 5000}
    tool = {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "a.rs"}}
    lines = [
        {"type": "user", "message": {"role": "user", "content": "review this"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [thinking, tool]}},
        {"type": "user", "toolUseResult": {"signature": "not a thinking block"}},
    ]
    project = _session(_claude_config_dir, sid, lines)
    art = tmp_path / "art"
    lane_mod.save_transcript(sid, art)
    kept = _transcript(art)
    assert kept[0] == lines[0] and kept[2] == lines[2]
    assert kept[1]["message"]["content"] == [
        {**thinking, "signature": "<5000 bytes removed>"},
        tool,
    ]
    assert not (art / "tool-results").exists(), "spilled tool output stays out"
    assert (project / f"{sid}.jsonl").exists(), "saving does not delete the session"


def test_no_session_no_transcript(tmp_path):
    lane_mod.save_transcript(lane_mod.new_session_id(), tmp_path / "art")
    assert not (tmp_path / "art" / lane_mod.TRANSCRIPT).exists()


def _bodies(tmp_path, index, files):
    bodies = tmp_path / "art" / lane_mod.API_BODIES
    bodies.mkdir(parents=True)
    for name, body in files.items():
        (bodies / name).write_text(json.dumps(body))
    if index is not None:
        (bodies / "index.jsonl").write_text("".join(json.dumps(x) + "\n" for x in index))
    return bodies


def _first(tmp_path):
    with gzip.open(tmp_path / "art" / lane_mod.FIRST_REQUEST, "rt") as f:
        return json.load(f)


def test_first_main_thread_request_is_kept_and_the_rest_dropped(tmp_path):
    bodies = _bodies(
        tmp_path,
        [
            {"query_source": "side", "request_file": "a.request.json"},
            {"query_source": "sdk", "request_file": "b.request.json"},
            {"query_source": "sdk", "request_file": "c.request.json"},
            {"query_source": "sdk", "request_file": "../escape.request.json"},
        ],
        {
            "a.request.json": {"system": "side"},
            "b.request.json": {"system": "main", "tools": [{"name": "Read"}]},
            "c.request.json": {"system": "later"},
            "b.response.json": {"content": []},
        },
    )
    lane_mod.keep_first_request(bodies, tmp_path / "art")
    assert _first(tmp_path) == {"system": "main", "tools": [{"name": "Read"}]}
    assert not bodies.exists()


def test_without_an_index_the_oldest_request_is_kept(tmp_path):
    bodies = _bodies(tmp_path, None, {"new.request.json": {"n": 2}, "old.request.json": {"n": 1}})
    past = time.time() - 60
    os.utime(bodies / "old.request.json", (past, past))
    lane_mod.keep_first_request(bodies, tmp_path / "art")
    assert _first(tmp_path) == {"n": 1}
    assert not bodies.exists()


def test_a_correctable_lane_keeps_its_whole_session_once(tmp_path, _claude_config_dir):
    """The correction turns resume the lane's session: one transcript for all of them, in the
    lane's own dir, written before the session is deleted."""
    sid = lane_mod.new_session_id()
    project = _session(_claude_config_dir, sid, [{"type": "user", "n": 0}])
    calls = []

    def runner(spec, art, worktree):
        calls.append(spec)
        with (project / f"{sid}.jsonl").open("a") as f:
            f.write(json.dumps({"type": "user", "n": len(calls)}) + "\n")
        text = "bad" if len(calls) < 2 else "{}"
        return LaneResult(exit_code=0, stdout="", stderr="", duration_s=1, result_text=text)

    spec = _spec(
        tmp_path,
        session_id=sid,
        check=lambda t: None if t.result_text == "{}" else "fix it",
        corrections=2,
    )
    lane_mod.run_with_corrections(runner, spec, tmp_path / "art", tmp_path)
    assert [x["n"] for x in _transcript(tmp_path / "art")] == [0, 1, 2]
    assert not (tmp_path / "art" / "correction-1" / lane_mod.TRANSCRIPT).exists()
    assert not lane_mod.session_files(sid)


FAKE_CLAUDE = r"""#!/bin/sh
cat >/dev/null
for a in "$@"; do
  [ "$prev" = --session-id ] && sid=$a
  [ "$prev" = --resume ] && sid=$a
  prev=$a
done
d="$CLAUDE_CONFIG_DIR/projects/-wt"
mkdir -p "$d"
echo '{"type":"user","message":{"role":"user","content":"p"}}' >> "$d/$sid.jsonl"
if [ -n "$OTEL_LOG_RAW_API_BODIES" ]; then
  b="${OTEL_LOG_RAW_API_BODIES#file:}"
  mkdir -p "$b"
  echo '{"system":"s","tools":[]}' > "$b/r1.request.json"
  echo '{"content":[]}' > "$b/p1.response.json"
  echo '{"query_source":"sdk","request_file":"r1.request.json"}' > "$b/index.jsonl"
fi
printf '{"type":"result","result":"{}","session_id":"%s"}' "$sid"
"""


def test_the_real_runner_leaves_transcript_and_first_request(tmp_path, _claude_config_dir):
    script = tmp_path / "claude"
    script.write_text(FAKE_CLAUDE)
    script.chmod(0o755)
    art = tmp_path / "art"
    res = lane_mod.run_claude_lane(_spec(tmp_path, claude_bin=str(script)), art, tmp_path)
    assert res.ok and res.session_id
    assert _transcript(art) == [{"type": "user", "message": {"role": "user", "content": "p"}}]
    assert _first(tmp_path) == {"system": "s", "tools": []}
    assert not (art / lane_mod.API_BODIES).exists()
    assert not lane_mod.session_files(res.session_id), "the lane's session is deleted"
    meta = json.loads((art / "lane-meta.json").read_text())
    assert meta["session_id"] == res.session_id
    assert meta["argv"][meta["argv"].index("--session-id") + 1] == res.session_id


def test_a_resumed_turn_records_no_request_bodies(tmp_path, _claude_config_dir):
    """A correction turn's first request is the whole session so far: not recorded, and the
    session (the caller's) is left for `run_with_corrections` to save."""
    script = tmp_path / "claude"
    script.write_text(FAKE_CLAUDE)
    script.chmod(0o755)
    sid = lane_mod.new_session_id()
    art = tmp_path / "art" / "correction-1"
    spec = _spec(tmp_path, claude_bin=str(script), session_id=sid, resume=True)
    assert lane_mod.run_claude_lane(spec, art, tmp_path).ok
    assert not (art / lane_mod.FIRST_REQUEST).exists()
    assert not (art / lane_mod.TRANSCRIPT).exists()
    assert lane_mod.session_files(sid), "the caller's session is not deleted by one turn"
