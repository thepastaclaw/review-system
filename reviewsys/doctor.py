"""Environment checks: gh auth, claude launcher, skills repo, proxy model probes."""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from . import quota
from .config import Config

PROXY_URL = os.environ.get("REVIEWSYS_PROXY_URL", "http://127.0.0.1:8317")
LANE_PROBE_TIMEOUT_S = 180


def _ok(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    return ok


def _first_line(r: subprocess.CompletedProcess[str]) -> str:
    """The first line a command said (stderr preferred), or "" when it said nothing."""
    said = (r.stderr or r.stdout).strip()
    return said.splitlines()[0] if said else ""


def _proxy_key() -> str | None:
    p = Path.home() / ".cli-proxy-api" / "client-keys.json"
    try:
        d = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return d.get("claude-code") or d.get("openclaw") or (next(iter(d.values())) if d else None)


def _messages(model: str, key: str, *, stop: bool) -> tuple[int, bytes]:
    payload: dict[str, object] = {
        "model": model,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "say ok"}],
    }
    if stop:
        payload["stop_sequences"] = ["</block>"]
    req = urllib.request.Request(
        f"{PROXY_URL}/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return int(resp.status), b""
    except urllib.error.HTTPError as exc:
        return int(exc.code), bytes(exc.read()[:4096])
    except (urllib.error.URLError, TimeoutError) as exc:
        return 0, str(exc).encode()


def probe_single_stop(model: str, key: str) -> tuple[bool, str]:
    """Reproducer for the 2026-08 CLIProxyAPI bug: a one-element stop_sequences array must be
    accepted. Some upstreams (Gemini via Antigravity, Meta) reject `stop` outright; Claude Code
    lanes never send one, so such a model passes if it answers a stop-free request."""
    status, err = _messages(model, key, stop=True)
    if status == 200:
        return True, "HTTP 200"
    lowered = err.lower()
    if b"stop" in lowered and b"not supported" in lowered:
        status, err = _messages(model, key, stop=False)
        if status == 200:
            return True, "HTTP 200 without stop (upstream rejects stop; lanes never send one)"
    return False, f"HTTP {status}: {err[:120]!r}"


def probe_lane(model: str, claude_bin: str) -> tuple[bool, str]:
    """One real `claude --print` through the launcher: catches a model the launcher's
    allowlist rejects, which the HTTP probe cannot see."""
    argv = [
        claude_bin,
        "--bare",
        "--permission-mode",
        "plan",
        "--model",
        model,
        "--effort",
        "low",
        "--output-format",
        "json",
        "--no-session-persistence",
        "--print",
        "Reply with exactly the word ok and nothing else.",
    ]
    env = {**os.environ, "GODEBUG": "netdns=go", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
    try:
        r = subprocess.run(
            argv, capture_output=True, text=True, env=env, timeout=LANE_PROBE_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)[:160]
    if r.returncode != 0:
        return False, _first_line(r)[:160] or f"exit {r.returncode}"
    try:
        out = json.loads(r.stdout)
    except json.JSONDecodeError:
        return False, "non-JSON output"
    if out.get("is_error"):
        return False, str(out.get("result"))[:160]
    return True, f"{out.get('duration_ms', '?')} ms"


def report_quota(cfg: Config, mgmt: quota.Management | None = None) -> bool:
    """One line per quota-gated Phase-1 rung: the account with the most left and its windows."""
    ok = True
    mgmt = mgmt or quota.Management()
    reserve = cfg.policy.quota_reserve
    for lm in cfg.policy.phase1_candidates:
        label = f"quota {lm.model}"
        if lm.quota is None:
            _ok(label, True, "not gated (pay per token)")
            continue
        try:
            st = quota.read(mgmt, lm.quota)
        except quota.QuotaError as exc:
            ok &= _ok(label, False, str(exc))
            continue
        detail = f"{st.account}: {st.describe()}"
        if st.remaining < reserve:
            detail += f" (below the {round(reserve * 100)}% reserve; rung skipped)"
        _ok(label, True, detail)
    return ok


def run(cfg: Config, *, probe_models: bool = True) -> bool:
    ok = True
    env = {**os.environ, "GODEBUG": "netdns=go"}
    r = subprocess.run(
        [cfg.gh_bin, "auth", "status"], capture_output=True, text=True, env=env, check=False
    )
    ok &= _ok("gh auth", r.returncode == 0, _first_line(r))
    r = subprocess.run(
        [cfg.gh_bin, "api", "rate_limit", "--jq", ".resources.core.remaining"],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )
    ok &= _ok(
        "gh api reachable",
        r.returncode == 0,
        f"remaining={r.stdout.strip()}" if r.returncode == 0 else r.stderr.strip()[:120],
    )
    ok &= _ok("claude launcher", Path(cfg.claude_bin).exists(), cfg.claude_bin)
    ok &= _ok("skills config", (cfg.skills_dir / "config.json").exists(), str(cfg.skills_dir))
    for rel in ("prompts/review-agent.md", "prompts/verifier-agent.md"):
        ok &= _ok(f"template {rel}", (cfg.skills_dir / rel).exists())
    for rc in cfg.repos:
        if rc.enabled:
            ok &= _ok(
                f"skill {rc.repo}",
                (cfg.skills_dir / rc.skill_path / "review.md").exists(),
                rc.skill_path,
            )
    for d in (cfg.work_dir, cfg.db_path.parent):
        try:
            d.mkdir(parents=True, exist_ok=True)
            ok &= _ok(f"writable {d}", os.access(d, os.W_OK))
        except OSError as exc:
            ok &= _ok(f"writable {d}", False, str(exc))
    if probe_models:
        key = _proxy_key()
        if not key:
            ok &= _ok("proxy client key", False, "~/.cli-proxy-api/client-keys.json unreadable")
        else:
            models = {
                cfg.policy.phase1_reviewer.model,
                cfg.policy.phase1_verifier.model,
                cfg.policy.phase2_reviewer.model,
                cfg.policy.phase2_verifier.model,
                cfg.policy.repair_model,
                cfg.policy.selector_model,
            }
            if cfg.policy.triage:
                models.add(cfg.policy.triage.model)
            models.update(c.model for c in cfg.policy.phase1_candidates)
            for m in sorted(models):
                good, detail = probe_single_stop(m, key)
                ok &= _ok(f"proxy model {m} (single stop_sequence)", good, detail)
        if cfg.policy.has_phase1_ladder:
            ok &= report_quota(cfg)
            for c in cfg.policy.phase1_candidates:
                good, detail = probe_lane(c.model, cfg.claude_bin)
                ok &= _ok(f"launcher lane {c.model}", good, detail)
    return bool(ok)
