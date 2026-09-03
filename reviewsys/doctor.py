"""Environment checks: gh auth, claude launcher, skills repo, proxy model probes."""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from .config import Config

PROXY_URL = os.environ.get("REVIEWSYS_PROXY_URL", "http://127.0.0.1:8317")


def _ok(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    return ok


def _proxy_key() -> str | None:
    p = Path.home() / ".cli-proxy-api" / "client-keys.json"
    try:
        d = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return d.get("claude-code") or d.get("openclaw") or (next(iter(d.values())) if d else None)


def probe_single_stop(model: str, key: str) -> tuple[bool, str]:
    """Reproducer for the 2026-08 CLIProxyAPI bug: a one-element stop_sequences array must be accepted."""
    body = json.dumps(
        {
            "model": model,
            "max_tokens": 8,
            "stop_sequences": ["</block>"],
            "messages": [{"role": "user", "content": "say ok"}],
        }
    ).encode()
    req = urllib.request.Request(
        f"{PROXY_URL}/v1/messages",
        data=body,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return resp.status == 200, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.read()[:120]!r}"
    except (urllib.error.URLError, TimeoutError) as exc:
        return False, str(exc)


def run(cfg: Config, *, probe_models: bool = True) -> bool:
    ok = True
    env = {**os.environ, "GODEBUG": "netdns=go"}
    r = subprocess.run(
        [cfg.gh_bin, "auth", "status"], capture_output=True, text=True, env=env, check=False
    )
    ok &= _ok(
        "gh auth",
        r.returncode == 0,
        (r.stderr or r.stdout).strip().splitlines()[0] if (r.stderr or r.stdout).strip() else "",
    )
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
            for m in sorted(models):
                good, detail = probe_single_stop(m, key)
                ok &= _ok(f"proxy model {m} (single stop_sequence)", good, detail)
    return bool(ok)
