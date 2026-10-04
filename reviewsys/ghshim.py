"""A pinned, read-only `gh` for `reviewsys replay`, shared by the harness and its lanes.

A replay re-reviews the exact head a past run reviewed. Everything the review reads about the
PR has to look as it did when that run started (the cutoff), and nothing may write to GitHub.
The v10 eval pinned only the harness's own client; lanes calling `gh pr diff N` saw the PR's
*current* head, `gh pr checks` its current CI, and every comment posted since. Now the
harness's client routes through `pinned_runner`, and every lane finds a `gh` shim first on its
PATH (`install_shim`) that runs `main` here. Both serve the same view:

- `gh pr diff` is `git diff <merge-base> <sha>` from the repo's mirror.
- The PR itself (REST `pulls/N` and `issues/N`, `gh pr view`, GraphQL `pullRequest`) is open,
  unmerged, at the replayed sha and at the base the source run saw, with the title and body
  the source run saw (when its evidence is on disk). Its files and commits come from git.
- CI (`gh pr checks`, the head's check-runs and status) comes from the source run's
  `evidence.json` snapshot; without one it is reported unavailable, never fetched live.
- Comments, reviews, review threads and thread comments created at or after the cutoff are
  dropped, the bot's own included.
- Every write is refused, and so is every command not known to be a read: an unknown command
  never reaches GitHub.

Not rewound (as of now): whether a thread is resolved, reactions, labels, and the state of
other PRs and issues (those created after the cutoff are refused, though). Every call and what
was done with it is appended to the pin's log.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .gh import GRAPHQL_MUTATION, GhRunner

Result = subprocess.CompletedProcess[str]
RealGh = Callable[[list[str]], Result]  # runs the real gh with these args (no argv[0])

SHIM_CHECK = "--reviewsys-shim-check"
# creation timestamps, REST and GraphQL spellings; `updated_at` is deliberately not one
_CREATED = ("created_at", "createdAt", "submitted_at", "submittedAt")
# subcommands that change something (on GitHub, or the lane's checkout): refused as writes
_WRITE_VERBS = frozenset(
    {
        "add",
        "archive",
        "cancel",
        "checkout",
        "clone",
        "close",
        "comment",
        "create",
        "delete",
        "deploy-key",
        "develop",
        "disable",
        "download",
        "edit",
        "enable",
        "fork",
        "lock",
        "login",
        "logout",
        "merge",
        "pin",
        "ready",
        "refresh",
        "remove",
        "rename",
        "reopen",
        "rerun",
        "review",
        "revert",
        "run",
        "set",
        "set-default",
        "setup-git",
        "sync",
        "transfer",
        "unarchive",
        "unlock",
        "unpin",
        "update-branch",
        "upload",
    }
)
_PR_URL = re.compile(r"https?://github\.com/([^/]+/[^/]+)/(?:pull|issues)/(\d+)")
_COMPUTED_VIEW = frozenset(
    {
        "headRefOid",
        "baseRefOid",
        "state",
        "closed",
        "closedAt",
        "mergedAt",
        "mergedBy",
        "mergeCommit",
        "commits",
        "files",
        "additions",
        "deletions",
        "changedFiles",
        "statusCheckRollup",
        "reviewDecision",
        "mergeable",
        "mergeStateStatus",
        "potentialMergeCommit",
        "autoMergeRequest",
        "updatedAt",
    }
)
_TEXT_FIELDS = (
    "number",
    "title",
    "state",
    "author",
    "url",
    "baseRefName",
    "headRefName",
    "isDraft",
    "labels",
    "body",
)


class Refused(Exception):
    """The view will not serve this call; the message tells the lane why."""


@dataclass(frozen=True, slots=True)
class Pin:
    """What a replay pins: the PR, the head, the moment it is shown as of, and where its git
    history and the real gh are."""

    repo: str
    number: int
    sha: str
    cutoff: str  # ISO-8601 Z; compares as a string with GitHub's timestamps
    merge_base: str  # where `gh pr diff` starts
    base_ref: str
    base_sha: str  # the base branch tip the source run saw
    git_dir: str  # the repo's mirror (any git dir holding both commits)
    real_gh: str
    ci: dict[str, Any] | None = None  # the source run's `ci` evidence snapshot
    pr: dict[str, Any] = field(default_factory=dict)  # its `PrMeta.as_dict()` evidence
    jq: str = ""  # jq binary for `--jq` on filtered responses ("" = none)
    log: str = ""  # JSONL of every call ("" = not logged)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: Path) -> Pin:
        return cls(**json.loads(path.read_text()))

    @property
    def where(self) -> str:
        return f"{self.repo}#{self.number} at {self.sha[:12]}"


# ---- entry points ----


def serve(
    pin: Pin, args: list[str], *, real: RealGh, stdin: str | None = None, source: str = "lane"
) -> Result:
    """Answer one `gh` invocation (`args` without argv[0]) from the pinned view."""
    try:
        res = _dispatch(pin, args, real, stdin)
        outcome = "served" if res.returncode == 0 else "error"
    except Refused as exc:
        res = _fail(f"gh (reviewsys replay): {exc}")
        outcome = "refused"
    _log(pin, source, args, outcome, res.returncode)
    return res


def pinned_runner(pin: Pin, inner: GhRunner | None = None) -> GhRunner:
    """A `Gh` runner serving the harness from the pinned view; `inner` runs the real gh."""
    from .gh import _default_runner

    run = inner or _default_runner

    def runner(argv: Sequence[str], stdin: str | None, timeout: int) -> Result:
        def real(args: list[str]) -> Result:
            return run([pin.real_gh, *args], None, timeout)

        return serve(pin, list(argv)[1:], real=real, stdin=stdin, source="harness")

    return runner


def main(argv: list[str] | None = None) -> int:
    """The lane-side shim: `python -m reviewsys.ghshim PIN.json gh-args...`."""
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("usage: python -m reviewsys.ghshim PIN.json [gh args...]", file=sys.stderr)
        return 2
    if args[1:] == [SHIM_CHECK]:
        print("ok")
        return 0
    pin = Pin.load(Path(args[0]))

    def real(gh_args: list[str]) -> Result:
        env = {**os.environ, "GODEBUG": "netdns=go", "GH_PROMPT_DISABLED": "1", "NO_COLOR": "1"}
        return subprocess.run(
            [pin.real_gh, *gh_args], capture_output=True, text=True, env=env, check=False
        )

    stdin = None if sys.stdin is None or sys.stdin.isatty() else _lazy_stdin(args[1:])
    res = serve(pin, args[1:], real=real, stdin=stdin)
    sys.stdout.write(res.stdout or "")
    sys.stderr.write(res.stderr or "")
    return res.returncode


def install_shim(pin_path: Path, bin_dir: Path) -> Path:
    """Write the `gh` shim into `bin_dir` (put first on a lane's PATH). It runs this module
    with the interpreter and package the replay itself runs on."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    pkg_root = Path(__file__).resolve().parent.parent
    path = bin_dir / "gh"
    path.write_text(
        "#!/bin/sh\n"
        "# reviewsys replay: a pinned, read-only gh (reviewsys/ghshim.py)\n"
        f"export PYTHONPATH={shlex.quote(str(pkg_root))}\n"
        f'exec {shlex.quote(sys.executable)} -m reviewsys.ghshim {shlex.quote(str(pin_path))} "$@"\n'
    )
    path.chmod(0o755)
    return path


def stage_real_gh(real: str, dest_dir: Path) -> Path:
    """A private copy of the real gh binary for the shim to run. The replay denies lanes
    executing the installed gh (so a lane that calls it by absolute path is stopped by the
    sandbox, not served the live PR), and that deny is by path: the copy is not covered. An APFS
    clone where possible; refreshed when the installed binary changes."""
    src = Path(os.path.realpath(shutil.which(real) or real))
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "gh"
    st = src.stat()
    if dest.exists():
        d = dest.stat()
        if (d.st_size, int(d.st_mtime)) == (st.st_size, int(st.st_mtime)):
            return dest
        dest.unlink()
    tmp = dest_dir / f".gh.{os.getpid()}"
    if subprocess.run(["cp", "-c", "-p", str(src), str(tmp)], capture_output=True).returncode:
        shutil.copy2(src, tmp)
    tmp.replace(dest)
    return dest


def _lazy_stdin(args: list[str]) -> str | None:
    """Only a GraphQL query read from `@-` is ever taken from stdin."""
    return sys.stdin.read() if any(a.endswith("=@-") for a in args) else None


# ---- dispatch ----


def _dispatch(pin: Pin, args: list[str], real: RealGh, stdin: str | None) -> Result:
    if not args or args[0] in ("--version", "version"):
        return real(["--version"])
    if args[0] == "help" or "--help" in args or "-h" in args:
        return real(args)
    cmd, sub = args[0], (args[1] if len(args) > 1 else "")
    if cmd == "api":
        return _api(pin, args[1:], real, stdin)
    if cmd == "pr" and sub == "diff":
        return _pr_diff(pin, args[2:], real, args)
    if cmd in ("pr", "issue") and sub == "view":
        return _view(pin, args[2:], real, args)
    if cmd == "pr" and sub == "checks":
        return _pr_checks(pin, args[2:])
    if cmd == "run" and sub == "view":
        return _run_view(args, real)
    if (cmd, sub) in (("repo", "view"), ("auth", "status")):
        return real(args)
    shown = " ".join(args[:2])
    if sub in _WRITE_VERBS or cmd in ("browse", "extension", "codespace"):
        raise Refused(
            f"refused `gh {shown}`: this is a read-only replay of {pin.where}; "
            "nothing may be written to GitHub or the checkout"
        )
    raise Refused(
        f"`gh {shown}` is not available in this replay of {pin.where}: only reads pinned to "
        "the reviewed commit are served (gh pr diff/view/checks, gh issue view, "
        "gh run view <id>, gh api GET)"
    )


# ---- argument parsing ----

_VALUE_FLAGS = frozenset(
    {
        "-R",
        "--repo",
        "--json",
        "-q",
        "--jq",
        "-t",
        "--template",
        "--color",
        "-i",
        "--interval",
        "-j",
        "--job",
        "-a",
        "--attempt",
    }
)


def _parse(args: list[str]) -> tuple[list[str], dict[str, str]]:
    """(positionals, flags) of a pr/issue/run subcommand; bool flags map to ""."""
    pos: list[str] = []
    flags: dict[str, str] = {}
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--") and "=" in a:
            k, _, v = a.partition("=")
            flags[k] = v
        elif a in _VALUE_FLAGS and i + 1 < len(args):
            flags[a] = args[i + 1]
            i += 1
        elif a.startswith("-") and a != "-":
            flags[a] = ""
        else:
            pos.append(a)
        i += 1
    return pos, flags


def _flag(flags: dict[str, str], *names: str) -> str | None:
    return next((flags[n] for n in names if n in flags), None)


def _target(pin: Pin, pos: list[str], flags: dict[str, str]) -> tuple[str, int]:
    """(repo, number) a pr/issue command is about: the pinned PR by default."""
    repo = _flag(flags, "-R", "--repo") or pin.repo
    if not pos:
        return repo, pin.number
    sel = pos[0]
    if m := _PR_URL.match(sel):
        return m.group(1), int(m.group(2))
    if sel.lstrip("#").isdigit():
        return repo, int(sel.lstrip("#"))
    raise Refused(
        f"`{sel}`: name the PR by number in this replay (the reviewed PR is #{pin.number})"
    )


def _is_pinned(pin: Pin, repo: str, number: int) -> bool:
    return repo.lower() == pin.repo.lower() and number == pin.number


def _same_repo(pin: Pin, repo: str) -> bool:
    return repo.lower() == pin.repo.lower()


def _existed(pin: Pin, repo: str, number: int, real: RealGh) -> None:
    """Refuse an issue or PR of the replayed repo that was opened after the cutoff."""
    res = real(["api", f"repos/{repo}/issues/{number}", "--jq", ".created_at"])
    if res.returncode == 0 and res.stdout.strip() >= pin.cutoff:
        raise Refused(f"{repo}#{number} did not exist yet at this replay's cutoff ({pin.cutoff})")


# ---- pr diff / view / checks, issue view, run view ----


def _pr_diff(pin: Pin, rest: list[str], real: RealGh, args: list[str]) -> Result:
    pos, flags = _parse(rest)
    if "--web" in flags or "-w" in flags:
        raise Refused("`--web` is not available in a replay")
    repo, number = _target(pin, pos, flags)
    if not _is_pinned(pin, repo, number):
        if _same_repo(pin, repo):
            _existed(pin, repo, number, real)
        return real(args)
    if "--name-only" in flags:
        return _ok(_git(pin, "diff", "--name-only", "-M", pin.merge_base, pin.sha))
    return _ok(_diff(pin))


def _view(pin: Pin, rest: list[str], real: RealGh, args: list[str]) -> Result:
    pos, flags = _parse(rest)
    if "--web" in flags or "-w" in flags:
        raise Refused("`--web` is not available in a replay")
    repo, number = _target(pin, pos, flags)
    fields = _flag(flags, "--json")
    jq = _flag(flags, "-q", "--jq")
    if _flag(flags, "-t", "--template") is not None:
        raise Refused("`--template` is not available in a replay; use --json with --jq")
    if not _is_pinned(pin, repo, number):
        if _same_repo(pin, repo):
            _existed(pin, repo, number, real)
        if fields is None:
            if "--comments" in flags or "-c" in flags:
                raise Refused(
                    "`--comments` is not available here in a replay; use --json comments "
                    "(comments after the replay's cutoff are hidden)"
                )
            return real(args)
        live = real([args[0], args[1], str(number), "-R", repo, "--json", fields])
        if live.returncode:
            return live
        data = _before_cutoff(
            pin, json.loads(live.stdout or "null"), pin_objects=_same_repo(pin, repo)
        )
        return _emit(pin, data, jq)
    want = (
        [f.strip() for f in fields.split(",") if f.strip()]
        if fields is not None
        else [*_TEXT_FIELDS, *(["comments"] if "--comments" in flags or "-c" in flags else [])]
    )
    data = _pinned_view(pin, want, real)
    if fields is not None:
        return _emit(pin, data, jq)
    return _ok(_render_view(data))


def _pinned_view(pin: Pin, want: list[str], real: RealGh) -> dict[str, Any]:
    known = {k for k in ("title", "body") if k in pin.pr}
    live_fields = [f for f in want if f not in _COMPUTED_VIEW and f not in known]
    data: dict[str, Any] = {}
    if live_fields:
        res = real(["pr", "view", str(pin.number), "-R", pin.repo, "--json", ",".join(live_fields)])
        if res.returncode:
            raise Refused(f"could not read {pin.repo}#{pin.number}: {res.stderr.strip()[:300]}")
        data = _before_cutoff(pin, json.loads(res.stdout or "{}"))
    files = _files(pin) if {"files", "additions", "deletions", "changedFiles"} & set(want) else []
    computed: dict[str, Any] = {
        "headRefOid": pin.sha,
        "baseRefOid": pin.base_sha,
        "state": "OPEN",
        "closed": False,
        "closedAt": None,
        "mergedAt": None,
        "mergedBy": None,
        "mergeCommit": None,
        "potentialMergeCommit": None,
        "autoMergeRequest": None,
        "mergeable": "UNKNOWN",
        "mergeStateStatus": "UNKNOWN",
        "reviewDecision": "",  # cannot be rewound
        "updatedAt": pin.cutoff,
        "statusCheckRollup": _rollup(pin),
        "additions": sum(f["additions"] for f in files),
        "deletions": sum(f["deletions"] for f in files),
        "changedFiles": len(files),
        "files": [
            {
                "path": f["filename"],
                "additions": f["additions"],
                "deletions": f["deletions"],
                "changeType": _CHANGE_TYPES.get(f["status"], "MODIFIED"),
            }
            for f in files
        ],
        **{k: pin.pr[k] for k in known},
    }
    if "commits" in want:
        computed["commits"] = [
            {
                "oid": c["sha"],
                "messageHeadline": c["commit"]["message"].split("\n", 1)[0],
                "messageBody": c["commit"]["message"].partition("\n")[2].strip(),
                "authoredDate": c["commit"]["author"]["date"],
                "committedDate": c["commit"]["committer"]["date"],
                "authors": [
                    {
                        "name": c["commit"]["author"]["name"],
                        "email": c["commit"]["author"]["email"],
                        "login": "",
                    }
                ],
            }
            for c in _commits(pin)
        ]
    return {f: computed[f] if f in computed else data.get(f) for f in want}


def _render_view(d: dict[str, Any]) -> str:
    def login(v: Any) -> str:
        return str((v or {}).get("login") or "") if isinstance(v, dict) else str(v or "")

    lines = [
        f"title:\t{d.get('title') or ''}",
        f"state:\t{d.get('state') or ''}",
        f"author:\t{login(d.get('author'))}",
        f"labels:\t{', '.join(str(lb.get('name')) for lb in d.get('labels') or [] if isinstance(lb, dict))}",
        f"number:\t{d.get('number') or ''}",
        f"url:\t{d.get('url') or ''}",
        f"base:\t{d.get('baseRefName') or ''}",
        f"head:\t{d.get('headRefName') or ''}",
        f"draft:\t{str(bool(d.get('isDraft'))).lower()}",
        "--",
        str(d.get("body") or ""),
    ]
    for c in d.get("comments") or []:
        lines += [
            "",
            f"-- {login(c.get('author'))} ({c.get('createdAt') or ''})",
            str(c.get("body") or ""),
        ]
    return "\n".join(lines) + "\n"


def _pr_checks(pin: Pin, rest: list[str]) -> Result:
    pos, flags = _parse(rest)
    repo, number = _target(pin, pos, flags)
    if not _is_pinned(pin, repo, number):
        raise Refused(
            f"CI of {repo}#{number} is not pinned in this replay; only #{pin.number}'s is served"
        )
    checks = _ci_checks(pin)
    if checks is None:
        return _fail(
            "CI status unavailable in this replay: the source run recorded no CI snapshot "
            f"for {pin.sha[:12]}"
        )
    rows = [
        {
            "name": c.get("name") or "",
            "state": str(c.get("conclusion") or c.get("status") or "").upper(),
            "bucket": _bucket(c),
            "link": c.get("url") or "",
            "description": "",
            "workflow": "",
            "event": "",
            "startedAt": None,
            "completedAt": None,
        }
        for c in checks
    ]
    note = f"note: CI as the source run recorded it at {(pin.ci or {}).get('fetched_at') or '?'} (replay)\n"
    fields = _flag(flags, "--json")
    if fields is not None:
        want = [f.strip() for f in fields.split(",") if f.strip()]
        res = _emit(pin, [{k: r.get(k) for k in want} for r in rows], _flag(flags, "-q", "--jq"))
        return Result(res.args, res.returncode, res.stdout, note + res.stderr)
    out = "".join(f"{r['name']}\t{r['bucket']}\t\t{r['link']}\t\n" for r in rows)
    buckets = {r["bucket"] for r in rows}
    rc = 1 if "fail" in buckets else 8 if "pending" in buckets else 0
    return Result([], rc, out, note)


def _run_view(args: list[str], real: RealGh) -> Result:
    pos, flags = _parse(args[2:])
    if "--web" in flags or "-w" in flags:
        raise Refused("`--web` is not available in a replay")
    if not (pos and pos[0].isdigit()) and not _flag(flags, "-j", "--job"):
        raise Refused("`gh run view` needs a run id (or --job id) in a replay")
    return real(args)


# ---- gh api ----


_API_VALUE_FLAGS = frozenset(
    {
        "-X",
        "--method",
        "-f",
        "--raw-field",
        "-F",
        "--field",
        "-q",
        "--jq",
        "-t",
        "--template",
        "-H",
        "--header",
        "--hostname",
        "-p",
        "--preview",
        "--cache",
        "--input",
    }
)


def _api(pin: Pin, args: list[str], real: RealGh, stdin: str | None) -> Result:
    rest: list[str] = []  # `--flag=value` spelled as two arguments
    for a in args:
        flag, eq, v = a.partition("=")
        rest += [flag, v] if a.startswith("--") and eq and flag in _API_VALUE_FLAGS else [a]
    endpoint = ""
    method: str | None = None
    fields: list[tuple[str, str, str]] = []  # (flag, key, value)
    keep: list[str] = []  # flags passed through to the real call (headers, hostname, ...)
    jq = template = None
    paginate = slurp = include = has_input = False
    i = 0

    def value() -> str:
        nonlocal i
        i += 1
        if i >= len(rest):
            raise Refused(f"`{rest[i - 1]}` needs a value")
        return rest[i]

    while i < len(rest):
        name = rest[i]
        val = value
        if name in ("-X", "--method"):
            method = val().upper()
        elif name in ("-f", "--raw-field", "-F", "--field"):
            k, _, v = val().partition("=")
            fields.append((name, k, v))
        elif name in ("-q", "--jq"):
            jq = val()
        elif name in ("-t", "--template"):
            template = val()
        elif name in ("-H", "--header", "--hostname", "-p", "--preview", "--cache"):
            keep += [name, val()]
        elif name == "--input":
            val()
            has_input = True
        elif name == "--paginate":
            paginate = True
        elif name == "--slurp":
            slurp = True
        elif name in ("-i", "--include"):
            include = True
        elif name in ("--silent", "--verbose"):
            keep.append(name)
        elif name.startswith("-") and not endpoint:
            raise Refused(f"`gh api {name}` is not understood by this replay")
        elif not endpoint:
            endpoint = name
        else:
            raise Refused(f"unexpected argument `{name}`")
        i += 1
    if not endpoint:
        raise Refused("`gh api` needs an endpoint")
    if endpoint == "graphql":
        query = _graphql_query(fields, stdin)
        if GRAPHQL_MUTATION.search(query):
            raise Refused(
                f"refused a GraphQL mutation: this is a read-only replay of {pin.where}; "
                "nothing may be written to GitHub"
            )
        if template is not None or include:
            raise Refused("`--template`/`--include` are not available on a replay's GraphQL reads")
        live = real(["api", "graphql", *_field_args(fields), *keep, *_page_args(paginate, slurp)])
        if live.returncode:
            return live
        data = json.loads(live.stdout or "null")
        if _mentions_pin(pin, query, fields):
            data = _pin_pull_request(pin, data)
        return _emit(pin, _unslurp(_before_cutoff(pin, data), paginate, slurp), jq)
    method = method or ("POST" if fields or has_input else "GET")
    if method != "GET" or has_input:
        raise Refused(
            f"refused `gh api {endpoint}` ({method}): this is a read-only replay of "
            f"{pin.where}; nothing may be written to GitHub"
        )
    path = endpoint.removeprefix("https://api.github.com").lstrip("/").split("?", 1)[0]
    parts = path.split("/")
    if parts[0] != "repos" or len(parts) < 3 or not _same_repo(pin, "/".join(parts[1:3])):
        # another repository, or not repository-scoped: nothing of this PR to pin
        return real(["api", *rest])
    if template is not None or include:
        raise Refused("`--template`/`--include` are not available on this replay's pinned reads")
    tail = parts[3:]
    synthetic = _synthetic(pin, tail)
    if synthetic is not None:
        if isinstance(synthetic, subprocess.CompletedProcess):
            return synthetic
        return _emit(pin, _paged(synthetic, paginate, slurp), jq)
    live = real(
        [
            "api",
            endpoint,
            "--method",
            "GET",
            *_field_args(fields),
            *keep,
            *_page_args(paginate, slurp),
        ]
    )
    if live.returncode:
        return live
    data = _before_cutoff(pin, json.loads(live.stdout or "null"), pin_objects=True)
    if tail[:2] in (["pulls", str(pin.number)], ["issues", str(pin.number)]) and isinstance(
        data, dict
    ):
        data = _pin_rest_pr(pin, data)
    return _emit(pin, _unslurp(data, paginate, slurp), jq)


def _synthetic(pin: Pin, tail: list[str]) -> Any:
    """The pinned answer for a repository endpoint git or the snapshot can serve, a failed
    Result for one that must fail, or None for "fetch it and filter"."""
    n = str(pin.number)
    if tail == ["pulls", n, "files"]:
        return _files(pin)
    if tail == ["pulls", n, "commits"]:
        return _commits(pin)
    if tail == ["pulls", n, "merge"]:
        return _fail("gh: Not Found (HTTP 404)")  # not merged, as of the cutoff
    if len(tail) == 3 and tail[0] == "commits" and _is_head(pin, tail[1]):
        kind = tail[2]
        if kind not in ("check-runs", "status", "statuses", "check-suites"):
            return None
        checks = _ci_checks(pin)
        if checks is None:
            return _fail(
                "gh: CI unavailable in this replay: the source run recorded no CI snapshot"
            )
        if kind == "check-runs":
            return {
                "total_count": len(checks),
                "check_runs": [
                    {
                        "name": c.get("name"),
                        "status": c.get("status"),
                        "conclusion": c.get("conclusion"),
                        "html_url": c.get("url"),
                        "details_url": c.get("url"),
                        "head_sha": pin.sha,
                    }
                    for c in checks
                ],
            }
        if kind == "status":  # every recorded check is served as a check run above
            return {"state": "success", "sha": pin.sha, "total_count": 0, "statuses": []}
        if kind == "statuses":
            return []
        return {"total_count": 0, "check_suites": []}
    return None


def _is_head(pin: Pin, ref: str) -> bool:
    return len(ref) >= 7 and pin.sha.startswith(ref.lower())


def _graphql_query(fields: list[tuple[str, str, str]], stdin: str | None) -> str:
    for flag, k, v in fields:
        if k != "query":
            continue
        if flag in ("-F", "--field") and v.startswith("@"):
            if v == "@-":
                return stdin or ""
            try:
                return Path(v[1:]).read_text()
            except OSError as exc:
                raise Refused(f"cannot read the query file {v[1:]}: {exc}") from exc
        return v
    return ""


def _field_args(fields: list[tuple[str, str, str]]) -> list[str]:
    return [a for flag, k, v in fields for a in (flag, f"{k}={v}")]


def _page_args(paginate: bool, slurp: bool) -> list[str]:
    # always slurped when paginated, so the pages parse as one JSON document
    return ["--paginate", "--slurp"] if paginate else []


def _unslurp(data: Any, paginate: bool, slurp: bool) -> Any:
    """What the caller asked for from what `_page_args` fetched: the slurped pages as they
    are, or (no `--slurp`) one list for array pages."""
    if not paginate or slurp or not isinstance(data, list):
        return data
    if all(isinstance(p, list) for p in data):
        return [row for page in data for row in page]
    return data[0] if len(data) == 1 else data


def _paged(data: Any, paginate: bool, slurp: bool) -> Any:
    return [data] if paginate and slurp else data


def _mentions_pin(pin: Pin, query: str, fields: list[tuple[str, str, str]]) -> bool:
    """A GraphQL read about the replayed PR: its number and repository name appear in the
    query or its variables."""
    text = query + " " + " ".join(f"{k}={v}" for _, k, v in fields)
    name = pin.repo.split("/", 1)[1]
    return bool(re.search(rf"\b{pin.number}\b", text)) and name.lower() in text.lower()


# ---- the view ----


def _before_cutoff(pin: Pin, data: Any, *, pin_objects: bool = False) -> Any:
    """`data` without anything created at or after the cutoff: list items with a later
    creation time, and review threads whose every comment is later. With `pin_objects` (a
    response from the replayed repository) the replayed PR is pinned wherever it appears as
    an object carrying its number."""
    if isinstance(data, list):
        kept = []
        for item in data:
            if isinstance(item, dict) and _late(pin, item):
                continue
            new = _before_cutoff(pin, item, pin_objects=pin_objects)
            if isinstance(item, dict) and _emptied_thread(item, new):
                continue
            kept.append(new)
        return kept
    if isinstance(data, dict):
        out = {k: _before_cutoff(pin, v, pin_objects=pin_objects) for k, v in data.items()}
        if pin_objects and out.get("number") == pin.number:
            if "head" in out:
                out = _pin_rest_pr(pin, out)
            elif "headRefOid" in out:
                out = _pin_graphql_pr(pin, out)
        return out
    return data


def _late(pin: Pin, item: dict[str, Any]) -> bool:
    return any(isinstance(item.get(k), str) and item[k] >= pin.cutoff for k in _CREATED)


def _emptied_thread(before: dict[str, Any], after: Any) -> bool:
    """A review thread (GraphQL) every comment of which came after the cutoff."""
    b, a = before.get("comments"), after.get("comments") if isinstance(after, dict) else None
    if not (isinstance(b, dict) and isinstance(a, dict)):
        return False
    return bool(b.get("nodes")) and not a.get("nodes")


def _pin_rest_pr(pin: Pin, d: dict[str, Any]) -> dict[str, Any]:
    out = {**d, "state": "open", "closed_at": None}
    if "head" in d:
        out.update(merged=False, merged_at=None, merge_commit_sha=None, merged_by=None)
    if isinstance(d.get("head"), dict):
        out["head"] = {**d["head"], "sha": pin.sha}
    if isinstance(d.get("base"), dict) and pin.base_sha:
        out["base"] = {**d["base"], "sha": pin.base_sha}
    if "pull_request" in d and isinstance(d["pull_request"], dict):
        out["pull_request"] = {**d["pull_request"], "merged_at": None}
    for k in ("title", "body"):
        if k in pin.pr:
            out[k] = pin.pr[k]
    return out


def _pin_graphql_pr(pin: Pin, d: dict[str, Any]) -> dict[str, Any]:
    out = dict(d)
    pinned = {
        "headRefOid": pin.sha,
        "state": "OPEN",
        "merged": False,
        "closed": False,
        "mergedAt": None,
        "closedAt": None,
        "mergeCommit": None,
        "mergedBy": None,
    }
    if pin.base_sha:
        pinned["baseRefOid"] = pin.base_sha
    for k, v in pinned.items():
        if k in out:
            out[k] = v
    for k in ("title", "body"):
        if k in out and k in pin.pr:
            out[k] = pin.pr[k]
    return out


def _pin_pull_request(pin: Pin, data: Any) -> Any:
    """Pin every `pullRequest` object of a GraphQL read about the replayed PR."""
    if isinstance(data, list):
        return [_pin_pull_request(pin, x) for x in data]
    if isinstance(data, dict):
        out = {k: _pin_pull_request(pin, v) for k, v in data.items()}
        if isinstance(out.get("pullRequest"), dict):
            out["pullRequest"] = _pin_graphql_pr(pin, out["pullRequest"])
        return out
    return data


def _ci_checks(pin: Pin) -> list[dict[str, Any]] | None:
    ci = pin.ci
    if not isinstance(ci, dict) or ci.get("error") or not isinstance(ci.get("checks"), list):
        return None
    return [c for c in ci["checks"] if isinstance(c, dict)]


_FAILING = frozenset(
    {"failure", "timed_out", "cancelled", "action_required", "startup_failure", "error"}
)


def _bucket(c: dict[str, Any]) -> str:
    conclusion = str(c.get("conclusion") or "").lower()
    if str(c.get("status") or "").lower() not in ("completed", "") or conclusion in ("", "pending"):
        return "pending"
    if conclusion in _FAILING:
        return "fail"
    if conclusion == "skipped":
        return "skipping"
    return "pass"


def _rollup(pin: Pin) -> list[dict[str, Any]]:
    return [
        {
            "__typename": "CheckRun",
            "name": c.get("name"),
            "status": str(c.get("status") or "").upper(),
            "conclusion": str(c.get("conclusion") or "").upper(),
            "detailsUrl": c.get("url"),
        }
        for c in _ci_checks(pin) or []
    ]


# ---- git ----

_CHANGE_TYPES = {
    "added": "ADDED",
    "removed": "DELETED",
    "modified": "MODIFIED",
    "renamed": "RENAMED",
    "copied": "COPIED",
    "changed": "CHANGED",
}
_STATUS = {"A": "added", "D": "removed", "M": "modified", "R": "renamed", "C": "copied"}


def _git(pin: Pin, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", pin.git_dir, *args],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if proc.returncode:
        raise Refused(f"git {args[0]} failed in the replay mirror: {proc.stderr.strip()[:300]}")
    return proc.stdout


def _diff(pin: Pin) -> str:
    return _git(pin, "diff", "--no-color", "-M", pin.merge_base, pin.sha)


def _patches(diff: str) -> dict[str, str]:
    """path -> the hunks of a unified diff, as GitHub's `patch` field holds them."""
    out: dict[str, str] = {}
    for chunk in re.split(r"(?m)^diff --git ", diff)[1:]:
        lines = chunk.split("\n")
        path = ""
        for line in lines[1:]:
            if line.startswith("+++ b/") or (line.startswith("--- a/") and not path):
                path = line[6:]
            elif line.startswith("rename to "):
                path = line[10:]
            elif line.startswith("@@"):
                break
        hunk = next((i for i, x in enumerate(lines) if x.startswith("@@")), None)
        if path and hunk is not None:
            out[path] = "\n".join(lines[hunk:]).rstrip("\n")
    return out


def _files(pin: Pin) -> list[dict[str, Any]]:
    """`pulls/N/files`, from git."""
    status: dict[str, tuple[str, str]] = {}  # path -> (status, previous path)
    toks = _git(pin, "diff", "--name-status", "-z", "-M", pin.merge_base, pin.sha).split("\0")
    i = 0
    while i < len(toks) and toks[i]:
        code = toks[i][0]
        if code in "RC":
            status[toks[i + 2]] = (_STATUS[code], toks[i + 1])
            i += 3
        else:
            status[toks[i + 1]] = (_STATUS.get(code, "changed"), "")
            i += 2
    counts: dict[str, tuple[int, int]] = {}
    toks = _git(pin, "diff", "--numstat", "-z", "-M", pin.merge_base, pin.sha).split("\0")
    i = 0
    while i < len(toks) and toks[i]:
        add, dele, path = toks[i].split("\t", 2)
        if path == "":  # a rename: old and new path follow
            path = toks[i + 2]
            i += 2
        counts[path] = (int(add) if add.isdigit() else 0, int(dele) if dele.isdigit() else 0)
        i += 1
    patches = _patches(_diff(pin))
    out = []
    for path, (st, prev) in status.items():
        a, d = counts.get(path, (0, 0))
        f: dict[str, Any] = {
            "filename": path,
            "status": st,
            "additions": a,
            "deletions": d,
            "changes": a + d,
        }
        if path in patches:
            f["patch"] = patches[path]
        if prev:
            f["previous_filename"] = prev
        out.append(f)
    return out


def _utc(ts: str) -> str:
    return datetime.fromisoformat(ts).astimezone(UTC).isoformat().replace("+00:00", "Z")


def _commits(pin: Pin) -> list[dict[str, Any]]:
    """`pulls/N/commits`, from git."""
    fmt = "%H%x00%P%x00%an%x00%ae%x00%aI%x00%cn%x00%ce%x00%cI%x00%B%x1e"
    raw = _git(pin, "log", "--reverse", f"--format={fmt}", f"{pin.merge_base}..{pin.sha}")
    out = []
    for rec in raw.split("\x1e"):
        rec = rec.strip("\n")
        if not rec:
            continue
        sha, parents, an, ae, ad, cn, ce, cd, msg = rec.split("\0", 8)
        out.append(
            {
                "sha": sha,
                "commit": {
                    "message": msg.strip(),
                    "author": {"name": an, "email": ae, "date": _utc(ad)},
                    "committer": {"name": cn, "email": ce, "date": _utc(cd)},
                },
                "parents": [{"sha": p} for p in parents.split()],
            }
        )
    return out


# ---- output ----


def _ok(out: str) -> Result:
    return Result([], 0, out, "")


def _fail(msg: str) -> Result:
    return Result([], 1, "", msg.rstrip("\n") + "\n")


def _emit(pin: Pin, data: Any, jq: str | None) -> Result:
    text = json.dumps(data)
    if jq is None:
        return _ok(text + "\n")
    if not pin.jq:
        raise Refused("`--jq` is not available in this replay (no jq binary); drop --jq")
    proc = subprocess.run(
        [pin.jq, "-r", jq], input=text, capture_output=True, text=True, check=False
    )
    return Result([], proc.returncode, proc.stdout, proc.stderr)


def _log(pin: Pin, source: str, args: list[str], outcome: str, rc: int) -> None:
    if not pin.log:
        return
    line = json.dumps(
        {
            "ts": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "source": source,
            "args": [a[:200] for a in args[:12]],
            "outcome": outcome,
            "rc": rc,
        }
    )
    try:
        with open(pin.log, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
