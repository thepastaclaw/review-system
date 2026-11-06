"""GitHub App authentication: the bot's PAT cannot write check runs, an App installation can.

The App signs a short-lived JWT with its private key (RS256, through `openssl`, so the runtime
stays stdlib-only), finds the installation that covers a repository, and trades the JWT for an
installation token scoped to Checks: write. Every call goes through `gh`, like the rest of the
system: the JWT as an explicit `Authorization: Bearer` header (gh sends it instead of its own
token), the installation token as `GH_TOKEN`.
"""

from __future__ import annotations

import base64
import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .db import parse_ts
from .gh import Gh, env_runner
from .models import FailKind, ReviewError

JWT_LIFETIME_S = 540  # GitHub caps it at 600; leaves room for clock drift
JWT_BACKDATE_S = 60  # GitHub's recommendation for clock drift
RENEW_MARGIN_S = 300  # renew a JWT or installation token this long before it expires
# a repository without the installation is looked up again after this long, so installing the
# App on another repository needs no restart
MISSING_RECHECK_S = 3600
TOKEN_PERMISSIONS = {"checks": "write"}
# check-run calls sit in the daemon's single-threaded tick: a hung call must not hold it long
CALL_TIMEOUT_S = 30

Signer = Callable[[bytes, Path], bytes]
TokenClient = Callable[[str], Gh]  # installation token -> a gh client using it


class AppAuthError(ReviewError):
    """The App itself cannot authenticate (key unreadable, JWT rejected): every repository
    is affected, unlike a failure of one installation."""

    def __init__(self, message: str) -> None:
        super().__init__(FailKind.FATAL, message)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def openssl_sign(data: bytes, key_path: Path) -> bytes:
    """RS256 signature of `data` with the PEM key at `key_path`."""
    try:
        proc = subprocess.run(
            ["openssl", "dgst", "-sha256", "-sign", str(key_path)],
            input=data,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AppAuthError(f"cannot run openssl: {exc}") from exc
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", "replace").strip()[:200]
        raise AppAuthError(f"cannot sign the App JWT with {key_path}: {err}")
    return proc.stdout


@dataclass(slots=True)
class _Token:
    value: str
    expires_at: float


class AppAuth:
    """Installation-token clients for one GitHub App, cached per installation."""

    def __init__(
        self,
        app_id: str,
        key_path: Path,
        *,
        gh_bin: str = "gh",
        app_gh: Gh | None = None,
        token_client: TokenClient | None = None,
        signer: Signer = openssl_sign,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.app_id = app_id
        self.key_path = key_path
        self.signer = signer
        self.clock = clock
        self._app_gh = app_gh or Gh(gh_bin, default_timeout=CALL_TIMEOUT_S)
        self._token_client = token_client or (
            lambda token: Gh(
                gh_bin, env_runner({"GH_TOKEN": token}), default_timeout=CALL_TIMEOUT_S
            )
        )
        self._jwt: _Token | None = None
        self._app: dict[str, object] | None = None
        self._tokens: dict[int, _Token] = {}
        # repo -> (installation id or None, when looked up)
        self._installations: dict[str, tuple[int | None, float]] = {}

    def jwt(self) -> str:
        now = self.clock()
        if self._jwt and self._jwt.expires_at - RENEW_MARGIN_S > now:
            return self._jwt.value
        header = _b64url(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
        claims = {
            "iat": int(now) - JWT_BACKDATE_S,
            "exp": int(now) + JWT_LIFETIME_S,
            "iss": self.app_id,
        }
        payload = _b64url(json.dumps(claims, separators=(",", ":")).encode())
        signing_input = f"{header}.{payload}".encode("ascii")
        token = f"{header}.{payload}.{_b64url(self.signer(signing_input, self.key_path))}"
        self._jwt = _Token(token, now + JWT_LIFETIME_S)
        return token

    def _as_app(self, endpoint: str, *, method: str = "GET", body: object = None) -> object:
        args = ["api", endpoint, "--method", method, "-H", f"Authorization: Bearer {self.jwt()}"]
        stdin = None
        if body is not None:
            args += ["--input", "-"]
            stdin = json.dumps(body)
        try:
            return self._app_gh.json(*args, stdin=stdin)
        except ReviewError as exc:
            if "HTTP 401" in exc.message:  # the JWT itself: wrong key or app id
                raise AppAuthError(f"GitHub rejected the App JWT: {exc.message}") from exc
            raise

    def app(self) -> dict[str, object]:
        """The App itself (`GET /app`), cached: proves the id and key belong together."""
        if self._app is None:
            data = self._as_app("app")
            self._app = data if isinstance(data, dict) else {}
        return self._app

    def installation(self, repo: str) -> int | None:
        """The id of the installation covering `repo`, or None when the App is not installed
        there. Raises ReviewError(INFRA) on a transient failure (nothing is cached then)."""
        now = self.clock()
        cached = self._installations.get(repo)
        if cached and (cached[0] is not None or now - cached[1] < MISSING_RECHECK_S):
            return cached[0]
        try:
            data = self._as_app(f"repos/{repo}/installation")
        except ReviewError as exc:
            if exc.kind is FailKind.INFRA or "HTTP 404" not in exc.message:
                raise
            data = None
        iid = int(data["id"]) if isinstance(data, dict) and data.get("id") else None
        self._installations[repo] = (iid, now)
        return iid

    def forget(self, repo: str) -> None:
        """Drop what is cached for `repo` (its installation was removed or lost access)."""
        cached = self._installations.pop(repo, None)
        if cached and cached[0] is not None:
            self._tokens.pop(cached[0], None)

    def token(self, installation_id: int) -> str:
        now = self.clock()
        tok = self._tokens.get(installation_id)
        if tok and tok.expires_at - RENEW_MARGIN_S > now:
            return tok.value
        data = self._as_app(
            f"app/installations/{installation_id}/access_tokens",
            method="POST",
            body={"permissions": TOKEN_PERMISSIONS},
        )
        if not isinstance(data, dict) or not data.get("token"):
            raise ReviewError(FailKind.CONTRACT, "installation token response has no token")
        value = str(data["token"])
        expires = data.get("expires_at")
        expires_at = parse_ts(str(expires)).timestamp() if expires else now + 3600
        self._tokens[installation_id] = _Token(value, expires_at)
        return value

    def gh_for(self, repo: str) -> Gh | None:
        """A `gh` client acting as the App's installation on `repo`, or None if not installed."""
        iid = self.installation(repo)
        if iid is None:
            return None
        return self._token_client(self.token(iid))
