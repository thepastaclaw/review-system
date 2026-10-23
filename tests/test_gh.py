"""Byte decoding at the real gh subprocess boundary, not just a string-only fake."""

import subprocess
import sys

import pytest

from reviewsys import github, publish
from reviewsys.gh import Gh


def test_pr_diff_accepts_blob_bytes_and_preserves_unicode(binary_diff_gh):
    diff = github.pr_diff(binary_diff_gh, "dashpay/platform", 5515)
    assert diff is not None
    assert "+café 東京 🦀\n" in diff
    assert "+Dc�" in diff
    parsed = publish.parse_diff(diff)
    assert publish.line_in_diff(parsed, "f.rs", 12)
    assert publish.line_in_diff(parsed, "key-hash.bin", 1)
    assert not publish.line_in_diff(parsed, "proof.bin", 1)


def test_api_decoding_is_still_strict(binary_diff_gh):
    with pytest.raises(UnicodeDecodeError):
        binary_diff_gh.api("repos/dashpay/platform/pulls/5515")


def test_default_runner_preserves_unicode_input_and_output(tmp_path):
    script = tmp_path / "gh-echo"
    script.write_text(
        f"#!{sys.executable}\nimport sys\nsys.stdout.buffer.write(sys.stdin.buffer.read())\n"
    )
    script.chmod(0o755)
    assert Gh(str(script)).run("api", stdin="café 東京 🦀") == "café 東京 🦀"


def test_pr_diff_too_large_still_uses_body_only_fallback():
    gh = Gh(
        runner=lambda argv, stdin, timeout: subprocess.CompletedProcess(
            argv, 1, b"", b"diff exceeded the maximum number of lines"
        )
    )
    assert github.pr_diff(gh, "dashpay/platform", 5515) is None
