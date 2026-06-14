"""Content-validity gate for the interpret CLI seam (MYC-488).

`_claude_via_cli` used to return ANY stdout (>= 1 char) as the document answer.
A `claude -p` rate-limit / session-limit / auth banner printed to stdout was
therefore served to the caller AS THE ANSWER (bug class PRODUCER-OUTPUT-
CONSUMED-WITHOUT-CONTENT-VALIDITY-CHECK; the bare-floor form of the MYC-420
router bug).

The fix gates success STRUCTURALLY on a `--output-format json` result envelope
with is_error=false + non-empty + non-truncated result. These tests pin that
no banner / error / truncated / empty body is ever returned as the answer, and
that a valid envelope is accepted even on a non-zero exit (SessionEnd hook
failing after Claude wrote the response).

Run with:
    pytest tests/integration/test_interpret_cli_envelope.py -v
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import interpret  # noqa: E402


class _Proc:
    """Minimal subprocess.CompletedProcess stand-in."""

    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _run_cli(monkeypatch, stdout: str, *, returncode: int = 0, stderr: str = ""):
    """Drive _claude_via_cli with a mocked CLI returning `stdout`."""
    monkeypatch.setattr(interpret, "_resolve_claude_cli", lambda: "/usr/bin/true")
    monkeypatch.setattr(
        interpret.subprocess,
        "run",
        lambda *a, **k: _Proc(stdout, stderr, returncode),
    )
    return interpret._claude_via_cli("markdown", "doc.pdf", "summarize", "sonnet")


def _envelope(**over) -> str:
    body = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "The document says X.",
        "stop_reason": "end_turn",
    }
    body.update(over)
    return json.dumps(body)


# ---------------------------------------------------------------------------
# Success: the envelope's `result` is the answer (NOT raw stdout).
# ---------------------------------------------------------------------------
def test_valid_success_envelope_returns_result(monkeypatch) -> None:
    ans, usage, err = _run_cli(monkeypatch, _envelope(result="The doc says X."))
    assert err is None
    assert ans == "The doc says X."
    assert usage.get("stop_reason") == "end_turn"


def test_valid_envelope_accepted_despite_nonzero_exit(monkeypatch) -> None:
    # A SessionEnd hook failing AFTER Claude wrote its response exits non-zero,
    # but the envelope on stdout is still valid → still accepted.
    ans, _usage, err = _run_cli(monkeypatch, _envelope(result="Answer."), returncode=1)
    assert err is None
    assert ans == "Answer."


def test_envelope_on_last_line_after_leaked_hook_text(monkeypatch) -> None:
    stdout = "some hook line leaked ahead of the envelope\n" + _envelope(result="OK.")
    ans, _usage, err = _run_cli(monkeypatch, stdout)
    assert err is None
    assert ans == "OK."


# ---------------------------------------------------------------------------
# Failure modes: NONE of these are ever returned as the answer.
# ---------------------------------------------------------------------------
def test_ratelimit_banner_is_not_consumed_as_answer(monkeypatch) -> None:
    # The exact MYC-420 failure: a banner printed to stdout. No envelope → error.
    ans, _usage, err = _run_cli(
        monkeypatch,
        "You've hit your session limit · resets 11:50pm",
        returncode=1,
    )
    assert ans == ""
    assert err is not None and "non-envelope" in err


def test_is_error_envelope_is_not_consumed(monkeypatch) -> None:
    stdout = _envelope(is_error=True, subtype="error_during_execution", result="boom")
    ans, _usage, err = _run_cli(monkeypatch, stdout)
    assert ans == ""
    assert err is not None and "error envelope" in err


def test_truncated_response_is_not_served_as_complete(monkeypatch) -> None:
    stdout = _envelope(result="A partial answer that got cut", stop_reason="max_tokens")
    ans, _usage, err = _run_cli(monkeypatch, stdout)
    assert ans == ""
    assert err is not None and "truncat" in err.lower()


def test_empty_result_envelope_fails(monkeypatch) -> None:
    ans, _usage, err = _run_cli(monkeypatch, _envelope(result=""))
    assert ans == ""
    assert err is not None and "empty result" in err.lower()


def test_empty_stdout_fails(monkeypatch) -> None:
    ans, _usage, err = _run_cli(monkeypatch, "", returncode=1, stderr="boom")
    assert ans == ""
    assert err is not None


# ---------------------------------------------------------------------------
# _extract_result_envelope unit behavior (the positive structural gate).
# ---------------------------------------------------------------------------
def test_extract_envelope_structural_gate() -> None:
    assert interpret._extract_result_envelope("") is None
    assert interpret._extract_result_envelope("not json at all") is None
    # A JSON object that is not a result envelope is not accepted.
    assert interpret._extract_result_envelope('{"type": "other"}') is None
    env = interpret._extract_result_envelope(_envelope(result="x"))
    assert env is not None and env["result"] == "x"
