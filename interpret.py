"""Interpret tool: parse first, then ask Claude over the markdown.

Bridges deterministic parsing and interpretive work. Flow:

1. Route the input through the parse backends to get clean markdown.
2. Send the markdown plus the operator's instruction to Claude.
3. Return Claude's response alongside the parse audit trail.

**Auth (Claude Max subscription by default):** This module mirrors the
canonical vault helper at `⚙️ Meta/scripts/_claude_router.py` (codified
2026-05-10 after the substrate audit migrated 4 vault scripts off
ANTHROPIC_API_KEY). It shells out to the `claude` CLI with the proven
flag set:

    claude -p <user> --append-system-prompt <system>
           --model <m> --no-session-persistence --disable-slash-commands

Critical insight from `_claude_router.py`: `claude -p` runs SessionStart
+ SessionEnd hooks even in print mode. A failing hook (e.g.
sync-my-skills.sh push) exits 1 AFTER Claude has already written its
response to stdout. We accept any exit code when stdout has substantial
content (>=20 chars). Throwing on non-zero exit discards a perfectly
good response.

**Fallback (API key, opt-in):** If the CLI fails OR
`CLAUDE_ROUTER_PREFER_API_KEY=1` is set, falls back to the Anthropic
SDK. Caller can disable the CLI entirely with `CLAUDE_ROUTER_DISABLE_CLI=1`.

**NVIDIA tier (separate, not used here):** For grunt-work text generation
(translation, format conversion, simple extraction), `⚙️ Meta/scripts/nvidia.sh`
routes to NVIDIA's free credits on Llama 3.3 70B / Qwen3 80B /
DeepSeek V4 / Nemotron. Not appropriate for `interpret` which needs
Claude-quality reasoning over parsed documents.

Why parse-first vs. handing the file to Claude directly:

- Markdown is 30 to 60% smaller than a raw PDF byte stream as multimodal input.
- Deterministic structure means Claude reasons over consistent input.
- The parsed text is reusable across multiple interpret calls.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backends.types import ParseResult
from router import RouteResult, route

# Soft ceiling: at some point you want to chunk anyway.
_TRUNCATE_AT_CHARS = 800_000

# Subprocess timeout.
_CLI_TIMEOUT_SECONDS = int(os.environ.get("PARSE_MCP_CLAUDE_CLI_TIMEOUT", "600"))

# MYC-488: CLI output is gated STRUCTURALLY on the `--output-format json` result
# envelope in `_claude_via_cli`, not on a raw stdout char-count. The old
# "any stdout >= 1 char is the answer" floor let a rate-limit / session-limit /
# auth banner be returned as the document answer (bug class PRODUCER-OUTPUT-
# CONSUMED-WITHOUT-CONTENT-VALIDITY-CHECK; the MYC-420 router fix, transferred
# to this CLI seam).


@dataclass
class InterpretResult:
    answer: str
    parse: RouteResult
    model: str
    instruction: str
    truncated: bool = False
    error: str | None = None
    auth: str = "unknown"  # "max-subscription" | "api-key" | "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "model": self.model,
            "instruction": self.instruction,
            "truncated": self.truncated,
            "error": self.error,
            "auth": self.auth,
            "parse": self.parse.to_dict(),
            "metadata": self.metadata,
        }


def _resolve_claude_cli() -> str | None:
    """Find the local `claude` CLI binary. Mirrors `_claude_router.py`."""
    if os.environ.get("CLAUDE_ROUTER_DISABLE_CLI") == "1":
        return None

    explicit = os.environ.get("CLAUDE_CLI_PATH")
    if explicit and Path(explicit).exists():
        return explicit

    candidate = shutil.which("claude")
    if candidate:
        return candidate

    home = Path.home()
    for fallback in [
        home / "local/node-v20.19.0-darwin-arm64/bin/claude",
        home / ".local/bin/claude",
        Path("/opt/homebrew/bin/claude"),
        Path("/usr/local/bin/claude"),
    ]:
        if fallback.exists():
            return str(fallback)

    return None


def _build_user_msg(markdown: str, filename: str | None, instruction: str) -> str:
    return (
        f"<document filename=\"{filename or 'input'}\">\n{markdown}\n</document>\n\n"
        f"Instruction: {instruction}"
    )


_SYSTEM_PROMPT = (
    "You are reading a document that has been pre-parsed to markdown. "
    "Answer the user's instruction strictly from the document content. "
    "If the document does not contain the answer, say so. Do not invent."
)


def _extract_result_envelope(stdout: str) -> dict | None:
    """Return the `claude -p --output-format json` result envelope, or None when
    stdout is not a parseable result envelope.

    None is itself a STRUCTURAL failure signal: in json mode a real response is
    ALWAYS a valid ``{"type":"result",...}`` envelope, so unparseable stdout can
    never be a success — this is the positive content-validity gate (MYC-488,
    transferring the MYC-420 principle). Tolerant of a stray hook line leaked
    ahead of the envelope (scans from the last line back).
    """
    if not stdout:
        return None
    try:
        obj = json.loads(stdout)
        if isinstance(obj, dict) and obj.get("type") == "result":
            return obj
    except json.JSONDecodeError:
        pass
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("type") == "result":
            return obj
    return None


def _claude_via_cli(
    markdown: str,
    filename: str | None,
    instruction: str,
    model: str,
) -> tuple[str, dict[str, Any], str | None]:
    """Call `claude -p --output-format json` to use the Max subscription.
    Returns ``(answer, usage, error)``; on any non-success the answer is "" and
    the caller (``interpret``) falls through to the API-key SDK path.

    MYC-488 content-validity gate (transfers the MYC-420 router principle to this
    CLI seam). Success is gated STRUCTURALLY on a parseable result envelope with
    ``is_error=false`` and a non-empty, non-truncated ``result`` — NOT on "any
    stdout >= 1 char". A rate-limit / session-limit / auth banner printed to
    stdout has no envelope, so it can NEVER be consumed as the document answer
    (the bug this fixes). The check is independent of exit code: a SessionEnd
    hook failing AFTER Claude wrote its response exits non-zero, but the envelope
    on stdout is still valid, so we still accept it. HARD dependency on
    ``--output-format json``: a CLI that ignores the flag and prints plain text
    yields no envelope → treated as unavailable → SDK fallback (the safe
    direction; an unrecognized banner is never returned as content).
    """
    cli = _resolve_claude_cli()
    if cli is None:
        return "", {}, "claude CLI not found in PATH"

    user = _build_user_msg(markdown, filename, instruction)

    try:
        proc = subprocess.run(
            [
                cli,
                "-p",
                user,
                "--append-system-prompt",
                _SYSTEM_PROMPT,
                "--model",
                model,
                "--output-format",
                "json",  # MYC-488: structured envelope, not raw stdout text
                "--no-session-persistence",
                "--disable-slash-commands",
            ],
            capture_output=True,
            text=True,
            timeout=_CLI_TIMEOUT_SECONDS,
            check=False,
            # CRITICAL: close stdin so claude doesn't wait on parent's stdin.
            stdin=subprocess.DEVNULL,
            # Neutral cwd so vault hooks don't match path-based triggers AND
            # any cwd-relative CLAUDE.md auto-discovery doesn't pollute the
            # subprocess context. The SessionEnd hook still fires but it
            # exits after Claude has written stdout.
            cwd=tempfile.gettempdir(),
        )
    except subprocess.TimeoutExpired:
        return "", {}, f"claude CLI timed out after {_CLI_TIMEOUT_SECONDS}s"
    except FileNotFoundError:
        return "", {}, "claude CLI not found"
    except OSError as exc:
        return "", {}, f"claude CLI launch failed: {exc}"

    stdout = (proc.stdout or "").strip()
    envelope = _extract_result_envelope(stdout)

    if envelope is not None and not bool(envelope.get("is_error")):
        text = envelope.get("result")
        text = "" if text is None else str(text)
        if not text.strip():
            # is_error:false but NO content — a real response is never empty
            # here. Fail rather than return "" as the answer.
            return "", {}, "claude CLI returned a success envelope with an empty result"
        stop_reason = envelope.get("stop_reason")
        if stop_reason == "max_tokens":
            # Cut off at the token cap — the answer is PARTIAL. Never serve a
            # truncated answer as complete (the same content-validity invariant
            # the MRP synthesis gate enforces).
            return "", {}, f"claude CLI response truncated at token cap (stop_reason={stop_reason})"
        usage: dict[str, Any] = {
            "exit_code": proc.returncode,
            "stop_reason": stop_reason,
        }
        if isinstance(envelope.get("usage"), dict):
            usage["tokens"] = envelope["usage"]
        return text.strip(), usage, None

    if envelope is not None:
        # Structured error envelope (is_error=true) — classify from its own
        # fields; never the answer.
        err = " ".join(
            str(envelope.get(k, ""))
            for k in ("subtype", "result", "stop_reason")
        ).strip()
        return "", {}, f"claude CLI error envelope: {err[:200]}"

    if stdout:
        # Non-envelope stdout in --output-format json mode = an error banner,
        # output corruption, or a CLI that ignores the flag. Failure by
        # construction — never consumed as a document answer.
        return "", {}, f"claude CLI returned non-envelope output (not a model response): {stdout[:200]}"

    stderr_tail = (proc.stderr or "")[-300:]
    return "", {}, f"claude CLI exit {proc.returncode} with empty stdout. stderr: {stderr_tail}"


def _claude_via_sdk(
    markdown: str,
    filename: str | None,
    instruction: str,
    model: str,
    max_tokens: int,
) -> tuple[str, dict[str, Any], str | None]:
    """Fallback path using the Anthropic SDK. Requires ANTHROPIC_API_KEY."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "", {}, "ANTHROPIC_API_KEY not set (and claude CLI unavailable)"

    try:
        import anthropic
    except ImportError:
        return "", {}, "anthropic SDK not installed"

    client = anthropic.Anthropic(api_key=api_key)
    user_blocks = [
        {
            "type": "text",
            "text": f"<document filename=\"{filename or 'input'}\">\n{markdown}\n</document>",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": f"Instruction: {instruction}"},
    ]
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_blocks}],
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )
        usage = getattr(resp, "usage", None)
        usage_dict = (
            {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
                "cache_creation_input_tokens": getattr(
                    usage, "cache_creation_input_tokens", None
                ),
            }
            if usage is not None
            else {}
        )
        return text, usage_dict, None
    except Exception as exc:
        return "", {}, f"{type(exc).__name__}: {exc}"


def interpret(
    data: bytes,
    *,
    filename: str | None,
    instruction: str,
    backend: str | None = None,
    model: str = "sonnet",
    max_tokens: int = 4096,
) -> InterpretResult:
    parse_result = route(data, filename=filename, backend=backend)
    if parse_result.final.error or not parse_result.final.markdown.strip():
        return InterpretResult(
            answer="",
            parse=parse_result,
            model=model,
            instruction=instruction,
            error=parse_result.final.error or "parse produced empty markdown",
        )

    markdown = parse_result.final.markdown
    truncated = False
    if len(markdown) > _TRUNCATE_AT_CHARS:
        markdown = markdown[:_TRUNCATE_AT_CHARS]
        truncated = True

    # Auth preference: Max subscription first (CLI), API key fallback only.
    # Codified rule: always use Max account when possible vs API key.
    # Pattern source: `⚙️ Meta/scripts/_claude_router.py`.
    prefer_api = os.environ.get("CLAUDE_ROUTER_PREFER_API_KEY") == "1"

    answer = ""
    usage: dict[str, Any] = {}
    auth = "unknown"
    err: str | None = None

    if not prefer_api:
        answer, usage, err = _claude_via_cli(markdown, filename, instruction, model)
        if err is None:
            auth = "max-subscription"
        else:
            # CLI unavailable or errored — try the SDK fallback.
            sdk_answer, sdk_usage, sdk_err = _claude_via_sdk(
                markdown, filename, instruction, model, max_tokens
            )
            if sdk_err is None:
                answer, usage, auth = sdk_answer, sdk_usage, "api-key"
                err = None
            else:
                err = f"cli: {err} | sdk: {sdk_err}"
    else:
        # Explicit opt-in to SDK path.
        answer, usage, err = _claude_via_sdk(
            markdown, filename, instruction, model, max_tokens
        )
        if err is None:
            auth = "api-key"

    return InterpretResult(
        answer=answer,
        parse=parse_result,
        model=model,
        instruction=instruction,
        truncated=truncated,
        error=err,
        auth=auth,
        metadata={"usage": usage},
    )
