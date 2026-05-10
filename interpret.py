"""Interpret tool: parse first, then ask Claude over the markdown.

This is the bridge between deterministic parsing and interpretive work.
The flow:

1. Route the input through the parse backends to get clean markdown.
2. Send the markdown plus the operator's instruction to Claude.
3. Return Claude's response alongside the parse audit trail so the
   caller knows which backend produced the markdown Claude saw.

Why bother (vs just handing the file to Claude directly)?

- markitdown markdown is 30 to 60% smaller than a raw PDF byte stream
  fed to Claude as a multimodal input. Token economics favor parse-first.
- The markdown is reusable: subsequent calls reuse the parsed text
  without re-extracting. The audit trail tells you which backend ran,
  so you can cache by hash.
- Deterministic structure (headings, tables) means Claude reasons over
  consistent input, not OCR noise.
- Saves the model from having to "see" the PDF at all in 95% of cases.
  The 5% where visual layout matters can use the `backend="docling"`
  override or fall back to direct multimodal Claude.

Auth: requires `ANTHROPIC_API_KEY` in the MCP server env.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from backends.types import ParseResult
from router import RouteResult, route

# Safety ceiling: Claude's input window is large but not infinite, and
# at some point you want to chunk anyway. 200K characters of markdown
# is a soft warning; the call still goes through but the response notes
# truncation pressure.
_TRUNCATE_AT_CHARS = 800_000


@dataclass
class InterpretResult:
    answer: str
    parse: RouteResult
    model: str
    instruction: str
    truncated: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "model": self.model,
            "instruction": self.instruction,
            "truncated": self.truncated,
            "error": self.error,
            "parse": self.parse.to_dict(),
            "metadata": self.metadata,
        }


def interpret(
    data: bytes,
    *,
    filename: str | None,
    instruction: str,
    backend: str | None = None,
    model: str = "claude-sonnet-4-6",
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

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return InterpretResult(
            answer="",
            parse=parse_result,
            model=model,
            instruction=instruction,
            truncated=truncated,
            error="ANTHROPIC_API_KEY not set",
        )

    try:
        import anthropic
    except ImportError:
        return InterpretResult(
            answer="",
            parse=parse_result,
            model=model,
            instruction=instruction,
            truncated=truncated,
            error="anthropic SDK not installed",
        )

    client = anthropic.Anthropic(api_key=api_key)

    system = (
        "You are reading a document that has been pre-parsed to markdown. "
        "Answer the user's instruction strictly from the document content. "
        "If the document does not contain the answer, say so. Do not invent."
    )
    # Cache the parsed markdown — reused across multiple interpret calls
    # against the same doc inside a session for free token reads.
    user_blocks = [
        {
            "type": "text",
            "text": f"<document filename=\"{filename or 'input'}\">\n{markdown}\n</document>",
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": f"Instruction: {instruction}",
        },
    ]

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
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
        return InterpretResult(
            answer=text,
            parse=parse_result,
            model=model,
            instruction=instruction,
            truncated=truncated,
            metadata={"usage": usage_dict},
        )
    except Exception as exc:
        return InterpretResult(
            answer="",
            parse=parse_result,
            model=model,
            instruction=instruction,
            truncated=truncated,
            error=f"{type(exc).__name__}: {exc}",
        )
