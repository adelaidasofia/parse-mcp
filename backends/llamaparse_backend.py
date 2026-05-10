"""LlamaParse backend (cloud, BYOK).

LlamaIndex LlamaParse (cloud SaaS). Cleanest output on visually-complex
PDFs (multi-column layouts, embedded images, dense tables). Cloud means
no model weight downloads, no GPU requirement, ~6 seconds regardless of
document size. Costs API credits per page.

Auth: requires `LLAMA_CLOUD_API_KEY` in the MCP server's env. Without
the key, `is_available()` returns False and the router skips this
backend.

Install: `pip install llama-cloud-services`.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from backends.types import ParseResult

NAME = "llamaparse"


def is_available() -> bool:
    if not os.environ.get("LLAMA_CLOUD_API_KEY"):
        return False
    try:
        from llama_cloud_services import LlamaParse  # noqa: F401
    except ImportError:
        return False
    return True


def parse(data: bytes, *, filename: str | None = None, hints: dict | None = None) -> ParseResult:
    bytes_in = len(data)
    fmt = (Path(filename).suffix.lstrip(".").lower() if filename else "") or "unknown"

    if bytes_in == 0:
        return ParseResult(markdown="", backend=NAME, format=fmt, bytes_in=0, error="empty input")

    api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
    if not api_key:
        return ParseResult(
            markdown="",
            backend=NAME,
            format=fmt,
            bytes_in=bytes_in,
            error="LLAMA_CLOUD_API_KEY not set",
        )

    try:
        from llama_cloud_services import LlamaParse
    except ImportError:
        return ParseResult(
            markdown="",
            backend=NAME,
            format=fmt,
            bytes_in=bytes_in,
            error="llama-cloud-services not installed",
        )

    start = time.monotonic()
    try:
        # LlamaParse takes a file path. Materialize the bytes first.
        suffix = Path(filename).suffix if filename else ""
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            parser = LlamaParse(api_key=api_key, result_type="markdown")
            documents = parser.load_data(tmp_path)
            text = "\n\n".join(getattr(d, "text", "") or "" for d in documents)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        latency_ms = int((time.monotonic() - start) * 1000)
        return ParseResult(
            markdown=text,
            backend=NAME,
            format=fmt,
            bytes_in=bytes_in,
            latency_ms=latency_ms,
            metadata={"text_length": len(text), "doc_count": len(documents)},
        )
    except Exception as exc:
        latency_ms = int((time.monotonic() - start) * 1000)
        return ParseResult(
            markdown="",
            backend=NAME,
            format=fmt,
            bytes_in=bytes_in,
            latency_ms=latency_ms,
            error=f"{type(exc).__name__}: {exc}",
        )
