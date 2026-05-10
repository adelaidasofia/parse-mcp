"""markitdown backend (default).

Microsoft markitdown (MIT). Fast, deterministic, free. Best for clean
structured docs: most PDFs, DOCX, PPTX, XLSX, HTML, CSV, JSON, XML,
EPub, ZIP. Weaker on complex tables, heavily-scanned PDFs, and visually
exotic layouts (escalate to Docling or LlamaParse for those).
"""
from __future__ import annotations

import io
import time
from pathlib import Path

from backends.types import ParseResult

NAME = "markitdown"


def is_available() -> bool:
    try:
        import markitdown  # noqa: F401
    except ImportError:
        return False
    return True


def parse(data: bytes, *, filename: str | None = None, hints: dict | None = None) -> ParseResult:
    bytes_in = len(data)
    fmt = (Path(filename).suffix.lstrip(".").lower() if filename else "") or "unknown"

    if bytes_in == 0:
        return ParseResult(
            markdown="", backend=NAME, format=fmt, bytes_in=0, error="empty input"
        )

    try:
        from markitdown import MarkItDown
    except ImportError:
        return ParseResult(
            markdown="",
            backend=NAME,
            format=fmt,
            bytes_in=bytes_in,
            error="markitdown not installed",
        )

    start = time.monotonic()
    try:
        converter = MarkItDown()
        stream = io.BytesIO(data)
        result = converter.convert_stream(
            stream,
            file_extension=Path(filename).suffix if filename else None,
        )
        text = getattr(result, "text_content", "") or ""
        latency_ms = int((time.monotonic() - start) * 1000)
        return ParseResult(
            markdown=text,
            backend=NAME,
            format=fmt,
            bytes_in=bytes_in,
            latency_ms=latency_ms,
            metadata={"text_length": len(text)},
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
