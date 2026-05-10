"""Docling backend (table-heavy + scanned escalation).

IBM Docling (open source). Best-in-class for complex tables (97.9%
extraction accuracy on benchmark sustainability reports), layout-aware
PDF parsing, and scanned documents that need OCR. Trade-off: downloads
several model weights on first run (slower cold start, requires disk).

Install: `pip install docling`. Without it, `is_available()` returns
False and the router skips this backend.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

from backends.types import ParseResult

NAME = "docling"


def is_available() -> bool:
    try:
        import docling  # noqa: F401
    except ImportError:
        return False
    return True


def parse(data: bytes, *, filename: str | None = None, hints: dict | None = None) -> ParseResult:
    bytes_in = len(data)
    fmt = (Path(filename).suffix.lstrip(".").lower() if filename else "") or "unknown"

    if bytes_in == 0:
        return ParseResult(markdown="", backend=NAME, format=fmt, bytes_in=0, error="empty input")

    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        return ParseResult(
            markdown="",
            backend=NAME,
            format=fmt,
            bytes_in=bytes_in,
            error="docling not installed",
        )

    start = time.monotonic()
    try:
        converter = DocumentConverter()
        # Docling's converter accepts paths, byte streams via DocumentStream.
        from docling.datamodel.base_models import DocumentStream

        stream = DocumentStream(name=filename or "input", stream=io.BytesIO(data))
        result = converter.convert(stream)
        # Docling exports the parsed document via to_markdown / export_to_markdown.
        doc = result.document
        if hasattr(doc, "export_to_markdown"):
            text = doc.export_to_markdown()
        elif hasattr(doc, "to_markdown"):
            text = doc.to_markdown()
        else:
            text = str(doc)
        latency_ms = int((time.monotonic() - start) * 1000)

        # Surface table count when available; the router uses this to
        # justify the docling escalation in the audit trail.
        try:
            tables = getattr(doc, "tables", None)
            table_count = len(tables) if tables is not None else None
        except Exception:
            table_count = None

        return ParseResult(
            markdown=text or "",
            backend=NAME,
            format=fmt,
            bytes_in=bytes_in,
            latency_ms=latency_ms,
            metadata={
                "text_length": len(text or ""),
                "table_count": table_count,
            },
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
