"""Docling backend (table-heavy + scanned escalation).

IBM Docling (open source). Best-in-class for complex tables (97.9%
extraction accuracy on benchmark sustainability reports), layout-aware
PDF parsing, and scanned documents that need OCR. Trade-off: downloads
several model weights on first run (slower cold start, requires disk).

Install: `pip install docling`. Without it, `is_available()` returns
False and the router skips this backend.

Pipeline tuning (MYC-1671). The converter is built with explicit
``PdfPipelineOptions`` rather than left at library defaults:

* **TableFormer ACCURATE + cell matching** — pinned explicitly. docling
  2.93.0 already defaults to these, but pinning means a future default
  change can't silently downgrade table fidelity.
* **OCR engine = Tesseract CLI when available.** On the parse-fidelity
  corpus (``tests/eval/``), Tesseract CLI beat docling's auto-selected
  engine on scanned PDFs (+0.05 text) and images (+0.04 text), with no
  regression on digital docs. It needs only the ``tesseract`` binary — no
  Python/torch OCR dependency. When the binary is absent we leave docling's
  default OCR so the backend still works (graceful, just lower fidelity).
* **``force_full_page_ocr`` is intentionally NOT enabled.** It OCRs over a
  PDF's native text layer and measurably regresses digital PDFs; the router
  reaches docling for scanned/hard docs via the quality gate instead.

The tuned ``DocumentConverter`` loads several models, so it is built once
and reused across calls (the previous code rebuilt it per parse).
"""
from __future__ import annotations

import io
import shutil
import time
from pathlib import Path

from backends.types import ParseResult

NAME = "docling"

# Built lazily on first parse and reused — model load is expensive.
_converter = None


def is_available() -> bool:
    try:
        import docling  # noqa: F401
    except ImportError:
        return False
    return True


def _ocr_engine() -> str:
    """Which OCR engine the tuned converter pins. See module docstring."""
    return "tesseract" if shutil.which("tesseract") else "auto"


def _build_converter():
    """Build the fidelity-tuned DocumentConverter (MYC-1671).

    Raises ImportError if docling is not installed; callers guard with
    ``is_available()`` / catch it and surface a clean ParseResult.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
        TesseractCliOcrOptions,
    )
    from docling.document_converter import (
        DocumentConverter,
        ImageFormatOption,
        PdfFormatOption,
    )

    opts = PdfPipelineOptions()
    opts.do_ocr = True
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.table_structure_options.do_cell_matching = True
    # Pin Tesseract CLI when the binary is present (measured fidelity winner);
    # otherwise keep docling's default OCR so a Tesseract-less host still works.
    if shutil.which("tesseract"):
        opts.ocr_options = TesseractCliOcrOptions()

    # Same tuned pipeline for born-digital/scanned PDFs and standalone images.
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts),
        }
    )


def _get_converter():
    global _converter
    if _converter is None:
        _converter = _build_converter()
    return _converter


def parse(data: bytes, *, filename: str | None = None, hints: dict | None = None) -> ParseResult:
    bytes_in = len(data)
    fmt = (Path(filename).suffix.lstrip(".").lower() if filename else "") or "unknown"

    if bytes_in == 0:
        return ParseResult(markdown="", backend=NAME, format=fmt, bytes_in=0, error="empty input")

    start = time.monotonic()
    try:
        from docling.datamodel.base_models import DocumentStream

        converter = _get_converter()
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
                # Record the pipeline knobs so the audit trail shows what ran.
                "ocr_engine": _ocr_engine(),
                "table_mode": "accurate",
            },
        )
    except ImportError:
        return ParseResult(
            markdown="",
            backend=NAME,
            format=fmt,
            bytes_in=bytes_in,
            error="docling not installed",
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
