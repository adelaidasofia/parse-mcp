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
  engine on scanned PDFs (+0.05 text) and images (+0.04 text) at the time
  MYC-1671 measured this (docling ~2.93). It needs only the ``tesseract``
  binary — no Python/torch OCR dependency. When the binary is absent we
  leave docling's default OCR so the backend still works (graceful, lower
  fidelity). **Update (parse-mcp #29, docling 2.94.0 -> 2.119.0):** docling's
  bundled default engine (RapidOCR, refactored upstream around 2.118.0) has
  since closed most of that gap — re-measured on the same corpus, the
  no-Tesseract path now trails Tesseract CLI by only ~0.006-0.017 text,
  inside the fidelity floor's epsilon. Tesseract CLI is still pinned when
  present (it is still the marginal winner, never worse), but the
  auto-selected fallback is no longer meaningfully "degraded" the way it was
  when this pipeline was first tuned — see ``_FORCE_NO_OCR_ENV`` below for
  why the floor's negative control no longer uses tesseract-absence as its
  detune mechanism.
* **``force_full_page_ocr`` is intentionally NOT enabled.** It OCRs over a
  PDF's native text layer and measurably regresses digital PDFs; the router
  reaches docling for scanned/hard docs via the quality gate instead.
* **Tesseract CLI's OCR ``lang=`` is always the detected installed set, never
  the class default.** ``TesseractCliOcrOptions()``'s bare default is
  ``lang=["eng", "spa", "fra", "deu"]``. docling 2.126.0 tolerated a language
  with no traineddata installed (silently dropped); docling >= 2.127.0 made
  that a hard ``OcrLanguageNotSupportedError`` at pipeline-init, so the bare
  default crashes every parse on any host lacking the non-English packs —
  including CI's ``apt-get install -y tesseract-ocr``, which ships only
  ``eng``. See ``_tesseract_available_langs`` (parse-mcp #51).

The tuned ``DocumentConverter`` loads several models, so it is built once
and reused across calls (the previous code rebuilt it per parse).
"""
from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

from backends.types import ParseResult

NAME = "docling"

logger = logging.getLogger(__name__)

# Built lazily on first parse and reused — model load is expensive.
_converter = None


def is_available() -> bool:
    try:
        import docling  # noqa: F401
    except ImportError:
        return False
    return True


# CI/test-only escape hatch for the parse-fidelity-floor negative control
# (MYC-1793, .github/workflows/parse-fidelity-floor.yml). That control exists
# to prove the floor gate is genuinely OCR-sensitive (anti-vacuity, same
# purpose as the markitdown-vs-docling control). It used to induce a real
# degradation by uninstalling the `tesseract` binary before running this
# backend, relying on docling's fallback engine being meaningfully worse.
# docling 2.94.0 -> 2.119.0 (parse-mcp #29) closed that gap to within the
# floor's epsilon (see module docstring), so tesseract-absence stopped being
# a reliable way to induce a detectable regression — the control started
# failing not because the floor broke, but because the *mechanism* it used to
# simulate "OCR degraded" no longer degrades anything on current docling.
# Forcing OCR off outright is a version-proof invariant (no scanned/image doc
# parses without OCR, regardless of which engine docling defaults to), so the
# workflow sets this instead of just removing the tesseract binary. Never set
# in production; the real parse path never touches this.
_FORCE_NO_OCR_ENV = "PARSE_MCP_DOCLING_FORCE_NO_OCR"


def _ocr_forced_off() -> bool:
    return os.environ.get(_FORCE_NO_OCR_ENV) == "1"


def _ocr_engine() -> str:
    """Which OCR engine the tuned converter pins. See module docstring."""
    if _ocr_forced_off():
        return "disabled"
    return "tesseract" if shutil.which("tesseract") else "auto"


def _warn_if_ocr_degraded() -> bool:
    """Log once if docling will fall back to a lower-fidelity OCR engine.

    The fail-loud companion to the graceful Tesseract fallback in
    ``_build_converter``. Without the ``tesseract`` binary, docling still
    works but on a lower-fidelity engine than the parse-fidelity matrix
    reports — a degraded config must announce itself, not silently lower
    quality. Returns True when degraded.
    """
    if _ocr_engine() == "auto":
        logger.warning(
            "docling OCR running in degraded mode: 'tesseract' binary not found. "
            "Scanned/image fidelity is lower than tests/eval/parse_fidelity_matrix.md "
            "reports; install tesseract for best results (see SETUP.md)."
        )
        return True
    return False


def _tesseract_available_langs(tesseract_cmd: str) -> list[str] | None:
    """Languages the local ``tesseract`` binary actually has traineddata for.

    MYC-4XXX / parse-mcp #51 (docling 2.126.0 -> 2.128.0). Upstream docling
    2.127.0 hardened OCR-language resolution: a requested language with no
    installed traineddata is now a hard ``OcrLanguageNotSupportedError`` at
    pipeline-init time. Docling 2.126.0's ``TesseractOcrCliModel`` only
    recorded which languages ``tesseract --list-langs`` reported and never
    cross-checked the request against it before invoking tesseract, so an
    unavailable language was silently dropped, never a failure. Upstream's own
    docstring for the new behavior: "a language with no model is an error,
    never a silent substitution" -- a deliberate hardening, not a bug.

    ``TesseractCliOcrOptions()``'s class default is ``lang=["eng", "spa",
    "fra", "deu"]`` and this backend never overrode it, so any host whose
    tesseract install has only English -- CI's ``apt-get install -y
    tesseract-ocr`` installs just the ``eng`` + ``osd`` traineddata; the extra
    languages are separate apt packages (``tesseract-ocr-spa`` etc.) -- now
    fails EVERY docling parse the instant OCR is needed: all 16 fixtures
    errored before any measurement (fidelity-floor: 9/9 gated cells MISSING in
    ~5s, previously a ~3min real run). Reproduced in a Linux container
    matching CI exactly (Python 3.13.15, apt tesseract-ocr 5.3.4-1build5):
    docling 2.126.0 passes 9/9, 2.127.0/2.128.0/2.129.0 all raise
    ``OcrLanguageNotSupportedError: ... no model for the OCR language 'spa'``.

    Detecting what is actually installed and requesting exactly that (rather
    than hardcoding ``lang=["eng"]``) keeps the backend version-proof against
    this class of upstream hardening AND keeps OCR genuinely multi-lingual on
    any host that installs more language packs. Returns ``None`` when
    detection itself fails (binary missing/errors/times out) so the caller
    can fall back to a safe explicit default instead of the class default.
    """
    try:
        out = subprocess.run(
            [tesseract_cmd, "--list-langs"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except Exception:
        return None
    # First line is a header ("List of available languages ..."); the rest
    # are one code per line. "osd" is the orientation/script-detection pack,
    # not a real OCR language -- never request it as one.
    langs = [
        line.strip()
        for line in out.stdout.splitlines()[1:]
        if line.strip() and line.strip() != "osd"
    ]
    return langs or None


def _build_converter():
    """Build the fidelity-tuned DocumentConverter (MYC-1671).

    Raises ImportError if docling is not installed; callers guard with
    ``is_available()`` / catch it and surface a clean ParseResult.
    """
    from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
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

    # docling is available here; warn if its OCR will run degraded (no tesseract).
    _warn_if_ocr_degraded()

    opts = PdfPipelineOptions()
    # Test-only: the parse-fidelity-floor negative control forces OCR off
    # entirely via _FORCE_NO_OCR_ENV. See that constant's docstring. This is
    # never set outside CI, so production parsing is unaffected.
    opts.do_ocr = not _ocr_forced_off()
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.table_structure_options.do_cell_matching = True
    # Pin Tesseract CLI when the binary is present (measured fidelity winner);
    # otherwise keep docling's default OCR so a Tesseract-less host still works.
    tesseract_cmd = shutil.which("tesseract")
    if opts.do_ocr and tesseract_cmd:
        # lang= ALWAYS explicit -- never the bare class default. See
        # _tesseract_available_langs's docstring: docling >= 2.127.0 makes
        # TesseractCliOcrOptions()'s default lang list (["eng","spa","fra",
        # "deu"]) a hard requirement, and most hosts (incl. CI's apt install)
        # only have "eng" traineddata, so the bare default crashes every OCR
        # parse. Detect what is actually installed; fall back to ["eng"]
        # (matches the fidelity corpus + this repo's validated baseline) only
        # if detection itself fails.
        langs = _tesseract_available_langs(tesseract_cmd) or ["eng"]
        opts.ocr_options = TesseractCliOcrOptions(lang=langs)

    # Pin the pre-2.123.0 PDF backend explicitly (parse-mcp scanned-table
    # regression). docling PR #3764 ("Default to threaded docling-parse
    # across SDK, CLI, service, and extraction", shipped in 2.123.0) swapped
    # PdfFormatOption's default backend from DoclingParseDocumentBackend
    # (pypdfium2-managed, random-access load_page()) to
    # ThreadedDoclingParseDocumentBackend (streaming iter_pages()). This repo
    # never set `backend=` explicitly, so the docling 2.119.0 -> 2.123.0 bump
    # (parse-mcp PR #43) silently inherited the new threaded default.
    # Measured on the committed parse-fidelity corpus: scanned_pdf/table
    # dropped baseline 0.9886 -> ~0.965-0.970 (drop 0.019-0.024, over the
    # gate's 0.02 epsilon on CI's Ubuntu runner). Only the PDF-backend path
    # regressed — image/* (ImageDocumentBackend, unaffected by this default)
    # and digital/table_heavy PDFs (native text layer, no OCR dependency)
    # held flat — consistent with a scanned-PDF-specific PDF-backend change,
    # not a general TableFormer/OCR-engine regression. Explicitly requesting
    # the old backend restores the committed baseline (see requirements-docling.txt
    # for the pin + eval instructions). Revisit if a future docling release
    # measurably fixes the threaded backend's scanned-PDF fidelity.
    # Same tuned pipeline for born-digital/scanned PDFs and standalone images.
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=opts, backend=DoclingParseDocumentBackend
            ),
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
