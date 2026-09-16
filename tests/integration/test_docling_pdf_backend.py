"""The PDF backend is pinned explicitly, never inherited from docling's default.

docling 2.123.0 (upstream PR #3764, "Default to threaded docling-parse across
SDK, CLI, service, and extraction") swapped ``PdfFormatOption``'s default
backend from ``DoclingParseDocumentBackend`` (pypdfium2-managed, random-access
``load_page()``) to ``ThreadedDoclingParseDocumentBackend`` (streaming
``iter_pages()``). ``backends/docling_backend.py`` did not set ``backend=``,
so the 2.119.0 -> 2.123.0 bump silently inherited the new default and
scanned_pdf/table fell from the committed 0.9886 baseline to 0.9645 -- under
the floor (MYC-4802).

The parse-fidelity floor caught that, but only as a score delta after a full
~3min scoring run, leaving the CAUSE to be inferred. These tests pin the
contract itself, so the next upstream default change names itself in seconds.

Teeth, stated honestly: on docling < 2.123.0 ``DoclingParseDocumentBackend``
is ALREADY the upstream default, so these assertions pass with or without the
explicit pin. They only bite on >= 2.123.0 -- which is exactly what CI
installs (``requirements-docling.txt``), and where the regression lives.
"""
from __future__ import annotations

import pytest

pytest.importorskip("docling", reason="optional backend; pinned in requirements-docling.txt")

from docling.backend.docling_parse_backend import (  # noqa: E402
    DoclingParseDocumentBackend,
)
from docling.datamodel.base_models import InputFormat  # noqa: E402

from backends import docling_backend  # noqa: E402


@pytest.fixture(scope="module")
def format_options():
    return docling_backend._build_converter().format_to_options


def test_pdf_backend_is_the_pinned_docling_parse_backend(format_options):
    # The regression: without an explicit backend=, docling >= 2.123.0 hands
    # back ThreadedDoclingParseDocumentBackend here.
    assert format_options[InputFormat.PDF].backend is DoclingParseDocumentBackend


def test_pdf_backend_is_not_the_threaded_default(format_options):
    # Named separately so a failure reads as "we inherited the threaded
    # default" rather than a bare identity mismatch.
    backend_name = format_options[InputFormat.PDF].backend.__name__
    assert "Threaded" not in backend_name, (
        f"PDF backend is {backend_name!r} -- the explicit pin in "
        "backends/docling_backend.py was lost, so docling's threaded default "
        "was inherited. This regresses scanned_pdf/table below the fidelity "
        "floor (MYC-4802)."
    )


def test_image_backend_is_left_at_its_own_default(format_options):
    # The image path was NOT affected by upstream PR #3764 and is deliberately
    # left alone; pinning it here would be cargo-culting the PDF fix.
    assert format_options[InputFormat.IMAGE].backend.__name__ == "ImageDocumentBackend"
