"""Docling OCR engine resolution + fail-loud degraded-mode warning.

The docling backend pins Tesseract CLI when the binary is present (the
measured fidelity winner) and falls back to docling's default OCR when it is
absent. A silent fallback would lower scanned/image fidelity below what the
parse-fidelity matrix reports without anyone knowing, so the backend warns.
These tests pin both halves of that contract and need no docling install
(the helpers use only ``shutil`` + ``logging``).
"""
from __future__ import annotations

import logging
import shutil

from backends import docling_backend


def test_ocr_engine_reports_tesseract_when_binary_present(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/tesseract")
    assert docling_backend._ocr_engine() == "tesseract"


def test_ocr_engine_reports_auto_when_binary_absent(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert docling_backend._ocr_engine() == "auto"


def test_warns_when_tesseract_absent(monkeypatch, caplog):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with caplog.at_level(logging.WARNING, logger="backends.docling_backend"):
        degraded = docling_backend._warn_if_ocr_degraded()
    assert degraded is True
    assert any(
        "degraded mode" in r.message and "tesseract" in r.message
        for r in caplog.records
    )


def test_no_warning_when_tesseract_present(monkeypatch, caplog):
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/tesseract")
    with caplog.at_level(logging.WARNING, logger="backends.docling_backend"):
        degraded = docling_backend._warn_if_ocr_degraded()
    assert degraded is False
    assert caplog.records == []
