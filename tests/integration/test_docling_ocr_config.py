"""Docling OCR engine resolution + fail-loud degraded-mode warning.

The docling backend pins Tesseract CLI when the binary is present (the
measured fidelity winner) and falls back to docling's default OCR when it is
absent. A silent fallback would lower scanned/image fidelity below what the
parse-fidelity matrix reports without anyone knowing, so the backend warns.
These tests pin both halves of that contract and need no docling install
(the helpers use only ``shutil`` + ``logging``).

Also covers ``PARSE_MCP_DOCLING_FORCE_NO_OCR`` (parse-mcp #29): the
parse-fidelity-floor negative control's version-proof detune knob, added
after docling 2.94.0 -> 2.119.0 closed the fidelity gap between Tesseract CLI
and docling's bundled default OCR engine to within the floor's epsilon,
making tesseract-absence alone too weak a regression to prove the floor gate
is still OCR-sensitive. See ``docling_backend._FORCE_NO_OCR_ENV``.
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


def test_ocr_not_forced_off_by_default(monkeypatch):
    monkeypatch.delenv(docling_backend._FORCE_NO_OCR_ENV, raising=False)
    assert docling_backend._ocr_forced_off() is False


def test_ocr_forced_off_when_env_set(monkeypatch):
    monkeypatch.setenv(docling_backend._FORCE_NO_OCR_ENV, "1")
    assert docling_backend._ocr_forced_off() is True


def test_ocr_engine_reports_disabled_when_forced_off_even_with_tesseract(monkeypatch):
    # The force-off knob must win regardless of tesseract's presence -- it is
    # the version-proof detune, not a tesseract-absence proxy.
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/tesseract")
    monkeypatch.setenv(docling_backend._FORCE_NO_OCR_ENV, "1")
    assert docling_backend._ocr_engine() == "disabled"


def test_no_degraded_warning_when_ocr_deliberately_forced_off(monkeypatch, caplog):
    # Forcing OCR off is a deliberate CI-only probe, not an accidental
    # tesseract-missing degradation -- it must not trip the "degraded mode"
    # warning meant for the real graceful-fallback path.
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setenv(docling_backend._FORCE_NO_OCR_ENV, "1")
    with caplog.at_level(logging.WARNING, logger="backends.docling_backend"):
        degraded = docling_backend._warn_if_ocr_degraded()
    assert degraded is False
    assert caplog.records == []


def test_ocr_force_off_env_requires_exact_value(monkeypatch):
    # Only the literal "1" arms it -- an accidental truthy-looking value
    # ("true", "yes") must NOT silently disable OCR in a real deployment.
    monkeypatch.setenv(docling_backend._FORCE_NO_OCR_ENV, "true")
    assert docling_backend._ocr_forced_off() is False
