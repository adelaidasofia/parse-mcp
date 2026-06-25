"""Router escalation is QUALITY-gated, not just empty/error-gated (MYC-1671).

The router used to stop at the first backend whose markdown was non-empty and
error-free (the stop-on-first-success rule in ``router._run_chain``). A scanned
PDF that markitdown turns into *garbage-but-non-empty* therefore terminated the
chain, and the low-fidelity output was served silently — never escalating to
docling. These tests pin the new contract through the public ``route()``
interface:

* escalate past a non-empty-but-low-fidelity result, AND
* never escalate past a genuinely good one (no needless docling cost), AND
* keep trusting small clean docs (a short note is not "low fidelity").

Backends are stubbed at the module boundary (the seam where each backend wraps
its external library), so the tests are deterministic and need no real parser.
"""
from __future__ import annotations

import router
from backends import docling_backend, llamaparse_backend, markitdown_backend
from backends.types import ParseResult


def _stub(monkeypatch, mod, *, markdown, error=None, available=True):
    """Make ``mod`` a fake backend that returns a fixed ParseResult."""
    monkeypatch.setattr(mod, "is_available", lambda: available)

    def _parse(data, *, filename=None, hints=None):
        return ParseResult(
            markdown=markdown,
            backend=mod.NAME,
            format="pdf",
            bytes_in=len(data),
            latency_ms=1,
            error=error,
        )

    monkeypatch.setattr(mod, "parse", _parse)


# A scanned-PDF-sized input: large enough that near-zero text is suspicious.
SCANNED_BYTES = b"%PDF-1.4\n" + b"\x00" * 60_000
# What docling recovers: real, dense text.
RICH = "# Internal Operations Memo\n\n" + ("Real recovered sentence. " * 300)
# What markitdown emits on a scanned page with a junk text layer: a stray
# token. Non-empty and error-free, so the OLD gate accepted it.
GARBAGE = "f"


def test_escalates_past_nonempty_garbage(monkeypatch):
    _stub(monkeypatch, markitdown_backend, markdown=GARBAGE)
    _stub(monkeypatch, docling_backend, markdown=RICH)
    _stub(monkeypatch, llamaparse_backend, markdown="", available=False)

    res = router.route(SCANNED_BYTES, filename="scan.pdf")

    assert res.final.backend == "docling"  # escalated past the garbage
    assert res.final.markdown == RICH
    # Audit trail keeps every attempt, in order.
    assert [r.backend for r in res.chain] == ["markitdown", "docling"]


def test_does_not_escalate_past_good_result(monkeypatch):
    # markitdown yields rich text even on a large doc -> trust it, no docling.
    _stub(monkeypatch, markitdown_backend, markdown=RICH)
    _stub(monkeypatch, docling_backend, markdown="SHOULD NOT BE REACHED")
    _stub(monkeypatch, llamaparse_backend, markdown="", available=False)

    res = router.route(SCANNED_BYTES, filename="report.pdf")

    assert res.final.backend == "markitdown"
    assert [r.backend for r in res.chain] == ["markitdown"]


def test_small_clean_doc_is_trusted(monkeypatch):
    # A tiny input with a short-but-complete extraction has high text density;
    # it must NOT be escalated just for being short.
    _stub(monkeypatch, markitdown_backend, markdown="# Note\n\nShort but complete.")
    _stub(monkeypatch, docling_backend, markdown="SHOULD NOT BE REACHED")
    _stub(monkeypatch, llamaparse_backend, markdown="", available=False)

    res = router.route(b"%PDF-1.4 tiny doc", filename="tiny.pdf")

    assert res.final.backend == "markitdown"
    assert [r.backend for r in res.chain] == ["markitdown"]
