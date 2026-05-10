"""Router: pick a backend per document, fall back on failure.

The router's job is to give callers ONE answer regardless of which
backend produced it, while preserving an audit trail of what ran, in
what order, and why.

Routing strategy (in order):

1. **Honor explicit override.** If the caller passes `backend="docling"`
   (or any specific name), try that backend only. No fallback. The
   caller is asking for diagnostic behavior, not best-of.
2. **Format-driven preference.** Some formats route differently from
   the markitdown default (see `_FORMAT_PREFERENCE`).
3. **Fallback chain.** If the preferred backend errors or returns empty
   markdown, try the next backend. Skip backends whose `is_available()`
   reports False so missing optional deps don't show as failures.
4. **Stop-on-first-success.** Return as soon as a backend produces
   non-empty markdown with no error.

The audit trail accumulates every attempt (success and failure) under
the result's `chain` field so the operator can see why an escalation
fired and how long each step took.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from backends import markitdown_backend, docling_backend, llamaparse_backend
from backends.types import ParseResult

# Registry: name to module. Order matters — earlier entries are checked
# first when a backend isn't explicitly requested. Markitdown leads
# because it's free, fast, and handles 80%+ of inputs cleanly.
_REGISTRY: dict[str, Any] = {
    markitdown_backend.NAME: markitdown_backend,
    docling_backend.NAME: docling_backend,
    llamaparse_backend.NAME: llamaparse_backend,
}

# Formats where markitdown's quality drops noticeably and Docling/LlamaParse
# tend to win. The router still attempts markitdown first (it's free and
# usually adequate); these formats just escalate sooner on empty/error.
_FORMAT_PREFERENCE: dict[str, list[str]] = {
    # Image inputs land here (markitdown does EXIF-only by default).
    "png": ["docling", "markitdown"],
    "jpg": ["docling", "markitdown"],
    "jpeg": ["docling", "markitdown"],
    "tiff": ["docling", "markitdown"],
    # Heavy-table formats: try markitdown first, escalate to docling, then llamaparse.
    "pdf": ["markitdown", "docling", "llamaparse"],
    "xlsx": ["markitdown", "docling"],
    "xls": ["markitdown", "docling"],
}

_DEFAULT_CHAIN = ["markitdown", "docling", "llamaparse"]


@dataclass
class RouteResult:
    """Result of a routed parse: the winning ParseResult plus full audit."""

    final: ParseResult
    chain: list[ParseResult] = field(default_factory=list)
    chosen_strategy: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "markdown": self.final.markdown,
            "backend": self.final.backend,
            "format": self.final.format,
            "bytes_in": self.final.bytes_in,
            "latency_ms": self.final.latency_ms,
            "error": self.final.error,
            "metadata": self.final.metadata,
            "chosen_strategy": self.chosen_strategy,
            "chain": [r.to_dict() for r in self.chain],
        }


def list_backends() -> list[dict[str, Any]]:
    """Report which backends are available in the current env.

    Used by the `list_backends` MCP tool so the operator can see whether
    Docling weights have been pulled and whether LLAMA_CLOUD_API_KEY is
    set without trying to parse a file first.
    """
    out = []
    for name, mod in _REGISTRY.items():
        out.append(
            {
                "name": name,
                "available": mod.is_available(),
                "module": mod.__name__,
            }
        )
    return out


def route(
    data: bytes,
    *,
    filename: str | None = None,
    backend: str | None = None,
    hints: dict | None = None,
) -> RouteResult:
    """Pick a backend, run it, fall back on empty/error.

    `backend`: when set, force that backend only (no fallback). The
    diagnostic mode for the `benchmark` tool and for clients debugging
    a specific parser.
    """
    hints = hints or {}

    if backend is not None:
        if backend not in _REGISTRY:
            err = ParseResult(
                markdown="",
                backend=backend,
                format=_format_of(filename),
                bytes_in=len(data),
                error=f"unknown backend: {backend}",
            )
            return RouteResult(final=err, chain=[err], chosen_strategy="explicit")
        result = _REGISTRY[backend].parse(data, filename=filename, hints=hints)
        return RouteResult(final=result, chain=[result], chosen_strategy="explicit")

    fmt = _format_of(filename)
    chain_names = _FORMAT_PREFERENCE.get(fmt, _DEFAULT_CHAIN)
    return _run_chain(data, filename, hints, chain_names)


def _run_chain(
    data: bytes,
    filename: str | None,
    hints: dict,
    chain_names: list[str],
) -> RouteResult:
    chain: list[ParseResult] = []
    last: ParseResult | None = None
    for name in chain_names:
        mod = _REGISTRY.get(name)
        if mod is None:
            continue
        if not mod.is_available():
            chain.append(
                ParseResult(
                    markdown="",
                    backend=name,
                    format=_format_of(filename),
                    bytes_in=len(data),
                    error=f"{name} not available",
                )
            )
            continue
        result = mod.parse(data, filename=filename, hints=hints)
        chain.append(result)
        last = result
        if result.error is None and result.markdown.strip():
            return RouteResult(final=result, chain=chain, chosen_strategy="default")

    if last is None:
        last = ParseResult(
            markdown="",
            backend="none",
            format=_format_of(filename),
            bytes_in=len(data),
            error="no backend available",
        )
        chain.append(last)
    return RouteResult(final=last, chain=chain, chosen_strategy="default")


def benchmark(
    data: bytes, *, filename: str | None = None, hints: dict | None = None
) -> list[ParseResult]:
    """Run every available backend on the same input. Diagnostic only.

    Returns a list of ParseResult, one per backend that's available.
    Skipped (unavailable) backends are not included in the result; check
    `list_backends()` to see them.
    """
    hints = hints or {}
    out: list[ParseResult] = []
    for name, mod in _REGISTRY.items():
        if not mod.is_available():
            continue
        start = time.monotonic()
        try:
            r = mod.parse(data, filename=filename, hints=hints)
        except Exception as exc:
            r = ParseResult(
                markdown="",
                backend=name,
                format=_format_of(filename),
                bytes_in=len(data),
                latency_ms=int((time.monotonic() - start) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )
        out.append(r)
    return out


def _format_of(filename: str | None) -> str:
    if not filename:
        return "unknown"
    from pathlib import Path

    return Path(filename).suffix.lstrip(".").lower() or "unknown"
