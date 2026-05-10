"""Shared types for parse backends."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParseResult:
    """Outcome of a single backend call.

    - `markdown` is empty string on failure; never None.
    - `error` is None on success.
    - `format` is the canonical token for the input ("pdf", "docx", etc).
    - `backend` is the name of the backend that produced this result.
    - `latency_ms` and `bytes_in` populate the audit trail.
    - `metadata` carries backend-specific extras (page count, table count,
      detected_layout, model_used, etc).
    """

    markdown: str
    backend: str
    format: str = "unknown"
    bytes_in: int = 0
    latency_ms: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "markdown": self.markdown,
            "backend": self.backend,
            "format": self.format,
            "bytes_in": self.bytes_in,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "metadata": self.metadata,
        }
