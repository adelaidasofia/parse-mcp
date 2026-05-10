"""Pluggable parse backends.

Every backend exposes the same surface:

    parse(data: bytes, *, filename: str | None, hints: dict) -> ParseResult
    is_available() -> bool
    NAME: str   # canonical token e.g. "markitdown" / "docling" / "llamaparse"

Backends never raise on parse failure. They return ParseResult with
`error` set so the router can fall back to the next backend in the
chain. Missing-dependency is reported by `is_available() == False`,
which the router uses to skip the backend entirely.
"""
from __future__ import annotations

from backends.types import ParseResult

__all__ = ["ParseResult"]
