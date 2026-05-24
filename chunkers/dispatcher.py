"""Chunker dispatcher — pick a chunker by doc_type, run it, return chunks.

One public function: `chunk_text(text, doc_type="auto", config=None)`.

Auto-routing flow:
1. If `doc_type="auto"` (default) → call `detect_doc_type(text)`.
2. Look up the matching chunker in `_REGISTRY`.
3. Run the chunker, return chunks + the resolved doc_type.

The registry maps name → chunker instance. Add a new chunker by
registering it here; the auto-detect heuristic in `detect.py` handles
the routing.
"""

from __future__ import annotations

from chunkers.base import Chunk, ChunkConfig, Chunker
from chunkers.book import BookChunker
from chunkers.default import DefaultChunker
from chunkers.detect import detect_doc_type
from chunkers.manual import ManualChunker
from chunkers.paper import PaperChunker
from chunkers.qa import QAChunker
from chunkers.resume import ResumeChunker
from chunkers.table import TableChunker


_REGISTRY: dict[str, Chunker] = {
    "default": DefaultChunker(),
    "paper": PaperChunker(),
    "book": BookChunker(),
    "manual": ManualChunker(),
    "qa": QAChunker(),
    "resume": ResumeChunker(),
    "table": TableChunker(),
}


def get_chunker(doc_type: str) -> Chunker:
    """Return a chunker for `doc_type`. Unknown types → DefaultChunker."""
    return _REGISTRY.get(doc_type, _REGISTRY["default"])


def chunk_text(
    text: str,
    *,
    doc_type: str = "auto",
    config: ChunkConfig | None = None,
) -> tuple[list[Chunk], str]:
    """Chunk `text` with the doc-type-appropriate chunker.

    `doc_type="auto"` (default) runs `detect_doc_type()` first.

    Returns (chunks, resolved_doc_type). The caller can inspect
    `resolved_doc_type` to see which chunker actually ran — useful for
    diagnostics and for surfaces that want to display "detected as: paper".
    """
    if not text or not text.strip():
        return [], doc_type if doc_type != "auto" else "default"
    if doc_type == "auto":
        resolved = detect_doc_type(text)
    else:
        resolved = doc_type
    chunker = get_chunker(resolved)
    return chunker.chunk(text, config), resolved


def list_doc_types() -> list[str]:
    """Names of all registered chunkers (for diagnostics)."""
    return sorted(_REGISTRY.keys())
