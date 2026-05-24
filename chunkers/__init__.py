"""Per-doc-type chunkers — pick the chunker by document shape.

Single document parsed to markdown via markitdown/docling/llamaparse arrives
flat. The chunker stage turns that flat markdown into retrieval-ready
chunks. Different document shapes want different chunking strategies:

- **paper**: keep abstract whole; section-aware splits; preserve citations
- **book**: drop the TOC; merge by chapter depth (default 5)
- **manual**: section-ID-aware merge; never cross-section join
- **qa**: pair questions with following answer blocks
- **resume**: domain-section detection (Experience, Education, Skills)
- **table**: each row a chunk, with column-role hints
- **default**: paragraph-based uniform chunker (safe fallback)

The dispatcher picks the right chunker from `doc_type`. When `doc_type="auto"`,
`detect_doc_type()` runs structural heuristics over the markdown first.

Source pattern: infiniflow/ragflow rag/app/* (Apache-2.0). Clean Python
reimplementation matching the doc-type-specific chunking primitive — no
license contamination, smaller surface (the upstream chunkers carry years
of incidental ETL logic specific to ragflow's vector store).
"""

from __future__ import annotations

from chunkers.base import Chunk, ChunkConfig, Chunker
from chunkers.detect import detect_doc_type
from chunkers.dispatcher import chunk_text, get_chunker

__all__ = [
    "Chunk",
    "ChunkConfig",
    "Chunker",
    "chunk_text",
    "detect_doc_type",
    "get_chunker",
]
