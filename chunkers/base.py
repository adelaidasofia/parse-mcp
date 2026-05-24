"""Base chunker types.

Every chunker returns a list of Chunk objects. Each Chunk carries:
- body: the text content
- heading: the most-immediate enclosing heading (or None)
- doc_type: the detected/specified document type
- section_id: stable identifier for the section the chunk belongs to (or None)
- start_line / end_line: line range in the source markdown (1-indexed, inclusive)
- metadata: per-chunker extras (chunker_specific data)

The Chunker protocol has one method: chunk(text, config) -> list[Chunk].
Implementations vary in how they slice, but the output shape is uniform so
downstream retrieval surfaces (memory-runtime-pro indexer) get a stable contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Chunk:
    """A single retrievable unit of text + metadata."""

    body: str
    heading: str | None
    doc_type: str
    section_id: str | None = None
    start_line: int = 1
    end_line: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "body": self.body,
            "heading": self.heading,
            "doc_type": self.doc_type,
            "section_id": self.section_id,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ChunkConfig:
    """Tunable parameters every chunker honors (best-effort).

    Chunkers that override defaults document the reason in their module.

    - target_tokens: target chunk size in whitespace-tokens. Best-effort:
      a section shorter than target stays whole; longer is split at
      paragraph boundaries first, then sentence boundaries.
    - max_tokens: hard cap. Above this a chunk is force-split even
      mid-paragraph.
    - min_tokens: don't emit a chunk shorter than this UNLESS it's the
      only chunk for its section.
    - overlap_tokens: trailing N tokens from chunk K-1 prepended to
      chunk K. 0 = no overlap.
    - keep_headings: emit the heading at the top of every chunk it
      belongs to (default True for retrieval — headings are signal).
    """

    target_tokens: int = 400
    max_tokens: int = 800
    min_tokens: int = 50
    overlap_tokens: int = 0
    keep_headings: bool = True


class Chunker(Protocol):
    """Chunker contract: text + config → ordered list of chunks."""

    name: str
    doc_type: str

    def chunk(self, text: str, config: ChunkConfig | None = None) -> list[Chunk]:
        ...  # pragma: no cover — protocol


# ---------------------------------------------------------------------------
# Shared helpers used by multiple concrete chunkers
# ---------------------------------------------------------------------------


def count_tokens(text: str) -> int:
    """Cheap whitespace-token count. NOT a real tokenizer; "good enough"
    for chunk-size budgeting at the chunker layer. Downstream embedders
    do their own tokenization.
    """
    return len(text.split())


def split_paragraphs(text: str) -> list[tuple[int, str]]:
    """Split into (start_line_1indexed, paragraph) tuples.

    A paragraph is one or more consecutive non-empty lines separated by
    a blank line. Lines with only whitespace count as blank.
    """
    out: list[tuple[int, str]] = []
    current: list[str] = []
    start = 1
    line_no = 0
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if raw.strip() == "":
            if current:
                out.append((start, "\n".join(current)))
                current = []
        else:
            if not current:
                start = line_no
            current.append(raw)
    if current:
        out.append((start, "\n".join(current)))
    return out
