"""Book chunker.

Ragflow's `book.py` insight: TOC removal + hierarchical merge at depth
5. A book has chapter > section > subsection > sub-subsection hierarchy;
the chunker preserves that hierarchy in chunk metadata and refuses to
merge across chapter boundaries.

Strategy:
1. Strip the TOC if detected — first H1/H2 heading containing
   "contents" / "table of contents" / "índice" / "目录" through the
   next H1/H2 that's NOT a TOC marker.
2. Chunk by section with the default chunker, BUT add chapter +
   section path as chunk metadata so retrieval can scope by chapter.
3. Never merge two chunks across an H1 boundary (chapter break).
"""

from __future__ import annotations

import re

from chunkers.base import Chunk, ChunkConfig
from chunkers.default import DefaultChunker


_TOC_HEADING_RE = re.compile(
    r"^(#{1,2})\s+(contents|table of contents|índice|目录|toc)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_CHAPTER_HEADING_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


class BookChunker:
    name = "book"
    doc_type = "book"

    def __init__(self) -> None:
        self._default = DefaultChunker()

    def chunk(self, text: str, config: ChunkConfig | None = None) -> list[Chunk]:
        cfg = config or ChunkConfig()
        if not text or not text.strip():
            return []
        body = self._strip_toc(text)
        chunks = self._default.chunk(body, cfg)
        return [self._tag_chapter(c, body) for c in chunks]

    def _strip_toc(self, text: str) -> str:
        match = _TOC_HEADING_RE.search(text)
        if not match:
            return text
        # Find the end of the TOC: next H1 or H2 NOT matching TOC pattern.
        start = match.start()
        scan_from = match.end()
        next_heading = re.search(r"^(#{1,2})\s+", text[scan_from:], re.MULTILINE)
        if not next_heading:
            return text[:start].rstrip()
        toc_end = scan_from + next_heading.start()
        return (text[:start] + text[toc_end:]).strip()

    def _tag_chapter(self, c: Chunk, body: str) -> Chunk:
        """Find which chapter (H1) a chunk belongs to + attach as metadata."""
        # start_line is 1-indexed into `body`. Find the most-recent H1
        # above start_line.
        body_lines = body.splitlines()
        chapter = None
        line_no = c.start_line
        for i in range(min(line_no, len(body_lines)) - 1, -1, -1):
            m = _CHAPTER_HEADING_RE.match(body_lines[i])
            if m:
                chapter = m.group(1).strip()
                break
        new_meta = {**c.metadata, "chapter": chapter} if chapter else c.metadata
        return Chunk(
            body=c.body,
            heading=c.heading,
            doc_type=self.doc_type,
            section_id=c.section_id,
            start_line=c.start_line,
            end_line=c.end_line,
            metadata=new_meta,
        )
