"""Academic paper chunker.

Ragflow's `paper.py` insight: "The abstract of the paper will be sliced
as an entire chunk, and will not be sliced partly." (verbatim).

Strategy:
1. Detect the abstract section — first heading matching `(abstract|摘要|summary|tl;dr)` is treated as the abstract.
2. Emit the entire abstract as one chunk (never split even if oversize).
3. Detect references section — anything below `References` / `Bibliography` /
   `Works Cited` is emitted as ONE chunk regardless of size (preserves
   citation continuity for downstream retrieval).
4. Everything between gets the default uniform chunker treatment.

A paper is identified by detect.py heuristic: presence of an abstract +
numbered section structure + references/bibliography section. The
chunker itself is permissive — it works on any document with that
shape, papers or not.
"""

from __future__ import annotations

import re

from chunkers.base import Chunk, ChunkConfig
from chunkers.default import DefaultChunker


_ABSTRACT_HEADINGS = re.compile(
    r"^#{1,6}\s+(abstract|摘要|summary|tl;dr|tldr)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_REFERENCES_HEADINGS = re.compile(
    r"^#{1,6}\s+(references|bibliography|works cited|citations)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


class PaperChunker:
    name = "paper"
    doc_type = "paper"

    def __init__(self) -> None:
        self._default = DefaultChunker()

    def chunk(self, text: str, config: ChunkConfig | None = None) -> list[Chunk]:
        cfg = config or ChunkConfig()
        if not text or not text.strip():
            return []

        abstract_span = self._find_section_span(text, _ABSTRACT_HEADINGS)
        refs_span = self._find_section_span(text, _REFERENCES_HEADINGS)

        out: list[Chunk] = []

        # Build the chunks in source order: head matter, abstract, body, refs.
        body_chunks_buf: list[Chunk] = []

        # 1) Head matter: everything before abstract (if any) or before refs.
        cutoffs = [s for s in (abstract_span, refs_span) if s is not None]
        head_end = min(s[0] for s in cutoffs) if cutoffs else len(text)
        head_text = text[:head_end].strip()
        if head_text:
            body_chunks_buf.extend(self._default.chunk(head_text, cfg))
            for c in body_chunks_buf:
                out.append(self._tag(c))
            body_chunks_buf = []

        # 2) Abstract as a single chunk.
        if abstract_span is not None:
            abs_start, abs_end, abs_heading = abstract_span
            abs_body = text[abs_start:abs_end].strip()
            line_offset = text[:abs_start].count("\n") + 1
            out.append(
                Chunk(
                    body=abs_body,
                    heading=abs_heading,
                    doc_type=self.doc_type,
                    section_id="abstract",
                    start_line=line_offset,
                    end_line=line_offset + abs_body.count("\n"),
                    metadata={"section_role": "abstract"},
                )
            )

        # 3) Body between abstract end and refs start.
        body_start = abstract_span[1] if abstract_span else head_end
        body_end = refs_span[0] if refs_span else len(text)
        body_text = text[body_start:body_end].strip()
        if body_text:
            body_line_offset = text[:body_start].count("\n") + 1
            for c in self._default.chunk(body_text, cfg):
                shifted = Chunk(
                    body=c.body,
                    heading=c.heading,
                    doc_type=self.doc_type,
                    section_id=c.section_id,
                    start_line=c.start_line + body_line_offset - 1,
                    end_line=c.end_line + body_line_offset - 1,
                    metadata={**c.metadata, "section_role": "body"},
                )
                out.append(shifted)

        # 4) References as a single chunk (preserves citation continuity).
        if refs_span is not None:
            r_start, r_end, r_heading = refs_span
            r_body = text[r_start:r_end].strip()
            line_offset = text[:r_start].count("\n") + 1
            out.append(
                Chunk(
                    body=r_body,
                    heading=r_heading,
                    doc_type=self.doc_type,
                    section_id="references",
                    start_line=line_offset,
                    end_line=line_offset + r_body.count("\n"),
                    metadata={"section_role": "references"},
                )
            )

        return out

    def _tag(self, c: Chunk) -> Chunk:
        """Re-tag doc_type from default → paper."""
        return Chunk(
            body=c.body,
            heading=c.heading,
            doc_type=self.doc_type,
            section_id=c.section_id,
            start_line=c.start_line,
            end_line=c.end_line,
            metadata={**c.metadata, "section_role": "head_matter"},
        )

    def _find_section_span(
        self, text: str, heading_re: re.Pattern
    ) -> tuple[int, int, str] | None:
        """Return (body_start, section_end, heading_text) for the first
        section matching `heading_re`. body_start is the offset of the
        first character AFTER the heading line. section_end is the
        offset of the next same-or-higher-level heading, or len(text).
        """
        match = heading_re.search(text)
        if not match:
            return None
        heading_text = match.group(1)
        # Body starts at end of heading line + newline.
        line_end = text.find("\n", match.end())
        if line_end == -1:
            return (match.end(), len(text), heading_text)
        body_start = line_end + 1
        # Walk forward; find next heading at the same or higher level.
        level = len(_count_hashes(match.group(0)))
        section_end = len(text)
        for next_match in re.finditer(r"^(#{1,6})\s+", text[body_start:], re.MULTILINE):
            next_level = len(next_match.group(1))
            if next_level <= level:
                section_end = body_start + next_match.start()
                break
        return (body_start, section_end, heading_text)


def _count_hashes(heading_line: str) -> str:
    return re.match(r"^(#+)", heading_line).group(1)  # type: ignore[union-attr]
