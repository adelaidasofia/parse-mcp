"""Manual / runbook chunker.

Ragflow's `manual.py` insight: section-ID-aware merge. A manual has
numbered sections (1, 1.1, 1.1.1) and the chunker MUST NOT merge
content across section boundaries — section 1.2's content getting
glued onto section 1.1's chunk is a precision bug, not a recall win.

Strategy:
1. Detect numbered section headings (e.g., `## 1. Setup`, `### 1.1
   Install dependencies`).
2. Track a section_id derived from the numbered prefix.
3. Default-chunk per section; tag every chunk with its section_id +
   numbered_section_path.
4. Never merge two chunks with different section_ids.
"""

from __future__ import annotations

import re

from chunkers.base import Chunk, ChunkConfig, count_tokens
from chunkers.default import DefaultChunker


_NUMBERED_HEADING_RE = re.compile(
    r"^(#{1,6})\s+((?:\d+\.)+\d*)\s*\.?\s*(.*?)\s*$",
    re.MULTILINE,
)


class ManualChunker:
    name = "manual"
    doc_type = "manual"

    def __init__(self) -> None:
        self._default = DefaultChunker()

    def chunk(self, text: str, config: ChunkConfig | None = None) -> list[Chunk]:
        cfg = config or ChunkConfig()
        if not text or not text.strip():
            return []
        sections = self._split_numbered_sections(text)
        if not sections:
            # No numbered structure → fall through to default uniform chunker.
            return [
                self._retag(c) for c in self._default.chunk(text, cfg)
            ]
        out: list[Chunk] = []
        for sec_path, heading, body, start_line in sections:
            section_id = sec_path or "head"
            inner_chunks = self._default.chunk(body, cfg)
            for c in inner_chunks:
                # Re-tag with manual's metadata + don't merge across sections.
                out.append(
                    Chunk(
                        body=c.body,
                        heading=heading,
                        doc_type=self.doc_type,
                        section_id=section_id,
                        start_line=c.start_line + start_line - 1,
                        end_line=c.end_line + start_line - 1,
                        metadata={
                            **c.metadata,
                            "numbered_section": sec_path,
                        },
                    )
                )
            # Empty sections: emit a placeholder so the section_id is
            # discoverable even with no body (matches manual.py behavior).
            if not inner_chunks and heading:
                out.append(
                    Chunk(
                        body=heading,
                        heading=heading,
                        doc_type=self.doc_type,
                        section_id=section_id,
                        start_line=start_line,
                        end_line=start_line,
                        metadata={"numbered_section": sec_path, "empty_section": True},
                    )
                )
        return out

    def _retag(self, c: Chunk) -> Chunk:
        return Chunk(
            body=c.body,
            heading=c.heading,
            doc_type=self.doc_type,
            section_id=c.section_id,
            start_line=c.start_line,
            end_line=c.end_line,
            metadata=c.metadata,
        )

    def _split_numbered_sections(
        self, text: str
    ) -> list[tuple[str | None, str | None, str, int]]:
        """Walk numbered headings; return (sec_path, heading, body, start_line).

        sec_path is the dotted-numeric prefix ("1.1", "2.3.4"). None for
        head matter before the first numbered heading.
        """
        out: list[tuple[str | None, str | None, str, int]] = []
        matches = list(_NUMBERED_HEADING_RE.finditer(text))
        if not matches:
            return []

        # Head matter
        first_start = matches[0].start()
        if first_start > 0:
            head_body = text[:first_start].strip()
            if head_body:
                out.append((None, None, head_body, 1))

        for i, m in enumerate(matches):
            sec_path = m.group(2).rstrip(".")
            title = m.group(3).strip() or None
            heading = f"{sec_path} {title}" if title else sec_path
            body_start = text.find("\n", m.end())
            if body_start == -1:
                body_start = m.end()
            else:
                body_start += 1
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()
            start_line = text[:body_start].count("\n") + 1
            out.append((sec_path, heading, body, start_line))
        return out
