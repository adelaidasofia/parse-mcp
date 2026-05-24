"""Default chunker — paragraph-based uniform chunking.

Safe fallback when no doc-type-specific chunker matches. Splits on
markdown headings first, then paragraphs within a section, packed to
`target_tokens`. Force-splits at sentence boundaries when a paragraph
exceeds `max_tokens`.

Output guarantees:
- Heading context preserved: every chunk knows its enclosing heading
- Stable section_id: SHA-style hash of (heading_path, position) so
  re-running on the same input produces the same section_ids
- Line ranges: 1-indexed inclusive start_line + end_line
"""

from __future__ import annotations

import hashlib
import re

from chunkers.base import Chunk, ChunkConfig, count_tokens, split_paragraphs


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


class DefaultChunker:
    name = "default"
    doc_type = "default"

    def chunk(self, text: str, config: ChunkConfig | None = None) -> list[Chunk]:
        cfg = config or ChunkConfig()
        if not text or not text.strip():
            return []
        sections = self._split_sections(text)
        out: list[Chunk] = []
        for heading, heading_path, section_body, start_line in sections:
            section_id = self._section_id(heading_path, start_line)
            chunks = self._pack_section(
                body=section_body,
                heading=heading,
                section_id=section_id,
                section_start_line=start_line,
                cfg=cfg,
            )
            out.extend(chunks)
        return out

    def _split_sections(
        self, text: str
    ) -> list[tuple[str | None, tuple[str, ...], str, int]]:
        """Walk the doc; emit (heading, heading_path, body_below_heading, start_line).

        heading is the closest enclosing heading text (e.g. "Pricing").
        heading_path is the full hierarchy ("Overview", "Pricing").
        body_below_heading is the text under the heading, up to the next
        same-or-shallower heading.
        start_line is the 1-indexed line of the FIRST body line (not the
        heading line itself).
        """
        lines = text.splitlines()
        sections: list[tuple[str | None, tuple[str, ...], str, int]] = []
        heading_stack: list[tuple[int, str]] = []  # (level, text)
        cur_lines: list[str] = []
        cur_start = 1

        def flush(cur_idx: int) -> None:
            nonlocal cur_lines, cur_start
            if not cur_lines:
                return
            body = "\n".join(cur_lines).strip("\n")
            if not body.strip():
                cur_lines = []
                return
            heading_text = heading_stack[-1][1] if heading_stack else None
            heading_path = tuple(h for _, h in heading_stack)
            sections.append((heading_text, heading_path, body, cur_start))
            cur_lines = []

        for line_no, line in enumerate(lines, start=1):
            m = _HEADING_RE.match(line)
            if m:
                flush(line_no)
                level = len(m.group(1))
                heading_text = m.group(2).strip()
                # Pop heading stack to (level - 1).
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, heading_text))
                cur_start = line_no + 1
            else:
                if not cur_lines:
                    cur_start = line_no
                cur_lines.append(line)
        flush(len(lines) + 1)
        return sections

    def _pack_section(
        self,
        body: str,
        heading: str | None,
        section_id: str,
        section_start_line: int,
        cfg: ChunkConfig,
    ) -> list[Chunk]:
        """Pack paragraphs into chunks honoring target/max/min token caps."""
        paragraphs = split_paragraphs(body)
        if not paragraphs:
            return []

        chunks: list[Chunk] = []
        buf: list[tuple[int, str]] = []
        buf_tokens = 0

        def flush_buf() -> None:
            nonlocal buf, buf_tokens
            if not buf:
                return
            text_joined = "\n\n".join(p for _, p in buf)
            start = section_start_line + buf[0][0] - 1
            last_para = buf[-1][1]
            end = section_start_line + buf[-1][0] - 1 + len(last_para.splitlines()) - 1
            chunks.append(
                Chunk(
                    body=text_joined,
                    heading=heading,
                    doc_type=self.doc_type,
                    section_id=section_id,
                    start_line=start,
                    end_line=end,
                )
            )
            buf = []
            buf_tokens = 0

        for offset, para in paragraphs:
            ptokens = count_tokens(para)
            if ptokens >= cfg.max_tokens:
                # Flush current buffer + force-split this oversize paragraph.
                flush_buf()
                for piece in _force_split_sentences(para, cfg.max_tokens):
                    chunks.append(
                        Chunk(
                            body=piece,
                            heading=heading,
                            doc_type=self.doc_type,
                            section_id=section_id,
                            start_line=section_start_line + offset - 1,
                            end_line=section_start_line + offset - 1
                            + len(piece.splitlines())
                            - 1,
                        )
                    )
                continue
            if buf_tokens + ptokens > cfg.target_tokens and buf:
                flush_buf()
            buf.append((offset, para))
            buf_tokens += ptokens
        flush_buf()

        # Merge undersized trailing chunk with previous when it would
        # fall below min_tokens (unless it's the only one).
        if len(chunks) >= 2 and count_tokens(chunks[-1].body) < cfg.min_tokens:
            last = chunks.pop()
            prev = chunks.pop()
            merged_body = prev.body + "\n\n" + last.body
            chunks.append(
                Chunk(
                    body=merged_body,
                    heading=prev.heading,
                    doc_type=prev.doc_type,
                    section_id=prev.section_id,
                    start_line=prev.start_line,
                    end_line=last.end_line,
                )
            )
        return chunks

    def _section_id(self, heading_path: tuple[str, ...], start_line: int) -> str:
        """Stable section_id from heading path + line position.

        Same input → same id. Survives re-runs but changes when section
        position or heading text changes.
        """
        key = "::".join(heading_path) + f"@{start_line}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _force_split_sentences(text: str, max_tokens: int) -> list[str]:
    """Split an oversize paragraph at sentence boundaries, packing pieces
    up to max_tokens each. Falls back to whitespace if no sentence
    boundary found.
    """
    sentences = _SENTENCE_RE.split(text)
    if len(sentences) == 1:
        # No sentence boundaries — split on whitespace into max_tokens groups.
        words = text.split()
        return [
            " ".join(words[i : i + max_tokens])
            for i in range(0, len(words), max_tokens)
        ] or [text]
    out: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for s in sentences:
        st = count_tokens(s)
        if buf_tokens + st > max_tokens and buf:
            out.append(" ".join(buf))
            buf = []
            buf_tokens = 0
        buf.append(s)
        buf_tokens += st
    if buf:
        out.append(" ".join(buf))
    return out
