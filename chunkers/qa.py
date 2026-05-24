"""Q&A / FAQ chunker.

Ragflow's `qa.py` insight: heading-stack pairing. Questions and answers
must be retrieved together — a chunk that has the question but not the
answer (or vice versa) is broken for downstream use. The chunker pairs
each detected question with all body content until the next question
(or end of doc).

Question detection:
- Heading line ending in `?`
- Bullet/numbered list item ending in `?`
- Heading prefixed with `Q:` / `Q.` / `Q1.` / `Question:`

The answer block is everything between this question and the next.
"""

from __future__ import annotations

import re

from chunkers.base import Chunk, ChunkConfig, count_tokens


_QUESTION_PREFIX_RE = re.compile(
    r"^(?:#{1,6}\s+)?(?:Q\d*[:.]\s*|Question[:\s]+)",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _is_question_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    # Heading ending in ?
    m = re.match(r"^#{1,6}\s+(.+?)\s*$", stripped)
    if m and m.group(1).rstrip().endswith("?"):
        return True
    # List item ending in ?
    m = re.match(r"^[-*+]\s+(.+?)\s*$", stripped)
    if m and m.group(1).rstrip().endswith("?"):
        return True
    m = re.match(r"^\d+[.)]\s+(.+?)\s*$", stripped)
    if m and m.group(1).rstrip().endswith("?"):
        return True
    # Q: / Question: prefix
    if _QUESTION_PREFIX_RE.match(stripped):
        return True
    return False


def _clean_question(line: str) -> str:
    stripped = line.strip()
    stripped = re.sub(r"^#{1,6}\s+", "", stripped)
    stripped = re.sub(r"^[-*+]\s+", "", stripped)
    stripped = re.sub(r"^\d+[.)]\s+", "", stripped)
    stripped = _QUESTION_PREFIX_RE.sub("", stripped, count=1)
    return stripped.strip()


class QAChunker:
    name = "qa"
    doc_type = "qa"

    def chunk(self, text: str, config: ChunkConfig | None = None) -> list[Chunk]:
        cfg = config or ChunkConfig()
        if not text or not text.strip():
            return []

        lines = text.splitlines()
        question_lines: list[tuple[int, str]] = []
        for idx, line in enumerate(lines, start=1):
            if _is_question_line(line):
                question_lines.append((idx, line))

        if not question_lines:
            # No questions detected → emit one chunk for the whole doc
            # so caller still gets a result. doc_type stays "qa" so
            # callers can detect "this was a Q&A request but no Qs".
            return [
                Chunk(
                    body=text.strip(),
                    heading=None,
                    doc_type=self.doc_type,
                    section_id=None,
                    start_line=1,
                    end_line=len(lines),
                    metadata={"qa_pairs_found": 0},
                )
            ]

        out: list[Chunk] = []
        for i, (q_line_no, q_raw) in enumerate(question_lines):
            question = _clean_question(q_raw)
            next_q_line = (
                question_lines[i + 1][0] if i + 1 < len(question_lines) else len(lines) + 1
            )
            # Answer = lines AFTER the question line, BEFORE next question line.
            answer_lines = lines[q_line_no:next_q_line - 1]
            answer = "\n".join(answer_lines).strip()
            body = f"Q: {question}\n\nA: {answer}" if answer else f"Q: {question}"
            tokens = count_tokens(body)
            out.append(
                Chunk(
                    body=body,
                    heading=question,
                    doc_type=self.doc_type,
                    section_id=f"qa-{i+1:04d}",
                    start_line=q_line_no,
                    end_line=next_q_line - 1,
                    metadata={
                        "question": question,
                        "answer_token_count": count_tokens(answer),
                        "oversize": tokens > cfg.max_tokens,
                    },
                )
            )
        return out
