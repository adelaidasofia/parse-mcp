"""Auto-detect doc_type from markdown structure.

Heuristics (highest-confidence first):

1. **resume** — short doc (<3000 tokens) with 2+ resume-vocab headings
   (Experience, Education, Skills, etc).
2. **qa** — at least 3 detected question lines (heading-ends-in-? or
   `Q:` prefix) AND fraction of question-lines among lines > 0.05.
3. **table** — body contains at least one markdown-table separator
   line AND >40% of non-empty lines are pipe-table rows.
4. **paper** — has both an `Abstract` heading AND a `References` /
   `Bibliography` heading.
5. **manual** — at least 3 numbered headings of form `## 1. ...` or
   `### 1.1 ...`.
6. **book** — has a Table of Contents heading AND ≥3 H1 headings
   (chapters).
7. **default** — fallback.
"""

from __future__ import annotations

import re

from chunkers.qa import _is_question_line
from chunkers.resume import _DOMAIN_VOCAB
from chunkers.manual import _NUMBERED_HEADING_RE


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_TABLE_SEP_RE = re.compile(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$", re.MULTILINE)
_PIPE_ROW_RE = re.compile(r"^\|.*\|.*$", re.MULTILINE)
_ABSTRACT_RE = re.compile(
    r"^#{1,6}\s+(abstract|摘要|summary|tl;dr|tldr)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_REFS_RE = re.compile(
    r"^#{1,6}\s+(references|bibliography|works cited|citations)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_TOC_RE = re.compile(
    r"^#{1,2}\s+(contents|table of contents|índice|目录|toc)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def detect_doc_type(text: str) -> str:
    """Return one of: paper | book | manual | qa | resume | table | default."""
    if not text or not text.strip():
        return "default"

    token_count = len(text.split())

    # Walk headings once; reuse across heuristics.
    headings = [(len(m.group(1)), m.group(2).strip()) for m in _HEADING_RE.finditer(text)]

    # 1) resume: short doc, multiple resume-vocab headings.
    if token_count < 3000:
        resume_hits = 0
        for _level, heading in headings:
            for patterns in _DOMAIN_VOCAB.values():
                if any(re.fullmatch(p, heading, re.IGNORECASE) for p in patterns):
                    resume_hits += 1
                    break
        if resume_hits >= 2:
            return "resume"

    # 2) qa: dense question-line presence.
    lines = text.splitlines()
    non_empty = [ln for ln in lines if ln.strip()]
    q_lines = [ln for ln in non_empty if _is_question_line(ln)]
    if len(q_lines) >= 3 and len(q_lines) / max(1, len(non_empty)) > 0.05:
        return "qa"

    # 3) table: lots of pipe-table rows.
    if _TABLE_SEP_RE.search(text):
        pipe_rows = len(_PIPE_ROW_RE.findall(text))
        if pipe_rows / max(1, len(non_empty)) > 0.4:
            return "table"

    # 4) paper: abstract + references both present.
    if _ABSTRACT_RE.search(text) and _REFS_RE.search(text):
        return "paper"

    # 5) manual: 3+ numbered headings.
    if len(list(_NUMBERED_HEADING_RE.finditer(text))) >= 3:
        return "manual"

    # 6) book: TOC heading + 3+ H1 (chapters).
    if _TOC_RE.search(text):
        h1_count = sum(1 for level, _ in headings if level == 1)
        if h1_count >= 3:
            return "book"

    return "default"
