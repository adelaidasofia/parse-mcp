"""Resume chunker.

Ragflow's `resume.py` insight: domain-section detection. A resume
isn't paragraph-uniform; it's a fixed set of role-specific blocks
(Contact, Summary, Experience, Education, Skills, Projects,
Certifications, etc). The chunker detects those domain sections and
emits one chunk per section, with the section role in metadata so
retrieval can filter by it.

Strategy:
1. Walk markdown headings.
2. Match each heading against the canonical resume-section vocabulary.
3. Emit one chunk per detected section.
4. Unmatched headings get the default chunker treatment.
"""

from __future__ import annotations

import re

from chunkers.base import Chunk, ChunkConfig
from chunkers.default import DefaultChunker


# Domain vocabulary. Each key is the canonical role; values are the
# regex patterns that map to it (case-insensitive). Pulled from common
# resume heading conventions (English + Spanish).
_DOMAIN_VOCAB: dict[str, list[str]] = {
    "contact": [
        r"contact",
        r"personal info(rmation)?",
        r"datos personales",
        r"información de contacto",
    ],
    "summary": [
        r"summary",
        r"profile",
        r"objective",
        r"about me",
        r"perfil",
        r"resumen",
    ],
    "experience": [
        r"(work|professional)?\s*experience",
        r"employment(\s+history)?",
        r"career(\s+history)?",
        r"experiencia(\s+(profesional|laboral))?",
    ],
    "education": [r"education", r"academic", r"educación", r"formación( académica)?"],
    "skills": [r"skills", r"competencies", r"competencias", r"habilidades"],
    "projects": [r"projects", r"portfolio", r"proyectos"],
    "certifications": [r"certifications?", r"certificaciones"],
    "publications": [r"publications", r"papers", r"publicaciones"],
    "awards": [r"awards", r"honors?", r"recognitions?", r"premios"],
    "languages": [r"languages", r"idiomas"],
    "interests": [r"interests", r"hobbies", r"intereses"],
    "references": [r"references", r"referencias"],
}


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _match_domain(heading: str) -> str | None:
    h = heading.strip().lower()
    for role, patterns in _DOMAIN_VOCAB.items():
        for pat in patterns:
            if re.fullmatch(pat, h, re.IGNORECASE):
                return role
    return None


class ResumeChunker:
    name = "resume"
    doc_type = "resume"

    def __init__(self) -> None:
        self._default = DefaultChunker()

    def chunk(self, text: str, config: ChunkConfig | None = None) -> list[Chunk]:
        cfg = config or ChunkConfig()
        if not text or not text.strip():
            return []

        lines = text.splitlines()
        heading_positions: list[tuple[int, int, str]] = []  # (line_idx, level, heading)
        for idx, line in enumerate(lines):
            m = _HEADING_RE.match(line)
            if m:
                heading_positions.append((idx, len(m.group(1)), m.group(2).strip()))

        out: list[Chunk] = []
        if not heading_positions:
            return self._default.chunk(text, cfg)

        # Head matter: lines BEFORE the first heading get a "contact" tag
        # heuristically (most resumes lead with contact info before any heading).
        first_heading_idx = heading_positions[0][0]
        if first_heading_idx > 0:
            head_body = "\n".join(lines[:first_heading_idx]).strip()
            if head_body:
                out.append(
                    Chunk(
                        body=head_body,
                        heading=None,
                        doc_type=self.doc_type,
                        section_id="contact",
                        start_line=1,
                        end_line=first_heading_idx,
                        metadata={"resume_role": "contact", "head_matter": True},
                    )
                )

        for i, (line_idx, _level, heading) in enumerate(heading_positions):
            next_idx = (
                heading_positions[i + 1][0] if i + 1 < len(heading_positions) else len(lines)
            )
            body_lines = lines[line_idx + 1 : next_idx]
            body = "\n".join(body_lines).strip()
            if not body:
                continue
            role = _match_domain(heading) or "other"
            out.append(
                Chunk(
                    body=body,
                    heading=heading,
                    doc_type=self.doc_type,
                    section_id=role if role != "other" else f"section-{i+1:04d}",
                    start_line=line_idx + 2,
                    end_line=next_idx,
                    metadata={"resume_role": role},
                )
            )
        return out
