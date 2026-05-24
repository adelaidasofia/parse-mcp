"""Tests for the chunkers/ package — per-doc-type chunking + auto-detect.

Run with:
    pytest tests/integration/test_chunkers.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from chunkers import chunk_text, detect_doc_type, get_chunker  # noqa: E402
from chunkers.base import Chunk, ChunkConfig, count_tokens, split_paragraphs  # noqa: E402
from chunkers.book import BookChunker  # noqa: E402
from chunkers.default import DefaultChunker  # noqa: E402
from chunkers.dispatcher import list_doc_types  # noqa: E402
from chunkers.manual import ManualChunker  # noqa: E402
from chunkers.paper import PaperChunker  # noqa: E402
from chunkers.qa import QAChunker  # noqa: E402
from chunkers.resume import ResumeChunker  # noqa: E402
from chunkers.table import TableChunker  # noqa: E402


# ---------------------------------------------------------------------------
# base helpers
# ---------------------------------------------------------------------------


def test_count_tokens_basic():
    assert count_tokens("one two three") == 3
    assert count_tokens("") == 0
    assert count_tokens("   ") == 0


def test_split_paragraphs_basic():
    text = "para one line one\npara one line two\n\npara two\n\n\npara three"
    paras = split_paragraphs(text)
    assert len(paras) == 3
    assert paras[0][0] == 1
    assert paras[1][0] == 4
    assert paras[2][0] == 7


def test_chunk_dataclass_roundtrip():
    c = Chunk(body="x", heading=None, doc_type="default")
    d = c.to_dict()
    assert d["body"] == "x"
    assert d["doc_type"] == "default"


# ---------------------------------------------------------------------------
# default chunker
# ---------------------------------------------------------------------------


def test_default_empty():
    assert DefaultChunker().chunk("") == []
    assert DefaultChunker().chunk("   \n  \n") == []


def test_default_splits_on_headings():
    text = (
        "intro paragraph here\n\n"
        "## Pricing\n\nfirst pricing line\nmore pricing\n\n"
        "## Timeline\n\ntimeline content\n"
    )
    chunks = DefaultChunker().chunk(text)
    headings = sorted({c.heading for c in chunks if c.heading})
    assert "Pricing" in headings
    assert "Timeline" in headings


def test_default_force_splits_oversize_paragraph():
    huge = " ".join(["word"] * 600)
    text = f"## Big\n\n{huge}"
    chunks = DefaultChunker().chunk(text, ChunkConfig(max_tokens=100))
    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c.body) <= 110


def test_default_section_id_stable_across_runs():
    text = "## Topic\n\nparagraph content for topic section"
    a = DefaultChunker().chunk(text)
    b = DefaultChunker().chunk(text)
    assert a[0].section_id == b[0].section_id


def test_default_section_id_differs_for_different_position():
    a = DefaultChunker().chunk("## Topic\n\nfoo\n").pop()
    b = DefaultChunker().chunk("intro\n\n## Topic\n\nfoo\n").pop()
    # Same heading text, different start_line → different section_id.
    assert a.section_id != b.section_id


# ---------------------------------------------------------------------------
# paper chunker
# ---------------------------------------------------------------------------


def test_paper_abstract_is_whole_chunk():
    # NOTE: Python implicit string concat binds tighter than `*`, so we
    # build the abstract body separately to avoid the surprise.
    abstract_body = ("this is the abstract body of the paper. " * 20).strip()
    text = (
        "# Title\n\n"
        "## Abstract\n\n"
        f"{abstract_body}\n\n"
        "## Introduction\n\n"
        "body content here in the intro section."
    )
    chunks = PaperChunker().chunk(text, ChunkConfig(target_tokens=50, max_tokens=100))
    abstract_chunks = [c for c in chunks if c.metadata.get("section_role") == "abstract"]
    assert len(abstract_chunks) == 1
    assert "this is the abstract body" in abstract_chunks[0].body
    # Even though abstract exceeds target/max, it stays one chunk.
    assert count_tokens(abstract_chunks[0].body) > 100


def test_paper_references_is_whole_chunk():
    refs_body = ("[1] Author. (2024). Title. " * 30).strip()
    text = (
        "## Introduction\n\nintro body here\n\n"
        "## References\n\n"
        f"{refs_body}"
    )
    chunks = PaperChunker().chunk(text, ChunkConfig(target_tokens=50, max_tokens=100))
    refs_chunks = [c for c in chunks if c.metadata.get("section_role") == "references"]
    assert len(refs_chunks) == 1
    assert refs_chunks[0].section_id == "references"


def test_paper_without_abstract_or_refs_falls_through():
    text = "## Section\n\nordinary body content here"
    chunks = PaperChunker().chunk(text)
    assert all(c.doc_type == "paper" for c in chunks)
    assert all(c.metadata.get("section_role") != "abstract" for c in chunks)


# ---------------------------------------------------------------------------
# book chunker
# ---------------------------------------------------------------------------


def test_book_strips_toc():
    text = (
        "## Table of Contents\n\n"
        "Chapter 1: Intro\nChapter 2: Body\nChapter 3: End\n\n"
        "# Chapter 1\n\n"
        "Chapter one body content goes here.\n\n"
        "# Chapter 2\n\n"
        "Chapter two body content goes here.\n"
    )
    chunks = BookChunker().chunk(text)
    joined = "\n".join(c.body for c in chunks)
    assert "Table of Contents" not in joined
    assert "Chapter 1: Intro" not in joined  # TOC line stripped
    assert "Chapter one body" in joined


def test_book_tags_chapter():
    text = "# Chapter Alpha\n\nbody alpha goes here\n\n# Chapter Beta\n\nbody beta goes here\n"
    chunks = BookChunker().chunk(text)
    chapters = {c.metadata.get("chapter") for c in chunks if c.metadata.get("chapter")}
    assert "Chapter Alpha" in chapters or "Chapter Beta" in chapters


# ---------------------------------------------------------------------------
# manual chunker
# ---------------------------------------------------------------------------


def test_manual_detects_numbered_sections():
    text = (
        "## 1. Setup\n\nsetup body here line one\nsetup line two\n\n"
        "## 1.1 Dependencies\n\nlist of deps here\n\n"
        "## 2. Configuration\n\nconfig body here\n"
    )
    chunks = ManualChunker().chunk(text)
    sections = sorted({c.metadata.get("numbered_section") for c in chunks if c.metadata.get("numbered_section")})
    assert "1" in sections
    assert "1.1" in sections
    assert "2" in sections


def test_manual_never_merges_across_sections():
    text = (
        "## 1. Section One\n\n"
        "tiny body\n\n"
        "## 2. Section Two\n\n"
        "another tiny body\n"
    )
    chunks = ManualChunker().chunk(text, ChunkConfig(min_tokens=50, target_tokens=400))
    sections_in_chunk_bodies = []
    for c in chunks:
        s = c.metadata.get("numbered_section")
        if s and s not in sections_in_chunk_bodies:
            sections_in_chunk_bodies.append(s)
    # Each section is its own chunk; no merge across sections.
    assert sorted(sections_in_chunk_bodies) == ["1", "2"]


def test_manual_falls_through_without_numbered_headings():
    text = "## Plain Heading\n\nbody one\n\n## Another\n\nbody two\n"
    chunks = ManualChunker().chunk(text)
    assert all(c.doc_type == "manual" for c in chunks)


# ---------------------------------------------------------------------------
# qa chunker
# ---------------------------------------------------------------------------


def test_qa_pairs_question_with_answer_via_heading():
    text = (
        "# FAQ\n\n"
        "## What is the refund policy?\n\n"
        "Refunds within 30 days.\n\n"
        "## When does the trial end?\n\n"
        "Trial lasts 14 days.\n"
    )
    chunks = QAChunker().chunk(text)
    questions = [c.metadata.get("question") for c in chunks]
    assert "What is the refund policy?" in questions
    assert "When does the trial end?" in questions
    refund = next(c for c in chunks if c.metadata.get("question") == "What is the refund policy?")
    assert "Refunds within 30 days" in refund.body
    assert refund.body.startswith("Q: ")


def test_qa_detects_q_prefix_format():
    text = (
        "Q: What is X?\n\n"
        "A: X is a thing.\n\n"
        "Q: How does Y work?\n\n"
        "Y works like this.\n"
    )
    chunks = QAChunker().chunk(text)
    assert len(chunks) == 2
    assert chunks[0].metadata["question"] == "What is X?"


def test_qa_detects_bullet_questions():
    text = (
        "- What is the price?\n"
        "  The price is $100.\n"
        "- When does shipping happen?\n"
        "  Shipping is Tuesday.\n"
    )
    chunks = QAChunker().chunk(text)
    questions = [c.metadata.get("question") for c in chunks]
    assert "What is the price?" in questions
    assert "When does shipping happen?" in questions


def test_qa_no_questions_returns_one_chunk():
    text = "Plain paragraph with no questions just statements.\n"
    chunks = QAChunker().chunk(text)
    assert len(chunks) == 1
    assert chunks[0].metadata["qa_pairs_found"] == 0


# ---------------------------------------------------------------------------
# resume chunker
# ---------------------------------------------------------------------------


def test_resume_detects_domain_sections():
    text = (
        "Jane Doe\njane@example.com\n\n"
        "## Summary\n\nSenior engineer with 10 years.\n\n"
        "## Experience\n\nAcme Corp 2020-now\nDoubled growth.\n\n"
        "## Education\n\nMIT BS 2015\n\n"
        "## Skills\n\nPython, Go, distributed systems\n"
    )
    chunks = ResumeChunker().chunk(text)
    roles = {c.metadata.get("resume_role") for c in chunks}
    assert "summary" in roles
    assert "experience" in roles
    assert "education" in roles
    assert "skills" in roles


def test_resume_head_matter_tagged_contact():
    text = (
        "Jane Doe\njane@example.com\n+1-555-0100\n\n"
        "## Experience\n\nAcme Corp\n"
    )
    chunks = ResumeChunker().chunk(text)
    contact_chunks = [c for c in chunks if c.metadata.get("resume_role") == "contact"]
    assert len(contact_chunks) == 1
    assert "Jane Doe" in contact_chunks[0].body


def test_resume_spanish_headings():
    text = (
        "## Experiencia\n\nAcme empresa\n\n"
        "## Educación\n\nUNI 2010\n\n"
        "## Habilidades\n\nPython, Go\n"
    )
    chunks = ResumeChunker().chunk(text)
    roles = {c.metadata.get("resume_role") for c in chunks}
    assert "experience" in roles
    assert "education" in roles
    assert "skills" in roles


# ---------------------------------------------------------------------------
# table chunker
# ---------------------------------------------------------------------------


def test_table_emits_row_per_chunk():
    text = (
        "| Name | Price | Stock |\n"
        "| --- | --- | --- |\n"
        "| Widget | 10 | 50 |\n"
        "| Gadget | 20 | 30 |\n"
        "| Sprocket | 5 | 100 |\n"
    )
    chunks = TableChunker().chunk(text)
    assert len(chunks) == 3
    bodies = [c.body for c in chunks]
    assert any("Name: Widget" in b for b in bodies)
    assert any("Price: 20" in b for b in bodies)


def test_table_falls_through_for_non_table_text():
    text = "## Heading\n\nparagraph one\n\nparagraph two\n"
    chunks = TableChunker().chunk(text)
    # No table → default chunker treatment → at least one chunk.
    assert len(chunks) >= 1


def test_table_handles_pipe_in_data_rows():
    text = (
        "| Col A | Col B |\n"
        "| --- | --- |\n"
        "| val one | val two |\n"
    )
    chunks = TableChunker().chunk(text)
    assert len(chunks) == 1
    assert "Col A: val one" in chunks[0].body
    assert "Col B: val two" in chunks[0].body


# ---------------------------------------------------------------------------
# detect_doc_type
# ---------------------------------------------------------------------------


def test_detect_paper():
    text = (
        "# Title\n\n## Abstract\n\nabstract body here for the paper introduction\n\n"
        "## Introduction\n\nbody\n\n"
        "## References\n\n[1] Author 2024.\n"
    )
    assert detect_doc_type(text) == "paper"


def test_detect_manual():
    text = (
        "## 1. Install\n\nbody\n\n"
        "## 1.1 Dependencies\n\nbody\n\n"
        "## 2. Configure\n\nbody\n"
    )
    assert detect_doc_type(text) == "manual"


def test_detect_qa():
    text = (
        "## What is X?\n\nX is a thing.\n\n"
        "## How does Y work?\n\nY works.\n\n"
        "## Why Z?\n\nbecause.\n"
    )
    assert detect_doc_type(text) == "qa"


def test_detect_resume():
    text = (
        "Jane Doe\n\n## Experience\n\nAcme\n\n## Education\n\nMIT\n"
    )
    assert detect_doc_type(text) == "resume"


def test_detect_table():
    text = (
        "Pricing schedule:\n\n"
        "| SKU | Price |\n"
        "| --- | --- |\n"
        "| A | 10 |\n"
        "| B | 20 |\n"
        "| C | 30 |\n"
        "| D | 40 |\n"
        "| E | 50 |\n"
    )
    assert detect_doc_type(text) == "table"


def test_detect_book():
    text = (
        "# Table of Contents\n\nCh 1\nCh 2\nCh 3\n\n"
        "# Chapter 1\n\nbody body body\n\n"
        "# Chapter 2\n\nbody body body\n\n"
        "# Chapter 3\n\nbody body body\n"
    )
    assert detect_doc_type(text) == "book"


def test_detect_default_for_plain_prose():
    text = "## Heading\n\nordinary paragraph text without any special structure\n"
    assert detect_doc_type(text) == "default"


def test_detect_empty():
    assert detect_doc_type("") == "default"


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------


def test_dispatcher_auto_routes_to_paper():
    text = "## Abstract\n\nabstract body\n\n## References\n\n[1]\n"
    chunks, resolved = chunk_text(text, doc_type="auto")
    assert resolved == "paper"
    assert any(c.doc_type == "paper" for c in chunks)


def test_dispatcher_explicit_override():
    """Force `table` chunker even on non-table text."""
    text = "## Heading\n\nparagraph one\n\nparagraph two\n"
    chunks, resolved = chunk_text(text, doc_type="table")
    assert resolved == "table"


def test_dispatcher_unknown_doc_type_falls_back_to_default():
    text = "## Heading\n\nparagraph\n"
    chunks, resolved = chunk_text(text, doc_type="bogus-type")
    # Resolved value passes through; chunker falls back to DefaultChunker.
    assert resolved == "bogus-type"
    assert len(chunks) >= 1


def test_dispatcher_empty_text():
    chunks, resolved = chunk_text("")
    assert chunks == []


def test_get_chunker_unknown_returns_default():
    assert get_chunker("nope").name == "default"


def test_list_doc_types():
    types = list_doc_types()
    assert {"default", "paper", "book", "manual", "qa", "resume", "table"}.issubset(set(types))
