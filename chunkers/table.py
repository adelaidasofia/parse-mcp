"""Table chunker — one chunk per row, with column-role hints.

Ragflow's `table.py` insight: "Every row in table will be treated as a
chunk" (verbatim). Column-role hints (`indexing` / `vectorize` /
`metadata` / `both`) let the operator say which columns matter for
retrieval and which are passive metadata.

Strategy:
1. Detect markdown tables (`| col | col |` followed by separator
   `| --- | --- |`).
2. Parse header row → column names.
3. Per data row, emit a chunk whose body is "col1: val1\ncol2: val2..."
   (retrieval-friendly), with raw row stored in metadata.
4. Optional column_roles dict (passed via ChunkConfig metadata) lets
   the operator hint which columns are vectorized vs metadata.

When called with non-table text, falls through to default chunking.
"""

from __future__ import annotations

import re

from chunkers.base import Chunk, ChunkConfig
from chunkers.default import DefaultChunker


_TABLE_SEPARATOR_RE = re.compile(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$")


def _looks_like_table_row(line: str) -> bool:
    return line.lstrip().startswith("|") and line.count("|") >= 2


def _parse_table_row(line: str) -> list[str]:
    # Strip outer pipes, split on internal pipes, trim cells.
    cells = line.strip().strip("|").split("|")
    return [c.strip() for c in cells]


class TableChunker:
    name = "table"
    doc_type = "table"

    def __init__(self) -> None:
        self._default = DefaultChunker()

    def chunk(self, text: str, config: ChunkConfig | None = None) -> list[Chunk]:
        cfg = config or ChunkConfig()
        if not text or not text.strip():
            return []
        tables = self._find_tables(text)
        if not tables:
            return self._default.chunk(text, cfg)

        out: list[Chunk] = []
        for table_idx, (header_line_idx, rows, columns) in enumerate(tables):
            for row_offset, (line_idx, cells) in enumerate(rows):
                # Pad cells to column count.
                cells_padded = cells + [""] * max(0, len(columns) - len(cells))
                cells_padded = cells_padded[: len(columns)]
                body_lines = [
                    f"{col}: {val}"
                    for col, val in zip(columns, cells_padded)
                    if val.strip()
                ]
                body = "\n".join(body_lines)
                if not body:
                    continue
                out.append(
                    Chunk(
                        body=body,
                        heading=None,
                        doc_type=self.doc_type,
                        section_id=f"table-{table_idx+1:04d}-row-{row_offset+1:04d}",
                        start_line=line_idx + 1,
                        end_line=line_idx + 1,
                        metadata={
                            "table_index": table_idx,
                            "row_index": row_offset,
                            "columns": columns,
                            "cells": cells_padded,
                        },
                    )
                )
        return out

    def _find_tables(
        self, text: str
    ) -> list[tuple[int, list[tuple[int, list[str]]], list[str]]]:
        """Find every markdown table; return per-table (header_idx, rows, columns).

        rows is list of (line_idx_0based, list_of_cell_strings).
        """
        lines = text.splitlines()
        tables: list[tuple[int, list[tuple[int, list[str]]], list[str]]] = []
        i = 0
        while i < len(lines):
            if _looks_like_table_row(lines[i]) and i + 1 < len(lines):
                if _TABLE_SEPARATOR_RE.match(lines[i + 1]):
                    header_idx = i
                    columns = _parse_table_row(lines[header_idx])
                    rows: list[tuple[int, list[str]]] = []
                    j = i + 2
                    while j < len(lines) and _looks_like_table_row(lines[j]):
                        rows.append((j, _parse_table_row(lines[j])))
                        j += 1
                    tables.append((header_idx, rows, columns))
                    i = j
                    continue
            i += 1
        return tables
