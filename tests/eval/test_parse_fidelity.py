"""Parse-fidelity scorer tests.

The scorer is the trust-critical half of the eval harness: if TEDS or the
edit-distance metrics are wrong, the backend-routing matrix that downstream
routing decisions lean on is wrong. So these tests assert COMPUTED values
(and cross-check the hand-rolled tree-edit-distance against known cases),
never the shape of a data structure.

Pure-Python, backend-free: runs in CI with markitdown-only installed.
"""
from __future__ import annotations

from pathlib import Path

from tests.eval.score_parse_fidelity import (
    Node,
    _levenshtein,
    _tree_edit_distance,
    aggregate,
    evaluate,
    normalized_edit_distance,
    reading_order_score,
    score_document,
    teds,
    text_score,
)

_FIXTURES = Path(__file__).parent / "fixtures"


def test_text_score_identical_is_one():
    doc = "# Quarterly Report\n\nRevenue grew 12% year over year."
    assert text_score(doc, doc) == 1.0


def test_levenshtein_known_value():
    # Textbook case: kitten -> sitting is 3 edits (k->s, e->i, +g).
    assert _levenshtein("kitten", "sitting") == 3


def test_levenshtein_on_sequences():
    # The reading-order metric runs Levenshtein over lists of block indices.
    assert _levenshtein([0, 1, 2, 3], [0, 2, 1, 3]) == 2


def test_text_score_partial_between_zero_and_one():
    gt = "The contract term is five years from the effective date."
    pred = "The contract term is five years from the affective date."
    score = text_score(pred, gt)
    assert 0.0 < score < 1.0


def test_text_score_whitespace_insensitive():
    gt = "Line one.\n\nLine two."
    pred = "Line one.\n\n\n   Line two.   "
    assert text_score(pred, gt) == 1.0


def test_text_score_empty_prediction_is_zero():
    # markitdown on a scanned/image PDF returns "" — fidelity must read 0.
    assert text_score("", "Some real content here.") == 0.0
    assert normalized_edit_distance("", "Some real content here.") == 1.0


# --------------------------------------------------------------------------
# Tree-edit-distance engine (Zhang-Shasha) — the spine of TEDS.
# Cross-checked against hand-computed cases + algebraic properties so a
# broken engine cannot pass.
# --------------------------------------------------------------------------


def test_tree_edit_distance_identity_is_zero():
    t = Node("r", [Node("a"), Node("b", [Node("c")])])
    t2 = Node("r", [Node("a"), Node("b", [Node("c")])])
    assert _tree_edit_distance(t, t2) == 0.0


def test_tree_edit_distance_single_rename():
    a = Node("r", [Node("x")])
    b = Node("r", [Node("y")])
    assert _tree_edit_distance(a, b) == 1.0


def test_tree_edit_distance_single_insert():
    a = Node("r", [Node("x")])
    b = Node("r", [Node("x"), Node("y")])
    assert _tree_edit_distance(a, b) == 1.0  # one insertion


def test_tree_edit_distance_relabel_root_and_delete_child():
    a = Node("r", [Node("a"), Node("b")])
    b = Node("q", [Node("a")])
    # relabel r->q (1) + delete b (1) = 2
    assert _tree_edit_distance(a, b) == 2.0


def test_tree_edit_distance_is_symmetric():
    a = Node("table", [Node("tr", [Node("c1"), Node("c2")])])
    b = Node("table", [Node("tr", [Node("c1")]), Node("tr", [Node("z")])])
    assert _tree_edit_distance(a, b) == _tree_edit_distance(b, a)


# --------------------------------------------------------------------------
# TEDS over markdown tables.
# --------------------------------------------------------------------------

_TABLE_GT = """\
| Quarter | Revenue | Margin |
|---------|---------|--------|
| Q1      | 1.2M    | 18%    |
| Q2      | 1.5M    | 21%    |
"""


def test_teds_identical_table_is_one():
    assert teds(_TABLE_GT, _TABLE_GT) == 1.0


def test_teds_one_cell_wrong_is_between_zero_and_one():
    pred = _TABLE_GT.replace("21%", "12%")  # one cell misread
    score = teds(pred, _TABLE_GT)
    assert 0.0 < score < 1.0


def test_teds_structural_damage_scores_lower_than_one_cell_error():
    one_cell = teds(_TABLE_GT.replace("21%", "12%"), _TABLE_GT)
    # A backend that drops an entire row loses structure, not just content.
    dropped_row = teds(
        "| Quarter | Revenue | Margin |\n|---|---|---|\n| Q1 | 1.2M | 18% |\n",
        _TABLE_GT,
    )
    assert dropped_row < one_cell


def test_teds_no_table_in_ground_truth_is_none():
    # Non-table fixtures must not contribute a bogus 0/1 to the table column.
    assert teds("# Heading\n\nProse only.", "# Heading\n\nProse only.") is None


def test_teds_prediction_found_no_table_is_zero():
    # GT has a table; the backend emitted only prose -> worst table score.
    assert teds("Revenue was 1.2M in Q1.", _TABLE_GT) == 0.0


# --------------------------------------------------------------------------
# Reading-order metric (block-level adaptation of OmniDocBench).
# --------------------------------------------------------------------------

_MULTIBLOCK = (
    "First paragraph about the introduction.\n\n"
    "Second paragraph about the method.\n\n"
    "Third paragraph about the results.\n\n"
    "Fourth paragraph about the conclusion."
)


def test_reading_order_identical_is_one():
    assert reading_order_score(_MULTIBLOCK, _MULTIBLOCK) == 1.0


def test_reading_order_reversed_below_one():
    blocks = _MULTIBLOCK.split("\n\n")
    reversed_doc = "\n\n".join(reversed(blocks))
    assert reading_order_score(reversed_doc, _MULTIBLOCK) < 1.0


def test_reading_order_interleaved_columns_scores_low():
    # Ground truth: column A fully, then column B fully (correct reading order).
    gt = (
        "Alpha block one.\n\nAlpha block two.\n\nAlpha block three.\n\n"
        "Beta block one.\n\nBeta block two.\n\nBeta block three."
    )
    # markitdown-style failure: it reads the two columns interleaved row-by-row.
    interleaved = (
        "Alpha block one.\n\nBeta block one.\n\nAlpha block two.\n\n"
        "Beta block two.\n\nAlpha block three.\n\nBeta block three."
    )
    # A single adjacent swap of two blocks.
    one_swap = (
        "Alpha block two.\n\nAlpha block one.\n\nAlpha block three.\n\n"
        "Beta block one.\n\nBeta block two.\n\nBeta block three."
    )
    interleaved_score = reading_order_score(interleaved, gt)
    swap_score = reading_order_score(one_swap, gt)
    assert interleaved_score < swap_score < 1.0


def test_reading_order_penalizes_dropped_blocks():
    # Order preserved but half the blocks missing: a backend cannot get
    # full reading-order credit for blocks it never reproduced. This guards
    # the collapse case (multicolumn markitdown scoring a bogus 1.0).
    partial = "First paragraph about the introduction.\n\nSecond paragraph about the method."
    assert reading_order_score(partial, _MULTIBLOCK) < reading_order_score(
        _MULTIBLOCK, _MULTIBLOCK
    )


def test_reading_order_ignores_table_blocks():
    # Reading order measures PROSE flow; table structure is TEDS's job. A
    # table rendered with different spacing must not drag reading order down.
    gt = (
        "# Title\n\nIntro paragraph here.\n\n"
        "| A | B |\n|---|---|\n| 1 | 2 |\n\nClosing paragraph here."
    )
    pred = (
        "# Title\n\nIntro paragraph here.\n\n"
        "|A|B|\n|--|--|\n|1|2|\n\nClosing paragraph here."
    )
    assert reading_order_score(pred, gt) == 1.0


def test_reading_order_single_block_is_none():
    # One block -> no ordering to assess.
    assert reading_order_score("Just one paragraph.", "Just one paragraph.") is None


def test_reading_order_empty_prediction_is_zero():
    assert reading_order_score("", _MULTIBLOCK) == 0.0


# --------------------------------------------------------------------------
# score_document: the per-(fixture, backend) bundle the matrix is built from.
# --------------------------------------------------------------------------


def test_score_document_bundles_all_metrics():
    scored = score_document(_TABLE_GT, _TABLE_GT)
    assert scored["text"] == 1.0
    assert scored["table"] == 1.0
    # _TABLE_GT is a single table block -> reading order is N/A.
    assert scored["reading_order"] is None


def test_score_document_table_none_for_prose():
    scored = score_document(_MULTIBLOCK, _MULTIBLOCK)
    assert scored["table"] is None
    assert scored["text"] == 1.0
    assert scored["reading_order"] == 1.0


# --------------------------------------------------------------------------
# Backend matrix driver — runs in CI with markitdown-only installed.
# These tests double as the executable proof of the harness's whole reason
# for existing: markitdown has no OCR, so it scores zero on scanned/image.
# --------------------------------------------------------------------------


def test_markitdown_reads_digital_pdf_but_scores_zero_on_scanned():
    res = evaluate(_FIXTURES, backends=["markitdown"])
    by_id = {f["id"]: f for f in res["fixtures"]}

    digital = by_id["memo_digital_pdf"]["results"]["markitdown"]
    assert digital["status"] == "ok"
    assert digital["scores"]["text"] > 0.9

    # Image-only PDF: markitdown ran fine but produced nothing (no OCR).
    scanned = by_id["memo_scanned_pdf"]["results"]["markitdown"]
    assert scanned["status"] == "ok"
    assert scanned["scores"]["text"] == 0.0


def test_evaluate_marks_unavailable_backend_without_crashing():
    # llamaparse needs LLAMA_CLOUD_API_KEY + the cloud lib; absent in CI.
    res = evaluate(_FIXTURES, backends=["llamaparse"])
    first = res["fixtures"][0]["results"]["llamaparse"]
    assert first["status"] == "unavailable"
    assert first["scores"] is None


def test_evaluate_covers_every_fixture():
    res = evaluate(_FIXTURES, backends=["markitdown"])
    # The harness must score the whole corpus, not silently drop fixtures.
    assert len(res["fixtures"]) >= 15
    assert all("doc_class" in f and f["results"] for f in res["fixtures"])


def test_evaluate_records_provenance():
    # Anti-rot: the matrix must carry what produced it so staleness is visible.
    res = evaluate(_FIXTURES, backends=["markitdown"])
    prov = res["provenance"]
    assert prov["n_fixtures"] >= 15
    assert len(prov["fixture_set_sha256"]) == 16  # corpus identity, not a literal
    assert "markitdown" in prov["backend_versions"]


def test_evaluate_and_aggregate_emit_latency():
    # The cost axis the router needs to escalate-not-default.
    res = evaluate(_FIXTURES, backends=["markitdown"])
    assert "latency_ms" in res["fixtures"][0]["results"]["markitdown"]
    agg = aggregate(res)
    assert "markitdown" in agg["latency"]["overall"]


def test_provenance_records_docling_ocr_engine():
    # Anti-rot (MYC-1671): docling's scanned/image fidelity depends on its OCR
    # engine, so the matrix records which engine produced the docling column.
    # Asserts the relationship (engine is recorded + a valid value), not a
    # literal — the resolved engine differs by host (tesseract binary present
    # or not) and must not be a change-detector.
    res = evaluate(_FIXTURES, backends=["markitdown", "docling"])
    assert res["provenance"]["docling_ocr_engine"] in {"tesseract", "auto"}


def test_provenance_omits_ocr_engine_when_docling_not_run():
    # A markitdown-only run never touches docling, so it records no OCR engine.
    res = evaluate(_FIXTURES, backends=["markitdown"])
    assert "docling_ocr_engine" not in res["provenance"]
