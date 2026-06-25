"""Tests for the parse-fidelity FLOOR gate (MYC-1793).

The floor gate is the half that makes the MYC-1670 eval BITE: it asserts a
fresh docling run scores at or above the committed baseline (minus epsilon)
per doc-class, and FAILS LOUD when docling did not actually run. These tests
are pure-Python (no docling) so they run under the markitdown-only base CI;
the real-pipeline + Tesseract-removed negative control run in the dedicated
parse-fidelity workflow (.github/workflows/parse-fidelity-floor.yml).
"""
from __future__ import annotations

import pytest

from tests.eval.fidelity_floor import (
    GATED_METRICS,
    check_floor,
    normalize_baseline,
)

# A tiny baseline resembling the committed docling column (real measured values).
_BASE = {
    ("scanned_pdf", "text"): 0.915,
    ("image", "text"): 0.941,
    ("scanned_pdf", "table"): 0.989,
    ("image", "table"): 0.895,
}


def test_gated_metrics_are_text_and_table():
    # Reading-order is intentionally NOT gated: it stays 1.0 across all
    # measured detunes, so it would only add noise to the floor.
    assert GATED_METRICS == ("text", "table")


def test_passes_when_current_equals_baseline():
    r = check_floor(_BASE, _BASE, epsilon=0.02)
    assert r.ok
    assert not r.regressions
    assert not r.missing


def test_passes_within_epsilon():
    cur = {k: v - 0.01 for k, v in _BASE.items()}  # 0.01 drop, eps 0.02
    assert check_floor(cur, _BASE, epsilon=0.02).ok


def test_boundary_exactly_epsilon_is_ok():
    # Floor is baseline - epsilon; landing exactly on it must pass (>=).
    cur = {k: round(v - 0.02, 6) for k, v in _BASE.items()}
    assert check_floor(cur, _BASE, epsilon=0.02).ok


def test_flags_regression_beyond_epsilon():
    cur = dict(_BASE)
    cur[("scanned_pdf", "text")] = 0.868  # the measured no-Tesseract value
    r = check_floor(cur, _BASE, epsilon=0.02)
    assert not r.ok
    regs = {(x["doc_class"], x["metric"]) for x in r.regressions}
    assert ("scanned_pdf", "text") in regs
    assert ("image", "text") not in regs  # unchanged cell not flagged


def test_fail_loud_when_measurement_missing():
    # docling unavailable -> current lacks the cell -> MUST fail, not silent pass.
    cur = dict(_BASE)
    del cur[("scanned_pdf", "text")]
    r = check_floor(cur, _BASE, epsilon=0.02)
    assert not r.ok
    assert ("scanned_pdf", "text") in r.missing


def test_only_gated_metrics_checked():
    base = dict(_BASE)
    base[("scanned_pdf", "reading_order")] = 1.0
    cur = dict(_BASE)
    cur[("scanned_pdf", "reading_order")] = 0.0  # tanked, but not a gated metric
    r = check_floor(cur, base, epsilon=0.02, metrics=("text", "table"))
    assert r.ok  # reading_order ignored


def test_none_baseline_cells_skipped():
    base = dict(_BASE)
    base[("multicolumn", "table")] = None  # N/A in the committed matrix
    cur = dict(_BASE)  # no multicolumn-table measurement at all
    r = check_floor(cur, base, epsilon=0.02)
    assert r.ok  # None baseline -> not gated, and not counted as "missing"


# --- normalize_baseline: both committed-matrix shapes + a fresh report ---


def test_normalize_aggregate_shape():
    # parse-mcp committed shape: {"aggregate": {"classes", "by_class": {cls:{backend:{metric:(mean,n)}}}}}
    mj = {
        "aggregate": {
            "classes": ["scanned_pdf"],
            "by_class": {
                "scanned_pdf": {
                    "docling": {
                        "text": (0.915, 5),
                        "table": (0.989, 4),
                        "reading_order": (1.0, 5),
                    }
                }
            },
        }
    }
    norm = normalize_baseline(mj, "docling")
    assert norm[("scanned_pdf", "text")] == 0.915
    assert norm[("scanned_pdf", "table")] == 0.989
    assert ("scanned_pdf", "reading_order") not in norm  # not gated by default


def test_normalize_extracts_named_backend_column():
    # The floor is read from a SPECIFIC backend column. The negative control
    # relies on this: markitdown gated against the docling column (not its own
    # ~0 column, which would pass vacuously).
    mj = {
        "aggregate": {
            "classes": ["scanned_pdf"],
            "by_class": {
                "scanned_pdf": {
                    "docling": {"text": (0.915, 5)},
                    "markitdown": {"text": (0.0, 5)},
                }
            },
        }
    }
    assert normalize_baseline(mj, "docling")[("scanned_pdf", "text")] == 0.915
    assert normalize_baseline(mj, "markitdown")[("scanned_pdf", "text")] == 0.0
    # markitdown's ~0 current cannot clear the docling floor -> regression.
    current = {("scanned_pdf", "text"): 0.0}
    docling_floor = normalize_baseline(mj, "docling")
    assert not check_floor(current, docling_floor, epsilon=0.02).ok


def test_normalize_report_shape_recomputes_means():
    # mrp committed shape (and ANY fresh evaluate() report): {"fixtures": [...]}
    rep = {
        "fixtures": [
            {
                "doc_class": "scanned_pdf",
                "results": {
                    "docling": {"status": "ok", "scores": {"text": 0.90, "table": 1.0, "reading_order": 1.0}}
                },
            },
            {
                "doc_class": "scanned_pdf",
                "results": {
                    "docling": {"status": "ok", "scores": {"text": 0.92, "table": 0.98, "reading_order": 1.0}}
                },
            },
        ]
    }
    norm = normalize_baseline(rep, "docling")
    assert abs(norm[("scanned_pdf", "text")] - 0.91) < 1e-9
    assert abs(norm[("scanned_pdf", "table")] - 0.99) < 1e-9


def test_normalize_unavailable_backend_yields_no_cells():
    # A fresh report where docling is unavailable -> empty -> check_floor fails loud.
    rep = {
        "fixtures": [
            {"doc_class": "scanned_pdf", "results": {"docling": {"status": "unavailable", "scores": None}}}
        ]
    }
    norm = normalize_baseline(rep, "docling")
    assert norm == {}
    assert not check_floor(norm, _BASE, epsilon=0.02).ok  # fail loud, not vacuous pass


def test_normalize_rejects_unknown_shape():
    with pytest.raises(ValueError):
        normalize_baseline({"weird": 1}, "docling")
