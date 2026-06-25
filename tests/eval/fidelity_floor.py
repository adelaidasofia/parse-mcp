"""Parse-fidelity FLOOR gate (MYC-1793) — make the MYC-1670 eval BITE.

MYC-1670 shipped the scorer + a committed ``parse_fidelity_matrix.json``;
MYC-1671 tuned docling against it. But nothing GATED it: ``make ci`` runs the
scorer's pure-Python unit tests under a markitdown-only install, so a docling
(or markitdown) version bump can change TableFormer / OCR behaviour and ship
with every fidelity check still green. A model/backend swap is a production
inference-path change and must be gated like one.

This module is that gate. Given a fresh evaluation (docling actually run over
the fixture corpus) and the committed baseline matrix, it asserts every gated
metric per doc-class is **>= baseline - epsilon**, and FAILS LOUD when the
backend did not actually run (a missing measurement is a failure, never a
silent pass — that is the whole anti-vacuity point).

Two halves:

* Pure-Python core (``check_floor`` / ``normalize_baseline``) — stdlib only,
  so its unit tests run in the base CI. ``normalize_baseline`` accepts BOTH
  committed-matrix shapes: parse-mcp's aggregate form and the raw evaluation
  report form (mrp's ``runtime_fidelity_matrix.json`` and any fresh
  ``evaluate()`` output), so the same logic gates both repos.
* CLI (``main``) — lazy-imports the scorer's ``evaluate`` (which needs docling)
  and runs the real gate. Invoked only by the dedicated parse-fidelity
  workflow, never by base CI.

Epsilon default 0.02 is calibrated against measurement: with Tesseract the
docling column reproduces the committed matrix exactly (0 jitter); removing
Tesseract drops scanned/image TEXT by ~0.045 — so 0.02 catches that real
regression with headroom while tolerating cross-host OCR jitter.
"""
from __future__ import annotations

import json
from pathlib import Path

# Gated metrics. text catches OCR regressions (Tesseract removed / engine
# swap); table TEDS catches TableFormer regressions (FAST mode / cell-matching
# off). reading-order stays 1.0 across measured detunes, so gating it would
# only add noise.
GATED_METRICS = ("text", "table")

# Floor tolerance below the committed baseline. See module docstring.
DEFAULT_EPSILON = 0.02


class FloorReport:
    """Outcome of a floor check.

    ``regressions`` — gated cells where current < baseline - epsilon.
    ``missing``     — gated cells the baseline has but the current run did NOT
                      produce (e.g. backend unavailable / errored). Treated as
                      failures: the gate cannot certify fidelity it did not
                      measure.
    ``checked``     — gated cells that were actually compared.
    """

    def __init__(self, regressions, checked, missing):
        self.regressions = regressions
        self.checked = checked
        self.missing = missing

    @property
    def ok(self) -> bool:
        return not self.regressions and not self.missing

    def report_str(self, *, epsilon: float) -> str:
        lines = []
        if self.ok:
            lines.append(
                f"PASS — {len(self.checked)} gated cell(s) at or above "
                f"baseline - {epsilon:.3f}."
            )
            return "\n".join(lines)
        lines.append(
            f"FAIL — {len(self.regressions)} regression(s), "
            f"{len(self.missing)} missing measurement(s) "
            f"(epsilon={epsilon:.3f})."
        )
        for r in self.regressions:
            lines.append(
                f"  REGRESSION {r['doc_class']}/{r['metric']}: "
                f"current {r['current']:.4f} < floor {r['floor']:.4f} "
                f"(baseline {r['baseline']:.4f}, drop {r['drop']:.4f})"
            )
        for cls, metric in self.missing:
            lines.append(
                f"  MISSING {cls}/{metric}: baseline has it but the run "
                f"produced no measurement (backend unavailable or errored)."
            )
        return "\n".join(lines)


def check_floor(current, baseline, *, epsilon=DEFAULT_EPSILON, metrics=GATED_METRICS):
    """Compare a fresh per-(class, metric) score map against a baseline map.

    ``current`` / ``baseline`` are ``{(doc_class, metric): float}``. A cell
    regresses when ``current < baseline - epsilon``. A baseline cell with no
    current counterpart is a *missing* failure (fail-loud), NOT a skip.
    Baseline cells valued ``None`` (metric N/A for that class) are not gated.
    """
    regressions, checked, missing = [], [], []
    for (cls, metric), base in sorted(baseline.items()):
        if metric not in metrics or base is None:
            continue
        cur = current.get((cls, metric))
        if cur is None:
            missing.append((cls, metric))
            continue
        checked.append((cls, metric))
        floor = base - epsilon
        if cur < floor:
            regressions.append(
                {
                    "doc_class": cls,
                    "metric": metric,
                    "baseline": base,
                    "current": cur,
                    "floor": floor,
                    "drop": base - cur,
                }
            )
    return FloorReport(regressions, checked, missing)


def _coerce(value):
    """Unwrap an aggregate ``(mean, n)`` pair to its mean; pass floats through."""
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def normalize_baseline(matrix_json: dict, backend: str, metrics=GATED_METRICS) -> dict:
    """Extract ``{(doc_class, metric): mean}`` for ``backend`` from a matrix.

    Accepts both shapes the codebase commits / emits:

    * **aggregate** (parse-mcp ``parse_fidelity_matrix.json``): a top-level
      ``"aggregate"`` with ``by_class[cls][backend][metric] == (mean, n)``.
    * **report** (mrp ``runtime_fidelity_matrix.json`` and any fresh
      ``evaluate()`` output): a top-level ``"fixtures"`` list of per-document
      ``results[backend].scores`` — means are recomputed per doc-class here.

    A run where ``backend`` is unavailable / errored contributes no cells, so
    the returned map is empty and ``check_floor`` against a real baseline fails
    loud rather than passing vacuously.
    """
    if "aggregate" in matrix_json:
        agg = matrix_json["aggregate"]
        out = {}
        for cls in agg.get("classes", []):
            cell = agg.get("by_class", {}).get(cls, {}).get(backend, {})
            for m in metrics:
                val = _coerce(cell.get(m))
                if val is not None:
                    out[(cls, m)] = val
        return out
    if "fixtures" in matrix_json:
        acc: dict[tuple, list] = {}
        for fx in matrix_json["fixtures"]:
            res = fx.get("results", {}).get(backend, {})
            scores = res.get("scores")
            if not scores:
                continue
            for m in metrics:
                v = scores.get(m)
                if v is not None:
                    acc.setdefault((fx["doc_class"], m), []).append(v)
        return {k: sum(v) / len(v) for k, v in acc.items()}
    raise ValueError(
        "unrecognized matrix shape: expected a top-level 'aggregate' "
        "(parse-mcp) or 'fixtures' (report) key"
    )


def load_matrix(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# CLI — the real gate. Lazy-imports the scorer's evaluate (needs docling).
# --------------------------------------------------------------------------


def _run_evaluation(fixtures_dir, backend):
    """Run the parse-mcp scorer's evaluate over the corpus for one backend.

    Returns ``(report_dict, provenance_dict)``. Isolated so the mrp copy of
    this module overrides only this one function (it drives the runtime's own
    backends via generate_runtime_matrix.evaluate instead).
    """
    import sys

    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from tests.eval.score_parse_fidelity import evaluate  # noqa: PLC0415

    report = evaluate(fixtures_dir, backends=[backend])
    return report, report.get("provenance", {})


def main(argv=None) -> int:
    import argparse

    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="Gate parse fidelity against the committed baseline matrix."
    )
    ap.add_argument("--matrix", default=str(here / "parse_fidelity_matrix.json"))
    ap.add_argument("--fixtures-dir", default=str(here / "fixtures"))
    ap.add_argument("--backend", default="docling", help="backend to RUN now")
    # The committed-matrix column the floor is read from. Defaults to --backend
    # (the real gate: docling-now vs docling-baseline). A negative control sets
    # it explicitly, e.g. --backend markitdown --baseline-backend docling, to
    # assert a weaker pipeline cannot clear the docling floor — comparing a
    # backend against its OWN column is vacuous (it just reproduces itself).
    ap.add_argument("--baseline-backend", default=None)
    ap.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    ap.add_argument("--metrics", nargs="*", default=list(GATED_METRICS))
    args = ap.parse_args(argv)

    metrics = tuple(args.metrics)
    baseline_backend = args.baseline_backend or args.backend
    baseline = normalize_baseline(load_matrix(args.matrix), baseline_backend, metrics)
    if not baseline:
        print(
            f"FAIL — baseline matrix {args.matrix} has no gated cells for "
            f"backend '{baseline_backend}'. Nothing to gate against.",
            flush=True,
        )
        return 1

    report, prov = _run_evaluation(args.fixtures_dir, args.backend)
    current = normalize_baseline(report, args.backend, metrics)

    engine = prov.get("docling_ocr_engine")
    versions = prov.get("backend_versions", {})
    print(
        f"parse-fidelity floor — run={args.backend} "
        f"version={versions.get(args.backend)} ocr_engine={engine} "
        f"vs baseline={baseline_backend} epsilon={args.epsilon}",
        flush=True,
    )
    result = check_floor(current, baseline, epsilon=args.epsilon, metrics=metrics)
    print(result.report_str(epsilon=args.epsilon), flush=True)
    # Per-cell gauge: current vs baseline vs floor for every gated cell, so a
    # passing run still shows its jitter margin (and a regression is legible).
    for (cls, metric), base in sorted(baseline.items()):
        if metric not in metrics or base is None:
            continue
        cur = current.get((cls, metric))
        cur_s = f"{cur:.4f}" if cur is not None else "MISSING"
        ok = cur is not None and cur >= base - args.epsilon
        print(
            f"  {cls}/{metric}: current={cur_s} baseline={base:.4f} "
            f"floor={base - args.epsilon:.4f} [{'ok' if ok else 'FAIL'}]",
            flush=True,
        )
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
