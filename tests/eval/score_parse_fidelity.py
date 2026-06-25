"""Parse-fidelity scorer + backend matrix driver.

The router picks a backend per document from a static format-preference
table (``router._FORMAT_PREFERENCE``) that carries no quality data. This
module replaces that guesswork with measurement: score each backend's
markdown output against ground-truth markdown for a fixture set spanning
document classes, then emit a ``backend x doc-class x metric`` matrix that
shows which backend actually wins per class — the evidence base for tuning
the routing table.

Metric definitions follow OmniDocBench / PubTabNet:

* **Text** — normalized edit distance (Levenshtein over whitespace-
  normalized text), reported here as a similarity ``1 - NED`` in [0, 1].
  Ref: OmniDocBench (Ouyang et al., CVPR 2025) "Text Edit Distance".
* **Table** — TEDS, Tree-Edit-Distance-based Similarity over the table's
  HTML tree: ``1 - TED / max(|T_pred|, |T_gt|)``. The tree edit distance
  is computed with Zhang-Shasha (Zhang & Shasha 1989), which returns the
  optimal ordered-tree edit distance — the same quantity APTED computes
  in the reference TEDS implementation, just without APTED's speedups
  (fine for table-sized trees). Ref: Zhong et al., "Image-based table
  recognition" (PubTabNet, ECCV 2020); OmniDocBench "Table TEDS".
* **Reading order** — OmniDocBench measures the edit distance between the
  predicted and ground-truth reading sequence of matched blocks. We have
  markdown, not layout bounding boxes, so we adapt: segment both docs
  into blocks, match predicted blocks to ground-truth blocks by text
  similarity, then take the normalized edit distance between the matched
  ground-truth index sequence (in prediction order) and its sorted order.
  Reported as a similarity ``1 - NED`` in [0, 1]. The adaptation is
  documented loudly because it is NOT bbox-based reading order.

All three metrics are reported as quality scores in [0, 1], higher is
better, so "which backend wins per doc-class" reads off the matrix
directly.

This module imports only the standard library at import time; the backend
runner lazy-imports ``router`` so the metric functions (and their tests)
stay backend-free and run under a markitdown-only CI install.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Text metric
# --------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def _normalize_text(s: str) -> str:
    """Collapse whitespace runs to single spaces and strip.

    OmniDocBench normalizes whitespace before computing text edit distance
    so that cosmetic spacing differences (a backend emitting double
    newlines vs single) do not count as content errors. Case is preserved
    — case IS content fidelity.
    """
    return _WS.sub(" ", s or "").strip()


def _levenshtein(a, b) -> int:
    """Levenshtein edit distance between two sequences (str or list).

    Iterative two-row dynamic program, O(len(a) * len(b)) time and
    O(len(b)) space. Works on any sequence of comparable, hashable items,
    so it backs both the character-level text metric and the block-index
    reading-order metric.
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + cost, # substitution
            )
        prev = cur
    return prev[lb]


def normalized_edit_distance(a: str, b: str) -> float:
    """Levenshtein(a, b) / max(len(a), len(b)) in [0, 1]. 0 == identical."""
    a = _normalize_text(a)
    b = _normalize_text(b)
    if not a and not b:
        return 0.0
    denom = max(len(a), len(b))
    if denom == 0:
        return 0.0
    return _levenshtein(a, b) / denom


def text_score(prediction_md: str, ground_truth_md: str) -> float:
    """Text fidelity as a similarity in [0, 1]. 1.0 == identical text."""
    return 1.0 - normalized_edit_distance(prediction_md, ground_truth_md)


# --------------------------------------------------------------------------
# Table metric: TEDS via Zhang-Shasha tree edit distance
# --------------------------------------------------------------------------


class Node:
    """An ordered-tree node. ``label`` is any comparable value.

    For TEDS the label is a ``(tag, text)`` tuple where ``tag`` is one of
    ``doc`` / ``table`` / ``tr`` / ``cell`` and ``text`` is the normalized
    cell content (empty for structural nodes).
    """

    __slots__ = ("label", "children")

    def __init__(self, label, children=None):
        self.label = label
        self.children = list(children) if children else []


def _count(node: Node) -> int:
    return 1 + sum(_count(c) for c in node.children)


def _postorder(root: Node):
    """Return (nodes, lmld): post-order node list + leftmost-leaf index map.

    ``lmld[k]`` is the post-order index of the leftmost-leaf descendant of
    the node at post-order index ``k`` — the Zhang-Shasha ``l()`` function.
    """
    nodes: list[Node] = []
    idx: dict[int, int] = {}

    stack = [(root, False)]
    while stack:
        node, processed = stack.pop()
        if processed:
            idx[id(node)] = len(nodes)
            nodes.append(node)
        else:
            stack.append((node, True))
            for child in reversed(node.children):
                stack.append((child, False))

    def leftmost(n: Node) -> Node:
        while n.children:
            n = n.children[0]
        return n

    lmld = [idx[id(leftmost(n))] for n in nodes]
    return nodes, lmld


def _keyroots(lmld) -> list[int]:
    seen: set[int] = set()
    kr: list[int] = []
    for k in range(len(lmld) - 1, -1, -1):
        if lmld[k] not in seen:
            seen.add(lmld[k])
            kr.append(k)
    return sorted(kr)


def _default_rename(a, b) -> float:
    return 0.0 if a == b else 1.0


def _tree_edit_distance(root1: Node, root2: Node, rename_cost=_default_rename) -> float:
    """Optimal ordered-tree edit distance (Zhang & Shasha 1989).

    Insertion and deletion each cost 1; substitution cost comes from
    ``rename_cost(label_a, label_b)``. Returns the same quantity APTED
    computes for ordered trees — TEDS is defined on this distance.
    """
    o1, lml1 = _postorder(root1)
    o2, lml2 = _postorder(root2)
    n1, n2 = len(o1), len(o2)
    treedist = [[0.0] * n2 for _ in range(n1)]

    for i in _keyroots(lml1):
        for j in _keyroots(lml2):
            li, lj = lml1[i], lml2[j]
            m, n = i - li + 2, j - lj + 2
            fd = [[0.0] * n for _ in range(m)]
            for x in range(1, m):
                fd[x][0] = fd[x - 1][0] + 1.0  # delete o1[li+x-1]
            for y in range(1, n):
                fd[0][y] = fd[0][y - 1] + 1.0  # insert o2[lj+y-1]
            for x in range(1, m):
                for y in range(1, n):
                    di, dj = li + x - 1, lj + y - 1
                    if lml1[di] == li and lml2[dj] == lj:
                        cost = min(
                            fd[x - 1][y] + 1.0,
                            fd[x][y - 1] + 1.0,
                            fd[x - 1][y - 1]
                            + rename_cost(o1[di].label, o2[dj].label),
                        )
                        fd[x][y] = cost
                        treedist[di][dj] = cost
                    else:
                        a = lml1[di] - li
                        b = lml2[dj] - lj
                        fd[x][y] = min(
                            fd[x - 1][y] + 1.0,
                            fd[x][y - 1] + 1.0,
                            fd[a][b] + treedist[di][dj],
                        )
    return treedist[n1 - 1][n2 - 1]


_SEP_ROW = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$")


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [_normalize_text(c) for c in s.split("|")]


def _extract_markdown_tables(md: str) -> list[Node]:
    """Parse GFM pipe tables into ``table -> tr -> cell`` trees.

    A table is a run of ``|``-bearing lines whose second line is a
    separator (``---``). Header and body cells are both tagged ``cell``:
    markdown cannot reliably convey ``th`` vs ``td``, so collapsing them
    keeps the metric from punishing a header-detection difference that is
    not a fidelity error. (Documented departure from PubTabNet th/td.)
    """
    lines = (md or "").splitlines()
    tables: list[Node] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "|" in line and i + 1 < len(lines) and _SEP_ROW.match(lines[i + 1]):
            rows = [lines[i]]
            j = i + 2
            while j < len(lines) and "|" in lines[j] and lines[j].strip():
                rows.append(lines[j])
                j += 1
            table = Node(("table", ""))
            header = _split_row(rows[0])
            table.children.append(
                Node(("tr", ""), [Node(("cell", c)) for c in header])
            )
            for body_line in rows[1:]:
                cells = _split_row(body_line)
                table.children.append(
                    Node(("tr", ""), [Node(("cell", c)) for c in cells])
                )
            tables.append(table)
            i = j
        else:
            i += 1
    return tables


def _table_forest(md: str) -> Node | None:
    tables = _extract_markdown_tables(md)
    if not tables:
        return None
    return Node(("doc", ""), tables)


def _teds_rename(a, b) -> float:
    tag_a, text_a = a
    tag_b, text_b = b
    if tag_a != tag_b:
        return 1.0
    if tag_a == "cell":
        return normalized_edit_distance(text_a, text_b)
    return 0.0


def teds(prediction_md: str, ground_truth_md: str):
    """Table TEDS in [0, 1], or ``None`` when the ground truth has no table.

    ``1 - TED / max(|T_pred|, |T_gt|)``. Returns ``None`` (N/A) for
    non-table fixtures, and ``0.0`` when the ground truth has a table but
    the prediction produced none.
    """
    gt_tree = _table_forest(ground_truth_md)
    if gt_tree is None:
        return None
    pred_tree = _table_forest(prediction_md)
    if pred_tree is None:
        return 0.0
    ted = _tree_edit_distance(pred_tree, gt_tree, rename_cost=_teds_rename)
    denom = max(_count(pred_tree), _count(gt_tree))
    if denom == 0:
        return 1.0
    return max(0.0, 1.0 - ted / denom)


# --------------------------------------------------------------------------
# Reading-order metric
# --------------------------------------------------------------------------

def _is_table_line(line: str) -> bool:
    """A single line that belongs to a (pipe) table, not prose."""
    return bool(_SEP_ROW.match(line)) or line.count("|") >= 2


def _prose_blocks(md: str) -> list[str]:
    """Reading-order segments: non-empty, non-table lines, normalized.

    Segmentation is line-level, not blank-line-level, because backends do
    not agree on block delimiters — markitdown emits single newlines
    between blocks on some PDFs while docling emits blank lines. Line-level
    segmentation makes the order comparison resilient to that. Table lines
    are excluded: table fidelity is measured by TEDS, so a differently
    rendered table must not be scored twice.
    """
    out: list[str] = []
    for raw in (md or "").splitlines():
        line = raw.strip()
        if not line or _is_table_line(line):
            continue
        norm = _normalize_text(line)
        if norm:
            out.append(norm)
    return out


def _best_match(block: str, gt_blocks: list[str]) -> tuple[int, float]:
    best_i, best_sim = -1, 0.0
    for i, g in enumerate(gt_blocks):
        sim = 1.0 - normalized_edit_distance(block, g)
        if sim > best_sim:
            best_i, best_sim = i, sim
    return best_i, best_sim


def reading_order_score(
    prediction_md: str, ground_truth_md: str, *, match_threshold: float = 0.5
):
    """Reading-order fidelity as a similarity in [0, 1], or ``None`` (N/A).

    Adaptation of OmniDocBench's reading-order edit distance to markdown
    (no layout bounding boxes). It is ground-truth-anchored so a backend
    that collapses the document into one garbled block cannot score a
    spurious 1.0:

    1. Segment both docs into blocks (blank-line split).
    2. For each ground-truth block, locate its best-matching predicted
       block (the position where it surfaced in the prediction). A block
       is "located" only above ``match_threshold`` similarity — a long
       collapsed block fails the length-sensitive similarity check and so
       is not credited.
    3. ``order_sim`` = ``1 - NED`` between the located ground-truth indices
       (ordered by where they surfaced) and their correct sorted order.
    4. Weight by ``coverage`` = located / total ground-truth blocks. You
       cannot preserve the reading order of blocks you never reproduced,
       so a structural collapse is penalized, not rewarded.

    Returns ``None`` when the ground truth has fewer than two blocks (no
    order to assess) and ``0.0`` when nothing is located.
    """
    gt_blocks = _prose_blocks(ground_truth_md)
    if len(gt_blocks) < 2:
        return None
    pred_blocks = _prose_blocks(prediction_md)
    if not pred_blocks:
        return 0.0
    located: list[tuple[int, int]] = []  # (gt_index, pred_position)
    for gt_index, block in enumerate(gt_blocks):
        pred_pos, sim = _best_match(block, pred_blocks)
        if pred_pos >= 0 and sim >= match_threshold:
            located.append((gt_index, pred_pos))
    if not located:
        return 0.0
    observed = [gi for gi, _ in sorted(located, key=lambda t: t[1])]
    target = sorted(observed)
    order_sim = 1.0 - _levenshtein(observed, target) / len(observed)
    coverage = len(located) / len(gt_blocks)
    return max(0.0, order_sim * coverage)


# --------------------------------------------------------------------------
# Per-document bundle
# --------------------------------------------------------------------------


def score_document(prediction_md: str, ground_truth_md: str) -> dict:
    """Score one prediction against ground truth across all three metrics.

    Values are quality scores in [0, 1] (higher is better) or ``None``
    when a metric does not apply (``table`` for a prose fixture,
    ``reading_order`` for a single-block fixture).
    """
    return {
        "text": text_score(prediction_md, ground_truth_md),
        "table": teds(prediction_md, ground_truth_md),
        "reading_order": reading_order_score(prediction_md, ground_truth_md),
    }


# --------------------------------------------------------------------------
# Backend matrix driver
#
# Runs each parse backend over the fixture corpus and scores its markdown
# against ground truth, then aggregates into a backend x doc-class x metric
# matrix. Backends that are not installed (or have no API key) are recorded
# as unavailable rather than crashing the run.
# --------------------------------------------------------------------------

DEFAULT_BACKENDS = ["markitdown", "docling", "llamaparse"]
METRICS = ["text", "table", "reading_order"]


def _ensure_repo_root_on_path() -> None:
    """Put the repo root on sys.path so ``import router`` works under the CLI.

    Under pytest the package layout already puts the root on the path; this
    only matters when the module is run directly as a script.
    """
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)


def _backend_module(name: str):
    from backends import (  # noqa: PLC0415 — lazy so metrics stay import-light
        docling_backend,
        llamaparse_backend,
        markitdown_backend,
    )

    return {
        "markitdown": markitdown_backend,
        "docling": docling_backend,
        "llamaparse": llamaparse_backend,
    }[name]


def _backend_version(name: str):
    import importlib.metadata as ilm

    pkg = {
        "markitdown": "markitdown",
        "docling": "docling",
        "llamaparse": "llama-cloud-services",
    }.get(name)
    if not pkg:
        return None
    try:
        return ilm.version(pkg)
    except Exception:
        return None


def _provenance(fixtures_dir, backends) -> dict:
    """Record what produced this matrix so it cannot rot unnoticed.

    Backend + python versions pin the toolchain; the fixture-set hash pins
    the corpus identity. A backend upgrade or a corpus change moves these,
    flagging that the committed matrix must be regenerated.
    """
    import hashlib
    import platform

    fixtures_dir = Path(fixtures_dir)
    manifest_bytes = (fixtures_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    digest = hashlib.sha256(manifest_bytes)
    for fx in sorted(manifest["fixtures"], key=lambda x: x["id"]):
        digest.update((fixtures_dir / fx["file"]).read_bytes())
        digest.update((fixtures_dir / fx["ground_truth"]).read_bytes())
    return {
        "python": platform.python_version(),
        "backend_versions": {b: _backend_version(b) for b in backends},
        "fixture_set_sha256": digest.hexdigest()[:16],
        "n_fixtures": len(manifest["fixtures"]),
    }


def _run_one_backend(name: str, data: bytes, filename: str):
    """Return (markdown, status, latency_ms). status in {ok, unavailable, error}."""
    mod = _backend_module(name)
    if not mod.is_available():
        return "", "unavailable", 0
    result = mod.parse(data, filename=filename)
    return result.markdown, ("error" if result.error else "ok"), result.latency_ms


def evaluate(fixtures_dir, backends=None) -> dict:
    """Run ``backends`` over every fixture in ``fixtures_dir`` and score them.

    Returns ``{"backends": [...], "fixtures": [ {id, doc_class, content,
    has_table, file, results: {backend: {status, output_len, scores}}} ]}``.
    Unavailable backends are recorded with ``scores=None`` rather than
    crashing the run.
    """
    _ensure_repo_root_on_path()
    backends = list(backends) if backends else list(DEFAULT_BACKENDS)
    fixtures_dir = Path(fixtures_dir)
    manifest = json.loads((fixtures_dir / "manifest.json").read_text(encoding="utf-8"))

    gt_cache: dict[str, str] = {}
    out = {"backends": backends, "fixtures": []}
    for fx in manifest["fixtures"]:
        gt_path = fixtures_dir / fx["ground_truth"]
        gt = gt_cache.get(str(gt_path))
        if gt is None:
            gt = gt_path.read_text(encoding="utf-8")
            gt_cache[str(gt_path)] = gt
        data = (fixtures_dir / fx["file"]).read_bytes()

        entry = {k: fx[k] for k in ("id", "doc_class", "content", "has_table", "file")}
        entry["results"] = {}
        for name in backends:
            markdown, status, latency_ms = _run_one_backend(name, data, fx["file"])
            entry["results"][name] = {
                "status": status,
                "output_len": len(markdown),
                "latency_ms": latency_ms,
                "scores": score_document(markdown, gt) if status == "ok" else None,
            }
        out["fixtures"].append(entry)
    out["provenance"] = _provenance(fixtures_dir, backends)
    return out


def _mean(values: list[float]):
    return sum(values) / len(values) if values else None


def _median(values: list[float]):
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def aggregate(evaluation: dict) -> dict:
    """Mean each metric over (backend, doc-class) and overall.

    Cell value is ``(mean_or_None, n_contributing_fixtures)``.
    """
    backends = evaluation["backends"]
    classes: list[str] = []
    for fx in evaluation["fixtures"]:
        if fx["doc_class"] not in classes:
            classes.append(fx["doc_class"])

    def collect(predicate):
        table = {}
        for backend in backends:
            table[backend] = {}
            for metric in METRICS:
                vals = []
                for fx in evaluation["fixtures"]:
                    if not predicate(fx):
                        continue
                    res = fx["results"].get(backend)
                    if not res or res["scores"] is None:
                        continue
                    val = res["scores"].get(metric)
                    if val is not None:
                        vals.append(val)
                table[backend][metric] = (_mean(vals), len(vals))
        return table

    def latency_collect(predicate):
        out_l = {}
        for backend in backends:
            vals = []
            for fx in evaluation["fixtures"]:
                if not predicate(fx):
                    continue
                res = fx["results"].get(backend)
                if res and res["status"] == "ok":
                    vals.append(res.get("latency_ms", 0))
            out_l[backend] = _median(vals)
        return out_l

    by_class = {c: collect(lambda fx, c=c: fx["doc_class"] == c) for c in classes}
    overall = collect(lambda fx: True)
    latency = {
        "overall": latency_collect(lambda fx: True),
        "by_class": {
            c: latency_collect(lambda fx, c=c: fx["doc_class"] == c) for c in classes
        },
    }
    return {
        "classes": classes,
        "by_class": by_class,
        "overall": overall,
        "latency": latency,
    }


def compute_findings(evaluation: dict, agg: dict) -> dict:
    """Derive the per-doc-class winning backend (routing guidance) from data."""
    backends = evaluation["backends"]
    best_per_class = {}
    for c in agg["classes"]:
        composites = {}
        for b in backends:
            means = [
                agg["by_class"][c][b][m][0]
                for m in METRICS
                if agg["by_class"][c][b][m][0] is not None
            ]
            if means:
                composites[b] = sum(means) / len(means)
        if composites:
            winner = max(composites, key=composites.get)
            best_per_class[c] = {"winner": winner, "composites": composites}

    return {"best_per_class": best_per_class}


# --------------------------------------------------------------------------
# Matrix rendering
# --------------------------------------------------------------------------

def _fmt(pair) -> str:
    value = pair[0] if isinstance(pair, tuple) else pair
    return f"{value:.3f}" if value is not None else "—"


def _metric_table(evaluation: dict, agg: dict, metric: str) -> str:
    backends = evaluation["backends"]
    head = "| doc-class | " + " | ".join(backends) + " |"
    sep = "|" + "---|" * (len(backends) + 1)
    rows = [head, sep]
    for c in agg["classes"]:
        cells = [_fmt(agg["by_class"][c][b][metric]) for b in backends]
        rows.append("| " + c + " | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _provenance_line(evaluation: dict) -> str:
    prov = evaluation.get("provenance") or {}
    versions = prov.get("backend_versions", {})
    vstr = ", ".join(f"{b} {v}" for b, v in versions.items() if v) or "n/a"
    return (
        f"**Provenance.** python {prov.get('python', '?')} · backends: {vstr} · "
        f"fixture-set `{prov.get('fixture_set_sha256', '?')}` "
        f"({prov.get('n_fixtures', '?')} docs). Regenerate when any backend "
        "version changes — a stale matrix lies silently."
    )


def render_matrix_md(evaluation: dict, agg: dict) -> str:
    backends = evaluation["backends"]
    findings = compute_findings(evaluation, agg)
    n_fix = len(evaluation["fixtures"])

    avail = {}
    for b in backends:
        avail[b] = any(
            fx["results"][b]["status"] == "ok" for fx in evaluation["fixtures"]
        )

    lines = [
        "# Parse-fidelity matrix",
        "",
        "_Generated by `tests/eval/score_parse_fidelity.py`. "
        "Regenerate: `python tests/eval/score_parse_fidelity.py`._",
        "",
        f"Corpus: **{n_fix} fixtures** across {len(agg['classes'])} document "
        "classes, scored against derived ground-truth markdown. All scores are "
        "quality in [0, 1], higher is better. `—` = metric not applicable "
        "(no table / single block) or backend unavailable.",
        "",
        "Metrics: **text** = 1 − normalized edit distance; **table** = TEDS "
        "(Tree-Edit-Distance similarity); **reading-order** = 1 − normalized "
        "block-order edit distance. Definitions follow OmniDocBench / PubTabNet "
        "(see the scorer module docstring).",
        "",
        _provenance_line(evaluation),
        "",
        "## Backend availability in this run",
        "",
    ]
    for b in backends:
        note = ""
        if b == "llamaparse" and not avail[b]:
            note = " (needs `LLAMA_CLOUD_API_KEY` + `llama-cloud-services`)"
        lines.append(f"- `{b}`: {'available' if avail[b] else 'UNAVAILABLE'}{note}")

    lines += ["", "## Text fidelity (1 − NED) by doc-class", "",
              _metric_table(evaluation, agg, "text"),
              "", "## Table fidelity (TEDS) by doc-class", "",
              _metric_table(evaluation, agg, "table"),
              "", "## Reading-order fidelity by doc-class", "",
              _metric_table(evaluation, agg, "reading_order"),
              ""]

    # Overall per-backend means + median latency. Latency is the cost axis:
    # highest fidelity is not a free default — docling is far slower than
    # markitdown, so route cheap-first and escalate, do not default to the winner.
    lat = agg.get("latency", {}).get("overall", {})
    lines += ["## Overall (mean across all applicable fixtures)", "",
              "| backend | text | table TEDS | reading-order | median latency (ms) |",
              "|---|---|---|---|---|"]
    for b in backends:
        o = agg["overall"][b]
        lm = lat.get(b)
        lat_str = str(int(lm)) if lm is not None else "—"
        lines.append(
            f"| {b} | {_fmt(o['text'])} | {_fmt(o['table'])} | "
            f"{_fmt(o['reading_order'])} | {lat_str} |"
        )

    # Findings: which backend to prefer per doc-class (informs the router's
    # `_FORMAT_PREFERENCE` table).
    lines += ["", "## Findings: routing per doc-class", "",
              "Best backend by composite (mean of applicable metrics). This is "
              "the evidence base for the router's format-preference table.", ""]
    for c in agg["classes"]:
        bp = findings["best_per_class"].get(c)
        if not bp:
            continue
        ranked = sorted(bp["composites"].items(), key=lambda kv: kv[1], reverse=True)
        ranked_str = ", ".join(f"{b} {v:.3f}" for b, v in ranked)
        lines.append(f"- **{c}** → best: `{bp['winner']}`  (composite: {ranked_str})")
    lines.append("")

    # Per-fixture text appendix.
    lines += ["## Per-fixture text fidelity", "",
              "| fixture | doc-class | " + " | ".join(backends) + " |",
              "|" + "---|" * (len(backends) + 2)]
    for fx in evaluation["fixtures"]:
        cells = []
        for b in backends:
            res = fx["results"][b]
            if res["status"] != "ok" or res["scores"] is None:
                cells.append("—")
            else:
                cells.append(f"{res['scores']['text']:.3f}")
        lines.append(f"| {fx['id']} | {fx['doc_class']} | " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    import argparse

    _ensure_repo_root_on_path()
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Score parse backends -> fidelity matrix")
    ap.add_argument("--fixtures-dir", default=str(here / "fixtures"))
    ap.add_argument("--out-dir", default=str(here))
    ap.add_argument("--backends", nargs="*", default=None)
    args = ap.parse_args(argv)

    evaluation = evaluate(args.fixtures_dir, backends=args.backends)
    agg = aggregate(evaluation)
    md = render_matrix_md(evaluation, agg)

    out = Path(args.out_dir)
    (out / "parse_fidelity_matrix.md").write_text(md, encoding="utf-8")
    (out / "parse_fidelity_matrix.json").write_text(
        json.dumps(
            {"evaluation": evaluation, "aggregate": agg,
             "findings": compute_findings(evaluation, agg)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
