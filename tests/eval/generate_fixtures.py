"""Generate the parse-fidelity fixture corpus.

One-time / re-runnable generator. It is NOT run in CI and NOT imported by
the test suite — it needs ``fpdf2`` + ``Pillow``, which the scorer tests
do not. Each authored document model is rendered into several
document-class artifacts (digital text PDF, scanned/image-only PDF,
PNG/JPG image, multi-column PDF), and the ground-truth markdown is derived
from the SAME model, so the rendered bytes and the ground truth cannot
drift apart.

    python tests/eval/generate_fixtures.py

Writes:
    tests/eval/fixtures/<id>.<ext>             rendered document
    tests/eval/fixtures/ground_truth/<doc>.md  derived ground truth
    tests/eval/fixtures/manifest.json          fixture index for the runner

Content is intentionally ASCII so the PDF core fonts (latin-1) render it
without substitution, and synthetic so there is zero copyright/PII risk.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
GT_DIR = FIXTURES / "ground_truth"

_ARIAL = "/System/Library/Fonts/Supplemental/Arial.ttf"
# Pin the embedded PDF creation date so regeneration is byte-reproducible.
_FIXED_DATE = datetime(2026, 1, 1, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# Document models. Blocks: ("h1"|"h2", text) | ("p", text)
#                          | ("table", header_list, rows_list)
# --------------------------------------------------------------------------

DOCUMENTS = {
    "office_memo": {
        "content": "prose",
        "blocks": [
            ("h1", "Internal Operations Memo"),
            ("p", "To: All engineering staff. From: Platform team. Date: March 2025."),
            ("p", "Effective next quarter, deployment freezes will apply during the "
                  "final week of each fiscal period. Plan releases accordingly and "
                  "coordinate with the release manager before the freeze begins."),
            ("p", "Questions about the new policy should be routed to the platform "
                  "channel. We will host an open office hour every Friday to walk "
                  "through the rollout and answer concerns."),
        ],
    },
    "quarterly_financials": {
        "content": "table",
        "blocks": [
            ("h1", "FY2025 Quarterly Financials"),
            ("p", "All figures in millions of USD unless otherwise noted. Margins "
                  "are gross and exclude one-time restructuring charges."),
            ("table",
             ["Quarter", "Revenue", "Gross Margin", "Headcount"],
             [["Q1", "12.4", "61%", "210"],
              ["Q2", "14.1", "63%", "225"],
              ["Q3", "15.8", "64%", "240"],
              ["Q4", "18.2", "66%", "262"]]),
        ],
    },
    "procurement_contract": {
        "content": "regulated",
        "blocks": [
            ("h1", "Master Services Agreement Extract"),
            ("h2", "Section 4. Service Levels"),
            ("p", "The Provider shall maintain availability of no less than 99.9 "
                  "percent measured monthly. Service credits accrue per the table "
                  "below when availability falls beneath the committed threshold."),
            ("table",
             ["Monthly Uptime", "Service Credit"],
             [["99.0 to 99.9 percent", "5 percent"],
              ["95.0 to 99.0 percent", "15 percent"],
              ["Below 95.0 percent", "30 percent"]]),
            ("p", "Section 5. Indemnification. Each party shall indemnify the other "
                  "against third-party claims arising from its gross negligence or "
                  "willful misconduct, subject to the limitations in Section 8."),
        ],
    },
    "engineering_review": {
        "content": "multicolumn",
        "blocks": [
            ("h1", "Quarterly Engineering Review"),
            ("p", "Reliability work dominated the quarter. The team cut p95 latency "
                  "by a third after replacing the legacy serializer on the hot path."),
            ("p", "The migration to the new store completed on schedule, with zero "
                  "customer-visible downtime during the cutover window."),
            ("p", "Security hardening landed across all public endpoints, including "
                  "stricter input validation and per-tenant rate limits."),
            ("p", "Documentation received a long-overdue refresh, and the onboarding "
                  "guide now reflects the current build pipeline."),
            ("p", "Hiring slowed deliberately to protect onboarding quality, and two "
                  "senior engineers joined the platform group this quarter."),
            ("p", "Looking ahead, the roadmap prioritizes multi-region readiness and "
                  "a deeper investment in automated evaluation."),
        ],
    },
    "vendor_invoice": {
        "content": "table",
        "blocks": [
            ("h1", "Vendor Invoice 2025-0417"),
            ("p", "Bill To: Acme Holdings. Terms: Net 30. Purchase Order: PO-88213."),
            ("table",
             ["Line Item", "Qty", "Unit Price", "Amount"],
             [["Annual platform license", "1", "24000", "24000"],
              ["Onboarding services", "40", "180", "7200"],
              ["Priority support tier", "12", "500", "6000"]]),
            ("p", "Total due: 37200 USD. Remit within thirty days of the invoice "
                  "date to avoid late fees per the agreed terms."),
        ],
    },
    "research_note": {
        "content": "mixed",
        "blocks": [
            ("h1", "Evaluation Note: Retrieval Quality"),
            ("p", "We compared three retrieval configurations on an internal "
                  "benchmark of 500 queries. Scores are NDCG at 5, higher is better."),
            ("table",
             ["Configuration", "NDCG@5", "Latency p95"],
             [["Baseline BM25", "0.41", "40 ms"],
              ["Dense only", "0.58", "90 ms"],
              ["Hybrid rerank", "0.67", "140 ms"]]),
            ("p", "The hybrid configuration won on quality at an acceptable latency "
                  "cost and is recommended for the next release."),
        ],
    },
}

# Fixture render plan: (fixture_id, source_doc, doc_class, render_kind, ext)
PLAN = [
    ("memo_digital_pdf",       "office_memo",          "digital_pdf",    "text_pdf",     "pdf"),
    ("memo_scanned_pdf",       "office_memo",          "scanned_pdf",    "image_pdf",    "pdf"),
    ("memo_image_png",         "office_memo",          "image",          "png",          "png"),
    ("memo_image_jpg",         "office_memo",          "image",          "jpg",          "jpg"),
    ("financials_digital_pdf", "quarterly_financials", "table_heavy",    "text_pdf",     "pdf"),
    ("financials_scanned_pdf", "quarterly_financials", "scanned_pdf",    "image_pdf",    "pdf"),
    ("financials_image_png",   "quarterly_financials", "image",          "png",          "png"),
    ("contract_digital_pdf",   "procurement_contract", "digital_pdf",    "text_pdf",     "pdf"),
    ("contract_scanned_pdf",   "procurement_contract", "scanned_pdf",    "image_pdf",    "pdf"),
    ("review_singlecol_pdf",   "engineering_review",    "digital_pdf",    "text_pdf",     "pdf"),
    ("review_multicolumn_pdf", "engineering_review",    "multicolumn",    "multicol_pdf", "pdf"),
    ("invoice_digital_pdf",    "vendor_invoice",        "table_heavy",    "text_pdf",     "pdf"),
    ("invoice_scanned_pdf",    "vendor_invoice",        "scanned_pdf",    "image_pdf",    "pdf"),
    ("invoice_image_png",      "vendor_invoice",        "image",          "png",          "png"),
    ("research_digital_pdf",   "research_note",         "digital_pdf",    "text_pdf",     "pdf"),
    ("research_scanned_pdf",   "research_note",         "scanned_pdf",    "image_pdf",    "pdf"),
]


# --------------------------------------------------------------------------
# Ground-truth markdown (derived from the model)
# --------------------------------------------------------------------------

def render_markdown(doc: dict) -> str:
    out: list[str] = []
    for block in doc["blocks"]:
        if block[0] == "h1":
            out.append(f"# {block[1]}")
        elif block[0] == "h2":
            out.append(f"## {block[1]}")
        elif block[0] == "p":
            out.append(block[1])
        elif block[0] == "table":
            header, rows = block[1], block[2]
            md = ["| " + " | ".join(header) + " |",
                  "| " + " | ".join("---" for _ in header) + " |"]
            for row in rows:
                md.append("| " + " | ".join(str(c) for c in row) + " |")
            out.append("\n".join(md))
    return "\n\n".join(out) + "\n"


def has_table(doc: dict) -> bool:
    return any(b[0] == "table" for b in doc["blocks"])


# --------------------------------------------------------------------------
# PDF renderers (text layer)
# --------------------------------------------------------------------------

def _new_pdf() -> FPDF:
    pdf = FPDF(format="letter", unit="mm")
    pdf.creation_date = _FIXED_DATE
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    return pdf


def render_text_pdf(doc: dict, path: Path) -> None:
    pdf = _new_pdf()
    epw = pdf.epw
    for block in doc["blocks"]:
        if block[0] == "h1":
            pdf.set_font("Helvetica", "B", 18)
            pdf.multi_cell(0, 9, text=block[1])
            pdf.ln(2)
        elif block[0] == "h2":
            pdf.set_font("Helvetica", "B", 14)
            pdf.multi_cell(0, 8, text=block[1])
            pdf.ln(1)
        elif block[0] == "p":
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, text=block[1])
            pdf.ln(2)
        elif block[0] == "table":
            header, rows = block[1], block[2]
            cw = epw / len(header)
            pdf.set_font("Helvetica", "B", 10)
            for cell in header:
                pdf.cell(cw, 8, text=cell, border=1)
            pdf.ln()
            pdf.set_font("Helvetica", "", 10)
            for row in rows:
                for cell in row:
                    pdf.cell(cw, 8, text=str(cell), border=1)
                pdf.ln()
            pdf.ln(2)
    pdf.output(str(path))


def render_multicolumn_pdf(doc: dict, path: Path) -> None:
    pdf = _new_pdf()
    pdf.set_auto_page_break(auto=False)
    title = next((b[1] for b in doc["blocks"] if b[0] == "h1"), None)
    if title:
        pdf.set_font("Helvetica", "B", 18)
        pdf.multi_cell(0, 9, text=title)
        pdf.ln(3)
    top = pdf.get_y()
    paras = [b[1] for b in doc["blocks"] if b[0] == "p"]
    gap = 8
    col_w = (pdf.epw - gap) / 2
    half = (len(paras) + 1) // 2
    left, right = paras[:half], paras[half:]
    left_x = pdf.l_margin
    right_x = pdf.l_margin + col_w + gap
    pdf.set_font("Helvetica", "", 11)
    pdf.set_xy(left_x, top)
    for para in left:
        pdf.set_x(left_x)
        pdf.multi_cell(col_w, 6, text=para)
        pdf.ln(2)
    pdf.set_xy(right_x, top)
    for para in right:
        pdf.set_x(right_x)
        pdf.multi_cell(col_w, 6, text=para)
        pdf.ln(2)
    pdf.output(str(path))


# --------------------------------------------------------------------------
# Image renderers (rasterized -> no text layer; the OCR path)
# --------------------------------------------------------------------------

def _font(size: int):
    try:
        return ImageFont.truetype(_ARIAL, size)
    except OSError:
        return ImageFont.load_default(size=size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, fnt, max_w: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if draw.textlength(trial, font=fnt) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def render_page_image(doc: dict, width: int = 1000, margin: int = 55) -> Image.Image:
    canvas = Image.new("RGB", (width, 4000), "white")
    draw = ImageDraw.Draw(canvas)
    x0 = margin
    max_w = width - 2 * margin
    y = margin
    for block in doc["blocks"]:
        if block[0] in ("h1", "h2"):
            fnt = _font(36 if block[0] == "h1" else 28)
            for line in _wrap(draw, block[1], fnt, max_w):
                draw.text((x0, y), line, fill="black", font=fnt)
                y += int(fnt.size * 1.3)
            y += 14
        elif block[0] == "p":
            fnt = _font(23)
            for line in _wrap(draw, block[1], fnt, max_w):
                draw.text((x0, y), line, fill="black", font=fnt)
                y += int(fnt.size * 1.35)
            y += 16
        elif block[0] == "table":
            header, rows = block[1], block[2]
            fnt = _font(22)
            cw = max_w / len(header)
            for i, cell in enumerate(header):
                draw.text((x0 + i * cw + 6, y), cell, fill="black", font=fnt)
            y += 36
            draw.line((x0, y, x0 + max_w, y), fill="black", width=2)
            y += 8
            for row in rows:
                for i, cell in enumerate(row):
                    draw.text((x0 + i * cw + 6, y), str(cell), fill="black", font=fnt)
                y += 34
            y += 20
    return canvas.crop((0, 0, width, min(y + margin, 4000)))


def render_image_pdf(doc: dict, path: Path) -> None:
    render_page_image(doc).save(str(path), "PDF", resolution=150.0)


def render_png(doc: dict, path: Path) -> None:
    render_page_image(doc).save(str(path), "PNG")


def render_jpg(doc: dict, path: Path) -> None:
    render_page_image(doc).save(str(path), "JPEG", quality=85)


RENDERERS = {
    "text_pdf": render_text_pdf,
    "multicol_pdf": render_multicolumn_pdf,
    "image_pdf": render_image_pdf,
    "png": render_png,
    "jpg": render_jpg,
}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    GT_DIR.mkdir(parents=True, exist_ok=True)

    # Ground truth: one file per source document.
    for doc_id, doc in DOCUMENTS.items():
        (GT_DIR / f"{doc_id}.md").write_text(render_markdown(doc), encoding="utf-8")

    manifest = {
        "generated_by": "tests/eval/generate_fixtures.py",
        "note": "Synthetic parse-fidelity corpus. Ground truth is derived from "
                "the same model that renders each fixture, so they cannot drift. "
                "Regenerate with: python tests/eval/generate_fixtures.py",
        "fixtures": [],
    }
    for fid, doc_id, doc_class, kind, ext in PLAN:
        doc = DOCUMENTS[doc_id]
        out = FIXTURES / f"{fid}.{ext}"
        RENDERERS[kind](doc, out)
        manifest["fixtures"].append({
            "id": fid,
            "file": out.name,
            "doc_class": doc_class,
            "content": doc["content"],
            "has_table": has_table(doc),
            "render": kind,
            "source_doc": doc_id,
            "ground_truth": f"ground_truth/{doc_id}.md",
        })
        print(f"  wrote {out.name}  ({doc_class}/{kind}, {out.stat().st_size} bytes)")

    (FIXTURES / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n{len(manifest['fixtures'])} fixtures + "
          f"{len(DOCUMENTS)} ground-truth docs -> {FIXTURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
