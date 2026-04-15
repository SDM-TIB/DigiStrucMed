"""
Step 1b — Table Extraction
─────────────────────────────────────────────────────────────────────────────
Input  : PDF file path
Output : <output_dir>/tables/<table_id>.csv   (one CSV per table)
         <output_dir>/table_index.json
         Each index record: { table_id, page, source_file, csv_path,
                               caption, headers, row_count }

Versions
--------
v1 (default) — pdfplumber (table detection) + PyMuPDF (bbox caption lookup)
  • Finds table regions with pdfplumber's layout engine
  • Matches caption text blocks above/below each table using bounding boxes
  • Best for simple PDF layouts with clear table borders

v2 — Docling
  • Uses Docling's document model to detect and extract tables structurally
  • Docling understands nested headers, merged cells, and complex layouts
  • Should be used together with extract_text version="v2" so both steps share the
    same Docling-parsed document representation
  • Caption comes from Docling's table.label / table.caption attributes

Use matching versions for extract_text and extract_tables:
  extract_text v1 + extract_tables v1   (PyMuPDF text + pdfplumber tables)
  extract_text v2 + extract_tables v2   (Docling text + Docling tables)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz        # PyMuPDF — used for bbox positions and caption lookup (v1)
import pdfplumber  # table detection (v1)

if __package__ in (None, ""):
    import sys

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from pipeline.step1.utils import log, save_json


# ── Shared constants ─────────────────────────────────────────────────────────

_CAPTION_STRICT = re.compile(
    r"^\s*(?:Table|TABLE)\s+\d+[.\s—:\-]+(.+)$",
    re.IGNORECASE,
)
_CAPTION_LOOSE = re.compile(
    r"^\s*(?:Table|TABLE)\s+\d+[.\s—:\-].+",
    re.IGNORECASE | re.DOTALL,
)

_CAPTION_TOLERANCE    = 25.0
_CAPTION_BELOW_MARGIN = 120.0
_CAPTION_ABOVE_MARGIN = 220.0
_MIN_TABLE_CHARS      = 50


# ── v1 helpers: PyMuPDF + pdfplumber ────────────────────────────────────────

def _get_page_blocks(page) -> List[Dict]:
    blocks = []
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, txt, *_ = b
        if txt and str(txt).strip():
            blocks.append({
                "x0": float(x0), "y0": float(y0),
                "x1": float(x1), "y1": float(y1),
                "text": str(txt),
            })
    return blocks


def _find_caption(
    table_bbox: Tuple[float, float, float, float],
    page_blocks: List[Dict],
) -> str:
    """
    Locate the caption for a table by searching PyMuPDF text blocks above and
    below the table's bounding box.  Priority: strict match above → strict below
    → loose match closest above.
    """
    tx0, ttop, tx1, tbottom = table_bbox
    tol = _CAPTION_TOLERANCE

    def is_above(blk: Dict) -> bool:
        return blk["y1"] <= ttop + tol and blk["y1"] >= ttop - _CAPTION_ABOVE_MARGIN

    def is_below(blk: Dict) -> bool:
        return blk["y0"] >= tbottom - tol and blk["y0"] <= tbottom + _CAPTION_BELOW_MARGIN

    for blk in page_blocks:
        text = (blk.get("text") or "").strip()
        if not text:
            continue
        if is_above(blk) and _CAPTION_STRICT.search(text):
            return text
        if is_below(blk) and _CAPTION_STRICT.search(text):
            return text

    candidates: List[Tuple[float, str]] = []
    for blk in page_blocks:
        if not is_above(blk):
            continue
        text = (blk.get("text") or "").strip()
        if text and _CAPTION_LOOSE.search(text):
            candidates.append((blk["y1"], text))
    if candidates:
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]

    return ""


def _clean_rows(raw: List[List]) -> List[List[str]]:
    rows = []
    for row in (raw or []):
        if row is None:
            continue
        cells = [
            re.sub(r"\s+", " ", str(c).strip()) if c is not None else ""
            for c in row
        ]
        if any(cells):
            rows.append(cells)
    return rows


# ── v1 extraction ────────────────────────────────────────────────────────────

def _extract_v1(
    pdf_path: str,
    tables_dir: Path,
    source_name: str,
) -> List[Dict]:
    """pdfplumber table detection + PyMuPDF bbox caption lookup."""
    table_index: List[Dict] = []

    with fitz.open(pdf_path) as fitz_doc:
        total_pages = len(fitz_doc)
        with pdfplumber.open(pdf_path) as pdf:
            for page_index in range(total_pages):
                page_num = page_index + 1

                fitz_page   = fitz_doc.load_page(page_index)
                page_blocks = _get_page_blocks(fitz_page)

                if page_index >= len(pdf.pages):
                    continue
                plumber_page = pdf.pages[page_index]

                try:
                    found_tables = plumber_page.find_tables()
                except Exception:
                    continue

                for t_idx, table in enumerate(found_tables):
                    try:
                        raw = table.extract()
                    except Exception:
                        continue

                    rows = _clean_rows(raw)
                    if not rows:
                        continue

                    total_chars = sum(len(c) for row in rows for c in row)
                    if total_chars < _MIN_TABLE_CHARS:
                        continue

                    caption = ""
                    if hasattr(table, "bbox") and table.bbox and page_blocks:
                        caption = _find_caption(table.bbox, page_blocks)

                    table_id = f"table_p{page_num}_{t_idx}"
                    csv_path = tables_dir / f"{table_id}.csv"

                    with open(csv_path, "w", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerows(rows)

                    table_index.append({
                        "table_id":    table_id,
                        "page":        page_num,
                        "source_file": source_name,
                        "csv_path":    str(csv_path),
                        "caption":     caption,
                        "headers":     rows[0] if rows else [],
                        "row_count":   max(0, len(rows) - 1),
                    })

    return table_index


# ── v2 extraction (Docling) ──────────────────────────────────────────────────

def _docling_rows(table, ddoc) -> Optional[List[List[str]]]:
    """
    Extract rows from a Docling TableItem.

    Tries multiple Docling API shapes (the API evolves across versions):
      1. table.export_to_dataframe(doc=ddoc) — Docling ≤ 1.x
      2. table.export_to_dataframe()          — Docling ≥ 2.x
      3. table.export_to_csv()                — fallback text path
    """
    # Attempt 1 & 2: DataFrame export
    for kwargs in [{"doc": ddoc}, {}]:
        try:
            df = table.export_to_dataframe(**kwargs)
            col_headers = [str(c) for c in df.columns.tolist()]
            data_rows   = [[str(c) for c in row] for row in df.values.tolist()]
            rows = [col_headers] + data_rows if col_headers else data_rows
            if rows:
                return rows
        except Exception:
            continue

    # Attempt 3: CSV string fallback
    try:
        csv_str = table.export_to_csv()
        reader  = csv.reader(io.StringIO(csv_str))
        rows = [row for row in reader if any(row)]
        if rows:
            return rows
    except Exception:
        pass

    return None


def _extract_v2(
    pdf_path: str,
    tables_dir: Path,
    source_name: str,
) -> List[Dict]:
    """
    Docling-based table extraction.

    Docling models the document structure before rendering, giving it superior
    handling of:
      • Merged cells and nested headers
      • Tables without visible borders
      • Tables that span page boundaries

    The page-by-page conversion loop mirrors extract_text v2 so both steps share
    identical Docling processing for every page.
    """
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        raise ImportError(
            "docling is required for version='v2'. "
            "Install with: pip install docling"
        )

    converter   = DocumentConverter()
    table_index: List[Dict] = []

    with fitz.open(pdf_path) as fitz_doc:
        total_pages = len(fitz_doc)

    for page_num in range(1, total_pages + 1):
        try:
            res  = converter.convert(str(pdf_path), page_range=(page_num, page_num))
            ddoc = res.document
        except Exception:
            continue

        doc_tables = getattr(ddoc, "tables", None) or []
        for t_idx, table in enumerate(doc_tables):
            rows = _docling_rows(table, ddoc)
            if not rows:
                continue

            total_chars = sum(len(c) for row in rows for c in row)
            if total_chars < _MIN_TABLE_CHARS:
                continue

            # Caption from Docling's own metadata
            caption = (
                getattr(table, "label",   None)
                or getattr(table, "caption", None)
                or ""
            )
            if not isinstance(caption, str):
                caption = str(caption)

            table_id = f"table_p{page_num}_{t_idx}"
            csv_path = tables_dir / f"{table_id}.csv"

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)

            table_index.append({
                "table_id":    table_id,
                "page":        page_num,
                "source_file": source_name,
                "csv_path":    str(csv_path),
                "caption":     caption,
                "headers":     rows[0] if rows else [],
                "row_count":   max(0, len(rows) - 1),
            })

    return table_index


# ── Public entry-point ───────────────────────────────────────────────────────

def extract_tables(
    pdf_path: str,
    output_dir: str = "outputs/step1",
    version: str = "v1",
) -> List[Dict]:
    """
    Extract tables from a PDF into individual CSV files with caption metadata.

    Parameters
    ----------
    pdf_path   : path to source PDF
    output_dir : directory for CSVs and table_index.json
    version    : 'v1' (pdfplumber + PyMuPDF bbox captions)
               | 'v2' (Docling — use with extract_text version='v2')

    Returns the table index (list of metadata dicts) and writes:
    - one CSV per table  → outputs/step1/tables/<table_id>.csv
    - table_index.json   → outputs/step1/table_index.json

    Each index record:
        table_id    : unique ID  e.g. "table_p12_0"
        page        : 1-based page number
        source_file : PDF filename
        csv_path    : path to CSV file
        caption     : detected table caption (empty string if not found)
        headers     : first row as list of strings
        row_count   : number of data rows (excluding header)
    """
    tables_dir = Path(output_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    source_name = Path(pdf_path).name

    log("1b", f"Extracting tables from {pdf_path} [version={version}]")

    if version == "v2":
        table_index = _extract_v2(pdf_path, tables_dir, source_name)
    else:
        table_index = _extract_v1(pdf_path, tables_dir, source_name)

    save_json(table_index, str(Path(output_dir) / "table_index.json"))
    log("1b", f"[{version}] Extracted {len(table_index)} tables")
    return table_index


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Step 1b: extract tables to CSV + table_index.json (v1=pdfplumber, v2=Docling).",
    )
    ap.add_argument(
        "pdf",
        nargs="?",
        default="input/Heidenreich, 2022, AHA,ACC,HFSA guidelines.pdf",
        help="source PDF path",
    )
    ap.add_argument("--out", default="outputs/step1", help="directory for table_index.json and tables/")
    ap.add_argument(
        "--version",
        choices=["v1", "v2"],
        default="v1",
        help="v1=pdfplumber+PyMuPDF; v2=Docling (match extract_text --version)",
    )
    args = ap.parse_args()
    extract_tables(
        args.pdf,
        output_dir=args.out,
        version=args.version,
    )
