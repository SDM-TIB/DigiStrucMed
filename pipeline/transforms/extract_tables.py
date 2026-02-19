"""
Stage A – table extraction only.
Extracts tables from PDF pages using pdfplumber; uses PyMuPDF (fitz) for block
positions when resolving table captions. Use this module when focusing on
table extraction independently of body text.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz
import pdfplumber


@dataclass(frozen=True)
class TableExtractionConfig:
    """Options for table extraction."""
    min_table_chars: int = 50  # skip very small tables
    caption_tolerance: float = 25.0
    caption_vertical_margin: float = 120.0


class ExtractTables:
    """
    Extracts tables from PDFs. Returns per-page lists of tables (each with
    optional 'title' and 'rows'). Caption detection uses text block positions
    from PyMuPDF.
    """

    _TABLE_CAPTION_PATTERN = re.compile(
        r"^\s*(?:Table|TABLE)\s+\d+[.\s—:\-]+(.+)$", re.IGNORECASE
    )

    def __init__(self, config: Optional[TableExtractionConfig] = None):
        self.config = config or TableExtractionConfig()

    def _get_raw_blocks(self, page: Any) -> List[Dict]:
        """Get text blocks with bbox from a PyMuPDF page."""
        raw_blocks = []
        for b in page.get_text("blocks"):
            x0, y0, x1, y1, txt, *_ = b
            if not txt or not str(txt).strip():
                continue
            raw_blocks.append({
                "x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1),
                "text": str(txt),
            })
        return raw_blocks

    def _find_caption_for_table(
        self,
        table_bbox: Tuple[float, float, float, float],
        page_blocks: List[Dict],
    ) -> str:
        tx0, ttop, tx1, tbottom = table_bbox
        tol = self.config.caption_tolerance
        margin = self.config.caption_vertical_margin

        def block_above(blk: Dict) -> bool:
            return blk["y1"] <= ttop + tol and blk["y1"] >= ttop - margin

        def block_below(blk: Dict) -> bool:
            return blk["y0"] >= tbottom - tol and blk["y0"] <= tbottom + margin

        for blk in page_blocks:
            text = (blk.get("text") or "").strip()
            if not text:
                continue
            if block_above(blk):
                if self._TABLE_CAPTION_PATTERN.search(text):
                    return text.strip()
            if block_below(blk):
                if self._TABLE_CAPTION_PATTERN.search(text):
                    return text.strip()
        return ""

    def _extract_tables_for_page(
        self,
        pdf_plumber_doc: Any,
        page_index: int,
        page_blocks: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        tables_out: List[Dict] = []
        page_blocks = page_blocks or []
        try:
            if page_index >= len(pdf_plumber_doc.pages):
                return tables_out
            plumber_page = pdf_plumber_doc.pages[page_index]
            found = plumber_page.find_tables()
            for table in found:
                raw = table.extract()
                if not raw:
                    continue
                rows: List[List[str]] = []
                for row in raw:
                    if row is None:
                        continue
                    cells = [
                        re.sub(r"\s+", " ", str(c).strip()) if c is not None else ""
                        for c in row
                    ]
                    if any(cells):
                        rows.append(cells)
                if not rows:
                    continue
                total_chars = sum(len(str(c)) for row in rows for c in row)
                if total_chars < self.config.min_table_chars:
                    continue
                caption = ""
                if hasattr(table, "bbox") and table.bbox and page_blocks:
                    caption = self._find_caption_for_table(table.bbox, page_blocks)
                tables_out.append({"title": caption, "rows": rows})
        except Exception:
            pass
        return tables_out

    def extract_from_pdf(
        self,
        pdf_path: Path,
        page_numbers: Optional[List[int]] = None,
    ) -> Dict[int, List[Dict]]:
        """
        Extract tables from a PDF.

        Args:
            pdf_path: Path to the PDF file.
            page_numbers: 1-based page numbers to process. If None, all pages
                         are processed.

        Returns:
            Dict mapping 1-based page number -> list of table dicts.
            Each table dict has "title" (str) and "rows" (list of list of str).
        """
        result: Dict[int, List[Dict]] = {}
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            with pdfplumber.open(pdf_path) as pdf_plumber_doc:
                indices = range(total_pages)
                if page_numbers is not None:
                    indices = [i for i in indices if (i + 1) in page_numbers]
                for page_index in indices:
                    page_num = page_index + 1
                    page = doc.load_page(page_index)
                    page_blocks = self._get_raw_blocks(page)
                    tables = self._extract_tables_for_page(
                        pdf_plumber_doc, page_index, page_blocks
                    )
                    result[page_num] = tables
        return result
