"""
Stage A version 2: Extract text and tables from PDFs using Docling.
Same contract as v1: transform(PDFGuidelines) -> RawText;
writes outputs/STAGE_A_v2/text.json and tables.json (same structure).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    from pipeline.data import PDFGuidelines, RawText
except Exception:
    from pdf_guidelines import PDFGuidelines
    from raw_text import RawText

# Use fitz only for page count (lightweight)
import fitz
try:
    from docling.document_converter import DocumentConverter
except ImportError:
    DocumentConverter = None  # optional; required only for Stage A v2


class ExtractTextV2:
    """Stage A v2: Docling-based extraction. Same I/O contract as ExtractText (v1)."""

    def __init__(
        self,
        skip_first_pages: int = 3,
        skip_last_pages: int = 5,
        stage_output_dir: Optional[str] = "outputs/STAGE_A_v2",
    ):
        if DocumentConverter is None:
            raise ImportError(
                "docling is required for Stage A v2. Install with: pip install docling"
            )
        self.skip_first_pages = skip_first_pages
        self.skip_last_pages = skip_last_pages
        self.stage_output_dir = stage_output_dir
        self._converter = DocumentConverter()

    def transform(self, pdf_guidelines: PDFGuidelines) -> RawText:
        raw_text = RawText()
        for pdf_path in pdf_guidelines.get_files():
            for page_data in self._extract_pdf(pdf_path):
                if page_data["text"].strip():
                    raw_text.add_page(
                        page_num=page_data["page"],
                        text=page_data["text"],
                        source_file=str(pdf_path.name),
                        tables=page_data.get("tables", []),
                    )
        if self.stage_output_dir:
            self._write_stage_output(raw_text)
        return raw_text

    def _write_stage_output(self, raw_text: RawText) -> None:
        """Write Stage A output (text and tables) — same structure as v1."""
        out_path = Path(self.stage_output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        pages = raw_text.get_pages()
        texts_out = [
            {"source_file": p.get("source", ""), "page": p["page"], "text": p["text"]}
            for p in pages
        ]
        (out_path / "text.json").write_text(
            json.dumps(texts_out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tables_out = []
        for p in pages:
            for t in p.get("tables", []):
                tables_out.append({
                    "source_file": p.get("source", ""),
                    "page": p["page"],
                    "caption": t.get("title", ""),
                    "rows": t.get("rows", []),
                })
        (out_path / "tables.json").write_text(
            json.dumps(tables_out, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _extract_pdf(self, pdf_path: Path) -> List[Dict]:
        """Extract per-page text and tables using Docling; respect skip first/last pages."""
        pages_out: List[Dict] = []
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
        for page_num in range(1, total_pages + 1):
            if page_num <= self.skip_first_pages:
                continue
            if page_num > total_pages - self.skip_last_pages:
                continue
            try:
                conv_res = self._converter.convert(
                    str(pdf_path),
                    page_range=(page_num, page_num),
                )
            except Exception:
                continue
            doc = conv_res.document
            text = doc.export_to_text() if hasattr(doc, "export_to_text") else ""
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            tables: List[Dict] = []
            if hasattr(doc, "tables") and doc.tables:
                for table in doc.tables:
                    try:
                        df = table.export_to_dataframe(doc=doc)
                        rows = [
                            [str(c) for c in row]
                            for row in df.values.tolist()
                        ]
                        title = getattr(table, "label", "") or getattr(table, "caption", "") or ""
                        if not isinstance(title, str):
                            title = str(title)
                        tables.append({"title": title, "rows": rows})
                    except Exception:
                        continue
            pages_out.append({
                "page": page_num,
                "text": text.strip(),
                "tables": tables,
            })
        return pages_out
