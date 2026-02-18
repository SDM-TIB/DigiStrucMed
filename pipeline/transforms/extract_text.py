from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import fitz
import pdfplumber
try:
    from pipeline.data import PDFGuidelines, RawText
except Exception:
    from pdf_guidelines import PDFGuidelines
    from raw_text import RawText
@dataclass(frozen=True)
class ExtractionConfig:
    skip_first_pages: int = 3
    skip_last_pages: int = 5
    preserve_paragraphs: bool = True
    paragraph_separator: str = "\n\n"
    enable_column_detection: bool = True
    filter_headers_footers: bool = True
    filter_urls: bool = True
    filter_toc: bool = True
    filter_metadata_near_start: bool = True
    filter_references_near_end: bool = True
    drop_references_section: bool = True
    references_heading_scan_lines: int = 40
class ExtractText:
    def __init__(
        self,
        skip_first_pages: int = 3,
        skip_last_pages: int = 5,
        config: Optional[ExtractionConfig] = None,
    ):
        self.config = config or ExtractionConfig(
            skip_first_pages=skip_first_pages, skip_last_pages=skip_last_pages
        )
        self._compile_patterns()
    def _compile_patterns(self) -> None:
        self.url_pattern = re.compile(
            r"https?://[^\s]+|doi:\s*\S+|www\.\S+", re.IGNORECASE
        )
        self.download_pattern = re.compile(
            r"downloaded from|accessed on|retrieved from", re.IGNORECASE
        )
        self.toc_dots_pattern = re.compile(r"\.{3,}")
        self.page_num_pattern = re.compile(r"\b\d{1,4}\s*$")
        self.section_num_pattern = re.compile(r"^\d+(?:\.\d+){1,}")
        self.metadata_short_caps = re.compile(r"^[A-Z0-9\s\.\,\-\:\;]{1,60}$")
        self.metadata_trailing_year = re.compile(r"\d{4}\s*$")
        self.reference_pattern = re.compile(r"^\d+\.\s+[A-Z][a-z]+\s+[A-Z]")
        self.et_al_pattern = re.compile(r"\bet\s+al\.?\b", re.IGNORECASE)
        self.references_heading = re.compile(r"^\s*REFERENCES\s*$", re.IGNORECASE)
        self.artifact_block_pattern = re.compile(r"^\s*[IVX0-9\s\.\-]+\s*$")
        self.classification_code_pattern = re.compile(
            r"^[IVX]+[a-z]?\s+[A-C]?\s*$", re.IGNORECASE
        )
        self.table_figure_pattern = re.compile(
            r"\b(Table|TABLE|Figure|FIGURE|Fig\.|FIG\.)\s+\d+", re.IGNORECASE
        )
        self.page_indicator = re.compile(r"^page\s+\d+|^\d+\s+of\s+\d+", re.IGNORECASE)
        self.running_header_pattern = re.compile(
            r"^\s*[^\n]*(?:et\s+al\.?)[^\n]*\d{4}[^\n]*Guidelines?\s*",
            re.IGNORECASE,
        )
    def _remove_control_chars(self, text: str) -> str:
        if not text:
            return text
        return "".join(
            ch
            for ch in text
            if (ch in "\n\t\r") or (unicodedata.category(ch) != "Cc")
        )
    def _fix_encoding(self, text: str) -> str:
        if not text:
            return text
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\u00a0", " ")
        text = self._remove_control_chars(text)
        return text
    def _strip_running_header(self, page_text: str) -> str:
        if not page_text or not page_text.strip():
            return page_text
        return self.running_header_pattern.sub("", page_text, count=1).strip()
    def _dehyphenate(self, text: str) -> str:
        if not text:
            return text
        return re.sub(r"([A-Za-z])\-\n\s*([A-Za-z])", r"\1\2", text)
    def _merge_midword_newlines(self, text: str) -> str:
        if not text:
            return text
        text = re.sub(r"([A-Za-z0-9])\n\s*([A-Za-z0-9])", r"\1\2", text)
        return text
    def _normalize_whitespace(self, text: str) -> str:
        if not text:
            return text
        text = re.sub(r"\s*\n+\s*", " ", text)
        text = " ".join(text.split())
        return text
    def _merge_incomplete_paragraphs(self, page_text: str) -> str:
        if not page_text or not page_text.strip():
            return page_text
        sep = self.config.paragraph_separator
        paras = [p.strip() for p in page_text.split(sep) if p.strip()]
        if len(paras) <= 1:
            return page_text
        trailing_prep = re.compile(r"\b(for|with|to|in|on|at|of|from|by)\s*$", re.IGNORECASE)
        incomplete_verb = re.compile(
            r"\b(should be|must be|may be|can be|will be|would be|could be|shall be)\s*$",
            re.IGNORECASE,
        )
        merged: List[str] = []
        i = 0
        while i < len(paras):
            p = paras[i]
            while i + 1 < len(paras) and (
                trailing_prep.search(p) or incomplete_verb.search(p)
            ):
                p = p + " " + paras[i + 1]
                i += 1
            merged.append(p)
            i += 1
        return sep.join(merged)
    def _clean_block_text(self, text: str) -> str:
        t = self._fix_encoding(text)
        t = self._dehyphenate(t)
        t = self._merge_midword_newlines(t)
        t = self._normalize_whitespace(t)
        return t
    def _get_raw_blocks(self, page) -> List[Dict]:
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
    _TABLE_CAPTION_PATTERN = re.compile(
        r"^\s*(?:Table|TABLE)\s+\d+[.\s—:\-]+(.+)$", re.IGNORECASE
    )
    def _find_caption_for_table(
        self,
        table_bbox: Tuple[float, float, float, float],
        page_blocks: List[Dict],
    ) -> str:
        tx0, ttop, tx1, tbottom = table_bbox
        tolerance = 25.0
        def block_above(blk: Dict) -> bool:
            return blk["y1"] <= ttop + tolerance and blk["y1"] >= ttop - 120
        def block_below(blk: Dict) -> bool:
            return blk["y0"] >= tbottom - tolerance and blk["y0"] <= tbottom + 120
        for blk in page_blocks:
            text = (blk.get("text") or "").strip()
            if not text:
                continue
            if block_above(blk):
                match = self._TABLE_CAPTION_PATTERN.search(text)
                if match:
                    return text.strip()
            if block_below(blk):
                match = self._TABLE_CAPTION_PATTERN.search(text)
                if match:
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
            page = pdf_plumber_doc.pages[page_index]
            found = page.find_tables()
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
                if total_chars < 50:
                    continue
                caption = ""
                if hasattr(table, "bbox") and table.bbox and page_blocks:
                    caption = self._find_caption_for_table(table.bbox, page_blocks)
                tables_out.append({"title": caption, "rows": rows})
        except Exception:
            pass
        return tables_out
    def _order_blocks(self, blocks: List[Dict], page_width: float) -> List[Dict]:
        if not blocks:
            return blocks
        if not self.config.enable_column_detection:
            return sorted(blocks, key=lambda b: (b["y0"], b["x0"]))
        x0s = sorted(b["x0"] for b in blocks)
        if len(x0s) < 8 or page_width <= 0:
            return sorted(blocks, key=lambda b: (b["y0"], b["x0"]))
        gap_i = max(range(len(x0s) - 1), key=lambda i: x0s[i + 1] - x0s[i])
        max_gap = x0s[gap_i + 1] - x0s[gap_i]
        if max_gap < 0.22 * page_width:
            return sorted(blocks, key=lambda b: (b["y0"], b["x0"]))
        boundary = (x0s[gap_i] + x0s[gap_i + 1]) / 2.0
        left = [b for b in blocks if b["x0"] < boundary]
        right = [b for b in blocks if b["x0"] >= boundary]
        if len(left) < 3 or len(right) < 3:
            return sorted(blocks, key=lambda b: (b["y0"], b["x0"]))
        left = sorted(left, key=lambda b: (b["y0"], b["x0"]))
        right = sorted(right, key=lambda b: (b["y0"], b["x0"]))
        return left + right
    def _is_url_block(self, text: str) -> bool:
        return bool(text and (self.url_pattern.search(text) or self.download_pattern.search(text)))
    def _is_toc_block(self, text: str) -> bool:
        if not text:
            return False
        t = text.strip()
        if not t:
            return False
        if len(t) > 220:
            return False
        has_leader_dots = bool(self.toc_dots_pattern.search(t))
        has_page_num_end = bool(self.page_num_pattern.search(t))
        has_section_start = bool(self.section_num_pattern.match(t))
        if has_leader_dots and has_page_num_end:
            return True
        if has_section_start and has_page_num_end and len(t.split()) <= 18:
            return True
        return False
    def _is_metadata_block(self, text: str) -> bool:
        t = text.strip() if text else ""
        if not t:
            return False
        if len(t) > 60:
            return False
        if self.metadata_short_caps.match(t):
            return True
        if len(t) <= 30 and self.metadata_trailing_year.search(t):
            return True
        return False
    def _is_reference_line(self, line: str) -> bool:
        t = line.strip() if line else ""
        if not t:
            return False
        if self.reference_pattern.match(t):
            return True
        if self.et_al_pattern.search(t):
            return True
        return False
    def _is_header_footer_block(
        self,
        text: str,
        bbox: Tuple[float, float, float, float],
        page_rect: fitz.Rect,
    ) -> bool:
        if not text:
            return True
        t = text.strip()
        if not t:
            return True
        if not self.config.filter_headers_footers:
            return False
        x0, y0, x1, y1 = bbox
        page_h = float(page_rect.height) if page_rect else 0.0
        if page_h <= 0:
            return False
        in_top = y0 <= 0.12 * page_h
        in_bottom = y1 >= 0.90 * page_h
        if not (in_top or in_bottom):
            return False
        if len(t) < 3:
            return True
        if len(t) < 80:
            if self.page_indicator.search(t):
                return True
            if re.fullmatch(r"\d{1,4}", t):
                return True
            if re.search(r"^[A-Z]{2,}\s+[A-Z][a-z].*\s+\d{1,4}$", t):
                return True
            if re.search(r"^\d{1,4}\s+[A-Z]{2,}\s+[A-Z][a-z]", t):
                return True
            if t.isupper() and len(t.split()) <= 6:
                return True
        return False
    def _is_side_margin_block(
        self,
        text: str,
        bbox: Tuple[float, float, float, float],
        page_rect: fitz.Rect,
    ) -> bool:
        if not text or not text.strip() or page_rect is None:
            return False
        x0, y0, x1, y1 = bbox
        page_w = float(page_rect.width)
        page_h = float(page_rect.height)
        if page_w <= 0 or page_h <= 0:
            return False
        bw = float(x1 - x0)
        bh = float(y1 - y0)
        near_left = x0 <= 0.08 * page_w
        near_right = x1 >= 0.92 * page_w
        narrow = bw <= 0.12 * page_w
        tall = bh >= 0.18 * page_h
        if not (narrow and tall and (near_left or near_right)):
            return False
        letters = [ch for ch in text if ch.isalpha()]
        if not letters:
            return True
        upper_ratio = sum(1 for ch in letters if ch.isupper()) / max(1, len(letters))
        return upper_ratio >= 0.65
    def _is_artifact_block(self, text: str) -> bool:
        t = text.strip()
        if not t:
            return True
        if self.artifact_block_pattern.match(t):
            return True
        if len(t.split()) <= 2 and self.classification_code_pattern.match(t):
            return True
        return False
    def _is_table_metadata_block(self, text: str) -> bool:
        t = text.strip()
        if not t:
            return False
        match = self.table_figure_pattern.search(t)
        if match and match.start() < 150:
            after = t[match.end() :].strip()
            if len(after.split()) < 20:
                return True
        if "|" in t and len(t.split("|")) >= 2:
            parts = [p.strip() for p in t.split("|")]
            if all(len(p.split()) <= 4 for p in parts):
                return True
        return False
    def _should_keep_block(
        self,
        t: str,
        blk: Dict,
        page_rect: fitz.Rect,
        page_num: int,
        total_pages: int,
    ) -> bool:
        if self._is_artifact_block(t):
            return False
        if self._is_table_metadata_block(t):
            return False
        bbox = (blk["x0"], blk["y0"], blk["x1"], blk["y1"])
        if self._is_header_footer_block(t, bbox, page_rect):
            return False
        if self._is_side_margin_block(t, bbox, page_rect):
            return False
        if self.config.filter_urls and self._is_url_block(t):
            return False
        if self.config.filter_toc and self._is_toc_block(t):
            return False
        near_start = page_num <= self.config.skip_first_pages + 5
        if self.config.filter_metadata_near_start and near_start and self._is_metadata_block(t):
            return False
        return True
    def _should_drop_page(
        self,
        page_text: str,
        page_blocks: List[str],
        page_num: int,
        total_pages: int,
    ) -> Tuple[bool, Optional[str]]:
        text_stripped = page_text.strip()
        text_lower = page_text.lower()
        if len(text_stripped) < 80:
            return True, "nearly_blank"
        head = text_lower[:250]
        if re.search(r"\blist of\s+(tables|figures|abbreviations)\b", head):
            return True, "list_of"
        words = re.findall(r"[A-Za-z]{2,}", page_text)
        if len(words) > 50:
            all_caps = sum(1 for w in words if w.isupper())
            if all_caps / max(1, len(words)) > 0.35:
                return True, "abbreviation_list"
        in_reference_zone = page_num > total_pages - self.config.skip_last_pages - 10
        if in_reference_zone and len(page_blocks) >= 5:
            ref_lines = 0
            total_lines = 0
            for b in page_blocks:
                for ln in re.split(r"(?<=\.)\s+|\n", b):
                    ln = ln.strip()
                    if not ln:
                        continue
                    total_lines += 1
                    if self._is_reference_line(ln):
                        ref_lines += 1
            if total_lines >= 8 and ref_lines / total_lines >= 0.85:
                return True, "reference_only"
        return False, None
    def _detect_references_cutoff(self, doc: fitz.Document) -> Optional[int]:
        total_pages = len(doc)
        considered = [
            p for p in range(1, total_pages + 1)
            if (p > self.config.skip_first_pages) and (p <= total_pages - self.config.skip_last_pages)
        ]
        if not considered:
            return None
        def ref_density(text: str) -> float:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if len(lines) < 20:
                return 0.0
            ref_like = 0
            for ln in lines:
                if re.match(r"^\d+\.\s", ln):
                    ref_like += 1
                elif self._is_reference_line(ln):
                    ref_like += 1
            return ref_like / max(1, len(lines))
        scan_n = max(5, int(self.config.references_heading_scan_lines))
        for page_num in considered:
            page = doc.load_page(page_num - 1)
            txt = page.get_text("text") or ""
            txt = self._fix_encoding(txt)
            txt = self._dehyphenate(txt)
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            for ln in lines[:scan_n]:
                if self.references_heading.match(ln):
                    if ref_density(txt) >= 0.25:
                        return page_num
                    break
        ref_density_threshold = 0.13
        consec = 0
        earliest_in_run: Optional[int] = None
        for page_num in reversed(considered):
            page = doc.load_page(page_num - 1)
            txt = page.get_text("text") or ""
            txt = self._fix_encoding(txt)
            txt = self._dehyphenate(txt)
            d = ref_density(txt)
            if d >= ref_density_threshold:
                consec += 1
                earliest_in_run = page_num
            else:
                if consec >= 3 and earliest_in_run is not None:
                    return earliest_in_run
                consec = 0
                earliest_in_run = None
        if consec >= 3 and earliest_in_run is not None:
            return earliest_in_run
        return None
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
        return raw_text
    def _extract_pdf(self, pdf_path: Path) -> List[Dict]:
        pages: List[Dict] = []
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            references_cutoff = None
            if self.config.filter_references_near_end and self.config.drop_references_section:
                references_cutoff = self._detect_references_cutoff(doc)
            with pdfplumber.open(pdf_path) as pdf_plumber_doc:
                for page_index in range(total_pages):
                    page_num = page_index + 1
                    if page_num <= self.config.skip_first_pages:
                        continue
                    if page_num > total_pages - self.config.skip_last_pages:
                        continue
                    if references_cutoff is not None and page_num >= references_cutoff:
                        continue
                    page = doc.load_page(page_index)
                    page_rect = page.rect
                    page_width = float(page_rect.width)
                    raw_blocks = self._get_raw_blocks(page)
                    ordered = self._order_blocks(raw_blocks, page_width)
                    cleaned_blocks: List[str] = []
                    for blk in ordered:
                        t = self._clean_block_text(blk["text"])
                        if not t:
                            continue
                        if self._should_keep_block(t, blk, page_rect, page_num, total_pages):
                            cleaned_blocks.append(t)
                    if not cleaned_blocks:
                        continue
                    if self.config.preserve_paragraphs:
                        page_text = self.config.paragraph_separator.join(cleaned_blocks)
                    else:
                        page_text = " ".join(cleaned_blocks)
                    page_text = self._remove_control_chars(page_text)
                    page_text = self._strip_running_header(page_text)
                    page_text = self._merge_incomplete_paragraphs(page_text)
                    if not page_text.strip():
                        continue
                    drop, _reason = self._should_drop_page(
                        page_text, cleaned_blocks, page_num, total_pages
                    )
                    if drop:
                        continue
                    tables = self._extract_tables_for_page(
                        pdf_plumber_doc, page_index, page_blocks=ordered
                    )
                    pages.append({"page": page_num, "text": page_text, "tables": tables})
        return pages
