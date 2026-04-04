"""
Step 1a — Text Extraction
─────────────────────────────────────────────────────────────────────────────
Input  : PDF file path
Output : outputs/step1/text_blocks.json
         Each record: { page, source_file, text }

Versions
--------
v1 (default) — PyMuPDF with full noise-filtering:
  • Two-column layout detection (gap-based, no hard-coded margins)
  • Automatic references-section cutoff
  • Header / footer stripping (bbox-based)
  • Side-margin block removal
  • TOC (dot-leader) removal
  • URL / download-citation stripping
  • De-hyphenation across line breaks
  • Incomplete-paragraph merging
  • Abbreviation / all-caps page detection and drop

v2 — Docling (proven for complex PDF layouts; same output schema)
  docling proved itself a lot — use when v1 misses multi-column content or
  has layout parsing issues.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from utils import log, save_json


# ── Shared constants ─────────────────────────────────────────────────────────

# Heart-failure domain acronyms — expanded BEFORE NER so the model sees full
# medical terms rather than abbreviations it may not have been trained on.
# Keep in sync with pipeline/data/heart_failure_acronyms.json
_HF_ACRONYMS: Dict[str, str] = {
    "HFrEF": "heart failure with reduced ejection fraction",
    "HFmrEF": "heart failure with mildly reduced ejection fraction",
    "HFpEF": "heart failure with preserved ejection fraction",
    "HFimpEF": "heart failure with improved ejection fraction",
    "HF": "heart failure",
    "ACE-I": "ACE inhibitor",
    "ACEi": "angiotensin-converting enzyme inhibitors",
    "ARNI": "angiotensin receptor-neprilysin inhibitor",
    "ARNi": "angiotensin receptor-neprilysin inhibitors",
    "ARB": "angiotensin receptor blockers",
    "MRA": "mineralocorticoid receptor antagonist",
    "SGLT2i": "sodium-glucose cotransporter-2 inhibitor",
    "SGLT2": "sodium-glucose cotransporter-2",
    "RAASi": "renin-angiotensin-aldosterone system inhibitors",
    "RASS": "renin-angiotensin-aldosterone system",
    "CV": "cardiovascular",
    "LV": "left ventricular",
    "RV": "right ventricular",
    "LVEF": "left ventricular ejection fraction",
    "NYHA": "New York Heart Association",
    "AF": "atrial fibrillation",
    "CAD": "coronary artery disease",
    "ICD": "implantable cardioverter-defibrillator",
    "CRT": "cardiac resynchronization therapy",
    "LVAD": "left ventricular assist device",
    "MCS": "mechanical circulatory support",
    "DOAC": "direct-acting oral anticoagulants",
    "BNP": "B-type natriuretic peptide",
    "CKD": "chronic kidney disease",
    "eGFR": "estimated glomerular filtration rate",
    "T2DM": "type 2 diabetes mellitus",
    "DM": "diabetes mellitus",
    "HTN": "hypertension",
    "MI": "myocardial infarction",
    "ACS": "acute coronary syndrome",
    "VT": "ventricular tachycardia",
    "VF": "ventricular fibrillation",
    "SCD": "sudden cardiac death",
    "ECG": "electrocardiogram",
    "GDMT": "guideline-directed medical therapy",
}


def _expand_acronyms(text: str) -> str:
    """Expand known HF acronyms so downstream NER sees full terms."""
    if not text:
        return text
    for acronym, expansion in _HF_ACRONYMS.items():
        pattern = r"\b" + re.escape(acronym) + r"\b"
        text = re.sub(pattern, expansion, text)
    return text


# ── v1: PyMuPDF extractor ────────────────────────────────────────────────────

class _ExtractorV1:
    """
    Full-featured PyMuPDF extractor ported from pipeline/transforms/extract_text.py.
    All class machinery is internal; the public API is extract_text().
    """

    def __init__(self, skip_first: int, skip_last: int, min_chars: int) -> None:
        self.skip_first = skip_first
        self.skip_last = skip_last
        self.min_chars = min_chars
        self._compile_patterns()

    # ── Pattern compilation ──────────────────────────────────────────────────

    def _compile_patterns(self) -> None:
        self.url_pat = re.compile(
            r"https?://[^\s]+|doi:\s*\S+|www\.\S+", re.IGNORECASE
        )
        self.download_pat = re.compile(
            r"downloaded from|accessed on|retrieved from", re.IGNORECASE
        )
        self.toc_dots = re.compile(r"\.{3,}")
        self.page_num = re.compile(r"\b\d{1,4}\s*$")
        self.section_num = re.compile(r"^\d+(?:\.\d+){1,}")
        self.metadata_caps = re.compile(r"^[A-Z0-9\s\.\,\-\:\;]{1,60}$")
        self.trailing_year = re.compile(r"\d{4}\s*$")
        self.reference_line = re.compile(r"^\d+\.\s+[A-Z][a-z]+\s+[A-Z]")
        self.et_al = re.compile(r"\bet\s+al\.?\b", re.IGNORECASE)
        self.references_heading = re.compile(r"^\s*REFERENCES\s*$", re.IGNORECASE)
        self.artifact_block = re.compile(r"^\s*[IVX0-9\s\.\-]+\s*$")
        self.class_code = re.compile(r"^[IVX]+[a-z]?\s+[A-C]?\s*$", re.IGNORECASE)
        self.table_fig = re.compile(
            r"\b(Table|TABLE|Figure|FIGURE|Fig\.|FIG\.)\s+\d+", re.IGNORECASE
        )
        self.page_indicator = re.compile(
            r"^page\s+\d+|^\d+\s+of\s+\d+", re.IGNORECASE
        )
        self.running_header = re.compile(
            r"^\s*[^\n]*(?:et\s+al\.?)[^\n]*\d{4}[^\n]*Guidelines?\s*",
            re.IGNORECASE,
        )
        self.trailing_prep = re.compile(
            r"\b(for|with|to|in|on|at|of|from|by)\s*$", re.IGNORECASE
        )
        self.incomplete_verb = re.compile(
            r"\b(should be|must be|may be|can be|will be|would be|could be|"
            r"shall be)\s*$",
            re.IGNORECASE,
        )

    # ── Text cleaning ────────────────────────────────────────────────────────

    def _remove_control(self, text: str) -> str:
        return "".join(
            ch for ch in text
            if ch in "\n\t\r" or unicodedata.category(ch) != "Cc"
        )

    def _fix_encoding(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\u00a0", " ")
        return self._remove_control(text)

    def _dehyphenate(self, text: str) -> str:
        return re.sub(r"([A-Za-z])\-\n\s*([A-Za-z])", r"\1\2", text)

    def _merge_midword_newlines(self, text: str) -> str:
        return re.sub(r"([A-Za-z0-9])\n\s*([A-Za-z0-9])", r"\1\2", text)

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"\s*\n+\s*", " ", text)
        return " ".join(text.split())

    def _clean_block(self, text: str) -> str:
        t = self._fix_encoding(text)
        t = self._dehyphenate(t)
        t = self._merge_midword_newlines(t)
        t = self._normalize_whitespace(t)
        return t

    def _strip_running_header(self, text: str) -> str:
        return self.running_header.sub("", text, count=1).strip()

    def _merge_incomplete_paragraphs(self, text: str) -> str:
        sep = "\n\n"
        paras = [p.strip() for p in text.split(sep) if p.strip()]
        if len(paras) <= 1:
            return text
        merged: List[str] = []
        i = 0
        while i < len(paras):
            p = paras[i]
            while i + 1 < len(paras) and (
                self.trailing_prep.search(p) or self.incomplete_verb.search(p)
            ):
                p = p + " " + paras[i + 1]
                i += 1
            merged.append(p)
            i += 1
        return sep.join(merged)

    # ── Block-level filters ──────────────────────────────────────────────────

    def _get_raw_blocks(self, page) -> List[Dict]:
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

    def _order_blocks(self, blocks: List[Dict], page_width: float) -> List[Dict]:
        """Order blocks left-column-first for two-column layouts."""
        if not blocks:
            return blocks
        x0s = sorted(b["x0"] for b in blocks)
        if len(x0s) < 8 or page_width <= 0:
            return sorted(blocks, key=lambda b: (b["y0"], b["x0"]))
        gap_i = max(range(len(x0s) - 1), key=lambda i: x0s[i + 1] - x0s[i])
        max_gap = x0s[gap_i + 1] - x0s[gap_i]
        if max_gap < 0.22 * page_width:
            return sorted(blocks, key=lambda b: (b["y0"], b["x0"]))
        boundary = (x0s[gap_i] + x0s[gap_i + 1]) / 2.0
        left  = sorted([b for b in blocks if b["x0"] <  boundary], key=lambda b: (b["y0"], b["x0"]))
        right = sorted([b for b in blocks if b["x0"] >= boundary], key=lambda b: (b["y0"], b["x0"]))
        if len(left) < 3 or len(right) < 3:
            return sorted(blocks, key=lambda b: (b["y0"], b["x0"]))
        return left + right

    def _is_url_block(self, t: str) -> bool:
        return bool(self.url_pat.search(t) or self.download_pat.search(t))

    def _is_toc_block(self, t: str) -> bool:
        if not t or len(t) > 220:
            return False
        has_dots  = bool(self.toc_dots.search(t))
        has_pgnum = bool(self.page_num.search(t))
        has_sec   = bool(self.section_num.match(t))
        if has_dots and has_pgnum:
            return True
        if has_sec and has_pgnum and len(t.split()) <= 18:
            return True
        return False

    def _is_metadata_block(self, t: str) -> bool:
        if not t or len(t) > 60:
            return False
        if self.metadata_caps.match(t):
            return True
        if len(t) <= 30 and self.trailing_year.search(t):
            return True
        return False

    def _is_reference_line(self, line: str) -> bool:
        t = line.strip()
        return bool(self.reference_line.match(t) or self.et_al.search(t))

    def _is_header_footer(
        self, t: str, bbox: Tuple[float, float, float, float], page_rect
    ) -> bool:
        if not t.strip():
            return True
        x0, y0, x1, y1 = bbox
        page_h = float(page_rect.height) if page_rect else 0.0
        if page_h <= 0:
            return False
        in_top    = y0 <= 0.12 * page_h
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

    def _is_side_margin(
        self, t: str, bbox: Tuple[float, float, float, float], page_rect
    ) -> bool:
        if not t.strip() or page_rect is None:
            return False
        x0, y0, x1, y1 = bbox
        pw = float(page_rect.width)
        ph = float(page_rect.height)
        if pw <= 0 or ph <= 0:
            return False
        bw = x1 - x0
        bh = y1 - y0
        near_left  = x0 <= 0.08 * pw
        near_right = x1 >= 0.92 * pw
        narrow = bw <= 0.12 * pw
        tall   = bh >= 0.18 * ph
        if not (narrow and tall and (near_left or near_right)):
            return False
        letters = [ch for ch in t if ch.isalpha()]
        if not letters:
            return True
        return sum(1 for ch in letters if ch.isupper()) / len(letters) >= 0.65

    def _is_artifact(self, t: str) -> bool:
        if not t.strip():
            return True
        if self.artifact_block.match(t):
            return True
        return len(t.split()) <= 2 and bool(self.class_code.match(t))

    def _is_table_metadata(self, t: str) -> bool:
        m = self.table_fig.search(t)
        if m and m.start() < 150 and len(t[m.end():].split()) < 20:
            return True
        if "|" in t:
            parts = [p.strip() for p in t.split("|")]
            if all(len(p.split()) <= 4 for p in parts):
                return True
        return False

    def _should_keep(
        self, t: str, blk: Dict, page_rect, page_num: int, total_pages: int
    ) -> bool:
        if self._is_artifact(t):
            return False
        if self._is_table_metadata(t):
            return False
        bbox = (blk["x0"], blk["y0"], blk["x1"], blk["y1"])
        if self._is_header_footer(t, bbox, page_rect):
            return False
        if self._is_side_margin(t, bbox, page_rect):
            return False
        if self._is_url_block(t):
            return False
        if self._is_toc_block(t):
            return False
        near_start = page_num <= self.skip_first + 5
        if near_start and self._is_metadata_block(t):
            return False
        return True

    # ── Page-level filters ───────────────────────────────────────────────────

    def _should_drop_page(
        self, page_text: str, page_blocks: List[str], page_num: int, total_pages: int
    ) -> bool:
        if len(page_text.strip()) < 80:
            return True
        head = page_text.lower()[:250]
        if re.search(r"\blist of\s+(tables|figures|abbreviations)\b", head):
            return True
        words = re.findall(r"[A-Za-z]{2,}", page_text)
        if len(words) > 50:
            all_caps = sum(1 for w in words if w.isupper())
            if all_caps / max(1, len(words)) > 0.35:
                return True
        in_ref_zone = page_num > total_pages - self.skip_last - 10
        if in_ref_zone and len(page_blocks) >= 5:
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
            if total_lines >= 8 and ref_lines / max(1, total_lines) >= 0.85:
                return True
        return False

    # ── References cutoff ────────────────────────────────────────────────────

    def _detect_references_cutoff(self, doc) -> Optional[int]:
        total = len(doc)
        considered = [
            p for p in range(1, total + 1)
            if self.skip_first < p <= total - self.skip_last
        ]
        if not considered:
            return None

        def ref_density(text: str) -> float:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if len(lines) < 20:
                return 0.0
            ref_like = sum(
                1 for ln in lines
                if re.match(r"^\d+\.\s", ln) or self._is_reference_line(ln)
            )
            return ref_like / max(1, len(lines))

        for pn in considered:
            page = doc.load_page(pn - 1)
            txt = self._fix_encoding(self._dehyphenate(page.get_text("text") or ""))
            for ln in [ln.strip() for ln in txt.splitlines() if ln.strip()][:40]:
                if self.references_heading.match(ln) and ref_density(txt) >= 0.25:
                    return pn

        consec = 0
        earliest: Optional[int] = None
        for pn in reversed(considered):
            page = doc.load_page(pn - 1)
            txt = self._fix_encoding(self._dehyphenate(page.get_text("text") or ""))
            d = ref_density(txt)
            if d >= 0.13:
                consec += 1
                earliest = pn
            else:
                if consec >= 3 and earliest is not None:
                    return earliest
                consec = 0
                earliest = None
        if consec >= 3 and earliest is not None:
            return earliest
        return None

    # ── Main extraction ──────────────────────────────────────────────────────

    def extract(self, pdf_path: str) -> List[Dict]:
        source_name = Path(pdf_path).name
        pages: List[Dict] = []

        with fitz.open(pdf_path) as doc:
            total = len(doc)
            ref_cutoff = self._detect_references_cutoff(doc)

            for page_index in range(total):
                page_num = page_index + 1
                if page_num <= self.skip_first:
                    continue
                if page_num > total - self.skip_last:
                    continue
                if ref_cutoff and page_num >= ref_cutoff:
                    continue

                page = doc.load_page(page_index)
                raw_blocks = self._get_raw_blocks(page)
                ordered    = self._order_blocks(raw_blocks, float(page.rect.width))

                cleaned: List[str] = []
                for blk in ordered:
                    t = self._clean_block(blk["text"])
                    if not t:
                        continue
                    if self._should_keep(t, blk, page.rect, page_num, total):
                        cleaned.append(t)

                if not cleaned:
                    continue

                page_text = "\n\n".join(cleaned)
                page_text = self._remove_control(page_text)
                page_text = self._strip_running_header(page_text)
                page_text = self._merge_incomplete_paragraphs(page_text)

                if not page_text.strip():
                    continue
                if len(page_text.strip()) < self.min_chars:
                    continue
                if self._should_drop_page(page_text, cleaned, page_num, total):
                    continue

                page_text = _expand_acronyms(page_text)
                pages.append({
                    "page": page_num,
                    "source_file": source_name,
                    "text": page_text,
                })

        return pages


# ── v2: Docling extractor ────────────────────────────────────────────────────

class _ExtractorV2:
    """
    Docling-based extractor ported from pipeline/transforms/extract_text_v2.py.
    Docling is preferred for complex layouts (multi-column, mixed tables+text)
    because it models the document structure before rendering text.
    """

    def __init__(self, skip_first: int, skip_last: int, min_chars: int) -> None:
        self.skip_first = skip_first
        self.skip_last  = skip_last
        self.min_chars  = min_chars

    def extract(self, pdf_path: str) -> List[Dict]:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError:
            raise ImportError(
                "docling is required for version='v2'. "
                "Install with: pip install docling"
            )

        source_name = Path(pdf_path).name
        converter   = DocumentConverter()
        pages_out: List[Dict] = []

        with fitz.open(pdf_path) as doc:
            total = len(doc)

        for page_num in range(1, total + 1):
            if page_num <= self.skip_first:
                continue
            if page_num > total - self.skip_last:
                continue
            try:
                res  = converter.convert(str(pdf_path), page_range=(page_num, page_num))
                ddoc = res.document
                text = ddoc.export_to_text() if hasattr(ddoc, "export_to_text") else ""
                if not isinstance(text, str):
                    text = str(text) if text is not None else ""
                text = text.strip()
            except Exception:
                continue

            if not text or len(text) < self.min_chars:
                continue

            text = _expand_acronyms(text)
            pages_out.append({
                "page": page_num,
                "source_file": source_name,
                "text": text,
            })

        return pages_out


# ── Public entry-point ───────────────────────────────────────────────────────

def extract_text(
    pdf_path: str,
    output_dir: str = "outputs/step1",
    skip_first_pages: int = 3,
    skip_last_pages: int = 5,
    min_chars: int = 80,
    version: str = "v1",
) -> List[Dict]:
    """
    Extract page-level text blocks from a PDF.

    Parameters
    ----------
    pdf_path         : path to the source PDF
    output_dir       : directory for text_blocks.json
    skip_first_pages : cover / copyright pages to skip (default 3)
    skip_last_pages  : reference / index pages to skip from end (default 5)
    min_chars        : minimum characters to keep a page block
    version          : 'v1' (PyMuPDF, full noise filtering)
                     | 'v2' (Docling, better for complex layouts)

    Returns list of { page, source_file, text } and writes text_blocks.json.
    """
    log("1a", f"Extracting text from {pdf_path} [version={version}]")

    if version == "v2":
        extractor: _ExtractorV1 | _ExtractorV2 = _ExtractorV2(
            skip_first_pages, skip_last_pages, min_chars
        )
    else:
        extractor = _ExtractorV1(skip_first_pages, skip_last_pages, min_chars)

    blocks = extractor.extract(pdf_path)

    out_path = Path(output_dir) / "text_blocks.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(blocks, str(out_path))
    log("1a", f"[{version}] Extracted {len(blocks)} pages → {out_path}")
    return blocks


if __name__ == "__main__":
    import sys
    _pdf = sys.argv[1] if len(sys.argv) > 1 else "input/guideline.pdf"
    _ver = sys.argv[2] if len(sys.argv) > 2 else "v1"
    extract_text(_pdf, version=_ver)
