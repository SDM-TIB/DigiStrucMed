"""
Rules and conversion logic for turning table rows into subject–predicate–object (SPO) triples.
Used by the content_preparation stage (Stage B).
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple

from pipeline.transforms.table_to_sentences import (
    classify_table,
    TYPE_CAUSE_REFERENCE,
    TYPE_LIST,
    TYPE_LVEF_CLASSIFICATION,
    TYPE_RECOMMENDATION,
    TYPE_DRUG_DOSING,
    _normalize_header,
    _row_to_header_key,
    _strip_numbered_prefix,
    _strip_citation_tail,
)


def _header_like(header: str, *patterns: str) -> bool:
    h = _normalize_header(header)
    return any(p in h for p in patterns)


def _find_column_index(headers: List[str], *patterns: str) -> Optional[int]:
    """Return first column index whose header matches any of the patterns."""
    for i, cell in enumerate(headers):
        if _header_like(str(cell), *patterns):
            return i
    return None


def is_reference_column(header: str) -> bool:
    """True if the column is a reference/link/URL column (to be ignored for subject/object)."""
    h = _normalize_header(header)
    if not h:
        return False
    if any(p in h for p in ("reference", "link", "url", "citation", "source link")):
        return True
    if re.search(r"reference\s*/\s*link", h):
        return True
    return False


def two_col_single_reference_content_index(headers: List[str]) -> Optional[int]:
    """
    If exactly 2 columns and exactly one is a reference column, return the index
    of the non-reference (content) column. Otherwise return None.
    """
    if len(headers) != 2:
        return None
    ref0 = is_reference_column(str(headers[0]))
    ref1 = is_reference_column(str(headers[1]))
    if ref0 and not ref1:
        return 1
    if ref1 and not ref0:
        return 0
    return None


def strip_table_number_prefix(caption: str) -> str:
    """Remove leading 'Table N. ' or 'TABLE N. ' from caption when used as subject."""
    if not caption:
        return (caption or "").strip()
    return re.sub(r"^\s*(?:Table|TABLE)\s+\d+[.\s—:\-]+", "", caption.strip(), flags=re.IGNORECASE).strip() or caption.strip()


def has_recommendations_for_prefix(text: Optional[str]) -> bool:
    """True if text starts with 'Recommendations for ' (after normalizing whitespace)."""
    if not text:
        return False
    s = re.sub(r"\s+", " ", str(text).strip())
    return s.upper().startswith("RECOMMENDATIONS FOR ")


def subject_from_recommendation_title(text: Optional[str]) -> Optional[str]:
    """
    For tables with red title 'Recommendations for ... [Referenced ...]':
    return what comes after 'Recommendations for ' and before 'Referenced' (or end).
    """
    if not text or not str(text).strip():
        return None
    s = re.sub(r"\s+", " ", str(text).strip())
    match = re.search(r"Recommendations\s+for\s+", s, re.IGNORECASE)
    if not match:
        return None
    start = match.end()
    rest = s[start:]
    ref_match = re.search(r"\bReferenced\b", rest, re.IGNORECASE)
    if ref_match:
        rest = rest[: ref_match.start()]
    rest = rest.strip()
    rest = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", rest)
    return rest or None


def normalize_header_to_predicate(header: str) -> str:
    """Turn a column header into a predicate phrase, e.g. 'Initial Daily Dose' -> 'has initial daily dose'. For meaning/phrase use 'is'."""
    h = (header or "").strip().lower()
    if not h:
        return "has value"
    if "meaning" in h or "phrase" in h:
        return "is"
    return "has " + h


def is_valid_object_value(text: str) -> bool:
    """False if the value is empty or symbol-only (e.g. X, •, -), so it should not be used as object."""
    s = (text or "").strip()
    if not s:
        return False
    if len(s) <= 3 and re.match(r"^[\sXx•\-–—↑↓↔✓✔\d,\.;:\'\"]+$", s):
        return False
    if not re.search(r"[a-zA-Z]", s):
        return False
    return True


def _unified_row_to_triples(
    headers: List[str],
    cells: List[str],
    current_col0_subject: Optional[str],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    One rule for all tables: subject = column 0 (carry forward when empty).
    For each other column: predicate = 'has <header>', object = cell value,
    only when column is not reference and value is valid (not symbol-only).
    """
    triples: List[Dict[str, Any]] = []
    subject = (cells[0] or "").strip() or current_col0_subject
    if not subject:
        return (triples, current_col0_subject)
    if (cells[0] or "").strip():
        current_col0_subject = (cells[0] or "").strip()
    for i in range(1, min(len(cells), len(headers))):
        if is_reference_column(str(headers[i])):
            continue
        if not is_valid_object_value(cells[i]):
            continue
        pred = normalize_header_to_predicate(str(headers[i]))
        triples.append({
            "subject": subject,
            "predicate": pred,
            "object": cells[i],
        })
    return (triples, current_col0_subject)


def _first_data_row_index(rows: List[List[str]], table_type: str) -> int:
    """Row index where data starts (after header)."""
    if len(rows) <= 1:
        return 1
    first = rows[0]
    first_cell = str(first[0]).strip() if first else ""
    if len(first_cell) > 50 and ("recommendation" in first_cell.lower() or "referenced" in first_cell.lower()):
        return 1
    if table_type == TYPE_RECOMMENDATION and len(first) >= 3:
        fc = first_cell.upper()
        if fc in ("COR", "LOE", "CLASS", "LEVEL") or "RECOMMENDATION" in first_cell.upper():
            return 1
    if table_type in (TYPE_LVEF_CLASSIFICATION, TYPE_DRUG_DOSING, TYPE_CAUSE_REFERENCE):
        return 1
    return 1


def table_rows_to_spo(
    rows: List[List[str]],
    table_type: str,
    table_title: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convert table rows to SPO triples based on column titles and table type.
    Returns list of dicts with keys: subject, predicate, object.
    """
    result: List[Dict[str, Any]] = []
    if not rows:
        return result
    headers = [str(c).strip() for c in rows[0]]
    if len(rows) > 1:
        for i in range(min(len(headers), len(rows[1]))):
            if not headers[i]:
                headers[i] = str(rows[1][i] or "").strip()
    start = _first_data_row_index(rows, table_type)
    rec_header_row: Optional[List[Any]] = None
    if table_type == TYPE_RECOMMENDATION and len(rows) > 1 and has_recommendations_for_prefix(rows[0][0] if rows and len(rows[0]) > 0 else None):
        start = 2
        rec_header_row = rows[1]
    current_col0_subject: Optional[str] = None

    for row_idx in range(start, len(rows)):
        row = rows[row_idx]
        if not row:
            continue
        cells = [str(c).strip() if c is not None else "" for c in row]
        spo: Dict[str, Any] = {"subject": None, "predicate": None, "object": None}

        if table_type == TYPE_RECOMMENDATION:
            rec_headers = [str(c).strip() for c in rec_header_row] if rec_header_row else headers
            rec_col = _find_column_index(rec_headers, "recommendation") or (len(cells) - 1)
            rec_text = cells[rec_col] if rec_col < len(cells) else ""
            rec_text = _strip_numbered_prefix(rec_text)
            rec_text = _strip_citation_tail(rec_text)
            if len(rec_text) < 15:
                continue
            first_cell = str(rows[0][0]).strip() if rows and len(rows[0]) > 0 else ""
            red_title = first_cell if has_recommendations_for_prefix(first_cell) else (table_title or "")
            spo["subject"] = subject_from_recommendation_title(red_title) or "guideline"
            spo["predicate"] = "recommends"
            spo["object"] = rec_text

        elif table_type == TYPE_LIST:
            if cells[0]:
                raw_title = (table_title or "table").strip()
                spo["subject"] = strip_table_number_prefix(re.sub(r"\s+", " ", raw_title))
                spo["predicate"] = "lists"
                spo["object"] = cells[0]

        else:
            content_col = two_col_single_reference_content_index(headers)
            if content_col is not None and (table_title or "").strip():
                subject_title = strip_table_number_prefix(re.sub(r"\s+", " ", (table_title or "").strip()))
                if content_col < len(cells) and is_valid_object_value(cells[content_col]):
                    pred = normalize_header_to_predicate(str(headers[content_col]))
                    result.append({
                        "subject": subject_title,
                        "predicate": pred,
                        "object": cells[content_col],
                    })
                continue
            row_triples, current_col0_subject = _unified_row_to_triples(
                headers, cells, current_col0_subject
            )
            result.extend(row_triples)
            continue

        if spo.get("subject") or spo.get("predicate") or spo.get("object"):
            result.append(spo)

    return result


def tables_pages_to_spo_list(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert full Stage A tables output (pages with tables) to a flat list of SPO triples.
    Skips evidence-gap / future-research tables and "Continued" when previous was evidence-gap.
    """
    triples: List[Dict[str, Any]] = []
    prev_title_lower: Optional[str] = None
    for page_data in pages:
        page_num = page_data.get("page")
        for table in page_data.get("tables", []):
            rows = table.get("rows", [])
            if not rows:
                continue
            raw_title = (table.get("title") or "").strip()
            t = raw_title.lower()
            if "evidence gap" in t or "future research" in t or "research direction" in t:
                prev_title_lower = t
                continue
            if "continued" in t and prev_title_lower and (
                "evidence gap" in prev_title_lower or "future research" in prev_title_lower or "research direction" in prev_title_lower
            ):
                prev_title_lower = t
                continue
            prev_title_lower = t
            table_type = classify_table(rows)
            table_title = re.sub(r"\s+", " ", raw_title) if raw_title else ""
            for spo in table_rows_to_spo(rows, table_type, table_title=raw_title or None):
                out = {
                    "subject": spo.get("subject"),
                    "predicate": spo.get("predicate"),
                    "object": spo.get("object"),
                }
                if table_title:
                    out["table"] = table_title
                out["page"] = page_num
                triples.append(out)
    return triples
