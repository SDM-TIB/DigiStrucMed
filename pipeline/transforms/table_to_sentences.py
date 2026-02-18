"""
[Transform] table_to_sentences

Classify extracted tables and convert rows to sentences for factual statement extraction.
Used by ChunkText (Stage B) to add table-derived chunks.
"""

import re
from typing import List, Tuple, Optional

# Table type identifiers
TYPE_RECOMMENDATION = "recommendation"
TYPE_STAGES_DEFINITION = "stages_definition"
TYPE_LVEF_CLASSIFICATION = "lvef_classification"
TYPE_DRUG_DOSING = "drug_dosing"
TYPE_PHENOTYPIC = "phenotypic"
TYPE_REFERENCE = "reference"
TYPE_CAUSE_REFERENCE = "cause_reference"  # Cause + Reference columns; title links to causes
TYPE_LIST = "list"
TYPE_GENERIC_TWO_COL = "generic_two_col"


def _normalize_header(h: str) -> str:
    return (h or "").strip().lower()


def _row_to_header_key(row: List[str]) -> str:
    """First row cells joined and normalized for pattern matching."""
    return " ".join(_normalize_header(str(c)) for c in row)


def classify_table(rows: List[List[str]]) -> str:
    """
    Classify table type from header row(s) and content.
    Returns one of TYPE_* constants.
    """
    if not rows:
        return TYPE_GENERIC_TWO_COL

    # First row often title; second row headers. Check first 2 rows.
    row0 = _row_to_header_key(rows[0])
    row1 = _row_to_header_key(rows[1]) if len(rows) > 1 else ""

    headers = row0 + " " + row1
    num_cols = len(rows[0]) if rows[0] else 0

    # Recommendation: COR, LOE, Recommendations
    if "cor" in headers and "loe" in headers and "recommendation" in headers:
        return TYPE_RECOMMENDATION
    if re.search(r"\bcor\b", headers) and re.search(r"\bloe\b", headers):
        return TYPE_RECOMMENDATION

    # Stages / definition
    if "stage" in headers and ("definition" in headers or "criteria" in headers):
        return TYPE_STAGES_DEFINITION
    if "stages" in headers and "definition" in headers:
        return TYPE_STAGES_DEFINITION

    # LVEF classification
    if "type of hf" in headers or "classification" in headers:
        if "lvef" in headers or "criteria" in headers:
            return TYPE_LVEF_CLASSIFICATION
    if "hfref" in row0 or "hfpef" in row0:
        return TYPE_LVEF_CLASSIFICATION

    # Drug dosing
    if "drug" in headers and ("dose" in headers or "maximum" in headers or "daily" in headers):
        return TYPE_DRUG_DOSING
    if "initial" in headers and "duration" in headers and num_cols >= 3:
        return TYPE_DRUG_DOSING

    # Phenotypic
    if "phenotypic" in headers or "ask specifically" in headers or "family member" in headers:
        return TYPE_PHENOTYPIC

    # Reference
    if "consideration" in headers and "reference" in headers:
        return TYPE_REFERENCE

    # Cause + Reference (e.g. Table 5. Other Potential Nonischemic Causes of HF)
    if "cause" in headers and "reference" in headers:
        return TYPE_CAUSE_REFERENCE

    # List (single column or Cardiac/Noncardiac)
    if num_cols == 1:
        return TYPE_LIST
    first_col_cells = [str(r[0]).strip().lower() for r in rows[:5] if r]
    if first_col_cells and any(x in first_col_cells for x in ("cardiac", "noncardiac")):
        return TYPE_LIST

    # Generic two-column
    if num_cols == 2:
        return TYPE_GENERIC_TWO_COL

    return TYPE_GENERIC_TWO_COL


def _strip_citation_tail(text: str) -> str:
    """Remove trailing citation patterns like '.1–6' or ' 1,2' or '36–39'."""
    # e.g. "symptoms.1–6" -> "symptoms", "therapy.1,2" -> "therapy"
    text = re.sub(r"[.\s]+\d+[–\-]\d+\s*$", "", text)
    text = re.sub(r"[.\s]+\d+(?:\s*,\s*\d+)*\s*$", "", text)
    return text.strip()


def _strip_numbered_prefix(text: str) -> str:
    """Remove leading '1. ' or '2. ' from recommendation text."""
    return re.sub(r"^\s*\d+\.\s*", "", text).strip()


def _dehyphenate(text: str) -> str:
    """Join line-break hyphenation (e.g. 're- duced' -> 'reduced', 'includ- ing' -> 'including')."""
    return re.sub(r"\b(\w+)-\s+(\w+)\b", r"\1\2", text)


def convert_recommendation_row(row: List[str]) -> Optional[str]:
    """Last column is recommendation text. Strip numbering and citations."""
    if not row:
        return None
    # COR, LOE, Recommendations -> use last column
    text = str(row[-1]).strip() if row else ""
    text = _strip_numbered_prefix(text)
    text = _strip_citation_tail(text)
    return text if len(text) >= 20 else None


def convert_stages_definition_row(row: List[str]) -> Optional[str]:
    """Subject (col0) is/criteria (col1)."""
    if len(row) < 2:
        return None
    subj = str(row[0]).strip()
    obj = str(row[1]).strip()
    if not subj or not obj:
        return None
    return f"{subj} is {obj}"


def convert_lvef_row(row: List[str]) -> Optional[str]:
    """Type (col0), criteria (col1)."""
    if len(row) < 2:
        return None
    subj = str(row[0]).strip()
    obj = str(row[1]).strip()
    if not subj or not obj:
        return None
    return f"{subj} is defined as {obj}"


def convert_drug_dosing_row(
    row: List[str],
    headers: Optional[List[str]] = None,
) -> Optional[str]:
    """Drug, Initial dose, Max dose, Duration -> one sentence."""
    if len(row) < 2:
        return None
    drug = str(row[0]).strip()
    if not drug:
        return None
    parts = [str(c).strip() for c in row[1:] if c and str(c).strip()]
    if not parts:
        return None  # skip header or empty dose rows
    return f"{drug} has " + ", ".join(parts)


def convert_phenotypic_row(row: List[str]) -> Optional[str]:
    """Phenotypic category, finding, ask about -> one sentence."""
    cells = [str(c).strip() for c in row if c]
    if not cells:
        return None
    if len(cells) == 1:
        return cells[0] if len(cells[0]) >= 20 else None
    if len(cells) == 2:
        return f"{cells[0]}: {cells[1]}"
    return ". ".join(c for c in cells if c)


def convert_reference_row(row: List[str]) -> Optional[str]:
    """Consideration, Reference."""
    if len(row) < 2:
        return None
    consideration = str(row[0]).strip()
    ref = str(row[1]).strip()
    if not consideration or not ref:
        return None
    return f"{consideration} is addressed in {ref}"


def _normalize_table_title_for_sentence(caption: str) -> str:
    """Remove 'Table N. ' prefix for use in factual sentences."""
    if not caption:
        return ""
    return re.sub(r"^\s*(?:Table|TABLE)\s+\d+[.\s—:\-]+", "", caption, flags=re.IGNORECASE).strip()


def convert_cause_reference_row(
    row: List[str],
    table_title: Optional[str] = None,
) -> Optional[str]:
    """Cause (col0) + Reference (col1); link to table title for factual statement.

    E.g. title 'Table 5. Other Potential Nonischemic Causes of HF' + row (cause, ref)
    -> 'Chemotherapy and other cardiotoxic medications is a cause listed in Other Potential Nonischemic Causes of HF (references 23-25).'
    """
    if len(row) < 2:
        return None
    cause = str(row[0]).strip()
    ref = str(row[1]).strip()
    if not cause or not ref:
        return None
    title_phrase = _normalize_table_title_for_sentence(table_title or "")
    if title_phrase:
        return f"{cause} is a cause listed in {title_phrase} (references {ref})"
    return f"{cause} (references {ref})"


def convert_list_row(row: List[str], section: Optional[str] = None) -> Optional[str]:
    """Single column or list item."""
    cell = str(row[0]).strip() if row else ""
    if not cell:
        return None
    if section and cell.lower() == section.lower():
        return None  # skip section header row (e.g. "Cardiac" alone)
    if section:
        return f"{section}: {cell}"
    return cell


def convert_generic_two_col_row(row: List[str]) -> Optional[str]:
    """Col0 is col1. Skip reference-style rows (long title + short org)."""
    if len(row) < 2:
        return None
    a = str(row[0]).strip()
    b = str(row[1]).strip()
    if not a or not b:
        return None
    # Skip guideline/reference rows: very long first col, very short second (e.g. "Title" is "ACCF/AHA")
    if len(a) > 60 and len(b) < 25:
        return None
    return f"{a} is {b}"


def table_rows_to_sentences(
    rows: List[List[str]],
    table_type: str,
    table_title: Optional[str] = None,
) -> List[Tuple[str, int]]:
    """
    Convert table rows to (sentence, row_index) list.
    Skips header/title rows heuristically.
    table_title: optional caption (e.g. "Table 5. Other Potential Nonischemic Causes of HF")
    used for cause_reference and similar types to link title to row content.
    """
    result: List[Tuple[str, int]] = []
    if not rows:
        return result

    # Heuristic: skip first row if it looks like title/header
    start = 0
    if len(rows) > 1:
        first = rows[0]
        first_cell = str(first[0]).strip() if first else ""
        # Title row: first cell long and contains "Recommendation" or "Referenced"
        if len(first_cell) > 50 and (
            "recommendation" in first_cell.lower() or "referenced" in first_cell.lower()
        ):
            start = 1
        # Header row: COR, LOE, Recommendations
        elif table_type == TYPE_RECOMMENDATION and len(first) >= 3:
            fc = first_cell.upper()
            if fc in ("COR", "LOE", "CLASS", "LEVEL") or "RECOMMENDATION" in first_cell.upper():
                start = 1
        # LVEF and drug_dosing: first row is always header
        elif table_type == TYPE_LVEF_CLASSIFICATION or table_type == TYPE_DRUG_DOSING:
            start = 1
        # Cause / Reference: first row is header (Cause, Reference)
        elif table_type == TYPE_CAUSE_REFERENCE and len(first) >= 2:
            if _normalize_header(first_cell) == "cause" or "reference" in _row_to_header_key(first):
                start = 1

    for i in range(start, len(rows)):
        row = rows[i]
        if not row:
            continue

        sentence = None
        if table_type == TYPE_RECOMMENDATION:
            sentence = convert_recommendation_row(row)
        elif table_type == TYPE_STAGES_DEFINITION:
            sentence = convert_stages_definition_row(row)
        elif table_type == TYPE_LVEF_CLASSIFICATION:
            sentence = convert_lvef_row(row)
        elif table_type == TYPE_DRUG_DOSING:
            sentence = convert_drug_dosing_row(row)
        elif table_type == TYPE_PHENOTYPIC:
            sentence = convert_phenotypic_row(row)
        elif table_type == TYPE_REFERENCE:
            sentence = convert_reference_row(row)
        elif table_type == TYPE_CAUSE_REFERENCE:
            sentence = convert_cause_reference_row(row, table_title=table_title)
        elif table_type == TYPE_LIST:
            section = None
            if i > 0 and row[0]:
                cell0 = str(row[0]).strip()
                if cell0.lower() in ("cardiac", "noncardiac"):
                    section = cell0
            sentence = convert_list_row(row, section)
        else:
            sentence = convert_generic_two_col_row(row)

        # When table has a caption, use it as subject/context for generic two-col
        if sentence and table_title and table_type == TYPE_GENERIC_TWO_COL:
            title_phrase = _normalize_table_title_for_sentence(table_title)
            if title_phrase:
                sentence = f"In {title_phrase}, {sentence}"

        if sentence and len(sentence) >= 15:
            sentence = _dehyphenate(sentence)
            result.append((sentence, i))

    return result
