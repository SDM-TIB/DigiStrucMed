import re
from typing import List, Tuple, Optional
TYPE_RECOMMENDATION = "recommendation"
TYPE_STAGES_DEFINITION = "stages_definition"
TYPE_LVEF_CLASSIFICATION = "lvef_classification"
TYPE_DRUG_DOSING = "drug_dosing"
TYPE_PHENOTYPIC = "phenotypic"
TYPE_REFERENCE = "reference"
TYPE_CAUSE_REFERENCE = "cause_reference"
TYPE_LIST = "list"
TYPE_GENERIC_TWO_COL = "generic_two_col"
def _normalize_header(h: str) -> str:
    return (h or "").strip().lower()
def _row_to_header_key(row: List[str]) -> str:
    return " ".join(_normalize_header(str(c)) for c in row)
def classify_table(rows: List[List[str]]) -> str:
    if not rows:
        return TYPE_GENERIC_TWO_COL
    row0 = _row_to_header_key(rows[0])
    row1 = _row_to_header_key(rows[1]) if len(rows) > 1 else ""
    headers = row0 + " " + row1
    num_cols = len(rows[0]) if rows[0] else 0
    if "cor" in headers and "loe" in headers and "recommendation" in headers:
        return TYPE_RECOMMENDATION
    if re.search(r"\bcor\b", headers) and re.search(r"\bloe\b", headers):
        return TYPE_RECOMMENDATION
    if "stage" in headers and ("definition" in headers or "criteria" in headers):
        return TYPE_STAGES_DEFINITION
    if "stages" in headers and "definition" in headers:
        return TYPE_STAGES_DEFINITION
    if "type of hf" in headers or "classification" in headers:
        if "lvef" in headers or "criteria" in headers:
            return TYPE_LVEF_CLASSIFICATION
    if "hfref" in row0 or "hfpef" in row0:
        return TYPE_LVEF_CLASSIFICATION
    if "drug" in headers and ("dose" in headers or "maximum" in headers or "daily" in headers):
        return TYPE_DRUG_DOSING
    if "initial" in headers and "duration" in headers and num_cols >= 3:
        return TYPE_DRUG_DOSING
    if "phenotypic" in headers or "ask specifically" in headers or "family member" in headers:
        return TYPE_PHENOTYPIC
    if "consideration" in headers and "reference" in headers:
        return TYPE_REFERENCE
    if "cause" in headers and "reference" in headers:
        return TYPE_CAUSE_REFERENCE
    if num_cols == 1:
        return TYPE_LIST
    first_col_cells = [str(r[0]).strip().lower() for r in rows[:5] if r]
    if first_col_cells and any(x in first_col_cells for x in ("cardiac", "noncardiac")):
        return TYPE_LIST
    if num_cols == 2:
        return TYPE_GENERIC_TWO_COL
    return TYPE_GENERIC_TWO_COL
def _strip_citation_tail(text: str) -> str:
    text = re.sub(r"[.\s]+\d+[–\-]\d+\s*$", "", text)
    text = re.sub(r"[.\s]+\d+(?:\s*,\s*\d+)*\s*$", "", text)
    return text.strip()
def _strip_numbered_prefix(text: str) -> str:
    return re.sub(r"^\s*\d+\.\s*", "", text).strip()
def _dehyphenate(text: str) -> str:
    return re.sub(r"\b(\w+)-\s+(\w+)\b", r"\1\2", text)
def convert_recommendation_row(row: List[str]) -> Optional[str]:
    if not row:
        return None
    text = str(row[-1]).strip() if row else ""
    text = _strip_numbered_prefix(text)
    text = _strip_citation_tail(text)
    return text if len(text) >= 20 else None
def convert_stages_definition_row(row: List[str]) -> Optional[str]:
    if len(row) < 2:
        return None
    subj = str(row[0]).strip()
    obj = str(row[1]).strip()
    if not subj or not obj:
        return None
    return f"{subj} is {obj}"
def convert_lvef_row(row: List[str]) -> Optional[str]:
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
    if len(row) < 2:
        return None
    drug = str(row[0]).strip()
    if not drug:
        return None
    parts = [str(c).strip() for c in row[1:] if c and str(c).strip()]
    if not parts:
        return None
    return f"{drug} has " + ", ".join(parts)
def convert_phenotypic_row(row: List[str]) -> Optional[str]:
    cells = [str(c).strip() for c in row if c]
    if not cells:
        return None
    if len(cells) == 1:
        return cells[0] if len(cells[0]) >= 20 else None
    if len(cells) == 2:
        return f"{cells[0]}: {cells[1]}"
    return ". ".join(c for c in cells if c)
def convert_reference_row(row: List[str]) -> Optional[str]:
    if len(row) < 2:
        return None
    consideration = str(row[0]).strip()
    ref = str(row[1]).strip()
    if not consideration or not ref:
        return None
    return f"{consideration} is addressed in {ref}"
def _normalize_table_title_for_sentence(caption: str) -> str:
    if not caption:
        return ""
    return re.sub(r"^\s*(?:Table|TABLE)\s+\d+[.\s—:\-]+", "", caption, flags=re.IGNORECASE).strip()
def convert_cause_reference_row(
    row: List[str],
    table_title: Optional[str] = None,
) -> Optional[str]:
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
    cell = str(row[0]).strip() if row else ""
    if not cell:
        return None
    if section and cell.lower() == section.lower():
        return None
    if section:
        return f"{section}: {cell}"
    return cell
def convert_generic_two_col_row(row: List[str]) -> Optional[str]:
    if len(row) < 2:
        return None
    a = str(row[0]).strip()
    b = str(row[1]).strip()
    if not a or not b:
        return None
    if len(a) > 60 and len(b) < 25:
        return None
    return f"{a} is {b}"
def table_rows_to_sentences(
    rows: List[List[str]],
    table_type: str,
    table_title: Optional[str] = None,
) -> List[Tuple[str, int]]:
    result: List[Tuple[str, int]] = []
    if not rows:
        return result
    start = 0
    if len(rows) > 1:
        first = rows[0]
        first_cell = str(first[0]).strip() if first else ""
        if len(first_cell) > 50 and (
            "recommendation" in first_cell.lower() or "referenced" in first_cell.lower()
        ):
            start = 1
        elif table_type == TYPE_RECOMMENDATION and len(first) >= 3:
            fc = first_cell.upper()
            if fc in ("COR", "LOE", "CLASS", "LEVEL") or "RECOMMENDATION" in first_cell.upper():
                start = 1
        elif table_type == TYPE_LVEF_CLASSIFICATION or table_type == TYPE_DRUG_DOSING:
            start = 1
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
        if sentence and table_title and table_type == TYPE_GENERIC_TWO_COL:
            title_phrase = _normalize_table_title_for_sentence(table_title)
            if title_phrase:
                sentence = f"In {title_phrase}, {sentence}"
        if sentence and len(sentence) >= 15:
            sentence = _dehyphenate(sentence)
            result.append((sentence, i))
    return result
