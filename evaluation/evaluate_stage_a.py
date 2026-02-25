"""
Stage A evaluation: analyze and score Stage A extraction outputs (text + tables).

Evaluates one or more versions (v1, v2, v3, ...). Each version is read from
outputs/STAGE_A_{version}/text.json and tables.json.

Metrics: schema validity, text metrics, table metrics, table SPO usability.
No other pipeline stages (e.g. Stage B) are run; this is Stage A only.

Usage (from project root):
  python evaluation/evaluate_stage_a.py
  python evaluation/evaluate_stage_a.py --versions v1 v2 v3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVAL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"

# Default versions to evaluate (can add v3, v4, ... later)
DEFAULT_VERSIONS = ["v1", "v2"]


# ---------------------------------------------------------------------------
# Load Stage A output
# ---------------------------------------------------------------------------

def load_stage_a_output(text_path: Path, tables_path: Path) -> Tuple[Optional[List], Optional[List]]:
    """
    Load text.json and tables.json.
    Returns (texts_list, tables_list). (None, None) if text.json missing.
    """
    if not text_path.exists():
        return None, None
    with open(text_path, "r", encoding="utf-8") as f:
        texts = json.load(f)
    if not isinstance(texts, list):
        texts = texts.get("pages", texts) if isinstance(texts, dict) else []
    tables_list: List[Dict] = []
    if tables_path.exists():
        with open(tables_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        tables_list = raw if isinstance(raw, list) else []
    return texts, tables_list


# ---------------------------------------------------------------------------
# Schema & validity
# ---------------------------------------------------------------------------

def validate_text_schema(texts: List[Dict]) -> Dict[str, Any]:
    errors = []
    seen_keys: set = set()
    for i, item in enumerate(texts):
        if not isinstance(item, dict):
            errors.append(f"text[{i}] not a dict")
            continue
        for key in ("source_file", "page", "text"):
            if key not in item:
                errors.append(f"text[{i}] missing key '{key}'")
        if "page" in item and "source_file" in item:
            key = (item["source_file"], item["page"])
            if key in seen_keys:
                errors.append(f"text duplicate (source_file, page) at index {i}")
            else:
                seen_keys.add(key)
        if "text" in item and not isinstance(item["text"], str):
            errors.append(f"text[{i}].text not a string")
    return {
        "valid": len(errors) == 0,
        "error_count": len(errors),
        "errors": errors[:50],
        "total_pages": len(texts),
        "unique_page_keys": len(seen_keys),
    }


def validate_tables_schema(tables_list: List[Dict]) -> Dict[str, Any]:
    errors = []
    for i, t in enumerate(tables_list):
        if not isinstance(t, dict):
            errors.append(f"tables[{i}] not a dict")
            continue
        for key in ("source_file", "page", "caption", "rows"):
            if key not in t:
                errors.append(f"tables[{i}] missing key '{key}'")
        if "rows" in t and not isinstance(t["rows"], list):
            errors.append(f"tables[{i}].rows not a list")
        elif "rows" in t:
            for j, row in enumerate(t["rows"]):
                if not isinstance(row, list):
                    errors.append(f"tables[{i}].rows[{j}] not a list")
    return {
        "valid": len(errors) == 0,
        "error_count": len(errors),
        "errors": errors[:50],
        "total_tables": len(tables_list),
    }


def check_encoding_and_control_chars(texts: List[Dict]) -> Dict[str, Any]:
    total_cc = 0
    total_chars = 0
    for item in texts:
        s = item.get("text", "")
        if not isinstance(s, str):
            continue
        total_chars += len(s)
        for ch in s:
            if unicodedata.category(ch) == "Cc" and ch not in "\n\t\r":
                total_cc += 1
    return {
        "control_char_count": total_cc,
        "total_char_count": total_chars,
        "control_char_ratio": total_cc / max(1, total_chars),
    }


# ---------------------------------------------------------------------------
# Text metrics
# ---------------------------------------------------------------------------

def text_metrics(texts: List[Dict]) -> Dict[str, Any]:
    if not texts:
        return {"page_count": 0, "total_chars": 0, "total_words": 0}
    total_chars = 0
    total_words = 0
    page_lengths_chars: List[int] = []
    page_lengths_words: List[int] = []
    empty_pages = 0
    near_empty_pages = 0
    toc_like_lines = 0
    ref_section_like = 0
    url_like_count = 0
    url_pattern = re.compile(r"https?://[^\s]+|doi:\s*\S+|www\.\S+", re.IGNORECASE)
    toc_dots = re.compile(r"\.{3,}")
    page_num_at_end = re.compile(r"\b\d{1,4}\s*$")
    references_heading = re.compile(r"^\s*REFERENCES\s*$", re.IGNORECASE)

    for item in texts:
        t = item.get("text", "") or ""
        total_chars += len(t)
        words = len(t.split())
        total_words += words
        page_lengths_chars.append(len(t))
        page_lengths_words.append(words)
        stripped = t.strip()
        if len(stripped) == 0:
            empty_pages += 1
        elif len(stripped) < 20:
            near_empty_pages += 1
        for line in t.splitlines():
            line = line.strip()
            if toc_dots.search(line) and page_num_at_end.search(line):
                toc_like_lines += 1
            if references_heading.match(line):
                ref_section_like += 1
        url_like_count += len(url_pattern.findall(t))

    page_count = len(texts)
    return {
        "page_count": page_count,
        "total_chars": total_chars,
        "total_words": total_words,
        "avg_chars_per_page": total_chars / max(1, page_count),
        "avg_words_per_page": total_words / max(1, page_count),
        "min_chars_per_page": min(page_lengths_chars) if page_lengths_chars else 0,
        "max_chars_per_page": max(page_lengths_chars) if page_lengths_chars else 0,
        "empty_pages": empty_pages,
        "near_empty_pages": near_empty_pages,
        "toc_like_line_count": toc_like_lines,
        "references_heading_count": ref_section_like,
        "url_like_count": url_like_count,
        "multi_paragraph_pages": sum(1 for it in texts if "\n\n" in (it.get("text") or "")),
    }


# ---------------------------------------------------------------------------
# Table-in-text detection (body text containing table-like content)
# ---------------------------------------------------------------------------

def detect_table_in_text(texts: List[Dict]) -> Dict[str, Any]:
    """
    Detect if body text contains table-like content (e.g. v2 dumping tables into text).
    We want tables only in tables.json; table-like content in text can mean poor separation.
    """
    if not texts:
        return {
            "pages_with_table_like_text": 0,
            "table_like_line_count": 0,
            "table_mention_in_text_count": 0,
            "pipe_row_count": 0,
            "table_like_ratio_of_lines": 0.0,
            "explanation": "No text to scan.",
        }
    pipe_pattern = re.compile(r"\|")
    table_mention = re.compile(r"\b(?:Table|TABLE)\s+\d+", re.IGNORECASE)
    min_pipes_for_row = 2
    min_consecutive_rows_for_block = 3

    total_lines = 0
    pipe_row_count = 0
    table_mention_count = 0
    pages_with_table_like_text = 0
    table_like_blocks_count = 0

    for item in texts:
        t = item.get("text", "") or ""
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        total_lines += len(lines)
        page_pipe_rows = 0
        page_mentions = 0
        consecutive_pipe_rows = 0
        for line in lines:
            n_pipes = len(pipe_pattern.findall(line))
            if n_pipes >= min_pipes_for_row:
                pipe_row_count += 1
                page_pipe_rows += 1
                consecutive_pipe_rows += 1
            else:
                if consecutive_pipe_rows >= min_consecutive_rows_for_block:
                    table_like_blocks_count += 1
                consecutive_pipe_rows = 0
            if table_mention.search(line):
                table_mention_count += 1
                page_mentions += 1
        if consecutive_pipe_rows >= min_consecutive_rows_for_block:
            table_like_blocks_count += 1
        if page_pipe_rows >= min_consecutive_rows_for_block or page_mentions > 0:
            pages_with_table_like_text += 1

    table_like_line_count = pipe_row_count
    ratio = table_like_line_count / max(1, total_lines)

    return {
        "pages_with_table_like_text": pages_with_table_like_text,
        "table_like_line_count": table_like_line_count,
        "table_mention_in_text_count": table_mention_count,
        "pipe_row_count": pipe_row_count,
        "table_like_blocks_count": table_like_blocks_count,
        "total_lines_scanned": total_lines,
        "table_like_ratio_of_lines": round(ratio, 4),
        "explanation": (
            "Body text should be mostly prose. High values suggest tables were left in text "
            "(poor table/text separation)."
        ),
    }


# ---------------------------------------------------------------------------
# Table metrics
# ---------------------------------------------------------------------------

def table_metrics(tables_list: List[Dict]) -> Dict[str, Any]:
    if not tables_list:
        return {"table_count": 0}
    table_count = len(tables_list)
    with_caption = sum(1 for t in tables_list if (t.get("caption") or "").strip())
    caption_like_table_n = sum(
        1 for t in tables_list
        if re.search(r"^\s*(?:Table|TABLE)\s+\d+", (t.get("caption") or "").strip(), re.IGNORECASE)
    )
    row_counts: List[int] = []
    col_counts: List[int] = []
    empty_cells_total = 0
    total_cells = 0
    inconsistent_cols = 0
    tables_with_zero_rows = 0
    tables_with_one_row = 0

    for t in tables_list:
        rows = t.get("rows") or []
        if len(rows) == 0:
            tables_with_zero_rows += 1
            continue
        if len(rows) == 1:
            tables_with_one_row += 1
        row_counts.append(len(rows))
        num_cols_0 = len(rows[0]) if rows[0] else 0
        col_counts.append(num_cols_0)
        for r in rows:
            cells = r if isinstance(r, list) else []
            total_cells += len(cells)
            for c in cells:
                if (c is None) or (isinstance(c, str) and not str(c).strip()):
                    empty_cells_total += 1
        for r in rows[1:]:
            if len(r) != num_cols_0:
                inconsistent_cols += 1
                break

    return {
        "table_count": table_count,
        "tables_with_caption": with_caption,
        "caption_ratio": with_caption / max(1, table_count),
        "tables_with_table_N_caption": caption_like_table_n,
        "tables_with_zero_rows": tables_with_zero_rows,
        "tables_with_one_row": tables_with_one_row,
        "total_rows": sum(row_counts),
        "avg_rows_per_table": sum(row_counts) / max(1, len(row_counts)) if row_counts else 0,
        "min_max_rows": (min(row_counts), max(row_counts)) if row_counts else (0, 0),
        "avg_cols_per_table": sum(col_counts) / max(1, len(col_counts)) if col_counts else 0,
        "empty_cell_ratio": empty_cells_total / max(1, total_cells),
        "tables_with_inconsistent_columns": inconsistent_cols,
    }


def table_spo_usability(tables_list: List[Dict]) -> Dict[str, Any]:
    """Measure how usable extracted tables are for SPO (classify + triples from rows only)."""
    try:
        from pipeline.transforms.table_to_sentences import classify_table
        from pipeline.models.table_spo_rules import table_rows_to_spo
    except ImportError:
        return {"triples_total": 0, "tables_yielding_zero_triples": 0, "error": "import failed"}
    triples_total = 0
    tables_yielding_zero = 0
    table_types: Dict[str, int] = defaultdict(int)
    for t in tables_list:
        rows = t.get("rows") or []
        if not rows:
            tables_yielding_zero += 1
            continue
        try:
            table_type = classify_table(rows)
            table_types[table_type] += 1
            spo_list = table_rows_to_spo(rows, table_type, table_title=(t.get("caption") or "").strip() or None)
            n = len(spo_list)
            triples_total += n
            if n == 0:
                tables_yielding_zero += 1
        except Exception:
            tables_yielding_zero += 1
    return {
        "triples_total": triples_total,
        "tables_yielding_zero_triples": tables_yielding_zero,
        "table_type_counts": dict(table_types),
        "triples_per_table_avg": triples_total / max(1, len(tables_list)),
    }


# ---------------------------------------------------------------------------
# Scoring (0–1), Stage A only
# ---------------------------------------------------------------------------

def score_schema(text_valid: bool, table_valid: bool) -> float:
    if text_valid and table_valid:
        return 1.0
    if text_valid or table_valid:
        return 0.5
    return 0.0


def score_text(
    metrics: Dict[str, Any],
    table_in_text: Dict[str, Any],
    schema_ok: bool,
) -> float:
    if not schema_ok or metrics.get("page_count", 0) == 0:
        return 0.0
    page_count = metrics["page_count"]
    total_words = metrics["total_words"]
    empty = metrics.get("empty_pages", 0)
    near_empty = metrics.get("near_empty_pages", 0)
    toc_like = metrics.get("toc_like_line_count", 0)
    url_like = metrics.get("url_like_count", 0)
    table_like_ratio = table_in_text.get("table_like_ratio_of_lines", 0.0)
    pages_with_table_text = table_in_text.get("pages_with_table_like_text", 0)
    table_in_text_penalty = min(0.25, table_like_ratio * 2 + (pages_with_table_text / max(1, page_count)) * 0.1)
    content_score = min(1.0, total_words / 50000)
    empty_penalty = (empty + near_empty) / max(1, page_count)
    noise_penalty = min(0.3, (toc_like * 0.01) + (url_like * 0.002))
    return max(0.0, min(1.0, content_score * (1 - empty_penalty) - noise_penalty - table_in_text_penalty))


def score_tables(metrics: Dict[str, Any], spo: Dict[str, Any], schema_ok: bool) -> float:
    if not schema_ok:
        return 0.0
    n_tables = metrics.get("table_count", 0)
    if n_tables == 0:
        return 0.5
    caption_ratio = metrics.get("caption_ratio", 0)
    zero_row = metrics.get("tables_with_zero_rows", 0)
    one_row = metrics.get("tables_with_one_row", 0)
    empty_ratio = metrics.get("empty_cell_ratio", 0)
    triples = spo.get("triples_total", 0)
    usability = min(1.0, triples / max(1, n_tables) / 5)
    structure = 1.0 - (zero_row + one_row) / max(1, n_tables) - empty_ratio * 0.5
    return max(0.0, min(1.0, 0.4 * caption_ratio + 0.3 * usability + 0.3 * structure))


def composite_score(
    schema_score: float,
    text_score: float,
    table_score: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Composite for Stage A only: schema, text, table (no downstream)."""
    w = weights or {"schema": 1.0 / 3, "text": 1.0 / 3, "table": 1.0 / 3}
    return w["schema"] * schema_score + w["text"] * text_score + w["table"] * table_score


# ---------------------------------------------------------------------------
# Evaluate one version
# ---------------------------------------------------------------------------

def evaluate_one_version(version: str, text_path: Path, tables_path: Path) -> Dict[str, Any]:
    texts, tables_list = load_stage_a_output(text_path, tables_path)
    if texts is None:
        return {
            "version": version,
            "error": "text.json not found",
            "schema_valid": False,
        }
    if tables_list is None:
        tables_list = []

    text_schema = validate_text_schema(texts)
    table_schema = validate_tables_schema(tables_list)
    encoding = check_encoding_and_control_chars(texts)
    text_met = text_metrics(texts)
    table_in_text = detect_table_in_text(texts)
    table_met = table_metrics(tables_list)
    table_spo = table_spo_usability(tables_list)

    schema_ok = text_schema["valid"] and table_schema["valid"]
    s_schema = score_schema(text_schema["valid"], table_schema["valid"])
    s_text = score_text(text_met, table_in_text, schema_ok)
    s_table = score_tables(table_met, table_spo, schema_ok)
    composite = composite_score(s_schema, s_text, s_table)

    return {
        "version": version,
        "schema": {
            "text": text_schema,
            "tables": table_schema,
            "encoding": encoding,
        },
        "text_metrics": text_met,
        "table_in_text": table_in_text,
        "table_metrics": table_met,
        "table_spo_usability": table_spo,
        "scores": {
            "schema": round(s_schema, 4),
            "text": round(s_text, 4),
            "table": round(s_table, 4),
            "composite": round(composite, 4),
        },
    }


def write_report(reports: List[Dict[str, Any]], paths_evaluated: Dict[str, Tuple[Path, Path]]) -> None:
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    by_version = {r["version"]: r for r in reports}
    comparison: Dict[str, Any] = {}
    for ver, r in by_version.items():
        comparison[f"{ver}_composite"] = r.get("scores", {}).get("composite", 0)
        comparison[f"{ver}_pages"] = r.get("text_metrics", {}).get("page_count", 0)
        comparison[f"{ver}_words"] = r.get("text_metrics", {}).get("total_words", 0)
        comparison[f"{ver}_tables"] = r.get("table_metrics", {}).get("table_count", 0)
        comparison[f"{ver}_triples"] = r.get("table_spo_usability", {}).get("triples_total", 0)
        ti = r.get("table_in_text", {})
        comparison[f"{ver}_pages_with_table_in_text"] = ti.get("pages_with_table_like_text", 0)
        comparison[f"{ver}_pipe_row_count"] = ti.get("pipe_row_count", 0)
        comparison[f"{ver}_table_like_ratio"] = ti.get("table_like_ratio_of_lines", 0)

    report = {
        "evaluation": "stage_a",
        "versions": by_version,
        "comparison": comparison,
    }
    json_path = EVAL_OUTPUT_DIR / "stage_a_evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Wrote {json_path}")

    run_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    report_lines = [
        "=" * 70,
        "STAGE A EVALUATION REPORT",
        "=" * 70,
        "",
        f"Run at: {run_time}",
        f"Versions evaluated: {', '.join(r['version'] for r in reports)}",
        "",
        "USE CASE",
        "-" * 70,
        "Stage A extracts text and tables from guideline PDFs. Downstream we use:",
        "  - Body text: for recommendation/clinical sentences and entity recognition.",
        "  - Tables (in tables.json): for structured SPO triples (e.g. drug dosing, COR/LOE).",
        "We want: (1) body text = mostly prose, not tables; (2) tables only in tables.json;",
        "(3) minimal noise (TOC, references, URLs) in text; (4) valid schema.",
        "",
        "WHAT WE MEASURE",
        "-" * 70,
        "  Schema       Output files have correct structure (required for pipeline).",
        "  Text score  Body text quality: enough content, few empty pages, little noise,",
        "              and little table-like content in text (tables should be separate).",
        "  Table score Tables in tables.json: captions, structure, and SPO usability.",
        "  Table-in-text  How much table-like content (e.g. pipe-separated rows) appears",
        "                 in body text. High = possible poor table/text separation.",
        "",
        "Inputs evaluated (paths that were read):",
    ]
    for ver in [r["version"] for r in reports]:
        text_p, tables_p = paths_evaluated.get(ver, (None, None))
        if text_p and tables_p:
            report_lines.append(f"  {ver}:")
            report_lines.append(f"    text:   {text_p}")
            report_lines.append(f"    tables: {tables_p}")
        else:
            report_lines.append(f"  {ver}: (not found)")
    report_lines.extend([
        "",
        "RESULTS",
        "-" * 70,
        "",
        "Composite scores (0-1, higher is better):",
    ])
    for r in reports:
        report_lines.append(f"  {r['version']}: {r.get('scores', {}).get('composite', 0):.4f}")
    report_lines.extend([
        "",
        "Score breakdown (schema | text | table):",
    ])
    for r in reports:
        s = r.get("scores", {})
        report_lines.append(
            "  {}  {:+.4f} | {:+.4f} | {:+.4f}".format(
                r["version"], s.get("schema", 0), s.get("text", 0), s.get("table", 0)
            )
        )
    report_lines.extend([
        "",
        "Body text (for recommendations / NER):",
        "  Metric          Meaning",
        "  pages           Number of pages with extracted text.",
        "  total_words     Total word count (proxy for content available for chunking).",
        "  empty/near-empty  Pages with no or very little text (bad).",
    ])
    for r in reports:
        tm = r.get("text_metrics", {})
        report_lines.append(
            "  {}  pages: {},  total words: {},  empty/near-empty: {}/{}".format(
                r["version"],
                tm.get("page_count", 0),
                tm.get("total_words", 0),
                tm.get("empty_pages", 0),
                tm.get("near_empty_pages", 0),
            )
        )
    report_lines.extend([
        "",
        "Table-in-text (tables leaking into body text; lower is better):",
        "  Metric                    Meaning",
        "  pages_with_table_like     Pages with table-like content (pipe rows or 'Table N' in text).",
        "  pipe_row_count            Lines with 2+ pipes (markdown-style table rows in body text).",
        "  table_like_ratio          Share of all lines that are pipe-rows (high = tables in text).",
    ])
    for r in reports:
        ti = r.get("table_in_text", {})
        report_lines.append(
            "  {}  pages_with_table_like: {},  pipe_row_count: {},  table_like_ratio: {}".format(
                r["version"],
                ti.get("pages_with_table_like_text", 0),
                ti.get("pipe_row_count", 0),
                ti.get("table_like_ratio_of_lines", 0),
            )
        )
    report_lines.extend([
        "",
        "Tables (in tables.json; for SPO triples):",
        "  Metric          Meaning",
        "  tables           Number of extracted tables.",
        "  with_caption     Tables that have a caption (helps SPO and classification).",
        "  SPO triples      Triples derivable from table rows (recommendation/drug/list).",
    ])
    for r in reports:
        tbl = r.get("table_metrics", {})
        spo = r.get("table_spo_usability", {})
        report_lines.append(
            "  {}  tables: {},  with caption: {},  SPO triples: {}".format(
                r["version"],
                tbl.get("table_count", 0),
                tbl.get("tables_with_caption", 0),
                spo.get("triples_total", 0),
            )
        )

    report_lines.extend([
        "",
        "ANALYSIS",
        "-" * 70,
    ])
    if len(reports) >= 2:
        best = max(reports, key=lambda r: r.get("scores", {}).get("composite", 0))
        report_lines.append(f"  Highest composite: {best['version']} ({best.get('scores', {}).get('composite', 0):.4f})")
        for r in reports:
            tm = r.get("text_metrics", {})
            ti = r.get("table_in_text", {})
            report_lines.append(
                f"  {r['version']} text: {tm.get('page_count', 0)} pages, {tm.get('total_words', 0)} words; "
                f"table-in-text: {ti.get('pages_with_table_like_text', 0)} pages, {ti.get('pipe_row_count', 0)} pipe-rows."
            )
        for r in reports:
            tbl = r.get("table_metrics", {})
            spo = r.get("table_spo_usability", {})
            report_lines.append(
                f"  {r['version']} tables: {tbl.get('table_count', 0)} tables, "
                f"{tbl.get('tables_with_caption', 0)} with caption, {spo.get('triples_total', 0)} SPO triples."
            )
    report_lines.extend([
        "",
        "Output files written:",
        f"  - {json_path} (full metrics and scores)",
        f"  - {EVAL_OUTPUT_DIR / 'stage_a_evaluation_report.txt'} (this report)",
        "",
        "=" * 70,
        "Evaluation report complete.",
        "=" * 70,
    ])

    report_path = EVAL_OUTPUT_DIR / "stage_a_evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Wrote {report_path}")
    print("\n" + "\n".join(report_lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage A evaluation: analyze and score extraction outputs (text + tables) for one or more versions."
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        default=DEFAULT_VERSIONS,
        help="Version names to evaluate (e.g. v1 v2 v3). Each reads outputs/STAGE_A_{version}/",
    )
    args = parser.parse_args()
    versions = args.versions

    reports = []
    paths_evaluated: Dict[str, Tuple[Path, Path]] = {}
    for ver in versions:
        stage_dir = OUTPUTS_DIR / f"STAGE_A_{ver}"
        text_path = stage_dir / "text.json"
        tables_path = stage_dir / "tables.json"
        paths_evaluated[ver] = (text_path, tables_path)
        print(f"Loading STAGE_A_{ver}...")
        report = evaluate_one_version(ver, text_path, tables_path)
        reports.append(report)
        if "error" in report:
            print(f"  Warning: {report['error']}")

    write_report(reports, paths_evaluated)


if __name__ == "__main__":
    main()
