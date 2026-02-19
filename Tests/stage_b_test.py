"""
Stage B: Content preparation.
Inputs: raw_text_and_tables (stage_a_raw_text.json + stage_a_tables.json).
Outputs: stage_b_text_chunks.json, stage_b_table_triples.json.
"""
import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.data import RawText
from pipeline.models import ParsingRules
from pipeline.transforms import ContentPreparation
from pipeline.transforms.table_to_sentences import classify_table

INPUT_FILE = Path(__file__).parent / "outputs" / "stage_a_raw_text.json"
TABLES_FILE = Path(__file__).parent / "outputs" / "stage_a_tables.json"
OUTPUT_TEXT_CHUNKS = Path(__file__).parent / "outputs" / "stage_b_text_chunks.json"
OUTPUT_TABLE_TRIPLES = Path(__file__).parent / "outputs" / "stage_b_table_triples.json"
TABLE_COMPARISON_FILE = Path(__file__).parent / "outputs" / "stage_b_table_comparison.json"


def build_table_comparison(stage_a_data: dict, chunks: list) -> dict:
    by_page_table = defaultdict(lambda: defaultdict(list))
    for c in chunks:
        if not c.get("from_table"):
            continue
        page = c["page"]
        ti = c.get("table_index", 0)
        by_page_table[page][ti].append({
            "row_index": c.get("row_index"),
            "sentence": c["text"],
            "chunk_id": c.get("chunk_id"),
        })
    for page in by_page_table:
        for ti in by_page_table[page]:
            by_page_table[page][ti].sort(key=lambda x: (x["row_index"] is None, x["row_index"] or 0))
    report = {"by_page": {}}
    for page_data in stage_a_data.get("pages", []):
        page_num = page_data["page"]
        tables = page_data.get("tables", [])
        if not tables:
            continue
        page_key = str(page_num)
        report["by_page"][page_key] = []
        for ti, table in enumerate(tables):
            rows = table.get("rows", [])
            table_type = classify_table(rows)
            sentences = by_page_table.get(page_num, {}).get(ti, [])
            report["by_page"][page_key].append({
                "table_index": ti,
                "table_type": table_type,
                "stage_a_row_count": len(rows),
                "stage_a_rows": rows,
                "stage_b_sentences": sentences,
                "stage_b_sentence_count": len(sentences),
            })
    report["metadata"] = {
        "description": "Compare Stage A tables to Stage B table-derived sentences",
        "total_pages_with_tables": len(report["by_page"]),
    }
    return report


def test_stage_b():
    print("=" * 70)
    print("STAGE B: raw_text_and_tables -> content_preparation -> text_chunks + table_triples")
    print("=" * 70)
    min_chunk_chars = 40
    OUTPUT_TEXT_CHUNKS.parent.mkdir(exist_ok=True)

    print(f"\n[1] Loading Stage A output (raw_text_and_tables): {INPUT_FILE} + {TABLES_FILE}...")
    if not INPUT_FILE.exists():
        print("    ERROR: Stage A output not found! Run stagee_a_extract_text_table.py first.")
        return None
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        stage_a_data = json.load(f)
    tables_by_page_source = {}
    if TABLES_FILE.exists():
        with open(TABLES_FILE, "r", encoding="utf-8") as f:
            tables_data = json.load(f)
        for p in tables_data.get("pages", []):
            key = (p["page"], p.get("source", ""))
            tables_by_page_source[key] = p.get("tables", [])
    raw_text = RawText()
    pages_with_tables = []
    for page in stage_a_data["pages"]:
        page_num = page["page"]
        source = page.get("source", "")
        key = (page_num, source)
        tables = tables_by_page_source.get(key, page.get("tables", []))
        raw_text.add_page(
            page_num=page_num,
            text=page["text"],
            source_file=source,
            tables=tables,
        )
        pages_with_tables.append({"page": page_num, "source": source, "tables": tables})
    stage_a_data_for_comparison = {"pages": pages_with_tables}
    print(f"    Loaded: {raw_text}")

    print(f"\n[2] Initializing content_preparation transform...")
    print(f"    Minimum chunk chars: {min_chunk_chars}")
    parsing_rules = ParsingRules()
    content_prep = ContentPreparation(
        parsing_rules=parsing_rules,
        min_chars=min_chunk_chars,
    )

    print("\n[3] Running content preparation (text chunks + table triples)...")
    result = content_prep.transform(raw_text)
    text_chunks = result.get_text_chunks()
    table_triples = result.get_table_triples()
    table_derived_count = sum(1 for c in text_chunks.get_chunks() if c.get("from_table"))
    print(f"    Text chunks: {text_chunks.count()} (table-derived: {table_derived_count})")
    print(f"    Table triples: {len(table_triples)}")

    print(f"\n[4] Saving outputs...")
    output_chunks_data = {
        "metadata": {
            "stage": "b",
            "description": "Text chunks from content_preparation",
            "total_chunks": text_chunks.count(),
            "table_derived_chunks": table_derived_count,
            "min_chunk_chars": min_chunk_chars,
        },
        "chunks": text_chunks.get_chunks(),
    }
    OUTPUT_TEXT_CHUNKS.write_text(
        json.dumps(output_chunks_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output_triples_data = {
        "metadata": {
            "stage": "b",
            "description": "Table SPO triples from content_preparation",
            "total_triples": len(table_triples),
        },
        "triples": table_triples,
    }
    OUTPUT_TABLE_TRIPLES.write_text(
        json.dumps(output_triples_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"    Text chunks -> {OUTPUT_TEXT_CHUNKS}")
    print(f"    Table triples -> {OUTPUT_TABLE_TRIPLES}")

    print(f"\n[5] Building table comparison report -> {TABLE_COMPARISON_FILE}...")
    comparison = build_table_comparison(stage_a_data_for_comparison, text_chunks.get_chunks())
    TABLE_COMPARISON_FILE.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("STAGE B COMPLETE")
    print("=" * 70)
    print(f"  Text chunks: {text_chunks.count()} (table-derived: {table_derived_count})")
    print(f"  Table triples: {len(table_triples)}")
    print(f"  Outputs: {OUTPUT_TEXT_CHUNKS.name}, {OUTPUT_TABLE_TRIPLES.name}")
    print(f"  Table comparison: {TABLE_COMPARISON_FILE.name}")
    print("=" * 70)
    return result


if __name__ == "__main__":
    test_stage_b()
