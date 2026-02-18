import sys
import json
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.data import RawText, TextChunks
from pipeline.models import ParsingRules
from pipeline.transforms import ChunkText
from pipeline.transforms.table_to_sentences import classify_table
INPUT_FILE = Path(__file__).parent / "outputs" / "stage_a_raw_text.json"
OUTPUT_FILE = Path(__file__).parent / "outputs" / "stage_b_text_chunks.json"
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
        "description": "Compare Stage A tables to Stage B table-derived sentences for transformation verification",
        "total_pages_with_tables": len(report["by_page"]),
    }
    return report
def test_stage_b():
    print("=" * 70)
    print("STAGE b: raw_text -> chunk_text -> text_chunks")
    print("=" * 70)
    min_chunk_chars = 40
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    print(f"\n[1] Loading Stage a output from {INPUT_FILE}...")
    if not INPUT_FILE.exists():
        print("    ERROR: Stage a output not found! Run stage_a_extract_text.py first.")
        return None
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        stage_a_data = json.load(f)
    raw_text = RawText()
    for page in stage_a_data["pages"]:
        raw_text.add_page(
            page_num=page["page"],
            text=page["text"],
            source_file=page["source"],
            tables=page.get("tables", []),
        )
    print(f"    Loaded: {raw_text}")
    print(f"\n[2] Initializing chunk_text transform...")
    print(f"    Minimum chunk chars: {min_chunk_chars}")
    parsing_rules = ParsingRules()
    chunker = ChunkText(
        parsing_rules=parsing_rules,
        min_chars=min_chunk_chars
    )
    print("\n[3] Chunking text...")
    text_chunks = chunker.transform(raw_text)
    print(f"    Result: {text_chunks}")
    table_derived_count = sum(1 for c in text_chunks.get_chunks() if c.get("from_table"))
    print(f"    Table-derived chunks: {table_derived_count}")
    print(f"\n[4] Saving output to {OUTPUT_FILE}...")
    output_data = {
        "metadata": {
            "stage": "b",
            "description": "Text chunks from chunk_text",
            "total_chunks": text_chunks.count(),
            "table_derived_chunks": table_derived_count,
            "min_chunk_chars": min_chunk_chars
        },
        "chunks": text_chunks.get_chunks()
    }
    OUTPUT_FILE.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\n[5] Building table comparison report -> {TABLE_COMPARISON_FILE}...")
    comparison = build_table_comparison(stage_a_data, text_chunks.get_chunks())
    TABLE_COMPARISON_FILE.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("\n" + "=" * 70)
    print("STAGE b COMPLETE")
    print("=" * 70)
    print(f"  Total chunks: {text_chunks.count()}")
    print(f"  Table-derived chunks: {table_derived_count}")
    print(f"  Output file: {OUTPUT_FILE}")
    print(f"  Table comparison report: {TABLE_COMPARISON_FILE}")
    print("=" * 70)
    return text_chunks
if __name__ == "__main__":
    test_stage_b()
