"""
Stage B v1: Content preparation.
Inputs: outputs/STAGE_A_v1/text.json + tables.json.
Outputs: outputs/STAGE_B_v1/stage_b_text_chunks.json, stage_b_table_triples.json.
"""
import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.data import RawText
from pipeline.models import ParsingRules
from pipeline.transforms import ContentPreparation

PROJECT_ROOT = Path(__file__).parent.parent
STAGE_A_V1_DIR = PROJECT_ROOT / "outputs" / "STAGE_A_v1"
STAGE_B_V1_DIR = PROJECT_ROOT / "outputs" / "STAGE_B_v1"

TEXT_FILE = STAGE_A_V1_DIR / "text.json"
TABLES_FILE = STAGE_A_V1_DIR / "tables.json"
OUTPUT_TEXT_CHUNKS = STAGE_B_V1_DIR / "stage_b_text_chunks.json"
OUTPUT_TABLE_TRIPLES = STAGE_B_V1_DIR / "stage_b_table_triples.json"


def load_stage_a_v1(text_path: Path, tables_path: Path) -> RawText | None:
    """Load Stage A v1 output (text.json + tables.json) into RawText."""
    if not text_path.exists():
        return None, None
    with open(text_path, "r", encoding="utf-8") as f:
        texts = json.load(f)
    if not isinstance(texts, list):
        texts = texts.get("pages", texts) if isinstance(texts, dict) else []
    tables_list = []
    if tables_path.exists():
        with open(tables_path, "r", encoding="utf-8") as f:
            tables_list = json.load(f)
    if not isinstance(tables_list, list):
        tables_list = []

    tables_by_key = defaultdict(list)
    for t in tables_list:
        key = (t.get("source_file", ""), t.get("page", 0))
        tables_by_key[key].append({"title": t.get("caption", ""), "rows": t.get("rows", [])})

    raw_text = RawText()
    pages_with_tables = []
    for item in texts:
        source_file = item.get("source_file", "")
        page_num = item.get("page", 0)
        text = item.get("text", "")
        key = (source_file, page_num)
        tables = tables_by_key.get(key, [])
        raw_text.add_page(
            page_num=page_num,
            text=text,
            source_file=source_file,
            tables=tables,
        )
    return raw_text


def test_stage_b():
    print("=" * 70)
    print("STAGE B v1: raw_text_and_tables -> content_preparation -> text_chunks + table_triples")
    print("=" * 70)
    min_chunk_chars = 40
    STAGE_B_V1_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[1] Loading Stage A v1 output: {TEXT_FILE} + {TABLES_FILE}...")
    raw_text = load_stage_a_v1(TEXT_FILE, TABLES_FILE)
    if raw_text is None:
        print("    ERROR: Stage A v1 output not found! Run stagee_a_extract_text_table.py first.")
        return None
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

    print("\n" + "=" * 70)
    print("STAGE B v1 COMPLETE")
    print("=" * 70)
    print(f"  Text chunks: {text_chunks.count()} (table-derived: {table_derived_count})")
    print(f"  Table triples: {len(table_triples)}")
    print(f"  Outputs: {OUTPUT_TEXT_CHUNKS.name}, {OUTPUT_TABLE_TRIPLES.name}")
    print(f"  Dir: {STAGE_B_V1_DIR}")
    print("=" * 70)
    return result


if __name__ == "__main__":
    test_stage_b()
