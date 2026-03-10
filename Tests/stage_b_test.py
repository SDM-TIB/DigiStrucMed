"""
Stage B v1: Content preparation.

By default, reads Stage A output from the directory configured in config.py
and writes Stage B output to the configured Stage B directory.

You can override this by:
- Passing an explicit input directory (containing text.json + tables.json)
- Passing an explicit output directory
- Specifying a Stage A version, which is resolved via config.stage_a_dir(...)
"""
import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from pipeline.data import RawText
from pipeline.models import ParsingRules
from pipeline.transforms import ContentPreparation


def load_stage_a_v1(text_path: Path, tables_path: Path) -> RawText | None:
    """Load Stage A v1 output (text.json + tables.json) into RawText."""
    if not text_path.exists():
        return None
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


def _resolve_stage_b_paths(
    stage_a_version: str | None = None,
    stage_b_version: str | None = None,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """
    Resolve input/output directories for Stage B.

    Input: from previous stage (Stage A).
    Output: where this stage writes.

    Priority:
    1. Explicit input_dir / output_dir (if provided)
    2. stage_a_version -> config.stage_a_dir() for input
    3. stage_b_version -> config.stage_b_dir() for output
    4. Defaults from config
    """
    if input_dir is not None:
        in_dir = Path(input_dir)
    else:
        in_dir = config.stage_a_dir(stage_a_version)

    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        out_dir = config.stage_b_dir(stage_b_version)

    return in_dir, out_dir


def test_stage_b(
    stage_a_version: str | None = None,
    stage_b_version: str | None = None,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
):
    effective_b_version = stage_b_version or config.DEFAULT_STAGE_B_VERSION
    print("=" * 70)
    print(f"STAGE B {effective_b_version}: raw_text_and_tables -> content_preparation -> text_chunks + table_triples")
    print("=" * 70)
    min_chunk_chars = 40

    input_root, output_root = _resolve_stage_b_paths(
        stage_a_version=stage_a_version,
        stage_b_version=stage_b_version,
        input_dir=input_dir,
        output_dir=output_dir,
    )
    text_file = input_root / "text.json"
    tables_file = input_root / "tables.json"

    print(f"\n[1] Loading Stage A output from: {text_file} + {tables_file}...")
    raw_text = load_stage_a_v1(text_file, tables_file)
    if raw_text is None:
        print("    ERROR: Stage A output not found! Check input directory or run Stage A first.")
        return None
    print(f"    Loaded: {raw_text}")

    print(f"\n[2] Initializing content_preparation transform...")
    print(f"    Minimum chunk chars: {min_chunk_chars}")
    print(f"    Stage B output dir: {output_root}")
    parsing_rules = ParsingRules()
    content_prep = ContentPreparation(
        parsing_rules=parsing_rules,
        min_chars=min_chunk_chars,
        stage_output_dir=str(output_root),
    )

    print("\n[3] Running content preparation (text chunks + table triples)...")
    result = content_prep.transform(raw_text)
    text_chunks = result.get_text_chunks()
    table_triples = result.get_table_triples()
    table_derived_count = sum(1 for c in text_chunks.get_chunks() if c.get("from_table"))
    print(f"    Text chunks: {text_chunks.count()} (table-derived: {table_derived_count})")
    print(f"    Table triples: {len(table_triples)}")
    print(f"    Output written to {output_root} (stage_b_text_chunks.json, stage_b_table_triples.json)")

    print("\n" + "=" * 70)
    print(f"STAGE B {effective_b_version} COMPLETE")
    print("=" * 70)
    print(f"  Text chunks: {text_chunks.count()} (table-derived: {table_derived_count})")
    print(f"  Table triples: {len(table_triples)}")
    print(f"  Outputs: {output_root / 'stage_b_text_chunks.json'}, {output_root / 'stage_b_table_triples.json'}")
    print("=" * 70)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Stage B v1 (content preparation).")
    parser.add_argument(
        "--stage-a-version",
        type=str,
        default=None,
        help="Stage A version to read from (e.g. v1, v2). If not set, uses config.DEFAULT_STAGE_A_VERSION.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Explicit input directory containing text.json and tables.json. Overrides --stage-a-version.",
    )
    parser.add_argument(
        "--stage-b-version",
        type=str,
        default=None,
        help="Stage B version to write to (e.g. v1, v2). Default: config.DEFAULT_STAGE_B_VERSION.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Explicit output directory. Overrides --stage-b-version.",
    )
    args = parser.parse_args()

    test_stage_b(
        stage_a_version=args.stage_a_version,
        stage_b_version=args.stage_b_version,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
