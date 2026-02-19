import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.data import PDFGuidelines, RawText
from pipeline.transforms import ExtractText

OUTPUTS_DIR = Path(__file__).parent / "outputs"
RAW_TEXT_FILE = OUTPUTS_DIR / "stage_a_raw_text.json"
TABLES_FILE = OUTPUTS_DIR / "stage_a_tables.json"


def test_stage_a():
    print("=" * 70)
    print("STAGE A: PDF -> raw_text_and_tables (raw text + tables)")
    print("=" * 70)
    pdf_dir = "data"
    skip_first_pages = 3
    skip_last_pages = 5
    OUTPUTS_DIR.mkdir(exist_ok=True)
    print("\n[1] Loading PDF guidelines...")
    pdf_guidelines = PDFGuidelines(pdf_dir=pdf_dir)
    print(f"    Found: {pdf_guidelines}")
    if pdf_guidelines.count() == 0:
        print("    ERROR: No PDF files found!")
        return None
    print(f"\n[2] Initializing text extractor...")
    print(f"    Skip first pages: {skip_first_pages}")
    print(f"    Skip last pages: {skip_last_pages}")
    extractor = ExtractText(
        skip_first_pages=skip_first_pages,
        skip_last_pages=skip_last_pages
    )
    print("\n[3] Extracting text and tables from PDFs...")
    raw_text = extractor.transform(pdf_guidelines)
    print(f"    Result: {raw_text}")
    pages = raw_text.get_pages()
    total_tables = sum(len(p.get("tables", [])) for p in pages)

    # raw_text_and_tables: two files — raw text and tables
    # File 1: stage_a_raw_text.json (text only, no "tables" key)
    print(f"\n[4] Saving raw_text_and_tables: text -> {RAW_TEXT_FILE}...")
    raw_text_data = {
        "metadata": {
            "stage": "a",
            "description": "Raw text (part of raw_text_and_tables)",
            "total_pages": len(pages),
            "skip_first_pages": skip_first_pages,
            "skip_last_pages": skip_last_pages,
        },
        "pages": [
            {"page": p["page"], "text": p["text"], "source": p.get("source", "")}
            for p in pages
        ],
    }
    RAW_TEXT_FILE.write_text(
        json.dumps(raw_text_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # File 2: stage_a_tables.json (tables only)
    print(f"    Saving raw_text_and_tables: tables -> {TABLES_FILE}...")
    tables_data = {
        "metadata": {
            "stage": "a",
            "description": "Tables (part of raw_text_and_tables)",
            "total_pages": len(pages),
            "total_tables": total_tables,
            "skip_first_pages": skip_first_pages,
            "skip_last_pages": skip_last_pages,
        },
        "pages": [
            {
                "page": p["page"],
                "source": p.get("source", ""),
                "tables": p.get("tables", []),
            }
            for p in pages
        ],
    }
    TABLES_FILE.write_text(
        json.dumps(tables_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("STAGE A COMPLETE (raw_text_and_tables)")
    print("=" * 70)
    print(f"  Total pages: {raw_text.count()}")
    print(f"  Total tables: {total_tables}")
    print(f"  Raw text:  {RAW_TEXT_FILE}")
    print(f"  Tables:    {TABLES_FILE}")
    print("=" * 70)
    return raw_text


if __name__ == "__main__":
    test_stage_a()
