"""Run Stage A v2 (Docling) directly: read from input, write to outputs/STAGE_A_v2."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.data import PDFGuidelines
from pipeline.transforms import ExtractTextV2

STAGE_A_OUTPUT_DIR = "outputs/STAGE_A_v2"


def test_stage_a_v2():
    print("=" * 70)
    print("STAGE A v2 (Docling): PDF -> raw text + tables (read from input, write to outputs)")
    print("=" * 70)
    pdf_dir = "input"
    skip_first_pages = 3
    skip_last_pages = 5
    print("\n[1] Loading PDF guidelines...")
    pdf_guidelines = PDFGuidelines(pdf_dir=pdf_dir)
    print(f"    Found: {pdf_guidelines}")
    if pdf_guidelines.count() == 0:
        print("    ERROR: No PDF files found!")
        return None
    print(f"\n[2] Initializing Stage A v2 extractor (Docling)...")
    print(f"    Skip first pages: {skip_first_pages}")
    print(f"    Skip last pages: {skip_last_pages}")
    print(f"    Output dir: {STAGE_A_OUTPUT_DIR}")
    extractor = ExtractTextV2(
        skip_first_pages=skip_first_pages,
        skip_last_pages=skip_last_pages,
        stage_output_dir=STAGE_A_OUTPUT_DIR,
    )
    print("\n[3] Extracting text and tables from PDFs...")
    raw_text = extractor.transform(pdf_guidelines)
    print(f"    Result: {raw_text}")
    pages = raw_text.get_pages()
    total_tables = sum(len(p.get("tables", [])) for p in pages)

    print("\n" + "=" * 70)
    print("STAGE A v2 COMPLETE")
    print("=" * 70)
    print(f"  Total pages: {raw_text.count()}")
    print(f"  Total tables: {total_tables}")
    print(f"  Output: {STAGE_A_OUTPUT_DIR}/text.json, {STAGE_A_OUTPUT_DIR}/tables.json")
    print("=" * 70)
    print("Done. Check outputs/STAGE_A_v2/text.json and tables.json")
    return raw_text


if __name__ == "__main__":
    test_stage_a_v2()
