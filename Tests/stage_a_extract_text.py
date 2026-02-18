import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.data import PDFGuidelines, RawText
from pipeline.transforms import ExtractText
def test_stage_a():
    print("=" * 70)
    print("STAGE a: PDF -> extract_text -> raw_text")
    print("=" * 70)
    pdf_dir = "data"
    skip_first_pages = 3
    skip_last_pages = 5
    output_file = Path(__file__).parent / "outputs" / "stage_a_raw_text.json"
    output_file.parent.mkdir(exist_ok=True)
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
    print("\n[3] Extracting text from PDFs...")
    raw_text = extractor.transform(pdf_guidelines)
    print(f"    Result: {raw_text}")
    print(f"\n[4] Saving output to {output_file}...")
    output_data = {
        "metadata": {
            "stage": "a",
            "description": "Raw text extracted from PDFs",
            "total_pages": raw_text.count(),
            "skip_first_pages": skip_first_pages,
            "skip_last_pages": skip_last_pages
        },
        "pages": raw_text.get_pages()
    }
    output_file.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("\n" + "=" * 70)
    print("STAGE a COMPLETE")
    print("=" * 70)
    print(f"  Total pages extracted: {raw_text.count()}")
    print(f"  Output file: {output_file}")
    print("=" * 70)
    return raw_text
if __name__ == "__main__":
    test_stage_a()
