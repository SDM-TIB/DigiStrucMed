"""
Stage A: Extract text and tables from PDFs.

Supports v1 (PyMuPDF) and v2 (Docling). By default reads from input/ and
writes to the configured Stage A directory for the given version.

Use --output-dir to override the output location (e.g. when called by orchestrator).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from pipeline.data import PDFGuidelines
from pipeline.transforms import ExtractText, ExtractTextV2


def test_stage_a(
    version: str | None = None,
    output_dir: str | Path | None = None,
    pdf_dir: str | Path | None = None,
    skip_first_pages: int = 3,
    skip_last_pages: int = 5,
):
    """Run Stage A with the given version (v1 or v2)."""
    ver = version or config.DEFAULT_STAGE_A_VERSION
    out_dir = Path(output_dir) if output_dir is not None else config.stage_a_dir(ver)
    out_dir_str = str(out_dir)
    pdf_dir_str = str(pdf_dir) if pdf_dir is not None else str(config.PDF_DIR)

    print("=" * 70)
    print(f"STAGE A {ver}: PDF -> raw text + tables")
    print("=" * 70)

    print("\n[1] Loading PDF guidelines...")
    pdf_guidelines = PDFGuidelines(pdf_dir=pdf_dir_str)
    print(f"    Found: {pdf_guidelines}")
    if pdf_guidelines.count() == 0:
        print("    ERROR: No PDF files found!")
        return None

    print(f"\n[2] Initializing Stage A {ver} extractor...")
    print(f"    Skip first pages: {skip_first_pages}")
    print(f"    Skip last pages: {skip_last_pages}")
    print(f"    Output dir: {out_dir_str}")

    if ver == "v2":
        extractor = ExtractTextV2(
            skip_first_pages=skip_first_pages,
            skip_last_pages=skip_last_pages,
            stage_output_dir=out_dir_str,
        )
    else:
        extractor = ExtractText(
            skip_first_pages=skip_first_pages,
            skip_last_pages=skip_last_pages,
            stage_output_dir=out_dir_str,
        )

    print("\n[3] Extracting text and tables from PDFs...")
    raw_text = extractor.transform(pdf_guidelines)
    print(f"    Result: {raw_text}")
    pages = raw_text.get_pages()
    total_tables = sum(len(p.get("tables", [])) for p in pages)

    print("\n" + "=" * 70)
    print(f"STAGE A {ver} COMPLETE")
    print("=" * 70)
    print(f"  Total pages: {raw_text.count()}")
    print(f"  Total tables: {total_tables}")
    print(f"  Output: {out_dir_str}/text.json, {out_dir_str}/tables.json")
    print("=" * 70)
    return raw_text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Stage A (text + table extraction).")
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Stage A version (v1 or v2). Default: config.DEFAULT_STAGE_A_VERSION.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: config.stage_a_dir(version).",
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=None,
        help=f"Directory containing PDFs. Default: config.PDF_DIR ({config.PDF_DIR})",
    )
    args = parser.parse_args()

    test_stage_a(
        version=args.version,
        output_dir=args.output_dir,
        pdf_dir=args.pdf_dir or None,
    )
