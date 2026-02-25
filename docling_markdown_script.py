#!/usr/bin/env python3
"""
Docling native export script: read PDFs (and other supported docs) from input/,
convert with Docling's default pipeline, and write Docling's standard outputs
(.md and .json) into outputs/docling_markdown/.

This script uses Docling's actual API and structure (no custom reshaping),
so you can see Docling's real capabilities and document model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from docling.document_converter import DocumentConverter
except ImportError:
    print("Docling is required. Install with: pip install docling", file=sys.stderr)
    sys.exit(1)


def main(
    input_dir: str | Path = "input",
    output_dir: str | Path = "outputs/docling_markdown",
) -> None:
    input_path = Path(input_dir)
    out_path = Path(output_dir)

    if not input_path.is_dir():
        print(f"Input directory not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Collect supported inputs (PDF and other formats Docling handles by default)
    allowed = (".pdf", ".docx", ".pptx", ".html", ".htm", ".md", ".png", ".jpg", ".jpeg", ".tiff", ".tif")
    input_files = sorted(
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in allowed
    )

    if not input_files:
        print(f"No supported files in {input_path}. Supported: {allowed}", file=sys.stderr)
        sys.exit(1)

    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Input: {input_path.absolute()}")
    print(f"Output: {out_path.absolute()}")
    print(f"Converting {len(input_files)} file(s) with Docling (native export)...")

    # Docling default converter – no custom pipeline
    converter = DocumentConverter()
    results = list(converter.convert_all([str(f) for f in input_files]))

    for res in results:
        try:
            name = Path(res.input.file).stem if res.input and getattr(res.input, "file", None) else None
        except Exception:
            name = None
        if not name:
            name = "document"

        doc = getattr(res, "document", None)
        if doc is None:
            print(f"  Skip {name}: no document in result")
            continue

        # Docling standard: Markdown
        md_file = out_path / f"{name}.md"
        md_file.write_text(doc.export_to_markdown(), encoding="utf-8")
        print(f"  Markdown: {md_file.name}")

        # Docling standard: JSON (document model as dict)
        json_file = out_path / f"{name}.json"
        json_file.write_text(
            json.dumps(doc.export_to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  JSON:    {json_file.name}")

    print("Done.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Convert input docs with Docling and write native .md and .json to outputs/docling_markdown/")
    p.add_argument("--input", "-i", default="input", help="Input directory (default: input)")
    p.add_argument("--output", "-o", default="outputs/docling_markdown", help="Output directory (default: outputs/docling_markdown)")
    args = p.parse_args()
    main(input_dir=args.input, output_dir=args.output)
