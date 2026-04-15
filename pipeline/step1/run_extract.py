"""
Step 1 runner — extract text and tables from a PDF.

Defaults target ``outputs/pipeline-output18/step1`` (override with ``--out``).

Usage (repo root):
  python -m pipeline.step1.run_extract
  python -m pipeline.step1.run_extract path/to/other.pdf --version v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_OUT = Path("outputs/pipeline-output18/step1")
DEFAULT_PDF = Path("input") / "Heidenreich, 2022, AHA,ACC,HFSA guidelines.pdf"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", nargs="?", type=Path, default=DEFAULT_PDF)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="step1 output directory")
    ap.add_argument("--version", choices=["v1", "v2"], default="v1")
    ap.add_argument("--min-chars", type=int, default=80)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from pipeline.step1.extract_text import extract_text
    from pipeline.step1.extract_tables import extract_tables

    pdf = args.pdf.resolve()
    out = args.out.resolve()
    if not pdf.is_file():
        print(f"ERROR: PDF not found: {pdf}", file=sys.stderr)
        return 1

    extract_text(
        str(pdf),
        output_dir=str(out),
        min_chars=args.min_chars,
        version=args.version,
    )
    extract_tables(
        str(pdf),
        output_dir=str(out),
        version=args.version,
    )
    print(f"\nStep 1 done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
