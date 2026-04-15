"""
Run Step 2 normalization (v1): tables + text into ``<run>/step2/``.

Requires:
  <run>/step1/table_index.json
  <run>/step1/text_blocks.json (optional)

Usage (repo root):
  python -m pipeline.step2.run_normalize
  python -m pipeline.step2.run_normalize --run outputs/other_run
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

DEFAULT_RUN = Path("outputs/pipeline-output18")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run",
        type=Path,
        default=DEFAULT_RUN,
        help="Run folder containing step1/ (default: outputs/pipeline-output18)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <run>/step2)",
    )
    ap.add_argument("--config", type=Path, default=None, help="Optional guideline_config.json path")
    args = ap.parse_args()

    run_dir = args.run.resolve()
    out_dir = (args.out or (run_dir / "step2")).resolve()
    step1 = run_dir / "step1"
    if not (step1 / "table_index.json").exists():
        print(f"ERROR: missing {step1 / 'table_index.json'}", file=sys.stderr)
        return 1

    from pipeline.step2.normalize_tables import run_table_normalization
    from pipeline.step2.normalize_text import run_text_normalization

    cfg = str(args.config) if args.config else None
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Step 2 — normalize (tables + text)")
    print(f"  run: {run_dir}")
    print(f"  in : {step1}")
    print(f"  out: {out_dir}")
    print("Running normalize_tables (v1)...")
    r = run_table_normalization(str(run_dir), str(out_dir), config_path=cfg)
    if r.get("status") == "error":
        print(r.get("message", r), file=sys.stderr)
        return 1
    print("  summary:", r.get("summary", r))

    disease_id = "Unknown"
    p = out_dir / "S_disease.csv"
    if p.exists():
        with p.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            disease_id = (rows[0].get("disease_id") or "Unknown").strip() or "Unknown"

    print(f"Running normalize_text (v1), disease_id={disease_id!r}...")
    tr = run_text_normalization(run_dir, out_dir, disease_id=disease_id, config_path=cfg)
    print("  text:", tr)

    print(f"\nStep 2 done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
