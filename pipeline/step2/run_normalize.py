from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

DEFAULT_RUN = Path("outputs/pipeline-output2")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Step 2 normalization: tables + text")
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN, help="Run folder containing step1/")
    ap.add_argument("--out", type=Path, default=None, help="Output directory (default: <run>/step2)")
    ap.add_argument("--therapy-min-freq", type=int, default=2)
    ap.add_argument("--text-min-freq", type=int, default=1)
    ap.add_argument("--hf-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--hf-token", default=None, help="HF token (or set HF_TOKEN env)")
    ap.add_argument("--llm-batch-size", type=int, default=15)
    args = ap.parse_args()

    run_dir = args.run.resolve()
    out_dir = (args.out or (run_dir / "step2")).resolve()
    step1 = run_dir / "step1"

    if not (step1 / "table_index.json").exists():
        print(f"ERROR: missing {step1 / 'table_index.json'}", file=sys.stderr)
        return 1

    from pipeline.step2.normalize_tables import run_table_normalization
    from pipeline.step2.normalize_text import run_text_normalization

    out_dir.mkdir(parents=True, exist_ok=True)

    print("Step 2 — normalize (tables + text)")
    print(f"  run: {run_dir}")
    print(f"  in : {step1}")
    print(f"  out: {out_dir}")

    print("Running normalize_tables...")
    r = run_table_normalization(run_dir, out_dir, therapy_min_freq=args.therapy_min_freq)
    if r.get("status") == "error":
        print(r.get("message", str(r)), file=sys.stderr)
        return 1
    print("  summary:", r.get("summary", r))

    condition_id = "Unknown"
    p = out_dir / "S_condition.csv"
    if p.exists():
        with p.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            condition_id = (rows[0].get("condition_id") or "Unknown").strip() or "Unknown"

    print(f"Running normalize_text, condition_id={condition_id!r}...")
    tr = run_text_normalization(run_dir, out_dir, condition_id=condition_id, min_freq=args.text_min_freq)
    print("  text:", tr)

    from pipeline.step2.dedup_entities import dedup_step2_dir
    print("Running entity dedup (drug + therapy)...")
    dedup_step2_dir(out_dir)

    from pipeline.step2.validate_entities_llm import validate_run
    hf_token = (args.hf_token or os.environ.get("HF_TOKEN") or "").strip()
    if not hf_token:
        print("ERROR: Step 2 now requires LLM validation. Set HF_TOKEN or pass --hf-token.", file=sys.stderr)
        return 1
    print("Running mandatory LLM entity validation...")
    validation_run_dir = out_dir.parent
    validation_step2_subdir = out_dir.name
    vr = validate_run(
        validation_run_dir,
        hf_model=args.hf_model,
        hf_token=hf_token,
        step2_subdir=validation_step2_subdir,
        batch_size=args.llm_batch_size,
    )
    if isinstance(vr, dict) and vr.get("error"):
        print(f"ERROR: LLM validation failed: {vr['error']}", file=sys.stderr)
        return 1

    print(f"\nStep 2 done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
