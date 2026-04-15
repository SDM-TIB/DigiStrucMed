"""
Step 4 runner — Relation extraction (structural + LLM).

Produces 8 relation CSVs in ``<run>/step4/``:
  Structural: S_contains, S_disease_phenotype, S_disease_stage
  LLM-derived: S_treats, S_recommendation, S_drug_adverse_event, S_disease_assessment, S_disease_cause

Requires GPU + HF_TOKEN for the LLM-derived relations (Llama 3.1).

Usage (repo root):
  python -m pipeline.step4.run_step4
  python -m pipeline.step4.run_step4 --run outputs/pipeline-output18
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_RUN = Path("outputs/pipeline-output18")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run", type=Path, default=DEFAULT_RUN,
        help="Pipeline run folder (default: outputs/pipeline-output18)",
    )
    ap.add_argument(
        "--step2-dir", type=Path, default=None,
        help="Step 2 output directory (default: <run>/step2)",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Step 4 output directory (default: <run>/step4)",
    )
    ap.add_argument("--hf-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--hf-token", default=None, help="Defaults to HF_TOKEN env")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    run_dir = args.run.resolve()
    step2_dir = (args.step2_dir or (run_dir / "step2")).resolve()
    out_dir = (args.out or (run_dir / "step4")).resolve()

    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    print("Step 4 — Relation extraction (structural + LLM)")
    print(f"  run       : {run_dir}")
    print(f"  step2 in  : {step2_dir}")
    print(f"  output    : {out_dir}")

    from pipeline.step4.extract_relations import run_relation_extraction

    report = run_relation_extraction(
        run_dir,
        step2_dir=step2_dir,
        out_dir=out_dir,
        hf_model=args.hf_model,
        hf_token=args.hf_token,
        batch_size=args.batch_size,
    )

    if report.get("status") == "error":
        print(f"ERROR: {report.get('message')}", file=sys.stderr)
        return 1

    print(f"\nStep 4 done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
