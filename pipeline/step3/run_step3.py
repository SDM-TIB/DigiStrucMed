"""
Step 3 runner — UMLS entity linking + optional Llama 3.1 disambiguation.

Reads from Step 2 output (``--step2-dir``), writes Step 3 artifacts to a
dedicated ``--out`` directory (default ``<run>/step3``).

Usage (repo root):
  python -m pipeline.step3.run_step3
  python -m pipeline.step3.run_step3 --run outputs/pipeline-output18
  python -m pipeline.step3.run_step3 --llama
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_RUN = Path("outputs/pipeline-output18")
DEFAULT_UMLS = Path("input/UMLS.csv")


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
        help="Step 3 output directory (default: <run>/step3)",
    )
    ap.add_argument("--umls", type=Path, default=DEFAULT_UMLS)
    ap.add_argument("--sim-threshold", type=float, default=0.92)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--jsonl", action="store_true")
    ap.add_argument("--no-expand-acronyms", action="store_true")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument(
        "--llama", action="store_true",
        help="Run Llama 3.1 on needs_disambiguation (needs HF_TOKEN + GPU)",
    )
    ap.add_argument("--hf-model", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    run_dir = args.run.resolve()
    step2_dir = (args.step2_dir or (run_dir / "step2")).resolve()
    out_dir = (args.out or (run_dir / "step3")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from pipeline.step3.entity_link_sources import run_entity_linking
    from pipeline.step3.disambiguate_llama import disambiguate_llama

    print("Step 3 — entity linking + disambiguation")
    print(f"  step2 input : {step2_dir}")
    print(f"  output      : {out_dir}")

    run_entity_linking(
        step2_dir,
        args.umls,
        out_dir=out_dir,
        sim_threshold=args.sim_threshold,
        use_cache=not args.no_cache,
        write_jsonl=args.jsonl,
        expand_acronyms=not args.no_expand_acronyms,
        config_path=args.config,
    )

    if args.llama:
        grounded = out_dir / "grounded_entities.json"
        if not grounded.is_file():
            print(f"ERROR: missing {grounded}", file=sys.stderr)
            return 1
        disambiguate_llama(
            grounded,
            out_dir,
            hf_model=args.hf_model or "meta-llama/Llama-3.1-8B-Instruct",
            batch_size=args.batch_size,
        )

    print(f"\nStep 3 done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
