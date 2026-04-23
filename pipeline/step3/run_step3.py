"""Step 3: UMLS linking (+ optional ``--llama``). Single ``--run`` or ``--batch-dir``."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

DEFAULT_RUN = Path("outputs/pipeline-output2_new")
DEFAULT_UMLS = Path("input/UMLS.csv")


def _link_one(
    run_dir: Path,
    umls: Path,
    *,
    sim_threshold: float,
    use_cache: bool,
    write_jsonl: bool,
    expand_acronyms: bool,
    config_path: Path | None,
) -> dict:
    from pipeline.step3.entity_link_sources import run_entity_linking

    step2_dir = (run_dir / "step2").resolve()
    if not step2_dir.is_dir():
        alt = run_dir / "step2_v2"
        if alt.is_dir():
            step2_dir = alt.resolve()
    out_dir = (run_dir / "step3").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = run_entity_linking(
        step2_dir,
        umls,
        out_dir=out_dir,
        sim_threshold=sim_threshold,
        use_cache=use_cache,
        write_jsonl=write_jsonl,
        expand_acronyms=expand_acronyms,
        config_path=config_path,
    )
    return report


def _make_linking_zip(batch_dir: Path, runs: list[Path]) -> Path:
    out_zip = batch_dir.parent / "step3_linking_outputs.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in runs:
            s3 = r / "step3"
            if not s3.is_dir():
                continue
            for f in sorted(s3.rglob("*")):
                if f.is_file():
                    arcname = f"{r.name}/step3/{f.relative_to(s3)}"
                    zf.write(f, arcname)
    mb = out_zip.stat().st_size / 1024 / 1024
    print(f"\nCreated {out_zip} ({mb:.1f} MB)")
    return out_zip


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 3: UMLS entity linking (+ optional Llama).")
    group = ap.add_mutually_exclusive_group()
    group.add_argument(
        "--run", type=Path, default=None,
        help="Single guideline run folder (e.g. outputs/pipeline-output2_new/<slug>)",
    )
    group.add_argument(
        "--batch-dir", type=Path, default=None,
        help="Root with multiple guideline slug folders (e.g. outputs/pipeline-output2_new)",
    )
    ap.add_argument(
        "--step2-dir", type=Path, default=None,
        help="Step 2 output directory (single-run only; default: <run>/step2)",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Step 3 output directory (single-run only; default: <run>/step3)",
    )
    ap.add_argument("--umls", type=Path, default=DEFAULT_UMLS)
    ap.add_argument("--sim-threshold", type=float, default=0.92)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--jsonl", action="store_true")
    ap.add_argument("--no-expand-acronyms", action="store_true")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument(
        "--llama", action="store_true",
        help="Also run Llama 3.1 disambiguation (needs HF_TOKEN + GPU)",
    )
    ap.add_argument("--hf-model", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    if args.batch_dir is not None:
        batch = args.batch_dir.resolve()
        if not batch.is_dir():
            print(f"ERROR: {batch} is not a directory", file=sys.stderr)
            return 1
        runs = sorted(
            p for p in batch.iterdir()
            if p.is_dir() and ((p / "step2").is_dir() or (p / "step2_v2").is_dir())
        )
        if not runs:
            print(f"ERROR: no slug folders with step2/ found in {batch}", file=sys.stderr)
            return 1

        print(f"Step 3 batch — UMLS entity linking for {len(runs)} guidelines")
        print(f"  source : {batch}")
        print(f"  UMLS   : {args.umls}\n")

        t0 = time.time()
        for i, r in enumerate(runs, 1):
            print(f"[{i}/{len(runs)}] {r.name}")
            try:
                report = _link_one(
                    r, args.umls,
                    sim_threshold=args.sim_threshold,
                    use_cache=not args.no_cache,
                    write_jsonl=args.jsonl,
                    expand_acronyms=not args.no_expand_acronyms,
                    config_path=args.config if args.config else None,
                )
                sc = report.get("status_counts", {})
                print(f"  mentions={report.get('unique_mentions', 0)}  "
                      f"direct={sc.get('direct', 0)}  "
                      f"disambig={sc.get('needs_disambiguation', 0)}  "
                      f"no_match={sc.get('no_match', 0)}")
            except Exception as exc:
                print(f"  ERROR: {exc}")

        elapsed = time.time() - t0
        print(f"\nEntity linking done: {len(runs)} guidelines in {elapsed:.0f}s")

        _make_linking_zip(batch, runs)

        if args.llama:
            from pipeline.step3.disambiguate_llama import disambiguate_llama
            for i, r in enumerate(runs, 1):
                grounded = r / "step3" / "grounded_entities.json"
                if not grounded.is_file():
                    continue
                print(f"\n[{i}/{len(runs)}] Llama disambig: {r.name}")
                disambiguate_llama(
                    grounded, r / "step3",
                    hf_model=args.hf_model or "meta-llama/Llama-3.1-8B-Instruct",
                    batch_size=args.batch_size,
                )
        return 0

    run_dir = (args.run or DEFAULT_RUN).resolve()
    step2_dir = (args.step2_dir or (run_dir / "step2")).resolve()
    if not step2_dir.is_dir():
        alt = run_dir / "step2_v2"
        if alt.is_dir():
            step2_dir = alt.resolve()
    out_dir = (args.out or (run_dir / "step3")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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
