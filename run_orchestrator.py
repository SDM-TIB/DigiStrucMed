"""
Orchestrator: Run the pipeline with per-stage version control.

Each stage reads its input from the OUTPUT directory of the previous stage
(using that stage's version). Example: A:v1 -> B:v2 -> C:v3 means:
  - Stage A v1 writes to outputs/STAGE_A_v1/
  - Stage B v2 reads from outputs/STAGE_A_v1/, writes to outputs/STAGE_B_v2/
  - Stage C v3 reads from outputs/STAGE_B_v2/, writes to outputs/STAGE_C_v3/

Usage:
  python run_orchestrator.py --A v1 --B v2 --C v3 --D v1 --E v1
  python run_orchestrator.py  # Uses config.json for versions
  python run_orchestrator.py --config my_config.json  # Override config file
  python run_orchestrator.py --from B --to D  # Run only stages B, C, D
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

TESTS_DIR = PROJECT_ROOT / "Tests"


def _run_stage_script(script_name: str, *args: str) -> bool:
    """Run a stage test script. Returns True if exit code is 0."""
    cmd = [sys.executable, str(TESTS_DIR / script_name)] + list(args)
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode == 0


def run_stage_a(version: str, output_dir: Path) -> bool:
    """Run Stage A with the given version. Returns True on success."""
    return _run_stage_script(
        "stage_a_test.py",
        "--version", version,
        "--output-dir", str(output_dir),
    )


def run_stage_b(input_dir: Path, output_dir: Path) -> bool:
    """Run Stage B. Reads from input_dir, writes to output_dir."""
    return _run_stage_script(
        "stage_b_test.py",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
    )


def run_stage_c(input_dir: Path, output_dir: Path, version: str) -> bool:
    """Run Stage C with the given version. Reads from input_dir, writes to output_dir."""
    return _run_stage_script(
        "stage_c_recognize_entities.py",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--stage-c-version", version,
    )


def run_stage_d(input_dir: Path, output_dir: Path) -> bool:
    """Run Stage D. Reads from input_dir, writes to output_dir."""
    return _run_stage_script(
        "stage_d_infer_entities.py",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
    )


def run_stage_e(input_dir: Path, output_dir: Path) -> bool:
    """Run Stage E. Reads from input_dir, writes to output_dir."""
    return _run_stage_script(
        "stage_e_validate.py",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
    )


def resolve_versions(
    args: argparse.Namespace,
    config_file: Path | None = None,
) -> dict[str, str]:
    """
    Build version map. Single config: config.json (or --config file).
    CLI flags override config.
    """
    if config_file and config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        versions = dict(data.get("versions", config.get_config()["versions"]))
    else:
        versions = dict(config.get_config()["versions"])

    # CLI overrides
    if args.A is not None:
        versions["A"] = args.A
    if args.B is not None:
        versions["B"] = args.B
    if args.C is not None:
        versions["C"] = args.C
    if args.D is not None:
        versions["D"] = args.D
    if args.E is not None:
        versions["E"] = args.E

    return versions


def run_orchestrator(
    versions: dict[str, str],
    stages_from: str = "A",
    stages_to: str = "E",
) -> bool:
    """
    Run the pipeline with the given version map.
    stages_from / stages_to: inclusive range (A, B, C, D, E).
    Returns True if all stages succeeded.
    """
    order = ["A", "B", "C", "D", "E"]
    from_idx = order.index(stages_from) if stages_from in order else 0
    to_idx = order.index(stages_to) if stages_to in order else len(order) - 1
    stages_to_run = order[from_idx : to_idx + 1]

    print("\n" + "=" * 70)
    print("ORCHESTRATOR: Version map")
    print("=" * 70)
    for stage in order:
        v = versions.get(stage, "v1")
        run = "yes" if stage in stages_to_run else "skip"
        print(f"  Stage {stage}: version {v}  [{run}]")
    print("=" * 70)

    # Stage A reads from input/ (PDFs); each subsequent stage reads from previous output
    for stage in stages_to_run:
        ver = versions.get(stage, "v1")
        if stage == "A":
            output_dir = config.stage_a_dir(ver)
            print(f"\n>>> Running Stage A {ver} -> {output_dir}")
            ok = run_stage_a(version=ver, output_dir=output_dir)
        elif stage == "B":
            input_dir = config.stage_a_dir(versions["A"])
            output_dir = config.stage_b_dir(ver)
            print(f"\n>>> Running Stage B {ver} <- {input_dir} -> {output_dir}")
            ok = run_stage_b(input_dir=input_dir, output_dir=output_dir)
        elif stage == "C":
            input_dir = config.stage_b_dir(versions["B"])
            output_dir = config.stage_c_dir(ver)
            print(f"\n>>> Running Stage C {ver} <- {input_dir} -> {output_dir}")
            ok = run_stage_c(input_dir=input_dir, output_dir=output_dir, version=ver)
        elif stage == "D":
            input_dir = config.stage_c_dir(versions["C"])
            output_dir = config.stage_d_dir(ver)
            print(f"\n>>> Running Stage D {ver} <- {input_dir} -> {output_dir}")
            ok = run_stage_d(input_dir=input_dir, output_dir=output_dir)
        else:  # E
            input_dir = config.stage_d_dir(versions["D"])
            output_dir = config.stage_e_dir(ver)
            print(f"\n>>> Running Stage E {ver} <- {input_dir} -> {output_dir}")
            ok = run_stage_e(input_dir=input_dir, output_dir=output_dir)

        if not ok:
            print(f"\n[ORCHESTRATOR] Stage {stage} failed. Stopping.")
            return False

    print("\n" + "=" * 70)
    print("ORCHESTRATOR COMPLETE")
    print("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run the pipeline with per-stage version control. "
        "Each stage reads from the previous stage's output directory."
    )
    parser.add_argument("--A", type=str, default=None, help="Stage A version (e.g. v1, v2)")
    parser.add_argument("--B", type=str, default=None, help="Stage B version")
    parser.add_argument("--C", type=str, default=None, help="Stage C version")
    parser.add_argument("--D", type=str, default=None, help="Stage D version")
    parser.add_argument("--E", type=str, default=None, help="Stage E version")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config file (JSON). Default: config.json. Same format: paths, versions.",
    )
    parser.add_argument(
        "--from",
        dest="stages_from",
        type=str,
        default="A",
        choices=["A", "B", "C", "D", "E"],
        help="First stage to run (default: A)",
    )
    parser.add_argument(
        "--to",
        dest="stages_to",
        type=str,
        default="E",
        choices=["A", "B", "C", "D", "E"],
        help="Last stage to run (default: E)",
    )
    args = parser.parse_args()

    config_file = Path(args.config) if args.config else config.CONFIG_FILE
    versions = resolve_versions(args, config_file)
    ok = run_orchestrator(
        versions=versions,
        stages_from=args.stages_from,
        stages_to=args.stages_to,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
