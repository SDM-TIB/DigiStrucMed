import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from pipeline.data import CandidateStatements, ValidatedFactsAndQualifiers
from pipeline.models import ValidationModel
from pipeline.inference import Validate

PROJECT_ROOT = Path(__file__).parent.parent


def _resolve_stage_e_paths(
    stage_d_version: str | None = None,
    stage_e_version: str | None = None,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """
    Resolve input/output paths for Stage E.
    Input: Stage D output (stage_d_candidate_statements.json).
    Output: Stage E output directory.
    """
    if input_dir is not None:
        in_path = Path(input_dir) / "stage_d_candidate_statements.json"
    else:
        in_dir = config.stage_d_dir(stage_d_version)
        in_path = in_dir / "stage_d_candidate_statements.json"

    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        out_dir = config.stage_e_dir(stage_e_version)

    return in_path, out_dir


def test_stage_e(
    stage_d_version: str | None = None,
    stage_e_version: str | None = None,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
):
    effective_e_version = stage_e_version or config.DEFAULT_STAGE_E_VERSION
    input_file, output_root = _resolve_stage_e_paths(
        stage_d_version=stage_d_version,
        stage_e_version=stage_e_version,
        input_dir=input_dir,
        output_dir=output_dir,
    )
    output_file = output_root / "stage_e_validated_output.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STAGE E: candidate_statements -> validate -> validated_facts_AND_qualifiers")
    print("(extraction only; experts validate later)")
    print("=" * 70)
    validation_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    batch_size = 4
    max_extraction_tokens = 400
    print(f"\n[1] Loading Stage D output from {input_file}...")
    if not input_file.exists():
        print("    ERROR: Stage D output not found! Run stage_d_infer_entities.py first.")
        return None
    with open(input_file, "r", encoding="utf-8") as f:
        stage_d_data = json.load(f)
    candidate_statements = CandidateStatements()
    for stmt in stage_d_data["statements"]:
        candidate_statements.add_statement(stmt)
    table_triples = stage_d_data.get("table_triples", [])
    print(f"    Loaded: {candidate_statements}")
    print(f"    Table triples: {len(table_triples)}")
    print(f"\n[2] Initializing extraction model: {validation_model_name}...")
    validation_model = ValidationModel(model_name=validation_model_name)
    print(f"\n[3] Initializing validate device...")
    print(f"    Batch size: {batch_size}")
    print(f"    Max tokens: {max_extraction_tokens}")
    validator = Validate(
        validation_model=validation_model,
        batch_size=batch_size,
        max_new_tokens=max_extraction_tokens
    )
    print("\n[4] Extracting factual statements from text...")
    validated_facts = validator.validate(candidate_statements)
    print(f"    Result: {validated_facts}")
    if table_triples:
        print("\n[4b] Running table_triples through LLM (validate/split)...")
        validator.validate_table_triples(validated_facts, table_triples)
        print(f"    Result after table triples: {validated_facts}")
    print(f"\n[5] Saving output to {output_file}...")
    output_data = {
        "metadata": {
            "stage": "e",
            "description": "Extracted factual statements (subject, predicate, object, entities) from text and table triples for expert validation",
            "total_statements": validated_facts.count(),
            "extraction_model": validation_model_name,
            "table_triples_through_llm": len(table_triples),
        },
        "validated_statements": validated_facts.get_all()
    }
    output_file.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("\n" + "=" * 70)
    print("STAGE e COMPLETE")
    print("=" * 70)
    print(f"  Total extracted statements: {validated_facts.count()}")
    print(f"  Output file: {output_file}")
    print("=" * 70)
    return validated_facts
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Stage E (LLM validation).")
    parser.add_argument(
        "--stage-d-version",
        type=str,
        default=None,
        help="Stage D version to read from. Default: config.DEFAULT_STAGE_D_VERSION.",
    )
    parser.add_argument(
        "--stage-e-version",
        type=str,
        default=None,
        help="Stage E version to write to. Default: config.DEFAULT_STAGE_E_VERSION.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Explicit Stage D output directory. Overrides --stage-d-version.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Explicit Stage E output directory. Overrides --stage-e-version.",
    )
    args = parser.parse_args()

    test_stage_e(
        stage_d_version=args.stage_d_version,
        stage_e_version=args.stage_e_version,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
