import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.data import CandidateStatements, ValidatedFactsAndQualifiers
from pipeline.models import ValidationModel
from pipeline.inference import Validate
def test_stage_e():
    print("=" * 70)
    print("STAGE e: candidate_statements -> validate -> validated_facts_AND_qualifiers")
    print("(extraction only; experts validate later)")
    print("=" * 70)
    validation_model_name = "meta-llama/Llama-3.2-3B-Instruct"
    batch_size = 4
    max_extraction_tokens = 400
    input_file = Path(__file__).parent / "outputs" / "stage_d_candidate_statements.json"
    output_file = Path(__file__).parent / "outputs" / "stage_e_validated_output.json"
    output_file.parent.mkdir(exist_ok=True)
    print(f"\n[1] Loading Stage d output from {input_file}...")
    if not input_file.exists():
        print("    ERROR: Stage d output not found! Run stage_d_infer_entities.py first.")
        return None
    with open(input_file, "r", encoding="utf-8") as f:
        stage_d_data = json.load(f)
    candidate_statements = CandidateStatements()
    for stmt in stage_d_data["statements"]:
        candidate_statements.add_statement(stmt)
    print(f"    Loaded: {candidate_statements}")
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
    print("\n[4] Extracting factual statements...")
    validated_facts = validator.validate(candidate_statements)
    print(f"    Result: {validated_facts}")
    print(f"\n[5] Saving output to {output_file}...")
    output_data = {
        "metadata": {
            "stage": "e",
            "description": "Extracted factual statements (subject, predicate, object, exception, duration) for expert validation",
            "total_statements": validated_facts.count(),
            "extraction_model": validation_model_name
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
    test_stage_e()
