import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.data import StatementsWithMedicalEntities, CandidateStatements
from pipeline.models import EntitiesLinker
from pipeline.inference import InferEntities
STAGE_D_CONFIG = {
    "umls_csv_path": None,
    "filter_unmatched": False,
    "use_partial_umls_match": False,
}
INPUT_FILE = Path(__file__).parent / "outputs" / "stage_c_statements_with_entities.json"
OUTPUT_FILE = Path(__file__).parent / "outputs" / "stage_d_candidate_statements.json"
def test_stage_d():
    print("=" * 70)
    print("STAGE d: statements_with_medical_entities -> infer:entities -> candidate_statements")
    print("=" * 70)
    cfg = STAGE_D_CONFIG
    umls_csv_path = cfg["umls_csv_path"]
    input_file = INPUT_FILE
    if not input_file.exists():
        root = Path(__file__).parent.parent
        input_file = root / "stage_c_statements_with_entities_NER.json"
    if not input_file.exists():
        print("    ERROR: Stage c output not found. Run stage_c_recognize_entities.py first.")
        return None
    output_file = OUTPUT_FILE
    output_file.parent.mkdir(exist_ok=True)
    print(f"\n[1] Loading Stage c output from {input_file}...")
    if not input_file.exists():
        print("    ERROR: Stage c output not found! Run stage_c_recognize_entities.py first.")
        return None
    with open(input_file, "r", encoding="utf-8") as f:
        stage_c_data = json.load(f)
    statements_with_entities = StatementsWithMedicalEntities()
    for stmt in stage_c_data["statements"]:
        statements_with_entities.add_statement(stmt)
    print(f"    Loaded: {statements_with_entities}")
    print(f"\n[2] Initializing EntitiesLinker...")
    if umls_csv_path and Path(umls_csv_path).exists():
        print(f"    Using UMLS: {umls_csv_path}")
        entities_linker = EntitiesLinker(
            knowledge_base="umls",
            umls_csv_path=umls_csv_path,
            filter_unmatched=cfg["filter_unmatched"],
            use_partial_umls_match=cfg.get("use_partial_umls_match", False),
        )
    else:
        print("    Using rule-based linking (no UMLS)")
        entities_linker = EntitiesLinker(knowledge_base="rule-based")
    print(f"\n[3] Initializing infer_entities device...")
    inferrer = InferEntities(entities_linker=entities_linker)
    print("\n[4] Inferring candidate statements...")
    candidate_statements = inferrer.infer(statements_with_entities)
    print(f"    Result: {candidate_statements}")
    print(f"\n[5] Saving output to {output_file}...")
    output_data = {
        "metadata": {
            "stage": "d",
            "description": "Candidate statements with UMLS-linked entities",
            "total_statements": candidate_statements.count(),
            "total_candidates": candidate_statements.count_candidates(),
            "umls_linking": umls_csv_path is not None and Path(umls_csv_path).exists()
        },
        "statements": candidate_statements.get_all()
    }
    output_file.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("\n" + "=" * 70)
    print("STAGE d COMPLETE")
    print("=" * 70)
    print(f"  Total statements: {candidate_statements.count()}")
    print(f"  Candidate statements: {candidate_statements.count_candidates()}")
    print(f"  Output file: {output_file}")
    print("=" * 70)
    return candidate_statements
if __name__ == "__main__":
    test_stage_d()
