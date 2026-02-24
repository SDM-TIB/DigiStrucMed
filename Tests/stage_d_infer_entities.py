import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.data import StatementsWithMedicalEntities, CandidateStatements
from pipeline.models import EntitiesLinker
from pipeline.inference.infer_entities import InferEntities

PROJECT_ROOT = Path(__file__).parent.parent
STAGE_C_V1_DIR = PROJECT_ROOT / "outputs" / "STAGE_C_v1"
STAGE_D_V1_DIR = PROJECT_ROOT / "outputs" / "STAGE_D_v1"

STAGE_D_CONFIG = {
    "umls_csv_path": str(PROJECT_ROOT / "data" / "UMLS.csv"),
    "filter_unmatched": False,
    "use_partial_umls_match": False,
}
INPUT_FILE = STAGE_C_V1_DIR / "stage_c_statements_with_entities.json"
INPUT_FILE_FALLBACK_TESTS = Path(__file__).parent / "outputs" / "stage_c_statements_with_entities.json"
INPUT_FILE_FALLBACK_NER = PROJECT_ROOT / "stage_c_statements_with_entities_NER.json"
OUTPUT_FILE = STAGE_D_V1_DIR / "stage_d_candidate_statements.json"


def _resolve_stage_c_input() -> Path:
    if INPUT_FILE.exists():
        return INPUT_FILE
    if INPUT_FILE_FALLBACK_TESTS.exists():
        return INPUT_FILE_FALLBACK_TESTS
    if INPUT_FILE_FALLBACK_NER.exists():
        return INPUT_FILE_FALLBACK_NER
    return INPUT_FILE


def test_stage_d():
    print("=" * 70)
    print("STAGE D v1: statements_with_medical_entities -> infer_entities -> candidate_statements")
    print("=" * 70)
    cfg = STAGE_D_CONFIG
    umls_csv_path = cfg["umls_csv_path"]
    input_file = _resolve_stage_c_input()
    if not input_file.exists():
        print("    ERROR: Stage C v1 output not found. Run stage_c_recognize_entities.py first.")
        return None
    output_file = OUTPUT_FILE
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[1] Loading Stage C v1 output from {input_file}...")
    if not input_file.exists():
        print("    ERROR: Stage C v1 output not found! Run stage_c_recognize_entities.py first.")
        return None
    with open(input_file, "r", encoding="utf-8") as f:
        stage_c_data = json.load(f)
    statements_with_entities = StatementsWithMedicalEntities()
    for stmt in stage_c_data["statements"]:
        statements_with_entities.add_statement(stmt)
    table_triples_raw = stage_c_data.get("table_triples", [])
    print(f"    Loaded: {statements_with_entities}")
    print(f"    Table triples: {len(table_triples_raw)}")
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
    print("\n[4b] Linking CUI for table_triples entities...")
    table_triples_enriched = []
    for triple in table_triples_raw:
        out = dict(triple)
        entities = triple.get("entities", [])
        if entities:
            linker_input = [
                {"text": e.get("text", ""), "label": e.get("label", "")}
                for e in entities
            ]
            for i, e in enumerate(entities):
                if i < len(linker_input):
                    if "score" in e:
                        linker_input[i]["score"] = e["score"]
                    if "start" in e:
                        linker_input[i]["start"] = e["start"]
                    if "end" in e:
                        linker_input[i]["end"] = e["end"]
            linked = entities_linker.link_entities(linker_input)
            out["entities"] = linked
        table_triples_enriched.append(out)
    print(f"    Enriched {len(table_triples_enriched)} table triples with CUI")
    print(f"\n[5] Saving output to {output_file}...")
    output_data = {
        "metadata": {
            "stage": "d",
            "description": "Candidate statements and table triples with UMLS-linked entities",
            "total_statements": candidate_statements.count(),
            "total_candidates": candidate_statements.count_candidates(),
            "total_table_triples": len(table_triples_enriched),
            "umls_linking": umls_csv_path is not None and Path(umls_csv_path).exists()
        },
        "statements": candidate_statements.get_all(),
        "table_triples": table_triples_enriched
    }
    output_file.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("\n" + "=" * 70)
    print("STAGE D v1 COMPLETE")
    print("=" * 70)
    print(f"  Total statements: {candidate_statements.count()}")
    print(f"  Candidate statements: {candidate_statements.count_candidates()}")
    print(f"  Table triples (with CUI): {len(table_triples_enriched)}")
    print(f"  Output: {output_file}")
    print("=" * 70)
    return candidate_statements
if __name__ == "__main__":
    test_stage_d()


