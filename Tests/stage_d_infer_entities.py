import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from pipeline.data import StatementsWithMedicalEntities, CandidateStatements
from pipeline.models import EntitiesLinker
from pipeline.inference.infer_entities import InferEntities

PROJECT_ROOT = Path(__file__).parent.parent
STAGE_D_CONFIG = {
    "umls_csv_path": str(PROJECT_ROOT / "data" / "UMLS.csv"),
    "filter_unmatched": False,
    "use_partial_umls_match": False,
}
INPUT_FILE_FALLBACK_TESTS = Path(__file__).parent / "outputs" / "stage_c_statements_with_entities.json"
INPUT_FILE_FALLBACK_NER = PROJECT_ROOT / "stage_c_statements_with_entities_NER.json"


def _resolve_stage_d_paths(
    stage_c_version: str | None = None,
    stage_d_version: str | None = None,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """
    Resolve input/output paths for Stage D.
    Input: Stage C output (stage_c_statements_with_entities.json).
    Output: Stage D output directory.
    """
    if input_dir is not None:
        in_path = Path(input_dir) / "stage_c_statements_with_entities.json"
    else:
        in_dir = config.stage_c_dir(stage_c_version)
        in_path = in_dir / "stage_c_statements_with_entities.json"

    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        out_dir = config.stage_d_dir(stage_d_version)

    return in_path, out_dir


def test_stage_d(
    stage_c_version: str | None = None,
    stage_d_version: str | None = None,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
):
    effective_d_version = stage_d_version or config.DEFAULT_STAGE_D_VERSION
    input_path, output_root = _resolve_stage_d_paths(
        stage_c_version=stage_c_version,
        stage_d_version=stage_d_version,
        input_dir=input_dir,
        output_dir=output_dir,
    )
    # Fallbacks for backward compatibility
    if not input_path.exists() and INPUT_FILE_FALLBACK_TESTS.exists():
        input_path = INPUT_FILE_FALLBACK_TESTS
    if not input_path.exists() and INPUT_FILE_FALLBACK_NER.exists():
        input_path = INPUT_FILE_FALLBACK_NER

    input_file = input_path
    output_file = output_root / "stage_d_candidate_statements.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"STAGE D {effective_d_version}: statements_with_medical_entities -> infer_entities -> candidate_statements")
    print("=" * 70)
    cfg = STAGE_D_CONFIG
    umls_csv_path = cfg["umls_csv_path"]
    if not input_file.exists():
        print("    ERROR: Stage C output not found. Run stage_c_recognize_entities.py first.")
        return None
    print(f"\n[1] Loading Stage C output from {input_file}...")
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
    print(f"STAGE D {effective_d_version} COMPLETE")
    print("=" * 70)
    print(f"  Total statements: {candidate_statements.count()}")
    print(f"  Candidate statements: {candidate_statements.count_candidates()}")
    print(f"  Table triples (with CUI): {len(table_triples_enriched)}")
    print(f"  Output: {output_file}")
    print("=" * 70)
    return candidate_statements
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Stage D (entity linking / UMLS).")
    parser.add_argument(
        "--stage-c-version",
        type=str,
        default=None,
        help="Stage C version to read from. Default: config.DEFAULT_STAGE_C_VERSION.",
    )
    parser.add_argument(
        "--stage-d-version",
        type=str,
        default=None,
        help="Stage D version to write to. Default: config.DEFAULT_STAGE_D_VERSION.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Explicit Stage C output directory. Overrides --stage-c-version.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Explicit Stage D output directory. Overrides --stage-d-version.",
    )
    args = parser.parse_args()

    test_stage_d(
        stage_c_version=args.stage_c_version,
        stage_d_version=args.stage_d_version,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )


