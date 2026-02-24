import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.data import TextChunks, StatementsWithMedicalEntities
from pipeline.models import NeuralModel
from pipeline.inference import RecognizeEntities

# Project root and versioned output dirs (align with outputs/STAGE_A_v1, etc.)
PROJECT_ROOT = Path(__file__).parent.parent
STAGE_B_V1_DIR = PROJECT_ROOT / "outputs" / "STAGE_B_v1"
STAGE_C_V1_DIR = PROJECT_ROOT / "outputs" / "STAGE_C_v1"

STAGE_C_CONFIG = {
    "neural_model_name": "d4data/biomedical-ner-all",
    "min_ner_score": 0.55,
    "acronym_file": None,
}
# Prefer outputs/STAGE_B_v1, fallback to Tests/outputs for backward compat
INPUT_CHUNKS = STAGE_B_V1_DIR / "stage_b_text_chunks.json"
INPUT_CHUNKS_FALLBACK = Path(__file__).parent / "outputs" / "stage_b_text_chunks.json"
INPUT_TABLE_TRIPLES = STAGE_B_V1_DIR / "stage_b_table_triples.json"
OUTPUT_FILE = STAGE_C_V1_DIR / "stage_c_statements_with_entities.json"


def test_stage_c():
    print("=" * 70)
    print("STAGE C v1: text_chunks + table_triples -> NER -> statements_with_entities + table_triples")
    print("(NER + acronym expansion; both Stage B outputs loaded and written)")
    print("=" * 70)
    cfg = STAGE_C_CONFIG
    neural_model_name = cfg["neural_model_name"]
    min_ner_score = cfg["min_ner_score"]
    acronym_file = cfg["acronym_file"]
    input_chunks = INPUT_CHUNKS if INPUT_CHUNKS.exists() else INPUT_CHUNKS_FALLBACK
    output_file = OUTPUT_FILE
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[1] Loading Stage B v1 outputs: text chunks + table triples...")
    if not input_chunks.exists():
        print("    ERROR: Stage B v1 text chunks not found! Run stage_b_test.py first (or put stage_b_text_chunks.json in outputs/STAGE_B_v1/).")
        return None
    with open(input_chunks, "r", encoding="utf-8") as f:
        stage_b_chunks_data = json.load(f)
    text_chunks = TextChunks()
    for chunk in stage_b_chunks_data["chunks"]:
        text_chunks.add_chunk(
            page=chunk["page"],
            text=chunk["text"],
            source=chunk.get("source", ""),
            chunk_id=chunk.get("chunk_id")
        )
    print(f"    Loaded text chunks: {text_chunks}")
    table_triples = []
    triples_file = INPUT_TABLE_TRIPLES
    if triples_file.exists():
        with open(triples_file, "r", encoding="utf-8") as f:
            triples_data = json.load(f)
        table_triples = triples_data.get("triples", []) if isinstance(triples_data, dict) else triples_data
        print(f"    Loaded table triples: {len(table_triples)}")
    else:
        print("    Table triples file not found; using empty list.")
    print(f"\n[2] Initializing neural model: {neural_model_name}...")
    neural_model = NeuralModel(model_name=neural_model_name)
    print(f"\n[3] Initializing recognize_entities...")
    print(f"    Min NER score: {min_ner_score}")
    print(f"    Acronym expansion: enabled (integrated into NER)")
    recognizer = RecognizeEntities(
        neural_model=neural_model,
        min_score=min_ner_score,
        acronym_file=acronym_file
    )
    print("\n[4] Running NER with acronym expansion...")
    statements_with_entities = recognizer.infer(text_chunks)
    print(f"    Result: {statements_with_entities}")
    print("    Enriching table triples with NER on subject/object...")
    table_triples = recognizer.enrich_triples_with_entities(table_triples)
    print(f"\n[5] Saving output (statements + table_triples) to {output_file}...")
    output_data = {
        "metadata": {
            "stage": "c",
            "description": "Statements with medical entities (NER + acronym expansion) and table triples from Stage B",
            "total_statements": statements_with_entities.count(),
            "total_entities": statements_with_entities.get_entity_count(),
            "total_table_triples": len(table_triples),
            "neural_model": neural_model_name,
            "min_ner_score": min_ner_score
        },
        "statements": statements_with_entities.get_all(),
        "table_triples": table_triples
    }
    output_file.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("\n" + "=" * 70)
    print("STAGE C v1 COMPLETE")
    print("=" * 70)
    print(f"  Total statements: {statements_with_entities.count()}")
    print(f"  Total entities: {statements_with_entities.get_entity_count()}")
    print(f"  Table triples: {len(table_triples)}")
    print(f"  Output: {output_file}")
    print("=" * 70)
    return statements_with_entities


if __name__ == "__main__":
    test_stage_c()
