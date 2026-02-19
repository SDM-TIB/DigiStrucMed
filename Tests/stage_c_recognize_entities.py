import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.data import TextChunks, StatementsWithMedicalEntities
from pipeline.models import NeuralModel
from pipeline.inference import RecognizeEntities
STAGE_C_CONFIG = {
    "neural_model_name": "d4data/biomedical-ner-all",
    "min_ner_score": 0.55,
    "acronym_file": None,
}
INPUT_FILE = Path(__file__).parent / "outputs" / "stage_b_text_chunks.json"
OUTPUT_FILE = Path(__file__).parent / "outputs" / "stage_c_statements_with_entities.json"
def test_stage_c():
    print("=" * 70)
    print("STAGE c: text_chunks -> recognize_entities -> statements_with_medical_entities")
    print("(NER_model_with_acronym_expander)")
    print("=" * 70)
    cfg = STAGE_C_CONFIG
    neural_model_name = cfg["neural_model_name"]
    min_ner_score = cfg["min_ner_score"]
    acronym_file = cfg["acronym_file"]
    input_file = INPUT_FILE
    output_file = OUTPUT_FILE
    output_file.parent.mkdir(exist_ok=True)
    print(f"\n[1] Loading Stage b output from {input_file}...")
    if not input_file.exists():
        print("    ERROR: Stage b output not found! Run stage_b_test.py first.")
        return None
    with open(input_file, "r", encoding="utf-8") as f:
        stage_b_data = json.load(f)
    text_chunks = TextChunks()
    for chunk in stage_b_data["chunks"]:
        text_chunks.add_chunk(
            page=chunk["page"],
            text=chunk["text"],
            source=chunk.get("source", ""),
            chunk_id=chunk.get("chunk_id")
        )
    print(f"    Loaded: {text_chunks}")
    print(f"\n[2] Initializing neural model: {neural_model_name}...")
    neural_model = NeuralModel(model_name=neural_model_name)
    print(f"\n[3] Initializing recognize_entities device...")
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
    print(f"\n[5] Saving output to {output_file}...")
    output_data = {
        "metadata": {
            "stage": "c",
            "description": "Statements with medical entities (NER + acronym expansion)",
            "total_statements": statements_with_entities.count(),
            "total_entities": statements_with_entities.get_entity_count(),
            "neural_model": neural_model_name,
            "min_ner_score": min_ner_score
        },
        "statements": statements_with_entities.get_all()
    }
    output_file.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("\n" + "=" * 70)
    print("STAGE c COMPLETE")
    print("=" * 70)
    print(f"  Total statements: {statements_with_entities.count()}")
    print(f"  Total entities: {statements_with_entities.get_entity_count()}")
    print(f"  Output file: {output_file}")
    print("=" * 70)
    return statements_with_entities
if __name__ == "__main__":
    test_stage_c()
