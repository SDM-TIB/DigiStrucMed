"""
Stage C v4: NER only – no label filtering, RoBERTa model (v2 engine).

Same as v3 (no label filter, all entity types pass through) but uses the v2 model:
Jean-Baptiste/roberta-large-ner-english. Outputs PER, ORG, LOC, MISC instead of
biomedical labels. Filter by UMLS in Stage D.

Usage:
  python Tests/stage_c_v4_ner_only.py
  python Tests/stage_c_v4_ner_only.py --input-dir outputs/STAGE_B_v1
  python Tests/stage_c_v4_ner_only.py --output-dir outputs/STAGE_C_v4
"""
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from pipeline.data import TextChunks, StatementsWithMedicalEntities
from pipeline.models import NeuralModel
from pipeline.inference import RecognizeEntities

V4_OUTPUT_DIR = config.OUTPUTS_ROOT / "STAGE_C_v4"
NEURAL_MODEL = "Jean-Baptiste/roberta-large-ner-english"
MIN_NER_SCORE = 0.55


def _print_ner_cache_info(neural_model_name: str) -> None:
    """Print where the NER model is loaded from."""
    local_path = config.ner_model_local_path(neural_model_name)
    if local_path is not None:
        print(f"    NER model: loaded from local cache")
        print(f"    Location: {local_path}")
    else:
        if os.environ.get("HUGGINGFACE_HUB_CACHE"):
            cache_path = Path(os.environ["HUGGINGFACE_HUB_CACHE"])
        elif os.environ.get("HF_HOME"):
            cache_path = Path(os.environ["HF_HOME"]) / "hub"
        else:
            cache_path = Path.home() / ".cache" / "huggingface" / "hub"
        print(f"    NER model: Hugging Face cache (run download_ner_model.py for local)")
        print(f"    Cache: {cache_path}")


def _resolve_input_paths(input_dir: Path | None) -> tuple[Path, Path]:
    """Return (chunks_path, triples_path)."""
    if input_dir is not None:
        root = Path(input_dir)
    else:
        root = config.stage_b_dir()
    chunks = root / "stage_b_text_chunks.json"
    triples = root / "stage_b_table_triples.json"
    if not chunks.exists():
        fallback = Path(__file__).parent / "outputs"
        if (fallback / "stage_b_text_chunks.json").exists():
            return fallback / "stage_b_text_chunks.json", fallback / "stage_b_table_triples.json"
    return chunks, triples


def run_stage_c_v4(
    input_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
):
    in_dir = Path(input_dir) if input_dir else None
    out_root = Path(output_dir) if output_dir else V4_OUTPUT_DIR
    out_root.mkdir(parents=True, exist_ok=True)
    output_file = out_root / "stage_c_statements_with_entities.json"

    print("=" * 70)
    print("STAGE C v4: NER only (no label filter) – RoBERTa model (v2 engine)")
    print("All entity types pass through (PER, ORG, LOC, MISC); filter by UMLS in Stage D.")
    print("=" * 70)

    chunks_path, triples_path = _resolve_input_paths(in_dir)
    if not chunks_path.exists():
        print(f"\n    ERROR: Stage B output not found: {chunks_path}")
        print("    Run stage_b_test.py first (or specify --input-dir).")
        return None

    print(f"\n[1] Loading Stage B outputs...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        stage_b_data = json.load(f)
    text_chunks = TextChunks()
    for chunk in stage_b_data["chunks"]:
        text_chunks.add_chunk(
            page=chunk["page"],
            text=chunk["text"],
            source=chunk.get("source", ""),
            chunk_id=chunk.get("chunk_id"),
        )
    print(f"    Loaded: {text_chunks}")

    table_triples = []
    if triples_path.exists():
        with open(triples_path, "r", encoding="utf-8") as f:
            triples_data = json.load(f)
        table_triples = triples_data.get("triples", []) if isinstance(triples_data, dict) else triples_data
        print(f"    Table triples: {len(table_triples)}")
    else:
        print("    Table triples: (none)")

    print(f"\n[2] Initializing neural model: {NEURAL_MODEL}...")
    neural_model = NeuralModel(model_name=NEURAL_MODEL)
    _print_ner_cache_info(NEURAL_MODEL)

    print(f"\n[3] Initializing recognize_entities (v4: no label filter, RoBERTa)...")
    print(f"    Min NER score: {MIN_NER_SCORE}")
    print(f"    Label filter: disabled (all entity types kept)")
    recognizer = RecognizeEntities(
        neural_model=neural_model,
        min_score=MIN_NER_SCORE,
        acronym_file=None,
        filter_labels=False,
        verbose=True,
    )

    print("\n[4] Running NER with acronym expansion...")
    statements_with_entities = recognizer.infer(text_chunks)
    print(f"    Result: {statements_with_entities}")
    print("    Enriching table triples with NER on subject/object...")
    table_triples = recognizer.enrich_triples_with_entities(table_triples)

    print(f"\n[5] Saving output to {output_file}...")
    output_data = {
        "metadata": {
            "stage": "c",
            "version": "v4",
            "description": "NER only, no label filter. RoBERTa model (v2 engine). PER/ORG/LOC/MISC. Filter by UMLS in Stage D.",
            "label_filter": False,
            "total_statements": statements_with_entities.count(),
            "total_entities": statements_with_entities.get_entity_count(),
            "total_table_triples": len(table_triples),
            "neural_model": NEURAL_MODEL,
            "min_ner_score": MIN_NER_SCORE,
        },
        "statements": statements_with_entities.get_all(),
        "table_triples": table_triples,
    }
    output_file.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("STAGE C v4 COMPLETE (NER only, no label filter, RoBERTa)")
    print("=" * 70)
    print(f"  Total statements: {statements_with_entities.count()}")
    print(f"  Total entities: {statements_with_entities.get_entity_count()}")
    print(f"  Table triples: {len(table_triples)}")
    print(f"  Output: {output_file}")
    print("=" * 70)
    return statements_with_entities


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Stage C v4: NER only, no label filter. RoBERTa model (v2 engine).",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Input directory with stage_b_text_chunks.json and stage_b_table_triples.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory (default: {V4_OUTPUT_DIR})",
    )
    args = parser.parse_args()
    run_stage_c_v4(input_dir=args.input_dir, output_dir=args.output_dir)
