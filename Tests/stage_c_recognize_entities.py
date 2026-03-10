import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _print_ner_cache_info(neural_model_name: str) -> None:
    """Print where the NER model is loaded from (local or Hugging Face cache)."""
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
import config
from pipeline.data import TextChunks, StatementsWithMedicalEntities
from pipeline.models import NeuralModel
from pipeline.inference import RecognizeEntities

# Default model/configs for Stage C (versioned)
STAGE_C_CONFIGS = {
    "v1": {
        "neural_model_name": "d4data/biomedical-ner-all",
        "min_ner_score": 0.55,
        "acronym_file": None,
    },
    "v2": {
        # RoBERTa-based NER model (same schema, different backbone)
        # This model is not biomedical-specific but gives a contrasting RoBERTa engine.
        "neural_model_name": "Jean-Baptiste/roberta-large-ner-english",
        "min_ner_score": 0.55,
        "acronym_file": None,
    },
}


def _get_stage_c_config(version: str | None) -> dict:
    """Return the config for the requested Stage C version, defaulting to v1."""
    ver = version or config.DEFAULT_STAGE_C_VERSION
    return STAGE_C_CONFIGS.get(ver, STAGE_C_CONFIGS["v1"])


def _resolve_stage_c_paths(
    stage_b_version: str | None = None,
    stage_c_version: str | None = None,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """
    Resolve input/output locations for Stage C.

    Inputs:
      - Prefer explicit input_dir (containing Stage B outputs)
      - Otherwise use config.stage_b_dir(stage_b_version)
      - If the main chunks file is missing, fall back to Tests/outputs for backward compatibility

    Output:
      - Prefer explicit output_dir
      - Otherwise use config.stage_c_dir(stage_c_version)
    """
    if input_dir is not None:
        in_root = Path(input_dir)
    else:
        in_root = config.stage_b_dir(stage_b_version)

    chunks_path = in_root / "stage_b_text_chunks.json"
    triples_path = in_root / "stage_b_table_triples.json"

    if not chunks_path.exists():
        # Backward compat: Tests/outputs
        fallback_root = Path(__file__).parent / "outputs"
        fb_chunks = fallback_root / "stage_b_text_chunks.json"
        if fb_chunks.exists():
            chunks_path = fb_chunks
            triples_path = fallback_root / "stage_b_table_triples.json"

    if output_dir is not None:
        out_root = Path(output_dir)
    else:
        out_root = config.stage_c_dir(stage_c_version)

    return chunks_path, triples_path, out_root


def test_stage_c(
    stage_b_version: str | None = None,
    stage_c_version: str | None = None,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
):
    print("=" * 70)
    print("STAGE C: text_chunks + table_triples -> NER -> statements_with_entities + table_triples")
    print("(NER + acronym expansion; both Stage B outputs loaded and written)")
    print("=" * 70)

    # Decide which Stage C version we are running (v1, v2, ...)
    effective_stage_c_version = stage_c_version or config.DEFAULT_STAGE_C_VERSION
    cfg = _get_stage_c_config(effective_stage_c_version)
    neural_model_name = cfg["neural_model_name"]
    min_ner_score = cfg["min_ner_score"]
    acronym_file = cfg["acronym_file"]

    input_chunks, input_table_triples, output_root = _resolve_stage_c_paths(
        stage_b_version=stage_b_version,
        stage_c_version=effective_stage_c_version,
        input_dir=input_dir,
        output_dir=output_dir,
    )

    output_file = output_root / "stage_c_statements_with_entities.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[1] Loading Stage B outputs: text chunks + table triples...")
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
    if input_table_triples.exists():
        with open(input_table_triples, "r", encoding="utf-8") as f:
            triples_data = json.load(f)
        table_triples = triples_data.get("triples", []) if isinstance(triples_data, dict) else triples_data
        print(f"    Loaded table triples: {len(table_triples)}")
    else:
        print("    Table triples file not found; using empty list.")
    print(f"\n[2] Initializing neural model: {neural_model_name}...")
    neural_model = NeuralModel(model_name=neural_model_name)
    _print_ner_cache_info(neural_model_name)
    print(f"\n[3] Initializing recognize_entities...")
    print(f"    Min NER score: {min_ner_score}")
    print(f"    Acronym expansion: enabled (integrated into NER)")
    recognizer = RecognizeEntities(
        neural_model=neural_model,
        min_score=min_ner_score,
        acronym_file=acronym_file,
        verbose=True,
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
    print(f"STAGE C {effective_stage_c_version} COMPLETE")
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
        description="Run Stage C v1 (NER + acronym expansion) on Stage B outputs.",
    )
    parser.add_argument(
        "--stage-b-version",
        type=str,
        default=None,
        help="Stage B version to read from (e.g. v1). If not set, uses config.DEFAULT_STAGE_B_VERSION.",
    )
    parser.add_argument(
        "--stage-c-version",
        type=str,
        default=None,
        help="Stage C version to write to (e.g. v1). If not set, uses config.DEFAULT_STAGE_C_VERSION.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Explicit input directory containing Stage B outputs (stage_b_text_chunks.json, stage_b_table_triples.json). Overrides --stage-b-version.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Explicit output directory for Stage C artifacts. Defaults to config.stage_c_dir().",
    )
    args = parser.parse_args()

    test_stage_c(
        stage_b_version=args.stage_b_version,
        stage_c_version=args.stage_c_version,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
