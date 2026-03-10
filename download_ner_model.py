#!/usr/bin/env python3
"""
Pre-download NER models to a local project folder (models/ner/).

Run this once to cache models locally. Useful for:
- Offline use
- Faster subsequent loads (no HF hub lookup)
- Keeping models in the project

Usage:
  python download_ner_model.py              # downloads v1 (biomedical, ~250MB)
  python download_ner_model.py --version v2 # downloads v2 (RoBERTa, ~1.3GB)
  python download_ner_model.py --version v1 v2  # both
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config

# Model IDs and approximate sizes
NER_MODELS = {
    "v1": {
        "id": "d4data/biomedical-ner-all",
        "size_mb": 250,
        "params": "66M",
        "ram_gb": "~2",
    },
    "v2": {
        "id": "Jean-Baptiste/roberta-large-ner-english",
        "size_mb": 1300,
        "params": "354M",
        "ram_gb": "~4–8",
    },
}


def _model_name_to_dir(model_id: str) -> str:
    """Convert 'org/model-name' to safe folder name."""
    return model_id.replace("/", "__")


def download_model(version: str, target_dir: Path) -> bool:
    """Download a model to target_dir. Returns True on success."""
    if version not in NER_MODELS:
        print(f"Unknown version: {version}. Use v1 or v2.")
        return False
    info = NER_MODELS[version]
    model_id = info["id"]
    folder_name = _model_name_to_dir(model_id)
    out_dir = target_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading {model_id} (~{info['size_mb']}MB, {info['params']} params)...")
    print(f"  Target: {out_dir}")
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForTokenClassification.from_pretrained(model_id)
        tokenizer.save_pretrained(out_dir)
        model.save_pretrained(out_dir)
        print(f"  Done. Model saved to {out_dir}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download NER models to models/ner/ for local use."
    )
    parser.add_argument(
        "--version",
        nargs="+",
        default=["v1"],
        choices=["v1", "v2"],
        help="Which model(s) to download (default: v1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.NER_MODELS_DIR,
        help=f"Output directory (default: {config.NER_MODELS_DIR})",
    )
    args = parser.parse_args()
    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("NER model download")
    print("=" * 60)
    print("\nModel sizes & laptop compatibility:")
    for v, info in NER_MODELS.items():
        print(f"  {v}: {info['id']}")
        print(f"      ~{info['size_mb']}MB disk, {info['params']} params, {info['ram_gb']}GB RAM")
    print("\n  v1: runs on average laptop (8GB RAM). Biomedical, ~250MB.")
    print("  v2: needs 8–16GB RAM. General NER, ~1.3GB.")
    ok = all(download_model(v, target) for v in args.version)
    print("\n" + "=" * 60)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
