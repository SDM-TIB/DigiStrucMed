#!/usr/bin/env python3
"""
Diagnostic: Show what the v2 NER model actually extracts (before label filtering).

Run: python Tests/diagnose_stage_c_v2.py

This proves v2 detects entities - they're just PER/ORG/LOC/MISC, which get
filtered out by keep_labels (biomedical only).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.models import NeuralModel

# v2 model - RoBERTa CoNLL NER
MODEL_ID = "Jean-Baptiste/roberta-large-ner-english"

# Sample biomedical text (similar to guideline chunks)
SAMPLE = (
    "The American College of Cardiology and American Heart Association recommend "
    "ACE inhibitors for patients with heart failure. Dr. Smith from Boston "
    "presented the findings at the Mayo Clinic."
)

def main():
    print("=" * 60)
    print("Stage C v2 NER diagnostic")
    print("=" * 60)
    print(f"\nModel: {MODEL_ID}")
    print(f"Sample text: {SAMPLE[:80]}...")
    print("\nLoading model (this may take a moment)...")
    model = NeuralModel(model_name=MODEL_ID)
    print("Running NER (NO label filter - raw output)...")
    # NeuralModel.extract_entities with keep_labels=None returns everything
    entities = model.extract_entities(
        text=SAMPLE,
        min_score=0.55,
        max_entities=50,
        keep_labels=None,  # No filter - get all labels
    )
    print(f"\nRaw entities extracted: {len(entities)}")
    for ent in entities:
        print(f"  - {ent.get('text')!r}  label={ent.get('label')}  score={ent.get('score')}")
    labels = set(e.get("label") for e in entities if e.get("label"))
    print(f"\nLabels found: {labels}")
    print("\n" + "=" * 60)
    print("EXPLANATION:")
    print("  v2 outputs PER, ORG, LOC, MISC (CoNLL NER).")
    print("  RecognizeEntities keeps only: Medication, Disease_disorder,")
    print("  Diagnostic_procedure, Sign_symptom, Therapeutic_procedure.")
    print("  So v2 entities are filtered out -> 0 in stage output.")
    print("=" * 60)


if __name__ == "__main__":
    main()
