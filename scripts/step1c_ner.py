"""
Step 1c — Named Entity Recognition  [NEURAL]
─────────────────────────────────────────────────────────────────────────────
Input  : outputs/step1/text_blocks.json
Output : outputs/step1/entity_mentions.json
         Each record: { text, type, score, start, end,
                        source_page, source_file, source_text }

Model  : d4data/biomedical-ner-all  (BERT-based, HuggingFace)

Improvements over the naive BERT call (ported from
pipeline/inference/recognize_entities.py):

  1. Acronym pre-expansion — HF acronyms (HFrEF, NYHA, SGLT2i …) are
     expanded to full terms BEFORE the model runs so the model sees the
     surface form it was trained on.  Acronym map is kept in sync with
     step1a's _HF_ACRONYMS table.

  2. Word-boundary validation — entity spans that don't align to real word
     boundaries in the source text are rejected (avoids BERT subword artefacts
     like "##tion" being returned as an entity fragment).

  3. Preferred-phrase protection — the four canonical HF phenotype phrases
     ("heart failure with reduced/preserved/mildly reduced/improved ejection
     fraction") are kept whole; BERT often fragments them into "heart failure"
     + "ejection fraction" separately.

  4. Sliding-window chunking — instead of hard-truncating at 512 chars, long
     blocks are split into overlapping 450-char windows with 50-char overlap.
     Entities from all windows are merged and de-duplicated.  This prevents
     silent truncation of long guideline paragraphs.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from utils import log, save_json, load_json


# ── Acronym map (keep in sync with step1a's _HF_ACRONYMS) ───────────────────

_HF_ACRONYMS: Dict[str, str] = {
    "HFrEF":   "heart failure with reduced ejection fraction",
    "HFmrEF":  "heart failure with mildly reduced ejection fraction",
    "HFpEF":   "heart failure with preserved ejection fraction",
    "HFimpEF": "heart failure with improved ejection fraction",
    "HF":      "heart failure",
    "ACE-I":   "ACE inhibitor",
    "ACEi":    "angiotensin-converting enzyme inhibitors",
    "ARNI":    "angiotensin receptor-neprilysin inhibitor",
    "ARNi":    "angiotensin receptor-neprilysin inhibitors",
    "ARB":     "angiotensin receptor blockers",
    "MRA":     "mineralocorticoid receptor antagonist",
    "SGLT2i":  "sodium-glucose cotransporter-2 inhibitor",
    "SGLT2":   "sodium-glucose cotransporter-2",
    "RAASi":   "renin-angiotensin-aldosterone system inhibitors",
    "RASS":    "renin-angiotensin-aldosterone system",
    "NYHA":    "New York Heart Association",
    "AF":      "atrial fibrillation",
    "CAD":     "coronary artery disease",
    "ICD":     "implantable cardioverter-defibrillator",
    "CRT":     "cardiac resynchronization therapy",
    "LVAD":    "left ventricular assist device",
    "MCS":     "mechanical circulatory support",
    "DOAC":    "direct-acting oral anticoagulants",
    "BNP":     "B-type natriuretic peptide",
    "CKD":     "chronic kidney disease",
    "eGFR":    "estimated glomerular filtration rate",
    "T2DM":    "type 2 diabetes mellitus",
    "DM":      "diabetes mellitus",
    "HTN":     "hypertension",
    "MI":      "myocardial infarction",
    "ACS":     "acute coronary syndrome",
    "VT":      "ventricular tachycardia",
    "VF":      "ventricular fibrillation",
    "SCD":     "sudden cardiac death",
    "ECG":     "electrocardiogram",
    "GDMT":    "guideline-directed medical therapy",
}


# Protected full-phrase entities: BERT often fragments these into sub-spans.
# If the text contains one of these verbatim, we inject it as a top-level
# Disease_disorder entity and suppress any shorter substring that was found.
_PREFERRED_PHRASES: Set[str] = {
    "heart failure with reduced ejection fraction",
    "heart failure with preserved ejection fraction",
    "heart failure with mildly reduced ejection fraction",
    "heart failure with improved ejection fraction",
}

# Labels produced by d4data/biomedical-ner-all that are relevant for HF domain
KEEP_LABELS: Set[str] = {
    "Disease_disorder",
    "Medication",
    "Sign_symptom",
    "Diagnostic_procedure",
    "Therapeutic_procedure",
    "Biological_structure",
    "Lab_value",
}

# Sliding-window parameters for long text blocks
_WINDOW_CHARS   = 450
_OVERLAP_CHARS  = 50


# ── Helpers ──────────────────────────────────────────────────────────────────

def _expand_acronyms(text: str) -> str:
    """Expand known HF acronyms before NER (longest match first to avoid partial overlaps)."""
    for acronym, expansion in sorted(_HF_ACRONYMS.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(r"\b" + re.escape(acronym) + r"\b", expansion, text)
    return text


def _has_valid_boundary(entity_text: str, source_text: str) -> bool:
    """
    Validate that the entity span aligns to real word boundaries in the source.
    Rejects BERT subword artefacts that don't appear as standalone words.
    """
    if not entity_text or not source_text:
        return True
    try:
        match = re.search(
            r"(?<![a-zA-Z])" + re.escape(entity_text) + r"(?![a-zA-Z])",
            source_text,
            re.IGNORECASE,
        )
        return match is not None
    except re.error:
        return True


def _chunk_text(text: str, window: int = _WINDOW_CHARS, overlap: int = _OVERLAP_CHARS) -> List[tuple[int, str]]:
    """
    Split text into (offset, chunk) pairs with overlap.
    Allows NER to cover long guideline paragraphs without silent truncation.
    """
    if len(text) <= window:
        return [(0, text)]
    chunks = []
    start = 0
    while start < len(text):
        end = start + window
        chunks.append((start, text[start:end]))
        if end >= len(text):
            break
        start += window - overlap
    return chunks


def _merge_entities(entity_lists: List[List[Dict]]) -> List[Dict]:
    """
    Merge entities from multiple window runs.
    De-duplicates by (text_lower, label); keeps highest-score copy.
    """
    best: Dict[tuple, Dict] = {}
    for entities in entity_lists:
        for ent in entities:
            key = (ent["text"].lower(), ent["type"])
            existing = best.get(key)
            if existing is None or ent["score"] > existing["score"]:
                best[key] = ent
    return list(best.values())


def _apply_preferred_phrases(entities: List[Dict], source_text: str) -> List[Dict]:
    """
    Inject whole preferred-phrase entities and suppress their sub-string
    components (e.g. drop bare "heart failure" when the full HFrEF expansion
    is present).
    """
    text_lower = source_text.lower()
    injected: List[Dict] = []
    injected_lowers: Set[str] = set()

    for phrase in _PREFERRED_PHRASES:
        if phrase in text_lower:
            injected.append({
                "text":  phrase,
                "type":  "Disease_disorder",
                "score": 1.0,
                "start": text_lower.index(phrase),
                "end":   text_lower.index(phrase) + len(phrase),
            })
            injected_lowers.add(phrase)

    if not injected:
        return entities

    # Remove entities whose text is a strict substring of an injected phrase
    filtered: List[Dict] = []
    for ent in entities:
        t = ent["text"].lower()
        is_substring = any(t != p and t in p for p in injected_lowers)
        if not is_substring:
            filtered.append(ent)

    # Merge, de-duplicate by text+type
    seen: Set[tuple] = {(e["text"].lower(), e["type"]) for e in filtered}
    for ent in injected:
        key = (ent["text"].lower(), ent["type"])
        if key not in seen:
            filtered.append(ent)
            seen.add(key)

    return filtered


_MIN_ENTITY_CHARS = 4

_GENERIC_WORDS: Set[str] = {
    "medical", "therapy", "treatment", "treatments", "clinical", "functional",
    "structural", "negative", "favorable", "features", "complex", "measure",
    "blood", "heart", "failure", "reduced", "full", "reported", "available",
    "increased", "improved", "higher", "evidence", "changes", "oral", "factor",
    "risk", "class", "assessment", "care", "disease", "events", "positive",
    "severe", "moderate", "mild", "advanced", "chronic", "acute", "general",
    "specific", "primary", "secondary", "major", "minor", "total", "standard",
    "normal", "common", "related", "associated", "management", "evaluation",
    "level", "value", "rate", "effect", "effects", "outcome", "outcomes",
    "study", "studies", "data", "results", "patients", "population", "group",
    "time", "year", "years", "month", "months", "week", "day", "days",
    "high", "low", "mean", "median", "baseline", "control", "significant",
    # Bare medical adjectives that UMLS has entries for but are not entities
    "cardiac", "renal", "hepatic", "pulmonary", "vascular", "coronary",
    "systemic", "arterial", "venous", "peripheral", "central", "lateral",
    "anterior", "posterior", "inferior", "superior", "proximal", "distal",
    "congenital", "acquired", "progressive", "recurrent", "persistent",
    "bilateral", "unilateral", "invasive", "noninvasive", "subcutaneous",
    "intravenous", "elevated", "decreased", "impaired",
    # Process/action words that are not medical entities
    "medication", "medications", "injection", "examination", "detection",
    "recognition", "communication", "education", "recommendation",
    "recommendations", "reconstruction", "rehabilitation", "enhancement",
    "improvement", "improvements", "reduction", "reductions", "conditions",
    "restriction", "elevation", "elevations", "information", "screening",
    "remodeling", "measurement", "implantation", "intervention",
    "interventions", "complication", "complications",
}


def _is_valid_entity(text_span: str) -> bool:
    """
    Post-NER quality gate. Rejects:
      - spans shorter than _MIN_ENTITY_CHARS
      - pure numbers / reference markers
      - generic English words that UMLS happens to have entries for
      - spans that are mostly non-alpha (table/formula fragments)
      - single-word entities that are just adjectives/generic nouns
    """
    t = text_span.strip()
    if len(t) < _MIN_ENTITY_CHARS:
        return False
    if re.match(r"^[\d\s\.\-,;:/%()]+$", t):
        return False
    alpha = sum(1 for c in t if c.isalpha())
    if len(t) > 0 and alpha / len(t) < 0.6:
        return False
    if t.lower() in _GENERIC_WORDS:
        return False
    return True


def _should_process(text: str) -> bool:
    """Skip blocks that are clearly not natural-language sentences."""
    if not text or not text.strip():
        return False
    text = text.strip()
    alpha = sum(1 for c in text if c.isalpha())
    if len(text) > 0 and alpha / len(text) < 0.5:
        return False
    if re.match(r"^[\d\s\.\-,;:]+$", text):
        return False
    words = text.split()
    unique = {w.lower() for w in words if len(w) > 2}
    if len(words) > 10 and len(unique) < len(words) * 0.3:
        return False
    return True


# ── Main NER function ────────────────────────────────────────────────────────

def run_ner(
    text_blocks_path: str = "outputs/step1/text_blocks.json",
    output_dir: str = "outputs/step1",
    model_name: str = "d4data/biomedical-ner-all",
    min_score: float = 0.55,
    keep_labels: Optional[Set[str]] = None,
) -> List[Dict]:
    """
    Run BERT-based NER over all text blocks.

    Improvements over naive approach:
      • Acronym expansion before model call
      • Sliding-window chunking for long paragraphs (no silent truncation)
      • Word-boundary validation of span results
      • Preferred-phrase protection for canonical HF phenotype terms

    Parameters
    ----------
    text_blocks_path : path to text_blocks.json (output of Step 1a)
    output_dir       : directory for entity_mentions.json
    model_name       : HuggingFace model ID (d4data/biomedical-ner-all)
    min_score        : confidence threshold
    keep_labels      : entity type filter; None = use KEEP_LABELS default

    Returns list of entity mention dicts and writes entity_mentions.json.
    """
    from transformers import pipeline as hf_pipeline

    if keep_labels is None:
        keep_labels = KEEP_LABELS

    blocks: List[Dict] = load_json(text_blocks_path)

    log("1c", f"Loading NER model: {model_name}")
    ner = hf_pipeline(
        "ner",
        model=model_name,
        aggregation_strategy="simple",
        device=-1,   # CPU; set to 0 for GPU on Colab
    )

    mentions: List[Dict] = []

    for block in blocks:
        raw_text: str = block.get("text", "")
        if not _should_process(raw_text):
            continue

        # Expand acronyms so the model sees full medical terms
        expanded_text = _expand_acronyms(raw_text)

        # Sliding-window NER to avoid silent BERT truncation
        window_results: List[List[Dict]] = []
        for offset, chunk in _chunk_text(expanded_text):
            try:
                raw_ents = ner(chunk)
            except Exception as exc:
                log("1c", f"NER error on page {block.get('page')}: {exc}")
                continue

            window_ents: List[Dict] = []
            for ent in raw_ents:
                label: str   = ent["entity_group"]
                score: float = ent["score"]

                if score < min_score:
                    continue
                if keep_labels and label not in keep_labels:
                    continue

                text_span = ent["word"].strip()
                if not text_span:
                    continue
                # Normalise whitespace (BERT sometimes returns "##tion"-style artefacts)
                text_span = " ".join(text_span.split()).replace(" ##", "")

                # Word-boundary validation
                if not _has_valid_boundary(text_span, chunk):
                    continue

                if not _is_valid_entity(text_span):
                    continue

                window_ents.append({
                    "text":  text_span,
                    "type":  label,
                    "score": round(float(score), 4),
                    "start": ent["start"] + offset,
                    "end":   ent["end"]   + offset,
                })
            window_results.append(window_ents)

        # Merge windows and de-duplicate
        merged = _merge_entities(window_results)

        # Preferred-phrase injection and sub-string suppression
        merged = _apply_preferred_phrases(merged, expanded_text)

        for ent in merged:
            mentions.append({
                "text":        ent["text"],
                "type":        ent["type"],
                "score":       ent.get("score", 1.0),
                "start":       ent.get("start"),
                "end":         ent.get("end"),
                "source_page": block["page"],
                "source_file": block["source_file"],
                # First 200 chars for context in Step 1e disambiguation prompt
                "source_text": expanded_text[:200],
            })

    out_path = Path(output_dir) / "entity_mentions.json"
    save_json(mentions, str(out_path))
    log("1c", f"Found {len(mentions)} entity mentions")
    return mentions


if __name__ == "__main__":
    run_ner()
