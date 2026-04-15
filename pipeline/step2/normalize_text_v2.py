"""
normalize_text_v2_fixed.py — Fully generic text extraction (no predefined term lists).

Noise-controlled version of v2: same generic approach, but frequency thresholds
and stricter stopword/length filters are applied so the output volume matches v1
without relying on any disease-specific term list.

Noise root-causes fixed vs original v2:
  1. extract_assessments: captured hapax fragments like "value for a drug",
     "encompasses clinical evaluation". Fix: frequency threshold (min_freq=2)
     + minimum word count (>=2 meaningful words).
  2. extract_adverse_events: captured generic sentence fragments by matching any
     word with a medical suffix in a harm-context sentence.  Fix: frequency
     threshold + stricter length/word-count gates.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


def _slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (s or "").strip())
    return s.strip("_")[:80] or "unknown"


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?;])\s+", text)
    return [p.strip() for p in parts if p.strip()]


@dataclass(frozen=True)
class Hit:
    page: int
    source_file: str
    span: str


# ═══════════════════════════════════════════════════════════════
# Assessment extraction — structural cues + frequency gate
# ═══════════════════════════════════════════════════════════════

_ASSESS_CUE = re.compile(
    r'\b(?:'
    r'assess(?:ed|ment)?\s+(?:with|by|using|of|for)\s+|'
    r'evaluat(?:ed|ion|e)\s+(?:with|by|using|of|for)\s+|'
    r'monitor(?:ed|ing)?\s+(?:with|by|using|of|for)\s+|'
    r'test(?:ed|ing)?\s+(?:with|by|using|for)\s+|'
    r'diagnos(?:ed|is|tic)\s+(?:with|by|using)\s+|'
    r'measur(?:ed|ement|ing)\s+(?:with|by|of|using)\s+|'
    r'screen(?:ed|ing)?\s+(?:with|by|for|using)\s+'
    r')'
    r'([A-Za-z][A-Za-z\s\-]{2,50})',
    flags=re.IGNORECASE,
)

_ASSESS_NOUN = re.compile(
    r'\b([A-Za-z][A-Za-z\s\-]{2,40}\s+'
    r'(?:test(?:ing)?|assessment|evaluation|imaging|'
    r'echocardiograph\w*|electrocardiogra\w*|catheteriz\w*|'
    r'biopsy|monitoring|screening|examination|'
    r'scan|angiograph\w*|ultrasound|radiograph\w*))\\b',
    flags=re.IGNORECASE,
)

_ASSESS_NOUN = re.compile(
    r'\b([A-Za-z][A-Za-z\s\-]{2,40}\s+'
    r'(?:test(?:ing)?|assessment|evaluation|imaging|'
    r'echocardiograph\w*|electrocardiogra\w*|catheteriz\w*|'
    r'biopsy|monitoring|screening|examination|'
    r'scan|angiograph\w*|ultrasound|radiograph\w*))\b',
    flags=re.IGNORECASE,
)

_ASSESS_STOPWORDS = {
    "the", "a", "an", "this", "that", "no", "any", "some", "their",
    "and", "or", "with", "for", "of", "is", "are", "initial",
    "routine", "regular", "serial", "baseline", "follow-up",
    # Additional filler that v2 was not blocking:
    "value", "encompasses", "contraindications", "interactions",
    "clinical", "diagnostic", "further", "appropriate", "adequate",
}


def _clean_assessment(name: str) -> str | None:
    name = name.strip().rstrip(" .,;:()")
    words = name.split()
    while words and words[0].lower() in _ASSESS_STOPWORDS:
        words.pop(0)
    meaningful = [w for w in words if w.lower() not in _ASSESS_STOPWORDS]
    if len(meaningful) < 2 or len(" ".join(words)) < 6:
        return None
    cleaned = " ".join(words)
    if _looks_truncated(cleaned):
        return None
    return cleaned


def _looks_truncated(name: str) -> bool:
    """Detect tokens that appear to be truncated mid-word (e.g. 'physical examina')."""
    words = name.split()
    if not words:
        return False
    last = words[-1]
    if len(last) < 6:
        return False
    if last[-1] in "aeiouy" and last[-2:] not in ("le", "re", "se", "ne", "te", "de", "ee"):
        return True
    return False


def _assessment_dedup_key(name: str) -> str:
    """Aggressive normalization for dedup: strip hyphens, spaces, trailing suffixes."""
    key = re.sub(r"[\s\-]+", "", name).lower()
    key = re.sub(r"(?:ic|ical|ically|ive|ing|tion|sion|ment|ance|ence|ogy|"
                 r"ism|ist|ous|ible|able|ity|al|ly|s|es|ed)$", "", key, flags=re.I)
    return key


def extract_assessments(text_blocks_path: Path, min_freq: int = 2) -> list[Hit]:
    """
    FIX vs original v2:
      - Requires at least 2 meaningful words after stopword stripping (eliminates
        single-word fragments like "natriuretic peptide" → kept, "evaluation" → dropped).
      - Applies frequency threshold: only emit assessment if seen in ≥ min_freq
        distinct text blocks.  Eliminates hapax fragments.
    """
    blocks = json.loads(text_blocks_path.read_text(encoding="utf-8"))

    # Pass 1: collect candidates with block indices
    candidates: dict[str, list[int]] = defaultdict(list)
    names_by_key: dict[str, Hit] = {}

    for block_idx, b in enumerate(blocks):
        page = int(b.get("page") or 0)
        src = str(b.get("source_file") or "")
        for sent in _sentences(b.get("text") or ""):
            for pattern in (_ASSESS_CUE, _ASSESS_NOUN):
                for m in pattern.finditer(sent):
                    name = _clean_assessment(m.group(1))
                    if name:
                        key = _assessment_dedup_key(name)
                        candidates[key].append(block_idx)
                        if key not in names_by_key:
                            names_by_key[key] = Hit(page=page, source_file=src, span=name)

    # Pass 2: frequency filter
    return [
        names_by_key[key]
        for key, blocks_seen in candidates.items()
        if len(set(blocks_seen)) >= min_freq
    ]


# ═══════════════════════════════════════════════════════════════
# Assessment value extraction — 100% generic patterns
# ═══════════════════════════════════════════════════════════════

_VALUE_COMPARATOR_RE = re.compile(
    r"(?:(?:<=|>=|<|>|=)\s*\d+(?:\.\d+)?(?:\s*[a-zA-Z%/]+)?|"
    r"\d+(?:\.\d+)?\s*(?:%|mg/dL|mmHg|mm\s*Hg|pg/mL|ng/L|ms|bpm|mL/min))",
    flags=re.IGNORECASE,
)

_VALUE_QUAL_RE = re.compile(
    r"\b(?:persistently\s+)?(?:elevated|reduced|normal|abnormal|worsening|improved|severe|mild|moderate)\b",
    flags=re.IGNORECASE,
)


def extract_assessment_values(text_blocks_path: Path, hits: list[Hit]) -> dict[str, str]:
    """Generic value extractor; no disease-specific keywords."""
    blocks = json.loads(text_blocks_path.read_text(encoding="utf-8"))
    sentences = []
    for b in blocks:
        sentences.extend(_sentences(b.get("text") or ""))

    values: dict[str, str] = {}
    for h in hits:
        key = _assessment_dedup_key(h.span)
        candidates: list[tuple[int, str]] = []
        needle = h.span.lower()
        for sent in sentences:
            low = sent.lower()
            if needle not in low:
                continue
            m = _VALUE_COMPARATOR_RE.search(sent)
            if m:
                candidates.append((3, m.group(0).strip()))
                continue
            m = _VALUE_QUAL_RE.search(sent)
            if m:
                candidates.append((1, m.group(0).strip()))
        if candidates:
            best_score = max(sc for sc, _ in candidates)
            top = [v for sc, v in candidates if sc == best_score]
            values[key] = Counter(top).most_common(1)[0][0]
        else:
            values[key] = ""
    return values


# ═══════════════════════════════════════════════════════════════
# Adverse event extraction — harm context + suffix + frequency gate
# ═══════════════════════════════════════════════════════════════

_HARM_TRIGGER = re.compile(
    r"\b(adverse (effects?|events?|reactions?)|side effects?|harmful|"
    r"contraindicat\w+|toxicit\w*|avoid(ed)?|"
    r"monitor(ed|ing)?\s+for|caution|warning|"
    r"risk of|may cause|can cause|associated with)\b",
    flags=re.IGNORECASE,
)

_LIST_CUE = re.compile(
    r"\b(including|such as|e\.g\.|for example|particularly|especially)\b",
    flags=re.IGNORECASE,
)

_EVENT_SUFFIX = re.compile(
    r"(emia|tension|dysfunction|failure|edema|bleed|infection|arrhythm|"
    r"angioedema|bradycard|tachycard|shock|stroke|fracture|hypersensit|"
    r"toxicit|opathy|itis|osis|penia|uria|spasm|trophy|nausea|"
    r"cough|dizziness|fatigue|vomiting|diarrhea|constipation|headache|"
    r"rash|gynecomastia|prolongation)",
    flags=re.IGNORECASE,
)

_AE_STOP_WORDS = {
    "patient", "patients", "therapy", "treatment", "drug", "drugs",
    "clinical trial", "study", "guideline", "recommendation",
    "class", "level", "evidence",
    # Additional filler:
    "heart failure", "heart", "failure", "disease", "condition",
}


def extract_adverse_events(text_blocks_path: Path, min_freq: int = 2) -> list[Hit]:
    """
    FIX vs original v2:
      - Frequency threshold (min_freq=2): only emit if seen in ≥ min_freq blocks.
      - Stricter word-count gate: items from enumeration must be 1–4 words.
      - Strategy 2 (suffix scan) now also requires multi-word terms (≥2 words)
        to avoid single-word false positives like "nausea" extracted everywhere.
    """
    blocks = json.loads(text_blocks_path.read_text(encoding="utf-8"))

    candidates: dict[str, list[int]] = defaultdict(list)
    names_by_key: dict[str, Hit] = {}

    for block_idx, b in enumerate(blocks):
        page = int(b.get("page") or 0)
        src = str(b.get("source_file") or "")

        for sent in _sentences(b.get("text") or ""):
            if not _HARM_TRIGGER.search(sent):
                continue

            # Strategy 1: enumeration lists after "including / such as / e.g."
            m = _LIST_CUE.search(sent)
            if m:
                tail = sent[m.end():]
                items = re.split(r",|\band\b|\bor\b", tail)
                for it in items:
                    it = it.strip(" .;:()[]")
                    if not it or len(it) < 3 or len(it) > 60:
                        continue
                    if any(ch.isdigit() for ch in it):
                        continue
                    words = re.split(r"\s+", it)
                    # FIX: tighten to 1–4 words (original allowed 1–6)
                    if len(words) < 1 or len(words) > 4:
                        continue
                    low = " ".join(w.lower() for w in words)
                    if low in _AE_STOP_WORDS:
                        continue
                    if _EVENT_SUFFIX.search(it):
                        candidates[low].append(block_idx)
                        if low not in names_by_key:
                            names_by_key[low] = Hit(page=page, source_file=src, span=it)

            # Strategy 2: multi-word terms with medical suffixes in harm-context sentences
            # FIX: require ≥2 words to avoid adding every single-word suffix match
            for word_m in re.finditer(r'\b([A-Za-z][a-z]+(?:\s+[a-z]+){1,2})\b', sent):
                term = word_m.group(1).strip()
                if len(term) < 6 or len(term) > 50:
                    continue
                if _EVENT_SUFFIX.search(term):
                    low = term.lower()
                    if low not in _AE_STOP_WORDS:
                        candidates[low].append(block_idx)
                        if low not in names_by_key:
                            names_by_key[low] = Hit(page=page, source_file=src, span=term)

    # Frequency filter
    return [
        names_by_key[key]
        for key, blocks_seen in candidates.items()
        if len(set(blocks_seen)) >= min_freq
    ]


# ═══════════════════════════════════════════════════════════════
# Writers — same CSV output format as v1
# ═══════════════════════════════════════════════════════════════

def _append_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    return len(rows)


def write_assessments(
    out_dir: Path,
    disease_id: str,
    hits: list[Hit],
    value_map: dict[str, str] | None = None,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    value_map = value_map or {}
    rows_a = []
    for h in hits:
        key = _assessment_dedup_key(h.span)
        rows_a.append(
            {
                "assessment_id": f"assessment_{_slug(h.span)}",
                "assessmentName": h.span,
                "assessmentValue": value_map.get(key, ""),
            }
        )
    n = _append_csv(
        out_dir / "S_assessment.csv",
        ["assessment_id", "assessmentName", "assessmentValue"],
        rows_a,
    )
    return n


def write_adverse_events(out_dir: Path, hits: list[Hit]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"ae_id": f"ae_{_slug(h.span)}", "adverseEventName": h.span,
             "adverseEventSeverity": ""} for h in hits]
    return _append_csv(out_dir / "S_adverse_event.csv",
                       ["ae_id", "adverseEventName", "adverseEventSeverity"], rows)


def run_text_normalization(
    run_dir: Path,
    out_dir: Path,
    disease_id: str = "Unknown",
    min_freq: int = 2,
) -> dict:
    """
    Args:
        min_freq: minimum block frequency for assessments and adverse events.
                  Set to 1 to match original v2 behaviour (no filter).
                  Default 2 eliminates hapax fragments.
    """
    step1 = run_dir / "step1"
    text_blocks = step1 / "text_blocks.json"
    if not text_blocks.exists():
        return {"text_blocks": str(text_blocks), "status": "missing"}

    assessments = extract_assessments(text_blocks, min_freq=min_freq)
    assessment_values = extract_assessment_values(text_blocks, assessments)
    adverse_events = extract_adverse_events(text_blocks, min_freq=min_freq)

    n_assess = write_assessments(
        out_dir,
        disease_id=disease_id,
        hits=assessments,
        value_map=assessment_values,
    )
    n_ae = write_adverse_events(out_dir, hits=adverse_events)

    return {
        "text_blocks": str(text_blocks),
        "version": "v2-generic-fixed",
        "quality_gates": {"min_freq": min_freq},
        "rows_written": {
            "S_assessment.csv": n_assess,
            "S_adverse_event.csv": n_ae,
        },
    }
