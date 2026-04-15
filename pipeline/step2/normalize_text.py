"""
normalize_text.py — Ontology-slot-driven extraction from unstructured text (refactored v2).

Key improvements over v1:
  1. Assessment extraction uses OPEN-ENDED pattern categories, not a closed list.
     Clinical assessment categories are defined as pattern groups (imaging, lab,
     physical, functional, etc.) so new specific terms are captured automatically.
  2. Adverse event extraction uses broader event-like patterns plus a curated
     symptom list covering common clinical adverse effects (nausea, cough,
     dizziness, fatigue, etc.) that were previously missed.
  3. Both extractors produce de-duplicated, lowercase-normalized output.
  4. Source provenance (page, source_file) is tracked for traceability.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from collections import Counter


def _load_config(config_path=None):
    """Load guideline-specific configuration from JSON file."""
    if config_path is None:
        config_path = Path(__file__).parent / "guideline_config.json"
    else:
        config_path = Path(config_path)
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {}

_CFG = _load_config()


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
# Assessment extraction (open-ended)
# ═══════════════════════════════════════════════════════════════
# Instead of a closed list of terms, we define CATEGORIES of clinical
# assessments. Each category has a regex pattern that captures
# variations across different guidelines.

_ASSESSMENT_CATEGORIES = _CFG.get("assessment_categories", {})

_ASSESSMENT_PATTERN = re.compile(
    r"\b(" + "|".join(_ASSESSMENT_CATEGORIES.values()) + r")\b",
    flags=re.IGNORECASE,
)


_SUFFIX_STRIP_RE = re.compile(
    r"((?:ic|ical|ically|ive|ing|tion|sion|ment|ance|ence|ogy|"
    r"ism|ist|ous|ible|able|ity|al|ly|s|es|ed)$)",
    re.IGNORECASE,
)

_TRUNC_WORD_RE = re.compile(r"\b[a-z]{2,}[a-z]$", re.IGNORECASE)


def _assessment_dedup_key(name: str) -> str:
    """Aggressive normalization for dedup: strip hyphens, spaces, trailing
    morphological suffixes, then lowercase.  This collapses 'troponin' /
    'troponins', 'echocardiographic' / 'echocardiography', etc.
    """
    key = re.sub(r"[\s\-]+", "", name).lower()
    key = _SUFFIX_STRIP_RE.sub("", key)
    return key


def _looks_truncated(name: str) -> bool:
    """Detect tokens that appear to be truncated mid-word (e.g. 'physical examina')."""
    words = name.split()
    if not words:
        return False
    last = words[-1]
    if len(last) < 4:
        return False
    if last[-1] in "aeiouy" and not last.endswith("ee"):
        vowel_end = True
    else:
        vowel_end = False
    if vowel_end and len(last) >= 6 and last[-2:] not in ("le", "re", "se", "ne", "te", "de"):
        return True
    return False


def extract_assessments(text_blocks_path: Path) -> list[Hit]:
    blocks = json.loads(text_blocks_path.read_text(encoding="utf-8"))
    hits: dict[str, Hit] = {}
    for b in blocks:
        page = int(b.get("page") or 0)
        src = str(b.get("source_file") or "")
        for sent in _sentences(b.get("text") or ""):
            for m in _ASSESSMENT_PATTERN.finditer(sent):
                name = m.group(0).strip()
                if len(name) < 4:
                    continue
                alpha = sum(1 for ch in name if ch.isalpha())
                if alpha < 4:
                    continue
                if _looks_truncated(name):
                    continue
                key = _assessment_dedup_key(name)
                if key not in hits:
                    hits[key] = Hit(page=page, source_file=src, span=name)
    return list(hits.values())


# ═══════════════════════════════════════════════════════════════
# Assessment value extraction (name -> measured/result value)
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
    """Extract generic value/result snippets for each assessment mention."""
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
# Adverse event extraction (broadened)
# ═══════════════════════════════════════════════════════════════

# Harm-context trigger (same as v1 but broader)
_HARM_TRIGGER = re.compile(
    r"\b(adverse (effects?|events?|reactions?)|side effects?|harmful|"
    r"contraindicat\w+|toxicit\w*|avoid(ed)?|"
    r"monitor(ed|ing)?\s+for|caution|warning|"
    r"risk of|may cause|can cause|associated with)\b",
    flags=re.IGNORECASE,
)

_LIST_CUE = re.compile(r"\b(including|such as|e\.g\.|for example|particularly|especially)\b", flags=re.IGNORECASE)

# Common clinical adverse events (the closed list from v1 missed many of these)
_COMMON_AE_TERMS = set(_CFG.get("common_adverse_event_terms", []))

_EVENT_SUFFIX = re.compile(
    r"(emia|tension|dysfunction|failure|edema|bleed|infection|arrhythm|"
    r"angioedema|bradycard|tachycard|shock|stroke|fracture|hypersensit|"
    r"toxicit|opathy|itis|osis|penia|uria|spasm|trophy|trophy|nausea|"
    r"cough|dizziness|fatigue|vomiting|diarrhea|constipation|headache|"
    r"rash|gynecomastia|prolongation)",
    flags=re.IGNORECASE,
)

_STOP_WORDS = set(_CFG.get("stop_words", []))


def extract_adverse_events(text_blocks_path: Path) -> list[Hit]:
    blocks = json.loads(text_blocks_path.read_text(encoding="utf-8"))
    hits: dict[str, Hit] = {}

    for b in blocks:
        page = int(b.get("page") or 0)
        src = str(b.get("source_file") or "")
        for sent in _sentences(b.get("text") or ""):
            if not _HARM_TRIGGER.search(sent):
                continue

            # Strategy 1: Extract from enumeration lists
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
                    if len(words) > 6:
                        continue
                    low = " ".join(w.lower() for w in words)
                    if low in _STOP_WORDS:
                        continue
                    if _EVENT_SUFFIX.search(it) or low in _COMMON_AE_TERMS:
                        if low not in hits:
                            hits[low] = Hit(page=page, source_file=src, span=it)

            # Strategy 2: Scan for known AE terms anywhere in harm-context sentences
            sent_lower = sent.lower()
            for term in _COMMON_AE_TERMS:
                if term.lower() in sent_lower and term.lower() not in hits:
                    hits[term.lower()] = Hit(page=page, source_file=src, span=term)

    return list(hits.values())


# ═══════════════════════════════════════════════════════════════
# Writers
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
    rows = [{"ae_id": f"ae_{_slug(h.span)}", "adverseEventName": h.span, "adverseEventSeverity": ""} for h in hits]
    return _append_csv(out_dir / "S_adverse_event.csv", ["ae_id", "adverseEventName", "adverseEventSeverity"], rows)


def run_text_normalization(run_dir: Path, out_dir: Path, disease_id: str = None, config_path: str = None) -> dict:
    global _CFG
    if config_path:
        _CFG = _load_config(config_path)
    if disease_id is None:
        disease_id = _CFG.get("disease", {}).get("disease_id", "Unknown")
    step1 = run_dir / "step1"
    text_blocks = step1 / "text_blocks.json"
    if not text_blocks.exists():
        return {"text_blocks": str(text_blocks), "status": "missing"}

    assessments = extract_assessments(text_blocks)
    assessment_values = extract_assessment_values(text_blocks, assessments)
    adverse_events = extract_adverse_events(text_blocks)

    n_assess = write_assessments(
        out_dir,
        disease_id=disease_id,
        hits=assessments,
        value_map=assessment_values,
    )
    n_ae = write_adverse_events(out_dir, hits=adverse_events)

    return {
        "text_blocks": str(text_blocks),
        "rows_written": {
            "S_assessment.csv": n_assess,
            "S_adverse_event.csv": n_ae,
        },
        "assessment_categories_used": len(_ASSESSMENT_CATEGORIES),
        "common_ae_terms_available": len(_COMMON_AE_TERMS),
    }
