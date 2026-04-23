"""Step 2: generic text-side entity extraction (frequency + fragment gates; no term lists)."""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


def _slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (s or "").strip()).strip("_")
    if len(s) <= 80:
        return s or "unknown"
    truncated = s[:80]
    last_sep = truncated.rfind("_")
    if last_sep > 40:
        truncated = truncated[:last_sep]
    return truncated.strip("_") or "unknown"


_TRAILING_FILLER = re.compile(
    r'\s+(?:and|or|with|for|of|in|the|a|an|to|is|are|was|were|by|at|on|from|'
    r'due|has|have|had|that|which|who|also|but|not|as|if|than|into|during|'
    r'between|without|within|about|after|before|through|over|under|'
    r'their|its|our|his|her|these|those|such|other|more|less|very|'
    r'including|particularly|especially|may|can|should|will|would|could|'
    r'been|being|having)\s*$',
    re.IGNORECASE,
)

_LEADING_FILLER = re.compile(
    r'^(?:and|or|with|for|of|in|the|a|an|to|is|are|was|were|by|at|on|from|'
    r'due|that|which|who|also|but|not|as|if|than|into)\s+',
    re.IGNORECASE,
)


def _strip_fragment_edges(text: str) -> str:
    t = text.strip()
    for _ in range(5):
        prev = t
        t = _TRAILING_FILLER.sub('', t).strip()
        t = _LEADING_FILLER.sub('', t).strip()
        if t == prev:
            break
    return t.strip(" ,;:.()")


def _is_fragment(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if len(t) < 4:
        return True
    alpha = sum(1 for ch in t if ch.isalpha())
    if alpha < 3:
        return True
    words = t.split()
    if len(words) < 1:
        return True
    last = words[-1].lower()
    if last in ("and", "or", "with", "for", "of", "in", "the", "a", "an",
                "to", "is", "are", "was", "were", "by", "at", "on", "from",
                "due", "has", "have", "had", "that", "which", "who"):
        return True
    first = words[0].lower()
    if first in ("and", "or", "with", "for", "of", "in", "to", "by", "from",
                 "due", "at", "on", "into"):
        return True
    return False


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
    "value", "encompasses", "contraindications", "interactions",
    "clinical", "diagnostic", "further", "appropriate", "adequate",
}

_ASSESS_REJECT_RE = re.compile(
    r'(?:'
    r'Randomized\b|Multicenter\b|Prospective\b|Investigators?\b|'
    r'\bTrial\b|\bStudy\b|\bProgram\b|Chief\b|'
    r'Circ\s+Cardiovasc|JACC\s+Cardiovasc|J\s+Thorac|Eur\s+Heart\s+J|'
    r'N\s+Engl\s+J\s+Med|Am\s+J\s+Cardiol|Lancet|BMJ\b|JAMA\b|'
    r'Referenced\s+studies|Online\s+Data\s+Supplement|'
    r'Recommendations?\s+for\b|'
    r'WHEN\s+TO\b|ADVANCED\s+THERAPIES|'
    r'^[A-Z\s]{8,}$'
    r')',
    re.IGNORECASE,
)


def _clean_assessment(name: str) -> str | None:
    name = _strip_fragment_edges(name)
    if not name:
        return None
    words = name.split()
    while words and words[0].lower() in _ASSESS_STOPWORDS:
        words.pop(0)
    meaningful = [w for w in words if w.lower() not in _ASSESS_STOPWORDS]
    if len(meaningful) < 2 or len(" ".join(words)) < 6:
        return None
    cleaned = " ".join(words)
    if _is_fragment(cleaned):
        return None
    if _looks_truncated(cleaned):
        return None
    if _ASSESS_REJECT_RE.search(cleaned):
        return None
    if len(cleaned) > 80:
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
    "heart failure", "heart", "failure", "disease", "condition",
    "management", "diagnosis", "prevention", "screening",
}

_AE_REJECT_RE = re.compile(
    r'(?:'
    r'^(?:of|with|for|due to|the|in|and|or)\s|'
    r'\s(?:of|with|for|are|is)\s*$|'
    r'Management\s+of|Treatment\s+of|Diagnosis\s+of|The\s+diagnosis|'
    r'Approved\s+for|'
    r'heart failure (?:are|treatment|management|patients)|'
    r'failure due to|failure are|'
    r'^hypertension$|^diabetes$|^atrial fibrillation$|'
    r'^coronary artery disease$|^chronic kidney disease$|'
    r'^rheumatoid arthritis$|^dilated cardiomyopathy$|'
    r'^(?:chronic|stable|advanced|new.onset|incident|worsening|decompensated|acute)'
    r'\s+heart\s+failure$|'
    r'^heart\s+failure\s+(?:hospitali|trial|treatment|management|patient)|'
    r'^(?:symptomatic|asymptomatic)\s+heart\s+failure$|'
    r'^heart\s+failure$|'
    r'^(?:presence|absence)\s+of\s|'
    r'^risk\s+of\s|'
    r'^(?:hypertrophic|isch[ae]mic|non.isch[ae]mic|dilated|restrictive|'
    r'arrhythmogenic|peripartum)\s+cardiomyopathy$|'
    r'^(?:left|right)\s+ventricular\s+(?:dysfunction|cardiomyopathy)$|'
    r'^(?:ventricular|systolic|diastolic)\s+dysfunction$|'
    r'^cardiac\s+sarcoidosis$|'
    r'^(?:secondary|right\s+ventricular)\s+cardiomyopathy$|'
    r'cardiomyopathy\s+associated$|'
    r'^(?:cause|related)\s+bradycardia$|'
    r'^dysfunction\s+may\s|'
    r'^individuals?\s+with\s|'
    r'\bcardiomyopathy$|'
    r'failure\s+hospitali'
    r')',
    re.IGNORECASE,
)

_AE_DRUG_SUFFIXES = re.compile(
    r'\b\w+(?:mab|nib|zole|pril|sartan|olol|statin|dipine|zodone|pine|'
    r'xaban|gatran|parin|flozin)\b',
    re.IGNORECASE,
)


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
                    it = _strip_fragment_edges(it)
                    if not it or len(it) < 3 or len(it) > 60:
                        continue
                    if _is_fragment(it):
                        continue
                    if any(ch.isdigit() for ch in it):
                        continue
                    words = re.split(r"\s+", it)
                    if len(words) < 1 or len(words) > 4:
                        continue
                    low = " ".join(w.lower() for w in words)
                    if low in _AE_STOP_WORDS:
                        continue
                    if _AE_REJECT_RE.search(it):
                        continue
                    if _AE_DRUG_SUFFIXES.fullmatch(it.strip()):
                        continue
                    if _EVENT_SUFFIX.search(it):
                        candidates[low].append(block_idx)
                        if low not in names_by_key:
                            names_by_key[low] = Hit(page=page, source_file=src, span=it)

            # Strategy 2: multi-word terms with medical suffixes in harm-context sentences
            for word_m in re.finditer(r'\b([A-Za-z][a-z]+(?:\s+[a-z]+){1,2})\b', sent):
                term = _strip_fragment_edges(word_m.group(1).strip())
                if not term or len(term) < 6 or len(term) > 50:
                    continue
                if _is_fragment(term):
                    continue
                if _EVENT_SUFFIX.search(term):
                    low = term.lower()
                    if low not in _AE_STOP_WORDS:
                        if _AE_REJECT_RE.search(term):
                            continue
                        if _AE_DRUG_SUFFIXES.fullmatch(term.strip()):
                            continue
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
# Recommendation extraction from text paragraphs
# ═══════════════════════════════════════════════════════════════

_REC_CUE = re.compile(
    r'\b(?:'
    r'(?:is|are)\s+recommended\b|'
    r'(?:should|shall)\s+be\s+(?:considered|performed|initiated|given|used|offered|assessed)\b|'
    r'(?:we|it\s+is)\s+recommend(?:ed|s)?\b|'
    r'(?:is|are)\s+indicated\b|'
    r'(?:is|are)\s+(?:strongly\s+)?suggested\b|'
    r'(?:should|shall)\s+(?:be\s+)?(?:used|given|started|considered|administered|offered|measured|performed)\b|'
    r'(?:is|are)\s+(?:not\s+)?recommended\b|'
    r'class\s+(?:I|II|III|IIa|IIb)\b|'
    r'level\s+of\s+evidence\s+[A-C]\b'
    r')',
    re.IGNORECASE,
)

_REC_REJECT_RE = re.compile(
    r'(?:'
    r'^(?:Table|Figure|Box)\s+\d|'
    r'^Recommendations?\s+for\s|'
    r'^(?:See|Refer|Modified)\s|'
    r'Referenced\s+studies|'
    r'Online\s+Data|'
    r'^\d+\.\d+\s|'
    r'^[A-Z\s]{10,}$'
    r')',
    re.IGNORECASE,
)


def extract_recommendations(text_blocks_path: Path) -> list[Hit]:
    blocks = json.loads(text_blocks_path.read_text(encoding="utf-8"))
    seen_keys: set[str] = set()
    hits: list[Hit] = []

    for b in blocks:
        page = int(b.get("page") or 0)
        src = str(b.get("source_file") or "")
        for sent in _sentences(b.get("text") or ""):
            if not _REC_CUE.search(sent):
                continue
            sent = sent.strip()
            if len(sent) < 20 or len(sent) > 600:
                continue
            if _REC_REJECT_RE.search(sent):
                continue
            alpha = sum(1 for c in sent if c.isalpha())
            if alpha < 15:
                continue
            key = re.sub(r'\s+', ' ', sent.lower().strip())[:120]
            if key in seen_keys:
                continue
            seen_keys.add(key)
            hits.append(Hit(page=page, source_file=src, span=sent))

    return hits


# ═══════════════════════════════════════════════════════════════
# Therapy extraction from text paragraphs
# ═══════════════════════════════════════════════════════════════

_THERAPY_CUE = re.compile(
    r'\b(?:'
    r'(?:treatment|therapy|treated)\s+with\s+|'
    r'use\s+of\s+|'
    r'(?:initiat|start|continu|discontinu|switch)\w*\s+(?:on\s+|with\s+)?|'
    r'(?:first.line|second.line|third.line)\s+(?:therapy|treatment)\b|'
    r'(?:pharmacological|non.pharmacological|device|interventional)\s+therap'
    r')',
    re.IGNORECASE,
)

_THERAPY_NAME_RE = re.compile(
    r'\b('
    r'(?:ACE\s+inhibitor|ACEi|angiotensin.converting\s+enzyme\s+inhibitor)s?|'
    r'(?:ARB|angiotensin\s+receptor\s+blocker)s?|'
    r'(?:ARNi|angiotensin\s+receptor.neprilysin\s+inhibitor)s?|'
    r'(?:SGLT2\s*i(?:nhibitor)?|sodium.glucose\s+co.?transporter.?\s*2\s+inhibitor)s?|'
    r'(?:MRA|mineralocorticoid\s+receptor\s+antagonist)s?|'
    r'beta.?blockers?|beta.?adrenergic\s+(?:blockers?|antagonists?)|'
    r'(?:loop|thiazide)\s+diuretics?|diuretics?|'
    r'(?:calcium\s+channel|potassium.?channel)\s+blockers?|'
    r'(?:cardiac\s+resynchronization\s+therapy|CRT(?:.D)?)|'
    r'(?:implantable\s+cardioverter.?defibrillator|ICD)s?|'
    r'(?:mechanical\s+circulatory\s+support|MCS)|'
    r'(?:intra.?aortic\s+balloon\s+pump|IABP)|'
    r'(?:ventricular\s+assist\s+device|VAD|LVAD)s?|'
    r'heart\s+transplant(?:ation)?|'
    r'catheter\s+ablation|surgical\s+ablation|'
    r'(?:cardiac|exercise)\s+rehabilitation|'
    r'(?:anticoagula(?:tion|nt)|antiplatelet\s+therapy)|'
    r'(?:antiarrhythmic|inotropic|vasodilator|vasopressor)\s+(?:therapy|agents?|drugs?)|'
    r'(?:guideline.directed\s+medical\s+therapy|GDMT)|'
    r'(?:cardiac\s+)?pacemaker(?:\s+implantation)?|'
    r'(?:rate|rhythm)\s+control(?:\s+therapy)?|'
    r'(?:oral\s+)?anticoagulants?|'
    r'(?:percutaneous\s+coronary\s+intervention|PCI)|'
    r'(?:coronary\s+artery\s+bypass\s+graft(?:ing)?|CABG)|'
    r'(?:aortic|mitral|tricuspid)\s+valve\s+(?:repair|replacement|intervention)|'
    r'iron\s+(?:supplementation|therapy|replacement)|'
    r'oxygen\s+therapy|'
    r'statin\s+therapy|statins?|'
    r'amiodarone|digoxin|ivabradine|hydralazine|nitrate|'
    r'sacubitril.valsartan|vericiguat|omecamtiv\s+mecarbil'
    r')\b',
    re.IGNORECASE,
)

_THERAPY_REJECT = {
    "therapy", "treatment", "agent", "agents", "drug", "drugs",
    "support", "intervention", "interventions", "management",
}

_THERAPY_FRAGMENT_RE = re.compile(
    r'(?:'
    r'(?:failure|patients?|symptoms?|candidate|candidates|individuals?)\s+'
    r'(?:despite|on|for|with|receiving|after)\s|'
    r'(?:despite|according\s+to|increase\s+in|reduction\s+in|'
    r'addition\s+to|replacement\s+for|benefit\s+(?:of|to)|'
    r'randomized\s+to|effect\s+of|role\s+of|limitation|'
    r'selecting\s+patients?\s+for|already\s+on|'
    r'management\s+of\s+patients?)\s|'
    r'^(?:effect|role|benefit|limitation|increase|reduction|'
    r'addition|replacement|selecting|management)\s|'
    r'\b(?:Trial|Study|Spanish|Swedish|MADIT|PARADIGM|'
    r'Antiarrhythmics\s+versus|AATAC)\b|'
    r'\b(?:intensi\s*fi\s*cation|fi\s*cation)\b|'
    r'\bLongQT\s+syndrome\b|'
    r'^(?:chronic|guided|unhealthy)\s+\w+\s+therapy$'
    r')',
    re.IGNORECASE,
)


def extract_therapies(text_blocks_path: Path) -> list[Hit]:
    blocks = json.loads(text_blocks_path.read_text(encoding="utf-8"))
    seen_keys: dict[str, Hit] = {}
    freq: dict[str, set[int]] = defaultdict(set)

    for block_idx, b in enumerate(blocks):
        page = int(b.get("page") or 0)
        src = str(b.get("source_file") or "")
        text = b.get("text") or ""
        for m in _THERAPY_NAME_RE.finditer(text):
            raw = m.group(1).strip()
            if raw.lower() in _THERAPY_REJECT:
                continue
            if _is_fragment(raw):
                continue
            context_start = max(0, m.start() - 60)
            context = text[context_start:m.end() + 10].replace('\n', ' ')
            if _THERAPY_FRAGMENT_RE.search(context):
                continue
            key = re.sub(r'[\s\-]+', ' ', raw).lower().strip()
            freq[key].add(block_idx)
            if key not in seen_keys:
                seen_keys[key] = Hit(page=page, source_file=src, span=raw)

    return [seen_keys[k] for k, blks in freq.items() if len(blks) >= 1]


# ═══════════════════════════════════════════════════════════════
# Drug extraction from text paragraphs
# ═══════════════════════════════════════════════════════════════

_DRUG_SUFFIX_RE = re.compile(
    r'\b([A-Z][a-z]+(?:olol|pril|sartan|dipine|statin|flozin|zole|mab|nib|'
    r'tide|gliptin|parin|xaban|gatran|lukast|fenac|profin|etine|zodone|'
    r'vaptan|ciclat|mivir|semide|azine|idone|amide|opram|xetine|pine|'
    r'vudine|navir|afil|cycline))\b'
)

_DRUG_KNOWN_RE = re.compile(
    r'\b('
    r'Metoprolol|Bisoprolol|Carvedilol|Nebivolol|Atenolol|Propranolol|'
    r'Lisinopril|Enalapril|Ramipril|Captopril|Perindopril|'
    r'Valsartan|Losartan|Candesartan|Irbesartan|Telmisartan|Olmesartan|'
    r'Sacubitril|Entresto|'
    r'Empagliflozin|Dapagliflozin|Canagliflozin|Sotagliflozin|'
    r'Spironolactone|Eplerenone|Finerenone|'
    r'Amiodarone|Dronedarone|Flecainide|Propafenone|Sotalol|Lidocaine|'
    r'Digoxin|Digitoxin|'
    r'Ivabradine|Vericiguat|Hydralazine|'
    r'Furosemide|Bumetanide|Torsemide|Hydrochlorothiazide|Metolazone|'
    r'Warfarin|Apixaban|Rivaroxaban|Dabigatran|Edoxaban|'
    r'Aspirin|Clopidogrel|Ticagrelor|Prasugrel|'
    r'Atorvastatin|Rosuvastatin|Simvastatin|Pravastatin|'
    r'Amlodipine|Nifedipine|Diltiazem|Verapamil|'
    r'Dopamine|Dobutamine|Milrinone|Levosimendan|Norepinephrine|'
    r'Heparin|Enoxaparin|Fondaparinux|'
    r'Metformin|Insulin|Sitagliptin|Liraglutide|Semaglutide|'
    r'Dexamethasone|Tocilizumab|Remdesivir|'
    r'Colchicine|Allopurinol|Febuxostat|'
    r'Ferric\s+carboxymaltose|Iron\s+sucrose'
    r')\b'
)

_DRUG_STOP = {"the", "this", "that", "with", "from", "have", "class", "pepine"}

_DRUG_REJECT_RE = re.compile(
    r'(?:'
    r'^Beta\s+\d|'
    r'^Alpha\s+\d|'
    r'\breceptor\b|'
    r'\bsympathomimetic\b|'
    r'\s+[a-d]$'
    r')',
    re.IGNORECASE,
)


def extract_drugs(text_blocks_path: Path) -> list[Hit]:
    blocks = json.loads(text_blocks_path.read_text(encoding="utf-8"))
    seen_keys: dict[str, Hit] = {}
    freq: dict[str, set[int]] = defaultdict(set)

    for block_idx, b in enumerate(blocks):
        page = int(b.get("page") or 0)
        src = str(b.get("source_file") or "")
        text = b.get("text") or ""

        for pattern in (_DRUG_KNOWN_RE, _DRUG_SUFFIX_RE):
            for m in pattern.finditer(text):
                name = m.group(1).strip()
                if name.lower() in _DRUG_STOP:
                    continue
                if len(name) < 4:
                    continue
                if _DRUG_REJECT_RE.search(name):
                    continue
                name = re.sub(r'\s+[a-d]$', '', name).strip()
                key = name.lower()
                freq[key].add(block_idx)
                if key not in seen_keys:
                    seen_keys[key] = Hit(page=page, source_file=src, span=name)

    return [seen_keys[k] for k, blks in freq.items() if len(blks) >= 2]


# ═══════════════════════════════════════════════════════════════
# Cause extraction from text paragraphs
# ═══════════════════════════════════════════════════════════════

_CAUSE_CUE = re.compile(
    r'\b(?:'
    r'(?:caused?\s+by|due\s+to|secondary\s+to|attributable\s+to|'
    r'result(?:ing|s)?\s+from|consequence\s+of|'
    r'(?:common|frequent|main|major|primary|important)\s+(?:cause|etiolog|aetiology))\s+'
    r'([A-Za-z][A-Za-z\s,\-]{3,80})'
    r')',
    re.IGNORECASE,
)

_CAUSE_ENUM_CUE = re.compile(
    r'\b(?:causes?|etiolog(?:y|ies)|aetiology|etiology)\s+(?:of\s+\w+\s+\w+\s+)?'
    r'(?:include|are|:|such\s+as|including)\s*[:\s]?'
    r'([A-Za-z][A-Za-z\s,\-/()]{5,200})',
    re.IGNORECASE,
)

_CAUSE_KNOWN_RE = re.compile(
    r'\b('
    r'(?:isch[ae]mic|non.?isch[ae]mic|hypertensive|valvular|idiopathic|'
    r'familial|genetic|alcoholic|viral|peripartum|toxic|infiltrative|'
    r'restrictive|dilated|stress.induced|tachycardia.?induced|'
    r'radiation.?induced|drug.?induced|chemotherapy.?induced)\s+'
    r'(?:heart\s+(?:disease|failure)|cardiomyopathy|aetiology|etiology)|'
    r'isch[ae]mic\s+heart\s+disease|'
    r'coronary\s+artery\s+disease|'
    r'myocardial\s+infarction|'
    r'valvular\s+(?:heart\s+)?disease|'
    r'(?:severe\s+)?(?:aortic|mitral|tricuspid)\s+(?:stenosis|regurgitation)|'
    r'(?:systemic|pulmonary)\s+hypertension|'
    r'(?:viral|acute)\s+myocarditis|'
    r'(?:congenital|rheumatic)\s+heart\s+disease|'
    r'atrial\s+fibrillation|'
    r'diabetes\s+mellitus|'
    r'chronic\s+kidney\s+disease|'
    r'thyroid\s+disease|hyperthyroidism|hypothyroidism|'
    r'amyloidosis|sarcoidosis|haemochromatosis|'
    r'Chagas\s+disease|'
    r'(?:obstructive\s+)?sleep\s+apn[oe]a'
    r')\b',
    re.IGNORECASE,
)


_CAUSE_REJECT_RE = re.compile(
    r'(?:'
    r'^Table\s+\d|^Figure\s+\d|^Box\s+\d|'
    r'^Class\s+(?:I|II|III)|^Level\s+|^NYHA\s|'
    r'\b(?:Trial|Study|Registry|Register|Survey|Program|Programme|'
    r'Investigators?|Consortium|Collaboration|Committee|Cohort|'
    r'Database|Initiative|Network)\b|'
    r'\b(?:MADIT|PARADIGM|CABANA|EUROASPIRE|IMPROVE|SPIRE|FINGER|'
    r'CHAMPION|COMET|DAPA|EMPEROR|SHIFT|DEFINITE|MUSTT|SCD.?HeFT|'
    r'RALES|EMPHASIS|CHARM|Val.?HeFT|PARAGON|DELIVER|GALACTIC|'
    r'COMPASS|ASCOT|SPRINT|HOPE|ONTARGET|ATMOSPHERE|GUIDE.?IT)\b|'
    r'\brandomized\b|\bmulticent(?:re|er)\b|\bprospective\b|\bretrospective\b|'
    r'\bcontrolled study\b|\bclinical trial\b|\bmeta.analysis\b|'
    r'\bmanagement\s+of\b|\btreatment\s+of\b|\bdiagnosis\s+of\b|'
    r'\bshould\s+be\b|\bmay\s+be\b|\bcan\s+be\b|\bis\s+recommended\b|'
    r'\bgoals\s+of\s+care\b|\bprevented\b|\breversed\b|'
    r'\bhigher\s+est\b|\blower\s+complication\b|\bincreased\s+survival\b|'
    r'\bpopulation\s+growth\b|\bageing\b|\bemployment\b|\benvironment\b|'
    r'\bdietary\s+habits\b|\blifestyle\b|\btobacco\b|'
    r'\bfear\s+of\b|\bpsychosocial\b|'
    r'\b(?:mortality|survival|incidence|death\s+rate|prognosis)\b|'
    r'\b(?:fewer|decrease|increase|lower\s+rate|higher\s+rate)\s+(?:in|of)\b|'
    r'\blimited\s+expectation\b|\blater\s+arrival\b|'
    r'\boverlap\s+in\s|growing\s+number\b|'
    r'\bprevailing\s+opinion\b|\bnon.isodiametric\b|'
    r'\black\s+of\s+(?:data|methodological)\b|'
    r'\bwhereas\b|\bsome\s+cases\b|\bbetter\s+understanding\b|'
    r'\bcan\s+be\s+reversed\b|\bstrong\s+predictor\b|'
    r'\bseveral\s+factors\b|\bfact\s+that\b'
    r')',
    re.IGNORECASE,
)

_CAUSE_SINGLE_WORD_REJECT = {
    "especially", "assigned", "large", "unselected", "often",
    "question", "replacement", "prevented", "respiratory",
    "wedging", "particularly", "significant", "commonly",
    "frequently", "approximately", "generally", "recently",
    "statins", "counseling", "monitoring", "foetal",
    "developed", "including", "concerns", "specific",
}


def _looks_mid_word_truncated(text: str) -> bool:
    """Detect if text ends mid-word (e.g. 'electroca', 'cardiomyop')."""
    t = text.rstrip(' ,;:.()')
    if not t or len(t) < 6:
        return False
    last_word = t.split()[-1]
    if len(last_word) < 4:
        return False
    if last_word[-1] in 'aeiouy' and last_word[-2:] not in (
            'le', 're', 'se', 'ne', 'te', 'de', 'ee', 'ae', 'ue', 'ie', 'oe',
            'ly', 'ry', 'ny', 'ty', 'dy', 'gy', 'ky', 'py', 'my', 'sy'):
        return True
    if re.search(r'(?:rhy|electroca|cardiomyop|dysfunctio|arrhythmi|'
                 r'fibrillat|tachycardi|bradycardi|insufficien|'
                 r'hypertensio|hypotensio|haemorrhag|thromboembol)$', t, re.I):
        return True
    return False


def _is_valid_cause(item: str) -> bool:
    item = _strip_fragment_edges(item)
    if not item or len(item) < 5 or len(item) > 80:
        return False
    if _is_fragment(item):
        return False
    if _looks_mid_word_truncated(item):
        return False
    alpha = sum(1 for c in item if c.isalpha())
    if alpha < 4:
        return False
    words = item.split()
    if len(words) > 8:
        return False
    if re.search(r'\.{3,}|(?:\.\s){3,}', item):
        return False
    if re.search(r'\bfewer\s+sudden\b', item, re.I):
        return False
    if len(words) == 1 and item.lower() in _CAUSE_SINGLE_WORD_REJECT:
        return False
    if len(words) < 2 and not re.search(r'(?:emia|itis|osis|pathy|tion|ism)\b', item, re.I):
        return False
    if _CAUSE_REJECT_RE.search(item):
        return False
    if re.search(r'\b(?:that|which|who|where|when|because|although|however)\b', item, re.I):
        return False
    return True


def extract_causes(text_blocks_path: Path) -> list[Hit]:
    blocks = json.loads(text_blocks_path.read_text(encoding="utf-8"))
    seen_keys: dict[str, Hit] = {}
    freq: dict[str, set[int]] = defaultdict(set)

    for block_idx, b in enumerate(blocks):
        page = int(b.get("page") or 0)
        src = str(b.get("source_file") or "")
        text = b.get("text") or ""

        for m in _CAUSE_KNOWN_RE.finditer(text):
            name = _strip_fragment_edges(m.group(1).strip())
            if not name or _is_fragment(name) or len(name) < 5:
                continue
            key = re.sub(r'\s+', ' ', name).lower().strip()
            freq[key].add(block_idx)
            if key not in seen_keys:
                seen_keys[key] = Hit(page=page, source_file=src, span=name)

        for sent in _sentences(text):
            for pat in (_CAUSE_CUE, _CAUSE_ENUM_CUE):
                m = pat.search(sent)
                if not m:
                    continue
                raw = m.group(1).strip()
                items = re.split(r',\s*|\band\b', raw) if ',' in raw else [raw]
                for item in items:
                    item = _strip_fragment_edges(item.strip())
                    if not _is_valid_cause(item):
                        continue
                    key = re.sub(r'\s+', ' ', item).lower().strip()
                    freq[key].add(block_idx)
                    if key not in seen_keys:
                        seen_keys[key] = Hit(page=page, source_file=src, span=item)

    return [seen_keys[k] for k, blks in freq.items() if len(blks) >= 1]


# ═══════════════════════════════════════════════════════════════
# Stage extraction from text paragraphs
# ═══════════════════════════════════════════════════════════════

_STAGE_RE = re.compile(
    r'\b('
    r'(?:NYHA|New\s+York\s+Heart\s+Association)\s+'
    r'(?:functional\s+)?[Cc]lass\s+(?:I{1,3}V?|[1-4])\b(?:\s*[-–/]\s*(?:I{1,3}V?|[1-4]))?|'
    r'(?:ACC/AHA|AHA/ACC)\s+[Ss]tage\s+[A-D]\b|'
    r'[Ss]tage\s+[A-D]\s*[:]\s*[A-Z][a-zA-Z\s\-]{3,40}|'
    r'[Ss]tage\s+[A-D]\b(?:\s+(?:HF|heart\s+failure))?|'
    r'NYHA\s+(?:I{1,3}V?|[1-4])\b(?:\s*[-–/]\s*(?:I{1,3}V?|[1-4]))?|'
    r'(?:Killip|Forrester|INTERMACS)\s+(?:class|profile|stage)\s+(?:I{1,3}V?|[1-7])'
    r')\b',
    re.IGNORECASE,
)


def extract_stages(text_blocks_path: Path) -> list[Hit]:
    blocks = json.loads(text_blocks_path.read_text(encoding="utf-8"))
    seen_keys: dict[str, Hit] = {}

    for b in blocks:
        page = int(b.get("page") or 0)
        src = str(b.get("source_file") or "")
        text = b.get("text") or ""
        for m in _STAGE_RE.finditer(text):
            name = m.group(1).strip()
            key = re.sub(r'\s+', ' ', name).lower().strip()
            key = re.sub(r'new york heart association', 'nyha', key)
            if key not in seen_keys:
                seen_keys[key] = Hit(page=page, source_file=src, span=name)

    return list(seen_keys.values())


# ═══════════════════════════════════════════════════════════════
# Phenotype extraction from text paragraphs
# ═══════════════════════════════════════════════════════════════

_PHENOTYPE_RE = re.compile(
    r'\b('
    r'HFrEF|HFpEF|HFmrEF|HFimpEF|HFrecEF|'
    r'HF\s+with\s+(?:reduced|preserved|mildly\s+reduced|improved|recovered)\s+'
    r'(?:ejection\s+fraction|EF)|'
    r'heart\s+failure\s+with\s+(?:reduced|preserved|mildly\s+reduced|improved|recovered)\s+'
    r'(?:ejection\s+fraction|EF)|'
    r'(?:dilated|hypertrophic|restrictive|arrhythmogenic|isch[ae]mic|'
    r'non.?isch[ae]mic|peripartum|stress|takotsubo)\s+cardiomyopathy|'
    r'(?:left|right)\s+(?:ventricular|atrial)\s+(?:dysfunction|failure|dilation|hypertrophy)|'
    r'(?:left|right)\s+bundle\s+branch\s+block|LBBB|RBBB|'
    r'(?:systolic|diastolic)\s+(?:dysfunction|heart\s+failure)|'
    r'(?:acute|chronic|acute\s+on\s+chronic)\s+(?:decompensated\s+)?heart\s+failure|'
    r'(?:low|high)\s+(?:cardiac\s+)?output\s+(?:state|syndrome)|'
    r'cardiogenic\s+shock|'
    r'(?:cardiac|cardio[\s-]?renal|cardio[\s-]?hepatic)\s+syndrome|'
    r'(?:LVEF|ejection\s+fraction)\s*(?:<=?|>=?|<|>)\s*\d+\s*%?'
    r')\b',
    re.IGNORECASE,
)


def extract_phenotypes(text_blocks_path: Path) -> list[Hit]:
    blocks = json.loads(text_blocks_path.read_text(encoding="utf-8"))
    seen_keys: dict[str, Hit] = {}

    for b in blocks:
        page = int(b.get("page") or 0)
        src = str(b.get("source_file") or "")
        text = b.get("text") or ""
        for m in _PHENOTYPE_RE.finditer(text):
            name = m.group(1).strip()
            key = re.sub(r'\s+', ' ', name).lower().strip()
            if key not in seen_keys:
                seen_keys[key] = Hit(page=page, source_file=src, span=name)

    return list(seen_keys.values())


# ═══════════════════════════════════════════════════════════════
# Writers — same CSV output format as v1
# ═══════════════════════════════════════════════════════════════

# Module-level context used to stamp `guideline_id` on every written row,
# set by run_text_normalization() and cleared at the end. Keeps per-row
# provenance consistent with normalize_tables.py.
_CURRENT_GUIDELINE_ID: str | None = None


def _stamp_rows(fieldnames: list[str], rows: list[dict]) -> tuple[list[str], list[dict]]:
    if "guideline_id" not in fieldnames:
        return fieldnames, rows
    gid = _CURRENT_GUIDELINE_ID or ""
    stamped = []
    for r in rows:
        if r.get("guideline_id"):
            stamped.append(r)
        else:
            nr = dict(r)
            nr["guideline_id"] = gid
            stamped.append(nr)
    return fieldnames, stamped


def _append_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames, rows = _stamp_rows(fieldnames, rows)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    return len(rows)


def write_assessments(
    out_dir: Path,
    condition_id: str,
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
        ["guideline_id", "assessment_id", "assessmentName", "assessmentValue"],
        rows_a,
    )
    return n


def write_adverse_events(out_dir: Path, hits: list[Hit]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"ae_id": f"ae_{_slug(h.span)}", "adverseEventName": h.span,
             "adverseEventSeverity": ""} for h in hits]
    return _append_csv(out_dir / "S_adverse_event.csv",
                       ["guideline_id", "ae_id", "adverseEventName", "adverseEventSeverity"], rows)


def write_recommendations(out_dir: Path, hits: list[Hit]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"rec_id": f"rec_txt_{_slug(h.span)[:60]}_{i}",
             "recommendationText": h.span}
            for i, h in enumerate(hits)]
    return _append_csv(out_dir / "S_recommendation.csv",
                       ["guideline_id", "rec_id", "recommendationText"], rows)


_DEVICE_PROCEDURE_RE = re.compile(
    r'(?:CRT|ICD|pacemaker|defibrillator|ablation|transplant|'
    r'resynchroniz|cardiac\s+pacing|'
    r'VAD|LVAD|IABP|MCS|mechanical\s+circulatory|'
    r'percutaneous|PCI|CABG|bypass|valve\s+(?:repair|replacement)|'
    r'device|implant|surgery|surgical|'
    r'renal\s+replacement|dialysis|'
    r'radiation\s+therapy)',
    re.IGNORECASE,
)


def _classify_therapy_type(name: str) -> str:
    if _DEVICE_PROCEDURE_RE.search(name):
        return "device/procedure"
    if re.search(r'(?:rehabilitation|exercise|training|lifestyle|diet)', name, re.I):
        return "non-pharmacological"
    return "pharmacological"


def write_therapies(out_dir: Path, hits: list[Hit]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"therapy_id": f"therapy_txt_{_slug(h.span)}",
             "therapyType": _classify_therapy_type(h.span),
             "therapy_name": h.span}
            for h in hits]
    return _append_csv(out_dir / "S_therapy.csv",
                       ["guideline_id", "therapy_id", "therapyType", "therapy_name"], rows)


def write_drugs(out_dir: Path, hits: list[Hit]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"drug_id": f"drug_txt_{_slug(h.span)}",
             "agentName": h.span,
             "minDose": "", "maxDose": "", "duration": "",
             "drugCategory": ""}
            for h in hits]
    return _append_csv(out_dir / "S_drug.csv",
                       ["guideline_id", "drug_id", "agentName", "minDose", "maxDose",
                        "duration", "drugCategory"], rows)


def write_causes(out_dir: Path, hits: list[Hit], condition_id: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"cause_id": f"cause_{_slug(h.span)}",
             "causeName": h.span}
            for h in hits]
    return _append_csv(out_dir / "S_cause.csv",
                       ["guideline_id", "cause_id", "causeName"], rows)


def write_stages(out_dir: Path, hits: list[Hit]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for h in hits:
        name = h.span
        level_m = re.search(r'(?:class|stage|profile)\s+(I{1,3}V?|[1-7A-D])', name, re.I)
        level = level_m.group(1) if level_m else ""
        scheme = "NYHA" if "NYHA" in name.upper() or "new york" in name.lower() else \
                 "ACC/AHA" if re.search(r'ACC|AHA|Stage\s+[A-D]', name) else \
                 "Killip" if "Killip" in name else \
                 "INTERMACS" if "INTERMACS" in name else "unknown"
        rows.append({
            "stage_id": f"stage_{_slug(name)}",
            "stageScheme": scheme,
            "stageLevel": level,
            "stageName": name,
            "StageCriteriaText": "",
        })
    return _append_csv(out_dir / "S_stage.csv",
                       ["guideline_id", "stage_id", "stageScheme", "stageLevel", "stageName",
                        "StageCriteriaText"], rows)


def write_phenotypes(out_dir: Path, hits: list[Hit]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for h in hits:
        name = h.span
        code_m = re.match(r'^(HFrEF|HFpEF|HFmrEF|HFimpEF|HFrecEF|LBBB|RBBB)\b', name)
        code = code_m.group(1) if code_m else _slug(name)[:30]
        rows.append({
            "phenotype_id": f"phenotype_{_slug(name)}",
            "phenotypeCode": code,
            "phenotypeCriteria": name,
        })
    return _append_csv(out_dir / "S_phenotype.csv",
                       ["guideline_id", "phenotype_id", "phenotypeCode", "phenotypeCriteria"], rows)


def _read_guideline_id(out_dir: Path) -> str:
    """Read the guideline_id produced by run_table_normalization().

    Falls back to 'guideline_unknown' if S_guideline.csv is missing/empty.
    """
    p = out_dir / "S_guideline.csv"
    if not p.exists():
        return "guideline_unknown"
    try:
        with p.open(encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                gid = (row.get("guideline_id") or "").strip()
                if gid:
                    return gid
    except Exception:
        pass
    return "guideline_unknown"


def run_text_normalization(
    run_dir: Path,
    out_dir: Path,
    condition_id: str = "Unknown",
    min_freq: int = 2,
) -> dict:
    """
    Extracts all entity types from text blocks: assessments, adverse events,
    recommendations, therapies, drugs, causes, stages, and phenotypes.

    Args:
        min_freq: minimum block frequency for assessments and adverse events.
                  Set to 1 to match original v2 behaviour (no filter).
                  Default 2 eliminates hapax fragments.
    """
    global _CURRENT_GUIDELINE_ID
    step1 = run_dir / "step1"
    text_blocks = step1 / "text_blocks.json"
    if not text_blocks.exists():
        return {"text_blocks": str(text_blocks), "status": "missing"}

    _CURRENT_GUIDELINE_ID = _read_guideline_id(out_dir)
    try:
        assessments = extract_assessments(text_blocks, min_freq=min_freq)
        assessment_values = extract_assessment_values(text_blocks, assessments)
        adverse_events = extract_adverse_events(text_blocks, min_freq=min_freq)
        recommendations = extract_recommendations(text_blocks)
        therapies = extract_therapies(text_blocks)
        drugs = extract_drugs(text_blocks)
        causes = extract_causes(text_blocks)
        stages = extract_stages(text_blocks)
        phenotypes = extract_phenotypes(text_blocks)

        n_assess = write_assessments(
            out_dir,
            condition_id=condition_id,
            hits=assessments,
            value_map=assessment_values,
        )
        n_ae = write_adverse_events(out_dir, hits=adverse_events)
        n_rec = write_recommendations(out_dir, hits=recommendations)
        n_ther = write_therapies(out_dir, hits=therapies)
        n_drug = write_drugs(out_dir, hits=drugs)
        n_cause = write_causes(out_dir, causes, condition_id)
        n_stage = write_stages(out_dir, stages)
        n_pheno = write_phenotypes(out_dir, phenotypes)

        return {
            "text_blocks": str(text_blocks),
            "version": "v2-generic-full",
            "guideline_id": _CURRENT_GUIDELINE_ID,
            "quality_gates": {"min_freq": min_freq},
            "rows_written": {
                "S_assessment.csv": n_assess,
                "S_adverse_event.csv": n_ae,
                "S_recommendation.csv": n_rec,
                "S_therapy.csv": n_ther,
                "S_drug.csv": n_drug,
                "S_cause.csv": n_cause,
                "S_stage.csv": n_stage,
                "S_phenotype.csv": n_pheno,
            },
        }
    finally:
        _CURRENT_GUIDELINE_ID = None
