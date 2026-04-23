"""Step 2: generic table normalization to ontology-shaped entity CSVs (noise-filtered v2)."""
from __future__ import annotations
import csv, io, json, re, shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _slug(s):
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (s or "").strip()).strip("_")
    if len(s) <= 80:
        return s or "unknown"
    truncated = s[:80]
    last_sep = truncated.rfind("_")
    if last_sep > 40:
        truncated = truncated[:last_sep]
    return truncated.strip("_") or "unknown"


_TRAILING_FILLER_RE = re.compile(
    r'\s+(?:and|or|with|for|of|in|the|a|an|to|is|are|was|were|by|at|on|from|'
    r'due|has|have|had|that|which|who|also|but|not|as|if|than|into)\s*$',
    re.IGNORECASE,
)

_LEADING_FILLER_RE = re.compile(
    r'^(?:and|or|with|for|of|in|the|a|an|to|is|are|was|were|by|at|on|from|'
    r'due|that|which|who|also|but|not|as|if|than|into)\s+',
    re.IGNORECASE,
)


def _strip_fragment_edges(text: str) -> str:
    t = text.strip()
    for _ in range(5):
        prev = t
        t = _TRAILING_FILLER_RE.sub('', t).strip()
        t = _LEADING_FILLER_RE.sub('', t).strip()
        if t == prev:
            break
    return t.strip(" ,;:.()")


def _is_fragment(text: str) -> bool:
    t = (text or "").strip()
    if not t or len(t) < 4:
        return True
    alpha = sum(1 for ch in t if ch.isalpha())
    if alpha < 3:
        return True
    words = t.split()
    last = words[-1].lower()
    if last in ("and", "or", "with", "for", "of", "in", "the", "a", "an",
                "to", "is", "are", "was", "were", "by", "at", "on", "from"):
        return True
    first = words[0].lower()
    if first in ("and", "or", "with", "for", "of", "in", "to", "by", "from"):
        return True
    return False


@dataclass
class TableEntry:
    table_id: str
    page: int
    csv_path: str
    caption: str
    headers: list


def _hl(t): return [h.lower().strip() for h in t.headers]
def _cl(t): return (t.caption or "").lower()

def _headers_are_generic(hl):
    """True when Docling assigned numeric column names instead of real headers."""
    return all(re.fullmatch(r"\d+", h) for h in hl) if hl else False


def _references_dir(out_dir) -> Path:
    """Subfolder for S_reference_*.csv (secondary / passthrough tables)."""
    d = Path(out_dir) / "references"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ════════════════════════════════════════════════════════════════
# Role Detectors — generic clinical guideline table patterns
# (identical to original v2 except detect_staging_classification)
# ════════════════════════════════════════════════════════════════

def detect_recommendation(t):
    hl = _hl(t)
    if not hl: return False
    h0 = hl[0]
    # AHA format: first header is a long recommendation title
    if h0.startswith("recommendation") and len(h0) > 30: return True
    # Evidence strength columns: AHA uses COR/LOE, ESC uses Class/Level
    has_cor = any("cor" in h for h in hl)
    has_loe = any("loe" in h for h in hl)
    has_class = any(h.startswith("class") for h in hl)
    has_level = any(h.startswith("level") for h in hl)
    has_rec = any("recommendation" in h for h in hl)
    # AHA: COR + LOE columns
    if has_cor and has_loe: return True
    # ESC: Class + Level columns (e.g. "Class a", "Level b")
    if has_class and has_level: return True
    # Recommendation column + any evidence strength column
    if has_rec and (has_cor or has_loe or has_class or has_level): return True
    cap = _cl(t)
    if "recommendation" in cap and ("cor" in cap or "loe" in cap or "class" in cap or "level" in cap): return True
    # ESC single-row format: header IS the recommendation, other cols are Roman numerals (I, II, III, IV) or letters (A, B, C)
    if len(hl) >= 2 and len(h0) > 40:
        others = [h.strip() for h in hl[1:] if h.strip()]
        if others and all(re.fullmatch(r'[iIvVxX]{1,4}|[a-cA-C]', o) for o in others):
            return True
    return False

def detect_drug_dosing(t):
    hl = _hl(t); cap = _cl(t)
    has_drug = any("drug" in h or "agent" in h or "medication" in h for h in hl)
    has_dose = any("dose" in h or "daily" in h or "dosing" in h for h in hl)
    if has_drug and has_dose: return True
    # ESC format: "Starting dose" + "Target dose" without explicit drug column
    has_starting = any("starting" in h for h in hl)
    has_target = any("target" in h for h in hl)
    if has_starting and has_target and any("dose" in h for h in hl): return True
    # Caption-based detection
    if ("drug" in cap or "diuretic" in cap or "medication" in cap) and "dose" in cap: return True
    if "commonly used" in cap and ("drug" in cap or "medication" in cap): return True
    # Tables with "infusion rate" + drug-like column
    if has_drug and any("infusion" in h or "rate" in h for h in hl): return True
    return False

def detect_harmful_drug(t):
    hl = _hl(t); cap = _cl(t)
    has_drug = any("drug" in h or "therapeutic class" in h for h in hl)
    has_mech = any("mechanism" in h or "magnitude" in h for h in hl)
    has_side = any("side effect" in h or "adverse" in h for h in hl)
    if has_drug and has_mech: return True
    if "harm" in cap or "exacerbat" in cap or "may cause" in cap or "worsen" in cap: return True
    # Tables listing drugs with their side effects/adverse reactions
    if has_drug and has_side: return True
    return False

def detect_inotropic_agent(t):
    hl = _hl(t); cap = _cl(t)
    if any("inotropic" in h or "infusion" in h for h in hl) and any("dose" in h or "dosing" in h for h in hl): return True
    if "inotropic" in cap or "intravenous" in cap: return True
    return False

def detect_stage(t):
    hl = _hl(t); cap = _cl(t)
    if any("stage" in h for h in hl) and any("definition" in h or "criteria" in h for h in hl): return True
    if "stage" in cap and ("definition" in cap or "criteria" in cap): return True
    return False

def detect_staging_classification(t):
    """
    FIX vs original v2: require BOTH a structural header match AND an uppercase
    abbreviation in the caption.  Original only required the abbreviation, which
    was too loose (many tables have uppercase acronyms that are not staging systems).
    """
    hl = _hl(t); cap = _cl(t)
    has_header = any("profile" in h or "features" in h or "hemodynamics" in h for h in hl)
    has_abbrev = bool(re.search(r'\b[A-Z]{3,}(?:/[A-Z]{2,})?\b', t.caption or ""))
    return has_header and has_abbrev

def detect_phenotype(t):
    hl = _hl(t); cap = _cl(t)
    if any("phenotype" in h or "lvef" in h for h in hl): return True
    if any(h.startswith("type of") for h in hl): return True
    if "classification" in cap and "lvef" in cap: return True
    # Rescue: first data row may contain real headers when Docling used generic cols
    first = getattr(t, "_first_row_lower", None)
    if first:
        if any("lvef" in v or "phenotype" in v for v in first): return True
        if any(v.startswith("type of") for v in first): return True
    return False

def detect_cause(t):
    hl = _hl(t); cap = _cl(t)
    if any("cause" in h or "etiology" in h for h in hl):
        if "natriuretic" not in cap: return True
    if ("cause" in cap or "etiolog" in cap) and "natriuretic" not in cap: return True
    return False

def detect_comorbidity(t):
    hl = _hl(t); cap = _cl(t)
    if any("condition" in h or "prevalence" in h for h in hl):
        if "comorbidit" in cap or "co-occur" in cap or "chronic condition" in cap: return True
    if "comorbidit" in cap: return True
    return False

def detect_risk_score(t):
    hl = _hl(t); cap = _cl(t)
    if any("risk score" in h for h in hl) and any("year" in h for h in hl): return True
    if "risk score" in cap or ("risk" in cap and "predict" in cap): return True
    return False

def detect_self_care_barrier(t):
    hl = _hl(t); cap = _cl(t)
    if any("barrier" in h for h in hl) and any("screening" in h or "intervention" in h for h in hl): return True
    if "barrier" in cap or "self-care" in cap: return True
    return False

def detect_vulnerable_population(t):
    cap = _cl(t)
    return "vulnerable" in cap or "special population" in cap or "disparit" in cap

def detect_therapy_benefit(t):
    hl = _hl(t); cap = _cl(t)
    if any("nnt" in h or "evidence-based therapy" in h for h in hl): return True
    if "benefit" in cap and "evidence" in cap: return True
    return False

def detect_cardiotoxic_agent(t):
    hl = _hl(t); cap = _cl(t)
    if any("cardiac function" in h or "monitoring" in h for h in hl):
        if "cancer" in cap or "cardiotox" in cap or "cardiomyopathy" in cap: return True
    if "cardiotox" in cap: return True
    return False

def detect_pregnancy_management(t):
    hl = _hl(t); cap = _cl(t)
    if any("preconception" in h or "postpartum" in h for h in hl): return True
    return "pregnancy" in cap

def detect_genetic_factor(t):
    hl = _hl(t); cap = _cl(t)
    if any("phenotypic category" in h for h in hl): return True
    return "genetic" in cap and "cardiomyopathy" in cap

def detect_precipitating_factor(t):
    return "precipitat" in _cl(t) and "factor" in _cl(t)

def detect_natriuretic_cause(t):
    cap = _cl(t)
    return "natriuretic peptide" in cap or ("natriuretic" in cap and "elevated" in cap)

def detect_associated_guideline(t):
    hl = _hl(t); cap = _cl(t)
    if any("title" in h for h in hl) and any("organization" in h or "publication year" in h for h in hl): return True
    return "associated guideline" in cap

def detect_performance_measure(t):
    hl = _hl(t); cap = _cl(t)
    if any("measure" in h and ("no" in h or "title" in h) for h in hl): return True
    return "performance" in cap or "quality" in cap

def detect_palliative_care(t):
    cap = _cl(t)
    return "palliative" in cap or "supportive care" in cap

def detect_advanced_hf_def(t):
    cap = _cl(t)
    return "advanced" in cap and ("definition" in cap or "criteria" in cap)

def detect_clinical_indicator(t):
    return "clinical indicator" in _cl(t)

def detect_shock_criteria(t):
    hl = _hl(t); cap = _cl(t)
    if any("sbp" in h or "hypoperfusion" in h for h in hl): return True
    return "shock" in cap and ("criteria" in cap or "hemodynamic" in cap)

def detect_transitional_care(t):
    cap = _cl(t)
    return "transitional" in cap or "care plan" in cap

def detect_mcs_indication(t):
    cap = _cl(t)
    if any(k in cap for k in ["mechanical", "durable", "lvad", "support"]):
        return "indication" in cap or "contraindication" in cap
    return False

def detect_evidence_gap(t):
    cap = _cl(t)
    return "evidence gap" in cap or "future research" in cap


_ROLE_DETECTORS = [
    ("recommendation", detect_recommendation), ("drug_dosing", detect_drug_dosing),
    ("harmful_drug", detect_harmful_drug), ("inotropic_agent", detect_inotropic_agent),
    ("stage", detect_stage), ("staging_classification", detect_staging_classification),
    ("phenotype", detect_phenotype), ("cause", detect_cause),
    ("comorbidity", detect_comorbidity), ("risk_score", detect_risk_score),
    ("self_care_barrier", detect_self_care_barrier),
    ("vulnerable_population", detect_vulnerable_population),
    ("therapy_benefit", detect_therapy_benefit),
    ("cardiotoxic_agent", detect_cardiotoxic_agent),
    ("pregnancy_management", detect_pregnancy_management),
    ("genetic_factor", detect_genetic_factor),
    ("precipitating_factor", detect_precipitating_factor),
    ("natriuretic_cause", detect_natriuretic_cause),
    ("associated_guideline", detect_associated_guideline),
    ("performance_measure", detect_performance_measure),
    ("palliative_care", detect_palliative_care),
    ("advanced_hf_def", detect_advanced_hf_def),
    ("clinical_indicator", detect_clinical_indicator),
    ("shock_criteria", detect_shock_criteria),
    ("transitional_care", detect_transitional_care),
    ("mcs_indication", detect_mcs_indication),
    ("evidence_gap", detect_evidence_gap),
]

_APPENDIX_SKIP = [
    "reviewer", "committee", "abbreviation", "appendix",
    "author relationship", "disclosure", "writing group",
]

def _is_appendix_or_metadata(t):
    cap = _cl(t); hl = _hl(t)
    if any(p in cap for p in _APPENDIX_SKIP): return True
    if hl and any(p in hl[0] for p in ["reviewer", "writing committee"]): return True
    return False

def classify_table(t):
    if _is_appendix_or_metadata(t): return None
    for name, det in _ROLE_DETECTORS:
        if det(t): return name
    return None


# ════════════════════════════════════════════════════════════════
# CSV helpers
# ════════════════════════════════════════════════════════════════

def _read_csv_rows(p):
    if not p.exists(): return []
    try:
        text = p.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
        import io
        return list(csv.DictReader(io.StringIO(text)))
    except Exception:
        return []

# Module-level context used to stamp `guideline_id` on every written row.
# Set by run_table_normalization() before any normalizer writes CSVs, and
# cleared at the end of the orchestration. Keeps normalizer signatures
# unchanged while guaranteeing provenance on every emitted row.
_CURRENT_GUIDELINE_ID: str | None = None


def _stamp_rows(fieldnames, rows):
    """Inject guideline_id into rows when the schema requires it.

    Does nothing if guideline_id is not part of the target schema or if no
    current guideline id is set (e.g. for reference/unmatched CSVs).
    """
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


def _append_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames, rows = _stamp_rows(fieldnames, rows)
    wh = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if wh: w.writeheader()
        w.writerows(rows)
    return len(rows)

def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames, rows = _stamp_rows(fieldnames, rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


# ════════════════════════════════════════════════════════════════
# Generic helpers (no config)
# ════════════════════════════════════════════════════════════════

_THERAPY_TYPE_KEYWORDS = {
    "device/procedure": [
        "icd", "crt", "pacemaker", "lvad", "vad", "iabp",
        "mechanical circulatory", "implantable", "defibrillator",
        "transplant", "revascularization", "surgery", "surgical",
        "ablation", "percutaneous", "pci", "cabg", "bypass",
        "valve repair", "valve replacement", "device",
        "stent", "angioplasty", "implantation",
        "resynchroniz", "cardiac pacing", "leadless",
        "renal replacement", "dialysis", "radiation",
        "mechanical prosthes",
    ],
    "non-pharmacological": [
        "exercise", "rehabilitation", "sodium restriction",
        "self-care", "diet", "training", "lifestyle",
        "oxygen therapy", "telemonitoring", "palliative",
        "structured telephone", "airway pressure",
        "salt restriction", "fluid restriction",
    ],
}

def _classify_therapy_type(name):
    nl = name.lower()
    for ttype, keywords in _THERAPY_TYPE_KEYWORDS.items():
        if any(k in nl for k in keywords):
            return ttype
    return "pharmacological"


def _extract_scheme_from_caption(caption):
    """Extract staging/classification scheme from a table caption generically."""
    m = re.search(r'\b([A-Z]{2,}(?:\s*/\s*[A-Z]{2,})*)\s+(?:stage|classification|class)', caption, re.I)
    if m:
        return m.group(1).replace(" ", "")
    m = re.search(r'\(([A-Z]{2,}(?:/[A-Z]{2,})*)\)', caption)
    if m:
        return m.group(1)
    m = re.search(r'\b([A-Z]{3,}(?:/[A-Z]{2,})*)\b', caption)
    if m:
        return m.group(1)
    return "unknown"


def _clean_condition_name(raw: str) -> str | None:
    """Shared post-processing for extracted condition/topic names.

    Returns cleaned name or None if the result is invalid.
    """
    name = raw.strip()
    name = re.sub(r'\s*\([^)]*\)\s*', ' ', name).strip()
    name = re.split(r'\s*[:\u2014\u2013]\s+', name)[0].strip()
    name = re.sub(r',\s*\d+\s*$', '', name).strip()
    name = re.sub(r'\s*Developed\b.*', '', name, flags=re.I).strip()
    name = re.sub(r'\s*With\s+the\s+(?:Collaboration|Contribution|Special)\b.*', '', name, flags=re.I).strip()
    name = re.sub(r'\s*The\s+Task\s+Force\b.*', '', name, flags=re.I).strip()

    name = re.sub(r'^(?:patients?\s+with)\s+', '', name, flags=re.I).strip()

    _IN_PATIENTS = re.search(
        r'\b(?:and\s+exercise\s+)?in\s+patients?\s+with\s+(.{3,60})$', name, re.I)
    if _IN_PATIENTS:
        name = _IN_PATIENTS.group(1).strip()

    name = re.sub(
        r'\s+(?:prevention|management|treatment|diagnosis)\s+'
        r'(?:in\s+clinical\s+practice|in\s+the\s+community)?\s*$',
        '', name, flags=re.I).strip()

    _THERAPY_WORDS = re.compile(
        r'^(?:cardiac\s+)?(?:pacing|resynchronization|therapy|rehabilitation|'
        r'transplantation|implantation|defibrillat|ablation|interventions?)\b', re.I)
    if _THERAPY_WORDS.search(name) and not re.search(
            r'(?:disease|failure|syndrome|disorder|arrhythmia|cardiomyopath)', name, re.I):
        return None

    _PROCEDURAL_RE = re.compile(
        r'^(?:anticoagulation|pacemaker|implant|electrophysiol|catheter|'
        r'sleep\s+eval|monitoring|imaging|biomarker|screening)\b', re.I)
    if _PROCEDURAL_RE.search(name):
        return None

    name = re.sub(r'\s+', ' ', name).strip().rstrip(' ,;:')

    if re.match(r'^(?:ESC|AHA|ACC|the\s|All\s|This\s|Get\s)', name, re.I):
        return None
    if re.search(r'experts?\s+involved|development\s+of\s+these|Task\s+Force', name, re.I):
        return None
    _BOILERPLATE_RE = re.compile(
        r'^(?:Preamble|Classes\s+of|Levels?\s+of|Evidence|messages?\s+from|'
        r'always\s+access|cated\s+with|full\s+text|New\s+recommendations|'
        r'Clinical\s+Practice|Supplementary\s+Data)\b', re.I)
    if _BOILERPLATE_RE.search(name):
        return None
    if len(name) < 3 or len(name) > 80:
        return None
    return name


def _is_toc_block(text: str) -> bool:
    """Detect table-of-contents blocks (many dotted leaders or page numbers)."""
    dotted = len(re.findall(r'\.{3,}', text))
    spaced_dots = len(re.findall(r'(?:\.\s){3,}', text))
    pipes = text.count('|')
    if dotted >= 3 or spaced_dots >= 3:
        return True
    if pipes >= 30 and spaced_dots + dotted >= 1:
        return True
    return False


_DISEASE_MENTION_RE = re.compile(
    r'\b((?:(?:acute|chronic|advanced|congestive|decompensated)\s+)?'
    r'heart\s+failure(?:\s+with\s+(?:reduced|preserved|mildly\s+reduced)\s+'
    r'ejection\s+fraction)?'
    r'|cardiomyopath\w+'
    r'|(?:ventricular|atrial|supraventricular)\s+(?:arrhythmia|tachycardia|fibrillation)\w*'
    r'|(?:aortic|mitral|tricuspid)\s+(?:stenosis|regurgitation|valve\s+disease)'
    r'|coronary\s+artery\s+disease'
    r'|(?:pulmonary|systemic)\s+hypertension'
    r'|myocardial\s+infarction'
    r'|sudden\s+cardiac\s+death'
    r'|(?:dilated|hypertrophic|restrictive)\s+cardiomyopathy'
    r'|(?:peripheral|cerebrovascular)\s+(?:artery|vascular)\s+disease'
    r')\b', re.I)


def _extract_condition_identity(text_blocks_path):
    """Extract condition/topic name from document title on first pages."""
    blocks = json.loads(text_blocks_path.read_text(encoding="utf-8"))

    _TITLE_PATTERNS = [
        # "Guidelines for the management/diagnosis/treatment of X"
        re.compile(
            r'(?:guideline|recommendation)s?\s+for\s+(?:the\s+)?'
            r'(?:management|diagnosis|treatment|prevention|evaluation)\s+of\s+'
            r'([^.\n]{3,80})', re.I),
        # "Guidelines on X" (ESC style)
        re.compile(
            r'(?:guideline|recommendation)s?\s+on\s+'
            r'([^.\n]{3,80})', re.I),
        # "Guidance for ... referral of patients with X"
        re.compile(
            r'(?:guidance|statement)\s+for\s+[^.\n]{5,40}\s+(?:patients?\s+with|of)\s+'
            r'([^.\n]{3,60})', re.I),
        # "Management of X patients" or "Management of X"
        re.compile(
            r'Management\s+of\s+([^.\n]{3,60}?)(?:\s+patients|\s*[:\n])', re.I),
    ]

    for b in blocks[:10]:
        text = (b.get("text") or "").strip()
        if not text or _is_toc_block(text):
            continue
        for pat in _TITLE_PATTERNS:
            m = pat.search(text)
            if not m:
                continue
            name = _clean_condition_name(m.group(1))
            if name:
                return _slug(name), name

    # Fallback: scan early blocks for explicit condition mentions
    for b in blocks[:15]:
        text = (b.get("text") or "").strip()
        if not text or _is_toc_block(text):
            continue
        m = _DISEASE_MENTION_RE.search(text)
        if m:
            name = m.group(1).strip()
            if len(name) >= 5:
                return _slug(name), name

    return "Unknown", "Unknown Condition"


def _condition_from_guideline_title(title: str) -> tuple[str, str] | None:
    """Derive condition/topic from an already-extracted guideline title as fallback."""
    if not title or title == "Unknown Guideline":
        return None

    _DERIVE_PATS = [
        re.compile(r'(?:management|diagnosis|treatment|prevention)\s+of\s+(.+)', re.I),
        re.compile(r'[Gg]uidelines?\s+(?:on|for)\s+(.+)', re.I),
        re.compile(r'[Gg]uidance\s+for\s+.{5,30}\s+(?:patients?\s+with|of)\s+(.+)', re.I),
        re.compile(r'Management\s+of\s+(.+)', re.I),
    ]

    for pat in _DERIVE_PATS:
        m = pat.search(title)
        if not m:
            continue
        name = _clean_condition_name(m.group(1))
        if name:
            return _slug(name), name
    return None


# ════════════════════════════════════════════════════════════════
# Normalizers
# ════════════════════════════════════════════════════════════════

def _is_evidence_col(h):
    """Check if a header looks like an evidence-strength column (COR/LOE/Class/Level)."""
    hl = h.lower().strip()
    return (hl in ("cor", "loe")
            or "cor" in hl.split() or "loe" in hl.split()
            or hl.startswith("class") or hl.startswith("level"))


def _is_single_row_rec_table(raw_rows):
    """Detect ESC single-row format: header IS the recommendation, other cols are Roman/letter grades."""
    if len(raw_rows) < 1: return False
    h0 = raw_rows[0][0].strip() if raw_rows[0] else ""
    if len(h0) < 40: return False
    others = [c.strip() for c in raw_rows[0][1:] if c.strip()]
    return others and all(re.fullmatch(r'[IiVvXx]{1,4}|[A-Ca-c]', o) for o in others)


def normalize_recommendation(t, csv_path, out_dir):
    raw_rows = []
    try:
        text = csv_path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
        reader = csv.reader(io.StringIO(text))
        for row in reader:
            raw_rows.append(row)
    except Exception:
        return 0
    if not raw_rows: return 0

    # Format 1: ESC single-row — header IS the recommendation text, other cols are grades
    if _is_single_row_rec_table(raw_rows):
        out = []
        # The header row itself is a recommendation
        rt = raw_rows[0][0].strip()
        if len(rt) >= 30:
            out.append({"rec_id": f"rec_{t.table_id}_h", "recommendationText": rt})
        # Additional data rows may also be single-row recommendations
        for i, row in enumerate(raw_rows[1:]):
            rt = row[0].strip() if row else ""
            if len(rt) >= 30:
                others = [c.strip() for c in row[1:] if c.strip()]
                if others and all(re.fullmatch(r'[IiVvXx]{1,4}|[A-Ca-c]', o) for o in others):
                    out.append({"rec_id": f"rec_{t.table_id}_{i}", "recommendationText": rt})
        if out:
            return _append_csv(out_dir / "S_recommendation.csv",
                               ["guideline_id", "rec_id", "recommendationText"], out)
        return 0

    if len(raw_rows) < 2: return 0

    # Detect header row: check row 0 and row 1 for evidence-strength columns
    first_vals = [c.strip().lower() for c in raw_rows[0]]
    has_ev_r0 = any(_is_evidence_col(v) for v in raw_rows[0])

    if has_ev_r0:
        headers = [c.strip() for c in raw_rows[0]]
        data_rows = raw_rows[1:]
    else:
        if len(raw_rows) > 1:
            has_ev_r1 = any(_is_evidence_col(v) for v in raw_rows[1])
            if has_ev_r1:
                headers = [c.strip() for c in raw_rows[1]]
                data_rows = raw_rows[2:]
            else:
                headers = [c.strip() for c in raw_rows[0]]
                data_rows = raw_rows[1:]
        else:
            return 0

    # Find evidence-strength column indices
    ev_indices = set()
    for i, h in enumerate(headers):
        if _is_evidence_col(h):
            ev_indices.add(i)

    # Find recommendation text column
    rec_idx = None
    rec_candidates = [(i, h) for i, h in enumerate(headers)
                      if "recommendation" in h.lower() and i not in ev_indices]
    if rec_candidates:
        for i, h in rec_candidates:
            tail = h.rsplit(".", 1)[-1].strip().lower()
            if tail in ("recommendations", "recommendation"):
                rec_idx = i
                break
        if rec_idx is None:
            rec_idx = rec_candidates[-1][0]
    # Fallback: pick the column with the longest header text that's not an evidence column
    if rec_idx is None:
        non_ev = [(i, h) for i, h in enumerate(headers) if i not in ev_indices]
        if non_ev:
            rec_idx = max(non_ev, key=lambda x: len(x[1]))[0]
        else:
            rec_idx = 0

    _EVIDENCE_ONLY_RE = re.compile(
        r"^(?:\d+:\s*(?:no benefit|benefit|harm)|"
        r"value statement\s*:\s*\w+\s+value|"
        r"class\s+[IVX]+|level\s+[A-C]|"
        r"(?:COR|LOE)\s*[:\-]?\s*[IVX\dA-C])",
        re.IGNORECASE,
    )

    _REC_NOISE_RE = re.compile(
        r"(?:"
        r"^Table\s+\d|^Figure\s+\d|"
        r"Referenced\s+studies\s+that\s+support|"
        r"Online\s+Data\s+Supplement|"
        r"section\s+\d|"
        r"Classes?\s+of\s+recommendation|"
        r"\.{3,}|"
        r"^\s*•\s*$|"
        r"^Risk\s+factors?\s+and\s+(?:clinical\s+conditions|interventions)|"
        r"^Policy\s+interventions|"
        r"^Goals\s+for\s+Optimization|"
        r"^Management\s+of\s+(?:Anemia|Iron)"
        r")",
        re.IGNORECASE,
    )

    out = []
    for i, row in enumerate(data_rows):
        if len(row) <= rec_idx: continue
        rt = row[rec_idx].strip()
        if not rt or len(rt) < 30: continue
        if _EVIDENCE_ONLY_RE.match(rt): continue
        if _REC_NOISE_RE.search(rt): continue
        alpha = sum(1 for ch in rt if ch.isalpha())
        if alpha < 20: continue
        out.append({"rec_id": f"rec_{t.table_id}_{i}", "recommendationText": rt})
    return _append_csv(out_dir / "S_recommendation.csv",
                       ["guideline_id", "rec_id", "recommendationText"], out)


# Caption filler words that must NOT become drugCategory
_CAP_FILLER = {
    "table", "commonly", "used", "oral", "with", "that", "have", "been",
    "from", "drug", "drugs", "dose", "dosing", "initial", "target", "their",
    "these", "some", "other", "both", "often", "more", "less", "also", "each",
}

_NAME_COL_KEYWORDS = ["drug", "agent", "medication", "compound", "substance"]

_PK_NOISE_RE = re.compile(
    r"^(?:t\s*1/2|half[\-\s]*life|[A-Z],?\s*[A-Z],?\s*[A-Z]$|NR$|NA$)",
    re.IGNORECASE,
)


def _pick_name_column(flds):
    """Prefer short header matches ("Drug", "Agent") over compound descriptors."""
    scored = []
    for i, f in enumerate(flds):
        fl = f.lower()
        for kw in _NAME_COL_KEYWORDS:
            if kw in fl:
                scored.append((len(fl), i, f))
                break
    if scored:
        scored.sort()
        return scored[0][2]
    return flds[0]


def _is_subheader_row(row, name_col):
    vals = [(v or "").strip() for v in row.values()]
    non_empty = [v for v in vals if v]
    if not non_empty:
        return False
    if len(set(non_empty)) == 1:
        return True
    name_val = (row.get(name_col) or "").strip()
    other = [v for k, v in row.items() if k != name_col and (v or "").strip() and v.strip() != name_val]
    return len(other) == 0


def _is_noise_drug_name(name):
    if not name:
        return True
    if _PK_NOISE_RE.match(name):
        return True
    alpha = sum(1 for ch in name if ch.isalpha())
    if alpha < 3:
        return True
    if len(name) > 80:
        return True
    words = name.split()
    if len(words) > 8:
        return True
    if any(w in name.lower() for w in ("recommended", "guideline", "patient", "treatment of", "are recommended")):
        return True
    return False


def normalize_drug_dosing(t, csv_path, out_dir):
    rows = _read_csv_rows(csv_path)
    if not rows: return 0
    flds = list(rows[0].keys())
    drug_col = _pick_name_column(flds)
    dose_cols = [f for f in flds if "dose" in f.lower() or "daily" in f.lower()]
    dur_col = next((f for f in flds if "duration" in f.lower() or "action" in f.lower()), None)
    drug_rows = []; therapy_rows = []; cat = "general"

    cap = _cl(t)
    for w in re.findall(r'\b[a-z]{4,}\b', cap):
        if w not in _CAP_FILLER:
            cat = w
            break

    for row in rows:
        dn = (row.get(drug_col) or "").strip()
        if not dn: continue
        if _is_subheader_row(row, drug_col):
            cat = dn
            therapy_rows.append({"therapy_id": f"therapy_{_slug(dn)}",
                                 "therapyType": _classify_therapy_type(dn), "therapy_name": dn})
            continue
        if _is_noise_drug_name(dn):
            continue
        md = (row.get(dose_cols[0]) or "").strip() if dose_cols else ""
        xd = (row.get(dose_cols[1]) or "").strip() if len(dose_cols) >= 2 else ""
        dur = (row.get(dur_col) or "").strip() if dur_col else ""
        drug_rows.append({"drug_id": f"drug_{_slug(dn)}", "agentName": dn, "minDose": md,
                          "maxDose": xd, "duration": dur, "drugCategory": cat})
    n = _append_csv(out_dir / "S_drug.csv",
                    ["guideline_id", "drug_id", "agentName", "minDose", "maxDose", "duration", "drugCategory"],
                    drug_rows)
    if therapy_rows:
        _append_csv(out_dir / "S_therapy.csv",
                    ["guideline_id", "therapy_id", "therapyType", "therapy_name"], therapy_rows)
    return n


def normalize_therapy_benefit(t, csv_path, out_dir):
    rows = _read_csv_rows(csv_path)
    if not rows: return 0
    flds = list(rows[0].keys())
    tc = next((f for f in flds if "therapy" in f.lower() or "evidence" in f.lower()), flds[0])
    tr = []; rr = []
    for row in rows:
        nm = (row.get(tc) or "").strip()
        if not nm: continue
        tr.append({"therapy_id": f"therapy_{_slug(nm)}", "therapyType": _classify_therapy_type(nm),
                    "therapy_name": nm})
        rr.append({k: (v or "").strip() for k, v in row.items()})
    _append_csv(out_dir / "S_therapy.csv",
                ["guideline_id", "therapy_id", "therapyType", "therapy_name"], tr)
    return _append_csv(_references_dir(out_dir) / "S_reference_therapy_benefit.csv", flds, rr)


def normalize_harmful_drug(t, csv_path, out_dir):
    rows = _read_csv_rows(csv_path)
    if not rows: return 0
    flds = list(rows[0].keys())
    dc = next((f for f in flds if "drug" in f.lower() or "class" in f.lower()), flds[0])
    ar = []
    for row in rows:
        nm = (row.get(dc) or "").strip()
        if not nm: continue
        ar.append({"ae_id": f"ae_{_slug(nm)}", "adverseEventName": nm, "adverseEventSeverity": "harmful"})
    return _append_csv(out_dir / "S_adverse_event.csv",
                       ["guideline_id", "ae_id", "adverseEventName", "adverseEventSeverity"], ar)


def normalize_stage(t, csv_path, out_dir, condition_id="Unknown"):
    rows = _read_csv_rows(csv_path)
    if not rows: return 0
    flds = list(rows[0].keys())
    sc = next((f for f in flds if "stage" in f.lower()), flds[0])
    cc = next((f for f in flds if "definition" in f.lower() or "criteria" in f.lower()),
              flds[1] if len(flds) > 1 else flds[0])
    scheme = _extract_scheme_from_caption(t.caption or "")
    srx = re.compile(r"(?:stage|class|profile|level)\s*([A-Za-z0-9]+(?:[-\u2013][A-Za-z0-9]+)?)", re.I)
    sr = []
    for row in rows:
        st = (row.get(sc) or "").strip()
        if not st: continue
        m = srx.search(st)
        lv = m.group(1) if m else _slug(st)[:20]
        sid = f"stage_{_slug(lv)}"
        cr = (row.get(cc) or "").strip()
        sr.append({"stage_id": sid, "stageScheme": scheme, "stageLevel": lv,
                    "stageName": st, "StageCriteriaText": cr})
    n = _append_csv(out_dir / "S_stage.csv",
                    ["guideline_id", "stage_id", "stageScheme", "stageLevel", "stageName", "StageCriteriaText"], sr)
    return n


_PHENOTYPE_CODE_RE = re.compile(r'\b(HF\w{1,10}|NYHA\s*[IVX]+|EF\s*[<>=]\s*\d+%?)\b', re.I)
_PHENOTYPE_REJECT_RE = re.compile(
    r'^(?:Diuretics?|ACE|ARB|ARNI|MRA|Beta|Candesartan|Irbesartan|Losartan|'
    r'Digoxin|Ivabradine|Sacubitril|Vericiguat|Dapagliflozin|Empagliflozin|'
    r'Spironolactone|Eplerenone|Amiodarone|Sotalol|Metoprolol|Carvedilol|Bisoprolol|'
    r'Participation|Follow|Annual|Six.monthly|Monthly|Weekly|'
    r'CRT|ICD|Table|Figure|\d+)\b',
    re.I)

def normalize_phenotype(t, csv_path, out_dir, condition_id="Unknown"):
    rows = _read_csv_rows(csv_path)
    if not rows: return 0
    flds = list(rows[0].keys())

    if _headers_are_generic([f.strip() for f in flds]):
        real_hdr = {f: (rows[0].get(f) or "").strip() for f in flds}
        col_map = {f: real_hdr[f] for f in flds}
        tc = next((f for f in flds if "type" in col_map[f].lower() or "phenotype" in col_map[f].lower()), flds[0])
        cc = next((f for f in flds if "criteria" in col_map[f].lower() or "lvef" in col_map[f].lower()),
                  flds[1] if len(flds) > 1 else flds[0])
        rows = rows[1:]
    else:
        tc = next((f for f in flds if "type" in f.lower() or "phenotype" in f.lower()), flds[0])
        cc = next((f for f in flds if "criteria" in f.lower() or "lvef" in f.lower()),
                  flds[1] if len(flds) > 1 else flds[0])

    pr = []
    for row in rows:
        tt = (row.get(tc) or "").strip()
        if not tt: continue
        if _PHENOTYPE_REJECT_RE.match(tt):
            continue
        m = _PHENOTYPE_CODE_RE.search(tt)
        code = m.group(1) if m else _slug(tt)[:20]
        if _PHENOTYPE_REJECT_RE.match(code):
            continue
        pid = f"phenotype_{_slug(code)}"
        cr = (row.get(cc) or "").strip()
        pr.append({"phenotype_id": pid, "phenotypeCode": code, "phenotypeCriteria": cr})
    n = _append_csv(out_dir / "S_phenotype.csv",
                    ["guideline_id", "phenotype_id", "phenotypeCode", "phenotypeCriteria"], pr)
    return n


_CAUSE_TABLE_REJECT_RE = re.compile(
    r'(?:'
    r'\b(?:Trial|Study|Registry|Register|Investigators?|Randomized|'
    r'Multicent(?:re|er)|Prospective|Retrospective|Meta.analysis|'
    r'MADIT|PARADIGM|CABANA|EUROASPIRE|SCD.?HeFT|FINGER|CHAMPION|'
    r'COMET|DAPA|EMPEROR|SHIFT|DEFINITE|MUSTT)\b|'
    r'^Table\s+\d|^Figure\s+\d|^Class\s+(?:I|II|III)|'
    r'\bshould\s+be\b|\bmay\s+be\b|\bis\s+recommended\b|'
    r'\bmanagement\s+of\b|\btreatment\s+of\b'
    r')',
    re.IGNORECASE,
)


def _is_noise_cause(text):
    t = text.strip()
    if not t:
        return True
    if re.fullmatch(r"\d+", t):
        return True
    alpha = sum(1 for ch in t if ch.isalpha())
    if alpha < 3:
        return True
    words = t.split()
    if len(words) == 1 and len(t) < 6:
        return True
    if _CAUSE_TABLE_REJECT_RE.search(t):
        return True
    return False


def normalize_cause(t, csv_path, out_dir, condition_id="Unknown"):
    rows = _read_csv_rows(csv_path)
    if not rows: return 0
    flds = list(rows[0].keys())
    cc = next((f for f in flds if "cause" in f.lower() or "etiol" in f.lower()), flds[0])
    cr = []
    for row in rows:
        ct = (row.get(cc) or "").strip()
        if _is_noise_cause(ct):
            continue
        cid = f"cause_{_slug(ct)}"
        cr.append({"cause_id": cid, "causeName": ct})
    n = _append_csv(out_dir / "S_cause.csv",
                    ["guideline_id", "cause_id", "causeName"], cr)
    return n


def normalize_generic_reference(role, t, csv_path, out_dir):
    rows = _read_csv_rows(csv_path)
    if not rows: return 0
    flds = [k for k in rows[0].keys() if k != "source_table"]
    clean = []
    for row in rows:
        d = {k: (v or "").strip() for k, v in row.items() if k != "source_table"}
        clean.append(d)
    return _append_csv(_references_dir(out_dir) / f"S_reference_{role}.csv", flds, clean)


def copy_unmatched(t, csv_path, out_dir):
    ud = out_dir / "S_unmatched"
    ud.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        shutil.copy2(csv_path, ud / csv_path.name)
    meta = {"table_id": t.table_id, "page": t.page, "caption": t.caption, "headers": t.headers}
    (ud / f"{csv_path.stem}_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


# ════════════════════════════════════════════════════════════════
# Therapy extraction from text — with quality gates
# ════════════════════════════════════════════════════════════════

# Noun phrases ending in known therapy-class suffixes
_THERAPY_SUFFIX_PATTERN = re.compile(
    r'\b((?:[A-Za-z][\w\-]+\s+){1,4}'
    r'(?:inhibitors?|blockers?|antagonists?|agonists?|'
    r'therap(?:y|ies)|agents?|'
    r'transplant(?:ation)?|rehabilitation|training|restriction|'
    r'support|defibrillators?|pacemakers?|'
    r'resynchronization|anticoagulat\w*|diuretics?|analgesics?))\b',
    flags=re.IGNORECASE,
)

_THERAPY_FILLER = {
    "the", "a", "an", "this", "that", "no", "any", "some", "their", "our", "its",
    "and", "or", "with", "for", "of", "in", "to", "is", "are", "was", "were",
    "based", "directed", "optimal", "medical", "current", "other", "new",
}

# Reject matches that START with these words — they are meta-language, not therapy names
_THERAPY_PREFIX_REJECT = re.compile(
    r'^(?:although|despite|while|including|safety\s+of|efficacy\s+of|'
    r'sequencing|titration|uptitration|administration\s+of|shown\s+that|'
    r'switching\s+to|discussions?\s+about|clinical\s+trials?|'
    r'further|predated?|characterized?|beyond|trial|study|studies|evidence|'
    r'consider|recommended?\s+as|first\s+line|second\s+line|be\s+improved|'
    r'novel\s+strategies?|device\s+management|clinical\s+outcomes?|'
    r'Referenced\s+studies|Recommendations?\s+for|'
    r'Comparison\s+of|Prospective\s+|Multicenter\s+|'
    r'failure\s+Referenced|disease\s+Referenced|Fraction\s+Referenced|'
    r'Amyloidosis\s+Referenced|'
    r'Testing\s+Referenced|'
    r'EVALUATION\s+OF|Association\s+Heart|'
    r'COmparison\s+of|data\s+to\s+support|'
    r'accordance\s+with|intolerant\s+to|patients\s+with|angioedema\s+with|'
    r'use\s+of|whom\s+|tolerate\s+an|indicates\s+|'
    r'Update\s+on|Guidelines\s+for|Prevention\s+and|'
    r'optimize\s+|on\s+optimal|'
    r'Pharmacological\s+therapy$|^handling\s+and)\b',
    flags=re.IGNORECASE,
)

_THERAPY_NOISE_RE = re.compile(
    r'(?:'
    r'^therapy$|^therapies$|^treatment$|^agents?$|^support$|'
    r'receptor\s+blockers?$|channel\s+inhibitor$|'
    r'suitable\s+for|'
    r'need\s+intensified'
    r')',
    re.IGNORECASE,
)


def extract_therapies_from_text(text_blocks_path, min_freq: int = 2):
    """
    Discover therapy class names from structural text cues.

    FIX vs original v2: two quality gates applied after regex matching:
      1. Prefix rejection: discard matches starting with meta-language words
         (e.g. "while optimizing …", "safety of …", "Although some therapies").
      2. Frequency threshold (min_freq=2): only emit a therapy if the same
         normalised name appears in at least `min_freq` distinct text blocks.
         This eliminates sentence fragments that happen to match once.
    """
    blocks = json.loads(text_blocks_path.read_text(encoding="utf-8"))

    # Pass 1: collect candidate → list of block indices
    candidates: dict[str, list[int]] = {}
    names_by_key: dict[str, str] = {}

    for block_idx, b in enumerate(blocks):
        text = b.get("text") or ""
        for m in _THERAPY_SUFFIX_PATTERN.finditer(text):
            name = m.group(1).strip()

            # Strip leading filler words
            words = name.split()
            while words and words[0].lower() in _THERAPY_FILLER:
                words.pop(0)
            if not words:
                continue
            name = " ".join(words)

            # Reject meta-language prefixes
            if _THERAPY_PREFIX_REJECT.match(name):
                continue

            name = _strip_fragment_edges(name)

            if not name or len(name) < 5 or len(name) > 80:
                continue

            if _is_fragment(name):
                continue

            if _THERAPY_NOISE_RE.search(name):
                continue

            key = name.lower()
            candidates.setdefault(key, []).append(block_idx)
            names_by_key.setdefault(key, name)

    # Pass 2: apply frequency threshold
    found = {}
    for key, block_indices in candidates.items():
        if len(set(block_indices)) >= min_freq:
            name = names_by_key[key]
            found[key] = {
                "therapy_id": f"therapy_{_slug(name)}",
                "therapyType": _classify_therapy_type(name),
                "therapy_name": name,
            }

    return list(found.values())


# ════════════════════════════════════════════════════════════════
# Guideline metadata — generic org detection
# ════════════════════════════════════════════════════════════════

def _clean_single_line(text):
    t = re.sub(r"\s+", " ", text).strip()
    t = re.sub(r'COVID\s*1\s*9', 'COVID-19', t)
    t = re.sub(r'(\w)(guidelines)', r'\1 \2', t, flags=re.I)
    t = re.sub(r'(fi|fl)\s+(cation|eld|rst|nal|brill|nding)', r'\1\2', t)
    return t


def extract_guideline_metadata(text_blocks_path):
    """Extract guideline metadata from first pages using multiple strategies."""
    blocks = json.loads(text_blocks_path.read_text(encoding="utf-8"))
    title = ""; source = ""; date = ""

    _TITLE_PATS = [
        # "YYYY ORG/ORG Guidelines for/on ..."
        re.compile(r'(\d{4}\s+(?:[A-Z]{2,}(?:/[A-Z]{2,})*)\s+[^\n]{10,}(?:guideline|recommendation)[^\n]*)', re.I),
        # "YYYY ESC Guidelines on/for ..."
        re.compile(r'(\d{4}\s+ESC\s+Guidelines?\s+(?:on|for)\s+[^\n]{10,})', re.I),
        # "Guideline for Management/Diagnosis/Treatment of ..."
        re.compile(r'((?:Guideline|Recommendation)s?\s+for\s+(?:the\s+)?(?:Management|Diagnosis|Treatment|Prevention)\s+of\s+[^\n]{5,80})', re.I),
        # "Guidance for ... referral ..."
        re.compile(r'(Guidance\s+for\s+[^\n]{10,80})', re.I),
        # "Management of X patients with Y"
        re.compile(r'(Management\s+of\s+[^\n]{5,80})', re.I),
    ]

    _SOURCE_PAT = re.compile(
        r'((?:[A-Z]{2,}(?:/[A-Z]{2,})+)|'
        r'European\s+Society\s+of\s+Cardiology|'
        r'American\s+(?:Heart\s+Association|College\s+of\s+Cardiology)|'
        r'Heart\s+Failure\s+(?:Association|Society))',
        re.I)

    for b in blocks[:10]:
        text = (b.get("text") or "").strip()
        if not text:
            continue
        if not title:
            for pat in _TITLE_PATS:
                m = pat.search(text)
                if m:
                    candidate = _clean_single_line(m.group(1))
                    candidate = re.split(r'\s*[:\u2014\u2013]\s+(?:A Report|Developed|With|Authors)', candidate)[0].strip()
                    if len(candidate) >= 15 and len(candidate) <= 200:
                        title = candidate
                        break
        if not source:
            m = _SOURCE_PAT.search(text)
            if m:
                source = _clean_single_line(m.group(1))[:120]
                if source in ("OF", "THE", "BY", "AND"):
                    source = ""
        if not date:
            m = re.search(r'\b(20\d{2})\b', text)
            if m:
                date = m.group(1).strip()
        if title and source and date:
            break

    return {"guideline_id": f"guideline_{_slug(title[:80])}" if title else "guideline_unknown",
            "guidelineTitle": title or "Unknown Guideline", "guidelineSource": source or "",
            "guidelineDate": date or ""}


# ════════════════════════════════════════════════════════════════
# Header scaffolding
# ════════════════════════════════════════════════════════════════

_HEADER_SCHEMA = {
    "S_guideline.csv": ["guideline_id", "guidelineTitle", "guidelineSource", "guidelineDate"],
    "S_recommendation.csv": ["guideline_id", "rec_id", "recommendationText"],
    "S_drug.csv": ["guideline_id", "drug_id", "agentName", "minDose", "maxDose", "duration", "drugCategory"],
    "S_therapy.csv": ["guideline_id", "therapy_id", "therapyType", "therapy_name"],
    "S_condition.csv": ["guideline_id", "condition_id", "conditionName"],
    "S_stage.csv": ["guideline_id", "stage_id", "stageScheme", "stageLevel", "stageName", "StageCriteriaText"],
    "S_cause.csv": ["guideline_id", "cause_id", "causeName"],
    "S_phenotype.csv": ["guideline_id", "phenotype_id", "phenotypeCode", "phenotypeCriteria"],
    "S_assessment.csv": ["guideline_id", "assessment_id", "assessmentName", "assessmentValue"],
    "S_adverse_event.csv": ["guideline_id", "ae_id", "adverseEventName", "adverseEventSeverity"],
    "S_annotation_concept.csv": ["guideline_id", "concept_id", "conceptName"],
}

_LEGACY_RELATION_FILES = (
    "S_condition_stage.csv", "S_condition_cause.csv", "S_condition_phenotype.csv",
    "S_condition_assessment.csv", "S_contains.csv", "S_treats.csv", "S_drug_adverse_event.csv",
)

def _reset_schema_csvs(out_dir):
    out_dir = Path(out_dir)
    for fn in list(_HEADER_SCHEMA) + list(_LEGACY_RELATION_FILES):
        p = out_dir / fn
        if p.exists():
            p.unlink()
    for p in out_dir.glob("S_reference_*.csv"):
        if p.exists():
            p.unlink()
    refd = out_dir / "references"
    if refd.is_dir():
        for p in refd.glob("S_reference_*.csv"):
            p.unlink()

def scaffold_headers(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for fn, flds in _HEADER_SCHEMA.items():
        p = out_dir / fn
        if not p.exists():
            with p.open("w", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=flds).writeheader()


# ════════════════════════════════════════════════════════════════
# Path resolution
# ════════════════════════════════════════════════════════════════

def resolve_csv_path(t, run_dir):
    s = t.csv_path.replace("\\", "/")
    rel = Path(s)
    fname = rel.name
    candidates = [
        Path(s),                                    # absolute path as-is
        run_dir.parent / rel,                       # relative to parent of run_dir
        run_dir / s,                                # relative to run_dir
        run_dir / "step1" / "tables" / fname,       # standard layout
        run_dir / "step1" / fname,                  # flat layout
    ]
    for c in candidates:
        if c.exists():
            return c
    return run_dir / "step1" / "tables" / fname


# ════════════════════════════════════════════════════════════════
# Main orchestrator
# ════════════════════════════════════════════════════════════════

_SPECIFIC_NORMALIZERS = {
    "recommendation": normalize_recommendation, "drug_dosing": normalize_drug_dosing,
    "harmful_drug": normalize_harmful_drug, "stage": normalize_stage,
    "phenotype": normalize_phenotype, "cause": normalize_cause,
    "therapy_benefit": normalize_therapy_benefit,
}

_GENERIC_ROLES = {
    "comorbidity", "risk_score", "self_care_barrier", "vulnerable_population",
    "cardiotoxic_agent", "pregnancy_management", "genetic_factor", "precipitating_factor",
    "natriuretic_cause", "associated_guideline", "performance_measure", "palliative_care",
    "advanced_hf_def", "clinical_indicator", "shock_criteria", "transitional_care",
    "mcs_indication", "evidence_gap", "staging_classification", "inotropic_agent",
}


def run_table_normalization(run_dir, out_dir, therapy_min_freq: int = 2):
    """
    Fully generic orchestrator — all metadata extracted from the document.

    Args:
        run_dir: pipeline run directory (must contain step1/table_index.json)
        out_dir: directory to write ontology-shaped CSVs
        therapy_min_freq: minimum number of distinct text blocks a therapy phrase
            must appear in to be included.  Default 2 eliminates hapax fragments.
    """
    global _CURRENT_GUIDELINE_ID
    run_dir = Path(run_dir); out_dir = Path(out_dir)
    step1 = run_dir / "step1"
    index_path = step1 / "table_index.json"
    text_blocks_path = step1 / "text_blocks.json"

    if not index_path.exists():
        return {"status": "error", "message": f"table_index.json not found at {index_path}"}

    condition_id, condition_name = "Unknown", "Unknown Condition"
    if text_blocks_path.exists():
        condition_id, condition_name = _extract_condition_identity(text_blocks_path)

    # Extract guideline metadata up front so every row written below carries
    # the correct guideline_id as provenance (prevents entity-ID collisions
    # across guidelines in the final merged KG).
    gm = {"guideline_id": "guideline_unknown", "guidelineTitle": "",
          "guidelineSource": "", "guidelineDate": ""}
    if text_blocks_path.exists():
        gm = extract_guideline_metadata(text_blocks_path)
    _CURRENT_GUIDELINE_ID = gm.get("guideline_id") or "guideline_unknown"

    try:
        _reset_schema_csvs(out_dir)
        scaffold_headers(out_dir)

        raw = json.loads(index_path.read_text(encoding="utf-8"))
        tables = [TableEntry(table_id=t["table_id"], page=int(t.get("page") or 0),
                             csv_path=t.get("csv_path", ""), caption=t.get("caption", ""),
                             headers=t.get("headers", [])) for t in raw]

        classified = {}; unmatched = []; skipped = 0
        deferred_generic = []  # tables with generic headers for rescue pass

        for t in tables:
            role = classify_table(t)
            if role is None:
                if _is_appendix_or_metadata(t):
                    skipped += 1; continue
                if _headers_are_generic(_hl(t)):
                    deferred_generic.append(t)
                    continue
                csv_path = resolve_csv_path(t, run_dir)
                copy_unmatched(t, csv_path, out_dir)
                unmatched.append({"table_id": t.table_id, "page": t.page,
                                  "caption": t.caption, "headers": t.headers})
                continue
            classified.setdefault(role, []).append(t.table_id)
            csv_path = resolve_csv_path(t, run_dir)
            if not csv_path.exists(): continue
            if role in _SPECIFIC_NORMALIZERS:
                if role in ["stage", "phenotype", "cause"]:
                    _SPECIFIC_NORMALIZERS[role](t, csv_path, out_dir, condition_id)
                else:
                    _SPECIFIC_NORMALIZERS[role](t, csv_path, out_dir)
            elif role in _GENERIC_ROLES:
                normalize_generic_reference(role, t, csv_path, out_dir)

        # Rescue pass: re-classify tables whose headers were generic numerics
        # by reading the first data row as surrogate headers.
        for t in deferred_generic:
            csv_path = resolve_csv_path(t, run_dir)
            if csv_path.exists():
                rows = _read_csv_rows(csv_path)
                if rows:
                    first_vals = [str(v).lower().strip() for v in rows[0].values()]
                    t._first_row_lower = first_vals
                    t.headers = [str(v).strip() for v in rows[0].values()]
            role = classify_table(t)
            if role is not None:
                classified.setdefault(role, []).append(t.table_id)
                if not csv_path.exists(): continue
                if role in _SPECIFIC_NORMALIZERS:
                    if role in ["stage", "phenotype", "cause"]:
                        _SPECIFIC_NORMALIZERS[role](t, csv_path, out_dir, condition_id)
                    else:
                        _SPECIFIC_NORMALIZERS[role](t, csv_path, out_dir)
                elif role in _GENERIC_ROLES:
                    normalize_generic_reference(role, t, csv_path, out_dir)
            else:
                copy_unmatched(t, csv_path, out_dir)
                unmatched.append({"table_id": t.table_id, "page": t.page,
                                  "caption": t.caption, "headers": t.headers})

        _write_csv(out_dir / "S_guideline.csv",
                   ["guideline_id", "guidelineTitle", "guidelineSource", "guidelineDate"], [gm])

        if condition_id == "Unknown" or condition_name == "Unknown Condition":
            derived = _condition_from_guideline_title(gm.get("guidelineTitle", ""))
            if derived:
                condition_id, condition_name = derived

        _write_csv(out_dir / "S_condition.csv", ["guideline_id", "condition_id", "conditionName"],
                   [{"condition_id": condition_id, "conditionName": condition_name}])

        # Therapies from text (with frequency + prefix gates)
        if text_blocks_path.exists():
            tt = extract_therapies_from_text(text_blocks_path, min_freq=therapy_min_freq)
            ep = out_dir / "S_therapy.csv"
            eids = set()
            if ep.exists():
                for row in _read_csv_rows(ep):
                    eids.add(row.get("therapy_id", ""))
            nt = [x for x in tt if x["therapy_id"] not in eids]
            if nt:
                _append_csv(ep,
                            ["guideline_id", "therapy_id", "therapyType", "therapy_name"], nt)

        tc = sum(len(v) for v in classified.values())
        report = {
            "run_dir": str(run_dir), "out_dir": str(out_dir),
            "version": "v2-generic-fixed",
            "condition_extracted": {"condition_id": condition_id, "condition_name": condition_name},
            "guideline_id": _CURRENT_GUIDELINE_ID,
            "quality_gates": {"therapy_min_freq": therapy_min_freq},
            "total_tables": len(tables), "classified": classified,
            "unmatched": unmatched, "skipped_metadata": skipped,
            "summary": {"total": len(tables), "classified": tc,
                         "unmatched": len(unmatched), "skipped": skipped,
                         "coverage_pct": round(100 * tc / max(len(tables), 1), 1)},
        }
        (out_dir / "normalization_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report
    finally:
        _CURRENT_GUIDELINE_ID = None
