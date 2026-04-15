"""
normalize_tables_v2.py — Fully generic table normalization (no guideline_config.json).

Step 2 focuses exclusively on **entity extraction**.  All relation / link tables
(S_treats, S_contains, S_drug_adverse_event, S_disease_*, etc.) are generated
by Step 4 (LLM-based relation extraction) which uses textual context for accuracy.

Noise-controlled version of v2: same generic approach, but every text-extraction
path applies quality gates (frequency threshold, prefix rejection, length limits)
so the output matches the cleanliness of v1 without hardcoding disease terms.
"""
from __future__ import annotations
import csv, json, re, shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _slug(s):
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (s or "").strip())
    return s.strip("_")[:80] or "unknown"


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
    if h0.startswith("recommendation") and len(h0) > 30: return True
    has_cor = any("cor" in h for h in hl)
    has_loe = any("loe" in h for h in hl)
    has_rec = any("recommendation" in h for h in hl)
    if has_cor and has_loe: return True
    if has_rec and (has_cor or has_loe): return True
    cap = _cl(t)
    if "recommendation" in cap and ("cor" in cap or "loe" in cap): return True
    return False

def detect_drug_dosing(t):
    hl = _hl(t); cap = _cl(t)
    has_drug = any("drug" in h for h in hl)
    has_dose = any("dose" in h or "daily" in h for h in hl)
    if has_drug and has_dose: return True
    if ("drug" in cap or "diuretic" in cap) and "dose" in cap: return True
    if "commonly used" in cap and "drug" in cap: return True
    return False

def detect_harmful_drug(t):
    hl = _hl(t); cap = _cl(t)
    has_drug = any("drug" in h or "therapeutic class" in h for h in hl)
    has_mech = any("mechanism" in h or "magnitude" in h for h in hl)
    if has_drug and has_mech: return True
    if "harm" in cap or "exacerbat" in cap or "may cause" in cap or "worsen" in cap: return True
    return False

def detect_inotropic_agent(t):
    hl = _hl(t); cap = _cl(t)
    if any("inotropic" in h or "infusion" in h for h in hl) and any("dose" in h for h in hl): return True
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
    with p.open("r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))

def _append_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    wh = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if wh: w.writeheader()
        w.writerows(rows)
    return len(rows)

def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


# ════════════════════════════════════════════════════════════════
# Generic helpers (no config)
# ════════════════════════════════════════════════════════════════

_THERAPY_TYPE_KEYWORDS = {
    "device": ["icd", "crt", "pacemaker", "lvad", "mechanical circulatory", "implantable", "defibrillator"],
    "procedural": ["transplant", "revascularization", "surgery", "ablation"],
    "lifestyle": ["exercise", "rehabilitation", "sodium restriction", "self-care", "diet", "training"],
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


def _extract_disease_identity(text_blocks_path):
    """Extract disease name and ID from the document title."""
    blocks = json.loads(text_blocks_path.read_text(encoding="utf-8"))
    for b in blocks[:15]:
        text = (b.get("text") or "").strip()
        if not text:
            continue
        m = re.search(
            r'(?:guideline|recommendation)s?\s+for\s+(?:the\s+)?'
            r'(?:management|diagnosis|treatment|prevention|evaluation)\s+of\s+'
            r'([^.\n]{3,60})',
            text, re.I)
        if m:
            name = m.group(1).strip().rstrip(" ,;:")
            name = re.sub(r'\s*\([^)]*\)\s*', ' ', name).strip()
            # Truncate at subtitle separators (e.g. "Heart Failure: A Report...")
            name = re.split(r'\s*[:\u2014\u2013]\s+', name)[0].strip()
            if len(name) >= 3:
                return _slug(name), name
        m = re.search(r'(\b[A-Z][a-zA-Z\s]{3,40})\s+(?:Guideline|Recommendation)', text)
        if m:
            name = m.group(1).strip()
            if len(name) >= 3:
                return _slug(name), name
    return "Unknown", "Unknown Disease"


# ════════════════════════════════════════════════════════════════
# Normalizers
# ════════════════════════════════════════════════════════════════

def normalize_recommendation(t, csv_path, out_dir):
    raw_rows = []
    with csv_path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            raw_rows.append(row)
    if len(raw_rows) < 2: return 0

    first_vals = [c.strip().lower() for c in raw_rows[0]]
    has_cor_header = any("cor" in v for v in first_vals)
    has_loe_header = any("loe" in v for v in first_vals)

    if has_cor_header and has_loe_header:
        headers = [c.strip() for c in raw_rows[0]]
        data_rows = raw_rows[1:]
    else:
        if len(raw_rows) > 1:
            second_vals = [c.strip().lower() for c in raw_rows[1]]
            if any("cor" in v for v in second_vals):
                headers = [c.strip() for c in raw_rows[1]]
                data_rows = raw_rows[2:]
            else:
                headers = [c.strip() for c in raw_rows[0]]
                data_rows = raw_rows[1:]
        else:
            return 0

    cor_idx = None; loe_idx = None; rec_idx = None
    for i, h in enumerate(headers):
        hl = h.lower()
        if "cor" == hl or "cor" in hl.split(): cor_idx = i
        elif "loe" == hl or "loe" in hl.split(): loe_idx = i
    rec_candidates = [(i, h) for i, h in enumerate(headers)
                      if "recommendation" in h.lower() and i not in (cor_idx, loe_idx)]
    if rec_candidates:
        for i, h in rec_candidates:
            tail = h.rsplit(".", 1)[-1].strip().lower()
            if tail in ("recommendations", "recommendation"):
                rec_idx = i
                break
        if rec_idx is None:
            rec_idx = rec_candidates[-1][0]
    if rec_idx is None:
        rec_idx = len(headers) - 1

    _EVIDENCE_ONLY_RE = re.compile(
        r"^(?:\d+:\s*(?:no benefit|benefit|harm)|"
        r"value statement\s*:\s*\w+\s+value|"
        r"class\s+[IVX]+|level\s+[A-C]|"
        r"(?:COR|LOE)\s*[:\-]?\s*[IVX\dA-C])",
        re.IGNORECASE,
    )

    out = []
    for i, row in enumerate(data_rows):
        if len(row) <= rec_idx: continue
        rt = row[rec_idx].strip()
        if not rt or len(rt) < 30: continue
        if _EVIDENCE_ONLY_RE.match(rt): continue
        out.append({"rec_id": f"rec_{t.table_id}_{i}", "recommendationText": rt})
    return _append_csv(out_dir / "S_recommendation.csv",
                       ["rec_id", "recommendationText"], out)


# Caption filler words that must NOT become drugCategory
_CAP_FILLER = {
    "table", "commonly", "used", "oral", "with", "that", "have", "been",
    "from", "drug", "drugs", "dose", "dosing", "initial", "target", "their",
    "these", "some", "other", "both", "often", "more", "less", "also", "each",
}

_NAME_COL_KEYWORDS = ["drug", "agent", "medication", "compound"]

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
    return alpha < 3


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
                    ["drug_id", "agentName", "minDose", "maxDose", "duration", "drugCategory"],
                    drug_rows)
    if therapy_rows:
        _append_csv(out_dir / "S_therapy.csv", ["therapy_id", "therapyType", "therapy_name"], therapy_rows)
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
    _append_csv(out_dir / "S_therapy.csv", ["therapy_id", "therapyType", "therapy_name"], tr)
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
                       ["ae_id", "adverseEventName", "adverseEventSeverity"], ar)


def normalize_stage(t, csv_path, out_dir, disease_id="Unknown"):
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
                    ["stage_id", "stageScheme", "stageLevel", "stageName", "StageCriteriaText"], sr)
    return n


def normalize_phenotype(t, csv_path, out_dir, disease_id="Unknown"):
    rows = _read_csv_rows(csv_path)
    if not rows: return 0
    flds = list(rows[0].keys())

    # When Docling used generic numeric column names, the first data row
    # contains the real headers — use those for column detection and skip it.
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

    hrx = re.compile(r'\b([A-Z][A-Za-z0-9]{1,15})\b')
    pr = []
    for row in rows:
        tt = (row.get(tc) or "").strip()
        if not tt: continue
        m = hrx.search(tt)
        code = m.group(1) if m else _slug(tt)[:20]
        pid = f"phenotype_{_slug(code)}"
        cr = (row.get(cc) or "").strip()
        pr.append({"phenotype_id": pid, "phenotypeCode": code, "phenotypeCriteria": cr})
    n = _append_csv(out_dir / "S_phenotype.csv",
                    ["phenotype_id", "phenotypeCode", "phenotypeCriteria"], pr)
    return n


def _is_noise_cause(text):
    t = text.strip()
    if not t:
        return True
    if re.fullmatch(r"\d+", t):
        return True
    alpha = sum(1 for ch in t if ch.isalpha())
    return alpha < 3


def normalize_cause(t, csv_path, out_dir, disease_id="Unknown"):
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
    n = _append_csv(out_dir / "S_cause.csv", ["cause_id", "causeName"], cr)
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
    r'novel\s+strategies?|device\s+management|clinical\s+outcomes?)\b',
    flags=re.IGNORECASE,
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

            # Basic length gate
            if len(name) < 5 or len(name) > 80:
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
    return re.sub(r"\s+", " ", text).strip()


def extract_guideline_metadata(text_blocks_path):
    """Extract guideline metadata using generic patterns (no org list)."""
    blocks = json.loads(text_blocks_path.read_text(encoding="utf-8"))
    title = ""; source = ""; date = ""

    for b in blocks[:15]:
        text = (b.get("text") or "").strip()
        if not text: continue
        if not title:
            m = re.search(
                r'(\d{4}\s+[^\n]{10,}(?:guideline|recommendation)[^\n]*)',
                text, re.I)
            if m: title = _clean_single_line(m.group(1))
        if not source:
            m = re.search(
                r'((?:[A-Z]{2,}(?:/[A-Z]{2,})*|'
                r'[A-Z][a-z]+\s+(?:Association|Society|College|Foundation|Institute))'
                r'(?:\s+(?:clinical\s+practice|joint)\s+\w+)?)',
                text)
            if m: source = _clean_single_line(m.group(1))[:200]
        if not date:
            m = re.search(
                r'((?:January|February|March|April|May|June|July|August|'
                r'September|October|November|December)\s+\d{1,2},?\s+\d{4}|\b(20\d{2})\b)', text)
            if m: date = m.group(1).strip()
        if title and source and date: break

    return {"guideline_id": f"guideline_{_slug(title[:80])}" if title else "guideline_unknown",
            "guidelineTitle": title or "Unknown Guideline", "guidelineSource": source or "",
            "guidelineDate": date or ""}


# ════════════════════════════════════════════════════════════════
# Header scaffolding
# ════════════════════════════════════════════════════════════════

_HEADER_SCHEMA = {
    "S_guideline.csv": ["guideline_id", "guidelineTitle", "guidelineSource", "guidelineDate"],
    "S_recommendation.csv": ["rec_id", "recommendationText"],
    "S_drug.csv": ["drug_id", "agentName", "minDose", "maxDose", "duration", "drugCategory"],
    "S_therapy.csv": ["therapy_id", "therapyType", "therapy_name"],
    "S_disease.csv": ["disease_id", "diseaseName"],
    "S_stage.csv": ["stage_id", "stageScheme", "stageLevel", "stageName", "StageCriteriaText"],
    "S_cause.csv": ["cause_id", "causeName"],
    "S_phenotype.csv": ["phenotype_id", "phenotypeCode", "phenotypeCriteria"],
    "S_assessment.csv": ["assessment_id", "assessmentName", "assessmentValue"],
    "S_adverse_event.csv": ["ae_id", "adverseEventName", "adverseEventSeverity"],
    "S_annotation_concept.csv": ["concept_id", "conceptName"],
}

_LEGACY_RELATION_FILES = (
    "S_disease_stage.csv", "S_disease_cause.csv", "S_disease_phenotype.csv",
    "S_disease_assessment.csv", "S_contains.csv", "S_treats.csv", "S_drug_adverse_event.csv",
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
    for c in [run_dir.parent / rel, run_dir / s, run_dir / "step1" / "tables" / rel.name]:
        if c.exists(): return c
    return run_dir / s


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
    run_dir = Path(run_dir); out_dir = Path(out_dir)
    step1 = run_dir / "step1"
    index_path = step1 / "table_index.json"
    text_blocks_path = step1 / "text_blocks.json"

    if not index_path.exists():
        return {"status": "error", "message": f"table_index.json not found at {index_path}"}

    disease_id, disease_name = "Unknown", "Unknown Disease"
    if text_blocks_path.exists():
        disease_id, disease_name = _extract_disease_identity(text_blocks_path)

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
                _SPECIFIC_NORMALIZERS[role](t, csv_path, out_dir, disease_id)
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
        role = classify_table(t)
        if role is not None:
            classified.setdefault(role, []).append(t.table_id)
            if not csv_path.exists(): continue
            if role in _SPECIFIC_NORMALIZERS:
                if role in ["stage", "phenotype", "cause"]:
                    _SPECIFIC_NORMALIZERS[role](t, csv_path, out_dir, disease_id)
                else:
                    _SPECIFIC_NORMALIZERS[role](t, csv_path, out_dir)
            elif role in _GENERIC_ROLES:
                normalize_generic_reference(role, t, csv_path, out_dir)
        else:
            copy_unmatched(t, csv_path, out_dir)
            unmatched.append({"table_id": t.table_id, "page": t.page,
                              "caption": t.caption, "headers": t.headers})

    # Post-processing
    _write_csv(out_dir / "S_disease.csv", ["disease_id", "diseaseName"],
               [{"disease_id": disease_id, "diseaseName": disease_name}])

    gm = {"guideline_id": "guideline_unknown", "guidelineTitle": "",
          "guidelineSource": "", "guidelineDate": ""}
    if text_blocks_path.exists():
        gm = extract_guideline_metadata(text_blocks_path)
    _write_csv(out_dir / "S_guideline.csv",
               ["guideline_id", "guidelineTitle", "guidelineSource", "guidelineDate"], [gm])

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
            _append_csv(ep, ["therapy_id", "therapyType", "therapy_name"], nt)

    tc = sum(len(v) for v in classified.values())
    report = {
        "run_dir": str(run_dir), "out_dir": str(out_dir),
        "version": "v2-generic-fixed",
        "disease_extracted": {"disease_id": disease_id, "disease_name": disease_name},
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
