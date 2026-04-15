"""
LLM-based relation extraction from guideline text (Step 4).

Reads ``text_blocks.json`` (Step 1) and entity CSVs (Step 2), finds passages
mentioning 2+ known entities, sends them to a local Llama model, and outputs
relation CSVs using the **existing** ontology schema.

Output CSVs (all written to ``<run>/step4/``):
  Structural (deterministic from entity CSVs):
    S_contains.csv, S_disease_phenotype.csv, S_disease_stage.csv
  LLM-derived (from text passages):
    S_treats.csv, S_recommendation.csv, S_drug_adverse_event.csv, S_disease_assessment.csv, S_disease_cause.csv
"""
from __future__ import annotations

import csv
import importlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


# ── Entity index ────────────────────────────────────────────────────────────

_ENTITY_SOURCES = [
    ("S_drug.csv",           "drug_id",       "agentName",        "Drug"),
    ("S_therapy.csv",        "therapy_id",    "therapy_name",     "Therapy"),
    ("S_phenotype.csv",      "phenotype_id",  "phenotypeCode",    "Phenotype"),
    ("S_assessment.csv",     "assessment_id", "assessmentName",   "Assessment"),
    ("S_cause.csv",          "cause_id",      "causeName",        "Cause"),
    ("S_stage.csv",          "stage_id",      "stageName",        "Stage"),
    ("S_adverse_event.csv",  "ae_id",         "adverseEventName", "AdverseEvent"),
    ("S_disease.csv",        "disease_id",    "diseaseName",      "Disease"),
    ("S_guideline.csv",      "guideline_id",  "guidelineTitle",   "Guideline"),
    ("S_recommendation.csv", "rec_id",        "recommendationText", "Recommendation"),
]


@dataclass
class Entity:
    entity_type: str
    entity_id: str
    name: str


def load_entity_index(step2_dir: Path) -> dict[str, Entity]:
    """Build ``{lowercase_name: Entity}`` from Step 2 CSVs."""
    index: dict[str, Entity] = {}
    for fname, id_col, name_col, etype in _ENTITY_SOURCES:
        p = step2_dir / fname
        if not p.exists():
            continue
        with p.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                name = (row.get(name_col) or "").strip()
                eid = (row.get(id_col) or "").strip()
                if name and eid:
                    index[name.lower()] = Entity(etype, eid, name)
    return index


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ── Passage filtering ───────────────────────────────────────────────────────

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')


def _split_sentences(text: str) -> list[str]:
    """Split a page-level text block into rough sentence groups."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    sents: list[str] = []
    for para in paras:
        for s in _SENT_SPLIT.split(para):
            s = s.strip()
            if len(s) > 30:
                sents.append(s)
    return sents


def _is_noisy_header_or_keyword_sentence(sent: str) -> bool:
    """Drop structurally noisy lines that spuriously co-mention entities."""
    s = sent.strip()
    if not s:
        return True
    # Typical delimiter-heavy table header rows.
    if s.count("|") >= 4:
        return True
    # Very short all-caps lines are usually section headers/boilerplate.
    alphas = [ch for ch in s if ch.isalpha()]
    if len(alphas) >= 10:
        upper_ratio = sum(1 for ch in alphas if ch.isupper()) / len(alphas)
        if upper_ratio > 0.75 and len(s.split()) <= 10:
            return True
    return False


@dataclass
class Passage:
    text: str
    page: int
    entities_found: list[tuple[str, Entity]]


def find_relation_passages(
    text_blocks: list[dict],
    entity_index: dict[str, Entity],
    min_entities: int = 2,
) -> list[Passage]:
    """Find text passages mentioning at least ``min_entities`` known entities."""
    skip_types = {"Guideline", "Recommendation"}
    entity_names_sorted = sorted(
        (n for n, e in entity_index.items() if e.entity_type not in skip_types),
        key=len, reverse=True,
    )
    patterns = {
        name: re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE)
        for name in entity_names_sorted
        if len(name) >= 3
    }

    passages: list[Passage] = []
    for block in text_blocks:
        page = block.get("page", 0)
        text = block.get("text", "")
        if not text:
            continue
        for sent in _split_sentences(text):
            if _is_noisy_header_or_keyword_sentence(sent):
                continue
            found: list[tuple[str, Entity]] = []
            seen_ids: set[str] = set()
            sent_lower = sent.lower()
            for name, pat in patterns.items():
                ent = entity_index[name]
                if ent.entity_id in seen_ids:
                    continue
                if pat.search(sent_lower):
                    found.append((name, ent))
                    seen_ids.add(ent.entity_id)
            if len(found) >= min_entities:
                passages.append(Passage(text=sent[:1500], page=page,
                                        entities_found=found))
    return passages


# ── LLM prompt ──────────────────────────────────────────────────────────────

_ALLOWED_RELATIONS = [
    "treats",
    "hasAdverseEvent",
    "evaluatedBy",
    "hasCause",
]

RELATION_PROMPT = """\
You are a biomedical knowledge extraction system. Given a clinical text passage \
and a list of known entities, extract factual relations between them.

ALLOWED RELATION TYPES (use ONLY these):
- treats(Drug/Therapy, Disease): a drug or therapy is used to treat the disease
- hasAdverseEvent(Drug, AdverseEvent): a drug causes an adverse event
- evaluatedBy(Disease, Assessment): the disease is evaluated using an assessment
- hasCause(Disease, Cause): a cause/etiology of the disease

KNOWN ENTITIES in this passage:
{entities_block}

TEXT PASSAGE:
"{passage_text}"

Extract all factual relations present in the text. Return ONLY a JSON array.
Each element: {{"subject": "<entity name>", "subject_type": "<type>", \
"relation": "<relation>", "object": "<entity name>", "object_type": "<type>"}}

If no relations are found, return an empty array: []

JSON array:"""


def _build_entities_block(entities: list[tuple[str, Entity]]) -> str:
    lines = []
    for name, ent in entities:
        lines.append(f"- {ent.name} (type: {ent.entity_type}, id: {ent.entity_id})")
    return "\n".join(lines)


def _build_prompt(passage: Passage) -> str:
    return RELATION_PROMPT.format(
        entities_block=_build_entities_block(passage.entities_found),
        passage_text=passage.text[:1200],
    )


# ── LLM extraction ─────────────────────────────────────────────────────────

@dataclass
class ExtractedRelation:
    subject: str
    subject_type: str
    relation: str
    obj: str
    object_type: str
    page: int
    source_passage: str = ""


def _hf_llm_module():
    """Resolve HF helper module across repo and Colab layouts."""
    for module_name in ("pipeline.step3._hf_llm", "hf_llm"):
        try:
            return importlib.import_module(module_name)
        except Exception:
            continue
    raise ModuleNotFoundError(
        "Could not import HF helper module. Expected one of: "
        "'pipeline.step3._hf_llm' or 'hf_llm'."
    )


def _parse_llm_relations(raw: str, passage: Passage) -> list[ExtractedRelation]:
    """Parse LLM JSON output into ExtractedRelation objects."""
    items = _hf_llm_module().parse_json_array(raw)
    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rel = (item.get("relation") or "").strip()
        if rel not in _ALLOWED_RELATIONS:
            continue
        results.append(ExtractedRelation(
            subject=(item.get("subject") or "").strip(),
            subject_type=(item.get("subject_type") or "").strip(),
            relation=rel,
            obj=(item.get("object") or "").strip(),
            object_type=(item.get("object_type") or "").strip(),
            page=passage.page,
            source_passage=passage.text,
        ))
    return results


def extract_relations_llm(
    passages: list[Passage],
    entity_index: dict[str, Entity],
    hf_model: str,
    hf_token: str,
    batch_size: int = 4,
    max_new_tokens: int = 512,
) -> list[ExtractedRelation]:
    """Send passages to Llama in batches, extract relations."""
    hf_llm = _hf_llm_module()
    check_hf_local_backend = hf_llm.check_hf_local_backend
    format_exception = hf_llm.format_exception
    hf_local_generate = hf_llm.hf_local_generate
    hf_local_generate_batch = hf_llm.hf_local_generate_batch
    probe_hf_local_backend = hf_llm.probe_hf_local_backend

    err = check_hf_local_backend()
    if err:
        raise RuntimeError(f"Local HF stack unavailable: {err}")
    probe_err = probe_hf_local_backend(hf_model, hf_token)
    if probe_err:
        raise RuntimeError(f"Model probe failed: {probe_err}")

    prompts = [_build_prompt(p) for p in passages]
    all_relations: list[ExtractedRelation] = []
    total = len(passages)
    bs = max(1, int(batch_size))

    print(f"[Step 4] Extracting relations from {total} passages (batch_size={bs})")

    for start in range(0, total, bs):
        chunk_passages = passages[start:start + bs]
        chunk_prompts = prompts[start:start + bs]
        t0 = time.time()

        batch_ok: list[str] | None
        try:
            batch_ok = hf_local_generate_batch(
                chunk_prompts, hf_model, hf_token,
                max_new_tokens=max_new_tokens,
            )
            if len(batch_ok) != len(chunk_passages):
                raise ValueError(f"batch returned {len(batch_ok)}, expected {len(chunk_passages)}")
        except Exception as exc:
            print(f"[Step 4]   Batch failed ({format_exception(exc)}), falling back to one-by-one")
            batch_ok = None

        if batch_ok is not None:
            for passage, raw in zip(chunk_passages, batch_ok):
                rels = _parse_llm_relations(raw, passage)
                all_relations.extend(rels)
        else:
            for passage, prompt in zip(chunk_passages, chunk_prompts):
                try:
                    raw = hf_local_generate(
                        prompt, hf_model, hf_token,
                        max_new_tokens=max_new_tokens,
                    )
                    rels = _parse_llm_relations(raw, passage)
                    all_relations.extend(rels)
                except Exception as exc:
                    print(f"[Step 4]   LLM error page {passage.page}: {format_exception(exc)}")

        done = min(start + len(chunk_passages), total)
        elapsed = time.time() - t0
        print(f"[Step 4]   Progress: {done}/{total} passages ({elapsed:.1f}s)")

    return all_relations


# ── Resolve extracted names to entity IDs ───────────────────────────────────

def _resolve_entity(name: str, expected_type: str,
                    entity_index: dict[str, Entity]) -> Entity | None:
    key = name.lower().strip()
    ent = entity_index.get(key)
    if ent:
        return ent
    for k, e in entity_index.items():
        if expected_type and e.entity_type.lower() != expected_type.lower():
            continue
        if k in key or key in k:
            return e
    return None


# ── Write output CSVs ──────────────────────────────────────────────────────

def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def _slug(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', text).strip('_')[:60]


# ── Structural (deterministic) relation generators ─────────────────────────

def _generate_contains(step2_dir: Path) -> list[dict]:
    """S_contains: every recommendation belongs to the guideline about the disease."""
    guidelines = _read_csv_rows(step2_dir / "S_guideline.csv")
    diseases = _read_csv_rows(step2_dir / "S_disease.csv")
    recs = _read_csv_rows(step2_dir / "S_recommendation.csv")

    guideline_id = guidelines[0].get("guideline_id", "Unknown") if guidelines else "Unknown"
    disease_id = diseases[0].get("disease_id", "Unknown") if diseases else "Unknown"

    rows = []
    if recs:
        for r in recs:
            rid = r.get("rec_id", "")
            if rid:
                rows.append({
                    "contains_id": f"contains_{rid}",
                    "guideline_id": guideline_id,
                    "disease_id": disease_id,
                    "rec_id": rid,
                })
    if not rows:
        rows.append({
            "contains_id": f"contains_{guideline_id}_{disease_id}",
            "guideline_id": guideline_id,
            "disease_id": disease_id,
            "rec_id": "",
        })
    return rows


def _generate_disease_phenotype(step2_dir: Path) -> list[dict]:
    """S_disease_phenotype: each phenotype is a phenotype of the disease."""
    diseases = _read_csv_rows(step2_dir / "S_disease.csv")
    phenotypes = _read_csv_rows(step2_dir / "S_phenotype.csv")
    disease_id = diseases[0].get("disease_id", "Unknown") if diseases else "Unknown"

    rows = []
    seen: set[str] = set()
    for p in phenotypes:
        pid = p.get("phenotype_id", "")
        if pid and pid not in seen:
            seen.add(pid)
            rows.append({"disease_id": disease_id, "phenotype_id": pid})
    return rows


def _generate_disease_stage(step2_dir: Path) -> list[dict]:
    """S_disease_stage: each stage is a stage of the disease."""
    diseases = _read_csv_rows(step2_dir / "S_disease.csv")
    stages = _read_csv_rows(step2_dir / "S_stage.csv")
    disease_id = diseases[0].get("disease_id", "Unknown") if diseases else "Unknown"

    rows = []
    seen: set[str] = set()
    for s in stages:
        sid = s.get("stage_id", "")
        if sid and sid not in seen:
            seen.add(sid)
            rows.append({"disease_id": disease_id, "stage_id": sid})
    return rows


def _recommendation_ids_by_page(step2_dir: Path) -> dict[int, list[str]]:
    """Index recommendation IDs by source page inferred from rec_id format."""
    recs = _read_csv_rows(step2_dir / "S_recommendation.csv")
    page_map: dict[int, list[str]] = {}
    for row in recs:
        rid = (row.get("rec_id") or "").strip()
        if not rid:
            continue
        m = re.search(r"_p(\d+)_", rid)
        if not m:
            continue
        page = int(m.group(1))
        page_map.setdefault(page, []).append(rid)
    return page_map


# ── Post-processing quality filters ────────────────────────────────────────

def _tokenize_entity_name(name: str) -> list[str]:
    toks = [t for t in re.findall(r"[a-z0-9]+", name.lower()) if len(t) >= 3]
    return toks


def _entity_mentioned_with_min_overlap(name: str, text: str, min_hits: int = 1) -> bool:
    low = text.lower()
    if name.lower() in low:
        return True
    toks = set(_tokenize_entity_name(name))
    if not toks:
        return False
    hits = sum(1 for t in toks if t in low)
    return hits >= min_hits


def _has_valid_drug_ae_context(rel: ExtractedRelation, drug_name: str, ae_name: str) -> bool:
    """Generic validation: require local co-mention in at least one sentence."""
    passage = (rel.source_passage or "").strip()
    if not passage:
        return False
    if not _entity_mentioned_with_min_overlap(drug_name, passage, min_hits=1):
        return False
    if not _entity_mentioned_with_min_overlap(ae_name, passage, min_hits=1):
        return False
    for sent in _split_sentences(passage):
        if _entity_mentioned_with_min_overlap(drug_name, sent, min_hits=1) and \
           _entity_mentioned_with_min_overlap(ae_name, sent, min_hits=1):
            return True
    return False


# ── LLM-derived relation writer ────────────────────────────────────────────

def write_relation_csvs(
    relations: list[ExtractedRelation],
    entity_index: dict[str, Entity],
    out_dir: Path,
    step2_dir: Path,
) -> dict[str, int]:
    """Write all relation CSVs: 3 structural + 5 LLM-derived."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {}

    # ── Structural relations ──
    stats["S_contains"] = _write_csv(
        out_dir / "S_contains.csv",
        ["contains_id", "guideline_id", "disease_id", "rec_id"],
        _generate_contains(step2_dir))
    stats["S_disease_phenotype"] = _write_csv(
        out_dir / "S_disease_phenotype.csv",
        ["disease_id", "phenotype_id"],
        _generate_disease_phenotype(step2_dir))
    stats["S_disease_stage"] = _write_csv(
        out_dir / "S_disease_stage.csv",
        ["disease_id", "stage_id"],
        _generate_disease_stage(step2_dir))

    # ── LLM-derived relations ──
    treats_rows: list[dict] = []
    recommends_rows: list[dict] = []
    drug_ae_rows: list[dict] = []
    disease_assess_rows: list[dict] = []
    disease_cause_rows: list[dict] = []
    seen_treats: set[tuple[str, str, str]] = set()
    seen_recommends: set[tuple[str, str, str]] = set()
    seen_drug_ae: set[tuple[str, str]] = set()
    seen_assess: set[tuple[str, str]] = set()
    seen_cause: set[tuple[str, str]] = set()
    rec_ids_by_page = _recommendation_ids_by_page(step2_dir)

    disease_entities = [e for e in entity_index.values() if e.entity_type == "Disease"]
    default_disease_id = disease_entities[0].entity_id if disease_entities else "Unknown"

    for rel in relations:
        subj = _resolve_entity(rel.subject, rel.subject_type, entity_index)
        obj = _resolve_entity(rel.obj, rel.object_type, entity_index)
        if not subj or not obj:
            continue

        if rel.relation == "treats":
            therapy_ent = subj if subj.entity_type in ("Therapy", "Drug") else obj
            disease_ent = obj if obj.entity_type == "Disease" else (
                subj if subj.entity_type == "Disease" else None)
            disease_id = disease_ent.entity_id if disease_ent else default_disease_id

            drug_id = therapy_ent.entity_id if therapy_ent.entity_type == "Drug" else ""
            therapy_id = therapy_ent.entity_id if therapy_ent.entity_type == "Therapy" else ""

            if not drug_id and not therapy_id:
                continue

            key = (disease_id, therapy_id or drug_id, drug_id)
            if key in seen_treats:
                continue
            seen_treats.add(key)
            tid = f"treats_step4_{therapy_ent.entity_id}"
            treats_rows.append({
                "treats_id": tid,
                "disease_id": disease_id,
                "therapy_id": therapy_id,
                "drug_id": drug_id,
            })
            # Recommendation-centric link (new): bind treatment to rec(s) on same page.
            for rec_id in rec_ids_by_page.get(rel.page, []):
                rkey = (rec_id, therapy_id, drug_id)
                if rkey in seen_recommends:
                    continue
                seen_recommends.add(rkey)
                recommends_rows.append({
                    "rec_id": rec_id,
                    "disease_id": disease_id,
                    "therapy_id": therapy_id,
                    "drug_id": drug_id,
                })

        elif rel.relation == "hasAdverseEvent":
            drug_ent = subj if subj.entity_type == "Drug" else (
                obj if obj.entity_type == "Drug" else None)
            ae_ent = obj if obj.entity_type == "AdverseEvent" else (
                subj if subj.entity_type == "AdverseEvent" else None)
            if not drug_ent or not ae_ent:
                continue
            if not _has_valid_drug_ae_context(rel, drug_ent.name, ae_ent.name):
                continue
            key = (drug_ent.entity_id, ae_ent.entity_id)
            if key in seen_drug_ae:
                continue
            seen_drug_ae.add(key)
            drug_ae_rows.append({"drug_id": drug_ent.entity_id, "ae_id": ae_ent.entity_id})

        elif rel.relation == "evaluatedBy":
            disease_ent = subj if subj.entity_type == "Disease" else (
                obj if obj.entity_type == "Disease" else None)
            assess_ent = obj if obj.entity_type == "Assessment" else (
                subj if subj.entity_type == "Assessment" else None)
            if not assess_ent:
                continue
            d_id = disease_ent.entity_id if disease_ent else default_disease_id
            key = (d_id, assess_ent.entity_id)
            if key in seen_assess:
                continue
            seen_assess.add(key)
            disease_assess_rows.append({"disease_id": d_id, "assessment_id": assess_ent.entity_id})

        elif rel.relation == "hasCause":
            disease_ent = subj if subj.entity_type == "Disease" else (
                obj if obj.entity_type == "Disease" else None)
            cause_ent = obj if obj.entity_type == "Cause" else (
                subj if subj.entity_type == "Cause" else None)
            if not cause_ent:
                continue
            d_id = disease_ent.entity_id if disease_ent else default_disease_id
            key = (d_id, cause_ent.entity_id)
            if key in seen_cause:
                continue
            seen_cause.add(key)
            disease_cause_rows.append({"disease_id": d_id, "cause_id": cause_ent.entity_id})

    stats["S_treats"] = _write_csv(
        out_dir / "S_treats.csv",
        ["treats_id", "disease_id", "therapy_id", "drug_id"],
        treats_rows)
    stats["S_recommendation"] = _write_csv(
        out_dir / "S_recommendation.csv",
        ["rec_id", "disease_id", "therapy_id", "drug_id"],
        recommends_rows)
    stats["S_drug_adverse_event"] = _write_csv(
        out_dir / "S_drug_adverse_event.csv",
        ["drug_id", "ae_id"],
        drug_ae_rows)
    stats["S_disease_assessment"] = _write_csv(
        out_dir / "S_disease_assessment.csv",
        ["disease_id", "assessment_id"],
        disease_assess_rows)
    stats["S_disease_cause"] = _write_csv(
        out_dir / "S_disease_cause.csv",
        ["disease_id", "cause_id"],
        disease_cause_rows)

    return stats


# ── Public orchestrator ─────────────────────────────────────────────────────

def run_relation_extraction(
    run_dir: str | Path,
    step2_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    hf_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    hf_token: str | None = None,
    batch_size: int = 4,
) -> dict:
    """Full Step 4 pipeline: load entities, find passages, LLM extract, write CSVs."""
    import os

    run_dir = Path(run_dir)
    step2 = Path(step2_dir) if step2_dir else run_dir / "step2"
    target = Path(out_dir) if out_dir else run_dir / "step4"
    target.mkdir(parents=True, exist_ok=True)
    token = (hf_token or os.environ.get("HF_TOKEN") or "").strip() or None

    if not token:
        raise ValueError("HF_TOKEN required for Llama. Set env var or pass hf_token=...")

    text_blocks_path = run_dir / "step1" / "text_blocks.json"
    if not text_blocks_path.exists():
        return {"status": "error", "message": f"text_blocks.json not found: {text_blocks_path}"}

    # 1. Load entity index
    t0 = time.time()
    print(f"[Step 4] Loading entity index from {step2}")
    entity_index = load_entity_index(step2)
    type_counts: dict[str, int] = {}
    for ent in entity_index.values():
        type_counts[ent.entity_type] = type_counts.get(ent.entity_type, 0) + 1
    print(f"[Step 4]   {len(entity_index)} entities: {type_counts}")

    # 2. Load text blocks and find passages
    print(f"[Step 4] Scanning text_blocks.json for relation-rich passages...")
    text_blocks = json.loads(text_blocks_path.read_text(encoding="utf-8"))
    passages = find_relation_passages(text_blocks, entity_index, min_entities=2)
    print(f"[Step 4]   {len(passages)} passages with 2+ entities (from {len(text_blocks)} pages)")

    if not passages:
        print("[Step 4] No relation-rich passages found. Writing structural + empty LLM CSVs.")
        stats = write_relation_csvs([], entity_index, target, step2)
        report = {"status": "ok", "passages": 0, "relations_extracted": 0, "csv_stats": stats}
        _save_json(report, target / "relation_report.json")
        return report

    # 3. LLM extraction
    relations = extract_relations_llm(
        passages, entity_index, hf_model, token,
        batch_size=batch_size,
    )
    print(f"[Step 4] Extracted {len(relations)} raw relations")

    # 4. Write CSVs (structural + LLM-derived)
    print(f"[Step 4] Writing relation CSVs to {target}")
    stats = write_relation_csvs(relations, entity_index, target, step2)
    elapsed = time.time() - t0

    for csv_name, count in stats.items():
        print(f"[Step 4]   {csv_name}: {count} rows")

    report = {
        "status": "ok",
        "step2_dir": str(step2.resolve()),
        "out_dir": str(target.resolve()),
        "model": hf_model,
        "total_pages": len(text_blocks),
        "passages_with_entities": len(passages),
        "raw_relations_extracted": len(relations),
        "csv_stats": stats,
        "elapsed_seconds": round(elapsed, 1),
    }
    _save_json(report, target / "relation_report.json")
    print(f"[Step 4] Done in {elapsed:.0f}s. Report: {target / 'relation_report.json'}")
    return report


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
