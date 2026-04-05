"""
Step 3b — Text Path  [OPEN QUESTION — versioned]
─────────────────────────────────────────────────────────────────────────────
Input  : outputs/step1/resolved_entities.json   (from Step 1e)
         outputs/step1/text_blocks.json          (from Step 1a)
         OntologyIndex                            (from Step 2)
Output : outputs/step3/text_assertions_<version>.json
         outputs/step3/text_mappings_<version>.ttl   (RML)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Invariant (both versions):
  No raw LLM output enters the KG directly.
  All text assertions → structured JSON records → RML → engine → RDF.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VERSION v1  [SYMBOLIC — ontology-driven]
  LLM role : none (CUI resolution already done in Step 1e).
  Triples  : derived from entity co-occurrence + dynamic ontology domain/range
             matching.  Nothing is hardcoded from the ontology — works with any
             OWL ontology.

VERSION v2  [NEUROSYMBOLIC — symbolic base + LLM augmentation]
  Base     : starts with all v1 symbolic assertions.
  LLM adds : for text blocks with linked entities but few symbolic hits,
             the LLM proposes additional SPO candidates (using only properties
             relevant to the entities present, not a fixed list).
  Gate     : every LLM candidate is validated against ontology domain/range
             before being accepted.  Rejects are logged.

Use run_text_path(version="v1"|"v2") to switch.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from step2_load_ontology import OntologyIndex, PropertyInfo
from utils import log, save_json

try:
    from rapidfuzz import fuzz as _rfuzz

    def _sim(a: str, b: str) -> float:
        return _rfuzz.ratio(a.lower().strip(), b.lower().strip()) / 100.0
except ImportError:

    def _sim(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


# ── Shared RML header ──────────────────────────────────────────────────────

_RML_HEADER = """\
@prefix rr:  <http://www.w3.org/ns/r2rml#> .
@prefix rml: <http://semweb.mmlab.be/ns/rml#> .
@prefix ql:  <http://semweb.mmlab.be/ns/ql#> .
@prefix ex:  <http://digistructmed.org/ontology/> .

"""

# ═══════════════════════════════════════════════════════════════════════════
#  ONTOLOGY-DRIVEN INFRASTRUCTURE  (shared by v1 and v2)
#  Nothing below is specific to any particular ontology.
# ═══════════════════════════════════════════════════════════════════════════

_CLASS_MATCH_THRESHOLD = 0.60


def build_ancestors_map(ontology: OntologyIndex) -> dict[str, set[str]]:
    """
    class URI → set of ancestor URIs (including self).
    Follows rdfs:subClassOf chains transitively, skipping blank nodes.
    """
    parents: dict[str, list[str]] = {}
    for uri, info in ontology.classes.items():
        parents[uri] = [
            p for p in info.get("subClassOf", [])
            if p in ontology.classes
        ]

    cache: dict[str, set[str]] = {}

    def _anc(uri: str, visited: set[str] | None = None) -> set[str]:
        if uri in cache:
            return cache[uri]
        if visited is None:
            visited = set()
        if uri in visited:
            return set()
        visited.add(uri)
        result = {uri}
        for p in parents.get(uri, []):
            result |= _anc(p, visited)
        cache[uri] = result
        return result

    for uri in ontology.classes:
        _anc(uri)
    return cache


def match_entity_to_classes(
    entity_text: str,
    ontology: OntologyIndex,
    *,
    ner_type: str = "",
    threshold: float = _CLASS_MATCH_THRESHOLD,
) -> list[tuple[str, float]]:
    """
    Match an entity to ontology classes using label similarity, substring
    containment, and token overlap with class comments.

    Uses *ner_type* as a soft secondary signal (compared against class
    labels and comments) — but never hardcoded to any specific NER model
    or ontology.

    Returns [(class_uri, score), …] sorted descending.
    """
    ent_lower = entity_text.lower().strip()
    if not ent_lower:
        return []

    ent_tokens = set(re.findall(r"[a-z]{3,}", ent_lower))
    ner_tokens = set(re.findall(r"[a-z]{3,}", ner_type.lower().replace("_", " ")))

    scored: list[tuple[str, float]] = []

    for uri, info in ontology.classes.items():
        label = (info.get("label") or "").lower().strip()
        comment = (info.get("comment") or "").lower()
        if not label:
            continue

        # 1. Fuzzy similarity: entity text vs class label
        score = _sim(ent_lower, label)

        # 2. Exact substring containment (strong signal — but require
        #    the shorter string to be at least 5 chars to avoid "as" matching
        #    "ischemic heart disease" via "dise-as-e")
        if ent_lower == label:
            score = max(score, 1.0)
        elif len(ent_lower) >= 5 and (ent_lower in label or label in ent_lower):
            score = max(score, 0.88)

        # 3. Token overlap with class label (require at least 1 shared token)
        label_tokens = set(re.findall(r"[a-z]{3,}", label))
        shared = ent_tokens & label_tokens
        if label_tokens and ent_tokens and shared:
            overlap = len(shared) / max(len(ent_tokens), len(label_tokens))
            score = max(score, overlap * 0.75)

        # 4. Token overlap with class comment (weaker signal).
        #    Disabled for single-word entities to prevent "cardiac" matching
        #    InfiltrativeDisease via its comment containing "cardiac".
        if ent_tokens and comment and len(ent_tokens) >= 2:
            comment_tokens = set(re.findall(r"[a-z]{3,}", comment))
            c_overlap = len(ent_tokens & comment_tokens) / max(len(ent_tokens), 1)
            if c_overlap >= 0.5:
                score = max(score, c_overlap * 0.55)

        # 5. NER-type boost: if NER type tokens appear in class label or comment
        if ner_tokens:
            if ner_tokens & label_tokens:
                score = max(score, 0.52)
            elif any(t in comment for t in ner_tokens if len(t) > 3):
                score = max(score, 0.48)

        if score >= threshold:
            scored.append((uri, round(score, 4)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:5]


def _get_domain_classes(prop: PropertyInfo, ontology: OntologyIndex) -> set[str]:
    """Domain class URIs, filtering out XSD types and unknown URIs."""
    return {
        d for d in prop.domain
        if d in ontology.classes
    }


def _get_range_classes(prop: PropertyInfo, ontology: OntologyIndex) -> set[str]:
    """Range class URIs for object properties, filtering out XSD and unknowns."""
    return {
        r for r in prop.range_
        if not r.startswith("http://www.w3.org/2001/XMLSchema#")
        and r in ontology.classes
    }


def _domain_range_compatible(
    subj_ancestors: set[str],
    obj_ancestors: set[str],
    prop: PropertyInfo,
    ontology: OntologyIndex,
) -> bool:
    """
    Check if subject and object entity types are compatible with the
    property's domain and range via subclass hierarchy.
    Empty/unresolvable domain or range is treated as compatible.
    """
    if prop.prop_type == "datatype":
        return False

    domain_cls = _get_domain_classes(prop, ontology)
    range_cls = _get_range_classes(prop, ontology)

    domain_ok = (not domain_cls) or bool(subj_ancestors & domain_cls)
    range_ok = (not range_cls) or bool(obj_ancestors & range_cls)
    return domain_ok and range_ok


def _find_valid_predicates(
    subj_ancestors: set[str],
    obj_ancestors: set[str],
    ontology: OntologyIndex,
) -> list[tuple[PropertyInfo, float]]:
    """
    Find the single most specific ontology object property compatible with
    subject and object types.  Returns at most 1 result to avoid the
    predicate-explosion problem (multiple predicates per entity pair).
    """
    results: list[tuple[PropertyInfo, float]] = []

    for _uri, prop in ontology.object_properties.items():
        if not _domain_range_compatible(subj_ancestors, obj_ancestors, prop, ontology):
            continue

        domain_cls = _get_domain_classes(prop, ontology)
        range_cls = _get_range_classes(prop, ontology)

        if not domain_cls or not range_cls:
            continue

        d_match = bool(subj_ancestors & domain_cls)
        r_match = bool(obj_ancestors & range_cls)
        if not (d_match and r_match):
            continue

        d_direct = len(subj_ancestors & domain_cls)
        r_direct = len(obj_ancestors & range_cls)
        specificity = d_direct + r_direct

        score = round(0.60 + min(specificity * 0.05, 0.35), 3)
        results.append((prop, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:1]


# ═══════════════════════════════════════════════════════════════════════════
#  VERSION v1 — Ontology-driven symbolic co-occurrence
# ═══════════════════════════════════════════════════════════════════════════

def _run_v1(
    entities: list[dict],
    text_blocks: list[dict],
    ontology: OntologyIndex,
) -> list[dict]:
    """
    Derive triples from entity co-occurrence using ontology domain/range.
    No LLM calls — fully symbolic.  Works for any OWL ontology.
    """
    ancestors_map = build_ancestors_map(ontology)

    linked = [e for e in entities if e.get("cui_final")]
    if not linked:
        log("3b", "v1: no linked entities — no assertions to generate")
        return []

    entity_classes: dict[str, list[tuple[str, float]]] = {}
    entity_ancestors: dict[str, set[str]] = {}

    for e in linked:
        text = e["text"]
        if text in entity_classes:
            continue
        matches = match_entity_to_classes(
            text, ontology, ner_type=e.get("type", ""),
        )
        entity_classes[text] = matches
        anc: set[str] = set()
        for cls_uri, _ in matches:
            anc |= ancestors_map.get(cls_uri, {cls_uri})
        entity_ancestors[text] = anc

    matched_count = sum(1 for v in entity_classes.values() if v)
    log("3b", f"v1: {matched_count}/{len(entity_classes)} unique entity texts matched to ontology classes")

    by_page: dict[int, list[dict]] = defaultdict(list)
    for e in linked:
        by_page[e["source_page"]].append(e)

    assertions: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for page, page_ents in by_page.items():
        for i, subj in enumerate(page_ents):
            s_anc = entity_ancestors.get(subj["text"], set())
            if not s_anc:
                continue
            s_classes = entity_classes.get(subj["text"], [])
            s_best = s_classes[0][1] if s_classes else 0.5

            for obj in page_ents[i + 1 :]:
                o_anc = entity_ancestors.get(obj["text"], set())
                if not o_anc:
                    continue
                o_classes = entity_classes.get(obj["text"], [])
                o_best = o_classes[0][1] if o_classes else 0.5

                for direction in ("fwd", "rev"):
                    sa, oa = (s_anc, o_anc) if direction == "fwd" else (o_anc, s_anc)
                    se, oe = (subj, obj) if direction == "fwd" else (obj, subj)
                    sb, ob = (s_best, o_best) if direction == "fwd" else (o_best, s_best)
                    sc, oc = (s_classes, o_classes) if direction == "fwd" else (o_classes, s_classes)

                    for prop, pred_conf in _find_valid_predicates(sa, oa, ontology):
                        key = (se["cui_final"], prop.uri, oe["cui_final"])
                        if key in seen:
                            continue
                        seen.add(key)

                        conf = round(pred_conf * min(sb, ob), 3)

                        assertions.append({
                            "subject_text": se["text"],
                            "subject_cui": se["cui_final"],
                            "subject_type": se.get("type", ""),
                            "subject_class": sc[0][0] if sc else "",
                            "predicate_uri": prop.uri,
                            "predicate_label": prop.label,
                            "object_text": oe["text"],
                            "object_cui": oe["cui_final"],
                            "object_type": oe.get("type", ""),
                            "object_class": oc[0][0] if oc else "",
                            "page": page,
                            "confidence": conf,
                            "source": "v1_ontology_driven",
                        })

    log("3b", f"v1: {len(assertions)} assertions from ontology-driven co-occurrence")
    return assertions


# ═══════════════════════════════════════════════════════════════════════════
#  VERSION v2 — Symbolic base + LLM augmentation with domain/range gate
# ═══════════════════════════════════════════════════════════════════════════

_SPO_PROMPT = """\
You are a biomedical knowledge graph expert extracting factual relationships.

RULES:
1. Subject and object MUST be exact entity texts from the Known Entities list.
2. Use ONLY the property URIs listed below — copy the URI exactly.
3. The subject entity type must match the property's Domain, and the object must match the Range.
4. Only extract relationships that are explicitly stated or clearly implied by the text.
5. Do NOT guess. If unsure, return [].

Text:
\"\"\"{text}\"\"\"

Known entities:
{entities_block}

Available properties (Domain → Range):
{props_block}

Respond with ONLY a JSON array. Each element:
  {{"subject": "<entity text>", "predicate_uri": "<full property URI>", "object": "<entity text>"}}

JSON:"""


_LLM_SKIP_PROPERTIES: set[str] = set()


def _relevant_properties_for_page(
    page_entity_ancestors: list[set[str]],
    ontology: OntologyIndex,
) -> list[PropertyInfo]:
    """
    Select ontology properties relevant to the entities on this page:
    only properties whose domain and range overlap with any of the entity
    class ancestors.  Filters out inverse/structural properties that confuse
    the LLM (isStageOf, isPhenotypeOf, progressesTo, continuesTherapyOf).
    """
    if not _LLM_SKIP_PROPERTIES:
        for _u, p in ontology.object_properties.items():
            lbl = p.label.lower()
            if any(kw in lbl for kw in ("is stage of", "is phenotype of",
                                         "progresses to", "continues therapy")):
                _LLM_SKIP_PROPERTIES.add(p.uri)

    all_ancestors: set[str] = set()
    for anc in page_entity_ancestors:
        all_ancestors |= anc

    relevant: list[PropertyInfo] = []
    for _uri, prop in ontology.object_properties.items():
        if prop.uri in _LLM_SKIP_PROPERTIES:
            continue
        domain_cls = _get_domain_classes(prop, ontology)
        range_cls = _get_range_classes(prop, ontology)
        d_ok = (not domain_cls) or bool(all_ancestors & domain_cls)
        r_ok = (not range_cls) or bool(all_ancestors & range_cls)
        if d_ok and r_ok:
            relevant.append(prop)
    return relevant


def _call_llm_spo(
    text: str,
    page_ents: list[dict],
    relevant_props: list[PropertyInfo],
    llm_backend: str,
    hf_token: Optional[str],
    hf_model: str,
) -> list[dict]:
    from hf_llm import (
        format_exception,
        hf_inference_chat,
        hf_local_generate,
        parse_json_array,
    )

    if llm_backend == "openai":
        log("3b", "Backend 'openai' is no longer supported; using 'hf_local' instead.")
        llm_backend = "hf_local"

    entities_block = "\n".join(
        f'  - "{e["text"]}"  (type: {e["type"]}, CUI: {e["cui_final"]})'
        for e in page_ents
    )
    props_block = "\n".join(
        f'  - {p.uri}: "{p.label}"  '
        f'(Domain: {", ".join(d.split("/")[-1] for d in p.domain[:2]) or "any"} → '
        f'Range: {", ".join(r.split("/")[-1] for r in p.range_[:2]) or "any"})'
        for p in relevant_props[:40]
    )
    prompt = _SPO_PROMPT.format(
        text=text[:1000],
        entities_block=entities_block,
        props_block=props_block,
    )
    try:
        if llm_backend == "hf_inference" and hf_token:
            raw = hf_inference_chat(prompt, hf_model, hf_token, max_new_tokens=600)
        elif llm_backend in ("hf_local", "llama"):
            raw = hf_local_generate(prompt, hf_model, hf_token, max_new_tokens=600)
        else:
            return []
        return parse_json_array(raw)
    except Exception as exc:
        log("3b", f"LLM SPO error: {format_exception(exc)}")
        return []


def _ontology_gate(
    candidate: dict,
    ontology: OntologyIndex,
    entity_map: dict[str, dict],
    entity_ancestors: dict[str, set[str]],
) -> Optional[dict]:
    """
    Validate an LLM-proposed SPO against ontology domain/range.
    Returns a validated assertion dict, or None if rejected.
    """
    pred_uri = candidate.get("predicate_uri", "")
    prop = ontology.object_properties.get(pred_uri)
    if prop is None:
        return None

    subj_key = candidate.get("subject", "")
    obj_key = candidate.get("object", "")
    subj_ent = entity_map.get(subj_key)
    obj_ent = entity_map.get(obj_key)
    if not subj_ent or not obj_ent:
        return None

    subj_anc = entity_ancestors.get(subj_key, set())
    obj_anc = entity_ancestors.get(obj_key, set())

    if not _domain_range_compatible(subj_anc, obj_anc, prop, ontology):
        return None

    domain_cls = _get_domain_classes(prop, ontology)
    range_cls = _get_range_classes(prop, ontology)
    specificity = 0.0
    if domain_cls and (subj_anc & domain_cls):
        specificity += 0.10
    if range_cls and (obj_anc & range_cls):
        specificity += 0.10

    subj_classes = match_entity_to_classes(subj_key, ontology, ner_type=subj_ent.get("type", ""))
    obj_classes = match_entity_to_classes(obj_key, ontology, ner_type=obj_ent.get("type", ""))

    return {
        "subject_text": subj_ent["text"],
        "subject_cui": subj_ent.get("cui_final"),
        "subject_type": subj_ent["type"],
        "subject_class": subj_classes[0][0] if subj_classes else "",
        "predicate_uri": pred_uri,
        "predicate_label": prop.label,
        "object_text": obj_ent["text"],
        "object_cui": obj_ent.get("cui_final"),
        "object_type": obj_ent["type"],
        "object_class": obj_classes[0][0] if obj_classes else "",
        "confidence": round(0.70 + specificity, 3),
        "source": "v2_llm_ontology_gated",
    }


def _run_v2(
    entities: list[dict],
    text_blocks: list[dict],
    ontology: OntologyIndex,
    llm_backend: str = "hf_local",
    hf_token: Optional[str] = None,
    hf_model: str = "meta-llama/Llama-3.1-8B-Instruct",
) -> list[dict]:
    """
    Symbolic base (v1) + LLM augmentation with domain/range gate.
    LLM is called only for text blocks where the symbolic pass found few
    assertions (neurosymbolic: LLM fills gaps, symbolic validates).
    """
    from hf_llm import probe_hf_local_backend

    if llm_backend == "openai":
        log("3b", "Backend 'openai' is no longer supported; using 'hf_local'.")
        llm_backend = "hf_local"
    if llm_backend == "llama":
        llm_backend = "hf_local"

    can_run_llm = False
    if llm_backend == "hf_inference" and hf_token:
        can_run_llm = True
    elif llm_backend == "hf_local":
        probe_error = probe_hf_local_backend(hf_model, hf_token)
        if probe_error:
            log("3b", f"v2: local LLM probe failed — falling back to v1. {probe_error}")
        else:
            can_run_llm = True

    assertions = _run_v1(entities, text_blocks, ontology)
    symbolic_count = len(assertions)

    if not can_run_llm:
        log("3b", f"v2: no usable LLM backend — returning v1 symbolic assertions only ({symbolic_count})")
        return assertions

    ancestors_map = build_ancestors_map(ontology)
    linked = [e for e in entities if e.get("cui_final")]
    by_page: dict[int, list[dict]] = defaultdict(list)
    for e in linked:
        by_page[e["source_page"]].append(e)

    entity_ancestors: dict[str, set[str]] = {}
    for e in linked:
        if e["text"] not in entity_ancestors:
            matches = match_entity_to_classes(e["text"], ontology, ner_type=e.get("type", ""))
            anc: set[str] = set()
            for cls_uri, _ in matches:
                anc |= ancestors_map.get(cls_uri, {cls_uri})
            entity_ancestors[e["text"]] = anc

    high_conf_per_page: dict[int, int] = defaultdict(int)
    for a in assertions:
        if a.get("confidence", 0) >= 0.70:
            high_conf_per_page[a["page"]] += 1

    seen: set[tuple] = {
        (a["subject_cui"], a["predicate_uri"], a["object_cui"])
        for a in assertions
    }
    llm_accepted = 0
    llm_rejected = 0

    pages_to_query = [
        block for block in text_blocks
        if len(by_page.get(block["page"], [])) >= 2
        and high_conf_per_page.get(block["page"], 0) < 3
    ]
    total_llm_pages = len(pages_to_query)
    log("3b", f"v2: {total_llm_pages} pages queued for LLM (out of {len(text_blocks)} text blocks)")

    for idx, block in enumerate(pages_to_query):
        page = block["page"]
        page_ents = by_page.get(page, [])

        page_anc_list = [entity_ancestors.get(e["text"], set()) for e in page_ents]
        relevant_props = _relevant_properties_for_page(page_anc_list, ontology)
        if not relevant_props:
            continue

        if (idx + 1) % 10 == 0 or idx == 0:
            log("3b", f"v2: LLM page {idx + 1}/{total_llm_pages} (page {page}, "
                f"+{llm_accepted} accepted, -{llm_rejected} rejected)")

        entity_map = {e["text"]: e for e in page_ents}
        raw_candidates = _call_llm_spo(
            block["text"], page_ents, relevant_props,
            llm_backend=llm_backend, hf_token=hf_token, hf_model=hf_model,
        )

        for cand in raw_candidates:
            validated = _ontology_gate(cand, ontology, entity_map, entity_ancestors)
            if validated is None:
                llm_rejected += 1
                continue
            key = (validated["subject_cui"], validated["predicate_uri"], validated["object_cui"])
            if key in seen:
                continue
            seen.add(key)
            validated["page"] = page
            assertions.append(validated)
            llm_accepted += 1

    log("3b", f"v2: {symbolic_count} symbolic + {llm_accepted} LLM-gated "
        f"({llm_rejected} LLM rejected) = {len(assertions)} total")
    return assertions


# ═══════════════════════════════════════════════════════════════════════════
#  RML GENERATOR  (shared)
# ═══════════════════════════════════════════════════════════════════════════

def _generate_text_rml(assertions_path: str, version: str) -> str:
    return (
        _RML_HEADER
        + f"<#TextAssertions_{version}>\n"
        + "    a rr:TriplesMap ;\n"
        + "    rml:logicalSource [\n"
        + f'        rml:source             "{assertions_path}" ;\n'
        + "        rml:referenceFormulation ql:JSONPath ;\n"
        + '        rml:iterator           "$[*]"\n'
        + "    ] ;\n"
        + "    rr:subjectMap [\n"
        + '        rr:template "http://digistructmed.org/instance/{subject_cui}" ;\n'
        + "        rr:class    ex:Entity\n"
        + "    ] ;\n"
        + "    rr:predicateObjectMap [\n"
        + '        rr:predicateMap [ rml:reference "predicate_uri" ] ;\n'
        + "        rr:objectMap    [\n"
        + '            rr:template "http://digistructmed.org/instance/{object_cui}"\n'
        + "        ]\n"
        + "    ] .\n"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def run_text_path(
    resolved_entities_path: str = "outputs/step1/resolved_entities.json",
    text_blocks_path: str = "outputs/step1/text_blocks.json",
    ontology: OntologyIndex | None = None,
    ontology_path: str = "input/hf_guideline_ontology.ttl",
    output_dir: str = "outputs/step3",
    version: str = "v1",
    llm_backend: str = "hf_local",
    hf_token: Optional[str] = None,
    hf_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    confidence_threshold: float = 0.4,
) -> dict:
    """
    Run the text path strategy.

    version="v1" — ontology-driven symbolic (no LLM triple generation)
    version="v2" — symbolic base + LLM augmentation, ontology-gated
    """
    assert version in ("v1", "v2"), "version must be 'v1' or 'v2'"

    if ontology is None:
        from step2_load_ontology import load_ontology
        ontology = load_ontology(ontology_path)

    entities: list[dict] = json.loads(Path(resolved_entities_path).read_text(encoding="utf-8"))
    text_blocks: list[dict] = json.loads(Path(text_blocks_path).read_text(encoding="utf-8"))

    log("3b", f"Running text path version: {version}")

    if version == "v1":
        assertions = _run_v1(entities, text_blocks, ontology)
    else:
        assertions = _run_v2(
            entities, text_blocks, ontology,
            llm_backend=llm_backend, hf_token=hf_token, hf_model=hf_model,
        )

    assertions = [a for a in assertions if a.get("confidence", 0) >= confidence_threshold]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assertions_path = out_dir / f"text_assertions_{version}.json"
    save_json(assertions, str(assertions_path))

    rml_content = _generate_text_rml(str(assertions_path), version)
    rml_path = out_dir / f"text_mappings_{version}.ttl"
    rml_path.write_text(rml_content, encoding="utf-8")

    log("3b", f"[{version}] {len(assertions)} text assertions → {assertions_path}")
    log("3b", f"[{version}] RML → {rml_path}")

    return {
        "version": version,
        "assertions_path": str(assertions_path),
        "rml_path": str(rml_path),
        "count": len(assertions),
    }


if __name__ == "__main__":
    import sys
    ver = sys.argv[1] if len(sys.argv) > 1 else "v1"
    run_text_path(version=ver)
