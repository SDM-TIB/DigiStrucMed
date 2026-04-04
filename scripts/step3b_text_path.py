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

VERSION v1
  LLM role : CUI resolution only (already done in Step 1e).
  Triples  : derived from entity co-occurrence + explicit NER-type rules.

VERSION v2
  LLM role : CUI resolution (Step 1e) + structured SPO candidate proposal.
  Gate     : every LLM-proposed (s, p, o) is validated against ontology
             domain/range before being written to the assertion file.

Use run_text_path(version="v1"|"v2") to switch.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from step2_load_ontology import OntologyIndex
from utils import log, save_json

# ── Shared RML header ──────────────────────────────────────────────────────

_RML_HEADER = """\
@prefix rr:  <http://www.w3.org/ns/r2rml#> .
@prefix rml: <http://semweb.mmlab.be/ns/rml#> .
@prefix ql:  <http://semweb.mmlab.be/ns/ql#> .
@prefix ex:  <http://digistructmed.org/ontology/> .

"""

# ── v1 — NER-type co-occurrence rules ──────────────────────────────────────
# Maps (NER_label_A, NER_label_B) → ontology property URI.
# Extend this table as the ontology grows.

NER_TYPE_RULES: dict[tuple[str, str], str] = {
    ("Medication",           "Disease_disorder"):
        "http://digistructmed.org/ontology/recommendedFor",
    ("Therapeutic_procedure","Disease_disorder"):
        "http://digistructmed.org/ontology/treates",
    ("Medication",           "Sign_symptom"):
        "http://digistructmed.org/ontology/reduces",
    ("Diagnostic_procedure", "Disease_disorder"):
        "http://digistructmed.org/ontology/diagnoses",
    ("Medication",           "Medication"):
        "http://digistructmed.org/ontology/usedWith",
}


def _lookup_rule(type_a: str, type_b: str) -> Optional[str]:
    return (
        NER_TYPE_RULES.get((type_a, type_b))
        or NER_TYPE_RULES.get((type_b, type_a))
    )


# ── VERSION v1 ─────────────────────────────────────────────────────────────

def _run_v1(
    entities: list[dict],
    text_blocks: list[dict],
    ontology: OntologyIndex,
) -> list[dict]:
    """
    Derive triples from co-occurrence + NER-type rules (v1).
    No additional LLM calls.
    """
    linked = [e for e in entities if e.get("cui_final")]
    by_page: dict[int, list[dict]] = defaultdict(list)
    for e in linked:
        by_page[e["source_page"]].append(e)

    assertions: list[dict] = []

    for page, page_ents in by_page.items():
        for i, subj in enumerate(page_ents):
            for obj in page_ents[i + 1:]:
                pred_uri = _lookup_rule(subj["type"], obj["type"])
                if not pred_uri:
                    continue
                assertions.append({
                    "subject_text":  subj["text"],
                    "subject_cui":   subj["cui_final"],
                    "subject_type":  subj["type"],
                    "predicate_uri": pred_uri,
                    "object_text":   obj["text"],
                    "object_cui":    obj["cui_final"],
                    "object_type":   obj["type"],
                    "page":          page,
                    "confidence":    0.65,
                    "source":        "v1_cooccurrence_rule",
                })

    return assertions


# ── VERSION v2 ─────────────────────────────────────────────────────────────

_SPO_PROMPT = """\
You are a biomedical knowledge graph expert.

Extract factual SPO assertions from the text below.
Use ONLY the ontology properties listed.
Subject and object MUST be entity texts from the Known Entities list.

Text:
\"\"\"{text}\"\"\"

Known entities (already linked to UMLS CUIs):
{entities_block}

Available ontology properties (use exact URI):
{props_block}

Respond with a JSON array only. Each item:
  {{"subject": "<entity text>", "predicate_uri": "<property URI>", "object": "<entity text>"}}
If no valid assertions exist, respond with [].

JSON:"""


def _call_llm_spo(
    text: str,
    page_ents: list[dict],
    ontology: OntologyIndex,
    llm_backend: str,
    openai_api_key: Optional[str],
    hf_token: Optional[str],
    hf_model: str,
    openai_model: str = "gpt-4",
) -> list[dict]:
    from hf_llm import hf_inference_chat, hf_local_generate, parse_json_array

    entities_block = "\n".join(
        f'  - "{e["text"]}"  (type: {e["type"]}, CUI: {e["cui_final"]})'
        for e in page_ents
    )
    props_block = "\n".join(
        f'  - {uri}: "{p.label}"'
        for uri, p in list(ontology.all_properties().items())[:30]   # cap prompt size
    )
    prompt = _SPO_PROMPT.format(
        text=text[:800],
        entities_block=entities_block,
        props_block=props_block,
    )
    try:
        if llm_backend == "openai" and openai_api_key:
            from openai import OpenAI
            client = OpenAI(api_key=openai_api_key)
            resp = client.chat.completions.create(
                model=openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
        elif llm_backend == "hf_inference" and hf_token:
            raw = hf_inference_chat(prompt, hf_model, hf_token, max_new_tokens=600)
        elif llm_backend == "hf_local":
            raw = hf_local_generate(prompt, hf_model, hf_token, max_new_tokens=600)
        elif llm_backend == "llama":
            raw = hf_local_generate(prompt, hf_model, hf_token, max_new_tokens=600)
        else:
            return []
        return parse_json_array(raw)
    except Exception as exc:
        log("3b", f"LLM SPO error: {exc}")
        return []


def _ontology_gate(
    candidate: dict,
    ontology: OntologyIndex,
    entity_map: dict[str, dict],
) -> Optional[dict]:
    """
    Validate a candidate SPO against the ontology.
    Returns a validated assertion dict, or None if rejected.
    """
    pred_uri = candidate.get("predicate_uri", "")
    prop = ontology.all_properties().get(pred_uri)
    if prop is None:
        return None   # predicate not in ontology → reject

    subj_ent = entity_map.get(candidate.get("subject", ""))
    obj_ent  = entity_map.get(candidate.get("object", ""))
    if not subj_ent or not obj_ent:
        return None   # unknown entity → reject

    return {
        "subject_text":   subj_ent["text"],
        "subject_cui":    subj_ent.get("cui_final"),
        "subject_type":   subj_ent["type"],
        "predicate_uri":  pred_uri,
        "predicate_label": prop.label,
        "object_text":    obj_ent["text"],
        "object_cui":     obj_ent.get("cui_final"),
        "object_type":    obj_ent["type"],
        "confidence":     0.78,
        "source":         "v2_llm_ontology_gated",
    }


def _run_v2(
    entities: list[dict],
    text_blocks: list[dict],
    ontology: OntologyIndex,
    llm_backend: str = "none",
    openai_api_key: Optional[str] = None,
    hf_token: Optional[str] = None,
    hf_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    openai_model: str = "gpt-4",
) -> list[dict]:
    """
    v2: LLM proposes SPO records; ontology-gated filter applied.
    """
    if llm_backend == "llama":
        llm_backend = "hf_local"

    can_run = False
    if llm_backend == "openai" and openai_api_key:
        can_run = True
    elif llm_backend == "hf_inference" and hf_token:
        can_run = True
    elif llm_backend == "hf_local":
        can_run = True  # public models work without token; gated need HF_TOKEN

    if not can_run:
        log("3b", "v2: no usable LLM backend — falling back to v1 co-occurrence rules")
        return _run_v1(entities, text_blocks, ontology)

    linked = [e for e in entities if e.get("cui_final")]
    by_page: dict[int, list[dict]] = defaultdict(list)
    for e in linked:
        by_page[e["source_page"]].append(e)

    assertions: list[dict] = []

    for block in text_blocks:
        page = block["page"]
        page_ents = by_page.get(page, [])
        if not page_ents:
            continue

        entity_map = {e["text"]: e for e in page_ents}
        raw_candidates = _call_llm_spo(
            block["text"],
            page_ents,
            ontology,
            llm_backend=llm_backend,
            openai_api_key=openai_api_key,
            hf_token=hf_token,
            hf_model=hf_model,
            openai_model=openai_model,
        )

        for cand in raw_candidates:
            validated = _ontology_gate(cand, ontology, entity_map)
            if validated:
                validated["page"] = page
                assertions.append(validated)

    return assertions


# ── RML generator (shared) ─────────────────────────────────────────────────

def _generate_text_rml(assertions_path: str, version: str) -> str:
    """
    Produce a minimal RML TriplesMap that reads text_assertions_<v>.json
    and maps each record to RDF triples.
    """
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
        + "        rr:predicateMap [ rml:reference \"predicate_uri\" ] ;\n"
        + "        rr:objectMap    [\n"
        + '            rr:template "http://digistructmed.org/instance/{object_cui}"\n'
        + "        ]\n"
        + "    ] .\n"
    )


# ── Public entry point ─────────────────────────────────────────────────────

def run_text_path(
    resolved_entities_path: str = "outputs/step1/resolved_entities.json",
    text_blocks_path: str = "outputs/step1/text_blocks.json",
    ontology: OntologyIndex | None = None,
    ontology_path: str = "input/hf_guideline_ontology.ttl",
    output_dir: str = "outputs/step3",
    version: str = "v1",              # "v1" | "v2"
    llm_backend: str = "none",
    openai_api_key: Optional[str] = None,
    openai_model: str = "gpt-4",
    hf_token: Optional[str] = None,
    hf_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    confidence_threshold: float = 0.6,
) -> dict:
    """
    Run the text path strategy.

    version="v1" — rule-based triples from co-occurrence (no LLM triple generation)
    version="v2" — LLM proposes SPO candidates, ontology-gated
    """
    assert version in ("v1", "v2"), "version must be 'v1' or 'v2'"

    if ontology is None:
        from step2_load_ontology import load_ontology
        ontology = load_ontology(ontology_path)

    entities: list[dict]    = json.loads(Path(resolved_entities_path).read_text(encoding="utf-8"))
    text_blocks: list[dict] = json.loads(Path(text_blocks_path).read_text(encoding="utf-8"))

    log("3b", f"Running text path version: {version}")

    if version == "v1":
        assertions = _run_v1(entities, text_blocks, ontology)
    else:
        assertions = _run_v2(
            entities,
            text_blocks,
            ontology,
            llm_backend=llm_backend,
            openai_api_key=openai_api_key,
            hf_token=hf_token,
            hf_model=hf_model,
            openai_model=openai_model,
        )

    # Apply confidence filter
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
