"""Build ontology-driven ``extraction_plan.json`` (classes, properties, role_hints, sources)."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef


EX = Namespace("http://digistructmed.org/ontology/")


@dataclass(frozen=True)
class PropRow:
    iri: str
    label: str | None
    domain: str | None
    range: str | None
    kind: str  # "object" | "datatype" | "general"


def _qname_or_str(g: Graph, iri: URIRef) -> str:
    try:
        return g.namespace_manager.normalizeUri(iri)
    except Exception:
        return str(iri)


def _first_literal(g: Graph, subj: URIRef, pred: URIRef) -> str | None:
    for o in g.objects(subj, pred):
        if hasattr(o, "value"):
            return str(o)
    return None


def _iter_props(g: Graph, rdf_type: URIRef, kind: str) -> Iterable[PropRow]:
    for p in g.subjects(RDF.type, rdf_type):
        if not isinstance(p, URIRef):
            continue
        label = _first_literal(g, p, RDFS.label)
        dom = next(iter(g.objects(p, RDFS.domain)), None)
        rng = next(iter(g.objects(p, RDFS.range)), None)
        yield PropRow(
            iri=str(p),
            label=label,
            domain=_qname_or_str(g, dom) if isinstance(dom, URIRef) else (str(dom) if dom else None),
            range=_qname_or_str(g, rng) if isinstance(rng, URIRef) else (str(rng) if rng else None),
            kind=kind,
        )


def build_extraction_plan(ontology_path: str) -> dict:
    g = Graph()
    g.parse(ontology_path, format="turtle")

    classes: list[dict] = []
    class_iris = set()
    for c in g.subjects(RDF.type, OWL.Class):
        if isinstance(c, URIRef):
            class_iris.add(c)
    for c in g.subjects(RDF.type, RDFS.Class):
        if isinstance(c, URIRef):
            class_iris.add(c)

    for c in class_iris:
        if str(c).startswith(str(OWL)):
            continue
        label = _first_literal(g, c, RDFS.label)
        comment = _first_literal(g, c, RDFS.comment)
        classes.append(
            {
                "iri": str(c),
                "qname": _qname_or_str(g, c),
                "label": label,
                "comment": comment,
                "subClassOf": [
                    _qname_or_str(g, sup)
                    for sup in g.objects(c, RDFS.subClassOf)
                    if isinstance(sup, URIRef)
                ],
            }
        )

    obj_props = sorted(_iter_props(g, OWL.ObjectProperty, "object"), key=lambda r: r.iri)
    dt_props = sorted(_iter_props(g, OWL.DatatypeProperty, "datatype"), key=lambda r: r.iri)
    rdf_props = sorted(_iter_props(g, RDF.Property, "general"), key=lambda r: r.iri)

    canonical_sources = [
        {"name": "S_guideline.csv", "class": "ex:Guideline",
         "fields": ["guideline_id", "guidelineTitle", "guidelineSource", "guidelineDate"]},
        {"name": "S_recommendation.csv", "class": "ex:Recommendation",
         "fields": ["rec_id", "recommendationText"]},
        {"name": "S_stage.csv", "class": "ex:Stage",
         "fields": ["stage_id", "stageScheme", "stageLevel", "stageName", "StageCriteriaText"]},
        {"name": "S_condition.csv", "class": "ex:Condition",
         "fields": ["condition_id", "conditionName"]},
        {"name": "S_drug.csv", "class": "ex:Drug",
         "fields": ["drug_id", "agentName", "drugCategory", "minDose", "maxDose", "duration"]},
        {"name": "S_therapy.csv", "class": "ex:Therapy",
         "fields": ["therapy_id", "therapyType", "therapy_name"]},
        {"name": "S_assessment.csv", "class": "ex:Assessment",
         "fields": ["assessment_id", "assessmentName", "assessmentValue"]},
        {"name": "S_adverse_event.csv", "class": "ex:AdverseEvent",
         "fields": ["ae_id", "adverseEventName", "adverseEventSeverity"]},
        {"name": "S_cause.csv", "class": "ex:Cause",
         "fields": ["cause_id", "causeName"]},
        {"name": "S_phenotype.csv", "class": "ex:Phenotype",
         "fields": ["phenotype_id", "phenotypeCode", "phenotypeCriteria"]},
        {"name": "S_annotation_concept.csv", "class": "ex:AnnotationConcept",
         "fields": ["concept_id", "conceptName"]},
        {"name": "S_contains.csv", "class": "ex:Contains",
         "fields": ["contains_id", "guideline_id", "condition_id", "rec_id"]},
        {"name": "S_treats.csv", "class": "ex:Treats",
         "fields": ["treats_id", "condition_id", "therapy_id", "drug_id"]},
        {"name": "S_condition_stage.csv", "property": "ex:hasStage",
         "fields": ["condition_id", "stage_id"]},
        {"name": "S_condition_cause.csv", "property": "ex:hasCause",
         "fields": ["condition_id", "cause_id"]},
        {"name": "S_condition_phenotype.csv", "property": "ex:hasPhenotype",
         "fields": ["condition_id", "phenotype_id"]},
        {"name": "S_condition_assessment.csv", "property": "ex:evaluatedBy",
         "fields": ["condition_id", "assessment_id"]},
        {"name": "S_drug_adverse_event.csv", "property": "ex:hasAdverseEvent",
         "fields": ["drug_id", "ae_id"]},
        {"name": "S_unmatched/", "class": "PASSTHROUGH",
         "fields": ["original_csv_copied_with_metadata"]},
    ]

    # Generic clinical guideline role hints (no disease-specific terms)
    role_hints = {
        "recommendation": {
            "description": "Guideline recommendation text tables",
            "header_patterns": ["recommendation"],
            "caption_patterns": ["recommendation"],
            "first_header_startswith": "recommendation",
        },
        "drug_dosing": {
            "description": "Drug dosing tables with initial/target doses",
            "header_patterns": ["drug", "dose", "initial", "target"],
            "caption_patterns": ["drug", "dose"],
        },
        "harmful_drug": {
            "description": "Drugs that may cause or exacerbate a condition",
            "header_patterns": ["drug", "therapeutic class", "magnitude", "mechanism"],
            "caption_patterns": ["harm", "exacerbat", "cause or"],
        },
        "stage": {
            "description": "Condition stage definitions",
            "header_patterns": ["stage", "definition", "criteria"],
            "caption_patterns": ["stage"],
        },
        "phenotype": {
            "description": "Condition phenotype classification",
            "header_patterns": ["phenotype", "criteria"],
            "caption_patterns": ["classification", "phenotype"],
        },
        "cause": {
            "description": "Causes or etiologies of a condition",
            "header_patterns": ["cause", "etiology", "aetiology"],
            "caption_patterns": ["cause", "etiolog", "aetiolog"],
        },
        "comorbidity": {
            "description": "Co-occurring conditions and prevalence",
            "header_patterns": ["condition", "prevalence"],
            "caption_patterns": ["co-occur", "comorbidit", "chronic condition"],
        },
        "risk_score": {
            "description": "Multivariable risk prediction scores",
            "header_patterns": ["risk score", "year published", "reference"],
            "caption_patterns": ["risk score", "predict"],
        },
        "staging_classification": {
            "description": "Clinical staging systems",
            "header_patterns": ["profile", "description", "features", "hemodynamics"],
            "caption_patterns": ["profile", "staging"],
        },
        "self_care_barrier": {
            "description": "Self-care barriers with screening tools and interventions",
            "header_patterns": ["barrier", "screening", "intervention"],
            "caption_patterns": ["barrier", "self-care"],
        },
        "vulnerable_population": {
            "description": "Risk and outcomes in special populations",
            "header_patterns": ["vulnerable", "population", "outcome"],
            "caption_patterns": ["vulnerable", "special population", "disparit"],
        },
        "therapy_benefit": {
            "description": "Evidence-based therapy benefit comparison",
            "header_patterns": ["nnt", "relative risk", "evidence-based therapy", "mortality"],
            "caption_patterns": ["benefit", "evidence-based", "nnt"],
        },
        "inotropic_agent": {
            "description": "IV inotropic agents with dosing and effects",
            "header_patterns": ["inotropic", "dose", "infusion", "adverse"],
            "caption_patterns": ["inotropic", "intravenous"],
        },
        "cardiotoxic_agent": {
            "description": "Cancer therapies associated with cardiomyopathy",
            "header_patterns": ["class", "cardiac function", "monitoring"],
            "caption_patterns": ["cancer therap", "cardiotox", "cardiomyopathy"],
        },
        "pregnancy_management": {
            "description": "Condition management across pregnancy continuum",
            "header_patterns": ["preconception", "during pregnancy", "postpartum"],
            "caption_patterns": ["pregnancy"],
        },
        "genetic_factor": {
            "description": "Genetic cardiomyopathy factors",
            "header_patterns": ["phenotypic category", "family member", "finding"],
            "caption_patterns": ["genetic", "cardiomyopathy"],
        },
        "precipitating_factor": {
            "description": "Common factors precipitating hospitalization",
            "header_patterns": [],
            "caption_patterns": ["precipitat", "factor"],
        },
        "performance_measure": {
            "description": "Clinical performance and quality measures",
            "header_patterns": ["measure no", "measure title", "care setting"],
            "caption_patterns": ["performance", "quality", "measure"],
        },
        "palliative_care": {
            "description": "Palliative and supportive care domains",
            "header_patterns": ["palliative", "supportive", "domain"],
            "caption_patterns": ["palliative", "supportive care"],
        },
        "transitional_care": {
            "description": "Components of a transitional care plan",
            "header_patterns": ["transitional", "care plan"],
            "caption_patterns": ["transitional", "care plan"],
        },
        "advanced_definition": {
            "description": "Definitions and criteria for advanced condition",
            "header_patterns": ["criteria", "guideline-directed"],
            "caption_patterns": ["advanced", "definition of advanced"],
        },
        "mcs_indication": {
            "description": "Mechanical circulatory support indications",
            "header_patterns": ["indication", "contraindication"],
            "caption_patterns": ["mechanical", "durable", "support"],
        },
        "clinical_indicator": {
            "description": "Clinical indicators of advanced condition",
            "header_patterns": ["repeated hospitalization", "refractory"],
            "caption_patterns": ["clinical indicator", "advanced"],
        },
        "shock_criteria": {
            "description": "Cardiogenic shock clinical/hemodynamic criteria",
            "header_patterns": ["sbp", "hypoperfusion", "hemodynamic"],
            "caption_patterns": ["shock", "hemodynamic criteria"],
        },
        "natriuretic_peptide_cause": {
            "description": "Causes of elevated natriuretic peptide levels",
            "header_patterns": ["cardiac"],
            "caption_patterns": ["natriuretic peptide", "elevated"],
        },
        "associated_guideline": {
            "description": "Cross-reference to other guidelines",
            "header_patterns": ["title", "organization", "publication year"],
            "caption_patterns": ["associated guideline"],
        },
    }

    return {
        "ontology_path": str(Path(ontology_path)),
        "version": "v2-generic",
        "classes": sorted(classes, key=lambda x: x["qname"]),
        "properties": {
            "object": [r.__dict__ for r in obj_props],
            "datatype": [r.__dict__ for r in dt_props],
            "general": [r.__dict__ for r in rdf_props],
        },
        "canonical_sources": canonical_sources,
        "role_hints": role_hints,
        "notes": [
            "This plan is purely ontology-driven with no guideline-specific config.",
            "Role hints use generic clinical guideline vocabulary only.",
            "Condition identity and metadata are extracted from the document at runtime.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology", default="ontology.ttl")
    ap.add_argument("--out", required=True, help="Path to write extraction_plan.json")
    args = ap.parse_args()

    plan = build_extraction_plan(args.ontology)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
