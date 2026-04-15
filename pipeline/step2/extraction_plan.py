"""
extraction_plan.py — Ontology-driven extraction plan builder.

Reads the OWL/TTL ontology and produces a JSON plan of:
  - classes (with hierarchy)
  - object & datatype properties (with domain/range)
  - canonical target sources (ontology-shaped CSVs)
  - role_hints: header/caption patterns for automatic table-role detection

This plan is GUIDELINE-AGNOSTIC: it derives targets from the ontology alone.
Individual normalizers use the plan to decide what to extract.

Lecture 9 grounding: ⟨O, S, M⟩ — the ontology O defines the target schema,
the sources S are populated by the normalizer, and mappings M (RML) are
written against the stable canonical sources.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef


EX = Namespace("http://digistructmed.org/ontology/")


def _load_config(config_path=None):
    """Load guideline-specific configuration from JSON file."""
    if config_path is None:
        config_path = Path(__file__).parent / "guideline_config.json"
    else:
        config_path = Path(config_path)
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {}


@dataclass(frozen=True)
class PropRow:
    iri: str
    label: str | None
    domain: str | None
    range: str | None
    kind: str  # "object" | "datatype"


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


def build_extraction_plan(ontology_path: str, config_path: str = None) -> dict:
    """
    Build a compact, ontology-driven extraction plan.
    """
    cfg = _load_config(config_path)
    disease_abbrevs = cfg.get("disease", {}).get("abbreviations", [])
    disease_name = cfg.get("disease", {}).get("disease_name", "the target disease")
    staging_systems = cfg.get("detection_hints", {}).get("staging_systems", [])
    pheno_headers = cfg.get("detection_hints", {}).get("phenotype_header_terms", ["phenotype"])
    pheno_captions = cfg.get("detection_hints", {}).get("phenotype_caption_terms", [])
    mcs_kw = cfg.get("detection_hints", {}).get("mcs_keywords", [])
    orgs = cfg.get("guideline_organizations", [])

    g = Graph()
    g.parse(ontology_path, format="turtle")

    classes: list[dict] = []
    # Support both owl:Class and rdfs:Class
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
    # Also check rdf:Property (used in new ontology style)
    rdf_props = sorted(_iter_props(g, RDF.Property, "general"), key=lambda r: r.iri)

    # Canonical target sources — ontology-shaped CSVs.
    # Each source corresponds to a class or a link table for an object property.
    canonical_sources = [
        # Entity tables
        {"name": "S_guideline.csv", "class": "ex:Guideline",
         "fields": ["guideline_id", "guidelineTitle", "guidelineSource", "guidelineDate"]},
        {"name": "S_recommendation.csv", "class": "ex:Recommendation",
         "fields": ["rec_id", "recommendationText"]},
        {"name": "S_stage.csv", "class": "ex:Stage",
         "fields": ["stage_id", "stageScheme", "stageLevel", "stageName", "StageCriteriaText"]},
        {"name": "S_disease.csv", "class": "ex:Disease",
         "fields": ["disease_id", "diseaseName"]},
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
        # Reification tables (n-ary relationships)
        {"name": "S_contains.csv", "class": "ex:Contains",
         "fields": ["contains_id", "guideline_id", "disease_id", "rec_id"]},
        {"name": "S_treats.csv", "class": "ex:Treats",
         "fields": ["treats_id", "disease_id", "therapy_id", "drug_id"]},
        # Link tables (binary object properties)
        {"name": "S_disease_stage.csv", "property": "ex:hasStage",
         "fields": ["disease_id", "stage_id"]},
        {"name": "S_disease_cause.csv", "property": "ex:hasCause",
         "fields": ["disease_id", "cause_id"]},
        {"name": "S_disease_phenotype.csv", "property": "ex:hasPhenotype",
         "fields": ["disease_id", "phenotype_id"]},
        {"name": "S_disease_assessment.csv", "property": "ex:evaluatedBy",
         "fields": ["disease_id", "assessment_id"]},
        {"name": "S_drug_adverse_event.csv", "property": "ex:hasAdverseEvent",
         "fields": ["drug_id", "ae_id"]},
        # Passthrough for unmatched tables
        {"name": "S_unmatched/", "class": "PASSTHROUGH",
         "fields": ["original_csv_copied_with_metadata"]},
    ]

    # Role hints — guideline-agnostic header/caption patterns used by
    # normalize_tables.py to classify tables into ontology roles.
    # These are OPEN patterns (substring/regex), NOT hardcoded table IDs.
    # Build disease-aware role hints from config
    _abbr_lower = [a.lower() for a in disease_abbrevs]
    _orgs_lower = [o.lower() for o in orgs]

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
            "caption_patterns": ["drug", "dose"] + cfg.get("detection_hints", {}).get("drug_dosing_caption_extras", []),
        },
        "harmful_drug": {
            "description": f"Drugs that may cause or exacerbate {disease_name}",
            "header_patterns": ["drug", "therapeutic class", "magnitude", "mechanism"],
            "caption_patterns": ["harm", "exacerbat", "cause or"],
        },
        "stage": {
            "description": f"Disease stage definitions ({', '.join(staging_systems[:3])})",
            "header_patterns": ["stage", "definition", "criteria"],
            "caption_patterns": ["stage"],
        },
        "phenotype": {
            "description": f"{disease_name} phenotype classification",
            "header_patterns": pheno_headers + ["criteria"],
            "caption_patterns": pheno_captions + ["phenotype"],
        },
        "cause": {
            "description": f"Causes or etiologies of {disease_name}",
            "header_patterns": ["cause", "etiology", "aetiology"],
            "caption_patterns": ["cause", "etiolog", "aetiolog"],
        },
        "comorbidity": {
            "description": "Co-occurring conditions and prevalence",
            "header_patterns": ["beneficiar", "condition", "prevalence", "n", "%"],
            "caption_patterns": ["co-occur", "comorbidit", "chronic condition"],
        },
        "risk_score": {
            "description": "Multivariable risk prediction scores",
            "header_patterns": ["risk score", "year published", "reference"],
            "caption_patterns": ["risk score", "predict"],
        },
        "staging_classification": {
            "description": f"Clinical staging systems ({', '.join(staging_systems[:4])})",
            "header_patterns": ["profile", "description", "features", "bedside", "hemodynamics"],
            "caption_patterns": staging_systems,
        },
        "self_care_barrier": {
            "description": "Self-care barriers with screening tools and interventions",
            "header_patterns": ["barrier", "screening", "intervention"],
            "caption_patterns": ["barrier", "self-care"],
        },
        "vulnerable_population": {
            "description": "Risk and outcomes in special populations",
            "header_patterns": ["vulnerable", "population"] + [f"risk of {a.lower()}" for a in disease_abbrevs[:1]] + ["outcome"],
            "caption_patterns": ["vulnerable", "special population", "disparit"],
        },
        "therapy_benefit": {
            "description": "Evidence-based therapy benefit comparison (NNT, RRR)",
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
            "description": f"{disease_name} management across pregnancy continuum",
            "header_patterns": ["preconception", "during pregnancy", "postpartum"],
            "caption_patterns": ["pregnancy"],
        },
        "genetic_factor": {
            "description": "Genetic cardiomyopathy factors",
            "header_patterns": ["phenotypic category", "family member", "finding"],
            "caption_patterns": ["genetic", "cardiomyopathy"],
        },
        "precipitating_factor": {
            "description": f"Common factors precipitating {disease_name} hospitalization",
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
        "advanced_hf_definition": {
            "description": f"Definitions and criteria for advanced {disease_name}",
            "header_patterns": ["criteria", "guideline-directed"],
            "caption_patterns": [f"advanced {a.lower()}" for a in disease_abbrevs[:1]] + ["definition of advanced"],
        },
        "mcs_indication": {
            "description": "MCS indications and contraindications",
            "header_patterns": ["indication", "contraindication"],
            "caption_patterns": mcs_kw,
        },
        "clinical_indicator": {
            "description": f"Clinical indicators of advanced {disease_name}",
            "header_patterns": ["repeated hospitalization", "refractory"],
            "caption_patterns": ["clinical indicator"] + [f"advanced {a.lower()}" for a in disease_abbrevs[:1]],
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
            "description": "Cross-reference to other associated guidelines",
            "header_patterns": ["title", "organization", "publication year"],
            "caption_patterns": ["associated guideline"] + [f"other {'/'.join(_orgs_lower[:2])}"] if _orgs_lower else ["associated guideline"],
        },
    }

    return {
        "ontology_path": str(Path(ontology_path)),
        "classes": sorted(classes, key=lambda x: x["qname"]),
        "properties": {
            "object": [r.__dict__ for r in obj_props],
            "datatype": [r.__dict__ for r in dt_props],
            "general": [r.__dict__ for r in rdf_props],
        },
        "canonical_sources": canonical_sources,
        "role_hints": role_hints,
        "notes": [
            "This plan is ontology-driven: it lists the target classes and slots (properties) to populate.",
            "role_hints provide guideline-agnostic patterns for classifying extracted tables.",
            "Unmatched tables are preserved in S_unmatched/ as passthrough for manual review.",
            "The canonical_sources list is a guideline-agnostic target schema.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology", default="ontology.ttl")
    ap.add_argument("--out", required=True, help="Path to write extraction_plan.json")
    ap.add_argument("--config", default=None, help="Path to guideline_config.json")
    args = ap.parse_args()

    plan = build_extraction_plan(args.ontology, config_path=args.config)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
