"""
Stage C evaluation: analyze and score NER outputs (statements with entities + table triples).

Evaluates one or more versions (v1, v2, v3, ...). Each version is read from
outputs/STAGE_C_{version}/stage_c_statements_with_entities.json.

Metrics: schema validity, entity coverage, entity quality, biomedical relevance,
table triples enrichment. No other pipeline stages are run; this is Stage C only.

Usage (from project root):
  python evaluation/evaluate_stage_c.py
  python evaluation/evaluate_stage_c.py --versions v1 v2 v3
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVAL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"

# Default versions to evaluate
DEFAULT_VERSIONS = ["v1", "v2"]

# Biomedical entity labels we care about for guideline extraction (from RecognizeEntities)
BIOMEDICAL_LABELS = frozenset({
    "Medication",
    "Disease_disorder",
    "Diagnostic_procedure",
    "Sign_symptom",
    "Therapeutic_procedure",
})


# ---------------------------------------------------------------------------
# Load Stage C output
# ---------------------------------------------------------------------------

def load_stage_c_output(path: Path) -> Tuple[Optional[Dict], Optional[List], Optional[List]]:
    """
    Load stage_c_statements_with_entities.json.
    Returns (metadata, statements, table_triples). (None, None, None) if file missing.
    """
    if not path.exists():
        return None, None, None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    metadata = data.get("metadata", {})
    statements = data.get("statements", [])
    if not isinstance(statements, list):
        statements = []
    table_triples = data.get("table_triples", [])
    if not isinstance(table_triples, list):
        table_triples = []
    return metadata, statements, table_triples


# ---------------------------------------------------------------------------
# Schema & validity
# ---------------------------------------------------------------------------

def validate_schema(metadata: Dict, statements: List[Dict], table_triples: List[Dict]) -> Dict[str, Any]:
    errors = []
    if not isinstance(metadata, dict):
        errors.append("metadata not a dict")
    if not isinstance(statements, list):
        errors.append("statements not a list")
    if not isinstance(table_triples, list):
        errors.append("table_triples not a list")

    for i, stmt in enumerate(statements):
        if not isinstance(stmt, dict):
            errors.append(f"statements[{i}] not a dict")
            continue
        for key in ("chunk_id", "page", "source", "text", "entities"):
            if key not in stmt:
                errors.append(f"statements[{i}] missing key '{key}'")
        if "entities" in stmt and not isinstance(stmt["entities"], list):
            errors.append(f"statements[{i}].entities not a list")

    for i, triple in enumerate(table_triples):
        if not isinstance(triple, dict):
            errors.append(f"table_triples[{i}] not a dict")
            continue
        if "entities" not in triple:
            errors.append(f"table_triples[{i}] missing key 'entities'")

    return {
        "valid": len(errors) == 0,
        "error_count": len(errors),
        "errors": errors[:50],
        "total_statements": len(statements),
        "total_table_triples": len(table_triples),
    }


# ---------------------------------------------------------------------------
# Entity metrics (statements)
# ---------------------------------------------------------------------------

def entity_metrics(statements: List[Dict]) -> Dict[str, Any]:
    """Compute entity coverage and quality from statements."""
    total_entities = 0
    statements_with_entities = 0
    entities_per_statement: List[int] = []
    all_scores: List[float] = []
    label_counts: Counter = Counter()
    biomedical_entity_count = 0
    unique_entity_texts: set = set()

    for stmt in statements:
        entities = stmt.get("entities") or []
        n = len(entities)
        total_entities += n
        if n > 0:
            statements_with_entities += 1
            entities_per_statement.append(n)
        for ent in entities:
            if isinstance(ent, dict):
                score = ent.get("score")
                if score is not None:
                    try:
                        all_scores.append(float(score))
                    except (TypeError, ValueError):
                        pass
                label = (ent.get("label") or "").strip()
                if label:
                    label_counts[label] += 1
                    if label in BIOMEDICAL_LABELS:
                        biomedical_entity_count += 1
                text = (ent.get("text") or "").strip().lower()
                if text:
                    unique_entity_texts.add(text)

    n_stmts = len(statements)
    return {
        "total_entities": total_entities,
        "statements_with_entities": statements_with_entities,
        "statements_without_entities": n_stmts - statements_with_entities,
        "entity_coverage_ratio": statements_with_entities / max(1, n_stmts),
        "entities_per_statement_avg": total_entities / max(1, n_stmts),
        "entities_per_statement_max": max(entities_per_statement) if entities_per_statement else 0,
        "unique_entity_texts": len(unique_entity_texts),
        "label_distribution": dict(label_counts),
        "biomedical_entity_count": biomedical_entity_count,
        "biomedical_ratio": biomedical_entity_count / total_entities if total_entities else 0.0,
        "score_avg": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "score_min": min(all_scores) if all_scores else 0.0,
        "score_max": max(all_scores) if all_scores else 0.0,
    }


# ---------------------------------------------------------------------------
# Table triples enrichment metrics
# ---------------------------------------------------------------------------

def table_triples_metrics(table_triples: List[Dict]) -> Dict[str, Any]:
    """Measure how well table triples are enriched with NER entities."""
    total_triples = len(table_triples)
    triples_with_entities = 0
    total_entities_in_triples = 0
    triples_entity_scores: List[float] = []

    for triple in table_triples:
        entities = triple.get("entities") or []
        n = len(entities)
        if n > 0:
            triples_with_entities += 1
            total_entities_in_triples += n
            for ent in entities:
                if isinstance(ent, dict) and ent.get("score") is not None:
                    try:
                        triples_entity_scores.append(float(ent["score"]))
                    except (TypeError, ValueError):
                        pass

    return {
        "total_triples": total_triples,
        "triples_with_entities": triples_with_entities,
        "triples_without_entities": total_triples - triples_with_entities,
        "enrichment_ratio": triples_with_entities / max(1, total_triples),
        "total_entities_in_triples": total_entities_in_triples,
        "entities_per_triple_avg": total_entities_in_triples / max(1, total_triples),
        "triple_entity_score_avg": sum(triples_entity_scores) / len(triples_entity_scores) if triples_entity_scores else 0.0,
    }


# ---------------------------------------------------------------------------
# Scoring (0–1), Stage C only
# ---------------------------------------------------------------------------

def score_schema(schema_valid: bool) -> float:
    return 1.0 if schema_valid else 0.0


def score_entity_coverage(metrics: Dict[str, Any], schema_ok: bool) -> float:
    """Score based on how many statements have entities and entity density."""
    if not schema_ok:
        return 0.0
    total = metrics.get("total_entities", 0)
    coverage = metrics.get("entity_coverage_ratio", 0.0)
    # We want both: entities present and good coverage across statements
    density_score = min(1.0, total / 2000)  # 2000+ entities is strong
    coverage_score = coverage  # fraction of statements with >=1 entity
    return 0.5 * density_score + 0.5 * coverage_score


def score_biomedical_relevance(metrics: Dict[str, Any], schema_ok: bool) -> float:
    """Score based on fraction of entities that are biomedical (vs generic PER/ORG)."""
    if not schema_ok:
        return 0.0
    total = metrics.get("total_entities", 0)
    if total == 0:
        return 0.0  # No entities = no biomedical relevance
    bio_ratio = metrics.get("biomedical_ratio", 0.0)
    return bio_ratio


def score_entity_quality(metrics: Dict[str, Any], schema_ok: bool) -> float:
    """Score based on entity confidence scores and uniqueness."""
    if not schema_ok:
        return 0.0
    total = metrics.get("total_entities", 0)
    if total == 0:
        return 0.0
    score_avg = metrics.get("score_avg", 0.0)
    unique = metrics.get("unique_entity_texts", 0)
    # Higher avg score = more confident; more unique = less repetition
    score_component = min(1.0, score_avg)  # 0.55+ is our min, so 0.8+ is good
    uniqueness = min(1.0, unique / max(1, total))  # ideally unique ≈ total
    return 0.7 * score_component + 0.3 * uniqueness


def score_table_enrichment(triple_metrics: Dict[str, Any], schema_ok: bool) -> float:
    """Score based on how many table triples have NER entities."""
    if not schema_ok:
        return 0.0
    total = triple_metrics.get("total_triples", 0)
    if total == 0:
        return 0.5  # No triples to enrich
    enrichment = triple_metrics.get("enrichment_ratio", 0.0)
    return enrichment


def composite_score(
    schema_score: float,
    entity_coverage_score: float,
    biomedical_score: float,
    entity_quality_score: float,
    table_enrichment_score: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Composite for Stage C: schema, entity coverage, biomedical relevance, quality, table enrichment."""
    w = weights or {
        "schema": 0.15,
        "entity_coverage": 0.30,
        "biomedical": 0.25,
        "entity_quality": 0.15,
        "table_enrichment": 0.15,
    }
    return (
        w["schema"] * schema_score
        + w["entity_coverage"] * entity_coverage_score
        + w["biomedical"] * biomedical_score
        + w["entity_quality"] * entity_quality_score
        + w["table_enrichment"] * table_enrichment_score
    )


# ---------------------------------------------------------------------------
# Evaluate one version
# ---------------------------------------------------------------------------

def evaluate_one_version(version: str, path: Path) -> Dict[str, Any]:
    metadata, statements, table_triples = load_stage_c_output(path)
    if metadata is None:
        return {
            "version": version,
            "error": "stage_c_statements_with_entities.json not found",
            "schema_valid": False,
        }

    schema = validate_schema(metadata, statements, table_triples)
    entity_met = entity_metrics(statements)
    triple_met = table_triples_metrics(table_triples)

    schema_ok = schema["valid"]
    s_schema = score_schema(schema_ok)
    s_coverage = score_entity_coverage(entity_met, schema_ok)
    s_biomedical = score_biomedical_relevance(entity_met, schema_ok)
    s_quality = score_entity_quality(entity_met, schema_ok)
    s_table = score_table_enrichment(triple_met, schema_ok)
    composite = composite_score(s_schema, s_coverage, s_biomedical, s_quality, s_table)

    return {
        "version": version,
        "metadata": metadata,
        "schema": schema,
        "entity_metrics": entity_met,
        "table_triples_metrics": triple_met,
        "scores": {
            "schema": round(s_schema, 4),
            "entity_coverage": round(s_coverage, 4),
            "biomedical_relevance": round(s_biomedical, 4),
            "entity_quality": round(s_quality, 4),
            "table_enrichment": round(s_table, 4),
            "composite": round(composite, 4),
        },
    }


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_report(reports: List[Dict[str, Any]], paths_evaluated: Dict[str, Path]) -> None:
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    by_version = {r["version"]: r for r in reports}
    comparison: Dict[str, Any] = {}
    for ver, r in by_version.items():
        comparison[f"{ver}_composite"] = r.get("scores", {}).get("composite", 0)
        em = r.get("entity_metrics", {})
        comparison[f"{ver}_total_entities"] = em.get("total_entities", 0)
        comparison[f"{ver}_statements_with_entities"] = em.get("statements_with_entities", 0)
        comparison[f"{ver}_biomedical_ratio"] = em.get("biomedical_ratio", 0)
        comparison[f"{ver}_entity_coverage"] = em.get("entity_coverage_ratio", 0)
        tm = r.get("table_triples_metrics", {})
        comparison[f"{ver}_triples_with_entities"] = tm.get("triples_with_entities", 0)
        comparison[f"{ver}_enrichment_ratio"] = tm.get("enrichment_ratio", 0)

    report = {
        "evaluation": "stage_c",
        "versions": by_version,
        "comparison": comparison,
    }
    json_path = EVAL_OUTPUT_DIR / "stage_c_evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Wrote {json_path}")

    run_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    report_lines = [
        "=" * 70,
        "STAGE C EVALUATION REPORT",
        "=" * 70,
        "",
        f"Run at: {run_time}",
        f"Versions evaluated: {', '.join(r['version'] for r in reports)}",
        "",
        "USE CASE",
        "-" * 70,
        "Stage C runs NER on text chunks and table triples. Downstream we use:",
        "  - Entities in statements: for linking to UMLS, fact extraction, recommendations.",
        "  - Entities in table triples: for structured SPO with medical concepts.",
        "We want: (1) good entity coverage across statements; (2) biomedical relevance",
        "(Disease_disorder, Medication, etc. vs generic PER/ORG); (3) high confidence",
        "scores; (4) table triples enriched with entities.",
        "",
        "WHAT WE MEASURE",
        "-" * 70,
        "  Schema           Output has correct structure (statements, entities, table_triples).",
        "  Entity coverage  How many statements have >=1 entity; total entity count.",
        "  Biomedical       Fraction of entities with biomedical labels (vs PER/ORG/LOC/MISC).",
        "  Entity quality   Avg confidence score; uniqueness of entity texts.",
        "  Table enrichment Fraction of table triples that have NER entities.",
        "  Composite       Weighted combination of the above.",
        "",
        "Inputs evaluated (paths that were read):",
    ]
    for ver in [r["version"] for r in reports]:
        p = paths_evaluated.get(ver)
        report_lines.append(f"  {ver}: {p}" if p else f"  {ver}: (not found)")
    report_lines.extend([
        "",
        "RESULTS",
        "-" * 70,
        "",
        "Composite scores (0-1, higher is better):",
    ])
    for r in reports:
        report_lines.append(f"  {r['version']}: {r.get('scores', {}).get('composite', 0):.4f}")
    report_lines.extend([
        "",
        "Score breakdown (schema | coverage | biomedical | quality | table_enrichment):",
    ])
    for r in reports:
        s = r.get("scores", {})
        report_lines.append(
            "  {}  {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f}".format(
                r["version"],
                s.get("schema", 0),
                s.get("entity_coverage", 0),
                s.get("biomedical_relevance", 0),
                s.get("entity_quality", 0),
                s.get("table_enrichment", 0),
            )
        )
    report_lines.extend([
        "",
        "Entity metrics (statements):",
        "  Metric                    Meaning",
        "  total_entities            Total NER entities across all statements.",
        "  statements_with_entities  Statements that have >=1 entity.",
        "  entity_coverage_ratio     Fraction of statements with entities.",
        "  biomedical_ratio          Fraction of entities with biomedical labels.",
        "  score_avg                 Average entity confidence score.",
    ])
    for r in reports:
        em = r.get("entity_metrics", {})
        report_lines.append(
            "  {}  entities: {},  stmts_with_entities: {},  coverage: {:.2%},  biomedical: {:.2%},  score_avg: {:.3f}".format(
                r["version"],
                em.get("total_entities", 0),
                em.get("statements_with_entities", 0),
                em.get("entity_coverage_ratio", 0),
                em.get("biomedical_ratio", 0),
                em.get("score_avg", 0),
            )
        )
    report_lines.extend([
        "",
        "Table triples enrichment:",
        "  Metric                Meaning",
        "  triples_with_entities  Table triples that have >=1 NER entity.",
        "  enrichment_ratio      Fraction of triples with entities.",
    ])
    for r in reports:
        tm = r.get("table_triples_metrics", {})
        report_lines.append(
            "  {}  triples_with_entities: {},  enrichment_ratio: {:.2%}".format(
                r["version"],
                tm.get("triples_with_entities", 0),
                tm.get("enrichment_ratio", 0),
            )
        )
    report_lines.extend([
        "",
        "Label distribution (entity types):",
    ])
    for r in reports:
        em = r.get("entity_metrics", {})
        dist = em.get("label_distribution", {})
        if dist:
            report_lines.append(f"  {r['version']}: {dict(dist)}")
        else:
            report_lines.append(f"  {r['version']}: (no entities)")

    report_lines.extend([
        "",
        "ANALYSIS",
        "-" * 70,
    ])
    if len(reports) >= 2:
        best = max(reports, key=lambda r: r.get("scores", {}).get("composite", 0))
        report_lines.append(f"  Highest composite: {best['version']} ({best.get('scores', {}).get('composite', 0):.4f})")
        for r in reports:
            em = r.get("entity_metrics", {})
            meta = r.get("metadata", {})
            report_lines.append(
                f"  {r['version']} model: {meta.get('neural_model', '?')}; "
                f"entities: {em.get('total_entities', 0)}, biomedical_ratio: {em.get('biomedical_ratio', 0):.2%}"
            )
    report_lines.extend([
        "",
        "Output files written:",
        f"  - {json_path} (full metrics and scores)",
        f"  - {EVAL_OUTPUT_DIR / 'stage_c_evaluation_report.txt'} (this report)",
        "",
        "=" * 70,
        "Evaluation report complete.",
        "=" * 70,
    ])

    report_path = EVAL_OUTPUT_DIR / "stage_c_evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Wrote {report_path}")
    print("\n" + "\n".join(report_lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage C evaluation: analyze and score NER outputs for one or more versions."
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        default=DEFAULT_VERSIONS,
        help="Version names to evaluate (e.g. v1 v2 v3). Each reads outputs/STAGE_C_{version}/stage_c_statements_with_entities.json",
    )
    args = parser.parse_args()
    versions = args.versions

    reports = []
    paths_evaluated: Dict[str, Path] = {}
    for ver in versions:
        path = OUTPUTS_DIR / f"STAGE_C_{ver}" / "stage_c_statements_with_entities.json"
        paths_evaluated[ver] = path
        print(f"Loading STAGE_C_{ver}...")
        report = evaluate_one_version(ver, path)
        reports.append(report)
        if "error" in report:
            print(f"  Warning: {report['error']}")

    write_report(reports, paths_evaluated)


if __name__ == "__main__":
    main()
