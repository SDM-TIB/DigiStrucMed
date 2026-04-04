"""
Step 3a — Table Mapping Generation  [SYMBOLIC — deterministic]
─────────────────────────────────────────────────────────────────────────────
Input  : outputs/step1/table_index.json    (from Step 1b)
         OntologyIndex                      (from Step 2)
Output : outputs/step3/table_mappings.ttl          (valid RML)
         outputs/step3/table_mapping_review.json   (ambiguous → review gate)

Method : Fuzzy-match each column header against ontology property labels
         (rdfs:label / rdfs:comment).  Three outcome buckets:
           score ≥ ACCEPT_THRESHOLD  → include in RML automatically
           REVIEW_THRESHOLD ≤ score < ACCEPT  → flag for human review
           score < REVIEW_THRESHOLD  → reject (no match)
         After review, the human edits table_mappings.ttl before Step 4.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import csv
import json
from difflib import SequenceMatcher
from pathlib import Path

from step2_load_ontology import OntologyIndex, PropertyInfo
from utils import log, save_json

# Similarity thresholds (tune if needed)
ACCEPT_THRESHOLD = 0.85     # auto-include in RML
REVIEW_THRESHOLD = 0.55     # flag for human review
REJECT_BELOW    = 0.55     # discard silently

# ── RML header ─────────────────────────────────────────────────────────────

RML_HEADER = """\
@prefix rr:  <http://www.w3.org/ns/r2rml#> .
@prefix rml: <http://semweb.mmlab.be/ns/rml#> .
@prefix ql:  <http://semweb.mmlab.be/ns/ql#> .
@prefix ex:  <http://digistructmed.org/ontology/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

"""


# ── Fuzzy matching ──────────────────────────────────────────────────────────

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def match_header(
    header: str,
    ontology: OntologyIndex,
) -> dict | None:
    """
    Return the best-matching ontology property for a column header, or None.

    Returns dict with keys: uri, label, prop_type, range_, score.
    """
    best_score = 0.0
    best = None

    for uri, prop in ontology.all_properties().items():
        for candidate_text in [prop.label, prop.comment]:
            if not candidate_text:
                continue
            s = _sim(header, candidate_text)
            if s > best_score:
                best_score = s
                best = {
                    "uri": uri,
                    "label": prop.label,
                    "prop_type": prop.prop_type,
                    "range_": prop.range_,
                    "score": round(s, 4),
                }

    return best if (best and best_score >= REJECT_BELOW) else None


# ── RML TriplesMap generator ────────────────────────────────────────────────

def _morph_instance_template(column_name: str) -> str:
    """Morph-KGC / R2RML template: one {ColumnName} placeholder matching pandas CSV header."""
    return "http://digistructmed.org/instance/{" + column_name + "}"


def _rml_triples_map(
    table_id: str,
    csv_path: str,
    col_mappings: list[dict],
    subject_header: str,
) -> str:
    """Emit one RML TriplesMap. CSV column names must match the file on disk (spaces, case)."""
    if not col_mappings:
        return ""

    src = Path(csv_path).resolve().as_posix()
    subj_t = _morph_instance_template(subject_header)

    poms: list[str] = []
    for cm in col_mappings:
        match = cm["match"]
        href = cm["header"]
        cref = _ttl_dquote(href)
        if match["prop_type"] == "object":
            ot = _morph_instance_template(href)
            poms.append(
                f'    rr:predicateObjectMap [\n'
                f'        rr:predicate <{match["uri"]}> ;\n'
                f'        rr:objectMap  [ rr:template "{_ttl_dquote(ot)}" ]\n'
                f'    ] ;'
            )
        else:
            xsd_type = match["range_"][0] if match["range_"] else str(
                "http://www.w3.org/2001/XMLSchema#string"
            )
            poms.append(
                f'    rr:predicateObjectMap [\n'
                f'        rr:predicate <{match["uri"]}> ;\n'
                f'        rr:objectMap  [ rml:reference "{cref}" ;\n'
                f'                        rr:datatype <{xsd_type}> ]\n'
                f'    ] ;'
            )

    pom_block = "\n".join(poms)
    return (
        f"<#{table_id}_Map>\n"
        f"    a rr:TriplesMap ;\n"
        f"    rml:logicalSource [\n"
        f'        rml:source             "{_ttl_dquote(src)}" ;\n'
        f"        rml:referenceFormulation ql:CSV\n"
        f"    ] ;\n"
        f"    rr:subjectMap [\n"
        f'        rr:template "{_ttl_dquote(subj_t)}" ;\n'
        f"        rr:class    ex:Entity\n"
        f"    ] ;\n"
        f"{pom_block}\n"
        f"    .\n"
    )


def _safe_ref(header: str) -> str:
    """Legacy slug (URI fragments only). Do not use for CSV column names."""
    return header.strip().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")


def _read_csv_headers(csv_path: str) -> list[str]:
    """First row of the CSV as written by Step 1b — source of truth for Morph-KGC."""
    p = Path(csv_path)
    if not p.is_file():
        return []
    with p.open(newline="", encoding="utf-8", errors="replace") as f:
        row = next(csv.reader(f), [])
    return [c.strip() for c in row]


def _ttl_dquote(s: str) -> str:
    """Escape a string for use inside Turtle double quotes."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


# ── Main ───────────────────────────────────────────────────────────────────

def generate_table_mappings(
    table_index_path: str = "outputs/step1/table_index.json",
    ontology: OntologyIndex | None = None,
    ontology_path: str = "input/hf_guideline_ontology.ttl",
    output_dir: str = "outputs/step3",
) -> dict:
    """
    Generate RML mappings for all extracted tables.

    Returns dict with keys: rml_path, review_path, counts.
    """
    if ontology is None:
        from step2_load_ontology import load_ontology
        ontology = load_ontology(ontology_path)

    table_index: list[dict] = json.loads(
        Path(table_index_path).read_text(encoding="utf-8")
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rml_blocks: list[str] = [RML_HEADER]
    review_required: list[dict] = []
    rejected: list[dict] = []
    accepted_count = 0

    for table in table_index:
        table_id = table["table_id"]
        csv_path = table["csv_path"]
        # Ground truth for Morph-KGC / pandas: first row of the CSV on disk (spaces, not slugs).
        csv_headers = _read_csv_headers(csv_path)
        if not csv_headers:
            log("3a", f"  skip {table_id}: no CSV or empty header row → {csv_path}")
            continue

        subject_header = csv_headers[0]
        fallback_headers: list[str] = table.get("headers") or []

        col_mappings: list[dict] = []

        for col_idx, header in enumerate(csv_headers):
            if not header.strip() or col_idx == 0:
                continue

            match = match_header(header, ontology)

            if match is None:
                rejected.append({"table": table_id, "header": header, "reason": "no_ontology_match"})
                continue

            if match["score"] < REVIEW_THRESHOLD:
                rejected.append({"table": table_id, "header": header,
                                  "best_candidate": match, "reason": "below_review_threshold"})
            elif match["score"] < ACCEPT_THRESHOLD:
                review_required.append({"table": table_id, "header": header, "match": match,
                                         "col_idx": col_idx})
                col_mappings.append({"header": header, "match": match})  # tentatively include
            else:
                col_mappings.append({"header": header, "match": match})
                accepted_count += 1

        if fallback_headers and [h.strip() for h in fallback_headers] != [
            h.strip() for h in csv_headers
        ]:
            log(
                "3a",
                f"  note {table_id}: table_index headers differ from CSV first row; using CSV.",
            )

        rml_block = _rml_triples_map(
            table_id, csv_path, col_mappings, subject_header=subject_header
        )
        if rml_block:
            rml_blocks.append(rml_block)

    rml_path = out_dir / "table_mappings.ttl"
    rml_path.write_text("\n".join(rml_blocks), encoding="utf-8")

    review_path = out_dir / "table_mapping_review.json"
    save_json(
        {"review_required": review_required, "rejected": rejected},
        str(review_path),
    )

    log("3a", f"RML written → {rml_path}")
    if review_required:
        log("3a", f"  ⚠  {len(review_required)} column(s) need human review → {review_path}")
    if rejected:
        log("3a", f"  ✗  {len(rejected)} column(s) rejected")

    return {
        "rml_path": str(rml_path),
        "review_path": str(review_path),
        "counts": {
            "accepted": accepted_count,
            "review": len(review_required),
            "rejected": len(rejected),
        },
    }


if __name__ == "__main__":
    generate_table_mappings()
