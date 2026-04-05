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
import re
from difflib import SequenceMatcher
from pathlib import Path

from step2_load_ontology import OntologyIndex, PropertyInfo
from utils import log, save_json

# Similarity thresholds — lowered from Run 1 defaults (0.85/0.55) because
# guideline headers like "Initial Daily Dose" vs property "initial dose"
# need a softer accept gate.  The review gate catches borderline matches.
ACCEPT_THRESHOLD = 0.72     # auto-include in RML
REVIEW_THRESHOLD = 0.50     # flag for human review
REJECT_BELOW    = 0.50     # discard silently

_NON_PROPERTY_HEADERS = {
    "reference", "references", "reference/link", "reference link",
    "organization", "source", "notes", "comments", "footnote",
    "abbreviation", "meaning/phrase", "meaning", "measure title",
    "care setting", "title",
}

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

    Uses both full-string similarity and token overlap to handle cases like
    "Initial Daily Dose" matching "initial dose" (token overlap is high even
    though the full string similarity is moderate).

    Returns dict with keys: uri, label, prop_type, range_, score.
    """
    best_score = 0.0
    best = None
    h_tokens = set(re.findall(r"[a-z]{3,}", header.lower()))

    for uri, prop in ontology.all_properties().items():
        for candidate_text in [prop.label, prop.comment]:
            if not candidate_text:
                continue
            s = _sim(header, candidate_text)

            if h_tokens:
                c_tokens = set(re.findall(r"[a-z]{3,}", candidate_text.lower()))
                shared = h_tokens & c_tokens
                if shared and c_tokens:
                    token_score = len(shared) / max(len(h_tokens), len(c_tokens))
                    s = max(s, token_score * 0.85)

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


def _column_has_freetext(csv_path: str, header: str, sample_rows: int = 10) -> bool:
    """
    Check if a CSV column contains free-text values (spaces, long strings,
    punctuation) that cannot be used directly in URI templates.
    Morph-KGC crashes when it tries to coerce text like "Critical
    cardiogenic shock" into a numeric URI component.
    """
    p = Path(csv_path)
    if not p.is_file():
        return False
    try:
        with p.open(newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= sample_rows:
                    break
                val = (row.get(header) or "").strip()
                if not val:
                    continue
                if " " in val or len(val) > 60 or re.search(r"[,;:!?(){}\[\]]", val):
                    return True
    except Exception:
        return False
    return False


def _rml_triples_map(
    table_id: str,
    csv_path: str,
    col_mappings: list[dict],
    subject_header: str,
) -> str:
    """
    Emit one RML TriplesMap.  CSV column names must match the file on disk.

    Object-property columns whose CSV values are free text are downgraded to
    xsd:string literals so Morph-KGC doesn't crash trying to cast text into
    numeric URI components.
    """
    if not col_mappings:
        return ""

    src = Path(csv_path).resolve().as_posix()
    subj_t = _morph_instance_template(subject_header)

    poms: list[str] = []
    for cm in col_mappings:
        match = cm["match"]
        href = cm["header"]
        cref = _ttl_dquote(href)

        is_object = match["prop_type"] == "object"
        force_literal = is_object and _column_has_freetext(csv_path, href)

        if is_object and not force_literal:
            ot = _morph_instance_template(href)
            poms.append(
                f'    rr:predicateObjectMap [\n'
                f'        rr:predicate <{match["uri"]}> ;\n'
                f'        rr:objectMap  [ rr:template "{_ttl_dquote(ot)}" ]\n'
                f'    ] ;'
            )
        else:
            xsd_type = match["range_"][0] if match["range_"] else "http://www.w3.org/2001/XMLSchema#string"
            if is_object or xsd_type not in (
                "http://www.w3.org/2001/XMLSchema#string",
                "http://www.w3.org/2001/XMLSchema#integer",
                "http://www.w3.org/2001/XMLSchema#decimal",
                "http://www.w3.org/2001/XMLSchema#boolean",
            ):
                xsd_type = "http://www.w3.org/2001/XMLSchema#string"
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
        f'        rr:template "{_ttl_dquote(subj_t)}"\n'
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
    all_table_mappings: list[dict] = []

    for table in table_index:
        table_id = table["table_id"]
        csv_path = table["csv_path"]
        # Ground truth for Morph-KGC / pandas: first row of the CSV on disk (spaces, not slugs).
        csv_headers = _read_csv_headers(csv_path)
        if not csv_headers:
            log("3a", f"  skip {table_id}: no CSV or empty header row → {csv_path}")
            continue

        subject_header = csv_headers[0].strip()
        if not subject_header:
            log("3a", f"  skip {table_id}: empty first-column header (no subject for RML)")
            continue
        fallback_headers: list[str] = table.get("headers") or []

        col_mappings: list[dict] = []

        for col_idx, header in enumerate(csv_headers):
            if not header.strip() or col_idx == 0:
                continue

            if header.lower().strip() in _NON_PROPERTY_HEADERS:
                rejected.append({"table": table_id, "header": header, "reason": "non_property_header"})
                continue

            match = match_header(header, ontology)

            if match is None:
                rejected.append({"table": table_id, "header": header, "reason": "no_ontology_match"})
                continue

            numeric_ranges = {
                "http://www.w3.org/2001/XMLSchema#integer",
                "http://www.w3.org/2001/XMLSchema#decimal",
                "http://www.w3.org/2001/XMLSchema#float",
                "http://www.w3.org/2001/XMLSchema#double",
                "http://www.w3.org/2001/XMLSchema#nonNegativeInteger",
            }
            if match["range_"] and set(match["range_"]) & numeric_ranges:
                if _column_has_freetext(csv_path, header):
                    rejected.append({"table": table_id, "header": header,
                                     "best_candidate": match, "reason": "text_column_vs_numeric_range"})
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

        if col_mappings:
            all_table_mappings.append({
                "table_id": table_id,
                "csv_path": csv_path,
                "subject_header": subject_header,
                "columns": [
                    {"header": cm["header"], "predicate_uri": cm["match"]["uri"],
                     "prop_type": cm["match"]["prop_type"],
                     "range": cm["match"]["range_"]}
                    for cm in col_mappings
                ],
            })

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

    # Save complete mapping index for rdflib fallback (Step 4 uses this when
    # Morph-KGC is unavailable or crashes on complex CSV values).
    mapping_index_path = out_dir / "table_mapping_index.json"
    save_json(all_table_mappings, str(mapping_index_path))

    log("3a", f"RML written → {rml_path}")
    log("3a", f"Mapping index → {mapping_index_path}")
    if review_required:
        log("3a", f"  ⚠  {len(review_required)} column(s) need human review → {review_path}")
    if rejected:
        log("3a", f"  ✗  {len(rejected)} column(s) rejected")

    return {
        "rml_path": str(rml_path),
        "review_path": str(review_path),
        "mapping_index_path": str(mapping_index_path),
        "counts": {
            "accepted": accepted_count,
            "review": len(review_required),
            "rejected": len(rejected),
        },
    }


if __name__ == "__main__":
    generate_table_mappings()
