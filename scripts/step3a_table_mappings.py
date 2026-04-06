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

def _is_metadata_header(header: str, ontology: "OntologyIndex") -> bool:
    """
    Detect metadata column headers that are not ontology properties.
    Uses structural heuristics instead of a hardcoded list.
    """
    h = header.lower().strip().rstrip(":")
    tokens = set(h.split())

    metadata_indicators = {"reference", "source", "footnote", "abbreviation"}
    if tokens & metadata_indicators:
        return True

    if h in {"notes", "comments", "organization", "title"}:
        return True

    if tokens >= {"meaning", "phrase"} or tokens >= {"care", "setting"}:
        return True

    if "measure" in tokens and any(w in tokens for w in ("title", "no", "no.", "domain")):
        return True

    all_prop_labels = {p.label.lower() for p in ontology.all_properties().values()}
    if h not in all_prop_labels and len(tokens) == 1 and h not in {
        c.get("label", "").lower() for c in ontology.classes.values()
    }:
        pass

    return False

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


def _csv_sample_values(csv_path: str, header: str, n: int = 3) -> list[str]:
    """Return up to n non-empty sample values from a column."""
    samples: list[str] = []
    try:
        with Path(csv_path).open(newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                val = (row.get(header) or "").strip()
                if val and val not in samples:
                    samples.append(val)
                    if len(samples) >= n:
                        break
    except Exception:
        pass
    return samples


def _llm_match_columns(
    unmatched: list[dict],
    ontology: OntologyIndex,
    llm_backend: str = "hf_local",
    hf_token: str | None = None,
    hf_model: str = "meta-llama/Llama-3.1-8B-Instruct"
) -> list[dict]:
    """
    Use the LLM to match table columns that fuzzy-matching could not resolve.
    For each unmatched column, the LLM sees the column header, sample values,
    and the full list of ontology properties, and must pick the best match
    or respond NONE.
    """
    if not unmatched:
        return []

    try:
        from hf_llm import (
            probe_hf_local_backend,
            hf_local_generate,
            hf_inference_chat,
        )
    except ImportError:
        log("3a", "  LLM fallback: hf_llm module not available — skipping")
        return []

    if llm_backend in ("hf_local", "llama"):
        probe_err = probe_hf_local_backend(hf_model, hf_token)
        if probe_err:
            log("3a", f"  LLM fallback: probe failed — {probe_err}")
            return []

    def _query(prompt: str) -> str:
        if llm_backend in ("hf_local", "llama"):
            return hf_local_generate(prompt, hf_model, hf_token, max_new_tokens=512)
        return hf_inference_chat(prompt, hf_model, hf_token, max_new_tokens=512)

    prop_list = []
    for uri, prop in ontology.all_properties().items():
        domain_label = ", ".join(
            ontology.class_label(d) for d in prop.domain
        ) or "any"
        range_label = ", ".join(
            ontology.class_label(r) if prop.prop_type == "object"
            else r.split("#")[-1]
            for r in prop.range_
        ) or "any"
        prop_list.append(
            f"  URI: {uri}\n  label: {prop.label}\n"
            f"  type: {prop.prop_type}\n  domain: {domain_label}\n"
            f"  range: {range_label}"
        )
    props_text = "\n---\n".join(prop_list)

    matched: list[dict] = []
    batch_size = 5
    for i in range(0, len(unmatched), batch_size):
        batch = unmatched[i:i + batch_size]
        columns_text = ""
        for item in batch:
            samples = _csv_sample_values(item["csv_path"], item["header"])
            columns_text += (
                f"\nColumn: \"{item['header']}\"\n"
                f"Table: {item['table_id']}\n"
                f"Sample values: {samples}\n"
            )

        prompt = (
            f"You are mapping table columns to ontology properties.\n\n"
            f"ONTOLOGY PROPERTIES:\n{props_text}\n\n"
            f"COLUMNS TO MATCH:{columns_text}\n\n"
            f"For each column, respond with the best matching property URI "
            f"or NONE if no property fits.\n"
            f"Respond ONLY as a JSON array of objects:\n"
            f'[{{"column": "<header>", "table": "<table_id>", '
            f'"property_uri": "<URI or NONE>"}}]\n\nJSON:'
        )

        try:
            raw = _query(prompt)
            import json as _json
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                proposals = _json.loads(raw[start:end])
                for prop in proposals:
                    uri = prop.get("property_uri", "NONE")
                    if uri == "NONE" or not uri:
                        continue
                    onto_prop = ontology.all_properties().get(uri)
                    if not onto_prop:
                        continue
                    col_header = prop.get("column", "")
                    tbl_id = prop.get("table", "")
                    for item in batch:
                        if item["header"] == col_header and item["table_id"] == tbl_id:
                            matched.append({
                                "table_id": tbl_id,
                                "csv_path": item["csv_path"],
                                "header": col_header,
                                "match": {
                                    "uri": uri,
                                    "label": onto_prop.label,
                                    "prop_type": onto_prop.prop_type,
                                    "range_": onto_prop.range_,
                                    "score": 0.60,
                                },
                                "source": "llm",
                            })
                            break
        except Exception as exc:
            log("3a", f"  LLM batch {i // batch_size + 1} failed: {exc}")

    log("3a", f"  LLM fallback: {len(matched)} additional column(s) matched out of {len(unmatched)} attempted")
    return matched


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


# ── Enum-subject detection ─────────────────────────────────────────────────

def _compute_common_prefix(strings: list[str]) -> str:
    """Find the longest common prefix among a list of strings (case-insensitive)."""
    if not strings:
        return ""
    lower = [s.lower() for s in strings]
    prefix = lower[0]
    for s in lower[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    last_space = prefix.rfind(" ")
    if last_space > 0:
        return prefix[:last_space + 1]
    return prefix


def _build_enum_label_index(
    ontology: OntologyIndex,
) -> dict[str, tuple[str, str]]:
    """
    Build a lookup: normalised-label → (member_uri, enum_class_uri).
    Dynamically detects the common prefix of member labels within each enum
    class and strips it to create short-form keys for CSV value matching.
    """
    index: dict[str, tuple[str, str]] = {}
    for enum_cls, member_uris in ontology.enumerations.items():
        labels: list[str] = []
        for mu in member_uris:
            info = ontology.named_individuals.get(mu, {})
            label = info.get("label", "")
            if not label:
                label = mu.split("/")[-1].split("#")[-1]
            labels.append(label)

        common_prefix = _compute_common_prefix(labels) if len(labels) > 1 else ""

        for mu, label in zip(member_uris, labels):
            norm = label.strip().lower()
            index[norm] = (mu, enum_cls)

            if common_prefix and norm.startswith(common_prefix):
                short = norm[len(common_prefix):].strip()
                if short:
                    index[short] = (mu, enum_cls)
    return index


def _match_csv_value_to_enum(
    value: str,
    enum_index: dict[str, tuple[str, str]],
) -> tuple[str, str] | None:
    """
    Return (member_uri, enum_class_uri) if the CSV cell matches an enum member.
    Uses exact match against the full label or the short (prefix-stripped) form.
    No substring matching — that causes false positives with short enum labels.
    """
    norm = value.strip().lower()
    if not norm:
        return None
    if norm in enum_index:
        return enum_index[norm]
    norm_nospace = re.sub(r"[\s\-_:]+", "", norm)
    for label, info in enum_index.items():
        if re.sub(r"[\s\-_:]+", "", label) == norm_nospace:
            return info
    return None


def _detect_enum_subject(
    csv_path: str,
    csv_headers: list[str],
    ontology: OntologyIndex,
) -> dict | None:
    """
    Check if the subject column (col 0) contains ontology enum values.

    Returns a dict with restructuring info if detected:
      { "enum_columns": [{col_idx, header, prop_uri, enum_class, member_map}, ...],
        "text_columns": [{col_idx, header}, ...] }
    Or None if this is a normal table.
    """
    if not ontology.enumerations or len(csv_headers) < 2:
        return None

    enum_index = _build_enum_label_index(ontology)
    if not enum_index:
        return None

    p = Path(csv_path)
    if not p.is_file():
        return None

    try:
        with p.open(newline="", encoding="utf-8", errors="replace") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None

    if not rows:
        return None

    enum_columns: list[dict] = []
    text_columns: list[dict] = []

    for col_idx, header in enumerate(csv_headers):
        if not header.strip():
            continue

        values = [
            (r.get(header) or "").strip()
            for r in rows[:20]
            if (r.get(header) or "").strip()
        ]
        if not values:
            text_columns.append({"col_idx": col_idx, "header": header})
            continue

        matches = 0
        member_map: dict[str, str] = {}
        class_counts: dict[str, int] = {}
        for v in values:
            result = _match_csv_value_to_enum(v, enum_index)
            if result:
                matches += 1
                member_map[v] = result[0]
                class_counts[result[1]] = class_counts.get(result[1], 0) + 1

        if not class_counts:
            text_columns.append({"col_idx": col_idx, "header": header})
            continue

        dominant_class = max(class_counts, key=class_counts.get)
        dominant_count = class_counts[dominant_class]

        distinct_members = len(set(member_map.values()))
        is_enum = (
            dominant_count / len(values) >= 0.8
            and len(class_counts) == 1
            and distinct_members >= 2
        )

        if is_enum:
            prop_uri = _find_property_for_enum_range(dominant_class, ontology)
            enum_columns.append({
                "col_idx": col_idx,
                "header": header,
                "prop_uri": prop_uri,
                "enum_class": dominant_class,
                "member_map": member_map,
            })
        else:
            text_columns.append({"col_idx": col_idx, "header": header})

    if not enum_columns:
        return None

    subject_is_enum = any(ec["col_idx"] == 0 for ec in enum_columns)
    if not subject_is_enum:
        return None

    distinct_enum_classes = {ec["enum_class"] for ec in enum_columns}
    if len(distinct_enum_classes) < 2 and len(enum_columns) > 1:
        return None

    return {
        "enum_columns": enum_columns,
        "text_columns": text_columns,
    }


def _find_property_for_enum_range(
    enum_class_uri: str,
    ontology: OntologyIndex,
) -> str | None:
    """Find the object property whose rdfs:range is the given enum class."""
    for uri, prop in ontology.object_properties.items():
        if enum_class_uri in prop.range_:
            return uri
    return None


def _subject_column_is_enum(
    csv_path: str,
    subject_header: str,
    ontology: OntologyIndex,
) -> bool:
    """
    Check if the subject column (col 0) contains only ontology enum values
    like COR codes ("1", "2a", "2b") or LOE codes ("A", "B-R").
    These should not be used as entity identifiers — they produce
    resources like inst:1 or inst:2a that merge different recommendations.

    Returns True if ≥80% of sampled subject values are enum members.
    """
    if not ontology.enumerations:
        return False
    enum_index = _build_enum_label_index(ontology)
    if not enum_index:
        return False

    p = Path(csv_path)
    if not p.is_file():
        return False
    try:
        with p.open(newline="", encoding="utf-8", errors="replace") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return False

    values = [
        (r.get(subject_header) or "").strip()
        for r in rows[:20]
        if (r.get(subject_header) or "").strip()
    ]
    if not values:
        return False

    matches = sum(1 for v in values if _match_csv_value_to_enum(v, enum_index))
    return matches / len(values) >= 0.8


def _extract_section_slug(headers: list[str]) -> str:
    """
    Extract a meaningful section name from the table column headers by
    finding their longest common prefix (the shared section title) and
    slugifying it. Fully dynamic — no hardcoded suffixes or prefixes.
    """
    if not headers:
        return ""

    common = _compute_common_prefix(headers)
    if not common or len(common) < 4:
        return re.sub(r"[^a-zA-Z0-9]+", "_", headers[0]).strip("_").lower()[:60]

    common = common.rstrip(". \t")

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", common).strip("_").lower()

    if len(slug) > 60:
        slug = slug[:60].rsplit("_", 1)[0]

    return slug if len(slug) > 3 else ""


def _extract_rec_number(text: str) -> str:
    """
    Extract the recommendation number from the start of recommendation text.
    "1. For patients with stage C HF..." → "1"
    "8. For patients who have LVEF ≤ 35%..." → "8"
    """
    m = re.match(r"^\s*(\d+)\.", text)
    return m.group(1) if m else ""


def _restructure_enum_table(
    table_id: str,
    csv_path: str,
    enum_info: dict,
    ontology: OntologyIndex,
) -> tuple[str, dict | None]:
    """
    Rewrite the CSV to add a _subject_id column and generate an RML mapping
    that treats enum columns as object properties pointing to named individuals.

    Subject IDs are meaningful: rec_{section}_{number}, e.g.
    "rec_dietary_sodium_restriction_1" for recommendation 1 from the
    Dietary Sodium Restriction section.

    Returns (rml_block, mapping_index_entry) or ("", None) on failure.
    """
    p = Path(csv_path)
    try:
        with p.open(newline="", encoding="utf-8", errors="replace") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return "", None

    if not rows:
        return "", None

    enum_columns = enum_info["enum_columns"]
    text_columns = enum_info["text_columns"]

    domain_class = None
    for ec in enum_columns:
        if ec["prop_uri"]:
            prop = ontology.object_properties.get(ec["prop_uri"])
            if prop and prop.domain:
                domain_class = prop.domain[0]
                break

    enum_index = _build_enum_label_index(ontology)
    enum_col_headers = {ec["header"] for ec in enum_columns}

    original_headers = list(rows[0].keys()) if rows else []
    section_slug = _extract_section_slug(original_headers)
    new_header = "_subject_id"
    new_csv_path = p.parent / f"{table_id}_restructured.csv"

    text_header = text_columns[0]["header"] if text_columns else None

    with new_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([new_header] + original_headers)
        for i, row in enumerate(rows):
            rec_num = ""
            if text_header:
                rec_num = _extract_rec_number(row.get(text_header, ""))

            if section_slug and rec_num:
                subj_id = f"rec_{section_slug}_{rec_num}"
            elif section_slug:
                subj_id = f"rec_{section_slug}_{i}"
            else:
                subj_id = f"rec_{table_id}_{rec_num or str(i)}"

            out_vals = [subj_id]
            for h in original_headers:
                raw = row.get(h, "")
                if h in enum_col_headers:
                    match_result = _match_csv_value_to_enum(raw, enum_index)
                    if match_result:
                        uri_local = match_result[0].split("/")[-1].split("#")[-1]
                        out_vals.append(uri_local)
                    else:
                        out_vals.append(raw)
                else:
                    out_vals.append(raw)
            writer.writerow(out_vals)

    src = new_csv_path.resolve().as_posix()
    subj_t = _morph_instance_template(new_header)

    poms: list[str] = []

    if domain_class:
        poms.append(
            f'    rr:predicateObjectMap [\n'
            f'        rr:predicate <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ;\n'
            f'        rr:objectMap  [ rr:constant <{domain_class}> ]\n'
            f'    ] ;'
        )

    ont_ns = "http://digistructmed.org/ontology/"
    for uri in ontology.classes:
        ont_ns = uri.rsplit("/", 1)[0] + "/"
        break

    for ec in enum_columns:
        if not ec["prop_uri"]:
            continue
        header = ec["header"]
        enum_template = ont_ns + "{" + header + "}"
        poms.append(
            f'    rr:predicateObjectMap [\n'
            f'        rr:predicate <{ec["prop_uri"]}> ;\n'
            f'        rr:objectMap  [ rr:template "{_ttl_dquote(enum_template)}" ]\n'
            f'    ] ;'
        )

    text_col_props: list[dict] = []
    for tc in text_columns:
        header = tc["header"]
        cref = _ttl_dquote(header)
        best_match = match_header(header, ontology)
        if best_match:
            prop_uri = best_match["uri"]
        else:
            prop_uri = None
            best_score = 0.0
            for uri, prop in ontology.datatype_properties.items():
                score = SequenceMatcher(None, header.lower(), prop.label.lower()).ratio()
                if score > best_score:
                    best_score = score
                    prop_uri = uri
            if not prop_uri:
                prop_uri = next(iter(ontology.datatype_properties), header)

        text_col_props.append({"header": header, "prop_uri": prop_uri})
        poms.append(
            f'    rr:predicateObjectMap [\n'
            f'        rr:predicate <{prop_uri}> ;\n'
            f'        rr:objectMap  [ rml:reference "{cref}" ;\n'
            f'                        rr:datatype <http://www.w3.org/2001/XMLSchema#string> ]\n'
            f'    ] ;'
        )

    pom_block = "\n".join(poms)
    rml = (
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

    n_enum = len(enum_columns)
    n_text = len(text_columns)
    mapping_entry = {
        "table_id": table_id,
        "csv_path": str(new_csv_path),
        "subject_header": new_header,
        "restructured": True,
        "enum_columns": n_enum,
        "text_columns": n_text,
        "domain_class": domain_class,
        "columns": [
            {"header": ec["header"], "predicate_uri": ec["prop_uri"],
             "prop_type": "object", "range": [ec["enum_class"]]}
            for ec in enum_columns if ec["prop_uri"]
        ] + [
            {"header": tcp["header"],
             "predicate_uri": tcp["prop_uri"],
             "prop_type": "datatype",
             "range": ["http://www.w3.org/2001/XMLSchema#string"]}
            for tcp in text_col_props
        ],
    }

    log("3a", f"  {table_id}: enum-subject detected → restructured "
        f"({n_enum} enum cols, {n_text} text cols, domain={ontology.class_label(domain_class) if domain_class else '?'})")

    return rml, mapping_entry


# ── COR/LOE/Recommendation 3-column detector ──────────────────────────────

def _detect_cor_loe_rec_table(
    csv_path: str,
    csv_headers: list[str],
    ontology: "OntologyIndex",
) -> tuple[str, dict | None]:
    """
    Detect tables whose columns match the pattern *.COR | *.LOE | *.Recommendation(s).
    These are recommendation tables that the enum detector misses because they
    have no text-subject column to anchor restructuring.

    Returns (rml_block, mapping_entry) or ("", None) if not a COR/LOE/Rec table.
    """
    if len(csv_headers) < 2:
        return "", None

    norm = [h.strip().lower().rstrip(".") for h in csv_headers]
    suffixes = [n.rsplit(".", 1)[-1] if "." in n else n for n in norm]

    cor_idx = loe_idx = rec_idx = None
    for i, suf in enumerate(suffixes):
        if suf == "cor":
            cor_idx = i
        elif suf == "loe":
            loe_idx = i
        elif suf in ("recommendation", "recommendations"):
            rec_idx = i

    if cor_idx is None and loe_idx is None:
        return "", None
    if rec_idx is None and cor_idx is not None and loe_idx is not None:
        for i, suf in enumerate(suffixes):
            if i not in (cor_idx, loe_idx) and suf not in ("cor", "loe"):
                rec_idx = i
                break

    if cor_idx is None or loe_idx is None:
        return "", None

    p = Path(csv_path)
    try:
        with p.open(newline="", encoding="utf-8", errors="replace") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return "", None

    if not rows:
        return "", None

    table_id = p.stem.replace("_restructured", "")
    section_slug = _extract_section_slug(csv_headers)

    new_header = "_subject_id"
    new_csv_path = p.parent / f"{table_id}_rec.csv"
    cor_header = csv_headers[cor_idx]
    loe_header = csv_headers[loe_idx]
    rec_header = csv_headers[rec_idx] if rec_idx is not None else None

    enum_index = _build_enum_label_index(ontology)

    cor_enum_classes = set()
    for mu, (_, ecls) in enum_index.items():
        if "ClassOfRecommendation" in ecls:
            cor_enum_classes.add(mu)

    skipped_subheaders = 0
    rec_row_idx = 0

    with new_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        out_headers = [new_header, cor_header, loe_header]
        if rec_header:
            out_headers.append(rec_header)
        writer.writerow(out_headers)

        for i, row in enumerate(rows):
            cor_val = (row.get(cor_header) or "").strip()
            cor_match = _match_csv_value_to_enum(cor_val, enum_index)

            if not cor_match:
                skipped_subheaders += 1
                continue

            rec_num = ""
            if rec_header:
                rec_num = _extract_rec_number(row.get(rec_header, ""))

            if section_slug and rec_num:
                subj_id = f"rec_{section_slug}_{rec_num}"
            elif section_slug:
                subj_id = f"rec_{section_slug}_{rec_row_idx}"
            else:
                subj_id = f"rec_{table_id}_{rec_num or str(rec_row_idx)}"
            rec_row_idx += 1

            cor_val = cor_match[0].split("/")[-1].split("#")[-1]

            loe_val = (row.get(loe_header) or "").strip()
            loe_match = _match_csv_value_to_enum(loe_val, enum_index)
            if loe_match:
                loe_val = loe_match[0].split("/")[-1].split("#")[-1]

            out_vals = [subj_id, cor_val, loe_val]
            if rec_header:
                out_vals.append(row.get(rec_header, ""))
            writer.writerow(out_vals)

    if skipped_subheaders:
        log("3a", f"    skipped {skipped_subheaders} section subheader row(s) (COR not in enum)")

    ont_ns = "http://digistructmed.org/ontology/"
    rec_cls = ont_ns + "Recommendation"
    src = new_csv_path.resolve().as_posix()
    subj_t = _morph_instance_template(new_header)

    poms = []
    poms.append(
        f'    rr:predicateObjectMap [\n'
        f'        rr:predicate <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ;\n'
        f'        rr:objectMap  [ rr:constant <{rec_cls}> ]\n'
        f'    ] ;'
    )
    cor_template = ont_ns + "{" + cor_header + "}"
    poms.append(
        f'    rr:predicateObjectMap [\n'
        f'        rr:predicate <{ont_ns}hasCOR> ;\n'
        f'        rr:objectMap  [ rr:template "{_ttl_dquote(cor_template)}" ]\n'
        f'    ] ;'
    )
    loe_template = ont_ns + "{" + loe_header + "}"
    poms.append(
        f'    rr:predicateObjectMap [\n'
        f'        rr:predicate <{ont_ns}hasLOE> ;\n'
        f'        rr:objectMap  [ rr:template "{_ttl_dquote(loe_template)}" ]\n'
        f'    ] ;'
    )
    if rec_header:
        cref = _ttl_dquote(rec_header)
        poms.append(
            f'    rr:predicateObjectMap [\n'
            f'        rr:predicate <{ont_ns}recommendationText> ;\n'
            f'        rr:objectMap  [ rml:reference "{cref}" ;\n'
            f'                        rr:datatype <http://www.w3.org/2001/XMLSchema#string> ]\n'
            f'    ] ;'
        )

    pom_block = "\n".join(poms)
    rml = (
        f"<#{table_id}_RecMap>\n"
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

    n_cols = 2 + (1 if rec_header else 0)
    mapping_entry = {
        "table_id": table_id,
        "csv_path": str(new_csv_path),
        "subject_header": new_header,
        "restructured": True,
        "recommendation_table": True,
        "domain_class": rec_cls,
        "columns": [
            {"header": cor_header, "predicate_uri": ont_ns + "hasCOR",
             "prop_type": "object", "range": [ont_ns + "ClassOfRecommendation"]},
            {"header": loe_header, "predicate_uri": ont_ns + "hasLOE",
             "prop_type": "object", "range": [ont_ns + "LevelOfEvidence"]},
        ] + ([
            {"header": rec_header, "predicate_uri": ont_ns + "recommendationText",
             "prop_type": "datatype", "range": ["http://www.w3.org/2001/XMLSchema#string"]}
        ] if rec_header else []),
    }

    log("3a", f"  {table_id}: COR/LOE/Rec table detected -> restructured "
        f"({n_cols} cols, {len(rows)} recs)")
    return rml, mapping_entry


# ── Main ───────────────────────────────────────────────────────────────────

def generate_table_mappings(
    table_index_path: str = "outputs/step1/table_index.json",
    ontology: OntologyIndex | None = None,
    ontology_path: str = "input/hf_guideline_ontology.ttl",
    output_dir: str = "outputs/step3",
    llm_backend: str = "hf_local",
    hf_token: str | None = None,
    hf_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    use_llm_fallback: bool = True,
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

        # COR/LOE/Rec 3-column recommendation tables (before enum detection)
        rec_rml, rec_entry = _detect_cor_loe_rec_table(csv_path, csv_headers, ontology)
        if rec_rml and rec_entry:
            rml_blocks.append(rec_rml)
            all_table_mappings.append(rec_entry)
            accepted_count += len(rec_entry.get("columns", []))
            continue

        enum_info = _detect_enum_subject(csv_path, csv_headers, ontology)

        if enum_info is None and _subject_column_is_enum(csv_path, subject_header, ontology):
            log("3a", f"  skip {table_id}: subject column contains enum values "
                f"(not entity identifiers) — header: {subject_header[:60]}")
            continue
        if enum_info is not None:
            rml_block, mapping_entry = _restructure_enum_table(
                table_id, csv_path, enum_info, ontology,
            )
            if rml_block and mapping_entry:
                rml_blocks.append(rml_block)
                all_table_mappings.append(mapping_entry)
                n_enum_cols = len(enum_info["enum_columns"])
                n_text_cols = len(enum_info["text_columns"])
                accepted_count += n_enum_cols + n_text_cols
            continue

        fallback_headers: list[str] = table.get("headers") or []

        col_mappings: list[dict] = []

        for col_idx, header in enumerate(csv_headers):
            if not header.strip() or col_idx == 0:
                continue

            if _is_metadata_header(header, ontology):
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

            if match["prop_type"] == "object" and _column_has_freetext(csv_path, header):
                rejected.append({"table": table_id, "header": header,
                                 "best_candidate": match,
                                 "reason": "freetext_in_object_property_column"})
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

    # ── LLM fallback for unmatched columns ─────────────────────────────────
    if use_llm_fallback:
        no_match = [r for r in rejected if r.get("reason") == "no_ontology_match"]
        if no_match:
            llm_unmatched = []
            table_csv_map: dict[str, str] = {}
            table_subj_map: dict[str, str] = {}
            for table in table_index:
                table_csv_map[table["table_id"]] = table["csv_path"]
                headers = _read_csv_headers(table["csv_path"])
                if headers:
                    table_subj_map[table["table_id"]] = headers[0].strip()

            for r in no_match:
                tid = r["table"]
                if tid in table_csv_map and tid in table_subj_map:
                    llm_unmatched.append({
                        "table_id": tid,
                        "csv_path": table_csv_map[tid],
                        "header": r["header"],
                        "subject_header": table_subj_map[tid],
                    })

            llm_matches = _llm_match_columns(
                llm_unmatched, ontology,
                llm_backend=llm_backend, hf_token=hf_token, hf_model=hf_model,
            )

            for lm in llm_matches:
                match = lm["match"]
                if match["prop_type"] == "object" and _column_has_freetext(lm["csv_path"], lm["header"]):
                    continue

                tid = lm["table_id"]
                subj_h = lm.get("subject_header", table_subj_map.get(tid, ""))

                existing = next((m for m in all_table_mappings if m["table_id"] == tid), None)
                col_entry = {
                    "header": lm["header"],
                    "predicate_uri": match["uri"],
                    "prop_type": match["prop_type"],
                    "range": match["range_"],
                }
                if existing:
                    existing["columns"].append(col_entry)
                else:
                    all_table_mappings.append({
                        "table_id": tid,
                        "csv_path": lm["csv_path"],
                        "subject_header": subj_h,
                        "columns": [col_entry],
                    })

                rml_block = _rml_triples_map(
                    tid, lm["csv_path"],
                    [{"header": lm["header"], "match": match}],
                    subject_header=subj_h,
                )
                if rml_block:
                    rml_blocks.append(rml_block)
                accepted_count += 1

            rejected = [r for r in rejected if r.get("reason") != "no_ontology_match"
                        or not any(lm["header"] == r["header"] and lm["table_id"] == r["table"]
                                   for lm in llm_matches)]

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
