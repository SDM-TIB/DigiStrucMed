"""
Step 4 — KG Materialization  [SYMBOLIC — RML engine]
─────────────────────────────────────────────────────────────────────────────
Input  : M = { outputs/step3/table_mappings.ttl,
               outputs/step3/text_mappings_<version>.ttl }
         S = { table CSVs, text_assertions_<version>.json }
         T-Box : input/hf_guideline_ontology.ttl
Output : outputs/step4/output_<version>.ttl   (A-Box + T-Box merged)

Tool   : Morph-KGC (primary)  — pip install morph-kgc
         rdflib fallback        (always available, covers text + tables)
─────────────────────────────────────────────────────────────────────────────
CUI linking:
  For every entity with a cui_final, emit (deduplicated):
    <inst:cui_CXXXXXXX> a ex:CUI ;
        ex:cuiCode "<CUI>" .
    <inst:entity_X> ex:hasUMLSConcept <inst:cui_CXXXXXXX> .

Entity typing:
  Each entity gets an rdf:type from its best-matching ontology class
  (field "subject_class" / "object_class" in the assertion records,
  computed dynamically by Step 3b from the ontology — nothing hardcoded).
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from rdflib import Graph, Namespace, RDF, RDFS, Literal, URIRef

from utils import log

EX   = Namespace("http://digistructmed.org/ontology/")
INST = Namespace("http://digistructmed.org/instance/")


def _slug(text: str) -> str:
    """Create an IRI-safe slug from arbitrary text."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip())
    return s.strip("_")[:80] or "unknown"


# ── CUI linking (deduplicated) ────────────────────────────────────────────

def _add_cui_triples(g: Graph, resolved_entities_path: str) -> int:
    """
    Emit ex:CUI instances and ex:hasUMLSConcept links for all resolved
    entities.  CUIs are RDF resources, not string literals.
    Deduplicates: each (entity_slug, CUI) pair is emitted once.
    """
    entities: list[dict] = json.loads(
        Path(resolved_entities_path).read_text(encoding="utf-8")
    )
    seen_cui: set[str] = set()
    seen_link: set[tuple[str, str]] = set()
    added = 0

    for ent in entities:
        cui = ent.get("cui_final")
        if not cui:
            continue

        ent_slug = _slug(ent["text"])
        ent_uri = INST[f"entity_{ent_slug}"]
        cui_uri = INST[f"cui_{cui}"]

        if cui not in seen_cui:
            g.add((cui_uri, RDF.type, EX.CUI))
            g.add((cui_uri, EX.cuiCode, Literal(cui)))
            if ent.get("text"):
                g.add((cui_uri, EX.preferredName, Literal(ent["text"])))
            seen_cui.add(cui)

        link_key = (ent_slug, cui)
        if link_key not in seen_link:
            g.add((ent_uri, EX.hasUMLSConcept, cui_uri))
            seen_link.add(link_key)
            added += 1

    return added


# ── Entity typing from assertion records ──────────────────────────────────

_SKIP_TYPING_CLASSES: set[str] = set()


def _load_constrained_classes(ontology_path: str) -> set[str]:
    """
    Identify ontology classes that carry mandatory cardinality constraints
    (owl:qualifiedCardinality, owl:someValuesFrom).  Instances typed as
    these classes must satisfy those constraints — so we should only type
    entities when we are confident, not from fuzzy co-occurrence.
    """
    from rdflib import Graph as RG, RDF as R, RDFS as RS, OWL as O
    g = RG()
    g.parse(ontology_path, format="turtle")
    constrained: set[str] = set()
    for cls in g.subjects(R.type, O.Class):
        for restriction in g.objects(cls, RS.subClassOf):
            if (restriction, R.type, O.Restriction) not in g:
                continue
            for card_prop in (O.qualifiedCardinality, O.minQualifiedCardinality,
                              O.someValuesFrom):
                if g.value(restriction, card_prop) is not None:
                    constrained.add(str(cls))
                    break
    return constrained


def _add_entity_types(
    g: Graph,
    assertions: list[dict],
    ontology_path: str = "input/hf_guideline_ontology.ttl",
) -> int:
    """
    Add rdf:type triples for entity instances using the ontology class
    computed by Step 3b (fields subject_class / object_class).

    Skips typing for classes that carry mandatory cardinality constraints
    unless the assertion confidence is high (>= 0.70).  This prevents
    table-sourced or low-confidence entities from being typed as e.g.
    HeartFailure and then failing hasStage cardinality checks.
    """
    constrained = _load_constrained_classes(ontology_path)
    seen: set[tuple[str, str]] = set()
    added = 0

    for a in assertions:
        conf = a.get("confidence", 0)
        for role in ("subject", "object"):
            cui = a.get(f"{role}_cui")
            cls = a.get(f"{role}_class", "")
            text = a.get(f"{role}_text", "")
            if not cui or not cls:
                continue

            if cls in constrained and conf < 0.70:
                continue

            ent_slug = _slug(text) if text else _slug(cui)
            key = (ent_slug, cls)
            if key in seen:
                continue
            seen.add(key)
            ent_uri = INST[f"entity_{ent_slug}"]
            g.add((ent_uri, RDF.type, URIRef(cls)))
            added += 1

    return added


# ── Morph-KGC materializer ────────────────────────────────────────────────

def _morph_materialize(
    table_mappings: str,
    text_mappings: str,
    output_path: str,
    output_dir: str,
) -> Graph:
    import morph_kgc

    config = (
        "[CONFIGURATION]\n"
        f"output_file = {output_path}\n"
        "output_format = N-TRIPLES\n\n"
        "[TableMappings]\n"
        f"mappings = {table_mappings}\n\n"
        "[TextMappings]\n"
        f"mappings = {text_mappings}\n"
    )
    cfg_path = Path(output_dir) / "morph_config.ini"
    cfg_path.write_text(config)

    g = morph_kgc.materialize(str(cfg_path))
    log("4", f"Morph-KGC: {len(g)} triples materialised")
    return g


# ── rdflib fallback materializer (text + tables) ─────────────────────────

def _rdflib_materialize(
    text_assertions_path: str,
    table_mapping_index_path: str | None = None,
) -> Graph:
    """
    Fallback materializer using rdflib directly.  Handles:
      1. Text assertions JSON (role triples)
      2. Table CSVs via the mapping index saved by Step 3a
    """
    g = Graph()
    g.bind("ex", EX)
    g.bind("inst", INST)

    # 1. Text assertions
    if Path(text_assertions_path).is_file():
        assertions: list[dict] = json.loads(
            Path(text_assertions_path).read_text(encoding="utf-8")
        )
        for a in assertions:
            s_cui = a.get("subject_cui") or _slug(a.get("subject_text", "unknown"))
            o_cui = a.get("object_cui") or _slug(a.get("object_text", "unknown"))
            pred = URIRef(a.get("predicate_uri", str(EX.relatedTo)))
            s_uri = INST[f"entity_{_slug(a.get('subject_text', s_cui))}"]
            o_uri = INST[f"entity_{_slug(a.get('object_text', o_cui))}"]
            g.add((s_uri, pred, o_uri))

        log("4", f"rdflib fallback: {len(assertions)} role triples from text assertions")
    else:
        log("4", f"rdflib fallback: text assertions not found at {text_assertions_path}")

    # 2. Table CSVs via mapping index
    if table_mapping_index_path and Path(table_mapping_index_path).is_file():
        mapping_index: list[dict] = json.loads(
            Path(table_mapping_index_path).read_text(encoding="utf-8")
        )
        table_triples = 0
        for tm in mapping_index:
            csv_path = tm["csv_path"]
            if not Path(csv_path).is_file():
                log("4", f"rdflib fallback: CSV not found, skipping: {csv_path}")
                continue
            subj_header = tm["subject_header"]
            col_defs = tm["columns"]

            with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    subj_val = (row.get(subj_header) or "").strip()
                    if not subj_val:
                        continue
                    subj_uri = INST[_slug(subj_val)]

                    for col in col_defs:
                        cell = (row.get(col["header"]) or "").strip()
                        if not cell:
                            continue
                        pred = URIRef(col["predicate_uri"])
                        if col["prop_type"] == "object":
                            g.add((subj_uri, pred, INST[_slug(cell)]))
                        else:
                            g.add((subj_uri, pred, Literal(cell)))
                        table_triples += 1

        log("4", f"rdflib fallback: {table_triples} triples from table CSVs")
    else:
        log("4", "rdflib fallback: no table mapping index — table triples skipped")

    return g


# ── Main ──────────────────────────────────────────────────────────────────

def materialize(
    table_mappings_path: str    = "outputs/step3/table_mappings.ttl",
    text_mappings_path: str     = "outputs/step3/text_mappings_v1.ttl",
    text_assertions_path: str   = "outputs/step3/text_assertions_v1.json",
    resolved_entities_path: str = "outputs/step1/resolved_entities.json",
    ontology_path: str          = "input/hf_guideline_ontology.ttl",
    output_dir: str             = "outputs/step4",
    version: str                = "v1",
    table_mapping_index_path: str | None = "outputs/step3/table_mapping_index.json",
) -> str:
    """
    Execute M on S → RDF A-Box; merge with T-Box; add CUI links + entity types.

    Returns path to the output Turtle file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(out_dir / f"output_{version}.ttl")

    log("4", f"Materialising KG [{version}]")

    # ── Try Morph-KGC first, fall back to rdflib on any error ─────────────
    g: Graph | None = None
    try:
        import morph_kgc  # noqa: F401
        g = _morph_materialize(
            table_mappings_path, text_mappings_path, output_path, str(out_dir)
        )
    except ImportError:
        log("4", "Morph-KGC not installed — using rdflib fallback")
    except Exception as exc:
        log("4", f"Morph-KGC failed ({type(exc).__name__}: {exc}) — using rdflib fallback")

    if g is None:
        g = _rdflib_materialize(text_assertions_path, table_mapping_index_path)

    # ── Merge T-Box ───────────────────────────────────────────────────────
    tbox = Graph()
    tbox.parse(ontology_path, format="turtle")
    g += tbox
    log("4", f"T-Box merged: {len(tbox)} triples")

    # ── Add CUI instance triples (deduplicated) ──────────────────────────
    n_cui = _add_cui_triples(g, resolved_entities_path)
    log("4", f"CUI triples added: {n_cui}")

    # ── Add entity rdf:type from assertion class fields ───────────────────
    if Path(text_assertions_path).is_file():
        assertions = json.loads(
            Path(text_assertions_path).read_text(encoding="utf-8")
        )
        n_typed = _add_entity_types(g, assertions)
        log("4", f"Entity type triples added: {n_typed}")

    # ── Serialise (clean invalid URIs) ───────────────────────────────────
    # Morph-KGC can produce URIs like inst:{} from empty CSV cells.  These
    # crash rdflib's Turtle serializer.  Build a clean graph, keeping only
    # triples whose every term is a valid URI / Literal / BNode.
    g.bind("ex",   EX)
    g.bind("inst", INST)

    clean = Graph()
    clean.bind("ex", EX)
    clean.bind("inst", INST)
    dropped = 0
    for triple in g:
        ok = True
        for term in triple:
            if isinstance(term, URIRef):
                u = str(term)
                if "{" in u or "}" in u or not u or u.endswith("/instance/"):
                    ok = False
                    break
        if ok:
            clean.add(triple)
        else:
            dropped += 1
    if dropped:
        log("4", f"Dropped {dropped} triples with invalid URIs")

    try:
        clean.serialize(output_path, format="turtle")
    except Exception as e:
        log("4", f"Turtle serialization failed ({e}); trying N-Triples format")
        nt_path = output_path.replace(".ttl", ".nt")
        clean.serialize(nt_path, format="nt")
        clean2 = Graph()
        clean2.parse(nt_path, format="nt")
        clean2.bind("ex", EX)
        clean2.bind("inst", INST)
        clean2.serialize(output_path, format="turtle")

    log("4", f"Output KG: {len(clean)} total triples → {output_path}")
    return output_path


if __name__ == "__main__":
    import sys
    ver = sys.argv[1] if len(sys.argv) > 1 else "v1"
    materialize(
        text_mappings_path=f"outputs/step3/text_mappings_{ver}.ttl",
        text_assertions_path=f"outputs/step3/text_assertions_{ver}.json",
        version=ver,
    )
