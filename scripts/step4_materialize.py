"""
Step 4 — KG Materialization  [SYMBOLIC — RML engine]
─────────────────────────────────────────────────────────────────────────────
Input  : M = { outputs/step3/table_mappings.ttl,
               outputs/step3/text_mappings_<version>.ttl }
         S = { table CSVs, text_assertions_<version>.json }
         T-Box : input/hf_guideline_ontology.ttl
Output : outputs/step4/output_<version>.ttl   (A-Box + T-Box merged)

Tool   : Morph-KGC (primary)  — pip install morph-kgc
         rdflib fallback        (always available)
─────────────────────────────────────────────────────────────────────────────
CUI linking:
  For every entity with a cui_final, emit:
    <inst:cui_CXXXXXXX> a ex:CUI ;
        ex:cuiCode "<CUI>" .
    <inst:entity_X> ex:hasUMLSConcept <inst:cui_CXXXXXXX> .
  CUIs are RDF resources — never plain literals.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
from pathlib import Path

from rdflib import Graph, Namespace, RDF, Literal, URIRef

from utils import log

EX   = Namespace("http://digistructmed.org/ontology/")
INST = Namespace("http://digistructmed.org/instance/")


# ── CUI linking ─────────────────────────────────────────────────────────────

def _add_cui_triples(g: Graph, resolved_entities_path: str) -> int:
    """
    Emit ex:CUI instances and ex:hasUMLSConcept links for all resolved entities.
    CUIs are RDF resources, not string literals.
    """
    entities: list[dict] = json.loads(
        Path(resolved_entities_path).read_text(encoding="utf-8")
    )
    added = 0
    for ent in entities:
        cui = ent.get("cui_final")
        if not cui:
            continue

        # Sanitise entity text for URI
        ent_slug = ent["text"].replace(" ", "_").replace("/", "_")[:40]
        ent_uri  = INST[f"entity_{ent_slug}"]
        cui_uri  = INST[f"cui_{cui}"]

        g.add((cui_uri, RDF.type, EX.CUI))
        g.add((cui_uri, EX.cuiCode, Literal(cui)))
        g.add((ent_uri, EX.hasUMLSConcept, cui_uri))
        added += 1

    return added


# ── Morph-KGC materializer ──────────────────────────────────────────────────

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


# ── rdflib fallback materializer ────────────────────────────────────────────

def _rdflib_materialize(
    text_assertions_path: str,
) -> Graph:
    """
    Minimal fallback: read text_assertions JSON and emit role triples directly.
    Does NOT process table CSVs (use Morph-KGC for full materialisation).
    """
    g = Graph()
    g.bind("ex", EX)
    g.bind("inst", INST)

    assertions: list[dict] = json.loads(
        Path(text_assertions_path).read_text(encoding="utf-8")
    )
    for a in assertions:
        s_cui = a.get("subject_cui") or a.get("subject_text", "unknown")
        o_cui = a.get("object_cui")  or a.get("object_text", "unknown")
        pred  = URIRef(a.get("predicate_uri", str(EX.relatedTo)))

        s_uri = INST[f"cui_{s_cui}"]
        o_uri = INST[f"cui_{o_cui}"]
        g.add((s_uri, pred, o_uri))

    log("4", f"rdflib fallback: {len(g)} role triples from text assertions")
    return g


# ── Main ────────────────────────────────────────────────────────────────────

def materialize(
    table_mappings_path: str    = "outputs/step3/table_mappings.ttl",
    text_mappings_path: str     = "outputs/step3/text_mappings_v1.ttl",
    text_assertions_path: str   = "outputs/step3/text_assertions_v1.json",
    resolved_entities_path: str = "outputs/step1/resolved_entities.json",
    ontology_path: str          = "input/hf_guideline_ontology.ttl",
    output_dir: str             = "outputs/step4",
    version: str                = "v1",
) -> str:
    """
    Execute M on S → RDF A-Box; merge with T-Box; add CUI links.

    Returns path to the output Turtle file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(out_dir / f"output_{version}.ttl")

    log("4", f"Materialising KG [{version}]")

    # ── Try Morph-KGC first ────────────────────────────────────────────────
    try:
        import morph_kgc  # noqa: F401
        g = _morph_materialize(
            table_mappings_path, text_mappings_path, output_path, str(out_dir)
        )
    except ImportError:
        log("4", "Morph-KGC not installed — using rdflib fallback")
        log("4", "  Install with: pip install morph-kgc")
        g = _rdflib_materialize(text_assertions_path)

    # ── Merge T-Box ────────────────────────────────────────────────────────
    tbox = Graph()
    tbox.parse(ontology_path, format="turtle")
    g += tbox
    log("4", f"T-Box merged: {len(tbox)} triples")

    # ── Add CUI instance triples ───────────────────────────────────────────
    n_cui = _add_cui_triples(g, resolved_entities_path)
    log("4", f"CUI triples added: {n_cui}")

    # ── Serialise ──────────────────────────────────────────────────────────
    g.bind("ex",   EX)
    g.bind("inst", INST)
    g.serialize(output_path, format="turtle")

    log("4", f"Output KG: {len(g)} total triples → {output_path}")
    return output_path


if __name__ == "__main__":
    import sys
    ver = sys.argv[1] if len(sys.argv) > 1 else "v1"
    materialize(
        text_mappings_path=f"outputs/step3/text_mappings_{ver}.ttl",
        text_assertions_path=f"outputs/step3/text_assertions_{ver}.json",
        version=ver,
    )
