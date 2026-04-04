"""
Step 5 — SHACL Validation  [SYMBOLIC]
─────────────────────────────────────────────────────────────────────────────
Input  : outputs/step4/output_<version>.ttl   (materialised KG from Step 4)
         SHACL shapes  (auto-derived from OWL ontology, saved as shapes.ttl)
Output : outputs/step5/validation_report_<version>.json
         { conforms, total_triples, total_violations, violations[], metrics{} }

Primary validator : TravSHACL  (SDM-TIB/Trav-SHACL, pip install travshacl)
  — validates an RDF graph via a SPARQL endpoint
  — intelligent shape-traversal order; detects violations early
  — requires: endpoint_url pointing to the KG loaded in a SPARQL server

Fallback validator : pySHACL  (pip install pyshacl)
  — used automatically when no SPARQL endpoint is configured
  — reads the KG directly from a .ttl file (works in Colab / CI without a
    running SPARQL server)
  — implements the same SHACL standard; results are equivalent

Shape derivation:
  owl:FunctionalProperty    → sh:maxCount 1
  rdfs:domain / range       → sh:class / sh:datatype
  owl:oneOf enumerations    → sh:in

  TravSHACL: https://github.com/SDM-TIB/Trav-SHACL
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from rdflib import Graph, Namespace, RDF, RDFS, OWL, XSD, BNode, URIRef, Literal

from utils import log, save_json

EX = Namespace("http://digistructmed.org/ontology/")
SH = Namespace("http://www.w3.org/ns/shacl#")


# ── Shape derivation ─────────────────────────────────────────────────────────

def derive_shapes(
    ontology_path: str = "input/hf_guideline_ontology.ttl",
    output_dir: str = "outputs/step5",
) -> str:
    """
    Auto-derive SHACL shapes from the OWL ontology.

    Conversions applied:
      rdfs:domain  → NodeShape + sh:property with sh:path
      rdfs:range   → sh:class (object prop) or sh:datatype (datatype prop)
      owl:FunctionalProperty → sh:maxCount 1
      owl:oneOf    → sh:in enumeration
    """
    g = Graph()
    g.parse(ontology_path, format="turtle")

    shapes = Graph()
    shapes.bind("sh",  SH)
    shapes.bind("ex",  EX)
    shapes.bind("xsd", XSD)
    shapes.bind("owl", OWL)

    # 1. rdfs:domain / rdfs:range → NodeShape per domain class
    for p in (
        list(g.subjects(RDF.type, OWL.ObjectProperty))
        + list(g.subjects(RDF.type, OWL.DatatypeProperty))
    ):
        prop_type = (
            "object" if (p, RDF.type, OWL.ObjectProperty) in g else "datatype"
        )
        for domain_cls in g.objects(p, RDFS.domain):
            shape_uri = URIRef(str(domain_cls) + "_Shape")
            prop_shape = BNode()

            shapes.add((shape_uri, RDF.type,        SH.NodeShape))
            shapes.add((shape_uri, SH.targetClass,  domain_cls))
            shapes.add((shape_uri, SH.property,     prop_shape))
            shapes.add((prop_shape, SH.path,         p))

            for range_val in g.objects(p, RDFS.range):
                if prop_type == "datatype" or str(range_val).startswith(str(XSD)):
                    shapes.add((prop_shape, SH.datatype, range_val))
                else:
                    shapes.add((prop_shape, SH["class"], range_val))

    # 2. owl:FunctionalProperty → sh:maxCount 1
    for p in g.subjects(RDF.type, OWL.FunctionalProperty):
        inner = BNode()
        fn_shape = BNode()
        shapes.add((fn_shape, RDF.type,    SH.NodeShape))
        shapes.add((fn_shape, SH.property, inner))
        shapes.add((inner, SH.path,        p))
        shapes.add((inner, SH.maxCount,    Literal(1)))

    # 3. owl:oneOf enumerations → sh:in
    for cls in g.subjects(RDF.type, OWL.Class):
        one_of_node = g.value(cls, OWL.oneOf)
        if not one_of_node:
            continue
        members: list[URIRef] = []
        node = one_of_node
        while node and str(node) != str(RDF.nil):
            first = g.value(node, RDF.first)
            if first:
                members.append(first)
            node = g.value(node, RDF.rest)
        if members:
            shape_uri = URIRef(str(cls) + "_EnumShape")
            shapes.add((shape_uri, RDF.type,       SH.NodeShape))
            shapes.add((shape_uri, SH.targetClass, cls))
            list_node = BNode()
            shapes.add((shape_uri, SH["in"], list_node))
            current = list_node
            for i, m in enumerate(members):
                shapes.add((current, RDF.first, m))
                if i < len(members) - 1:
                    nxt = BNode()
                    shapes.add((current, RDF.rest, nxt))
                    current = nxt
                else:
                    shapes.add((current, RDF.rest, RDF.nil))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shapes_path = out_dir / "shapes.ttl"
    shapes.serialize(str(shapes_path), format="turtle")
    log("5", f"Derived {len(shapes)} shape triples → {shapes_path}")
    return str(shapes_path)


# ── TravSHACL backend ────────────────────────────────────────────────────────

def _validate_travshacl(
    shapes_dir: str,
    endpoint_url: str,
    version: str,
) -> list[dict]:
    """
    Run TravSHACL against a SPARQL endpoint.

    It requires:
      shapes_dir   — directory containing one .ttl file per SHACL shape
      endpoint_url — SPARQL endpoint URL where the KG is loaded
                     e.g. "http://localhost:3030/kg/sparql" (Apache Jena Fuseki)

    Returns a list of violation dicts (empty list = conforms).
    """
    try:
        from TravSHACL import GraphTraversal, ShapeSchema
        from TravSHACL.core.GraphTraversal import BFSOrder
    except ImportError:
        raise ImportError(
            "TravSHACL not installed. Run: pip install travshacl\n"
            "GitHub: https://github.com/SDM-TIB/Trav-SHACL"
        )

    schema = ShapeSchema(
        shapeDir=shapes_dir,
        endpoint=endpoint_url,
        endpointType="SPARQLEndpoint",
        graphTraversal=BFSOrder.ORDER,
        # heuristics: order shapes to detect violations as early as possible
        useSelectiveQueries=True,
        maxSplit=256,
        outputDir=None,
        ORDERBYinQueries=True,
        SHACL2SPARQLorder=False,
    )

    result = GraphTraversal(BFSOrder.ORDER, schema).traverse_graph()

    violations: list[dict] = []
    for shape_name, report in result.items():
        for entity in report.get("invalid", []):
            violations.append({
                "focus_node":  str(entity),
                "result_path": shape_name,
                "severity":    str(SH.Violation),
                "message":     f"Entity does not satisfy shape: {shape_name}",
            })
    return violations


# ── pySHACL backend (fallback) ───────────────────────────────────────────────

def _validate_pyshacl(
    kg: Graph,
    shacl_graph: Graph,
) -> tuple[bool, list[dict]]:
    """
    Run pySHACL against an in-memory rdflib Graph.

    Used as the fallback when no SPARQL endpoint is available (Colab / CI).
    pySHACL implements the same SHACL W3C standard as TravSHACL.
    """
    try:
        import pyshacl
    except ImportError:
        raise ImportError(
            "pySHACL not installed. Run: pip install pyshacl"
        )

    conforms, results_graph, _ = pyshacl.validate(
        kg,
        shacl_graph=shacl_graph,
        inference="rdfs",
        abort_on_first=False,
    )

    violations: list[dict] = []
    for result in results_graph.subjects(RDF.type, SH.ValidationResult):
        violations.append({
            "focus_node":  str(results_graph.value(result, SH.focusNode)      or ""),
            "result_path": str(results_graph.value(result, SH.resultPath)     or ""),
            "severity":    str(results_graph.value(result, SH.resultSeverity) or ""),
            "message":     str(results_graph.value(result, SH.resultMessage)  or ""),
        })
    return conforms, violations


# ── Public entry-point ───────────────────────────────────────────────────────

def validate(
    kg_path: str,
    shapes_path: Optional[str] = None,
    ontology_path: str = "input/hf_guideline_ontology.ttl",
    output_dir: str = "outputs/step5",
    version: str = "v1",
    # TravSHACL option — leave None to use pySHACL fallback
    sparql_endpoint: Optional[str] = None,
) -> dict:
    """
    Validate the materialised KG against SHACL shapes.

    Backend selection
    -----------------
    sparql_endpoint is set  → TravSHACL.
      The KG must already be loaded into the endpoint (e.g. via Jena Fuseki).
      shapes_path must point to a *directory* of per-shape .ttl files for
      TravSHACL's shape-directory convention.

    sparql_endpoint is None → pySHACL (local .ttl file, no server needed).
      Works in Colab / CI out of the box.

    If shapes_path is None in either case, shapes are auto-derived from the
    OWL ontology first.
    """
    if shapes_path is None:
        shapes_path = derive_shapes(ontology_path, output_dir)

    log("5", f"Validating [{version}]: {kg_path}")

    if sparql_endpoint:
        # ── TravSHACL path ──────────────────────────────────────────────────
        log("5", f"Backend: TravSHACL  endpoint={sparql_endpoint}")
        # TravSHACL expects a directory; if shapes_path is a file, use its parent
        shapes_dir = (
            shapes_path if Path(shapes_path).is_dir()
            else str(Path(shapes_path).parent)
        )
        violations = _validate_travshacl(shapes_dir, sparql_endpoint, version)
        conforms   = len(violations) == 0
        kg         = Graph().parse(kg_path, format="turtle")
        n_triples  = len(kg)
        backend    = "TravSHACL"

    else:
        # ── pySHACL fallback path ───────────────────────────────────────────
        log("5", "Backend: pySHACL (fallback — no SPARQL endpoint provided)")
        kg = Graph()
        kg.parse(kg_path, format="turtle")

        shacl_graph = Graph()
        shacl_graph.parse(shapes_path, format="turtle")

        conforms, violations = _validate_pyshacl(kg, shacl_graph)
        n_triples = len(kg)
        backend   = "pySHACL"

    report = {
        "version":          version,
        "backend":          backend,
        "kg_path":          kg_path,
        "sparql_endpoint":  sparql_endpoint,
        "conforms":         conforms,
        "total_triples":    n_triples,
        "total_violations": len(violations),
        "violations":       violations,
        "metrics": {
            "violation_rate":    round(len(violations) / max(n_triples, 1), 4),
            "conformance_ratio": round(1.0 - len(violations) / max(n_triples, 1), 4),
        },
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"validation_report_{version}.json"
    save_json(report, str(report_path))

    status = "✓ CONFORMS" if conforms else f"✗ {len(violations)} violations"
    log("5", f"[{version}] {backend}  {status} — {n_triples} triples  →  {report_path}")
    return report


# ── Comparison helper ────────────────────────────────────────────────────────

def compare_versions(
    report_v1_path: str = "outputs/step5/validation_report_v1.json",
    report_v2_path: str = "outputs/step5/validation_report_v2.json",
) -> None:
    """Print a side-by-side comparison of v1 vs v2 validation results."""
    r1 = json.loads(Path(report_v1_path).read_text(encoding="utf-8"))
    r2 = json.loads(Path(report_v2_path).read_text(encoding="utf-8"))

    print("\n" + "═" * 60)
    print(f"{'Metric':<30} {'v1':>14}  {'v2':>10}")
    print("─" * 60)
    for key in ("backend", "total_triples", "total_violations"):
        print(f"{key:<30} {str(r1.get(key, '—')):>14}  {str(r2.get(key, '—')):>10}")
    print(f"{'conformance_ratio':<30} "
          f"{r1['metrics'].get('conformance_ratio', '—'):>14}  "
          f"{r2['metrics'].get('conformance_ratio', '—'):>10}")
    print(f"{'conforms':<30} {str(r1.get('conforms', '—')):>14}  "
          f"{str(r2.get('conforms', '—')):>10}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    import sys
    ver      = sys.argv[1] if len(sys.argv) > 1 else "v1"
    endpoint = sys.argv[2] if len(sys.argv) > 2 else None   # e.g. http://localhost:3030/kg/sparql
    validate(
        kg_path=f"outputs/step4/output_{ver}.ttl",
        version=ver,
        sparql_endpoint=endpoint,
    )
