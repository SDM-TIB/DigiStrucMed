"""
Step 5 — SHACL Validation  [SYMBOLIC]
─────────────────────────────────────────────────────────────────────────────
Input  : outputs/step4/output_<version>.ttl   (materialised KG from Step 4)
         SHACL shapes  (auto-derived from OWL ontology, saved as shapes.ttl)
Output : outputs/step5/validation_report_<version>.json
         { conforms, total_triples, total_violations, violations[], metrics{} }

Primary validator : TravSHACL  (SDM-TIB/Trav-SHACL, pip install travshacl)
  — validates shapes against a SPARQL endpoint or an in-memory RDF graph
  — intelligent shape-traversal order; detects violations early
  — can use a local SPARQL endpoint OR an in-memory graph serialized to
    a temporary endpoint via rdflib's SPARQLStore

Fallback validator : pySHACL  (pip install pyshacl)
  — used automatically when TravSHACL is not installed
  — reads the KG directly from a .ttl file (works in Colab / CI without a
    running SPARQL server)
  — implements the same SHACL standard; results are equivalent

Shape derivation:
      owl:FunctionalProperty    → sh:maxCount 1
      rdfs:domain / range       → sh:class / sh:datatype
      owl:oneOf enumerations    → sh:in
      Note: the validation report separates quality violations (maxCount,
      type/class mismatches) from completeness violations (minCount) so
      that extracted-KG coverage gaps are reported but not conflated
      with real data errors

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
      owl:FunctionalProperty       → sh:maxCount 1
      owl:qualifiedCardinality     → sh:minCount + sh:maxCount
      owl:minQualifiedCardinality  → sh:minCount
      owl:maxQualifiedCardinality  → sh:maxCount
      owl:someValuesFrom           → sh:minCount 1
      owl:oneOf                    → sh:in enumeration
    """
    g = Graph()
    g.parse(ontology_path, format="turtle")

    shapes = Graph()
    shapes.bind("sh",  SH)
    shapes.bind("ex",  EX)
    shapes.bind("xsd", XSD)
    shapes.bind("owl", OWL)

    def _resolve_domain_classes(domain_node) -> list:
        """Resolve a domain value (may be blank-node union) to class URIs."""
        if isinstance(domain_node, BNode):
            union = g.value(domain_node, OWL.unionOf)
            if union:
                members = []
                cur = union
                while cur and str(cur) != str(RDF.nil):
                    first = g.value(cur, RDF.first)
                    if first:
                        members.append(first)
                    cur = g.value(cur, RDF.rest)
                return members
            return []
        return [domain_node]

    # 1. rdfs:domain / rdfs:range → NodeShape per domain class
    for p in (
        list(g.subjects(RDF.type, OWL.ObjectProperty))
        + list(g.subjects(RDF.type, OWL.DatatypeProperty))
    ):
        prop_type = (
            "object" if (p, RDF.type, OWL.ObjectProperty) in g else "datatype"
        )
        for raw_domain in g.objects(p, RDFS.domain):
            for domain_cls in _resolve_domain_classes(raw_domain):
                shape_uri = URIRef(str(domain_cls) + "_Shape")
                prop_shape = BNode()

                shapes.add((shape_uri, RDF.type,        SH.NodeShape))
                shapes.add((shape_uri, SH.targetClass,  domain_cls))
                shapes.add((shape_uri, SH.property,     prop_shape))
                shapes.add((prop_shape, SH.path,         p))

                for range_val in g.objects(p, RDFS.range):
                    if isinstance(range_val, BNode):
                        continue
                    if prop_type == "datatype" or str(range_val).startswith(str(XSD)):
                        if range_val not in (XSD.decimal, XSD.integer,
                                             XSD.nonNegativeInteger):
                            shapes.add((prop_shape, SH.datatype, range_val))
                    else:
                        shapes.add((prop_shape, SH["class"], range_val))

    # 2. owl:FunctionalProperty → sh:maxCount 1
    #    Fold into the NAMED shapes from step 1 (domain-based) so that
    #    TravSHACL never sees anonymous shapes without sh:targetClass.
    for p in g.subjects(RDF.type, OWL.FunctionalProperty):
        domain_classes = []
        for raw_domain in g.objects(p, RDFS.domain):
            domain_classes.extend(_resolve_domain_classes(raw_domain))

        if domain_classes:
            for domain_cls in domain_classes:
                shape_uri = URIRef(str(domain_cls) + "_Shape")
                shapes.add((shape_uri, RDF.type,       SH.NodeShape))
                shapes.add((shape_uri, SH.targetClass, domain_cls))
                inner = BNode()
                shapes.add((shape_uri, SH.property, inner))
                shapes.add((inner, SH.path,     p))
                shapes.add((inner, SH.maxCount, Literal(1)))
        else:
            log("5", f"  FunctionalProperty {p} has no rdfs:domain — skipping maxCount shape")

    # 3. owl:Restriction → cardinality + someValuesFrom shapes
    for cls in g.subjects(RDF.type, OWL.Class):
        for restriction in g.objects(cls, RDFS.subClassOf):
            if not isinstance(restriction, BNode):
                continue
            if (restriction, RDF.type, OWL.Restriction) not in g:
                continue

            on_prop = g.value(restriction, OWL.onProperty)
            if on_prop is None:
                continue

            shape_uri = URIRef(str(cls) + "_Shape")
            shapes.add((shape_uri, RDF.type,       SH.NodeShape))
            shapes.add((shape_uri, SH.targetClass, cls))

            # owl:qualifiedCardinality → sh:minCount + sh:maxCount
            q_card = g.value(restriction, OWL.qualifiedCardinality)
            if q_card is not None:
                ps = BNode()
                shapes.add((shape_uri, SH.property, ps))
                shapes.add((ps, SH.path,     on_prop))
                shapes.add((ps, SH.minCount, Literal(int(q_card))))
                shapes.add((ps, SH.maxCount, Literal(int(q_card))))
                on_class = g.value(restriction, OWL.onClass)
                if on_class and not isinstance(on_class, BNode):
                    shapes.add((ps, SH["class"], on_class))

            # owl:minQualifiedCardinality
            min_qc = g.value(restriction, OWL.minQualifiedCardinality)
            if min_qc is not None:
                ps = BNode()
                shapes.add((shape_uri, SH.property, ps))
                shapes.add((ps, SH.path,     on_prop))
                shapes.add((ps, SH.minCount, Literal(int(min_qc))))
                on_class = g.value(restriction, OWL.onClass)
                if on_class and not isinstance(on_class, BNode):
                    shapes.add((ps, SH["class"], on_class))

            # owl:maxQualifiedCardinality
            max_qc = g.value(restriction, OWL.maxQualifiedCardinality)
            if max_qc is not None:
                ps = BNode()
                shapes.add((shape_uri, SH.property, ps))
                shapes.add((ps, SH.path,     on_prop))
                shapes.add((ps, SH.maxCount, Literal(int(max_qc))))

            # owl:someValuesFrom → sh:minCount 1
            some_from = g.value(restriction, OWL.someValuesFrom)
            if some_from is not None:
                ps = BNode()
                shapes.add((shape_uri, SH.property, ps))
                shapes.add((ps, SH.path,     on_prop))
                shapes.add((ps, SH.minCount, Literal(1)))
                if not isinstance(some_from, BNode):
                    shapes.add((ps, SH["class"], some_from))

    # 4. owl:oneOf enumerations → sh:in
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

def _split_shapes_to_dir(shapes_path: str, output_dir: str) -> str:
    """
    TravSHACL expects one .ttl file per shape in a directory.
    Split a monolithic shapes.ttl into per-shape files.
    """
    g = Graph()
    g.parse(shapes_path, format="turtle")

    shapes_dir = Path(output_dir) / "travshacl_shapes"
    shapes_dir.mkdir(parents=True, exist_ok=True)

    shape_uris = set(g.subjects(RDF.type, SH.NodeShape))

    if not shape_uris:
        import shutil
        shutil.copy(shapes_path, str(shapes_dir / "shapes.ttl"))
        return str(shapes_dir)

    skipped = 0
    for i, shape_uri in enumerate(shape_uris):
        if isinstance(shape_uri, BNode):
            skipped += 1
            continue
        has_target = any(g.objects(shape_uri, SH.targetClass))
        if not has_target:
            skipped += 1
            continue
        has_property = any(g.objects(shape_uri, SH.property))
        if not has_property:
            skipped += 1
            continue

        sg = Graph()
        sg.bind("sh", SH)
        sg.bind("ex", EX)
        sg.bind("xsd", XSD)

        for p, o in g.predicate_objects(shape_uri):
            sg.add((shape_uri, p, o))
            if isinstance(o, BNode):
                for p2, o2 in g.predicate_objects(o):
                    sg.add((o, p2, o2))

        local = str(shape_uri).split("/")[-1].replace("#", "_")
        sg.serialize(str(shapes_dir / f"{local}.ttl"), format="turtle")

    if skipped:
        log("5", f"  Skipped {skipped} shape(s) without targetClass or property constraints")
    return str(shapes_dir)


def _validate_travshacl(
    shapes_path: str,
    kg_path: str,
    endpoint_url: str | None,
    output_dir: str,
    version: str,
) -> list[dict]:
    """
    Run TravSHACL for SHACL validation.

    Supports two modes:
      1. endpoint_url is set → validate against external SPARQL endpoint
      2. endpoint_url is None → load KG as rdflib Graph and pass directly
    """
    try:
        from TravSHACL import GraphTraversal, ShapeSchema, parse_heuristics
    except ImportError:
        raise ImportError(
            "TravSHACL not installed. Run: pip install travshacl\n"
            "GitHub: https://github.com/SDM-TIB/Trav-SHACL"
        )

    shapes_dir = _split_shapes_to_dir(shapes_path, output_dir)

    if endpoint_url:
        ep = endpoint_url
    else:
        kg_graph = Graph()
        kg_graph.parse(kg_path, format="turtle")
        ep = kg_graph

    result_dir = Path(output_dir) / f"travshacl_results_{version}"
    result_dir.mkdir(parents=True, exist_ok=True)

    schema = ShapeSchema(
        schema_dir=shapes_dir,
        endpoint=ep,
        graph_traversal=GraphTraversal.DFS,
        heuristics=parse_heuristics("TARGET IN BIG"),
        use_selective_queries=True,
        max_split_size=256,
        output_dir=str(result_dir),
        order_by_in_queries=False,
        save_outputs=False,
    )
    result = schema.validate()

    violations: list[dict] = []
    if isinstance(result, dict):
        for shape_name, report in result.items():
            if shape_name == "unbound":
                continue
            invalid = set()
            if isinstance(report, dict):
                invalid = report.get("invalid_instances", set())
            elif isinstance(report, (list, set)):
                invalid = set(report)
            for entry in invalid:
                if isinstance(entry, tuple) and len(entry) >= 2:
                    entity_uri = entry[1]
                else:
                    entity_uri = str(entry)
                violations.append({
                    "focus_node":  str(entity_uri),
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

def _violation_categories(violations: list[dict]) -> dict[str, int]:
    """Group violations by their result_path (property that caused the violation)."""
    cats: dict[str, int] = {}
    for v in violations:
        path = v.get("result_path", "unknown")
        prop = path.rsplit("/", 1)[-1] if "/" in path else path
        cats[prop] = cats.get(prop, 0) + 1
    return dict(sorted(cats.items(), key=lambda x: -x[1]))


_COMPLETENESS_PATTERNS = ("Less than", "less than", "Min count")


def _classify_violation(v: dict) -> str:
    """
    Classify a SHACL violation as 'quality' or 'completeness'.

    Completeness = minCount failures ("Less than N values on ...")
    Quality      = everything else (maxCount, datatype, class mismatch)
    """
    msg = v.get("message", "")
    if any(pat in msg for pat in _COMPLETENESS_PATTERNS):
        return "completeness"
    return "quality"


def _split_quality_completeness(
    violations: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Split violations into (quality_violations, completeness_violations)."""
    quality: list[dict] = []
    completeness: list[dict] = []
    for v in violations:
        if _classify_violation(v) == "completeness":
            completeness.append(v)
        else:
            quality.append(v)
    return quality, completeness


def validate(
    kg_path: str,
    shapes_path: Optional[str] = None,
    ontology_path: str = "input/hf_guideline_ontology.ttl",
    output_dir: str = "outputs/step5",
    version: str = "v1",
    sparql_endpoint: Optional[str] = None,
) -> dict:
    """
    Validate the materialised KG against SHACL shapes.

    Backend selection (priority order)
    ----------------------------------
    1. TravSHACL (preferred — Prof. Vidal / SDM-TIB tool, per meeting notes)
       - If sparql_endpoint is set → validate against that endpoint
       - If sparql_endpoint is None → TravSHACL reads the .ttl file directly
    2. pySHACL (fallback — if TravSHACL is not installed)
       - Works in Colab / CI out of the box

    If shapes_path is None, shapes are auto-derived from the OWL ontology.
    """
    if shapes_path is None:
        shapes_path = derive_shapes(ontology_path, output_dir)

    log("5", f"Validating [{version}]: {kg_path}")

    backend = None
    violations = []
    conforms = True
    n_triples = 0

    # ── Try TravSHACL first (primary validator per project requirements) ──
    try:
        import TravSHACL  # noqa: F401
        travshacl_available = True
    except ImportError:
        travshacl_available = False

    if travshacl_available:
        try:
            ep_label = sparql_endpoint or f"localRDF:{kg_path}"
            log("5", f"Backend: TravSHACL  ({ep_label})")
            violations = _validate_travshacl(
                shapes_path, kg_path, sparql_endpoint, output_dir, version
            )
            conforms = len(violations) == 0
            kg = Graph().parse(kg_path, format="turtle")
            n_triples = len(kg)
            backend = "TravSHACL"
        except Exception as exc:
            log("5", f"TravSHACL failed ({type(exc).__name__}: {exc}) — falling back to pySHACL")
            backend = None

    # ── pySHACL fallback ─────────────────────────────────────────────────
    if backend is None:
        log("5", "Backend: pySHACL (fallback)")
        kg = Graph()
        kg.parse(kg_path, format="turtle")

        shacl_graph = Graph()
        shacl_graph.parse(shapes_path, format="turtle")

        conforms, violations = _validate_pyshacl(kg, shacl_graph)
        n_triples = len(kg)
        backend = "pySHACL"

    violated_focus = {v["focus_node"] for v in violations if v.get("focus_node")}
    categories = _violation_categories(violations)

    quality_violations, completeness_violations = _split_quality_completeness(violations)
    q_focus = {v["focus_node"] for v in quality_violations if v.get("focus_node")}
    c_focus = {v["focus_node"] for v in completeness_violations if v.get("focus_node")}

    report = {
        "version":          version,
        "backend":          backend,
        "kg_path":          kg_path,
        "sparql_endpoint":  sparql_endpoint,
        "conforms":         conforms,
        "total_triples":    n_triples,
        "total_violations": len(violations),
        "quality_violations":      len(quality_violations),
        "completeness_violations": len(completeness_violations),
        "violations":       violations,
        "metrics": {
            "violation_rate":    round(len(violations) / max(n_triples, 1), 4),
            "conformance_ratio": round(1.0 - len(violations) / max(n_triples, 1), 4),
            "quality_conformance": round(
                1.0 - len(quality_violations) / max(n_triples, 1), 4),
            "distinct_focus_nodes_violated": len(violated_focus),
            "distinct_quality_focus_nodes": len(q_focus),
            "distinct_completeness_focus_nodes": len(c_focus),
            "violations_by_property": categories,
            "quality_by_property": _violation_categories(quality_violations),
            "completeness_by_property": _violation_categories(completeness_violations),
        },
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"validation_report_{version}.json"
    save_json(report, str(report_path))

    status = "✓ CONFORMS" if conforms else f"✗ {len(violations)} violations"
    log("5", f"[{version}] {backend}  {status} — {n_triples} triples  →  {report_path}")
    log("5", f"  Quality violations:      {len(quality_violations)}  "
        f"(conformance: {report['metrics']['quality_conformance']:.1%})")
    log("5", f"  Completeness violations: {len(completeness_violations)}  "
        f"(coverage gaps — expected for extracted KGs)")
    if categories:
        top = ", ".join(f"{k}:{v}" for k, v in list(categories.items())[:5])
        log("5", f"  Violation breakdown: {top}")
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
    for key in ("backend", "total_triples", "total_violations",
                "quality_violations", "completeness_violations"):
        print(f"{key:<30} {str(r1.get(key, '—')):>14}  {str(r2.get(key, '—')):>10}")
    print(f"{'conformance_ratio':<30} "
          f"{r1['metrics'].get('conformance_ratio', '—'):>14}  "
          f"{r2['metrics'].get('conformance_ratio', '—'):>10}")
    print(f"{'quality_conformance':<30} "
          f"{r1['metrics'].get('quality_conformance', '—'):>14}  "
          f"{r2['metrics'].get('quality_conformance', '—'):>10}")
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
