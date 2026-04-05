"""
Step 2 — Ontology Loading  [SYMBOLIC]
─────────────────────────────────────────────────────────────────────────────
Input  : hf_guideline_ontology.ttl   (OWL/RDFS)
Output : OntologyIndex  (in-memory, used by Steps 3a, 3b, 5)

Parses and indexes:
  - Classes          (URI → label, comment, subClassOf)
  - Object properties (URI → label, domain, range)
  - Datatype properties (URI → label, domain, range/datatype)
  - Named individuals (URI → label, types)
  - Enumerations     (owl:oneOf)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rdflib import BNode, Graph, Namespace, RDF, RDFS, OWL, XSD

from utils import log

EX = Namespace("http://digistructmed.org/ontology/")


def _resolve_class_list(g: Graph, node) -> list[str]:
    """
    Resolve a domain/range value that may be a direct class URI or a
    blank-node owl:unionOf / owl:intersectionOf.  Returns plain class URIs.
    """
    if not isinstance(node, BNode):
        return [str(node)]

    for list_prop in (OWL.unionOf, OWL.intersectionOf):
        rdf_list = g.value(node, list_prop)
        if rdf_list is None:
            continue
        members: list[str] = []
        cur = rdf_list
        while cur and str(cur) != str(RDF.nil):
            first = g.value(cur, RDF.first)
            if first is not None:
                members.extend(_resolve_class_list(g, first))
            cur = g.value(cur, RDF.rest)
        if members:
            return members

    return []


@dataclass
class PropertyInfo:
    uri: str
    label: str
    comment: str
    domain: list[str]
    range_: list[str]
    prop_type: str          # "object" | "datatype"

    # Convenience: all labels (label + comment words) for fuzzy matching in 3a
    @property
    def match_tokens(self) -> list[str]:
        tokens = [self.label.lower()]
        tokens += [w.lower() for w in self.comment.split() if len(w) > 3]
        return tokens


@dataclass
class OntologyIndex:
    classes: dict[str, dict] = field(default_factory=dict)
    object_properties: dict[str, PropertyInfo] = field(default_factory=dict)
    datatype_properties: dict[str, PropertyInfo] = field(default_factory=dict)
    named_individuals: dict[str, dict] = field(default_factory=dict)
    enumerations: dict[str, list[str]] = field(default_factory=dict)
    _source_path: str = ""

    # ── Quick lookups ──────────────────────────────────────────────────────

    def all_properties(self) -> dict[str, PropertyInfo]:
        return {**self.object_properties, **self.datatype_properties}

    def class_label(self, uri: str) -> str:
        return self.classes.get(uri, {}).get("label", uri.split("/")[-1])

    def prop_label(self, uri: str) -> str:
        p = self.all_properties().get(uri)
        return p.label if p else uri.split("/")[-1]


# ── Parser ─────────────────────────────────────────────────────────────────

def load_ontology(ttl_path: str) -> OntologyIndex:
    """
    Parse an OWL/RDFS Turtle file and return a populated OntologyIndex.
    """
    log("2", f"Parsing ontology: {ttl_path}")
    g = Graph()
    g.parse(ttl_path, format="turtle")
    log("2", f"Graph loaded: {len(g)} triples")

    idx = OntologyIndex(_source_path=ttl_path)

    # ── Classes ────────────────────────────────────────────────────────────
    for cls in g.subjects(RDF.type, OWL.Class):
        uri = str(cls)
        idx.classes[uri] = {
            "label": str(g.value(cls, RDFS.label) or ""),
            "comment": str(g.value(cls, RDFS.comment) or ""),
            "subClassOf": [str(s) for s in g.objects(cls, RDFS.subClassOf)],
        }

    # ── Object properties ──────────────────────────────────────────────────
    for p in g.subjects(RDF.type, OWL.ObjectProperty):
        uri = str(p)
        domain_uris: list[str] = []
        for d in g.objects(p, RDFS.domain):
            domain_uris.extend(_resolve_class_list(g, d))
        range_uris: list[str] = []
        for r in g.objects(p, RDFS.range):
            range_uris.extend(_resolve_class_list(g, r))
        idx.object_properties[uri] = PropertyInfo(
            uri=uri,
            label=str(g.value(p, RDFS.label) or ""),
            comment=str(g.value(p, RDFS.comment) or ""),
            domain=domain_uris,
            range_=range_uris,
            prop_type="object",
        )

    # ── Datatype properties ────────────────────────────────────────────────
    for p in g.subjects(RDF.type, OWL.DatatypeProperty):
        uri = str(p)
        domain_uris = []
        for d in g.objects(p, RDFS.domain):
            domain_uris.extend(_resolve_class_list(g, d))
        range_uris = []
        for r in g.objects(p, RDFS.range):
            range_uris.extend(_resolve_class_list(g, r))
        idx.datatype_properties[uri] = PropertyInfo(
            uri=uri,
            label=str(g.value(p, RDFS.label) or ""),
            comment=str(g.value(p, RDFS.comment) or ""),
            domain=domain_uris,
            range_=range_uris,
            prop_type="datatype",
        )

    # ── Named individuals ──────────────────────────────────────────────────
    for ind in g.subjects(RDF.type, OWL.NamedIndividual):
        uri = str(ind)
        idx.named_individuals[uri] = {
            "label": str(g.value(ind, RDFS.label) or ""),
            "types": [
                str(t) for t in g.objects(ind, RDF.type)
                if t != OWL.NamedIndividual
            ],
        }

    # ── Enumerations (owl:oneOf) ───────────────────────────────────────────
    # Check both direct owl:oneOf and owl:equivalentClass → owl:oneOf
    def _extract_one_of(rdf_node) -> list[str]:
        members: list[str] = []
        cur = rdf_node
        while cur and str(cur) != str(RDF.nil):
            first = g.value(cur, RDF.first)
            if first:
                members.append(str(first))
            cur = g.value(cur, RDF.rest)
        return members

    for cls in g.subjects(RDF.type, OWL.Class):
        one_of = g.value(cls, OWL.oneOf)
        if one_of:
            members = _extract_one_of(one_of)
            if members:
                idx.enumerations[str(cls)] = members
            continue

        equiv = g.value(cls, OWL.equivalentClass)
        if equiv is not None:
            one_of = g.value(equiv, OWL.oneOf)
            if one_of:
                members = _extract_one_of(one_of)
                if members:
                    idx.enumerations[str(cls)] = members

    log("2", (
        f"Indexed: {len(idx.classes)} classes, "
        f"{len(idx.object_properties)} object props, "
        f"{len(idx.datatype_properties)} datatype props, "
        f"{len(idx.named_individuals)} named individuals, "
        f"{len(idx.enumerations)} enumerations"
    ))
    return idx


if __name__ == "__main__":
    import sys
    ttl = sys.argv[1] if len(sys.argv) > 1 else "input/hf_guideline_ontology.ttl"
    idx = load_ontology(ttl)
    print(f"\nSample classes: {list(idx.classes.values())[:3]}")
