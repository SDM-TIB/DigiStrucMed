"""
Visualize a materialized knowledge graph (TTL/NT) as an interactive HTML graph.

Uses Cytoscape.js (no eval(), CSP-safe, works offline when inlined).

    python visualize_kg.py
    python visualize_kg.py --kg outputs/pipeline-output13/step4/output_v2.ttl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from rdflib import Graph, Literal, URIRef

EX_ONTO = "http://digistructmed.org/ontology/"
EX_INST = "http://digistructmed.org/instance/"
CYTO_CDN = "https://unpkg.com/cytoscape@3.30.4/dist/cytoscape.min.js"


LABEL_PROPS = [
    URIRef(EX_ONTO + "preferredName"),
    URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
    URIRef(EX_ONTO + "profileDescription"),
    URIRef(EX_ONTO + "drugClassName"),
]


def _short(term) -> str:
    s = str(term)
    for ns in (EX_INST, EX_ONTO):
        if s.startswith(ns):
            return s[len(ns):]
    return s.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _build_label_map(g: Graph) -> dict[str, str]:
    """Build URI -> human-readable label from known label properties."""
    labels: dict[str, str] = {}
    for prop in LABEL_PROPS:
        for s, _, o in g.triples((None, prop, None)):
            if isinstance(s, URIRef) and isinstance(o, Literal):
                text = str(o).replace("\n", " ").strip()
                if not text or len(text) > 80:
                    continue
                uri = str(s)
                if uri not in labels or prop == LABEL_PROPS[0]:
                    labels[uri] = text
    return labels


def _label(term, label_map: dict[str, str] | None = None) -> str:
    if isinstance(term, Literal):
        s = str(term).replace("\n", " ").strip()
        return s[:55] + "..." if len(s) > 55 else s
    uri = str(term)
    short = _short(term)
    if label_map and uri in label_map:
        name = label_map[uri]
        if short != name:
            return f"{name} ({short})"
        return name
    # Make entity_* URIs more readable
    if short.startswith("entity_"):
        return short[7:].replace("_", " ")
    return short


def _pred_label(p: URIRef) -> str:
    s = str(p)
    if s.startswith(EX_ONTO):
        return s[len(EX_ONTO):]
    return s.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def build_graph_data(kg_path: Path, max_edges: int, seed: int) -> dict:
    fmt = "turtle" if kg_path.suffix.lower() in {".ttl", ".turtle"} else "nt"
    g = Graph()
    g.parse(str(kg_path), format=fmt)

    label_map = _build_label_map(g)
    print(f"  Label map: {len(label_map)} human-readable names found")

    triples = list(g.triples((None, None, None)))
    random.Random(seed).shuffle(triples)

    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    pred_counts: Counter = Counter()

    def ensure_node(term):
        nid = str(term)
        if nid not in nodes:
            kind = "uri" if isinstance(term, URIRef) else "lit"
            nodes[nid] = {"id": nid, "label": _label(term, label_map), "kind": kind}
        return nid

    for s, p, o in triples:
        if len(edges) >= max_edges:
            break
        if not isinstance(s, URIRef) or not isinstance(p, URIRef):
            continue
        if isinstance(o, URIRef):
            sid = ensure_node(s)
            oid = ensure_node(o)
            pl = _pred_label(p)
            pred_counts[pl] += 1
            edges.append({"source": sid, "target": oid, "label": pl, "kind": "obj"})

    if len(edges) < max_edges:
        for s, p, o in triples:
            if len(edges) >= max_edges:
                break
            if not isinstance(s, URIRef) or not isinstance(p, URIRef) or not isinstance(o, Literal):
                continue
            sid = ensure_node(s)
            oid = ensure_node(o)
            pl = _pred_label(p)
            pred_counts[pl] += 1
            edges.append({"source": sid, "target": oid, "label": pl, "kind": "lit"})

    top_preds = [{"p": p, "c": c} for p, c in pred_counts.most_common(40)]

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "total_triples": len(triples),
        "top_preds": top_preds,
    }


def render_html(gd: dict, cyto_js: str) -> str:
    elements = []
    for n in gd["nodes"]:
        elements.append({"data": {"id": n["id"], "label": n["label"], "kind": n["kind"]}})
    for i, e in enumerate(gd["edges"]):
        elements.append({"data": {"id": f"e{i}", "source": e["source"], "target": e["target"], "label": e["label"], "kind": e["kind"]}})

    el_json = json.dumps(elements, ensure_ascii=False)
    preds_json = json.dumps(gd["top_preds"], ensure_ascii=False)
    nn = len(gd["nodes"])
    ne = len(gd["edges"])
    nt = gd["total_triples"]

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DigiStructMed KG Viewer</title>
<script>{cyto_js}</script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{height:100vh;display:grid;grid-template-columns:280px 1fr;grid-template-rows:44px 1fr;font-family:system-ui,sans-serif;background:#0d1117;color:#c9d1d9}}
#bar{{grid-column:1/-1;display:flex;align-items:center;gap:10px;padding:0 14px;background:#161b22;border-bottom:1px solid #30363d}}
#bar h1{{font-size:13px;color:#58a6ff}}
.pill{{font-size:11px;padding:2px 10px;border:1px solid #30363d;border-radius:99px;color:#8b949e}}
#side{{background:#0d1117;border-right:1px solid #30363d;padding:12px;overflow-y:auto}}
#side h3{{font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.08em;margin:12px 0 6px}}
input,select{{width:100%;padding:6px 8px;border:1px solid #30363d;border-radius:8px;background:#161b22;color:#c9d1d9;font-size:12px;outline:none}}
input:focus,select:focus{{border-color:#58a6ff}}
button{{padding:6px 12px;border:1px solid #30363d;border-radius:8px;background:#21262d;color:#c9d1d9;cursor:pointer;font-size:12px}}
button:hover{{border-color:#58a6ff}}
.row{{display:flex;gap:6px;margin:6px 0}}
.small{{font-size:11px;color:#8b949e;line-height:1.4}}
#info{{margin-top:10px;padding:10px;border:1px solid #30363d;border-radius:10px;background:#161b22;display:none}}
#info.show{{display:block}}
#info .lbl{{font-size:10px;color:#8b949e;margin-bottom:4px}}
#info .val{{font-size:12px;word-break:break-all}}
#cy{{width:100%;height:100%}}
</style>
</head>
<body>
<div id="bar">
  <h1>KG Viewer</h1>
  <span class="pill">nodes {nn}</span>
  <span class="pill">edges {ne}</span>
  <span class="pill">triples {nt}</span>
</div>
<div id="side">
  <h3>Search</h3>
  <div class="row"><input id="q" placeholder="Type a node label..."></div>
  <div class="row"><button onclick="doSearch()">Find</button><button onclick="cy.fit(undefined,30)">Fit all</button></div>

  <h3>Filter by predicate</h3>
  <select id="pred" onchange="applyFilter()"><option value="">All</option></select>

  <h3>Show</h3>
  <div class="row">
    <button onclick="showOnly('obj')">URI edges only</button>
    <button onclick="showOnly('')">All</button>
  </div>
  <div class="small" style="margin-top:8px">Click a node to see details. Scroll to zoom. Drag to pan.</div>

  <div id="info">
    <div class="lbl">Selected node</div>
    <div class="val" id="infoV"></div>
  </div>
</div>
<div id="cy"></div>
<script>
var ELEMENTS = {el_json};
var PREDS = {preds_json};

var cy = cytoscape({{
  container: document.getElementById('cy'),
  elements: ELEMENTS,
  style: [
    {{
      selector: 'node[kind="uri"]',
      style: {{
        'label': 'data(label)',
        'font-size': 8,
        'color': '#c9d1d9',
        'text-valign': 'center',
        'text-halign': 'center',
        'background-color': '#1f6feb',
        'border-color': '#58a6ff',
        'border-width': 1,
        'width': 'mapData(degree, 0, 50, 20, 60)',
        'height': 'mapData(degree, 0, 50, 20, 60)',
        'text-wrap': 'ellipsis',
        'text-max-width': 80,
      }}
    }},
    {{
      selector: 'node[kind="lit"]',
      style: {{
        'label': 'data(label)',
        'font-size': 7,
        'color': '#e3b341',
        'text-valign': 'center',
        'text-halign': 'center',
        'background-color': '#3d2e00',
        'border-color': '#e3b341',
        'border-width': 1,
        'shape': 'round-rectangle',
        'width': 16,
        'height': 16,
        'text-wrap': 'ellipsis',
        'text-max-width': 70,
      }}
    }},
    {{
      selector: 'edge[kind="obj"]',
      style: {{
        'label': 'data(label)',
        'font-size': 6,
        'color': '#8b949e',
        'text-rotation': 'autorotate',
        'line-color': '#30638e',
        'target-arrow-color': '#30638e',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier',
        'width': 1,
        'arrow-scale': 0.6,
        'text-background-color': '#0d1117',
        'text-background-opacity': 0.7,
        'text-background-padding': '2px',
      }}
    }},
    {{
      selector: 'edge[kind="lit"]',
      style: {{
        'label': 'data(label)',
        'font-size': 5,
        'color': '#8b949e',
        'text-rotation': 'autorotate',
        'line-color': '#3d2e00',
        'target-arrow-color': '#3d2e00',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier',
        'width': 0.7,
        'arrow-scale': 0.5,
        'line-style': 'dashed',
      }}
    }},
    {{
      selector: ':selected',
      style: {{
        'background-color': '#f78166',
        'border-color': '#f78166',
        'line-color': '#f78166',
        'target-arrow-color': '#f78166',
      }}
    }}
  ],
  layout: {{
    name: 'cose',
    animate: false,
    nodeOverlap: 20,
    idealEdgeLength: 80,
    nodeRepulsion: 8000,
    gravity: 0.25,
    numIter: 300,
    fit: true,
    padding: 30,
  }},
  wheelSensitivity: 0.3,
  minZoom: 0.05,
  maxZoom: 4,
}});

// populate predicate dropdown
(function() {{
  var sel = document.getElementById('pred');
  PREDS.forEach(function(p) {{
    var o = document.createElement('option');
    o.value = p.p;
    o.textContent = p.p + ' (' + p.c + ')';
    sel.appendChild(o);
  }});
}})();

function applyFilter() {{
  var v = document.getElementById('pred').value;
  if (!v) {{
    cy.elements().show();
  }} else {{
    cy.edges().hide();
    cy.nodes().hide();
    var matched = cy.edges().filter(function(e) {{ return e.data('label') === v; }});
    matched.show();
    matched.connectedNodes().show();
  }}
}}

function showOnly(kind) {{
  if (!kind) {{
    cy.elements().show();
    return;
  }}
  cy.edges().hide();
  cy.nodes().hide();
  var matched = cy.edges().filter(function(e) {{ return e.data('kind') === kind; }});
  matched.show();
  matched.connectedNodes().show();
}}

function doSearch() {{
  var q = (document.getElementById('q').value || '').toLowerCase().trim();
  if (!q) return;
  var found = cy.nodes().filter(function(n) {{
    return (n.data('label') || '').toLowerCase().indexOf(q) >= 0;
  }});
  if (found.length > 0) {{
    cy.animate({{ fit: {{ eles: found, padding: 60 }}, duration: 400 }});
    cy.nodes().unselect();
    found.select();
  }}
}}

cy.on('tap', 'node', function(evt) {{
  var d = evt.target.data();
  document.getElementById('infoV').textContent = d.id;
  document.getElementById('info').className = 'show';
}});

cy.on('tap', function(evt) {{
  if (evt.target === cy) {{
    document.getElementById('info').className = '';
  }}
}});
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default="outputs/pipeline-output14/step4/output_v2.ttl")
    ap.add_argument("--out", default=None, help="Output HTML path (default: kg_graph.html next to the KG file)")
    ap.add_argument("--max-edges", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    kg_path = Path(args.kg)
    out_path = Path(args.out) if args.out else kg_path.parent / "kg_graph.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not kg_path.is_file():
        raise SystemExit(f"KG not found: {kg_path}")

    print(f"Parsing {kg_path} ...")
    gd = build_graph_data(kg_path, args.max_edges, args.seed)
    print(f"  {len(gd['nodes'])} nodes, {len(gd['edges'])} edges, {gd['total_triples']} triples")

    vendor = out_path.parent / "cytoscape.min.js"
    if not vendor.is_file():
        import urllib.request
        print(f"Downloading cytoscape.js to {vendor} ...")
        urllib.request.urlretrieve(CYTO_CDN, str(vendor))

    cyto_js = vendor.read_text(encoding="utf-8", errors="replace")

    html = render_html(gd, cyto_js)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote: {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
