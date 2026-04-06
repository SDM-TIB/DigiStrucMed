"""
Standalone visualizer: reads hf_guideline_ontology.ttl and produces an
EER-style interactive HTML graph with proper specialization junctions.

Usage:
    python visualize_ontology.py
"""
from __future__ import annotations
import json, re
from pathlib import Path
from collections import defaultdict

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD
from rdflib.term import BNode

EX = "http://digistructmed.org/ontology/"
TTL_PATH = Path("input/hf_guideline_ontology.ttl")
# Write to input/ so you can open the "usual" HTML directly.
# (This file is fully self-contained: it embeds the graph JSON.)
OUT_PATH = Path("input/ontology_graph.html")


def _local(uri) -> str:
    s = str(uri)
    for ns in (EX,
               "http://www.w3.org/2001/XMLSchema#",
               "http://www.w3.org/2002/07/owl#",
               "http://www.w3.org/2000/01/rdf-schema#",
               "http://www.w3.org/1999/02/22-rdf-syntax-ns#"):
        s = s.replace(ns, "")
    return s


def _rdf_list_members(g: Graph, head) -> list:
    """Ordered members of an RDF collection (rdf:first / rdf:rest)."""
    out = []
    while head and head != RDF.nil:
        first = g.value(head, RDF.first)
        if first is not None:
            out.append(first)
        head = g.value(head, RDF.rest)
    return out


def expand_property_domains(g: Graph, prop, classes: dict) -> list[str]:
    """
    Resolve rdfs:domain to named ontology classes: a single class IRI or
    owl:unionOf of class IRIs (blank-node domain).
    """
    out: list[str] = []
    for d in g.objects(prop, RDFS.domain):
        union_head = g.value(d, OWL.unionOf)
        if union_head is not None:
            for mem in _rdf_list_members(g, union_head):
                loc = _local(mem)
                if loc in classes:
                    out.append(loc)
        else:
            loc = _local(d)
            if loc in classes:
                out.append(loc)
    return out


def rollup_umls_concept_domains(obj_props: list, taxonomies: list) -> list:
    """
    If hasUMLSConcept is shown for a class that has superclass links, attach the
    edge to the topmost ancestor in the taxonomy instead (L4: one relationship
    at the parent level). E.g. Drug, DeviceTherapy, InotropicAgent → Therapy.
    """
    child_parents = defaultdict(list)
    for t in taxonomies:
        child_parents[t["child"]].append(t["parent"])

    def top_ancestor(name: str) -> str:
        seen: set[str] = set()
        cur = name
        while cur in child_parents:
            pars = child_parents[cur]
            if not pars:
                break
            nxt = sorted(pars)[0]
            if nxt in seen:
                break
            seen.add(nxt)
            cur = nxt
        return cur

    umls_prop, cui_range = "hasUMLSConcept", "CUI"
    kept = []
    root_domains: set[str] = set()
    for p in obj_props:
        if p.get("name") == umls_prop and p.get("range") == cui_range:
            root_domains.add(top_ancestor(p["domain"]))
        else:
            kept.append(p)
    for dom in sorted(root_domains):
        kept.append({
            "name": umls_prop,
            "label": "has UMLS concept",
            "domain": dom,
            "range": cui_range,
        })
    return kept


def parse_ttl(path: Path):
    g = Graph()
    g.parse(str(path), format="turtle")

    classes = {}
    class_uris = set(g.subjects(RDF.type, OWL.Class)) | set(g.subjects(RDF.type, RDFS.Class))
    for uri in class_uris:
        # Anonymous union/restriction expressions are also typed owl:Class; skip bnodes.
        if isinstance(uri, BNode):
            continue
        name = _local(uri)
        if not name or name.startswith("_"):
            continue
        lbl = g.value(uri, RDFS.label)
        cmt = g.value(uri, RDFS.comment)
        classes[name] = {
            "id": name,
            "label": str(lbl) if lbl else name,
            "comment": str(cmt) if cmt else "",
        }

    taxonomies = []
    for subj, obj in g.subject_objects(RDFS.subClassOf):
        s, o = _local(subj), _local(obj)
        if s in classes and o in classes:
            taxonomies.append({"child": s, "parent": o})

    obj_props = []
    for p in set(g.subjects(RDF.type, OWL.ObjectProperty)):
        name = _local(p)
        if not name:
            continue
        lbl = g.value(p, RDFS.label)
        domains = expand_property_domains(g, p, classes)
        ranges = [_local(r) for r in g.objects(p, RDFS.range) if _local(r) in classes]
        for dom in domains:
            for rng in ranges:
                obj_props.append({
                    "name": name,
                    "label": str(lbl) if lbl else name,
                    "domain": dom,
                    "range": rng,
                })

    dt_props = []
    for p in set(g.subjects(RDF.type, OWL.DatatypeProperty)):
        name = _local(p)
        if not name:
            continue
        lbl = g.value(p, RDFS.label)
        domains = expand_property_domains(g, p, classes)
        rng_uri = g.value(p, RDFS.range)
        rng = _local(rng_uri) if rng_uri else "string"
        for dom in domains:
            dt_props.append({
                "name": name,
                "label": str(lbl) if lbl else name,
                "domain": dom,
                "range": rng,
            })

    # ── Collapse leaf-only subclasses into attributes ──────────
    # If a subclass has NO properties as domain, NO properties as range,
    # NO children of its own, and ALL siblings are equally "empty" —
    # they are just values, not real entity types (L4: avoid entity types
    # that are only data type values). Collapse them into a single
    # attribute on the parent.
    prop_domains = set()
    prop_ranges = set()
    for p in obj_props:
        prop_domains.add(p["domain"])
        prop_ranges.add(p["range"])
    for p in dt_props:
        prop_domains.add(p["domain"])

    parent_children_raw = defaultdict(list)
    child_parents_raw = defaultdict(list)
    for t in taxonomies:
        parent_children_raw[t["parent"]].append(t["child"])
        child_parents_raw[t["child"]].append(t["parent"])

    children_of_raw = defaultdict(list)
    for t in taxonomies:
        children_of_raw[t["parent"]].append(t["child"])

    collapsed_groups = {}
    collapsed_classes = set()

    # Parents we never collapse: guideline section buckets (subclasses are real entity types, not "values")
    _no_collapse_parents = frozenset({"ClinicalAssessment"})

    for parent, children in parent_children_raw.items():
        if parent in _no_collapse_parents:
            continue
        if len(children) < 3:
            continue
        all_leaves = True
        for ch in children:
            has_own_props = ch in prop_domains or ch in prop_ranges
            has_children = len(children_of_raw.get(ch, [])) > 0
            if has_own_props or has_children:
                all_leaves = False
                break
        if all_leaves:
            labels = []
            for ch in children:
                lbl = classes[ch]["label"] if ch in classes else ch
                labels.append(lbl)
            collapsed_groups[parent] = {
                "values": sorted(labels),
                "children": list(children),
            }
            for ch in children:
                collapsed_classes.add(ch)

    taxonomies = [t for t in taxonomies if t["child"] not in collapsed_classes]
    for cname in collapsed_classes:
        if cname in classes:
            del classes[cname]

    for parent, info in collapsed_groups.items():
        examples = ", ".join(info["values"][:5])
        if len(info["values"]) > 5:
            examples += ", ..."
        if "CUI" in classes:
            obj_props.append({
                "name": "hasUMLSConcept",
                "label": "has UMLS concept",
                "domain": parent,
                "range": "CUI",
            })
        else:
            dt_props.append({
                "name": "name",
                "label": "name",
                "domain": parent,
                "range": f"string  (e.g. {examples})",
            })

    obj_props = rollup_umls_concept_domains(obj_props, taxonomies)

    # Detect disjoint groups: children that share a parent AND have owl:disjointWith among them
    disjoint_pairs = set()
    for s, o in g.subject_objects(OWL.disjointWith):
        sl, ol = _local(s), _local(o)
        if sl in classes and ol in classes:
            disjoint_pairs.add((min(sl, ol), max(sl, ol)))

    parent_children = defaultdict(list)
    for t in taxonomies:
        parent_children[t["parent"]].append(t["child"])

    junctions = []
    junction_edges = []
    used_tax = set()

    for parent, children in parent_children.items():
        if len(children) < 2:
            continue
        any_disjoint = any(
            (min(a, b), max(a, b)) in disjoint_pairs
            for i, a in enumerate(children)
            for b in children[i+1:]
        )
        jtype = "d" if any_disjoint else "o"
        jid = f"junc_{parent}"
        junctions.append({"id": jid, "parent": parent, "type": jtype})
        junction_edges.append({"from": jid, "to": parent})
        for ch in children:
            junction_edges.append({"from": ch, "to": jid})
            used_tax.add((ch, parent))

    remaining_tax = [t for t in taxonomies if (t["child"], t["parent"]) not in used_tax]

    # Compute hierarchy levels (depth from root). Roots = level 0 (rightmost).
    child_set = {t["child"] for t in taxonomies}
    parent_set = {t["parent"] for t in taxonomies}
    roots = parent_set - child_set
    all_named = set(classes.keys())
    orphans = all_named - child_set - parent_set

    parent_of = {}
    for t in taxonomies:
        parent_of.setdefault(t["child"], []).append(t["parent"])
    children_of = {}
    for t in taxonomies:
        children_of.setdefault(t["parent"], []).append(t["child"])

    levels = {}
    for r in roots | orphans:
        levels[r] = 0

    def assign_levels(node, lvl):
        if node in levels:
            levels[node] = max(levels[node], lvl)
        else:
            levels[node] = lvl
        for ch in children_of.get(node, []):
            assign_levels(ch, lvl + 1)

    for r in roots:
        assign_levels(r, 0)

    max_lvl = max(levels.values()) if levels else 0
    class_levels = {}
    for cname in classes:
        class_levels[cname] = levels.get(cname, 0)

    # Junction levels: between parent and children
    junction_levels = {}
    for j in junctions:
        p_lvl = class_levels.get(j["parent"], 0)
        ch_lvls = [class_levels.get(ch, p_lvl + 1) for ch in parent_children.get(j["parent"], [])]
        junction_levels[j["id"]] = p_lvl + 1 if ch_lvls else p_lvl

    # Instances (NamedIndividual)
    individuals = []
    for ind in g.subjects(RDF.type, OWL.NamedIndividual):
        if isinstance(ind, BNode):
            continue
        name = _local(ind)
        if not name:
            continue
        lbl = g.value(ind, RDFS.label)
        types = [_local(t) for t in g.objects(ind, RDF.type)
                 if _local(t) in classes and _local(t) != "NamedIndividual"]
        individuals.append({
            "id": name,
            "label": str(lbl) if lbl else name,
            "types": types,
        })

    for c in classes.values():
        c["level"] = class_levels.get(c["id"], 0)

    for j in junctions:
        j["level"] = junction_levels.get(j["id"], 1)

    return {
        "classes": list(classes.values()),
        "taxonomies": remaining_tax,
        "junctions": junctions,
        "junction_edges": junction_edges,
        "obj_props": obj_props,
        "dt_props": dt_props,
        "individuals": individuals,
        "max_level": max_lvl,
    }


def render_html(data: dict) -> str:
    d = json.dumps(data, ensure_ascii=False)
    nc = len(data["classes"])
    nj = len(data["junctions"])
    no = len(data["obj_props"])
    ni = len(data["individuals"])
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DigiStructMed — HF Ontology EER Graph v2</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#f0f4fa;--surface:#ffffff;--surface2:#e8eef7;--border:#c8d1dc;
  --text:#1a2332;--dim:#5a6578;--accent:#0b57d0;--green:#137333;--amber:#b35900;--red:#cf222e;--purple:#6e40c9
}}
html,body{{height:100%;overflow:hidden;background:var(--bg);color:var(--text);font-family:'Inter',sans-serif}}
#app{{display:grid;grid-template-columns:248px 1fr;grid-template-rows:48px 1fr;height:100vh}}
#toolbar{{grid-column:1/-1;display:flex;align-items:center;gap:8px;padding:0 14px;background:var(--surface);border-bottom:1px solid var(--border);flex-wrap:wrap;box-shadow:0 1px 0 rgba(0,0,0,.04)}}
#toolbar h1{{font-size:13px;font-weight:700;color:var(--accent);white-space:nowrap}}
.sep{{width:1px;height:22px;background:var(--border);flex-shrink:0}}
.btn{{padding:4px 10px;border-radius:6px;border:1px solid var(--border);background:var(--surface);color:var(--text);cursor:pointer;font-size:11px;transition:.15s;white-space:nowrap}}
.btn:hover{{border-color:var(--accent);color:var(--accent);background:#f3f8ff}}
.btn.active{{background:var(--accent);color:#fff;border-color:var(--accent);font-weight:600}}
#search{{padding:4px 8px;border-radius:6px;border:1px solid var(--border);background:var(--surface);color:var(--text);font-size:11px;width:150px;outline:none}}
#search:focus{{border-color:var(--accent);box-shadow:0 0 0 2px rgba(11,87,208,.15)}}
.badge{{font-size:10px;padding:2px 7px;border-radius:99px;font-weight:600}}
.b-blue{{background:#e3f0ff;color:#0550ae}}
.b-green{{background:#dafbe1;color:var(--green)}}
.b-amber{{background:#fff5d8;color:var(--amber)}}
.b-purple{{background:#f3e8ff;color:var(--purple)}}
#sidebar{{background:var(--surface);border-right:1px solid var(--border);overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:10px}}
#sidebar h3{{font-size:10px;font-weight:700;color:var(--dim);text-transform:uppercase;letter-spacing:.08em;margin-bottom:3px}}
.legend-item{{display:flex;align-items:center;gap:6px;font-size:10px;color:var(--text);margin:2px 0}}
.sw{{width:24px;height:14px;border-radius:3px;border:2px solid;flex-shrink:0}}
.sw-circ{{width:20px;height:20px;border-radius:50%;border:2px solid;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700}}
.entity-list{{display:flex;flex-direction:column;gap:1px}}
.entity-item{{padding:4px 6px;border-radius:4px;cursor:pointer;font-size:10px;font-family:'JetBrains Mono',monospace;border:1px solid transparent;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--dim)}}
.entity-item:hover{{background:var(--surface2);color:var(--accent);border-color:var(--border)}}
.entity-item.sel{{background:#e3f0ff;color:var(--accent);border-color:var(--accent);font-weight:600}}
#graph-wrap{{position:relative;overflow:hidden;background:#ffffff}}
#graph{{width:100%;height:100%;background:#ffffff}}
.sw-oval{{width:22px;height:14px;border-radius:50%;flex-shrink:0}}
#info{{position:absolute;right:10px;bottom:10px;width:280px;max-height:45vh;overflow-y:auto;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px;display:none;flex-direction:column;gap:6px;box-shadow:0 8px 32px rgba(26,35,50,.12)}}
#info.vis{{display:flex}}
#info .x{{position:absolute;top:6px;right:8px;cursor:pointer;color:var(--dim);font-size:14px}}
#info h4{{font-size:12px;font-weight:700;color:var(--accent);font-family:'JetBrains Mono',monospace}}
#info .desc{{font-size:10px;color:var(--dim);line-height:1.4}}
.vis-tooltip{{background:var(--surface)!important;border:1px solid var(--border)!important;color:var(--text)!important;border-radius:6px!important;font-size:11px!important;padding:6px 10px!important;box-shadow:0 4px 16px rgba(0,0,0,.1)!important}}
</style>
</head>
<body>
<div id="app">
<div id="toolbar">
<h1>&#9670; HF Ontology EER</h1><div class="sep"></div>
<button class="btn active" id="btn-cls" onclick="toggleLayer('cls')">Classes</button>
<button class="btn active" id="btn-rel" onclick="toggleLayer('rel')">Relationships</button>
<button class="btn" id="btn-ind" onclick="toggleLayer('ind')">Example instances</button>
<button class="btn active" id="btn-dt" onclick="toggleLayer('dt')">Datatype Props</button>
<button class="btn" id="btn-umls" onclick="toggleUmls()" title="Show UMLS: class CUI, hasUMLSConcept links, and datatype props on CUI (hidden by default for a cleaner graph)">UMLS / CUI</button>
<div class="sep"></div>
<button class="btn" onclick="network.fit()">&#x2922; Fit</button>
<button class="btn" id="btn-ph" onclick="togglePh()">&#9889; Physics</button>
<div class="sep"></div>
<input id="search" placeholder="Search…" oninput="filterList(this.value)">
<div class="sep"></div>
<span class="badge b-blue">{nc} classes</span>
<span class="badge b-green">{no} obj props</span>
<span class="badge b-amber">{nj} specializations</span>
<span class="badge b-purple">{ni} examples</span>
</div>
<div id="sidebar">
<div>
<h3>Legend</h3>
<div class="legend-item"><div class="sw" style="background:#e3f2ff;border-color:#0b57d0"></div>Class (owl:Class)</div>
<div class="legend-item"><div class="sw-circ" style="background:#e6ffef;border-color:#137333;color:#137333">d</div>Disjoint specialization</div>
<div class="legend-item"><div class="sw-circ" style="background:#fff8e6;border-color:#b35900;color:#b35900">o</div>Overlapping specialization</div>
<div class="legend-item"><div class="sw" style="background:#f6edff;border-color:#6e40c9"></div>Example instance (NamedIndividual)</div>
<div class="legend-item" style="margin-top:4px"><svg width="24" height="14"><line x1="0" y1="7" x2="18" y2="7" stroke="#137333" stroke-width="2.5"/><polygon points="18,3 24,7 18,11" fill="#137333"/></svg>subClassOf</div>
<div class="legend-item"><svg width="24" height="14"><line x1="0" y1="7" x2="18" y2="7" stroke="#0b57d0" stroke-width="1.5"/><polygon points="18,3 24,7 18,11" fill="#0b57d0"/></svg>Object property</div>
<div class="legend-item"><div class="sw-oval" style="background:#fff4d6;border:2px dashed #b35900"></div>Attribute (datatype / literal)</div>
<div class="legend-item"><svg width="24" height="14"><line x1="0" y1="7" x2="22" y2="7" stroke="#8c6b1a" stroke-width="1" stroke-dasharray="3,2"/></svg>Datatype property</div>
<div class="legend-item" style="margin-top:6px;font-size:10px;color:var(--dim);line-height:1.35">UMLS / CUI layer is <b>off</b> at load (toolbar) — turn on to see vocabulary anchors.</div>
</div>
<div>
<h3>Classes <span id="cnt" style="color:var(--dim)">({nc})</span></h3>
<div class="entity-list" id="elist"></div>
</div>
</div>
<div id="graph-wrap">
<div id="graph"></div>
<div id="info"><span class="x" onclick="closeI()">&#215;</span><h4 id="ititle"></h4><div class="desc" id="idesc"></div></div>
</div>
</div>
<script>
const D={d};
const nDS=new vis.DataSet(),eDS=new vis.DataSet();
const layers={{cls:true,rel:true,ind:false,dt:true}};
let showUmls=false;
let phOn=false;

function layerOff(g){{
  if(g==='cls')return !layers.cls;
  if(g==='rel')return !layers.rel;
  if(g==='dt')return !layers.dt;
  if(g==='ind')return !layers.ind;
  return false;
}}
function nodeHidden(n){{
  return layerOff(n.group)||(!!n.u&&!showUmls);
}}
function edgeHidden(e){{
  return layerOff(e.group)||(!!e.u&&!showUmls);
}}

function diamondSvg(text){{
  var fs=11,cw=fs*0.58,lines=[text];
  if(text.length>16){{var m=Math.floor(text.length/2),s=text.lastIndexOf(' ',m);if(s<4)s=text.indexOf(' ',m);if(s>0)lines=[text.slice(0,s),text.slice(s+1)];}}
  var ml=0;lines.forEach(function(l){{if(l.length>ml)ml=l.length;}});
  var w=Math.max(110,ml*cw+60),h=Math.max(58,w*0.52),cx=w/2,cy=h/2,ts='';
  if(lines.length===1){{var e=lines[0].replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    ts='<text x="'+cx+'" y="'+cy+'" text-anchor="middle" dominant-baseline="central" font-family="Inter,sans-serif" font-size="'+fs+'" font-weight="500" fill="#0d5c2d">'+e+'</text>';
  }}else{{var lh=fs*1.35,sy=cy-lh*(lines.length-1)/2;
    lines.forEach(function(line,i){{var e=line.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      ts+='<text x="'+cx+'" y="'+(sy+i*lh)+'" text-anchor="middle" dominant-baseline="central" font-family="Inter,sans-serif" font-size="'+fs+'" font-weight="500" fill="#0d5c2d">'+e+'</text>';
    }});
  }}
  return 'data:image/svg+xml,'+encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="'+w+'" height="'+h+'">'
    +'<polygon points="'+cx+',2 '+(w-2)+','+cy+' '+cx+','+(h-2)+' 2,'+cy+'" fill="#d4edda" stroke="#137333" stroke-width="2.5"/>'
    +ts+'</svg>');
}}

function build(){{
  nDS.clear();eDS.clear();

  // Seed layout from taxonomy levels (left=specialized, right=roots) to reduce edge crossings vs random start
  const XGAP=380,YGAP=86;
  const maxL=D.max_level??0;
  const byLv={{}};D.classes.forEach(c=>{{const L=c.level??0;(byLv[L]=byLv[L]||[]).push(c);}});
  Object.keys(byLv).forEach(k=>byLv[k].sort((a,b)=>(a.label||a.id).localeCompare(b.label||b.id)));
  const classPos={{}};const juncPos={{}};const dtPerDom={{}};function rndOff(s){{let h=0;for(let i=0;i<s.length;i++)h=((h<<5)-h)+s.charCodeAt(i)|0;return h;}}
  Object.keys(byLv).forEach(k=>{{
    const L=+k;const arr=byLv[k];const x=(maxL-L)*XGAP;
    arr.forEach((c,i)=>{{const y=(i-(arr.length-1)/2)*YGAP;classPos[c.id]={{x,y}};}});
  }});
  D.junctions.forEach(j=>{{
    const p=classPos[j.parent];
    juncPos[j.id]=p?{{x:p.x-XGAP*0.44,y:p.y}}:{{x:0,y:0}};
  }});

  // Class nodes — light fills + strong borders + dark text
  D.classes.forEach(c=>{{
    const isRoot=c.level===0;
    const pos=classPos[c.id]||{{x:0,y:0}};
    nDS.add({{
      id:'c_'+c.id, label:c.label, title:'<b>'+c.label+'</b>'+(c.comment?'<br>'+c.comment:''),
      x:pos.x,y:pos.y,shape:'box', group:'cls',
      color:{{background:isRoot?'#cfe5ff':'#e3f2ff',border:isRoot?'#0550ae':'#0b57d0',highlight:{{background:'#b6d9ff',border:'#0344a3'}},hover:{{background:'#d7ebff',border:'#0550ae'}}}},
      font:{{color:'#032563',size:isRoot?14:11,face:'JetBrains Mono,monospace',bold:isRoot}},
      borderWidth:isRoot?3:2, mass:isRoot?3:1,
      u:c.id==='CUI', hidden:nodeHidden({{group:'cls',u:c.id==='CUI'}}), _data:c,
    }});
  }});

  // Junction nodes (EER specialization circles)
  D.junctions.forEach(j=>{{
    const isD=j.type==='d';
    const pos=juncPos[j.id]||{{x:0,y:0}};
    nDS.add({{
      id:j.id, label:j.type, title:(isD?'Disjoint':'Overlapping')+' specialization of '+j.parent,
      x:pos.x,y:pos.y,shape:'circle', size:14, group:'cls',
      color:{{background:isD?'#e6ffef':'#fff8e6',border:isD?'#137333':'#b35900',highlight:{{background:isD?'#ccf7d9':'#ffeebd',border:isD?'#0d5c2d':'#8f4700'}},hover:{{background:isD?'#d8fae4':'#fff4d6',border:isD?'#137333':'#b35900'}}}},
      font:{{color:isD?'#04491f':'#6b3b00',size:11,face:'Inter',bold:true}},
      borderWidth:2, u:false, hidden:nodeHidden({{group:'cls',u:false}}),
    }});
  }});

  // Junction edges
  D.junction_edges.forEach((e,i)=>{{
    const isToParent=D.junctions.some(j=>j.id===e.from);
    eDS.add({{
      id:'je_'+i, from:'c_'+e.from.replace('c_',''), to:isToParent?'c_'+e.to:e.to,
      from: e.from.startsWith('junc_')?e.from:'c_'+e.from,
      to: e.to.startsWith('junc_')?e.to:'c_'+e.to,
      color:{{color:'#137333',highlight:'#0d5c2d',hover:'#18a957'}},
      width:2.5, arrows:{{to:{{enabled:true,scaleFactor:1,type:'triangle'}}}},
      smooth:false, u:false, hidden:edgeHidden({{group:'cls',u:false}}), group:'cls',
    }});
  }});

  // Remaining taxonomy (single-child subClassOf)
  D.taxonomies.forEach((t,i)=>{{
    eDS.add({{
      id:'tax_'+i, from:'c_'+t.child, to:'c_'+t.parent,
      label:'', color:{{color:'#137333',highlight:'#0d5c2d',hover:'#18a957'}},
      width:2, arrows:{{to:{{enabled:true,scaleFactor:1,type:'triangle'}}}},
      smooth:false, u:false, hidden:edgeHidden({{group:'cls',u:false}}), group:'cls',
    }});
  }});

  // Object property diamond nodes (EER-style relationship diamonds)
  D.obj_props.forEach((p,i)=>{{
    const opU=p.name==='hasUMLSConcept'||p.range==='CUI'||p.domain==='CUI';
    const diamId='rel_'+i;
    const bFrom=classPos[p.domain]||{{x:0,y:0}};
    const bTo=classPos[p.range]||{{x:0,y:0}};
    let mx=(bFrom.x+bTo.x)/2, my=(bFrom.y+bTo.y)/2;
    if(p.domain===p.range){{mx+=80;my-=70;}}
    nDS.add({{
      id:diamId, label:'', title:p.name+': '+p.domain+' → '+p.range,
      x:mx,y:my, shape:'image', image:diamondSvg(p.label),
      shapeProperties:{{useImageSize:true}},
      group:'rel',
      u:opU, hidden:nodeHidden({{group:'rel',u:opU}}),
    }});
    eDS.add({{
      id:'ope_f_'+i, from:'c_'+p.domain, to:diamId,
      color:{{color:'#0b57d0',highlight:'#0344a3',hover:'#1565d8'}},
      width:1.5, arrows:{{to:{{enabled:false}}}},
      smooth:false, u:opU, hidden:edgeHidden({{group:'rel',u:opU}}), group:'rel',
    }});
    eDS.add({{
      id:'ope_t_'+i, from:diamId, to:'c_'+p.range,
      color:{{color:'#0b57d0',highlight:'#0344a3',hover:'#1565d8'}},
      width:1.5, arrows:{{to:{{enabled:true,scaleFactor:.8,type:'arrow'}}}},
      smooth:false, u:opU, hidden:edgeHidden({{group:'rel',u:opU}}), group:'rel',
    }});
  }});

  // Datatype property nodes — fan out by domain to separate attribute spokes
  D.dt_props.forEach((p,i)=>{{
    const dpTitle = p.range ? (p.name+': '+p.domain+' → '+p.range) : (p.name+': '+p.domain);
    const b=classPos[p.domain]||{{x:0,y:0}};
    const slot=(dtPerDom[p.domain]=(dtPerDom[p.domain]??0)+1)-1;
    const spread=38;const row=8;
    const dx=-270,dy=(slot%row-(row-1)/2)*spread+Math.floor(slot/row)*22+(rndOff(p.name)%13);
    const dtU=p.domain==='CUI';
    nDS.add({{
      id:'dp_'+i, label:p.label, title:dpTitle,
      x:b.x+dx,y:b.y+dy,group:'dt', shape:'ellipse',
      color:{{background:'#fff4d6',border:'#b35900',highlight:{{background:'#ffe8a8',border:'#8f4700'}},hover:{{background:'#ffedb8',border:'#9a4d00'}}}},
      font:{{color:'#5c3b00',size:10,face:'JetBrains Mono,monospace'}},
      borderWidth:2, borderDashes:true,
      u:dtU, hidden:nodeHidden({{group:'dt',u:dtU}}),
    }});
    eDS.add({{
      id:'dpe_'+i, from:'c_'+p.domain, to:'dp_'+i,
      color:{{color:'#8c6b1a',highlight:'#6b4f00',hover:'#a67c1a'}}, width:1.2, dashes:true,
      arrows:{{to:{{enabled:false}}}}, smooth:{{type:'cubicBezier',roundness:0.35}},
      u:dtU, hidden:edgeHidden({{group:'dt',u:dtU}}), group:'dt',
    }});
  }});

  // Individual nodes — near first asserted type
  D.individuals.forEach((ind,i)=>{{
    const t0=ind.types&&ind.types[0];
    const bp=t0&&classPos[t0]?classPos[t0]:{{x:-XGAP*2,y:(i%14)*YGAP-420}};
    nDS.add({{
      id:'i_'+ind.id, label:ind.label,
      x:bp.x-160,y:bp.y+(i%6)*28+(rndOff(ind.id)%17),shape:'box', group:'ind',
      color:{{background:'#f6edff',border:'#6e40c9',highlight:{{background:'#ead9ff',border:'#53389b'}},hover:{{background:'#f0e2ff',border:'#6e40c9'}}}},
      font:{{color:'#4a206e',size:10,face:'JetBrains Mono,monospace',italic:true}},
      borderWidth:1.5, u:false, hidden:nodeHidden({{group:'ind',u:false}}),
    }});
    ind.types.forEach(t=>{{
      eDS.add({{
        id:'it_'+ind.id+'_'+t, from:'i_'+ind.id, to:'c_'+t,
        label:'rdf:type', color:{{color:'#6e40c9',highlight:'#53389b'}},
        width:1, dashes:[4,3],
        arrows:{{to:{{enabled:true,scaleFactor:.7,type:'arrow'}}}},
        font:{{color:'#4a206e',size:8,face:'Inter',strokeWidth:2,strokeColor:'#ffffff'}},
        smooth:false, u:false, hidden:edgeHidden({{group:'ind',u:false}}), group:'ind',
      }});
    }});
  }});
}}

build();

const container=document.getElementById('graph');
const network=new vis.Network(container,{{nodes:nDS,edges:eDS}},{{
  nodes:{{
    shadow:{{enabled:true,size:8,x:2,y:2,color:'rgba(26,35,50,.08)'}},
    font:{{color:'#1a2332',face:'Inter',size:12}},
  }},
  groups:{{
    dt:{{
      shape:'ellipse',
      borderWidth:2,
      color:{{
        background:'#fff4d6',border:'#b35900',
        highlight:{{background:'#ffe8a8',border:'#8f4700'}},
        hover:{{background:'#ffedb8',border:'#9a4d00'}}
      }},
      font:{{color:'#5c3b00',size:10,face:'JetBrains Mono,monospace'}},
    }},
  }},
  edges:{{selectionWidth:1.5}},
  interaction:{{hover:true,tooltipDelay:200,keyboard:true,zoomView:true,dragView:true}},
  physics:{{
    enabled:true,
    forceAtlas2Based:{{
      gravitationalConstant:-115,
      centralGravity:0.0012,
      springLength:275,
      springConstant:0.032,
      damping:0.52,
      avoidOverlap:0.72,
    }},
    solver:'forceAtlas2Based',
    stabilization:{{iterations:720,updateInterval:4,fit:true}},
  }},
  layout:{{improvedLayout:true}},
}});

network.on('stabilizationIterationsDone',()=>{{
  network.setOptions({{physics:{{enabled:false}}}});
  phOn=false;
  network.fit();
}});

function refreshVisibility(){{
  nDS.get().forEach(n=>nDS.update({{id:n.id,hidden:nodeHidden(n)}}));
  eDS.get().forEach(e=>eDS.update({{id:e.id,hidden:edgeHidden(e)}}));
  network.stabilize(180);
}}

function toggleLayer(l){{
  layers[l]=!layers[l];
  document.getElementById('btn-'+l).classList.toggle('active',layers[l]);
  refreshVisibility();
}}

function toggleUmls(){{
  showUmls=!showUmls;
  const b=document.getElementById('btn-umls');
  if(b)b.classList.toggle('active',showUmls);
  refreshVisibility();
}}

function togglePh(){{
  phOn=!phOn;
  network.setOptions({{physics:{{enabled:phOn}}}});
  document.getElementById('btn-ph').classList.toggle('active',phOn);
}}

network.on('click',p=>{{
  if(p.nodes.length>0){{
    const nid=p.nodes[0];
    if(nid.startsWith('c_'))showI(nid.slice(2));
    else if(nid.startsWith('i_'))showI(nid.slice(2));
  }} else closeI();
}});

function showI(id){{
  const c=D.classes.find(x=>x.id===id);
  const ind=D.individuals.find(x=>x.id===id);
  const item=c||ind;
  if(!item)return;
  document.getElementById('ititle').textContent=item.label||item.id;
  document.getElementById('idesc').textContent=item.comment||('Types: '+(item.types||[]).join(', '));
  document.getElementById('info').classList.add('vis');
  document.querySelectorAll('.entity-item').forEach(el=>el.classList.toggle('sel',el.textContent===item.id));
}}
function closeI(){{document.getElementById('info').classList.remove('vis');document.querySelectorAll('.entity-item').forEach(el=>el.classList.remove('sel'))}}

function buildList(f){{
  const el=document.getElementById('elist');el.innerHTML='';
  const list=f?D.classes.filter(c=>c.id.toLowerCase().includes(f.toLowerCase())):D.classes;
  document.getElementById('cnt').textContent='('+list.length+')';
  list.forEach(c=>{{
    const d=document.createElement('div');d.className='entity-item';d.textContent=c.id;
    d.onclick=()=>{{network.focus('c_'+c.id,{{scale:1.1,animation:{{duration:400}}}});showI(c.id)}};
    el.appendChild(d);
  }});
}}
function filterList(v){{buildList(v)}}
buildList('');
</script>
</body>
</html>"""


def main():
    print(f"Parsing {TTL_PATH} ...")
    data = parse_ttl(TTL_PATH)
    print(f"  Classes: {len(data['classes'])}")
    print(f"  Junctions (EER specializations): {len(data['junctions'])}")
    print(f"  Object properties: {len(data['obj_props'])}")
    print(f"  Datatype properties: {len(data['dt_props'])}")
    print(f"  Example instances (NamedIndividual): {len(data['individuals'])}")

    html = render_html(data)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Written: {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
