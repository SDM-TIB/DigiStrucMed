# DigiStructMed

Automated knowledge graph construction from medical PDF guidelines. Extracts text and tables from clinical PDFs, recognizes and links biomedical entities to UMLS, maps them to an OWL ontology via RML, materializes RDF triples, and validates the result with SHACL shapes.

## Pipeline

The pipeline runs as a sequence of steps, each reading the previous step's output. Versions are configurable per stage via `config.json`.

```
PDF  ──►  1a Text extraction  ──►  1b Table extraction
               │                         │
               ▼                         │
          1c NER (biomedical)            │
               │                         │
               ▼                         │
          1d Entity linking (UMLS)       │
               │                         │
               ▼                         │
          1e LLM disambiguation          │
               │                         │
               ▼                         ▼
          2  Load ontology ──────►  3a Table mappings (RML)
               │                    3b Text path (RML)
               │                         │
               ▼                         ▼
                    4  Materialize KG (RDF)
                           │
                           ▼
                    5  SHACL validation
```

| Step | Script | What it does |
|------|--------|--------------|
| 1a | `step1a_extract_text.py` | Extract text blocks from PDF (PyMuPDF v1 / Docling v2) |
| 1b | `step1b_extract_tables.py` | Extract tables as CSVs (pdfplumber v1 / Docling v2) |
| 1c | `step1c_ner.py` | Biomedical NER using `d4data/biomedical-ner-all` with acronym expansion, sliding-window chunking, and entity filtering |
| 1d | `step1d_entity_linking.py` | Link entity mentions to UMLS concepts via token index + fuzzy matching |
| 1e | `step1e_disambiguate.py` | LLM disambiguation for ambiguous mentions (Llama-3.1-8B, conditional) |
| 2 | `step2_load_ontology.py` | Parse and index the OWL ontology (classes, properties, enumerations) |
| 3a | `step3a_table_mappings.py` | Fuzzy-match table headers to ontology properties, generate RML |
| 3b | `step3b_text_path.py` | Generate text assertions via entity co-occurrence + ontology matching (v1 symbolic, v2 adds LLM augmentation) |
| 4 | `step4_materialize.py` | Materialize RDF triples using SDM-RDFizer / Morph-KGC / rdflib fallback |
| 5 | `step5_validate.py` | Validate KG against SHACL shapes derived from the ontology (TravSHACL / pySHACL) |

## Running

```bash
# Full pipeline with config.json versions
python run_orchestrator.py

# Override versions per stage
python run_orchestrator.py --A v2 --B v2 --C v1

# Run a subset of stages
python run_orchestrator.py --from B --to D

# Visualize the output KG
python visualize_kg.py --kg outputs/step4/output_v2.ttl
```

## Project structure

```
scripts/          Step scripts (step1a – step5, utils, hf_llm)
input/            Source PDFs, ontology, UMLS CSV
outputs/          Per-stage versioned output directories
config.json       Stage version configuration
config.py         Resolved paths and version helpers
run_orchestrator.py   End-to-end pipeline runner
visualize_kg.py       Interactive KG visualization (Cytoscape.js)
```

## Requirements

- Python 3.10+
- See `requirements.txt` — key dependencies: PyMuPDF, Docling, Transformers, RDFLib, RDFizer, pySHACL/TravSHACL, RapidFuzz