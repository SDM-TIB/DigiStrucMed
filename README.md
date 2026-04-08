# DigiStructMed

Automated knowledge graph construction from medical PDF guidelines. Extracts text and tables from clinical PDFs, recognizes and links biomedical entities to UMLS, maps them to an OWL ontology via RML, materializes RDF triples, and validates the result with SHACL shapes.

## Pipeline

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
| 5 | `step5_validate.py` | Validate KG against SHACL shapes derived from the ontology (TravSHACL / pySHACL fallback) |

## Running

The pipeline is designed to run on **Google Colab** (GPU recommended for LLM steps).

### Full pipeline — from PDF to validated KG

Upload `DigiStructMed_thesis_colab.zip` to Colab, unzip it, place your input PDF and ontology in the `input/` folder, then open and run:

```
COLAB_Pipeline.ipynb
```

### Resume from Step 3a — skip extraction/NER, start from table mappings

If extraction and NER outputs are already available (e.g. from a previous run), use the lighter notebook:

```
COLAB_From_Step3a.ipynb
```

### Build the Colab zip (local)

```bash
python create_colab_zip.py
# Writes DigiStructMed_thesis_colab.zip — upload this to Colab
```

### Visualize outputs (local)

```bash
# Interactive KG viewer (Cytoscape.js, offline-ready)
python visualize_kg.py --kg outputs/<run-dir>/step4/output_v2.ttl

# Regenerate the ontology graph
python visualize_ontology.py
# Output: input/ontology_graph.html
```

## Project structure

```
scripts/              Step scripts (step1a – step5, utils, hf_llm)
input/                Source PDFs, ontology, UMLS CSV, ontology_graph.html
outputs/              Pipeline run directories (pipeline-output<N>/)
COLAB_Pipeline.ipynb        Full pipeline notebook (Colab)
COLAB_From_Step3a.ipynb     Resume-from-step3a notebook (Colab)
create_colab_zip.py         Packages scripts + input stubs for Colab upload
visualize_kg.py             Interactive KG visualization (Cytoscape.js)
visualize_ontology.py       Regenerate ontology_graph.html
input/FIXES_LOG.md          Per-run change log and metrics
```

## Requirements

- Python 3.10+
- See `requirements.txt` — key dependencies: PyMuPDF, Docling, Transformers, RDFLib, RDFizer, pySHACL/TravSHACL, RapidFuzz
