# Stage A evaluation

Evaluates **Stage A** extraction outputs only (text + tables). Compares one or more versions (v1, v2, and any future versions). No other pipeline stages (e.g. Stage B) are run.

## Prerequisites

- Stage A outputs must exist under `outputs/STAGE_A_{version}/`:
  - `text.json`
  - `tables.json`
- Run from **project root**.

## Usage

**Default (evaluate v1 and v2):**

```bash
python evaluation/evaluate_stage_a.py
```

**Specific versions (e.g. v1, v2, v3):**

```bash
python evaluation/evaluate_stage_a.py --versions v1 v2 v3
```

## Outputs

Written to `outputs/evaluation/`:

- **stage_a_evaluation_report.json** – Full metrics and scores per version (schema, text_metrics, table_metrics, table_spo_usability, scores) plus a comparison section.
- **stage_a_evaluation_report.txt** – Evaluation report: run time, paths evaluated, results (scores and metrics), short analysis, and confirmation that the evaluation completed.

## Scores (0–1)

- **schema** – Validity of `text.json` and `tables.json` structure.
- **text** – Content volume, empty pages, noise (TOC-like lines, URLs).
- **table** – Caption presence, row/column structure, SPO usability (triples derivable from table rows).
- **composite** – Equal weight: schema + text + table (Stage A only).

## How we measure each metric (and why)

| Metric | How we measure it | Why it matters |
|--------|-------------------|----------------|
| **Schema** | Check every text item has `source_file`, `page`, `text`; every table item has `source_file`, `page`, `caption`, `rows`; no duplicate (source_file, page). | Pipeline and Stage B expect this structure; invalid schema breaks loading. |
| **Text score** | Count pages and total words; count empty/near-empty pages, TOC-like lines (dots + page number), references headings, URLs. Penalize if body text contains many table-like lines (see Table-in-text). | We need enough prose for recommendations/NER and minimal noise; tables should not appear in body text. |
| **Table-in-text** | Scan body text for lines with 2+ pipes (`\|`) and for “Table N” mentions; count pages with such content and ratio of pipe-rows to total lines. | If tables are dumped into text, separation is poor and text score is reduced. |
| **Table score** | Count tables, how many have a caption, row/column consistency, empty-cell ratio. Run SPO conversion on each table’s rows and count triples produced. | Tables in `tables.json` feed SPO triples; captions and structure improve classification and yield. |
| **Composite** | One-third schema + one-third text + one-third table. | Single number to compare versions; all three aspects matter for Stage A quality. |

---

## Stage C evaluation

Evaluates **Stage C** NER outputs (statements with entities + enriched table triples). Compares v1 (biomedical NER) vs v2 (general NER) and any future versions.

### Prerequisites

- Stage C outputs under `outputs/STAGE_C_{version}/`: `stage_c_statements_with_entities.json`
- Run from **project root**

### Usage

```bash
python evaluation/evaluate_stage_c.py
python evaluation/evaluate_stage_c.py --versions v1 v2 v3
```

### Outputs

- **stage_c_evaluation_report.json** – Full metrics and scores per version
- **stage_c_evaluation_report.txt** – Human-readable report

### Scores (0–1)

- **schema** – Validity of statements and table_triples structure
- **entity_coverage** – How many statements have entities; total entity count
- **biomedical_relevance** – Fraction of entities with biomedical labels (Disease_disorder, Medication, etc.)
- **entity_quality** – Avg confidence score; uniqueness of entity texts
- **table_enrichment** – Fraction of table triples with NER entities
- **composite** – Weighted: schema 15%, coverage 30%, biomedical 25%, quality 15%, table 15%

### Metrics

| Metric | How we measure it | Why it matters |
|--------|-------------------|----------------|
| Schema | Check statements have chunk_id, page, source, text, entities; triples have entities | Downstream stages expect this structure |
| Entity coverage | Total entities; statements_with_entities; entity_coverage_ratio | More entities = more concepts for linking/facts |
| Biomedical relevance | Fraction of entities with labels in {Medication, Disease_disorder, Diagnostic_procedure, Sign_symptom, Therapeutic_procedure} | v1 yields these; v2 (RoBERTa CoNLL) yields PER/ORG/LOC/MISC which get filtered out |
| Entity quality | Avg/min/max score; unique entity texts | Higher scores = more confident; less repetition |
| Table enrichment | Triples with ≥1 entity; enrichment_ratio | Enriched triples feed Stage D linking |

### Why v1 vs v2 differs

- **v1** (d4data/biomedical-ner-all): Biomedical NER; outputs Disease_disorder, Medication, etc. → high biomedical_ratio, high entity coverage
- **v2** (Jean-Baptiste/roberta-large-ner-english): CoNLL NER (PER, ORG, LOC, MISC); filtered by keep_labels → 0 entities in output, low scores
