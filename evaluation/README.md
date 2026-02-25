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
