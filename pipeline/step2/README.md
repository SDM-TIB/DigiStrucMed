# Step 2 — Normalization + extraction plan

Ontology-driven normalization following the ⟨O, S, M⟩ paradigm. All runnable Python for this step lives under `pipeline/step2/` (except shared `pipeline/step1/utils.py`).

## Layout

- **`normalize_tables.py`**, **`normalize_text.py`** — v1 config-aware normalizers (`guideline_config.json` in this folder).
- **`normalize_tables_v2.py`**, **`normalize_text_v2.py`** — generic v2 normalizers.
- **`run_normalize.py`** — run v1 table + text normalization (default run: `outputs/pipeline-output18`).
- **`extraction_plan.py`**, **`extraction_plan_v2.py`**, **`guideline_config.json`** — plan builder + config.

## Commands

```bash
python -m pipeline.step2.run_normalize
python -m pipeline.step2.run_normalize --run outputs/other_run
python -m pipeline.step2.extraction_plan --ontology ontology.ttl --out outputs/run/step2/extraction_plan.json
```

## Step 1 (PDF extraction)

See `pipeline/step1/` — `python -m pipeline.step1.run_extract` (defaults to `outputs/pipeline-output18/step1`).

## Step 3 (UMLS + optional Llama)

See `pipeline/step3/` — `python -m pipeline.step3.run_step3` and `COLAB_Step3_EL_Llama.ipynb` at repo root.

### Architecture (normalizers)

1. **`extraction_plan.py`** reads the ontology TTL and produces `extraction_plan.json` (classes, properties, `role_hints`).
2. **`normalize_tables.py`** classifies every extracted table into a role and writes ontology-shaped CSVs.
3. **`normalize_text.py`** fills text slots (assessments, adverse events, etc.).

### PDF extraction (before step 2)

Use the same `--version` for text and tables:

```bash
python -m pipeline.step1.extract_text --out outputs/my-run/step1
python -m pipeline.step1.extract_tables --out outputs/my-run/step1
python -m pipeline.step1.extract_text --out outputs/my-run/step1 --version v2
python -m pipeline.step1.extract_tables --out outputs/my-run/step1 --version v2
```

### Output layout (`step2/`)

`S_recommendation.csv`, `S_drug.csv`, `S_disease_stage.csv`, `S_phenotype.csv`, `S_cause.csv`, link CSVs, `S_assessment.csv`, `references/S_reference_*.csv`, `S_unmatched/`, `normalization_report.json`, etc.
