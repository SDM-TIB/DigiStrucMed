# Step 1 — PDF extraction

- **`extract_text.py`** / **`extract_tables.py`** — same `--version` for both (`v1` PyMuPDF/pdfplumber, `v2` Docling).
- **`run_extract.py`** — runs both; default output `outputs/pipeline-output18/step1`.
- **`utils.py`** — JSON + logging (also used by step 2 and step 3).

```bash
python -m pipeline.step1.run_extract
python -m pipeline.step1.extract_text --help
```
