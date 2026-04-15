# Step 3 — UMLS linking + Llama 3.1

- **`entity_link_sources.py`** — CSV columns → UMLS; writes `grounded_entities.json`, `S_annotation_concept.csv`, `entity_linking_report.json`.
- **`disambiguate_llama.py`** — Llama 3.1 over `needs_disambiguation` rows (`HF_TOKEN`, GPU recommended).
- **`run_step3.py`** — link, then optional `--llama`.

```bash
python -m pipeline.step3.run_step3
python -m pipeline.step3.run_step3 --llama
```

Colab: `COLAB_Step3_EL_Llama.ipynb` at repository root.
