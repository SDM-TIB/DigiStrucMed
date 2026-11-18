# From a patient question to a checked video prompt

A patient asks a question; the pipeline answers with a video prompt in which every spoken line comes from a clinical-guideline
fact in the knowledge graph and has been checked against that fact.

## Part 1 — facts from the knowledge graph (`graph_facts.py`, code only)

1. **Find concepts**: match the question's words to the graph's own names (spelling and word forms, abbreviations only in
   capitals, near-identical meaning match with a local MiniLM model). `find_concepts`
2. **Query the graph** with fixed SPARQL: Query A — claims (`CLAIMS`), Query B — recommendation readings (`READINGS`), plus
   `STATEMENTS` (sentence, grade, page) and `BROADER` (stop rule: step to a broader concept, at most 3 levels).
3. **Rank and filter**: direct before broader, specific concepts first, grade, recommendation before explanation; drop
   off-topic facts and near-copies; keep at most 10. `choose`
4. **Fact text with source**: triple, type, grade, guideline, page and the guideline sentence. `fact_text`

## Part 2 — from facts to a checked script (`graph_script.py`)

5. **Slots** (code): one per fact, with type, grade, caution ("may") and abbreviation meanings from the graph. `slots_of`
6. **Writer** gpt-oss-20b (Groq): one plain line per fact, at most 6 lines; skips off-topic, repeated or conflicting facts.
7. **Code checks**: numbers in the fact, caution kept, unclear abbreviations kept, on-screen words from line or fact.
8. **Judge** gpt-oss-120b (Groq, `new_judge.py`): each line against its own fact only; quote must be in the fact; asked three
   ways. A failed line is rewritten once with the reason, else dropped.
9. **Video prompt**: built by code from the kept lines, with a source table.

`text_tools.py` holds small shared helpers (JSON reading, word normalisation, the MiniLM encoder).

## Run

Needs GraphDB with the repository `Guideline_KG` at `http://127.0.0.1:7200`, `GROQ_API_KEY` (environment or `GAAPO/.env`),
and Python with `openai`, `numpy`, `torch`, `transformers` (MiniLM in the local cache), `python-dotenv`, optionally `truststore`.
Run from the folder that contains `GALLM/`:

    python -m GALLM.graph_script "When can I drive again after getting an ICD?" --name icd-driving
    python GALLM/demo_page.py        # then open http://127.0.0.1:8770

Every model call is saved, so a question asked again costs no tokens.

## Tests

    python -m unittest GALLM.tests.test_graph_facts GALLM.tests.test_graph_script GALLM.tests.test_new_judge

## Not in this repository

The graph's name index (`data/graph_names*.json/.npy`, built from GraphDB on the first run; it contains UMLS names, which stay
local), run outputs and keys.
