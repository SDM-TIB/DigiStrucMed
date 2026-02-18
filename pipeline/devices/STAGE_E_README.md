# Stage E: factual statement extraction (no factuality validation)

## Role

**Stage E extracts factual statements from text.**  
The LLM outputs **subject, predicate, object, exception, duration** for each fact.  
**Factuality is not assessed**—medical experts validate statements manually later.

## Pipeline position

```
Stage d: … → CandidateStatements
Stage e: CandidateStatements → Validate (LLM: extraction only) → ValidatedFactsAndQualifiers
        (experts validate later)
```

## What `validate.py` does

1. **Input:** `CandidateStatements` — one entry per chunk with `text` and linked `entities`.

2. **For each chunk:**  
   The LLM extracts **every** explicit factual statement from the paragraph.  
   A single paragraph may yield **zero, one, or several** statements.

3. **Output schema (per statement):**
   - **subject** — who or what the fact is about (e.g. patients with HFrEF, HFrEF)
   - **predicate** — verb or relation (e.g. receive, avoid, consider)
   - **object** — what the predicate applies to (e.g. ACE inhibitors)
   - **exception** — explicit exception if stated (e.g. unless contraindicated), else null
   - **duration** — explicit timeframe if stated (e.g. for 6 weeks), else null  
   Plus optional **source_text**, **chunk_id**, **page**, **source_pdf**, **entities** for traceability.

4. **Output:** `ValidatedFactsAndQualifiers` — list of extracted statements.  
   No confidence score; no accept/reject. Experts perform validation separately.

## Prompt design (chain-of-thought, NSSC-style)

Following the NSSC paper: the LLM is prompted to use **chain-of-thought (CoT)** so it reveals its reasoning step-by-step before giving the final answer.

- **Step 1:** Read the text; decide if it contains any factual statement or is only metadata/header.
- **Step 2:** If there are facts, list each one in natural language (subject, predicate, object, exception, duration).
- **Step 3:** Count how many distinct statements were identified.
- **Step 4:** Output the JSON array (one object per statement). The parser extracts the array from the full response (reasoning + array).
- Only explicit facts (no inference): “Step 1: identify each fact; Step 2: output the array.”
- Only explicit facts; no inference or added medical knowledge.
