"""
Step 1e — LLM Disambiguation  [NEURAL — conditional]
─────────────────────────────────────────────────────────────────────────────
Input  : outputs/step1/grounded_entities.json   (from Step 1d)
           → only processes records where linking_status == "needs_disambiguation"
Output : outputs/step1/resolved_entities.json
           → same records with cui_final filled in (or "no_match_after_llm")

LLM backends (pick one):
  openai       — OpenAI Chat Completions (OPENAI_API_KEY)
  hf_local     — Load a HF causal LM locally (Llama 3, etc.). Uses HF_TOKEN for
                 gated models; model cached after first load.
  hf_inference — Hugging Face Serverless Inference API (same token; no local GPU)
  none         — Skip LLM calls (pass-through)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from utils import log, save_json

from hf_llm import hf_inference_chat, hf_local_generate, parse_cui_answer

# ── CoT Prompt Template ────────────────────────────────────────────────────

COT_PROMPT = """\
You are a medical terminology expert helping to link clinical text to UMLS concepts.

Entity text    : "{entity_text}"
Entity type    : "{entity_type}"
Sentence context: "{context}"

Candidate UMLS concepts:
{candidates_block}

Task:
1. Read the entity text and its context carefully.
2. For each candidate, consider whether the CUI label and definition match the \
clinical meaning in the sentence.
3. Select the single best CUI.
4. If no candidate is appropriate, respond with NO_MATCH.

Respond with ONLY the CUI string (e.g. C0018801) or NO_MATCH. No explanation.

Answer:"""


def _build_candidates_block(candidates: list[dict]) -> str:
    lines: list[str] = []
    for i, c in enumerate(candidates, start=1):
        defn = c.get("definition", "—")[:180]
        lines.append(
            f"{i}. CUI: {c['cui']}\n"
            f"   Label: {c['label']}\n"
            f"   Semantic type: {c.get('semantic_type', '—')}\n"
            f"   Definition: {defn}"
        )
    return "\n\n".join(lines)


def _call_openai(prompt: str, client, model: str = "gpt-4") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def disambiguate(
    grounded_path: str = "outputs/step1/grounded_entities.json",
    output_dir: str = "outputs/step1",
    llm_backend: str = "none",
    # OpenAI
    openai_api_key: Optional[str] = None,
    openai_model: str = "gpt-4",
    # Hugging Face (Hub token — gated models + Inference API)
    hf_token: Optional[str] = None,
    hf_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    max_new_tokens: int = 64,
    # Legacy alias (maps to hf_model if set)
    llama_model: Optional[str] = None,
) -> list[dict]:
    """
    Resolve ambiguous CUI candidates using an LLM.

    llm_backend:
      "openai"       — requires openai_api_key
      "hf_local"     — full model on GPU/CPU; hf_token for gated checkpoints
      "hf_inference" — HF Serverless chat completions; requires hf_token
      "llama"        — treated as "hf_local" (backward compatibility)
      "none"         — skip
    """
    if llm_backend == "llama":
        llm_backend = "hf_local"

    model_id = llama_model or hf_model

    entities: list[dict] = json.loads(
        Path(grounded_path).read_text(encoding="utf-8")
    )

    to_resolve = [e for e in entities if e.get("linking_status") == "needs_disambiguation"]
    log("1e", f"{len(to_resolve)} entities need disambiguation (backend: {llm_backend})")

    if not to_resolve or llm_backend == "none":
        log("1e", "Skipping LLM disambiguation (no entities or backend=none)")
        save_json(entities, str(Path(output_dir) / "resolved_entities.json"))
        return entities

    client = None
    if llm_backend == "openai":
        if not openai_api_key:
            log("1e", "WARNING: openai backend but no OPENAI_API_KEY — skipping")
            save_json(entities, str(Path(output_dir) / "resolved_entities.json"))
            return entities
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key)

    elif llm_backend == "hf_inference":
        if not hf_token:
            log("1e", "WARNING: hf_inference requires HF_TOKEN — skipping")
            save_json(entities, str(Path(output_dir) / "resolved_entities.json"))
            return entities

    elif llm_backend == "hf_local":
        pass  # token optional for public models

    # ── Resolve each ambiguous entity ──────────────────────────────────────
    resolved = 0
    for entity in to_resolve:
        prompt = COT_PROMPT.format(
            entity_text=entity["text"],
            entity_type=entity.get("type", ""),
            context=entity.get("source_text", "")[:300],
            candidates_block=_build_candidates_block(entity.get("cui_candidates", [])),
        )

        try:
            if client:
                raw = _call_openai(prompt, client, model=openai_model)
            elif llm_backend == "hf_inference":
                raw = hf_inference_chat(
                    prompt, model_id, hf_token, max_new_tokens=max_new_tokens
                )
            elif llm_backend == "hf_local":
                raw = hf_local_generate(
                    prompt, model_id, hf_token, max_new_tokens=max_new_tokens
                )
            else:
                raw = "NO_MATCH"

            answer = parse_cui_answer(raw)

            if answer and answer != "NO_MATCH" and answer.startswith("C"):
                entity["cui_final"] = answer
                entity["linking_status"] = "disambiguated"
                resolved += 1
            else:
                entity["cui_final"] = None
                entity["linking_status"] = "no_match_after_llm"

        except Exception as exc:
            log("1e", f"LLM error for '{entity['text']}': {exc}")
            entity["linking_status"] = "llm_error"

    out_path = Path(output_dir) / "resolved_entities.json"
    save_json(entities, str(out_path))
    total_linked = sum(1 for e in entities if e.get("cui_final"))
    log("1e", f"Resolved {resolved}/{len(to_resolve)} — total with CUI: {total_linked}/{len(entities)}")
    return entities


if __name__ == "__main__":
    import sys
    disambiguate(openai_api_key=sys.argv[1] if len(sys.argv) > 1 else None)
