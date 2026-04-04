"""
Step 1e — LLM Disambiguation  [NEURAL — conditional]
─────────────────────────────────────────────────────────────────────────────
Input  : outputs/step1/grounded_entities.json   (from Step 1d)
           → only processes records where linking_status == "needs_disambiguation"
Output : outputs/step1/resolved_entities.json
           → same records with cui_final filled in (or "no_match_after_llm");
           cui_candidates are removed on write (Step 1d grounded file keeps them).

LLM: Llama (or other causal LM) loaded locally via Hugging Face transformers.
     Set HF_TOKEN for gated Hub models (pass hf_token= or environment variable).
     Ambiguous mentions are processed in batches (default 8) with progress logs.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from utils import log, save_json

from hf_llm import (
    check_hf_local_backend,
    format_exception,
    hf_local_generate,
    hf_local_generate_batch,
    parse_cui_answer,
    probe_hf_local_backend,
)

DEFAULT_DISAMBIGUATION_BATCH_SIZE = 8

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


def _save_resolved_entities(entities: list[dict], output_dir: str) -> None:
    """Write resolved_entities.json without cui_candidates (cui_final only)."""
    for e in entities:
        e.pop("cui_candidates", None)
    save_json(entities, str(Path(output_dir) / "resolved_entities.json"))


def disambiguate(
    grounded_path: str = "outputs/step1/grounded_entities.json",
    output_dir: str = "outputs/step1",
    hf_token: Optional[str] = None,
    hf_model: str = "meta-llama/Llama-3.2-3B-Instruct",
    max_new_tokens: int = 64,
    llama_model: Optional[str] = None,
    batch_size: int = DEFAULT_DISAMBIGUATION_BATCH_SIZE,
) -> list[dict]:
    """
    Resolve ambiguous CUI candidates using a locally loaded HF causal LM (e.g. Llama).

    When at least one mention needs disambiguation, HF_TOKEN must be set (argument or
    environment) for gated models on the Hub.

    batch_size: how many prompts to run per GPU forward (default 8).
    """
    model_id = llama_model or hf_model
    token = (hf_token or os.environ.get("HF_TOKEN") or "").strip() or None

    entities: list[dict] = json.loads(
        Path(grounded_path).read_text(encoding="utf-8")
    )

    to_resolve = [e for e in entities if e.get("linking_status") == "needs_disambiguation"]
    log("1e", f"{len(to_resolve)} entities need disambiguation (hf_local / Llama)")

    if not to_resolve:
        _save_resolved_entities(entities, output_dir)
        return entities

    if not token:
        raise ValueError(
            "Step 1e requires HF_TOKEN (Hugging Face Hub) for Llama/gated models. "
            "Set the environment variable or pass hf_token=..."
        )

    backend_error = check_hf_local_backend()
    if backend_error:
        raise RuntimeError(f"Step 1e: local LLM stack unavailable. {backend_error}")

    probe_error = probe_hf_local_backend(model_id, token)
    if probe_error:
        raise RuntimeError(f"Step 1e: model load/generation probe failed. {probe_error}")

    def _apply_raw_to_entity(entity: dict, raw: str) -> bool:
        """Update entity from one model output. Returns True if a CUI was assigned."""
        answer = parse_cui_answer(raw)
        if answer and answer != "NO_MATCH" and answer.startswith("C"):
            entity["cui_final"] = answer
            entity["linking_status"] = "disambiguated"
            return True
        entity["cui_final"] = None
        entity["linking_status"] = "no_match_after_llm"
        return False

    total_r = len(to_resolve)
    resolved = 0
    bs = max(1, int(batch_size))

    for start in range(0, total_r, bs):
        chunk = to_resolve[start : start + bs]
        prompts = [
            COT_PROMPT.format(
                entity_text=e["text"],
                entity_type=e.get("type", ""),
                context=e.get("source_text", "")[:300],
                candidates_block=_build_candidates_block(e.get("cui_candidates", [])),
            )
            for e in chunk
        ]

        batch_ok: list[str] | None
        try:
            batch_ok = hf_local_generate_batch(
                prompts, model_id, token, max_new_tokens=max_new_tokens
            )
            if len(batch_ok) != len(chunk):
                raise ValueError(
                    f"batch returned {len(batch_ok)} strings, expected {len(chunk)}"
                )
        except Exception as batch_exc:
            log(
                "1e",
                f"Batch generate failed ({format_exception(batch_exc)}); retrying chunk one-by-one.",
            )
            batch_ok = None

        if batch_ok is not None:
            for entity, raw in zip(chunk, batch_ok):
                if _apply_raw_to_entity(entity, raw):
                    resolved += 1
        else:
            for entity, p in zip(chunk, prompts):
                try:
                    raw = hf_local_generate(
                        p, model_id, token, max_new_tokens=max_new_tokens
                    )
                except Exception as exc:
                    log(
                        "1e",
                        f"LLM error for '{entity['text']}': {format_exception(exc)}",
                    )
                    entity["linking_status"] = "llm_error"
                    continue
                if _apply_raw_to_entity(entity, raw):
                    resolved += 1

        done = min(start + len(chunk), total_r)
        log("1e", f"Progress: {done}/{total_r} entities disambiguated (batch size {bs})")

    _save_resolved_entities(entities, output_dir)
    total_linked = sum(1 for e in entities if e.get("cui_final"))
    log("1e", f"Resolved {resolved}/{len(to_resolve)} — total with CUI: {total_linked}/{len(entities)}")
    return entities


if __name__ == "__main__":
    disambiguate(
        hf_token=os.environ.get("HF_TOKEN"),
        hf_model=os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-3B-Instruct"),
    )
