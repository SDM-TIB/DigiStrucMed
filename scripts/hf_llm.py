"""
Shared Hugging Face LLM helpers for Step 1e (disambiguation) and Step 3b (SPO).

Modes
-----
hf_local     — Load a causal LM from the Hub (Llama 3, etc.). Requires HF_TOKEN
               for gated models. Model stays cached in-process after first load.

hf_inference — Serverless Inference API (no local GPU). Same token; calls Hub.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

# One cached (tokenizer, model) per (model_id, token) for local mode
_local_cache: dict[tuple[str, str | None], tuple[Any, Any]] = {}


def _get_local_model(model_id: str, hf_token: Optional[str]) -> tuple[Any, Any]:
    key = (model_id, hf_token)
    if key in _local_cache:
        return _local_cache[key]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    _local_cache[key] = (tok, model)
    return tok, model


def hf_local_generate(
    prompt: str,
    model_id: str,
    hf_token: Optional[str],
    max_new_tokens: int = 256,
) -> str:
    """Generate with a locally loaded HF causal LM (Llama 3, Mistral, etc.)."""
    import torch

    tok, model = _get_local_model(model_id, hf_token)
    messages = [{"role": "user", "content": prompt}]

    if getattr(tok, "chat_template", None) is not None:
        inputs = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
    else:
        inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    # Decode only the new tokens
    gen = out[0, inputs.shape[-1] :]
    text = tok.decode(gen, skip_special_tokens=True)
    return text.strip()


def hf_inference_chat(
    prompt: str,
    model_id: str,
    hf_token: Optional[str],
    max_new_tokens: int = 256,
) -> str:
    """
    Hugging Face Serverless Inference API (chat completions).
    No GPU required; uses your HF token quota.
    """
    try:
        from huggingface_hub import InferenceClient
    except ImportError as e:
        raise ImportError("pip install huggingface_hub") from e

    client = InferenceClient(token=hf_token)
    out = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model=model_id,
        max_tokens=max_new_tokens,
        temperature=0.0,
    )
    choice0 = out.choices[0]
    msg = getattr(choice0, "message", None)
    content = getattr(msg, "content", None) if msg is not None else None
    if content:
        return str(content).strip()
    # Dict-like fallback
    if isinstance(choice0, dict):
        return str(choice0.get("message", {}).get("content", "")).strip()
    return ""


def hf_inference_text_generation(
    prompt: str,
    model_id: str,
    hf_token: Optional[str],
    max_new_tokens: int = 256,
) -> str:
    """Fallback: text_generation endpoint for completion-style models."""
    from huggingface_hub import InferenceClient

    client = InferenceClient(model=model_id, token=hf_token)
    return client.text_generation(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    ).strip()


def parse_cui_answer(raw: str) -> str:
    """Extract a UMLS CUI (C + 7 digits) or NO_MATCH from model output."""
    if not raw:
        return "NO_MATCH"
    up = raw.upper()
    if "NO_MATCH" in up or "NO MATCH" in raw.upper():
        return "NO_MATCH"
    m = re.search(r"\b(C\d{7})\b", raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return "NO_MATCH"


def parse_json_array(raw: str) -> list:
    """Best-effort parse of JSON array from LLM output (strips markdown fences)."""
    t = raw.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        data = json.loads(t)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        # Try to find first [...] block
        m = re.search(r"\[[\s\S]*\]", t)
        if m:
            try:
                data = json.loads(m.group(0))
                return data if isinstance(data, list) else []
            except json.JSONDecodeError:
                pass
        return []
