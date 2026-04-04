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
import traceback
from typing import Any, Optional

# One cached (tokenizer, model) per (model_id, token) for local mode
_local_cache: dict[tuple[str, str | None], tuple[Any, Any]] = {}


def check_hf_local_backend() -> Optional[str]:
    """
    Return None when local HF generation dependencies look usable, otherwise
    return a human-readable error message describing the first failure.
    """
    try:
        import torch  # noqa: F401
    except Exception as exc:
        return (
            "PyTorch is unavailable for local generation. "
            f"{type(exc).__name__}: {exc}"
        )

    try:
        import transformers  # noqa: F401
    except Exception as exc:
        return (
            "transformers is unavailable for local generation. "
            f"{type(exc).__name__}: {exc}"
        )

    return None


def format_exception(exc: BaseException) -> str:
    """
    Produce a compact but informative exception string.
    Some HF/torch exceptions stringify to "", so fall back to repr().
    """
    text = str(exc).strip()
    if text:
        return f"{type(exc).__name__}: {text}"
    return f"{type(exc).__name__}: {exc!r}"


def probe_hf_local_backend(
    model_id: str,
    hf_token: Optional[str],
    max_new_tokens: int = 8,
) -> Optional[str]:
    """
    End-to-end smoke test for local generation.
    Returns None on success, otherwise a readable failure summary.
    """
    dep_error = check_hf_local_backend()
    if dep_error:
        return dep_error

    try:
        _ = hf_local_generate(
            "Reply with NO_MATCH only.",
            model_id,
            hf_token,
            max_new_tokens=max_new_tokens,
        )
        return None
    except Exception as exc:
        tb = traceback.format_exc(limit=3).strip().splitlines()
        tail = tb[-1] if tb else ""
        msg = format_exception(exc)
        return f"{msg}. Probe failed while loading/generating with model '{model_id}'. {tail}".strip()


def _pick_torch_dtype(torch) -> Any:
    """
    Choose a local-generation dtype that is broadly compatible with Colab GPUs.

    T4/V100 often do not support bfloat16 well; float16 is the safer CUDA default.
    """
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _get_local_model(model_id: str, hf_token: Optional[str]) -> tuple[Any, Any]:
    key = (model_id, hf_token)
    if key in _local_cache:
        return _local_cache[key]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model_kwargs = {
        "token": hf_token,
        "torch_dtype": _pick_torch_dtype(torch),
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    _local_cache[key] = (tok, model)
    return tok, model


def _model_input_device(model: Any) -> Any:
    """
    Device for input_ids / attention_mask.
    device_map='auto' models often lack .device; prefer embedding weights, then any parameter.
    """
    import torch

    try:
        emb = model.get_input_embeddings()
        w = getattr(emb, "weight", None)
        if w is not None and hasattr(w, "device"):
            return w.device
    except Exception:
        pass
    p = next(model.parameters(), None)
    if p is not None:
        return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _generation_sequences(out: Any, torch: Any) -> Any:
    """Handle raw LongTensor vs GenerateOutput / ModelOutput with .sequences."""
    if torch.is_tensor(out):
        return out
    seq = getattr(out, "sequences", None)
    if seq is not None:
        return seq
    return out


def _tokenize_chat_prompt_ids(tok: Any, prompt: str, torch: Any) -> Any:
    """CPU 1-D LongTensor of token ids for one user message (chat template if present)."""
    messages = [{"role": "user", "content": prompt}]
    if getattr(tok, "chat_template", None) is not None:
        try:
            enc = tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except Exception:
            enc = tok(prompt, return_tensors="pt")
    else:
        enc = tok(prompt, return_tensors="pt")
    if isinstance(enc, str):
        enc = tok(prompt, return_tensors="pt")
    if isinstance(enc, torch.Tensor):
        return enc.view(-1).long().clone()
    moved = enc
    batch = dict(moved) if not isinstance(moved, dict) else moved
    input_ids = batch.get("input_ids")
    if input_ids is None:
        raise ValueError("Tokenizer output has no input_ids after chat template / encode.")
    if input_ids.dim() == 2:
        input_ids = input_ids.squeeze(0)
    return input_ids.long().clone()


def hf_local_generate(
    prompt: str,
    model_id: str,
    hf_token: Optional[str],
    max_new_tokens: int = 256,
) -> str:
    """Generate with a locally loaded HF causal LM (Llama 3, Mistral, etc.)."""
    import torch

    tok, model = _get_local_model(model_id, hf_token)
    device = _model_input_device(model)
    ids_1d = _tokenize_chat_prompt_ids(tok, prompt, torch)
    input_ids = ids_1d.unsqueeze(0).to(device)
    attn = torch.ones_like(input_ids, dtype=torch.long)

    in_len = int(input_ids.shape[-1])
    pad_id = tok.pad_token_id or tok.eos_token_id

    gen_kw: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attn,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": pad_id,
    }

    with torch.no_grad():
        raw_out = model.generate(**gen_kw)

    seq = _generation_sequences(raw_out, torch)
    if not torch.is_tensor(seq):
        raise TypeError(
            f"Unexpected generate() return type {type(raw_out)!r}; expected tensor or object with .sequences"
        )
    row = seq[0] if seq.dim() == 2 else seq
    gen = row[in_len:]
    text = tok.decode(gen, skip_special_tokens=True)
    return text.strip()


def hf_local_generate_batch(
    prompts: list[str],
    model_id: str,
    hf_token: Optional[str],
    max_new_tokens: int = 256,
) -> list[str]:
    """
    Batched greedy generation for the same model. Left-pads prompts to one tensor.
    Returns one decoded string per prompt (new tokens only).
    """
    if not prompts:
        return []

    import torch

    tok, model = _get_local_model(model_id, hf_token)
    device = _model_input_device(model)
    pad_id = tok.pad_token_id or tok.eos_token_id

    prev_pad = getattr(tok, "padding_side", "right")
    tok.padding_side = "left"
    try:
        rows_cpu = [_tokenize_chat_prompt_ids(tok, p, torch) for p in prompts]
    finally:
        tok.padding_side = prev_pad

    max_l = max(int(t.shape[0]) for t in rows_cpu)
    padded_ids: list[Any] = []
    padded_attn: list[Any] = []
    for ids in rows_cpu:
        L = int(ids.shape[0])
        pl = max_l - L
        if pl > 0:
            pads = torch.full((pl,), pad_id, dtype=ids.dtype)
            pids = torch.cat([pads, ids])
            am = torch.cat(
                [
                    torch.zeros(pl, dtype=torch.long),
                    torch.ones(L, dtype=torch.long),
                ]
            )
        else:
            pids = ids
            am = torch.ones(L, dtype=torch.long)
        padded_ids.append(pids)
        padded_attn.append(am)

    input_ids = torch.stack(padded_ids).to(device)
    attention_mask = torch.stack(padded_attn).to(device)
    in_len = int(input_ids.shape[-1])

    gen_kw: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": pad_id,
    }

    with torch.no_grad():
        raw_out = model.generate(**gen_kw)

    seq = _generation_sequences(raw_out, torch)
    if not torch.is_tensor(seq):
        raise TypeError(
            f"Unexpected generate() return type {type(raw_out)!r}; expected tensor or object with .sequences"
        )
    if seq.dim() != 2:
        raise TypeError(f"Expected 2D sequences tensor, got shape {tuple(seq.shape)}")

    gen_block = seq[:, in_len:]
    out_texts: list[str] = []
    for i in range(seq.shape[0]):
        text = tok.decode(gen_block[i], skip_special_tokens=True)
        out_texts.append(text.strip())
    return out_texts


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
