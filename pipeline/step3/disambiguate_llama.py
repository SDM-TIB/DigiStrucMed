"""Step 3b: Llama disambiguation for ``needs_disambiguation`` rows in ``grounded_entities.json``.

Writes ``resolved_entities.json``, ``disambiguation_report.json``, and refreshes
``S_annotation_concept.csv``. Needs ``HF_TOKEN`` (or ``--hf-token``) for gated Hub models.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[2]

from pipeline.step3._hf_llm import (  # noqa: E402
    check_hf_local_backend,
    format_exception,
    hf_local_generate,
    hf_local_generate_batch,
    parse_cui_answer,
    probe_hf_local_backend,
)

from pipeline.step3.build_annotation_links import build_links_for_run  # noqa: E402
from pipeline.step3.entity_link_sources import write_s_annotation_concept  # noqa: E402
from pipeline.step1.utils import log, save_json  # noqa: E402

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_BATCH = 8

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
        defn = (c.get("definition") or "-")[:180]
        lines.append(
            f"{i}. CUI: {c.get('cui', '')}\n"
            f"   Label: {c.get('label', '')}\n"
            f"   Semantic type: {c.get('semantic_type', '-')}\n"
            f"   Definition: {defn}"
        )
    return "\n\n".join(lines)


def _label_from_candidates(cui: str, candidates: list[dict]) -> str:
    for c in candidates:
        if str(c.get("cui", "")) == str(cui):
            return str(c.get("label") or cui)
    return cui


def disambiguate_llama(
    grounded_path: str | Path,
    shaped_dir: str | Path | None = None,
    *,
    hf_token: Optional[str] = None,
    hf_model: str = DEFAULT_MODEL,
    max_new_tokens: int = 64,
    batch_size: int = DEFAULT_BATCH,
) -> dict:
    grounded_path = Path(grounded_path)
    shaped_dir = Path(shaped_dir or grounded_path.parent)
    token = (hf_token or os.environ.get("HF_TOKEN") or "").strip() or None

    entities: list[dict] = json.loads(grounded_path.read_text(encoding="utf-8"))
    to_resolve = [
        e
        for e in entities
        if e.get("linking_status") == "needs_disambiguation"
        and (e.get("cui_candidates") or [])
    ]
    log("3b", f"{len(to_resolve)} mentions need Llama disambiguation (model={hf_model})")

    if not to_resolve:
        _finalize_and_write(entities, shaped_dir, grounded_path.parent, resolved=0)
        return {"resolved": 0, "skipped": "none needed"}

    if not token:
        raise ValueError(
            "HF_TOKEN is required for Llama 3.1 on the Hub. "
            "Set the environment variable or pass hf_token=..."
        )

    err = check_hf_local_backend()
    if err:
        raise RuntimeError(f"Local HF stack unavailable: {err}")
    probe_err = probe_hf_local_backend(hf_model, token)
    if probe_err:
        raise RuntimeError(f"Model probe failed: {probe_err}")

    resolved = 0
    bs = max(1, int(batch_size))
    total_r = len(to_resolve)

    for start in range(0, total_r, bs):
        chunk = to_resolve[start : start + bs]
        prompts = [
            COT_PROMPT.format(
                entity_text=e.get("text", ""),
                entity_type=e.get("type") or e.get("source_column", ""),
                context=(e.get("source_text") or "")[:300],
                candidates_block=_build_candidates_block(e.get("cui_candidates") or []),
            )
            for e in chunk
        ]

        batch_ok: list[str] | None
        try:
            batch_ok = hf_local_generate_batch(
                prompts, hf_model, token, max_new_tokens=max_new_tokens
            )
            if len(batch_ok) != len(chunk):
                raise ValueError(
                    f"batch returned {len(batch_ok)} strings, expected {len(chunk)}"
                )
        except Exception as batch_exc:
            log(
                "3b",
                f"Batch failed ({format_exception(batch_exc)}); falling back to one-by-one.",
            )
            batch_ok = None

        if batch_ok is not None:
            for entity, raw in zip(chunk, batch_ok):
                if _apply_answer(entity, raw):
                    resolved += 1
        else:
            for entity, p in zip(chunk, prompts):
                try:
                    raw = hf_local_generate(p, hf_model, token, max_new_tokens=max_new_tokens)
                except Exception as exc:
                    log("3b", f"LLM error for {entity.get('text')!r}: {format_exception(exc)}")
                    entity["linking_status"] = "llm_error"
                    continue
                if _apply_answer(entity, raw):
                    resolved += 1

        done = min(start + len(chunk), total_r)
        log("3b", f"Progress: {done}/{total_r}")

    _finalize_and_write(entities, shaped_dir, grounded_path.parent, resolved=resolved)
    return {
        "model": hf_model,
        "to_resolve": total_r,
        "resolved_new_cui": resolved,
        "output_dir": str(shaped_dir),
    }


def _apply_answer(entity: dict, raw: str) -> bool:
    answer = parse_cui_answer(raw)
    cands = entity.get("cui_candidates") or []
    if answer and answer != "NO_MATCH" and answer.startswith("C"):
        entity["cui_final"] = answer
        entity["cui_label"] = _label_from_candidates(answer, cands)
        entity["linking_status"] = "disambiguated"
        return True
    entity["cui_final"] = None
    entity["linking_status"] = "no_match_after_llm"
    return False


def _finalize_and_write(
    entities: list[dict],
    shaped_dir: Path,
    report_dir: Path,
    *,
    resolved: int,
) -> None:
    n_csv = write_s_annotation_concept(shaped_dir, entities)
    for e in entities:
        e.pop("cui_candidates", None)
    out_res = shaped_dir / "resolved_entities.json"
    save_json(entities, str(out_res))
    linked = sum(1 for e in entities if e.get("cui_final"))
    rep = {
        "resolved_entities": str(out_res),
        "llama_disambiguated_count": resolved,
        "total_entities": len(entities),
        "with_cui_final": linked,
        "s_annotation_concept_rows": n_csv,
    }
    try:
        link_counts = build_links_for_run(shaped_dir.parent.resolve())
        if link_counts:
            rep["annotation_link_csvs"] = link_counts
    except Exception as exc:
        log("3b", f"annotation link CSVs skipped: {exc}")
    save_json(rep, str(report_dir / "disambiguation_report.json"))
    log(
        "3b",
        f"Wrote resolved_entities.json + S_annotation_concept.csv ({n_csv} CUIs); "
        f"CUI-bearing rows={linked}/{len(entities)}",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Llama UMLS disambiguation (Step 3b).")
    ap.add_argument(
        "--grounded",
        type=Path,
        default=Path("outputs/pipeline-output18/step2/grounded_entities.json"),
    )
    ap.add_argument(
        "--shaped-dir",
        type=Path,
        default=None,
        help="Defaults to parent of grounded file",
    )
    ap.add_argument("--hf-model", default=os.environ.get("HF_MODEL", DEFAULT_MODEL))
    ap.add_argument("--hf-token", default=None, help="Defaults to HF_TOKEN env")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    args = ap.parse_args()

    shaped = args.shaped_dir or args.grounded.parent
    disambiguate_llama(
        args.grounded,
        shaped,
        hf_token=args.hf_token,
        hf_model=args.hf_model,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
