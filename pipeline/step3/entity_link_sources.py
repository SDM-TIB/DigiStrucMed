r"""
Link strings from ontology-shaped CSVs to UMLS (CUI + label).

Reads Step 2 CSVs from ``shaped_dir`` and writes all Step 3 artifacts to
``out_dir`` (which defaults to ``shaped_dir`` for backward compatibility):

  - ``grounded_entities.json``  — full linker state for Llama disambiguation
  - ``S_annotation_concept.csv`` — unique CUI/label pairs with ``cui_final``
  - ``entity_linking_report.json``
  - ``entity_linking_details.jsonl`` (optional)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

_ROOT = Path(__file__).resolve().parents[2]
_STEP2 = Path(__file__).resolve().parent.parent / "step2"

from pipeline.step3._entity_linking import (  # noqa: E402
    load_bk_and_inverted_index,
    _link_one_mention,
)
from pipeline.step1.utils import log, save_json  # noqa: E402

class _el:
    load_bk_and_inverted_index = staticmethod(load_bk_and_inverted_index)
    _link_one_mention = staticmethod(_link_one_mention)

# (csv_file, column_to_link, min_text_length, semantic_entity_type)
_DEFAULT_SOURCES: tuple[tuple[str, str, int, str], ...] = (
    ("S_drug.csv", "agentName", 2, "Drug"),
    ("S_therapy.csv", "therapy_name", 2, "Therapy"),
    ("S_cause.csv", "causeName", 2, "Disease Cause"),
    ("S_phenotype.csv", "phenotypeCode", 2, "Disease Phenotype"),
    ("S_adverse_event.csv", "adverseEventName", 2, "Adverse Event"),
    ("S_disease.csv", "diseaseName", 2, "Disease"),
)


def _config_disease_abbreviations(config_path: Path | None) -> dict[str, str]:
    if config_path is None:
        config_path = _STEP2 / "guideline_config.json"
    else:
        config_path = Path(config_path)
    if not config_path.is_file():
        return {}
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    d = cfg.get("disease") or {}
    disease_name = (d.get("disease_name") or "").strip()
    out: dict[str, str] = {}
    for a in d.get("abbreviations") or []:
        a = str(a).strip()
        if a and disease_name:
            out[a] = disease_name
    return out


def _merge_acronym_table(config_path: Path | None) -> dict[str, str]:
    from pipeline.step1.extract_text import _HF_ACRONYMS

    merged = dict(_HF_ACRONYMS)
    for k, v in _config_disease_abbreviations(config_path).items():
        merged.setdefault(k, v)
    return merged


def _expand_acronyms(text: str, table: dict[str, str]) -> str:
    if not text or not table:
        return text
    for acronym in sorted(table.keys(), key=len, reverse=True):
        expansion = table[acronym]
        pattern = r"\b" + re.escape(acronym) + r"\b"
        text = re.sub(pattern, expansion, text)
    return text.strip()


def _row_context(row: dict, linked_col: str) -> str:
    """Build a short descriptive string from all other columns in the row."""
    parts = []
    for k, v in row.items():
        if k == linked_col or not v or k.endswith("_id"):
            continue
        parts.append(f"{k}: {v.strip()}")
    return "; ".join(parts)[:400]


def _iter_mentions(
    shaped_dir: Path,
    *,
    acronym_table: dict[str, str] | None,
) -> Iterator[dict]:
    seen: set[str] = set()
    for fname, col, min_len, entity_type in _DEFAULT_SOURCES:
        path = shaped_dir / fname
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if col not in row:
                    continue
                raw = (row.get(col) or "").strip()
                if len(raw) < min_len:
                    continue
                linked_text = _expand_acronyms(raw, acronym_table) if acronym_table else raw
                key = linked_text.lower()
                if key in seen:
                    continue
                seen.add(key)
                rid = next(
                    (row[k] for k in row if k.endswith("_id") and row.get(k)),
                    f"{fname}:{col}",
                )
                ctx = _row_context(row, col)
                m: dict = {
                    "text": linked_text,
                    "source_file": fname,
                    "source_row_id": str(rid),
                    "source_column": col,
                    "type": entity_type,
                    "source_text": ctx if ctx else f"from {fname}",
                }
                if acronym_table and linked_text != raw:
                    m["text_raw"] = raw
                yield m


def _label_for_cui(cui: str, candidates: list | None) -> str:
    if not candidates:
        return str(cui)
    for c in candidates:
        if str(c.get("cui", "")) == str(cui):
            return str(c.get("label") or cui)
    return str(cui)


def write_s_annotation_concept(shaped_dir: Path, entities: list[dict]) -> int:
    """Write ``S_annotation_concept.csv`` from any records with ``cui_final``."""
    by_cui: dict[str, str] = {}
    for e in entities:
        cui = e.get("cui_final")
        if not cui:
            continue
        if e.get("cui_label"):
            lab = str(e["cui_label"])
        else:
            cands = e.get("cui_candidates")
            lab = _label_for_cui(
                str(cui), cands if isinstance(cands, list) else None
            )
        by_cui[str(cui)] = lab
    out_csv = shaped_dir / "S_annotation_concept.csv"
    rows = [{"concept_id": c, "conceptName": n} for c, n in sorted(by_cui.items())]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["concept_id", "conceptName"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def _enrich_for_llm(m: dict) -> None:
    """Ensure fields expected by ``disambiguate_llama`` are set."""
    m.setdefault("type", m.get("source_column", ""))
    m.setdefault("source_text", f"from {m.get('source_file', '')}")


def run_entity_linking(
    shaped_dir: str | Path,
    umls_csv: str | Path,
    *,
    out_dir: str | Path | None = None,
    sim_threshold: float = 0.92,
    top_k: int = 15,
    max_score: int = 12_000,
    cache_dir: str | Path | None = "outputs/cache",
    use_cache: bool = True,
    write_jsonl: bool = False,
    expand_acronyms: bool = True,
    config_path: str | Path | None = None,
    save_grounded: bool = True,
) -> dict:
    shaped = Path(shaped_dir)
    target = Path(out_dir) if out_dir else shaped
    target.mkdir(parents=True, exist_ok=True)
    umls_path = Path(umls_csv)
    if not shaped.is_dir():
        raise FileNotFoundError(shaped)
    if not umls_path.is_file():
        raise FileNotFoundError(umls_path)

    cfg_p = Path(config_path) if config_path else None
    acronym_table = _merge_acronym_table(cfg_p) if expand_acronyms else None
    mentions = list(_iter_mentions(shaped, acronym_table=acronym_table))
    for m in mentions:
        _enrich_for_llm(m)
    log(
        "EL",
        f"{len(mentions)} unique mention strings from CSV columns"
        + (f" (acronym expansion: {len(acronym_table or {})} keys)" if acronym_table else ""),
    )

    bk, inverted, freq = _el.load_bk_and_inverted_index(
        str(umls_path),
        cache_dir=str(cache_dir) if cache_dir else None,
        use_cache=use_cache,
    )

    results: list[dict] = []
    for i, m in enumerate(mentions):
        out = _el._link_one_mention(
            m, bk, inverted, freq, top_k, sim_threshold, max_score
        )
        _enrich_for_llm(out)
        if out.get("linking_status") == "direct" and out.get("cui_final"):
            out["cui_label"] = _label_for_cui(
                str(out["cui_final"]), out.get("cui_candidates") or []
            )
        results.append(out)
        if (i + 1) % 500 == 0:
            log("EL", f"linked {i + 1}/{len(mentions)}")

    status_counts: dict[str, int] = defaultdict(int)
    for r in results:
        status_counts[str(r.get("linking_status") or "unknown")] += 1

    n = len(results)
    direct = status_counts.get("direct", 0)
    disamb = status_counts.get("needs_disambiguation", 0)
    none_ = status_counts.get("no_match", 0)

    n_concepts = write_s_annotation_concept(target, results)

    if save_grounded:
        save_json(results, str(target / "grounded_entities.json"))

    report = {
        "step2_dir": str(shaped.resolve()),
        "out_dir": str(target.resolve()),
        "umls_csv": str(umls_path.resolve()),
        "unique_mentions": n,
        "status_counts": dict(status_counts),
        "rate_direct": round(direct / n, 4) if n else 0.0,
        "rate_needs_disambiguation": round(disamb / n, 4) if n else 0.0,
        "rate_no_match": round(none_ / n, 4) if n else 0.0,
        "distinct_cuis_written": n_concepts,
        "sim_threshold": sim_threshold,
        "acronym_expansion": bool(acronym_table),
        "acronym_table_size": len(acronym_table) if acronym_table else 0,
        "grounded_entities_path": str(target / "grounded_entities.json") if save_grounded else None,
    }
    save_json(report, str(target / "entity_linking_report.json"))

    if write_jsonl:
        jl = target / "entity_linking_details.jsonl"
        with jl.open("w", encoding="utf-8") as jf:
            for r in results:
                jf.write(json.dumps(r, ensure_ascii=False) + "\n")

    log(
        "EL",
        f"Wrote {n_concepts} concepts -> S_annotation_concept.csv; "
        f"direct={direct}/{n} ({report['rate_direct']:.1%}), "
        f"no_match={none_}/{n}; grounded_entities.json={'yes' if save_grounded else 'no'}",
    )
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--shaped-dir", type=Path,
        default=Path("outputs/pipeline-output18/step2"),
        help="Step 2 output directory (source CSVs)",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Step 3 output directory (default: same as shaped-dir for backward compat)",
    )
    ap.add_argument("--umls", type=Path, default=Path("input/UMLS.csv"))
    ap.add_argument("--sim-threshold", type=float, default=0.92)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--jsonl", action="store_true")
    ap.add_argument("--no-expand-acronyms", action="store_true")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--no-grounded-json", action="store_true", help="Skip grounded_entities.json")
    args = ap.parse_args()

    run_entity_linking(
        args.shaped_dir,
        args.umls,
        out_dir=args.out,
        sim_threshold=args.sim_threshold,
        use_cache=not args.no_cache,
        write_jsonl=args.jsonl,
        expand_acronyms=not args.no_expand_acronyms,
        config_path=args.config,
        save_grounded=not args.no_grounded_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
