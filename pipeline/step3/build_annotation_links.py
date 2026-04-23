from __future__ import annotations

import csv
import json
import re
from pathlib import Path


_SOURCE_TO_SPEC = {
    "S_condition.csv": ("conditionName", "condition_id", "S_condition_annotation.csv"),
    "S_drug.csv": ("agentName", "drug_id", "S_drug_annotation.csv"),
    "S_therapy.csv": ("therapy_name", "therapy_id", "S_therapy_annotation.csv"),
    "S_cause.csv": ("causeName", "cause_id", "S_cause_annotation.csv"),
    "S_phenotype.csv": ("phenotypeCode", "phenotype_id", "S_phenotype_annotation.csv"),
    "S_adverse_event.csv": ("adverseEventName", "ae_id", "S_adverse_event_annotation.csv"),
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_links_for_run(run_dir: Path) -> dict[str, int]:
    step2 = run_dir / "step2"
    step3 = run_dir / "step3"
    grounded = step3 / "resolved_entities.json"
    if not grounded.is_file():
        grounded = step3 / "grounded_entities.json"
    if not grounded.is_file():
        return {}

    try:
        entities = json.loads(grounded.read_text(encoding="utf-8"))
    except Exception:
        return {}

    guideline_rows = _read_csv_rows(step2 / "S_guideline.csv")
    guideline_id = (
        (guideline_rows[0].get("guideline_id") or "").strip()
        if guideline_rows
        else "guideline_unknown"
    )

    name_index: dict[str, dict[str, set[str]]] = {}
    for source_csv, (name_col, id_col, _) in _SOURCE_TO_SPEC.items():
        ids_by_name: dict[str, set[str]] = {}
        for row in _read_csv_rows(step2 / source_csv):
            name = _norm(row.get(name_col, ""))
            eid = (row.get(id_col) or "").strip()
            if not name or not eid:
                continue
            ids_by_name.setdefault(name, set()).add(eid)
        name_index[source_csv] = ids_by_name

    out_rows: dict[str, set[tuple[str, str]]] = {
        spec[2]: set() for spec in _SOURCE_TO_SPEC.values()
    }

    for e in entities:
        source_file = (e.get("source_file") or "").strip()
        concept_id = (e.get("cui_final") or "").strip()
        if source_file not in _SOURCE_TO_SPEC or not concept_id:
            continue

        name_col, id_col, out_csv = _SOURCE_TO_SPEC[source_file]
        ids_by_name = name_index.get(source_file, {})

        keys = [_norm(e.get("text", ""))]
        cui_label = _norm(e.get("cui_label", ""))
        if cui_label and cui_label not in keys:
            keys.append(cui_label)

        for k in keys:
            for entity_id in ids_by_name.get(k, set()):
                out_rows[out_csv].add((entity_id, concept_id))

    step3.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    for source_csv, (_, id_col, out_csv) in _SOURCE_TO_SPEC.items():
        rows = sorted(out_rows.get(out_csv, set()))
        out_path = step3 / out_csv
        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["guideline_id", id_col, "concept_id"])
            w.writeheader()
            for entity_id, concept_id in rows:
                w.writerow(
                    {
                        "guideline_id": guideline_id,
                        id_col: entity_id,
                        "concept_id": concept_id,
                    }
                )
        written[out_csv] = len(rows)
    return written


def build_links_batch(root: Path) -> dict[str, dict[str, int]]:
    report: dict[str, dict[str, int]] = {}
    if not root.exists():
        return report
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        stats = build_links_for_run(run_dir)
        if stats:
            report[run_dir.name] = stats
    return report


def main() -> int:
    root = Path("outputs/pipeline-output2_new")
    report = build_links_batch(root)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
