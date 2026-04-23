"""Merge duplicate Step 2 entity rows (same guideline, same semantic key).

Skips S_disease / S_guideline. Recommendations group by normalized text when
``group_by_text`` is set. For drugs, when DrugBank data is loaded, rows whose
``agentName`` has no DrugBank match are dropped. See ``_DEDUP_TARGETS``.
"""
from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path


_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")
_NORM_RE = re.compile(r"[^a-z0-9]+")

_ABBREV: dict[str, str] = {
    "cabg": "coronary artery bypass grafting",
    "pci": "percutaneous coronary intervention",
    "crt": "cardiac resynchronization therapy",
    "icd": "implantable cardioverter defibrillator",
    "arb": "angiotensin receptor blocker",
    "arni": "angiotensin receptor neprilysin inhibitor",
    "acei": "angiotensin converting enzyme inhibitor",
    "mra": "mineralocorticoid receptor antagonist",
    "sglt2i": "sglt2 inhibitor",
    "lvad": "left ventricular assist device",
    "lbbb": "left bundle branch block",
    "rbbb": "right bundle branch block",
    "hfref": "heart failure with reduced ejection fraction",
    "hfpef": "heart failure with preserved ejection fraction",
    "hfmref": "heart failure with mildly reduced ejection fraction",
    "hfimpef": "heart failure with improved ejection fraction",
    "dcm": "dilated cardiomyopathy",
    "hcm": "hypertrophic cardiomyopathy",
    "acm": "arrhythmogenic cardiomyopathy",
    "rcm": "restrictive cardiomyopathy",
    "lvh": "left ventricular hypertrophy",
}

_SUFFIX_ABBREV: dict[str, str] = {
    "_ef": "_ejection_fraction",
}

_DRUGBANK_NAME_TO_ID: dict[str, str] | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _norm_drug_name(s: str) -> str:
    return _NORM_RE.sub("", (s or "").lower().strip())


def _localname(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _load_drugbank_name_to_id() -> dict[str, str]:
    global _DRUGBANK_NAME_TO_ID
    if _DRUGBANK_NAME_TO_ID is not None:
        return _DRUGBANK_NAME_TO_ID

    csv_path = _repo_root() / "input" / "DrugBank.csv"
    if csv_path.is_file():
        mapping: dict[str, str] = {}
        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            for raw in csv.DictReader(f):
                row = {
                    str(k).replace("\ufeff", "").strip().lower(): v
                    for k, v in (raw or {}).items()
                    if k is not None
                }
                name = (row.get("name") or "").strip()
                did = (row.get("drug_id") or "").strip()
                if not name or not did:
                    continue
                key = _norm_drug_name(name)
                if key and key not in mapping:
                    mapping[key] = did
        _DRUGBANK_NAME_TO_ID = mapping
        return _DRUGBANK_NAME_TO_ID

    xml_path = _repo_root() / "input" / "full database.xml"
    if not xml_path.is_file():
        _DRUGBANK_NAME_TO_ID = {}
        return _DRUGBANK_NAME_TO_ID

    mapping = {}
    for _event, elem in ET.iterparse(str(xml_path), events=("end",)):
        if _localname(elem.tag) != "drug":
            continue

        primary_id = ""
        name = ""
        for child in list(elem):
            lname = _localname(child.tag)
            if lname == "drugbank-id" and child.attrib.get("primary") == "true":
                primary_id = (child.text or "").strip()
            elif lname == "name":
                name = (child.text or "").strip()
            if primary_id and name:
                break

        if primary_id and name:
            key = _norm_drug_name(name)
            if key and key not in mapping:
                mapping[key] = primary_id

        elem.clear()

    _DRUGBANK_NAME_TO_ID = mapping
    return _DRUGBANK_NAME_TO_ID


def _drugbank_id_for_name(name: str) -> str:
    key = _norm_drug_name(name)
    if not key:
        return ""
    return _load_drugbank_name_to_id().get(key, "")


def _slug(s: str) -> str:
    return _SLUG_RE.sub("_", s.strip()).strip("_")


def _norm_key(s: str) -> str:
    return _NORM_RE.sub("_", s.lower().strip()).strip("_")


def _expand_abbrev(normed: str) -> str:
    plain = normed.replace("_", "")
    expanded = _ABBREV.get(plain)
    if expanded:
        return _NORM_RE.sub("_", expanded).strip("_")
    result = normed
    for suffix, replacement in _SUFFIX_ABBREV.items():
        if result.endswith(suffix):
            result = result[: -len(suffix)] + replacement
            break
    return result


def _canonical_id(entity_kind: str, row: dict, id_col: str, name_col: str) -> str:
    name = (row.get(name_col) or "").strip()
    if entity_kind == "drug":
        dbid = _drugbank_id_for_name(name)
        if dbid:
            return dbid

    eid = (row.get(id_col) or "").strip()
    if eid:
        cleaned = eid.replace(f"{entity_kind}_txt_", f"{entity_kind}_")
        if entity_kind == "drug" and cleaned.upper().startswith("DB") and cleaned[2:].isdigit():
            return cleaned.upper()
        return cleaned
    return f"{entity_kind}_{_slug(name)}" if name else ""


def _grouping_key(entity_kind: str, canonical: str) -> str:
    suffix = canonical
    prefix = f"{entity_kind}_"
    if suffix.lower().startswith(prefix.lower()):
        suffix = suffix[len(prefix):]
    normed = _norm_key(suffix)
    expanded = _expand_abbrev(normed)
    return f"{entity_kind}_{expanded}"


def _stage_group_key(row: dict, fallback: str) -> str:
    scheme = _norm_key((row.get("stageScheme") or "").replace("/", "_"))
    level = _norm_key(row.get("stageLevel") or "")
    if level:
        if scheme == "nyha":
            return f"stage_{scheme}_{level}"
        return f"stage_{level}"
    name = _norm_key(row.get("stageName") or "")
    return f"stage_{name}" if name else fallback


def _pick_best_name(values: list[str]) -> str:
    values = [v for v in values if v]
    if not values:
        return ""
    return max(values, key=len)


def _merge_data(values: list[str]) -> str:
    seen: list[str] = []
    for v in values:
        v = (v or "").strip()
        if not v or v == "text_mention":
            continue
        if v not in seen:
            seen.append(v)
    return " | ".join(seen)


def _merge_stage_scheme(values: list[str]) -> str:
    parts: list[str] = []
    for v in values:
        for p in (v or "").split("|"):
            p = p.strip()
            if p and p not in parts:
                parts.append(p)
    if len(parts) > 1:
        parts = [p for p in parts if p.lower() != "unknown"] or parts
    return " | ".join(parts)


def _pick_best_id(ids: list[str]) -> str:
    ids = [i for i in ids if i]
    if not ids:
        return ""
    return min(ids, key=len)


def _merge_bucket(bucket: list[dict], id_col: str, name_col: str,
                  data_cols: tuple[str, ...], canonical_ids: list[str]) -> dict:
    merged = dict(bucket[0])
    merged[id_col] = _pick_best_id(canonical_ids)
    if name_col:
        merged[name_col] = _pick_best_name([r.get(name_col, "") for r in bucket])
    for col in data_cols:
        if col == "stageScheme":
            merged[col] = _merge_stage_scheme([r.get(col, "") for r in bucket])
        else:
            merged[col] = _merge_data([r.get(col, "") for r in bucket])
    return merged


def _dedup_file(
    path: Path,
    entity_kind: str,
    id_col: str,
    name_col: str,
    data_cols: tuple[str, ...],
    *,
    group_by_text: str = "",
) -> tuple[int, int, int]:
    if not path.is_file():
        return 0, 0, 0
    with path.open(encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        header = list(rdr.fieldnames or [])
        rows = list(rdr)
    if not rows:
        return 0, 0, 0
    before = len(rows)

    if entity_kind == "drug":
        db_map = _load_drugbank_name_to_id()
        if db_map:
            rows = [
                r
                for r in rows
                if _drugbank_id_for_name((r.get(name_col) or "").strip())
            ]
        if not rows:
            with path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
                w.writeheader()
            return before, 0, 0

    buckets: dict[tuple[str, str], list[dict]] = {}
    orig_ids: dict[tuple[str, str], list[str]] = {}
    order: list[tuple[str, str]] = []
    for r in rows:
        gid = (r.get("guideline_id") or "").strip()
        cid = _canonical_id(entity_kind, r, id_col, name_col)
        if not cid:
            continue
        if group_by_text:
            raw = (r.get(group_by_text) or "").strip()
            gkey = f"{entity_kind}_{_norm_key(raw)}" if raw else _grouping_key(entity_kind, cid)
        elif entity_kind == "stage":
            gkey = _stage_group_key(r, _grouping_key(entity_kind, cid))
        else:
            gkey = _grouping_key(entity_kind, cid)
        key = (gid, gkey)
        if key not in buckets:
            buckets[key] = []
            orig_ids[key] = []
            order.append(key)
        buckets[key].append(r)
        orig_ids[key].append(cid)

    deduped: list[dict] = []
    merged_count = 0
    for key in order:
        bucket = buckets[key]
        if len(bucket) > 1:
            merged_count += 1
        deduped.append(_merge_bucket(bucket, id_col, name_col, data_cols, orig_ids[key]))

    if entity_kind == "drug":
        db_map = _load_drugbank_name_to_id()
        if db_map:
            kept: list[dict] = []
            for r in deduped:
                n = (r.get(name_col) or "").strip()
                dbid = _drugbank_id_for_name(n)
                if dbid:
                    r[id_col] = dbid
                    kept.append(r)
            deduped = kept

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in deduped:
            w.writerow({c: (r.get(c, "") or "") for c in header})

    return before, len(deduped), merged_count


# (filename, entity_kind, id_col, name_col, data_cols, group_by_text)
_DEDUP_TARGETS: tuple[tuple[str, str, str, str, tuple[str, ...], str], ...] = (
    ("S_drug.csv",            "drug",       "drug_id",       "agentName",
        ("minDose", "maxDose", "duration", "drugCategory"), ""),
    ("S_therapy.csv",         "therapy",    "therapy_id",    "therapy_name",
        ("therapyType",), ""),
    ("S_cause.csv",           "cause",      "cause_id",      "causeName",
        (), ""),
    ("S_phenotype.csv",       "phenotype",  "phenotype_id",  "phenotypeCode",
        ("phenotypeCriteria",), ""),
    ("S_stage.csv",           "stage",      "stage_id",      "stageName",
        ("stageScheme", "stageLevel", "StageCriteriaText"), ""),
    ("S_recommendation.csv",  "rec",        "rec_id",        "recommendationText",
        (), "recommendationText"),
    ("S_adverse_event.csv",   "ae",         "ae_id",         "adverseEventName",
        ("adverseEventSeverity",), ""),
    ("S_assessment.csv",      "assessment", "assessment_id", "assessmentName",
        ("assessmentValue",), ""),
)


def dedup_step2_dir(step2_dir: Path, verbose: bool = True) -> dict[str, tuple[int, int, int]]:
    report: dict[str, tuple[int, int, int]] = {}
    for fname, kind, id_col, name_col, data_cols, gbt in _DEDUP_TARGETS:
        report[fname] = _dedup_file(
            step2_dir / fname, kind, id_col, name_col, data_cols,
            group_by_text=gbt,
        )
    if verbose:
        for name, (b, a, m) in report.items():
            if b:
                suffix = f"  ({m} merged)" if m else ""
                print(f"  [dedup] {name}: {b} -> {a} rows{suffix}")
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Dedup Step 2 CSVs in one or many step2 folders.")
    ap.add_argument("path", type=Path, help="step2 folder or parent of <slug>/step2/")
    args = ap.parse_args()

    root = args.path.resolve()
    if (root / "S_drug.csv").is_file():
        targets = [root]
    else:
        targets = sorted(p / "step2" for p in root.iterdir()
                         if p.is_dir() and (p / "step2").is_dir())

    if not targets:
        raise SystemExit(f"No step2 folders found under {root}")

    for t in targets:
        print(f"\n[dedup] {t}")
        dedup_step2_dir(t)
