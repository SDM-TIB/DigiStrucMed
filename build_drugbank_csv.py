"""Build input/DrugBank.csv from DrugBank full-database XML (one-time / refresh).

Reads ``input/full database.xml`` by default, streams <drug> elements, and writes
``input/DrugBank.csv`` with columns ``drug_id,name`` (primary DrugBank ID + name).
``pipeline/step2/dedup_entities.py`` loads this CSV first to avoid parsing the
large XML on every run (e.g. Colab).

Usage (repo root):
    python build_drugbank_csv.py
    python build_drugbank_csv.py --xml "input/full database.xml" --out input/DrugBank.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _localname(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    ap.add_argument("--xml", type=Path, default=root / "input" / "full database.xml")
    ap.add_argument("--out", type=Path, default=root / "input" / "DrugBank.csv")
    args = ap.parse_args()

    xml_path = args.xml.resolve()
    out_path = args.out.resolve()
    if not xml_path.is_file():
        print(f"ERROR: missing {xml_path}", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["drug_id", "name"])
        w.writeheader()
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
                w.writerow({"drug_id": primary_id, "name": name})
                n += 1
            elem.clear()

    print(f"Wrote {n} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
