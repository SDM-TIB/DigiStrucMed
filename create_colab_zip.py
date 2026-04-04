"""
Build a single ZIP for Google Colab: notebook + scripts + requirements.

Output (default): DigiStructMed_thesis_colab.zip  (in repo root)

Upload that ZIP to Colab, unzip, add input/guideline.pdf and
input/hf_guideline_ontology.ttl, then open COLAB_Pipeline.ipynb.

Usage (from repo root):
    python create_colab_zip.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_NAME = "DigiStructMed_thesis_colab.zip"

# Files at repo root to include
ROOT_FILES = [
    "COLAB_Pipeline.ipynb",
    "requirements.txt",
    "PIPELINE_CONSENSUS.md",
    "pipeline_overview.html",
]

# Directories to pack recursively (only *.py under scripts/)
DIRS = ["scripts"]

INSTRUCTIONS = """DigiStructMed — Colab upload bundle
================================

What is in this ZIP
  - COLAB_Pipeline.ipynb   → run this in Colab (Runtime → Run all)
  - scripts/               → pipeline steps (imported by the notebook)
  - requirements.txt       → reference; the notebook installs deps in cell 1

Before running
  1. Unzip this archive on the Colab machine (or upload and unzip).
  2. Run section "0b. Upload inputs" — it prompts you to upload the PDF, the
     ontology (.ttl), and optionally a UMLS CSV; it creates input/ for you.
     Or skip uploads and copy files into input/ manually.
  3. Optional: HF_TOKEN for LLM steps (see config cell).

The notebook adds scripts/ to sys.path automatically when cwd is the unzip folder.

"""


def _should_skip(path: Path) -> bool:
    parts = path.parts
    if "__pycache__" in parts:
        return True
    if path.suffix in {".pyc", ".pyo"}:
        return True
    return False


def main() -> None:
    out_path = ROOT / OUT_NAME
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("COLAB_UPLOAD_INSTRUCTIONS.txt", INSTRUCTIONS)

        for name in ROOT_FILES:
            p = ROOT / name
            if p.is_file():
                zf.write(p, arcname=name)

        for dname in DIRS:
            d = ROOT / dname
            if not d.is_dir():
                continue
            for f in d.rglob("*"):
                if not f.is_file() or _should_skip(f):
                    continue
                arc = f.relative_to(ROOT).as_posix()
                zf.write(f, arcname=arc)

    kb = out_path.stat().st_size // 1024
    print(f"Wrote {out_path} ({kb} KB)")
    print("  Upload this file to Colab, unzip, add input PDF + ontology, run the notebook.")


if __name__ == "__main__":
    main()
