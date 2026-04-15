"""
Build a single ZIP for Google Colab: notebooks + scripts + pipeline package + input stub.

Output (default): DigiStructMed_thesis_colab.zip  (in repo root)

Upload that ZIP to Colab, unzip, add your PDF + ontology (+ UMLS for step 3) under input/,
then open the notebook for the step you need (Step 1 extract, Step 3 linking, Step 4 relations, etc.).

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
    "COLAB_Step1_Extract.ipynb",
    "COLAB_Step3_EL_Llama.ipynb",
    "COLAB_Step4_Relations.ipynb",
    "COLAB_From_Step3a.ipynb",
    "requirements.txt",
    "PIPELINE_CONSENSUS.md",
    "pipeline_overview.html",
]

# Directories to pack recursively (relative to repo root)
DIRS = [
    "scripts",
    "pipeline",  # step1–step4 (extract, normalize, UMLS + Llama, relation extraction)
    "outputs/pipeline-output18",  # bundled sample run (step1/ step2/ …) for Colab
]

INSTRUCTIONS = """DigiStructMed — Colab upload bundle
================================

What is in this ZIP
  - COLAB_Pipeline.ipynb         → full legacy pipeline (scripts/)
  - COLAB_Step1_Extract.ipynb    → Step 1 PDF extraction (GPU + Docling v2); outputs/.../step1/
  - COLAB_Step3_EL_Llama.ipynb   → Step 3 UMLS + optional Llama 3.1 disambiguation
  - COLAB_Step4_Relations.ipynb  → Step 4 LLM relation extraction → outputs/.../step4/
  - COLAB_From_Step3a.ipynb      → optional workflow starting after Step 3a
  - scripts/                     → helpers (hf_llm.py, etc.) used by notebooks
  - pipeline/                    → pipeline.step1 … pipeline.step4 (match local Thesis)
  - outputs/pipeline-output18/   → sample run folder (step1, step2, etc.) if present at zip time
  - requirements.txt             → reference; notebooks install deps in cells
  - input/README_INPUT.txt       → what to place in input/

Before running
  1. Unzip on Colab (or upload ZIP and unzip to /content/<YourThesisFolder>/).
  2. Put files in input/ (see input/README_INPUT.txt):
       - Guideline PDF (e.g. Heidenreich, 2022, AHA,ACC,HFSA guidelines.pdf)
       - Ontology TTL (e.g. hf_guideline_ontology.ttl)
       - UMLS.csv for Step 3 entity linking (optional if you skip Step 3)
  3. HF_TOKEN for Docling Hub models (Step 1), Llama 3.1 (Step 3 / Step 4), etc.

Colab path tip: same layout as locally —
  outputs/pipeline-output18/step1/ after Step 1
  outputs/pipeline-output18/step2/ after Step 2 (entity CSVs only)
  outputs/pipeline-output18/step4/ after Step 4 (relation CSVs)

Step 4 can run without Step 3; it only needs step1/text_blocks.json and step2/*.csv.

The notebooks set cwd to the unzip folder and add the repo root to sys.path.

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

        # Empty input/ tree so Colab "0b upload" cells match local layout
        zf.writestr(
            "input/README_INPUT.txt",
            "Place here (upload via notebook or Colab file browser):\n"
            "  - Guideline PDF (default local name: Heidenreich, 2022, AHA,ACC,HFSA guidelines.pdf)\n"
            "  - Ontology TTL (e.g. hf_guideline_ontology.ttl or input/hf_guideline_ontology.ttl)\n"
            "  - UMLS subset CSV as input/UMLS.csv (for Step 3 entity linking)\n",
        )
        zf.writestr("outputs/.gitkeep", "")
        # Placeholder only if the run folder is missing or empty (no files packed above)
        _run = ROOT / "outputs" / "pipeline-output18"
        if not _run.is_dir() or not any(_run.iterdir()):
            zf.writestr("outputs/pipeline-output18/.gitkeep", "")

    kb = out_path.stat().st_size // 1024
    print(f"Wrote {out_path} ({kb} KB)")
    print("  Upload to Colab, unzip, fill input/, then run the notebook for your step")
    print("  (e.g. COLAB_Step1_Extract.ipynb, COLAB_Step4_Relations.ipynb, COLAB_Pipeline.ipynb).")


if __name__ == "__main__":
    main()
