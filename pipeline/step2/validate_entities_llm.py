"""LLM keep/reject validation over Step 2 entity CSVs (after normalize + dedup)."""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


_VALIDATION_PROMPTS = {
    "S_adverse_event.csv": {
        "name_col": "adverseEventName",
        "prompt": (
            "You are a clinical entity validator. Given 'adverseEventName' values, "
            "decide if each is a plausible adverse event, side effect, or complication.\n\n"
            "Default to KEEP, but REJECT if it clearly is NOT an adverse event:\n"
            "- A broken sentence fragment (e.g. 'failure with reduced', 'dysfunction may be')\n"
            "- A disease name that is NOT caused by treatment (e.g. 'cardiac sarcoidosis', "
            "'hypertrophic cardiomyopathy', 'dilated cardiomyopathy' — these are diseases, "
            "not adverse events)\n"
            "- A section header or trial name\n"
            "- A word ending with 'associated' (e.g. 'cardiomyopathy associated')\n\n"
            "KEEP real adverse events caused by drugs or devices:\n"
            "- Drug side effects: hypotension, bleeding, hyperkalemia, bradycardia, "
            "angioedema, QT prolongation, cough, renal dysfunction\n"
            "- Harmful outcomes: stroke, ischaemic stroke, sudden cardiac death\n"
            "- Device complications: lead failure, infection, pneumothorax\n"
            "- Toxicity: embryofoetal toxicity, maternal toxicity\n\n"
            "When in doubt, KEEP.\n\n"
        ),
    },
    "S_therapy.csv": {
        "name_col": "therapy_name",
        "prompt": (
            "You are a clinical entity validator. Given 'therapy_name' values, "
            "decide if each is a plausible therapy, treatment, procedure, or intervention.\n\n"
            "Default to KEEP, but REJECT if it clearly is NOT a therapy name:\n"
            "- A sentence fragment describing a patient state (e.g. 'failure despite guideline-directed "
            "medical therapy', 'candidate for cardiac resynchronization therapy', "
            "'already on guideline-directed medical therapy')\n"
            "- A phrase starting with 'effect of', 'role of', 'benefit of', 'limitation of'\n"
            "- A clinical trial name (e.g. 'PARADIGM-HF', 'Antiarrhythmics versus Implantable "
            "Defibrillators', 'Spanish Atrial Fibrillation and Resynchronization')\n"
            "- A section heading like 'Recommendations for' or 'Table 5'\n"
            "- A garbled/OCR token (e.g. 'intensi fi cation')\n\n"
            "KEEP real therapy names:\n"
            "- Drug classes: ACEi, ARB, ARNi, SGLT2i, MRA, beta-blockers, diuretics, "
            "anticoagulation, antiplatelet therapy, GDMT\n"
            "- Devices: CRT, ICD, pacemaker, mechanical circulatory support\n"
            "- Procedures: heart transplant, catheter ablation, cardiac rehabilitation\n\n"
            "When in doubt, REJECT.\n\n"
        ),
    },
    "S_assessment.csv": {
        "name_col": "assessmentName",
        "prompt": (
            "You are a clinical entity validator. Given 'assessmentName' values, "
            "decide if each is a plausible clinical assessment, test, screening, "
            "or diagnostic evaluation.\n\n"
            "Default to KEEP. Only REJECT if it is CLEARLY not an assessment:\n"
            "- A broken sentence fragment with no medical meaning\n"
            "- A clinical trial name (e.g. 'Randomized Aldactone Evaluation Study')\n"
            "- A journal name (e.g. 'Circ Cardiovasc Imaging')\n"
            "- A pure section heading like 'WHEN TO CONSIDER REFERRAL'\n\n"
            "KEEP all of the following — they ARE valid assessments:\n"
            "- Diagnostic tests: echocardiography, ECG, BNP testing, troponin, MRI\n"
            "- Screening: natriuretic peptide-based screening, pre-participation "
            "cardiovascular assessment, pre-pregnancy risk assessment\n"
            "- Evaluations: pre-discharge assessment, risk assessment, "
            "cardiopulmonary exercise testing, 6-minute walk test, "
            "genetic testing, sleep evaluation, haemodynamic assessment\n"
            "- Monitoring: anti-Xa level monitoring, telemonitoring, "
            "ambulatory blood pressure monitoring\n\n"
            "When in doubt, KEEP.\n\n"
        ),
    },
    "S_recommendation.csv": {
        "name_col": "recommendationText",
        "prompt": (
            "You are a clinical entity validator. Given 'recommendationText' values, "
            "decide if each is a plausible clinical recommendation.\n\n"
            "Default to KEEP. Only REJECT if it is CLEARLY not a recommendation:\n"
            "- A section header with no actionable content "
            "(e.g. 'Recommendations for the diagnosis of HF')\n"
            "- A table/figure caption (e.g. 'Table 5 Cardiovascular disease risk')\n"
            "- Boilerplate (e.g. 'Referenced studies that support recommendations')\n"
            "- Fewer than 5 words and not a complete instruction\n\n"
            "KEEP any text that tells clinicians what to do, what to consider, "
            "or what is recommended/indicated, even if the phrasing is imperfect.\n\n"
            "When in doubt, KEEP.\n\n"
        ),
    },
    "S_condition.csv": {
        "name_col": "conditionName",
        "prompt": (
            "You are a clinical entity validator. Given 'conditionName' values, "
            "decide if each names a condition or clinical syndrome.\n\n"
            "Default to KEEP. Only REJECT if it is CLEARLY not a condition:\n"
            "- A therapy or procedure (e.g. 'cardiac pacing', 'resynchronization therapy')\n"
            "- A broken fragment (e.g. 'Get With the', 'patients with')\n"
            "- A guideline scope phrase that is not a disease "
            "(e.g. 'prevention in clinical practice', 'sports cardiology')\n\n"
            "KEEP all conditions including compound descriptions:\n"
            "heart failure, HFpEF, HFrEF, cardiovascular disease, "
            "cardiovascular diseases during pregnancy, ventricular arrhythmias, "
            "coronary artery disease, atrial fibrillation, hypertension, diabetes, "
            "cardiomyopathy, sudden cardiac death, advanced heart failure, etc.\n\n"
            "When in doubt, KEEP.\n\n"
        ),
    },
    "S_drug.csv": {
        "name_col": "agentName",
        "prompt": (
            "You are a clinical entity validator. Given 'agentName' values, "
            "decide if each is a plausible drug name or drug class.\n\n"
            "Default to KEEP. Only REJECT if it is CLEARLY not a drug:\n"
            "- A broken sentence fragment with no pharmacological meaning\n"
            "- A section header or table caption\n"
            "- A pure receptor/protein name with no drug context "
            "(e.g. 'Beta 1 receptor', 'troponin')\n\n"
            "KEEP all of the following — they ARE valid drugs:\n"
            "- Specific drugs: Metoprolol, Empagliflozin, Sacubitril, Amiodarone, "
            "Lisinopril, Carvedilol, Dapagliflozin, Ivabradine, Digoxin\n"
            "- Drug classes: ACE inhibitors, beta-blockers, SGLT2 inhibitors, MRA, "
            "ARB, loop diuretics, anticoagulants, antiarrhythmics, statins\n"
            "- Abbreviations: ACEi, ARB, ARNi, SGLT2i, MRA\n\n"
            "When in doubt, KEEP.\n\n"
        ),
    },
    "S_cause.csv": {
        "name_col": "causeName",
        "prompt": (
            "You are a clinical entity validator. Given 'causeName' values, "
            "decide if each names a plausible medical cause or etiology of a disease.\n\n"
            "Default to KEEP, but REJECT if it clearly is NOT a cause/etiology:\n"
            "- A broken or truncated sentence fragment (e.g. 'giving rise to typical electroca', "
            "'decrease in the incidence of shockable rhy')\n"
            "- An outcome or statistic (e.g. 'increased mortality', 'fewer sudden cardiac death', "
            "'limited expectation for survival')\n"
            "- A clinical trial or registry name (e.g. 'MADIT', 'FINGER registry')\n"
            "- A management/treatment phrase (e.g. 'should be managed with diuretics')\n"
            "- An opinion or methodology phrase (e.g. 'prevailing opinion of experts', "
            "'lack of methodological standardization')\n"
            "- A vague non-medical phrase (e.g. 'several factors', 'some cases')\n\n"
            "KEEP real causes/etiologies: ischaemic heart disease, coronary artery disease, "
            "hypertension, valvular disease, viral myocarditis, genetic mutation, "
            "atrial fibrillation, diabetes mellitus, amyloidosis, Chagas disease, "
            "tachycardia-induced cardiomyopathy, drug-induced cardiomyopathy, etc.\n\n"
            "When in doubt, REJECT.\n\n"
        ),
    },
    "S_stage.csv": {
        "name_col": "stageName",
        "prompt": (
            "You are a clinical entity validator. Given 'stageName' values, "
            "decide if each is a plausible disease stage or classification.\n\n"
            "Default to KEEP. Only REJECT if it is CLEARLY not a stage:\n"
            "- A broken sentence fragment or boilerplate text\n"
            "- A therapy, drug, or recommendation\n\n"
            "KEEP any staging entry: Stage A, Stage B, Stage C, Stage D, "
            "NYHA Class I-IV, ACC/AHA stages, Pre-HF, At Risk for HF, etc.\n\n"
            "When in doubt, KEEP.\n\n"
        ),
    },
    "S_phenotype.csv": {
        "name_col": "phenotypeCriteria",
        "prompt": (
            "You are a clinical entity validator. Given 'phenotypeCriteria' values, "
            "decide if each describes a plausible clinical phenotype or subtype.\n\n"
            "Default to KEEP. Only REJECT if it is CLEARLY not a phenotype:\n"
            "- A broken sentence fragment with no clinical meaning\n"
            "- A table caption, section header, or boilerplate\n"
            "- A single generic word (e.g. 'Participation', 'Follow')\n\n"
            "KEEP phenotypes and subtypes: HFrEF, HFpEF, HFmrEF, HFimpEF, "
            "dilated cardiomyopathy, ischaemic cardiomyopathy, "
            "left bundle branch block, reduced ejection fraction, etc.\n\n"
            "When in doubt, REJECT.\n\n"
        ),
    },
}

_BATCH_PROMPT_TEMPLATE = """{system_prompt}
For each entity below, respond with ONLY "KEEP" or "REJECT" on a separate line.
Do NOT add explanations. One word per line.

Entities:
{entities}

Responses (one per line):"""


def _norm_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)  # OCR hyphen line-break fix
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _must_reject(csv_name: str, text: str, condition_names: set[str]) -> bool:
    t = _norm_text(text)
    low = t.lower()
    if not t:
        return True
    if re.search(r"\b(table|figure)\s+\d|\brecommendations?\s+for\b|referenced studies|online data supplement", low):
        return True
    if csv_name == "S_therapy.csv":
        if re.search(r"^(management|treatment|diagnosis)\s+of\b", low):
            return True
    if csv_name == "S_phenotype.csv":
        if "|" in t or len(t) > 120:
            return True
    if csv_name == "S_cause.csv":
        if re.search(r"\b(their|this|that|these|those)\b", low):
            return True
        nlow = re.sub(r"[^a-z0-9\s]+", "", low)
        if any(cn and (nlow == cn or nlow in cn or cn in nlow) for cn in condition_names):
            return True
    return False


def _hf_llm_module():
    for module_name in ("pipeline.step3._hf_llm", "hf_llm"):
        try:
            return importlib.import_module(module_name)
        except Exception:
            continue
    raise ModuleNotFoundError("Could not import HF helper module.")


def _validate_batch(
    names: list[str],
    system_prompt: str,
    hf_model: str,
    hf_token: str,
    max_new_tokens: int = 256,
) -> list[bool]:
    hf_llm = _hf_llm_module()

    numbered = "\n".join(f"{i+1}. {n}" for i, n in enumerate(names))
    prompt = _BATCH_PROMPT_TEMPLATE.format(
        system_prompt=system_prompt,
        entities=numbered,
    )

    try:
        raw = hf_llm.hf_local_generate(prompt, hf_model, hf_token, max_new_tokens=max_new_tokens)
    except Exception as exc:
        print(f"  [validate] LLM error: {exc} — keeping all rows in this batch")
        return [True] * len(names)

    lines = [ln.strip().upper() for ln in raw.strip().splitlines() if ln.strip()]
    results = []
    for i in range(len(names)):
        if i < len(lines):
            clean = re.sub(r"^\d+[\.\)]\s*", "", lines[i]).strip().upper()
            results.append(clean != "REJECT")
        else:
            results.append(True)

    return results


def validate_csv(
    csv_path: Path,
    csv_name: str,
    name_col: str,
    system_prompt: str,
    hf_model: str,
    hf_token: str,
    condition_names: set[str],
    batch_size: int = 15,
) -> dict:
    if not csv_path.exists():
        return {"file": str(csv_path), "status": "missing"}

    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        return {"file": str(csv_path), "status": "empty", "kept": 0, "rejected": 0}

    names = [_norm_text(row.get(name_col) or "") for row in rows]
    keep_mask = [True] * len(rows)

    total = len(names)
    for start in range(0, total, batch_size):
        batch_names = names[start:start + batch_size]
        batch_results = _validate_batch(batch_names, system_prompt, hf_model, hf_token)
        for j, keep in enumerate(batch_results):
            keep_mask[start + j] = keep

    # Deterministic post-LLM guardrails (stronger for noisy classes).
    for i, row in enumerate(rows):
        name = _norm_text(row.get(name_col) or "")
        if _must_reject(csv_name, name, condition_names):
            keep_mask[i] = False
        row[name_col] = name

    kept_rows = [row for row, keep in zip(rows, keep_mask) if keep]
    rejected = sum(1 for k in keep_mask if not k)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept_rows)

    return {
        "file": str(csv_path),
        "status": "ok",
        "original": len(rows),
        "kept": len(kept_rows),
        "rejected": rejected,
    }


def validate_run(
    run_dir: str | Path,
    hf_model: str,
    hf_token: str,
    step2_subdir: str = "step2",
    batch_size: int = 15,
) -> dict:
    run = Path(run_dir)
    s2 = run / step2_subdir
    if not s2.is_dir():
        return {"error": f"step2 dir not found: {s2}"}

    hf_llm = _hf_llm_module()
    err = hf_llm.check_hf_local_backend()
    if err:
        raise RuntimeError(f"Local HF stack unavailable: {err}")
    probe_err = hf_llm.probe_hf_local_backend(hf_model, hf_token)
    if probe_err:
        raise RuntimeError(f"Model probe failed: {probe_err}")

    results = {}
    condition_names: set[str] = set()
    cond_csv = s2 / "S_condition.csv"
    if cond_csv.exists():
        with cond_csv.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                name = re.sub(r"[^a-z0-9\s]+", "", (row.get("conditionName") or "").strip().lower())
                name = re.sub(r"\s+", " ", name).strip()
                if name:
                    condition_names.add(name)

    for csv_name, config in _VALIDATION_PROMPTS.items():
        csv_path = s2 / csv_name
        print(f"  [validate] {run.name}/{csv_name}...", end=" ", flush=True)
        t0 = time.time()
        r = validate_csv(
            csv_path,
            csv_name=csv_name,
            name_col=config["name_col"],
            system_prompt=config["prompt"],
            hf_model=hf_model,
            hf_token=hf_token,
            condition_names=condition_names,
            batch_size=batch_size,
        )
        elapsed = time.time() - t0
        print(f"{r.get('status', '?')} kept={r.get('kept', '?')}/{r.get('original', '?')} ({elapsed:.1f}s)")
        results[csv_name] = r

    report_path = s2 / "validation_report.json"
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 2 LLM entity validation.")
    ap.add_argument("--run", type=Path, required=True, help="Run directory (e.g. outputs/pipeline-output2_new/<slug>)")
    ap.add_argument("--step2", default="step2", help="Step2 subdirectory name")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env)")
    ap.add_argument("--batch-size", type=int, default=15)
    args = ap.parse_args()

    import os
    hf_token = args.token or os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("ERROR: set --token or HF_TOKEN environment variable", file=sys.stderr)
        return 1

    results = validate_run(
        args.run,
        hf_model=args.model,
        hf_token=hf_token,
        step2_subdir=args.step2,
        batch_size=args.batch_size,
    )

    ok = sum(1 for r in results.values() if isinstance(r, dict) and r.get("status") == "ok")
    print(f"\nValidation complete: {ok}/{len(results)} CSVs processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
