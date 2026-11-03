"""The grounded pipeline on every topic the knowledge graph covers (13 fully, 14 partly; GALLM/runs/kg-coverage).
Writer gpt-oss-120b, checker gpt-oss-20b (GALLM/editor.config.json), both at temperature 0. Prompt: the thesis's best
GAAPO prompt plus the scene_number line, the best scorer in the re-test. Every model call is cached and Groq's daily
limit is waited out, and each topic is saved as it finishes, so a stopped run continues where it stopped.
--style doctor (2026-11-01): the doctor-talks-to-patient style (GALLM/prompts/doctor_style.txt) in three scenes,
on the 6 medicine topics, with the same facts and checker.
--style doctor-long (2026-11-03): the same doctor in six scenes, about one minute (GALLM/prompts/doctor_style_long.txt).
Run from the thesis repo root: python -m GALLM.run_grounded_topics [--style doctor] [--only ruler-012]"""
import argparse
import json
import traceback
from pathlib import Path

from GAAPO.storyboard import inspect_format

from .editor_gaapo import CachedLLM, PatientLLM
from .fact_check import KEPT
from .grounded_pipeline import FactIndex, run
from .models import ChatLLM

OLD = '  - `on_screen_text`\n'
NEW = OLD + '- `scene_number` must be a whole number such as 1, not text such as "1".\n'
MEDICINE_TOPICS = ['ruler-012', 'ruler-013', 'ruler-014', 'ruler-015', 'ruler-016', 'ruler-017']  # as in grounded-v1
DOCTOR_STYLES = {'doctor': ('doctor_style.txt', 'three scenes', 'doctor-talk'),  # prompt file, scenes, default output
                 'doctor-long': ('doctor_style_long.txt', 'six scenes', 'doctor-talk-long')}


def summary(rid, record):
    first, final = record.get('first_verdicts', []), record.get('fact_report', [])
    supported = lambda verdicts, labels=('SUPPORTED',): sum(v['label'] in labels for v in verdicts)
    storyboard = record.get('final_storyboard')
    valid = bool(storyboard) and inspect_format(json.dumps(
        {k: v for k, v in storyboard.items() if k != 'not_covered'}))['strict_valid']
    return {'topic': rid, 'status': record['status'], 'facts': len(record.get('facts', [])),
            'first_draft_supported': supported(first), 'first_draft_no_claim': supported(first, ('NO_CLAIM',)),
            'first_draft_sentences': len(first),
            'final_supported': supported(final), 'final_no_claim': supported(final, ('NO_CLAIM',)),
            'removed': len(final) - supported(final, KEPT),
            'valid_format': valid, 'not_covered': record.get('not_covered', [])}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--style', choices=('thesis', *DOCTOR_STYLES), default='thesis')
    parser.add_argument('--output', help='Default: GALLM/runs/grounded-27, or GALLM/runs/doctor-talk(-long) for a doctor style')
    parser.add_argument('--only', nargs='*', help='Run just these topic ids (a quick check before the full run)')
    args = parser.parse_args()
    doctor = DOCTOR_STYLES.get(args.style)
    try:
        import truststore  # Norton's HTTPS scanning needs the system certificates
        truststore.inject_into_ssl()
    except ImportError:
        pass
    from dotenv import load_dotenv
    load_dotenv(Path('GAAPO/.env'), override=False)
    config = json.loads(Path('GALLM/editor.config.json').read_text(encoding='utf-8'))
    cache = Path('GALLM/runs/editor-model-calls.jsonl')
    editor = CachedLLM(PatientLLM(ChatLLM.from_settings(dict(config['optimizer_llm']))), cache)
    checker = CachedLLM(PatientLLM(ChatLLM.from_settings(dict(config['checker_llm']))),
                        str(cache.with_suffix('')) + '-checker-' + config['checker_llm']['model'].replace('/', '_') + '.jsonl')
    thesis_prompt = Path(config['initial_prompts'][0]).read_text(encoding='utf-8').strip()
    assert thesis_prompt.count(OLD) == 1
    gaapo_prompt = (Path('GALLM/prompts', doctor[0]).read_text(encoding='utf-8').strip() if doctor
                    else thesis_prompt.replace(OLD, NEW))
    coverage = json.loads(Path('GALLM/runs/kg-coverage/coverage_review.json').read_text(encoding='utf-8'))
    topics = args.only or (MEDICINE_TOPICS if doctor else [rid for rid, c in coverage.items() if c['coverage'] in ('C', 'P')])
    index = FactIndex.from_graph()
    out = Path(args.output or ('GALLM/runs/' + doctor[2] if doctor else 'GALLM/runs/grounded-27'))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, rid in enumerate(topics, 1):
        path = out / f'{rid}.json'
        if not path.exists():
            print(f'[{number}/{len(topics)}] {rid} ...', flush=True)
            try:
                record = run(coverage[rid]['request'], index, editor, checker, gaapo_prompt,
                             doctor[1] if doctor else 'one scene')
            except Exception:
                print(rid, 'FAILED\n' + traceback.format_exc()[-600:], flush=True)
                continue
            path.write_text(json.dumps({**record, 'coverage': coverage[rid]['coverage']}, ensure_ascii=False, indent=1),
                            encoding='utf-8')
        rows.append({**summary(rid, json.loads(path.read_text(encoding='utf-8'))), 'coverage': coverage[rid]['coverage']})
        r = rows[-1]
        print(f"{rid} ({r['coverage']}) {r['status']} | facts {r['facts']} | first draft {r['first_draft_supported']}/"
              f"{r['first_draft_sentences']} supported, {r['first_draft_no_claim']} no claim | removed {r['removed']} | "
              f"valid format {r['valid_format']}", flush=True)
    if not args.only:
        (out / 'summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'done: {len(rows)} of {len(topics)} topics; new model calls: writer {editor.new}, checker {checker.new}')


if __name__ == '__main__':
    main()
