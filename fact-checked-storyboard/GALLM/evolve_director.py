"""Prompt engineering for the director with GAAPO (the team's genetic-algorithm prompt optimiser in gallm/GAAPO).

    python -m GALLM.evolve_director --tiny      a quick check that everything runs (1 round, few calls)
    python -m GALLM.evolve_director             the real search (4 prompts per round, 3 rounds)

What evolves: only the director's instructions (director.START_PROMPT is the first one). GAAPO makes new versions by
mutation, crossover, learning from mistakes (APO) and learning from scores (OPRO); director.DirectorEvaluator scores them.
The checked scripts are fixed inputs, saved earlier by graph_script.py:
  train       questions APO learns from (it reads why storyboards lost points)
  validation  questions every candidate is scored on; GAAPO keeps the best
  test        questions no prompt was chosen on: the fair check of the winner against the starting prompt
Output, in GALLM/runs/director-evolution/<time>/: GAAPO's own files (candidates.jsonl, generations.json, best_prompt.txt)
and report.json + report.md: start against winner on validation and test, and the winner's storyboard and video prompt for
every question. Model calls are cached, so a rerun of the same prompts costs nothing."""
import argparse
import json
import os
import time
from pathlib import Path

from GALLM import director as dr
from GALLM.graph_script import JUDGE_MODEL, WRITER_MODEL, DiskCache, GroqModel

RUNS = Path('GALLM/runs/part2')
SPLITS = {  # question records saved by graph_script.py / the demo page; each holds a checked script
    'train': ['page-20261120-145145', 'train-beta-blockers', 'train-alcohol'],
    'validation': ['icd-driving-v3', 'page-20261119-184611', 'train-exercise'],
    'test': ['train-icd-checkups', 'train-warning-signs'],
}
CACHE = 'GALLM/runs/director-calls'


def setup():
    os.environ.pop('SSLKEYLOGFILE', None)  # Norton sets it, and Python's HTTPS then crashes
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    if 'GROQ_API_KEY' not in os.environ:
        from dotenv import load_dotenv
        load_dotenv(Path('GAAPO/.env'), override=False)


def load(splits):
    from GAAPO.data import Example
    records, sets, missing = {}, {}, []
    for split, names in splits.items():
        sets[split] = []
        for name in names:
            path = RUNS / f'{name}.json'
            record = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
            if not record or not record.get('video_prompt'):
                missing.append(name)
                continue
            records[name] = record
            sets[split].append(Example(id=name, user_request=record['question'],
                                       output=[line['text'] for line in dr.script_lines(record)]))
    return records, sets, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tiny', action='store_true', help='1 round, 2 validation questions, 2 new prompts: a run check')
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--population', type=int, default=4)
    args = parser.parse_args()
    setup()
    from GAAPO.core import GAAPOConfig, GAAPOOptimizer

    splits = {k: list(v) for k, v in SPLITS.items()}
    if args.tiny:
        splits['validation'] = splits['validation'][:2]
    records, sets, missing = load(splits)
    if missing:
        print('left out (no checked script saved yet):', ', '.join(missing))
    for split in ('train', 'validation'):
        if not sets[split]:
            raise SystemExit(f'No {split} questions with a checked script; make them with graph_script.py first.')

    director = DiskCache(GroqModel(WRITER_MODEL), Path(CACHE + '-director.jsonl'))
    judge = DiskCache(GroqModel(JUDGE_MODEL), Path(CACHE + '-judge.jsonl'))
    # GAAPO's prompt-improving calls (mutation, APO, OPRO). The tiny run of 2026-11-21 had gpt-oss-20b do them: its rewrites
    # ("expert persona", "creative backstory") made the instructions longer and vaguer and lost every comparison, so the
    # larger model rewrites now. Low reasoning: these prompts hold whole storyboards, and long thinking used up the answer.
    optimizer_llm = DiskCache(GroqModel(JUDGE_MODEL, extra={'reasoning_effort': 'low'}), Path(CACHE + '-optimizer.jsonl'))
    evaluator = dr.DirectorEvaluator(records, director, judge)

    methods = {'mutation': {'enabled': True, 'candidates_per_generation': 1 if args.tiny else 2},
               'crossover': {'enabled': not args.tiny, 'candidates_per_generation': 1},
               # 0.6, not 0.7: the starting prompt's judge part is fixed at 0.5, so it never scores above 0.7 and APO
               # treated every one of its storyboards as a mistake; now only storyboards that really lost points count
               'apo': {'enabled': True, 'candidates_per_generation': 1, 'batch_size': 3, 'error_threshold': 0.6},
               'opro': {'enabled': not args.tiny, 'candidates_per_generation': 1, 'dropout': 0.2},
               'fewshot': {'enabled': False, 'candidates_per_generation': 0}}  # its example outputs are scripts, not storyboards
    config = GAAPOConfig(population_size=args.population, generations=1 if args.tiny else args.generations,
                         parents_per_generation=2, generation_methods=methods, evaluation={'method': 'complete'})
    out = Path('GALLM/runs/director-evolution') / time.strftime('%Y%m%d-%H%M%S' + ('-tiny' if args.tiny else ''))
    started = time.time()
    optimizer = GAAPOOptimizer(config, optimizer_llm, evaluator, progress=lambda m: print(m, flush=True), checkpoint_directory=out)
    best = optimizer.optimize(dr.START_PROMPT, sets['train'], sets['validation'])

    def scores(prompt, examples):
        return {e.id: evaluator.result(prompt, e.id)['score'] for e in examples}
    report = {'seconds': round(time.time() - started), 'winner_id': best.id, 'winner_method': best.generation_method,
              'winner_prompt': best.prompt_text, 'start_prompt': dr.START_PROMPT,
              'tokens': {'director': director.llm.tokens, 'judge': judge.llm.tokens, 'optimizer': optimizer_llm.llm.tokens},
              'scores': {split: {'start': scores(dr.START_PROMPT, sets[split]), 'winner': scores(best.prompt_text, sets[split])}
                         for split in ('validation', 'test') if sets[split]},
              'questions': {}}
    for name, record in records.items():
        result = evaluator.result(best.prompt_text, name)
        report['questions'][name] = {'question': record['question'], 'score': result['score'], 'storyboard': result['text'],
                                     'judge': result.get('judge'), 'code': result.get('code'),
                                     'video_prompt': dr.video_prompt(record, result['scenes']) if result.get('scenes') else None}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    mean = lambda d: sum(d.values()) / len(d) if d else 0.0
    lines = [f'# Director prompt search ({"tiny check" if args.tiny else "full run"})', '',
             f'Winner: {best.id} ({best.generation_method}). Time: {report["seconds"]} s. Tokens: {report["tokens"]}.', '',
             '| questions | start prompt | winner |', '|---|---|---|']
    lines += [f'| {split} | {mean(s["start"]):.3f} | {mean(s["winner"]):.3f} |' for split, s in report['scores'].items()]
    lines += ['', '## Winner prompt', '', '```', best.prompt_text, '```']
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:7 + len(report['scores'])]))
    print(f'written to {out}')


if __name__ == '__main__':
    main()
