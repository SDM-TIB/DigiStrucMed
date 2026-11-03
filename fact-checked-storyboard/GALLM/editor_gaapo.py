"""GAAPO for the editor prompt (agreed 2026-10-30). Mahsa's GAAPO is unchanged; what it optimizes is the
"GAAPO-selected prompt" inside the grounded editor (GALLM.grounded_pipeline), and the fitness comes from the
fact check. The fine-tuned model is not used. The safety rules, the guideline facts per topic (looked up once)
and the checker stay fixed. Every model call is cached on disk, so a run stopped by Groq's daily limit
continues where it stopped (GAAPO is seeded, so it asks the same questions again).
Run with python -m GALLM.editor_gaapo --smoke-test --output GALLM/runs/editor-smoke."""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from GAAPO.core import GAAPOConfig, GAAPOOptimizer
from GAAPO.data import Example, validate_splits
from GAAPO.experiment import INITIAL
from GAAPO.models import MockPromptGenerationLLM
from GAAPO.run_logging import LoggedLLM, RunLog
from GAAPO.storyboard import StoryboardEvaluator, format_validity

from .fact_check import check_sentences
from .gaapo_rerun import GAAPOHelper, planned_calls, shrink_to_smoke_test
from .grounded_pipeline import checked_sentences, editor_prompt, read_storyboard, with_context
from .models import ChatLLM
from .scorecard import readability, unless_silent


class PatientLLM:
    """Waits out Groq's free daily token limit (a rolling 24 hours) instead of stopping the run."""
    def __init__(self, llm, progress=print):
        self.llm, self.progress = llm, progress

    def generate(self, prompt, system=None, **sampling):
        from openai import RateLimitError
        while True:
            try:
                return self.llm.generate(prompt, system=system, **sampling)
            except RateLimitError as error:
                if 'per day' not in str(error):
                    raise
                wait = re.search(r'try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?', str(error))
                seconds = (int(wait.group(1) or 0) * 3600 + int(wait.group(2) or 0) * 60 + float(wait.group(3) or 0)
                           if wait else 600)
                self.progress(f'Groq daily limit reached; waiting {seconds / 60:.0f} min, then continuing ...')
                time.sleep(seconds + 10)


class CachedLLM:
    """Answers a repeated question from disk; only new questions reach the model."""
    def __init__(self, llm, path):
        self.llm, self.path = llm, Path(path)
        self.cache = {}
        if self.path.exists():
            for line in self.path.read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                self.cache[row['key']] = row['answer']
        self.new = 0

    def generate(self, prompt, system=None, **sampling):
        key = hashlib.sha256(json.dumps([prompt, system, sorted(sampling.items())]).encode('utf-8')).hexdigest()
        if key not in self.cache:
            self.cache[key] = self.llm.generate(prompt, system=system, **sampling)
            self.new += 1
            with self.path.open('a', encoding='utf-8') as f:
                f.write(json.dumps({'key': key, 'answer': self.cache[key]}, ensure_ascii=False) + '\n')
        return self.cache[key]


class GroundedEditor:
    """GAAPO's "model": writes a storyboard for a request from the prompt being tested plus that topic's facts."""
    def __init__(self, llm, facts):
        self.llm, self.facts = llm, facts

    def generate(self, prompt, request):
        facts = self.facts.get(request)
        return self.llm.generate(editor_prompt(request, facts, prompt), temperature=0) if facts else ''


class FactScore:
    """The fact check as two metrics; each storyboard is checked once (one call for all its sentences)."""
    def __init__(self, checker, facts):
        self.checker, self.facts, self.seen = checker, facts, {}

    def verdicts(self, output, example):
        key = (output, example.user_request)
        if key not in self.seen:
            storyboard, facts = read_storyboard(output), self.facts.get(example.user_request, [])
            sentences = [s for _, _, s in checked_sentences(storyboard)] if storyboard else []
            self.seen[key] = check_sentences(self.checker, sentences, [with_context(f) for f in facts])
        return self.seen[key]

    def support(self, output, example):
        """Share of what a patient hears or reads that a guideline fact supports (0 when nothing is said)."""
        v = self.verdicts(output, example)
        return sum(x['label'] == 'SUPPORTED' for x in v) / len(v) if v else 0.0

    def coverage(self, output, example):
        """How many of the topic's facts the storyboard uses, up to 3."""
        facts = self.facts.get(example.user_request, [])
        used = {str(x['fact']) for x in self.verdicts(output, example) if x['label'] == 'SUPPORTED' and x['fact']}
        return min(1.0, len(used) / min(3, len(facts))) if facts else 0.0


def metrics(score):
    """Fact support counts double: it is the point of the pipeline. Refusals and empty answers score 0."""
    checks = {'fact_support': score.support, 'fact_coverage': score.coverage,
              'readability': lambda output, example: readability(output), 'format_validity': format_validity}
    return ({name: unless_silent(check) for name, check in checks.items()},
            {'fact_support': 2.0, 'fact_coverage': 1.0, 'readability': 1.0, 'format_validity': 1.0})


def load_examples(data):
    """Train: the 6 grounded-v1 storyboards (fact-checked, so few-shot copies only checked text).
    Tuning and test: topics with guideline facts, their facts chosen once (GALLM/data/editor_topics.json)."""
    topics = json.loads(Path(data['topics']).read_text(encoding='utf-8'))
    facts = {t['request']: t['facts'] for t in topics['tuning'] + topics['test']}
    train = []
    for path in sorted(Path(data['train_run']).glob('ruler-0??.json')):
        r = json.loads(path.read_text(encoding='utf-8'))
        facts[r['request']] = r['facts']
        train.append(Example(path.stem, r['request'], {k: v for k, v in r['final_storyboard'].items() if k != 'not_covered'}))
    as_examples = lambda rows: tuple(Example(t['id'], t['request'], {'guideline_facts': [f['text'] for f in t['facts']]}) for t in rows)
    return tuple(train), as_examples(topics['tuning']), as_examples(topics['test']), facts


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default='GALLM/editor.config.json')
    parser.add_argument('--dry-run', action='store_true', help='Fake models: checks the wiring in seconds')
    parser.add_argument('--smoke-test', action='store_true', help='One generation, one tuning and one test topic')
    parser.add_argument('--output', default='GALLM/runs/editor-gaapo')
    parser.add_argument('--cache', default='GALLM/runs/editor-model-calls.jsonl',
                        help='Shared by smoke test and full run, so no question is paid for twice')
    args = parser.parse_args()
    log = RunLog(args.output)
    progress, output = log.progress, Path(args.output)
    data = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.smoke_test:
        shrink_to_smoke_test(data)
    config = GAAPOConfig.from_dict(data)
    train, tuning, test, facts = load_examples(data)
    if args.smoke_test:
        tuning, test = tuning[:1], test[:1]
    validate_splits(train, tuning, test)
    initial = tuple(Path(p).read_text(encoding='utf-8').strip() if p.endswith('.txt') else INITIAL
                    for p in data['initial_prompts'])
    progress(f'Planned editor calls (each plus one fact check): '
             f'{planned_calls(config, len(initial), len(train), len(tuning), len(test), 3)}')
    if args.dry_run:
        def fake_storyboard(prompt, **_):
            return json.dumps({'title': 'Test', 'scenes': [{'scene_number': 1, 'narration': 'It helps.',
                               'visual_description': 'A heart.', 'on_screen_text': 'Helps'}]})
        fake = type('Fake', (), {'generate': lambda self, prompt, system=None, **s: (
            '{"verdicts": [{"sentence": 1, "label": "SUPPORTED", "fact": 1}, {"sentence": 2, "label": "SUPPORTED", "fact": 1}]}'
            if system else fake_storyboard(prompt))})()
        editor_llm = checker = fake
        helper = MockPromptGenerationLLM()
    else:
        try:
            import truststore  # Norton's HTTPS scanning needs the system certificates
            truststore.inject_into_ssl()
        except ImportError:
            pass
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / 'GAAPO' / '.env', override=False)
        chat = ChatLLM.from_settings(dict(data['optimizer_llm']))
        chat.client.models.list()  # fail now on a bad or expired key
        cached = CachedLLM(PatientLLM(chat, progress), args.cache)
        editor_llm, checker = cached, cached
        if data.get('checker_llm'):  # a second model: its own daily allowance, and the writer does not grade itself
            checker = CachedLLM(PatientLLM(ChatLLM.from_settings(dict(data['checker_llm'])), progress),
                                str(Path(args.cache).with_suffix('')) + '-checker-' + data['checker_llm']['model'].replace('/', '_') + '.jsonl')
        helper = GAAPOHelper(cached)
        progress(f'Groq key works; {len(cached.cache)} answers already cached; checker: '
                 f'{(data.get("checker_llm") or data["optimizer_llm"])["model"]}.')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'experiment.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    (output / 'data_split_ids.json').write_text(json.dumps({name: [e.id for e in split] for name, split in
        [('train', train), ('tuning', tuning), ('test', test)]}, indent=2), encoding='utf-8')
    checks, weights = metrics(FactScore(checker, facts))
    evaluator = StoryboardEvaluator(GroundedEditor(editor_llm, facts), checks, weights, progress=progress,
                                    result_callback=lambda record: log.write('storyboards.jsonl', record))
    optimizer = GAAPOOptimizer(config, LoggedLLM(helper, log), evaluator, progress=progress,
                               checkpoint_directory=args.output)
    try:
        optimizer.optimize(initial[0], train, tuning, initial_variants=initial[1:])
    finally:
        optimizer.save(args.output)
    progress('Optimization complete. Scoring on the unseen test topics ...')
    result = {'mahsa_best_start': evaluator.score(initial[0], test),
              'gaapo_initial': evaluator.score(INITIAL, test),
              'editor_gaapo_best': evaluator.score(optimizer.population[0].prompt_text, test)}
    result.update(test_topics=len(test), editor_calls=evaluator.calls, dry_run=args.dry_run, smoke_test=args.smoke_test)
    (output / 'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'comparison': result, 'generations': optimizer.generation_logs}, indent=2))


if __name__ == '__main__':
    main()
