"""Rerun GAAPO with the new scorecard: Mahsa's optimizer and settings, with prompts chosen on
the 30 ruler validation examples and tested on the 15 ruler test examples. Her files stay
unchanged. Run with python -m GALLM.gaapo_rerun --smoke-test --output GALLM/runs/rerun-smoke."""
import argparse
import json
from pathlib import Path

from GAAPO.core import GAAPOConfig, GAAPOOptimizer
from GAAPO.data import load_dataset, validate_splits
from GAAPO.experiment import INITIAL
from GAAPO.models import MedicalStoryboardModel, MockMedicalStoryboardModel, MockPromptGenerationLLM
from GAAPO.run_logging import LoggedLLM, RunLog

from .experiment import load_compare_prompts
from .models import ChatLLM
from .scorecard import ScorecardEvaluator, load_ruler


class GAAPOHelper:
    """GAAPO's helper call (one user message, temperature 0) with ChatLLM's free-tier retries."""
    def __init__(self, chat):
        self.chat = chat

    def generate(self, prompt):
        return self.chat.generate(prompt, temperature=0)


def shrink_to_smoke_test(data):
    """GAAPO's own smoke-test settings: one generation, every method once."""
    data['gaapo'].update(population_size=5, generations=1, parents_per_generation=2)
    for settings in data['generation_methods'].values():
        settings.update(enabled=True, candidates_per_generation=1)
    data['generation_methods']['apo']['batch_size'] = 1


def planned_calls(config, initial_prompts, train, validation, test, test_prompts):
    """1B-model calls of a run with complete evaluation (as Mahsa ran it): every prompt on all
    validation examples, APO's training batches, and each compared prompt on the test examples."""
    methods = [s for s in config.generation_methods.values() if s.get('enabled', True)]
    per_generation = sum(s.get('candidates_per_generation', 0) for s in methods)
    apo = config.generation_methods.get('apo', {})
    apo_calls = apo.get('candidates_per_generation', 0) * min(apo.get('batch_size', 8), train) if apo.get('enabled', True) else 0
    return (initial_prompts + config.generations * per_generation) * validation + config.generations * apo_calls + test_prompts * test


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default='GALLM/rerun.config.json')
    parser.add_argument('--dry-run', action='store_true', help='Fake models: checks the wiring in seconds')
    parser.add_argument('--smoke-test', action='store_true', help='One generation, one validation and test example')
    parser.add_argument('--output', default='GALLM/runs/rerun')
    args = parser.parse_args()
    log = RunLog(args.output)
    progress = log.progress
    data = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.smoke_test:
        shrink_to_smoke_test(data)
    config = GAAPOConfig.from_dict(data)
    train = load_dataset(data['datasets']['train'])
    validation, test = (load_ruler(data['datasets']['ruler'], split) for split in ('validation', 'test'))
    if args.smoke_test:
        train, validation, test = train[:3], validation[:1], test[:1]
    validate_splits(train, validation, test)
    initial = (data.get('initial_prompt', INITIAL), *data.get('initial_variants', ()))
    others = load_compare_prompts(data, progress)
    progress(f'Planned 1B-model calls: {planned_calls(config, len(initial), len(train), len(validation), len(test), 2 + len(others))}')
    if args.dry_run:
        model, llm = MockMedicalStoryboardModel(), MockPromptGenerationLLM()
    else:
        try:
            # Trust the operating system's certificates, which antivirus HTTPS scanning (e.g. Norton) relies on.
            import truststore
            truststore.inject_into_ssl()
        except ImportError:
            pass
        from dotenv import load_dotenv
        # Explicit process environment wins over local files; never copy secrets into run logs.
        load_dotenv(Path(__file__).resolve().parent.parent / 'GAAPO' / '.env', override=False)
        chat = ChatLLM.from_settings(data['optimizer_llm'])
        chat.client.models.list()  # fail now on a bad or expired key, not after the first 90 storyboards
        llm = GAAPOHelper(chat)
        progress('Helper key works. Loading fixed medical model and adapter ...')
        model = MedicalStoryboardModel.from_adapter(**data['target_model'])
        progress('Medical model loaded. Starting GAAPO with the new scorecard.')

    output = Path(args.output)
    (output / 'experiment.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    (output / 'data_split_ids.json').write_text(json.dumps({name: [e.id for e in split]
        for name, split in [('train', train), ('validation', validation), ('test', test)]}, indent=2), encoding='utf-8')

    def save_storyboard(record):
        log.write('storyboards.jsonl', record)
        if record['status'] == 'speech_ready':
            directory = output / 'speech_storyboards'
            directory.mkdir(exist_ok=True)
            (directory / f"call_{record['call']:03d}.json").write_text(
                json.dumps(record['storyboard'], ensure_ascii=False, indent=2), encoding='utf-8')

    evaluator = ScorecardEvaluator(model, progress=progress, result_callback=save_storyboard)
    optimizer = GAAPOOptimizer(config, LoggedLLM(llm, log), evaluator, progress=progress,
                               checkpoint_directory=args.output)
    try:
        optimizer.optimize(initial[0], train, validation, initial_variants=initial[1:])
    finally:
        optimizer.save(args.output)
    progress('Optimization complete. Scoring the prompts on the held-out ruler test examples ...')
    result = {'baseline': evaluator.score(optimizer.baseline.prompt_text, test),
              'gaapo_new_score': evaluator.score(optimizer.population[0].prompt_text, test)}
    for name, prompt in others.items():
        result[name] = evaluator.score(prompt, test)
    result.update(test_examples=len(test), model_calls=evaluator.calls, dry_run=args.dry_run, smoke_test=args.smoke_test)
    (output / 'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'comparison': result, 'generations': optimizer.generation_logs}, indent=2))


if __name__ == '__main__':
    main()
