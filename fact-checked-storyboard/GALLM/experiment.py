"""Run with python -m GALLM.experiment --dry-run --output GALLM/runs/dry."""
import argparse
import json
from pathlib import Path

from GAAPO.data import Example, load_dataset, validate_splits
from GAAPO.experiment import INITIAL
from GAAPO.models import MedicalStoryboardModel, MockMedicalStoryboardModel
from GAAPO.run_logging import RunLog
from GAAPO.storyboard import LLMRubricMetric, StoryboardEvaluator, format_validity, structure_completeness

from .core import GALLMConfig, GALLMOptimizer, planned_target_calls
from .models import ChatLLM, LoggedChatLLM, MockChatLLM

METRICS = {'format_validity': format_validity, 'structure_completeness': structure_completeness}


def compare_test(optimizer, baseline, train, validation, test, others=None):
    """Score the starting prompt, GALLM's winner and any saved prompts (e.g. GAAPO's) on held-out test."""
    if not optimizer.completed:
        raise RuntimeError('Test evaluation is allowed only after optimization')
    validate_splits(train, validation, test)
    score = optimizer.evaluator.score
    result = {'baseline': score(baseline, test), 'gallm': score(optimizer.population_best().prompt_text, test)}
    for name, prompt in (others or {}).items():
        result[name] = score(prompt, test)
    result['test_examples'] = len(test)
    return result


def load_compare_prompts(data, progress):
    prompts = {}
    for name, path in data.get('compare_prompts', {}).items():
        if Path(path).exists():
            prompts[name] = Path(path).read_text(encoding='utf-8')
        else:
            progress(f'Comparison prompt {name} not found at {path}; skipping it')
    return prompts


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default='GALLM/config.example.json')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke-test', action='store_true', help='Population 2, one generation, one validation/test example')
    parser.add_argument('--use-config-datasets', action='store_true',
                        help='In dry-run mode, use configured datasets with mock models')
    parser.add_argument('--output', default='GALLM/runs/experiment')
    args = parser.parse_args()
    log = RunLog(args.output)
    progress = log.progress
    data = json.loads(Path(args.config).read_text(encoding='utf-8'))
    smoke = args.smoke_test or data.get('smoke_test', False)
    if args.dry_run:
        data.setdefault('gallm', {}).update(population_size=4, generations=2)
    if smoke:
        data.setdefault('gallm', {}).update(population_size=2, generations=1, tournament_size=2)
    config = GALLMConfig.from_dict(data)
    evaluation_settings = data.get('storyboard_evaluation', {})
    metrics = {name: METRICS[name] for name in evaluation_settings.get('metrics', ['format_validity'])}
    weights = evaluation_settings.get('weights')
    if args.dry_run:
        label = {'title': 'Synthetic fixture', 'scenes': []}
        train = tuple(Example(f'tr{i}', f'Synthetic training request {i}', label) for i in range(3))
        validation = tuple(Example(f'va{i}', f'Synthetic validation request {i}', label) for i in range(4))
        test = tuple(Example(f'te{i}', f'Synthetic test request {i}', label) for i in range(4))
        if args.use_config_datasets:
            train, validation, test = (load_dataset(data['datasets'][name])
                                       for name in ('train', 'validation', 'test'))
        model, llm = MockMedicalStoryboardModel(), MockChatLLM()
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
        load_dotenv(Path(__file__).resolve().parent.parent / '.env', override=False)
        train, validation, test = (load_dataset(data['datasets'][name]) for name in ('train', 'validation', 'test'))
        validate_splits(train, validation, test)
        llm = ChatLLM.from_settings(data['optimizer_llm'])
        rubrics = evaluation_settings.get('rubrics', {})
        if rubrics:
            from GAAPO.models import configured_llm
            judge = configured_llm(data['judge_llm'])
            metrics.update({name: LLMRubricMetric(judge, rubric) for name, rubric in rubrics.items()})
        progress('Loading fixed medical model and adapter ...')
        model = MedicalStoryboardModel.from_adapter(**data['target_model'])
        progress('Medical model loaded. Starting GALLM.')
    if smoke:
        train, validation, test = train[:3], validation[:1], test[:1]
    validate_splits(train, validation, test)
    baseline = data.get('initial_prompt', INITIAL)
    others = load_compare_prompts(data, progress)
    planned = planned_target_calls(config, len(validation), len(test), 2 + len(others))
    progress(f'Planned target-model calls: {planned} (fewer if a generated prompt repeats an earlier one)')

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

    evaluator = StoryboardEvaluator(model, metrics, weights, progress=progress, result_callback=save_storyboard)
    progress(f'Scoring the starting prompt on {len(validation)} validation examples')
    baseline_validation = evaluator.score(baseline, validation)
    optimizer = GALLMOptimizer(config, LoggedChatLLM(llm, log), evaluator, progress=progress,
                               checkpoint_directory=args.output)
    try:
        best = optimizer.optimize(train, validation)
    finally:
        optimizer.save(args.output)
    progress('Optimization complete. Comparing prompts on held-out test data ...')
    result = compare_test(optimizer, baseline, train, validation, test, others)
    result.update(validation={'baseline': baseline_validation, 'gallm': best.evaluation_score},
                  target_calls=evaluator.calls, planned_target_calls=planned,
                  helper_calls=optimizer.helper_calls, dry_run=args.dry_run, smoke_test=smoke)
    (output / 'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'comparison': result, 'generations': optimizer.generation_logs}, indent=2))


if __name__ == '__main__':
    main()
