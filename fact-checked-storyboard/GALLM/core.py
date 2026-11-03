"""GALLM: a standard genetic algorithm whose operators are LLM calls.

Follows Loss & Dhuvad, "An LLM-Based Genetic Algorithm for Prompt Engineering" (GECCO'25)
and its research code (github.com/leloss/gallm, gallm-fit.py). Data, the fixed target model
and the fitness come from GAAPO, so GALLM and GAAPO runs score prompts identically.
"""
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from GAAPO.core import PromptCandidate, select_best_prompts
from GAAPO.evaluation import evaluate

# Operator prompts are verbatim from gallm-fit.py, including "with using" in the mutation prompt.
SYSTEM_MESSAGE = ('You are an AI that helps people solve problems. Avoid comments outside the proposed '
                  'response as the user will use your answer to integrate with another downstream system.')
INIT_PROMPT = 'Create a {target} prompt that solves the underlying problem exemplified by the following examples:\n'
CROSSOVER_PROMPT = ('Given the following drafts for two prompts that aim to solve the same particular '
                    'problem, create a better prompt using only ideas from them.')
MUTATION_PROMPT = ('Given the following draft for a prompt that aims to solve a particular problem, '
                   'create a better prompt with using ideas from it.')


def init_request(examples, target):
    text = INIT_PROMPT.format(target=target)
    for e in examples:
        text += 'INPUT: ' + e.user_request + '\nOUTPUT: ' + json.dumps(e.output, ensure_ascii=False) + '\n\n'
    return text


def crossover_request(first, second):
    return CROSSOVER_PROMPT + '\nPROMPT 1:\n' + first + '\n\nPROMPT 2:\n' + second + '\n'


def mutation_request(prompt):
    return MUTATION_PROMPT + '\nPROMPT:\n' + prompt + '\n'


def tournament(population, rng, size):
    """gallm-fit's run_selection: one winner per slot, each the best of `size` random entrants."""
    return [max(rng.sample(population, size), key=lambda c: c.evaluation_score)
            for _ in range(len(population))]


def planned_target_calls(config, validation, test, test_prompts=2, baseline_validation=True):
    """Target-model calls of a complete run; generated prompts that repeat an earlier text cost less."""
    per_generation = (config.population_size - 1) * validation  # the elite keeps its score
    return ((validation if baseline_validation else 0) + config.population_size * validation
            + config.generations * per_generation + test_prompts * test)


@dataclass
class GALLMConfig:
    population_size: int = 20
    generations: int = 20
    mutation_rate: float = 0.1
    tournament_size: int = 3
    init_examples: int = 16
    random_seed: int = 42
    target_name: str = 'Llama 3.2 1B Instruct'
    init_sampling: dict = field(default_factory=lambda: {'temperature': 1.0, 'top_p': 0.9})
    operator_sampling: dict = field(default_factory=lambda: {'temperature': 1.0, 'top_p': 0.9})

    def __post_init__(self):
        # Selected parents are paired (0,1), (2,3), ...; an odd population leaves one unpaired.
        if self.population_size < 2 or self.population_size % 2:
            raise ValueError('Population size must be an even number of at least 2')
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError('Tournament size must lie between 1 and the population size')
        if self.generations < 0 or self.init_examples < 1 or not 0 <= self.mutation_rate <= 1:
            raise ValueError('Invalid generations, example count or mutation rate')

    @classmethod
    def from_dict(cls, data):
        return cls(**data.get('gallm', {}))


class GALLMOptimizer:
    def __init__(self, config, helper_llm, evaluator, progress=None, checkpoint_directory=None):
        self.config, self.llm, self.evaluator = config, helper_llm, evaluator
        self.progress = progress or (lambda message: None)
        self.checkpoint_directory = checkpoint_directory
        self.rng = random.Random(config.random_seed)
        self.history, self.generation_logs, self.population = [], [], []
        self.scores = {}
        self.helper_calls = 0
        self.completed = False

    def _helper(self, request, sampling):
        self.helper_calls += 1
        return self.llm.generate(request, system=SYSTEM_MESSAGE, **sampling).strip()

    def _candidate(self, text, generation, method, parents=(), metadata=None):
        candidate = PromptCandidate(f'g{len(self.history):06d}', text, generation,
                                    [p.id for p in parents], method, metadata=metadata or {})
        candidate.metadata['parent_prompts'] = [p.prompt_text for p in parents]
        self.history.append(candidate)
        if self.checkpoint_directory:
            self.save(self.checkpoint_directory)
        return candidate

    def _evaluate(self, candidates, validation):
        # The target decodes greedily, so an identical prompt text always earns the same score.
        # Reusing it saves model calls without changing any result.
        for c in candidates:
            if c.prompt_text in self.scores:
                c.evaluation_score = self.scores[c.prompt_text]
                c.metadata.update(validation_examples=len(validation), score_cached=True)
            else:
                evaluate([c], validation, self.evaluator, {'method': 'complete'}, self.rng)
                self.scores[c.prompt_text] = c.evaluation_score

    def optimize(self, train, validation):
        from GAAPO.data import validate_splits
        if self.history:
            raise RuntimeError('Use a fresh optimizer for each experiment')
        train, validation = tuple(train), tuple(validation)
        validate_splits(train, validation)
        config = self.config
        examples = train[:config.init_examples]
        request = init_request(examples, config.target_name)
        self.progress(f'Writing {config.population_size} first prompts from {len(examples)} training examples')
        self.population = [self._candidate(self._helper(request, config.init_sampling), 0, 'initial',
                                           metadata={'operator': 'gallm_init',
                                                     'example_ids': [e.id for e in examples]})
                           for _ in range(config.population_size)]
        self.progress(f'Initial evaluation: {len(self.population)} prompts on {len(validation)} validation examples')
        self._evaluate(self.population, validation)
        self._log(0)
        for generation in range(1, config.generations + 1):
            best = select_best_prompts(self.population, 1)[0]
            selected = tournament(self.population, self.rng, config.tournament_size)
            children = []
            for first, second in zip(selected[0::2], selected[1::2]):
                for p1, p2 in ((first, second), (second, first)):
                    self.progress(f'Generation {generation}: crossover {p1.id} x {p2.id}')
                    text = self._helper(crossover_request(p1.prompt_text, p2.prompt_text), config.operator_sampling)
                    metadata = {'operator': 'gallm_llm_crossover', 'mutated': False}
                    if self.rng.random() < config.mutation_rate:
                        metadata.update(mutated=True, before_mutation=text)
                        text = self._helper(mutation_request(text), config.operator_sampling)
                    children.append(self._candidate(text, generation, 'crossover', (p1, p2), metadata))
            # gallm-fit overwrites the first child with the best individual (elitism).
            children[0].metadata['replaced_by_elite'] = best.id
            self.population = [best] + children[1:]
            self.progress(f'Generation {generation}: evaluating {len(children) - 1} new prompts')
            self._evaluate(children[1:], validation)
            self._log(generation)
            self.progress(f'Generation {generation}: best score={self.population_best().evaluation_score:.3f}')
        self.completed = True
        return self.population_best()

    def population_best(self):
        return select_best_prompts(self.population, 1)[0]

    def _log(self, generation):
        scores = [c.evaluation_score for c in self.population]
        best = self.population_best()
        self.generation_logs.append({'generation': generation, 'best_score': best.evaluation_score,
                                     'best_id': best.id, 'mean_score': sum(scores) / len(scores),
                                     'target_calls': getattr(self.evaluator, 'calls', None),
                                     'helper_calls': self.helper_calls})
        if self.checkpoint_directory:
            self.save(self.checkpoint_directory)
            snapshot = {'generation': generation, 'population': [asdict(c) for c in self.population],
                        'summary': self.generation_logs[-1]}
            (Path(self.checkpoint_directory) / f'generation_{generation:03d}.json').write_text(
                json.dumps(snapshot, indent=2, ensure_ascii=False), encoding='utf-8')

    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / 'candidates.jsonl').write_text(''.join(json.dumps(asdict(c), ensure_ascii=False) + '\n'
                                                     for c in self.history), encoding='utf-8')
        (path / 'generations.json').write_text(json.dumps(self.generation_logs, indent=2), encoding='utf-8')
        (path / 'config.json').write_text(json.dumps(asdict(self.config), indent=2), encoding='utf-8')
        scored = [c for c in self.population if c.evaluation_score is not None]
        if scored:
            (path / 'best_prompt.txt').write_text(select_best_prompts(scored, 1)[0].prompt_text, encoding='utf-8')
