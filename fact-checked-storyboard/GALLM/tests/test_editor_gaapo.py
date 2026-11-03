import json
import tempfile
import unittest
from pathlib import Path

from GAAPO.data import Example
from GAAPO.storyboard import StoryboardEvaluator
from GALLM.editor_gaapo import CachedLLM, FactScore, GroundedEditor, metrics

REQUEST = 'Create a brief patient-education scene that explains how beta-blockers protect the heart.'
FACTS = {REQUEST: [{'text': 'Beta blockers slow the heart rate and lower blood pressure.', 'headers': []},
                   {'text': 'Beta blockers may reverse some of the heart damage.', 'headers': []}]}


class Counting:
    """Counts real model calls; answers like the editor (no system message) or the checker (system message)."""
    def __init__(self, verdicts='{"verdicts": [{"sentence": 1, "label": "SUPPORTED", "fact": 1},'
                                ' {"sentence": 2, "label": "NOT_SUPPORTED", "fact": null}]}'):
        self.calls, self.verdicts = 0, verdicts

    def generate(self, prompt, system=None, **sampling):
        self.calls += 1
        if system:
            return self.verdicts
        return json.dumps({'title': 'Beta-blockers', 'scenes': [{'scene_number': 1,
            'narration': 'Beta-blockers slow your heart.', 'visual_description': 'A calm heart.',
            'on_screen_text': 'They cure everything'}]})


class Tests(unittest.TestCase):
    def test_cache_answers_repeats_from_disk(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Counting()
            first = CachedLLM(model, Path(folder) / 'cache.jsonl')
            answer = first.generate('Same question', temperature=0)
            self.assertEqual(first.generate('Same question', temperature=0), answer)
            again = CachedLLM(model, Path(folder) / 'cache.jsonl')  # a restarted run
            self.assertEqual(again.generate('Same question', temperature=0), answer)
            self.assertEqual((model.calls, again.new), (1, 0))

    def test_daily_limit_is_waited_out(self):
        from unittest import mock
        from GALLM.editor_gaapo import PatientLLM

        class RateLimitError(Exception):
            pass
        limit = RateLimitError('Rate limit reached ... on tokens per day (TPD): Limit 200000. Please try again in 1m30.5s.')
        answers = iter([limit, 'done'])

        class Flaky:
            def generate(self, prompt, system=None, **sampling):
                answer = next(answers)
                if isinstance(answer, Exception):
                    raise answer
                return answer
        with mock.patch('GALLM.editor_gaapo.time.sleep') as sleep, mock.patch('openai.RateLimitError', RateLimitError):
            self.assertEqual(PatientLLM(Flaky(), progress=lambda m: None).generate('q'), 'done')
        sleep.assert_called_once_with(100.5)  # 1 min 30.5 s plus 10 s to be safe

    def test_storyboard_is_fact_checked_once_for_both_metrics(self):
        model = Counting()
        checks, weights = metrics(FactScore(model, FACTS))
        evaluator = StoryboardEvaluator(GroundedEditor(model, FACTS), checks, weights)
        output, fitness = evaluator.assess('Be clear.', Example('t1', REQUEST, {'guideline_facts': []}))
        self.assertEqual(model.calls, 2)  # one storyboard, one fact check for both of its sentences
        # support 1/2 (x2), coverage 1 of 2 facts, readability 1, strict JSON 1 -> (1 + 0.5 + 1 + 1) / 5
        self.assertAlmostEqual(fitness, 3.5 / 5)

    def test_topic_without_facts_gets_no_storyboard_and_scores_zero(self):
        model = Counting()
        checks, weights = metrics(FactScore(model, {}))
        evaluator = StoryboardEvaluator(GroundedEditor(model, {}), checks, weights)
        self.assertEqual(evaluator.assess('Be clear.', Example('t2', 'Why is the sky blue?', {'x': 1}))[1], 0.0)
        self.assertEqual(model.calls, 0)


if __name__ == '__main__':
    unittest.main()
