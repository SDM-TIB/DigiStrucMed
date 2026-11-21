import json
import re
import tempfile
import unittest

from GALLM import director as dr

RECORD = {'question': 'When can I drive again after getting an ICD?',
          'scenes': [{'narration': "Hello. Let's look at what the heart guidelines say about your question.", 'medical': False},
                     {'narration': 'If you got an ICD for primary prevention, wait four weeks before driving again.',
                      'on_screen_text': 'Drive after 4 weeks', 'medical': True},
                     {'narration': "Please check the details with your care team, and don't wait until things get worse.", 'medical': False}],
          'sources': [{'scene': 2, 'source': 'vijgen 2009; p. 8'}], 'video_prompt': 'x'}


def board(props=('none', 'calendar', 'none'), patient=('asks the question', 'nods', 'looks relieved'), shots=('wide', 'close-up on prop', 'medium')):
    return [{'line': n, 'shot': s, 'doctor': 'points to the prop' if p != 'none' else 'smiles warmly', 'patient': pt, 'prop': p,
             'heading': n == 2, 'pause': 0.5} for n, (p, pt, s) in enumerate(zip(props, patient, shots), 1)]


class Director:
    """Gives the storyboard in `boards` for the first instruction text found in the prompt, else the default one."""
    def __init__(self, boards, default):
        self.boards, self.default, self.prompts = boards, default, []

    def generate(self, prompt, system=None, **_):
        self.prompts.append(prompt)
        for text, scenes in self.boards.items():
            if prompt.startswith(text):
                return json.dumps({'scenes': scenes})
        return json.dumps({'scenes': self.default})


class Judge:
    """Prefers the storyboard that shows a heart model; records every comparison."""
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, system=None, **_):
        self.prompts.append(prompt)
        a = re.search(r'Storyboard A:\n(.*?)\n\nStoryboard B:', prompt, re.S).group(1)
        b = re.search(r'Storyboard B:\n(.*?)\n\nWhich makes', prompt, re.S).group(1)
        better = 'A' if 'heart model' in a else 'B' if 'heart model' in b else 'tie'
        return json.dumps({'better': better, 'why': 'shows the heart'})


class Shape(unittest.TestCase):
    lines = dr.script_lines(RECORD)

    def test_locked_lines_keep_words_heading_and_source(self):
        self.assertEqual([(line['n'], line['medical'], line['heading']) for line in self.lines],
                         [(1, False, ''), (2, True, 'Drive after 4 weeks'), (3, False, '')])
        self.assertEqual(self.lines[1]['source'], 'vijgen 2009; p. 8')

    def test_a_good_storyboard_passes_the_gate(self):
        self.assertEqual(dr.gate(board(), self.lines), [])

    def test_a_missing_or_moved_line_or_unknown_value_fails(self):
        self.assertTrue(dr.gate(board()[:2], self.lines))
        self.assertTrue(dr.gate(list(reversed(board())), self.lines))
        bad = board()
        bad[1]['prop'] = 'a chart showing 4 weeks'
        self.assertIn('prop', dr.gate(bad, self.lines)[0])
        self.assertTrue(dr.gate(None, self.lines))

    def test_the_director_cannot_add_words(self):
        scenes = board()
        scenes[1]['says'] = 'You can drive after one day.'  # anything the director writes is ignored
        text = dr.scene_texts(scenes, self.lines, RECORD['question'])
        self.assertNotIn('one day', text)
        for line in self.lines:
            self.assertIn(f'says: "{line["text"]}"', text)
        self.assertIn('The patient asks: "When can I drive again after getting an ICD?"', text)
        self.assertIn('On-screen heading: "Drive after 4 weeks"', text)
        prompt = dr.video_prompt(RECORD, scenes)
        self.assertIn('Spoken words EXACTLY as written below', prompt)

    def test_code_quality_rewards_a_fitting_object_and_a_patient_who_takes_part(self):
        good, parts = dr.code_quality(board(), self.lines)
        self.assertEqual(parts, {'objects fit': 1.0, 'doctor shows the object': 1.0, 'patient takes part': 1.0})
        dull, _ = dr.code_quality(board(props=('none',) * 3, patient=('listens',) * 3, shots=('medium',) * 3), self.lines)
        self.assertGreater(good, dull)
        asked_twice, parts = dr.code_quality(board(patient=('asks the question', 'asks the question', 'nods')), self.lines)
        self.assertEqual(parts['patient takes part'], 0.0)

    def test_no_object_is_fine_but_a_decorative_one_is_not(self):
        _, parts = dr.code_quality(board(props=('none', 'none', 'none')), self.lines)
        self.assertEqual(parts['objects fit'], 0.5)   # a calendar would have fitted "four weeks"
        _, parts = dr.code_quality(board(props=('none', 'wine glass', 'none')), self.lines)
        self.assertEqual(parts['objects fit'], 0.0)   # nothing about alcohol in the line
        record = {**RECORD, 'scenes': [RECORD['scenes'][0], {'narration': 'Ask your care team which choice suits you.',
                                                             'medical': True}, RECORD['scenes'][2]]}
        _, parts = dr.code_quality(board(props=('none', 'none', 'none')), dr.script_lines(record))
        self.assertEqual(parts['objects fit'], 1.0)   # "none" is fine when no object fits the words
        self.assertEqual(sorted(dr.fitting_props('An ICD watches your heart rhythm.')),
                         ['ICD device model', 'heart model', 'tablet showing a heartbeat line'])

    def test_variety_no_longer_counts(self):
        same, _ = dr.code_quality(board(shots=('medium',) * 3), self.lines)
        varied, _ = dr.code_quality(board(shots=('wide', 'close-up on prop', 'medium')), self.lines)
        self.assertEqual(same, varied)

    def test_an_object_showing_what_the_line_warns_against_fails_the_gate(self):
        record = {**RECORD, 'scenes': [RECORD['scenes'][0], {'narration': 'Stop smoking to protect your heart.', 'medical': True},
                                       RECORD['scenes'][2]]}
        lines = dr.script_lines(record)
        problems = dr.gate(board(props=('none', 'cigarette pack', 'none')), lines)
        self.assertTrue(any('advises against' in p for p in problems))
        describes = {**RECORD, 'scenes': [RECORD['scenes'][0], {'narration': 'Smoking harms your heart and blood vessels.',
                                                                'medical': True}, RECORD['scenes'][2]]}
        self.assertEqual(dr.gate(board(props=('none', 'cigarette pack', 'none')), dr.script_lines(describes)), [])


class Evaluator(unittest.TestCase):
    def setUp(self):
        from GAAPO.data import Example
        self.example = Example(id='icd', user_request=RECORD['question'], output=['x'])
        self.heart = board(props=('none', 'heart model', 'none'))

    def test_start_prompt_needs_no_judge_and_a_better_storyboard_wins(self):
        director = Director({'BETTER': self.heart}, board())
        judge = Judge()
        ev = dr.DirectorEvaluator({'icd': RECORD}, director, judge)
        start = ev.score(dr.START_PROMPT, [self.example])
        self.assertEqual(judge.prompts, [])  # the start prompt is compared with nothing
        better = ev.score('BETTER', [self.example])
        self.assertEqual(len(judge.prompts), 2)  # asked in both orders
        self.assertGreater(better, start)
        self.assertEqual(ev.score('BETTER', [self.example]), better)  # scored once, then remembered
        self.assertEqual(len(judge.prompts), 2)

    def test_a_judge_that_always_picks_the_first_gives_a_tie(self):
        class FirstJudge:
            def generate(self, prompt, system=None, **_):
                return json.dumps({'better': 'A', 'why': 'first one'})
        ev = dr.DirectorEvaluator({'icd': RECORD}, Director({'OTHER': self.heart}, board()), FirstJudge())
        verdict = ev.result('OTHER', 'icd')['judge']
        self.assertEqual((verdict['points'], verdict['votes']), (0.5, [1.0, 0.0]))

    def test_an_unusable_storyboard_scores_zero_and_says_why(self):
        ev = dr.DirectorEvaluator({'icd': RECORD}, Director({'BROKEN': board()[:1]}, board()), Judge())
        output, score = ev.assess('BROKEN', self.example)
        self.assertEqual(score, 0.0)
        self.assertIn('UNUSABLE', output)


class WithGAAPO(unittest.TestCase):
    def test_gaapo_keeps_the_better_prompt(self):
        from GAAPO.core import GAAPOConfig, GAAPOOptimizer
        from GAAPO.data import Example
        records = {name: {**RECORD, 'question': f'{RECORD["question"]} ({name})'} for name in ('t1', 'v1')}
        examples = {name: Example(id=name, user_request=r['question'], output=['x']) for name, r in records.items()}

        class Optimizer:  # GAAPO's mutation step asks for a revised prompt
            def generate(self, prompt):
                return 'BETTER: show a heart model when the heart is explained.'
        director = Director({'BETTER': board(props=('none', 'heart model', 'none'))}, board())
        ev = dr.DirectorEvaluator(records, director, Judge())
        methods = {'mutation': {'enabled': True, 'candidates_per_generation': 1},
                   **{m: {'enabled': False, 'candidates_per_generation': 0} for m in ('crossover', 'apo', 'opro', 'fewshot')}}
        config = GAAPOConfig(population_size=2, generations=1, parents_per_generation=1, generation_methods=methods)
        with tempfile.TemporaryDirectory() as folder:
            best = GAAPOOptimizer(config, Optimizer(), ev, checkpoint_directory=folder).optimize(
                dr.START_PROMPT, [examples['t1']], [examples['v1']])
        self.assertTrue(best.prompt_text.startswith('BETTER'))
        self.assertEqual(best.generation_method, 'mutation')


if __name__ == '__main__':
    unittest.main()
