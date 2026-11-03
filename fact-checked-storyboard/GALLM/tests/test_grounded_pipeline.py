import json
import unittest

from GALLM.grounded_pipeline import (FactIndex, choose_facts, editor_prompt, is_fact, read_storyboard, remove_flagged, run,
                                     without_its_group)
from GALLM.tests.test_fact_check import filled


def unit(text, cls=(), headers=()):
    return {'text': text, 'file': 'f.nt', 'class': list(cls), 'level': [], 'headers': list(headers), 'entities': []}


FACTS = [unit('Beta blockers slow the heart rate, lower blood pressure and may reverse some of the heart damage.'),
         unit('Mineralocorticoid Antagonists (MRA) spironolactone, eplerenone Help the body get rid of salt and fluids.'),
         unit('ACE-I = angiotensin-converting enzyme inhibitor; ARB = angiotensin receptor blocker; HF = heart failure;')]


class Model:
    """Plays the editor and the checker: fixed answers in order, and a verdict per checked sentence."""
    def __init__(self, answers, verdicts):
        self.answers, self.verdicts, self.prompts = list(answers), verdicts, []

    def generate(self, prompt, system=None, **sampling):
        self.prompts.append(prompt)
        if system:  # the fact checker
            sentence = prompt.split('Sentence: "', 1)[1].split('"\n', 1)[0]
            return json.dumps(filled(self.verdicts.get(sentence, 'NOT_SUPPORTED')))
        return self.answers.pop(0)


def storyboard(narration, caption='Slow and steady'):
    return json.dumps({'title': 'Beta-blockers', 'scenes': [{'scene_number': 1, 'narration': narration,
                       'visual_description': 'A calm heart.', 'on_screen_text': caption}], 'not_covered': []})


class Tests(unittest.TestCase):
    def test_noise_and_groupless_bans_are_not_facts(self):
        self.assertFalse(is_fact(FACTS[2]))
        self.assertTrue(without_its_group(unit('ACE inhibitors, ARBs, or direct renin inhibitors are not recommended.', ['III'])))
        self.assertFalse(without_its_group(unit('In pregnant women, ACE inhibitors are not recommended.', ['III'])))

    def test_search_finds_the_topic(self):
        index = FactIndex(FACTS)
        self.assertEqual(len(index.units), 2)
        self.assertIn('Beta blockers', index.find('How do beta-blockers protect the heart?')[0]['text'])
        self.assertIn('MRA', index.find('How do MRAs help the body?')[0]['text'])

    def test_choose_facts_keeps_valid_numbers_once(self):
        model = Model(['{"facts": [2, 2, 9, "x"]}'], {})
        self.assertEqual(choose_facts(model, 'MRAs', FACTS[:2]), [FACTS[1]])

    def test_unsupported_sentence_is_rewritten_then_removed(self):
        first = storyboard('Beta-blockers slow the heart. They cure heart failure.')
        repaired = storyboard('Beta-blockers slow the heart. They make you run faster.')
        model = Model(['{"facts": [1]}', first, repaired], {'Beta-blockers slow the heart.': 'SUPPORTED',
                                                              'Slow and steady': 'SUPPORTED'})
        record = run('How do beta-blockers protect the heart?', FactIndex(FACTS), model, model, 'Be friendly.')
        self.assertEqual(record['status'], 'done')
        self.assertEqual(record['final_storyboard']['scenes'][0]['narration'], 'Beta-blockers slow the heart.')
        rewrite = [p for p in model.prompts if 'not supported by the guideline facts' in p]
        self.assertEqual(len(rewrite), 1)
        self.assertIn('"They cure heart failure." (NOT_SUPPORTED)', rewrite[0])
        self.assertEqual({r['label'] for r in record['fact_report']}, {'SUPPORTED', 'NOT_SUPPORTED'})

    def test_text_scene_number_becomes_a_number(self):
        first = storyboard('Beta-blockers slow the heart.').replace('"scene_number": 1', '"scene_number": "1"')
        model = Model(['{"facts": [1]}', first], {'Beta-blockers slow the heart.': 'SUPPORTED', 'Slow and steady': 'SUPPORTED'})
        record = run('How do beta-blockers protect the heart?', FactIndex(FACTS), model, model, 'Be friendly.')
        self.assertEqual(record['first_storyboard']['scenes'][0]['scene_number'], '1')  # the writer's answer stays as given
        self.assertEqual(record['final_storyboard']['scenes'][0]['scene_number'], 1)
        self.assertEqual(record['final_storyboard']['scenes'][0]['narration'], 'Beta-blockers slow the heart.')

    def test_greeting_without_a_claim_stays_and_is_not_repaired(self):
        first = storyboard('Hello, I am your doctor. Beta-blockers slow the heart.')
        model = Model(['{"facts": [1]}', first], {'Hello, I am your doctor.': 'NO_CLAIM', 'Beta-blockers slow the heart.': 'SUPPORTED',
                                                  'Slow and steady': 'SUPPORTED'})
        record = run('How do beta-blockers protect the heart?', FactIndex(FACTS), model, model, 'Be friendly.')
        self.assertEqual(record['final_storyboard']['scenes'][0]['narration'], 'Hello, I am your doctor. Beta-blockers slow the heart.')
        self.assertNotIn('repaired_storyboard', record)

    def test_a_field_left_empty_gets_one_more_try(self):
        first = storyboard('They cure heart failure.')
        refill = storyboard('Beta-blockers slow the heart.', caption='Something else')
        model = Model(['{"facts": [1]}', first, first, refill], {'Beta-blockers slow the heart.': 'SUPPORTED',
                                                                  'Slow and steady': 'SUPPORTED'})
        record = run('How do beta-blockers protect the heart?', FactIndex(FACTS), model, model, 'Be friendly.')
        scene = record['final_storyboard']['scenes'][0]
        self.assertEqual(scene['narration'], 'Beta-blockers slow the heart.')
        self.assertEqual(scene['on_screen_text'], 'Slow and steady')  # only the empty field comes from the new answer
        fill = [p for p in model.prompts if 'fields of your storyboard are empty' in p]
        self.assertEqual(len(fill), 1)
        self.assertIn('- scene 1 narration, removed: "They cure heart failure." (NOT_SUPPORTED)', fill[0])
        self.assertEqual([r['label'] for r in record['refill_verdicts']], ['SUPPORTED'])

    def test_a_refill_that_fails_the_check_stays_empty(self):
        first = storyboard('They cure heart failure.')
        model = Model(['{"facts": [1]}', first, first, storyboard('They make you run faster.')], {'Slow and steady': 'SUPPORTED'})
        record = run('How do beta-blockers protect the heart?', FactIndex(FACTS), model, model, 'Be friendly.')
        self.assertEqual(record['final_storyboard']['scenes'][0]['narration'], '')
        self.assertEqual(model.answers, [])  # one try only: a second one would have nothing left to pop

    def test_scene_count_is_one_unless_the_style_asks(self):
        self.assertIn('4. Exactly one scene unless', editor_prompt('Beta-blockers', FACTS[:1], 'Be friendly.'))
        self.assertIn('4. Exactly three scenes unless', editor_prompt('Beta-blockers', FACTS[:1], 'Be friendly.', 'three scenes'))

    def test_no_facts_means_no_storyboard(self):
        record = run('Why is the sky blue?', FactIndex(FACTS), Model([], {}), Model([], {}), 'Be friendly.')
        self.assertEqual(record['status'], 'no_facts')

    def test_storyboard_must_be_json_with_scenes(self):
        self.assertIsNone(read_storyboard('Here is your storyboard: Scene 1 ...'))
        self.assertEqual(read_storyboard('```json\n' + storyboard('Hi.') + '\n```')['title'], 'Beta-blockers')
        self.assertEqual(remove_flagged(json.loads(storyboard('A. B.')), [])['scenes'][0]['narration'], '')


if __name__ == '__main__':
    unittest.main()
