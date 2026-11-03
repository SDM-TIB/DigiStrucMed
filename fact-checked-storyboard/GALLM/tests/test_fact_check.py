import json
import unittest

from GALLM.fact_check import check_sentence, check_sentences, decide, parse

FACTS = ['Beta blockers slow the heart rate, lower blood pressure and may reverse some of the heart damage.']


def filled(label='SUPPORTED', **boxes):
    """A checklist as the checker fills it in, leading to label unless boxes change it."""
    change = {'SUPPORTED': {}, 'NO_CLAIM': {'medical_statement': False}, 'NOT_SUPPORTED': {'states_it': False},
              'CONTRADICTED': {'conflicts': True}}[label]
    return {'medical_statement': True, 'fact': 1, 'fact_is_about': 'everyone', 'sentence_is_about': 'everyone',
            'same_people': True, 'added': [], 'conflicts': False, 'states_it': True, 'reason': 'test', **change, **boxes}


class Judge:
    """Stands in for the checker model: returns a fixed answer and remembers what it was asked."""
    def __init__(self, answer):
        self.answer, self.asked = answer, []

    def generate(self, prompt, system=None, **sampling):
        self.asked.append((prompt, system, sampling))
        return self.answer


class Tests(unittest.TestCase):
    def test_label_is_decided_from_the_checklist(self):
        judge = Judge('```json\n' + json.dumps(filled('CONTRADICTED')) + '\n```')
        verdict = check_sentence(judge, 'Beta-blockers make the heart beat faster.', FACTS)
        self.assertEqual((verdict['label'], verdict['fact'], verdict['checklist']['conflicts']), ('CONTRADICTED', 1, True))
        prompt, system, sampling = judge.asked[0]
        self.assertIn('1. Beta blockers slow the heart rate', prompt)
        self.assertIn('fact_is_about', prompt)
        self.assertIn('ONLY by the numbered facts', system)
        self.assertEqual(sampling, {'temperature': 0})

    def test_a_group_fact_said_for_everyone_is_cut_even_if_the_model_says_same_people(self):
        self.assertEqual(decide(filled(fact_is_about='patients with HFrEF', sentence_is_about='everyone'), FACTS), 'NOT_SUPPORTED')
        self.assertEqual(decide(filled(fact_is_about='patients with HFrEF', sentence_is_about='people with HFrEF'), FACTS), 'SUPPORTED')
        self.assertEqual(decide(filled(fact_is_about='patients with heart failure'), FACTS), 'SUPPORTED')  # the videos' audience
        self.assertEqual(decide(filled(fact_is_about='patients with HFrEF', sentence_is_about='people with HFrEF',
                                       same_people='false'), FACTS), 'NOT_SUPPORTED')

    def test_same_people_counts_only_when_the_fact_names_a_group(self):
        self.assertEqual(decide(filled(same_people=False), FACTS), 'SUPPORTED')  # the fact is about everyone
        self.assertEqual(decide(filled(fact_is_about=None, same_people=False), FACTS), 'NOT_SUPPORTED')  # left empty: against it

    def test_anything_added_is_cut(self):
        self.assertEqual(decide(filled(added=['every morning']), FACTS), 'NOT_SUPPORTED')
        self.assertEqual(decide(filled(added='none'), FACTS), 'SUPPORTED')

    def test_no_claim_only_when_the_model_says_so(self):
        self.assertEqual(decide(filled('NO_CLAIM'), FACTS), 'NO_CLAIM')
        self.assertEqual(decide(filled(medical_statement='false'), FACTS), 'NO_CLAIM')
        self.assertEqual(decide(filled(medical_statement=None, states_it=None), FACTS), 'NOT_SUPPORTED')

    def test_no_fact_or_a_wrong_number_is_not_supported(self):
        self.assertEqual(decide(filled(fact=None), FACTS), 'NOT_SUPPORTED')
        self.assertEqual(decide(filled(fact=7), FACTS), 'NOT_SUPPORTED')

    def test_unreadable_checklist_is_an_error_not_a_pass(self):
        self.assertEqual(check_sentence(Judge('I think it is fine.'), 'They slow the heart.', FACTS)['label'], 'ERROR')

    def test_empty_answer_is_an_error_not_a_crash(self):
        class Empty(Judge):
            def generate(self, prompt, system=None, **sampling):
                raise ValueError('Helper model returned an empty response')  # as ChatLLM does
        self.assertEqual(check_sentence(Empty(None), 'It can reduce hospital visits and death.', FACTS)['label'], 'ERROR')

    def test_verdict_inside_fences_and_spaced_label(self):
        self.assertEqual(parse('```json\n{"label": "not supported", "fact": null}\n```')['label'], 'NOT_SUPPORTED')

    def test_thinking_before_the_answer_is_ignored(self):
        answer = '<think>Fact 1 {maybe} says slow...</think>\n{"label": "CONTRADICTED", "fact": 1}'
        self.assertEqual(parse(answer)['label'], 'CONTRADICTED')

    def test_no_claim_is_a_label(self):
        self.assertEqual(parse('{"label": "no claim", "fact": null}')['label'], 'NO_CLAIM')

    def test_unreadable_answer_is_an_error_not_a_pass(self):
        self.assertEqual(parse('I think it is fine.')['label'], 'ERROR')
        self.assertEqual(parse('{"label": "MAYBE"}')['label'], 'ERROR')

    def test_batch_checks_all_sentences_in_one_call(self):
        judge = Judge('{"verdicts": [{"sentence": 2, "label": "CONTRADICTED", "fact": 1},'
                      ' {"sentence": 1, "label": "SUPPORTED", "fact": 1}]}')
        verdicts = check_sentences(judge, ['They slow the heart.', 'They speed it up.', 'They cure HF.'], FACTS)
        self.assertEqual([v['label'] for v in verdicts], ['SUPPORTED', 'CONTRADICTED', 'ERROR'])  # 3 got no verdict
        self.assertEqual(len(judge.asked), 1)
        self.assertIn('3. They cure HF.', judge.asked[0][0])

    def test_no_facts_means_not_supported_without_asking(self):
        judge = Judge('{"label": "SUPPORTED"}')
        self.assertEqual(check_sentence(judge, 'In Germany, call 112.', [])['label'], 'NOT_SUPPORTED')
        self.assertEqual(judge.asked, [])


if __name__ == '__main__':
    unittest.main()
