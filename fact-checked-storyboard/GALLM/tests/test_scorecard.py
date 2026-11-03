import json
import tempfile
import unittest
from pathlib import Path

from GAAPO.data import Example, validate_splits
from GAAPO.storyboard import StoryboardEvaluator
from GALLM.scorecard import (SCORECARD, WEIGHTS, RulerExample, ScorecardEvaluator, key_point_coverage,
                             key_points_said, load_ruler, number_precision, quantities, readability, sentences,
                             sources, storyboard_text, too_long, unsupported_numbers)


def storyboard(narration, caption='Stay safe', title='Heart health'):
    return json.dumps({'title': title, 'scenes': [{
        'scene_number': 1, 'narration': narration, 'visual_description': 'A doctor talks to a patient.',
        'on_screen_text': caption}]})


SOURCES_112 = ['Create a brief patient-education scene about when to call 112.',
               'Sudden, severe shortness of breath.', 'Do not hesitate to call 112.']

# Anchors from ruler example 043 (when to call 112), first three key points
EXAMPLE_112 = {'anchors': [['breath*'],
                           ['chest pain', 'chest tight*', 'tight chest', 'pressure in the chest'],
                           ['dizz*', 'faint*', 'racing', 'irregular', 'pass* out']]}
RULER_112 = {'request': SOURCES_112[0], 'key_points': ['Sudden, severe shortness of breath.', 'Chest pain or tightness.',
                                                       'Dizziness, fainting or a racing heart.'], **EXAMPLE_112}


class Replay:
    """Stands in for the 1B model: always returns the same output."""
    def __init__(self, output):
        self.output = output

    def generate(self, prompt, request):
        return self.output


def combined_score(output):
    example = RulerExample('ruler-043', RULER_112['request'], None, RULER_112)
    return StoryboardEvaluator(Replay(output), SCORECARD, WEIGHTS).assess('any prompt', example)[1]


class Tests(unittest.TestCase):
    def test_correct_storyboard_keeps_full_marks(self):
        good = storyboard('Call 112 at once if you cannot breathe.', 'Call 112 now', 'When to call 112')
        self.assertEqual(number_precision(good, SOURCES_112), 1.0)

    def test_invented_waiting_time_is_caught(self):
        bad = storyboard('Wait three days before calling 112 if you cannot breathe.', 'Call 112 now')
        self.assertEqual(unsupported_numbers(bad, SOURCES_112), [(3, 'day')])
        self.assertEqual(number_precision(bad, SOURCES_112), 0.5)

    def test_unit_must_match(self):
        sources = ['No driving for at least 3 months.']
        self.assertEqual(number_precision(storyboard('Do not drive for 3 months.'), sources), 1.0)
        self.assertEqual(unsupported_numbers(storyboard('Do not drive for 3 weeks.'), sources), [(3, 'week')])

    def test_range_allows_both_ends(self):
        sources = ['Men at most 10–20 g alcohol per day, women no more than 10 g.']
        self.assertEqual(number_precision(storyboard('Men: up to 20 grams a day.'), sources), 1.0)
        self.assertEqual(number_precision(storyboard('Men: up to 30 g a day.'), sources), 0.0)

    def test_common_words_are_not_numbers(self):
        self.assertEqual(quantities('One of the signs; symptoms tend to often get worse for someone.'), set())

    def test_no_numbers_means_nothing_wrong(self):
        self.assertEqual(number_precision(storyboard('Rest and breathe slowly.'), []), 1.0)

    def test_plain_text_answers_are_still_read(self):
        self.assertEqual(unsupported_numbers('Scene: wait 2 hours, then call 112.', SOURCES_112), [(2, 'hour')])

    def test_allowed_facts_count_as_sources(self):
        example = {'request': 'Create a brief patient-education scene about acute heart failure.',
                   'key_points': ['Fluid builds up in the lungs.'],
                   'allowed_facts': [{'id': '041-b', 'text': 'See a doctor urgently if you gain more than 2 kg in 3 days.'}]}
        weight = storyboard('See a doctor if you gain 2 kg in 3 days.')
        self.assertEqual(number_precision(weight, sources(example)), 1.0)
        self.assertEqual(number_precision(weight, sources({**example, 'allowed_facts': []})), 0.0)

    def test_broken_output_is_read_whole_without_scene_numbers(self):
        # Real 1B pattern: a broken first block with the content, then an empty valid one
        broken = '{"title":"Pills", "-scenes": {"scene_number ":2, "narrATION":"Take them daily."}} {"title":"","scenes":{}}'
        self.assertIn('daily', storyboard_text(broken))
        self.assertEqual(number_precision(broken, []), 1.0)

    def test_non_breaking_hyphen_range(self):
        self.assertEqual(quantities('10‑20 g'), {(10, 'g'), (20, 'g')})

    def test_key_point_said_when_any_anchor_appears(self):
        good = storyboard('Call 112 at once if you suddenly cannot breathe or have chest pain at rest.')
        self.assertEqual(key_points_said(good, EXAMPLE_112), [True, True, False])
        self.assertAlmostEqual(key_point_coverage(good, EXAMPLE_112), 2 / 3)

    def test_wrong_topic_says_nothing(self):
        driving = storyboard('After getting an ICD, do not drive for at least two weeks.', 'No driving yet')
        self.assertEqual(key_point_coverage(driving, EXAMPLE_112), 0.0)

    def test_star_endings_plurals_and_number_words(self):
        example = {'anchors': [['breath*'], ['ventricle'], ['3 month*'], ['2 lead*']]}
        text = 'Breathlessness, both ventricles, three months, a two‑lead device.'
        self.assertEqual(key_points_said(text, example), [True, True, True, True])

    def test_anchors_match_whole_words_only(self):
        example = {'anchors': [['fail'], ['press'], ['organ']]}
        self.assertEqual(key_points_said('Heart failure, blood pressure, a pill organizer.', example),
                         [False, False, False])

    def test_short_storyboard_is_easy_to_read(self):
        self.assertEqual(readability(storyboard('Rest and breathe slowly.')), 1.0)

    def test_field_over_mahsas_limit_is_caught(self):
        caption = 'Weigh yourself every morning and write the number down'
        self.assertEqual(too_long(storyboard('Rest.', caption)), [(caption, 9, 8)])
        self.assertEqual(readability(storyboard('Rest.', caption)), 7 / 8)  # 4 fields, 4 sentences

    def test_long_sentence_is_caught_inside_the_field_limit(self):
        one = ('If your ankles swell, you gain weight quickly, you feel short of breath at night, '
               'or you feel very tired, please call your doctor today.')
        self.assertEqual(too_long(storyboard(one)), [(one, 25, 20)])
        short = 'Do your ankles swell? Do you gain weight fast? Are you short of breath at night? Call your doctor today.'
        self.assertEqual(readability(storyboard(short)), 1.0)

    def test_symptom_list_in_broken_output_is_caught(self):
        # Real 1B pattern (042): no clean JSON, so fields are unknown and only sentences are judged
        broken = ('{"title":"Warning signs", "-scenes": {"narrATION":Heart rate goes up,\\nsymptoms such as '
                  'dizziness, confusion, fatigue, shortness of breath, chest pain, nausea, vomiting, loss of '
                  'appetite, weight gain, swelling, cramps, weakness, numbness", '
                  '"on_screen_TEXT":Weigh yourself every morning and write the number down}')
        self.assertEqual([words for _, words, _ in too_long(broken)], [22])
        self.assertEqual(readability(broken), 3 / 4)

    def test_stray_braces_and_quotes_are_not_sentences(self):
        self.assertEqual(sentences('{ } "", \\n }'), [])

    def test_combined_score_through_mahsas_evaluator(self):
        good = storyboard('Call 112 at once if you suddenly cannot breathe or have chest pain at rest.',
                          'Call 112 now', 'When to call 112')
        # Full marks on format, fields, numbers and reading; 2 of 3 key points said
        self.assertAlmostEqual(combined_score(good), (4 + 2 / 3) / 5)

    def test_empty_answer_scores_zero(self):
        # It used to keep "no wrong numbers" and "nothing too long": 2 of 5, halved = 0.2
        self.assertEqual(combined_score(''), 0.0)

    def test_refusals_score_zero(self):
        # Real rerun-02 refusals (call 905) and a grounding-pilot one buried after empty JSON
        self.assertEqual(combined_score("I can't answer this question."), 0.0)
        self.assertEqual(combined_score('{"title": Short title",scenes":[{" ,narration":"","visualDescription":""}]'
                                        '.I cannot create content that suggests taking medication based on unproven '
                                        'claims. Can I help you with something else?'), 0.0)

    def test_storyboard_that_says_cannot_still_counts(self):
        good = storyboard("You can't feel the device. Call 112 if you cannot breathe.", 'Call 112 now', 'When to call 112')
        self.assertGreater(combined_score(good), 0.5)

    def test_training_example_skips_key_points_and_checks_numbers_against_its_reference(self):
        reference = json.loads(storyboard('See your doctor if you gain 2 kg in 3 days.'))
        example = Example('tr1', 'Create a brief patient-education scene about daily weighing.', reference)
        scored = []
        evaluator = ScorecardEvaluator(Replay(storyboard('Gain 5 kg in 3 days? See your doctor.')),
                                       result_callback=lambda r: r['status'] == 'scored' and scored.append(r))
        self.assertAlmostEqual(evaluator.assess('any prompt', example)[1], (1 + 1 + 0.5 + 1) / 4)  # 5 kg is made up
        self.assertNotIn('key_point_coverage', scored[0]['metrics'])

    def test_ruler_examples_pass_gaapos_split_check(self):
        rows = [{**RULER_112, 'id': f'ruler-{i}', 'request': f'Create a brief scene about topic {i}.', 'split': split}
                for i, split in enumerate(['validation', 'test'])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ruler.json'
            path.write_text(json.dumps(rows), encoding='utf-8')
            validation, test = load_ruler(path, 'validation'), load_ruler(path, 'test')
        validate_splits(validation, test)
        self.assertEqual(validation[0].output, {'key_points': RULER_112['key_points']})


if __name__ == '__main__':
    unittest.main()
