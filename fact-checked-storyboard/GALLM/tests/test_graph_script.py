import json
import re
import unittest

from GALLM import graph_script as gs

DRIVE_4W = 'Patients receiving ICDs for primary prevention should be restricted from private driving for 4 weeks after the implantation of the device.'
DRIVE_3M = 'Patients receiving ICDs for secondary prevention should be restricted from private driving for 3 months after the index arrhythmia.'
REFUSE = 'Therefore, patients refusing an ICD for primary prevention should have no driving restriction for private driving.'
MAYBE = 'A second system integrity check may be considered after 4 weeks before resumption of driving.'


def fact(n, statement, rec=True, classes=()):
    return {'fact': f'f{n}', 'text': f'(x) — recommendation — g\n    statement: "{statement}"', 'statement': statement,
            'recommendation': rec, 'classes': list(classes), 'levels': [], 'pages': [12], 'guideline': 'vijgen 2009'}


class Writer:
    """Answers the writing prompt with `first`, and the rewrite prompt with `repair`."""
    def __init__(self, first, repair=None):
        self.first, self.repair, self.prompts = first, repair, []

    def generate(self, prompt, system=None, **_):
        self.prompts.append(prompt)
        return json.dumps(self.repair if prompt.startswith('Some lines') else self.first)


class Judge:
    """Plays new_judge's model: PLAIN for lines with no fact, NOT for lines with a word in `bad`, else SUPPORTED with a real
    quote. Records every prompt, so a test can see which facts the judge was shown."""
    def __init__(self, bad=()):
        self.bad, self.prompts = bad, []

    def generate(self, prompt, system=None, **_):
        self.prompts.append(prompt)
        if 'not_in_facts' in prompt:
            return json.dumps({'not_in_facts': ''})
        line = re.search(r'Line \((?:spoken|on screen)\): (.*)\n', prompt).group(1)
        facts = re.search(r'Facts:\n(.*?)\nLine \(', prompt, re.S).group(1).strip()
        if not facts:
            return json.dumps({'verdict': 'PLAIN', 'fact': None, 'quote': '', 'reason': 'a greeting'})
        if any(word in line for word in self.bad):
            return json.dumps({'verdict': 'NOT', 'fact': None, 'quote': '', 'reason': 'not in the fact'})
        quote = ' '.join(facts.split('. ', 1)[1].split()[:6])
        return json.dumps({'verdict': 'SUPPORTED', 'fact': 1, 'quote': quote, 'reason': 'same claim'})


def answer(lines, skipped=()):
    return {'title': 'Driving with an ICD', 'opening': 'Hello, today we talk about driving with an ICD.',
            'lines': lines, 'skipped': list(skipped), 'closing': 'Ask your care team if you have questions.'}


class Slots(unittest.TestCase):
    def test_one_slot_per_fact_with_kind_source_and_caution(self):
        slots = gs.slots_of([fact(1, DRIVE_4W, classes=['I']), fact(2, MAYBE, rec=False)])
        self.assertEqual([s['n'] for s in slots], [1, 2])
        self.assertEqual((slots[0]['kind'], slots[0]['grade'], slots[0]['caution']), ('recommendation', 'Class I', False))
        self.assertEqual((slots[1]['kind'], slots[1]['caution']), ('explanation', True))
        self.assertEqual(slots[0]['source'], 'vijgen 2009; p. 12')

    def test_the_statement_is_read_from_the_fact_text_when_missing(self):
        old = fact(1, DRIVE_4W)
        del old['statement']
        self.assertEqual(gs.slots_of([old])[0]['statement'], DRIVE_4W)


class CodeChecks(unittest.TestCase):
    slot = gs.slots_of([fact(1, DRIVE_4W)])[0]
    maybe = gs.slots_of([fact(1, MAYBE)])[0]

    def test_a_faithful_line_passes(self):
        self.assertEqual(gs.code_checks({'narration': 'After an ICD for prevention, do not drive for four weeks.'}, self.slot), [])

    def test_a_number_the_fact_does_not_have_fails(self):
        problems = gs.code_checks({'narration': 'After your ICD, wait 6 weeks before driving.'}, self.slot)
        self.assertIn('Numbers not in the fact: 6.', problems)

    def test_one_as_a_word_is_no_number(self):
        self.assertEqual(gs.code_checks({'narration': 'One thing to know: no private driving for 4 weeks.'}, self.slot), [])

    def test_a_may_fact_said_as_a_must_fails(self):
        self.assertTrue(gs.code_checks({'narration': 'You need a second device check after 4 weeks.'}, self.maybe))
        self.assertEqual(gs.code_checks({'narration': 'A second device check may be done after 4 weeks.'}, self.maybe), [])

    def test_on_screen_words_must_come_from_the_line_or_fact(self):
        line = {'narration': 'Do not drive for 4 weeks.', 'on_screen_text': 'No driving for 4 weeks'}
        self.assertIsNone(gs.screen_problem(line, self.slot))
        self.assertIn('safe', gs.screen_problem({**line, 'on_screen_text': 'Safe driving tips'}, self.slot))


class Run(unittest.TestCase):
    part1 = {'concepts': [], 'facts': [fact(1, DRIVE_4W), fact(2, DRIVE_3M), fact(3, REFUSE)]}

    def test_kept_rewritten_and_skipped_lines_end_up_in_the_prompt_with_sources(self):
        writer = Writer(answer([
            {'fact': 1, 'narration': 'After an ICD for prevention, do not drive for 4 weeks.', 'on_screen_text': 'No driving for 4 weeks', 'visual': 'a car key'},
            {'fact': 2, 'narration': 'After a shock, do not drive for 6 months.', 'on_screen_text': 'Wait 6 months', 'visual': 'a calendar'}],
            skipped=[{'fact': 3, 'why': 'about people who refuse an ICD'}]),
            repair={'lines': [{'fact': 2, 'narration': 'After an ICD for secondary prevention, do not drive for 3 months.', 'on_screen_text': 'No driving for 3 months'}]})
        judge = Judge()
        record = gs.run('When can I drive after an ICD?', self.part1, writer, judge)
        self.assertEqual(record['status'], 'done')
        first, second = record['lines']
        self.assertEqual((first['status'], second['status']), ('kept', 'rewritten'))
        self.assertEqual(second['first_try']['verdict']['decided_by'], 'code check')  # "6" is not in the fact
        self.assertEqual(record['skipped'][0]['fact'], 3)
        prompt = record['video_prompt']
        for text in (gs.OPENING, first['narration'], second['narration'], gs.CLOSING):
            self.assertIn(f'Narration: "{text}"', prompt)
        self.assertNotIn('6 months', prompt)
        self.assertEqual([s['statement'] for s in record['sources']], [DRIVE_4W, DRIVE_3M])

    def test_the_judge_sees_only_the_lines_own_fact(self):
        writer = Writer(answer([{'fact': 1, 'narration': 'After an ICD for prevention, do not drive for 4 weeks.'},
                                {'fact': 2, 'narration': 'After a shock-causing rhythm, do not drive for 3 months.'}]))
        judge = Judge()
        gs.run('When can I drive after an ICD?', self.part1, writer, judge)
        for prompt in judge.prompts:
            self.assertFalse(DRIVE_4W in prompt and DRIVE_3M in prompt, 'a judge prompt held two facts')
        self.assertFalse(any(REFUSE in p for p in judge.prompts))

    def test_a_line_that_fails_twice_is_dropped(self):
        writer = Writer(answer([{'fact': 1, 'narration': 'After an ICD, driving is always unsafe for 4 weeks.'}]),
                        repair={'lines': [{'fact': 1, 'narration': 'Driving is always unsafe for 4 weeks.'}]})
        record = gs.run('When can I drive after an ICD?', self.part1, writer, Judge(bad=('always',)))
        self.assertEqual(record['lines'][0]['status'], 'dropped')
        self.assertIsNone(record['video_prompt'])  # nothing medical is left, so there is no prompt
        self.assertEqual(record['why_no_prompt'], 'No checked medical line is left to say.')

    def test_no_facts_means_no_writer_call(self):
        writer = Writer(answer([]))
        record = gs.run('What is the emergency number?', {'concepts': [], 'facts': []}, writer, Judge())
        self.assertEqual(record['status'], 'no_facts')
        self.assertEqual(writer.prompts, [])

    def test_never_more_lines_than_the_cap(self):
        facts = [fact(n, f'Rule {n}: do not drive for {n} weeks.') for n in range(1, 10)]
        writer = Writer(answer([{'fact': n, 'narration': f'Do not drive for {n} weeks.'} for n in range(1, 10)]))
        record = gs.run('When can I drive?', {'concepts': [], 'facts': facts}, writer, Judge())
        self.assertEqual([line['slot'] for line in record['lines']], list(range(1, gs.MOST_LINES + 1)))

    def test_a_line_for_a_fact_that_does_not_exist_is_ignored(self):
        writer = Writer(answer([{'fact': 9, 'narration': 'Something from nowhere.'},
                                {'fact': 1, 'narration': 'After an ICD for prevention, do not drive for 4 weeks.'}]))
        record = gs.run('When can I drive after an ICD?', self.part1, writer, Judge())
        self.assertEqual([line['slot'] for line in record['lines']], [1])


class Cache(unittest.TestCase):
    def test_a_repeated_question_costs_no_call(self):
        import tempfile
        from pathlib import Path

        class Model:
            model, tokens, calls = 'm', 0, 0

            def generate(self, prompt, system=None, **_):
                self.calls += 1
                self.tokens += 10
                return 'answer'
        with tempfile.TemporaryDirectory() as folder:
            model = Model()
            cached = gs.DiskCache(model, Path(folder, 'calls.jsonl'))
            cached.generate('p')
            cached.generate('p')
            self.assertEqual((model.calls, gs.used(cached)), (1, 10))
            self.assertEqual(gs.DiskCache(Model(), Path(folder, 'calls.jsonl')).generate('p'), 'answer')


class Abbreviations(unittest.TestCase):
    rows = [{'s': 'ALVD', 'full': 'asymptomatic LV dysfunction', 'n': '15'},
            {'s': 'VA', 'full': 'veno-arterial', 'n': '94'}, {'s': 'VA', 'full': 'ventricular arrhythmia', 'n': '16'},
            {'s': 'HFrEF', 'full': 'HF with reduced ejection fraction', 'n': '174'},
            {'s': 'HFrEF', 'full': 'heart failure with reduced ejection fraction', 'n': '81'},
            {'s': 'PA', 'full': 'physical activity', 'n': '20'}, {'s': 'PA', 'full': 'pulmonary artery', 'n': '1'}]

    def setUp(self):
        self.meanings = gs.graph_abbreviations(lambda query: self.rows)

    def test_meanings_come_from_the_graph_and_two_meanings_mean_none(self):
        self.assertEqual(self.meanings['ALVD'], 'asymptomatic LV dysfunction')
        self.assertIsNone(self.meanings['VA'])  # veno-arterial or ventricular arrhythmia
        self.assertEqual(self.meanings['HFrEF'], 'HF with reduced ejection fraction')  # spelling variants of one meaning
        self.assertEqual(self.meanings['PA'], 'physical activity')  # a rare second form is left out

    def test_the_judge_sees_the_graphs_meaning(self):
        statement = 'Smoking cessation is recommended in all patients including those with ALVD.'
        writer = Writer(answer([{'fact': 1, 'narration': 'Quitting smoking is recommended for everyone, even with aortic valve disease.'}]))
        judge = Judge()
        gs.run('Quitting smoking?', {'concepts': [], 'facts': [fact(1, statement)]}, writer, judge, meanings=self.meanings)
        self.assertTrue(any('ALVD means asymptomatic LV dysfunction' in p for p in judge.prompts))
        self.assertIn('ALVD means asymptomatic LV dysfunction', writer.prompts[0])

    def test_an_abbreviation_with_two_meanings_must_stay(self):
        slot = gs.slots_of([fact(1, 'In patients with VA due to coronary artery spasm, smoking cessation is indicated.')], self.meanings)[0]
        self.assertTrue(gs.code_checks({'narration': 'If you have heart rhythm problems from artery spasm, stop smoking.'}, slot))
        self.assertEqual(gs.code_checks({'narration': 'If you have VA from artery spasm, stop smoking.'}, slot), [])
