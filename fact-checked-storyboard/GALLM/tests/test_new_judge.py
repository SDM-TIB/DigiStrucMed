"""The code around the new judge, with made-up model answers. No model is called."""
from GALLM.new_judge import judge_line, quote_is_real

FACTS = ['Beta blockers slow the heart rate, lower blood pressure.', 'Another fact.']
SUPPORTED = '{"verdict": "SUPPORTED", "fact": 1, "quote": "slow the heart rate", "reason": "same"}'
EMPTY, NAMED = '{"not_in_facts": ""}', '{"not_in_facts": "every week"}'


class Model:
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), 0

    def generate(self, prompt, system=None, temperature=0):
        self.calls += 1
        return self.answers.pop(0)


def test_quote_must_really_be_in_the_fact():
    assert quote_is_real('Slow the heart rate', FACTS[0])
    assert quote_is_real('Beta blockers ... lower blood pressure', FACTS[0])
    assert not quote_is_real('slows the heart', FACTS[0])
    assert not quote_is_real('Beta blockers ... cure the heart', FACTS[0])
    assert not quote_is_real('', FACTS[0])


def test_a_made_up_quote_is_cut_without_asking_further():
    model = Model('{"verdict": "SUPPORTED", "fact": 1, "quote": "cures everyone", "reason": "same"}')
    verdict = judge_line(model, 'x', FACTS)
    assert verdict['label'] == 'NOT_SUPPORTED' and verdict['decided_by'] == 'quote check' and model.calls == 1


def test_a_wrong_fact_number_is_cut():
    model = Model('{"verdict": "SUPPORTED", "fact": 9, "quote": "slow the heart rate", "reason": "same"}')
    assert judge_line(model, 'x', FACTS)['label'] == 'NOT_SUPPORTED'


def test_plain_talk_skips_the_second_question():
    model = Model('{"verdict": "PLAIN", "fact": null, "quote": "", "reason": "a title"}')
    assert judge_line(model, 'x', FACTS)['label'] == 'NO_CLAIM' and model.calls == 1


def test_one_objection_of_three_does_not_cut():
    assert judge_line(Model(SUPPORTED, NAMED, EMPTY, EMPTY), 'x', FACTS)['label'] == 'SUPPORTED'


def test_two_objections_of_three_cut():
    verdict = judge_line(Model(SUPPORTED, NAMED, EMPTY, NAMED), 'x', FACTS)
    assert verdict['label'] == 'NOT_SUPPORTED' and verdict['decided_by'] == 'second question'


def test_only_supported_and_plain_can_stay():
    for word, label in (('PARTLY', 'NOT_SUPPORTED'), ('NOT', 'NOT_SUPPORTED'), ('AGAINST', 'CONTRADICTED'), ('ODD', 'NOT_SUPPORTED')):
        answer = '{"verdict": "' + word + '", "fact": 1, "quote": "", "reason": "r"}'
        assert judge_line(Model(answer), 'x', FACTS)['label'] == label


def test_an_unreadable_answer_is_not_kept():
    assert judge_line(Model('no json here'), 'x', FACTS)['label'] == 'ERROR'


def test_the_pipeline_gives_the_judge_the_line_before():
    from GALLM.grounded_pipeline import check

    class Seeing(Model):
        def generate(self, prompt, system=None, temperature=0):
            self.calls += 1
            self.seen = getattr(self, 'seen', []) + [prompt]
            return '{"verdict": "PLAIN", "fact": null, "quote": "", "reason": "plain"}'

    board = {'scenes': [{'narration': 'First line. Second line.', 'on_screen_text': 'Title'}]}
    facts = [{'text': 'A fact.', 'headers': [], 'group': None, 'against': None}]
    model = Seeing()
    verdicts = check(model, board, facts, topic='a topic')
    assert [v['label'] for v in verdicts] == ['NO_CLAIM'] * 3
    assert 'Line before: First line.' in model.seen[1] and 'Line before: Second line.' in model.seen[2]


def test_the_old_checker_is_still_the_way_back():
    from GALLM.grounded_pipeline import check
    board = {'scenes': [{'narration': 'First line.'}]}
    facts = [{'text': 'A fact.', 'headers': [], 'group': None, 'against': None}]
    model = Model('{"kind": "greeting", "reason": "hello"}')
    assert check(model, board, facts, topic='a topic', judge='today')[0]['label'] == 'NO_CLAIM'
