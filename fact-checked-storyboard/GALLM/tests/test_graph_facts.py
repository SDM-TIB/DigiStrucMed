import re
import unittest

from GALLM import graph_facts as gf

C = gf.CONCEPT
F = 'http://digistructmed.org/digilearn/resource/fact/'


class FakeKG:
    """A tiny graph that answers the module's fixed queries the way GraphDB would."""

    def __init__(self, claims, broader, statements, names, readings=None):
        self.claims, self.broader, self.statements, self.names = claims, broader, statements, names
        self.readings = readings or {}
        self.asked = []

    @staticmethod
    def ids(query):
        return re.findall(r'<[^>]*/(?:concept|fact)/([^>]+)>', query)

    def __call__(self, query):
        self.asked.append(query)
        ids = set(self.ids(query))
        if 'dl:RecommendationReading' in query:
            rows = []
            for reading, (fact, advice, operation, roles) in self.readings.items():
                if ids & {c for cs in roles.values() for c in cs}:
                    for role, cs in roles.items():
                        rows += [{'reading': 'r/' + reading, 'fact': F + fact, 'role': gf.DL + role, 'concept': C + c,
                                  'advice': advice, 'operation': operation} for c in cs]
            return rows
        if 'dl:broader ?parent' in query:
            return [{'c': C + c, 'parent': C + p} for c in sorted(ids) for p in self.broader.get(c, [])]
        if '?part ?concept' in query:
            rows = []
            for claim, (kind, fact, parts) in self.claims.items():
                if ids & {c for cs in parts.values() for c in cs}:
                    for part, cs in parts.items():
                        rows += [{'claim': 'x/' + claim, 'type': gf.DL + kind, 'fact': F + fact, 'part': gf.DL + part,
                                  'concept': C + c} for c in cs]
            return rows
        if 'dl:text ?text' in query:
            rows = []
            for fact in sorted(ids):
                text, rec, grade, page = self.statements[fact]
                row = {'fact': F + fact, 'text': text, 'rec': 'true' if rec else 'false', 'title': 'ESC HF', 'year': '2023'}
                if grade:
                    row['grade'] = grade
                if page:
                    row['page'] = page
                rows.append(row)
            return rows
        if 'dl:name ?name' in query:
            return [{'c': C + c, 'name': self.names[c]} for c in sorted(ids) if c in self.names]
        if 'rdfs:label' in query:
            return [{'x': gf.DL + 'TreatmentClaim', 'label': 'treatment claim'}]
        if 'COUNT(DISTINCT ?claim)' in query:
            return [{'n': '100'}]
        raise AssertionError('unexpected query: ' + query[:80])


def kg():
    claims = {
        'c1': ('TreatmentClaim', 'f1', {'treatmentConcept': {'DAPA'}, 'conditionConcept': {'HFREF'}}),
        'c2': ('TreatmentClaim', 'f2', {'treatmentConcept': {'DAPA'}, 'conditionConcept': {'T2D'}}),
        'c3': ('TreatmentClaim', 'f3', {'treatmentConcept': {'OTHER'}, 'conditionConcept': {'HFREF'}}),
        'c4': ('TreatmentClaim', 'f4', {'treatmentConcept': {'BETA'}, 'conditionConcept': {'GRAND'}}),
    }
    broader = {'ORGANIZER': ['DEVICE'], 'DEVICE': ['GRAND'], 'LONG': ['L1'], 'L1': ['L2'], 'L2': ['L3'], 'L3': ['L4']}
    statements = {'f1': ('Dapagliflozin is recommended in HFrEF.', True, 'I', '69'),
                  'f2': ('Dapagliflozin lowers glucose in T2D.', False, '', '12'),
                  'f3': ('Other drug treats HFrEF.', True, '', '70'),
                  'f4': ('Beta blockers help devices.', False, '', '5'),
                  'f5': ('Smoking cessation is obligatory.', True, 'I', '30')}
    names = {'DAPA': 'dapagliflozin', 'HFREF': 'HFrEF', 'T2D': 'type 2 diabetes', 'OTHER': 'other drug', 'BETA': 'beta blocker',
             'ORGANIZER': 'pill organizer', 'DEVICE': 'device', 'GRAND': 'grand concept', 'QUIT': 'Cessation of smoking',
             'SMOKER': 'Smoker'}
    readings = {'r1': ('f5', 'do', 'COUNSEL', {'hasTarget': {'QUIT'}, 'hasPopulation': {'SMOKER'}})}
    return FakeKG(claims, broader, statements, names, readings)


class Norm(unittest.TestCase):
    def test_american_spelling_and_singular(self):
        self.assertEqual(gf.norm('Ischaemic hearts'), ('ischemic', 'heart'))
        self.assertEqual(gf.norm('ARNIs'), ('arni',))

    def test_ruler_task_opening_is_not_the_topic(self):
        words = gf.topic_words('Create a brief patient-education scene that explains SGLT2 inhibitors.')
        self.assertEqual(words, ('sglt2', 'inhibitor'))


class Names(unittest.TestCase):
    def test_names_come_from_the_kg_with_weights(self):
        rows = {'name': [{'c': C + 'C1', 's': 'Heart failure'}],
                'words': [{'c': C + 'C1', 's': 'heart failure', 'n': '40'}, {'c': C + 'C2', 's': 'heart failure', 'n': '2'}],
                'abbreviation': [{'c': C + 'C3', 's': 'HFrEF', 'full': 'heart failure with reduced ejection fraction'}]}
        query = lambda q: next(v for k, v in rows.items() if gf.NAME_QUERIES[k] == q)
        names = gf.build_names(query)
        self.assertEqual(names['heart failure'], {'C1': 41, 'C2': 2})
        self.assertEqual(names['hfref'], {'C3': 1})
        self.assertEqual(names['heart failure with reduced ejection fraction'], {'C3': 1})


class FindConcepts(unittest.TestCase):
    names = {'heart failure': {'C1': 5}, 'heart': {'C2': 3}, 'sglt2 inhibitor': {'C3': 4}, 'patient': {'C9': 9},
             'patient education': {'C8': 9}, 'valve': {'C4': 1, 'C5': 7}}

    def test_longest_name_first_and_generic_words_skipped(self):
        found = gf.find_concepts('Create a brief patient-education scene that explains SGLT2 inhibitors in heart failure '
                                 'for patients', self.names)
        self.assertEqual([f['cui'] for f in found], ['C3', 'C1'])
        self.assertTrue(all(f['method'] == 'name' for f in found))

    def test_the_concept_the_guidelines_use_most_wins(self):
        self.assertEqual(gf.find_concepts('valve', self.names)[0]['cui'], 'C5')

    def test_a_generic_word_alone_is_not_a_concept(self):
        self.assertEqual(gf.find_concepts('how the heart works', self.names), [])

    def test_a_short_word_is_an_abbreviation_only_in_capitals(self):
        names = {'tip': {'TIPS': 3}, 'icd': {'ICD': 9}, 'lead': {'METAL': 2}}
        self.assertEqual(gf.find_concepts('practical tips for quitting', names), [])
        self.assertEqual(gf.find_concepts('this can lead to heart failure', names), [])
        self.assertEqual([f['cui'] for f in gf.find_concepts('people with an ICD or two ICDs', names)], ['ICD'])
        self.assertEqual([f['cui'] for f in gf.find_concepts('what an S-ICD or EV-ICD is', names)], ['ICD'])

    def test_rejected_and_short_words_never_reach_the_meaning_lookup(self):
        seen = []
        similar = lambda phrases: seen.extend(phrases) or []
        gf.find_concepts('practical tips so the leads last', {'tip': {'TIPS': 3}, 'lead': {'METAL': 2}}, similar)
        self.assertNotIn('tip', seen)
        self.assertNotIn('lead', seen)

    def test_leftover_words_go_to_the_meaning_lookup(self):
        seen = []
        similar = lambda phrases: seen.extend(phrases) or []
        gf.find_concepts('pill organizer for heart failure', self.names, similar)
        self.assertIn('pill organizer', seen)
        self.assertNotIn('heart', ' '.join(seen))

    def test_meaning_lookup_keeps_only_near_identical_names(self):
        import numpy as np

        class Encoder:
            table = {'ischemia': [1, 0], 'pump': [0.6, 0.8], 'ischaemia x': [0.99, 0.141]}

            def __call__(self, texts):
                return np.array([self.table[t] for t in texts], dtype=float)

        names = {'ischemia': {'C7': 1}, 'pump': {'C6': 1}}
        lookup = gf.MeaningLookup(names, encoder=Encoder(), vectors=np.array([[1, 0], [0.6, 0.8]], dtype=float))
        found = lookup(['ischaemia x'])
        self.assertEqual([(f['cui'], f['method']) for f in found], [('C7', 'similar')])


class StopRule(unittest.TestCase):
    def test_a_concept_with_claims_stays(self):
        reached, claims = gf.walk(['DAPA'], kg())
        self.assertEqual(reached['DAPA'], [('DAPA', ['DAPA'])])
        self.assertEqual(len(claims), 2)

    def test_steps_out_along_broader_and_stops_at_first_claims(self):
        reached, claims = gf.walk(['ORGANIZER'], kg())
        self.assertEqual(reached['ORGANIZER'], [('GRAND', ['ORGANIZER', 'DEVICE', 'GRAND'])])
        self.assertEqual(set(claims), {'x/c4'})

    def test_gives_up_after_the_ceiling(self):
        fake = kg()
        reached, claims = gf.walk(['LONG'], fake)
        self.assertEqual(reached['LONG'], [])
        self.assertFalse(any('L4' in q for q in fake.asked))

    def test_no_parent_no_claims(self):
        reached, _ = gf.walk(['NOTHING'], kg())
        self.assertEqual(reached['NOTHING'], [])


class Choose(unittest.TestCase):
    def facts(self, cuis):
        fake = kg()
        reached, claims = gf.walk(cuis, fake)
        statements = gf.statements_of({c['fact'] for c in claims.values()}, fake)
        return gf.facts_of(reached, claims, statements)

    def test_joining_fact_first_then_recommendation_before_explanation(self):
        facts = self.facts(['DAPA', 'HFREF'])
        chosen = [k.rsplit('/', 1)[-1] for k, _ in gf.choose(facts)]
        self.assertEqual(chosen, ['f1', 'f3', 'f2'])

    def test_never_more_than_the_cap(self):
        facts = self.facts(['DAPA', 'HFREF'])
        self.assertEqual(len(gf.choose(facts, most=2)), 2)

    def test_a_rare_concept_outweighs_a_common_one(self):
        reached = {'RARE': [('RARE', ['RARE'])], 'COMMON': [('COMMON', ['COMMON'])]}
        claims = {f'x/{i}': {'type': 'TreatmentClaim', 'fact': f'f{i}', 'parts': {'conditionConcept': {'COMMON'}}}
                  for i in range(50)}
        claims['x/r'] = {'type': 'TreatmentClaim', 'fact': 'fr', 'parts': {'treatmentConcept': {'RARE'}}}
        weight = gf.specificity(reached, claims, 1000)
        self.assertGreater(weight['RARE'], weight['COMMON'])
        statements = {f: {'text': 't', 'recommendation': True, 'classes': {'I'} if f != 'fr' else set(), 'levels': set(),
                          'pages': set(), 'guideline': 'g', 'year': ''} for f in [f'f{i}' for i in range(50)] + ['fr']}
        facts = gf.facts_of(reached, claims, statements, weight)
        self.assertEqual(gf.choose(facts, most=1)[0][0], 'fr')  # a graded fact about the common concept comes after it
        self.assertEqual([k for k, _ in gf.choose(facts, weight=weight)], ['fr'])  # and off-topic ones do not fill the places

    def test_direct_before_via_broader(self):
        facts = self.facts(['BETA', 'ORGANIZER'])
        self.assertEqual(facts[F + 'f4']['hops'], 0)


ARNI = ('Recommendations: Sacubitril/valsartan is recommended as a replacement for an ACE-I in patients with HFrEF to reduce '
        'the risk of HF hospitalization and death.')
ARNI_ROW = ('Sacubitril/valsartan is recommended as a replacement for an ACE inhibitor to reduce the risk of HF hospitalization '
            'and death in patients with HFrEF.707,730; I; B')


class WordForms(unittest.TestCase):
    names = {'driving': {'DRIVE': 3}, 'implantable defibrillator': {'ICD': 1}, 'icd': {'ICD': 5}, 'care': {'CARE': 1}}

    def test_drive_finds_the_driving_concept(self):
        found = gf.find_concepts('When can I drive again after getting an ICD?', self.names)
        self.assertEqual({f['cui'] for f in found}, {'DRIVE', 'ICD'})

    def test_a_word_the_kg_names_is_not_changed(self):
        self.assertEqual(gf.word_form('care', {'care': {}, 'caring': {}}), 'care')

    def test_a_word_with_no_known_form_stays(self):
        self.assertEqual(gf.word_form('again', self.names), 'again')


class NearCopies(unittest.TestCase):
    """Real pairs from the ruler topics of 2026-11-18."""

    def test_the_same_recommendation_from_a_table_row_is_a_copy(self):
        self.assertTrue(gf.near_copy(ARNI, ARNI_ROW))

    def test_another_medicine_in_the_same_sentence_is_not_a_copy(self):
        self.assertFalse(gf.near_copy(
            'Recommendations: In patients with T2DM and CKD, finerenone is recommended to reduce the risk of HF hospitalization.',
            'Recommendations: In patients with T2DM and CKD, SGLT2 inhibitors are recommended to reduce the risk of HF hospitalization.'))

    def test_not_recommended_is_never_a_copy_of_recommended(self):
        self.assertFalse(gf.near_copy('Driving is recommended after one week.', 'Driving is not recommended after one week.'))

    def test_a_narrower_group_is_not_a_copy_of_the_general_advice(self):
        self.assertFalse(gf.near_copy('Smoking cessation is recommended.',
                                      'Smoking cessation is recommended in all patients with LEAD.'))

    def test_a_copy_gives_its_place_to_the_next_fact(self):
        statement = lambda text, grade: {'text': text, 'recommendation': True, 'classes': {grade}, 'levels': set(),
                                         'pages': set(), 'guideline': 'g', 'year': ''}
        texts = {'a': (ARNI, 'I'), 'b': (ARNI_ROW, 'I'), 'c': ('A low starting dose may be used.', 'IIb')}
        facts = {k: {'statement': statement(*v), 'claims': [], 'joins': {'X'}, 'hops': 0, 'paths': {}, 'weight': 1.0, 'top': 1.0}
                 for k, v in texts.items()}
        self.assertEqual([k for k, _ in gf.choose(facts, most=2)], ['a', 'c'])


class Run(unittest.TestCase):
    def test_end_to_end_on_the_fake_graph(self):
        names = {'dapagliflozin': {'DAPA': 3}, 'hfref': {'HFREF': 2}, 'pill organizer': {'ORGANIZER': 1}}
        record = gf.run('Explain dapagliflozin in HFrEF and a pill organizer', names, query=kg())
        self.assertEqual([c['cui'] for c in record['concepts']], ['DAPA', 'HFREF', 'ORGANIZER'])
        first = record['facts'][0]
        self.assertTrue(first['fact'].endswith('f1'))
        self.assertIn('(dapagliflozin, treatment claim, HFrEF)', first['text'])
        self.assertIn('recommendation, Class I', first['text'])
        self.assertIn('p. 69', first['text'])
        via = [f for f in record['facts'] if f['hops'] > 0]
        self.assertTrue(via and 'via pill organizer -> broader -> device -> broader -> grand concept' in via[0]['text'])
        self.assertLessEqual(len(record['facts']), gf.MOST_FACTS)

    def test_lifestyle_advice_comes_from_the_readings(self):
        names = {'quitting smoking': {'QUIT': 4}}
        record = gf.run('Give practical tips for quitting smoking', names, query=kg())
        self.assertEqual(len(record['facts']), 1)
        self.assertIn('(do counsel, Cessation of smoking; population: Smoker)', record['facts'][0]['text'])
        self.assertIn('recommendation, Class I', record['facts'][0]['text'])


if __name__ == '__main__':
    unittest.main()
