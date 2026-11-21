import datetime
import json
import unittest

from GALLM import make_video as mv
from GALLM import speech_check as sc
from GALLM import timeline as tl

RECORD = {'question': 'When can I drive again after getting an ICD?',
          'scenes': [{'narration': "Hello. Let's look at what the heart guidelines say about your question.", 'medical': False},
                     {'narration': 'You can drive again after about four weeks if you got an ICD for primary prevention.',
                      'on_screen_text': 'Drive after four weeks', 'medical': True},
                     {'narration': "Please check the details with your care team, and don't wait until things get worse.",
                      'medical': False}],
          'sources': [{'scene': 2, 'source': 'vijgen 2009 updated recommendations for driving by patients with implant; p. 8',
                       'kind': 'recommendation'}],
          'video_prompt': 'x'}
LABEL = 'This video was made with AI and uses a computer voice.'


def segments(storyboard=None):
    return tl.apply_storyboard(tl.segments_of(RECORD, LABEL, mv.readable_source), storyboard)


class Order(unittest.TestCase):
    def test_patient_asks_first_then_the_doctor_with_the_label_after_the_opening(self):
        segs = segments()
        self.assertEqual([(s['id'], s['speaker']) for s in segs],
                         [('q', 'patient'), ('s1', 'doctor'), ('label', 'doctor'), ('s2', 'doctor'), ('s3', 'doctor')])
        self.assertEqual(segs[0]['text'], RECORD['question'])
        self.assertEqual(segs[0]['patient'], 'asks the question')

    def test_the_words_are_exactly_the_checked_lines(self):
        texts = [s['text'] for s in segments()]
        for scene in RECORD['scenes']:
            self.assertIn(scene['narration'], texts)

    def test_the_storyboard_only_sets_the_four_visual_fields(self):
        board = [{'line': 2, 'shot': 'close-up on prop', 'doctor': 'points to the prop', 'patient': 'looks at the prop',
                  'prop': 'calendar', 'heading': False, 'says': 'You can drive after one day.'}]
        seg = next(s for s in segments(board) if s['id'] == 's2')
        self.assertEqual((seg['shot'], seg['doctor'], seg['patient'], seg['prop']),
                         ('close-up on prop', 'points to the prop', 'looks at the prop', 'calendar'))
        self.assertFalse(seg['show_heading'])
        self.assertNotIn('one day', json.dumps(segments(board)))


class Storyboard(unittest.TestCase):
    def test_the_code_storyboard_follows_the_rules(self):
        from GALLM import director as dr
        board = dr.code_storyboard(RECORD)
        self.assertEqual(dr.gate(board, dr.script_lines(RECORD)), [])
        line2 = board[1]
        self.assertIn(line2['prop'], dr.fitting_props(RECORD['scenes'][1]['narration']))
        self.assertIn(line2['doctor'], dr.SHOWS_PROP)
        self.assertEqual(board[-1]['patient'], 'smiles')
        smoking = {**RECORD, 'scenes': [RECORD['scenes'][0], {'narration': 'Stop smoking to protect your heart.', 'medical': True},
                                        RECORD['scenes'][2]]}
        board = dr.code_storyboard(smoking)
        self.assertNotEqual(board[1]['prop'], 'cigarette pack')
        self.assertEqual(dr.gate(board, dr.script_lines(smoking)), [])

    def test_held_objects_and_no_second_question(self):
        board = [{'line': 1, 'shot': 'medium', 'doctor': 'smiles warmly', 'patient': 'asks the question', 'prop': 'none'},
                 {'line': 2, 'shot': 'close-up on doctor', 'doctor': 'holds up the prop', 'patient': 'looks at the prop',
                  'prop': 'car keys', 'heading': True}]
        segs = {s['id']: s for s in segments(board)}
        self.assertEqual(segs['s1']['patient'], 'listens')        # the patient asked already, in their own segment
        self.assertEqual(segs['q']['patient'], 'asks the question')
        self.assertEqual((segs['s2']['prop_at'], segs['s1']['prop_at']), ('doctor-hand', 'desk'))


class Timing(unittest.TestCase):
    durations = {'q': 2.5, 's1': 3.8, 'label': 3.0, 's2': 5.6, 's3': 4.4}

    def test_same_inputs_give_the_same_timeline(self):
        a = tl.build(segments(), self.durations)
        b = tl.build(segments(), self.durations)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_segments_follow_each_other_with_pauses_and_reading_time(self):
        timeline = tl.build(segments(), self.durations)
        segs = timeline['segments']
        for before, after in zip(segs, segs[1:]):
            self.assertEqual(before['t1'], after['t0'])
        for s in segs:
            self.assertAlmostEqual(s['voice_at'], s['t0'] + tl.PAUSE_BEFORE, places=3)
            self.assertGreaterEqual(s['t1'] - s['t0'], tl.PAUSE_BEFORE + s['voice'] + tl.PAUSE_AFTER - 0.001)
            self.assertGreaterEqual(s['t1'] - s['t0'], tl.reading_time(s['text']) + 0.4 - 0.001)
        self.assertEqual(timeline['seconds'], segs[-1]['t1'])

    def test_the_sound_track_pieces_cover_it_in_order_one_line_each(self):
        timeline = tl.build(segments(), self.durations)
        pieces = mv.windows(timeline, rate=24000)
        self.assertEqual([p[0] for p in pieces], [s['id'] for s in timeline['segments']])
        for (_, a, b), (_, c, _) in zip(pieces, pieces[1:]):
            self.assertEqual(b, c)                         # no gap, no overlap
        for (_, a, b), seg in zip(pieces, timeline['segments']):
            self.assertLessEqual(a, int(seg['voice_at'] * 24000))
            self.assertGreaterEqual(b, int((seg['voice_at'] + seg['voice']) * 24000))   # the whole voice is inside

    def test_loudness_travels_with_its_segment(self):
        timeline = tl.build(segments(), self.durations, loudness={'q': [0.0, 0.5, 1.0]})
        self.assertEqual(timeline['segments'][0]['loudness'], [0.0, 0.5, 1.0])
        self.assertNotIn('loudness', timeline['segments'][1])

    def test_word_times_move_onto_the_film_clock(self):
        timeline = tl.build(segments(), self.durations, words={'q': [['When', 0.1, 0.3]]})
        q = timeline['segments'][0]
        self.assertEqual(q['words'], [['When', round(q['voice_at'] + 0.1, 3), round(q['voice_at'] + 0.3, 3)]])


class Sources(unittest.TestCase):
    def test_a_readable_source_line_with_the_year_and_no_warning_on_screen(self):
        line = mv.readable_source(RECORD['sources'][0])
        self.assertEqual(line, 'Source: Vijgen et al., EHRA consensus on driving with an ICD (Europace), 2009, p. 8 · guideline recommendation')
        self.assertNotIn('review', line)
        self.assertEqual(mv.readable_source({'source': 'UMLS definition of "Implantable defibrillator"', 'kind': 'definition'}),
                         'Definition of "Implantable defibrillator": UMLS (U.S. National Library of Medicine)')

    def test_the_reviewer_sees_the_age_warning(self):
        review = mv.source_review(RECORD, today=datetime.date(2026, 11, 22))
        self.assertEqual((review[0]['year'], review[0]['age_years']), (2009, 17))
        self.assertIn('STRONG WARNING', review[0]['warning'])
        recent = {'sources': [{'scene': 1, 'source': 'mcdonagh 2023 focused update; p. 3'}]}
        self.assertEqual(mv.source_review(recent, today=datetime.date(2026, 11, 22))[0]['warning'], '')
        older = {'sources': [{'scene': 1, 'source': 'mcdonagh 2021 esc heart failure; p. 3'}]}
        review = mv.source_review(older, today=datetime.date(2026, 11, 22))[0]
        self.assertIn('newer version', review['warning'])
        self.assertTrue(review['readable_name_missing'])


class SpokenNumbers(unittest.TestCase):
    norm = staticmethod(sc.lemo_asr_check().norm_en)

    def same(self, a, b):
        return sc.spoken_words(a, self.norm) == sc.spoken_words(b, self.norm)

    def test_numbers_match_however_whisper_writes_them(self):
        self.assertTrue(self.same('An ejection fraction of 35% or less.', 'An ejection fraction of thirty-five percent or less.'))
        self.assertTrue(self.same('Wait four weeks.', 'Wait 4 weeks.'))
        self.assertTrue(self.same('Take 1.5 tablets.', 'Take 1 point 5 tablets.'))
        self.assertTrue(self.same('About 1,000 people.', 'About 1000 people.'))

    def test_a_changed_number_or_word_still_fails(self):
        self.assertFalse(self.same('Wait four weeks.', 'Wait six weeks.'))
        self.assertFalse(self.same('35% or less', '25% or less'))
        self.assertFalse(self.same('You may drive.', 'You must drive.'))


class Tools(unittest.TestCase):
    def test_only_a_bad_caption_is_a_content_failure(self):
        self.assertEqual(mv.reading_status(0, 'OK  caption-q  at 0.00s'), mv.PASS)
        self.assertEqual(mv.reading_status(1, 'OK  caption-q ...\nBAD caption-s2 at 5.00s  need 7s  got 4s'), mv.CONTENT_FAIL)
        self.assertEqual(mv.reading_status(1, 'the headless browser for playwright-core 1.63.0 is not installed.'), mv.TOOL_ERROR)
        self.assertEqual(mv.reading_status(2, 'readcheck: NOT CHECKED'), mv.TOOL_ERROR)

    def test_the_browser_matching_the_renderer_is_found_on_disk(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            lemo, home = Path(folder, 'lemo'), Path(folder, 'home')
            (lemo / 'node_modules' / 'playwright-core').mkdir(parents=True)
            (lemo / 'node_modules' / 'playwright-core' / 'browsers.json').write_text(json.dumps(
                {'browsers': [{'name': 'chromium', 'revision': '1243'}, {'name': 'chromium-headless-shell', 'revision': '1243'}]}))
            self.assertIsNone(mv.headless_browser(lemo, environ={}, home=home))
            old = home / 'AppData' / 'Local' / 'ms-playwright' / 'chromium_headless_shell-1223' / 'chrome-headless-shell-win64'
            old.mkdir(parents=True)
            (old / 'chrome-headless-shell.exe').write_text('x')
            self.assertIsNone(mv.headless_browser(lemo, environ={}, home=home))   # another version is not used
            right = home / 'AppData' / 'Local' / 'ms-playwright' / 'chromium_headless_shell-1243' / 'chrome-headless-shell-win64'
            right.mkdir(parents=True)
            (right / 'chrome-headless-shell.exe').write_text('x')
            self.assertEqual(mv.headless_browser(lemo, environ={}, home=home), right / 'chrome-headless-shell.exe')
            copy = lemo / 'browsers' / 'chromium_headless_shell-1243' / 'chrome-headless-shell-win64'
            copy.mkdir(parents=True)
            (copy / 'chrome-headless-shell.exe').write_text('x')
            self.assertEqual(mv.headless_browser(lemo, environ={}, home=home), copy / 'chrome-headless-shell.exe')   # outside AppData first


class Voices(unittest.TestCase):
    def test_two_voices_and_slower_speech(self):
        lines = mv.lines_file(segments())
        self.assertEqual({l['id']: l['voice'] for l in lines}['q'], mv.VOICES['patient'])
        self.assertEqual({l['voice'] for l in lines if l['id'] != 'q'}, {mv.VOICES['doctor']})
        self.assertEqual({l['speed'] for l in lines}, {0.8})


if __name__ == '__main__':
    unittest.main()
