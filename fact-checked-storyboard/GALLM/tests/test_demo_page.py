import json
import tempfile
import unittest
from pathlib import Path

from GALLM import demo_page as dp
from GALLM import make_video as mv
from GALLM.tests.test_timeline import RECORD


class VideoPlan(unittest.TestCase):
    def test_the_page_shows_what_the_video_will_show_in_spoken_order(self):
        plan = dp.video_plan(RECORD)
        self.assertEqual(plan['problems'], [])
        self.assertEqual([s['id'] for s in plan['segments']], ['q', 's1', 'label', 's2', 's3'])
        self.assertEqual(plan['segments'][0]['text'], RECORD['question'])
        self.assertEqual(plan['segments'][2]['text'], mv.SPOKEN_LABEL)
        self.assertEqual(set(plan['segments'][0]), set(dp.SHOWN))

    def test_no_video_without_checked_lines(self):
        self.assertIsNone(dp.video_plan({**RECORD, 'scenes': []}))
        self.assertIsNone(dp.video_plan({**RECORD, 'video_prompt': ''}))


class VideoResult(unittest.TestCase):
    def test_a_stopped_run_says_which_check_stopped_it(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder, 'report.json')
            report.write_text(json.dumps({'checks': {'speech_lines': {'status': 'CONTENT_FAIL'}}}))
            output = ['1/6 voices ...', '2/6 speech check ...',
                      'STOPPED - CONTENT_FAIL: a line does not say its script words exactly', f'No new video was made. Report: {report}']
            result = dp.video_result('page-x', 1, output, videos=Path(folder))
        self.assertEqual(result['status'], mv.CONTENT_FAIL)
        self.assertEqual(result['why'], 'a line does not say its script words exactly')
        self.assertEqual(result['checks'], {'speech_lines': 'CONTENT_FAIL'})

    def test_a_tool_that_cannot_start_is_a_tool_error(self):
        result = dp.video_result('page-x', 2, ['Traceback ...', 'FileNotFoundError: no python'])
        self.assertEqual((result['status'], result['why']), (mv.TOOL_ERROR, 'FileNotFoundError: no python'))

    def test_a_finished_film_gives_its_address_and_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'page-x').mkdir()
            Path(folder, 'page-x', 'report.json').write_text(json.dumps(
                {'seconds': 55.5, 'checks': {'speech_lines': {'status': 'PASS'}, 'reading_time': {'status': 'PASS'}},
                 'sources': [{'scene': 2, 'warning': 'STRONG WARNING', 'readable_name_missing': False},
                             {'scene': 3, 'warning': '', 'readable_name_missing': False}]}))
            result = dp.video_result('page-x', 0, ['6/6 done'], videos=Path(folder))
        self.assertEqual((result['status'], result['video'], result['seconds']), (mv.PASS, '/video/page-x.mp4', 55.5))
        self.assertEqual(result['checks'], {'speech_lines': 'PASS', 'reading_time': 'PASS'})
        self.assertEqual([s['scene'] for s in result['reviewer']], [2])


class ByteRange(unittest.TestCase):
    def test_the_video_player_gets_the_part_it_asks_for(self):
        self.assertIsNone(dp.byte_range(None, 100))
        self.assertIsNone(dp.byte_range('bytes=-', 100))
        self.assertEqual(dp.byte_range('bytes=0-', 100), (0, 99))
        self.assertEqual(dp.byte_range('bytes=10-19', 100), (10, 19))
        self.assertEqual(dp.byte_range('bytes=90-500', 100), (90, 99))
        self.assertEqual(dp.byte_range('bytes=-10', 100), (90, 99))
        self.assertEqual(dp.byte_range('bytes=100-', 100), 'bad')


if __name__ == '__main__':
    unittest.main()
