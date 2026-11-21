"""The timeline compiler: one checked script + its storyboard + the voice lengths -> timeline.json, the single file the
picture, the sound and the captions are all built from. Fixed code: the same inputs always give the same timeline.

Spoken order: the patient asks their own question (patient voice), then the checked lines (doctor voice), with the
spoken AI label right after the doctor's opening line. Every segment says who speaks, the exact caption (= the spoken
words), its checked heading and source line, and what the storyboard shows: camera shot, doctor action, patient action
and one object. Without a storyboard, a calm default is used (medium shot, doctor explains, patient listens, no object).

Timing rule (as before): pause, voice, pause, and never shorter than the time needed to read the longest text on screen
(non-space characters / 15 + 1.5 s)."""
import re

PAUSE_BEFORE, PAUSE_AFTER = 0.6, 1.0
READ_CPS, READ_PAD = 15, 1.5
DEFAULT_SCENE = {'shot': 'medium', 'doctor': 'explains', 'patient': 'listens', 'prop': 'none'}
QUESTION_SCENE = {'shot': 'close-up on patient', 'doctor': 'leans forward', 'patient': 'asks the question', 'prop': 'none'}
LABEL_SCENE = {'shot': 'medium', 'doctor': 'explains', 'patient': 'listens', 'prop': 'none'}


def reading_time(text):
    return sum(1 for ch in text if not ch.isspace()) / READ_CPS + READ_PAD


def segments_of(record, spoken_label, readable_source=lambda source: source):
    """The spoken segments, in order, before timing. `readable_source` turns a source entry into its on-screen line."""
    sources = {s['scene']: s for s in record.get('sources', [])}
    segments = [{'id': 'q', 'speaker': 'patient', 'text': record['question'], 'heading': '', 'source': '',
                 'medical': False, 'line': None, **QUESTION_SCENE}]
    for n, scene in enumerate(record['scenes'], 1):
        source = sources.get(n)
        segments.append({'id': f's{n}', 'speaker': 'doctor', 'text': scene['narration'],
                         'heading': scene.get('on_screen_text') or '', 'source': readable_source(source) if source else '',
                         'medical': bool(scene.get('medical')), 'line': n, **DEFAULT_SCENE})
        if n == 1 and spoken_label:
            segments.append({'id': 'label', 'speaker': 'doctor', 'text': spoken_label, 'heading': '', 'source': '',
                             'medical': False, 'line': None, **LABEL_SCENE})
    return segments


def apply_storyboard(segments, storyboard):
    """Copy the director's choices onto the script lines they belong to (storyboard scene 'line' n -> script line n).
    Only the four visual fields are taken; the director never supplies words."""
    by_line = {s.get('line'): s for s in storyboard or [] if isinstance(s, dict)}
    for seg in segments:
        choice = by_line.get(seg['line'])
        if choice:
            for field in ('shot', 'doctor', 'patient', 'prop'):
                if choice.get(field):
                    seg[field] = choice[field]
            seg['show_heading'] = bool(choice.get('heading')) and bool(seg['heading'])
        else:
            seg['show_heading'] = bool(seg['heading'])
        if seg['speaker'] == 'doctor' and seg['patient'] == 'asks the question':
            seg['patient'] = 'listens'   # the patient already asked, in their own segment, before the doctor speaks
        seg['prop_at'] = 'doctor-hand' if seg['doctor'] == 'holds up the prop' else 'desk'
    return segments


def build(segments, durations, words=None, fps=24, size='1920x1080', loudness=None):
    """Times every segment. durations: voice length per segment id (seconds). words: Whisper word timings per id
    ([word, start, end] from the start of that line's recording), turned into times on the film's clock. loudness: per
    id, the voice's loudness (0..1) for each video frame from the start of the voice; it moves the speaker's mouth."""
    t, out = 0.0, []
    for seg in segments:
        voice = float(durations[seg['id']])
        on_screen = [seg['text'], seg['heading'] if seg.get('show_heading', True) else '', seg['source']]
        read = max(reading_time(x) for x in on_screen if x)
        length = max(PAUSE_BEFORE + voice + PAUSE_AFTER, read + 0.4)
        voice_at = t + PAUSE_BEFORE
        entry = {**seg, 't0': round(t, 3), 't1': round(t + length, 3), 'voice_at': round(voice_at, 3), 'voice': round(voice, 3)}
        if words and seg['id'] in words:
            entry['words'] = [[w, round(voice_at + a, 3), round(voice_at + b, 3)] for w, a, b in words[seg['id']]]
        if loudness and seg['id'] in loudness:
            entry['loudness'] = loudness[seg['id']]
        out.append(entry)
        t += length
    return {'version': 1, 'fps': fps, 'size': size, 'seconds': round(t, 3), 'segments': out}


def guideline_year(source_text):
    found = re.findall(r'\b(19[5-9]\d|20\d\d)\b', source_text or '')
    return int(found[0]) if found else None
