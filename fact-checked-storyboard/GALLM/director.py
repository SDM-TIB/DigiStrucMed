"""The director: turns a checked script into scenes of a doctor talking with a patient, and scores how good those scenes are,
so GAAPO can evolve the director's instructions (the prompt engineering).

Safety comes from the design, not from the prompt: the director never writes a word. For every checked line it only picks,
from fixed lists, the camera shot, what the doctor does, what the patient does and one prop. Code then fills in the
checked words exactly, the patient's own question, and on-screen headings that already passed the screen check. So no
instruction GAAPO invents can add a medical claim to the video.

Only the director's instructions evolve. The wrapper around them (the lines, the allowed values, the answer format) is fixed
by code, so an evolved prompt cannot break the format or the rules.

Score of one storyboard (0..1), the fitness GAAPO maximises (VIDEO_PLAN.md section 8, research P1 and P2):
  safety gate (code)   wrong format, a line missing, moved or doubled, a value not on the lists, or an object showing what
                       its line warns against (a cigarette pack on "stop smoking") -> 0
  code quality (0.4)   objects fit what is said ("none" is fine when nothing fits; Houts et al. 2006: pictures help only
                       when they match the message), the doctor shows the object when there is one, the patient asks the
                       question and reacts. Variety no longer counts.
  judge (0.6)          gpt-oss-120b compares the storyboard with the starting prompt's storyboard for the same question,
                       asked in BOTH orders (Zheng et al. 2023: judges favour a position): a win or a loss counts only when
                       both answers agree, anything else is a tie (0.5). Picking one of two separates prompts far better
                       than marks out of 10 (an earlier GAAPO run with absolute marks had an almost flat fitness)."""
import hashlib
import json
import re

from GALLM.text_tools import json_object

PROPS = ['none', 'heart model', 'calendar', 'clock', 'car keys', 'ICD device model', 'pill box', 'glass of water',
         'walking shoes', 'blood pressure cuff', 'bathroom scale', 'phone', 'blank notepad', 'tablet showing a heartbeat line',
         'plate of healthy food', 'stethoscope', 'cigarette pack', 'wine glass', 'bed and pillow']
DOCTOR = ['explains', 'points to the prop', 'holds up the prop', 'smiles warmly', 'nods', 'leans forward', 'gestures calmly']
PATIENT = ['asks the question', 'listens', 'nods', 'looks relieved', 'thinks', 'looks at the prop', 'smiles']
SHOTS = ['wide', 'medium', 'close-up on doctor', 'close-up on patient', 'close-up on prop']
CODE_WEIGHT, JUDGE_WEIGHT = 0.4, 0.6
# Which object fits which spoken words (whole words or phrases, lower case). An object fits a line when one of its words is
# in the line. Plain communication rules, not medical claims; extend the list rather than loosen it.
PROP_FITS = {
    'heart model': ['heart', 'pump', 'pumps', 'pumping', 'muscle', 'valve', 'valves', 'chamber', 'chambers', 'artery', 'arteries'],
    'calendar': ['day', 'days', 'week', 'weeks', 'month', 'months', 'year', 'years'],
    'clock': ['hour', 'hours', 'minute', 'minutes', 'daily', 'morning', 'evening', 'time'],
    'car keys': ['drive', 'driving', 'car', 'cars'],
    'ICD device model': ['icd', 'defibrillator', 'device', 'implant', 'implanted', 'shock', 'shocks'],
    'pill box': ['medicine', 'medicines', 'pill', 'pills', 'tablet', 'tablets', 'dose', 'doses', 'drug', 'drugs'],
    'glass of water': ['water', 'fluid', 'fluids', 'thirst', 'thirsty'],
    'walking shoes': ['walk', 'walking', 'exercise', 'exercises', 'active', 'activity', 'training'],
    'blood pressure cuff': ['blood pressure'],
    'bathroom scale': ['weight', 'weigh', 'kilo', 'kilos', 'kilogram', 'kilograms', 'pound', 'pounds'],
    'phone': ['call', 'phone', 'emergency', '112', '911'],
    'blank notepad': ['write', 'note', 'notes', 'diary', 'record', 'track'],
    'tablet showing a heartbeat line': ['rhythm', 'heartbeat', 'heart rate', 'pulse', 'ecg', 'fibrillation'],
    'plate of healthy food': ['food', 'eat', 'eating', 'diet', 'salt', 'meal', 'meals'],
    'stethoscope': ['check', 'checks', 'check-up', 'check-ups', 'exam', 'examination', 'listen'],
    'cigarette pack': ['smoke', 'smoking', 'cigarette', 'cigarettes', 'tobacco', 'nicotine'],
    'wine glass': ['alcohol', 'wine', 'beer', 'drinking'],
    'bed and pillow': ['sleep', 'sleeping', 'rest', 'night', 'bed', 'apnoea', 'apnea'],
}
# Objects that show a risky thing. They may illustrate a line that describes it, never a line that advises against it.
WARNED_PROPS = {'cigarette pack', 'wine glass'}
ADVICE_AGAINST = ['avoid', 'stop', 'stopping', 'quit', 'quitting', 'do not', "don't", 'not', 'no', 'never', 'limit', 'cut down',
                  'reduce', 'less', 'without']
SHOWS_PROP = ('points to the prop', 'holds up the prop')

START_PROMPT = '''You are directing a short patient-education video. A friendly doctor explains to a patient in a calm consultation room.
For every script line, choose the camera shot, what the doctor does, what the patient does and one prop.
- Pick a prop that helps the patient picture what the doctor is saying in that line; choose "none" when nothing fits.
- Never show something the line says not to do or warns against.
- Let the patient take part: they ask their question at the start and react naturally while listening.
- Vary the camera shots so the video does not look static, but keep it calm.'''

WRAPPER = '''{instructions}

The patient asked: "{question}"

Script lines, in order. They are already medically checked; you never write or change spoken words.
{lines}

Allowed values (use them exactly as written):
shot: {shots}
doctor: {doctor}
patient: {patient} ("asks the question" only in scene 1)
prop: {props}

Answer with JSON only: one scene per script line, in the same order:
{{"scenes": [{{"line": 1, "shot": "...", "doctor": "...", "patient": "...", "prop": "...", "heading": true or false, "pause": 0.0 to 2.0}}]}}
"heading": true shows the line's checked heading on screen; only lines marked [heading] have one.'''

COMPARE = '''Two storyboards for the same short patient-education video: a doctor explains to a patient. The spoken words are the same
in both and already medically checked; judge only what is shown and done.

Patient question: "{question}"

Storyboard A:
{a}

Storyboard B:
{b}

Which makes the better video for a patient?
1. Does each prop and action fit what the doctor is saying at that moment and help the patient picture it?
2. Does it never show something the line warns against or says not to do?
3. Does it feel like a real, warm conversation (the patient takes part, natural reactions)?
4. Is it calm and easy to follow (no object or camera change just for variety)?
If they are about equal, answer "tie".
Answer with JSON only: {{"better": "A" or "B" or "tie", "why": "..."}}'''

VIDEO_PROMPT = '''Create a calm patient-education video: a friendly doctor explains to a patient in a calm consultation room, answering the patient's question: "{question}"

STRICT RULES - this is a medically fact-checked script:
- Spoken words EXACTLY as written below, word for word, in this order. Add, remove or change nothing.
- No other spoken words, captions or on-screen text, apart from the headings given below.
- No readable text or numbers in the picture (calendars, clocks and devices without readable writing).
- The same doctor, patient and room in every scene.

{scenes}

Voice: one calm, warm, clear English voice for the doctor; the patient's own voice for their question. Very soft background music or none.'''


def script_lines(record):
    """The locked lines of a checked record: its scenes, in order, with their checked heading (if any) and source."""
    sources = {s['scene']: s for s in record.get('sources', [])}
    return [{'n': n, 'text': s['narration'], 'medical': bool(s.get('medical')), 'heading': s.get('on_screen_text') or '',
             'source': (sources.get(n) or {}).get('source', '')} for n, s in enumerate(record['scenes'], 1)]


def director_prompt(instructions, record):
    lines = '\n'.join(f'{line["n"]}. {"[heading] " if line["heading"] else ""}"{line["text"]}"' for line in script_lines(record))
    quote = lambda values: ', '.join(f'"{v}"' for v in values)
    return WRAPPER.format(instructions=instructions.strip(), question=record['question'], lines=lines, shots=quote(SHOTS),
                          doctor=quote(DOCTOR), patient=quote(PATIENT), props=quote(PROPS))


def read_storyboard(raw):
    answer = json_object(raw)
    scenes = answer.get('scenes') if isinstance(answer, dict) else None
    return scenes if isinstance(scenes, list) else None


def gate(scenes, lines):
    """Problems that make a storyboard unusable. The words cannot be wrong (code fills them), so the gate checks the shape:
    one scene per line in order, and every choice from its list."""
    if not scenes:
        return ['no readable storyboard']
    problems = []
    if [s.get('line') if isinstance(s, dict) else None for s in scenes] != [line['n'] for line in lines]:
        problems.append('scenes are not exactly one per script line, in order')
    for s in scenes:
        if not isinstance(s, dict):
            continue
        for key, allowed in (('shot', SHOTS), ('doctor', DOCTOR), ('patient', PATIENT), ('prop', PROPS)):
            if s.get(key) not in allowed:
                problems.append(f'line {s.get("line")}: {key} "{s.get(key)}" is not an allowed value')
    texts = {line['n']: line['text'] for line in lines}
    for s in scenes:
        if isinstance(s, dict) and s.get('prop') in WARNED_PROPS and s.get('line') in texts and advises_against(texts[s['line']]):
            problems.append(f'line {s.get("line")}: "{s["prop"]}" shows what the line advises against')
    return problems


def has_words(text, words):
    """True when one of the words or phrases appears in the text as whole words."""
    low = ' ' + re.sub(r"[^a-z0-9'\- ]+", ' ', text.lower()) + ' '
    return any(f' {w} ' in low for w in words)


def advises_against(text):
    return has_words(text, ADVICE_AGAINST)


def fitting_props(text):
    return {prop for prop, words in PROP_FITS.items() if has_words(text, words)}


def code_quality(scenes, lines):
    """0..1 from things code can count. Each part is 0..1; the score is their mean.
    objects fit: per medical line, a fitting object 1; no object 1 when nothing fits, 0.5 when one would have fitted;
                 an object that does not fit the words 0 (a decorative picture distracts).
    doctor shows the object: the doctor points to or holds up the object when there is one, and does not point at nothing.
    patient takes part: asks their question at the start (only there) and reacts while listening."""
    pairs = [(s, line) for s, line in zip(scenes, lines) if line['medical']]
    fit = []
    for s, line in pairs:
        fits = fitting_props(line['text'])
        fit.append((1.0 if s['prop'] in fits else 0.0) if s['prop'] != 'none' else (0.5 if fits else 1.0))
    shows = [1.0 if (s['prop'] != 'none') == (s['doctor'] in SHOWS_PROP) else 0.0 for s in scenes]
    reacting = sum(1 for s in scenes[1:] if s['patient'] not in ('listens', 'asks the question'))
    asks_first = scenes[0]['patient'] == 'asks the question'
    asks_later = any(s['patient'] == 'asks the question' for s in scenes[1:])
    parts = {
        'objects fit': sum(fit) / len(fit) if fit else 1.0,
        'doctor shows the object': sum(shows) / len(shows),
        'patient takes part': 0.0 if asks_later else (0.5 * asks_first + 0.5 * min(1.0, reacting / 2)),
    }
    return sum(parts.values()) / len(parts), parts


HAND_PROPS = {'heart model', 'ICD device model', 'pill box', 'car keys', 'phone', 'glass of water', 'blank notepad',
              'tablet showing a heartbeat line', 'stethoscope', 'walking shoes'}   # small enough to hold up


def code_storyboard(record):
    """A plain storyboard made by code with the same rules the score uses, for a video when no director storyboard is
    given (no AI, no tokens): each medical line shows an object that fits its words (a different one from the line
    before when another fits), never one the line advises against; the doctor holds up small objects and points to
    the others; the patient looks at the object, otherwise listens or nods, and smiles at the end."""
    lines = script_lines(record)
    scenes, previous = [], 'none'
    for i, line in enumerate(lines):
        first, last = i == 0, i == len(lines) - 1
        fits = [p for p in PROP_FITS if p in fitting_props(line['text'])] if line['medical'] else []
        if advises_against(line['text']):
            fits = [p for p in fits if p not in WARNED_PROPS]
        prop = next((p for p in fits if p != previous), fits[0] if fits else 'none')
        if prop == 'none':
            doctor = 'smiles warmly' if first or last else 'explains'
            patient = 'smiles' if last else 'listens' if first or i % 2 else 'nods'
            shot = 'medium'
        else:
            doctor = 'holds up the prop' if prop in HAND_PROPS else 'points to the prop'
            patient = 'looks at the prop'
            shot = 'close-up on doctor' if doctor == 'holds up the prop' else 'medium'
        scenes.append({'line': line['n'], 'shot': shot, 'doctor': doctor, 'patient': patient, 'prop': prop,
                       'heading': bool(line['heading']), 'pause': 0.5})
        previous = prop
    return scenes


def scene_texts(scenes, lines, question):
    """The storyboard in words, built by code: what the judge compares and what the video prompt says. Every spoken word
    comes from the checked lines or the patient's own question."""
    out = []
    for i, (s, line) in enumerate(zip(scenes, lines), 1):
        prop = s['prop'] if s['prop'] != 'none' else ''
        doctor = s['doctor'].replace('the prop', f'the {prop}' if prop else 'the desk')
        parts = [f'Scene {i} ({s["shot"]}{", prop: " + prop if prop else ""})']
        if s['patient'] == 'asks the question':
            parts.append(f'The patient asks: "{question}"')
        parts.append(f'The doctor {doctor} and says: "{line["text"]}"')
        if s['patient'] not in ('asks the question',):
            parts.append(f'The patient {s["patient"].replace("the prop", f"the {prop}" if prop else "the doctor")}.')
        if s.get('heading') is True and line['heading']:
            parts.append(f'On-screen heading: "{line["heading"]}"')
        pause = max(0.0, min(2.0, float(s.get('pause') or 0)))
        if pause:
            parts.append(f'Pause {pause:.1f} s.')
        out.append('\n'.join(parts))
    return '\n\n'.join(out)


def video_prompt(record, scenes):
    return VIDEO_PROMPT.format(question=record['question'].replace('"', "'"),
                               scenes=scene_texts(scenes, script_lines(record), record['question']))


def key(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode('utf-8')).hexdigest()[:16]


class DirectorEvaluator:
    """GAAPO's evaluator: score(prompt, examples) -> 0..1 and assess(prompt, example) -> (output, score).
    Examples are GAAPO Examples whose id names a checked record in `records`."""

    def __init__(self, records, director, judge, baseline=START_PROMPT, seed=7):
        self.records, self.director, self.judge, self.baseline, self.seed = records, director, judge, baseline, seed
        self.results = {}  # (prompt key, example id) -> result, so a prompt is never scored twice on a question

    def storyboard(self, instructions, record):
        return read_storyboard(self.director.generate(director_prompt(instructions, record)))

    def result(self, prompt, example_id):
        k = (key(prompt), example_id)
        if k in self.results:
            return self.results[k]
        record = self.records[example_id]
        lines = script_lines(record)
        scenes = self.storyboard(prompt, record)
        problems = gate(scenes, lines)
        if problems:
            result = {'score': 0.0, 'problems': problems, 'text': 'UNUSABLE: ' + '; '.join(problems[:3])}
        else:
            quality, parts = code_quality(scenes, lines)
            text = scene_texts(scenes, lines, record['question'])
            verdict = self.compare(prompt, example_id, record, text)
            result = {'score': round(CODE_WEIGHT * quality + JUDGE_WEIGHT * verdict['points'], 4), 'code': parts,
                      'judge': verdict, 'text': text, 'scenes': scenes}
        self.results[k] = result
        return result

    def compare(self, prompt, example_id, record, text):
        """Points against the starting prompt's storyboard, asked in both orders (new one as A, then as B): better 1 only
        when both answers say better, worse 0 only when both say worse, anything else (a tie, or the answer flipping with
        the order) 0.5. So a judge that prefers a position cannot decide the score."""
        if key(prompt) == key(self.baseline):
            return {'points': 0.5, 'why': 'this is the starting prompt'}
        base = self.result(self.baseline, example_id)
        if base['score'] == 0.0:  # the starting prompt failed here: any usable storyboard is better
            return {'points': 1.0, 'why': 'the starting prompt gave no usable storyboard'}
        if text == base['text']:
            return {'points': 0.5, 'why': 'same storyboard as the starting prompt'}
        votes, whys = [], []
        for a, b, mine in ((text, base['text'], 'A'), (base['text'], text, 'B')):
            answer = json_object(self.judge.generate(COMPARE.format(question=record['question'], a=a, b=b))) or {}
            better = str(answer.get('better', 'tie')).strip().upper()
            votes.append(1.0 if better == mine else 0.5 if better not in ('A', 'B') else 0.0)
            whys.append(str(answer.get('why', ''))[:200])
        points = votes[0] if votes[0] == votes[1] else 0.5
        return {'points': points, 'votes': votes, 'why': ' | '.join(whys)}

    def score(self, prompt, examples):
        examples = list(examples)
        return sum(self.result(prompt, e.id)['score'] for e in examples) / len(examples) if examples else 0.0

    def assess(self, prompt, example):
        """For GAAPO's learn-from-mistakes step (APO): the storyboard plus why it lost points."""
        result = self.result(prompt, example.id)
        notes = '' if 'problems' in result else (
            '\n\nCode checks (0..1): ' + ', '.join(f'{k} {v:.2f}' for k, v in result['code'].items())
            + f'\nJudge, compared with the starting prompt: {result["judge"]["points"]} - {result["judge"]["why"]}')
        return result['text'] + notes, result['score']
