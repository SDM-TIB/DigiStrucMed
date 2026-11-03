"""Scorecard checks that read what a storyboard says, not only its shape.

First check: every number in the storyboard must come from the approved sources
(the request, the example's key points and its allowed facts). A wrong number in
patient advice is the error we fear most, so this check is about precision.

Second check: did the storyboard say the example's key points? Each key point has a
few anchor words (approved 2026-10-27) and counts as said when any of them appears.
It cannot tell "call 112" from "don't call 112", so it measures coverage, not truth.

Third check: is it easy for patients to follow? Every field keeps Mahsa's word limit and
every sentence stays short (approved 2026-10-27). Medical jargon is not checked.

SCORECARD joins the three with Mahsa's two into one score for her StoryboardEvaluator.
A refusal or an empty answer scores 0 on every check (found in rerun-02: refusals earned
0.2, more than the average real storyboard).
"""
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from GAAPO.data import Example
from GAAPO.storyboard import StoryboardEvaluator, format_validity, inspect_format, structure_completeness

WORDS = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
         'ten': 10, 'eleven': 11, 'twelve': 12, 'fifteen': 15, 'twenty': 20, 'thirty': 30,
         'forty': 40, 'fifty': 50, 'sixty': 60, 'hundred': 100, 'one': 1}
UNITS = {'second': 'second', 'minute': 'minute', 'min': 'minute', 'hour': 'hour', 'hr': 'hour',
         'day': 'day', 'week': 'week', 'month': 'month', 'year': 'year', 'time': 'time',
         'gram': 'g', 'g': 'g', 'mg': 'mg', 'kg': 'kg', 'liter': 'l', 'litre': 'l', 'l': 'l',
         'ml': 'ml', 'percent': '%', '%': '%'}
NUMBER = r'\d+(?:\.\d+)?|(?:' + '|'.join(WORDS) + r')\b'  # "10g" is a number, "tend" is not
UNIT = '|'.join(sorted((u + 's?' if u.isalpha() else re.escape(u) for u in UNITS), key=len, reverse=True))
DASHES = r'\-‐‑‒–—―'  # for [...] classes; the data also uses the non-breaking hyphen U+2011
QUANTITY = re.compile(rf'\b({NUMBER})(?:\s*(?:[{DASHES}]|to)\s*({NUMBER}))?(?:[\s{DASHES}]*({UNIT})(?![a-z]))?', re.I)
NUMBER_WORDS = {w: str(n) for w, n in WORDS.items() if w != 'one'}  # "the right one" is no number
SCENE_NUMBER = re.compile(r'\bscene[\s_-]*(?:number)?\W{0,4}\d+', re.I)  # "scene_number ":2, "Scene 2:"
# Mahsa's limits: title from GAAPO/import_chat_documents.py, the rest from GAAPO/models.py
WORD_LIMITS = {'title': 10, 'narration': 35, 'visual_description': 15, 'on_screen_text': 8}
SENTENCE_LIMIT = 20  # plain-language rule of thumb: about 15-20 words a sentence
# A sentence ends at . ! ? (before closing quotes or brackets), a line break, or a "key": in broken JSON
SENTENCE_BREAK = re.compile(r'(?<=[.!?])["\')\]},]*\s+|"[\w -]{1,30}"\s*:|\\n|\n')
REFUSAL = re.compile(r"\bI\s+(?:can(?:no|'|’)?t|can not|am unable|won(?:'|’)t)\b|\bI['’]m (?:not able|unable)\b", re.I)


def storyboard_parts(output):
    """(field, text) for what a viewer sees or hears; scene numbers are structure, not content.
    Unless the whole output is one JSON object it is one part with no field: tidying may pick
    an empty later fragment."""
    report = inspect_format(output)
    value = report['value']
    if not isinstance(value, dict) or report['wrapped_json']:
        return [(None, SCENE_NUMBER.sub(' ', output))] if isinstance(output, str) else []
    parts = [('title', value.get('title'))]
    for scene in value.get('scenes') or []:
        if isinstance(scene, dict):
            parts += [(k, scene.get(k)) for k in ('narration', 'visual_description', 'on_screen_text')]
    return [(field, text) for field, text in parts if isinstance(text, str)]


def storyboard_text(output):
    """All the words of a storyboard in one string."""
    return ' '.join(text for _, text in storyboard_parts(output))


def quantities(text):
    """(value, unit) pairs; a range such as "10–20 g" gives both ends with the unit."""
    found = set()
    for match in QUANTITY.finditer(text):
        low, high, unit = match.groups()
        if unit:
            unit = unit.lower()
            unit = UNITS.get(unit) or UNITS.get(unit[:-1])  # "days" -> "day"
        if low.lower() == 'one' and unit is None:
            continue  # "one of", "the right one": too common to be a stated number
        for number in (low, high):
            if number:
                found.add((WORDS.get(number.lower()) or float(number), unit))
    return found


def sources(example):
    """What a storyboard for a ruler example may state: allowed facts are true per the
    students' source but not required, so stating them is never an error."""
    return [example['request'], *example['key_points'], *(f['text'] for f in example.get('allowed_facts', []))]


def unsupported_numbers(output, sources):
    """Numbers in the storyboard that no source states; a unit must match when one is given."""
    allowed = set().union(*(quantities(s) for s in sources))
    values = {v for v, _ in allowed}
    return sorted(((v, u) for v, u in quantities(storyboard_text(output))
                   if (v, u) not in allowed and not (u is None and v in values)), key=str)


def number_precision(output, sources):
    """1.0 when every number is backed by a source; each unbacked number lowers it."""
    found = quantities(storyboard_text(output))
    return 1.0 if not found else 1 - len(unsupported_numbers(output, sources)) / len(found)


def normalize(text):
    """Lower case, "40%" -> "40 percent", hyphens and slashes -> spaces, "three" -> "3"."""
    text = re.sub(r'(\d)([a-z%])', r'\1 \2', text.lower()).replace('%', ' percent')
    text = re.sub(f'[{DASHES}/]', ' ', text)
    text = re.sub(r'\b(' + '|'.join(NUMBER_WORDS) + r')\b', lambda m: NUMBER_WORDS[m.group(1)], text)
    return re.sub(r'\s+', ' ', text)


@lru_cache(maxsize=None)
def anchor_pattern(anchor):
    """Whole words, plural s/es allowed; a trailing * allows any ending ("breath*" finds "breathing")."""
    words = normalize(anchor).split()
    return re.compile(r'\b' + ' '.join(re.escape(w[:-1]) + r'\w*' if w.endswith('*')
                                       else re.escape(w) + '(?:s|es)?' for w in words) + r'\b')


def key_points_said(output, example):
    """One True/False per key point: does the storyboard contain any of its anchors?"""
    text = normalize(storyboard_text(output))
    return [any(anchor_pattern(a).search(text) for a in anchors) for anchors in example['anchors']]


def key_point_coverage(output, example):
    """Share of the example's key points the storyboard touches."""
    said = key_points_said(output, example)
    return sum(said) / len(said)


def word_count(text):
    return len(text.split())  # as Mahsa counts in GAAPO/import_chat_documents.py


def sentences(text):
    """Stray braces and quotes from broken JSON are not sentences: each needs a letter or digit."""
    return [s.strip() for s in SENTENCE_BREAK.split(text) if re.search(r'[^\W_]', s)]


def reading_pieces(output):
    """(text, word limit) for each field and each sentence. Broken output has no reliable
    fields, so only its sentences are judged; the format check already punishes it."""
    pieces = []
    for field, text in storyboard_parts(output):
        if field in WORD_LIMITS:
            pieces.append((text, WORD_LIMITS[field]))
        pieces += [(sentence, SENTENCE_LIMIT) for sentence in sentences(text)]
    return pieces


def too_long(output):
    """(text, words, limit) for every field or sentence over its limit."""
    return [(text, word_count(text), limit) for text, limit in reading_pieces(output)
            if word_count(text) > limit]


def readability(output):
    """Share of fields and sentences within their word limits; 1.0 when there is nothing to
    judge, since saying nothing is the key-point check's job to punish."""
    pieces = reading_pieces(output)
    return 1.0 if not pieces else 1 - len(too_long(output)) / len(pieces)


def says_nothing(output):
    """An empty answer, or a short one that refuses ("I can't help with this request ...")."""
    words = storyboard_text(output).split()
    return len(words) < 3 or (len(words) < 60 and bool(REFUSAL.search(output)))


def unless_silent(check):
    """No points for saying nothing: an empty or refusing answer has no wrong numbers and no long
    sentences, so these checks used to give it full marks."""
    return lambda output, example: 0.0 if says_nothing(output) else check(output, example)


@dataclass(frozen=True)
class RulerExample(Example):
    """A GAAPO example that also carries its ruler row: key points, anchors and allowed facts."""
    ruler: dict = field(default=None, compare=False)


def load_ruler(path='GALLM/data/ruler_draft.json', split=None):
    """Ruler rows as GAAPO examples. The ruler has no reference storyboard: its label is the key points."""
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    return tuple(RulerExample(row['id'], row['request'], {'key_points': row['key_points']}, row)
                 for row in rows if split in (None, row['split']))


# One score: Mahsa's two checks plus our three, equal weights (approved 2026-10-28).
# StoryboardEvaluator itself halves the score when the scene count is wrong.
SCORECARD = {
    'format_validity': unless_silent(format_validity),
    'structure_completeness': unless_silent(structure_completeness),
    'number_precision': unless_silent(lambda output, example: number_precision(output, sources(example.ruler))),
    'key_point_coverage': lambda output, example: key_point_coverage(output, example.ruler),
    'readability': unless_silent(lambda output, example: readability(output)),
}
WEIGHTS = {name: 1.0 for name in SCORECARD}


def reference_sources(example):
    """Mahsa's training examples have no key points; their request and reference storyboard are the sources."""
    return [example.user_request, storyboard_text(json.dumps(example.output))]


# GAAPO's APO scores Mahsa's training examples to pick errors to learn from. They have no key
# points, so there the other four checks apply, with the reference storyboard as number source.
TRAINING_SCORECARD = {**{name: m for name, m in SCORECARD.items() if name != 'key_point_coverage'},
                      'number_precision': unless_silent(
                          lambda output, example: number_precision(output, reference_sources(example)))}


class ScorecardEvaluator(StoryboardEvaluator):
    """Mahsa's evaluator with SCORECARD on ruler examples and TRAINING_SCORECARD on hers."""
    def __init__(self, model, **kwargs):
        super().__init__(model, SCORECARD, WEIGHTS, **kwargs)

    def assess(self, prompt, example):
        self.metrics = SCORECARD if isinstance(example, RulerExample) else TRAINING_SCORECARD
        self.weights = {name: 1.0 for name in self.metrics}
        return super().assess(prompt, example)
