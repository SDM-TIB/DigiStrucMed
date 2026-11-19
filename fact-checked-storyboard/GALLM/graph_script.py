"""Part 2 (agreed 2026-11-18): graph facts -> a checked video script -> a video prompt. Research behind each step:
`Documents/for me/reports/Part 2 script research.md`.

    question -> part 1 graph facts (graph_facts.py) -> code: one slot per fact, in rank order (Cheng 2026, Kasner & Dusek 2022)
    -> writer gpt-oss-20b: one spoken line per slot, no merging (Cheng 2026: best bounded writer of the open models)
    -> code checks: numbers in the fact, caution kept, on-screen words from the line or fact (Devaraj 2022, Ayadi 2026)
    -> judge gpt-oss-120b: the line against ITS OWN fact only (Kamoi 2024; new_judge.py, 1 mistake in 104 code-made tests)
    -> a failed line is rewritten once with the reason, checked again, else dropped (Kamoi 2024, Tyen 2024)
    -> the video prompt, built by code from the kept lines, with each line's guideline source beside it.

Writer and judge are two different models, so the judge never grades its own writing (Huang 2024, Ji 2026).
Both run on Groq's free plan; every call is cached, so a question asked again costs nothing."""
import json
import os
import re
import time

from .text_tools import json_object
from .new_judge import judge_line

WRITER_MODEL, JUDGE_MODEL = 'openai/gpt-oss-20b', 'openai/gpt-oss-120b'
GROQ = 'https://api.groq.com/openai/v1'
MOST_WORDS, SCREEN_WORDS = 18, 6
MOST_LINES = 6  # about 45 seconds of speech with the opening and closing (agreed length 2026-11-18)
KEPT = ('SUPPORTED', 'NO_CLAIM')
# "one" is left out: "one thing you can do" is no number.
NUMBER_WORDS = {w: str(n) for n, w in enumerate('zero one two three four five six seven eight nine ten eleven twelve'.split()) if w != 'one'}
# A fact that only allows or suggests something; the line must not turn it into a must. General words, not per relation.
CAUTION_IN_FACT = re.compile(r'\b(may|might|could|consider(?:ed)?|optional|can be)\b', re.I)
CAUTION_IN_LINE = re.compile(r'\b(may|might|could|can|consider|considered|possibly|option|options|sometimes|talk|ask|discuss)\b', re.I)
SMALL = set('a an the and or of to in on for with by is are be your you it this that at as from after before'.split())
# Opening and closing are written by code: the writer kept putting claims in them ("how it helps your heart"), and fixed
# words need no judge call (TFTS: code owns what recurs).
OPENING = "Hello. Let's look at what the heart guidelines say about your question."
CLOSING = 'If you have questions, please ask your care team.'
# An abbreviation as guidelines write it: two or more capitals, maybe with digits or small letters (HFrEF, SGLT2i, ALVD).
ABBREVIATION = re.compile(r'\b[A-Za-z0-9]*[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*\b')

WRITER = '''You write the spoken lines of a short patient-education video. A friendly heart doctor speaks to the patient as "you".

The patient asked: "{question}"

Below are numbered guideline facts, chosen and ordered by code. Write one spoken line for each fact you use, in this order.
Rules:
1. A line says what its one fact says, in plain words, and nothing more: no other medical facts, numbers, medicine names or advice from your own knowledge, even when you are sure they are true.
2. Keep the fact's group ("people with ...", "after ...") and its caution ("may", "can be considered").
3. A line has at most {most} words and sounds like a doctor talking to a patient, not like a guideline. Use everyday words a patient knows instead of technical terms ("task force", "system integrity check", "secondary prevention", "prognosis", "cessation", "obligatory"), meaning exactly the same. Do not copy the guideline sentence.
   Abbreviations: a fact may say what an abbreviation means (from the knowledge graph); then say that meaning in plain words. Never guess what an abbreviation means: if the fact does not say, keep the abbreviation as written or leave the fact out.
4. Never join two facts in one line.
5. Use at most {lines} facts: the ones that best answer the patient's question. Skip the others and say why, for example:
   - it does not help answer the question, or it is a table fragment;
   - it says the same as a fact you already used (name that fact). A fact about a different situation (after a shock, after a device replacement, for another group) is not the same: use it if it helps answer the question;
   - it is mainly about a different topic than the question, even if it also mentions the topic (for example a blood pressure target in a video about smoking);
   - it gives a different number or rule for the same situation as another fact (a conflict, not a duplicate; name that fact). Say only one, so the video never tells the patient two different things: use the guideline's own advice ("should", "is recommended", "the task force recommends") and skip a fact that only reports what others did or used to do ("most societies recommended", "previously", "on the basis of these data").
6. on_screen_text: at most {screen} words, taken from that line.
7. visual: one simple, calm picture for that line, with no words or numbers in it. Never show the thing the line says not to do.

Facts:
{slots}

Answer with JSON only:
{{"title": "...", "lines": [{{"fact": <number>, "narration": "...", "on_screen_text": "...", "visual": "..."}}], "skipped": [{{"fact": <number>, "why": "..."}}]}}'''
REPAIR = '''Some lines of your patient-education video did not pass the fact check. Rewrite only these lines, with the same rules:
a line says only what its own fact says, keeps the fact's group and caution, adds no number or medicine the fact does not
have, and has at most {most} words. It sounds like a doctor talking to a patient, in everyday words, not like the guideline
sentence. Never guess what an abbreviation means: use only a meaning the fact gives, or keep the abbreviation as written.
on_screen_text has at most {screen} words, taken from the new line.

{items}

Answer with JSON only: {{"lines": [{{"fact": <number>, "narration": "...", "on_screen_text": "..."}}]}}'''
VIDEO_PROMPT = '''Create a calm patient-education video, about {seconds} seconds long, that answers this patient question: "{question}"

STRICT RULE - this is a medically fact-checked script:
- Use the narration EXACTLY as written below, word for word, in this order. Do not add, remove, reword, shorten or paraphrase any sentence.
- Do not add any other spoken words: no extra greeting, intro, outro, summary or call to action.
- Do not add any medical facts, numbers, statistics or claims anywhere (narration, captions, on-screen text or pictures).
- Captions must match the narration exactly.
- On-screen headings must be exactly the texts given below. No other words, numbers or medicine names on screen, apart from the captions.
- Keep the same people and the same room in every scene.

{scenes}

Voice: one calm, warm, clear English voice, the same in every scene. Very soft background music or none.
Look: calm, reassuring, simple flat illustrations, easy to read for patients.'''


# ---- models ------------------------------------------------------------------------------------------------------------------

class GroqModel:
    """gpt-oss on Groq's free plan through the OpenAI client. Counts the tokens Groq reports, waits out the per-minute limit
    (429 with retry-after is handled by the client; 413 "Limit 8000, Requested ..." has none, so it waits a minute), and stops
    with a clear message at the daily limit, because a live demo must not hang for hours."""
    window_seconds = 60

    def __init__(self, model, api_key=None, client=None):
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=GROQ, api_key=api_key or os.environ['GROQ_API_KEY'], max_retries=8)
        self.client, self.model = client, model
        self.tokens, self.calls = 0, 0

    def generate(self, prompt, system=None, temperature=0, top_p=1.0):
        messages = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': prompt}]
        for attempt in range(4):
            try:
                answer = self.client.chat.completions.create(model=self.model, messages=messages, temperature=temperature,
                                                             top_p=top_p, max_tokens=4096)
                break
            except Exception as error:
                text = str(error)
                if 'per day' in text:
                    raise RuntimeError(f'Groq daily token limit reached for {self.model}; it frees up over the next 24 hours.') from error
                if getattr(error, 'status_code', None) != 413 or attempt == 3:
                    raise
                time.sleep(self.window_seconds)
        usage = getattr(answer, 'usage', None)
        self.tokens += getattr(usage, 'total_tokens', 0) or 0
        self.calls += 1
        content = answer.choices[0].message.content
        if not content or not content.strip():
            raise ValueError('The model returned an empty answer')
        return content.strip()


class DiskCache:
    """Answers a question asked before from disk, so a rehearsed demo question costs no tokens."""
    def __init__(self, llm, path):
        import hashlib
        import threading
        self.llm, self.path, self.hash, self.lock = llm, path, hashlib.sha256, threading.Lock()
        self.cache = {}
        if path.exists():
            for row in path.read_text(encoding='utf-8').splitlines():
                row = json.loads(row)
                self.cache[row['key']] = row['answer']

    def generate(self, prompt, system=None, **sampling):
        key = self.hash(json.dumps([self.llm.model, prompt, system, sorted(sampling.items())]).encode('utf-8')).hexdigest()
        if key not in self.cache:
            answer = self.llm.generate(prompt, system=system, **sampling)
            with self.lock:
                self.cache[key] = answer
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open('a', encoding='utf-8') as f:
                    f.write(json.dumps({'key': key, 'answer': answer}, ensure_ascii=False) + '\n')
        return self.cache[key]


def used(llm):
    """Tokens the model behind any wrappers has used so far (cached answers cost none)."""
    while llm is not None and not hasattr(llm, 'tokens'):
        llm = getattr(llm, 'llm', None)
    return getattr(llm, 'tokens', None)


# ---- 1. slots (code) ---------------------------------------------------------------------------------------------------------

def graph_abbreviations(query=None):
    """Abbreviation -> its full form, from the graph's own abbreviation marks (dl:fullForm), or None when the graph gives it
    two different meanings (VA: "veno-arterial" 94 times, "ventricular arrhythmia" 16): then nobody may expand it. Spelling
    variants of one meaning ("HF with reduced ejection fraction", "heart failure with reduced ejection fraction") share most
    words and count as one. A form used for under a tenth of the marks is left out."""
    from collections import Counter, defaultdict
    from . import graph_facts as gf
    rows = (query or gf.sparql)(gf.PREFIX + 'SELECT ?s ?full (COUNT(?e) AS ?n) WHERE { ?e dl:fullForm ?full ; dl:surfaceForm ?s } '
                                             'GROUP BY ?s ?full')
    forms = defaultdict(Counter)
    for row in rows:
        forms[row['s'].strip()][row['full'].strip()] += int(row['n'])
    meaning = {}
    for abbreviation, counts in forms.items():
        total = sum(counts.values())
        common = [form for form, n in counts.most_common() if n >= total / 10]
        top = set(plain(common[0]))
        same = all(len(top & set(plain(form))) / len(top | set(plain(form))) >= 0.5 for form in common[1:])
        meaning[abbreviation] = common[0] if same else None
    return meaning


def abbreviations_in(text, meanings):
    return {a: meanings[a] for a in dict.fromkeys(ABBREVIATION.findall(text)) if a in meanings}


def slots_of(facts, meanings=None):
    """One slot per part-1 fact, in rank order. The statement is the guideline's own sentence; code adds what kind it is,
    its grade, its source, whether it is only a "may" (so the caution check knows), and what its abbreviations mean
    according to the graph."""
    slots = []
    for n, fact in enumerate(facts, 1):
        statement = fact.get('statement') or re.search(r'statement: "(.*)"', fact['text'], re.S).group(1)
        grade = ', '.join(x for x in ([f'Class {c}' for c in fact.get('classes', [])] + [f'Level {lv}' for lv in fact.get('levels', [])]) if x)
        slots.append({'n': n, 'fact': fact['fact'], 'statement': statement,
                      'kind': 'recommendation' if fact.get('recommendation') else 'explanation', 'grade': grade,
                      'source': '; '.join(x for x in (fact.get('guideline', ''), ', '.join(f'p. {p}' for p in fact.get('pages', []))) if x),
                      'caution': bool(CAUTION_IN_FACT.search(statement)), 'abbreviations': abbreviations_in(statement, meanings or {})})
    return slots


def meaning_note(slot):
    """What the slot's abbreviations mean, as the graph says, for the writer and the judge."""
    notes = [f'{a} means {full}' if full else f'{a}: two meanings in the graph, keep it as written'
             for a, full in slot.get('abbreviations', {}).items()]
    return f' ({"; ".join(notes)})' if notes else ''


def evidence(slot):
    """The text the judge sees: the guideline sentence and the graph's meaning of its abbreviations."""
    return slot['statement'] + meaning_note(slot)


def slot_lines(slots):
    return '\n'.join(f'{s["n"]}. [{s["kind"]}{", " + s["grade"] if s["grade"] else ""}{", a may: keep the caution" if s["caution"] else ""}] '
                     f'{s["statement"]}{meaning_note(s)}' for s in slots)


# ---- 3. code checks ----------------------------------------------------------------------------------------------------------

def numbers(text):
    found = re.findall(r'\d+(?:[.,]\d+)?', text)
    found += [NUMBER_WORDS[w] for w in re.findall(r'[a-z]+', text.lower()) if w in NUMBER_WORDS]
    return {n.replace(',', '.') for n in found}


def plain(text):
    return [w for w in re.findall(r'[a-z0-9]+', text.lower())]


def stem(word):
    if word == 'not':  # "No driving" on screen for "Do not drive" spoken
        return 'no'
    word = word[:-1] if len(word) > 3 and word.endswith('s') else word
    return word[:-3] if len(word) > 5 and word.endswith('ing') else word


def code_checks(line, slot):
    """Problems a program can see without a model. Each is a reason the judge would also give, found for free."""
    problems = []
    narration = str(line.get('narration') or '').strip()
    if not narration:
        return ['The line is empty.']
    if len(narration.split()) > MOST_WORDS + 4:
        problems.append(f'The line has {len(narration.split())} words; at most {MOST_WORDS} are allowed.')
    extra = numbers(narration) - numbers(slot['statement'])
    if extra:
        problems.append(f'Numbers not in the fact: {", ".join(sorted(extra))}.')
    if slot['caution'] and not CAUTION_IN_LINE.search(narration):
        problems.append('The fact only says "may" or "can be considered"; the line states it as certain.')
    for abbreviation, full in slot.get('abbreviations', {}).items():
        if full is None and abbreviation not in narration:
            problems.append(f'The graph gives {abbreviation} two meanings, so the line must keep it as written.')
    return problems


def screen_problem(line, slot):
    """On-screen text is checked by code only: short, and every word comes from its line or its fact."""
    screen = str(line.get('on_screen_text') or '').strip()
    if not screen:
        return None
    if len(screen.split()) > SCREEN_WORDS + 2:
        return 'too long'
    allowed = {stem(w) for w in plain(line.get('narration', '')) + plain(evidence(slot))}
    new = [w for w in plain(screen) if w not in SMALL and stem(w) not in allowed]
    return f'words not in the line or fact: {", ".join(new)}' if new else None


# ---- 4. judge ----------------------------------------------------------------------------------------------------------------

def judge(judge_llm, line, slot, question, previous):
    """Code checks first; only a line they pass goes to the judge, and the judge sees only this line's own fact."""
    problems = code_checks(line, slot)
    if problems:
        return {'label': 'NOT_SUPPORTED', 'why': ' '.join(problems), 'decided_by': 'code check'}
    return judge_line(judge_llm, line['narration'], [evidence(slot)], topic=question, previous=previous)


def judge_plain(judge_llm, sentence, question, previous):
    """Opening and closing must hold no medical claim: the judge sees no facts, so any claim fails."""
    if not str(sentence or '').strip():
        return {'label': 'NOT_SUPPORTED', 'why': 'empty', 'decided_by': 'code check'}
    return judge_line(judge_llm, sentence, [], topic=question, previous=previous)


# ---- the run -----------------------------------------------------------------------------------------------------------------

def read_lines(answer, slots):
    """The writer's lines, kept only when they name a real slot, at most one line per slot, in slot order."""
    by_n = {s['n']: s for s in slots}
    seen, lines = set(), []
    for item in (answer or {}).get('lines', []) or []:
        n = item.get('fact') if isinstance(item, dict) else None
        n = int(n) if str(n).isdigit() else None
        if n in by_n and n not in seen:
            seen.add(n)
            lines.append({'slot': n, 'narration': str(item.get('narration') or '').strip(),
                          'on_screen_text': str(item.get('on_screen_text') or '').strip(), 'visual': str(item.get('visual') or '').strip()})
    return sorted(lines, key=lambda line: line['slot'])


def build_prompt(question, scenes):
    if not any(scene['medical'] for scene in scenes):
        return None
    blocks, spoken = [], 0
    for number, scene in enumerate(scenes, 1):
        spoken += len(scene['narration'].split())
        lines = [f'Scene {number}']
        if scene.get('on_screen_text'):
            lines.append(f'Heading: "{scene["on_screen_text"]}"')
        lines.append(f'Narration: "{scene["narration"]}"')
        if scene.get('visual'):
            lines.append(f'Visual: {scene["visual"]}')
        blocks.append('\n'.join(lines))
    seconds = round(spoken / 2.5) + 2 * len(scenes)  # 150 spoken words a minute, plus a short pause per scene
    return VIDEO_PROMPT.format(seconds=seconds, question=question.replace('"', "'"), scenes='\n\n'.join(blocks))


def run(question, part1, writer, judge_llm, progress=lambda event: None, meanings=None):
    """part1: the record of graph_facts.run for this question; meanings: graph_abbreviations(). Returns everything the page
    shows."""
    started, spent = time.perf_counter(), (used(writer) or 0, used(judge_llm) or 0)
    record = {'question': question, 'part1': part1, 'status': 'started'}
    slots = slots_of(part1.get('facts', []), meanings)
    record['slots'] = slots
    progress({'step': 'facts', 'slots': slots, 'concepts': part1.get('concepts', [])})
    if not slots:
        record.update(status='no_facts', why='The graph has no fact for this question, so nothing medical can be said.')
        progress({'step': 'done', 'record': record})
        return record

    progress({'step': 'writing'})
    raw = writer.generate(WRITER.format(question=question, slots=slot_lines(slots), most=MOST_WORDS, screen=SCREEN_WORDS, lines=MOST_LINES))
    answer = json_object(raw)
    if answer is None:
        record.update(status='no_script', why='The writer gave no readable answer.', writer_answer=raw)
        progress({'step': 'done', 'record': record})
        return record
    lines = read_lines(answer, slots)[:MOST_LINES]  # the cap holds even if the writer writes more
    by_n = {s['n']: s for s in slots}
    record.update(title=str(answer.get('title') or ''), skipped=[s for s in answer.get('skipped', []) or [] if isinstance(s, dict)])
    progress({'step': 'written', 'lines': lines, 'skipped': record['skipped']})

    opening, closing = OPENING, CLOSING
    record['opening'] = {'sentence': opening, 'label': 'NO_CLAIM', 'why': 'Fixed words written by code.', 'decided_by': 'code'}
    progress({'step': 'checked', 'part': 'opening', 'result': record['opening']})

    previous = opening
    for line in lines:
        slot = by_n[line['slot']]
        line['verdict'] = judge(judge_llm, line, slot, question, previous)
        line['status'] = 'kept' if line['verdict']['label'] in KEPT else 'failed'
        progress({'step': 'checked', 'part': 'line', 'line': line})
        previous = line['narration']

    failed = [line for line in lines if line['status'] == 'failed']
    if failed:
        progress({'step': 'rewriting', 'lines': [line['slot'] for line in failed]})
        items = '\n\n'.join(f'Fact {line["slot"]}: "{evidence(by_n[line["slot"]])}"\nYour line: "{line["narration"]}"\n'
                            f'Problem: {line["verdict"].get("why")}' for line in failed)
        fixed = {line['slot']: line for line in read_lines(json_object(writer.generate(REPAIR.format(
            items=items, most=MOST_WORDS, screen=SCREEN_WORDS))), slots)}
        for line in failed:
            new = fixed.get(line['slot'])
            line['first_try'] = {'narration': line['narration'], 'verdict': line['verdict']}
            if new is None:
                line['status'] = 'dropped'
                line['verdict'] = {**line['verdict'], 'why': f'{line["verdict"].get("why")} The rewrite gave no line for this fact.'}
            else:
                line.update(narration=new['narration'], on_screen_text=new['on_screen_text'] or line['on_screen_text'])
                before = [x['narration'] for x in lines if x['slot'] < line['slot'] and x['status'] in ('kept', 'rewritten')]
                previous = before[-1] if before else opening
                line['verdict'] = judge(judge_llm, line, by_n[line['slot']], question, previous)
                line['status'] = 'rewritten' if line['verdict']['label'] in KEPT else 'dropped'
            progress({'step': 'checked', 'part': 'line', 'line': line})

    kept = [line for line in lines if line['status'] in ('kept', 'rewritten')]
    previous = kept[-1]['narration'] if kept else opening
    record['closing'] = {'sentence': closing, 'label': 'NO_CLAIM', 'why': 'Fixed words written by code.', 'decided_by': 'code'}
    progress({'step': 'checked', 'part': 'closing', 'result': record['closing']})

    scenes, sources = [], []
    if record['opening']['label'] in KEPT:
        scenes.append({'narration': opening, 'on_screen_text': '', 'visual': '', 'medical': False})
    for line in kept:
        slot = by_n[line['slot']]
        problem = screen_problem(line, slot)
        line['screen_check'] = problem or 'ok'
        scenes.append({'narration': line['narration'], 'on_screen_text': '' if problem else line['on_screen_text'],
                       'visual': line['visual'], 'medical': line['verdict']['label'] == 'SUPPORTED'})
        sources.append({'scene': len(scenes), 'line': line['narration'], 'statement': slot['statement'], 'kind': slot['kind'],
                        'grade': slot['grade'], 'source': slot['source']})
    if record['closing']['label'] in KEPT:
        scenes.append({'narration': closing, 'on_screen_text': '', 'visual': '', 'medical': False})

    prompt = build_prompt(question, scenes)
    record.update(lines=lines, scenes=scenes, sources=sources, video_prompt=prompt,
                  why_no_prompt=None if prompt else 'No checked medical line is left to say.',
                  tokens={'writer': (used(writer) or 0) - spent[0], 'judge': (used(judge_llm) or 0) - spent[1]},
                  seconds=round(time.perf_counter() - started, 1), status='done')
    progress({'step': 'done', 'record': record})
    return record


def save(record, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding='utf-8')


# ---- both parts from one question ----------------------------------------------------------------------------------------------

CACHE = 'GALLM/runs/part2-model-calls'
RUNS = 'GALLM/runs/part2'


class Pipeline:
    """Part 1 (live on GraphDB) and part 2 (live on Groq) for any question. Loads the name index and the meaning look-up once."""

    def __init__(self, writer=None, judge_llm=None):
        from pathlib import Path
        from . import graph_facts as gf
        os.environ.pop('SSLKEYLOGFILE', None)  # Norton sets it, and Python's HTTPS then crashes (dev-machine note)
        try:
            import truststore  # Norton's HTTPS scanning needs the system certificates
            truststore.inject_into_ssl()
        except ImportError:
            pass
        if 'GROQ_API_KEY' not in os.environ:
            from dotenv import load_dotenv
            load_dotenv(Path('GAAPO/.env'), override=False)
        self.gf = gf
        self.names = gf.load_names()
        self.similar = gf.MeaningLookup(self.names)
        self.labels, self.total = gf.labels_of(gf.CLAIM_TYPES), gf.claim_total()
        self.meanings = graph_abbreviations()
        self.writer = writer or DiskCache(GroqModel(WRITER_MODEL), Path(CACHE + '-writer.jsonl'))
        self.judge = judge_llm or DiskCache(GroqModel(JUDGE_MODEL), Path(CACHE + '-judge.jsonl'))

    def ask(self, question, progress=lambda event: None):
        progress({'step': 'graph'})
        part1 = self.gf.run(question, self.names, self.similar, labels=self.labels, total=self.total)
        return run(question, part1, self.writer, self.judge, progress, self.meanings)


def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('question')
    parser.add_argument('--name', default='question')
    args = parser.parse_args()
    show = lambda event: print(event['step'], event.get('part', ''), (event.get('line') or {}).get('status', ''), flush=True)
    record = Pipeline().ask(args.question, show)
    save(record, Path(RUNS, f'{args.name}.json'))
    print(json.dumps({k: record.get(k) for k in ('status', 'tokens', 'seconds')}))
    print(record.get('video_prompt') or record.get('why_no_prompt') or record.get('why'))


if __name__ == '__main__':
    main()
