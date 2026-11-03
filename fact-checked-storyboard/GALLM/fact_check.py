"""Fact check for storyboard sentences. Each sentence is judged only against the guideline facts found in
Mahsa's knowledge graph for that request, never against the checker's own knowledge:
SUPPORTED (a fact states it, or it follows directly), CONTRADICTED (a fact conflicts with it),
NOT_SUPPORTED (the facts do not say) or, in the one-sentence check, NO_CLAIM (no medical statement, such as a
doctor's greeting; added 2026-11-01 for the doctor-talk style). Only SUPPORTED and NO_CLAIM sentences may stay
in a patient storyboard. The one-sentence SUPPORTED was tightened the same day: the doctor-talk storyboards had
passed small additions ("take it daily", a group's benefit said for everyone). It then cut a group claim once and
passed a near-identical one minutes later, so the one-sentence check became a checklist: the model fills in small
boxes and the code below decides the label. Its first proof (2026-11-02) called device descriptions NO_CLAIM and
everyday words "added", so those two boxes were reworded, and the model's same_people answer now counts only when the
fact names a group."""
import json
import re

LABELS = ('SUPPORTED', 'CONTRADICTED', 'NOT_SUPPORTED', 'NO_CLAIM')
KEPT = ('SUPPORTED', 'NO_CLAIM')
SYSTEM = ('You check sentences for patient-education videos about heart failure against guideline facts. '
          'Judge ONLY by the numbered facts given. Do not use your own medical knowledge, even when you are sure.')
CHECKLIST = '''Guideline facts:
{facts}

Sentence: "{sentence}"

Fill in this checklist about the sentence, in order:
1. medical_statement: true if the sentence says anything about the body, an illness, a test, a device, a medicine
   or a treatment (what it is, what it does or causes, how it works, how it affects daily life), or gives health
   advice (what to take, eat, do or avoid). false only for a greeting, a thank-you, everyday politeness, naming the
   topic of the talk, or inviting the patient to ask their care team (also "ask your care team if <a medicine> is
   right for you"), when the sentence says nothing more than that. If any part of it says more, or you are unsure: true.
2. fact: the number of the one fact closest to the sentence (the one that says the same or the opposite), or null
   if no fact is about it.
3. fact_is_about: the people that fact holds for, from the fact or its section (for example "patients with HFrEF",
   "patients treated with diuretics", "patients after a heart attack"), or "everyone" if it names no group. The
   videos are for people with heart failure, so a fact for all heart-failure patients is about everyone. A fact that
   only says what a medicine does, for whoever takes it, names no group; a recommendation for a group does.
4. sentence_is_about: the people the sentence is about, or "everyone" if it names no group. A sentence spoken to
   "you" that names no group is about everyone.
5. same_people: true if the sentence is about the same people as the fact, or a smaller group of them (true if the
   fact is about everyone).
6. added: the information in the sentence that the fact does not give: a time, frequency, amount or condition, a
   cause or result, stronger wording ("always" or "cures" where the fact says "may" or "helps"), or advice to
   patients that the fact does not give to patients. [] if nothing is added. Words are not information: the fact's
   own information in everyday words (a plainer word, a common word for a medical one) is not added.
7. conflicts: true if the fact says something that conflicts with the sentence.
8. states_it: true if the fact says what the sentence says, or it follows directly from the fact.
Answer with JSON only: {{"medical_statement": ..., "fact": ..., "fact_is_about": "...", "sentence_is_about": "...",
"same_people": ..., "added": [...], "conflicts": ..., "states_it": ..., "reason": "<one short sentence>"}}'''
BOXES = ('medical_statement', 'fact', 'fact_is_about', 'sentence_is_about', 'same_people', 'added', 'conflicts', 'states_it')
EVERYONE = ('everyone', 'everybody', 'anyone', 'all', 'all patients', 'all people', 'you', 'patients', 'people',
            'people with heart failure', 'patients with heart failure', 'heart failure patients', 'all heart failure patients')
NOTHING = ('', 'none', 'nothing', 'null', '[]')


def numbered(facts):
    return '\n'.join(f'{n}. {fact}' for n, fact in enumerate(facts, 1))


def without_thinking(answer):
    """Some models (Qwen) think aloud in <think> tags before answering; only the answer counts."""
    return re.sub(r'<think>.*?</think>', '', answer or '', flags=re.S)


def parse(answer):
    """The judge's JSON verdict; label ERROR when there is none, which counts as not supported."""
    match = re.search(r'\{.*\}', without_thinking(answer), re.S)
    try:
        verdict = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        verdict = {}
    label = str(verdict.get('label', '')).strip().upper().replace(' ', '_')
    return {'label': label if label in LABELS else 'ERROR', 'fact': verdict.get('fact'),
            'reason': verdict.get('reason') or (answer or '')[:300]}


def decide(boxes, facts):
    """The label from a filled-in checklist. A box left empty or unclear counts against the sentence, except
    medical_statement: only an explicit false makes NO_CLAIM."""
    yes = lambda box: boxes.get(box) is True or str(boxes.get(box)).strip().lower() in ('true', 'yes')
    group = lambda box: re.sub(r'[\s\-‑]+', ' ', str(boxes.get(box) or 'everyone')).strip().lower().rstrip('.')
    added = boxes.get('added') if isinstance(boxes.get('added'), list) else [boxes.get('added')]
    added = [a for a in added if str(a if a is not None else '').strip().lower() not in NOTHING]
    fact = boxes.get('fact')
    if boxes.get('medical_statement') is False or str(boxes.get('medical_statement')).strip().lower() in ('false', 'no'):
        return 'NO_CLAIM'
    if not (str(fact).isdigit() and 0 < int(fact) <= len(facts)):
        return 'NOT_SUPPORTED'
    if yes('conflicts'):
        return 'CONTRADICTED'
    # a fact for everyone holds for any sentence's people; only a fact with a group needs the model's same_people
    everyone = bool(str(boxes.get('fact_is_about') or '').strip()) and group('fact_is_about') in EVERYONE
    if not yes('states_it') or added or not (everyone or yes('same_people')):
        return 'NOT_SUPPORTED'
    if not everyone and group('sentence_is_about') in EVERYONE:
        return 'NOT_SUPPORTED'  # the model wrote down a group for the fact and none for the sentence
    return 'SUPPORTED'


def check_sentence(llm, sentence, facts):
    """Verdict for one sentence, with the filled-in checklist; with no guideline facts nothing could support it."""
    if not facts:
        return {'label': 'NOT_SUPPORTED', 'fact': None, 'reason': 'No guideline facts were found for this request.'}
    try:
        answer = llm.generate(CHECKLIST.format(facts=numbered(facts), sentence=sentence), system=SYSTEM, temperature=0)
    except ValueError:  # an empty answer (gpt-oss can think until its token budget runs out); not cached, so asked again next run
        return {'label': 'ERROR', 'fact': None, 'reason': 'The checker gave an empty answer.'}
    match = re.search(r'\{.*\}', without_thinking(answer), re.S)
    try:
        boxes = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        boxes = None
    if not isinstance(boxes, dict):
        return {'label': 'ERROR', 'fact': None, 'reason': (answer or '')[:300]}
    return {'label': decide(boxes, facts), 'fact': boxes.get('fact'), 'reason': boxes.get('reason') or '',
            'checklist': {box: boxes.get(box) for box in BOXES}}


BATCH_QUESTION = '''Guideline facts:
{facts}

Sentences (judge each one on its own):
{sentences}

Label every sentence:
- SUPPORTED: one fact states it, or it follows directly from one fact. Simpler wording is fine.
- CONTRADICTED: a fact says something that conflicts with it.
- NOT_SUPPORTED: the facts do not say it either way.
Answer with JSON only: {{"verdicts": [{{"sentence": <its number>, "label": "...", "fact": <number of the fact used, or null>, "reason": "<one short sentence>"}}]}}'''


def check_sentences(llm, sentences, facts):
    """Verdicts for all sentences of one storyboard in a single call (fewer calls on Groq's free plan).
    A sentence the answer leaves out gets ERROR, which counts as not supported."""
    if not sentences:
        return []
    if not facts:
        return [check_sentence(llm, s, facts) for s in sentences]
    answer = llm.generate(BATCH_QUESTION.format(facts=numbered(facts), sentences=numbered(sentences)),
                          system=SYSTEM, temperature=0)
    match = re.search(r'\{.*\}', without_thinking(answer), re.S)
    try:
        listed = json.loads(match.group(0)).get('verdicts', []) if match else []
    except (json.JSONDecodeError, AttributeError):
        listed = []
    by_number = {}
    for item in listed if isinstance(listed, list) else []:
        if isinstance(item, dict) and str(item.get('sentence')).isdigit():
            by_number.setdefault(int(item['sentence']), parse(json.dumps(item)))
    missing = {'label': 'ERROR', 'fact': None, 'reason': 'No verdict for this sentence in the answer.'}
    return [by_number.get(n, missing) for n in range(1, len(sentences) + 1)]
