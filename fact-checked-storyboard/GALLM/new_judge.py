"""The judge (built 2026-11-04): checks one line against the facts it is given and decides whether it may stay.
Research behind the design: long prompts change strictness rather than accuracy (Vaugrante 2026), rules tuned without human
answers do not move a judge closer to humans (Han and Kim 2026), the question is asked a second way because framing can flip a
verdict (Hwang 2026), and code checks the quote because a model's explanation can be wrong (Joseph 2024, Zhang 2026).
The examples are the user's own practice answers. Two of them, P02 and P06, are also starting sentences of the code-made test,
so tests made from those two are left out of the score: see EXAMPLE_SEEDS."""
import re

from .text_tools import json_object, numbered

SYSTEM = ('You check lines from patient-education videos against guideline facts. '
          'Use only the facts given, never your own medical knowledge.')
EXAMPLE_SEEDS = ('They will assess your risks and decide together with you.',
                 'Mineralocorticoid antagonists, like spironolactone, help your body remove extra salt and water.')
JUDGE = '''Decide if this line can stay in the video. The topic and the line before say who and what the line is about, but they are not evidence.

Verdicts:
- PLAIN: nothing medical. Greetings, thanks, titles, naming the topic and "ask your care team" are plain.
- SUPPORTED: one fact says everything medical in the line. Simpler words are fine.
- PARTLY: a fact covers some of it, but the line adds a detail, is stronger than the fact, leaves out the fact's group, or adds a cause.
- NOT: no fact covers it.
- AGAINST: a fact says the opposite.

Examples:
Line: ACE inhibitors and your heart (on screen)
Verdict: PLAIN, a title.

Fact: In patients who are candidates for CRT, implantation of a CRT-D should be considered after individual risk assessment and using shared decision-making.
Line before: If you need both a pacemaker and a defibrillator, doctors may recommend a CRT-D.
Line: They will assess your risks and decide together with you.
Verdict: SUPPORTED, simpler words for risk assessment and shared decision-making.

Fact: Mineralocorticoid Antagonists (MRA) spironolactone, eplerenone Aldactone, Inspra Help the body get rid of salt and fluids.
Line: Mineralocorticoid antagonists, like spironolactone, help your body remove extra salt and water.
Verdict: SUPPORTED, the same claim.

Fact: Sacubitril/valsartan is recommended as a replacement for an ACE inhibitor to reduce the risk of HF hospitalization and death in patients with HFrEF.
Line: It can reduce hospital visits and death.
Verdict: PARTLY, the fact is for patients with HFrEF and the line names no group.

Now the real one.
Topic: {topic}
Line before: {previous}
Facts:
{facts}
Line ({where}): {sentence}

Answer with JSON only: {{"verdict": "...", "fact": <number of the fact used, or null>, "quote": "<exact words copied from that fact, or empty>", "reason": "<one short sentence>"}}'''
SECOND = '''Facts:
{facts}

Video topic: {topic}
Line: {sentence}

{question}
Answer with JSON only: {{"not_in_facts": "<the words, or empty>"}}'''
# One wording missed "every week" in round 2 and another cut a good line in round 1, so the same question is asked in three
# wordings and two of them must agree (Subasic 2026: several reworded prompts agree with humans better than one).
QUESTIONS = ('Name anything medical in the line that the facts do not say, or say only for a smaller group. Simpler words for what a fact says do not count.',
             'Name anything medical in the line that the facts do not say, say less strongly, or say only for a smaller group. Simpler or shorter words for what a fact says do not count.',
             'Copy any words from the line that add a detail, a stronger claim, a wider group or a cause beyond the facts. Rewording a fact is fine.')
STAY = {'PLAIN': 'NO_CLAIM', 'SUPPORTED': 'SUPPORTED'}
NOTHING = ('', 'none', 'nothing', 'null', 'n/a')


def plain_words(text):
    return re.findall(r'[a-z0-9]+', str(text).lower())


def quote_is_real(quote, fact):
    """The quote must be in the fact: all its words, side by side, so a made-up quote does not pass. A model may shorten a
    quote with dots, so each part is looked up on its own."""
    parts = [' '.join(plain_words(part)) for part in re.split(r'\.\.\.|…', str(quote))]
    parts = [part for part in parts if part]
    text = ' '.join(plain_words(fact))
    return bool(parts) and len(' '.join(parts)) > 3 and all(part in text for part in parts)


def ask(llm, prompt):
    try:
        return json_object(llm.generate(prompt, system=SYSTEM, temperature=0))
    except ValueError:  # gpt-oss can think until its token budget runs out
        return None


def judge_line(llm, sentence, facts, topic='', previous='', field='narration'):
    """The verdict for one line: only SUPPORTED and NO_CLAIM stay."""
    listed = numbered(facts)
    answer = ask(llm, JUDGE.format(topic=topic, previous=previous or '(none)', facts=listed, sentence=sentence,
                                   where='spoken' if field == 'narration' else 'on screen'))
    if answer is None:
        return {'label': 'ERROR', 'why': 'The judge gave no readable answer.', 'decided_by': 'judge'}
    verdict = str(answer.get('verdict', '')).strip().upper()
    note = answer.get('reason') or ''
    if verdict not in STAY:
        return {'label': 'CONTRADICTED' if verdict == 'AGAINST' else 'NOT_SUPPORTED', 'verdict': verdict, 'why': note, 'decided_by': 'judge'}
    if verdict == 'SUPPORTED':
        number, quote = answer.get('fact'), answer.get('quote') or ''
        fact = facts[int(number) - 1] if str(number).isdigit() and 1 <= int(number) <= len(facts) else ''
        if not quote_is_real(quote, fact):
            return {'label': 'NOT_SUPPORTED', 'verdict': verdict, 'why': f'The quote is not in the fact: {quote}', 'decided_by': 'quote check'}
    if verdict == 'PLAIN':  # round 1: the second question only ever cut plain lines wrongly ("care team"), and the judge alone caught 8 of 8 hidden claims
        return {'label': 'NO_CLAIM', 'verdict': verdict, 'why': note, 'decided_by': 'judge'}
    named = []
    for question in QUESTIONS:
        second = ask(llm, SECOND.format(facts=listed, topic=topic, sentence=sentence, question=question))
        extra = str((second or {}).get('not_in_facts', '')).strip()
        if second is None or extra.lower() not in NOTHING:
            named.append(extra or 'no readable answer')
    if len(named) >= 2:
        return {'label': 'NOT_SUPPORTED', 'verdict': verdict, 'why': f'Asked the other way, {len(named)} of 3 named: {"; ".join(named)}', 'decided_by': 'second question'}
    return {'label': STAY[verdict], 'verdict': verdict, 'fact': answer.get('fact'), 'quote': answer.get('quote') or '', 'reason': note, 'why': note,
            'decided_by': 'judge', 'named': named}  # fact and quote are for the source link in the report; they change no decision
