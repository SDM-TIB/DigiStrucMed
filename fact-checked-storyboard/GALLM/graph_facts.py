"""Part 1 of the graph pipeline (2026-11-18): from a topic request to at most 10 guideline facts, every one read from the
knowledge graph in GraphDB (repository Guideline_KG). No sentence search and no model in this part.

1. Concepts of the request: the KG's own names (UMLS names of the concepts the statements name, the guidelines' words the
   linker tied to a concept, abbreviations with their full forms), matched word for word in American spelling, longest name
   first. Words left over are matched by meaning (MiniLM, as KG-RAG does, Soman et al. 2024) to catch other spellings.
2. Fixed SPARQL: the claims (treats, prevents, side effect, interaction, outcome) that name the concepts, and the
   recommendation readings (do or avoid, what, for whom) whose target, group, condition or outcome is one of them, in English
   statements of professional guidelines. Lifestyle advice (smoking, exercise, driving) is in the readings, not the claims.
   The queries never change; only the concept codes are filled in.
3. Stop rule: a concept with no claims steps out along dl:broader one level at a time and stops at the first level that has
   claims (ceiling: 3 levels, the depth ToG and PoG settled on).
4. Each fact keeps its statement: recommendation or explanation (the KG's own type), class, level, guideline, page, and the
   path it came through.
5. At most 10 facts: reached directly before via broader; then by how specific the request concepts in the fact are (a
   concept that few claims name, like SGLT2 inhibitor, counts more than one that thousands name, like heart failure: the
   inverse document frequency of search engines, counted on the graph's claims); then graded statements, then recommendations.

Papers behind each step: reports/Pipeline step references.md (in the user's reports folder).

Run: python -m GALLM.graph_facts                     (all ruler topics)
     python -m GALLM.graph_facts --only ruler-016    (some)
     python -m GALLM.graph_facts --name x --request "Explain SGLT2 inhibitors"
"""
import argparse
import http.client
import json
import math
import re
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path

from GALLM.text_tools import COMMON, GENERIC, singular, words

ENDPOINT = 'http://127.0.0.1:7200/repositories/Guideline_KG'
DL = 'http://digistructmed.org/digilearn/terms/'
CONCEPT = 'http://digistructmed.org/digilearn/resource/concept/'
PREFIX = f'PREFIX dl: <{DL}>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n'
NAMES_FILE = Path('GALLM/data/graph_names.json')
VECTORS_FILE = Path('GALLM/data/graph_names_vectors.npy')
RULER = Path('GALLM/data/ruler_draft.json')
OUT = Path('GALLM/runs/graph-facts-2026-11-18')

MOST_FACTS = 10
MAX_WORDS = 8
MOST_LEVELS = 3
SIMILAR = 0.90  # meaning look-up: only near-identical names count (spelling variants), not related ideas
CLAIM_TYPES = ('TreatmentClaim', 'PreventionClaim', 'SideEffectClaim', 'InteractionClaim', 'OutcomeClaim')
PARTS = ('treatmentConcept', 'conditionConcept', 'secondMedication', 'outcomeConcept')
ROLES = ('hasTarget', 'hasPopulation', 'hasCondition', 'hasOutcome', 'hasException')
SHORT = 4  # a one-word name this short is an abbreviation only when the request writes it in capitals ("ICD", not "tips")
GRADE_ORDER = {'I': 0, 'IIa': 1, 'IIb': 2, 'III': 3}
# the ruler's fixed opening is the task, not the topic ("Create a brief patient-education scene that explains ...")
TASK = re.compile(r'^\s*create a brief patient[- ]education scene that (?:explains|introduces|gives)\b', re.I)
STOP = set('''a an the and or of to in on for with without how why what which who whom whose that this these those is are was were
be been can could may might must should would will it its their they them people person using use used about from by as at into
after before when whether also between while both two three four five first main some other gives give example examples simple
word words help helps work works way ways explain explains explained introduce introduces module scene brief create get gets make
makes do does doing done not no yes very more most less much many there here than then so such if only just each every own one
s t'''.split())
SKIP = GENERIC | COMMON | STOP


def sparql(query, endpoint=ENDPOINT):
    """Rows of a SELECT on GraphDB, each a dict of variable -> value (IRIs and literals as plain strings). Plain http.client:
    urllib.request builds an HTTPS certificate store first, and on this PC that crashes Python (OPENSSL_Applink, 2026-11-18)."""
    url = urllib.parse.urlsplit(endpoint)
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=300)
    try:
        connection.request('POST', url.path, body=urllib.parse.urlencode({'query': query}),
                           headers={'Accept': 'application/sparql-results+json',
                                    'Content-Type': 'application/x-www-form-urlencoded'})
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f'GraphDB answered {response.status}: {body[:300]!r}')
    finally:
        connection.close()
    result = json.loads(body)
    return [{k: v['value'] for k, v in row.items()} for row in result['results']['bindings']]


def cui_of(iri):
    return iri.rsplit('/', 1)[-1]


def iri(cui):
    return f'<{CONCEPT}{cui}>'


def norm(text):
    """The words of a name or a request as the index keys them: lower case, American spelling, singular."""
    return tuple(singular(w) for w in words(text, english=True))


def values(cuis):
    return ' '.join(iri(c) for c in sorted(cuis))


# ---- 1. the KG's names ---------------------------------------------------------------------------------------------------

NAME_QUERIES = {
    # UMLS names of every concept a statement names
    'name': PREFIX + 'SELECT DISTINCT ?c ?s WHERE { ?f dl:mentionsConcept ?c . ?c dl:name ?s }',
    # the guidelines' own words the linker tied to a concept, with how often
    'words': PREFIX + '''SELECT ?c ?s (COUNT(?m) AS ?n) WHERE { ?m dl:surfaceForm ?s ; dl:hasLinkSet ?l . ?l dl:effectiveConcept ?c }
                          GROUP BY ?c ?s''',
    # abbreviations the model marked, with the full form the guideline gives, both tied to the phrase's concept
    'abbreviation': PREFIX + '''SELECT DISTINCT ?c ?s ?full WHERE { ?e dl:fullForm ?full ; dl:surfaceForm ?s ; dl:headConcept ?c }''',
}


def build_names(query=sparql):
    """name key -> {cui: weight}, read from the KG. A concept's own name weighs 1, the guidelines' words weigh how often they
    were linked to it, so a phrase linked to two concepts goes to the one the guidelines use it for most."""
    names = defaultdict(Counter)
    for row in query(NAME_QUERIES['name']):
        names[' '.join(norm(row['s']))][cui_of(row['c'])] += 1
    for row in query(NAME_QUERIES['words']):
        names[' '.join(norm(row['s']))][cui_of(row['c'])] += int(row['n'])
    for row in query(NAME_QUERIES['abbreviation']):
        for text in (row['s'], row['full']):
            names[' '.join(norm(text))][cui_of(row['c'])] += 1
    return {key: dict(counts) for key, counts in names.items() if key}


def load_names(rebuild=False, query=sparql):
    if rebuild or not NAMES_FILE.exists():
        NAMES_FILE.write_text(json.dumps(build_names(query), ensure_ascii=False), encoding='utf-8')
    return json.loads(NAMES_FILE.read_text(encoding='utf-8'))


def topic_words(request):
    return norm(TASK.sub('', request))


def best(counts):
    return max(sorted(counts), key=lambda c: counts[c])


def word_forms(word):
    """Other forms of one English word, as a spelling variant: "drive" -> "driving", "driving" -> "drive". Plain word + "ing"
    is left out, because it turned "car" into "caring" (ruler-003, "a car engine")."""
    forms = [word[:-1] + 'ing' if word.endswith('e') and len(word) > 3 else '']
    if word.endswith('ing') and len(word) > 5:
        forms += [word[:-3], word[:-3] + 'e']
    return [f for f in forms if f]


def word_form(word, names):
    """The word as the KG names it. A user writes "drive" where the guidelines say "driving" (2026-11-18: the question "When
    can I drive again after getting an ICD?" found only the ICD). A word the KG already names stays as it is."""
    if word in names or word in SKIP:
        return word
    return next((f for f in word_forms(word) if f in names and f not in SKIP), word)


def find_concepts(request, names, similar=None):
    """The request's concepts, longest KG name first. Each: the words, the concept, and how it was found ('name' or 'similar'
    with its score). As in the linker, a generic word ("heart", "patients", "treatment") is never a concept on its own, but it
    is inside a longer name ("heart failure"); a run of stop words ("how to") never is."""
    tokens = tuple(word_form(t, names) for t in topic_words(request))
    # words written in capitals, hyphenated ones split: "S-ICD", "EV-ICD", "CRT-D" and "ICDs" all give "icd" / "crt"
    capitals = {singular(w.lower()) for w in re.findall(r'(?<![A-Za-z])[A-Z][A-Z0-9]+s?(?![a-z])', request)}
    found, covered, i = [], set(), 0
    while i < len(tokens):
        for n in range(min(MAX_WORDS, len(tokens) - i), 0, -1):
            key = ' '.join(tokens[i:i + n])
            short = n == 1 and len(key) <= SHORT and key not in capitals
            if key in names and not short and not (n == 1 and key in SKIP) and not all(t in STOP for t in tokens[i:i + n]):
                found.append({'words': key, 'cui': best(names[key]), 'method': 'name'})
                covered.update(range(i, i + n))
                i += n
                break
        else:
            i += 1
    if similar:
        # a phrase that is a KG name word for word was already judged above (and left out on purpose: "tips", "lead"), and a
        # short word is never guessed by meaning
        phrases = [p for p in leftovers(tokens, covered) if p not in names and not (' ' not in p and len(p) <= SHORT)]
        found += similar(phrases)
    return dedupe(found)


def leftovers(tokens, covered):
    """Runs of uncovered, non-generic words, as phrases of one to three words, for the meaning look-up."""
    runs, run = [], []
    for i, token in enumerate(tokens):
        if i in covered or token in SKIP or len(token) < 3:
            if run:
                runs.append(run)
            run = []
        else:
            run.append(token)
    if run:
        runs.append(run)
    return [' '.join(r[s:s + n]) for r in runs for n in range(min(3, len(r)), 0, -1) for s in range(len(r) - n + 1)]


def dedupe(found):
    seen, out = set(), []
    for f in found:
        if f['cui'] not in seen:
            seen.add(f['cui'])
            out.append(f)
    return out


class MeaningLookup:
    """Leftover words matched to the KG's names by meaning (MiniLM, as in KG-RAG). Only names at least SIMILAR alike count, so
    "ischaemic cardiomyopathy" finds "ischemic cardiomyopathy" but "pill organizer" does not become "pill"."""

    def __init__(self, names, encoder=None, vectors=None):
        import numpy as np
        self.np = np
        self.keys = sorted(names)
        self.names = names
        if encoder is None:
            from GALLM.text_tools import Encoder
            encoder = Encoder()
        self.encoder = encoder
        if vectors is None:
            vectors = self.cached_vectors()
        self.vectors = vectors

    def cached_vectors(self):
        if VECTORS_FILE.exists():
            vectors = self.np.load(VECTORS_FILE)
            if len(vectors) == len(self.keys):
                return vectors
        vectors = self.encoder(self.keys)
        self.np.save(VECTORS_FILE, vectors)
        return vectors

    def __call__(self, phrases):
        if not phrases:
            return []
        scores = self.encoder(phrases) @ self.vectors.T
        found, used = [], set()
        for i in self.np.argsort(-scores.max(axis=1)):
            j = int(scores[i].argmax())
            words_ = phrases[i].split()
            if scores[i, j] < SIMILAR or used & set(words_):
                continue
            used.update(words_)
            found.append({'words': phrases[i], 'cui': best(self.names[self.keys[j]]), 'method': 'similar',
                          'matched': self.keys[j], 'score': round(float(scores[i, j]), 3)})
        return found


# ---- 2. fixed SPARQL -------------------------------------------------------------------------------------------------------

# Query A — CLAIMS: claims that name a question concept (treats, prevents, side effect, interaction, outcome),
# with the guideline statement each came from. English statements from professional guidelines only.
CLAIMS = PREFIX + '''SELECT ?claim ?type ?fact ?part ?concept WHERE {
  VALUES ?c { %s }
  VALUES ?p { ''' + ' '.join(f'dl:{p}' for p in PARTS) + ''' }
  VALUES ?type { ''' + ' '.join(f'dl:{t}' for t in CLAIM_TYPES) + ''' }
  VALUES ?part { ''' + ' '.join(f'dl:{p}' for p in PARTS) + ''' }
  ?claim ?p ?c ; a ?type ; dl:inStatement ?fact ; ?part ?concept .
  ?fact dl:language "en" ; dl:inGuideline ?g .
  ?g dl:documentType "professional" .
}'''

# Query B — RECOMMENDATION READINGS: do / avoid what, for whom, under which condition (lifestyle advice such as
# smoking, exercise, driving lives here). Same English + professional-guideline filter.
READINGS = PREFIX + '''SELECT ?reading ?fact ?role ?concept ?advice ?operation WHERE {
  VALUES ?c { %s }
  VALUES ?r { ''' + ' '.join(f'dl:{r}' for r in ROLES) + ''' }
  VALUES ?role { ''' + ' '.join(f'dl:{r}' for r in ROLES) + ''' }
  ?reading a dl:RecommendationReading ; dl:status "PRESENT" ; dl:inStatement ?fact ; ?r ?e ; ?role ?e2 .
  ?e dl:headConcept ?c . ?e2 dl:headConcept ?concept .
  OPTIONAL { ?reading dl:advice ?advice }
  OPTIONAL { ?reading dl:operation ?operation }
  ?fact dl:language "en" ; dl:inGuideline ?g .
  ?g dl:documentType "professional" .
}'''

# Helper — STATEMENTS: for each statement found by A or B, the guideline's own sentence and what the graph stores about it:
# recommendation or explanation, class and level of recommendation (the grade), page, guideline title and year.
STATEMENTS = PREFIX + '''SELECT ?fact ?text ?rec ?grade ?level ?page ?title ?year WHERE {
  VALUES ?fact { %s }
  ?fact dl:text ?text ; dl:inGuideline ?g .
  ?g dl:title ?title .
  OPTIONAL { ?g dl:year ?year }
  OPTIONAL { ?fact dl:pageNumber ?page }
  OPTIONAL { ?fact dl:hasClassOfRecommendation ?gc . BIND(STRAFTER(STR(?gc), "class-of-recommendation/") AS ?grade) }
  OPTIONAL { ?fact dl:hasLevelOfEvidence ?lv . BIND(STRAFTER(STR(?lv), "level-of-evidence/") AS ?level) }
  BIND(EXISTS { ?fact a dl:Recommendation } AS ?rec)
}'''

# Helper — BROADER: the parent concept, used only when a concept has no facts (the stop rule in walk(), at most 3 levels).
BROADER = PREFIX + 'SELECT ?c ?parent WHERE { VALUES ?c { %s } ?c dl:broader ?parent }'

# Helper — DEFINITIONS: the UMLS definition of each question concept (4,275 concepts have one, mostly from MeSH). Part 2 may
# turn one into a plain "what it is" line, checked against the definition like any fact.
DEFINITIONS = PREFIX + 'SELECT ?c ?def WHERE { VALUES ?c { %s } ?c dl:definition ?def }'

# Helper — TERMS: the UMLS definitions of the concepts each kept statement names, so part 2 can say a technical word in
# plain words ("arrhythmia" -> "a heart rhythm problem") that the judge can check, the way it checks abbreviation meanings.
TERMS = PREFIX + 'SELECT ?f ?c ?def WHERE { VALUES ?f { %s } ?f dl:mentionsConcept ?c . ?c dl:definition ?def }'
MOST_TERMS = 3  # per statement
# Off since 2026-11-21: the first live run passed a wrong graph link to writer and judge ("lead", the ICD wire, defined as the
# metal), and the definitions of labels like "primary prevention" were too general to help. Switch on only after the
# concept links are validated.
USE_TERMS = False

NAMES_OF = PREFIX + 'SELECT ?c ?name WHERE { VALUES ?c { %s } ?c dl:name ?name }'

LABELS = PREFIX + 'SELECT ?x ?label WHERE { VALUES ?x { %s } ?x rdfs:label ?label }'

ALL_CLAIMS = PREFIX + ('SELECT (COUNT(DISTINCT ?claim) AS ?n) WHERE { VALUES ?type { '
                       + ' '.join(f'dl:{t}' for t in CLAIM_TYPES) + ' } ?claim a ?type }')


def claim_total(query=sparql):
    return int(query(ALL_CLAIMS)[0]['n'])


def claims_of(cuis, query=sparql):
    """claim or reading IRI -> {'type', 'fact', 'parts': {part or role: {cui}}} for every claim that names one of the concepts
    and every recommendation reading with one of them as target, group, condition, outcome or exception."""
    if not cuis:
        return {}
    claims = {}
    for row in query(CLAIMS % values(cuis)):
        claim = claims.setdefault(row['claim'], {'type': row['type'].rsplit('/', 1)[-1], 'fact': row['fact'],
                                                 'parts': defaultdict(set)})
        claim['parts'][row['part'].rsplit('/', 1)[-1]].add(cui_of(row['concept']))
    for row in query(READINGS % values(cuis)):
        reading = claims.setdefault(row['reading'], {'type': 'RecommendationReading', 'fact': row['fact'],
                                                     'parts': defaultdict(set), 'advice': row.get('advice', ''),
                                                     'operation': row.get('operation', '')})
        reading['parts'][row['role'].rsplit('/', 1)[-1]].add(cui_of(row['concept']))
    return claims


def walk(cuis, query=sparql):
    """The stop rule. Each request concept with claims stays as it is (path: itself). One without steps out along dl:broader
    a level at a time and stops at the first level where some parent has claims, or after MOST_LEVELS levels.
    Returns {request cui: [(reached cui, path)]} and all claims found."""
    reached, claims = {}, {}
    for cui in cuis:
        frontier, level = {cui: [cui]}, 0
        while frontier and level <= MOST_LEVELS:
            found = claims_of(set(frontier), query)
            hit = {c for claim in found.values() for part in claim['parts'].values() for c in part} & set(frontier)
            if hit:
                reached[cui] = [(c, frontier[c]) for c in sorted(hit)]
                claims.update({k: v for k, v in found.items()
                               if any(c in hit for part in v['parts'].values() for c in part)})
                break
            level += 1
            if level > MOST_LEVELS:
                break
            nxt = {}
            for row in query(BROADER % values(frontier)):
                parent = cui_of(row['parent'])
                if parent not in nxt:
                    nxt[parent] = frontier[cui_of(row['c'])] + [parent]
            frontier = nxt
        reached.setdefault(cui, [])
    return reached, claims


def statements_of(facts, query=sparql):
    """fact IRI -> its words, recommendation or explanation, classes, levels, pages, guideline and year."""
    out = {}
    for start in range(0, len(facts), 200):
        chunk = sorted(facts)[start:start + 200]
        for row in query(STATEMENTS % ' '.join(f'<{f}>' for f in chunk)):
            s = out.setdefault(row['fact'], {'text': row['text'], 'recommendation': row['rec'] == 'true', 'classes': set(),
                                             'levels': set(), 'pages': set(), 'guideline': row['title'],
                                             'year': row.get('year', '')})
            for key, name in (('grade', 'classes'), ('level', 'levels'), ('page', 'pages')):
                if row.get(key):
                    s[name].add(row[key])
    return out


def names_of(cuis, query=sparql):
    return {cui_of(r['c']): r['name'] for r in query(NAMES_OF % values(cuis))} if cuis else {}


def labels_of(local_names, query=sparql):
    rows = query(LABELS % ' '.join(f'dl:{n}' for n in sorted(local_names)))
    return {r['x'].rsplit('/', 1)[-1]: r['label'] for r in rows}


# ---- 3. choose at most 10 ---------------------------------------------------------------------------------------------------

def specificity(reached, claims, total):
    """request cui -> log(total claims / claims naming it): rare concepts weigh more (inverse document frequency)."""
    named = Counter(c for claim in claims.values() for c in {c for part in claim['parts'].values() for c in part})
    return {request_cui: max(math.log(total / (1 + sum(named[cui] for cui, _ in hits))), 0.0)
            for request_cui, hits in reached.items() if hits}


def facts_of(reached, claims, statements, weight=None):
    """One fact per statement, with the claims in it, the request concepts they name, their weight, and the shortest path."""
    weight = weight or {}
    via = {}  # reached cui -> (request cui, path)
    for request_cui, hits in reached.items():
        for cui, path in hits:
            via.setdefault(cui, (request_cui, path))
    facts = {}
    for iri_, claim in claims.items():
        concepts = {c for part in claim['parts'].values() for c in part}
        hits = {via[c][0] for c in concepts if c in via}
        if not hits or claim['fact'] not in statements:
            continue
        fact = facts.setdefault(claim['fact'], {'statement': statements[claim['fact']], 'claims': [], 'joins': set(),
                                                'hops': 99, 'paths': {}})
        fact['claims'].append({'iri': iri_, **claim})
        fact['joins'] |= hits
        for c in concepts:
            if c in via:
                request_cui, path = via[c]
                fact['hops'] = min(fact['hops'], len(path) - 1)
                fact['paths'][request_cui] = path
    for fact in facts.values():
        fact['weight'] = round(sum(weight.get(c, 1.0) for c in fact['joins']), 3)
        fact['top'] = max((weight.get(c, 1.0) for c in fact['joins']), default=0.0)
    return facts


def rank_key(fact):
    s = fact['statement']
    grade = min((GRADE_ORDER.get(g, 9) for g in s['classes']), default=9)
    return (fact['hops'], -round(fact.get('top', 0.0)), grade, -fact.get('weight', len(fact['joins'])),
            not s['recommendation'], len(s['text']))


# The same recommendation is often stored several times: from two guidelines, or as the sentence and as its table row
# ("... in patients with HFrEF. 707,730; I; B"). Compared without a leading "Recommendations:" and without the trailing
# reference numbers, a copy is a statement whose words all appear in another one that is at most a quarter longer. Extra
# words that negate ("not recommended") never make a copy. Checked by eye on the 45 ruler topics on 2026-11-18: 17 pairs
# found, all real copies; "finerenone" and "SGLT2 inhibitors" in otherwise equal sentences stay apart.
HEADER = re.compile(r'^\s*(?:recommendations?|what to do and what not to do)[^:]{0,80}:\s*', re.I)
CITATION = re.compile(r'(?<=[.)])\s*\d[\d,–\-]*(?:\s*;.*)?$')
NEGATION = {'not', 'no', 'never', 'without', 'avoid', 'nor', 'cannot', 'neither'}
COPY_SHARE = 0.8


def statement_words(text):
    core = CITATION.sub('', HEADER.sub('', str(text).strip()))
    return {w for w in re.findall(r'[a-z0-9]+', core.lower()) if len(w) > 1 or w.isdigit()}


def near_copy(a_text, b_text):
    a, b = statement_words(a_text), statement_words(b_text)
    small, big = (a, b) if len(a) <= len(b) else (b, a)
    return bool(small) and small <= big and len(small) / len(big) >= COPY_SHARE and not (big - small) & NEGATION


def choose(facts, request_cuis=None, most=MOST_FACTS, weight=None, stats=None):
    """The best `most` facts: direct before via broader; then the most specific request concept they name (rounded, so
    near-equal concepts tie); then grade; then the weight of all request concepts they name; then recommendation before
    explanation; then the shorter statement. A fact that names only request concepts less than half as specific as the
    request's most specific one is left out: fewer facts beat off-topic ones (KAPING, SubgraphRAG), so "loss of appetite
    in heart failure" does not fill its places with heart-failure drug advice. A near copy of a fact already chosen is
    skipped, so its place goes to the next fact. `stats`, if given, is filled with how many facts each cut left (the page's
    funnel)."""
    if weight:
        floor = max(weight.values()) / 2
        facts = {k: v for k, v in facts.items() if any(weight.get(c, 0.0) >= floor for c in v['joins'])}
    chosen, copies = [], 0
    for key, fact in sorted(facts.items(), key=lambda kv: (rank_key(kv[1]), kv[0])):
        if len(chosen) == most:
            break
        if any(near_copy(fact['statement']['text'], kept['statement']['text']) for _, kept in chosen):
            copies += 1
            continue
        chosen.append((key, fact))
    if stats is not None:
        stats.update(on_topic=len(facts), near_copies=copies, kept=len(chosen), below_cut=len(facts) - copies - len(chosen))
    return chosen


# The queries the page shows: the SPARQL texts actually sent for one question, by name. Long VALUES lists are cut to three
# items plus a comment, so the text stays readable and still runs.
QUERY_NAMES = [('Query A — claims', CLAIMS), ('Query B — recommendation readings', READINGS),
               ('Helper — statements (sentence, grade, page, guideline)', STATEMENTS), ('Helper — broader concept', BROADER),
               ('Helper — definitions of the question concepts', DEFINITIONS),
               ('Helper — definitions of the words in the kept statements', TERMS)]


def first_sentence(text, most=240):
    text = re.sub(r'\s+', ' ', str(text)).strip()
    cut = re.split(r'(?<=[a-z)\]])\. ', text, maxsplit=1)[0].rstrip('.') + '.'
    return cut if len(cut) <= most else cut[:most].rsplit(' ', 1)[0] + ' …'


def term_rows(fact_iris, query=sparql):
    """fact IRI -> {cui: definition} for the concepts the statement names that have a UMLS definition."""
    out = defaultdict(dict)
    if fact_iris:
        for row in query(TERMS % ' '.join(f'<{f}>' for f in sorted(fact_iris))):
            out[row['f']].setdefault(cui_of(row['c']), row['def'])
    return out


def terms_of(statement, defined, names, most=MOST_TERMS):
    """The technical words of one statement that the graph defines: a concept counts only if every word of its name is in
    the statement (so it is a word the patient would hear) and it is not a generic word. At most `most`, longest name first."""
    said = {singular(w) for w in words(statement)}
    found = []
    for cui, definition in defined.items():
        name = names.get(cui, '')
        parts = [singular(w) for w in words(name)]
        if parts and all(p in said for p in parts) and name.lower() not in SKIP:
            found.append({'cui': cui, 'term': name, 'definition': first_sentence(definition)})
    return sorted(found, key=lambda t: -len(t['term']))[:most]


def definitions_of(cuis, query=sparql):
    """cui -> its UMLS definition (one per concept), for the question's own concepts."""
    if not cuis:
        return {}
    found = {}
    for row in query(DEFINITIONS % values(cuis)):
        found.setdefault(cui_of(row['c']), row['def'])
    return found


def shorten_values(text, most=6):
    def cut(m):
        items = m.group(2).split()
        if len(items) <= most:
            return m.group(0)
        return f'VALUES ?{m.group(1)} {{ ' + ' '.join(items[:3]) + f'\n    # … and {len(items) - 3} more\n  }}'
    return re.sub(r'VALUES \?(c|fact|f) \{ ([^}]*) \}', cut, text)


def shown_queries(sent):
    """One entry per query name, in the order first sent: the first text sent and how often that query was sent (once per
    question concept or broader level for A and B, once per batch of statements for the helper)."""
    out = {}
    for text in sent:
        for name, template in QUERY_NAMES:
            if text.startswith(template.split('%s')[0]):
                out.setdefault(name, {'name': name, 'text': shorten_values(text), 'times': 0})['times'] += 1
                break
    return list(out.values())


# ---- 4. fact text ------------------------------------------------------------------------------------------------------------

def fact_text(fact, names, labels):
    """Rule-based text of a fact, as KAPING and KG-RAG write triples: (concept, claim type, concept) for each claim, then the
    statement's type, grade, guideline and page, and the path when it was reached through a broader concept."""
    s = fact['statement']
    triples = []
    show = lambda cs: ' / '.join(names.get(c, c) for c in sorted(cs)) or '?'
    for claim in fact['claims']:
        parts = claim['parts']
        if claim['type'] == 'RecommendationReading':
            who = '; '.join(f'{role[3:].lower()}: {show(parts[role])}' for role in ROLES[1:] if parts.get(role))
            action = ' '.join(x for x in (claim.get('advice', ''), claim.get('operation', '').lower()) if x) or 'reading'
            triples.append(f'({action}, {show(parts.get("hasTarget", set()))}' + (f'; {who})' if who else ')'))
            continue
        first = parts.get('treatmentConcept') or parts.get('secondMedication') or set()
        second = (parts.get('conditionConcept') or parts.get('outcomeConcept') or
                  (parts.get('secondMedication') if 'treatmentConcept' in parts else set()) or set())
        triples.append(f'({show(first)}, {labels.get(claim["type"], claim["type"])}, {show(second)})')
    kind = 'recommendation' if s['recommendation'] else 'explanation'
    grade = ', '.join(f'Class {g}' for g in sorted(s['classes'], key=lambda g: GRADE_ORDER.get(g, 9)))
    level = ', '.join(f'Level {lv}' for lv in sorted(s['levels']))
    pages = ', '.join(f'p. {p}' for p in sorted(s['pages'], key=int))
    title = s['guideline'] if not s['year'] or s['year'] in s['guideline'] else f'{s["guideline"]} {s["year"]}'
    source = '; '.join(x for x in (title, pages) if x)
    paths = [' -> broader -> '.join(names.get(c, c) for c in path) for path in fact['paths'].values() if len(path) > 1]
    head = '; '.join(sorted(set(triples)))
    tail = ', '.join(x for x in (kind, grade, level) if x)
    via = f' [via {"; ".join(paths)}]' if paths else ''
    return f'{head} — {tail} — {source}{via}\n    statement: "{s["text"]}"'


def fact_graph(fact, names, labels):
    """The fact's own piece of the graph, for drawing it on the demo page (21 Nov 2026): each claim or reading with its
    action (or claim type) and the concepts it links, by role, plus the broader-concept path that reached it. Read only:
    nothing here changes which facts are chosen."""
    claims = []
    for claim in fact['claims']:
        parts = claim['parts']
        if claim['type'] == 'RecommendationReading':
            action = ' '.join(x for x in (claim.get('advice', ''), claim.get('operation', '').lower()) if x) or 'reading'
            roles = [(role[3:].lower(), parts.get(role, set())) for role in ROLES]
        else:
            action = labels.get(claim['type'], claim['type'])
            roles = [(part.replace('Concept', '').replace('secondMedication', 'medication'), parts.get(part, set())) for part in PARTS]
        claims.append({'claim': action, 'concepts': [{'role': role, 'cui': c, 'name': names.get(c, c)}
                                                     for role, cs in roles for c in sorted(cs)]})
    paths = [[{'cui': c, 'name': names.get(c, c)} for c in path] for path in fact['paths'].values() if len(path) > 1]
    return {'claims': sorted(claims, key=lambda x: x['claim']), 'paths': paths}


# ---- run ---------------------------------------------------------------------------------------------------------------------

def run(request, names, similar=None, query=sparql, labels=None, total=None):
    sent = []

    def recorded(text):
        sent.append(text)
        return query(text)
    concepts = find_concepts(request, names, similar)
    cuis = [c['cui'] for c in concepts]
    reached, claims = walk(cuis, recorded)
    statements = statements_of({c['fact'] for c in claims.values()}, recorded)
    weight = specificity(reached, claims, total or claim_total(query))
    facts = facts_of(reached, claims, statements, weight)
    funnel = {}
    chosen = choose(facts, weight=weight, stats=funnel)
    defined = definitions_of(cuis, recorded)
    fact_terms = term_rows({k for k, _ in chosen}, recorded) if USE_TERMS else {}
    shown = set(cuis) | {c for hits in reached.values() for _, path in hits for c in path}
    shown |= {c for _, f in chosen for claim in f['claims'] for part in claim['parts'].values() for c in part}
    shown |= {c for d in fact_terms.values() for c in d}
    concept_names = names_of(shown, query)
    labels = labels if labels is not None else labels_of(CLAIM_TYPES, query)
    return {
        'request': request,
        'concepts': [{**c, 'name': concept_names.get(c['cui'], ''), 'weight': round(weight.get(c['cui'], 0.0), 3), 'claims_at': [
            {'cui': cui, 'name': concept_names.get(cui, ''), 'path': path} for cui, path in reached.get(c['cui'], [])]}
            for c in concepts],
        'claims_found': len(claims),
        'statements_found': len(facts),
        'funnel': {'claims_found': len(claims), 'statements': len(facts), **funnel},
        'queries': shown_queries(sent),
        'definitions': [{'cui': c, 'name': concept_names.get(c, ''), 'definition': defined[c]} for c in cuis if c in defined],
        'facts': [{'fact': k, 'text': fact_text(v, concept_names, labels), 'statement': v['statement']['text'],
                   'levels': sorted(v['statement']['levels']), 'recommendation': v['statement']['recommendation'],
                   'classes': sorted(v['statement']['classes']), 'pages': sorted(v['statement']['pages'], key=int),
                   'guideline': v['statement']['guideline'], 'joins': sorted(v['joins']), 'hops': v['hops'],
                   'terms': terms_of(v['statement']['text'], fact_terms.get(k, {}), concept_names),
                   'graph': fact_graph(v, concept_names, labels)}
                  for k, v in chosen],
    }


def summary_line(name, record):
    concepts = ', '.join(f'{c["name"] or c["words"]}' + ('' if c['method'] == 'name' else f' ~{c["score"]}')
                         + ('' if c['claims_at'] else ' [no claims]') for c in record['concepts'])
    graded = sum(1 for f in record['facts'] if f['classes'])
    rec = sum(1 for f in record['facts'] if f['recommendation'])
    via = sum(1 for f in record['facts'] if f['hops'] > 0)
    return (f'| {name} | {concepts or "—"} | {record["claims_found"]} | {len(record["facts"])} | {rec} | {graded} | {via} |')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', nargs='*', default=[], help='Ruler topic ids (default: all)')
    parser.add_argument('--name', help='Name for a topic given with --request')
    parser.add_argument('--request')
    parser.add_argument('--rebuild-names', action='store_true', help='Read the name index from GraphDB again')
    parser.add_argument('--no-similar', action='store_true', help='Skip the meaning look-up for spelling variants')
    parser.add_argument('--output', default=str(OUT))
    args = parser.parse_args()

    names = load_names(args.rebuild_names)
    similar = None if args.no_similar else MeaningLookup(names)
    ruler = {e['id']: e['request'] for e in json.loads(RULER.read_text(encoding='utf-8'))} if RULER.exists() else {}  # example topics, kept local
    topics = [(n, ruler[n]) for n in (args.only or ([] if args.request else sorted(ruler)))]
    if args.request:
        topics.append((args.name or 'request', args.request))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    labels = labels_of(CLAIM_TYPES)
    total = claim_total()
    lines = ['| topic | concepts found | claims + readings | facts kept | recommendations | graded | via broader |',
             '|---|---|---|---|---|---|---|']
    for name, request in topics:
        record = run(request, names, similar, labels=labels, total=total)
        (out / f'{name}.json').write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding='utf-8')
        lines.append(summary_line(name, record))
        print(lines[-1], flush=True)
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'written to {out}')


if __name__ == '__main__':
    main()
