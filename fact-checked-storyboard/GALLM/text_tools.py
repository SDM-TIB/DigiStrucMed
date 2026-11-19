"""Small helpers the new pipeline needs, copied on 2026-11-18 from the earlier GALLM modules (fact_check.py, link_concepts.py,
evidence_store.py) so the pipeline stands alone: reading a model's JSON answer, numbering facts, normalising words (American
spelling, singular), the generic-word lists, and the local MiniLM encoder for the meaning look-up."""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np  # noqa: F401 (used by the encoder's callers)


# ---- from fact_check.py ----

def numbered(facts):
    return '\n'.join(f'{n}. {fact}' for n, fact in enumerate(facts, 1))


def without_thinking(answer):
    """Some models (Qwen) think aloud in <think> tags before answering; only the answer counts."""
    return re.sub(r'<think>.*?</think>', '', answer or '', flags=re.S)


def json_object(answer):
    """The JSON object in a model's answer, or None."""
    match = re.search(r'\{.*\}', without_thinking(answer), re.S)
    try:
        value = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


# ---- from link_concepts.py ----

BRITISH = [('haem', 'hem'), ('aemia', 'emia'), ('aemic', 'emic'), ('paed', 'ped'), ('anaes', 'anes'), ('caesar', 'cesar'), ('faec', 'fec'),
           ('gynaec', 'gynec'), ('aetiolog', 'etiolog'), ('oedem', 'edem'), ('oesophag', 'esophag'), ('oestr', 'estr'), ('foet', 'fet'),
           ('pnoea', 'pnea'), ('rrhoea', 'rrhea'), ('coeliac', 'celiac'), ('tumour', 'tumor'), ('behaviour', 'behavior'),
           ('manoeuvr', 'maneuver'), ('leucocyt', 'leukocyt'), ('sulph', 'sulf')]


BRITISH_WORDS = {'centre': 'center', 'centres': 'centers', 'litre': 'liter', 'litres': 'liters', 'fibre': 'fiber', 'fibres': 'fibers'}


GENERIC = set('''treatment treatments therapy therapies management patient patients disease diseases condition conditions symptom symptoms
study studies risk use evaluation care procedure procedures test tests history level levels dose doses drug drugs medicine medicines
medication medications increase decrease report reports change changes response control function functions failure activity
activities sleep heart blood pressure rate weight time day days week weeks month months year years age life
pain effect effects outcome outcomes event events infection prevention screening monitoring assessment diagnosis surgery'''.split())


COMMON = set('''other others support section sections writing start period model models prevent reduction intervention interventions
data diagnosed term long cardiology virus help helps stage stages class type types form forms group groups member members unit units
program programs programme programmes plan plans order orders table figure note notes practice
battery batteries'''.split())


COMMON |= set('''people person persons individual individuals population populations american european documented reported reporting
reviewer reviewers principal rest resting relative healthcare consultant consultants'''.split()) | {'health care', 'expert witness'}


COMMON |= {'level of evidence', 'levels of evidence'}


def american(word):
    """A lower-case English word in American spelling: ischaemia -> ischemia, oedema -> edema, dyspnoea -> dyspnea."""
    for british, us in BRITISH:
        if british in word:
            word = word.replace(british, us)
    return BRITISH_WORDS.get(word, word)


def words(text, english=False):
    """Lower-case words, letters of any language (so German ä, ö, ü and ß stay inside a word). Footnote marks such as the ᵃ in
    "Loop diureticsᵃ" or the ³ in "reduced³" are modifier letters and superscript digits, and they separate words. English words
    come in American spelling, for the text and the UMLS names alike."""
    text = ''.join(' ' if unicodedata.category(c) in ('Lm', 'No') else c for c in text)
    found = re.findall(r'[^\W_]+', text.lower().replace('β', 'beta').replace('α', 'alpha'))
    return [american(w) for w in found] if english else found


def singular(token):
    if token.endswith('ies') and len(token) > 4:
        return token[:-3] + 'y'
    if token.endswith('ses') or token.endswith('xes'):
        return token[:-2]
    return token[:-1] if token.endswith('s') and not token.endswith('ss') and len(token) > 3 else token


# ---- from evidence_store.py ----

ENCODER = 'sentence-transformers/all-MiniLM-L6-v2'


class Encoder:
    """Turns text into meaning vectors on the local GPU, or on the processor when the GPU fails. Only ever reads the copy on this PC."""
    def __init__(self, name=ENCODER):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.name = torch, name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        self.model = AutoModel.from_pretrained(name, local_files_only=True).to(self.device).eval()

    def to_cpu(self):
        """A fresh copy of the model on the processor: after a CUDA error the GPU copy cannot be moved, so it is read again from disk."""
        from transformers import AutoModel
        self.device = 'cpu'
        self.model = AutoModel.from_pretrained(self.name, local_files_only=True).eval()

    def encode(self, texts):
        tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt').to(self.device)
        with self.torch.no_grad():
            hidden = self.model(**tokens).last_hidden_state
        mask = tokens['attention_mask'].unsqueeze(-1)
        vectors = (hidden * mask).sum(1) / mask.sum(1)
        return self.torch.nn.functional.normalize(vectors, dim=1).cpu().numpy()

    def __call__(self, texts, batch=64):
        out = []
        for start in range(0, len(texts), batch):
            try:
                out.append(self.encode(texts[start:start + batch]))
            except RuntimeError as error:  # torch's CUDA errors are RuntimeErrors; one stopped a whole pipeline run (2026-11-17)
                if self.device == 'cpu':
                    raise
                print(f'GPU failed ({(str(error).splitlines() or [""])[0][:80]}); search continues on the processor', flush=True)
                self.to_cpu()
                out.append(self.encode(texts[start:start + batch]))
        return np.vstack(out) if out else np.zeros((0, 384))
