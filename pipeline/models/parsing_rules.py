import re
from typing import List
class ParsingRules:
    def __init__(self):
        self.artifact_pattern = re.compile(r"^\s*[IVX0-9\s\.\-]+\s*$")
        self.classification_code_pattern = re.compile(r"^[IVX]+[a-z]?\s+[A-C]?\s*$", re.IGNORECASE)
        self.lowercase_starters = {
            "and", "or", "but", "if", "when", "while", "because",
            "although", "though", "unless", "until", "since"
        }
        self.mid_sentence_starters = {
            "with", "in", "on", "at", "by", "for", "from",
            "to", "of", "as", "after", "before", "during", "patients"
        }
        self.recommendation_split_pattern = re.compile(
            r'(?=(?:An?|The|A)\s+[A-Z][^\.\n]{0,50}?\s+(?:may|should|must|is\s+recommended|are\s+recommended|can\s+be))'
        )
        self.url_patterns = [
            re.compile(r"https?://", re.IGNORECASE),
            re.compile(r"www\.", re.IGNORECASE),
            re.compile(r"\.(com|org|edu|gov|uk)\b", re.IGNORECASE),
        ]
        self.table_figure_pattern = re.compile(
            r"\b(Table|TABLE|Figure|FIGURE|Fig\.|FIG\.)\s+\d+",
            re.IGNORECASE
        )
    def is_artifact_chunk(self, text: str) -> bool:
        t = text.strip()
        if self.artifact_pattern.match(t):
            return True
        if len(t.split()) <= 2 and self.classification_code_pattern.match(t):
            return True
        return False
    def starts_mid_sentence(self, text: str) -> bool:
        t = text.strip()
        if not t:
            return False
        if len(t) > 10 and not re.search(r'[.!?]$', t):
            last_punct = max(t.rfind('.'), t.rfind('!'), t.rfind('?'))
            if last_punct > 0:
                after_last_punct = t[last_punct+1:].strip()
                if len(after_last_punct.split()) < 3:
                    return True
        first_word = t.split()[0] if t.split() else ""
        if first_word.islower() and first_word not in self.lowercase_starters:
            if first_word in self.mid_sentence_starters:
                return True
        return False
    def split_multiple_recommendations(self, text: str) -> List[str]:
        parts = self.recommendation_split_pattern.split(text)
        recommendations = [p.strip() for p in parts if p.strip() and len(p.strip()) > 30]
        if len(recommendations) > 1:
            return recommendations
        return [text]
    def contains_url(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self.url_patterns)
    def is_table_metadata(self, text: str) -> bool:
        if not text:
            return False
        t = text.strip()
        match = self.table_figure_pattern.search(t)
        if match and match.start() < 150:
            after_mention = t[match.end():].strip()
            if len(after_mention.split()) < 20:
                return True
        if "|" in t and len(t.split("|")) >= 2:
            parts = [p.strip() for p in t.split("|")]
            if all(len(p.split()) <= 4 for p in parts):
                return True
        return False
    def is_truncated(self, text: str) -> bool:
        if not text or len(text) < 10:
            return False
        text_clean = text.strip()
        if not re.search(r'[.!?]$', text_clean):
            words = text_clean.split()
            if words:
                last_word = words[-1].strip(',:;')
                if len(last_word) <= 3 and last_word.islower() and last_word.isalpha():
                    return True
                if re.match(r'^[a-z]{1,3}$', last_word):
                    return True
                if len(last_word) > 3 and last_word[-3:].lower() in ['gur', 'rdi', 'egi', 'ste', 'ndr']:
                    return True
        trailing_prepositions = r'\b(for|with|to|in|on|at|of|from|by)\s*$'
        if re.search(trailing_prepositions, text_clean, re.IGNORECASE):
            return True
        if re.search(r'-\s*$', text_clean):
            return True
        incomplete_verb_endings = r'\b(should be|must be|may be|can be|will be|would be|could be|shall be)\s*$'
        if re.search(incomplete_verb_endings, text_clean, re.IGNORECASE):
            return True
        return False
    def should_merge_with_next(self, text: str, next_text: str = None) -> bool:
        is_incomplete = (
            len(text.strip()) < 80 or
            self.starts_mid_sentence(text) or
            self.is_truncated(text) or
            not re.search(r'[.!?]$', text.strip())
        )
        return is_incomplete and next_text is not None
    def normalize_text_for_llm(self, text: str) -> str:
        if not text or not text.strip():
            return text
        normalized = re.sub(r'\n+', ' ', text)
        normalized = ' '.join(normalized.split())
        return normalized.strip()
