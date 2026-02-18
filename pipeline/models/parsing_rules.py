"""
[RuleBasedModel] parsing_rules

Rule-based parsing for clear chunks: artifact detection, filtering, splitting, merge logic.
Does not modify sentence content; only whitespace is normalized.
"""

import re
from typing import List


class ParsingRules:
    """
    Rule-based text parsing for clear chunks.
    Provides artifact detection, filtering (URL/table/truncation), splitting, and merge logic.
    Sentence content is never modified; normalize_text_for_llm only normalizes whitespace.
    """
    
    def __init__(self):
        """Initialize parsing rules."""
        # Patterns for artifact detection
        self.artifact_pattern = re.compile(r"^\s*[IVX0-9\s\.\-]+\s*$")
        self.classification_code_pattern = re.compile(r"^[IVX]+[a-z]?\s+[A-C]?\s*$", re.IGNORECASE)
        
        # Patterns for mid-sentence detection
        self.lowercase_starters = {
            "and", "or", "but", "if", "when", "while", "because",
            "although", "though", "unless", "until", "since"
        }
        self.mid_sentence_starters = {
            "with", "in", "on", "at", "by", "for", "from",
            "to", "of", "as", "after", "before", "during", "patients"
        }
        
        # Pattern for splitting multiple recommendations
        self.recommendation_split_pattern = re.compile(
            r'(?=(?:An?|The|A)\s+[A-Z][^\.\n]{0,50}?\s+(?:may|should|must|is\s+recommended|are\s+recommended|can\s+be))'
        )
        
        # URL/metadata patterns
        self.url_patterns = [
            re.compile(r"https?://", re.IGNORECASE),
            re.compile(r"www\.", re.IGNORECASE),
            re.compile(r"\.(com|org|edu|gov|uk)\b", re.IGNORECASE),
        ]
        
        # Table/figure patterns
        self.table_figure_pattern = re.compile(
            r"\b(Table|TABLE|Figure|FIGURE|Fig\.|FIG\.)\s+\d+",
            re.IGNORECASE
        )
    
    def is_artifact_chunk(self, text: str) -> bool:
        """Detect if chunk is an artifact (classification codes, page numbers, etc.)."""
        t = text.strip()
        
        # Skip if mostly just numbers and classification codes
        if self.artifact_pattern.match(t):
            return True
        
        # Skip if it's just a classification code with minimal text
        if len(t.split()) <= 2 and self.classification_code_pattern.match(t):
            return True
        
        return False
    
    def starts_mid_sentence(self, text: str) -> bool:
        """Detect if text starts mid-sentence."""
        t = text.strip()
        if not t:
            return False
        
        # Check if text ends abruptly (no punctuation)
        if len(t) > 10 and not re.search(r'[.!?]$', t):
            last_punct = max(t.rfind('.'), t.rfind('!'), t.rfind('?'))
            if last_punct > 0:
                after_last_punct = t[last_punct+1:].strip()
                if len(after_last_punct.split()) < 3:
                    return True
        
        # Check first word
        first_word = t.split()[0] if t.split() else ""
        
        if first_word.islower() and first_word not in self.lowercase_starters:
            if first_word in self.mid_sentence_starters:
                return True
        
        return False
    
    def split_multiple_recommendations(self, text: str) -> List[str]:
        """
        Split text containing multiple medication recommendations.
        Universal pattern across all guideline formats.
        """
        parts = self.recommendation_split_pattern.split(text)
        
        # Filter out empty parts and very short fragments
        recommendations = [p.strip() for p in parts if p.strip() and len(p.strip()) > 30]
        
        if len(recommendations) > 1:
            return recommendations
        
        return [text]
    
    def contains_url(self, text: str) -> bool:
        """Check if text contains URLs or web links."""
        return any(pattern.search(text) for pattern in self.url_patterns)
    
    def is_table_metadata(self, text: str) -> bool:
        """Detect if text is a table header, figure caption, or PDF metadata."""
        if not text:
            return False
        
        t = text.strip()
        
        # Check for table/figure mentions early in text
        match = self.table_figure_pattern.search(t)
        if match and match.start() < 150:
            after_mention = t[match.end():].strip()
            if len(after_mention.split()) < 20:
                return True
        
        # Check for pipe-separated table columns
        if "|" in t and len(t.split("|")) >= 2:
            parts = [p.strip() for p in t.split("|")]
            if all(len(p.split()) <= 4 for p in parts):
                return True
        
        return False
    
    def is_truncated(self, text: str) -> bool:
        """
        Detect if text appears to be truncated (cut mid-word or mid-phrase).
        
        Args:
            text: Text to check
        
        Returns:
            True if text is likely truncated, False otherwise
        """
        if not text or len(text) < 10:
            return False
        
        text_clean = text.strip()
        
        # Pattern 1: Ends with incomplete word (no final punctuation + suspicious ending)
        if not re.search(r'[.!?]$', text_clean):
            # Check if last word looks incomplete (very short or partial word)
            words = text_clean.split()
            if words:
                last_word = words[-1].strip(',:;')
                
                # If last word is very short and lowercase (likely fragment)
                if len(last_word) <= 3 and last_word.islower() and last_word.isalpha():
                    return True
                
                # If last word has weird pattern (partial abbreviation, etc.)
                if re.match(r'^[a-z]{1,3}$', last_word):
                    return True
                
                # If last word ends abruptly (consonant clusters, no vowel at end)
                if len(last_word) > 3 and last_word[-3:].lower() in ['gur', 'rdi', 'egi', 'ste', 'ndr']:
                    return True
        
        # Pattern 2: Ends mid-phrase (preposition + no object)
        trailing_prepositions = r'\b(for|with|to|in|on|at|of|from|by)\s*$'
        if re.search(trailing_prepositions, text_clean, re.IGNORECASE):
            return True
        
        # Pattern 3: Text ends with hyphen or dash (line break mid-word)
        if re.search(r'-\s*$', text_clean):
            return True
        
        # Pattern 4: Ends with "be" or modal verb (incomplete verb phrase)
        incomplete_verb_endings = r'\b(should be|must be|may be|can be|will be|would be|could be|shall be)\s*$'
        if re.search(incomplete_verb_endings, text_clean, re.IGNORECASE):
            return True
        
        return False
    
    def should_merge_with_next(self, text: str, next_text: str = None) -> bool:
        """Determine if current chunk should be merged with next."""
        # Check if chunk is incomplete or very small
        is_incomplete = (
            len(text.strip()) < 80 or
            self.starts_mid_sentence(text) or
            self.is_truncated(text) or  # NEW: Check for truncation
            not re.search(r'[.!?]$', text.strip())
        )
        
        return is_incomplete and next_text is not None
    
    def normalize_text_for_llm(self, text: str) -> str:
        """
        Normalize only whitespace; do not change any words or sentence structure.
        Produces clear chunks without modifying sentence content.
        """
        if not text or not text.strip():
            return text
        normalized = re.sub(r'\n+', ' ', text)
        normalized = ' '.join(normalized.split())
        return normalized.strip()