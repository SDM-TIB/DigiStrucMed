"""
[Transform] chunk_text

Chunk raw text into units for downstream NLP.
Uses parsing_rules (RuleBasedModel).

Implements:
- Sentence-first chunking (max chars per chunk; no mid-sentence splits).
- Generic citation stripping (pattern-based).
- Optional noise filter (word-overlap similarity to recommendation-like text).
"""

import nltk
import re
from typing import List, Dict, Optional
from pipeline.data import RawText, TextChunks
from pipeline.models.parsing_rules import ParsingRules
from pipeline.transforms.table_to_sentences import (
    classify_table,
    table_rows_to_sentences,
)

# Default max characters per body chunk (sentence-first; no mid-sentence split).
DEFAULT_MAX_CHUNK_CHARS = 400

# Citation patterns (generic; no hardcoded numbers).
# Comma-separated number lists and recommendation classes; ranges only when citation-like (followed by .,;)] or end).
CITATION_PATTERNS = [
    re.compile(r"\b\d+(?:,\s*\d+)+\b"),                         # 1,2,3 or 4,5,6,7
    re.compile(r"\b\d+[–\-]\d+(?:,\s*\d+[–\-]\d+)*(?=[.,;)\]]|\s+[A-Z]|\s*$)"),  # 4–7, 53–59 (not 2-3 mg)
    re.compile(r"\b\d+[A-Z]?[-–]\s*[A-Z]{2}\b", re.IGNORECASE), # 1C-LD, 2A
]
# Trailing citation patterns (applied after main patterns to clean end of text).
CITATION_TRAILING_PATTERNS = [
    re.compile(r"\.\d{1,4}\s*$"),       # trailing .3 or .12 (ref after period)
    re.compile(r"[.,;]\s*\d{1,4}\s*$"), # trailing ", 41" or "; 12"
    re.compile(r"[–\-]\d{1,4}\s*$"),   # trailing –46 or -11 (en/em-dash ref)
]

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from nltk.tokenize import sent_tokenize


class ChunkText:
    """
    Chunk raw text into text units for downstream NLP.
    Uses rule-based parsing for artifact detection and text quality.
    """
    
    def __init__(
        self,
        parsing_rules: ParsingRules,
        min_chars: int = 40,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        filter_noise: bool = False,
    ):
        """
        Initialize chunk_text transform.

        Args:
            parsing_rules: Rule-based model for text parsing
            min_chars: Minimum characters for valid chunk
            max_chunk_chars: Max characters per body chunk (sentence-first chunking)
            filter_noise: If True, drop body chunks that are not recommendation-like (similarity filter)
        """
        self.parsing_rules = parsing_rules
        self.min_chars = min_chars
        self.max_chunk_chars = max_chunk_chars
        self.filter_noise = filter_noise
        self._noise_exemplar_vocab: Optional[set] = None
    
    def transform(self, raw_text: RawText) -> TextChunks:
        """
        Transform raw text to clean chunks.
        
        Args:
            raw_text: Extracted raw text pages
        
        Returns:
            TextChunks containing segmented text units
        """
        text_chunks = TextChunks()
        pages = raw_text.get_pages()
        
        chunk_id = 0
        for page in pages:
            text = page["text"]
            page_number = page["page"]
            source_file = page.get("source", "")
            
            # Segment into chunks
            chunks = self._segment_page(text)
            
            for chunk_text in chunks:
                # Apply parsing rules to filter artifacts
                if self.parsing_rules.is_artifact_chunk(chunk_text):
                    continue
                
                if self.parsing_rules.is_table_metadata(chunk_text):
                    continue
                
                if len(chunk_text) < self.min_chars:
                    continue
                
                # Try to split multiple recommendations
                sub_chunks = self.parsing_rules.split_multiple_recommendations(chunk_text)
                
                for sub_text in sub_chunks:
                    if len(sub_text) >= self.min_chars:
                        # Skip truncated chunks (incomplete text)
                        if self.parsing_rules.is_truncated(sub_text):
                            continue
                        
                        # Normalize whitespace only (parsing_rules does not modify sentence content)
                        normalized_text = self.parsing_rules.normalize_text_for_llm(sub_text)
                        # Strip citation patterns (generic regex)
                        normalized_text = self._strip_citations(normalized_text)
                        if not normalized_text.strip() or len(normalized_text.strip()) < self.min_chars:
                            continue
                        text_chunks.add_chunk(
                            page=page_number,
                            text=normalized_text.strip(),
                            source=source_file,
                            chunk_id=chunk_id
                        )
                        chunk_id += 1
        
        # Step 1: Merge small/incomplete chunks
        merged_chunks = self._merge_small_chunks(text_chunks.get_chunks())
        
        # Step 2: Deduplicate similar chunks
        deduplicated_chunks = self._deduplicate_chunks(merged_chunks)
        
        # Step 3: Enforce max length on very long chunks
        # Use 800 for initial limit to allow abbreviation lists, then split if truly too long
        length_enforced = self._enforce_max_length(deduplicated_chunks, max_length=800)
        
        # Step 4: Final cleanup - merge any new small chunks created by length enforcement
        final_merged = self._merge_small_chunks(length_enforced)
        
        # Step 5: Final deduplication pass (length enforcement might create new duplicates)
        final_chunks = self._deduplicate_chunks(final_merged)
        
        # Replace with processed chunks
        text_chunks.chunks = final_chunks

        # Optional: drop body chunks that are not recommendation-like (e.g. committee lists)
        if self.filter_noise:
            text_chunks.chunks = [
                c for c in text_chunks.chunks
                if not self._is_noise_chunk(c.get("text", ""))
            ]
        
        # Step 6: Add table-derived chunks (from each page's tables)
        chunk_id = len(text_chunks.chunks)
        for page in pages:
            page_number = page["page"]
            source_file = page.get("source", "")
            tables = page.get("tables", [])
            for ti, table in enumerate(tables):
                rows = table.get("rows", [])
                if not rows:
                    continue
                table_type = classify_table(rows)
                table_title = table.get("title") or ""
                for sentence, row_index in table_rows_to_sentences(
                    rows, table_type, table_title=table_title
                ):
                    if len(sentence) >= self.min_chars:
                        text_chunks.add_chunk(
                            page=page_number,
                            text=sentence,
                            source=source_file,
                            chunk_id=chunk_id,
                            from_table=True,
                            table_index=ti,
                            row_index=row_index,
                        )
                        chunk_id += 1
        
        # Reassign chunk IDs
        for i, chunk in enumerate(text_chunks.chunks):
            chunk["chunk_id"] = i
        
        return text_chunks
    
    def _normalize_newlines(self, text: str) -> str:
        """
        Normalize newlines in text before segmentation.
        Removes artifact newlines while preserving semantic structure.
        Uses general patterns - no hardcoding.
        
        Args:
            text: Raw text with potential unwanted newlines
            
        Returns:
            Text with normalized newlines
        """
        if not text:
            return text
        
        # Pattern 1: Multiple consecutive newlines -> single space (paragraph breaks)
        text = re.sub(r'\n{2,}', ' ', text)
        
        # Pattern 2: Single newline between lowercase words -> space (continuation)
        text = re.sub(r'([a-z])\n([a-z])', r'\1 \2', text)
        
        # Pattern 3: Single newline after punctuation -> space
        text = re.sub(r'([.,;:!?])\n', r'\1 ', text)
        
        # Pattern 4: Single newline before punctuation -> remove (artifact)
        text = re.sub(r'\n([.,;:!?])', r'\1', text)
        
        # Pattern 5: Newline between word and number -> space
        text = re.sub(r'([a-zA-Z])\n(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)\n([a-zA-Z])', r'\1 \2', text)
        
        # Pattern 6: Any remaining single newlines -> space
        # This catches all edge cases
        text = re.sub(r'\n', ' ', text)
        
        # Final cleanup: Fix spacing around punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)  # Add space after punctuation
        
        # Normalize multiple spaces to single space
        text = ' '.join(text.split())
        
        return text

    def _strip_citations(self, text: str) -> str:
        """
        Remove citation patterns from text (generic regex; no hardcoded numbers).
        Covers: number lists (1,2,3), ranges (4–7), recommendation classes (1C-LD),
        and trailing refs (e.g. .3, .12, , 41, –46 at end of chunk).
        """
        if not text or not text.strip():
            return text
        out = text
        for pat in CITATION_PATTERNS:
            out = pat.sub("", out)
        for pat in CITATION_TRAILING_PATTERNS:
            out = pat.sub("", out)
        # Clean spacing left by removed citations
        out = re.sub(r"\s+", " ", out).strip()
        out = re.sub(r"\s*,\s*,", ",", out)
        out = re.sub(r"^\s*,\s*", "", out)
        return out

    def _get_noise_exemplar_vocab(self) -> set:
        """Build a single exemplar vocabulary from generic recommendation-like sentences (no document-specific words)."""
        if self._noise_exemplar_vocab is not None:
            return self._noise_exemplar_vocab
        stop = {"the", "a", "an", "is", "are", "be", "to", "of", "in", "for", "with", "and", "or", "as", "by", "on", "at"}
        exemplars = [
            "In patients with heart failure, ACE inhibitors are recommended.",
            "Patients should receive beta blockers when indicated.",
            "Treatment should be considered for eligible individuals.",
        ]
        vocab = set()
        for s in exemplars:
            for w in re.findall(r"\b[a-zA-Z]{2,}\b", s.lower()):
                if w not in stop:
                    vocab.add(w)
        self._noise_exemplar_vocab = vocab
        return self._noise_exemplar_vocab

    def _is_noise_chunk(self, text: str) -> bool:
        """
        Heuristic: chunk is likely noise (e.g. committee list) if it has very low
        overlap with recommendation-like vocabulary. No hardcoded committee phrases.
        """
        if not text or len(text) < 50:
            return False
        vocab = self._get_noise_exemplar_vocab()
        words = set(re.findall(r"\b[a-zA-Z]{2,}\b", text.lower()))
        if not words:
            return True
        overlap = len(words & vocab) / len(words)
        # Low overlap and no sentence-ending punctuation often indicates lists/metadata
        has_ending = bool(re.search(r"[.!?]$", text.strip()))
        return overlap < 0.12 and not has_ending

    def _segment_page(self, text: str) -> List[str]:
        """
        Segment page text by sentence boundaries only (no mid-sentence splits).
        1. Normalize newlines, then sentence-tokenize.
        2. Merge truncated sentences with the next sentence.
        3. Group full sentences into chunks until max_chunk_chars would be exceeded.
        No hardcoded topic starters or relation rules; length-only grouping.
        """
        text = self._normalize_newlines(text)
        sentences = sent_tokenize(text)
        if not sentences:
            return []

        # Merge truncated sentences with next
        cleaned: List[str] = []
        i = 0
        while i < len(sentences):
            s = sentences[i].strip()
            if not s:
                i += 1
                continue
            while not self._is_sentence_complete(s) and i + 1 < len(sentences):
                s = s + " " + sentences[i + 1].strip()
                i += 1
            if self._is_sentence_complete(s):
                cleaned.append(s)
            i += 1

        # Sentence-first chunking: add sentences until length would exceed max_chunk_chars
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        max_len = self.max_chunk_chars

        for s in cleaned:
            add_len = len(s) + (1 if current else 0)
            if current_len + add_len > max_len and current:
                chunk_text = " ".join(current).strip()
                if chunk_text and self._is_sentence_complete(chunk_text):
                    chunks.append(chunk_text)
                current = []
                current_len = 0
            current.append(s)
            current_len += add_len

        if current:
            chunk_text = " ".join(current).strip()
            if chunk_text and self._is_sentence_complete(chunk_text):
                chunks.append(chunk_text)

        return chunks
    
    def _is_sentence_complete(self, sentence: str) -> bool:
        """
        Check if sentence is complete (not truncated mid-word).
        
        Uses general linguistic patterns to detect truncation:
        - Uncommon consonant clusters at word endings
        - Words ending mid-syllable
        - Invalid word patterns
        
        100% pattern-based, no hardcoded word lists.
        
        Args:
            sentence: Sentence to check
            
        Returns:
            True if sentence is complete, False if truncated
        """
        if not sentence:
            return False
        
        sentence_clean = sentence.strip()
        
        # Pattern 1: Ends with equals sign (abbreviation definition, likely truncated)
        # Example: "CV =", "HF ="
        if '=' in sentence_clean[-5:]:
            return False
        
        # Pattern 2: If ends with proper sentence-ending punctuation
        if sentence_clean and sentence_clean[-1] in '.!?':
            # Extract last word before punctuation
            words_before_punct = sentence_clean[:-1].split()
            if not words_before_punct:
                return True
            last_word = words_before_punct[-1]
        else:
            # Get last word (may have trailing punctuation)
            words = sentence_clean.split()
            if not words:
                return False
            last_word = words[-1].rstrip('.!?;:,)]}"\'-')
        
        if not last_word:
            return True  # Just punctuation
        
        # Pattern 3: Single character that's a consonant (likely truncated)
        # Example: "constrictive p"
        if len(last_word) == 1 and last_word.isalpha() and last_word.lower() not in 'aeiouy':
            return False
        
        # Pattern 4: Words with uncommon consonant clusters at the end
        # English words rarely end with certain consonant combinations
        # This is a general linguistic pattern
        uncommon_endings = [
            r'[bcdfghjklmnpqrstvwxz]{4,}$',  # 4+ consonants in a row (very rare)
            r'[pbtdkg][lrwy]$',              # Stop + liquid/glide at end (unusual)
            r'[sz][ptk]$',                    # Fricative + stop at end (unusual)
        ]
        
        for pattern in uncommon_endings:
            if re.search(pattern, last_word.lower()):
                return False
        
        # Pattern 5: Very short words (2 chars) with double consonants
        # English has very few valid 2-letter words with double consonants
        if len(last_word) == 2:
            # Consonant + consonant (rare in valid 2-letter English words)
            if re.match(r'^[bcdfghjklmnpqrstvwxz]{2}$', last_word.lower()):
                return False
        
        # If we reach here, word appears complete based on linguistic patterns
        return True
    
    def _merge_small_chunks(self, chunks: List[Dict], max_merged_length: int = 550) -> List[Dict]:
        """
        Merge very small chunks with adjacent chunks.
        More aggressive merging for very short or incomplete chunks.
        Respects max_merged_length to avoid creating very long chunks.
        
        UPDATED: Special handling for:
        - Mid-sentence chunks (prioritize semantic completeness over length)
        - Abbreviation lists (ending with '=')
        
        Args:
            chunks: List of chunks to process
            max_merged_length: Maximum length for merged chunks (avoids creating oversized chunks)
        """
        if not chunks:
            return []
        
        # First pass: merge very short chunks (< 60 chars) with next
        pass1 = []
        i = 0
        
        while i < len(chunks):
            current = chunks[i]
            current_text = current["text"].strip()
            
            # Very short chunks should always be merged if possible
            is_very_short = len(current_text) < 60
            
            # Check standard merge conditions
            needs_merge = self.parsing_rules.should_merge_with_next(
                current_text,
                chunks[i + 1]["text"] if i + 1 < len(chunks) else None
            )
            
            # Merge if very short OR standard conditions met
            should_merge = (is_very_short or needs_merge) and i + 1 < len(chunks)
            
            if should_merge:
                next_chunk = chunks[i + 1]
                next_text = next_chunk["text"].strip()
                merged_len = len(current_text) + len(next_text) + 1
                
                # Only merge if result won't exceed max length
                if merged_len <= max_merged_length:
                    # Allow merging across pages for very short chunks
                    if current["page"] == next_chunk["page"] or is_very_short:
                        merged_text = current_text + " " + next_text
                        pass1.append({
                            "chunk_id": current["chunk_id"],
                            "page": current["page"],
                            "text": merged_text,
                            "source": current.get("source", "")
                        })
                        i += 2
                        continue
            
            pass1.append(current)
            i += 1
        
        # Second pass: merge chunks that start with lowercase (mid-sentence)
        # UPDATED: Use more flexible length limits for mid-sentence and abbreviation chunks
        pass2 = []
        i = 0
        
        while i < len(pass1):
            current = pass1[i]
            current_text = current["text"].strip()
            
            # If this chunk starts with lowercase and we have a previous chunk
            if current_text and current_text[0].islower() and pass2:
                prev = pass2[-1]
                prev_text = prev["text"].strip()
                merged_len = len(prev_text) + len(current_text) + 1
                
                # Determine appropriate length limit based on chunk type
                # For abbreviation lists or mid-sentence continuations, use more flexible limit
                is_abbreviation_list = '=' in prev_text[-10:]  # Previous chunk ends with '='
                is_mid_sentence_continuation = not prev_text.endswith(('.', '!', '?'))
                is_short_fragment = len(current_text) < 150  # Short fragment likely continues previous
                
                # Use larger limit for special cases so we merge fragments and avoid broken boundaries
                if is_abbreviation_list or is_mid_sentence_continuation or is_short_fragment:
                    effective_limit = 800  # More flexible for semantic completeness
                else:
                    effective_limit = max_merged_length
                
                # Only merge if result won't exceed effective length limit
                if merged_len <= effective_limit:
                    # Merge with previous if on same page or previous doesn't end with punctuation
                    if prev["page"] == current["page"] or is_mid_sentence_continuation:
                        merged_text = prev_text + " " + current_text
                        pass2[-1] = {
                            "chunk_id": prev["chunk_id"],
                            "page": prev["page"],
                            "text": merged_text,
                            "source": prev.get("source", "")
                        }
                        i += 1
                        continue
            
            pass2.append(current)
            i += 1
        
        return pass2
    
    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Remove duplicate or highly similar chunks.
        Uses multiple similarity checks to catch different types of duplicates.
        
        Args:
            chunks: List of chunks to deduplicate
            
        Returns:
            List of unique chunks
        """
        if not chunks:
            return []
        
        unique_chunks = []
        seen_texts = []  # List instead of set for flexible comparison
        
        for chunk in chunks:
            text = chunk["text"].strip()
            text_lower = text.lower()
            
            # Check if this chunk is similar to any seen chunk
            is_duplicate = False
            replace_index = -1
            
            for i, seen in enumerate(seen_texts):
                seen_lower = seen.lower()
                
                # Check 1: Exact match
                if text == seen:
                    is_duplicate = True
                    break
                
                # Check 2: One is substring of the other
                if text_lower in seen_lower or seen_lower in text_lower:
                    # Keep the longer one
                    if len(text) > len(seen):
                        replace_index = i
                    else:
                        is_duplicate = True
                    break
                
                # Check 3: High overlap (Jaccard similarity on words)
                text_words = set(text_lower.split())
                seen_words = set(seen_lower.split())
                
                if len(text_words) > 5 and len(seen_words) > 5:  # Only for non-trivial chunks
                    intersection = len(text_words & seen_words)
                    union = len(text_words | seen_words)
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity > 0.8:  # 80% word overlap = very similar
                        # Keep the longer, more complete one
                        if len(text) > len(seen) and text.endswith(('.', '!', '?')):
                            replace_index = i
                        else:
                            is_duplicate = True
                        break
                
                # Check 4: First N chars match (catches prefixes)
                prefix_len = min(80, len(text), len(seen))
                if text_lower[:prefix_len] == seen_lower[:prefix_len]:
                    # Keep the longer one
                    if len(text) > len(seen):
                        replace_index = i
                    else:
                        is_duplicate = True
                    break
            
            if is_duplicate:
                continue
            
            if replace_index >= 0:
                # Replace shorter with longer
                unique_chunks[replace_index] = chunk
                seen_texts[replace_index] = text
            else:
                unique_chunks.append(chunk)
                seen_texts.append(text)
        
        return unique_chunks
    
    def _enforce_max_length(self, chunks: List[Dict], max_length: int = 550) -> List[Dict]:
        """
        Enforce maximum length on chunks by splitting very long ones.
        Tries to split at sentence boundaries, then at word boundaries if needed.
        
        Args:
            chunks: List of chunks
            max_length: Maximum characters per chunk
            
        Returns:
            List of chunks with length enforced
        """
        result = []
        
        for chunk in chunks:
            text = chunk["text"]
            
            if len(text) <= max_length:
                result.append(chunk)
                continue
            
            # Split long chunk at sentence boundaries
            sentences = sent_tokenize(text)
            
            current_part = []
            current_length = 0
            
            for sentence in sentences:
                sentence_len = len(sentence)
                
                # If adding this sentence exceeds max, finalize current part
                if current_length + sentence_len > max_length and current_part:
                    part_text = " ".join(current_part)
                    result.append({
                        "chunk_id": chunk["chunk_id"],
                        "page": chunk["page"],
                        "text": part_text,
                        "source": chunk.get("source", "")
                    })
                    current_part = []
                    current_length = 0
                
                # Handle single sentences that exceed max_length
                if sentence_len > max_length:
                    # Split long sentence at word boundaries
                    split_parts = self._split_long_text(sentence, max_length)
                    for part in split_parts:
                        if current_length + len(part) > max_length and current_part:
                            part_text = " ".join(current_part)
                            result.append({
                                "chunk_id": chunk["chunk_id"],
                                "page": chunk["page"],
                                "text": part_text,
                                "source": chunk.get("source", "")
                            })
                            current_part = []
                            current_length = 0
                        current_part.append(part)
                        current_length += len(part) + 1
                else:
                    current_part.append(sentence)
                    current_length += sentence_len + 1  # +1 for space
            
            # Add remaining part
            if current_part:
                part_text = " ".join(current_part)
                result.append({
                    "chunk_id": chunk["chunk_id"],
                    "page": chunk["page"],
                    "text": part_text,
                    "source": chunk.get("source", "")
                })
        
        return result
    
    def _split_long_text(self, text: str, max_length: int) -> List[str]:
        """
        Split a long text at word boundaries to fit within max_length.
        Uses natural break points (punctuation, conjunctions) when possible.
        
        Args:
            text: Long text to split
            max_length: Maximum length per part
            
        Returns:
            List of text parts, each <= max_length
        """
        if len(text) <= max_length:
            return [text]
        
        parts = []
        words = text.split()
        
        current_part = []
        current_length = 0
        
        for word in words:
            word_len = len(word)
            
            # Check if adding this word exceeds max
            if current_length + word_len + 1 > max_length and current_part:
                parts.append(" ".join(current_part))
                current_part = []
                current_length = 0
            
            current_part.append(word)
            current_length += word_len + 1
        
        # Add remaining
        if current_part:
            parts.append(" ".join(current_part))
        
        return parts