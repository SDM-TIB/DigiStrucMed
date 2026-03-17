"""
Stage B: Content preparation — chunk raw text and convert tables to SPO triples.
Consumes raw_text_and_tables (raw text + tables from Stage A) and produces
text chunks and table triples (two outputs).
"""
from __future__ import annotations

import json
import nltk
import re
from pathlib import Path
from typing import List, Dict, Optional

from pipeline.data import RawText, TextChunks, TextChunksAndTableTriples
from pipeline.models.parsing_rules import ParsingRules
from pipeline.models.table_spo_rules import tables_pages_to_spo_list

DEFAULT_MAX_CHUNK_CHARS = 400
CITATION_PATTERNS = [
    re.compile(r"\b\d+(?:,\s*\d+)+\b"),
    re.compile(r"\b\d+[–\-]\d+(?:,\s*\d+[–\-]\d+)*(?=[.,;)\]]|\s+[A-Z]|\s*$)"),
    re.compile(r"\b\d+[A-Z]?[-–]\s*[A-Z]{2}\b", re.IGNORECASE),
]
CITATION_TRAILING_PATTERNS = [
    re.compile(r"\.\d{1,4}\s*$"),
    re.compile(r"[.,;]\s*\d{1,4}\s*$"),
    re.compile(r"[–\-]\d{1,4}\s*$"),
]
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize


class ContentPreparation:
    """
    Stage B transform: takes raw text + tables (raw_text_and_tables) and produces
    (1) text chunks and (2) table SPO triples.

    stage_b_version controls chunking behaviour:
      v1 (default) – original logic, chunks all page text including any table body
                     text that leaked from the PDF extractor.
      v2           – same JSON schema, but filters out chunks whose content is
                     predominantly composed of table-cell material already captured in
                     the structured table triples path.  No hard-coded phrases: the
                     filter compares each chunk against actual cell content extracted
                     by Stage A.
    """

    # Minimum cell length to use in table-content matching (short cells like "1."
    # or "Yes" are too common to be reliable signals).
    _TABLE_CELL_MIN_LEN: int = 20
    # Fraction of a chunk's words that must appear in table-cell vocabulary before
    # the chunk is considered table-derived (v2 only).
    _TABLE_OVERLAP_THRESHOLD: float = 0.55

    def __init__(
        self,
        parsing_rules: ParsingRules,
        min_chars: int = 40,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        filter_noise: bool = False,
        stage_output_dir: Optional[str] = "outputs/STAGE_B_v1",
        stage_b_version: str = "v1",
    ):
        self.parsing_rules = parsing_rules
        self.min_chars = min_chars
        self.max_chunk_chars = max_chunk_chars
        self.filter_noise = filter_noise
        self.stage_output_dir = stage_output_dir
        self.stage_b_version = stage_b_version
        self._noise_exemplar_vocab: Optional[set] = None

    def transform(self, raw_text: RawText) -> TextChunksAndTableTriples:
        pages = raw_text.get_pages()
        text_chunks = self._build_text_chunks(pages)
        table_triples = tables_pages_to_spo_list(pages)
        result = TextChunksAndTableTriples(text_chunks=text_chunks, table_triples=table_triples)
        if self.stage_output_dir:
            self._write_stage_output(text_chunks, table_triples)
        return result

    def _write_stage_output(self, text_chunks: TextChunks, table_triples: List[Dict]) -> None:
        """Write Stage B v1 output to this stage's output folder. Creates folder if missing; overwrites files if existing."""
        out_path = Path(self.stage_output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        table_derived_count = sum(1 for c in text_chunks.get_chunks() if c.get("from_table"))
        (out_path / "stage_b_text_chunks.json").write_text(
            json.dumps({
                "metadata": {
                    "stage": "b",
                    "description": "Text chunks from content_preparation",
                    "total_chunks": text_chunks.count(),
                    "table_derived_chunks": table_derived_count,
                    "min_chunk_chars": self.min_chars,
                },
                "chunks": text_chunks.get_chunks(),
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out_path / "stage_b_table_triples.json").write_text(
            json.dumps({
                "metadata": {
                    "stage": "b",
                    "description": "Table SPO triples from content_preparation",
                    "total_triples": len(table_triples),
                },
                "triples": table_triples,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Table-content filtering helpers (used only in v2)
    # ------------------------------------------------------------------

    @staticmethod
    def _camel_split(text: str) -> List[str]:
        """Split a camelCase / PascalCase / all-caps run into individual tokens."""
        tokens = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", text)
        return [t.lower() for t in tokens if len(t) >= 3]

    def _build_table_vocab(self, tables: List[Dict]) -> set:
        """Return a lower-case word set built from all substantial table cells."""
        vocab: set = set()
        for table in tables:
            for row in table.get("rows", []):
                for cell in (row if isinstance(row, list) else []):
                    cell_text = str(cell).strip()
                    if len(cell_text) >= self._TABLE_CELL_MIN_LEN:
                        for word in re.findall(r"[a-zA-Z]{3,}", cell_text.lower()):
                            vocab.add(word)
        return vocab

    def _build_table_header_vocab(self, tables: List[Dict]) -> set:
        """
        Return a lower-case word set built from short header-like cells (4-19 chars).
        Used only in the suffix-trim pass to catch PDF column-header artifacts like
        'CORLOERecommendations' which are concatenations of short column names.
        """
        vocab: set = set()
        for table in tables:
            for row in table.get("rows", []):
                for cell in (row if isinstance(row, list) else []):
                    cell_text = str(cell).strip()
                    cell_len = len(cell_text)
                    if 4 <= cell_len < self._TABLE_CELL_MIN_LEN:
                        for word in re.findall(r"[a-zA-Z]{3,}", cell_text.lower()):
                            vocab.add(word)
                    elif cell_len >= self._TABLE_CELL_MIN_LEN:
                        for word in re.findall(r"[a-zA-Z]{3,}", cell_text.lower()):
                            vocab.add(word)
            # Also include the table title words as header vocab
            title = str(table.get("title", "")).strip()
            if title:
                for word in re.findall(r"[a-zA-Z]{3,}", title.lower()):
                    vocab.add(word)
        return vocab

    def _is_table_derived_chunk(self, chunk_text: str, table_vocab: set) -> bool:
        """
        Return True when the chunk is predominantly composed of table-cell words.

        Two complementary checks are applied:
        1. Vocabulary overlap – fraction of the chunk's words that appear in the
           per-page table-cell vocabulary.
        2. Substring containment – at least one full table-cell string (≥ 30 chars)
           is contained verbatim inside the chunk.  This catches concatenated-cell
           blocks that may use rare words not in the vocabulary.
        """
        if not table_vocab:
            return False
        words = re.findall(r"[a-zA-Z]{3,}", chunk_text.lower())
        if not words:
            return False
        overlap = sum(1 for w in words if w in table_vocab) / len(words)
        return overlap >= self._TABLE_OVERLAP_THRESHOLD

    def _build_table_substrings(self, tables: List[Dict]) -> List[str]:
        """Collect long cell strings for verbatim substring matching."""
        substrings: List[str] = []
        for table in tables:
            for row in table.get("rows", []):
                for cell in (row if isinstance(row, list) else []):
                    cell_text = str(cell).strip()
                    if len(cell_text) >= 30:
                        substrings.append(cell_text.lower())
        return substrings

    def _chunk_contains_table_cell(self, chunk_lower: str, substrings: List[str]) -> bool:
        return any(sub in chunk_lower for sub in substrings)

    def _trim_table_suffix(self, text: str, table_vocab: set) -> str:
        """
        Strip a trailing table-artifact suffix from a chunk.

        PDF extractors sometimes append concatenated table-header cells at the end
        of a narrative paragraph (e.g. "…interventions. CORLOERecommendations 2aB-NR1.",
        or mid-sentence: "…because CORLOERecommendations 8.").
        We scan the last 160 characters for a token whose camelCase-split sub-words
        are predominantly table-vocab words; if found, we trim from that token's start.
        Uses camelCase splitting so "CORLOERecommendations" → ["CORLOE","Recommendations"].
        """
        if not table_vocab or len(text) < 60:
            return text
        scan_window = 160
        tail = text[-scan_window:] if len(text) > scan_window else text
        tail_offset = max(0, len(text) - scan_window)

        # Tokenise tail into (start_in_tail, end_in_tail, word_string) triples
        token_spans = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[A-Za-z]{4,}", tail)]
        if not token_spans:
            return text

        # Walk tokens from the end; find the first (from end) whose camelCase tokens
        # are ≥ 40% table-vocab.  That marks the start of the artifact suffix.
        for start_pos, end_pos, word in reversed(token_spans):
            sub_tokens = self._camel_split(word)
            if not sub_tokens:
                continue
            vocab_hits = sum(1 for t in sub_tokens if t in table_vocab)
            if vocab_hits / len(sub_tokens) >= 0.4:
                # Trim from start_pos backward to the nearest whitespace / punctuation
                cut_in_text = tail_offset + start_pos
                # Back up past any preceding spaces or commas/semicolons
                while cut_in_text > 0 and text[cut_in_text - 1] in " ,;":
                    cut_in_text -= 1
                trimmed = text[:cut_in_text].strip()
                if len(trimmed) >= self.min_chars:
                    return trimmed
                # Trimming too much — leave the chunk unchanged
                return text
        return text

    # ------------------------------------------------------------------

    def _build_text_chunks(self, pages: List[Dict]) -> TextChunks:
        text_chunks = TextChunks()
        chunk_id = 0
        use_table_filter = self.stage_b_version == "v2"

        # Build document-wide table vocabs once.
        # global_table_vocab  – cells >= TABLE_CELL_MIN_LEN, used for chunk-level overlap filter.
        # global_header_vocab – all cells (including short header cells) + camelCase-split tokens,
        #                       used only in the post-merge suffix-trim pass.
        if use_table_filter:
            global_table_vocab: set = set()
            global_header_vocab: set = set()
            for p in pages:
                p_tables = p.get("tables", [])
                global_table_vocab |= self._build_table_vocab(p_tables)
                global_header_vocab |= self._build_table_header_vocab(p_tables)
        else:
            global_table_vocab = set()
            global_header_vocab = set()

        for page in pages:
            text = page["text"]
            page_number = page["page"]
            source_file = page.get("source", "")
            page_tables = page.get("tables", [])

            if use_table_filter and page_tables:
                table_vocab = self._build_table_vocab(page_tables)
                table_substrings = self._build_table_substrings(page_tables)
            else:
                table_vocab = set()
                table_substrings = []

            chunks = self._segment_page(text)
            for chunk_text in chunks:
                if self.parsing_rules.is_artifact_chunk(chunk_text):
                    continue
                if self.parsing_rules.is_table_metadata(chunk_text):
                    continue
                if len(chunk_text) < self.min_chars:
                    continue
                # v2: skip chunks that are table-body content already in triples
                if use_table_filter and (
                    self._is_table_derived_chunk(chunk_text, table_vocab)
                    or self._chunk_contains_table_cell(chunk_text.lower(), table_substrings)
                ):
                    continue
                sub_chunks = self.parsing_rules.split_multiple_recommendations(chunk_text)
                for sub_text in sub_chunks:
                    if len(sub_text) >= self.min_chars:
                        if self.parsing_rules.is_truncated(sub_text):
                            continue
                        normalized_text = self.parsing_rules.normalize_text_for_llm(sub_text)
                        normalized_text = self._strip_citations(normalized_text)
                        if not normalized_text.strip() or len(normalized_text.strip()) < self.min_chars:
                            continue
                        text_chunks.add_chunk(
                            page=page_number,
                            text=normalized_text.strip(),
                            source=source_file,
                            chunk_id=chunk_id,
                        )
                        chunk_id += 1
        merged_chunks = self._merge_small_chunks(text_chunks.get_chunks())
        deduplicated_chunks = self._deduplicate_chunks(merged_chunks)
        length_enforced = self._enforce_max_length(deduplicated_chunks, max_length=800)
        final_merged = self._merge_small_chunks(length_enforced)
        final_chunks = self._deduplicate_chunks(final_merged)

        # v2 post-merge pass: trim trailing table-artifact suffixes that were glued
        # onto good narrative chunks by _merge_small_chunks
        # (e.g. "…interventions. CORLOERecommendations 2aB-NR1.").
        # Uses the broader header vocab + camelCase splitting.
        if use_table_filter and global_header_vocab:
            cleaned: List[Dict] = []
            for chunk in final_chunks:
                trimmed_text = self._trim_table_suffix(chunk["text"], global_header_vocab)
                if trimmed_text.strip() and len(trimmed_text.strip()) >= self.min_chars:
                    cleaned.append({**chunk, "text": trimmed_text.strip()})
                elif chunk["text"].strip() and len(chunk["text"].strip()) >= self.min_chars:
                    cleaned.append(chunk)
            final_chunks = cleaned

        text_chunks.chunks = final_chunks
        if self.filter_noise:
            text_chunks.chunks = [
                c for c in text_chunks.chunks
                if not self._is_noise_chunk(c.get("text", ""))
            ]
        for i, chunk in enumerate(text_chunks.chunks):
            chunk["chunk_id"] = i
        return text_chunks

    def _normalize_newlines(self, text: str) -> str:
        if not text:
            return text
        text = re.sub(r"\n{2,}", " ", text)
        text = re.sub(r"([a-z])\n([a-z])", r"\1 \2", text)
        text = re.sub(r"([.,;:!?])\n", r"\1 ", text)
        text = re.sub(r"\n([.,;:!?])", r"\1", text)
        text = re.sub(r"([a-zA-Z])\n(\d)", r"\1 \2", text)
        text = re.sub(r"(\d)\n([a-zA-Z])", r"\1 \2", text)
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        text = re.sub(r"([.,;:!?])([A-Za-z])", r"\1 \2", text)
        text = " ".join(text.split())
        return text

    def _strip_citations(self, text: str) -> str:
        if not text or not text.strip():
            return text
        out = text
        for pat in CITATION_PATTERNS:
            out = pat.sub("", out)
        for pat in CITATION_TRAILING_PATTERNS:
            out = pat.sub("", out)
        out = re.sub(r"\s+", " ", out).strip()
        out = re.sub(r"\s*,\s*,", ",", out)
        out = re.sub(r"^\s*,\s*", "", out)
        return out

    def _get_noise_exemplar_vocab(self) -> set:
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
        if not text or len(text) < 50:
            return False
        vocab = self._get_noise_exemplar_vocab()
        words = set(re.findall(r"\b[a-zA-Z]{2,}\b", text.lower()))
        if not words:
            return True
        overlap = len(words & vocab) / len(words)
        has_ending = bool(re.search(r"[.!?]$", text.strip()))
        return overlap < 0.12 and not has_ending

    def _segment_page(self, text: str) -> List[str]:
        text = self._normalize_newlines(text)
        sentences = sent_tokenize(text)
        if not sentences:
            return []
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
        if not sentence:
            return False
        sentence_clean = sentence.strip()
        if "=" in sentence_clean[-5:]:
            return False
        if sentence_clean and sentence_clean[-1] in ".!?":
            words_before_punct = sentence_clean[:-1].split()
            if not words_before_punct:
                return True
            last_word = words_before_punct[-1]
        else:
            words = sentence_clean.split()
            if not words:
                return False
            last_word = words[-1].rstrip(".!?;:,)]}\"'-")
        if not last_word:
            return True
        if len(last_word) == 1 and last_word.isalpha() and last_word.lower() not in "aeiouy":
            return False
        uncommon_endings = [
            r"[bcdfghjklmnpqrstvwxz]{4,}$",
            r"[pbtdkg][lrwy]$",
            r"[sz][ptk]$",
        ]
        for pattern in uncommon_endings:
            if re.search(pattern, last_word.lower()):
                return False
        if len(last_word) == 2:
            if re.match(r"^[bcdfghjklmnpqrstvwxz]{2}$", last_word.lower()):
                return False
        return True

    def _merge_small_chunks(self, chunks: List[Dict], max_merged_length: int = 550) -> List[Dict]:
        if not chunks:
            return []
        pass1 = []
        i = 0
        while i < len(chunks):
            current = chunks[i]
            current_text = current["text"].strip()
            is_very_short = len(current_text) < 60
            needs_merge = self.parsing_rules.should_merge_with_next(
                current_text,
                chunks[i + 1]["text"] if i + 1 < len(chunks) else None,
            )
            should_merge = (is_very_short or needs_merge) and i + 1 < len(chunks)
            if should_merge:
                next_chunk = chunks[i + 1]
                next_text = next_chunk["text"].strip()
                merged_len = len(current_text) + len(next_text) + 1
                if merged_len <= max_merged_length:
                    if current["page"] == next_chunk["page"] or is_very_short:
                        merged_text = current_text + " " + next_text
                        pass1.append({
                            "chunk_id": current["chunk_id"],
                            "page": current["page"],
                            "text": merged_text,
                            "source": current.get("source", ""),
                        })
                        i += 2
                        continue
            pass1.append(current)
            i += 1
        pass2 = []
        i = 0
        while i < len(pass1):
            current = pass1[i]
            current_text = current["text"].strip()
            if current_text and current_text[0].islower() and pass2:
                prev = pass2[-1]
                prev_text = prev["text"].strip()
                merged_len = len(prev_text) + len(current_text) + 1
                is_abbreviation_list = "=" in prev_text[-10:]
                is_mid_sentence_continuation = not prev_text.endswith((".", "!", "?"))
                is_short_fragment = len(current_text) < 150
                if is_abbreviation_list or is_mid_sentence_continuation or is_short_fragment:
                    effective_limit = 800
                else:
                    effective_limit = max_merged_length
                if merged_len <= effective_limit:
                    if prev["page"] == current["page"] or is_mid_sentence_continuation:
                        merged_text = prev_text + " " + current_text
                        pass2[-1] = {
                            "chunk_id": prev["chunk_id"],
                            "page": prev["page"],
                            "text": merged_text,
                            "source": prev.get("source", ""),
                        }
                        i += 1
                        continue
            pass2.append(current)
            i += 1
        return pass2

    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        if not chunks:
            return []
        unique_chunks = []
        seen_texts = []
        for chunk in chunks:
            text = chunk["text"].strip()
            text_lower = text.lower()
            is_duplicate = False
            replace_index = -1
            for i, seen in enumerate(seen_texts):
                seen_lower = seen.lower()
                if text == seen:
                    is_duplicate = True
                    break
                if text_lower in seen_lower or seen_lower in text_lower:
                    if len(text) > len(seen):
                        replace_index = i
                    else:
                        is_duplicate = True
                    break
                text_words = set(text_lower.split())
                seen_words = set(seen_lower.split())
                if len(text_words) > 5 and len(seen_words) > 5:
                    intersection = len(text_words & seen_words)
                    union = len(text_words | seen_words)
                    similarity = intersection / union if union > 0 else 0
                    if similarity > 0.8:
                        if len(text) > len(seen) and text.endswith((".", "!", "?")):
                            replace_index = i
                        else:
                            is_duplicate = True
                        break
                prefix_len = min(80, len(text), len(seen))
                if text_lower[:prefix_len] == seen_lower[:prefix_len]:
                    if len(text) > len(seen):
                        replace_index = i
                    else:
                        is_duplicate = True
                    break
            if is_duplicate:
                continue
            if replace_index >= 0:
                unique_chunks[replace_index] = chunk
                seen_texts[replace_index] = text
            else:
                unique_chunks.append(chunk)
                seen_texts.append(text)
        return unique_chunks

    def _enforce_max_length(self, chunks: List[Dict], max_length: int = 550) -> List[Dict]:
        result = []
        for chunk in chunks:
            text = chunk["text"]
            if len(text) <= max_length:
                result.append(chunk)
                continue
            sentences = sent_tokenize(text)
            current_part = []
            current_length = 0
            for sentence in sentences:
                sentence_len = len(sentence)
                if current_length + sentence_len > max_length and current_part:
                    part_text = " ".join(current_part)
                    result.append({
                        "chunk_id": chunk["chunk_id"],
                        "page": chunk["page"],
                        "text": part_text,
                        "source": chunk.get("source", ""),
                    })
                    current_part = []
                    current_length = 0
                if sentence_len > max_length:
                    split_parts = self._split_long_text(sentence, max_length)
                    for part in split_parts:
                        if current_length + len(part) > max_length and current_part:
                            part_text = " ".join(current_part)
                            result.append({
                                "chunk_id": chunk["chunk_id"],
                                "page": chunk["page"],
                                "text": part_text,
                                "source": chunk.get("source", ""),
                            })
                            current_part = []
                            current_length = 0
                        current_part.append(part)
                        current_length += len(part) + 1
                else:
                    current_part.append(sentence)
                    current_length += sentence_len + 1
            if current_part:
                part_text = " ".join(current_part)
                result.append({
                    "chunk_id": chunk["chunk_id"],
                    "page": chunk["page"],
                    "text": part_text,
                    "source": chunk.get("source", ""),
                })
        return result

    def _split_long_text(self, text: str, max_length: int) -> List[str]:
        if len(text) <= max_length:
            return [text]
        parts = []
        words = text.split()
        current_part = []
        current_length = 0
        for word in words:
            word_len = len(word)
            if current_length + word_len + 1 > max_length and current_part:
                parts.append(" ".join(current_part))
                current_part = []
                current_length = 0
            current_part.append(word)
            current_length += word_len + 1
        if current_part:
            parts.append(" ".join(current_part))
        return parts


