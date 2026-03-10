from typing import List, Dict, Optional
import re
from pipeline.data import TextChunks, StatementsWithMedicalEntities
from pipeline.models import NeuralModel
from pipeline.transforms.acronym_expander import AcronymExpander
DEFAULT_KEEP_LABELS = [
    "Medication",
    "Disease_disorder",
    "Diagnostic_procedure",
    "Sign_symptom",
    "Therapeutic_procedure",
]
# Preferred full phrases for table triples: when subject/object equals one of these, use it as the entity (so Stage D links e.g. HFrEF not just "heart failure").
PREFERRED_TABLE_PHRASES = {
    "heart failure with reduced ejection fraction",
    "heart failure with preserved ejection fraction",
    "heart failure with mildly reduced ejection fraction",
    "heart failure with improved ejection fraction",
}
class RecognizeEntities:
    def __init__(
        self,
        neural_model: NeuralModel,
        min_score: float = 0.55,
        keep_labels: Optional[List[str]] = None,
        filter_labels: bool = True,
        acronym_file: str = None,
        verbose: bool = False,
    ):
        self.neural_model = neural_model
        self.min_score = min_score
        self.acronym_expander = AcronymExpander(acronym_file)
        self.filter_labels = filter_labels
        self.verbose = verbose
        if self.filter_labels:
            self.keep_labels = keep_labels if keep_labels else DEFAULT_KEEP_LABELS
            self.keep_labels_lower = [label.lower() for label in self.keep_labels]
        else:
            self.keep_labels = None
            self.keep_labels_lower = None
    def infer(self, text_chunks: TextChunks) -> StatementsWithMedicalEntities:
        result = StatementsWithMedicalEntities()
        chunks = text_chunks.get_chunks()
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            if self.verbose and total > 0:
                print(f"    Processing chunk {i + 1}/{total}...", flush=True)
            original_text = chunk.get("text", "")
            expanded_text = self.acronym_expander.expand(original_text)
            if not self._should_process_chunk(expanded_text):
                result.add_statement({
                    "chunk_id": chunk.get("chunk_id"),
                    "page": chunk.get("page"),
                    "source": chunk.get("source", ""),
                    "text": expanded_text,
                    "original_text": original_text,
                    "entities": []
                })
                continue
            extracted_entities = self.neural_model.extract_entities(
                text=expanded_text,
                min_score=self.min_score,
                max_entities=10000
            )
            validated_entities = self._validate_entities(extracted_entities, expanded_text)
            if self.filter_labels:
                validated_entities = self._filter_by_labels(validated_entities)
            result.add_statement({
                "chunk_id": chunk.get("chunk_id"),
                "page": chunk.get("page"),
                "source": chunk.get("source", ""),
                "text": expanded_text,
                "original_text": original_text,
                "entities": validated_entities
            })
        return result

    def enrich_triples_with_entities(self, table_triples: List[Dict]) -> List[Dict]:
        """Run NER on subject and object of each triple; add an 'entities' field to each triple.
        Subject and object are expanded, cleaned for NER, then written back so output reflects
        the text used for entity extraction."""
        enriched = []
        total = len(table_triples)
        for j, triple in enumerate(table_triples):
            if self.verbose and total > 0:
                print(f"    Enriching triple {j + 1}/{total}...", flush=True)
            out = dict(triple)
            subject = (triple.get("subject") or "").strip()
            obj = (triple.get("object") or "").strip()
            entities = []
            seen_texts = set()
            for i, text in enumerate((subject, obj)):
                if not text:
                    continue
                expanded = self.acronym_expander.expand(text)
                if not expanded or not expanded.strip():
                    expanded = text
                expanded = self._clean_table_phrase_for_ner(expanded)
                if i == 0:
                    out["subject"] = expanded
                else:
                    out["object"] = expanded
                extracted = self.neural_model.extract_entities(
                    text=expanded,
                    min_score=self.min_score,
                    max_entities=1000
                )
                validated = self._validate_entities(extracted, expanded)
                if self.filter_labels:
                    validated = self._filter_by_labels(validated)
                for ent in validated:
                    t = (ent.get("text") or "").strip()
                    if not t:
                        continue
                    t = " ".join(t.split())
                    key = t.lower()
                    if key in seen_texts:
                        continue
                    seen_texts.add(key)
                    entities.append({"text": t, "label": ent.get("label", ""), "score": ent.get("score")})
            # Prefer full phrases for subject/object: if cleaned subject equals a preferred phrase, add it and drop substring entities (e.g. "heart failure").
            out["entities"] = self._apply_preferred_phrases(entities, out.get("subject", ""), out.get("object", ""))
            enriched.append(out)
        return enriched

    def _should_process_chunk(self, text: str) -> bool:
        if not text or not text.strip():
            return False
        text = text.strip()
        words = text.split()
        alpha_chars = sum(1 for c in text if c.isalpha())
        total_chars = len(text)
        if total_chars > 0 and alpha_chars / total_chars < 0.5:
            return False
        if re.match(r'^[\d\s\.\-,;:]+$', text):
            return False
        unique_words = set(w.lower() for w in words if len(w) > 2)
        if len(words) > 10 and len(unique_words) < len(words) * 0.3:
            return False
        return True
    def _validate_entities(self, entities: List[Dict], original_text: str) -> List[Dict]:
        if not entities:
            return []
        validated = []
        seen_texts = set()
        for ent in entities:
            text = ent.get("text", "").strip()
            if not text:
                continue
            text = " ".join(text.split())
            text_key = text.lower()
            if text_key in seen_texts:
                continue
            if not self._has_valid_boundary(text, original_text):
                continue
            validated_ent = {"text": text, "label": ent.get("label", ""), "score": ent.get("score")}
            validated.append(validated_ent)
            seen_texts.add(text_key)
        return validated
    def _has_valid_boundary(self, entity_text: str, original_text: str) -> bool:
        if not entity_text or not original_text:
            return True
        escaped_text = re.escape(entity_text)
        pattern = rf'(?<![a-zA-Z]){escaped_text}(?![a-zA-Z])'
        try:
            match = re.search(pattern, original_text, re.IGNORECASE)
            return match is not None
        except re.error:
            return True
    def _filter_by_labels(self, entities: List[Dict]) -> List[Dict]:
        if not self.keep_labels_lower:
            return entities
        filtered = []
        for ent in entities:
            label = ent.get("label", "")
            label_lower = label.lower()
            if any(keep_label in label_lower or label_lower in keep_label 
                   for keep_label in self.keep_labels_lower):
                filtered.append(ent)
        return filtered

    def _apply_preferred_phrases(
        self, entities: List[Dict], cleaned_subject: str, cleaned_object: str
    ) -> List[Dict]:
        """When cleaned subject equals a preferred full phrase, add it and drop substring entities (e.g. 'heart failure'). Reuse score from dropped substring if any. Return entities without start/end."""
        subject_lower = (cleaned_subject or "").strip().lower()
        subject_matches = subject_lower in {p.lower() for p in PREFERRED_TABLE_PHRASES}
        result = []
        if subject_matches and cleaned_subject:
            # Reuse score from a dropped substring entity (e.g. "heart failure") if present
            preferred_score = None
            for e in entities:
                t = (e.get("text") or "").strip()
                if not t:
                    continue
                t_lower = t.lower()
                if t_lower == subject_lower:
                    continue
                if subject_lower.startswith(t_lower + " ") or subject_lower == t_lower:
                    s = e.get("score")
                    if s is not None and (preferred_score is None or (isinstance(s, (int, float)) and isinstance(preferred_score, (int, float)) and s > preferred_score)):
                        preferred_score = s
            result.append({"text": cleaned_subject.strip(), "label": "Disease_disorder", "score": preferred_score})
            for e in entities:
                t = (e.get("text") or "").strip()
                if not t:
                    continue
                t_lower = t.lower()
                if t_lower == subject_lower:
                    continue
                if subject_lower.startswith(t_lower + " ") or subject_lower == t_lower:
                    continue
                result.append({"text": t, "label": e.get("label", ""), "score": e.get("score")})
            return result
        for e in entities:
            result.append({"text": (e.get("text") or "").strip(), "label": e.get("label", ""), "score": e.get("score")})
        return result

    def _clean_table_phrase_for_ner(self, text: str) -> str:
        """Clean expanded table subject/object phrase before NER: trim, collapse whitespace,
        and remove redundant parenthetical duplicates (e.g. 'X (abbreviated X)' -> 'X').
        Used only for table triple enrichment; conservative."""
        if not text:
            return text
        s = text.strip()
        s = " ".join(s.split())
        if not s:
            return s
        # Single trailing parenthetical: "X (Y)" -> possibly keep only X if Y is redundant
        m = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", s)
        if not m:
            return s
        x, y = m.group(1).strip(), m.group(2).strip()
        x = " ".join(x.split())
        y = " ".join(y.split())
        if not x or not y:
            return s
        x_lower, y_lower = x.lower(), y.lower()
        # Duplicate: same content (normalized)
        if x_lower == y_lower:
            return x
        # Y is a substring of X -> redundant
        if y_lower in x_lower:
            return x
        # Y looks like shorthand of X: same leading words, Y shorter (e.g. "heart failure with reduced EF" vs "heart failure with reduced ejection fraction")
        x_words, y_words = x.split(), y.split()
        if len(y_words) >= 2 and len(y_words) <= len(x_words):
            if x_lower.startswith(y_lower) or y_lower.startswith(x_lower[:len(" ".join(y_words))].lower()):
                return x
            # First 2+ words of Y match first 2+ words of X and Y is not longer
            n = min(2, len(y_words), len(x_words))
            if n >= 2 and [w.lower() for w in y_words[:n]] == [w.lower() for w in x_words[:n]]:
                return x
        return s
