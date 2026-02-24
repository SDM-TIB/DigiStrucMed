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
class RecognizeEntities:
    def __init__(
        self,
        neural_model: NeuralModel,
        min_score: float = 0.55,
        keep_labels: Optional[List[str]] = None,
        filter_labels: bool = True,
        acronym_file: str = None
    ):
        self.neural_model = neural_model
        self.min_score = min_score
        self.acronym_expander = AcronymExpander(acronym_file)
        self.filter_labels = filter_labels
        if self.filter_labels:
            self.keep_labels = keep_labels if keep_labels else DEFAULT_KEEP_LABELS
            self.keep_labels_lower = [label.lower() for label in self.keep_labels]
        else:
            self.keep_labels = None
            self.keep_labels_lower = None
    def infer(self, text_chunks: TextChunks) -> StatementsWithMedicalEntities:
        result = StatementsWithMedicalEntities()
        chunks = text_chunks.get_chunks()
        for chunk in chunks:
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
        """Run NER on subject and object of each triple; add an 'entities' field to each triple."""
        enriched = []
        for triple in table_triples:
            out = dict(triple)
            subject = (triple.get("subject") or "").strip()
            obj = (triple.get("object") or "").strip()
            entities = []
            seen_texts = set()
            for text in (subject, obj):
                if not text:
                    continue
                expanded = self.acronym_expander.expand(text)
                if not expanded or not expanded.strip():
                    expanded = text
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
                    entities.append(ent)
            out["entities"] = entities
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
            validated_ent = ent.copy()
            validated_ent["text"] = text
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
