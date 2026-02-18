"""
[Device] validate (Stage e)

Extract factual statements from candidate chunks using an LLM.
No factuality validation here: experts validate statements manually later.
Output: subject, predicate, object, exception, duration (and optional source/entities).

Input: CandidateStatements (from Stage d)
Output: ValidatedFactsAndQualifiers (extracted statements for expert validation)

A single paragraph may yield multiple factual statements; all are extracted.
"""

import re
from typing import Optional, List, Dict
from pipeline.data import CandidateStatements, ValidatedFactsAndQualifiers
from pipeline.models import ValidationModel


class Validate:
    """
    Device for extracting factual statements from candidate text (Stage e).
    LLM extracts subject, predicate, object, exception, duration per statement.
    No validation: all extractions are passed through for expert review.
    """

    def __init__(
        self,
        validation_model: ValidationModel,
        max_new_tokens: int = 400,
        batch_size: int = 4
    ):
        """
        Initialize extraction device (Stage e).
        
        Args:
            validation_model: LLM used for extraction (no factuality scoring)
            max_new_tokens: Token limit (400 allows multiple statements per chunk)
            batch_size: Number of chunks to process in parallel
        """
        self.validation_model = validation_model
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
    
    def validate(
        self,
        candidate_statements: CandidateStatements
    ) -> ValidatedFactsAndQualifiers:
        """
        Extract factual statements from each candidate chunk. Multiple statements per chunk are allowed.
        Output is intended for manual validation by medical experts.
        
        Args:
            candidate_statements: Candidate chunks with entities (from Stage d)
        
        Returns:
            ValidatedFactsAndQualifiers with extracted statements (subject, predicate, object, exception, duration)
        """
        validated = ValidatedFactsAndQualifiers()
        statements = candidate_statements.get_all()
        total = len(statements)
        texts = [s["text"] for s in statements]
        entities_list = [s.get("entities", []) for s in statements]
        
        # Batch extraction (returns one list of statements per chunk)
        all_outputs: List[List[Dict]] = []
        total_batches = (total + self.batch_size - 1) // self.batch_size
        for batch_idx, i in enumerate(range(0, total, self.batch_size)):
            batch_texts = texts[i:i + self.batch_size]
            batch_entities = entities_list[i:i + self.batch_size]
            batch_results = self.validation_model.extract_statements_batch(
                texts=batch_texts,
                entities_list=batch_entities,
                max_new_tokens=self.max_new_tokens,
                batch_size=self.batch_size
            )
            all_outputs.extend(batch_results)
            if (batch_idx + 1) % 6 == 0 or (batch_idx + 1) == total_batches:
                print(f"[Stage e] {batch_idx + 1}/{total_batches} batches done")
        
        for idx, (stmt, extracted_list) in enumerate(zip(statements, all_outputs)):
            entities = stmt.get("entities", [])
            simplified_entities = self._filter_entities([
                {
                    "text": e.get("concept_name") or e.get("text"),
                    "label": e.get("label"),
                    "id": e.get("umls_id") or e.get("concept_id")
                }
                for e in entities
            ])
            source_text = stmt.get("text", "")
            chunk_id = stmt.get("chunk_id")
            source_pdf = stmt.get("source_pdf")
            page = stmt.get("page")
            
            for one in extracted_list:
                if not isinstance(one, dict):
                    continue
                subject = one.get("subject") or "unspecified"
                predicate = (one.get("predicate") or "").strip()
                obj = one.get("object")
                exception = one.get("exception")
                duration = one.get("duration")
                # Skip only malformed extractions (no predicate at all). No blacklist—experts validate.
                if not predicate:
                    continue
                duration = self._clean_duration(duration)
                entities_aligned = self._filter_entities_by_spo_alignment(
                    simplified_entities, subject, obj or ""
                )
                
                validated.add_validated({
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "exception": exception,
                    "duration": duration,
                    "entities": entities_aligned,
                    "source_text": source_text,
                    "chunk_id": chunk_id,
                    "source_pdf": source_pdf,
                    "page": page,
                })
        
        return validated
    
    def _filter_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Filter out fragmented/invalid entities.
        Removes NER artifacts like 2-char fragments, numbers, etc.
        
        Args:
            entities: List of entity dicts
        
        Returns:
            Filtered list of valid entities
        """
        filtered = []
        seen_texts = set()
        
        for entity in entities:
            text = entity.get("text", "")
            if not text or not isinstance(text, str):
                continue
            
            text_clean = text.strip()
            text_lower = text_clean.lower()
            
            # Skip very short entities (likely fragments)
            if len(text_clean) <= 2:
                continue
            
            # Skip pure numbers
            if text_clean.replace(".", "").replace(",", "").isdigit():
                continue
            
            # Skip entities that look like page/section numbers
            if re.match(r'^[\d\s\.\-]+$', text_clean):
                continue
            
            # Skip entities that are clearly fragmented (contain weird patterns)
            if re.match(r'^\d+\s*\.\s*\d+', text_clean):  # e.g., "14. 1 pregnancy"
                continue
            
            # Skip duplicates (case-insensitive)
            if text_lower in seen_texts:
                continue
            seen_texts.add(text_lower)
            
            filtered.append(entity)
        
        return filtered
    
    def _filter_entities_by_spo_alignment(
        self, entities: List[Dict], subject: str, obj: str
    ) -> List[Dict]:
        """
        Filter entities to only those that appear in the subject or object.
        This ensures entity-SPO alignment.
        
        Args:
            entities: List of entity dicts
            subject: Subject text (population)
            object: Object text (treatment/action)
        
        Returns:
            Filtered list of entities that appear in subject or object
        """
        if not entities:
            return []
        
        aligned_entities = []
        subject_lower = subject.lower() if subject else ""
        object_lower = obj.lower() if obj else ""
        
        for entity in entities:
            entity_text = entity.get("text", "")
            if not entity_text:
                continue
            
            entity_text_lower = entity_text.lower()
            
            # Check if entity text appears in subject or object
            # Use word boundary matching to avoid partial matches
            # e.g., "heart" should match "heart failure" but not "hear"
            appears_in_subject = entity_text_lower in subject_lower
            appears_in_object = entity_text_lower in object_lower
            
            if appears_in_subject or appears_in_object:
                aligned_entities.append(entity)
        
        return aligned_entities
    
    def _clean_duration(self, duration: Optional[str]) -> Optional[str]:
        """
        Clean and validate duration field.
        Filters out study periods mistakenly extracted as guideline durations.
        
        Args:
            duration: Raw duration string from LLM output
        
        Returns:
            Cleaned duration or None if invalid
        """
        if not duration or not isinstance(duration, str):
            return None
        
        duration_clean = duration.strip()
        duration_lower = duration_clean.lower()
        
        # Filter out study/observation periods (not actual guideline durations)
        # e.g., "11 years", "5-year study period", "during follow-up"
        study_period_patterns = [
            r'\b(study|studies|trial|observation|follow-?up|period)\b',
            r'\b(retrospective|prospective|longitudinal)\b',
            r'\b(registry|cohort|analysis)\b',
        ]
        
        for pattern in study_period_patterns:
            if re.search(pattern, duration_lower):
                return None
        
        # Filter out very long durations that are likely study periods
        # Most clinical guideline restrictions are days/weeks/months, not many years
        year_match = re.search(r'(\d+)\s*(?:-\s*\d+\s*)?years?', duration_lower)
        if year_match:
            years = int(year_match.group(1))
            if years > 5:  # More than 5 years is likely a study period
                return None
        
        # Filter out durations that are just numbers without units
        if re.fullmatch(r'\d+', duration_clean):
            return None
        
        # Must contain temporal keywords to be valid
        temporal_keywords = [
            "day", "week", "month", "year", "hour", "minute",
            "until", "before", "after", "within", "during", "following"
        ]
        has_temporal = any(kw in duration_lower for kw in temporal_keywords)
        
        if not has_temporal:
            return None
        
        return duration_clean