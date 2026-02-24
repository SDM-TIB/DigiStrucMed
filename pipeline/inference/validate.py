import re
from typing import Optional, List, Dict
from pipeline.data import CandidateStatements, ValidatedFactsAndQualifiers
from pipeline.models import ValidationModel
class Validate:
    def __init__(
        self,
        validation_model: ValidationModel,
        max_new_tokens: int = 400,
        batch_size: int = 4
    ):
        self.validation_model = validation_model
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
    def validate(
        self,
        candidate_statements: CandidateStatements
    ) -> ValidatedFactsAndQualifiers:
        validated = ValidatedFactsAndQualifiers()
        statements = candidate_statements.get_all()
        total = len(statements)
        texts = [s["text"] for s in statements]
        entities_list = [s.get("entities", []) for s in statements]
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
                    "id": e.get("cui_id") or e.get("umls_id") or e.get("concept_id")
                }
                for e in entities
            ])
            source_text = stmt.get("text", "")
            chunk_id = stmt.get("chunk_id")
            page = stmt.get("page")
            for one in extracted_list:
                if not isinstance(one, dict):
                    continue
                subject = one.get("subject") or "unspecified"
                predicate = (one.get("predicate") or "").strip()
                obj = one.get("object")
                exception = one.get("exception")
                duration = one.get("duration")
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
                    "entities": entities_aligned,
                    "source_text": source_text,
                    "chunk_id": chunk_id,
                    "exception": exception,
                    "duration": duration,
                    "page": page,
                })
        return validated

    def validate_table_triples(
        self,
        validated: ValidatedFactsAndQualifiers,
        table_triples: List[Dict],
    ) -> None:
        """Run each table_triple through LLM: validate as single fact or split object into multiple triples; append to validated (same 9-field format)."""
        total = len(table_triples)
        for idx, triple in enumerate(table_triples):
            subj = (triple.get("subject") or "").strip()
            pred = (triple.get("predicate") or "").strip()
            obj = (triple.get("object") or "").strip()
            if not pred:
                continue
            extracted = self.validation_model.validate_and_split_table_triple(
                subj, pred, obj, max_new_tokens=self.max_new_tokens
            )
            if not extracted:
                extracted = [{"subject": subj, "predicate": pred, "object": obj}]
            entities = triple.get("entities", [])
            simplified = self._filter_entities([
                {
                    "text": e.get("concept_name") or e.get("text", ""),
                    "label": e.get("label", ""),
                    "id": e.get("cui_id") or e.get("umls_id") or e.get("concept_id"),
                }
                for e in entities
            ])
            source_text = f"{subj} {pred} {obj}".strip()[:2000]
            page = triple.get("page")
            for one in extracted:
                if not isinstance(one, dict):
                    continue
                s = (one.get("subject") or "unspecified").strip()
                p = (one.get("predicate") or "").strip()
                o = one.get("object")
                if o is not None and isinstance(o, str):
                    o = o.strip()
                if not p:
                    continue
                aligned = self._filter_entities_by_spo_alignment(simplified, s, o or "")
                validated.add_validated({
                    "subject": s,
                    "predicate": p,
                    "object": o,
                    "entities": aligned,
                    "source_text": source_text,
                    "chunk_id": None,
                    "exception": one.get("exception"),
                    "duration": self._clean_duration(one.get("duration")),
                    "page": page,
                })
            if (idx + 1) % 50 == 0 or (idx + 1) == total:
                print(f"    Table triples (LLM): {idx + 1}/{total}")
        return

    def _filter_entities(self, entities: List[Dict]) -> List[Dict]:
        filtered = []
        seen_texts = set()
        for entity in entities:
            text = entity.get("text", "")
            if not text or not isinstance(text, str):
                continue
            text_clean = text.strip()
            text_lower = text_clean.lower()
            if len(text_clean) <= 2:
                continue
            if text_clean.replace(".", "").replace(",", "").isdigit():
                continue
            if re.match(r'^[\d\s\.\-]+$', text_clean):
                continue
            if re.match(r'^\d+\s*\.\s*\d+', text_clean):
                continue
            if text_lower in seen_texts:
                continue
            seen_texts.add(text_lower)
            filtered.append(entity)
        return filtered
    def _filter_entities_by_spo_alignment(
        self, entities: List[Dict], subject: str, obj: str
    ) -> List[Dict]:
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
            appears_in_subject = entity_text_lower in subject_lower
            appears_in_object = entity_text_lower in object_lower
            if appears_in_subject or appears_in_object:
                aligned_entities.append(entity)
        return aligned_entities
    def _clean_duration(self, duration: Optional[str]) -> Optional[str]:
        if not duration or not isinstance(duration, str):
            return None
        duration_clean = duration.strip()
        duration_lower = duration_clean.lower()
        study_period_patterns = [
            r'\b(study|studies|trial|observation|follow-?up|period)\b',
            r'\b(retrospective|prospective|longitudinal)\b',
            r'\b(registry|cohort|analysis)\b',
        ]
        for pattern in study_period_patterns:
            if re.search(pattern, duration_lower):
                return None
        year_match = re.search(r'(\d+)\s*(?:-\s*\d+\s*)?years?', duration_lower)
        if year_match:
            years = int(year_match.group(1))
            if years > 5:
                return None
        if re.fullmatch(r'\d+', duration_clean):
            return None
        temporal_keywords = [
            "day", "week", "month", "year", "hour", "minute",
            "until", "before", "after", "within", "during", "following"
        ]
        has_temporal = any(kw in duration_lower for kw in temporal_keywords)
        if not has_temporal:
            return None
        return duration_clean
