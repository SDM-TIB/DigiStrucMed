"""
[Device] infer:recognize_entities

Run NER over text_chunks to detect entity mentions.
Uses NER_model with acronym_expander (as shown in diagram).

Input: TextChunks (from Stage b)
Output: StatementsWithMedicalEntities (Stage c output)

Enhanced with:
- Acronym expansion before NER (from NSSC paper)
- Chunk pre-filtering (skip low-quality chunks)
- Entity quality validation
- Label filtering (keeps only medically-relevant entities)

Note: Acronym expansion is integrated into this stage (NER_model_with_acronym_expander).
"""

from typing import List, Dict, Optional
import re
from pipeline.data import TextChunks, StatementsWithMedicalEntities
from pipeline.models import NeuralModel
from pipeline.transforms.acronym_expander import AcronymExpander

# Default labels to keep for KG construction (medically-relevant entities)
DEFAULT_KEEP_LABELS = [
    "Medication",            # Drugs, treatments
    "Disease_disorder",      # Conditions, diseases
    "Diagnostic_procedure",  # Tests, exams, imaging
    "Sign_symptom",          # Symptoms, signs
    "Therapeutic_procedure", # Treatments, surgeries, interventions
]


class RecognizeEntities:
    """
    Device for recognizing biomedical entities in text chunks.
    Uses NER model with acronym expansion (NER_model_with_acronym_expander).
    
    Pipeline flow (matching diagram):
    text_chunks → recognize_entities (NER + acronym expansion) → statements_with_medical_entities
    
    Enhanced features:
    - Acronym expansion before NER (improves entity recognition)
    - Pre-filters chunks that are unlikely to contain useful entities
    - Validates entity quality (boundary checks, normalization)
    - Filters by label type (keeps only medically-relevant entities)
    
    Output: StatementsWithMedicalEntities - contains text + expanded text + entities
    """
    
    def __init__(
        self,
        neural_model: NeuralModel,
        min_score: float = 0.55,
        keep_labels: Optional[List[str]] = None,
        filter_labels: bool = True,
        acronym_file: str = None
    ):
        """
        Initialize entity recognition device.

        Args:
            neural_model: Neural model for entity extraction (e.g. NER)
            min_score: Minimum confidence score for entities
            keep_labels: List of label types to keep (None = use defaults)
            filter_labels: Whether to filter by labels (default True)
            acronym_file: Path to acronym JSON file (None = use default)
        """
        self.neural_model = neural_model
        self.min_score = min_score

        # Initialize acronym expander (part of NER_model_with_acronym_expander)
        self.acronym_expander = AcronymExpander(acronym_file)

        # Label filtering for KG construction
        self.filter_labels = filter_labels
        if self.filter_labels:
            self.keep_labels = keep_labels if keep_labels else DEFAULT_KEEP_LABELS
            self.keep_labels_lower = [label.lower() for label in self.keep_labels]
        else:
            self.keep_labels = None
            self.keep_labels_lower = None
    
    def infer(self, text_chunks: TextChunks) -> StatementsWithMedicalEntities:
        """
        Infer entities from text chunks.
        
        Pipeline: text_chunks → (acronym expansion) → NER → statements_with_medical_entities
        
        Args:
            text_chunks: Text chunks from Stage b (chunk_text)
        
        Returns:
            StatementsWithMedicalEntities containing text + entities
        """
        result = StatementsWithMedicalEntities()
        chunks = text_chunks.get_chunks()

        for chunk in chunks:
            original_text = chunk.get("text", "")

            # Step 1: Expand acronyms (part of NER_model_with_acronym_expander)
            expanded_text = self.acronym_expander.expand(original_text)

            # Pre-filter: skip chunks unlikely to contain entities
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

            # Step 2: Extract entities using NER model on expanded text (no cap)
            extracted_entities = self.neural_model.extract_entities(
                text=expanded_text,
                min_score=self.min_score,
                max_entities=10000
            )

            # Step 3: Post-process: validate and normalize entities
            validated_entities = self._validate_entities(extracted_entities, expanded_text)

            # Step 4: Filter by label type (keep only medically-relevant entities)
            if self.filter_labels:
                validated_entities = self._filter_by_labels(validated_entities)

            # Add statement with entities
            result.add_statement({
                "chunk_id": chunk.get("chunk_id"),
                "page": chunk.get("page"),
                "source": chunk.get("source", ""),
                "text": expanded_text,  # Expanded text (for LLM in Stage e)
                "original_text": original_text,  # Original text (for reference)
                "entities": validated_entities
            })
        
        return result
    
    def _should_process_chunk(self, text: str) -> bool:
        """
        Determine if chunk should be processed for NER.
        Filters out chunks unlikely to contain useful entities.
        
        Uses general patterns, not hardcoded content.
        """
        if not text or not text.strip():
            return False
        
        text = text.strip()
        words = text.split()

        # Calculate text quality indicators
        alpha_chars = sum(1 for c in text if c.isalpha())
        total_chars = len(text)
        
        # Very low alphabetic content (likely numbers, symbols, or formatting)
        if total_chars > 0 and alpha_chars / total_chars < 0.5:
            return False
        
        # Pure number sequences (page numbers, table data)
        if re.match(r'^[\d\s\.\-,;:]+$', text):
            return False
        
        # Very repetitive content (headers, formatting artifacts)
        unique_words = set(w.lower() for w in words if len(w) > 2)
        if len(words) > 10 and len(unique_words) < len(words) * 0.3:
            return False
        
        return True
    
    def _validate_entities(self, entities: List[Dict], original_text: str) -> List[Dict]:
        """
        Validate and normalize extracted entities.
        
        - Checks entity boundaries (not partial words)
        - Normalizes entity text (consistent whitespace)
        - Removes duplicates within chunk
        
        Uses general patterns, not hardcoded lists.
        """
        if not entities:
            return []
        
        validated = []
        seen_texts = set()
        
        for ent in entities:
            text = ent.get("text", "").strip()
            
            if not text:
                continue
            
            # Normalize whitespace
            text = " ".join(text.split())
            
            # Skip if already seen (case-insensitive)
            text_key = text.lower()
            if text_key in seen_texts:
                continue
            
            # Validate entity boundaries in original text
            if not self._has_valid_boundary(text, original_text):
                continue
            
            # Update entity with normalized text
            validated_ent = ent.copy()
            validated_ent["text"] = text
            
            validated.append(validated_ent)
            seen_texts.add(text_key)
        
        return validated
    
    def _has_valid_boundary(self, entity_text: str, original_text: str) -> bool:
        """
        Check if entity has valid word boundaries in original text.
        Prevents partial word matches (e.g., "ion" from "medication").
        
        Uses regex word boundary matching.
        """
        if not entity_text or not original_text:
            return True  # Can't validate, assume OK
        
        # Escape special regex characters
        escaped_text = re.escape(entity_text)
        
        # Check for word boundary match
        # Allow for punctuation/numbers adjacent to entity
        pattern = rf'(?<![a-zA-Z]){escaped_text}(?![a-zA-Z])'
        
        try:
            match = re.search(pattern, original_text, re.IGNORECASE)
            return match is not None
        except re.error:
            # If regex fails, assume valid
            return True
    
    def _filter_by_labels(self, entities: List[Dict]) -> List[Dict]:
        """
        Filter entities to keep only specified label types.
        
        This removes noise entities (Detailed_description, Lab_value, etc.)
        and keeps only medically-relevant entities for KG construction.
        
        Args:
            entities: List of entities to filter
        
        Returns:
            Filtered list of entities with only kept labels
        """
        if not self.keep_labels_lower:
            return entities
        
        filtered = []
        for ent in entities:
            label = ent.get("label", "")
            label_lower = label.lower()
            
            # Check if label matches any in keep list (case-insensitive)
            # Uses substring matching to handle variations like "Disease_disorder"
            if any(keep_label in label_lower or label_lower in keep_label 
                   for keep_label in self.keep_labels_lower):
                filtered.append(ent)
        
        return filtered
