"""
[NumericalModel] entities_linker

Entity linking/knowledge base model (e.g., UMLS).
Maps entity mentions to canonical IDs/concepts.

Enhanced with UMLS CSV support for real medical concept linking.
"""

from typing import List, Dict, Optional, Tuple
import re
import csv
from pathlib import Path


class EntitiesLinker:
    """
    Entity linking model that maps entity mentions to canonical concepts.
    
    Primary purpose: Map entity variants to the same canonical ID for KG deduplication.
    Example: "HF", "heart failure", "HFrEF" → same UMLS ID (e.g., "C0018801")
    
    Supports:
    - UMLS CSV database lookup (CUI, Label format)
    - Rule-based pattern matching (fallback)
    - Text normalization
    
    Note: Relationship extraction is done by the LLM during validation, not here.
    """
    
    def __init__(
        self,
        knowledge_base: str = "rule-based",
        umls_csv_path: Optional[str] = None,
        filter_unmatched: bool = False,
        use_partial_umls_match: bool = False,
        partial_match_max_scan: int = 2000,
    ):
        """
        Initialize entity linker.

        Args:
            knowledge_base: Type of KB ("rule-based", "umls")
            umls_csv_path: Path to UMLS CSV (CUI, Label format)
            filter_unmatched: If True, drop entities without UMLS match
            use_partial_umls_match: If True, try partial text match in UMLS (slower on large DB)
            partial_match_max_scan: Max UMLS entries to scan for partial match (when enabled)
        """
        self.knowledge_base = knowledge_base
        self.filter_unmatched = filter_unmatched
        self.use_partial_umls_match = use_partial_umls_match
        self.partial_match_max_scan = partial_match_max_scan
        self.umls_db = {}
        self.umls_by_normalized = {}  # normalized form -> umls_data for O(1) Strategy 2
        self.umls_loaded = False

        if umls_csv_path:
            self._load_umls_csv(umls_csv_path)
            self.knowledge_base = "umls"

        # Fallback patterns (rule-based when no UMLS or no match)
        self.medication_patterns = {
            r"\bACE[- ]?inhibitor[s]?\b": "ACE_inhibitor",
            r"\bACEi\b": "ACE_inhibitor",
            r"\bARB[s]?\b": "ARB",
            r"\bbeta[- ]?blocker[s]?\b": "beta_blocker",
            r"\bMRA[s]?\b": "mineralocorticoid_receptor_antagonist",
            r"\bdiuretic[s]?\b": "diuretic",
            r"\bSGLT2[- ]?inhibitor[s]?\b": "SGLT2_inhibitor",
            r"\bSGLT2i\b": "SGLT2_inhibitor",
            r"\bDOAC[s]?\b": "direct_acting_oral_anticoagulant",
            r"\bNSAID[s]?\b": "nonsteroidal_anti_inflammatory_drug",
            r"\bRAASi\b": "renin_angiotensin_aldosterone_system_inhibitor",
        }
        self.condition_patterns = {
            r"\bHFrEF\b": "heart_failure_reduced_ejection_fraction",
            r"\bHFpEF\b": "heart_failure_preserved_ejection_fraction",
            r"\bHFmrEF\b": "heart_failure_mildly_reduced_ejection_fraction",
            r"\bHFimpEF\b": "heart_failure_improved_ejection_fraction",
            r"\bheart failure\b": "heart_failure",
            r"\barrhythmia[s]?\b": "cardiac_arrhythmia",
            r"\batrial fibrillation\b": "atrial_fibrillation",
            r"\bGDMT\b": "guideline_directed_medical_therapy",
            r"\bLVEF\b": "left_ventricular_ejection_fraction",
            r"\beGFR\b": "estimated_glomerular_filtration_rate",
        }
    
    def _load_umls_csv(self, csv_path: str) -> None:
        """
        Load UMLS database from CSV file.
        
        Expected CSV format:
        "CUI","Label"
        "C0018801","heart failure"
        ...
        
        Builds lookup: label_lower -> {cui, label}
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            return
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                
                # Skip header
                header = next(reader, None)
                
                count = 0
                for row in reader:
                    if len(row) >= 2:
                        cui = row[0].strip().strip('"')
                        label = row[1].strip().strip('"')
                        
                        if cui and label and len(label) > 1:
                            label_lower = label.lower()
                            data = {"cui": cui, "label": label}
                            self.umls_db[label_lower] = data
                            norm = self._normalize_for_matching(label_lower)
                            if norm and norm not in self.umls_by_normalized:
                                self.umls_by_normalized[norm] = data
                            count += 1

            self.umls_loaded = True
            
        except Exception as e:
            self.umls_loaded = False
    
    def link_entities(
        self, 
        entities: List[Dict], 
        context_text: str = ""
    ) -> List[Dict]:
        """
        Link entities to canonical concepts.
        
        Args:
            entities: List of extracted entities
            context_text: Original text for context
        
        Returns:
            Entities with canonical concept information:
            - concept_name: The canonical name (from UMLS or original text)
            - umls_id: The UMLS CUI (or normalized ID if no UMLS match)
            - label: Entity type (Medication, Disease_disorder, etc.)
            - score, start, end: Preserved from input
            
            If filter_unmatched=True, only returns entities with UMLS match
        """
        linked_entities = []
        
        for entity in entities:
            entity_text = entity.get("text", "")
            entity_label = entity.get("label", "")
            
            # Try to find UMLS concept
            umls_match = self._find_umls_concept(entity_text)
            
            if umls_match:
                # Build clean entity with UMLS data
                linked = {
                    "concept_name": umls_match["label"],
                    "umls_id": umls_match["cui"],
                    "label": entity_label,
                }
                # Preserve other fields (score, start, end, etc.)
                for key in ["score", "start", "end"]:
                    if key in entity:
                        linked[key] = entity[key]
                
                linked_entities.append(linked)
            else:
                # No UMLS match - skip if filtering
                if self.filter_unmatched:
                    continue
                
                # Try pattern-based matching (fallback)
                concept_id, concept_name = self._find_pattern_concept(entity_text, entity_label)

                if concept_id:
                    linked = {
                        "concept_name": concept_name,
                        "umls_id": concept_id,
                        "label": entity_label,
                    }
                else:
                    # No mapping found - use normalized form
                    linked = {
                        "concept_name": entity_text,
                        "umls_id": self._normalize_text(entity_text),
                        "label": entity_label,
                    }

                # Preserve other fields
                for key in ["score", "start", "end"]:
                    if key in entity:
                        linked[key] = entity[key]

                linked_entities.append(linked)
        
        return linked_entities
    
    def _find_umls_concept(self, text: str) -> Optional[Dict]:
        """
        Find UMLS concept for entity text.
        Strategy 1: exact match. Strategy 2: normalized match (O(1) via index).
        Strategy 3 (optional): partial match, capped for performance.
        """
        if not self.umls_loaded:
            return None

        text_lower = text.lower().strip()
        if text_lower in self.umls_db:
            return self.umls_db[text_lower]

        normalized = self._normalize_for_matching(text_lower)
        if normalized in self.umls_by_normalized:
            return self.umls_by_normalized[normalized]

        if not self.use_partial_umls_match or len(text_lower) < 5:
            return None

        scanned = 0
        for umls_label, umls_data in self.umls_db.items():
            if scanned >= self.partial_match_max_scan:
                break
            scanned += 1
            if text_lower in umls_label and len(text_lower) >= len(umls_label) * 0.5:
                return umls_data
            if umls_label in text_lower and len(umls_label) >= 5:
                return umls_data
        return None
    
    def _find_pattern_concept(self, text: str, label: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Find concept using pattern-based matching (fallback).
        
        Returns:
            (concept_id, concept_name) or (None, None)
        """
        text_lower = text.lower()
        
        # Check medication patterns
        for pattern, concept_id in self.medication_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                concept_name = concept_id.replace('_', ' ').title()
                return concept_id, concept_name
        
        # Check condition patterns
        for pattern, concept_id in self.condition_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                concept_name = concept_id.replace('_', ' ').title()
                return concept_id, concept_name
        
        return None, None
    
    def _normalize_for_matching(self, text: str) -> str:
        """Normalize text for UMLS matching."""
        normalized = text.lower()
        normalized = re.sub(r'[^\w\s]', '', normalized)  # Remove punctuation
        normalized = re.sub(r'\s+', ' ', normalized)  # Normalize spaces
        return normalized.strip()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize entity text for use as ID."""
        normalized = text.lower()
        normalized = re.sub(r'[^\w\s-]', '', normalized)
        normalized = re.sub(r'[-\s]+', '_', normalized)
        return normalized
    
    def get_umls_stats(self) -> Dict:
        """Return UMLS database statistics."""
        return {
            "umls_loaded": self.umls_loaded,
            "total_concepts": len(self.umls_db),
            "knowledge_base": self.knowledge_base
        }