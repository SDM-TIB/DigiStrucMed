from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class ShaclConstraints:
    """
    Collection of SHACL-style constraints extracted from Stage E factual statements.
    Each constraint describes a rule for a medical concept (medication, condition, etc.)
    with properties like dosage range, frequency, contraindication, monitoring, etc.
    """
    constraints: List[Dict] = field(default_factory=list)

    def add_constraint(self, constraint: Dict) -> None:
        self.constraints.append(constraint)

    def get_all(self) -> List[Dict]:
        return self.constraints

    def count(self) -> int:
        return len(self.constraints)

    def get_by_cui(self, cui: str) -> List[Dict]:
        """Return all constraints targeting a specific UMLS CUI."""
        return [c for c in self.constraints if c.get("target_cui") == cui]

    def get_by_concept(self, concept: str) -> List[Dict]:
        """Return all constraints whose target_concept contains the given string (case-insensitive)."""
        concept_lower = concept.lower()
        return [
            c for c in self.constraints
            if concept_lower in (c.get("target_concept") or "").lower()
        ]

    def get_by_property(self, prop: str) -> List[Dict]:
        """Return all constraints for a specific property (dosage, frequency, etc.)."""
        prop_lower = prop.lower()
        return [c for c in self.constraints if (c.get("property") or "").lower() == prop_lower]

    def __repr__(self) -> str:
        return f"ShaclConstraints(count={self.count()})"
