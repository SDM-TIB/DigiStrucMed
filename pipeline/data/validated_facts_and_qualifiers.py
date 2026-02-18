"""
[Symbol] validated_facts_AND_qualifiers

Validated factual statements plus qualifiers.
"""

from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class ValidatedFactsAndQualifiers:
    """Container for validated guideline statements with qualifiers."""
    
    validated_statements: List[Dict] = field(default_factory=list)
    
    def add_validated(self, statement: Dict) -> None:
        """Add a validated statement."""
        self.validated_statements.append(statement)
    
    def get_all(self) -> List[Dict]:
        """Get all validated statements."""
        return self.validated_statements
    
    def count(self) -> int:
        """Get number of validated statements."""
        return len(self.validated_statements)
    
    def __repr__(self) -> str:
        return f"ValidatedFactsAndQualifiers(count={self.count()})"
