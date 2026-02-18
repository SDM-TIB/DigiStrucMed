"""
[Data] statements_with_medical_entities

Data structure containing text chunks paired with their extracted medical entities.
Output of Stage c (Learning from data to symbol - NER with acronym expansion).
"""

from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class StatementsWithMedicalEntities:
    """
    Container for text statements paired with their medical entities.
    
    Each entry contains:
    - chunk_id: Identifier for the chunk
    - page: Source page number
    - source: Source document
    - text: Text with acronyms expanded (for NER and LLM alignment)
    - original_text: Original chunk text before acronym expansion
    - entities: List of extracted medical entities
    
    This is the output of Stage c (NER with acronym expansion).
    """
    
    statements: List[Dict] = field(default_factory=list)
    
    def add_statement(self, statement: Dict) -> None:
        """
        Add a statement with entities.
        
        Args:
            statement: Dict containing text, original_text, and entities
        """
        self.statements.append(statement)
    
    def get_all(self) -> List[Dict]:
        """Get all statements with entities."""
        return self.statements
    
    def count(self) -> int:
        """Get number of statements."""
        return len(self.statements)
    
    def get_entity_count(self) -> int:
        """Get total number of entities across all statements."""
        return sum(len(s.get("entities", [])) for s in self.statements)
    
    def __repr__(self) -> str:
        total_entities = self.get_entity_count()
        return f"StatementsWithMedicalEntities(statements={self.count()}, entities={total_entities})"
