"""
[Data] candidate_statements

Data structure containing statements with linked entities.
Output of Stage d (entity linking). Input to Stage e (LLM for SPO extraction and validation).
"""

from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class CandidateStatements:
    """
    Container for statements with linked entities.

    Each entry: chunk_id, page, source, text, original_text, entities (UMLS-linked).
    Stage e (LLM) uses these for SPO extraction and decides whether a statement is factual.
    """
    
    statements: List[Dict] = field(default_factory=list)
    
    def add_statement(self, statement: Dict) -> None:
        """
        Add a candidate statement.
        
        Args:
            statement: Dict containing text, entities, and candidate info
        """
        self.statements.append(statement)
    
    def get_all(self) -> List[Dict]:
        """Get all statements."""
        return self.statements
    
    def get_candidates(self) -> List[Dict]:
        """Get all statements (all are sent to LLM; LLM decides if factual)."""
        return self.statements

    def count(self) -> int:
        """Get total number of statements."""
        return len(self.statements)

    def count_candidates(self) -> int:
        """Get number of statements (same as count(); LLM filters by factuality)."""
        return len(self.statements)

    def __repr__(self) -> str:
        return f"CandidateStatements(statements={self.count()})"
