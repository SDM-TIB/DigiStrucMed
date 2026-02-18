from typing import List, Dict
from dataclasses import dataclass, field
@dataclass
class CandidateStatements:
    statements: List[Dict] = field(default_factory=list)
    def add_statement(self, statement: Dict) -> None:
        self.statements.append(statement)
    def get_all(self) -> List[Dict]:
        return self.statements
    def get_candidates(self) -> List[Dict]:
        return self.statements
    def count(self) -> int:
        return len(self.statements)
    def count_candidates(self) -> int:
        return len(self.statements)
    def __repr__(self) -> str:
        return f"CandidateStatements(statements={self.count()})"
