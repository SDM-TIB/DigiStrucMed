from typing import List, Dict
from dataclasses import dataclass, field
@dataclass
class ValidatedFactsAndQualifiers:
    validated_statements: List[Dict] = field(default_factory=list)
    def add_validated(self, statement: Dict) -> None:
        self.validated_statements.append(statement)
    def get_all(self) -> List[Dict]:
        return self.validated_statements
    def count(self) -> int:
        return len(self.validated_statements)
    def __repr__(self) -> str:
        return f"ValidatedFactsAndQualifiers(count={self.count()})"
