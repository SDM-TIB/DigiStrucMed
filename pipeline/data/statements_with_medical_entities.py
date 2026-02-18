from typing import List, Dict
from dataclasses import dataclass, field
@dataclass
class StatementsWithMedicalEntities:
    statements: List[Dict] = field(default_factory=list)
    def add_statement(self, statement: Dict) -> None:
        self.statements.append(statement)
    def get_all(self) -> List[Dict]:
        return self.statements
    def count(self) -> int:
        return len(self.statements)
    def get_entity_count(self) -> int:
        return sum(len(s.get("entities", [])) for s in self.statements)
    def __repr__(self) -> str:
        total_entities = self.get_entity_count()
        return f"StatementsWithMedicalEntities(statements={self.count()}, entities={total_entities})"
