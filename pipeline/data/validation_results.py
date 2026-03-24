from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class ValidationResults:
    """
    Results of Stage G cross-validation between Stage E factual statements
    and Stage F SHACL constraints.

    Each result has a status:
      VALID       - fact satisfies all applicable SHACL constraints
      VIOLATED    - fact violates one or more constraints (e.g. dosage out of range)
      UNVERIFIED  - no SHACL constraints found for the entities in this fact
    """
    results: List[Dict] = field(default_factory=list)

    def add_result(self, result: Dict) -> None:
        self.results.append(result)

    def get_all(self) -> List[Dict]:
        return self.results

    def count(self) -> int:
        return len(self.results)

    def count_valid(self) -> int:
        return sum(1 for r in self.results if r.get("status") == "VALID")

    def count_violated(self) -> int:
        return sum(1 for r in self.results if r.get("status") == "VIOLATED")

    def count_unverified(self) -> int:
        return sum(1 for r in self.results if r.get("status") == "UNVERIFIED")

    def get_violations(self) -> List[Dict]:
        return [r for r in self.results if r.get("status") == "VIOLATED"]

    def get_valid(self) -> List[Dict]:
        return [r for r in self.results if r.get("status") == "VALID"]

    def get_unverified(self) -> List[Dict]:
        return [r for r in self.results if r.get("status") == "UNVERIFIED"]

    def summary(self) -> Dict:
        return {
            "total": self.count(),
            "valid": self.count_valid(),
            "violated": self.count_violated(),
            "unverified": self.count_unverified(),
            "valid_pct": round(self.count_valid() / self.count() * 100, 1) if self.count() else 0,
            "violated_pct": round(self.count_violated() / self.count() * 100, 1) if self.count() else 0,
        }

    def __repr__(self) -> str:
        return (
            f"ValidationResults(total={self.count()}, "
            f"valid={self.count_valid()}, "
            f"violated={self.count_violated()}, "
            f"unverified={self.count_unverified()})"
        )
