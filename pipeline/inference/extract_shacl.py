"""
Stage F inference: extract SHACL constraints from Stage E validated facts.

For every factual statement that has medical entities with UMLS CUIs,
runs ShaclExtractor (LLaMA) to produce formal SHACL-style constraint objects.
Statements with no medical entities are skipped silently.
"""
from __future__ import annotations

from typing import List

from pipeline.data import ValidatedFactsAndQualifiers
from pipeline.data.shacl_constraints import ShaclConstraints
from pipeline.models.shacl_extractor import ShaclExtractor


class ExtractShacl:
    """
    Stage F: iterate over Stage E facts, extract SHACL constraints per statement.
    Uses ShaclExtractor which shares the already-loaded LLaMA (ValidationModel).
    """

    def __init__(self, shacl_extractor: ShaclExtractor):
        self.shacl_extractor = shacl_extractor

    def extract(self, validated_facts: ValidatedFactsAndQualifiers) -> ShaclConstraints:
        all_constraints = ShaclConstraints()
        statements = validated_facts.get_all()
        total = len(statements)
        constraint_count = 0
        skipped = 0

        for i, stmt in enumerate(statements):
            constraints: List = self.shacl_extractor.extract_constraints(stmt)
            if constraints:
                for c in constraints:
                    all_constraints.add_constraint(c)
                constraint_count += len(constraints)
            else:
                skipped += 1

            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(
                    f"[Stage F] {i + 1}/{total} statements processed — "
                    f"{constraint_count} constraints extracted, {skipped} skipped (no medical entities)"
                )

        return all_constraints
