"""
[Device] infer:entities

Stage D: entity linking only.
Maps NER entity mentions to canonical concepts (UMLS or rule-based).
Subject–predicate–object extraction is done by the LLM (Stage e).

Input: StatementsWithMedicalEntities (from Stage c)
Output: CandidateStatements (text + linked entities; all passed to Stage e for LLM)
"""

from pipeline.data import StatementsWithMedicalEntities, CandidateStatements
from pipeline.models import EntitiesLinker


class InferEntities:
    """
    Device for entity linking: map NER entities to canonical concepts.
    Does not filter or extract SPO; the LLM does that in Stage e.
    """

    def __init__(self, entities_linker: EntitiesLinker):
        """
        Args:
            entities_linker: Entity linking model (UMLS or rule-based)
        """
        self.entities_linker = entities_linker

    def infer(self, statements_with_entities: StatementsWithMedicalEntities) -> CandidateStatements:
        """
        Link entities in each statement to canonical concepts.
        Returns CandidateStatements with linked entities; all statements are passed to Stage e (LLM).
        """
        result = CandidateStatements()
        statements = statements_with_entities.get_all()

        for stmt in statements:
            text = stmt.get("text", "")
            original_text = stmt.get("original_text", text)
            raw_entities = stmt.get("entities", [])

            linked_entities = self.entities_linker.link_entities(
                entities=raw_entities,
                context_text=text,
            )

            result.add_statement({
                "chunk_id": stmt.get("chunk_id"),
                "page": stmt.get("page"),
                "source": stmt.get("source", ""),
                "text": text,
                "original_text": original_text,
                "entities": linked_entities,
            })

        return result
