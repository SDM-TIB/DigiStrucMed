from pipeline.data import StatementsWithMedicalEntities, CandidateStatements
from pipeline.models import EntitiesLinker
class InferEntities:
    def __init__(self, entities_linker: EntitiesLinker):
        self.entities_linker = entities_linker
    def infer(self, statements_with_entities: StatementsWithMedicalEntities) -> CandidateStatements:
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
