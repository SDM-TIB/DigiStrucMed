from pathlib import Path
import json
from typing import Optional
from pipeline.data import (
    PDFGuidelines,
    RawText,
    TextChunks,
    StatementsWithMedicalEntities,
    CandidateStatements,
    ValidatedFactsAndQualifiers
)
from pipeline.models import (
    ParsingRules,
    NeuralModel,
    EntitiesLinker,
    ValidationModel
)
from pipeline.transforms import (
    ExtractText,
    ExtractTextV2,
    ContentPreparation,
)
from pipeline.inference import (
    RecognizeEntities,
    InferEntities,
    Validate
)
class Pipeline:
    def __init__(
        self,
        pdf_dir: str = "input",
        output_file: str = "candidate_statements.json",
        neural_model_name: str = "d4data/biomedical-ner-all",
        validation_model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        confidence_threshold: float = 0.0,
        min_chunk_chars: int = 40,
        min_ner_score: float = 0.55,
        max_validation_tokens: int = 400,
        batch_size: int = 4,
        skip_first_pages: int = 3,
        skip_last_pages: int = 5,
        umls_csv_path: str = None,
        filter_unmatched_entities: bool = False,
        acronym_file: str = None,
        skip_llm_validation: bool = True,
        output_dir: str = "outputs",
        stage_a_version: str = "v1",
    ):
        self.pdf_dir = pdf_dir
        self.output_file = output_file
        self.output_dir = output_dir
        self.stage_a_version = stage_a_version
        self.confidence_threshold = confidence_threshold
        self.skip_llm_validation = skip_llm_validation
        self.parsing_rules = ParsingRules()
        self.neural_model = NeuralModel(model_name=neural_model_name)
        if umls_csv_path:
            self.entities_linker = EntitiesLinker(
                knowledge_base="umls",
                umls_csv_path=umls_csv_path,
                filter_unmatched=filter_unmatched_entities
            )
        else:
            self.entities_linker = EntitiesLinker(knowledge_base="rule-based")
        if not skip_llm_validation:
            self.validation_model = ValidationModel(model_name=validation_model_name)
        else:
            self.validation_model = None
        stage_a_output_dir = f"{output_dir}/STAGE_A_{stage_a_version}"
        if stage_a_version == "v2":
            self.extract_text = ExtractTextV2(
                skip_first_pages=skip_first_pages,
                skip_last_pages=skip_last_pages,
                stage_output_dir=stage_a_output_dir,
            )
        else:
            self.extract_text = ExtractText(
                skip_first_pages=skip_first_pages,
                skip_last_pages=skip_last_pages,
                stage_output_dir=stage_a_output_dir,
            )
        self.content_preparation = ContentPreparation(
            parsing_rules=self.parsing_rules,
            min_chars=min_chunk_chars,
            stage_output_dir=str(Path(self.output_dir) / "STAGE_B_v1"),
        )
        self.recognize_entities = RecognizeEntities(
            neural_model=self.neural_model,
            min_score=min_ner_score,
            acronym_file=acronym_file
        )
        self.infer_entities = InferEntities(
            entities_linker=self.entities_linker
        )
        if not skip_llm_validation:
            self.validate = Validate(
                validation_model=self.validation_model,
                max_new_tokens=max_validation_tokens,
                batch_size=batch_size
            )
        else:
            self.validate = None
    def run(self):
        pdf_guidelines = PDFGuidelines(pdf_dir=self.pdf_dir)
        if pdf_guidelines.count() == 0:
            if self.skip_llm_validation:
                from pipeline.data import CandidateStatements
                return CandidateStatements()
            else:
                return ValidatedFactsAndQualifiers()
        raw_text = self.extract_text.transform(pdf_guidelines)
        content_result = self.content_preparation.transform(raw_text)
        text_chunks = content_result.get_text_chunks()
        table_triples = content_result.get_table_triples()
        statements_with_entities = self.recognize_entities.infer(text_chunks)
        table_triples = self.recognize_entities.enrich_triples_with_entities(table_triples)
        # Write Stage C output (statements + table_triples) to outputs/STAGE_C_v1
        stage_c_dir = Path(self.output_dir) / "STAGE_C_v1"
        stage_c_dir.mkdir(parents=True, exist_ok=True)
        stage_c_file = stage_c_dir / "stage_c_statements_with_entities.json"
        stage_c_data = {
            "metadata": {
                "stage": "c",
                "description": "Statements with medical entities (NER) and table triples from Stage B",
                "total_statements": statements_with_entities.count(),
                "total_entities": statements_with_entities.get_entity_count(),
                "total_table_triples": len(table_triples),
            },
            "statements": statements_with_entities.get_all(),
            "table_triples": table_triples,
        }
        stage_c_file.write_text(
            json.dumps(stage_c_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        candidate_statements = self.infer_entities.infer(statements_with_entities)
        if not self.skip_llm_validation:
            validated_facts = self.validate.validate(candidate_statements)
            output_data = {
                "metadata": {
                    "total_statements": validated_facts.count(),
                    "confidence_threshold": self.confidence_threshold,
                    "validation_method": "LLM"
                },
                "validated_statements": validated_facts.get_all()
            }
            Path(self.output_file).write_text(
                json.dumps(output_data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            return validated_facts
        else:
            output_data = {
                "metadata": {
                    "total_statements": candidate_statements.count(),
                    "total_candidates": candidate_statements.count_candidates(),
                    "validation_method": "manual"
                },
                "candidate_statements": candidate_statements.get_all()
            }
            Path(self.output_file).write_text(
                json.dumps(output_data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            return candidate_statements
