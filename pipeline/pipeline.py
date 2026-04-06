"""
Pipeline: directory-based flow for all stages A–E.
Each stage reads from the previous stage's output directory and writes to its own.
"""
from pathlib import Path
import json
from typing import Optional

import config
from pipeline.data import (
    PDFGuidelines,
    RawText,
    TextChunks,
    StatementsWithMedicalEntities,
    CandidateStatements,
    ValidatedFactsAndQualifiers,
)
from pipeline.models import (
    ParsingRules,
    NeuralModel,
    EntitiesLinker,
    BiEncoderLinker,
    ValidationModel,
)
from pipeline.transforms import ExtractText, ExtractTextV2, ContentPreparation
from pipeline.inference import RecognizeEntities, InferEntities, Validate
from pipeline.stage_io import (
    load_stage_a_output,
    load_stage_b_output,
    load_stage_c_output,
    load_stage_d_output,
)


class Pipeline:
    def __init__(
        self,
        pdf_dir: Optional[str] = None,
        output_file: str = "candidate_statements.json",
        neural_model_name: str = "d4data/biomedical-ner-all",
        validation_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        confidence_threshold: float = 0.0,
        min_chunk_chars: int = 40,
        min_ner_score: float = 0.55,
        max_validation_tokens: int = 400,
        batch_size: int = 4,
        skip_first_pages: int = 3,
        skip_last_pages: int = 5,
        umls_csv_path: Optional[str] = None,
        filter_unmatched_entities: bool = False,
        linker_use_partial_umls_match: bool = False,
        linker_max_candidates_per_entity: int = 8,
        linker_max_anchor_bucket_hits: int = 250,
        linker_max_candidate_pool_size: int = 300,
        linker_enable_type_constraints: bool = True,
        acronym_file: Optional[str] = None,
        skip_llm_validation: bool = True,
        output_dir: str = "outputs",
        stage_a_version: Optional[str] = None,
        stage_b_version: Optional[str] = None,
        stage_c_version: Optional[str] = None,
        stage_d_version: Optional[str] = None,
        stage_e_version: Optional[str] = None,
        biencoder_index_dir: Optional[str] = None,
        biencoder_model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        biencoder_top_k: int = 16,
        biencoder_min_link_score: Optional[float] = 0.72,
        biencoder_use_type_rerank: bool = True,
        biencoder_prefer_shorter_concept: bool = True,
        biencoder_disambiguator: Optional[object] = None,
    ):
        self.pdf_dir = pdf_dir or str(config.PDF_DIR)
        self.output_file = output_file
        self.output_dir = output_dir
        self.stage_a_version = stage_a_version or config.DEFAULT_STAGE_A_VERSION
        self.stage_b_version = stage_b_version or config.DEFAULT_STAGE_B_VERSION
        self.stage_c_version = stage_c_version or config.DEFAULT_STAGE_C_VERSION
        self.stage_d_version = stage_d_version or config.DEFAULT_STAGE_D_VERSION
        self.stage_e_version = stage_e_version or config.DEFAULT_STAGE_E_VERSION
        self.confidence_threshold = confidence_threshold
        self.skip_llm_validation = skip_llm_validation
        self.parsing_rules = ParsingRules()
        self.neural_model = NeuralModel(model_name=neural_model_name)
        if self.stage_d_version == "v2" and biencoder_index_dir:
            self.entities_linker = BiEncoderLinker(
                model_name=biencoder_model_name,
                index_dir=biencoder_index_dir,
                top_k=biencoder_top_k,
                min_link_score=biencoder_min_link_score,
                use_type_rerank=biencoder_use_type_rerank,
                prefer_shorter_concept=biencoder_prefer_shorter_concept,
                disambiguator=biencoder_disambiguator,
            )
        elif umls_csv_path:
            self.entities_linker = EntitiesLinker(
                knowledge_base="umls",
                umls_csv_path=umls_csv_path,
                filter_unmatched=filter_unmatched_entities,
                use_partial_umls_match=linker_use_partial_umls_match,
                max_candidates_per_entity=linker_max_candidates_per_entity,
                max_anchor_bucket_hits=linker_max_anchor_bucket_hits,
                max_candidate_pool_size=linker_max_candidate_pool_size,
                enable_type_constraints=linker_enable_type_constraints,
            )
        else:
            self.entities_linker = EntitiesLinker(knowledge_base="rule-based")
        if not skip_llm_validation:
            self.validation_model = ValidationModel(model_name=validation_model_name)
        else:
            self.validation_model = None

        stage_a_dir = config.stage_a_dir(self.stage_a_version)
        stage_b_dir = config.stage_b_dir(self.stage_b_version)
        if self.stage_a_version == "v2":
            self.extract_text = ExtractTextV2(
                skip_first_pages=skip_first_pages,
                skip_last_pages=skip_last_pages,
                stage_output_dir=str(stage_a_dir),
            )
        else:
            self.extract_text = ExtractText(
                skip_first_pages=skip_first_pages,
                skip_last_pages=skip_last_pages,
                stage_output_dir=str(stage_a_dir),
            )
        self.content_preparation = ContentPreparation(
            parsing_rules=self.parsing_rules,
            min_chars=min_chunk_chars,
            stage_output_dir=str(stage_b_dir),
            stage_b_version=self.stage_b_version,
        )
        self.recognize_entities = RecognizeEntities(
            neural_model=self.neural_model,
            min_score=min_ner_score,
            acronym_file=acronym_file,
        )
        self.infer_entities = InferEntities(entities_linker=self.entities_linker)
        if not skip_llm_validation:
            self.validate = Validate(
                validation_model=self.validation_model,
                max_new_tokens=max_validation_tokens,
                batch_size=batch_size,
            )
        else:
            self.validate = None

    def run(self):
        # Stage A: PDFs -> STAGE_A
        pdf_guidelines = PDFGuidelines(pdf_dir=self.pdf_dir)
        if pdf_guidelines.count() == 0:
            if self.skip_llm_validation:
                return CandidateStatements()
            return ValidatedFactsAndQualifiers()
        self.extract_text.transform(pdf_guidelines)

        # Stage B: load STAGE_A -> transform -> STAGE_B
        stage_a_dir = config.stage_a_dir(self.stage_a_version)
        raw_text = load_stage_a_output(stage_a_dir)
        if raw_text is None:
            if self.skip_llm_validation:
                return CandidateStatements()
            return ValidatedFactsAndQualifiers()
        self.content_preparation.transform(raw_text)

        # Stage C: load STAGE_B -> NER -> STAGE_C
        stage_b_dir = config.stage_b_dir(self.stage_b_version)
        text_chunks, table_triples = load_stage_b_output(stage_b_dir)
        if text_chunks.count() == 0:
            if self.skip_llm_validation:
                return CandidateStatements()
            return ValidatedFactsAndQualifiers()
        statements_with_entities = self.recognize_entities.infer(text_chunks)
        table_triples = self.recognize_entities.enrich_triples_with_entities(table_triples)
        stage_c_dir = config.stage_c_dir(self.stage_c_version)
        stage_c_dir.mkdir(parents=True, exist_ok=True)
        stage_c_file = stage_c_dir / "stage_c_statements_with_entities.json"
        stage_c_file.write_text(
            json.dumps(
                {
                    "metadata": {
                        "stage": "c",
                        "description": "Statements with medical entities (NER) and table triples from Stage B",
                        "total_statements": statements_with_entities.count(),
                        "total_entities": statements_with_entities.get_entity_count(),
                        "total_table_triples": len(table_triples),
                    },
                    "statements": statements_with_entities.get_all(),
                    "table_triples": table_triples,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # Stage D: load STAGE_C -> infer -> STAGE_D
        stage_c_dir = config.stage_c_dir(self.stage_c_version)
        statements_with_entities, table_triples_raw = load_stage_c_output(stage_c_dir)
        candidate_statements = self.infer_entities.infer(statements_with_entities)
        table_triples_enriched = []
        for triple in table_triples_raw:
            out = dict(triple)
            entities = triple.get("entities", [])
            if entities:
                triple_context = " ".join(
                    part.strip()
                    for part in (
                        triple.get("subject", ""),
                        triple.get("predicate", ""),
                        triple.get("object", ""),
                    )
                    if isinstance(part, str) and part.strip()
                )
                linker_input = [
                    {"text": e.get("text", ""), "label": e.get("label", "")}
                    for e in entities
                ]
                for i, e in enumerate(entities):
                    if i < len(linker_input):
                        if "score" in e:
                            linker_input[i]["score"] = e["score"]
                        if "start" in e:
                            linker_input[i]["start"] = e["start"]
                        if "end" in e:
                            linker_input[i]["end"] = e["end"]
                linked = self.entities_linker.link_entities(
                    linker_input,
                    context_text=triple_context,
                )
                out["entities"] = linked
            table_triples_enriched.append(out)
        stage_d_dir = config.stage_d_dir(self.stage_d_version)
        stage_d_dir.mkdir(parents=True, exist_ok=True)
        stage_d_file = stage_d_dir / "stage_d_candidate_statements.json"
        stage_d_file.write_text(
            json.dumps(
                {
                    "metadata": {
                        "stage": "d",
                        "description": "Candidate statements and table triples with UMLS-linked entities",
                        "total_statements": candidate_statements.count(),
                        "total_candidates": candidate_statements.count_candidates(),
                        "total_table_triples": len(table_triples_enriched),
                        "umls_linking": self.entities_linker.knowledge_base == "umls",
                    },
                    "statements": candidate_statements.get_all(),
                    "table_triples": table_triples_enriched,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        if self.skip_llm_validation:
            Path(self.output_file).write_text(
                json.dumps(
                    {
                        "metadata": {
                            "total_statements": candidate_statements.count(),
                            "total_candidates": candidate_statements.count_candidates(),
                            "validation_method": "manual",
                        },
                        "candidate_statements": candidate_statements.get_all(),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return candidate_statements

        # Stage E: load STAGE_D -> validate -> STAGE_E
        stage_d_dir = config.stage_d_dir(self.stage_d_version)
        candidate_statements, table_triples = load_stage_d_output(stage_d_dir)
        validated_facts = self.validate.validate(candidate_statements)
        if table_triples:
            self.validate.validate_table_triples(validated_facts, table_triples)
        stage_e_dir = config.stage_e_dir(self.stage_e_version)
        stage_e_dir.mkdir(parents=True, exist_ok=True)
        stage_e_file = stage_e_dir / "stage_e_validated_output.json"
        stage_e_file.write_text(
            json.dumps(
                {
                    "metadata": {
                        "stage": "e",
                        "description": "Extracted factual statements for expert validation",
                        "total_statements": validated_facts.count(),
                        "extraction_model": "LLM",
                        "table_triples_through_llm": len(table_triples),
                    },
                    "validated_statements": validated_facts.get_all(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        Path(self.output_file).write_text(
            json.dumps(
                {
                    "metadata": {
                        "total_statements": validated_facts.count(),
                        "confidence_threshold": self.confidence_threshold,
                        "validation_method": "LLM",
                    },
                    "validated_statements": validated_facts.get_all(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return validated_facts
