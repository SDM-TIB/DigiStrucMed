"""
Pipeline

Coordinates the execution of the entire pipeline matching the diagram flow.

Pipeline flow (matching diagram):
a. PDFGuidelines → ExtractText → RawText
b. RawText → ChunkText (uses ParsingRules) → TextChunks
c. TextChunks → RecognizeEntities (uses NeuralModel + AcronymExpander) → StatementsWithMedicalEntities
d. StatementsWithMedicalEntities → InferEntities (uses EntitiesLinker) → CandidateStatements
e. CandidateStatements → Validate (uses ValidationModel/LLM) → ValidatedFactsAndQualifiers (OPTIONAL - can be skipped for manual validation)
f. ValidatedFactsAndQualifiers → generate → KG_AND_qualifiers (future)

Key: Acronym expansion is integrated into NER (Stage c).
Both NER and LLM work on the same expanded text for entity-SPO alignment.

Note: Stage e (LLM validation) can be skipped. Candidate statements from Stage d can be manually validated.
"""

from pathlib import Path
import json
from typing import Optional

# Data components
from pipeline.data import (
    PDFGuidelines,
    RawText,
    TextChunks,
    StatementsWithMedicalEntities,
    CandidateStatements,
    ValidatedFactsAndQualifiers
)

# Model components
from pipeline.models import (
    ParsingRules,
    NeuralModel,
    EntitiesLinker,
    ValidationModel
)

# Transform components
from pipeline.transforms import (
    ExtractText,
    ChunkText
)

# Device components
from pipeline.devices import (
    RecognizeEntities,
    InferEntities,
    Validate
)


class Pipeline:
    """
    Orchestrates the complete knowledge graph extraction pipeline.
    
    Pipeline flow (matching diagram):
    a. PDFGuidelines → ExtractText → RawText
    b. RawText → ChunkText (using ParsingRules) → TextChunks
    c. TextChunks → RecognizeEntities (using NeuralModel + AcronymExpander) → StatementsWithMedicalEntities
    d. StatementsWithMedicalEntities → InferEntities (using EntitiesLinker) → CandidateStatements
    e. CandidateStatements → Validate (using ValidationModel/LLM) → ValidatedFactsAndQualifiers (OPTIONAL)
    
    Key benefit: Acronym expansion is integrated into NER.
    Both NER and LLM work on the same expanded text.
    
    Note: Stage e (LLM validation) can be skipped by setting skip_llm_validation=True.
    In this case, candidate statements are output for manual validation.
    """
    
    def __init__(
        self,
        pdf_dir: str = "data",
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
        extracted_texts_file: Optional[str] = "extracted_texts.json",
        extracted_tables_file: Optional[str] = "extracted_tables.json",
    ):
        """
        Initialize pipeline.
        
        Args:
            pdf_dir: Directory containing PDF guidelines
            output_file: Output file for candidate statements or validated JSON
            neural_model_name: Neural model identifier (e.g. HuggingFace NER model)
            validation_model_name: Validation LLM identifier (only used if skip_llm_validation=False)
            confidence_threshold: Minimum confidence for validated statements
            min_chunk_chars: Minimum characters for valid chunk
            min_ner_score: Minimum NER confidence score
            max_validation_tokens: Max tokens for LLM extraction (400 allows multiple statements per chunk)
            batch_size: Batch size for LLM validation (4-8 for T4 GPU, higher = faster)
            skip_first_pages: Number of initial PDF pages to skip (metadata, cover)
            skip_last_pages: Number of final PDF pages to skip (references, appendices)
            umls_csv_path: Path to UMLS CSV file for entity linking (optional)
            filter_unmatched_entities: If True, remove entities without UMLS match
            acronym_file: Path to custom acronym JSON file (optional)
            skip_llm_validation: If True, skip Stage e (LLM validation) and output candidate statements for manual validation
            extracted_texts_file: If set, write extracted body text (no tables) to this JSON after Stage a
            extracted_tables_file: If set, write extracted tables (with captions) to this JSON after Stage a
        """
        self.pdf_dir = pdf_dir
        self.output_file = output_file
        self.extracted_texts_file = extracted_texts_file
        self.extracted_tables_file = extracted_tables_file
        self.confidence_threshold = confidence_threshold
        self.skip_llm_validation = skip_llm_validation
        
        # Initialize models (these are the dependencies)
        self.parsing_rules = ParsingRules()
        self.neural_model = NeuralModel(model_name=neural_model_name)
        
        # Initialize entity linker with optional UMLS support
        if umls_csv_path:
            self.entities_linker = EntitiesLinker(
                knowledge_base="umls",
                umls_csv_path=umls_csv_path,
                filter_unmatched=filter_unmatched_entities
            )
        else:
            self.entities_linker = EntitiesLinker(knowledge_base="rule-based")
        
        # Only initialize validation model if LLM validation is enabled
        if not skip_llm_validation:
            self.validation_model = ValidationModel(model_name=validation_model_name)
        else:
            self.validation_model = None
        
        # Initialize transforms
        self.extract_text = ExtractText(
            skip_first_pages=skip_first_pages,
            skip_last_pages=skip_last_pages
        )
        self.chunk_text = ChunkText(
            parsing_rules=self.parsing_rules,
            min_chars=min_chunk_chars
        )
        
        # Initialize devices
        # Stage c: NER with acronym expansion (NER_model_with_acronym_expander)
        self.recognize_entities = RecognizeEntities(
            neural_model=self.neural_model,
            min_score=min_ner_score,
            acronym_file=acronym_file  # Acronym expansion integrated into NER
        )
        # Stage d: Entity linking (EntitiesLinker)
        self.infer_entities = InferEntities(
            entities_linker=self.entities_linker
        )
        # Stage e: Extraction (ValidationModel/LLM) - only if enabled. Experts validate later.
        if not skip_llm_validation:
            self.validate = Validate(
                validation_model=self.validation_model,
                max_new_tokens=max_validation_tokens,
                batch_size=batch_size
            )
        else:
            self.validate = None
    
    def run(self):
        """
        Run the complete pipeline.
        
        Returns:
            If skip_llm_validation=True: CandidateStatements for manual validation
            If skip_llm_validation=False: ValidatedFactsAndQualifiers containing LLM-validated statements
        """
        # Stage a: Load PDF guidelines
        pdf_guidelines = PDFGuidelines(pdf_dir=self.pdf_dir)
        
        if pdf_guidelines.count() == 0:
            if self.skip_llm_validation:
                from pipeline.data import CandidateStatements
                return CandidateStatements()
            else:
                return ValidatedFactsAndQualifiers()
        
        # Stage a (continued): Extract text from PDFs
        raw_text = self.extract_text.transform(pdf_guidelines)

        # Write split extraction (texts only / tables only) for stages A–D
        if self.extracted_texts_file:
            texts_out = [
                {"source_file": p.get("source", ""), "page": p["page"], "text": p["text"]}
                for p in raw_text.get_pages()
            ]
            Path(self.extracted_texts_file).write_text(
                json.dumps(texts_out, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        if self.extracted_tables_file:
            tables_out = []
            for p in raw_text.get_pages():
                for t in p.get("tables", []):
                    tables_out.append({
                        "source_file": p.get("source", ""),
                        "page": p["page"],
                        "caption": t.get("title", ""),
                        "rows": t.get("rows", []),
                    })
            Path(self.extracted_tables_file).write_text(
                json.dumps(tables_out, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        
        # Stage b: Chunk text
        text_chunks = self.chunk_text.transform(raw_text)
        
        # Stage c: Recognize entities (NER with acronym expansion)
        statements_with_entities = self.recognize_entities.infer(text_chunks)
        
        # Stage d: Infer candidate statements (entity linking)
        candidate_statements = self.infer_entities.infer(statements_with_entities)
        
        # Stage e: Validate statements using LLM (OPTIONAL)
        if not self.skip_llm_validation:
            validated_facts = self.validate.validate(candidate_statements)
            
            # Save validated results as JSON
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
            # Skip LLM validation - output candidate statements for manual validation
            # Save candidate statements as JSON
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
