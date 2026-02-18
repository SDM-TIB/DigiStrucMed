# Analysis of Guideline Extraction System for Thesis

## Overview
This document provides a comprehensive analysis of the neuro-symbolic hybrid system developed for extracting structured guidelines from medical PDF documents.

## Architecture

### Neuro-Symbolic Hybrid Approach
The system combines:
1. **Symbolic (Rule-based) Filtering**: Pre-processing to remove non-guideline content
2. **Neural (BERT-based) NER**: Entity recognition using biomedical-ner-all model
3. **LLM (LLaMA) Validation**: Structured extraction and verification

### Pipeline Components

```
PDFs → Text Extraction → Symbolic Filter → Chunking → NER → Entity Linking → LLM Validation → Knowledge Graph
```

#### 1. **Models/** (Model Components)
- `parsing_rules.py`: Rule-based symbolic filters
- `ner_model.py`: BERT-based entity recognition
- `entities_linker.py`: Entity linking and normalization
- `validation_model.py`: LLaMA-based validation

#### 2. **Transforms/** (Data Transformations)
- `extract_text.py`: PDF text extraction (currently using PyMuPDF)
- `clean_segment.py`: Text cleaning and chunking
- `sdm_rdfizer.py`: KG conversion

#### 3. **Devices/** (Orchestration)
- `recognize_entities.py`: NER orchestration
- `infer_entities.py`: Entity linking orchestration
- `validate.py`: LLM validation orchestration

## Key Challenges & Solutions

### 1. Table Header Leakage
**Problem**: Table headers containing directive words (e.g., "Class of recommendation IIb") were being extracted as guidelines.

**Solution**: Enhanced `is_table_or_metadata()` with:
- Figure caption detection (e.g., "Figure 13 Diuretic therapy...")
- Symbol-based table detection (lines starting with ≥, ≤, <, >)
- Dosage table detection (multiple measurement ranges like "50–70 mEq/L")

### 2. Noisy NER Entities
**Problem**: BERT confidently misclassified non-medical text:
- Classification codes: "IIb", "IIa" → labeled as medications
- Truncated words: "fur" (from "furosemide") → labeled as medication
- Short fragments from table parsing errors

**Solution**: Generalizable, confidence-based filtering:
- Multi-factor scoring combining BERT confidence, entity length, context
- Medical context markers (co-occurrence with medical terms)
- Structure-based heuristics (capitalization patterns, numeric content)
- NO hardcoded patterns or whitelists

### 3. Generic LLM Extractions
**Problem**: LLM producing overly generic actions like "consider" or "may be considered" without specifics.

**Initial Approach**: Complex prompt engineering and heavy post-validation

**Revised Approach** (Clean Architecture):
- Moved filtering logic from LLM prompt to symbolic pre-filter
- Enhanced `symbolic_filter.py` to detect URLs, explanatory text
- Simplified LLM prompt to focus on extraction
- Streamlined post-validation

**Final Approach** (Very Permissive):
- Accepts ALL actions (including generic "consider")
- Expands fallback to extract from ANY directive/medical text
- Minimal requirements (only needs statement_type)
- Aims for high recall, accepts some noise

### 4. Chunking Issues
**Problem**: Chunks starting mid-sentence or containing multiple distinct recommendations.

**Solution**: Enhanced chunking logic:
- Split multiple recommendations (e.g., "ACE-I may be considered... ARB may be considered...")
- Detect incomplete sentences (no punctuation at end, lowercase at start)
- Merge small chunks (threshold increased to 80 characters)

### 5. Generic Exceptions/Durations
**Problem**: Placeholder values like "none", "not specified", "as needed".

**Solution**: Post-validation filtering in `devices/validate.py`:
- Removes generic placeholders
- Validates actual medical content
- Ensures extracted fields have meaningful information

## Performance Optimization

### Colab Workflow
For GPU-limited local environments:

1. **Local (CPU)**: `export_for_colab.py`
   - PDF text extraction
   - Symbolic filtering
   - Text chunking
   - NER (BERT)
   - Entity linking
   - Output: `candidates_for_colab.json`

2. **Colab (GPU)**: `Colab_Validation.ipynb`
   - Upload `candidates_for_colab.json` and `project_code.zip`
   - LLM validation (LLaMA 3.2 3B Instruct)
   - Output: `kg_ready.json`

### Memory Optimization
- LLaMA loads with `device_map="auto"` and `torch.float16` on GPU
- Reduces VRAM usage from ~12GB to ~6GB
- Prevents RAM exhaustion crashes

## Validation Strategy Evolution

### Version 1: Conservative
- Strict action filtering (removed generic "consider")
- Required multiple fields for validity
- LLM prompt emphasized "only extract if explicit"

**Result**: High precision, low recall (~10-15 statements)

### Version 2: Less Aggressive
- Kept generic actions if part of medication recommendations
- Accepted partial information
- More permissive LLM prompt

**Result**: Improved recall (~15-20 statements)

### Version 3: Very Permissive (Current)
- Accepts ALL actions
- Fallback creates extractions from ANY directive/medical text
- Minimal requirements (only statement_type)
- LLM prompt: "Be VERY inclusive"

**Result**: Expected high recall (40-60+ statements), some noise acceptable

## Generalizability

### Design Principles
1. **No hardcoded patterns** specific to ESC guidelines
2. **Universal medical terminology** detection
3. **Cross-format compatibility** (ESC, WHO, NICE, AHA)
4. **Confidence-based filtering** rather than whitelist/blacklist
5. **Context-aware validation** using surrounding text

### Examples of Generalizable Features
- Figure detection: Works for any "Figure N Title (description)" pattern
- Table detection: Symbol-based (≥, ≤) works across formats
- Chunking: Recommendation splitting works for any "Drug X may/should..." pattern
- NER filtering: Medical context detection works for any guideline domain

## Output Format

### `kg_ready.json` Structure
```json
[
  {
    "page": 11,
    "text": "ACE-I may be considered for patients with HFmrEF...",
    "chunk_id": 150,
    "source_pdf": "McDonagh, 2023, ESC guidelines.pdf",
    "structured": {
      "statement_type": "recommendation",
      "population": "patients with HFmrEF",
      "action": "receive ACE inhibitors",
      "duration": null,
      "exception": "contraindicated"
    },
    "ner_entities": [
      {
        "text": "ace-i",
        "label": "Medication",
        "score": 0.997
      }
    ],
    "confidence": 0.85
  }
]
```

## Future Enhancements

### 1. UMLS Integration
- Normalize medical entities to UMLS concepts
- Improve entity disambiguation
- Enable cross-guideline comparison

### 2. Advanced Confidence Scoring
- Combine NER scores, LLM uncertainty, context relevance
- Multi-factor validation
- Threshold-based filtering

### 3. Relationship Extraction
- Extract relations between entities (treats, causes, prevents)
- Build richer knowledge graph
- Enable reasoning and inference

### 4. Multi-guideline Comparison
- Detect conflicts between guidelines
- Track changes over time
- Generate guideline comparison reports

### 5. Interactive Validation
- Web UI for human review
- Active learning loop
- Expert feedback integration

## Thesis Contributions

1. **Neuro-symbolic hybrid architecture** for medical guideline extraction
2. **Generalizable filtering approach** without domain-specific hardcoding
3. **Colab workflow** for GPU-limited environments
4. **Confidence-based entity filtering** for BERT NER
5. **Clean architecture** separating models, transforms, and orchestration

## Key Metrics

- **Precision**: Accuracy of extracted guidelines (true positives / total extracted)
- **Recall**: Coverage of actual guidelines (true positives / total actual)
- **F1 Score**: Harmonic mean of precision and recall
- **Entity Quality**: Ratio of valid medical entities to total NER entities
- **Extraction Completeness**: Percentage of statements with all fields populated

## Conclusion

This system demonstrates a practical neuro-symbolic approach to structured information extraction from medical PDFs. By combining rule-based filters, neural entity recognition, and LLM validation, it achieves generalizable extraction across guideline formats while maintaining quality through confidence-based filtering and validation.

The modular architecture enables easy extension and modification, while the Colab workflow addresses practical deployment constraints. Future work should focus on UMLS integration, advanced confidence scoring, and relationship extraction to build richer knowledge representations.
