"""
Stage I/O helpers: load from and write to versioned stage directories.
Used by pipeline.py and stage scripts for consistent directory-based flow.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from pipeline.data import (
    RawText,
    TextChunks,
    StatementsWithMedicalEntities,
    CandidateStatements,
    ValidatedFactsAndQualifiers,
    ShaclConstraints,
)


def load_stage_a_output(input_dir: Path) -> Optional[RawText]:
    """Load Stage A output (text.json + tables.json) from input_dir into RawText."""
    text_path = input_dir / "text.json"
    tables_path = input_dir / "tables.json"
    if not text_path.exists():
        return None
    with open(text_path, "r", encoding="utf-8") as f:
        texts = json.load(f)
    if not isinstance(texts, list):
        texts = texts.get("pages", texts) if isinstance(texts, dict) else []
    tables_list = []
    if tables_path.exists():
        with open(tables_path, "r", encoding="utf-8") as f:
            tables_list = json.load(f)
    if not isinstance(tables_list, list):
        tables_list = []

    tables_by_key = defaultdict(list)
    for t in tables_list:
        key = (t.get("source_file", ""), t.get("page", 0))
        tables_by_key[key].append({"title": t.get("caption", ""), "rows": t.get("rows", [])})

    raw_text = RawText()
    for item in texts:
        source_file = item.get("source_file", "")
        page_num = item.get("page", 0)
        text = item.get("text", "")
        key = (source_file, page_num)
        tables = tables_by_key.get(key, [])
        raw_text.add_page(
            page_num=page_num,
            text=text,
            source_file=source_file,
            tables=tables,
        )
    return raw_text


def load_stage_b_output(input_dir: Path) -> Tuple[TextChunks, List[Dict]]:
    """Load Stage B output (chunks + triples) from input_dir."""
    chunks_path = input_dir / "stage_b_text_chunks.json"
    triples_path = input_dir / "stage_b_table_triples.json"
    text_chunks = TextChunks()
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for chunk in data.get("chunks", []):
            text_chunks.add_chunk(
                page=chunk["page"],
                text=chunk["text"],
                source=chunk.get("source", ""),
                chunk_id=chunk.get("chunk_id"),
            )
    table_triples = []
    if triples_path.exists():
        with open(triples_path, "r", encoding="utf-8") as f:
            triples_data = json.load(f)
        table_triples = triples_data.get("triples", [])
        if not isinstance(table_triples, list):
            table_triples = []
    return text_chunks, table_triples


def load_stage_c_output(input_dir: Path) -> Tuple[StatementsWithMedicalEntities, List[Dict]]:
    """Load Stage C output (statements + table_triples) from input_dir."""
    file_path = input_dir / "stage_c_statements_with_entities.json"
    statements = StatementsWithMedicalEntities()
    table_triples = []
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for stmt in data.get("statements", []):
            statements.add_statement(stmt)
        table_triples = data.get("table_triples", [])
    return statements, table_triples


def load_stage_e_output(input_dir: Path) -> ValidatedFactsAndQualifiers:
    """Load Stage E output (validated factual statements) from input_dir."""
    file_path = input_dir / "stage_e_validated_output.json"
    facts = ValidatedFactsAndQualifiers()
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for stmt in data.get("validated_statements", []):
            facts.add_validated(stmt)
    return facts


def load_stage_f_output(input_dir: Path) -> ShaclConstraints:
    """Load Stage F output (SHACL constraints) from input_dir."""
    file_path = input_dir / "stage_f_shacl_constraints.json"
    constraints = ShaclConstraints()
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for c in data.get("constraints", []):
            constraints.add_constraint(c)
    return constraints


def load_stage_d_output(input_dir: Path) -> Tuple[CandidateStatements, List[Dict]]:
    """Load Stage D output (candidate statements + table_triples) from input_dir."""
    file_path = input_dir / "stage_d_candidate_statements.json"
    candidate_statements = CandidateStatements()
    table_triples = []
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for stmt in data.get("statements", []):
            candidate_statements.add_statement(stmt)
        table_triples = data.get("table_triples", [])
    return candidate_statements, table_triples
