"""Result of Stage B content preparation: text chunks and table SPO triples."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

from .text_chunks import TextChunks


@dataclass
class TextChunksAndTableTriples:
    """Holds both text chunks and table-derived SPO triples from Stage B."""
    text_chunks: TextChunks
    table_triples: List[Dict[str, Any]]

    def get_text_chunks(self) -> TextChunks:
        return self.text_chunks

    def get_table_triples(self) -> List[Dict[str, Any]]:
        return self.table_triples
