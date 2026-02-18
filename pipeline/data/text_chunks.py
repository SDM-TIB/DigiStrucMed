from typing import List, Dict, Optional
from dataclasses import dataclass, field
@dataclass
class TextChunks:
    chunks: List[Dict] = field(default_factory=list)
    def add_chunk(
        self,
        page: int,
        text: str,
        source: str = "",
        chunk_id: Optional[int] = None,
        from_table: bool = False,
        table_index: Optional[int] = None,
        row_index: Optional[int] = None,
    ) -> None:
        if chunk_id is None:
            chunk_id = len(self.chunks)
        chunk = {
            "chunk_id": chunk_id,
            "page": page,
            "text": text,
            "source": source,
        }
        if from_table:
            chunk["from_table"] = True
            if table_index is not None:
                chunk["table_index"] = table_index
            if row_index is not None:
                chunk["row_index"] = row_index
        self.chunks.append(chunk)
    def get_chunks(self) -> List[Dict]:
        return self.chunks
    def count(self) -> int:
        return len(self.chunks)
    def __repr__(self) -> str:
        return f"TextChunks(count={self.count()})"
