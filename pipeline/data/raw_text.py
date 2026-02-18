from typing import Dict, List, Optional
from dataclasses import dataclass, field
@dataclass
class RawText:
    pages: List[Dict] = field(default_factory=list)
    def add_page(
        self,
        page_num: int,
        text: str,
        source_file: str = "",
        tables: Optional[List[Dict]] = None,
    ) -> None:
        self.pages.append({
            "page": page_num,
            "text": text,
            "source": source_file,
            "tables": tables if tables is not None else [],
        })
    def get_pages(self) -> List[Dict]:
        return self.pages
    def count(self) -> int:
        return len(self.pages)
    def __repr__(self) -> str:
        return f"RawText(pages={self.count()})"
