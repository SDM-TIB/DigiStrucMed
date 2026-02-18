from pathlib import Path
from typing import List
class PDFGuidelines:
    def __init__(self, pdf_dir: str = "data"):
        self.pdf_dir = Path(pdf_dir)
        self.pdf_files: List[Path] = []
        if self.pdf_dir.exists():
            self.pdf_files = sorted(self.pdf_dir.glob("*.pdf"))
    def get_files(self) -> List[Path]:
        return self.pdf_files
    def count(self) -> int:
        return len(self.pdf_files)
    def __repr__(self) -> str:
        return f"PDFGuidelines(count={self.count()}, dir='{self.pdf_dir}')"
