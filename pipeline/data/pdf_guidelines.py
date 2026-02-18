"""
[Data] PDF_guidelines

Input PDFs (guideline documents).
Provides file paths for extraction.
"""

from pathlib import Path
from typing import List


class PDFGuidelines:
    """Container for PDF guideline file paths."""
    
    def __init__(self, pdf_dir: str = "data"):
        """
        Initialize with directory containing PDF guidelines.
        
        Args:
            pdf_dir: Directory path containing PDF files
        """
        self.pdf_dir = Path(pdf_dir)
        self.pdf_files: List[Path] = []
        
        if self.pdf_dir.exists():
            self.pdf_files = sorted(self.pdf_dir.glob("*.pdf"))
    
    def get_files(self) -> List[Path]:
        """Get list of PDF file paths."""
        return self.pdf_files
    
    def count(self) -> int:
        """Get number of PDF files."""
        return len(self.pdf_files)
    
    def __repr__(self) -> str:
        return f"PDFGuidelines(count={self.count()}, dir='{self.pdf_dir}')"
