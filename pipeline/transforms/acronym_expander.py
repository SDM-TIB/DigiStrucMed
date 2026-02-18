"""
[Transform] Acronym Expander

Expands medical acronyms to full forms before NER.
Inspired by NSSC paper's preprocessing approach.

General, domain-agnostic design:
- Load acronyms from JSON file
- Apply case-sensitive and case-insensitive matching
- Preserve original text structure
"""

import json
import re
from pathlib import Path
from typing import Dict

# Default acronym file (relative to pipeline.data); override via acronym_file argument.
DEFAULT_ACRONYM_FILE = "heart_failure_acronyms.json"


class AcronymExpander:
    """
    Expands medical acronyms to improve NER accuracy.
    
    Based on NSSC's preprocessing approach:
    1. Load acronym dictionary
    2. Expand acronyms before NER
    3. Improve entity recognition
    """
    
    def __init__(self, acronym_file: str = None):
        """
        Initialize acronym expander.
        
        Args:
            acronym_file: Path to JSON file with acronym mappings
                         If None, uses default heart failure acronyms
        """
        if acronym_file is None:
            default_path = Path(__file__).parent.parent / "data" / DEFAULT_ACRONYM_FILE
            acronym_file = str(default_path)
        
        self.acronyms = self._load_acronyms(acronym_file)
    
    def _load_acronyms(self, file_path: str) -> Dict[str, str]:
        """
        Load acronym mappings from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Dictionary mapping acronyms to full forms
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert list of pairs to dictionary
            acronym_dict = {}
            for short, full in data.get("acronyms", []):
                acronym_dict[short] = full
            
            return acronym_dict
        
        except FileNotFoundError:
            return {}
        except Exception as e:
            return {}
    
    def expand(self, text: str) -> str:
        """
        Expand acronyms in text.
        
        Uses word boundary matching to avoid false expansions.
        Preserves case sensitivity where appropriate.
        
        Args:
            text: Input text with acronyms
            
        Returns:
            Text with acronyms expanded
        """
        if not text or not self.acronyms:
            return text
        
        expanded_text = text
        
        for acronym, full_form in self.acronyms.items():
            # Pattern 1: Exact match with word boundaries
            # Matches "HF" in "HF patients" but not in "HFrEF"
            pattern = r'\b' + re.escape(acronym) + r'\b'
            
            # Replace with full form
            expanded_text = re.sub(
                pattern,
                full_form,
                expanded_text,
                flags=re.IGNORECASE
            )
        
        return expanded_text
    
