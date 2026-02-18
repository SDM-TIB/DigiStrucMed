import json
import re
from pathlib import Path
from typing import Dict
DEFAULT_ACRONYM_FILE = "heart_failure_acronyms.json"
class AcronymExpander:
    def __init__(self, acronym_file: str = None):
        if acronym_file is None:
            default_path = Path(__file__).parent.parent / "data" / DEFAULT_ACRONYM_FILE
            acronym_file = str(default_path)
        self.acronyms = self._load_acronyms(acronym_file)
    def _load_acronyms(self, file_path: str) -> Dict[str, str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            acronym_dict = {}
            for short, full in data.get("acronyms", []):
                acronym_dict[short] = full
            return acronym_dict
        except FileNotFoundError:
            return {}
        except Exception as e:
            return {}
    def expand(self, text: str) -> str:
        if not text or not self.acronyms:
            return text
        expanded_text = text
        for acronym, full_form in self.acronyms.items():
            pattern = r'\b' + re.escape(acronym) + r'\b'
            expanded_text = re.sub(
                pattern,
                full_form,
                expanded_text,
                flags=re.IGNORECASE
            )
        return expanded_text
