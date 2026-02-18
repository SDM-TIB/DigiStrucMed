"""
[StatisticalModel] neural_model

General neural model for entity extraction (e.g. NER).
Swap model_name to use different HuggingFace or other backends later.
"""

from transformers import pipeline
from typing import List, Dict, Optional

# Default min score for entity extraction (override via extract_entities(min_score=...) or pipeline min_ner_score)
DEFAULT_MIN_SCORE = 0.55


class NeuralModel:
    """
    Neural model for entity extraction (e.g. token-classification / NER).
    Uses a HuggingFace model; change model_name to swap to another model later.
    """

    def __init__(self, model_name: str = "d4data/biomedical-ner-all"):
        """
        Initialize the neural model.

        Args:
            model_name: HuggingFace model identifier (or other backend later)
        """
        self.pipeline = pipeline(
            task="token-classification",
            model=model_name,
            aggregation_strategy="simple"
        )

    def extract_entities(
        self,
        text: str,
        min_score: float = DEFAULT_MIN_SCORE,
        max_entities: int = 15,
        keep_labels: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Extract entities from text (e.g. biomedical NER).

        Args:
            text: Input text
            min_score: Minimum confidence score
            max_entities: Maximum entities to return
            keep_labels: Optional list of label types to keep

        Returns:
            List of entities with text, label, score, start, end
        """
        if not text or not text.strip():
            return []

        try:
            entities = self.pipeline(text)
        except Exception:
            return []

        results = []
        seen = set()

        for ent in entities:
            score = float(ent.get("score", 0.0))
            if score < min_score:
                continue

            label = str(ent.get("entity_group") or ent.get("entity") or "").strip()
            ent_text = (ent.get("word") or "").strip()

            if len(ent_text) < 2:
                continue

            if keep_labels:
                if not any(k.lower() in label.lower() for k in keep_labels):
                    continue

            start = ent.get("start", None)
            end = ent.get("end", None)

            key = ent_text.lower()
            if key in seen:
                continue
            seen.add(key)

            results.append({
                "text": ent_text,
                "label": label,
                "score": round(score, 3),
                "start": start,
                "end": end
            })

            if len(results) >= max_entities:
                break

        # Sort by position for merging
        results.sort(key=lambda x: (x["start"] if x["start"] is not None else 0))

        # Merge fragmented entities
        results = self._merge_fragmented_entities(results)

        # Filter remaining fragments and artifacts (uses same min_score as extraction)
        results = self._filter_fragmented_entities(results, text, min_score)

        # Sort by confidence (best first)
        results.sort(key=lambda x: -x["score"])
        return results[:max_entities]

    def _merge_fragmented_entities(self, entities: List[Dict]) -> List[Dict]:
        """Merge fragmented entities from BERT tokenization."""
        if not entities:
            return []

        merged = []
        i = 0

        while i < len(entities):
            current = entities[i]
            ent_text = current["text"]

            # Check if this is a fragment (starts with ##)
            if ent_text.startswith("##"):
                if i == 0 or not merged:
                    i += 1
                    continue

                prev = merged[-1]
                if (prev["label"] == current["label"] and
                    prev["end"] is not None and current["start"] is not None and
                    abs(prev["end"] - current["start"]) <= 2):

                    merged_text = prev["text"] + ent_text[2:]
                    prev["text"] = merged_text
                    prev["end"] = current["end"]
                    prev["score"] = round((prev["score"] + current["score"]) / 2, 3)
                i += 1
            else:
                # Check for hyphenated fragments
                if merged and i > 0:
                    prev = merged[-1]

                    if (prev["label"] == current["label"] and
                        prev["end"] is not None and current["start"] is not None):

                        gap_size = current["start"] - prev["end"]

                        if 0 <= gap_size <= 3:
                            separator = " " if gap_size == 1 else " - "
                            merged_text = prev["text"] + separator + ent_text
                            prev["text"] = merged_text
                            prev["end"] = current["end"]
                            prev["score"] = round((prev["score"] + current["score"]) / 2, 3)
                            i += 1
                            continue

                merged.append(current.copy())
                i += 1

        return merged

    def _filter_fragmented_entities(
        self, entities: List[Dict], original_text: str, min_score: float
    ) -> List[Dict]:
        """
        Filter obvious fragments and junk; keep entities with score >= min_score.
        Uses the same min_score as extraction (from pipeline min_ner_score).
        """
        import re

        filtered = []
        for ent in entities:
            text = ent["text"].strip()
            score = ent.get("score", 0.0)

            if text.startswith("##") or len(text) == 0:
                continue
            if len(text) == 1:
                continue
            if len(text) == 2:
                continue

            # Skip page/section numbers (allow measurements like "5 mg")
            if re.match(r"^[\d\s\.\-]+", text) and re.search(r"\d", text):
                if not re.search(r"\b\d+\s*[a-zA-Z]{1,5}\b", text):
                    continue

            # Skip number-heavy strings (allow unit patterns)
            alpha_count = sum(1 for c in text if c.isalpha())
            if alpha_count < len(text) * 0.4:
                if not re.search(r"\d+\s*[a-zA-Z]{1,5}", text):
                    continue

            if score < min_score:
                continue

            filtered.append(ent)

        return filtered
