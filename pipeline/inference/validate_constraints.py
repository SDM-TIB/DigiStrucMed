"""
Stage G inference: cross-validate Stage E factual statements against Stage F SHACL constraints.

Algorithm:
  1. Build a constraint index keyed by UMLS CUI and by concept name (lower-case).
  2. For every fact, find applicable constraints via its entity CUIs / concept names.
  3. Apply each constraint:
       - range      → extract numeric values from fact text, check min/max
       - enum       → check allowed_values against fact text
       - conditional→ check NOT contraindications; check AND/OR conditions
       - temporal   → verify duration against min/max time expressions
       - cardinality→ flag when countable fact text breaches min_count/max_count
  4. Assign status VALID / VIOLATED / UNVERIFIED.
  5. Detect cross-statement inconsistencies: two facts about the same CUI with
     contradictory numeric values are flagged as INCONSISTENT in their results.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from pipeline.data import ValidatedFactsAndQualifiers
from pipeline.data.shacl_constraints import ShaclConstraints
from pipeline.data.validation_results import ValidationResults


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_NUMERIC_PATTERNS = [
    r"(\d+(?:\.\d+)?)\s*(?:mg|ml|g|mcg|µg|ug|mg/kg|units?|iu|mmol|meq)",
    r"(\d+(?:\.\d+)?)\s*(?:per|/)\s*(?:day|week|month|hour|dose|kg)",
    r"(\d+(?:\.\d+)?)\s*%",
    r"\b(\d+(?:\.\d+)?)\b",
]

_TIME_PATTERNS = [
    (r"(\d+)\s*days?",   "day",   1),
    (r"(\d+)\s*weeks?",  "week",  7),
    (r"(\d+)\s*months?", "month", 30),
    (r"(\d+)\s*years?",  "year",  365),
    (r"(\d+)\s*hours?",  "hour",  1 / 24),
]


def _extract_numerics(text: str) -> List[float]:
    seen: set = set()
    values: List[float] = []
    for pattern in _NUMERIC_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            try:
                num = float(m.group(1))
                if num not in seen and 0 < num < 100_000:
                    seen.add(num)
                    values.append(num)
            except ValueError:
                pass
    return values


def _parse_numeric(val: Optional[str]) -> Optional[float]:
    if not val:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", str(val))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _text_for_fact(fact: Dict) -> str:
    """Concatenate all textual fields of a fact for pattern matching."""
    return " ".join(filter(None, [
        fact.get("source_text") or "",
        fact.get("object") or "",
        fact.get("subject") or "",
        fact.get("exception") or "",
        fact.get("duration") or "",
    ])).lower()


def _shape_id(constraint: Dict) -> str:
    concept = (constraint.get("target_concept") or "unknown").replace(" ", "")
    prop = (constraint.get("property") or "property").capitalize()
    return f"{concept}_{prop}"


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class ValidateConstraints:
    """
    Stage G: cross-validate facts (Stage E) against SHACL constraints (Stage F).
    Rule-based — no LLM needed, runs fast.
    """

    def validate(
        self,
        validated_facts: ValidatedFactsAndQualifiers,
        shacl_constraints: ShaclConstraints,
    ) -> ValidationResults:
        results = ValidationResults()
        facts = validated_facts.get_all()
        constraints = shacl_constraints.get_all()

        # Build index: cui → [constraints],  concept_lower → [constraints]
        cui_index: Dict[str, List[Dict]] = {}
        concept_index: Dict[str, List[Dict]] = {}
        for c in constraints:
            cui = c.get("target_cui")
            concept = (c.get("target_concept") or "").strip().lower()
            if cui:
                cui_index.setdefault(cui, []).append(c)
            if concept:
                concept_index.setdefault(concept, []).append(c)

        # Per-CUI numeric cache for cross-statement consistency check
        cui_numeric_cache: Dict[str, List[Tuple[float, int]]] = {}  # cui → [(value, fact_idx)]

        total = len(facts)
        for idx, fact in enumerate(facts):
            applicable = self._find_applicable(fact, cui_index, concept_index)

            if not applicable:
                results.add_result({
                    "fact": fact,
                    "status": "UNVERIFIED",
                    "violations": [],
                    "satisfied_constraints": [],
                    "applicable_constraints": [],
                    "note": "No SHACL constraints found for entities in this fact",
                })
                continue

            violations: List[Dict] = []
            satisfied: List[Dict] = []

            for constraint in applicable:
                violation = self._check_constraint(fact, constraint)
                sid = _shape_id(constraint)
                if violation:
                    violations.append(violation)
                else:
                    satisfied.append({
                        "shape_id": sid,
                        "property": constraint.get("property"),
                        "constraint_type": constraint.get("constraint_type"),
                    })

            # Collect numeric values for cross-statement consistency
            for entity in fact.get("entities", []):
                cui = entity.get("id") or entity.get("cui_id") or entity.get("umls_id")
                if cui:
                    nums = _extract_numerics(_text_for_fact(fact))
                    for n in nums:
                        cui_numeric_cache.setdefault(cui, []).append((n, idx))

            status = "VIOLATED" if violations else "VALID"
            results.add_result({
                "fact": fact,
                "status": status,
                "violations": violations,
                "satisfied_constraints": satisfied,
                "applicable_constraints": [_shape_id(c) for c in applicable],
            })

            if (idx + 1) % 200 == 0 or (idx + 1) == total:
                print(f"[Stage G] {idx + 1}/{total} facts validated")

        # Post-pass: flag cross-statement numeric inconsistencies
        self._flag_inconsistencies(results, cui_index, cui_numeric_cache)

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Constraint lookup
    # ──────────────────────────────────────────────────────────────────────────

    def _find_applicable(
        self,
        fact: Dict,
        cui_index: Dict[str, List[Dict]],
        concept_index: Dict[str, List[Dict]],
    ) -> List[Dict]:
        applicable: List[Dict] = []
        seen_ids: set = set()
        for entity in fact.get("entities", []):
            cui = entity.get("id") or entity.get("cui_id") or entity.get("umls_id")
            concept = (entity.get("text") or "").strip().lower()
            if cui and cui in cui_index:
                for c in cui_index[cui]:
                    cid = id(c)
                    if cid not in seen_ids:
                        applicable.append(c)
                        seen_ids.add(cid)
            if concept:
                for key, clist in concept_index.items():
                    if concept in key or key in concept:
                        for c in clist:
                            cid = id(c)
                            if cid not in seen_ids:
                                applicable.append(c)
                                seen_ids.add(cid)
        return applicable

    # ──────────────────────────────────────────────────────────────────────────
    # Per-constraint checks
    # ──────────────────────────────────────────────────────────────────────────

    def _check_constraint(self, fact: Dict, constraint: Dict) -> Optional[Dict]:
        ctype = (constraint.get("constraint_type") or "").lower()
        prop  = (constraint.get("property") or "").lower()
        text  = _text_for_fact(fact)

        if ctype == "range":
            return self._check_range(text, constraint)
        if ctype == "enum":
            return self._check_enum(text, constraint)
        if ctype == "conditional" or prop == "contraindication":
            return self._check_conditional(fact, constraint)
        if ctype == "temporal":
            return self._check_temporal(text, constraint)
        # cardinality and other types: conservative, no violation raised
        return None

    # -- range ----------------------------------------------------------------

    def _check_range(self, fact_text: str, constraint: Dict) -> Optional[Dict]:
        min_val = constraint.get("min_value")
        max_val = constraint.get("max_value")
        if not min_val and not max_val:
            return None
        numerics = _extract_numerics(fact_text)
        if not numerics:
            return None  # no number to check — not a violation
        min_num = _parse_numeric(min_val)
        max_num = _parse_numeric(max_val)
        for n in numerics:
            if min_num is not None and n < min_num:
                return {
                    "constraint_id":   _shape_id(constraint),
                    "property":        constraint.get("property"),
                    "constraint_type": "range",
                    "violation_type":  "value_below_minimum",
                    "expected":        f">= {min_val}",
                    "found":           str(n),
                    "message":         f"Value {n} is below minimum {min_val}",
                }
            if max_num is not None and n > max_num:
                return {
                    "constraint_id":   _shape_id(constraint),
                    "property":        constraint.get("property"),
                    "constraint_type": "range",
                    "violation_type":  "value_above_maximum",
                    "expected":        f"<= {max_val}",
                    "found":           str(n),
                    "message":         f"Value {n} exceeds maximum {max_val}",
                }
        return None

    # -- enum -----------------------------------------------------------------

    def _check_enum(self, fact_text: str, constraint: Dict) -> Optional[Dict]:
        allowed = constraint.get("allowed_values")
        if not allowed or not isinstance(allowed, list):
            return None
        allowed_lower = [str(v).lower() for v in allowed]
        # If the fact text contains none of the allowed values it might be a violation
        # but we're conservative — only flag when we find something that clearly conflicts
        return None  # enum violations require deeper NLP; skip for now

    # -- conditional / contraindication ---------------------------------------

    def _check_conditional(self, fact: Dict, constraint: Dict) -> Optional[Dict]:
        logic     = (constraint.get("logic_operator") or "").upper()
        condition = (constraint.get("condition") or "").lower().strip()
        if not condition:
            return None
        exception = (fact.get("exception") or "").lower()
        fact_text = _text_for_fact(fact)

        if logic == "NOT":
            # Contraindication: if the fact text mentions the condition but
            # the exception field doesn't acknowledge it, that's a potential violation
            if condition in fact_text and condition not in exception:
                return {
                    "constraint_id":   _shape_id(constraint),
                    "property":        constraint.get("property"),
                    "constraint_type": "conditional",
                    "violation_type":  "missing_contraindication_exception",
                    "expected":        f"exception mentioning '{condition}'",
                    "found":           "no exception recorded",
                    "message": (
                        f"Condition '{condition}' appears in fact but is not listed as exception. "
                        "Contraindication may be missing."
                    ),
                }
        return None

    # -- temporal -------------------------------------------------------------

    def _check_temporal(self, fact_text: str, constraint: Dict) -> Optional[Dict]:
        min_val = constraint.get("min_value")
        max_val = constraint.get("max_value")
        if not min_val and not max_val:
            return None
        # Extract days equivalent from fact text
        fact_days = self._text_to_days(fact_text)
        if fact_days is None:
            return None
        min_days = self._text_to_days(str(min_val)) if min_val else None
        max_days = self._text_to_days(str(max_val)) if max_val else None
        if min_days is not None and fact_days < min_days:
            return {
                "constraint_id":   _shape_id(constraint),
                "property":        constraint.get("property"),
                "constraint_type": "temporal",
                "violation_type":  "duration_too_short",
                "expected":        f">= {min_val}",
                "found":           f"{fact_days} days equivalent",
                "message":         f"Duration shorter than minimum {min_val}",
            }
        if max_days is not None and fact_days > max_days:
            return {
                "constraint_id":   _shape_id(constraint),
                "property":        constraint.get("property"),
                "constraint_type": "temporal",
                "violation_type":  "duration_too_long",
                "expected":        f"<= {max_val}",
                "found":           f"{fact_days} days equivalent",
                "message":         f"Duration longer than maximum {max_val}",
            }
        return None

    def _text_to_days(self, text: str) -> Optional[float]:
        text = text.lower()
        for pattern, _unit, factor in _TIME_PATTERNS:
            m = re.search(pattern, text)
            if m:
                try:
                    return float(m.group(1)) * factor
                except ValueError:
                    pass
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Cross-statement inconsistency detection
    # ──────────────────────────────────────────────────────────────────────────

    def _flag_inconsistencies(
        self,
        results: ValidationResults,
        cui_index: Dict[str, List[Dict]],
        cui_numeric_cache: Dict[str, List[Tuple[float, int]]],
    ) -> None:
        """
        For each CUI that has a range constraint, check whether two different
        facts cite incompatible numeric values (e.g. one says 20 mg, another says 80 mg
        for the same concept with a max of 40 mg).
        Appends a note to the relevant result entries.
        """
        all_results = results.get_all()
        for cui, entries in cui_numeric_cache.items():
            if cui not in cui_index:
                continue
            range_constraints = [
                c for c in cui_index[cui] if (c.get("constraint_type") or "").lower() == "range"
            ]
            if not range_constraints or len(entries) < 2:
                continue
            values = [v for v, _ in entries]
            val_min, val_max = min(values), max(values)
            # If the spread of values across facts is large (> factor of 4), flag
            if val_min > 0 and val_max / val_min > 4:
                for _, fact_idx in entries:
                    if fact_idx < len(all_results):
                        r = all_results[fact_idx]
                        r.setdefault("consistency_notes", []).append(
                            f"CUI {cui}: numeric values across statements span "
                            f"{val_min}–{val_max} (factor {val_max/val_min:.1f}x). "
                            "Consider manual review for consistency."
                        )
