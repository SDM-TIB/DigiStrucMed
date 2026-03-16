from typing import Callable, Dict, List, Optional, Set, Tuple
import csv
import json
import re
from pathlib import Path


COMMON_STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "of",
    "on", "or", "the", "to", "with", "without",
}

TYPE_KEYWORDS = {
    "medication": {
        "inhibitor", "inhibitors", "blocker", "blockers", "antagonist",
        "antagonists", "agonist", "agonists", "diuretic", "diuretics",
        "antibiotic", "antibiotics", "drug", "drugs", "medication",
        "medications", "agent", "agents", "tablet", "tablets", "capsule",
        "capsules", "injectable", "infusion",
    },
    "diagnostic_procedure": {
        "test", "tests", "assay", "assays", "screening", "scan", "scans",
        "imaging", "monitoring", "measurement", "measurements", "evaluation",
        "echocardiography", "echocardiogram", "electrocardiogram", "mri",
        "ultrasound", "catheterization", "biopsy", "diagnostic",
    },
    "therapeutic_procedure": {
        "treatment", "treatments", "therapy", "therapies", "surgery",
        "procedure", "procedures", "repair", "replacement", "implantation",
        "transplant", "transplantation", "ablation", "intervention",
        "interventions", "revascularization",
    },
    "sign_symptom": {
        "pain", "fever", "edema", "swelling", "nausea", "vomiting", "fatigue",
        "cough", "dyspnea", "breathlessness", "symptom", "symptoms",
        "headache", "dizziness", "palpitations",
    },
    "disease_disorder": {
        "disease", "disorder", "syndrome", "failure", "infection", "fibrillation",
        "hypertension", "diabetes", "injury", "infarction", "ischemia",
        "carcinoma", "cardiomyopathy", "effusion", "arrhythmia",
    },
}


class EntitiesLinker:
    def __init__(
        self,
        knowledge_base: str = "rule-based",
        umls_csv_path: Optional[str] = None,
        filter_unmatched: bool = False,
        use_partial_umls_match: bool = False,
        partial_match_max_scan: int = 2000,
        max_candidates_per_entity: int = 8,
        max_anchor_bucket_hits: int = 250,
        max_candidate_pool_size: int = 300,
        enable_type_constraints: bool = True,
    ):
        self.knowledge_base = knowledge_base
        self.filter_unmatched = filter_unmatched
        self.use_partial_umls_match = use_partial_umls_match
        self.partial_match_max_scan = partial_match_max_scan
        self.max_candidates_per_entity = max_candidates_per_entity
        self.max_anchor_bucket_hits = max_anchor_bucket_hits
        self.max_candidate_pool_size = max_candidate_pool_size
        self.enable_type_constraints = enable_type_constraints

        self.umls_entries: List[Tuple[str, str]] = []
        self.umls_by_exact: Dict[str, List[int]] = {}
        self.umls_by_normalized: Dict[str, List[int]] = {}
        self.umls_anchor_index: Dict[str, List[int]] = {}
        self.umls_loaded = False

        if umls_csv_path:
            self._load_umls_csv(umls_csv_path)
            self.knowledge_base = "umls"

    def _load_umls_csv(self, csv_path: str) -> None:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            return
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header and len(header) >= 2:
                    if header[0].strip().strip('"').lower() != "cui":
                        # The first row was actual data, not a header.
                        self._add_umls_row(header)
                for row in reader:
                    self._add_umls_row(row)
            self.umls_loaded = bool(self.umls_entries)
        except Exception:
            self.umls_loaded = False

    def _add_umls_row(self, row: List[str]) -> None:
        if len(row) < 2:
            return
        cui = row[0].strip().strip('"')
        label = row[1].strip().strip('"')
        if not cui or not label or len(label) <= 1:
            return
        entry_id = len(self.umls_entries)
        self.umls_entries.append((cui, label))

        label_lower = label.lower().strip()
        normalized = self._normalize_for_matching(label)
        self._append_index(self.umls_by_exact, label_lower, entry_id)
        if normalized:
            self._append_index(self.umls_by_normalized, normalized, entry_id)
            for token in self._extract_anchor_tokens(normalized):
                self._append_index(self.umls_anchor_index, token, entry_id)

    def _append_index(self, index: Dict[str, List[int]], key: str, entry_id: int) -> None:
        if not key:
            return
        bucket = index.get(key)
        if bucket is None:
            index[key] = [entry_id]
        else:
            bucket.append(entry_id)

    def link_entities(
        self,
        entities: List[Dict],
        context_text: str = "",
    ) -> List[Dict]:
        linked_entities: List[Dict] = []
        for entity in entities:
            entity_text = (entity.get("text") or "").strip()
            entity_label = (entity.get("label") or "").strip()
            umls_match = self._find_umls_concept(
                text=entity_text,
                entity_label=entity_label,
                context_text=context_text,
            )
            if not umls_match:
                continue
            linked = {
                "text": entity_text,
                "concept_name": umls_match["label"],
                "cui_id": umls_match["cui"],
                "label": entity_label,
            }
            if "score" in entity:
                linked["score"] = entity["score"]
            linked["link_score"] = round(float(umls_match["link_score"]), 3)
            linked["match_type"] = umls_match["match_type"]
            linked_entities.append(linked)
        return linked_entities

    def _find_umls_concept(
        self,
        text: str,
        entity_label: str = "",
        context_text: str = "",
    ) -> Optional[Dict]:
        if not self.umls_loaded:
            return None
        mention = self._build_mention_profile(text)
        candidate_ids = self._generate_candidate_ids(mention)
        if not candidate_ids:
            return None

        ranked = self._rank_candidates(
            candidate_ids=candidate_ids,
            mention=mention,
            entity_label=entity_label,
            context_text=context_text,
        )
        if not ranked:
            return None
        return ranked[0]

    def _build_mention_profile(self, text: str) -> Dict:
        raw = (text or "").strip()
        raw_lower = raw.lower()
        normalized = self._normalize_for_matching(raw)
        singular_normalized = self._singularize_phrase(normalized)
        tokens = self._tokenize(normalized)
        singular_tokens = self._tokenize(singular_normalized)
        return {
            "raw": raw,
            "raw_lower": raw_lower,
            "normalized": normalized,
            "singular_normalized": singular_normalized,
            "tokens": tokens,
            "singular_tokens": singular_tokens,
            "anchor_tokens": self._extract_anchor_tokens(normalized, singular_normalized),
        }

    def _generate_candidate_ids(self, mention: Dict) -> List[int]:
        candidate_ids: Set[int] = set()

        exact_keys = [mention["raw_lower"]]
        normalized_keys = [mention["normalized"], mention["singular_normalized"]]

        for key in exact_keys:
            candidate_ids.update(self.umls_by_exact.get(key, []))
        for key in normalized_keys:
            candidate_ids.update(self.umls_by_normalized.get(key, []))

        for token in mention["anchor_tokens"]:
            bucket = self.umls_anchor_index.get(token, [])
            if not bucket:
                continue
            candidate_ids.update(bucket[: self.max_anchor_bucket_hits])
            if len(candidate_ids) >= self.max_candidate_pool_size:
                break

        candidate_list = list(candidate_ids)
        if len(candidate_list) > self.max_candidate_pool_size:
            candidate_list = candidate_list[: self.max_candidate_pool_size]
        return candidate_list

    def _rank_candidates(
        self,
        candidate_ids: List[int],
        mention: Dict,
        entity_label: str,
        context_text: str,
    ) -> List[Dict]:
        context_tokens = set(self._tokenize(context_text))
        best_by_cui: Dict[str, Dict] = {}
        for entry_id in candidate_ids:
            cui, label = self.umls_entries[entry_id]
            label_normalized = self._normalize_for_matching(label)
            if not label_normalized:
                continue

            candidate_type = self._infer_candidate_type(label)
            if (
                self.enable_type_constraints
                and not self._is_type_compatible(entity_label, candidate_type)
            ):
                continue

            link_score, match_type = self._score_candidate(
                mention=mention,
                label=label,
                label_normalized=label_normalized,
                candidate_type=candidate_type,
                context_tokens=context_tokens,
            )
            if link_score <= 0:
                continue

            current = best_by_cui.get(cui)
            candidate = {
                "cui": cui,
                "label": label,
                "link_score": link_score,
                "match_type": match_type,
            }
            if current is None or candidate["link_score"] > current["link_score"]:
                best_by_cui[cui] = candidate

        ranked = list(best_by_cui.values())
        ranked.sort(key=lambda item: (-item["link_score"], len(item["label"])))
        return ranked[: self.max_candidates_per_entity]

    def _score_candidate(
        self,
        mention: Dict,
        label: str,
        label_normalized: str,
        candidate_type: Optional[str],
        context_tokens: Set[str],
    ) -> Tuple[float, str]:
        label_lower = label.lower().strip()
        label_singular = self._singularize_phrase(label_normalized)
        label_tokens = set(self._tokenize(label_normalized))
        mention_tokens = set(mention["tokens"])
        if mention["singular_tokens"]:
            mention_tokens |= set(mention["singular_tokens"])

        score = 0.0
        reasons: List[str] = []

        if mention["raw_lower"] == label_lower:
            score += 12.0
            reasons.append("exact")
        if mention["normalized"] and mention["normalized"] == label_normalized:
            score += 10.0
            reasons.append("normalized")
        if mention["singular_normalized"] and mention["singular_normalized"] == label_singular:
            score += 8.0
            reasons.append("singular")

        if mention["normalized"] and (
            mention["normalized"] in label_normalized or label_normalized in mention["normalized"]
        ):
            score += 4.0
            reasons.append("substring")

        if mention_tokens and label_tokens:
            overlap = len(mention_tokens & label_tokens)
            union = len(mention_tokens | label_tokens)
            if union:
                token_similarity = overlap / union
                score += token_similarity * 6.0
                if overlap:
                    reasons.append("token_overlap")
            if mention_tokens <= label_tokens or label_tokens <= mention_tokens:
                score += 1.5
                reasons.append("containment")

        if context_tokens and label_tokens:
            context_overlap = len(context_tokens & label_tokens) / max(len(label_tokens), 1)
            if context_overlap:
                score += context_overlap * 2.5
                reasons.append("context")

        if self.use_partial_umls_match and mention["normalized"] and label_normalized:
            if mention["normalized"] in label_normalized or label_normalized in mention["normalized"]:
                score += 1.0
                reasons.append("partial")

        if candidate_type:
            score += 0.5
            reasons.append(candidate_type)

        return score, "+".join(reasons) if reasons else "candidate"

    def _infer_candidate_type(self, label: str) -> Optional[str]:
        tokens = set(self._tokenize(label))
        if not tokens:
            return None
        for candidate_type in (
            "medication",
            "diagnostic_procedure",
            "therapeutic_procedure",
            "sign_symptom",
            "disease_disorder",
        ):
            if tokens & TYPE_KEYWORDS[candidate_type]:
                return candidate_type
        return None

    def _is_type_compatible(
        self,
        entity_label: str,
        candidate_type: Optional[str],
    ) -> bool:
        if not entity_label or not candidate_type:
            return True
        label_lower = entity_label.lower()
        if "medication" in label_lower:
            return candidate_type == "medication"
        if "disease" in label_lower or "disorder" in label_lower:
            return candidate_type == "disease_disorder"
        if "sign" in label_lower or "symptom" in label_lower:
            return candidate_type == "sign_symptom"
        if "diagnostic" in label_lower:
            return candidate_type == "diagnostic_procedure"
        if "therapeutic" in label_lower:
            return candidate_type == "therapeutic_procedure"
        return True

    def _extract_anchor_tokens(self, *texts: str) -> List[str]:
        tokens: Set[str] = set()
        for text in texts:
            for token in self._tokenize(text):
                if len(token) >= 4 and token not in COMMON_STOPWORDS:
                    tokens.add(token)
        return sorted(tokens, key=lambda token: (-len(token), token))[:3]

    def _tokenize(self, text: str) -> List[str]:
        normalized = self._normalize_for_matching(text)
        if not normalized:
            return []
        return [token for token in normalized.split() if token and token not in COMMON_STOPWORDS]

    def _normalize_for_matching(self, text: str) -> str:
        normalized = (text or "").lower()
        normalized = normalized.replace("/", " ")
        normalized = re.sub(r"[\(\)\[\]\{\},;:]+", " ", normalized)
        normalized = re.sub(r"[-_]+", " ", normalized)
        normalized = re.sub(r"[^\w\s]", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _singularize_phrase(self, text: str) -> str:
        if not text:
            return ""
        tokens = [self._singularize_token(token) for token in text.split()]
        return " ".join(token for token in tokens if token)

    def _singularize_token(self, token: str) -> str:
        if len(token) <= 3:
            return token
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if re.search(r"(xes|zes|ches|shes|sses)$", token):
            return token[:-2]
        if token.endswith("s") and not token.endswith(("ss", "us", "is")):
            return token[:-1]
        return token

    def _normalize_text(self, text: str) -> str:
        normalized = text.lower()
        normalized = re.sub(r"[^\w\s-]", "", normalized)
        normalized = re.sub(r"[-\s]+", "_", normalized)
        return normalized

    def get_umls_stats(self) -> Dict:
        return {
            "umls_loaded": self.umls_loaded,
            "total_concepts": len(self.umls_entries),
            "exact_index_keys": len(self.umls_by_exact),
            "normalized_index_keys": len(self.umls_by_normalized),
            "anchor_index_keys": len(self.umls_anchor_index),
            "knowledge_base": self.knowledge_base,
        }


# --- Stage D v2: Bi-encoder linker (optional deps: sentence_transformers, faiss) ---

def build_biencoder_index(
    umls_csv_path: str,
    model_name: str,
    index_dir: str,
    batch_size: int = 50000,
    max_concepts: Optional[int] = None,
) -> Tuple[int, str]:
    """
    Build a Faiss index and CUI map from a UMLS CSV for use with BiEncoderLinker.
    Call this once (e.g. in Colab with GPU) and save; then load in BiEncoderLinker.
    Returns (num_indexed, index_dir).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Stage D v2 requires sentence_transformers: pip install sentence-transformers")
    try:
        import faiss
    except ImportError:
        raise ImportError("Stage D v2 requires faiss: pip install faiss-cpu (or faiss-gpu)")

    Path(index_dir).mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name)
    labels: List[str] = []
    cuis: List[str] = []
    with open(umls_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header and len(header) >= 2 and header[0].strip().strip('"').lower() == "cui":
            pass
        elif header and len(header) >= 2:
            cuis.append(header[0].strip().strip('"'))
            labels.append(header[1].strip().strip('"'))
        for row in reader:
            if len(row) < 2:
                continue
            cui, label = row[0].strip().strip('"'), row[1].strip().strip('"')
            if not cui or not label or len(label) <= 1:
                continue
            cuis.append(cui)
            labels.append(label)
            if max_concepts and len(cuis) >= max_concepts:
                break

    n = len(cuis)
    if n == 0:
        raise ValueError("No concepts read from UMLS CSV")
    dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_labels = labels[start:end]
        emb = model.encode(batch_labels, normalize_embeddings=True, show_progress_bar=end - start > 1000)
        index.add(emb.astype("float32"))
    faiss.write_index(index, str(Path(index_dir) / "umls.faiss"))
    with open(Path(index_dir) / "cui_map.json", "w", encoding="utf-8") as f:
        json.dump({"cuis": cuis, "labels": labels}, f, ensure_ascii=False)
    return n, index_dir


def _load_biencoder_deps():
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        return SentenceTransformer, faiss
    except ImportError as e:
        raise ImportError(
            "Stage D v2 (bi-encoder) requires: pip install sentence-transformers faiss-cpu"
        ) from e


def make_llm_disambiguator(llm_run_inference, max_new_tokens: int = 50) -> Callable[[str], Optional[str]]:
    """
    Build a disambiguator callable for BiEncoderLinker (NSSC/BioLinker-style).
    llm_run_inference: a method like ValidationModel.run_inference(prompt, max_new_tokens) -> str.
    Returns a function prompt -> CUI string or None.
    """
    def disambiguate(prompt: str) -> Optional[str]:
        raw = llm_run_inference(prompt, max_new_tokens=max_new_tokens)
        if not raw:
            return None
        raw = raw.strip().upper()
        if raw in ("NONE", "N/A", ""):
            return None
        if raw.startswith("C") and len(raw) >= 8 and raw[1:].isdigit():
            return raw
        import re
        m = re.search(r"C\d{7,}", raw)
        return m.group(0) if m else None
    return disambiguate


def _ner_label_to_type_key(label: str) -> Optional[str]:
    """Map NER label to TYPE_KEYWORDS key (lowercase, underscores)."""
    if not label:
        return None
    key = label.strip().lower().replace(" ", "_")
    return key if key in TYPE_KEYWORDS else None


def _type_bonus(concept_name: str, ner_type_key: Optional[str]) -> float:
    """Return a small bonus if concept_name contains keywords for ner_type_key."""
    if not ner_type_key or ner_type_key not in TYPE_KEYWORDS:
        return 0.0
    words = set(re.findall(r"[a-z0-9]+", concept_name.lower()))
    overlap = words & TYPE_KEYWORDS[ner_type_key]
    return 0.05 if overlap else 0.0


class BiEncoderLinker:
    """
    Stage D v2: link entities using a pre-trained bi-encoder (e.g. SapBERT).
    Requires a pre-built index from build_biencoder_index() (e.g. built in Colab).
    Optimal mode: top_k retrieval + type-based reranking + optional score threshold
    and optional LLM disambiguation (NSSC/BioLinker-style).
    """

    def __init__(
        self,
        model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        index_dir: str = "",
        top_k: int = 16,
        max_length: int = 512,
        device: Optional[str] = None,
        min_link_score: Optional[float] = 0.72,
        use_type_rerank: bool = True,
        prefer_shorter_concept: bool = True,
        disambiguator: Optional[Callable[..., Optional[str]]] = None,
    ):
        SentenceTransformer, faiss = _load_biencoder_deps()
        self.model = SentenceTransformer(model_name, device=device or "cuda")
        self.retrieval_top_k = max(1, top_k)
        self.max_length = max_length
        self.min_link_score = min_link_score
        self.use_type_rerank = use_type_rerank
        self.prefer_shorter_concept = prefer_shorter_concept
        self.disambiguator = disambiguator
        index_path = Path(index_dir) / "umls.faiss"
        cui_map_path = Path(index_dir) / "cui_map.json"
        if not index_path.exists() or not cui_map_path.exists():
            raise FileNotFoundError(
                f"Stage D v2 index not found in {index_dir}. "
                "Run build_biencoder_index() in Colab first and upload the index directory."
            )
        self.index = faiss.read_index(str(index_path))
        with open(cui_map_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.cuis = data["cuis"]
        self.labels = data["labels"]
        self.knowledge_base = "umls"

    def _rerank_candidates(
        self,
        candidates: List[Tuple[str, str, float]],
        ner_label: str,
    ) -> Optional[Tuple[str, str, float]]:
        """Pick best (cui_id, concept_name, score) from candidates using type bonus and tie-break."""
        if not candidates:
            return None
        type_key = _ner_label_to_type_key(ner_label) if self.use_type_rerank else None
        scored = []
        for cui, name, score in candidates:
            bonus = _type_bonus(name, type_key)
            total = score + bonus
            scored.append((total, len(name), cui, name, score))
        scored.sort(key=lambda x: (-x[0], x[1] if self.prefer_shorter_concept else 0))
        _, _, cui, name, score = scored[0]
        return (cui, name, score)

    def link_entities(
        self,
        entities: List[Dict],
        context_text: str = "",
    ) -> List[Dict]:
        if not entities:
            return []
        import numpy as np
        texts = []
        for entity in entities:
            mention = (entity.get("text") or "").strip()
            if not mention:
                texts.append("")
                continue
            ctx = (context_text or "").strip()[: self.max_length - 50]
            if ctx:
                texts.append(f"{ctx} [SEP] {mention}")
            else:
                texts.append(mention)
        emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        D, I = self.index.search(emb.astype("float32"), self.retrieval_top_k)
        linked_entities: List[Dict] = []
        for i, entity in enumerate(entities):
            entity_text = (entity.get("text") or "").strip()
            if not entity_text:
                continue
            ner_label = (entity.get("label") or "").strip()
            candidates = []
            for k in range(D.shape[1]):
                idx = int(I[i, k])
                score = float(D[i, k]) if D.size > 0 else 0.0
                candidates.append((self.cuis[idx], self.labels[idx], score))
            if self.disambiguator:
                chosen = self._disambiguate_with_llm(
                    entity_text, context_text, candidates, ner_label
                )
            else:
                chosen = self._rerank_candidates(candidates, ner_label)
            if chosen is None:
                continue
            cui, concept_name, score = chosen
            below = self.min_link_score is not None and score < self.min_link_score
            rec = {
                "text": entity_text,
                "concept_name": concept_name if not below else "",
                "cui_id": cui if not below else "",
                "label": ner_label,
                "link_score": round(score, 4),
                "match_type": "biencoder_rerank" if self.retrieval_top_k > 1 else "biencoder",
            }
            if below:
                rec["below_threshold"] = True
            linked_entities.append(rec)
            if "score" in entity:
                linked_entities[-1]["score"] = entity["score"]
        return linked_entities

    def _disambiguate_with_llm(
        self,
        mention: str,
        context: str,
        candidates: List[Tuple[str, str, float]],
        ner_label: str,
    ) -> Optional[Tuple[str, str, float]]:
        """Use optional LLM disambiguator to pick best candidate (NSSC/BioLinker-style)."""
        if not self.disambiguator or not candidates:
            return self._rerank_candidates(candidates, ner_label)
        prompt = self._build_disambiguation_prompt(mention, context, candidates[:10], ner_label)
        try:
            out = self.disambiguator(prompt)
            if out and isinstance(out, str):
                c = out.strip().upper()
                for cui, name, score in candidates:
                    if (cui or "").upper() == c:
                        return (cui, name, score)
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                return (out[0], out[1], out[2] if len(out) > 3 else 0.0)
        except Exception:
            pass
        return self._rerank_candidates(candidates, ner_label)

    def _build_disambiguation_prompt(
        self,
        mention: str,
        context: str,
        candidates: List[Tuple[str, str, float]],
        ner_label: str,
    ) -> str:
        lines = [f"Mention: \"{mention}\"", f"Context: {context[:400]}", f"Entity type: {ner_label or 'unknown'}", "Candidates (CUI | concept name):"]
        for cui, name, score in candidates:
            lines.append(f"  {cui} | {name}")
        lines.append("Return only the correct CUI (e.g. C1234567), or NONE if no match.")
        return "\n".join(lines)

    def get_umls_stats(self) -> Dict:
        return {
            "umls_loaded": True,
            "total_concepts": len(self.cuis),
            "knowledge_base": "umls",
            "linker_version": "v2_biencoder",
        }
