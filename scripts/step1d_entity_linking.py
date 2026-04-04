"""
Step 1d — Entity Linking  [SYMBOLIC]
─────────────────────────────────────────────────────────────────────────────
Input  : outputs/step1/entity_mentions.json   (from Step 1c)
         UMLS background knowledge CSV         (cui, label, semantic_type,
                                                definition)
Output : outputs/step1/grounded_entities.json
         Each record inherits Step 1c fields plus:
           { cui_candidates: [{cui, label, semantic_type, definition, sim}],
             cui_final: str | None,
             linking_status: "direct" | "needs_disambiguation" | "no_match" }

Hybrid (symbolic + neural is split across steps):
  · Step 1d — Rule-based candidate generation: token inverted index over UMLS
    labels + fuzzy scores on a bounded candidate set only (no full-BK scan).
    Cheap heuristics decide direct link vs unresolved.
  · Step 1e — AI ambiguity resolution: LLM runs only for mentions with
    linking_status == needs_disambiguation, choosing among the candidate list
    from 1d (see step1e_disambiguate.py).

  Optional disk cache: after the first CSV load + index build, the BK and index
  are stored under outputs/cache/ so later runs skip re-reading millions of rows.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import pickle
import random
import re
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from utils import log, save_json

DEFAULT_SIM_THRESHOLD = 0.96
DEFAULT_TOP_K = 15
# Max BK rows to score per mention after index retrieval (avoids O(mentions × |BK|))
DEFAULT_MAX_SCORE = 12_000
DEFAULT_CACHE_DIR = "outputs/cache"
# Bump when BK row shape / index logic changes (invalidates old .pkl.gz caches).
_UMLS_LINKING_CACHE_VERSION = 4

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

try:
    from rapidfuzz import fuzz as _rfuzz

    def _sim_ratio(a: str, b: str) -> float:
        return _rfuzz.ratio(a.lower().strip(), b.lower().strip()) / 100.0

    _SIM_BACKEND = "rapidfuzz"
except ImportError:
    def _sim_ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

    _SIM_BACKEND = "difflib"


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= 2]


def _normalize_umls_header_key(k: object) -> str:
    """Strip BOM/whitespace and lowercase so DictReader keys align with cui/label."""
    if k is None:
        return ""
    s = str(k).replace("\ufeff", "").strip().strip("\"'").lower()
    return s


def _normalize_umls_row_keys(row: dict) -> dict:
    """
    Lowercase and strip CSV header keys so CUI/Label match cui/label.
    Values are kept as read from CSV (str); None becomes "".
    """
    out: dict[str, object] = {}
    for k, v in row.items():
        key = _normalize_umls_header_key(k)
        if not key:
            continue
        if v is None:
            out[key] = ""
        else:
            out[key] = v
    return out


def _canonicalize_concept_row(row: dict[str, object]) -> dict[str, object]:
    """
    Ensure every row has string ``cui`` and ``label`` for indexing and output.

    UMLS/RxNorm exports often use STR instead of Label; some files omit one key
    if BOM mangled the header.
    """
    r = dict(row)

    def _first_nonempty(keys: tuple[str, ...]) -> str:
        for key in keys:
            raw = r.get(key)
            if raw is None:
                continue
            s = str(raw).strip()
            if s:
                return s
        return ""

    cui = _first_nonempty(("cui", "cui_code", "code", "umls_cui"))
    label = _first_nonempty(
        ("label", "str", "string", "preferred_name", "name", "term", "synonym")
    )

    r["cui"] = cui
    r["label"] = label
    return r


def _preview_text(value: object, limit: int = 120) -> str:
    text = str(value or "").replace("\n", "\\n").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def load_umls_bk(umls_csv_path: str) -> list[dict]:
    """
    Load UMLS Background Knowledge from a CSV file.

    Expected columns (at minimum): cui, label, semantic_type, definition
    Header names are matched case-insensitively (e.g. CUI, Label).
    ``STR`` / ``name`` / etc. are mapped to ``label`` when ``label`` is empty.
    File is read as UTF-8 with optional BOM (utf-8-sig).
    Delimiter is sniffed (``,``, ``;``, tab) so European CSV exports still parse.
    """
    bk: list[dict] = []
    # utf-8-sig strips BOM so the first header is "CUI" not "\ufeffCUI"
    with open(umls_csv_path, newline="", encoding="utf-8-sig") as f:
        head = f.read(65536)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(head, delimiters=",\t;")
        except csv.Error:
            dialect = csv.get_dialect("excel")
        reader = csv.DictReader(f, dialect=dialect)
        raw_headers = list(reader.fieldnames or [])
        normalized_headers = [
            _normalize_umls_header_key(h) for h in raw_headers if h is not None
        ]
        for row in reader:
            bk.append(_canonicalize_concept_row(_normalize_umls_row_keys(row)))
    nonempty_labels = sum(1 for x in bk if str(x.get("label", "")).strip())
    nonempty_cuis = sum(1 for x in bk if str(x.get("cui", "")).strip())
    sample = bk[0] if bk else {}
    log(
        "1d",
        f"Loaded UMLS BK: {len(bk)} concepts from {umls_csv_path} "
        f"({nonempty_labels} rows with non-empty label)",
    )
    log(
        "1d",
        "CSV diagnostics: "
        f"delimiter={getattr(dialect, 'delimiter', ',')!r}; "
        f"headers={raw_headers[:8]!r}; "
        f"normalized={normalized_headers[:8]!r}",
    )
    if bk:
        log(
            "1d",
            "Sample parsed concept: "
            f"cui={_preview_text(sample.get('cui'))!r}; "
            f"label={_preview_text(sample.get('label'))!r}; "
            f"keys={sorted(sample.keys())[:8]!r}",
        )
    if bk and nonempty_cuis == 0:
        raise ValueError(
            "UMLS CSV loaded rows but no non-empty CUI values were found. "
            f"Headers seen: {raw_headers[:12]!r}"
        )
    if bk and nonempty_labels == 0:
        raise ValueError(
            "UMLS CSV loaded rows but no non-empty labels were found, so entity "
            "linking cannot build an index. "
            f"Headers seen: {raw_headers[:12]!r}; "
            f"sample row keys: {sorted(sample.keys())[:12]!r}"
        )
    return bk


def _build_inverted_index(bk: list[dict]) -> dict[str, set[int]]:
    """Token (from label) -> set of row indices."""
    inverted: dict[str, set[int]] = defaultdict(set)
    for i, concept in enumerate(bk):
        label = str(concept.get("label") or "")
        for tok in _tokens(label):
            inverted[tok].add(i)
    return dict(inverted)


def _token_freq(inverted: dict[str, set[int]]) -> dict[str, int]:
    return {t: len(s) for t, s in inverted.items()}


def _umls_file_fingerprint(umls_csv_path: str) -> str:
    """Invalidate cache when path, size, or mtime changes."""
    p = Path(umls_csv_path).resolve()
    st = p.stat()
    raw = f"{p}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:28]


def _load_bk_index_from_cache(cache_file: Path, expect_fp: str) -> tuple[list[dict], dict[str, set[int]], dict[str, int]] | None:
    try:
        with gzip.open(cache_file, "rb") as f:
            payload = pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError):
        return None
    if payload.get("cache_version") != _UMLS_LINKING_CACHE_VERSION:
        return None
    if payload.get("fingerprint") != expect_fp:
        return None
    bk = payload["bk"]
    inverted = payload["inverted"]
    freq = payload["freq"]
    return bk, inverted, freq


def _save_bk_index_cache(
    cache_file: Path,
    fingerprint: str,
    bk: list[dict],
    inverted: dict[str, set[int]],
    freq: dict[str, int],
) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": _UMLS_LINKING_CACHE_VERSION,
        "fingerprint": fingerprint,
        "bk": bk,
        "inverted": inverted,
        "freq": freq,
    }
    with gzip.open(cache_file, "wb", compresslevel=3) as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_bk_and_inverted_index(
    umls_csv_path: str,
    *,
    cache_dir: str | None,
    use_cache: bool,
) -> tuple[list[dict], dict[str, set[int]], dict[str, int]]:
    """
    Load BK from CSV and build the token index, or restore both from cache.

    Cache is keyed by resolved path + file size + mtime (not a full hash of
    4M rows — cheap and usually sufficient).
    """
    fp = _umls_file_fingerprint(umls_csv_path)
    if use_cache and cache_dir:
        cdir = Path(cache_dir)
        cache_file = cdir / f"umls_linking_{fp}.pkl.gz"
        restored = _load_bk_index_from_cache(cache_file, fp)
        if restored is not None:
            bk, inverted, freq = restored
            log(
                "1d",
                f"Restored BK + index from cache ({len(bk)} concepts) → {cache_file.name}",
            )
            return bk, inverted, freq
        log("1d", "No valid cache for this UMLS file — loading CSV and building index…")

    bk = load_umls_bk(umls_csv_path)
    log("1d", f"Building token index over labels (similarity={_SIM_BACKEND})…")
    t_idx0 = time.perf_counter()
    inverted = _build_inverted_index(bk)
    freq = _token_freq(inverted)
    log(
        "1d",
        f"Index ready: {len(inverted)} distinct tokens in {time.perf_counter() - t_idx0:.1f}s",
    )
    if len(inverted) == 0 and bk:
        s0 = bk[0]
        raise ValueError(
            "UMLS token index is empty after parsing labels. "
            f"First row keys={sorted(s0.keys())!r}; "
            f"cui={str(s0.get('cui'))[:40]!r}; "
            f"label={str(s0.get('label'))[:80]!r}"
        )

    if use_cache and cache_dir:
        cache_file = Path(cache_dir) / f"umls_linking_{fp}.pkl.gz"
        t_w0 = time.perf_counter()
        try:
            _save_bk_index_cache(cache_file, fp, bk, inverted, freq)
            log(
                "1d",
                f"Wrote cache ({time.perf_counter() - t_w0:.1f}s) → {cache_file}",
            )
        except OSError as e:
            log("1d", f"Could not write cache ({e}); continuing without.")

    return bk, inverted, freq


def _sample_indices(indices: set[int], k: int, seed_key: str) -> set[int]:
    if len(indices) <= k:
        return set(indices)
    rng = random.Random(hash(seed_key) & 0xFFFFFFFF)
    return set(rng.sample(list(indices), k))


def _candidate_indices(
    mention_text: str,
    inverted: dict[str, set[int]],
    freq: dict[str, int],
    max_score: int,
) -> set[int]:
    """
    Retrieve a bounded set of BK row indices likely to match the mention.

    Uses token overlap on labels (rarest tokens first); does not scan all concepts.
    """
    mtoks = _tokens(mention_text)
    valid = [t for t in mtoks if t in inverted]
    if not valid:
        return set()

    valid.sort(key=lambda t: freq.get(t, 0))

    a0 = inverted[valid[0]]
    if len(valid) == 1:
        return _sample_indices(a0, max_score, mention_text)

    cand = a0 & inverted[valid[1]]
    for t in valid[2:]:
        if len(cand) <= max_score:
            break
        cand &= inverted[t]

    if not cand:
        u = a0 | inverted[valid[1]]
        return _sample_indices(u, max_score, mention_text)

    if len(cand) > max_score:
        return _sample_indices(cand, max_score, mention_text)
    return cand


def _link_one_mention(
    mention: dict,
    bk: list[dict],
    inverted: dict[str, set[int]],
    freq: dict[str, int],
    top_k: int,
    sim_threshold: float,
    max_score: int,
) -> dict:
    entity_text = mention["text"]
    scored: list[dict] = []
    for idx in _candidate_indices(entity_text, inverted, freq, max_score):
        concept = bk[idx]
        label = str(concept.get("label") or "")
        s = _sim_ratio(entity_text, label)
        if s > 0.25:
            scored.append({**concept, "_sim": round(s, 4)})

    scored.sort(key=lambda x: x["_sim"], reverse=True)
    top_candidates = scored[:top_k]

    candidates_out = [
        {
            "cui": c.get("cui", ""),
            "label": c.get("label", ""),
            "semantic_type": c.get("semantic_type", ""),
            "definition": (c.get("definition") or "")[:200],
            "sim": c["_sim"],
        }
        for c in top_candidates
    ]

    if not top_candidates:
        status = "no_match"
        cui_final = None
    elif len(top_candidates) == 1 or top_candidates[0]["_sim"] >= sim_threshold:
        status = "direct"
        cui_final = top_candidates[0]["cui"]
    else:
        status = "needs_disambiguation"
        cui_final = None

    return {
        **mention,
        "cui_candidates": candidates_out,
        "cui_final": cui_final,
        "linking_status": status,
    }


def link_entities(
    mentions_path: str = "outputs/step1/entity_mentions.json",
    umls_csv_path: str | None = None,
    output_dir: str = "outputs/step1",
    top_k: int = DEFAULT_TOP_K,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    max_score: int = DEFAULT_MAX_SCORE,
    progress_every: int = 200,
    cache_dir: str | None = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> list[dict]:
    """
    Link every entity mention to UMLS candidate CUIs.

    If umls_csv_path is None, all entities are marked 'needs_disambiguation'
    so the system can still proceed without UMLS data (useful for testing).

    Parameters
    ----------
    max_score
        Maximum BK rows to run string similarity against per mention.  The
        default avoids scoring every concept in a multi-million-row subset.
    progress_every
        Log progress every N mentions (0 to disable).
    cache_dir
        Directory for gzip-pickled BK + inverted index.  ``None`` disables
        reading/writing cache files.
    use_cache
        If False, always load CSV and rebuild the index (ignore on-disk cache).
    """
    mentions: list[dict] = json.loads(
        Path(mentions_path).read_text(encoding="utf-8")
    )

    if not umls_csv_path:
        results = []
        for mention in mentions:
            results.append({
                **mention,
                "cui_candidates": [],
                "cui_final": None,
                "linking_status": "needs_disambiguation",
            })
        out_path = Path(output_dir) / "grounded_entities.json"
        save_json(results, str(out_path))
        log("1d", "No UMLS path — all mentions marked needs_disambiguation")
        return results

    t0 = time.perf_counter()
    bk, inverted, freq = load_bk_and_inverted_index(
        umls_csv_path,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )

    counts = {"direct": 0, "needs_disambiguation": 0, "no_match": 0}
    results: list[dict] = []

    n = len(mentions)
    for i, mention in enumerate(mentions):
        r = _link_one_mention(
            mention, bk, inverted, freq, top_k, sim_threshold, max_score
        )
        counts[r["linking_status"]] += 1
        results.append(r)
        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.perf_counter() - t0
            log(
                "1d",
                f"  … {i + 1}/{n} mentions ({elapsed:.0f}s elapsed)",
            )

    out_path = Path(output_dir) / "grounded_entities.json"
    save_json(results, str(out_path))
    log(
        "1d",
        (
            f"Linked {len(results)} entities — "
            f"direct: {counts['direct']}, "
            f"needs_disambiguation: {counts['needs_disambiguation']}, "
            f"no_match: {counts['no_match']} "
            f"(total {time.perf_counter() - t0:.1f}s)"
        ),
    )
    return results


if __name__ == "__main__":
    import sys

    umls = sys.argv[1] if len(sys.argv) > 1 else None
    link_entities(umls_csv_path=umls)
