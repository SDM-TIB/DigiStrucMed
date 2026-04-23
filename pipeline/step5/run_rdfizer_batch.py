"""Batch-run SDM-RDFizer per guideline under ``outputs/pipeline-output2_new/<slug>/``.

Uses ``input/MappingRules/mappingrules.rml.ttl`` and ``rdfizer_config.template.ini``.
Default is subprocess ``python -m rdfizer`` with UTF-8 env; pass ``--inprocess`` for
``rdfizer.semantify.semantify`` in this interpreter. Merges all ``step5/*.ttl`` into
``outputs/merged_kg/``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

# RDFizer Turtle serializer quirks: repair orphan `>.` before `ex:` and lone `;` before `<subject`.
_ORPHAN_DOT_RE = re.compile(
    r'(?P<end>>|"(?:[^"\\]|\\.)*"(?:\^\^[^\s.]+|@[^\s.]+)?)\.'
    r'(?P<sep>(?:\r?\n\s*){1,}\r?\n\s+)ex:',
    re.MULTILINE,
)

_ORPHAN_SEMI_RE = re.compile(
    r'(?P<end>>|"(?:[^"\\]|\\.)*"(?:\^\^[^\s.]+|@[^\s.]+)?);'
    r'(?P<sep>\r?\n)(?=<)',
    re.MULTILINE,
)


def _read_text_tolerant(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8-with-replace"


def _fix_orphan_periods(ttl_path: Path) -> tuple[int, str]:
    text, enc = _read_text_tolerant(ttl_path)
    fixed, n_dot = _ORPHAN_DOT_RE.subn(
        lambda m: f"{m.group('end')};{m.group('sep')}ex:", text
    )
    fixed, n_semi = _ORPHAN_SEMI_RE.subn(
        lambda m: f"{m.group('end')}.{m.group('sep')}", fixed
    )
    n = n_dot + n_semi
    if n or enc != "utf-8":
        ttl_path.write_bytes(fixed.encode("utf-8"))
    return n, enc

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "outputs" / "pipeline-output2_new"
MAPPING = REPO_ROOT / "input" / "MappingRules" / "mappingrules.rml.ttl"
TEMPLATE = REPO_ROOT / "pipeline" / "step5" / "rdfizer_config.template.ini"
MERGED_DIR = REPO_ROOT / "outputs" / "merged_kg"


def _render_config(slug: str, slug_dir: Path, out_dir: Path) -> Path:
    text = TEMPLATE.read_text(encoding="utf-8")
    text = (text
            .replace("{MAIN_DIR}", str(slug_dir).replace("\\", "/") + "/")
            .replace("{OUT_DIR}", str(out_dir).replace("\\", "/") + "/")
            .replace("{NAME}", slug)
            .replace("{MAPPING}", str(MAPPING).replace("\\", "/")))
    cfg = out_dir / "config.ini"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(text, encoding="utf-8")
    return cfg


def _run_inprocess(cfg: Path, log_path: Path, cwd: Path) -> int:
    print(f"    > rdfizer.semantify.semantify('{cfg}')  (in-process, cwd={cwd})")
    try:
        from rdfizer.semantify import semantify
    except ImportError as e:
        print(f"    ! `rdfizer` is not importable in {sys.executable}")
        print(f"      -> {e}")
        print( "      -> install with:  python -m pip install rdfizer")
        return 127
    original_cwd = os.getcwd()
    try:
        os.chdir(cwd)
        semantify(str(cfg), str(log_path))
        return 0
    except SystemExit as se:
        return int(getattr(se, "code", 1) or 0)
    except Exception:
        print("    ! semantify raised an exception:")
        traceback.print_exc()
        return 1
    finally:
        os.chdir(original_cwd)


def _run_subprocess(cfg: Path, cwd: Path) -> int:
    cmd = [sys.executable, "-m", "rdfizer", "-c", str(cfg)]
    print(f"    > {' '.join(cmd)}  (cwd={cwd}, PYTHONUTF8=1)")
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd), env=env,
                       encoding="utf-8", errors="replace")
    if p.returncode != 0:
        print(f"    ! exit {p.returncode}")
    if p.stdout:
        sys.stdout.write(p.stdout)
    if p.stderr:
        sys.stderr.write(p.stderr)
    return p.returncode


def _per_guideline(slugs: list[str], use_subprocess: bool, dry: bool) -> list[dict]:
    results: list[dict] = []
    for slug in slugs:
        slug_dir = ROOT / slug
        out_dir = slug_dir / "step5"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[step5] {slug}")
        cfg = _render_config(slug, slug_dir, out_dir)
        kg_path = out_dir / f"{slug}.ttl"
        log_path = out_dir / "rdfizer.log"

        if dry:
            print(f"    > (dry-run) config written to {cfg}")
            rc = 0
        else:
            t0 = time.time()
            if use_subprocess:
                rc = _run_subprocess(cfg, slug_dir)
            else:
                rc = _run_inprocess(cfg, log_path, slug_dir)
            elapsed = time.time() - t0
            print(f"    ({'ok' if rc == 0 else 'FAIL'} in {elapsed:.1f}s)")

        exists = kg_path.is_file()
        patched = 0
        enc = None
        triples = 0
        parse_error = None
        if exists:
            patched, enc = _fix_orphan_periods(kg_path)
            try:
                from rdflib import Graph
                g = Graph()
                g.parse(str(kg_path), format="turtle")
                triples = len(g)
            except Exception as e:
                parse_error = str(e)
        if patched or (enc and enc != "utf-8"):
            print(f"    (patched {patched} orphan-period(s), encoding={enc})")
        if parse_error:
            print(f"    ! rdflib parse failed: {parse_error}")
        results.append({"slug": slug, "rc": rc, "kg": str(kg_path),
                        "triples": triples, "exists": exists,
                        "patched": patched, "encoding": enc})
    return results


def _merge(results: list[dict]) -> dict:
    from rdflib import Graph

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    out_nt = MERGED_DIR / "merged.nt"
    out_ttl = MERGED_DIR / "merged.ttl"

    g = Graph()
    total_in = 0
    per_file: list[dict] = []
    for r in results:
        p = Path(r["kg"])
        if not p.is_file():
            print(f"  skip merge: {p} not found")
            continue
        gi = Graph()
        gi.parse(str(p), format="turtle")
        n_in = len(gi)
        total_in += n_in
        g += gi
        per_file.append({"slug": r["slug"], "triples": n_in})

    g.serialize(destination=str(out_nt), format="nt")
    g.serialize(destination=str(out_ttl), format="turtle")

    stats = {"files_merged": len(per_file),
             "triples_in": total_in,
             "unique_triples_out": len(g),
             "merged_nt": str(out_nt),
             "merged_ttl": str(out_ttl),
             "per_file": per_file}
    (MERGED_DIR / "merged_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\n[merge] {len(g)} unique triples (from {total_in}) -> {out_nt}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", action="append", default=[],
                    help="Run only this slug (can be used multiple times). Default: all.")
    ap.add_argument("--inprocess", action="store_true",
                    help="Call rdfizer.semantify.semantify() in-process (default is subprocess "
                         "with PYTHONUTF8=1, which handles non-ASCII chars like \u2264 correctly "
                         "and isolates each guideline's run).")
    ap.add_argument("--dry-run", action="store_true", help="Write configs only; don't run RDFizer.")
    ap.add_argument("--skip-merge", action="store_true", help="Skip the merge step.")
    args = ap.parse_args()

    if not ROOT.is_dir():
        raise SystemExit(f"Missing: {ROOT}")
    if not MAPPING.is_file():
        raise SystemExit(f"Missing: {MAPPING}")

    slugs = args.slug or sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    missing = [s for s in slugs if not (ROOT / s).is_dir()]
    if missing:
        raise SystemExit(f"Unknown slug(s): {missing}")

    print(f"Python:      {sys.executable}")
    print(f"Guidelines to process ({len(slugs)}):")
    for s in slugs:
        print(f"  - {s}")

    use_subprocess = not args.inprocess
    results = _per_guideline(slugs, use_subprocess, args.dry_run)

    print("\n[per-guideline results]")
    for r in results:
        ok = "OK  " if r["exists"] and r["rc"] == 0 else "FAIL"
        print(f"  {ok}  {r['slug']:<70}  triples={r['triples']:>7}")

    if args.skip_merge or args.dry_run:
        return
    _merge(results)


if __name__ == "__main__":
    main()
