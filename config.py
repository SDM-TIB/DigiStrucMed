"""
Unified configuration. Loads from config.json if present; falls back to built-in defaults.
"""
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).parent
CONFIG_FILE = PROJECT_ROOT / "config.json"

# Built-in defaults (used when config.json is missing or incomplete)
_DEFAULTS = {
    "paths": {
        "pdf_dir": "input",
        "outputs_root": "outputs",
    },
    "versions": {
        "A": "v1",
        "B": "v1",
        "C": "v1",
        "D": "v1",
        "E": "v1",
    },
}


def _load_config() -> dict:
    """Load config from config.json, merging with defaults."""
    out = {}
    for k, v in _DEFAULTS.items():
        out[k] = dict(v) if isinstance(v, dict) else v

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key in ("paths", "versions"):
                if key in data and isinstance(data[key], dict):
                    out[key].update(data[key])
        except (json.JSONDecodeError, IOError):
            pass

    return out


_CONFIG = _load_config()

# Resolved paths (relative to PROJECT_ROOT)
PDF_DIR = PROJECT_ROOT / _CONFIG["paths"]["pdf_dir"]
OUTPUTS_ROOT = PROJECT_ROOT / _CONFIG["paths"]["outputs_root"]

# Local NER model cache: run download_ner_model.py to pre-download models here.
NER_MODELS_DIR = PROJECT_ROOT / "models" / "ner"


def ner_model_local_path(model_id: str) -> Path | None:
    """Return local path if model is cached in NER_MODELS_DIR, else None."""
    folder = model_id.replace("/", "__")
    path = NER_MODELS_DIR / folder
    return path if path.is_dir() else None


# Default stage versions (from config)
DEFAULT_STAGE_A_VERSION = _CONFIG["versions"].get("A", "v1")
DEFAULT_STAGE_B_VERSION = _CONFIG["versions"].get("B", "v1")
DEFAULT_STAGE_C_VERSION = _CONFIG["versions"].get("C", "v1")
DEFAULT_STAGE_D_VERSION = _CONFIG["versions"].get("D", "v1")
DEFAULT_STAGE_E_VERSION = _CONFIG["versions"].get("E", "v1")


def stage_a_dir(version: str | None = None) -> Path:
    """Return the directory containing Stage A outputs for the given version."""
    v = version or DEFAULT_STAGE_A_VERSION
    return OUTPUTS_ROOT / f"STAGE_A_{v}"


def stage_b_dir(version: str | None = None) -> Path:
    """Return the directory containing Stage B outputs for the given version."""
    v = version or DEFAULT_STAGE_B_VERSION
    return OUTPUTS_ROOT / f"STAGE_B_{v}"


def stage_c_dir(version: str | None = None) -> Path:
    """Return the directory containing Stage C outputs for the given version."""
    v = version or DEFAULT_STAGE_C_VERSION
    return OUTPUTS_ROOT / f"STAGE_C_{v}"


def stage_d_dir(version: str | None = None) -> Path:
    """Return the directory containing Stage D outputs for the given version."""
    v = version or DEFAULT_STAGE_D_VERSION
    return OUTPUTS_ROOT / f"STAGE_D_{v}"


def stage_e_dir(version: str | None = None) -> Path:
    """Return the directory containing Stage E outputs for the given version."""
    v = version or DEFAULT_STAGE_E_VERSION
    return OUTPUTS_ROOT / f"STAGE_E_{v}"


def get_config() -> dict:
    """Return the current config (paths, versions). For orchestration overrides."""
    return dict(_CONFIG)
