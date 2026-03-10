from pathlib import Path


PROJECT_ROOT = Path(__file__).parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

# Local NER model cache: run download_ner_model.py to pre-download models here.
# Models are ~250MB (v1) or ~1.3GB (v2). Falls back to Hugging Face cache if not found.
NER_MODELS_DIR = PROJECT_ROOT / "models" / "ner"


def ner_model_local_path(model_id: str) -> Path | None:
    """Return local path if model is cached in NER_MODELS_DIR, else None."""
    folder = model_id.replace("/", "__")
    path = NER_MODELS_DIR / folder
    return path if path.is_dir() else None

# Default stage versions (can be adjusted in one place)
DEFAULT_STAGE_A_VERSION = "v1"
DEFAULT_STAGE_B_VERSION = "v1"
DEFAULT_STAGE_C_VERSION = "v1"
DEFAULT_STAGE_D_VERSION = "v1"
DEFAULT_STAGE_E_VERSION = "v1"


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

