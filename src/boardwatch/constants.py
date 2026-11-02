import os
from pathlib import Path


def _resolve_project_root(
    module_file: str | Path, current_directory: str | Path, override: str | None = None
) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    source_root = Path(module_file).resolve().parents[2]
    if (source_root / "pyproject.toml").is_file():
        return source_root
    return Path(current_directory).resolve()


PROJECT_ROOT = _resolve_project_root(__file__, Path.cwd(), os.environ.get("BOARDWATCH_HOME"))
DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw" / "tic-tac-toe-board-dataset"
PROCESSED_ROOT = DATA_ROOT / "processed"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
CHECKPOINT_PATH = ARTIFACT_ROOT / "boardwatch_fasterrcnn.pt"
REPORT_ROOT = PROJECT_ROOT / "reports"

DATASET_REPO = "zoncoen/tic-tac-toe-board-dataset"
DATASET_REVISION = "8d3782b5a9992abca0e5611e9473eac7a1f98822"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_REPO}"
DATASET_IMAGE_COUNT = 59
DEFAULT_SCORE_THRESHOLD = 0.55

# Source YOLO IDs 0..2 are shifted by one for TorchVision's background=0 convention.
CLASS_NAMES = ("empty_cell", "white_circle_cell", "black_cross_cell")
NUM_CLASSES = len(CLASS_NAMES) + 1
