"""
Central configuration for the Fetal NT Analysis project.

All paths are read from environment variables so the code runs on any machine.
Copy .env.example to .env and edit it, or export the variables in your shell:

    export FETUS_DATA_ROOT="/path/to/Dataset for Fetus Framework"
"""

import os
from pathlib import Path

# --- Project layout -----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# --- Dataset paths ------------------------------------------------------------
# Override with the FETUS_DATA_ROOT environment variable.

DATA_ROOT = Path(
    os.environ.get("FETUS_DATA_ROOT", PROJECT_ROOT / "data")
)

# Set2 contains the Standard / Non-standard folders used for training.
TRAIN_DIR = Path(
    os.environ.get(
        "FETUS_TRAIN_DIR",
        DATA_ROOT / "Set2-Training-Validation Sets ANN Scoring system",
    )
)

# Held-out test set with the same Standard / Non-standard folder structure.
TEST_DIR = Path(os.environ.get("FETUS_TEST_DIR", DATA_ROOT / "test"))

# Bounding-box annotations for the NT region.
ANNOTATION_FILE = Path(
    os.environ.get("FETUS_ANNOTATIONS", DATA_ROOT / "ObjectDetection.xlsx")
)

# --- Model hyper-parameters ---------------------------------------------------

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

MODEL_PATH = MODELS_DIR / "model_standard_vs_nonstandard.h5"

# --- Clinical thresholds ------------------------------------------------------

# NT height above this many pixels is flagged for follow-up.
# NOTE: this is a pixel threshold, not millimetres. Converting to mm requires
# the pixel-spacing metadata from the original DICOM, which the public dataset
# does not provide.
NT_PIXEL_THRESHOLD = 30

# Sigmoid output at or above this value is classified as "Standard".
CLASSIFICATION_THRESHOLD = 0.5

CLASS_NAMES = ["Non-standard", "Standard"]


def describe() -> str:
    """Return a readable summary of the active configuration."""
    return "\n".join(
        [
            f"Project root  : {PROJECT_ROOT}",
            f"Data root     : {DATA_ROOT}",
            f"Train dir     : {TRAIN_DIR}",
            f"Test dir      : {TEST_DIR}",
            f"Annotations   : {ANNOTATION_FILE}",
            f"Model path    : {MODEL_PATH}",
            f"Image size    : {IMG_SIZE}x{IMG_SIZE}",
        ]
    )


if __name__ == "__main__":
    print(describe())
