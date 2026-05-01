import os
import torch
from pathlib import Path

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Project root (repo folder containing this file)
PROJECT_ROOT = Path(__file__).resolve().parent

# External FaceForensics++ dataset path.
# Prefer the environment variable `FFPP_C23_DIR` if set, otherwise use a
# repository-local default folder `FaceForensics++_C23` under the project root.
FFPP_C23_DIR = Path(os.environ.get("FFPP_C23_DIR", PROJECT_ROOT / "FaceForensics++_C23"))
if not FFPP_C23_DIR.exists():
	print(f"Warning: FFPP_C23_DIR does not exist: {FFPP_C23_DIR}")
	print("Set the environment variable FFPP_C23_DIR to the dataset location if needed.")

# Data folders (relative to project root). Allow overriding catalog CSV via env var.
DATA_DIR = Path(os.environ.get("MEDIA_VERIF_DATA_DIR", PROJECT_ROOT / "data"))
FRAMES_DIR = DATA_DIR / "frames"
FACES_DIR = DATA_DIR / "faces"
CATALOG_CSV = Path(os.environ.get("CATALOG_CSV", DATA_DIR / "catalog.csv"))

# Image / training settings
# NOTE: The values below are LOCAL DEV/DEBUG overrides.
# Production training was conducted on a separate machine (RTX A4000)
# with: BATCH_SIZE=8, PGD_STEPS=5, PGD_EPSILON=8/255, EPOCHS=11.
# See the project report for production training configuration.
IMAGE_SIZE = 224

# BATCH_SIZE = 2
BATCH_SIZE = 2

LEARNING_RATE = 1e-4

# EPOCHS = 10
EPOCHS = 3

SEED = 42

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

PGD_EPSILON = 4 / 255
PGD_ALPHA = PGD_EPSILON / 2

# PGD_STEPS = 3
PGD_STEPS = 1

RUNS_DIR = PROJECT_ROOT / "runs"
CHECKPOINTS_DIR = RUNS_DIR / "checkpoints"

# Decision threshold for binary classification
# 0.65 reduces false positives; tuned for forensic applications
DECISION_THRESHOLD = 0.65
