"""Shared constants and paths for the thermal-conduction surrogate project."""
from pathlib import Path

# ---- Simulation -----------------------------------------------------------
GRID_SIZE = 100          # domain is GRID_SIZE x GRID_SIZE cells
N_ITERS = 200            # time steps per simulation (the CNN predicts the state after this many)
DIFFUSION_NUMBER = 0.2   # r = alpha * dt / dx^2. Explicit scheme is stable only for r <= 0.25
SOURCE_TEMP = 1.0        # source blocks are held at this temperature

# ---- Paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CSV_DIR = ROOT / "data" / "csv"
H5_PATH = ROOT / "data" / "dataset.h5"
CKPT_PATH = ROOT / "checkpoints" / "best.pt"
RESULTS_DIR = ROOT / "results"

# ---- Train / val / test split (shared so plot_results.py sees the same test set) ----
SPLIT_SEED = 0
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1           # the remainder (10%) is the test set
