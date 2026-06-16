import os
from pathlib import Path

BASE_DIR    = Path("/shared/diabetes-synth")
DATA_DIR    = BASE_DIR / "data"
MODELS_DIR  = BASE_DIR / "models"
SYNTH_DIR   = BASE_DIR / "synthetic"
RESULTS_DIR = BASE_DIR / "results"
MLRUNS_DIR  = BASE_DIR / "mlruns"
LOG_DIR     = BASE_DIR / "logs"

N_FOLDS = 5

# columns
TARGET    = "diabetes"
POS_VAL   = 1
STRAT_COL = "gender"          # used for M2 stratified non-diabetic sampling

# No DROP_COLS — every column is clinically meaningful
DROP_COLS = []

# holdout split inside each fold's train.csv
HOLDOUT_RATIO = 0.20
RANDOM_STATE  = 42

# ARGN training
M1_MAX_EPOCHS = 50             # diabetic-only; small dataset, early stop fires fast
M2_MAX_EPOCHS = 100
M3_MAX_EPOCHS = 100
M2_NONFR_FRAC = 0.10           # 10% non-diabetic rows for M2

# generation
M1_POOL_TARGET = 50_000        # target positive rows — M1 batched free generation
M2_POOL_TARGET = 150_000
M1_GEN_BATCH   = 100_000       # M1 is 100% positive so one pass usually enough
M2_GEN_BATCH   = 50_000
M3_GEN_BATCH   = 5_000_000    # single free pass for M3 — natural yield

POOL_PER_MODEL = 150_000       # legacy alias

# GPU assignment — overridable via env vars for parallel fold execution
GPU_M1 = int(os.environ.get("ARGN_GPU_M1", 0))
GPU_M2 = int(os.environ.get("ARGN_GPU_M2", 1))
GPU_M3 = int(os.environ.get("ARGN_GPU_M3", 2))

# Optuna
OPTUNA_TRIALS = 50
AUG_TARGETS   = [0.10, 0.15, 0.20, 0.30]   # target diabetic % after augmentation

# metrics
F_BETA = 2
