from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

BASE_DIR     = Path("/shared/diabetes-synth")
DATA_DIR     = BASE_DIR / "data"
SYNTH_DIR    = BASE_DIR / "synthetic"
RESULTS_DIR  = BASE_DIR / "results" / "downstream"
MODELS_DIR   = BASE_DIR / "models"  / "downstream"

N_FOLDS      = 5
TARGET       = "diabetes"
POS_VAL      = 1
RANDOM_STATE = 42


@dataclass
class AugmentationConfig:
    enabled: bool = False
    pool: str = "m1"                          # m1 | m2 | m3
    mode: str = "target_ratio"                # target_ratio | fraud_multiplier
    ratios: list = field(default_factory=lambda: [0.0, 0.10, 0.15, 0.20, 0.30, 0.50])
    tune_ratio: bool = False
    use_hard_fn: bool = True                  # filter pool to FN before augmenting
    fn_threshold: float = 0.5
    fn_selection: str = "sorted"              # sorted | random — how to pick from FN pool


@dataclass
class RunConfig:
    run_name: str                             # e.g. "01_baseline", "02_synthetic_m1"
    mode: str                                 # "baseline" | "synthetic"
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    n_trials: int = 50
    inner_folds: int = 3
    gpu_id: int = 0
    random_state: int = RANDOM_STATE

    @property
    def results_dir(self) -> Path:
        return RESULTS_DIR / self.run_name

    @property
    def models_dir(self) -> Path:
        return MODELS_DIR / self.run_name


def load_config(path: str) -> RunConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    aug_d = d.pop("augmentation", {})
    aug = AugmentationConfig(**aug_d)
    return RunConfig(augmentation=aug, **d)
