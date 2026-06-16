"""Logging + MLflow helpers — mirrors paysim-sarmad tracking."""
import contextlib
import json
import logging
import time
from pathlib import Path

import mlflow

from config import LOG_DIR, MLRUNS_DIR

log = logging.getLogger("diabetes")


def setup_fold_logging(fold: int) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"fold_{fold}.log"

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    file_h = logging.FileHandler(log_path, mode="a")
    file_h.setFormatter(fmt)

    stream_h = logging.StreamHandler()
    stream_h.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(file_h)
        root.addHandler(stream_h)
    else:
        root.handlers.clear()
        root.addHandler(file_h)
        root.addHandler(stream_h)


def init_mlflow(fold: int) -> None:
    import os
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(str(MLRUNS_DIR))
    mlflow.set_experiment(f"diabetes_synth_fold_{fold}")


@contextlib.contextmanager
def timed(label: str, fold: int, extra: dict | None = None):
    log.info(f"[fold={fold}] ▶ START {label}")
    t0 = time.time()
    yield
    elapsed = round(time.time() - t0, 1)
    info = {"elapsed_s": elapsed}
    if extra:
        info.update(extra)
    log.info(f"[fold={fold}] ✔ DONE  {label}  ({elapsed}s)")
