"""Train M1, M2, M3 TabularARGN models per fold."""
import multiprocessing as mp
import warnings
from pathlib import Path

import pandas as pd

from config import (
    MODELS_DIR,
    M1_MODEL, M2_MODEL, M3_MODEL,
    M1_MAX_EPOCHS, M2_MAX_EPOCHS, M3_MAX_EPOCHS,
    GPU_M1, GPU_M3,
)
import tracking as T

GPU_M1_M2 = GPU_M1


def _worker(data_path: str, workspace_dir: str, max_epochs: int, model: str, device: str) -> None:
    warnings.filterwarnings("ignore")
    import pandas as pd
    from mostlyai.engine import TabularARGN

    df = pd.read_parquet(data_path)
    argn = TabularARGN(
        model=model,
        max_epochs=max_epochs,
        workspace_dir=workspace_dir,
        device=device,
        verbose=1,
    )
    argn.fit(df)


def train_all(
    m1_data: pd.DataFrame,
    m2_data: pd.DataFrame,
    m3_data: pd.DataFrame,
    fold: int,
    m1_only: bool = False,
) -> tuple[Path, Path, Path]:
    warnings.filterwarnings("ignore")
    from mostlyai.engine import TabularARGN

    fold_dir = MODELS_DIR / f"fold_{fold}"
    ws_m1 = fold_dir / "m1"
    ws_m2 = fold_dir / "m2"
    ws_m3 = fold_dir / "m3"
    ws_m1.mkdir(parents=True, exist_ok=True)

    with T.timed("train_m1", fold, {"rows": len(m1_data), "diabetic": int((m1_data["diabetes"]==1).sum())}):
        argn_m1 = TabularARGN(
            model=M1_MODEL,
            max_epochs=M1_MAX_EPOCHS,
            workspace_dir=str(ws_m1),
            device=f"cuda:{GPU_M1_M2}",
            verbose=1,
        )
        argn_m1.fit(m1_data)

    if m1_only:
        T.log.info(f"[fold={fold}] m1_only=True — skipping M2 and M3")
        return ws_m1, ws_m2, ws_m3

    ws_m2.mkdir(parents=True, exist_ok=True)
    with T.timed("train_m2", fold, {"rows": len(m2_data)}):
        argn_m2 = TabularARGN(
            model=M2_MODEL,
            max_epochs=M2_MAX_EPOCHS,
            workspace_dir=str(ws_m2),
            device=f"cuda:{GPU_M1_M2}",
            verbose=1,
        )
        argn_m2.fit(m2_data)

    ws_m3.mkdir(parents=True, exist_ok=True)
    tmp_m3 = fold_dir / "_m3_tmp.parquet"
    m3_data.to_parquet(tmp_m3, index=False)

    T.log.info(f"[fold={fold}] Launching M3 on cuda:{GPU_M3} (subprocess)")
    ctx = mp.get_context("spawn")
    m3_proc = ctx.Process(
        target=_worker,
        args=(str(tmp_m3), str(ws_m3), M3_MAX_EPOCHS, M3_MODEL, f"cuda:{GPU_M3}"),
        name=f"argn-m3-fold{fold}",
    )
    m3_proc.start()
    T.log.info(f"[fold={fold}] Waiting for M3 subprocess (pid={m3_proc.pid}) …")
    m3_proc.join()
    tmp_m3.unlink(missing_ok=True)

    if m3_proc.exitcode != 0:
        raise RuntimeError(f"[fold={fold}] M3 subprocess failed (exit={m3_proc.exitcode})")
    T.log.info(f"[fold={fold}] M3 training done")

    return ws_m1, ws_m2, ws_m3


def load_argn(workspace_dir: Path, device: str) -> "TabularARGN":
    from mostlyai.engine import TabularARGN
    argn = TabularARGN(workspace_dir=str(workspace_dir), device=device, verbose=0)
    argn._fitted = True
    argn._workspace_path = workspace_dir
    argn.workspace_dir = str(workspace_dir)
    return argn
