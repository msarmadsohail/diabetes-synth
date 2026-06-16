"""Synthetic diabetic sample generation from trained ARGN models."""
import threading
import warnings
from pathlib import Path

import pandas as pd

from config import (
    SYNTH_DIR, TARGET, POS_VAL,
    M1_POOL_TARGET, M2_POOL_TARGET,
    M1_GEN_BATCH, M2_GEN_BATCH, M3_GEN_BATCH,
    GPU_M1, GPU_M2, GPU_M3,
)
import tracking as T


def _filter_pos(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[TARGET] == POS_VAL].reset_index(drop=True)


def _generate_free_batched(argn, target: int, batch: int, label: str, fold: int) -> pd.DataFrame:
    collected, total, rounds = [], 0, 0
    while total < target:
        chunk = argn.sample(n_samples=batch)
        pos_chunk = _filter_pos(chunk)
        collected.append(pos_chunk)
        total += len(pos_chunk)
        rounds += 1
        T.log.info(f"[fold={fold}] {label} pool: {total:,}/{target:,} positive rows (round {rounds})")
    return pd.concat(collected, ignore_index=True).head(target)


def _run_m1(ws_m1: Path, fold: int, out_path: Path, results: dict) -> None:
    warnings.filterwarnings("ignore")
    from train_argn import load_argn
    with T.timed("generate_m1", fold):
        m1 = load_argn(ws_m1, device=f"cuda:{GPU_M1}")
        # M1 is 100% diabetic — single batch, no filtering loop needed
        pool = m1.sample(n_samples=M1_GEN_BATCH)
        pool.to_csv(out_path, index=False)
        T.log.info(f"[fold={fold}] M1 pool saved: {len(pool):,} rows → {out_path}")
        results["m1"] = pool


def _run_m2(ws_m2: Path, fold: int, out_path: Path, results: dict) -> None:
    warnings.filterwarnings("ignore")
    from train_argn import load_argn
    with T.timed("generate_m2", fold):
        m2 = load_argn(ws_m2, device=f"cuda:{GPU_M2}")
        pool = _generate_free_batched(m2, M2_POOL_TARGET, M2_GEN_BATCH, "M2", fold)
        pool.to_csv(out_path, index=False)
        T.log.info(f"[fold={fold}] M2 pool saved: {len(pool):,} rows → {out_path}")
        results["m2"] = pool


def _run_m3(ws_m3: Path, fold: int, out_path: Path, results: dict) -> None:
    warnings.filterwarnings("ignore")
    from train_argn import load_argn
    with T.timed("generate_m3", fold):
        m3 = load_argn(ws_m3, device=f"cuda:{GPU_M3}")
        raw = m3.sample(n_samples=M3_GEN_BATCH)
        pool = _filter_pos(raw)
        pool.to_csv(out_path, index=False)
        T.log.info(f"[fold={fold}] M3 pool saved: {len(pool):,} rows (natural yield) → {out_path}")
        results["m3"] = pool


def generate_all(ws_m1: Path, ws_m2: Path, ws_m3: Path, fold: int, m1_only: bool = False) -> tuple[Path, Path, Path]:
    out_dir = SYNTH_DIR / f"fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    p_m1 = out_dir / "pool_m1.csv"
    p_m2 = out_dir / "pool_m2.csv"
    p_m3 = out_dir / "pool_m3.csv"
    results = {}

    if m1_only:
        T.log.info(f"[fold={fold}] Generating M1 only on cuda:{GPU_M1}")
        _run_m1(ws_m1, fold, p_m1, results)
        return p_m1, p_m2, p_m3

    # M1 + M2 in parallel, M3 after M2 on same GPU
    t1 = threading.Thread(target=_run_m1, args=(ws_m1, fold, p_m1, results))
    t2 = threading.Thread(target=_run_m2, args=(ws_m2, fold, p_m2, results))
    t1.start(); t2.start()
    t1.join(); t2.join()
    _run_m3(ws_m3, fold, p_m3, results)

    return p_m1, p_m2, p_m3


def load_pools(fold: int) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    out_dir = SYNTH_DIR / f"fold_{fold}"
    def _load(name):
        p = out_dir / name
        return pd.read_csv(p) if p.exists() else None
    return _load("pool_m1.csv"), _load("pool_m2.csv"), _load("pool_m3.csv")
