#!/usr/bin/env python
"""
Parallel fold runner for diabetes-synth — fully utilizes all 4 GPUs.

GPU layout per fold pair:
  Fold A → M1+M2 on GPU 0, M3 on GPU 1
  Fold B → M1+M2 on GPU 2, M3 on GPU 3

Schedule:
  Round 1: folds 0 + 1  (GPUs 0,1 | 2,3)
  Round 2: folds 2 + 3  (GPUs 0,1 | 2,3)
  Round 3: fold  4      (GPUs 0,1)
"""

import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, "/shared/diabetes-synth/src")


def _run_fold_worker(fold: int, gpu_primary: int, gpu_secondary: int) -> None:
    """
    Each fold gets 2 physical GPUs:
      gpu_primary   → M1 + M2 (sequential)
      gpu_secondary → M3 (subprocess)
    CUDA_VISIBLE_DEVICES remaps them to cuda:0 and cuda:1 inside this process.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_primary},{gpu_secondary}"
    os.environ["ARGN_GPU_M1"] = "0"
    os.environ["ARGN_GPU_M2"] = "0"
    os.environ["ARGN_GPU_M3"] = "1"

    import warnings
    warnings.filterwarnings("ignore")

    import tracking as T
    T.setup_fold_logging(fold)
    T.log.info(f"[fold={fold}] process started — physical GPUs {gpu_primary},{gpu_secondary} → cuda:0,cuda:1")

    from pipeline import run_fold
    run_fold(fold, skip_training=False, skip_generation=False,
             stop_after_generation=True, m1_only=False)


def _run_pair(fold_a: int, fold_b: int | None) -> None:
    ctx = mp.get_context("spawn")
    procs = []

    # Fold A: physical GPUs 0,1
    p_a = ctx.Process(
        target=_run_fold_worker,
        args=(fold_a, 0, 1),
        name=f"fold-{fold_a}",
    )
    p_a.start()
    procs.append((fold_a, p_a))

    if fold_b is not None:
        # Fold B: physical GPUs 2,3
        p_b = ctx.Process(
            target=_run_fold_worker,
            args=(fold_b, 2, 3),
            name=f"fold-{fold_b}",
        )
        p_b.start()
        procs.append((fold_b, p_b))

    for fold, p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Fold {fold} worker exited with code {p.exitcode}")
        print(f"[run_parallel] fold {fold} complete", flush=True)


def main() -> None:
    rounds = [
        (0, 1),
        (2, 3),
        (4, None),
    ]

    total_start = time.time()
    for fold_a, fold_b in rounds:
        label = f"folds {fold_a}+{fold_b}" if fold_b is not None else f"fold {fold_a}"
        print(f"\n{'='*60}", flush=True)
        print(f"[run_parallel] Starting {label}", flush=True)
        print(f"{'='*60}", flush=True)
        _run_pair(fold_a, fold_b)
        print(f"[run_parallel] {label} done", flush=True)

    elapsed = time.time() - total_start
    print(f"\n[run_parallel] All folds complete in {elapsed/3600:.2f}h", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
