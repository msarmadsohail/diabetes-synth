#!/usr/bin/env python
"""
Run all 5 folds in parallel across 4 GPUs.
Usage: python run_parallel.py configs/baseline.yaml

Fold → GPU mapping:
  fold 0 → GPU 0
  fold 1 → GPU 1
  fold 2 → GPU 2
  fold 3 → GPU 3
  fold 4 → GPU 0  (shares with fold 0, starts after fold 0 finishes)
"""
import multiprocessing as mp
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def _worker(config_path: str, fold: int, gpu_id: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import warnings
    warnings.filterwarnings("ignore")

    from config import load_config
    from pipeline import run_fold, _setup_logging

    cfg = load_config(config_path)
    cfg.gpu_id = 0  # remapped via CUDA_VISIBLE_DEVICES
    _setup_logging(cfg.run_name)
    run_fold(fold, cfg)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_parallel.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]

    # Load config just for the run_name and summary
    from config import load_config
    from evaluate import save_summary
    import json

    cfg = load_config(config_path)

    # All 5 folds → 4 GPUs: folds 0-3 simultaneously, fold 4 after
    rounds = [
        [0, 1, 2, 3],
        [4],
    ]

    all_results = [None] * 5
    ctx = mp.get_context("spawn")

    for round_folds in rounds:
        procs = []
        for fold in round_folds:
            gpu_id = fold % 4
            p = ctx.Process(
                target=_worker,
                args=(config_path, fold, gpu_id),
                name=f"fold-{fold}",
            )
            p.start()
            procs.append((fold, p))
            print(f"[run_parallel] fold {fold} started on GPU {gpu_id}", flush=True)

        for fold, p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Fold {fold} exited with code {p.exitcode}")
            print(f"[run_parallel] fold {fold} complete", flush=True)

    # Aggregate summary from per-fold results
    results_dir = cfg.results_dir
    all_fold_results = []
    for fold in range(5):
        fold_path = results_dir / f"fold_{fold}.json"
        if fold_path.exists():
            all_fold_results.append(json.loads(fold_path.read_text()))

    if all_fold_results:
        summary = save_summary(all_fold_results, results_dir / "summary.json")
        print(f"\n{'='*60}", flush=True)
        print(f"RUN COMPLETE: {cfg.run_name}", flush=True)
        for clf in summary:
            pr = summary[clf]["pr_auc"]
            print(f"  {clf}: pr_auc={pr['mean']:.4f} ± {pr['std']:.4f}", flush=True)
        print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
