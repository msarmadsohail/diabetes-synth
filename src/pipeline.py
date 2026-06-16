"""Per-fold orchestrator for diabetes TabularARGN pipeline."""
import json
import sys
import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import mlflow
import pandas as pd

from config import RESULTS_DIR, SYNTH_DIR, MODELS_DIR
import tracking as T
from data import load_fold, preprocess, split_holdout, build_m1, build_m2, build_m3, pos_stats
from train_argn import train_all
from generate import generate_all, load_pools


def run_fold(
    fold: int,
    skip_training: bool = False,
    skip_generation: bool = False,
    stop_after_generation: bool = True,
    m1_only: bool = False,
) -> dict:
    T.setup_fold_logging(fold)
    T.init_mlflow(fold)

    fold_start = time.time()
    T.log.info(f"{'='*60}")
    T.log.info(f"FOLD {fold} — pipeline start")
    T.log.info(f"{'='*60}")

    with mlflow.start_run(run_name=f"fold_{fold}"):
        mlflow.log_param("fold", fold)

        # ── 1. Data ───────────────────────────────────────────────────────────
        with T.timed("load_data", fold):
            train_raw, test_raw = load_fold(fold)
            train = preprocess(train_raw)
            test  = preprocess(test_raw)
            T.log.info(pos_stats(train, "train"))
            T.log.info(pos_stats(test,  "test"))
            mlflow.log_params({
                "train_rows": len(train),
                "test_rows":  len(test),
                "train_diabetic": int((train["diabetes"] == 1).sum()),
            })

        # ── 2. Holdout ────────────────────────────────────────────────────────
        train_argn, holdout = split_holdout(train)
        T.log.info(pos_stats(train_argn, "train_argn (holdout reserved)"))
        T.log.info(pos_stats(holdout,    "holdout (reserved)"))

        # ── 3. ARGN datasets ──────────────────────────────────────────────────
        m1_data = build_m1(train)
        m2_data = build_m2(train)
        m3_data = build_m3(train)
        T.log.info(pos_stats(m1_data, "M1"))
        T.log.info(pos_stats(m2_data, "M2"))
        T.log.info(pos_stats(m3_data, "M3"))
        mlflow.log_params({"m1_rows": len(m1_data), "m2_rows": len(m2_data), "m3_rows": len(m3_data)})

        # ── 4. Training ───────────────────────────────────────────────────────
        ws_m1 = MODELS_DIR / f"fold_{fold}" / "m1"
        ws_m2 = MODELS_DIR / f"fold_{fold}" / "m2"
        ws_m3 = MODELS_DIR / f"fold_{fold}" / "m3"

        if not skip_training:
            with T.timed("train_all_argns", fold):
                ws_m1, ws_m2, ws_m3 = train_all(m1_data, m2_data, m3_data, fold, m1_only=m1_only)
        else:
            T.log.info(f"[fold={fold}] skip_training=True — using existing weights")

        # ── 5. Generation ─────────────────────────────────────────────────────
        if not skip_generation:
            with T.timed("generate_all_pools", fold):
                generate_all(ws_m1, ws_m2, ws_m3, fold, m1_only=m1_only)
        else:
            T.log.info(f"[fold={fold}] skip_generation=True — using existing pools")

        pool_m1, pool_m2, pool_m3 = load_pools(fold)
        mlflow.log_params({
            "pool_m1_size": len(pool_m1) if pool_m1 is not None else 0,
            "pool_m2_size": len(pool_m2) if pool_m2 is not None else 0,
            "pool_m3_size": len(pool_m3) if pool_m3 is not None else 0,
        })

        # ── 6. Export + results ───────────────────────────────────────────────
        total_elapsed = round(time.time() - fold_start, 1)

        export_dir = SYNTH_DIR / "for_classifier"
        export_dir.mkdir(parents=True, exist_ok=True)
        parts = [pool_m1]
        if pool_m2 is not None: parts.append(pool_m2)
        if pool_m3 is not None: parts.append(pool_m3)
        combined = pd.concat(parts, ignore_index=True)
        export_path = export_dir / f"split_{fold}.csv"
        combined.to_csv(export_path, index=False)
        T.log.info(f"[fold={fold}] Combined pool exported: {len(combined):,} rows → {export_path}")

        fold_result = {
            "fold": fold,
            "total_elapsed_s": total_elapsed,
            "export_diabetic_rows": len(combined),
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"fold_{fold}_results.json"
        out_path.write_text(json.dumps(fold_result, indent=2))
        mlflow.log_artifact(str(out_path))

        T.log.info(f"[fold={fold}] Results saved → {out_path}")
        T.log.info(f"[fold={fold}] {'='*50}")
        T.log.info(f"[fold={fold}] FOLD {fold} DONE — total {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
        T.log.info(f"[fold={fold}] {'='*50}")

    return fold_result
