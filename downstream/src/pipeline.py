"""Per-fold pipeline — baseline and synthetic augmentation modes."""
from __future__ import annotations
import logging
import time
from pathlib import Path

import joblib
import numpy as np

from config import RunConfig, N_FOLDS, TARGET, POS_VAL
from data import load_fold, encode_features, get_xy, load_synthetic_pool
from models import get_classifier, CLASSIFIERS
from augmentation import filter_hard_fn, augment
from tuning import run_study
from evaluate import compute_metrics, save_fold_results, save_summary

log = logging.getLogger("diabetes.downstream")


def _setup_logging(run_name: str) -> None:
    from pathlib import Path
    log_dir = Path("/shared/diabetes-synth/logs/downstream")
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_dir / f"{run_name}.log", mode="a")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(sh)


def run_fold(fold: int, cfg: RunConfig) -> dict:
    t0 = time.time()
    log.info(f"[fold={fold}] {'='*50}")
    log.info(f"[fold={fold}] START — run={cfg.run_name}  mode={cfg.mode}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_df, test_df = load_fold(fold)
    train_df, encoders = encode_features(train_df, fit=True)
    test_df,  _        = encode_features(test_df, encoders=encoders, fit=False)

    X_train, y_train = get_xy(train_df)
    X_test,  y_test  = get_xy(test_df)
    log.info(f"[fold={fold}] train={len(X_train):,}  pos={int(y_train.sum()):,}  test={len(X_test):,}")

    # ── Synthetic pool ────────────────────────────────────────────────────────
    X_pool = None
    if cfg.mode == "synthetic":
        pool_df = load_synthetic_pool(cfg.augmentation.pool, fold)
        pool_df, _ = encode_features(pool_df, encoders=encoders, fit=False)
        X_pool_raw = pool_df.drop(columns=[TARGET]).values
        log.info(f"[fold={fold}] Synthetic pool ({cfg.augmentation.pool}): {len(X_pool_raw):,} rows")

    # ── Per-classifier loop ───────────────────────────────────────────────────
    fold_dir = cfg.models_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    fold_results = {}

    for clf_name in CLASSIFIERS:
        log.info(f"[fold={fold}] [{clf_name}] Tuning — {cfg.n_trials} trials")
        ct0 = time.time()

        pool_for_tuning = None
        if cfg.mode == "synthetic" and not cfg.augmentation.tune_ratio:
            # Sweep ratios — run separate study per ratio, pick best
            best_pr, best_ratio, best_params = -1, 0.0, {}
            for ratio in cfg.augmentation.ratios:
                # FN filter with a quick baseline fit when ratio > 0
                if ratio > 0 and cfg.augmentation.use_hard_fn:
                    base = get_classifier(clf_name, {}, gpu_id=cfg.gpu_id)
                    base.fit(X_train, y_train)
                    _, X_fn = filter_hard_fn(pool_df, base, X_pool_raw,
                                             threshold=cfg.augmentation.fn_threshold)
                    pool_for_study = X_fn if len(X_fn) > 0 else X_pool_raw
                else:
                    pool_for_study = X_pool_raw if cfg.mode == "synthetic" else None

                params, val_pr, _ = run_study(
                    clf_name, X_train, y_train, cfg,
                    X_pool=pool_for_study, ratio=ratio,
                )
                if val_pr > best_pr:
                    best_pr, best_ratio, best_params = val_pr, ratio, params

            log.info(f"[fold={fold}] [{clf_name}] Best ratio={best_ratio}  val_pr={best_pr:.4f}")

            # FN filter for final fit
            if best_ratio > 0 and cfg.augmentation.use_hard_fn:
                base = get_classifier(clf_name, best_params, gpu_id=cfg.gpu_id)
                base.fit(X_train, y_train)
                _, X_fn = filter_hard_fn(pool_df, base, X_pool_raw,
                                         threshold=cfg.augmentation.fn_threshold)
                X_pool_final = X_fn if len(X_fn) > 0 else X_pool_raw
            else:
                X_pool_final = X_pool_raw if cfg.mode == "synthetic" else None

            X_fit, y_fit = augment(
                X_train, y_train,
                X_pool_final if X_pool_final is not None else np.empty((0, X_train.shape[1])),
                best_ratio, mode=cfg.augmentation.mode,
                seed=cfg.random_state,
            )

        else:
            # Baseline or tune_ratio mode
            best_params, _, best_ratio = run_study(
                clf_name, X_train, y_train, cfg,
                X_pool=X_pool, ratio=0.0,
            )
            X_fit, y_fit = X_train, y_train

        # Final fit on full train (+ augmented if synthetic)
        model = get_classifier(clf_name, best_params, gpu_id=cfg.gpu_id)
        model.fit(X_fit, y_fit)

        # Evaluate
        probs = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, probs)
        elapsed = round(time.time() - ct0, 1)

        log.info(f"[fold={fold}] [{clf_name}] pr_auc={metrics['pr_auc']:.4f}  "
                 f"roc_auc={metrics['roc_auc']:.4f}  f1={metrics['f1']:.4f}  ({elapsed}s)")

        # Save model
        joblib.dump(model, fold_dir / f"{clf_name}.joblib")

        fold_results[clf_name] = {
            **metrics,
            "best_params": best_params,
            "best_ratio":  best_ratio if cfg.mode == "synthetic" else None,
            "elapsed_s":   elapsed,
        }

    total = round(time.time() - t0, 1)
    log.info(f"[fold={fold}] {'='*50}")
    log.info(f"[fold={fold}] FOLD {fold} DONE — {total:.1f}s ({total/60:.1f}min)")
    log.info(f"[fold={fold}] {'='*50}")

    save_fold_results(fold_results, cfg.results_dir / f"fold_{fold}.json")
    return fold_results


def run_all(cfg: RunConfig) -> None:
    _setup_logging(cfg.run_name)
    log.info(f"Run: {cfg.run_name}  mode: {cfg.mode}  folds: {N_FOLDS}")

    all_results = []
    for fold in range(N_FOLDS):
        result = run_fold(fold, cfg)
        all_results.append(result)

    summary = save_summary(all_results, cfg.results_dir / "summary.json")
    log.info(f"\n{'='*60}")
    log.info(f"RUN COMPLETE: {cfg.run_name}")
    for clf in summary:
        pr = summary[clf]["pr_auc"]
        log.info(f"  {clf}: pr_auc={pr['mean']:.4f} ± {pr['std']:.4f}")
    log.info(f"{'='*60}")
