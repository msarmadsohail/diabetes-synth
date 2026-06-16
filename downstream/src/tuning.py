from __future__ import annotations
import numpy as np
import optuna
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

from config import RunConfig, TARGET, POS_VAL
from models import get_classifier, suggest_params
from augmentation import augment


def run_study(
    clf_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: RunConfig,
    X_pool: np.ndarray | None = None,
    ratio: float = 0.0,
) -> tuple[dict, float]:
    """
    Run Optuna study for one classifier.
    Returns (best_params, best_val_pr_auc).
    """
    inner_cv = StratifiedKFold(
        n_splits=cfg.inner_folds, shuffle=True, random_state=cfg.random_state
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, clf_name)

        if cfg.augmentation.tune_ratio and X_pool is not None:
            trial_ratio = trial.suggest_categorical(
                "aug_ratio", cfg.augmentation.ratios
            )
        else:
            trial_ratio = ratio

        scores = []
        for tr_idx, val_idx in inner_cv.split(X_train, y_train):
            Xtr, ytr = X_train[tr_idx], y_train[tr_idx]
            Xval, yval = X_train[val_idx], y_train[val_idx]

            if X_pool is not None and trial_ratio > 0:
                Xtr, ytr = augment(
                    Xtr, ytr, X_pool, trial_ratio,
                    mode=cfg.augmentation.mode,
                    seed=cfg.random_state + trial.number,
                )

            model = get_classifier(clf_name, params, gpu_id=cfg.gpu_id)
            model.fit(Xtr, ytr)
            prob = model.predict_proba(Xval)[:, 1]
            scores.append(average_precision_score(yval, prob))

        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=False)

    best_params = study.best_params.copy()
    best_ratio  = best_params.pop("aug_ratio", ratio)
    return best_params, study.best_value, best_ratio
