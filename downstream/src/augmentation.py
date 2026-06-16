from __future__ import annotations
import numpy as np
import pandas as pd

from config import TARGET, POS_VAL, RANDOM_STATE


def n_rows_for_ratio(n_pos: int, n_total: int, target_ratio: float) -> int:
    """How many synthetic rows to add so minority class reaches target_ratio."""
    if target_ratio <= 0:
        return 0
    current = n_pos / n_total if n_total else 0.0
    if target_ratio <= current:
        return 0
    s = (target_ratio * n_total - n_pos) / (1.0 - target_ratio)
    return int(max(0, round(s)))


def n_rows_for_multiplier(n_pos: int, multiplier: float) -> int:
    """Add multiplier * n_pos synthetic rows."""
    return int(max(0, round(multiplier * n_pos)))


def filter_hard_fn(
    pool_df: pd.DataFrame,
    model,
    X_pool: np.ndarray,
    threshold: float = 0.5,
    selection: str = "sorted",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Keep only rows the model predicts as non-diabetic (false negatives — hard cases).

    selection="sorted"  → rows ordered by prob descending within FN pool,
                          i.e. most uncertain (closest to threshold) first.
                          Caller takes top-k to get the hardest boundary cases.
    selection="random"  → original behaviour, unordered.

    Returns (filtered_df, X_filtered, probs_filtered)
    """
    probs = model.predict_proba(X_pool)[:, 1]
    fn_mask = probs < threshold

    df_fn   = pool_df[fn_mask].reset_index(drop=True)
    X_fn    = X_pool[fn_mask]
    p_fn    = probs[fn_mask]

    if selection == "sorted":
        # highest prob within FN = closest to decision boundary = most uncertain
        order   = np.argsort(p_fn)[::-1]
        df_fn   = df_fn.iloc[order].reset_index(drop=True)
        X_fn    = X_fn[order]
        p_fn    = p_fn[order]

    return df_fn, X_fn, p_fn


def augment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pool: np.ndarray,
    ratio: float,
    mode: str = "target_ratio",
    selection: str = "sorted",
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample from pool and concatenate with train.

    selection="sorted" → take top-s rows (most uncertain first, no randomness).
    selection="random" → random sample (original behaviour).
    """
    n_pos   = int(y_train.sum())
    n_total = len(y_train)

    if mode == "target_ratio":
        s = n_rows_for_ratio(n_pos, n_total, ratio)
    elif mode == "fraud_multiplier":
        s = n_rows_for_multiplier(n_pos, ratio)
    else:
        raise ValueError(f"Unknown augmentation mode: {mode}")

    if s == 0 or len(X_pool) == 0:
        return X_train, y_train

    if selection == "sorted":
        # pool already sorted by uncertainty desc — take the top s
        idx = np.arange(min(s, len(X_pool)))
        if s > len(X_pool):
            # repeat from the top if we need more than available
            reps = s // len(X_pool) + 1
            idx  = np.tile(np.arange(len(X_pool)), reps)[:s]
    else:
        rng     = np.random.default_rng(seed)
        replace = s > len(X_pool)
        idx     = rng.choice(len(X_pool), size=s, replace=replace)

    X_aug = np.vstack([X_train, X_pool[idx]])
    y_aug = np.concatenate([y_train, np.ones(s, dtype=y_train.dtype)])
    return X_aug, y_aug
