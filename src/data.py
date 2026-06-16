"""Data loading and dataset construction for TabularARGN training."""
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DATA_DIR, TARGET, POS_VAL, STRAT_COL,
    DROP_COLS, HOLDOUT_RATIO, RANDOM_STATE,
    M2_NONFR_FRAC,
)


def load_fold(fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = DATA_DIR / "splits" / str(fold)
    train = pd.read_csv(base / "train.csv")
    test  = pd.read_csv(base / "test.csv")
    return train, test


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    if DROP_COLS:
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return df.reset_index(drop=True)


def split_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, holdout = train_test_split(
        df, test_size=HOLDOUT_RATIO,
        stratify=df[TARGET], random_state=RANDOM_STATE,
    )
    return train.reset_index(drop=True), holdout.reset_index(drop=True)


def build_m1(df: pd.DataFrame) -> pd.DataFrame:
    """Diabetic rows only."""
    return df[df[TARGET] == POS_VAL].reset_index(drop=True)


def build_m2(df: pd.DataFrame) -> pd.DataFrame:
    """Diabetic + 10% non-diabetic (stratified by gender)."""
    pos = df[df[TARGET] == POS_VAL]
    neg = df[df[TARGET] != POS_VAL]
    n_neg = max(1, int(len(neg) * M2_NONFR_FRAC))
    neg_sample = neg.groupby(STRAT_COL, group_keys=False).apply(
        lambda g: g.sample(frac=n_neg / len(neg), random_state=RANDOM_STATE)
    ).head(n_neg)
    return pd.concat([pos, neg_sample], ignore_index=True)


def build_m3(df: pd.DataFrame) -> pd.DataFrame:
    """Full training set — natural class ratio."""
    return df.reset_index(drop=True)


def pos_stats(df: pd.DataFrame, label: str) -> str:
    n_pos = int((df[TARGET] == POS_VAL).sum())
    pct = n_pos / len(df) * 100 if len(df) else 0
    return f"{label}: {len(df):,} rows | diabetic={n_pos:,} ({pct:.3f}%)"
