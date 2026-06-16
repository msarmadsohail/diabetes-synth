from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from config import DATA_DIR, SYNTH_DIR, TARGET, POS_VAL

CATEGORICAL_COLS = ["gender", "smoking_history"]


def load_fold(fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = DATA_DIR / "splits" / str(fold)
    return pd.read_csv(base / "train.csv"), pd.read_csv(base / "test.csv")


def encode_features(df: pd.DataFrame, encoders: dict | None = None, fit: bool = False):
    """Label-encode categoricals. If fit=True, fit and return encoders. Else use provided."""
    df = df.copy()
    if encoders is None:
        encoders = {}
    for col in CATEGORICAL_COLS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # handle unseen labels gracefully
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col])
    return df, encoders


def get_xy(df: pd.DataFrame) -> tuple:
    X = df.drop(columns=[TARGET]).values
    y = df[TARGET].values
    return X, y


def load_synthetic_pool(pool: str, fold: int) -> pd.DataFrame:
    path = SYNTH_DIR / f"fold_{fold}" / f"pool_{pool}.csv"
    df = pd.read_csv(path)
    return df[df[TARGET] == POS_VAL].reset_index(drop=True)
