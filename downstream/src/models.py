from __future__ import annotations
import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


CLASSIFIERS = ["xgboost"]


def get_classifier(name: str, params: dict, gpu_id: int = 0):
    if name == "xgboost":
        return XGBClassifier(
            **params,
            tree_method="hist",
            device=f"cuda:{gpu_id}",
            eval_metric="aucpr",
            use_label_encoder=False,
            verbosity=0,
            random_state=42,
        )
    elif name == "lightgbm":
        return LGBMClassifier(
            **params,
            device="gpu",
            gpu_device_id=gpu_id,
            verbose=-1,
            random_state=42,
        )
    elif name == "catboost":
        return CatBoostClassifier(
            **params,
            task_type="GPU",
            devices=str(gpu_id),
            eval_metric="AUC",
            verbose=0,
            random_state=42,
        )
    raise ValueError(f"Unknown classifier: {name}")


def suggest_params(trial: optuna.Trial, name: str) -> dict:
    if name == "xgboost":
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 9),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0),
        }
    elif name == "lightgbm":
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 9),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "is_unbalance":     True,
        }
    elif name == "catboost":
        return {
            "iterations":       trial.suggest_int("iterations", 100, 800),
            "depth":            trial.suggest_int("depth", 3, 9),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "l2_leaf_reg":      trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "auto_class_weights": "Balanced",
        }
    raise ValueError(f"Unknown classifier: {name}")
