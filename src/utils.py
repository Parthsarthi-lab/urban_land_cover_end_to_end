import logging
import os

import joblib


SPECTRAL_BASE_FEATURES = [
    "Bright",
    "Mean_G",
    "Mean_R",
    "Mean_NIR",
    "NDVI",
    "SD_G",
    "SD_R",
    "SD_NIR",
]


DEFAULT_RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "max_features": 9,
    "min_samples_split": 4,
    "n_jobs": 1,
}


def resolve_feature_columns(base_features: list[str], all_columns) -> list[str]:
    return [
        column
        for column in all_columns
        if any(column == base or column.startswith(f"{base}_") for base in base_features)
    ]


def save_model_artifact(model, save_path: str) -> str:
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, save_path)
    logging.info("Saved trained model to %s", save_path)
    return save_path
