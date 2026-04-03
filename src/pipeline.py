import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.utils import DEFAULT_RF_PARAMS, SPECTRAL_BASE_FEATURES, resolve_feature_columns, save_model_artifact


class DataValidationError(Exception):
    def __str__(self):
        return "Dataset failed validation checks. Please check logs"


class TrainingPipeline:
    def __init__(
        self,
        train_path: str = "data/training.csv",
        test_path: str = "data/testing.csv",
        target_col: str = "class",
        model_path: str = "artifacts/rf_model.joblib",
        random_state: int = 42,
        scoring: str = "f1_weighted",
        n_splits: int = 5,
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.target_col = target_col
        self.model_path = model_path
        self.random_state = random_state
        self.scoring = scoring
        self.n_splits = n_splits

    def load_data(self, data_path: str) -> pd.DataFrame:
        df = pd.read_csv(data_path)
        logging.info("Loaded dataset from %s", data_path)
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        processed_df = df.copy()
        if self.target_col in processed_df.columns:
            processed_df[self.target_col] = processed_df[self.target_col].astype(str).str.strip()
        return processed_df

    def validate_data(self, df: pd.DataFrame) -> None:
        if df.isnull().any().sum() != 0:
            logging.info("Missing values found in dataset.")
            logging.info(df.isnull().sum())
            raise DataValidationError()
        logging.info("Data validation successful.")

    def feature_selection(self, df: pd.DataFrame) -> list[str]:
        selected_columns = resolve_feature_columns(
            SPECTRAL_BASE_FEATURES,
            df.drop(columns=[self.target_col]).columns,
        )
        logging.info("Selected spectral columns: %s", selected_columns)
        return selected_columns

    def model_selection(self, df: pd.DataFrame, selected_features: list[str]) -> dict:
        if not selected_features:
            raise ValueError("selected_features is empty. Model selection cannot proceed.")

        X = df[selected_features]
        y = df[self.target_col]
        model = RandomForestClassifier(**DEFAULT_RF_PARAMS, random_state=self.random_state)
        cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring=self.scoring,
            n_jobs=1,
        )

        model.fit(X, y)

        results = {
            "best_params": dict(DEFAULT_RF_PARAMS),
            "best_score": float(scores.mean()),
            "cv_scores": scores.tolist(),
            "best_model": model,
        }

        logging.info("Model selection used one hyperparameter combination: %s", results["best_params"])
        logging.info("Cross-validation scores: %s", results["cv_scores"])
        logging.info("Mean cross-validation score: %.6f", results["best_score"])
        return results

    def evaluate_model(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        selected_features: list[str],
        model_selection_info: dict,
    ) -> dict:
        model = RandomForestClassifier(
            **model_selection_info["best_params"],
            random_state=self.random_state,
        )

        X_train = train_df[selected_features]
        y_train = train_df[self.target_col]
        X_test = test_df[selected_features]
        y_test = test_df[self.target_col]

        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        metrics = {
            "trained_model": model,
            "train_accuracy": accuracy_score(y_train, train_predictions),
            "train_f1_weighted": f1_score(y_train, train_predictions, average="weighted"),
            "train_report": classification_report(y_train, train_predictions),
            "test_accuracy": accuracy_score(y_test, test_predictions),
            "test_f1_weighted": f1_score(y_test, test_predictions, average="weighted"),
            "test_report": classification_report(y_test, test_predictions),
        }

        logging.info("Final model params: %s", model_selection_info["best_params"])
        logging.info("Training evaluation:")
        logging.info("Train accuracy: %.6f", metrics["train_accuracy"])
        logging.info("Train weighted F1: %.6f", metrics["train_f1_weighted"])
        logging.info("\n%s", metrics["train_report"])
        logging.info("Testing evaluation:")
        logging.info("Test accuracy: %.6f", metrics["test_accuracy"])
        logging.info("Test weighted F1: %.6f", metrics["test_f1_weighted"])
        logging.info("\n%s", metrics["test_report"])

        return metrics

    def save_model(self, model) -> str:
        return save_model_artifact(model, self.model_path)

    def run(self) -> dict:
        train_df = self.preprocess_data(self.load_data(self.train_path))
        test_df = self.preprocess_data(self.load_data(self.test_path))

        self.validate_data(train_df)
        self.validate_data(test_df)

        selected_features = self.feature_selection(train_df)
        model_selection_info = self.model_selection(train_df, selected_features)
        evaluation_info = self.evaluate_model(
            train_df,
            test_df,
            selected_features,
            model_selection_info,
        )
        model_path = self.save_model(evaluation_info["trained_model"])

        return {
            "selected_features": selected_features,
            "model_selection_info": model_selection_info,
            "evaluation_info": evaluation_info,
            "model_path": model_path,
        }
