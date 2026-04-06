from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


class AnalysisError(Exception):
    """Raised when the uploaded dataset cannot be processed for classification."""


@dataclass
class ModelResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    confusion_matrix: list[list[int]]
    report: str


def _validate_inputs(
    dataframe: pd.DataFrame,
    target_column: str | None,
    test_size: float,
    knn_neighbors: int,
) -> str:
    if dataframe.empty:
        raise AnalysisError("Uploaded CSV is empty.")

    if dataframe.shape[1] < 2:
        raise AnalysisError("Dataset must contain at least one feature column and one target.")

    if not 0.1 <= test_size <= 0.5:
        raise AnalysisError("Test size must be between 0.1 and 0.5.")

    if knn_neighbors <= 0:
        raise AnalysisError("KNN neighbors must be a positive integer.")

    resolved_target = target_column or dataframe.columns[-1]
    if resolved_target not in dataframe.columns:
        raise AnalysisError(
            f"Target column '{resolved_target}' was not found in uploaded CSV."
        )

    return resolved_target


def _compute_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    is_binary: bool,
) -> ModelResult:
    average = "binary" if is_binary else "weighted"

    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    roc_auc: float | None = None
    if is_binary and y_proba is not None:
        roc_auc = roc_auc_score(y_true, y_proba)

    return ModelResult(
        name=model_name,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        confusion_matrix=confusion_matrix(y_true, y_pred).astype(int).tolist(),
        report=classification_report(y_true, y_pred, zero_division=0),
    )


def run_classification_analysis(
    dataframe: pd.DataFrame,
    target_column: str | None = None,
    drop_columns: list[str] | None = None,
    test_size: float = 0.25,
    knn_neighbors: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    dataframe = dataframe.copy()
    drop_columns = drop_columns or []

    resolved_target = _validate_inputs(dataframe, target_column, test_size, knn_neighbors)

    for column in drop_columns:
        if column in dataframe.columns and column != resolved_target:
            dataframe = dataframe.drop(columns=[column])

    dataframe = dataframe.dropna(subset=[resolved_target])
    if dataframe.empty:
        raise AnalysisError("No rows remain after removing missing target values.")

    y_raw = dataframe[resolved_target]
    X_raw = dataframe.drop(columns=[resolved_target])

    # Convert feature columns to one-hot encoded values for algorithm compatibility.
    X = pd.get_dummies(X_raw, drop_first=False)
    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    if X.empty:
        raise AnalysisError("No usable feature rows remain after preprocessing.")

    y_raw = y_raw.loc[X.index]

    unique_classes = y_raw.nunique(dropna=True)
    if unique_classes < 2:
        raise AnalysisError("Target column must have at least two classes.")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # Keep class distribution stable whenever stratification is possible.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if unique_classes > 1 else None,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=200)),
        ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=knn_neighbors)),
        ("Gaussian Naive Bayes", GaussianNB()),
    ]

    model_results: list[ModelResult] = []
    is_binary = unique_classes == 2

    for model_name, model in models:
        if model_name in {"Logistic Regression", "K-Nearest Neighbors"}:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = (
                model.predict_proba(X_test_scaled)[:, 1]
                if is_binary and hasattr(model, "predict_proba")
                else None
            )
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = (
                model.predict_proba(X_test)[:, 1]
                if is_binary and hasattr(model, "predict_proba")
                else None
            )

        model_results.append(
            _compute_metrics(
                model_name=model_name,
                y_true=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                is_binary=is_binary,
            )
        )

    best_model = max(model_results, key=lambda result: result.accuracy)

    return {
        "dataset_shape": dataframe.shape,
        "target_column": resolved_target,
        "class_labels": label_encoder.classes_.tolist(),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "models": [result.__dict__ for result in model_results],
        "best_model": {
            "name": best_model.name,
            "accuracy": best_model.accuracy,
        },
    }
