import numpy as np
import torch

from typing import Any
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


def evaluate_model(y_true, y_pred) -> Any:
    """
    Evaluate the model performance on the test dataset.
    Calculates MAE, MAPE, MSE, R2, and RMSE.

    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: dictionary containing evaluation metrics
    """

    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred))
    }

    return metrics


def permutation_importance(model, X, y, n_repeats=10):
    baseline_mse = mean_squared_error(y, model.predict(X))
    importances = []

    for column in range(X.shape[2]):  # Iterate over features
        feature_importances = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[:, :, column] = np.random.permutation(X_permuted[:, :, column])
            permuted_mse = mean_squared_error(y, model.predict(X_permuted))
            importance = permuted_mse - baseline_mse
            feature_importances.append(importance)
        importances.append(np.mean(feature_importances))

    return importances
