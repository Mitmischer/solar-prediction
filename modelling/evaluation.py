from typing import Any

import numpy as np
import torch
from numpy import ndarray, dtype
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_absolute_error


# Evaluate lstm
def lstm_evaluate(model, dataloaders, device):
    model.eval()  # Set the model to evaluation mode
    model.to(device)
    predictions = []
    actuals = []

    with torch.no_grad():
        for features, labels in dataloaders["test"]:
            features = features.to(device)  # Move features to GPU
            labels = labels.to(device)  # Move labels to GPU
            outputs = model(features)
            outputs = outputs.view_as(labels)
            predictions.extend(outputs.view(-1).tolist())  # Flatten and store predictions
            actuals.extend(labels.view(-1).tolist())  # Flatten and store actual labels

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    metrics = {
        "r2": r2_score(actuals, predictions),
        "mse": mean_squared_error(actuals, predictions),
        "rmse": np.sqrt(mean_squared_error(actuals, predictions)),
        "mape": mean_absolute_percentage_error(actuals, predictions),
        "mae": mean_absolute_error(actuals, predictions)
    }

    return metrics

