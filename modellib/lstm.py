# LSTM model

import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Any
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


# Create sequences
def create_sequences(series: pd.Series, target_column: str, sequence_length: int = 24, batch_size: int = 8) -> (
np.ndarray, np.ndarray):
    features = series.values
    target = series[target_column].values

    data_gen = TimeseriesGenerator(
        features,
        target,
        sequence_length,
        batch_size
    )

    X, y = [], []
    for i in range(len(data_gen)):
        x, y_batch = data_gen[i]
        X.append(x)
        y.append(y_batch)

    X = np.concatenate(X)
    y = np.concatenate(y)

    return X, y


def create_lstm_model(input_shape, units=50, dropout_rate=0.1, optimizer=Adam(learning_rate=0.0001), loss='mse'):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units),
        Dropout(dropout_rate),
        Dense(1)
    ])

    optimizer = optimizer
    model.compile(optimizer=optimizer, loss=loss)

    return model


def train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=8):
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    with tf.device('/GPU:0'):
        history = model.fit(
            tf.constant(X_train), tf.constant(y_train),
            validation_data=(tf.constant(X_val), tf.constant(y_val)),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )

    return history


# Metrics
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
            X_permuted = tf.identity(X)
            X_permuted[:, :, column] = np.random.permutation(X_permuted[:, :, column])
            permuted_mse = mean_squared_error(y, model.predict(X_permuted))
            importance = permuted_mse - baseline_mse
            feature_importances.append(importance)
        importances.append(np.mean(feature_importances))

    return importances
