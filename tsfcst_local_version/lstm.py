import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


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


