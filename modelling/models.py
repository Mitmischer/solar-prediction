# import modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
# tensorboard
from torch.utils.tensorboard import SummaryWriter
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime


# Univariate models

# Prophet
def fit_prophet_model(data: pd.DataFrame, **kwargs):
    """
    Fits a Prophet model to the provided data.

    Parameters:
    data (pd.DataFrame): DataFrame containing 'ds' (date) and 'y' (value) columns.
    kwargs: Additional keyword arguments for the Prophet model.

    Returns:
    Prophet: Fitted Prophet model.
    """
    model = Prophet(**kwargs)
    model.fit(data)
    return model

def make_forecast(model: Prophet, periods: int, freq: str = 'H'):
    """
    Makes a forecast using the fitted Prophet model.

    Parameters:
    model (Prophet): Fitted Prophet model.
    periods (int): Number of periods to forecast.
    freq (str): Frequency of the forecast periods ('D' for days, 'M' for months, etc.).

    Returns:
    pd.DataFrame: Forecasted values.
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast

# LSTM
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2, device=False, bidirectional=False):
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.device = device

        # Building the LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out






