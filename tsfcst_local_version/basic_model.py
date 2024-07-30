import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import numpy as np

class TimeSeriesPredictionModel():
    """
    Time series prediction model implementation

    Parameters
    ----------
        model_class : class
            Choice of regressor
        model_params : dict
            Definition of model specific tuning parameters

    Functions
    ----------
        init: Initialize model with given parameters
        train : Train chosen model
        forecast : Apply trained model to prediction period and generate forecast DataFrame
    """
    def __init__(self, model_class, model_params: dict) -> None:
        """Initialize a new instance of time_series_prediction_model."""
        self.model_class = model_class
        self.model_params = model_params
        self.model = None
        self.is_univariate = 'endog' in model_class.__init__.__code__.co_varnames

    def train(self, X_train: pd.DataFrame = None, y_train: pd.Series = None, train_series: pd.Series = None) -> None:
        """Train chosen model."""

        if self.is_univariate:
            if train_series is None:
                raise ValueError("train_series must be provided for univariate models")
            self.train_series = train_series
            self.model = self.model_class(endog=self.train_series, **self.model_params)
            self.model = self.model.fit()
        else:
            if X_train is None or y_train is None:
                raise ValueError("X_train and y_train must be provided for multivariate models")
            self.X_train = X_train
            self.y_train = y_train
            self.model = self.model_class(**self.model_params)
            self.model.fit(self.X_train, self.y_train)

    def forecast(self, X_test: pd.DataFrame = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Apply trained model to prediction period and generate forecast DataFrame."""
        if self.is_univariate:
            #if steps is None:
            if start_date is None or end_date is None:
                raise ValueError("start_date and end_date must be provided for univariate models")
            #forecast = self.model.predict(start = start_date, end= end_date, typ = 'levels')
            # Make predictions
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            forecast = self.model.predict(start=start_date, end=end_date, typ='levels')

            # forecast = self.model.forecast(steps)
            forecast_df = pd.DataFrame(forecast, columns=['Forecast'])
        else:
            if X_test is None:
                raise ValueError("X_test must be provided for multivariate models")
            self.X_test = X_test
            forecast_df = pd.DataFrame(self.model.predict(self.X_test), index=self.X_test.index)
            forecast_df.index.name = 'Datum'
        return forecast_df
    
# Backtesting with sliding window

def backtesting(X_train: pd.DataFrame, y_train: pd.DataFrame,
                X_test: pd.DataFrame, y_test: pd.DataFrame,
                model: TimeSeriesPredictionModel, prediction_step_size: int=96):
    """
    Perform rolling forecast backtesting for a time series prediction model using
    specified train and test datasets, and a given model.

    This function splits the test data into multiple windows based on the
    prediction_step_size and sequentially forecasts each window. After each
    forecasting step, the window of test data used for the current prediction is
    added to the training data, and the earliest window of the training data is
    removed. The predictions are stored in a DataFrame alongside the original test data values.

    Args:
        X_train (pd.DataFrame): Training feature dataset.
        y_train (pd.DataFrame): Training target dataset.
        X_test (pd.DataFrame): Testing feature dataset.
        y_test (pd.DataFrame): Testing target dataset.
        model (TimeSeriesPredictionModel): The model used for time series forecasting.
        prediction_step_size (int): The number of time steps to predict at each iteration.

    Returns:
        pd.DataFrame: A DataFrame with two columns 'Original' and 'Predictions',
                      containing the actual values from y_test and the predictions
                      made by the model, respectively.
    """

    # initializing output df
    predictions = pd.DataFrame(index=y_test.index, columns=['Original', 'Predictions'])
    predictions['Original'] = y_test

    for i in range(0, len(X_test)-prediction_step_size, prediction_step_size):
        end_idx = i + prediction_step_size
        forecast_index= X_test.iloc[i:end_idx].index

        # fit model and predict
        model.train(X_train, y_train)
        forecast = model.forecast(X_test.iloc[i:end_idx])
        predictions.loc[forecast_index, 'Predictions'] = forecast.to_numpy()

        print(f'Finished Forecast for {forecast_index[-1].date()}')

        # delete old time window from train data
        X_train = X_train.drop(X_train.head(prediction_step_size).index)
        y_train = y_train.drop(y_train.head(prediction_step_size).index)

        # add next time window to train data
        X_train = pd.concat([X_train, X_test.iloc[i:end_idx]])
        y_train = pd.concat([y_train, y_test.iloc[i:end_idx]])

    return predictions


def evaluation(y_true, y_pred):

    """
    Calculate various error metrics to evaluate the accuracy of a regression model.

    This function computes the mean absolute error (MAE), mean absolute percentage error (MAPE),
    mean squared error (MSE), coefficient of determination (R^2 score), and root mean squared error (RMSE)
    between the actual and predicted values.

    Args:
        y_true (array-like): True values for the target variable.
        y_pred (array-like): Predicted values generated by the model.

    Returns:
        tuple: A tuple containing:
               - mae (float): Mean absolute error.
               - mape (float): Mean absolute percentage error.
               - mse (float): Mean squared error.
               - r2 (float): R^2 score, measuring the proportion of variation explained by the model.

    """

    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return mae, mape, mse, r2, rmse


def random_search_optimization(model_class, param_grid, train_series=None, X_train=None, y_train=None,
                               X_val=None, y_val=None, n_iter=10, scoring_function=None):
    """
    Perform random search optimization on model parameters.

    Parameters
    ----------
    model_class : class
        Choice of model class (e.g., ARIMA, LinearRegression)
    param_grid : dict
        Dictionary with parameter names as keys and lists of parameter settings to try as values
    train_series : pd.Series, optional
        Training data for univariate models
    X_train : pd.DataFrame, optional
        Training features for multivariate models
    y_train : pd.Series, optional
        Training target for multivariate models
    X_val : pd.DataFrame, optional
        Validation features for multivariate models
    y_val : pd.Series, optional
        Validation target for multivariate models
    n_iter : int
        Number of parameter settings that are sampled
    scoring_function : function
        Function to evaluate model performance, should return a single score

    Returns
    -------
    best_params : dict
        Best parameter combination found
    best_score : float
        Best score obtained
    """

    def sample_params(param_grid):
        return {key: random.choice(values) for key, values in param_grid.items()}

    best_score = float('-inf')
    best_params = None

    for i in range(n_iter):
        print(i)
        params = sample_params(param_grid)
        model = TimeSeriesPredictionModel(model_class, params)

        if train_series is not None:
            model.train(train_series=train_series)
            forecast = model.forecast(steps=len(train_series))
            score = scoring_function(train_series[-len(forecast):], forecast)
        else:
            model.train(X_train=X_train, y_train=y_train)
            forecast = model.forecast(X_test=X_val)
            score = scoring_function(y_val, forecast)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score