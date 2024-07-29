import tsfresh
import pandas as pd
import numpy as np

# multiple lagged features
def create_lagged_features(train_series: pd.Series | pd.DataFrame, val_series: pd.Series | pd.DataFrame,
                           test_series: pd.Series | pd.DataFrame, lags: list[int],
                           fill_method: str = 'ffill') -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Join train val test and Create lagged features for a time series on all splits of the data.
    Split the data back into train, validation, and test sets.
    Return three pd.DataFrames with lagged features added.

    :param train_series: The train time series data.
    :param val_series: The validation time series data.
    :param test_series: The test time series data.
    :param lags: A list of integers representing the lags to create.
    :param fill_method: Method to fill missing values. Options: 'ffill', 'bfill', 'zero', 'mean', 'drop'
    :return: The train, validation, and test time series data with lagged features.
    """

    # Join train val test
    full_series = pd.concat([train_series, val_series, test_series])

    def add_lags(series):
        if isinstance(series, pd.Series):
            original = series.to_frame()
        else:
            original = series.copy()

        lagged_features = []
        for lag in lags:
            if isinstance(series, pd.Series):
                lagged_features.append(series.shift(lag).rename(f"lag_{lag}"))
            else:
                lagged_features.extend(
                    [series[column].shift(lag).rename(f"{column}_lag_{lag}") for column in series.columns])

        result = pd.concat([original] + lagged_features, axis=1)

        # Handle missing values
        if fill_method == 'ffill':
            result = result.ffill()
        elif fill_method == 'bfill':
            result = result.bfill()
        elif fill_method == 'zero':
            result = result.fillna(0)
        elif fill_method == 'mean':
            result = result.fillna(result.mean())
        elif fill_method == 'drop':
            result = result.dropna()
        else:
            raise ValueError("Invalid fill_method. Choose from 'ffill', 'bfill', 'zero', 'mean', 'drop'")

        return result

    # Create lagged features
    full_lagged = add_lags(full_series)

    # Split back into train, validation, and test sets
    train_lagged = full_lagged.iloc[:len(train_series)]
    val_lagged = full_lagged.iloc[len(train_series):len(train_series) + len(val_series)]
    test_lagged = full_lagged.iloc[len(train_series) + len(val_series):]

    return train_lagged, val_lagged, test_lagged


def create_rolling_features(train_series: pd.Series | pd.DataFrame, val_series: pd.Series | pd.DataFrame,
                            test_series: pd.Series | pd.DataFrame, windows: list[int],
                            fill_method: str = 'ffill') -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create rolling features for a time series on all splits of the data.
    Return three pd.DataFrames with rolling features added.

    :param train_series: The train time series data.
    :param val_series: The validation time series data.
    :param test_series: The test time series data.
    :param windows: A list of integers representing the window sizes to create.
    :param fill_method: Method to fill missing values. Options: 'ffill', 'bfill', 'zero', 'mean', 'drop'
    :return: The train, validation, and test time series data with rolling features.
    """
    # Join train val test
    full_series = pd.concat([train_series, val_series, test_series])

    def add_rolling_features(series):
        if isinstance(series, pd.Series):
            original = series.to_frame()
        else:
            original = series.copy()

        rolling_features = []
        for window in windows:
            if isinstance(series, pd.Series):
                rolling_features.extend([
                    series.rolling(window=window).mean().rename(f"rolling_mean_{window}"),
                    series.rolling(window=window).std().rename(f"rolling_std_{window}")
                ])
            else:
                for column in series.columns:
                    rolling_features.extend([
                        series[column].rolling(window=window).mean().rename(f"{column}_rolling_mean_{window}"),
                        series[column].rolling(window=window).std().rename(f"{column}_rolling_std_{window}")
                    ])

        result = pd.concat([original] + rolling_features, axis=1)

        # Handle missing values
        if fill_method == 'ffill':
            result = result.ffill()
        elif fill_method == 'bfill':
            result = result.bfill()
        elif fill_method == 'zero':
            result = result.fillna(0)
        elif fill_method == 'mean':
            result = result.fillna(result.mean())
        elif fill_method == 'drop':
            result = result.dropna()
        else:
            raise ValueError("Invalid fill_method. Choose from 'ffill', 'bfill', 'zero', 'mean', 'drop'")

        return result

    # Create rolling features
    full_rolling = add_rolling_features(full_series)

    # Split back into train, validation, and test sets
    train_rolling = full_rolling.iloc[:len(train_series)]
    val_rolling = full_rolling.iloc[len(train_series):len(train_series) + len(val_series)]
    test_rolling = full_rolling.iloc[len(train_series) + len(val_series):]

    return train_rolling, val_rolling, test_rolling


def create_datetime_features(train_series: pd.Series | pd.DataFrame, val_series: pd.Series | pd.DataFrame,
                             test_series: pd.Series | pd.DataFrame, datetime_column: str = None) -> tuple[
    pd.Series, pd.Series, pd.Series]:
    """
    Create time features for a time series on all splits of the data.
    Use sine and cosine transformations to encode the time of day and year.
    Return three pd.DataFrames with time features added.

    :param train_series: The train time series data.
    :param val_series: The validation time series data.
    :param test_series: The test time series data.
    :param datetime_column: The name of the column containing datetime information. If None, assumes index is datetime.
    :return: The train, validation, and test time series data with time features.
    """

    # Join train val test
    full_series = pd.concat([train_series, val_series, test_series])

    def add_datetime_features(series):
        if isinstance(series, pd.Series):
            original = series.to_frame()
        else:
            original = series.copy()

        # Determine the datetime source
        if datetime_column is not None:
            if datetime_column not in original.columns:
                raise ValueError(f"Column '{datetime_column}' not found in the DataFrame.")
            datetime_source = original[datetime_column]
        elif isinstance(original.index, pd.DatetimeIndex):
            datetime_source = original.index
        else:
            raise ValueError("No datetime column specified and index is not a DatetimeIndex.")

        # Extract the time features
        datetime_features = pd.DataFrame(index=original.index)
        datetime_features['hour'] = datetime_source.hour
        datetime_features['dayofyear'] = datetime_source.dayofyear

        # Encode the time features using sine and cosine transformations
        for col in ['hour', 'dayofyear']:
            max_val = 24 if col == 'hour' else 366
            datetime_features[f'{col}_sin'] = np.sin(2 * np.pi * datetime_features[col] / max_val)
            datetime_features[f'{col}_cos'] = np.cos(2 * np.pi * datetime_features[col] / max_val)

        # Drop the original columns (keeping only sin and cos transformations)
        datetime_features = datetime_features.drop(['hour', 'dayofyear'], axis=1)

        return pd.concat([original, datetime_features], axis=1)

    # Create datetime features
    full_datetime = add_datetime_features(full_series)

    # Split back into train, validation, and test sets
    train_datetime = full_datetime.iloc[:len(train_series)]
    val_datetime = full_datetime.iloc[len(train_series):len(train_series) + len(val_series)]
    test_datetime = full_datetime.iloc[len(train_series) + len(val_series):]

    return train_datetime, val_datetime, test_datetime


