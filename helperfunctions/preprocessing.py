import pandas as pd
from statsmodels.tsa.stattools import adfuller

def adjust_dayofyear(row: pd.Series, date_col: str) -> pd.Series:
    """
    Adjust the day of the year for leap years and extract the month from the date.

    Parameters:
    row (pd.Series): A row of data containing a date column.
    date_col (str): The name of the column containing datetime values.

    Returns:
    pd.Series: A series with adjusted day of the year and month.

    Explanation:
    The adjustment ensures that the day of the year after February 28th in leap years
    is consistent with non-leap years by subtracting 1 from the day of the year
    for dates after February 28th. This removes the extra day (February 29th) from the count.
    """
    date = row[date_col]
    adjusted_day = date.dayofyear
    if date.is_leap_year and date.dayofyear > 59:  # Check for leap year and adjust the day of the year
        adjusted_day -= 1
    return pd.Series({"adjusted_dayofyear": adjusted_day, "month": date.month})


# Check for continuity
def check_continuity(ts: pd.Series, split: str, freq: str = '15min') -> None:
    """
    Check for missing timestamps in a time series data based on a specified frequency.

    Parameters:
    ts (pd.Series): Time series data with a datetime index.
    split (str): Identifier for the data split being checked, e.g. 'train', 'test', 'val'.
    freq (str): Frequency string (default is '15min').

    Returns:
    None
    """
    expected_range = pd.date_range(start=ts.index.min(), end=ts.index.max(), freq=freq)
    missing_times = expected_range.difference(ts.index)
    if not missing_times.empty:
        print(f"Missing timestamps detected in {split}: {missing_times}")
    else:
        print(f"No missing timestamps in {split}")


# Resample
import pandas as pd

def resample_ts(ts: pd.Series, freq: str = 'h') -> pd.Series:
    """
    Resample a time series to a specified frequency and aggregate the values by summing.

    Parameters:
    ts (pd.Series): Time series data with a datetime index.
    freq (str): Frequency string to resample the time series (default is 'h' for hourly).

    Returns:
    pd.Series: Resampled time series with values aggregated by sum.
    """
    return ts.resample(freq).sum()


# Normalise
def normalise_ts(train_ts, val_ts, test_ts: pd.Series) -> pd.Series:
    """
    Normalise a time series using the mean and standard deviation of the training set.

    Parameters:
    train_ts (pd.Series): Time series data for training.
    val_ts (pd.Series): Time series data for validation.
    test_ts (pd.Series): Time series data for testing.

    Returns:
    pd.Series: Normalised time series data.
    """
    mean = train_ts.mean()
    std = train_ts.std()
    normalised_train = (train_ts - mean) / std
    normalised_val = (val_ts - mean) / std
    normalised_test = (test_ts - mean) / std
    return normalised_train, normalised_val, normalised_test


# Stationarity - Augmented Dicky Fuller Test
def make_stationary(train_ts, val_ts, test_ts: pd.Series) -> None:
    """
    Perform the Augmented Dickey-Fuller test on train_ts. If train_ts is not stationary,
    difference train_ts, val_ts, and test_ts based on the train_ts result.

    Parameters:
    train_ts (pd.DataFrame): The training time series data.
    val_ts (pd.DataFrame): The validation time series data.
    test_ts (pd.DataFrame): The test time series data.

    Returns:
    None
    """

    def perform_adf(series, column_name):
        result = adfuller(series, autolag='AIC')
        print(f'ADF Statistic for {column_name}: {result[0]}')
        print(f'p-value for {column_name}: {result[1]}')
        for key, value in result[4].items():
            print(f'Critical Values for {column_name} {key}: {value}')
        return result[1] < 0.05  # Return True if the series is stationary

    def check_and_difference(series, column_name):
        differenced = False
        is_stationary = perform_adf(series, column_name)
        iteration = 0
        while not is_stationary:
            print(f'The time series {column_name} is not stationary. Differencing the series and re-testing...')
            series = series.diff().dropna()
            is_stationary = perform_adf(series, f'{column_name} (Differenced {iteration + 1})')
            differenced = True
            iteration += 1

        if is_stationary:
            print(f'The time series {column_name} is stationary after differencing {iteration} time(s).')
        else:
            print(f'The time series {column_name} is still not stationary after differencing {iteration} time(s).')

        return series, differenced

    for col in train_ts.columns:
        print(f'Checking stationarity for {col}')
        train_ts[col], differenced = check_and_difference(train_ts[col], col)
        if differenced:
            val_ts[col] = val_ts[col].diff().dropna()
            test_ts[col] = test_ts[col].diff().dropna()
        print("\n")  # Add a space between outputs


