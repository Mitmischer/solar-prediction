import pandas as pd
from statsmodels.tsa.stattools import adfuller


# split into train val test
def split_ts(ts: pd.Series, val_start: str, test_start: str, date_column: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Split a time series into training, validation, and test sets based on the specified dates.
    Set the datetime column as the index for each set.

    Parameters:
    ts (pd.Series): Time series data with a datetime index.
    val_start (str): The start date for the validation set.
    test_start (str): The start date for the test set.
    date_column (str): The name of the column containing datetime values.

    Returns:
    pd.Series: The training, validation, and test sets.
    """
    train = ts[ts[date_column] < val_start]
    val = ts[(ts[date_column] >= val_start) & (ts[date_column] < test_start)]
    test = ts[ts[date_column] >= test_start]

    # Set indices
    train_ts = train.set_index(keys=date_column, drop=True)
    val_ts = val.set_index(keys=date_column, drop=True)
    test_ts = test.set_index(keys=date_column, drop=True)

    return train_ts, val_ts, test_ts


# handle leap years
def handle_leap_years(ts: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust the day of the year for leap years and extract the month from the date.

    Parameters:
    ts (pd.DataFrame): The DataFrame containing time series data with datetime index.

    Returns:
    pd.DataFrame: A DataFrame with the original data and additional columns
                  for adjusted day of the year and month.
    """

    def remove_leapyears(row: pd.Series) -> pd.Series:
        date = row.name
        adjusted_day = date.dayofyear
        if date.is_leap_year and date.dayofyear > 59:
            adjusted_day -= 1
        return pd.Series({"adjusted_dayofyear": adjusted_day, "month": date.month})

    # Apply the function to the DataFrame
    new_columns = ts.apply(remove_leapyears, axis=1)

    # Add the new columns to the DataFrame
    ts["adjusted_dayofyear"] = new_columns["adjusted_dayofyear"]
    ts["month"] = new_columns["month"]

    return ts


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
        print(f"Missing timestamps detected in {split} dataset:")
        print(f"{missing_times}")
    else:
        print(f"No missing timestamps in {split} dataset")


# Interpolate missing timestamps
def interpolate_missing_timestamps(ts: pd.Series, freq: str = '15min') -> pd.Series:
    """
    Interpolate missing timestamps in a time series data based on a specified frequency.

    Parameters:
    ts (pd.Series): Time series data with a datetime index.
    freq (str): Frequency string (default is '15min').

    Returns:
    pd.Series: Time series data with missing timestamps interpolated.
    """
    expected_range = pd.date_range(start=ts.index.min(), end=ts.index.max(), freq=freq)
    interpolated = ts.reindex(expected_range)
    interpolated.interpolate(method='time', inplace=True)
    return interpolated


# Resample
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


# Make stationary - Augmented Dicky Fuller Test
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
