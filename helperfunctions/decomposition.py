import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt


# Make stationary - Augmented Dicky Fuller Test
def make_stationary_unitroot(train_ts, val_ts, test_ts: pd.Series) -> None:
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
        diff = False
        is_stationary = perform_adf(series, column_name)
        iteration = 0
        while not is_stationary:
            print(f'The time series {column_name} is not stationary. Differencing the series and re-testing...')
            series = series.diff().dropna()
            is_stationary = perform_adf(series, f'{column_name} (Differenced {iteration + 1})')
            diff = True
            iteration += 1

        if is_stationary:
            print(f'The time series {column_name} is stationary after differencing {iteration} time(s).')
        else:
            print(f'The time series {column_name} is still not stationary after differencing {iteration} time(s).')

        return series, diff

    for col in train_ts.columns:
        print(f'Checking stationarity for {col}')
        train_ts[col], differenced = check_and_difference(train_ts[col], col)
        if differenced:
            val_ts[col] = val_ts[col].diff().dropna()
            test_ts[col] = test_ts[col].diff().dropna()
        print("\n")  # Add a space between outputs


def check_stationarity_variance(ts: pd.DataFrame | pd.Series, window: int = 24) -> None:
    """
    Check the stationarity of a time series based on the variance of a rolling window.
    Print whether the time series has constant variance.

    Parameters:
    :param ts: Train Time series data (DataFrame or Series).
    :param window: The size of the rolling window (default is 24).
    :return: None
    """
    if isinstance(ts, pd.DataFrame):
        for column in ts.columns:
            print(f"\nChecking stationarity for column: {column}")
            check_stationarity_variance_single(ts[column], window)
    else:
        check_stationarity_variance_single(ts, window)


def check_stationarity_variance_single(ts: pd.Series, window: int = 24, plot: bool = True) -> None:
    """
    Check the stationarity of a single time series based on the variance of a rolling window.
    Print whether the time series has constant variance.
    Plot the rolling variance and print the results of the Augmented Dickey-Fuller and KPSS tests.
    """
    # Calculate rolling variance
    rolling_var = ts.rolling(window=window).var().dropna()

    # Perform Augmented Dickey-Fuller test on rolling variance
    adf_result = adfuller(rolling_var)

    # Perform Kwiatkowski-Phillips-Schmidt-Shin test on rolling variance
    kpss_result = kpss(rolling_var)

    # Plot rolling variance with transparent background
    if plot:
        plt.figure(figsize=(12, 6))

        plt.plot(rolling_var)
        plt.title(f'Rolling Variance (window={window})')
        plt.xlabel('Time')
        plt.ylabel('Variance')
        plt.show()

    # Check results and print conclusion
    if adf_result[1] < 0.05 <= kpss_result[1]:
        print("The time series appears to have constant variance (stationary).")
    else:
        print("The time series does not appear to have constant variance (non-stationary).")

    # Print test results
    print("\nAugmented Dickey-Fuller Test:")
    print(f"ADF Statistic: {adf_result[0]}")
    print(f"p-value: {adf_result[1]}")

    print("\nKwiatkowski-Phillips-Schmidt-Shin Test:")
    print(f"KPSS Statistic: {kpss_result[0]}")
    print(f"p-value: {kpss_result[1]}")


def detrend_ts(ts: pd.Series) -> pd.Series:
    """
    Detrend a time series by differencing.

    Parameters:
        :param ts: Time series data.
        :return: Detrended time series data.
    """
    return ts.diff().dropna()


def deseasonalise_ts(ts: pd.Series, period: int = 365) -> pd.Series:
    """
    Deseasonalise a time series by differencing.

    Parameters:
        :param ts: Time series data.
        :param period: Seasonal period.
        :return: Deseasonalised time series data.
    """
    return ts.diff(period).dropna()


def retrend_ts(original_ts: pd.Series, detrended_ts: pd.Series) -> pd.Series:
    """
    Retrend a detrended time series by adding the trend back.

    Parameters:
        :param original_ts: Original time series data.
        :param detrended_ts: Detrended time series data.
        :return: Retrended time series data.
    """
    return original_ts.shift(1) + detrended_ts


def reseasonalise_ts(original_ts: pd.Series, deseasonalised_ts: pd.Series, period: int = 365) -> pd.Series:
    """
    Reseasonalise a deseasonalised time series by adding the seasonal component back.

    Parameters:
        :param original_ts: Original time series data.
        :param deseasonalised_ts: Deseasonalised time series data.
        :param period: Seasonal period.
        :return: Reseasonalised time series data.
    """
    return original_ts.shift(period) + deseasonalised_ts
