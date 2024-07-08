import pandas as pd

# Load and Save time series
# Save to csv
def save_to_csv(data: pd.Series, filename: str):
    """
    Save a time series to a csv file.
    :param data: pandas Series
    :param filename: String ending with .csv
    :return: None
    """
    data.to_csv(filename)

# Load from csv
