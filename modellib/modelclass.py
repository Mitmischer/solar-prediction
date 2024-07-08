class TimeSeriesPredictionModel():
    """
    Time series prediction model implementation

    Parameters
    ----------
        model_name : class
            Choice of regressor
        model_params : dict
            Definition of model specific tuning parameters

    Functions
    ----------
        init: Initialize model with given parameters
        train : Train chosen model
        forcast : Apply trained model to prediction period and generate forecast DataFrame
    """

    def __init__(self, model_name,
                 model_params: dict) -> None:
        """Initialize a new instance of time_series_prediction_model."""
        self.model = model_name(**model_params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train chosen model."""
        self.X_train = X_train
        self.y_train = y_train
        self.model.fit(self.X_train, self.y_train)

    def forecast(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """Apply trained model to prediction period and generate forecast DataFrame."""
        self.X_test = X_test
        forecast_df = pd.DataFrame(self.model.predict(self.X_test), index=self.X_test.index)
        forecast_df.index.name = 'Datum'
        return forecast_df