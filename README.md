# Learning from Light: Advanced Time Series Analysis for German Solar Energy Forecasting

## Introduction
The rising demand for renewable energy and the variability in solar power generation due to weather conditions necessitate accurate prediction models for energy supply. This project outlines a comprehensive approach to forecasting solar power generation, utilizing both univariate and multivariate time series models. Key methodologies include preprocessing techniques such as stationarity adjustments, Fourier analysis, and anomaly detection using autoencoders. Various models, including Moving Average, NBEATS, NHITS, Random Forest, and CatBoost for univariate data, and LSTM for multivariate data, are evaluated. The study leverages weather data from six German cities and highlights the challenges and potential improvements in the prediction models.

## Background
The necessity for a consistent and affordable energy supply is critical for industrial expansion, particularly as renewable and weather-dependent energy sources such as solar power become more widely used. As part of the Opencampus Course "Advanced Time Series Prediction," this group project intends to test alternative time series algorithms for forecasting solar power supply in Germany, taking into account the high volatility and seasonality associated with solar power output.

## Data Sources
The solar power dataset was obtained from [Energy Charts](https://www.energy-charts.info/charts/power/chart.htm?l=de&c=DE&source=total&interval=year&legendItems=lyf&year=2024) and historical weather data from [Open-Meteo](https://open-meteo.com/en/docs#latitude=52.5244&longitude=13.4105&timezone=Europe%2FBerlin). Regarding the weather data, we considered six German cities (Templin, Kastellaun, Gütersloh, Ingolstadt, Erfurt, and Neumünster) that are recognized for their prominent solar energy centers while simultaneously covering a broad geographical range within Germany. Three datasets were created from the data, using different preprocessing, feature engineering, and feature selection techniques. The datasets have been made available on Hugging Face:  [Original Dataset](https://huggingface.co/datasets/Creatorin/solarpower), [PCA Dataset](https://huggingface.co/datasets/Creatorin/solar_pca), and [Feature-Selected Dataset](https://huggingface.co/datasets/Creatorin/solar_selected).The goal is to develop a reliable forecasting model that can accurately predict Germany's solar power supply, ultimately contributing to the advancement of renewable energy technologies.

## Preprocessing Techniques
To prepare the data for analysis, the following preprocessing steps were undertaken:

- **Stationarity:** The data was standardized, differentiated to remove trends, and normalized by annual volatility.
- **Fourier Analysis:** Various frequency cutoffs were tested to optimize the performance and quality of fit.
- **Anomaly Detection:** Data was sliced into smaller vectors, and an autoencoder was trained to detect anomalies based on reconstruction errors.

## Univariate Models
Several univariate models were tested, including:

- **Moving Average:** A simple mean of the past hours, yielding an \( R^2 \) score of 0.6150.
- **NBEATS and NHITS:** Advanced models with varying levels of computational complexity and performance.
- **Autocorrelation Features:** Lags with the highest autocorrelation values were identified to enhance model accuracy.
- **Random Forest and CatBoost:** Ensemble methods showed promising results, with \( R^2 \) scores of 0.8430 and 0.8457, respectively.

## Multivariate Models
The Long Short-Term Memory (LSTM) model was employed for multivariate analysis, incorporating additional weather data and engineered features, including lagged features, rolling features, and time-dependent features such as sine and cosine transformations of the datetime values. Despite promising initial results (\( R^2 \) score of 0.8826 for the original LSTM model), the introduction of feature engineering and Principal Component Analysis (PCA) did significantly impede performance (\( R^2 \) scores of 0.4499 and 0.2834, respectively).

## Challenges and Future Outlook
The greatest challenge we faced was related to the need for more GPU resources. Future efforts will focus on optimizing the models in terms of efficiency and performance, as well as exploring transformer models on different data subsets.

## Conclusion
This research underscores the importance of selecting appropriate preprocessing techniques and models for accurate solar power prediction. While univariate models like CatBoost showed robust performance, multivariate models require more careful feature selection and optimization. This is indeed a trade-off, but with some additional effort, these models outperform the traditional, univariate methods.

## Performance Metrics Table
| Type | Model | MAE | MSE | RMSE | R^2 |
|------|-------|-----|-----|------|-----|
| Univariate | Moving Av. | 1.4802 | 5.9520 | 2.4397 | 0.6150 |
| Univariate | Rdn. Forest | 0.81939 | 2.4157 | 1.5542 | 0.8430 |
| Univariate | CatBoost | 0.8177 | 2.3737 | 1.5407 | 0.8457 |
| Multivariate | LSTM (Orig) | 0.2031 | 0.1739 | 0.0417 | 0.8826 |
| Multivariate | LSTM (FE) | 1.0324 | 2.5386 | 1.5933 | 0.4499 |
| Multivariate | LSTM (PCA) | 1.0554 | 3.0770 | 1.7541 | 0.2834 |



