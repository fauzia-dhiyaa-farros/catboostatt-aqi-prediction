## Spatio-Temporal Prediction Modeling Of Air Quality Index (AQI) in DKI Jakarta using Catboost and Extreme Feature Attention Mechanism
This study aims to develop an Air Quality Index (AQI) prediction model for DKI Jakarta by combining the CatBoost Regressor algorithm with a Feature Attention Weighting mechanism that assigns attention weights of 0.5 for normal conditions and 1.0 for extreme conditions. The dataset consists of AQI data from DKI Jakarta and meteorological data from GHCN-H NOAA for the 2020–2024 period, totaling 106,197 entries, including PM₂.₅, PM₁₀, SO₂, CO, O₃, NO₂, temperature, 24 hour rainfall, relative humidity, wind direction, wind speed, and dew point temperature. The model was trained using time sequence windows (1, 8, 12, 24, and 48 hours) and a 70:15:15 (train:validation:test) data split, and compared against seven benchmark models: Linear Regression, Lasso Regression, LightGBM, FNN, CNN-LSTM, GRU-LSTM, and Transformer. The best performance was achieved at the 48-hour window with RMSE = 0.002 and MAPE = 0.00%, where the Friedman test indicated a significant difference (p = 0.010) and the Pairwise Test confirmed significant variations in RMSE (χ² = 11.200; p = 0.024), MAE (χ² = 13.280; p = 0.010), and MAPE (χ² = 13.280; p = 0.010). The spatiotemporal analysis revealed increases in PM₂.₅ and PM₁₀ concentrations during morning and evening hours due to traffic and industrial activities, with the highest concentrations recorded in Lubang Buaya (PM₂.₅ = 287 µg/m³; PM₁₀ = 187 µg/m³). Meanwhile, meteorological fluctuations driven by daily weather variations affected pollutant dispersion, particularly at Halim Perdanakusuma, where humidity reached 100%, temperature peaked at 38°C, and wind direction was 999°. Overall, CatBoost demonstrated high effectiveness, accuracy, and statistical relevance, making it a robust choice for spatiotemporal AQI prediction modeling in DKI Jakarta.

## Research Benefits
In the short term, the results of this study are expected to function as an information system for monitoring air quality conditions in DKI Jakarta by providing alerts to the public when pollutant concentrations or meteorological elements (weather) are predicted to exceed WHO-recommended limits during traffic and industrial peak hours.

In the long term, this system is expected to contribute to urban environmental planning and management in DKI Jakarta, such as:
- Establishing low-emission zones by restricting motorized vehicles in areas with high levels of air pollution.
- Developing environmentally friendly transportation systems, such as expanding public transportation networks and promoting the use of electric vehicles.
- Optimizing the use of Green Open Spaces (Ruang Terbuka Hijau) as pollutant-absorbing areas and as a means to improve air quality in urban environments.

## Setup Environment - Miniconda3
```
conda env create -f environment.yml
conda activate aqi-env

```

## Run Streamlit App Locally
```
streamlit run catboostatt_dashboard.py

```

## Run Steamlit App 
```
https://catboostatt-aqi-prediction.streamlit.app/

```

Copyright © Fauzia Dhiyaa' Farros 2025
