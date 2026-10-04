# Smart Electricity Demand Forecasting

An end-to-end machine learning application for **24-hour household electricity-demand forecasting**, peak-risk classification, anomaly screening, uncertainty estimation, and prediction logging.

> This project uses historical measurements from a single household. It is a forecasting prototype and does not represent live utility-grid monitoring or control.

## Project Overview

The system processes the UCI Individual Household Electric Power Consumption dataset and models hourly average power demand. It combines time-series feature engineering with XGBoost-based forecasting and classification, Isolation Forest anomaly screening, a Streamlit analytics dashboard, and SQLite logging.

## Key Features

- 24-hour recursive electricity-demand forecasting
- 15 temporal, lag, and rolling-window features
- Chronological 80/20 train-test validation
- Linear Regression, Random Forest, and XGBoost regression benchmarking
- Dedicated XGBoost peak-risk classifier
- Isolation Forest anomaly screening
- Approximate residual-based 95% prediction intervals
- Streamlit interactive analytics dashboard
- SQLite prediction logging

## Engineered Features

**Temporal:** `hour`, `dayofweek`, `quarter`, `month`, `year`, `is_weekend`

**Lag:** `lag_1h`, `lag_2h`, `lag_3h`, `lag_24h`, `lag_7d`

**Rolling:** `rolling_mean_3h`, `rolling_mean_6h`, `rolling_mean_24h`, `rolling_std_24h`

## Model Performance

| Model | RMSE (kW) | MAE (kW) | R² | MAPE |
|---|---:|---:|---:|---:|
| Linear Regression | 0.5100 | 0.3616 | 0.5126 | 50.34% |
| Random Forest | 0.4799 | 0.3302 | 0.5686 | 47.22% |
| **XGBoost Regressor** | **0.4674** | **0.3248** | **0.5907** | **44.89%** |

XGBoost was selected as the deployed regression model. MAPE should be interpreted cautiously because electricity demand can approach zero.

## Peak-Risk Classification

The dedicated XGBoost classifier uses a demand threshold of approximately **2.86 kW** and a probability threshold of **60%**.

- Precision: 0.1761
- Recall: 0.3972
- F1 score: 0.2440

The classifier should be interpreted as a peak-risk screening component rather than a highly accurate alarm system.

## Forecasting and Uncertainty

The application produces a genuine recursive 24-hour forecast: each predicted hour becomes part of the history used to construct features for subsequent hours.

Uncertainty bounds are estimated using residual standard deviation:

`prediction ± 1.96 × residual_std`

The lower bound is clipped at zero. These are approximate residual-based prediction intervals, not calibrated confidence intervals.

## Anomaly Detection

Isolation Forest provides unsupervised anomaly screening over forecast feature patterns. Because the dataset has no labeled anomaly ground truth, anomaly flags should not be interpreted as validated electrical faults and no supervised accuracy claim is made.

## Project Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard and forecasting workflow |
| `database.py` | SQLite prediction logging |
| `xgboost_power_model.json` | Trained demand forecasting model |
| `xgboost_peak_classifier.json` | Trained peak-risk classifier |
| `iso_forest_model.pkl` | Isolation Forest anomaly model |
| `energy_thresholds.json` | Peak-risk thresholds |
| `forecast_state.json` | Historical state required for recursive forecasting |
| `model_benchmark.json` | Regression benchmark metrics |
| `peak_metrics.json` | Peak detection/classifier metrics |
| `residual_std.npy` | Residual standard deviation for uncertainty bounds |

## Installation

Clone the repository:

```bash
git clone https://github.com/KishanAnand/Smart-Electricity-Demand-Forecasting.git
cd Smart-Electricity-Demand-Forecasting
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

## Dataset

The model was developed using the **UCI Individual Household Electric Power Consumption** dataset, containing more than two million one-minute household electricity measurements spanning nearly four years. Measurements were resampled to hourly averages, with `Global_active_power` used as the forecasting target.

The raw dataset is intentionally not included in this repository.

## Limitations

- Historical single-household data rather than live utility-grid data
- No weather variables
- Recursive multi-step forecasts can accumulate prediction error
- Peak classifier has modest F1 performance
- Isolation Forest flags unusual patterns but cannot validate electrical faults without labels
- Residual-based uncertainty intervals are approximate rather than calibrated
- Held-out one-step regression metrics should not be interpreted as direct 24-hour recursive forecast accuracy

## Future Improvements

Potential improvements include rolling/walk-forward 24-hour backtesting, newer/live data sources, relevant weather features, conformal or quantile prediction intervals, SHAP-based explainability, and monitoring against newly observed actual demand.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Streamlit, Plotly, Joblib, and SQLite.
